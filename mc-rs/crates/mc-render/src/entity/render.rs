//! The outer entity transform and the `ENTITIES` render phase.
//!
//! java: `client/renderer/entity/RendererLivingEntity.doRender`, `Render.java`,
//! `client/renderer/RenderHelper.java` (the two-light entity lighting).
//!
//! # The transform, and why it looks like that
//!
//! Vanilla builds the entity matrix as a sequence of `GlStateManager` calls:
//!
//! ```text
//! translate(x, y, z)                 // renderLivingAt — position relative to the view entity
//! rotate(180 - bodyYaw, Y)           // rotateCorpse
//! rotate(deathAngle, Z)              // rotateCorpse, only while dying
//! scale(-1, -1, 1)                   // doRender — model space is X-flipped and Y-down
//! scale(s, s, s)                     // preRenderCallback — per-renderer size (slimes, babies)
//! translate(0, -1.5078125, 0)        // doRender — lift the model's feet onto the ground
//! ```
//!
//! then draws parts at `scale = 0.0625`. [`entity_matrix`] is that sequence, with the
//! translation coming from [`crate::Camera::relative`] so geometry stays camera-relative
//! (see [`crate::Camera`] — the view matrix is rotation-only, and world coordinates reach
//! 30 million blocks, far past `f32`).
//!
//! The `-1.5078125` is not a rounded `-1.5`: it is `-1.5078125` exactly, which is
//! `-24.125 / 16`. Rounding it puts every mob an eighth of a pixel into the floor.
//!
//! # The phase
//!
//! [`EntityPhase`] is a [`RenderPhase`] registered at [`PhaseOrder::ENTITIES`]. Unlike terrain
//! it holds no retained mesh: the game loop calls [`EntityPhase::begin`], then
//! [`EntityPhase::push`] once per entity with that entity's transformed vertices and its
//! texture key, then [`EntityPhase::finish`]. Consecutive pushes sharing a texture merge into
//! one draw. A player is 216 vertices, so re-walking the part tree every frame is far cheaper
//! than the per-part uniform updates a retained approach would need.

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;

use glam::{Mat4, Vec3};
use mc_assets::{AtlasAnimationUpload, TextureMetadata};
use mc_data::EntityType;

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, RenderPhase};
use crate::sky::lightmap::{Lightmap, LightmapParams};

use super::layers::LayerBlend;
use super::manager::{compose_horse_texture, HorseTextureSelection};
use super::{missing_texture_pixels, EntityVertex, MISSING_TEXTURE_SIZE};

/// Vanilla's model lift, `-24.125 / 16` blocks. java: `RendererLivingEntity.doRender`
pub const MODEL_Y_OFFSET: f32 = -1.5078125;

/// The draw-time scale every living model uses: one model unit is a sixteenth of a block.
pub const MODEL_SCALE: f32 = 0.0625;

/// The packed lightmap coordinate returned by an entity that renders at maximum block and sky
/// brightness: `240 << 16 | 240`.
///
/// This is deliberately the same packed representation as `World.getCombinedLight`, not a
/// normalized colour. `RenderManager.renderEntityStatic` splits its result into the low and high
/// 16-bit lightmap coordinates immediately before invoking the entity renderer.
pub const FULL_BRIGHT_COMBINED_LIGHT: u32 = 0x00F0_00F0;

/// The packed lightmap coordinate forced by `LayerEnderDragonEyes`.
///
/// java: `LayerEnderDragonEyes.doRenderLayer` assigns `i = 61680`, then splits it into
/// `j = i % 65536` and `k = i / 65536` before calling `setLightmapTextureCoords`.  The entity
/// phase stores that pair in the same packed low/high-16-bit representation as its per-draw
/// lightmap uniform, so the exact Java value is the low-half-only coordinate `0x0000_F0F0`,
/// not the ordinary `Entity.getBrightnessForRender` full-bright value `0x00F0_00F0`.
pub const DRAGON_EYES_COMBINED_LIGHT: u32 = 61_680;

/// The resource path bound by `LayerEnderDragonEyes`.
pub const DRAGON_EYES_TEXTURE: &str = "textures/entity/enderdragon/dragon_eyes.png";

/// Sentinel written to the parallel light stream when a batch uses its per-draw lightmap state.
///
/// Packed Java lightmap coordinates only use the low 24 bits (`sky`/`block` bytes are each at
/// most `0xf0`), so `u32::MAX` is not a valid coordinate. Keeping the sentinel out of
/// [`EntityVertex`] preserves the 48-byte ABI used by every other entity emitter while allowing
/// block-model callers to supply a true per-corner lightmap coordinate.
pub const NO_VERTEX_LIGHT: u32 = u32::MAX;

/// Selects the packed lightmap state for a layer draw.
///
/// `LayerEnderDragonEyes` temporarily overrides the active lightmap with Java's literal 61680
/// before replaying the dragon model.  This is a draw-state override, not an entity override:
/// the dragon body, death-texture body, and every unrelated entity retain their sampled world
/// light.  The blend guard also prevents a caller that happens to reuse the eye texture for an
/// opaque draw from inheriting the layer's temporary state.
#[must_use]
pub fn layer_combined_light(texture: &str, blend: LayerBlend, combined_light: u32) -> u32 {
    if texture == DRAGON_EYES_TEXTURE
        && matches!(
            blend,
            LayerBlend::Additive | LayerBlend::AdditiveNoDepthWrite
        )
    {
        DRAGON_EYES_COMBINED_LIGHT
    } else {
        combined_light
    }
}

/// Selects the packed lightmap coordinate an entity supplies to `RenderManager`.
///
/// java: `Entity.getBrightnessForRender` normally delegates to the world light lookup. The
/// wither and magma cube override that base path here with `return 15728880;`
/// (`EntityWither.getBrightnessForRender:589-592`, `EntityMagmaCube.getBrightnessForRender:66-69`),
/// making both lightmap axes `240` even in a completely dark world. Keeping the override at this
/// pure state boundary lets the submission path carry the exact Java packed value when the entity
/// phase consumes lightmaps. The scalar `getBrightness` override on `EntityMagmaCube` is then
/// represented by [`entity_lightmap_brightness`] receiving this selected full-bright coordinate.
#[must_use]
pub const fn entity_combined_light(entity: EntityType, world_combined_light: u32) -> u32 {
    match entity {
        EntityType::WitherBoss | EntityType::LavaSlime => FULL_BRIGHT_COMBINED_LIGHT,
        _ => world_combined_light,
    }
}

/// Collapses Minecraft's packed two-axis lightmap coordinate to the scalar approximation used
/// by this renderer's entity shaders.
///
/// `RenderManager.renderEntityStatic` passes the low and high 16-bit halves straight to
/// `OpenGlHelper.setLightmapTextureCoords`.  The full lightmap texture also depends on time,
/// dimension, gamma, and potion effects, none of which belongs to an entity draw.  Until that
/// shared texture is bound here, use the same level-to-brightness curve as the TESR pass.  The
/// important exact boundary for entity overrides is retained: `0x00F0_00F0` maps to `1.0`.
///
/// The caller must select an entity's coordinate with [`entity_combined_light`] before it calls
/// [`EntityPhase::push_lit`] or [`EntityPhase::push_blended_lit`].
#[must_use]
pub fn entity_lightmap_brightness(combined_light: u32) -> f32 {
    let block = ((combined_light & 0xffff) as u16 / 16).min(15) as u8;
    let sky = ((combined_light >> 16) as u16 / 16).min(15) as u8;
    super::tilepass::light_brightness(block.max(sky))
}

/// Convert a packed Java lightmap coordinate into the normalized texel-center UV used by the
/// entity shader.
///
/// `OpenGlHelper.setLightmapTextureCoords` receives the low 16-bit block axis as X and the high
/// 16-bit sky axis as Y. The `+8` offset addresses the centre of the corresponding 16x16 texel;
/// keeping this pure helper beside the entity shader gives mixed-channel controls a canonical
/// axis-order oracle without requiring a GPU adapter.
#[must_use]
pub fn entity_packed_lightmap_uv(combined_light: u32) -> [f32; 2] {
    let block = (combined_light & 0xff) as f32;
    let sky = ((combined_light >> 16) & 0xff) as f32;
    [(block + 8.0) / 256.0, (sky + 8.0) / 256.0]
}

/// `RenderWither.preRenderCallback`'s uniform scale.
///
/// The ordinary wither is twofold. During its 220-tick spawn invulnerability, Java samples the
/// watched remaining tick count at the frame's partial tick and shrinks it towards 1.5.
#[must_use]
pub fn wither_pre_render_scale(invul_time: i32, partial_ticks: f32) -> f32 {
    let mut scale = 2.0;
    if invul_time > 0 {
        scale -= ((invul_time as f32 - partial_ticks) / 220.0) * 0.5;
    }
    scale
}

/// How an entity is placed in the world for this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EntityTransform {
    /// Camera-relative position of the entity's feet, in blocks.
    pub rel: Vec3,
    /// Interpolated body yaw in degrees. java: `renderYawOffset`
    pub body_yaw: f32,
    /// Death spin about Z in degrees, from [`death_rotation`]. Zero while alive.
    pub death_rotation: f32,
    /// Additional Y rotation applied after the death spin.
    ///
    /// `RenderPlayer.rotateCorpse` has one player-only branch for a live sleeping player:
    /// `bedOrientation(Y)`, `getDeathMaxRotation(Z)`, then `270(Y)`. The ordinary living
    /// transform leaves this at zero, so the field is a neutral renderer extension.
    pub post_yaw: f32,
    /// Uniform scale from `preRenderCallback` — babies, giants, the wither skeleton.
    ///
    /// The **non-uniform** `preRenderCallback` overrides (`RenderSlime`, `RenderMagmaCube`) do
    /// not fit here at all; they go through [`entity_matrix_scaled`].
    pub scale: f32,
    /// Extra Y lift applied *after* the flip, in blocks. Almost always [`MODEL_Y_OFFSET`].
    pub y_offset: f32,
    /// Whether `RendererLivingEntity.rotateCorpse` applies the Dinnerbone/Grumm inversion.
    ///
    /// Java translates by `height + 0.1F` and then rotates 180 degrees around Z before the
    /// renderer's ordinary `scale(-1,-1,1)`. Keeping this as a transform input (rather than a
    /// model mutation) makes every living layer share the same upside-down body frame.
    pub upside_down: bool,
    /// Live `Entity.height` used by the Dinnerbone/Grumm translation. Ignored when
    /// [`Self::upside_down`] is false.
    pub upside_down_height: f32,
}

impl Default for EntityTransform {
    fn default() -> Self {
        Self {
            rel: Vec3::ZERO,
            body_yaw: 0.0,
            death_rotation: 0.0,
            post_yaw: 0.0,
            scale: 1.0,
            y_offset: MODEL_Y_OFFSET,
            upside_down: false,
            upside_down_height: 0.0,
        }
    }
}

/// Builds the matrix a living entity's model is drawn under.
///
/// java: `RendererLivingEntity.doRender` — `renderLivingAt`, `rotateCorpse`, the
/// `scale(-1, -1, 1)` flip, `preRenderCallback`, then `translate(0, -1.5078125, 0)`.
#[must_use]
pub fn entity_matrix(t: &EntityTransform) -> Mat4 {
    entity_matrix_scaled(t, Vec3::ONE)
}

/// [`entity_matrix`] with the **non-uniform** half of `preRenderCallback`.
///
/// java: `RendererLivingEntity.prepareScale` (`:230-236`) is
/// `scale(-1,-1,1)` → `preRenderCallback(entity, partialTicks)` → `translate(0, -1.5078125, 0)`,
/// so a `preRenderCallback` scale is composed **before** the model lift and the lift scales with
/// it. Post-multiplying the finished matrix instead leaves the model's feet off the ground by
/// `1.5078125 * (scale - 1)` blocks — 4.5 blocks for a size-4 slime — which is why this is a
/// second entry point rather than something a caller can bolt on afterwards.
///
/// Two of 1.8.9's overrides need it, and they are the same three lines:
/// [`slime_pre_render_scale`] serves both. `EntityRenderer::scale` carries the uniform ones and
/// is `1.0` for the slime and the magma cube precisely because theirs is not uniform.
///
/// `Vec3::ONE` is exactly [`entity_matrix`].
#[must_use]
pub fn entity_matrix_scaled(t: &EntityTransform, pre_render_scale: Vec3) -> Mat4 {
    let mut m = Mat4::from_translation(t.rel);
    m *= Mat4::from_rotation_y((180.0 - t.body_yaw).to_radians());
    if t.death_rotation != 0.0 {
        m *= Mat4::from_rotation_z(t.death_rotation.to_radians());
    }
    if t.post_yaw != 0.0 {
        m *= Mat4::from_rotation_y(t.post_yaw.to_radians());
    }
    if t.upside_down {
        // java: `RendererLivingEntity.rotateCorpse`'s non-death Dinnerbone/Grumm branch.
        m *= Mat4::from_translation(Vec3::new(0.0, t.upside_down_height + 0.1, 0.0));
        m *= Mat4::from_rotation_z(core::f32::consts::PI);
    }
    m *= Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0));
    if t.scale != 1.0 {
        m *= Mat4::from_scale(Vec3::splat(t.scale));
    }
    if pre_render_scale != Vec3::ONE {
        m *= Mat4::from_scale(pre_render_scale);
    }
    m *= Mat4::from_translation(Vec3::new(0.0, t.y_offset, 0.0));
    m
}

/// `RenderSlime.preRenderCallback` / `RenderMagmaCube.preRenderCallback` — the squash-and-
/// stretch scale, which is the same three lines in both classes.
///
/// java: `RenderSlime.java:29-35` and `RenderMagmaCube.java:29-37`:
/// ```text
/// f  = (float)getSlimeSize();
/// f1 = (prevSquishFactor + (squishFactor - prevSquishFactor) * partialTicks) / (f * 0.5F + 1.0F);
/// f2 = 1.0F / (f1 + 1.0F);
/// scale(f2 * f, 1.0F / f2 * f, f2 * f);
/// ```
/// `squish` is already the interpolated factor — see
/// [`EntityAnimState::squish_factor`](super::anim::EntityAnimState::squish_factor), which is a
/// single value because vanilla keeps `prevSquishFactor` equal to `squishFactor`.
///
/// Two things this is load-bearing for beyond the bounce. **`size` scales the whole model**, so a
/// size-4 magma cube drawn without this is a quarter of its correct width inside its correct
/// hitbox; and the Y factor is `f / f2` against the XZ factor's `f2 * f`, so a squashing slime
/// spreads sideways rather than just getting shorter.
///
/// It is **not** volume-preserving, and two audit passes have tried to "fix" it into being so.
/// The product is `(f2 * f) * (f / f2) * (f2 * f) = f2 * f^3`, which varies with the squish;
/// only the X/Y *side profile* is conserved, because the reciprocal pairs the Y factor with one
/// of the two horizontal axes and the other horizontal axis is left over. The arithmetic above
/// is a faithful transcription of `RenderSlime.java:29-35` — change the prose, never the maths.
///
/// `size` is `getSlimeSize()`, which mc-rs decodes into
/// [`EntityAnimState::variant`](super::anim::EntityAnimState::variant) for `Slime` and
/// `LavaSlime`. A `size` of zero (an unset watched value) collapses the model to nothing, so it
/// is floored at one — vanilla's `entityInit` seeds index 16 with `1`.
#[must_use]
pub fn slime_pre_render_scale(size: i32, squish: f32) -> Vec3 {
    let f = size.max(1) as f32;
    let f1 = squish / (f * 0.5 + 1.0);
    let f2 = 1.0 / (f1 + 1.0);
    Vec3::new(f2 * f, 1.0 / f2 * f, f2 * f)
}

/// The death spin, in degrees about Z.
///
/// java: `RendererLivingEntity.rotateCorpse` — `sqrt((deathTime + partial - 1) / 20 * 1.6)`,
/// clamped to 1, times `getDeathMaxRotation` (90 for everything but the guardian and the
/// ender dragon).
#[must_use]
pub fn death_rotation(death_time: u32, partial_ticks: f32, max_rotation: f32) -> f32 {
    if death_time == 0 {
        return 0.0;
    }
    let f = (death_time as f32 + partial_ticks - 1.0) / 20.0 * 1.6;
    let f = mc_math::sqrt_f(f).min(1.0);
    f * max_rotation
}

/// The vertex layout [`EntityVertex`] is fed to the pipeline with.
pub const ENTITY_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<EntityVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 0,
            shader_location: 0,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x2,
            offset: 12,
            shader_location: 1,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x3,
            offset: 20,
            shader_location: 2,
        },
        // The primary-colour channel. See [`super::EntityVertex::color`].
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 32,
            shader_location: 3,
        },
    ],
};

/// The packed lightmap stream paired with [`ENTITY_VERTEX_LAYOUT`].
///
/// This is intentionally a second buffer rather than a field added to [`EntityVertex`]: the
/// latter is shared by model, item, line, outline, and debug emitters, while only a few world-model
/// callers need Java's per-vertex sky/block coordinates.  Ordinary batches receive
/// [`NO_VERTEX_LIGHT`] and continue through their per-draw lightmap uniform.
pub const ENTITY_VERTEX_LIGHT_LAYOUT: wgpu::VertexBufferLayout<'static> =
    wgpu::VertexBufferLayout {
        array_stride: core::mem::size_of::<u32>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 0,
            shader_location: 4,
        }],
    };

const ENTITY_WGSL: &str = r#"
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

struct LightmapUniform {
    // A uniform-buffer struct rounds its size up to its largest member alignment.  A scalar
    // followed by `vec3<u32>` therefore has a 32-byte WGSL layout (the vec3 starts at offset
    // 16), while the Rust record below is deliberately 16 bytes.  Store the packed coordinate
    // in one lane of a vec4 so both sides are exactly one 16-byte uniform record.
    combined_light: vec4<u32>,
};
@group(2) @binding(0) var<uniform> lightmap: LightmapUniform;
@group(2) @binding(1) var lightmap_tex: texture_2d<f32>;
@group(2) @binding(2) var lightmap_samp: sampler;

struct VsOut {
    // Glint/overlay pipelines replay this position with an exact depth comparison.
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    // xy is a normalized Java lightmap coordinate; z selects the per-vertex stream when 1.
    // Keeping xy as a floating-point varying gives adjacent corners Java's raster interpolation
    // instead of selecting one corner's scalar brightness for the whole quad.
    @location(3) lightmap: vec3<f32>,
};

fn packed_lightmap_uv(packed: u32) -> vec2<f32> {
    let block = f32(packed & 0xffu);
    let sky = f32((packed >> 16u) & 0xffu);
    // EntityRenderer's texture matrix is scale(1/256), translate(8, 8). The texture's first
    // coordinate is the low (block) lightmap axis and the second is the high (sky) axis.
    return (vec2<f32>(block, sky) + vec2<f32>(8.0, 8.0)) / 256.0;
}

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) vertex_light: u32,
) -> VsOut {
    var out: VsOut;
    // `pos` is already CAMERA-RELATIVE: the CPU walked the part tree under a matrix whose
    // translation came from Camera::relative. view_proj's view half is rotation-only.
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.normal = normal;
    out.color = color;
    let has_vertex_light = vertex_light != 0xffffffffu;
    let packed = select(lightmap.combined_light.x, vertex_light, has_vertex_light);
    out.lightmap = vec3<f32>(packed_lightmap_uv(packed), select(0.0, 1.0, has_vertex_light));
    return out;
}

// java: RenderHelper.enableStandardItemLighting — two directional lights at 0.6 diffuse over a
// 0.4 ambient floor. Vanilla's light vectors are (0.2, 1.0, -0.7) and (-0.2, 1.0, 0.7),
// normalised, and the fixed-function pipeline clamps each dot to >= 0.
const LIGHT0 = vec3<f32>(0.2, 1.0, -0.7);
const LIGHT1 = vec3<f32>(-0.2, 1.0, 0.7);

fn diffuse(normal: vec3<f32>) -> f32 {
    let n = normalize(normal);
    var light = 0.4;
    light = light + 0.6 * max(dot(n, normalize(LIGHT0)), 0.0);
    light = light + 0.6 * max(dot(n, normalize(LIGHT1)), 0.0);
    return light;
}

// Java samples the generated 16x16 lightmap at the interpolated sky/block coordinates. The
// per-draw uniform remains the fallback for ordinary entity geometry; falling-block model quads
// select their exact packed corner stream in the vertex stage and use the same interpolation path.
fn lightmap_color(lightmap_coord: vec3<f32>) -> vec3<f32> {
    return textureSample(lightmap_tex, lightmap_samp, lightmap_coord.xy).rgb;
}

// A02-07: the primary colour. java: GlStateManager.color plus GL_COLOR_MATERIAL's
// AMBIENT_AND_DIFFUSE mode. Fixed-function lighting clamps the lit primary colour before
// the default GL_MODULATE texture environment applies the sampled texel. Clamping the
// scalar light first is observably darker for tinted upward faces, where the two standard
// item lights add to more than 1.0.
//
// The lightmap is texture unit 1 (`EntityRenderer.enableLightmap`), a second GL_MODULATE
// stage after the texel, so it multiplies the already-clamped colour. Inside the clamp, an
// upward face (light 1.37) under a night lightmap came out 37% too bright.
fn lit_entity(in: VsOut, texel: vec4<f32>) -> vec4<f32> {
    let lit_color = clamp(diffuse(in.normal) * in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(
        texel.rgb * lit_color * lightmap_color(in.lightmap),
        texel.a * in.color.a,
    );
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // java: GlStateManager.alphaFunc(516, 0.1) — GL_GREATER, ref 0.1. Fixed-function GL tests
    // the combined fragment alpha; this pipeline's draws are opaque-tinted (`in.color.a == 1`),
    // so the texel's alpha is that value. The blended families with a real vertex alpha test
    // the product in `fs_translucent` and `fs_ghost`.
    // GL_GREATER rejects equality too: a texel alpha of exactly 0.1 is not visible.
    if (texel.a <= 0.1) {
        discard;
    }
    return lit_entity(in, texel);
}

// java: the same `alphaFunc(516, 0.1)`, against the combined `texel * color` alpha, for the
// blended families whose vertex alpha is real: `Render.renderShadow` (its per-cell alpha fades
// with distance, height and light, so a cell at or under 0.1 is discarded outright),
// `LayerSlimeGel` and `MobAppearance`'s pulsing apparition.
@fragment
fn fs_translucent(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a * in.color.a <= 0.1) {
        discard;
    }
    return lit_entity(in, texel);
}

// java: `RendererLivingEntity.renderModel` (`:246-278`) lowers the reference to
// `alphaFunc(516, 0.003921569)` for the invisible-teammate ghost drawn at
// `color(1, 1, 1, 0.15)`, still against the combined alpha.
@fragment
fn fs_ghost(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a * in.color.a <= 0.003921569) {
        discard;
    }
    return lit_entity(in, texel);
}

// java: `RendererLivingEntity.setBrightness` configures the lightmap texture unit as
// `INTERPOLATE(CONSTANT, PREVIOUS, CONSTANT_ALPHA)`, then modulates that result by the ordinary
// lightmap. Hurt/death uses the exact constant `(1, 0, 0, .3)`. This must happen in the textured
// source draw: an untextured replay paints transparent texels and cannot preserve a blended slime
// shell.
//
// The alpha is the texel's alone: all three combiner stages set `GL_COMBINE_ALPHA` to
// `GL_REPLACE` (unit 0 from `GL_TEXTURE0`, units 1 and 2 from `GL_PREVIOUS`), so the primary
// colour's alpha never reaches the fragment. A hurt invisible teammate (drawn at
// `color(1, 1, 1, 0.15)`) therefore flashes at the skin's own opacity in vanilla, not as a 15%
// ghost (H7 2026-11-11).
fn living_hurt_lit(in: VsOut, texel: vec4<f32>) -> vec4<f32> {
    let primary = clamp(
        diffuse(in.normal) * in.color.rgb,
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
    let affine = mix(texel.rgb * primary, vec3<f32>(1.0, 0.0, 0.0), 0.3);
    return vec4<f32>(affine * lightmap_color(in.lightmap), texel.a);
}

@fragment
fn fs_living_hurt(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a <= 0.1) {
        discard;
    }
    return living_hurt_lit(in, texel);
}

@fragment
fn fs_living_hurt_ghost(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // java: `renderModel` temporarily uses alphaFunc(GL_GREATER, 1/255) for a visible
    // invisible teammate, then restores the ordinary 0.1 cutout.
    if (texel.a <= 0.003921569) {
        discard;
    }
    return living_hurt_lit(in, texel);
}

@fragment
fn fs_living_hurt_glint(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    let affine = mix(
        texel.rgb * in.color.rgb,
        vec3<f32>(1.0, 0.0, 0.0),
        0.3,
    );
    return vec4<f32>(affine * lightmap_color(in.lightmap), texel.a * in.color.a);
}

// Creeper's colour multiplier is applied to the textured living-model draw in Java. The Rust
// phase retains its equivalent-depth replay, so sample solely to preserve the model cutout and
// source alpha, then blend lightmapped white over the already-rendered body by the multiplier
// alpha carried in `in.color.a`.
@fragment
fn fs_living_white_flash(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a <= 0.1) {
        discard;
    }
    return vec4<f32>(lightmap_color(in.lightmap), texel.a * in.color.a);
}

// java: the additive layers — LayerCreeperCharge (`:38`) and LayerWitherAura (`:45`) call
// `disableLighting()`, and LayerEndermanEyes/LayerSpiderEyes call `disableAlpha()` and force the
// lightmap to full bright (61680). This phase has no lightmap input at all, so "unlit" is the
// nearer of the two available approximations to vanilla's full-bright eyes, and it is exactly
// right for the two aura layers.
@fragment
fn fs_unlit(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    return vec4<f32>(
        texel.rgb * in.color.rgb * lightmap_color(in.lightmap),
        texel.a * in.color.a,
    );
}

// java: Render.renderEntityOnFire disables lighting but does *not* enable blending or disable
// alpha testing.  It therefore needs this exact combination rather than the additive unlit
// state used by glowing eye/aura layers.
@fragment
fn fs_unlit_cutout(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a <= 0.1) {
        discard;
    }
    return vec4<f32>(
        texel.rgb * in.color.rgb * lightmap_color(in.lightmap),
        texel.a * in.color.a,
    );
}

// java: `RenderItem.renderEffect` / `LayerArmorBase.renderGlint` disable lighting before
// replaying the model, so the entity's two directional lights do not touch the foil.
// `disableLighting()` is GL_LIGHTING only: the lightmap on texture unit 1, which
// `RenderGlobal.renderEntities` enables for the whole entity and tile-entity pass (`:581-740`),
// still multiplies the foil, and the `GL_SRC_COLOR, GL_ONE` blend then squares it. So at night
// the foil is far dimmer than by day (measured on fixed-function GL: lightmap 77/255 turns
// (37, 9, 94) into (3, 1, 9) over black).
// Alpha testing is intentionally absent: the glint texture's own alpha is consumed by the
// `GL_SRC_COLOR, GL_ONE` blend state, just as the fixed-function texture environment is.
@fragment
fn fs_glint(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    return vec4<f32>(
        texel.rgb * in.color.rgb * lightmap_color(in.lightmap),
        texel.a * in.color.a,
    );
}

// java: RenderTNTPrimed's flashing replay disables texture unit 0 and lighting. Its
// `color(1, 1, 1, f2)` never reaches the fragment: `renderBlockBrightness` draws
// `DefaultVertexFormats.ITEM` quads after `putColorRGB_F4(1, 1, 1)`, and that colour array (alpha
// 255) replaces the current colour. The lightmap on unit 1 is still enabled, so the fragment is
// the vertex colour times the entity's lightmap texel; `GL_SRC_ALPHA, GL_DST_ALPHA` then adds it
// to the opaque body, checked against a Java reference run of vanilla's pass.
@fragment
fn fs_tnt_flash(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb * lightmap_color(in.lightmap), in.color.a);
}

// Java's leash and fishing-line paths disable both texture and lighting before submitting
// POSITION_COLOR vertices. Keep their fragment path texture-independent rather than sampling a
// fallback image (which would make omitted-layer mutants look like a magenta rope).
@fragment
fn fs_untextured(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}

// java: RenderDragon.renderModel's exploding replay binds dragon_exploding.png and sets
// alphaFunc(GL_GREATER, deathTicks / 200).  The CPU puts that threshold in the primary alpha
// channel for this dedicated pass; do not multiply it into the output alpha because vanilla's
// alpha test is a discard gate, not a fade.
@fragment
fn fs_dragon_death_body(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a <= in.color.a) {
        discard;
    }
    // Same order as `fs`: clamp the lit primary colour, then the texel and lightmap stages.
    let lit_color = clamp(diffuse(in.normal) * in.color.rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(texel.rgb * lit_color * lightmap_color(in.lightmap), texel.a);
}

// java: LayerEnderDragonDeath disables texture unit 0 and lighting and supplies a per-vertex white
// or magenta colour. The lightmap on unit 1 stays enabled at the dragon's own light (the eyes
// layer's `setLightmap` restored it), so it still multiplies that colour; the additive blend
// state belongs to the layer.
@fragment
fn fs_dragon_death_shards(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color.rgb * lightmap_color(in.lightmap), in.color.a);
}
"#;

/// Minimal POSITION_COLOR shader used by both F3+B line geometry and the spectator outline
/// mask. Keeping it separate from the textured entity shader means the debug path does not bind
/// a dummy texture/lightmap and can use a line-list topology without changing normal batches.
const DEBUG_GEOMETRY_WGSL: &str = r#"
struct CameraUniform {
    view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen: vec4<f32>,
    frame: vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// Fullscreen composite for the blurred spectator outline target.
const OUTLINE_COMPOSITE_WGSL: &str = r#"
@group(0) @binding(0) var image: texture_2d<f32>;

@vertex
fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(image, vec2<i32>(pos.xy), 0);
}
"#;

/// One debug line in camera-relative coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugLine {
    /// Start point.
    pub start: Vec3,
    /// End point.
    pub end: Vec3,
    /// RGBA colour.
    pub color: [f32; 4],
}

/// One vertex in vanilla's debug hitbox strips. Keeping the strips as vertices (rather than
/// eagerly flattening them to independent line-list pairs) preserves the Java vertex-count
/// oracle: non-living entities contribute 18 AABB vertices plus the two look-ray vertices, while
/// living entities add the 18-vertex eye-height strip.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DebugVertex {
    /// Position in camera-relative coordinates.
    pub pos: Vec3,
    /// RGBA colour.
    pub color: [f32; 4],
}

fn append_debug_box(vertices: &mut Vec<DebugVertex>, min: Vec3, max: Vec3, color: [f32; 4]) {
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    // RenderGlobal.drawOutlinedBoundingBox emits two five-vertex line strips followed by four
    // independent vertical pairs. Keep those three strips contiguous for the line-list adapter
    // in EntityPhase while retaining the exact 18 source vertices from Java.
    for &index in &[0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7] {
        vertices.push(DebugVertex {
            pos: corners[index],
            color,
        });
    }
}

/// Build the exact Java F3+B vertex streams for one entity.
///
/// The AABB is always present. The red eye-height box is emitted only for living entities, while
/// the blue two-block look ray is emitted for both living and non-living entities. `position` is
/// the interpolated feet position and `look` is the frame-interpolated view direction.
#[must_use]
// The epsilon is the narrowed Java literal used by RenderGlobal's debug box.
#[allow(clippy::excessive_precision)]
pub fn debug_entity_vertices(
    position: Vec3,
    width: f32,
    height: f32,
    eye_height: f32,
    look: Vec3,
    living: bool,
) -> Vec<DebugVertex> {
    let half = width * 0.5;
    let min = Vec3::new(position.x - half, position.y, position.z - half);
    let max = Vec3::new(position.x + half, position.y + height, position.z + half);
    let mut vertices = Vec::with_capacity(if living { 38 } else { 20 });
    append_debug_box(&mut vertices, min, max, [1.0, 1.0, 1.0, 1.0]);
    if living {
        let epsilon = 0.009_999_999_776_482_582;
        let eye_min = position.y + eye_height - epsilon;
        let eye_max = position.y + eye_height + epsilon;
        append_debug_box(
            &mut vertices,
            Vec3::new(position.x - half, eye_min, position.z - half),
            Vec3::new(position.x + half, eye_max, position.z + half),
            [1.0, 0.0, 0.0, 1.0],
        );
    }
    let look = if look.length_squared() > f32::EPSILON {
        look.normalize()
    } else {
        Vec3::ZERO
    };
    let eye = Vec3::new(position.x, position.y + eye_height, position.z);
    vertices.extend([
        DebugVertex {
            pos: eye,
            color: [0.0, 0.0, 1.0, 1.0],
        },
        DebugVertex {
            pos: eye + look * 2.0,
            color: [0.0, 0.0, 1.0, 1.0],
        },
    ]);
    vertices
}

/// Build Java's F3+B geometry for one entity: 12 white AABB edges, a red eye-height slab, and a
/// blue two-block look ray. `position` is the entity's interpolated feet position and `look` is
/// the frame-interpolated view direction (normalised here defensively).
#[must_use]
// Preserve the narrowed Java epsilon rather than replacing it with a rounded Rust literal.
#[allow(clippy::excessive_precision)]
pub fn debug_entity_geometry(
    position: Vec3,
    width: f32,
    height: f32,
    eye_height: f32,
    look: Vec3,
) -> Vec<DebugLine> {
    let half = width * 0.5;
    let min = Vec3::new(position.x - half, position.y, position.z - half);
    let max = Vec3::new(position.x + half, position.y + height, position.z + half);
    let white = [1.0, 1.0, 1.0, 1.0];
    let red = [1.0, 0.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let corners = [
        Vec3::new(min.x, min.y, min.z),
        Vec3::new(max.x, min.y, min.z),
        Vec3::new(max.x, min.y, max.z),
        Vec3::new(min.x, min.y, max.z),
        Vec3::new(min.x, max.y, min.z),
        Vec3::new(max.x, max.y, min.z),
        Vec3::new(max.x, max.y, max.z),
        Vec3::new(min.x, max.y, max.z),
    ];
    let mut lines = Vec::with_capacity(21);
    // Bottom, top, then the four vertical edges: the same twelve segments produced by
    // `RenderGlobal.drawOutlinedBoundingBox`.
    for &(a, b) in &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        lines.push(DebugLine {
            start: corners[a],
            end: corners[b],
            color: white,
        });
    }

    let eye_min = position.y + eye_height - 0.009_999_999_776_482_582;
    let eye_max = position.y + eye_height + 0.009_999_999_776_482_582;
    let eye_corners = [
        Vec3::new(position.x - half, eye_min, position.z - half),
        Vec3::new(position.x + half, eye_min, position.z - half),
        Vec3::new(position.x + half, eye_min, position.z + half),
        Vec3::new(position.x - half, eye_min, position.z + half),
        Vec3::new(position.x - half, eye_max, position.z - half),
        Vec3::new(position.x + half, eye_max, position.z - half),
        Vec3::new(position.x + half, eye_max, position.z + half),
        Vec3::new(position.x - half, eye_max, position.z + half),
    ];
    for &(a, b) in &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
    ] {
        lines.push(DebugLine {
            start: eye_corners[a],
            end: eye_corners[b],
            color: red,
        });
    }
    let eye = Vec3::new(position.x, position.y + eye_height, position.z);
    let look = if look.length_squared() > f32::EPSILON {
        look.normalize()
    } else {
        Vec3::ZERO
    };
    lines.push(DebugLine {
        start: eye,
        end: eye + look * 2.0,
        color: blue,
    });
    lines
}

/// One uploaded entity texture.
struct TextureEntry {
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Raw decoded bytes retained so a live anaglyph toggle can rebuild the GPU copy without
    /// applying the integer channel transform to already-converted pixels.
    source_rgba: RefCell<Vec<u8>>,
    /// Sampler metadata that accompanied the raw source.
    options: TextureMetadata,
}

/// The storage/sampling format for ordinary entity PNGs.
///
/// java: `TextureUtil.uploadTextureImageAllocate` allocates level zero with plain `GL_RGBA`.
/// That is an unconverted byte-normalized texture: a stored RGB byte of 128 is sampled as
/// `128/255`, not passed through an sRGB-to-linear transfer.  Keep this format local to the
/// entity phase; block/item atlases, lightmaps, and UI textures have separate owners and are
/// deliberately not changed by the SimpleTexture parity fix.
const ENTITY_TEXTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

fn atlas_animation_rect_valid(upload: &AtlasAnimationUpload, width: u32, height: u32) -> bool {
    upload.mip_level == 0
        && upload.width > 0
        && upload.height > 0
        && upload.rgba.len() == (u64::from(upload.width) * u64::from(upload.height) * 4) as usize
        && upload
            .x
            .checked_add(upload.width)
            .is_some_and(|end| end <= width)
        && upload
            .y
            .checked_add(upload.height)
            .is_some_and(|end| end <= height)
}

/// An interned entity texture key: an index into [`TextureInterner::keys`].
///
/// 06-entity-batch U1: batches used to own a `String` copy of their `ResourceLocation` (one heap
/// allocation per new batch) and `record` re-hashed that string to find the bind group. Ids are
/// assigned once per distinct key for the life of the phase and never reused, so an id recorded
/// in a batch stays meaningful across same-key re-uploads and resource reloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TextureId(u32);

impl TextureId {
    const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Process-stable texture key table. Grows by one entry per distinct key ever pushed or
/// uploaded (the same bound as the resident texture map itself, which never evicts either).
#[derive(Debug, Default)]
struct TextureInterner {
    ids: HashMap<String, TextureId>,
    keys: Vec<String>,
}

impl TextureInterner {
    fn get(&self, key: &str) -> Option<TextureId> {
        self.ids.get(key).copied()
    }

    fn intern(&mut self, key: &str) -> TextureId {
        if let Some(id) = self.get(key) {
            return id;
        }
        let id = TextureId(
            u32::try_from(self.keys.len()).expect("fewer than 2^32 distinct entity textures"),
        );
        self.keys.push(key.to_owned());
        self.ids.insert(key.to_owned(), id);
        id
    }

    fn key(&self, id: TextureId) -> &str {
        &self.keys[id.index()]
    }
}

/// A contiguous run of vertices sharing one texture *and* one blend mode.
mod indexed_experiment;
// Shader packs: entities casting sun shadows. A child module so it can read this file's private batches and texture slots.
mod pack_shadow;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DrawBatch {
    texture: TextureId,
    /// Which of [`EntityPhase`]'s pipelines draws this run.
    blend: LayerBlend,
    /// The packed state last supplied to `OpenGlHelper.setLightmapTextureCoords` for this draw.
    ///
    /// 06-entity-batch U1: this is no longer draw state. Every vertex now carries its own packed
    /// coordinate in the parallel light stream (see [`extend_vertex_lights`]), so the shader's
    /// per-batch uniform fallback is never selected and two adjacent pushes that differ only in
    /// light share one draw with bit-identical fragments. The value is kept solely so the two
    /// strip topologies keep their exact pre-U1 merge rule (see [`batches_can_merge`]).
    combined_light: u32,
    start: u32,
    end: u32,
    /// Experimental compact quad storage; logical triangle order remains 0,1,2,0,2,3.
    indexed: bool,
}

/// One 16-byte lightmap uniform record.  Dynamic uniform offsets add the device-required
/// 256-byte stride between records; keeping the source record compact avoids pretending that
/// padding is semantic renderer state.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LightmapUniform {
    combined_light: u32,
    _padding: [u32; 3],
}

const LIGHTMAP_UNIFORM_STRIDE: u64 = 256;

/// Whether a push may extend the previous batch instead of starting a new draw.
///
/// 06-entity-batch U1: light is no longer part of the identity for triangle-list pipelines,
/// because the per-vertex stream carries it (pixel-identical: the vertex stage emits the same
/// `packed_lightmap_uv(packed)` value either way). The two strip topologies keep the pre-U1 rule
/// exactly: concatenating two independent strips adds bridging primitives, so a strip push must
/// not newly merge with a neighbour that it previously stayed apart from.
fn batches_can_merge(
    batch: &DrawBatch,
    texture: TextureId,
    blend: LayerBlend,
    combined_light: u32,
    start: u32,
) -> bool {
    batch.texture == texture
        && batch.blend == blend
        && (!blend_is_strip(blend) || batch.combined_light == combined_light)
        && batch.end == start
}

/// The pipelines whose primitive topology is a strip (see `EntityPhase::new`'s topology match).
const fn blend_is_strip(blend: LayerBlend) -> bool {
    matches!(blend, LayerBlend::FishingLine | LayerBlend::Leash)
}

/// Append one push's packed lightmap coordinates to the per-vertex light stream.
///
/// 06-entity-batch U1: ordinary pushes used to write [`NO_VERTEX_LIGHT`] and let the vertex
/// shader select the batch's dynamic uniform. They now write the batch coordinate itself, and an
/// explicit per-vertex stream keeps its values except that a sentinel entry is replaced by the
/// batch coordinate the shader would previously have selected for it. Either way the stream
/// never contains the sentinel, so the shader's `select` yields exactly the old value per vertex.
fn extend_vertex_lights(
    stream: &mut Vec<u32>,
    combined_light: u32,
    count: usize,
    packed_lights: Option<&[u32]>,
) {
    match packed_lights {
        Some(lights) => stream.extend(lights.iter().map(|&light| {
            if light == NO_VERTEX_LIGHT {
                combined_light
            } else {
                light
            }
        })),
        None => stream.extend(std::iter::repeat_n(combined_light, count)),
    }
}

/// Append one push to the frame's vertex, light and batch streams.
///
/// `separate` forces a new batch even when [`batches_can_merge`] would allow a merge (the second
/// leash strip). Kept free of `EntityPhase` so the batching rules are testable without a GPU.
#[allow(clippy::too_many_arguments)]
fn append_draw(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    batches: &mut Vec<DrawBatch>,
    texture: TextureId,
    blend: LayerBlend,
    combined_light: u32,
    new_verts: &[EntityVertex],
    packed_lights: Option<&[u32]>,
    separate: bool,
) {
    let start = verts.len() as u32;
    verts.extend_from_slice(new_verts);
    extend_vertex_lights(
        vertex_lights,
        combined_light,
        new_verts.len(),
        packed_lights,
    );
    let end = verts.len() as u32;
    match batches.last_mut() {
        Some(b)
            if !separate
                && !b.indexed
                && batches_can_merge(b, texture, blend, combined_light, start) =>
        {
            b.end = end;
        }
        _ => batches.push(DrawBatch {
            texture,
            blend,
            combined_light,
            start,
            end,
            indexed: false,
        }),
    }
}

// Compact quad-shaped pushes as they
// arrive instead of re-reading and re-writing the whole frame stream in `finish`.

/// Kill switch for [`append_draw_fused`]: `MC_NW_F1_FUSED_COMPACT=0` restores the original
/// append-raw-then-compact-in-`finish` path. Read once per process.
pub const FUSED_COMPACT_ENV: &str = "MC_NW_F1_FUSED_COMPACT";

fn fused_compact_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os(FUSED_COMPACT_ENV).is_none_or(|value| value != "0"))
}

/// Bitwise vertex identity, as `indexed_experiment`'s quad test uses it: float equality would
/// collapse signed zero and reject identical NaN payloads.
#[inline]
fn same_vertex(a: &EntityVertex, b: &EntityVertex) -> bool {
    bytemuck::bytes_of(a) == bytemuck::bytes_of(b)
}

/// The light [`extend_vertex_lights`] would store for vertex `index` of a push.
#[inline]
fn pushed_light(packed_lights: Option<&[u32]>, combined_light: u32, index: usize) -> u32 {
    match packed_lights {
        Some(lights) if lights[index] != NO_VERTEX_LIGHT => lights[index],
        _ => combined_light,
    }
}

/// Append `new_verts` as indexed quads (`0,1,2,5` of every six) when every six-vertex group is the
/// `0,1,2,0,2,3` quad pattern bit for bit, lights included: exactly the per-group test
/// `indexed_experiment::compact` applies in `finish`. Returns `false` and leaves both streams as
/// they were when any group fails.
fn try_append_compact(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    new_verts: &[EntityVertex],
    packed_lights: Option<&[u32]>,
    combined_light: u32,
) -> bool {
    let base = verts.len();
    let quads = new_verts.len() / 6;
    verts.reserve(quads * 4);
    vertex_lights.reserve(quads * 4);
    for (quad, group) in new_verts.chunks_exact(6).enumerate() {
        let at = quad * 6;
        let l: [u32; 6] =
            core::array::from_fn(|i| pushed_light(packed_lights, combined_light, at + i));
        if !(same_vertex(&group[0], &group[3])
            && same_vertex(&group[2], &group[4])
            && l[0] == l[3]
            && l[2] == l[4])
        {
            verts.truncate(base);
            vertex_lights.truncate(base);
            return false;
        }
        verts.extend_from_slice(&[group[0], group[1], group[2], group[5]]);
        vertex_lights.extend_from_slice(&[l[0], l[1], l[2], l[5]]);
    }
    true
}

/// Turn the last batch, stored as indexed quads, back into its original triangle list. Exact:
/// a compacted quad's dropped vertices were bit copies of the two it keeps.
fn expand_last_batch(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    batch: &mut DrawBatch,
) {
    let start = batch.start as usize;
    let quads = (batch.end - batch.start) as usize / 4;
    debug_assert_eq!(
        batch.end as usize,
        verts.len(),
        "only the last batch can expand"
    );
    verts.resize(start + quads * 6, EntityVertex::default());
    vertex_lights.resize(start + quads * 6, 0);
    // Back to front: quad `q` is read from `start + 4q` before anything at or after it is written.
    for quad in (0..quads).rev() {
        let read = start + quad * 4;
        let write = start + quad * 6;
        let v = [
            verts[read],
            verts[read + 1],
            verts[read + 2],
            verts[read + 3],
        ];
        let l = [
            vertex_lights[read],
            vertex_lights[read + 1],
            vertex_lights[read + 2],
            vertex_lights[read + 3],
        ];
        verts[write..write + 6].copy_from_slice(&[v[0], v[1], v[2], v[0], v[2], v[3]]);
        vertex_lights[write..write + 6].copy_from_slice(&[l[0], l[1], l[2], l[0], l[2], l[3]]);
    }
    batch.end = (start + quads * 6) as u32;
    batch.indexed = false;
}

/// [`append_draw`] for the phase's main streams with compaction fused in: same batches, same
/// merge rule, and after [`compact_after_fusion`] the same vertices, lights and `indexed` flags as
/// `append_draw` followed by `indexed_experiment::compact`. `open` is true while the last batch is
/// one this function stored compacted and may still extend (`indexed` alone cannot say that:
/// after `finish` an indexed batch must not take more pushes, exactly as before).
#[allow(clippy::too_many_arguments)]
fn append_draw_fused(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    batches: &mut Vec<DrawBatch>,
    open: &mut bool,
    texture: TextureId,
    blend: LayerBlend,
    combined_light: u32,
    new_verts: &[EntityVertex],
    packed_lights: Option<&[u32]>,
    separate: bool,
) {
    let start = verts.len() as u32;
    let quad_shaped = !blend_is_strip(blend) && new_verts.len().is_multiple_of(6);
    let mergeable = match batches.last() {
        Some(b) => !separate && batches_can_merge(b, texture, blend, combined_light, start),
        None => false,
    };
    if mergeable && *open {
        if quad_shaped
            && try_append_compact(
                verts,
                vertex_lights,
                new_verts,
                packed_lights,
                combined_light,
            )
        {
            if let Some(batch) = batches.last_mut() {
                batch.end = verts.len() as u32;
            }
            return;
        }
        // The batch now holds a push `finish` would not compact: store all of it raw and let
        // `compact_after_fusion` run the original whole-batch test on it.
        if let Some(batch) = batches.last_mut() {
            expand_last_batch(verts, vertex_lights, batch);
        }
        *open = false;
        verts.extend_from_slice(new_verts);
        extend_vertex_lights(
            vertex_lights,
            combined_light,
            new_verts.len(),
            packed_lights,
        );
        if let Some(batch) = batches.last_mut() {
            batch.end = verts.len() as u32;
        }
        return;
    }
    if mergeable && batches.last().is_some_and(|b| !b.indexed) {
        verts.extend_from_slice(new_verts);
        extend_vertex_lights(
            vertex_lights,
            combined_light,
            new_verts.len(),
            packed_lights,
        );
        if let Some(batch) = batches.last_mut() {
            batch.end = verts.len() as u32;
        }
        return;
    }
    let indexed = quad_shaped
        && try_append_compact(
            verts,
            vertex_lights,
            new_verts,
            packed_lights,
            combined_light,
        );
    if !indexed {
        verts.extend_from_slice(new_verts);
        extend_vertex_lights(
            vertex_lights,
            combined_light,
            new_verts.len(),
            packed_lights,
        );
    }
    batches.push(DrawBatch {
        texture,
        blend,
        combined_light,
        start,
        end: verts.len() as u32,
        indexed,
    });
    *open = indexed;
}

/// `indexed_experiment::compact` for streams built by [`append_draw_fused`]: batches stored
/// compacted are kept as they are (moved down only when an earlier raw batch shrank), and every
/// raw batch gets the original whole-batch test and compaction, unchanged.
fn compact_after_fusion(
    verts: &mut Vec<EntityVertex>,
    lights: &mut Vec<u32>,
    batches: &mut [DrawBatch],
) {
    assert_eq!(verts.len(), lights.len());
    let mut write = 0;
    for batch in batches {
        let start = batch.start as usize;
        let end = batch.end as usize;
        if batch.indexed {
            if write != start {
                verts.copy_within(start..end, write);
                lights.copy_within(start..end, write);
            }
            batch.start = write as u32;
            write += end - start;
            batch.end = write as u32;
            continue;
        }
        let indexed = !blend_is_strip(batch.blend)
            && (end - start).is_multiple_of(6)
            && end <= i32::MAX as usize
            && (start..end).step_by(6).all(|i| {
                same_vertex(&verts[i], &verts[i + 3])
                    && same_vertex(&verts[i + 2], &verts[i + 4])
                    && lights[i] == lights[i + 3]
                    && lights[i + 2] == lights[i + 4]
            });
        batch.start = write as u32;
        if indexed {
            for read in (start..end).step_by(6) {
                let quad = [
                    verts[read],
                    verts[read + 1],
                    verts[read + 2],
                    verts[read + 5],
                ];
                let light = [
                    lights[read],
                    lights[read + 1],
                    lights[read + 2],
                    lights[read + 5],
                ];
                verts[write..write + 4].copy_from_slice(&quad);
                lights[write..write + 4].copy_from_slice(&light);
                write += 4;
            }
            batch.indexed = true;
        } else {
            verts.copy_within(start..end, write);
            lights.copy_within(start..end, write);
            write += end - start;
        }
        batch.end = write as u32;
    }
    verts.truncate(write);
    lights.truncate(write);
}

/// [`flush_deferred_overlays`] into the main streams through [`append_draw_fused`].
fn flush_deferred_overlays_fused(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    batches: &mut Vec<DrawBatch>,
    open: &mut bool,
    deferred: &mut [DeferredStream],
) {
    for stream in deferred.iter_mut() {
        for batch in &stream.batches {
            let range = batch.start as usize..batch.end as usize;
            append_draw_fused(
                verts,
                vertex_lights,
                batches,
                open,
                batch.texture,
                batch.blend,
                batch.combined_light,
                &stream.verts[range.clone()],
                Some(&stream.lights[range]),
                false,
            );
        }
        stream.clear();
    }
}

/// 06-entity-batch U5: the shadow sheet bound by `Render.renderShadow` (the client's
/// `appearance::SHADOW_TEXTURE` is the same `ResourceLocation`).
pub const DEFERRABLE_SHADOW_TEXTURE: &str = "textures/misc/shadow.png";

/// Environment switch for the opt-in overlay deferral, read once per [`EntityPhase::new`].
pub const DEFER_OVERLAYS_ENV: &str = "MC_RS_ENTITY_DEFER_OVERLAYS";

/// One side queue of deferred overlay pushes (06-entity-batch U5).
#[derive(Debug, Default)]
struct DeferredStream {
    verts: Vec<EntityVertex>,
    lights: Vec<u32>,
    batches: Vec<DrawBatch>,
}

impl DeferredStream {
    fn clear(&mut self) {
        self.verts.clear();
        self.lights.clear();
        self.batches.clear();
    }
}

/// The sampler flags a resource-pack entity texture is drawn with: its `.mcmeta` flags
/// (`SimpleTexture`), except where a renderer overrides them after binding.
///
/// java: `RenderGuardian.doRender` binds `guardian_beam.png` and then sets `GL_TEXTURE_WRAP_S`
/// and `_T` to `GL_REPEAT` itself, so a pack's `"clamp": true` never reaches the beam, whose V
/// coordinate runs over two and a half texture heights per block. Its `"blur"` does.
fn resource_texture_options(path: &str, mut options: TextureMetadata) -> TextureMetadata {
    if path == super::objects::GUARDIAN_BEAM_TEXTURE {
        options.clamp = false;
    }
    options
}

/// Which deferred queue a push joins when overlay deferral is on: `Some(0)` for every enchantment
/// foil (`LayerBlend::Glint`), `Some(1)` for contact shadows, `None` for everything else. Glints
/// flush before shadows, so each entity keeps vanilla's glint-then-shadow order for itself.
fn deferred_overlay_slot(texture: &str, blend: LayerBlend) -> Option<usize> {
    match blend {
        LayerBlend::Glint | LayerBlend::DroppedItemGlint => Some(0),
        LayerBlend::Translucent if texture == DEFERRABLE_SHADOW_TEXTURE => Some(1),
        _ => None,
    }
}

/// Append every deferred queue to the frame streams, glints first, preserving each queue's push
/// order, then empty the queues (keeping their capacity).
fn flush_deferred_overlays(
    verts: &mut Vec<EntityVertex>,
    vertex_lights: &mut Vec<u32>,
    batches: &mut Vec<DrawBatch>,
    deferred: &mut [DeferredStream],
) {
    for stream in deferred.iter_mut() {
        for batch in &stream.batches {
            let range = batch.start as usize..batch.end as usize;
            append_draw(
                verts,
                vertex_lights,
                batches,
                batch.texture,
                batch.blend,
                batch.combined_light,
                &stream.verts[range.clone()],
                Some(&stream.lights[range]),
                false,
            );
        }
        stream.clear();
    }
}

/// The exact fixed-function blend equation used by both vanilla item and armour foil passes.
/// Keeping the descriptor in one helper makes the entity pipeline and its CPU oracle share an
/// auditable source of truth (`GL_SRC_COLOR, GL_ONE`), rather than letting one path drift back to
/// ordinary alpha blending.
fn glint_blend_state() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Src,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

/// The state both textured beams draw under, captured from the real renderers by
/// `tests/guardian_beam_java_parity.rs` and `tests/dragon_beam_java_parity.rs`.
///
/// `RenderGuardian.doRender` calls `disableBlend()` before `tryBlendFuncSeparate(770, 1, 1, 0)`,
/// so that function is never used; `RenderDragon.drawRechargeRay` follows `LayerEnderDragonEyes`,
/// which ends in `disableBlend()`. Both beams are opaque under the world pass's
/// `alphaFunc(GL_GREATER, 0.1)` (their textures are all fully opaque or fully transparent texels,
/// so they are crisp lines that write depth only where a texel is kept), with lighting off and the
/// lightmap stage on: at `setLightmapTextureCoords(240, 240)` for the guardian, at the dragon's
/// own light for the crystal beam, each the caller's combined light. `depthMask(true)` with the
/// ordinary `GL_LEQUAL`; `disableCull()` is [`entity_cull_mode`]'s `None`.
fn beam_pipeline_spec() -> EntityPipelineSpec {
    EntityPipelineSpec {
        fragment_entry: "fs_unlit_cutout",
        blend: None,
        depth_write: true,
        depth_compare: ordinary_depth_compare(),
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

#[derive(Clone, Copy, Debug)]
struct EntityPipelineSpec {
    fragment_entry: &'static str,
    blend: Option<wgpu::BlendState>,
    depth_write: bool,
    depth_compare: wgpu::CompareFunction,
    depth_bias: wgpu::DepthBiasState,
}

/// The shared GL_ONE/GL_ONE state for visible and invisible glowing layers.
fn additive_pipeline_spec(depth_write: bool) -> EntityPipelineSpec {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    EntityPipelineSpec {
        fragment_entry: "fs_unlit",
        blend: Some(wgpu::BlendState {
            color: component,
            alpha: component,
        }),
        depth_write,
        depth_compare: ordinary_depth_compare(),
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

/// `RenderTNTPrimed.doRender`'s second-pass fixed-function state.
///
/// java: `blendFunc(770, 772)`, i.e. `GL_SRC_ALPHA, GL_DST_ALPHA` (772 is `GL_DST_ALPHA`, not
/// `GL_ONE_MINUS_SRC_COLOR`). The opaque body under the flash wrote alpha 1, so the flash is added
/// to it: white by day. This was `OneMinusSrc`, which zeroed the body under the white flash and
/// left `alpha * white`, a grey of at most 0.8.
fn tnt_flash_pipeline_spec() -> EntityPipelineSpec {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::DstAlpha,
        operation: wgpu::BlendOperation::Add,
    };
    EntityPipelineSpec {
        fragment_entry: "fs_tnt_flash",
        blend: Some(wgpu::BlendState {
            color: component,
            alpha: component,
        }),
        depth_write: true,
        depth_compare: ordinary_depth_compare(),
        depth_bias: wgpu::DepthBiasState {
            constant: -3,
            slope_scale: -3.0,
            clamp: 0.0,
        },
    }
}

/// `LayerEnderDragonDeath`'s `blendFunc(SRC_ALPHA, ONE)` shard fans.
fn dragon_death_shards_blend_state() -> wgpu::BlendState {
    let component = wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    };
    wgpu::BlendState {
        color: component,
        alpha: component,
    }
}

/// `RendererLivingEntity.setBrightness`'s affine white colour multiplier.
fn white_flash_pipeline_spec() -> EntityPipelineSpec {
    EntityPipelineSpec {
        fragment_entry: "fs_living_white_flash",
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        depth_write: false,
        depth_compare: wgpu::CompareFunction::Equal,
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

/// `RendererLivingEntity.setBrightness`'s hurt/death combiner while retaining the source
/// model pass's depth-write and alpha-blend policy. Both sources write depth: the body, and the
/// slime gel (`LayerSlimeGel` never calls `depthMask(false)`; `RendererLivingEntity:174` sets
/// `depthMask(true)` before the layers).
fn living_hurt_pipeline_spec(translucent: bool) -> EntityPipelineSpec {
    EntityPipelineSpec {
        fragment_entry: "fs_living_hurt",
        blend: translucent.then_some(wgpu::BlendState::ALPHA_BLENDING),
        depth_write: true,
        depth_compare: ordinary_depth_compare(),
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

fn living_hurt_ghost_pipeline_spec() -> EntityPipelineSpec {
    EntityPipelineSpec {
        fragment_entry: "fs_living_hurt_ghost",
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        depth_write: false,
        depth_compare: ordinary_depth_compare(),
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

fn living_hurt_glint_pipeline_spec() -> EntityPipelineSpec {
    EntityPipelineSpec {
        fragment_entry: "fs_living_hurt_glint",
        blend: Some(glint_blend_state()),
        depth_write: false,
        depth_compare: glint_depth_compare(),
        depth_bias: wgpu::DepthBiasState::default(),
    }
}

const fn glint_depth_compare() -> wgpu::CompareFunction {
    wgpu::CompareFunction::Equal
}

const fn ordinary_depth_compare() -> wgpu::CompareFunction {
    wgpu::CompareFunction::LessEqual
}

/// `GL_CULL_FACE` for `blend`'s pipeline.
///
/// java: the world pass enables back-face culling (`EntityRenderer:1328`). Living entities draw
/// two-sided because `RendererLivingEntity.doRender` (`:92`) disables it for the body and its
/// layers; `RenderEntityItem` leaves it on, and has no negative-scale face switch (see
/// [`LayerBlend::DroppedItem`]). `RenderArrow` leaves it on too
/// ([`LayerBlend::OpaqueUnlitCulled`]). `LayerEnderDragonDeath` turns it back on for its rays
/// (`enableCull()`, captured by `tests/dragon_death_java_parity.rs`), so each shard shows one side.
/// The face matches the first-person item path's
/// [`crate::pipeline::first_person_item_cull_face`] for an unmirrored display transform.
const fn entity_cull_mode(blend: LayerBlend) -> Option<wgpu::Face> {
    match blend {
        LayerBlend::DroppedItem
        | LayerBlend::DroppedItemGlint
        | LayerBlend::OpaqueUnlitCulled
        | LayerBlend::DragonDeathShards => Some(wgpu::Face::Back),
        _ => None,
    }
}

/// Index of `blend`'s pipeline in [`EntityPhase::pipelines`].
const fn pipeline_index(blend: LayerBlend) -> usize {
    match blend {
        LayerBlend::Opaque => 0,
        LayerBlend::OpaqueUnlit => 1,
        LayerBlend::Additive => 2,
        LayerBlend::AdditiveNoDepthWrite => 3,
        LayerBlend::Translucent => 4,
        LayerBlend::TranslucentDepthWrite => 5,
        LayerBlend::MapTranslucentDepthWrite => 6,
        LayerBlend::Glint => 7,
        LayerBlend::LivingHurt => 8,
        LayerBlend::LivingHurtTranslucent => 9,
        LayerBlend::LivingHurtGhost => 10,
        LayerBlend::LivingHurtGlint => 11,
        LayerBlend::WhiteFlash => 12,
        LayerBlend::TntFlash => 13,
        LayerBlend::DragonDeathBody => 14,
        LayerBlend::DragonDeathShards => 15,
        LayerBlend::DragonBeam => 16,
        LayerBlend::FishingLine => 17,
        LayerBlend::Leash => 18,
        LayerBlend::GuardianBeam => 19,
        LayerBlend::DroppedItem => 20,
        LayerBlend::DroppedItemGlint => 21,
        LayerBlend::OpaqueUnlitCulled => 22,
        LayerBlend::Ghost => 23,
    }
}

/// The `ENTITIES` phase: draws whatever the game loop pushed this frame.
///
/// java: the entity half of `RenderGlobal.renderEntities`, plus `RenderManager.renderEntity`.
pub struct EntityPhase {
    /// One pipeline per [`LayerBlend`], indexed by [`pipeline_index`]. Vanilla expresses
    /// these as `GlStateManager` calls around one draw; wgpu bakes blend and depth-write state
    /// into the pipeline, so the state combinations have to be enumerated up front.
    pipelines: [wgpu::RenderPipeline; 24],
    texture_layout: wgpu::BindGroupLayout,
    lightmap_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    /// 06-entity-batch U1: texture keys are interned once; resident textures live in
    /// `texture_slots` at their key's [`TextureId`] index (`None` = not resident), so `record`
    /// resolves a batch with a `Vec` index instead of hashing a `String` per draw.
    texture_ids: TextureInterner,
    texture_slots: Vec<Option<TextureEntry>>,
    /// Resource paths whose PNG failed to load since the last [`Self::clear_textures`].
    ///
    /// java: `TextureManager.loadTexture` binds `TextureUtil.missingTexture` under a location
    /// that failed and keeps it in `mapTextureObjects`, so a missing texture is looked up once,
    /// not once per draw. Without this every frame re-walked the pack stack (one filesystem stat
    /// per pack) for each entity using it (GPURES-11 / ENTR-8).
    missing_textures: std::collections::HashSet<String>,
    /// Java's `GameSettings.anaglyph`, applied at every texture upload boundary.
    anaglyph: bool,
    /// Key of the 1x1 white texture used when an entity's texture has not arrived yet — a
    /// missing skin must not stall the frame (this section's stated hazard).
    fallback: String,
    verts: Vec<EntityVertex>,
    batches: Vec<DrawBatch>,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    uploaded: u32,
    quad_indices: Option<wgpu::Buffer>,
    quad_index_capacity: u32,
    #[cfg(test)]
    indexed_experiment: bool,
    /// Main-stream pushes are compacted as they arrive
    /// ([`append_draw_fused`]) instead of in `finish`. `MC_NW_F1_FUSED_COMPACT=0` turns it off.
    fused_compact: bool,
    /// Whether the last batch was stored compacted by [`append_draw_fused`] and may still grow.
    fused_open: bool,
    /// Packed sky/block coordinate for each entity vertex. 06-entity-batch U1: every push now
    /// writes a real coordinate here (never `NO_VERTEX_LIGHT`), so the batch uniform below is a
    /// single constant record bound once per pass; model renderers still supply per-corner values.
    vertex_lights: Vec<u32>,
    vertex_light_buffer: Option<wgpu::Buffer>,
    vertex_light_capacity: u64,
    vertex_light_uploaded: u32,
    lightmap_buffer: Option<wgpu::Buffer>,
    lightmap_bind_group: Option<wgpu::BindGroup>,
    lightmap_capacity: u64,
    /// The entity-side copy of Java's shared 16x16 lightmap. The client driver currently refreshes
    /// terrain and particles; [`Self::set_lightmap_params`] is the explicit handoff for the
    /// remaining entity-phase fanout.
    lightmap_texture: wgpu::Texture,
    lightmap_view: wgpu::TextureView,
    lightmap_sampler: wgpu::Sampler,
    /// Raw entity lightmap bytes, retained so an anaglyph toggle can replay the source exactly
    /// once instead of applying Java's integer RGB transform twice.
    lightmap_source_rgba: Vec<u8>,
    /// What the lightmap texture currently holds, when it came from parameters.
    uploaded_lightmap_params: Option<LightmapParams>,
    /// Dedicated colour-only geometry for spectator outlines. It is never mixed with normal
    /// entity batches: the mask is rendered to its own target before the vanilla post chain.
    outline_verts: Vec<EntityVertex>,
    outline_buffer: Option<wgpu::Buffer>,
    outline_capacity: u64,
    outline_uploaded: u32,
    outline_pipeline: wgpu::RenderPipeline,
    outline_mask: Option<OutlineMask>,
    outline_enabled: bool,
    /// F3+B POSITION_COLOR lines, depth-tested against the scene but not depth-writing.
    debug_verts: Vec<EntityVertex>,
    debug_buffer: Option<wgpu::Buffer>,
    debug_capacity: u64,
    debug_uploaded: u32,
    debug_pipeline: wgpu::RenderPipeline,
    debug_enabled: bool,
    /// Fullscreen composite of the blurred outline target over the resolved frame.
    composite_layout: wgpu::BindGroupLayout,
    composite_pipeline: wgpu::RenderPipeline,
    diagnostics_enabled: bool,
    diagnostic_stats: EntityFrameStats,
    enabled: bool,
    /// 06-entity-batch U5 (opt-in, default off): queue glint and contact-shadow pushes and append
    /// them after all other entity geometry at [`EntityPhase::finish`]. NOT pixel-identical in
    /// general (see `deferred_overlay_slot`'s users and the 06 findings, design §B).
    defer_overlays: bool,
    deferred: [DeferredStream; 2],
    /// Item-frame map images share these pages; see [`super::map_atlas`]. Empty, and free, until the first framed map.
    map_atlas: super::map_atlas::MapAtlas,
    /// Opaque map images queued by [`EntityPhase::push_map_deferred`], one
    /// stream per atlas page, appended page by page by [`EntityPhase::flush_deferred_maps`] (at
    /// the latest by `finish`): at most one draw per page whatever order the maps came in.
    deferred_maps: Vec<DeferredStream>,
    /// The shader packs' entity shadow pipeline ([`pack_shadow`]),
    /// built the first frame a pack draws entity shadows; `None` (always, with no pack) costs
    /// nothing and is never read by [`EntityPhase::record`].
    pack_shadow: Option<pack_shadow::EntityShadowGpu>,
}

struct OutlineMask {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
    chain: crate::post::PostChain,
    /// `Some(size)` while the mask and the chain's targets are 1x1
    /// stand-ins (nothing drawn yet): the window size they grow to the first frame the outline
    /// is drawn. `None`: full size, as before.
    dormant: Option<(u32, u32)>,
}

/// The spectator outline mask texture, as `install_outline` makes it.
fn outline_mask_texture(gpu: &Gpu, width: u32, height: u32) -> wgpu::Texture {
    gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("mc-render spectator outline mask"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Opt-in per-frame entity upload counters for O1 attribution.
///
/// The byte fields count the payload supplied to `wgpu::Queue::write_buffer`, not the rounded
/// capacity of the backing GPU buffers. Repeated [`EntityPhase::finish`] calls intentionally add
/// the full payload again: that is the duplicate-write cost O1 needs to expose. The current
/// vertex/batch fields describe the accumulated frame geometry after the most recent finish.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EntityFrameStats {
    /// Number of [`EntityPhase::finish`] calls since the last [`EntityPhase::begin`].
    pub finish_calls: u64,
    /// Sum of all submitted payload streams below.
    pub upload_bytes: u64,
    /// Bytes written for the accumulated entity vertex stream.
    pub vertex_upload_bytes: u64,
    /// Bytes written for the parallel packed-light vertex stream.
    pub light_upload_bytes: u64,
    /// Experimental static quad index payload; written only when its capacity grows.
    pub index_upload_bytes: u64,
    /// Bytes written for spectator-outline vertices.
    pub outline_upload_bytes: u64,
    /// Bytes written for F3+B debug vertices.
    pub debug_upload_bytes: u64,
    /// Bytes written for per-batch dynamic lightmap uniforms.
    pub dynamic_uniform_upload_bytes: u64,
    /// Accumulated entity vertices after the latest finish.
    pub vertex_count: usize,
    /// Accumulated draw batches after the latest finish.
    pub batch_count: usize,
    /// Current capacity of the entity vertex, light, outline, debug, and dynamic-uniform buffers.
    pub gpu_capacity_bytes_total: u64,
}

impl EntityFrameStats {
    /// Add one finish delta. Upload fields in `delta` are per-call payloads and accumulate;
    /// geometry counts and capacity are the latest post-finish snapshot.
    fn record_finish(&mut self, delta: Self) {
        self.finish_calls = self.finish_calls.saturating_add(delta.finish_calls);
        self.upload_bytes = self.upload_bytes.saturating_add(delta.upload_bytes);
        self.vertex_upload_bytes = self
            .vertex_upload_bytes
            .saturating_add(delta.vertex_upload_bytes);
        self.light_upload_bytes = self
            .light_upload_bytes
            .saturating_add(delta.light_upload_bytes);
        self.outline_upload_bytes = self
            .outline_upload_bytes
            .saturating_add(delta.outline_upload_bytes);
        self.debug_upload_bytes = self
            .debug_upload_bytes
            .saturating_add(delta.debug_upload_bytes);
        self.dynamic_uniform_upload_bytes = self
            .dynamic_uniform_upload_bytes
            .saturating_add(delta.dynamic_uniform_upload_bytes);
        self.index_upload_bytes = self
            .index_upload_bytes
            .saturating_add(delta.index_upload_bytes);
        self.vertex_count = delta.vertex_count;
        self.batch_count = delta.batch_count;
        self.gpu_capacity_bytes_total = delta.gpu_capacity_bytes_total;
    }
}

fn uploaded_payload_bytes<T>(items: &[T]) -> u64 {
    if items.is_empty() {
        0
    } else {
        std::mem::size_of_val(items) as u64
    }
}

fn upload_vertices(
    gpu: &Gpu,
    label: &str,
    vertices: &[EntityVertex],
    buffer: &mut Option<wgpu::Buffer>,
    capacity: &mut u64,
) -> u32 {
    let uploaded = vertices.len() as u32;
    if vertices.is_empty() {
        return uploaded;
    }
    let needed = std::mem::size_of_val(vertices) as u64;
    if *capacity < needed || buffer.is_none() {
        let new_capacity = needed.next_power_of_two().max(4096);
        *buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: new_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        *capacity = new_capacity;
    }
    if let Some(buffer) = buffer {
        // Through the per-frame upload ring: the frame encoder is this buffer's only reader.
        gpu.write_buffer_staged(buffer, 0, bytemuck::cast_slice(vertices));
    }
    uploaded
}

fn upload_vertex_lights(
    gpu: &Gpu,
    lights: &[u32],
    buffer: &mut Option<wgpu::Buffer>,
    capacity: &mut u64,
) -> u32 {
    let uploaded = lights.len() as u32;
    if lights.is_empty() {
        return uploaded;
    }
    let needed = std::mem::size_of_val(lights) as u64;
    if *capacity < needed || buffer.is_none() {
        let new_capacity = needed.next_power_of_two().max(4096);
        *buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render entity vertex lights"),
            size: new_capacity,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        *capacity = new_capacity;
    }
    if let Some(buffer) = buffer {
        gpu.write_buffer_staged(buffer, 0, bytemuck::cast_slice(lights));
    }
    uploaded
}

fn upload_lightmap_texture(gpu: &Gpu, texture: &wgpu::Texture, rgba: &[u8]) {
    const SIDE: u32 = 16;
    assert_eq!(
        rgba.len(),
        (SIDE * SIDE * 4) as usize,
        "invalid entity lightmap payload"
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
            bytes_per_row: Some(SIDE * 4),
            rows_per_image: Some(SIDE),
        },
        wgpu::Extent3d {
            width: SIDE,
            height: SIDE,
            depth_or_array_layers: 1,
        },
    );
}

/// Build the bytes Java's `TextureUtil.uploadTexture` receives for the entity lightmap.
///
/// `Lightmap::generate` is the shared, non-anaglyph source table.  The global anaglyph setting
/// is an upload-time RGB transform (alpha is retained), just as it is for terrain and particles;
/// keeping that boundary here prevents the entity sampler from drifting from those phases.
fn entity_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

impl EntityPhase {
    /// The texture key used when an entity's real texture is not resident yet.
    pub const FALLBACK_TEXTURE: &'static str = "mc-render/entity/fallback";
    /// The 1x1 white binding used by texture-disabled Java geometry (leashes and fishing lines).
    pub const SOLID_TEXTURE: &'static str = "mc-render/entity/solid";

    /// Builds the pipeline. Cheap enough to do at startup; holds no per-entity state.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Builds the pipeline with Java's global texture-upload anaglyph switch.
    ///
    /// The default [`Self::new`] remains off for compatibility with existing callers.  A client
    /// that owns `GameSettings` should use this constructor at install time or call
    /// [`Self::set_anaglyph`] before its first resource-backed upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render entity"),
                source: wgpu::ShaderSource::Wgsl(ENTITY_WGSL.into()),
            });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render entity texture"),
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
        let lightmap_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render entity lightmap"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            // The vertex stage selects between this per-draw coordinate and the
                            // parallel per-vertex stream before handing a normalized UV to the
                            // rasterizer.
                            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: true,
                                min_binding_size: NonZeroU64::new(core::mem::size_of::<
                                    LightmapUniform,
                                >(
                                )
                                    as u64),
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                });
        let initial_lightmap = Lightmap::generate(LightmapParams::default());
        let lightmap_source_rgba = initial_lightmap.rgba.clone();
        let lightmap_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render entity lightmap texture"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let initial_upload_rgba = entity_upload_rgba(&initial_lightmap.rgba, anaglyph);
        upload_lightmap_texture(gpu, &lightmap_texture, &initial_upload_rgba);
        let lightmap_view = lightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let lightmap_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render entity lightmap sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render entity layout"),
                bind_group_layouts: &[
                    Some(gpu.camera_layout()),
                    Some(&texture_layout),
                    Some(&lightmap_layout),
                ],
                immediate_size: 0,
            });
        // `LayerBlend` was a policy function with no mechanism behind it — one opaque
        // pipeline drew every layer, so glowing eyes and the charged-creeper aura would have come
        // out as flat opaque quads. These are the `GlStateManager` states the layer/item
        // renderers actually set, enumerated as pipelines.
        let make_pipeline = |blend: LayerBlend| {
            let (entry, target_blend, depth_write, depth_compare) = match blend {
                // java: the ambient state from `Minecraft.startGame` — blend off, depth writing on.
                LayerBlend::Opaque => ("fs", None, true, ordinary_depth_compare()),
                // java: Render.renderEntityOnFire leaves the ordinary world alpha test and
                // depth mask alone, changing only the fixed-function lighting enable.
                // `OpaqueUnlitCulled` is the same state with the world pass's culling kept
                // (`entity_cull_mode`); RenderArrow's own light arrives in the vertex colour.
                LayerBlend::OpaqueUnlit | LayerBlend::OpaqueUnlitCulled => {
                    ("fs_unlit_cutout", None, true, ordinary_depth_compare())
                }
                // java: `blendFunc(1, 1)` — GL_ONE/GL_ONE. `LayerCreeperCharge:40`,
                // `LayerWitherAura:45`, `LayerEndermanEyes:27`, `LayerSpiderEyes:26`. Depth
                // writing stays *on* for a visible entity (`depthMask(!isInvisible())`), which is
                // why the eyes need LEQUAL below rather than a depth bias.
                LayerBlend::Additive | LayerBlend::AdditiveNoDepthWrite => {
                    let spec = additive_pipeline_spec(blend == LayerBlend::Additive);
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                // java: `blendFunc(770, 771)` — GL_SRC_ALPHA/GL_ONE_MINUS_SRC_ALPHA — with
                // `depthMask(false)`: `Render.renderShadow`. `LayerSlimeGel` keeps depth writes
                // on and uses `TranslucentDepthWrite` below. Both test the combined alpha.
                LayerBlend::Translucent => (
                    "fs_translucent",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    false,
                    ordinary_depth_compare(),
                ),
                // java: `RendererLivingEntity.renderModel:262`'s invisible ghost: the same state
                // under its lowered `alphaFunc(516, 0.003921569)`.
                LayerBlend::Ghost => (
                    "fs_ghost",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    false,
                    ordinary_depth_compare(),
                ),
                // java: `MobAppearance.renderParticle` calls `depthMask(true)` then enables
                // standard SRC_ALPHA blending for its elder-guardian apparition. `LayerSlimeGel`
                // (`:25-26`) has the same state: blend on, the entity pass's depth mask left on.
                LayerBlend::TranslucentDepthWrite => (
                    "fs_translucent",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    true,
                    ordinary_depth_compare(),
                ),
                // java: `MapItemRenderer.renderMap` disables lighting, enables blending and
                // chooses `tryBlendFuncSeparate(ONE, ONE_MINUS_SRC_ALPHA, ZERO, ONE)`. Its map
                // texture contains low-alpha black unexplored texels, so using the ordinary
                // SRC_ALPHA pipeline would incorrectly premultiply that path.
                LayerBlend::MapTranslucentDepthWrite => (
                    "fs_unlit",
                    Some(wgpu::BlendState {
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
                    }),
                    true,
                    ordinary_depth_compare(),
                ),
                // java: `RenderItem.renderEffect` and `LayerArmorBase.renderGlint` both use
                // `blendFunc(768, 1)` (`GL_SRC_COLOR, GL_ONE`), disable lighting, call
                // `depthMask(false)`, and replay the model at exact depth (`depthFunc(514)`,
                // `GL_EQUAL`).  `disableLighting()` drops the two directional lights only; the
                // lightmap texture stage stays on, so `fs_glint` still multiplies by it.
                LayerBlend::Glint => (
                    "fs_glint",
                    Some(glint_blend_state()),
                    false,
                    glint_depth_compare(),
                ),
                // java: the hurt/death combiner is part of the model's original textured draw,
                // so preserve that draw's depth-write and blend policy instead of replaying an
                // untextured mesh at GL_EQUAL.
                LayerBlend::LivingHurt | LayerBlend::LivingHurtTranslucent => {
                    let spec =
                        living_hurt_pipeline_spec(blend == LayerBlend::LivingHurtTranslucent);
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                LayerBlend::LivingHurtGhost => {
                    let spec = living_hurt_ghost_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                LayerBlend::LivingHurtGlint => {
                    let spec = living_hurt_glint_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                LayerBlend::WhiteFlash => {
                    let spec = white_flash_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                LayerBlend::TntFlash => {
                    let spec = tnt_flash_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                LayerBlend::DragonDeathBody => {
                    ("fs_dragon_death_body", None, true, ordinary_depth_compare())
                }
                LayerBlend::DragonDeathShards => (
                    "fs_dragon_death_shards",
                    Some(dragon_death_shards_blend_state()),
                    false,
                    ordinary_depth_compare(),
                ),
                // java: `RenderDragon.drawRechargeRay`: opaque, alpha-tested, unlit, lightmapped.
                LayerBlend::DragonBeam => {
                    let spec = beam_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                // java: `RenderFish`/`RenderLiving.renderLeash` disable texture and lighting,
                // leave ordinary LEQUAL/depth-write state intact, and supply POSITION_COLOR
                // vertices. A dedicated fragment entry avoids alpha/cutout work entirely.
                LayerBlend::FishingLine | LayerBlend::Leash => {
                    ("fs_untextured", None, true, ordinary_depth_compare())
                }
                // java: `RenderGuardian.doRender`: opaque, alpha-tested, unlit, lightmapped.
                LayerBlend::GuardianBeam => {
                    let spec = beam_pipeline_spec();
                    (
                        spec.fragment_entry,
                        spec.blend,
                        spec.depth_write,
                        spec.depth_compare,
                    )
                }
                // java: `RenderEntityItem.doRender` (`:103-106`) — `enableBlend()`,
                // `tryBlendFuncSeparate(770, 771, 1, 0)`, depth mask and 0.1 alpha test as the
                // world pass left them; only the culling differs from `TranslucentDepthWrite`.
                LayerBlend::DroppedItem => (
                    "fs",
                    Some(wgpu::BlendState::ALPHA_BLENDING),
                    true,
                    ordinary_depth_compare(),
                ),
                // java: `RenderItem.renderEffect`'s foil state, as `Glint`, with culling kept on.
                LayerBlend::DroppedItemGlint => (
                    "fs_glint",
                    Some(glint_blend_state()),
                    false,
                    glint_depth_compare(),
                ),
            };
            let topology = match blend {
                LayerBlend::FishingLine => wgpu::PrimitiveTopology::LineStrip,
                LayerBlend::Leash => wgpu::PrimitiveTopology::TriangleStrip,
                _ => wgpu::PrimitiveTopology::TriangleList,
            };
            let depth_bias = if blend == LayerBlend::TntFlash {
                tnt_flash_pipeline_spec().depth_bias
            } else {
                wgpu::DepthBiasState::default()
            };
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("mc-render entity pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(ENTITY_VERTEX_LAYOUT), Some(ENTITY_VERTEX_LIGHT_LAYOUT)],
                    },
                    // java: RendererLivingEntity.doRender calls disableCull() — mobs are drawn
                    // two-sided, which matters for flat parts like the bat's wings and capes.
                    // A dropped item keeps the world pass's back-face culling.
                    primitive: wgpu::PrimitiveState {
                        topology,
                        cull_mode: entity_cull_mode(blend),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        depth_write_enabled: Some(depth_write),
                        // java: `Minecraft.startGame` (`:540`) — `GlStateManager.depthFunc(515)`,
                        // GL_LEQUAL, and nothing in the world pass changes it back. This was
                        // `Less`, which is a *hard* blocker on every layer whose geometry is the
                        // body redrawn under another texture: `LayerEndermanEyes` and
                        // `LayerSpiderEyes` re-render the main model at the identical depth, so
                        // under GL_LESS every one of their fragments fails the test and the layer
                        // is invisible no matter how correctly it was emitted.
                        depth_compare: Some(depth_compare),
                        stencil: wgpu::StencilState::default(),
                        bias: depth_bias,
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some(entry),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: target_blend,
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let pipelines = [
            make_pipeline(LayerBlend::Opaque),
            make_pipeline(LayerBlend::OpaqueUnlit),
            make_pipeline(LayerBlend::Additive),
            make_pipeline(LayerBlend::AdditiveNoDepthWrite),
            make_pipeline(LayerBlend::Translucent),
            make_pipeline(LayerBlend::TranslucentDepthWrite),
            make_pipeline(LayerBlend::MapTranslucentDepthWrite),
            make_pipeline(LayerBlend::Glint),
            make_pipeline(LayerBlend::LivingHurt),
            make_pipeline(LayerBlend::LivingHurtTranslucent),
            make_pipeline(LayerBlend::LivingHurtGhost),
            make_pipeline(LayerBlend::LivingHurtGlint),
            make_pipeline(LayerBlend::WhiteFlash),
            make_pipeline(LayerBlend::TntFlash),
            make_pipeline(LayerBlend::DragonDeathBody),
            make_pipeline(LayerBlend::DragonDeathShards),
            make_pipeline(LayerBlend::DragonBeam),
            make_pipeline(LayerBlend::FishingLine),
            make_pipeline(LayerBlend::Leash),
            make_pipeline(LayerBlend::GuardianBeam),
            make_pipeline(LayerBlend::DroppedItem),
            make_pipeline(LayerBlend::DroppedItemGlint),
            make_pipeline(LayerBlend::OpaqueUnlitCulled),
            make_pipeline(LayerBlend::Ghost),
        ];
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render entity sampler"),
            // Vanilla entity textures are point-sampled; bilinear turns a 64x64 skin to mush.
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            // `RenderItem.renderEffect` leaves the texture matrix's UVs unbounded.  The glint
            // resource is a tiled texture in vanilla, so entity-side sampling must repeat rather
            // than clamp its animated strips to the edge texel.  Ordinary entity UVs are inside
            // 0..1, so this does not alter their samples.
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        let geometry_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render debug geometry"),
                source: wgpu::ShaderSource::Wgsl(DEBUG_GEOMETRY_WGSL.into()),
            });
        let geometry_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render debug geometry layout"),
                bind_group_layouts: &[Some(gpu.camera_layout())],
                immediate_size: 0,
            });
        let outline_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render spectator outline mask"),
                layout: Some(&geometry_layout),
                vertex: wgpu::VertexState {
                    module: &geometry_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(ENTITY_VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &geometry_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let debug_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render debug bounding boxes"),
                layout: Some(&geometry_layout),
                vertex: wgpu::VertexState {
                    module: &geometry_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(ENTITY_VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(false),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: gpu.sample_count(),
                    ..Default::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &geometry_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let composite_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render spectator outline composite"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });
        let composite_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render spectator outline composite"),
                source: wgpu::ShaderSource::Wgsl(OUTLINE_COMPOSITE_WGSL.into()),
            });
        let composite_pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("mc-render spectator outline composite layout"),
                    bind_group_layouts: &[Some(&composite_layout)],
                    immediate_size: 0,
                });
        let composite_pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("mc-render spectator outline composite"),
                    layout: Some(&composite_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &composite_shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    fragment: Some(wgpu::FragmentState {
                        module: &composite_shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                });

        let mut phase = Self {
            pipelines,
            texture_layout,
            lightmap_layout,
            sampler,
            texture_ids: TextureInterner::default(),
            texture_slots: Vec::new(),
            missing_textures: std::collections::HashSet::new(),
            anaglyph,
            fallback: Self::FALLBACK_TEXTURE.to_string(),
            verts: Vec::new(),
            batches: Vec::new(),
            buffer: None,
            capacity: 0,
            uploaded: 0,
            quad_indices: None,
            quad_index_capacity: 0,
            #[cfg(test)]
            indexed_experiment: true,
            fused_compact: fused_compact_requested(),
            fused_open: false,
            vertex_lights: Vec::new(),
            vertex_light_buffer: None,
            vertex_light_capacity: 0,
            vertex_light_uploaded: 0,
            lightmap_buffer: None,
            lightmap_bind_group: None,
            lightmap_capacity: 0,
            lightmap_texture,
            lightmap_view,
            lightmap_sampler,
            lightmap_source_rgba,
            uploaded_lightmap_params: None,
            outline_verts: Vec::new(),
            outline_buffer: None,
            outline_capacity: 0,
            outline_uploaded: 0,
            outline_pipeline,
            outline_mask: None,
            outline_enabled: false,
            debug_verts: Vec::new(),
            debug_buffer: None,
            debug_capacity: 0,
            debug_uploaded: 0,
            debug_pipeline,
            debug_enabled: false,
            composite_layout,
            composite_pipeline,
            diagnostics_enabled: false,
            diagnostic_stats: EntityFrameStats::default(),
            enabled: true,
            defer_overlays: std::env::var_os(DEFER_OVERLAYS_ENV)
                .is_some_and(|value| value.to_str() == Some("1")),
            deferred: Default::default(),
            map_atlas: Default::default(),
            deferred_maps: Vec::new(),
            pack_shadow: None,
        };
        phase.upload_texture(
            gpu,
            Self::FALLBACK_TEXTURE,
            &missing_texture_pixels(),
            MISSING_TEXTURE_SIZE,
            MISSING_TEXTURE_SIZE,
        );
        // POSITION_COLOR-only submissions (leashes/fishing lines) still need a valid bind group
        // because the shared entity record binds texture slot 1 for every batch.  Their fragment
        // entry ignores it, but keeping a resident 1x1 white texture makes the omitted-layer and
        // late-resource paths deterministic instead of silently dropping the draw.
        phase.upload_texture(gpu, Self::SOLID_TEXTURE, &[255, 255, 255, 255], 1, 1);
        phase
    }

    /// Uploads (or replaces) an entity texture. `rgba` must be `width * height * 4` bytes.
    ///
    /// # Panics
    /// If `rgba` is not exactly `width * height * 4` bytes.
    pub fn upload_texture(&mut self, gpu: &Gpu, key: &str, rgba: &[u8], width: u32, height: u32) {
        self.upload_texture_with_options(
            gpu,
            key,
            rgba,
            width,
            height,
            &TextureMetadata::default(),
        );
    }

    /// Replace the entity phase's copy of Java's shared 16x16 lightmap.
    ///
    /// The packed sky/block coordinates carried by a falling-block vertex are only half of the
    /// contract: the final colour comes from this frame's lightmap texture (sun, gamma, torch
    /// flicker, night vision, and boss darkening). The client currently refreshes terrain and
    /// particles through the driver; this explicit API keeps the entity handoff ready without
    /// coupling this phase to client settings ownership.
    pub fn set_lightmap(&mut self, gpu: &Gpu, lightmap: &Lightmap) {
        self.uploaded_lightmap_params = None;
        self.lightmap_source_rgba.clone_from(&lightmap.rgba);
        let upload_rgba = entity_upload_rgba(&lightmap.rgba, self.anaglyph);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// Generate and upload one entity lightmap frame from Java's `EntityRenderer.updateLightmap`
    /// inputs.
    pub fn set_lightmap_params(&mut self, gpu: &Gpu, params: LightmapParams) {
        if self.uploaded_lightmap_params == Some(params) {
            return;
        }
        self.set_lightmap(gpu, &Lightmap::generate_cached(params));
        self.uploaded_lightmap_params = Some(params);
    }

    /// Upload an entity texture while preserving the Java `SimpleTexture` sampler flags from
    /// its `.mcmeta` sidecar.  A separate sampler belongs in the bind group because wgpu does
    /// not allow sampler state to vary per draw otherwise.
    pub fn upload_texture_with_options(
        &mut self,
        gpu: &Gpu,
        key: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
        options: &TextureMetadata,
    ) {
        assert_eq!(
            rgba.len(),
            (width * height * 4) as usize,
            "entity texture {key} is {} bytes, expected {}x{}x4",
            rgba.len(),
            width,
            height
        );
        // `TextureUtil.copyToBufferPos` applies `anaglyphColor` to the temporary upload array,
        // never to the decoded source held by a resource/reload owner.  Keep that same boundary
        // here so layered textures and ordinary `SimpleTexture` PNGs share one conversion and a
        // later setting toggle can rebuild from raw bytes.
        let source_rgba = rgba.to_vec();
        let upload_rgba = mc_assets::sprite::anaglyph_rgba(&source_rgba, self.anaglyph);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(key),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENTITY_TEXTURE_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            texture.as_image_copy(),
            &upload_rgba,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let option_sampler = if *options == TextureMetadata::default() {
            None
        } else {
            Some(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("mc-render entity metadata sampler"),
                mag_filter: if options.blur {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                min_filter: if options.blur {
                    wgpu::FilterMode::Linear
                } else {
                    wgpu::FilterMode::Nearest
                },
                mipmap_filter: wgpu::MipmapFilterMode::Nearest,
                address_mode_u: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                address_mode_v: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                address_mode_w: if options.clamp {
                    wgpu::AddressMode::ClampToEdge
                } else {
                    wgpu::AddressMode::Repeat
                },
                ..Default::default()
            }))
        };
        let sampler = option_sampler.as_ref().unwrap_or(&self.sampler);
        let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &self.texture_layout,
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
        });
        let id = self.texture_ids.intern(key);
        if self.texture_slots.len() <= id.index() {
            self.texture_slots.resize_with(id.index() + 1, || None);
        }
        self.texture_slots[id.index()] = Some(TextureEntry {
            texture,
            bind_group,
            width,
            height,
            source_rgba: RefCell::new(source_rgba),
            options: options.clone(),
        });
    }

    /// The resident texture uploaded under `key`, if any.
    fn texture_entry(&self, key: &str) -> Option<&TextureEntry> {
        let id = self.texture_ids.get(key)?;
        self.texture_slots.get(id.index())?.as_ref()
    }

    /// Whether this phase applies Java's global upload-time anaglyph transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global texture-upload transform and rebuild every resident texture.
    ///
    /// Rebuilding from retained raw pixels is important: running the integer transform on the
    /// current GPU bytes would double-convert an already enabled phase.  The source cache also
    /// keeps sampler flags attached to each `SimpleTexture`/`LayeredTexture` result.  Callers that
    /// only need to affect future uploads may use [`Self::set_anaglyph`] instead.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        let uploads: Vec<_> = self
            .texture_ids
            .keys
            .iter()
            .zip(&self.texture_slots)
            .filter_map(|(key, slot)| slot.as_ref().map(|entry| (key, entry)))
            .map(|(key, entry)| {
                (
                    key.clone(),
                    entry.source_rgba.borrow().clone(),
                    entry.width,
                    entry.height,
                    entry.options.clone(),
                )
            })
            .collect();
        for (key, rgba, width, height, options) in uploads {
            self.upload_texture_with_options(gpu, &key, &rgba, width, height, &options);
        }
        let upload_rgba = entity_upload_rgba(&self.lightmap_source_rgba, enabled);
        upload_lightmap_texture(gpu, &self.lightmap_texture, &upload_rgba);
    }

    /// Set the transform for subsequent uploads without touching resident GPU textures.
    ///
    /// This additive control is useful for callers that clear/reinstall their phase immediately
    /// after changing settings.  Use [`Self::set_anaglyph_with_gpu`] when existing entries must
    /// change in the same frame.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Loads a horse's generated texture from its resource-backed base, markings and armour
    /// sheets, composes them in Java's source-over order, and inserts the result in the entity
    /// texture cache under the stable [`HorseTextureSelection::upload_key`].
    ///
    /// Normal horses always use a [`LayeredTexture`]-style generated entry, including the bare
    /// unarmoured case. Donkeys, mules and undead horses use their direct type sheet until an
    /// armour layer is present; an invalid selection returns `false` so the caller can bind its
    /// ordinary renderer fallback. A layer failure is intentionally transactional: no partial
    /// generated entry is inserted, so a later pack/resource replacement can retry cleanly.
    pub fn load_horse_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        selection: &HorseTextureSelection,
    ) -> bool {
        if !selection.valid {
            return false;
        }
        let Some(key) = selection.cache_key() else {
            return false;
        };
        if self.has_texture(key) {
            return true;
        }

        let paths = selection.paths();
        let mut decoded: [Option<Vec<u8>>; 3] = [None, None, None];
        let mut dimensions = None;
        for (index, path) in paths.into_iter().enumerate() {
            let Some(path) = path else {
                continue;
            };
            let Ok(location) = mc_assets::ResourceLocation::parse(path) else {
                return false;
            };
            let Ok((rgba, width, height, _metadata)) = resources.load_png_rgba(&location) else {
                return false;
            };
            if dimensions.is_some_and(|(expected_width, expected_height)| {
                expected_width != width || expected_height != height
            }) {
                return false;
            }
            dimensions = Some((width, height));
            decoded[index] = Some(rgba);
        }
        let Some((width, height)) = dimensions else {
            return false;
        };
        let layers = [
            decoded[0].as_deref(),
            decoded[1].as_deref(),
            decoded[2].as_deref(),
        ];
        let Some(composed) = compose_horse_texture(selection, layers, width, height) else {
            return false;
        };
        // LayeredTexture draws ImageIO pixels directly and owns no SimpleTexture metadata
        // sampler. The generated image therefore uses the default nearest/repeat sampler even
        // when an individual source sheet carries an unrelated `.mcmeta` sidecar.
        self.upload_texture(gpu, key, &composed, width, height);
        true
    }

    /// Apply one S09 animated-atlas dirty rectangle to an entity texture copy.
    ///
    /// The live held-item/block path uses [`EntityPhase::upload_texture`] with the synthetic
    /// `BLOCK_ATLAS_TEXTURE` key; all other entity textures are independent pack files and are
    /// correctly rejected by the key lookup below.
    pub fn upload_atlas_animation(
        &mut self,
        gpu: &Gpu,
        key: &str,
        upload: &AtlasAnimationUpload,
    ) -> bool {
        let anaglyph = self.anaglyph;
        let Some(entry) = self.texture_entry(key) else {
            return false;
        };
        if !atlas_animation_rect_valid(upload, entry.width, entry.height) {
            return false;
        }
        // Keep the retained raw atlas current so a later live anaglyph toggle replays the latest
        // animation frame rather than reverting this key to its initial sheet frame.
        for row in 0..upload.height {
            let src_start = (row * upload.width * 4) as usize;
            let dst_start = (((upload.y + row) * entry.width + upload.x) * 4) as usize;
            let count = (upload.width * 4) as usize;
            entry.source_rgba.borrow_mut()[dst_start..dst_start + count]
                .copy_from_slice(&upload.rgba[src_start..src_start + count]);
        }
        // Atlas animator records are raw CPU pixels.  Convert this dirty rectangle at the same
        // upload boundary as the initial atlas copy; never mutate the shared animation record.
        let upload_rgba = mc_assets::sprite::anaglyph_rgba(&upload.rgba, anaglyph);
        gpu.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &entry.texture,
                mip_level: 0,
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
        true
    }

    /// True if `key` has been uploaded.
    #[must_use]
    pub fn has_texture(&self, key: &str) -> bool {
        self.texture_entry(key).is_some()
    }

    /// Loads `path` from the resource pack and uploads it under `path` as its own key, unless it
    /// is already resident. Returns whether the texture is resident afterwards.
    ///
    /// java: `TextureManager.bindTexture` — vanilla also loads an entity texture the first time
    /// a renderer asks for it (`SimpleTexture.loadTexture`) rather than pre-loading the set, and
    /// caches it in `mapTextureObjects` forever after. The keys here are the same
    /// `ResourceLocation` strings the descriptor table already stores
    /// ([`EntityRenderer::texture`](super::manager::EntityRenderer::texture),
    /// [`texture_for`](super::manager::texture_for),
    /// [`LayerKind::texture`](super::layers::LayerKind::texture),
    /// [`armor_texture`](super::layers::armor_texture)), so a caller never has to invent one.
    ///
    /// A missing or corrupt PNG is **not** an error: it returns `false` and the caller pushes
    /// under [`Self::FALLBACK_TEXTURE`] instead, which is this phase's standing rule that a
    /// texture that has not arrived must never stall or drop a frame.
    pub fn load_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        path: &str,
    ) -> bool {
        if self.has_texture(path) {
            return true;
        }
        if self.missing_textures.contains(path) {
            return false;
        }
        let loaded = mc_assets::ResourceLocation::parse(path)
            .ok()
            .and_then(|location| resources.load_png_rgba(&location).ok());
        let Some((rgba, width, height, options)) = loaded else {
            self.missing_textures.insert(path.to_owned());
            return false;
        };
        let options = resource_texture_options(path, options);
        self.upload_texture_with_options(gpu, path, &rgba, width, height, &options);
        true
    }

    /// Drop every resource-backed entity texture and restore the two built-in bindings.
    ///
    /// The client phase itself survives the cheap resource-pack reload path, so generated horse
    /// keys (and ordinary PNG keys) must not keep pointing at the previous pack's GPU objects.
    /// Re-uploading the fallback/solid entries keeps the phase render-safe between this call and
    /// the next frame's lazy resource loads.
    pub fn clear_textures(&mut self, gpu: &Gpu) {
        // Ids stay interned (a batch recorded before the reload keeps a valid index and resolves
        // to the fallback until its key is re-uploaded); only residency is dropped.
        self.texture_slots.iter_mut().for_each(|slot| *slot = None);
        self.missing_textures.clear();
        // The pages went with the slots above; the next frame re-uploads
        // every drawn map. java: `Minecraft.refreshResources` -> `clearLoadedMaps()`.
        self.map_atlas.clear();
        self.upload_texture(
            gpu,
            Self::FALLBACK_TEXTURE,
            &missing_texture_pixels(),
            MISSING_TEXTURE_SIZE,
            MISSING_TEXTURE_SIZE,
        );
        self.upload_texture(gpu, Self::SOLID_TEXTURE, &[255, 255, 255, 255], 1, 1);
    }

    /// The pixel size of an uploaded texture, if it is resident.
    #[must_use]
    pub fn texture_size(&self, key: &str) -> Option<(u32, u32)> {
        self.texture_entry(key).map(|t| (t.width, t.height))
    }

    /// Install the vanilla `entity_outline` post chain and its separate player-mask target.
    /// Missing resources are reported as `false` and leave the phase safe/no-op.
    pub fn install_outline(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        width: u32,
        height: u32,
    ) -> bool {
        let lazy = crate::w21_mr::lazy_outline_enabled();
        self.install_outline_with(gpu, resources, width, height, lazy)
    }

    /// [`Self::install_outline`] with the switch given explicitly.
    pub(crate) fn install_outline_with(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        width: u32,
        height: u32,
        lazy: bool,
    ) -> bool {
        let location = mc_assets::ResourceLocation::minecraft("shaders/post/entity_outline.json");
        let Ok(json) = resources.load_string(&location) else {
            return false;
        };
        let Ok(spec) = crate::post::ShaderGroupSpec::parse(&json) else {
            return false;
        };
        let (width, height) = (width.max(1), height.max(1));
        // The chain (its shaders, pipelines and target graph) is built
        // and checked here as before; with the switch on, its screen-sized targets and the mask
        // start at 1x1 and grow in `wake_outline` the first frame the outline is drawn.
        let dormant = lazy.then_some((width, height));
        let (built_width, built_height) = if dormant.is_some() {
            (1, 1)
        } else {
            (width, height)
        };
        let chain = crate::post::PostChain::new(
            gpu,
            resources,
            &spec,
            gpu.config.format,
            (built_width, built_height),
        );
        if chain.target_view("final").is_none() {
            return false;
        }
        let texture = outline_mask_texture(gpu, built_width, built_height);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.outline_mask = Some(OutlineMask {
            texture,
            view,
            width: built_width,
            height: built_height,
            chain,
            dormant,
        });
        self.outline_enabled = false;
        true
    }

    /// Grow a dormant mask and the chain's targets to the window size
    /// (the size `install_outline` and every later `resize` asked for). No-op once grown.
    fn wake_outline(&mut self, gpu: &Gpu) {
        let Some(mask) = &mut self.outline_mask else {
            return;
        };
        let Some((width, height)) = mask.dormant.take() else {
            return;
        };
        mask.chain.resize(gpu, width, height);
        mask.texture = outline_mask_texture(gpu, width, height);
        mask.view = mask
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        mask.width = width;
        mask.height = height;
    }

    /// Rebuild screen-sized outline resources after a window resize.
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        let Some(mask) = &mut self.outline_mask else {
            return;
        };
        let (width, height) = (width.max(1), height.max(1));
        // Not drawn yet: only remember the size to grow to.
        if let Some(wanted) = &mut mask.dormant {
            *wanted = (width, height);
            return;
        }
        mask.chain.resize(gpu, width, height);
        if mask.width == width && mask.height == height {
            return;
        }
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render spectator outline mask resized"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: gpu.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        mask.texture = texture;
        mask.view = view;
        mask.width = width;
        mask.height = height;
    }

    /// Whether spectator outlines are being drawn this frame.
    #[must_use]
    pub fn outline_enabled(&self) -> bool {
        self.outline_enabled
    }

    /// Set the spectator/key gate for the outline pass. The resource and shader gates are
    /// implicit in `install_outline`; the client supplies the live spectator/held conjunction.
    pub fn set_outline_enabled(&mut self, on: bool) {
        self.outline_enabled = on && self.outline_mask.is_some();
    }

    /// Set F3+B's debug-box flag.
    pub fn set_debug_bounding_box(&mut self, on: bool) {
        self.debug_enabled = on;
    }

    /// F3+B's debug-box flag as last set. Producers check it before building debug vertices
    /// that [`Self::push_debug_vertices`] would discard anyway.
    #[must_use]
    pub const fn debug_bounding_box(&self) -> bool {
        self.debug_enabled
    }

    /// Drops all per-frame geometry. Call once at the top of the frame.
    pub fn begin(&mut self) {
        if self.diagnostics_enabled {
            self.diagnostic_stats = EntityFrameStats::default();
        }
        self.verts.clear();
        self.vertex_lights.clear();
        self.batches.clear();
        self.fused_open = false;
        self.outline_verts.clear();
        self.debug_verts.clear();
        self.deferred.iter_mut().for_each(DeferredStream::clear);
        // One counter bump; maps drawn from here on are this frame's.
        self.deferred_maps
            .iter_mut()
            .for_each(DeferredStream::clear);
        self.map_atlas.begin_frame();
    }

    /// 06-entity-batch U5: opt into appending glint and contact-shadow pushes after all other
    /// entity geometry (one draw each instead of one per entity). Off by default: it changes the
    /// blend order against translucent draws and the 1/64-block shadow slab, so it is a visible
    /// deviation, however small. Also enabled by `MC_RS_ENTITY_DEFER_OVERLAYS=1` at construction.
    pub fn set_deferred_overlays(&mut self, enabled: bool) {
        self.defer_overlays = enabled;
    }

    /// Whether glint/shadow deferral (U5) is on.
    #[must_use]
    pub const fn deferred_overlays(&self) -> bool {
        self.defer_overlays
    }

    /// Queue a player's model geometry in the separate spectator-outline mask. The body is
    /// copied with an opaque team colour; normal texture/lightmap batches remain untouched.
    pub fn push_outline(&mut self, color: [f32; 4], verts: &[EntityVertex]) {
        if !self.outline_enabled || verts.is_empty() {
            return;
        }
        self.outline_verts
            .extend(verts.iter().copied().map(|mut vertex| {
                vertex.color = color;
                vertex
            }));
    }

    /// Queue one F3+B line segment as a pair of POSITION_COLOR vertices.
    pub fn push_debug_line(&mut self, line: DebugLine) {
        if !self.debug_enabled {
            return;
        }
        self.debug_verts.push(EntityVertex {
            pos: line.start.to_array(),
            color: line.color,
            ..EntityVertex::default()
        });
        self.debug_verts.push(EntityVertex {
            pos: line.end.to_array(),
            color: line.color,
            ..EntityVertex::default()
        });
    }

    /// Queue all F3+B segments for one visible entity.
    pub fn push_debug_geometry(&mut self, lines: &[DebugLine]) {
        for &line in lines {
            self.push_debug_line(line);
        }
    }

    /// Queue Java's strip-shaped F3+B vertices as line-list pairs for wgpu. The expansion keeps
    /// the source stream's three AABB strips (and, for living entities, three eye-box strips)
    /// separate so no artificial diagonal joins appear between the strips.
    pub fn push_debug_vertices(&mut self, vertices: &[DebugVertex], living: bool) {
        if !self.debug_enabled {
            return;
        }
        let expected = if living { 38 } else { 20 };
        if vertices.len() != expected {
            return;
        }
        let push_strip = |phase: &mut EntityPhase, strip: &[DebugVertex]| {
            for pair in strip.windows(2) {
                phase.push_debug_line(DebugLine {
                    start: pair[0].pos,
                    end: pair[1].pos,
                    color: pair[0].color,
                });
            }
        };
        push_strip(self, &vertices[0..5]);
        push_strip(self, &vertices[5..10]);
        // The final eight AABB vertices are the four vertical line-list pairs.
        for pair in vertices[10..18].chunks_exact(2) {
            self.push_debug_line(DebugLine {
                start: pair[0].pos,
                end: pair[1].pos,
                color: pair[0].color,
            });
        }
        let look_start = if living { 36 } else { 18 };
        if living {
            push_strip(self, &vertices[18..23]);
            push_strip(self, &vertices[23..28]);
            for pair in vertices[28..36].chunks_exact(2) {
                self.push_debug_line(DebugLine {
                    start: pair[0].pos,
                    end: pair[1].pos,
                    color: pair[0].color,
                });
            }
        }
        self.push_debug_line(DebugLine {
            start: vertices[look_start].pos,
            end: vertices[look_start + 1].pos,
            color: vertices[look_start].color,
        });
    }

    /// Appends one entity's vertices under `texture`, drawn opaquely.
    ///
    /// Consecutive pushes with the same texture merge into a single draw, so sorting entities by
    /// texture before pushing them is worth doing when a crowd shares a skin.
    pub fn push(&mut self, texture: &str, verts: &[EntityVertex]) {
        self.push_lit(texture, FULL_BRIGHT_COMBINED_LIGHT, verts);
    }

    /// Appends one opaque entity body with its packed `getBrightnessForRender` result.
    ///
    /// Callers that have world ownership must pass the result of
    /// [`entity_combined_light`], not their raw world sample, so the Wither/MagmaCube
    /// unconditional `0x00F0_00F0` override reaches the shader.  [`Self::push`] remains full
    /// bright for legacy object submissions which do not yet have a world-light query.
    pub fn push_lit(&mut self, texture: &str, combined_light: u32, verts: &[EntityVertex]) {
        self.push_blended_lit(texture, LayerBlend::Opaque, combined_light, verts);
    }

    /// Appends one batch under `texture` with an explicit blend mode.
    ///
    /// java: the `enableBlend`/`blendFunc`/`depthMask` triple each `LayerRenderer` wraps its own
    /// draw in. Batches merge only when both the texture *and* the blend mode match, because two
    /// different blend modes are two different pipelines and cannot share a draw call.
    ///
    /// Order matters and is the caller's: this phase draws batches in push order, which is the
    /// order vanilla's `renderModel` → `renderLayers` sequence produces, so a translucent or
    /// additive layer pushed after its body lands over it.
    pub fn push_blended(&mut self, texture: &str, blend: LayerBlend, verts: &[EntityVertex]) {
        self.push_blended_lit(texture, blend, FULL_BRIGHT_COMBINED_LIGHT, verts);
    }

    /// [`Self::push_blended`] with the packed lightmap coordinate active for this entity draw.
    ///
    /// java: `RenderManager.renderEntityStatic` sets this state immediately before it dispatches
    /// the renderer, and `RendererLivingEntity.renderLayers` inherits that same state.  This is
    /// therefore remains the fallback batch property for ordinary entity geometry: a body and its
    /// aura can share it, while world-model callers may opt into the parallel per-vertex stream.
    pub fn push_blended_lit(
        &mut self,
        texture: &str,
        blend: LayerBlend,
        combined_light: u32,
        verts: &[EntityVertex],
    ) {
        if verts.is_empty() {
            return;
        }
        let combined_light = layer_combined_light(texture, blend, combined_light);
        self.append_blended_lit(texture, blend, combined_light, verts, None);
    }

    /// [`Self::push_blended_lit`] with Java's packed lightmap coordinate per vertex.
    ///
    /// `BlockModelRenderer` writes four sky/block coordinates for every quad corner and the GPU
    /// linearly interpolates those coordinates across each triangle. `packed_lights` is already
    /// expanded into the submitted triangle-list order, so this method preserves that contract
    /// without changing [`EntityVertex`] or merging lighting into its primary colour.
    pub fn push_blended_lit_with_vertex_lights(
        &mut self,
        texture: &str,
        blend: LayerBlend,
        combined_light: u32,
        verts: &[EntityVertex],
        packed_lights: &[u32],
    ) {
        if verts.is_empty() {
            return;
        }
        if verts.len() != packed_lights.len() {
            debug_assert_eq!(
                verts.len(),
                packed_lights.len(),
                "per-vertex light stream must match entity geometry"
            );
            self.push_blended_lit(texture, blend, combined_light, verts);
            return;
        }
        let combined_light = layer_combined_light(texture, blend, combined_light);
        self.append_blended_lit(texture, blend, combined_light, verts, Some(packed_lights));
    }

    fn append_blended_lit(
        &mut self,
        texture: &str,
        blend: LayerBlend,
        combined_light: u32,
        verts: &[EntityVertex],
        packed_lights: Option<&[u32]>,
    ) {
        if self.defer_overlays {
            if let Some(slot) = deferred_overlay_slot(texture, blend) {
                let texture = self.texture_ids.intern(texture);
                let stream = &mut self.deferred[slot];
                append_draw(
                    &mut stream.verts,
                    &mut stream.lights,
                    &mut stream.batches,
                    texture,
                    blend,
                    combined_light,
                    verts,
                    packed_lights,
                    false,
                );
                return;
            }
        }
        let texture = self.batch_texture_id(texture);
        self.append_main(texture, blend, combined_light, verts, packed_lights, false);
    }

    /// Whether main-stream pushes compact as they arrive. Test builds
    /// also honour the `indexed_experiment` toggle, which turns compaction off altogether.
    fn fusion_active(&self) -> bool {
        #[cfg(test)]
        {
            self.fused_compact && self.indexed_experiment
        }
        #[cfg(not(test))]
        {
            self.fused_compact
        }
    }

    /// Append one push to the main streams: [`append_draw_fused`] while fusion is active,
    /// otherwise the original [`append_draw`].
    fn append_main(
        &mut self,
        texture: TextureId,
        blend: LayerBlend,
        combined_light: u32,
        verts: &[EntityVertex],
        packed_lights: Option<&[u32]>,
        separate: bool,
    ) {
        if self.fusion_active() {
            append_draw_fused(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
                &mut self.fused_open,
                texture,
                blend,
                combined_light,
                verts,
                packed_lights,
                separate,
            );
        } else {
            append_draw(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
                texture,
                blend,
                combined_light,
                verts,
                packed_lights,
                separate,
            );
        }
    }

    /// Intern `texture` for a batch. Consecutive pushes usually repeat the previous batch's key,
    /// so that case is one string compare and no hash.
    fn batch_texture_id(&mut self, texture: &str) -> TextureId {
        match self.batches.last() {
            Some(last) if self.texture_ids.key(last.texture) == texture => last.texture,
            _ => self.texture_ids.intern(texture),
        }
    }

    /// Appends two independent strips under one render state.
    ///
    /// Java's leash renderer submits two `GL_TRIANGLE_STRIP`s back-to-back.  A normal batch
    /// merge would connect them in one primitive and create a diagonal bridge, so the second
    /// strip is deliberately forced into its own draw batch even when its texture/blend/light
    /// tuple matches the first one.
    pub fn push_blended_lit_separate(
        &mut self,
        texture: &str,
        blend: LayerBlend,
        combined_light: u32,
        first: &[EntityVertex],
        second: &[EntityVertex],
    ) {
        let combined_light = layer_combined_light(texture, blend, combined_light);
        if !first.is_empty() {
            self.append_blended_lit(texture, blend, combined_light, first, None);
        }
        if second.is_empty() {
            return;
        }
        let texture = self.batch_texture_id(texture);
        self.append_main(texture, blend, combined_light, second, None, true);
    }

    // Item-frame map art.

    /// Where `map_id`'s image is drawn from in the shared map atlas ([`super::map_atlas`]),
    /// uploading it first when the atlas does not hold `revision` (`mc_world::MapData::revision`).
    ///
    /// `image` runs only for an upload: it returns the 128x128 RGBA8 pixels, or `None` when the
    /// map's colours cannot be converted, in which case an older upload of the same map stays
    /// (vanilla keeps its last `DynamicTexture`). `None` from this method means "draw the
    /// fallback texture". A world without framed maps never creates a page.
    pub fn map_atlas_slot<I: AsRef<[u8]>>(
        &mut self,
        gpu: &Gpu,
        map_id: i32,
        revision: u64,
        image: impl FnOnce() -> Option<I>,
    ) -> Option<super::map_atlas::MapImage> {
        use super::map_atlas::{Plan, PAGE_SIDE, SLOT_SIDE};
        let (slot, new_page, previous) = match self.map_atlas.plan(map_id, revision) {
            Plan::Ready(image) => return Some(image),
            Plan::Unavailable => return None,
            Plan::Upload {
                slot,
                new_page,
                previous,
            } => (slot, new_page, previous),
        };
        let Some(pixels) = image() else {
            if previous.is_none() {
                self.map_atlas.fail(map_id);
            }
            return previous;
        };
        let key = slot.texture_key();
        if new_page || !self.has_texture(key) {
            let blank = vec![0u8; (PAGE_SIDE * PAGE_SIDE * 4) as usize];
            self.upload_texture(gpu, key, &blank, PAGE_SIDE, PAGE_SIDE);
        }
        let opaque = self.map_atlas.stage(pixels.as_ref());
        let (x, y) = slot.origin();
        let upload = AtlasAnimationUpload {
            mip_level: 0,
            x,
            y,
            width: SLOT_SIDE,
            height: SLOT_SIDE,
            rgba: self.map_atlas.take_staging(),
        };
        self.upload_atlas_animation(gpu, key, &upload);
        self.map_atlas.restore_staging(upload.rgba);
        self.map_atlas.uploaded(map_id, opaque)
    }

    /// Queue map-image geometry to be appended after the rest of the producer's pushes, by
    /// [`Self::flush_deferred_maps`] (or at the latest by [`Self::finish`]).
    ///
    /// Only for **fully opaque** images under [`LayerBlend::MapTranslucentDepthWrite`]: with
    /// source alpha 1 that blend returns the source whatever is behind it, and the quad writes
    /// depth, so drawing it later changes nothing but the draw count. The images are queued per
    /// atlas page, so each page becomes one draw whatever order the maps came in, and the
    /// item-frame shells that used to sit between them merge too.
    pub fn push_map_deferred(
        &mut self,
        slot: super::map_atlas::MapSlot,
        blend: LayerBlend,
        combined_light: u32,
        verts: &[EntityVertex],
    ) {
        if verts.is_empty() {
            return;
        }
        let key = slot.texture_key();
        let combined_light = layer_combined_light(key, blend, combined_light);
        if self.deferred_maps.len() <= slot.page {
            self.deferred_maps
                .resize_with(slot.page + 1, DeferredStream::default);
        }
        let texture = match self.deferred_maps[slot.page]
            .batches
            .last()
            .map(|b| b.texture)
        {
            Some(texture) => texture,
            None => self.texture_ids.intern(key),
        };
        let stream = &mut self.deferred_maps[slot.page];
        append_draw(
            &mut stream.verts,
            &mut stream.lights,
            &mut stream.batches,
            texture,
            blend,
            combined_light,
            verts,
            None,
            false,
        );
    }

    /// Append the map images queued by [`Self::push_map_deferred`], page by page.
    pub fn flush_deferred_maps(&mut self) {
        if self
            .deferred_maps
            .iter()
            .all(|stream| stream.batches.is_empty())
        {
            return;
        }
        if self.fusion_active() {
            flush_deferred_overlays_fused(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
                &mut self.fused_open,
                &mut self.deferred_maps,
            );
            return;
        }
        flush_deferred_overlays(
            &mut self.verts,
            &mut self.vertex_lights,
            &mut self.batches,
            &mut self.deferred_maps,
        );
    }

    /// Forget every map image and free the atlas pages (world change, disconnect).
    pub fn clear_map_atlas(&mut self) {
        let pages = self.map_atlas.clear();
        for key in &super::map_atlas::PAGE_KEYS[..pages] {
            if let Some(id) = self.texture_ids.get(key) {
                if let Some(slot) = self.texture_slots.get_mut(id.index()) {
                    *slot = None;
                }
            }
        }
    }

    /// The map atlas's counters (pages, resident maps, uploads).
    #[must_use]
    pub fn map_atlas_stats(&self) -> super::map_atlas::MapAtlasStats {
        self.map_atlas.stats()
    }

    /// Uploads this frame's geometry. Call once after the last [`EntityPhase::push`].
    pub fn finish(&mut self, gpu: &Gpu) {
        // Queued map images the producer did not flush itself.
        self.flush_deferred_maps();
        // 06-entity-batch U5: deferred overlays (empty unless the option is on) join the frame
        // streams after everything pushed so far.
        let fused = self.fusion_active();
        if fused {
            flush_deferred_overlays_fused(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
                &mut self.fused_open,
                &mut self.deferred,
            );
        } else {
            flush_deferred_overlays(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
                &mut self.deferred,
            );
        }
        let use_indexed = {
            #[cfg(test)]
            {
                self.indexed_experiment
            }
            #[cfg(not(test))]
            {
                true
            }
        };
        if fused {
            // Quad pushes were compacted as they arrived; only raw
            // batches (strips, irregular geometry) still need the whole-batch test.
            compact_after_fusion(&mut self.verts, &mut self.vertex_lights, &mut self.batches);
        } else if use_indexed {
            indexed_experiment::compact(
                &mut self.verts,
                &mut self.vertex_lights,
                &mut self.batches,
            );
        }
        // Batches are final from here on: a later push starts a new one, as it always did.
        self.fused_open = false;
        let max_quads = self
            .batches
            .iter()
            .filter(|b| b.indexed)
            .map(|b| (b.end - b.start) / 4)
            .max()
            .unwrap_or(0);
        let index_upload_bytes = indexed_experiment::ensure_indices(
            gpu,
            max_quads,
            &mut self.quad_indices,
            &mut self.quad_index_capacity,
        );
        self.uploaded = upload_vertices(
            gpu,
            "mc-render entity vertices",
            &self.verts,
            &mut self.buffer,
            &mut self.capacity,
        );
        debug_assert_eq!(
            self.verts.len(),
            self.vertex_lights.len(),
            "entity vertex and light streams must stay aligned"
        );
        self.vertex_light_uploaded = upload_vertex_lights(
            gpu,
            &self.vertex_lights,
            &mut self.vertex_light_buffer,
            &mut self.vertex_light_capacity,
        );
        // One `bool` test a frame; the first frame that draws the
        // outline grows its targets to the window size before anything records into them.
        if self.outline_enabled {
            self.wake_outline(gpu);
        }
        self.outline_uploaded = upload_vertices(
            gpu,
            "mc-render spectator outline vertices",
            &self.outline_verts,
            &mut self.outline_buffer,
            &mut self.outline_capacity,
        );
        self.debug_uploaded = upload_vertices(
            gpu,
            "mc-render debug bounding-box vertices",
            &self.debug_verts,
            &mut self.debug_buffer,
            &mut self.debug_capacity,
        );

        // 06-entity-batch U1: every vertex carries its own packed coordinate, so the shader never
        // selects the uniform record and one constant record serves the whole pass. It is written
        // once when the buffer is created; nothing is uploaded per frame or per batch any more
        // (previously a zeroed `batches * 256`-byte Vec plus a `write_buffer` every frame).
        let lightmap_bytes = 0_u64;
        if !self.batches.is_empty() && self.lightmap_buffer.is_none() {
            let capacity = LIGHTMAP_UNIFORM_STRIDE;
            let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mc-render entity lightmaps"),
                size: capacity,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let uniform = LightmapUniform {
                combined_light: FULL_BRIGHT_COMBINED_LIGHT,
                _padding: [0; 3],
            };
            gpu.queue
                .write_buffer(&buffer, 0, bytemuck::bytes_of(&uniform));
            let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("mc-render entity lightmaps"),
                layout: &self.lightmap_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &buffer,
                            offset: 0,
                            size: NonZeroU64::new(core::mem::size_of::<LightmapUniform>() as u64),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.lightmap_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&self.lightmap_sampler),
                    },
                ],
            });
            self.lightmap_buffer = Some(buffer);
            self.lightmap_bind_group = Some(bind_group);
            self.lightmap_capacity = capacity;
        }
        if self.diagnostics_enabled {
            let capacity_total = self
                .capacity
                .saturating_add(self.vertex_light_capacity)
                .saturating_add(self.outline_capacity)
                .saturating_add(self.debug_capacity)
                .saturating_add(self.lightmap_capacity)
                .saturating_add(u64::from(self.quad_index_capacity) * 6 * 4);
            let vertex_upload_bytes = uploaded_payload_bytes(&self.verts);
            let light_upload_bytes = uploaded_payload_bytes(&self.vertex_lights);
            let outline_upload_bytes = uploaded_payload_bytes(&self.outline_verts);
            let debug_upload_bytes = uploaded_payload_bytes(&self.debug_verts);
            let upload_bytes = vertex_upload_bytes
                .saturating_add(light_upload_bytes)
                .saturating_add(outline_upload_bytes)
                .saturating_add(debug_upload_bytes)
                .saturating_add(lightmap_bytes)
                .saturating_add(index_upload_bytes);
            self.diagnostic_stats.record_finish(EntityFrameStats {
                finish_calls: 1,
                upload_bytes,
                vertex_upload_bytes,
                light_upload_bytes,
                index_upload_bytes,
                outline_upload_bytes,
                debug_upload_bytes,
                dynamic_uniform_upload_bytes: lightmap_bytes,
                vertex_count: self.vertex_count(),
                batch_count: self.batches.len(),
                gpu_capacity_bytes_total: capacity_total,
            });
        }
    }

    /// Logical input vertices queued this frame, expanding indexed quads to their original
    /// triangle-list references. Physical upload bytes remain separately measured.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.batches
            .iter()
            .map(indexed_experiment::logical_count)
            .sum()
    }

    /// Draw calls this frame will issue.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Camera-relative positions of the vertices queued this frame, in submission order.
    ///
    /// Read-only and never called by the renderer itself: it lets integration tests check
    /// *where* a body was drawn, not just that it was (e.g. that the local F5 body sits on the
    /// interpolated feet the camera orbits).
    pub fn vertex_positions(&self) -> impl Iterator<Item = [f32; 3]> + '_ {
        self.batches.iter().flat_map(move |batch| {
            (0..indexed_experiment::logical_count(batch))
                .map(move |i| self.verts[indexed_experiment::vertex_index(batch, i)].pos)
        })
    }

    /// Enable or disable the O1 upload counters. Disabled phases do not update the counters.
    pub fn set_diagnostics_enabled(&mut self, enabled: bool) {
        self.diagnostics_enabled = enabled;
        if !enabled {
            self.diagnostic_stats = EntityFrameStats::default();
        }
    }

    /// Return the current opt-in per-frame upload counters.
    #[must_use]
    pub fn diagnostic_stats(&self) -> Option<EntityFrameStats> {
        self.diagnostics_enabled.then_some(self.diagnostic_stats)
    }

    /// Turns the phase off without unregistering it. java: the `hideGUI`/spectator paths.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }
}

fn record_debug(phase: &EntityPhase, gpu: &Gpu, target: PassTarget<'_>) {
    let Some(buffer) = &phase.debug_buffer else {
        return;
    };
    if !phase.debug_enabled || phase.debug_uploaded == 0 {
        return;
    }
    // A pass of its own, or the shared world pass.
    let mut own = None;
    let pass = target.begin(gpu, "entity debug bounding boxes", &mut own);
    pass.set_pipeline(&phase.debug_pipeline);
    pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
    pass.set_vertex_buffer(0, buffer.slice(..));
    pass.draw(0..phase.debug_uploaded, 0..1);
}

/// Whether [`record_outline`] would record this frame: exactly its guards.
fn outline_records(phase: &EntityPhase) -> bool {
    // A dormant (1x1) mask never records; `finish` wakes it first.
    phase
        .outline_mask
        .as_ref()
        .is_some_and(|mask| mask.dormant.is_none())
        && phase.outline_buffer.is_some()
        && phase.outline_enabled
        && phase.outline_uploaded != 0
}

fn record_outline(
    phase: &EntityPhase,
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
) {
    let Some(mask) = &phase.outline_mask else {
        return;
    };
    let Some(buffer) = &phase.outline_buffer else {
        return;
    };
    // `mask.dormant`: see `outline_records`.
    if !phase.outline_enabled || phase.outline_uploaded == 0 || mask.dormant.is_some() {
        return;
    }
    {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render spectator outline mask"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &mask.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&phase.outline_pipeline);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_vertex_buffer(0, buffer.slice(..));
        pass.draw(0..phase.outline_uploaded, 0..1);
    }
    // `entity_outline.json` declares a `final` mask input and ping-pongs through `swap` and
    // `final`; copy and all post passes share this encoder so no GPU readback is involved.
    if !mask.chain.copy_to_target(enc, &mask.texture, "final") {
        return;
    }
    mask.chain.run(gpu, enc, &mask.view, &mask.view, 0.0);
    let Some(final_view) = mask.chain.target_view("final") else {
        return;
    };
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("mc-render spectator outline composite"),
        layout: &phase.composite_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(final_view),
        }],
    });
    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mc-render spectator outline composite"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
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
    pass.set_pipeline(&phase.composite_pipeline);
    pass.set_bind_group(0, &bind_group, &[]);
    pass.draw(0..3, 0..1);
}

impl RenderPhase for EntityPhase {
    fn name(&self) -> &'static str {
        "entity"
    }

    fn enabled(&self) -> bool {
        self.enabled
    }

    fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        EntityPhase::resize(self, gpu, width, height);
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        _cam: &Camera,
    ) {
        self.record_main(
            gpu,
            PassTarget::Own {
                enc: &mut *enc,
                view,
            },
        );
        record_debug(
            self,
            gpu,
            PassTarget::Own {
                enc: &mut *enc,
                view,
            },
        );
        record_outline(self, gpu, enc, view);
    }

    // The batches and the debug boxes join the world pass; the spectator
    // outline renders through its own targets and post passes, so a frame that draws one keeps
    // the phase on its own.
    fn shares_world_pass(&self) -> bool {
        !outline_records(self)
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_main(gpu, PassTarget::Shared(&mut *pass));
        record_debug(self, gpu, PassTarget::Shared(pass));
    }
}

impl EntityPhase {
    /// The entity batches, into a pass of their own or the shared world
    /// pass.
    fn record_main(&self, gpu: &Gpu, target: PassTarget<'_>) {
        if let (Some(buffer), Some(vertex_light_buffer), Some(lightmap_bind_group)) = (
            &self.buffer,
            &self.vertex_light_buffer,
            &self.lightmap_bind_group,
        ) {
            if self.uploaded != 0 && self.vertex_light_uploaded == self.uploaded {
                let mut own = None;
                let pass = target.begin(gpu, "entity", &mut own);
                pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
                pass.set_vertex_buffer(0, buffer.slice(..));
                pass.set_vertex_buffer(1, vertex_light_buffer.slice(..));
                if let Some(indices) = &self.quad_indices {
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                }
                // 06-entity-batch U1: one constant lightmap record for the whole pass (light is
                // per vertex). All 22 pipelines share one layout, so bindings survive
                // `set_pipeline` and an unchanged texture binding is not re-issued.
                pass.set_bind_group(2, lightmap_bind_group, &[0]);
                let fallback = self.texture_entry(&self.fallback);
                let mut bound: Option<usize> = None;
                let mut bound_texture: Option<&TextureEntry> = None;
                for batch in &self.batches {
                    let entry = self
                        .texture_slots
                        .get(batch.texture.index())
                        .and_then(Option::as_ref)
                        .or(fallback);
                    let Some(entry) = entry else {
                        continue;
                    };
                    // Only rebind when the blend mode actually changes. Layers are
                    // pushed immediately after their body, so an all-opaque frame still issues
                    // exactly one `set_pipeline`.
                    let index = pipeline_index(batch.blend);
                    if bound != Some(index) {
                        pass.set_pipeline(&self.pipelines[index]);
                        bound = Some(index);
                    }
                    if !bound_texture.is_some_and(|previous| std::ptr::eq(previous, entry)) {
                        pass.set_bind_group(1, &entry.bind_group, &[]);
                        bound_texture = Some(entry);
                    }
                    if batch.indexed {
                        pass.draw_indexed(
                            0..(batch.end - batch.start) / 4 * 6,
                            batch.start as i32,
                            0..1,
                        );
                    } else {
                        pass.draw(batch.start..batch.end, 0..1);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)]
    fn debug_vertex_stream_matches_java_nonliving_and_living_shapes() {
        let nonliving = debug_entity_vertices(
            Vec3::new(4.0, 5.0, 6.0),
            1.0,
            2.0,
            1.5,
            Vec3::new(0.0, 0.0, 4.0),
            false,
        );
        assert_eq!(nonliving.len(), 20, "AABB18 + look2");
        assert!(nonliving[..18]
            .iter()
            .all(|vertex| vertex.color == [1.0, 1.0, 1.0, 1.0]));
        assert!(nonliving[18..]
            .iter()
            .all(|vertex| vertex.color == [0.0, 0.0, 1.0, 1.0]));
        assert_eq!(nonliving[18].pos, Vec3::new(4.0, 6.5, 6.0));
        assert_eq!(nonliving[19].pos, Vec3::new(4.0, 6.5, 8.0));

        let living = debug_entity_vertices(
            Vec3::new(4.0, 5.0, 6.0),
            1.0,
            2.0,
            1.5,
            Vec3::new(0.0, 0.0, 4.0),
            true,
        );
        assert_eq!(living.len(), 38, "AABB18 + eye18 + look2");
        assert!(living[18..36]
            .iter()
            .all(|vertex| vertex.color == [1.0, 0.0, 0.0, 1.0]));
        assert!(living[36..]
            .iter()
            .all(|vertex| vertex.color == [0.0, 0.0, 1.0, 1.0]));
        assert!((living[18].pos.y - (6.5 - 0.009_999_999_776_482_582)).abs() < f32::EPSILON);
        assert!((living[23].pos.y - (6.5 + 0.009_999_999_776_482_582)).abs() < f32::EPSILON);
    }

    #[test]
    fn debug_vertex_stream_normalizes_look_without_changing_eye_origin() {
        let vertices =
            debug_entity_vertices(Vec3::ZERO, 2.0, 3.0, 1.0, Vec3::new(0.0, 0.0, 9.0), false);
        assert_eq!(vertices[18].pos, Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(vertices[19].pos, Vec3::new(0.0, 1.0, 2.0));
    }

    /// `EntityWither` and `EntityMagmaCube` have unconditional full-light overrides, while an
    /// ordinary entity must preserve the world coordinate that `Entity` returned. Both controls
    /// are needed: testing only the full-bright constants would pass if this helper made every
    /// entity glow.
    #[test]
    fn only_full_bright_entities_replace_world_light_with_the_java_coordinate() {
        let dark_world = 0;
        assert_eq!(
            entity_combined_light(EntityType::WitherBoss, dark_world),
            FULL_BRIGHT_COMBINED_LIGHT
        );
        assert_eq!(
            entity_combined_light(EntityType::LavaSlime, dark_world),
            FULL_BRIGHT_COMBINED_LIGHT
        );
        assert_eq!(FULL_BRIGHT_COMBINED_LIGHT, 15 << 20 | 15 << 4);

        let ordinary_world = 7 << 20 | 5 << 4;
        assert_eq!(
            entity_combined_light(EntityType::Cow, ordinary_world),
            ordinary_world,
            "a normal entity must retain Entity.getBrightnessForRender's world value"
        );
        assert_ne!(
            entity_combined_light(EntityType::Cow, ordinary_world),
            FULL_BRIGHT_COMBINED_LIGHT,
            "the normal control must not accidentally receive a full-bright override"
        );
    }

    /// java: `LayerEnderDragonEyes` hard-codes `i = 61680` before replaying the dragon model.
    /// The dark and already-lit controls prove that this is a layer-local override, rather than
    /// a second entity-wide brightness rule.
    #[test]
    fn dragon_eyes_force_java_61680_and_stay_full_bright_in_dark_cells() {
        let dark_world = 0;
        let ordinary_world = 7 << 20 | 5 << 4;

        assert_eq!(
            layer_combined_light(DRAGON_EYES_TEXTURE, LayerBlend::Additive, dark_world,),
            DRAGON_EYES_COMBINED_LIGHT
        );
        assert_eq!(
            layer_combined_light(
                DRAGON_EYES_TEXTURE,
                LayerBlend::AdditiveNoDepthWrite,
                ordinary_world,
            ),
            DRAGON_EYES_COMBINED_LIGHT
        );
        assert_eq!(DRAGON_EYES_COMBINED_LIGHT, 61_680);
        assert_eq!(entity_lightmap_brightness(DRAGON_EYES_COMBINED_LIGHT), 1.0);
        assert_eq!(entity_lightmap_brightness(dark_world), 0.05);
    }

    /// Negative controls for the dragon-eye state override: the ordinary dragon body and an
    /// unrelated additive texture retain the caller's world light, and an opaque reuse of the
    /// eye texture cannot accidentally inherit the temporary layer state.
    #[test]
    fn dragon_eye_light_override_does_not_bleed_into_ordinary_draws() {
        let dark_world = 0;
        assert_eq!(
            layer_combined_light(
                "textures/entity/enderdragon/dragon.png",
                LayerBlend::Opaque,
                dark_world,
            ),
            dark_world
        );
        assert_eq!(
            layer_combined_light(
                "textures/entity/zombie/zombie.png",
                LayerBlend::Additive,
                dark_world,
            ),
            dark_world
        );
        assert_eq!(
            layer_combined_light(DRAGON_EYES_TEXTURE, LayerBlend::Opaque, dark_world),
            dark_world
        );
    }

    /// `LayerEnderDragonEyes` keeps GL_ONE/GL_ONE and toggles only the depth-write policy for an
    /// invisible entity. Both descriptors remain LEQUAL-tested, with no depth bias or accidental
    /// ordinary alpha blend introduced by the full-bright lightmap override.
    #[test]
    fn dragon_eye_light_override_preserves_additive_blend_and_depth_descriptors() {
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        for (blend, depth_write) in [
            (LayerBlend::Additive, true),
            (LayerBlend::AdditiveNoDepthWrite, false),
        ] {
            let spec = additive_pipeline_spec(depth_write);
            let state = spec.blend.expect("dragon eyes enable additive blending");
            assert_eq!(spec.fragment_entry, "fs_unlit");
            assert_eq!(state.color, expected);
            assert_eq!(state.alpha, expected);
            assert_eq!(spec.depth_write, depth_write);
            assert_eq!(spec.depth_compare, wgpu::CompareFunction::LessEqual);
            assert_eq!(spec.depth_bias, wgpu::DepthBiasState::default());
            assert_eq!(pipeline_index(blend), if depth_write { 2 } else { 3 });
        }
        assert_ne!(
            wgpu::BlendState {
                color: expected,
                alpha: expected,
            },
            wgpu::BlendState::ALPHA_BLENDING
        );
    }

    /// This is deliberately a CPU/shader-shape test, not a headless wgpu draw: the observable
    /// contract is that the exact packed value is preserved to a per-draw binding and resolves
    /// to the brightest lightmap sample.  A real adapter is neither needed nor safe to require
    /// in this parallel render lane.
    #[test]
    fn wither_full_bright_coordinate_reaches_the_per_draw_lightmap_uniform() {
        let dark_world = 0;
        let wither_light = entity_combined_light(EntityType::WitherBoss, dark_world);
        assert_eq!(wither_light, FULL_BRIGHT_COMBINED_LIGHT);
        assert_eq!(entity_lightmap_brightness(wither_light), 1.0);
        assert_eq!(entity_lightmap_brightness(dark_world), 0.05);

        let ordinary = entity_combined_light(EntityType::Cow, 7 << 20 | 5 << 4);
        assert!(entity_lightmap_brightness(ordinary) < 1.0);
        assert!(ENTITY_WGSL.contains("@group(2) @binding(0) var<uniform> lightmap"));
        assert!(ENTITY_WGSL.contains("lightmap.combined_light"));
        assert!(ENTITY_WGSL.contains("@location(4) vertex_light: u32"));
        assert!(ENTITY_WGSL.contains("lightmap_color(in.lightmap)"));
        assert!(ENTITY_WGSL.contains("packed_lightmap_uv"));
    }

    #[test]
    fn entity_vertex_light_uv_keeps_java_block_x_sky_y_axes() {
        let mixed = (11_u32 << 20) | (4_u32 << 4);
        assert_eq!(
            entity_packed_lightmap_uv(mixed),
            [(4.0 * 16.0 + 8.0) / 256.0, (11.0 * 16.0 + 8.0) / 256.0]
        );
        assert_ne!(
            entity_packed_lightmap_uv(mixed),
            [(11.0 * 16.0 + 8.0) / 256.0, (4.0 * 16.0 + 8.0) / 256.0],
            "the low block axis must not be swapped with high sky"
        );
        assert!(ENTITY_WGSL.contains("let block = f32(packed & 0xffu)"));
        assert!(ENTITY_WGSL.contains("let sky = f32((packed >> 16u) & 0xffu)"));
    }

    /// java: `EntityMagmaCube.getBrightnessForRender` returns `15728880` and
    /// `EntityMagmaCube.getBrightness` returns `1.0`, even when the world sample is black.
    /// The ordinary entity control proves this narrow override does not make every mob glow.
    #[test]
    fn magma_cube_is_full_bright_in_a_dark_world_for_packed_and_scalar_consumers() {
        let dark_world = 0;
        let magma_light = entity_combined_light(EntityType::LavaSlime, dark_world);
        assert_eq!(magma_light, 15 << 20 | 15 << 4);
        assert_eq!(magma_light, FULL_BRIGHT_COMBINED_LIGHT);
        assert_eq!(entity_lightmap_brightness(magma_light), 1.0);

        let ordinary_light = entity_combined_light(EntityType::Cow, dark_world);
        assert_eq!(ordinary_light, dark_world);
        assert_eq!(entity_lightmap_brightness(ordinary_light), 0.05);
        assert_ne!(ordinary_light, FULL_BRIGHT_COMBINED_LIGHT);
        assert_ne!(entity_lightmap_brightness(ordinary_light), 1.0);
    }

    /// 06-entity-batch U1 replaced the per-batch lightmap uniform with the per-vertex stream:
    /// adjacent triangle-list pushes that differ only in light now share one draw, and the
    /// world-lit body still cannot inherit the preceding Wither's full-bright state because each
    /// vertex carries its own coordinate.
    #[test]
    fn different_lightmap_states_share_a_batch_but_keep_per_vertex_light() {
        let mut interner = TextureInterner::default();
        let wither = interner.intern("textures/entity/wither/wither.png");
        let (mut verts, mut lights, mut batches) = (Vec::new(), Vec::new(), Vec::new());
        let body = [EntityVertex::default(); 36];
        let world_lit = 7 << 20 | 5 << 4;
        for light in [FULL_BRIGHT_COMBINED_LIGHT, world_lit] {
            append_draw(
                &mut verts,
                &mut lights,
                &mut batches,
                wither,
                LayerBlend::Opaque,
                light,
                &body,
                None,
                false,
            );
        }
        assert_eq!(
            batches.len(),
            1,
            "light no longer splits a triangle-list draw"
        );
        assert_eq!((batches[0].start, batches[0].end), (0, 72));
        assert!(lights[..36]
            .iter()
            .all(|&l| l == FULL_BRIGHT_COMBINED_LIGHT));
        assert!(
            lights[36..].iter().all(|&l| l == world_lit),
            "a world-lit body must not inherit the preceding Wither's full-bright state"
        );
        assert!(!lights.contains(&NO_VERTEX_LIGHT));
    }

    /// The vertex stage computes `select(uniform, vertex_light, vertex_light != NO_VERTEX_LIGHT)`
    /// and the fragment stages read only the resulting `lightmap.xy`. Model the old phase
    /// (sentinel stream + one uniform per batch, light in the merge key) against the new one
    /// (expanded stream, constant uniform) over mixed pushes, and require every vertex's selected
    /// packed coordinate, every vertex position in the stream and every draw boundary's
    /// vertex order to be identical: the new batches must be unions of consecutive old batches.
    #[test]
    fn per_vertex_light_selects_exactly_the_old_per_batch_coordinate() {
        fn shader_packed(vertex_light: u32, uniform: u32) -> u32 {
            if vertex_light != NO_VERTEX_LIGHT {
                vertex_light
            } else {
                uniform
            }
        }
        let mut interner = TextureInterner::default();
        let a = interner.intern("a");
        let b = interner.intern("b");
        let corner = [1, 2, 3, NO_VERTEX_LIGHT, 5, 6];
        // (texture, blend, light, vertex count, explicit per-vertex stream)
        type Push<'a> = (TextureId, LayerBlend, u32, usize, Option<&'a [u32]>);
        let pushes: Vec<Push<'_>> = vec![
            (a, LayerBlend::Opaque, 0x00F0_00F0, 6, None),
            (a, LayerBlend::Opaque, 0x0050_0070, 12, None),
            (a, LayerBlend::Opaque, 0x0050_0070, 6, Some(&corner[..])),
            (b, LayerBlend::Translucent, 0, 6, None),
            (b, LayerBlend::Translucent, 0x0010_0020, 6, None),
            (a, LayerBlend::Glint, 0x00F0_00F0, 12, None),
            (a, LayerBlend::Opaque, 0, 6, None),
        ];

        // Old phase.
        let mut old_lights = Vec::new();
        let mut old_batches: Vec<(TextureId, LayerBlend, u32, u32, u32)> = Vec::new();
        let mut cursor = 0_u32;
        for &(texture, blend, light, count, packed) in &pushes {
            match packed {
                Some(lights) => old_lights.extend_from_slice(lights),
                None => old_lights.extend(std::iter::repeat_n(NO_VERTEX_LIGHT, count)),
            }
            let (start, end) = (cursor, cursor + count as u32);
            cursor = end;
            match old_batches.last_mut() {
                Some(last)
                    if last.0 == texture
                        && last.1 == blend
                        && last.2 == light
                        && last.4 == start =>
                {
                    last.4 = end;
                }
                _ => old_batches.push((texture, blend, light, start, end)),
            }
        }
        let old_selected: Vec<u32> = old_batches
            .iter()
            .flat_map(|&(_, _, light, start, end)| (start..end).map(move |i| (i, light)))
            .map(|(i, light)| shader_packed(old_lights[i as usize], light))
            .collect();

        // New phase.
        let (mut verts, mut lights, mut batches) = (Vec::new(), Vec::new(), Vec::new());
        for &(texture, blend, light, count, packed) in &pushes {
            let geometry: Vec<EntityVertex> = (0..count)
                .map(|i| EntityVertex {
                    pos: [verts.len() as f32 + i as f32, 0.0, 0.0],
                    ..EntityVertex::default()
                })
                .collect();
            append_draw(
                &mut verts,
                &mut lights,
                &mut batches,
                texture,
                blend,
                light,
                &geometry,
                packed,
                false,
            );
        }
        let new_selected: Vec<u32> = lights
            .iter()
            .map(|&light| shader_packed(light, FULL_BRIGHT_COMBINED_LIGHT))
            .collect();

        assert_eq!(
            new_selected, old_selected,
            "every vertex selects the old coordinate"
        );
        assert!(verts.iter().enumerate().all(|(i, v)| v.pos[0] == i as f32));
        // Every new boundary is an old boundary, with the same pipeline and texture on both sides.
        let old_bounds: Vec<u32> = old_batches.iter().map(|b| b.3).collect();
        for batch in &batches {
            assert!(old_bounds.contains(&batch.start));
            for old in old_batches
                .iter()
                .filter(|old| old.3 >= batch.start && old.4 <= batch.end)
            {
                assert_eq!((old.0, old.1), (batch.texture, batch.blend));
            }
        }
        assert_eq!(old_batches.len(), 6);
        assert_eq!(batches.len(), 4, "the two light-only splits merged");
    }

    /// Strip topologies keep the pre-U1 rule: concatenating two strips would add bridging
    /// primitives, so a light change still splits them, and the forced second leash strip stays
    /// separate even when everything matches.
    #[test]
    fn strip_topologies_never_newly_merge_across_a_light_change() {
        let mut interner = TextureInterner::default();
        let solid = interner.intern(EntityPhase::SOLID_TEXTURE);
        for blend in [LayerBlend::Leash, LayerBlend::FishingLine] {
            assert!(blend_is_strip(blend));
            let previous = DrawBatch {
                texture: solid,
                blend,
                combined_light: 0x0050_0070,
                start: 0,
                end: 8,
                indexed: false,
            };
            assert!(!batches_can_merge(&previous, solid, blend, 0, 8));
            assert!(batches_can_merge(&previous, solid, blend, 0x0050_0070, 8));
        }
        let (mut verts, mut lights, mut batches) = (Vec::new(), Vec::new(), Vec::new());
        let strip = [EntityVertex::default(); 8];
        append_draw(
            &mut verts,
            &mut lights,
            &mut batches,
            solid,
            LayerBlend::Leash,
            0,
            &strip,
            None,
            false,
        );
        append_draw(
            &mut verts,
            &mut lights,
            &mut batches,
            solid,
            LayerBlend::Leash,
            0,
            &strip,
            None,
            true,
        );
        assert_eq!(
            batches.len(),
            2,
            "the second leash strip is always its own draw"
        );
        assert!(!blend_is_strip(LayerBlend::Opaque));
        assert!(!blend_is_strip(LayerBlend::Translucent));
        assert!(!blend_is_strip(LayerBlend::Glint));
    }

    /// Interned ids are stable per key, distinct across keys, and round-trip to the key.
    #[test]
    fn texture_interner_assigns_stable_ids() {
        let mut interner = TextureInterner::default();
        let a = interner.intern("textures/entity/steve.png");
        let b = interner.intern("textures/misc/enchanted_item_glint.png");
        assert_ne!(a, b);
        assert_eq!(interner.intern("textures/entity/steve.png"), a);
        assert_eq!(
            interner.get("textures/misc/enchanted_item_glint.png"),
            Some(b)
        );
        assert_eq!(interner.get("missing"), None);
        assert_eq!(interner.key(a), "textures/entity/steve.png");
        assert_eq!((a.index(), b.index()), (0, 1));
    }

    /// 06-entity-batch U5 (opt-in): glints and contact shadows are queued per kind and appended
    /// after all other geometry -- glints first, each queue in push order -- so three dropped
    /// items become one body draw, one glint draw and one shadow draw, and every vertex keeps its
    /// own entity's light. Ordinary translucent bodies (ghosts, slime gel) are never deferred.
    #[test]
    fn deferred_overlays_append_glints_then_shadows_in_push_order() {
        const ATLAS: &str = "minecraft:textures/atlas/blocks.png";
        const GLINT: &str = "textures/misc/enchanted_item_glint.png";
        assert_eq!(deferred_overlay_slot(GLINT, LayerBlend::Glint), Some(0));
        assert_eq!(
            deferred_overlay_slot(DEFERRABLE_SHADOW_TEXTURE, LayerBlend::Translucent),
            Some(1)
        );
        assert_eq!(
            deferred_overlay_slot(DEFERRABLE_SHADOW_TEXTURE, LayerBlend::Opaque),
            None
        );
        assert_eq!(
            deferred_overlay_slot("textures/entity/slime/slime.png", LayerBlend::Translucent),
            None,
            "ghost and gel bodies keep their place"
        );

        let mut interner = TextureInterner::default();
        let (mut verts, mut lights, mut batches) = (Vec::new(), Vec::new(), Vec::new());
        let mut deferred: [DeferredStream; 2] = Default::default();
        let quad = |tag: f32| {
            [EntityVertex {
                pos: [tag, 0.0, 0.0],
                ..EntityVertex::default()
            }; 6]
        };
        for (item, light) in [
            (1.0_f32, 0x0010_0010_u32),
            (2.0, 0x0020_0020),
            (3.0, 0x0030_0030),
        ] {
            for (key, blend, tag) in [
                (ATLAS, LayerBlend::Opaque, item * 10.0),
                (
                    DEFERRABLE_SHADOW_TEXTURE,
                    LayerBlend::Translucent,
                    item * 10.0 + 1.0,
                ),
                (GLINT, LayerBlend::Glint, item * 10.0 + 2.0),
            ] {
                let texture = interner.intern(key);
                let (v, l, b) = match deferred_overlay_slot(key, blend) {
                    Some(slot) => {
                        let stream = &mut deferred[slot];
                        (&mut stream.verts, &mut stream.lights, &mut stream.batches)
                    }
                    None => (&mut verts, &mut lights, &mut batches),
                };
                append_draw(v, l, b, texture, blend, light, &quad(tag), None, false);
            }
        }
        flush_deferred_overlays(&mut verts, &mut lights, &mut batches, &mut deferred);
        let atlas = interner.intern(ATLAS);
        let glint = interner.intern(GLINT);
        let shadow = interner.intern(DEFERRABLE_SHADOW_TEXTURE);
        assert_eq!(
            batches
                .iter()
                .map(|b| (b.texture, b.blend, b.start, b.end))
                .collect::<Vec<_>>(),
            vec![
                (atlas, LayerBlend::Opaque, 0, 18),
                (glint, LayerBlend::Glint, 18, 36),
                (shadow, LayerBlend::Translucent, 36, 54),
            ]
        );
        let tags: Vec<f32> = verts.iter().step_by(6).map(|v| v.pos[0]).collect();
        assert_eq!(
            tags,
            vec![10.0, 20.0, 30.0, 12.0, 22.0, 32.0, 11.0, 21.0, 31.0]
        );
        let quad_lights: Vec<u32> = lights.iter().step_by(6).copied().collect();
        assert_eq!(
            quad_lights,
            [0x0010_0010, 0x0020_0020, 0x0030_0030].repeat(3)
        );
        assert!(deferred
            .iter()
            .all(|stream| stream.verts.is_empty() && stream.batches.is_empty()));
    }

    #[test]
    fn entity_glint_pipeline_is_src_color_plus_one_and_depth_equal() {
        let glint = glint_blend_state();
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Src,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        assert_eq!(
            glint.color, expected,
            "RenderItem/LayerArmorBase colour blend"
        );
        assert_eq!(glint.alpha, expected, "glBlendFunc applies to alpha too");
        assert_eq!(pipeline_index(LayerBlend::Glint), 7);
        assert!(ENTITY_WGSL.contains("fn fs_glint"));
        assert!(ENTITY_WGSL.contains("texel.rgb * in.color.rgb"));
        assert!(!glint.eq(&wgpu::BlendState::ALPHA_BLENDING));

        // GL_EQUAL is materially different from the ordinary entity LEQUAL path: the foil must
        // replay only the model's existing depth, not paint through an intervening surface.
        assert_eq!(glint_depth_compare(), wgpu::CompareFunction::Equal);
        assert_eq!(ordinary_depth_compare(), wgpu::CompareFunction::LessEqual);

        // Negative control: source-colour/one is not source-alpha/one-minus-alpha for a mixed
        // source/destination pair, so a translucent approximation cannot satisfy this oracle.
        let src = 0.5_f32;
        let dst = 0.25_f32;
        let exact = src * src + dst;
        let alpha_mutant = src * src + dst * (1.0 - src);
        assert!((exact - 0.5).abs() < f32::EPSILON);
        assert!((alpha_mutant - 0.375).abs() < f32::EPSILON);
        assert_ne!(exact, alpha_mutant);
    }

    #[test]
    fn invisible_eye_pipeline_is_additive_lequal_without_depth_writes() {
        let hidden = additive_pipeline_spec(false);
        let visible = additive_pipeline_spec(true);
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };

        for spec in [hidden, visible] {
            assert_eq!(spec.fragment_entry, "fs_unlit");
            let blend = spec.blend.expect("glowing eyes enable blending");
            assert_eq!(blend.color, expected);
            assert_eq!(blend.alpha, expected);
            assert_eq!(spec.depth_compare, wgpu::CompareFunction::LessEqual);
        }
        assert!(!hidden.depth_write);
        assert!(visible.depth_write);
        assert_eq!(pipeline_index(LayerBlend::AdditiveNoDepthWrite), 3);
    }

    /// java: `RenderGuardian.doRender` and `RenderDragon.drawRechargeRay` draw their beams with
    /// blending off, the world pass's `GL_GREATER 0.1` alpha test, lighting off, the lightmap
    /// stage on, depth writes and `GL_LEQUAL`, culling off. The guardian's was `SRC_ALPHA, ONE`
    /// blended, the dragon's untested and unlit: washed-out beams whose transparent texels drew or
    /// wrote depth.
    #[test]
    fn beams_are_opaque_alpha_tested_unlit_and_lightmapped() {
        let spec = beam_pipeline_spec();
        assert_eq!(spec.fragment_entry, "fs_unlit_cutout");
        assert_eq!(spec.blend, None);
        assert!(spec.depth_write);
        assert_eq!(spec.depth_compare, wgpu::CompareFunction::LessEqual);
        for blend in [LayerBlend::GuardianBeam, LayerBlend::DragonBeam] {
            assert_eq!(entity_cull_mode(blend), None, "{blend:?}");
            let (_, arm) = include_str!("render.rs")
                .split_once(&format!("LayerBlend::{blend:?} => {{"))
                .expect("the beam's pipeline arm");
            assert!(
                arm[..120].contains("beam_pipeline_spec()"),
                "{blend:?} must use the shared beam state"
            );
        }
        assert_eq!(pipeline_index(LayerBlend::GuardianBeam), 19);
        assert_eq!(pipeline_index(LayerBlend::DragonBeam), 16);
        let (_, cutout) = ENTITY_WGSL
            .split_once("fn fs_unlit_cutout")
            .expect("the cutout fragment entry");
        let (cutout, _) = cutout
            .split_once("@fragment")
            .expect("another entry follows");
        assert!(cutout.contains("if (texel.a <= 0.1)"));
        assert!(cutout.contains("discard;"));
        assert!(cutout.contains("lightmap_color(in.lightmap)"));
        assert!(!cutout.contains("diffuse("), "lighting is off");
    }

    /// java: `LayerEnderDragonDeath` draws its rays additively (`SRC_ALPHA, ONE`) without depth
    /// writes, with `enableCull()` (back faces culled) and texturing off but the lightmap stage
    /// still on at the dragon's own light. They were drawn two-sided and unlit.
    #[test]
    fn dragon_death_rays_are_culled_additive_and_lightmapped() {
        assert_eq!(
            entity_cull_mode(LayerBlend::DragonDeathShards),
            Some(wgpu::Face::Back)
        );
        let blend = dragon_death_shards_blend_state();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::One);
        let (_, shards) = ENTITY_WGSL
            .split_once("fn fs_dragon_death_shards")
            .expect("the death ray fragment entry");
        assert!(shards.contains("in.color.rgb * lightmap_color(in.lightmap)"));
        assert!(!shards.contains("textureSample"), "texturing is off");
        assert!(
            !ENTITY_WGSL.contains("fn fs_dragon_beam"),
            "no beam bypasses the lightmap"
        );
    }

    /// java: `RenderGuardian.doRender` sets `GL_REPEAT` on both axes after binding
    /// `guardian_beam.png`, overriding a pack's `.mcmeta` clamp but not its blur.
    #[test]
    fn the_guardian_beam_texture_repeats_whatever_its_mcmeta_says() {
        let pack = TextureMetadata {
            blur: true,
            clamp: true,
            ..TextureMetadata::default()
        };
        let beam = resource_texture_options(crate::entity::GUARDIAN_BEAM_TEXTURE, pack.clone());
        assert!(!beam.clamp);
        assert!(beam.blur);
        assert_eq!(
            resource_texture_options("textures/entity/guardian.png", pack.clone()),
            pack,
            "every other entity texture keeps its sidecar flags"
        );
    }

    #[test]
    fn living_white_flash_is_affine_alpha_over_at_equal_depth() {
        let spec = white_flash_pipeline_spec();
        assert_eq!(spec.fragment_entry, "fs_living_white_flash");
        assert_eq!(spec.blend, Some(wgpu::BlendState::ALPHA_BLENDING));
        assert!(!spec.depth_write);
        assert_eq!(spec.depth_compare, wgpu::CompareFunction::Equal);
        assert_eq!(pipeline_index(LayerBlend::WhiteFlash), 12);
        let (_, flash) = ENTITY_WGSL
            .split_once("fn fs_living_white_flash")
            .expect("dedicated textured living white-flash fragment");
        let (flash, _) = flash
            .split_once("// java: the additive layers")
            .expect("white flash ends before additive layers");
        assert!(flash.contains("textureSample(tex, samp, in.uv)"));
        assert!(flash.contains("if (texel.a <= 0.1)"));
        assert!(flash.contains("discard;"));
        assert!(flash.contains("lightmap_color(in.lightmap)"));
        assert!(flash.contains("texel.a * in.color.a"));

        let old = 0.2_f32;
        let weight = 0.25_f32;
        let affine = 1.0 * weight + old * (1.0 - weight);
        let white_multiply_mutant = old * 1.0;
        assert!((affine - 0.4).abs() < f32::EPSILON);
        assert_eq!(white_multiply_mutant, old);
        assert_ne!(affine, white_multiply_mutant);
    }

    #[test]
    fn living_hurt_is_textured_affine_with_source_depth_alpha_and_glint_policies() {
        let opaque = living_hurt_pipeline_spec(false);
        assert_eq!(opaque.fragment_entry, "fs_living_hurt");
        assert_eq!(opaque.blend, None);
        assert!(opaque.depth_write);
        assert_eq!(opaque.depth_compare, wgpu::CompareFunction::LessEqual);

        // java: the slime gel's hurt replay keeps `LayerSlimeGel`'s depth mask, which is on.
        let translucent = living_hurt_pipeline_spec(true);
        assert_eq!(translucent.fragment_entry, "fs_living_hurt");
        assert_eq!(translucent.blend, Some(wgpu::BlendState::ALPHA_BLENDING));
        assert!(translucent.depth_write);
        assert_eq!(translucent.depth_compare, wgpu::CompareFunction::LessEqual);
        assert_eq!(pipeline_index(LayerBlend::LivingHurt), 8);
        assert_eq!(pipeline_index(LayerBlend::LivingHurtTranslucent), 9);

        let ghost = living_hurt_ghost_pipeline_spec();
        assert_eq!(ghost.fragment_entry, "fs_living_hurt_ghost");
        assert_eq!(ghost.blend, Some(wgpu::BlendState::ALPHA_BLENDING));
        assert!(!ghost.depth_write);
        assert_eq!(ghost.depth_compare, wgpu::CompareFunction::LessEqual);
        assert_eq!(pipeline_index(LayerBlend::LivingHurtGhost), 10);

        let hurt_glint = living_hurt_glint_pipeline_spec();
        assert_eq!(hurt_glint.fragment_entry, "fs_living_hurt_glint");
        assert_eq!(hurt_glint.blend, Some(glint_blend_state()));
        assert!(!hurt_glint.depth_write);
        assert_eq!(hurt_glint.depth_compare, wgpu::CompareFunction::Equal);
        assert_eq!(pipeline_index(LayerBlend::LivingHurtGlint), 11);

        assert!(ENTITY_WGSL.contains("fn fs_living_hurt"));
        assert!(ENTITY_WGSL.contains("fn fs_living_hurt_ghost"));
        assert!(ENTITY_WGSL.contains("fn fs_living_hurt_glint"));
        assert!(ENTITY_WGSL.contains("let texel = textureSample(tex, samp, in.uv)"));
        assert!(ENTITY_WGSL.contains("if (texel.a <= 0.1)"));
        assert!(ENTITY_WGSL.contains("if (texel.a <= 0.003921569)"));
        assert!(ENTITY_WGSL.contains("mix(texel.rgb * primary, vec3<f32>(1.0, 0.0, 0.0), 0.3)"));
        assert!(ENTITY_WGSL.contains("texel.a * in.color.a"));

        let blue = [0.0_f32, 0.0, 1.0];
        let exact = [1.0 * 0.3 + blue[0] * 0.7, blue[1] * 0.7, blue[2] * 0.7];
        let multiplicative_mutant = [blue[0], 0.0, 0.0];
        assert_eq!(exact, [0.3, 0.0, 0.7]);
        assert_ne!(exact, multiplicative_mutant);
    }

    #[test]
    fn tnt_flash_pipeline_is_src_alpha_dst_alpha_and_depth_biased() {
        let spec = tnt_flash_pipeline_spec();
        // java: `blendFunc(770, 772)` = GL_SRC_ALPHA, GL_DST_ALPHA.
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::DstAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let blend = spec.blend.expect("TNT flash enables blending");
        assert_eq!(blend.color, expected);
        assert_eq!(blend.alpha, expected);
        assert_eq!(spec.fragment_entry, "fs_tnt_flash");
        assert!(spec.depth_write);
        assert_eq!(spec.depth_compare, wgpu::CompareFunction::LessEqual);
        assert_eq!(spec.depth_bias.constant, -3);
        assert_eq!(spec.depth_bias.slope_scale, -3.0);
        assert_eq!(spec.depth_bias.clamp, 0.0);
        assert_eq!(pipeline_index(LayerBlend::TntFlash), 13);
        assert!(ENTITY_WGSL.contains("fn fs_tnt_flash"));
        // The lightmap stage stays on (only unit 0 is disabled).
        let (_, flash) = ENTITY_WGSL
            .split_once("fn fs_tnt_flash")
            .expect("the flash fragment exists");
        let flash = flash.split("@fragment").next().unwrap_or(flash);
        assert!(flash.contains("lightmap_color(in.lightmap)"));

        let atlas = TextureInterner::default().intern("minecraft:textures/atlas/blocks.png");
        let body = DrawBatch {
            texture: atlas,
            blend: LayerBlend::Opaque,
            combined_light: 0,
            start: 0,
            end: 36,
            indexed: false,
        };
        assert!(!batches_can_merge(
            &body,
            atlas,
            LayerBlend::TntFlash,
            0,
            36,
        ));
        assert_ne!(blend, wgpu::BlendState::ALPHA_BLENDING);
    }

    #[test]
    fn wither_spawn_scale_uses_the_watched_tick_count_and_partial_tick() {
        assert_eq!(wither_pre_render_scale(0, 0.75), 2.0);
        assert_eq!(wither_pre_render_scale(220, 0.0), 1.5);
        assert_eq!(
            wither_pre_render_scale(220, 0.5),
            2.0 - ((220.0 - 0.5) / 220.0) * 0.5
        );
    }

    fn standard_item_light(normal: Vec3) -> f32 {
        let light0 = Vec3::new(0.2, 1.0, -0.7).normalize();
        let light1 = Vec3::new(-0.2, 1.0, 0.7).normalize();
        0.4 + 0.6 * normal.normalize().dot(light0).max(0.0)
            + 0.6 * normal.normalize().dot(light1).max(0.0)
    }

    #[test]
    fn tinted_item_lighting_clamps_the_lit_primary_colour_before_texture_modulation() {
        // RenderHelper's symmetric lights add above 1.0 on an upward face. With
        // GL_COLOR_MATERIAL(AMBIENT_AND_DIFFUSE), fixed-function GL lights the primary colour,
        // clamps that result, and only then modulates it by the texture. Clamping the scalar
        // first loses the extra brightness on every non-white channel.
        let light = standard_item_light(Vec3::Y);
        assert!(light > 1.0, "up-facing standard-item light was {light}");

        let tint = Vec3::new(0.2, 0.8, 1.0);
        let vanilla = (tint * light).clamp(Vec3::ZERO, Vec3::ONE);
        let old_order = tint * light.min(1.0);
        assert!(vanilla.x > old_order.x);
        assert_eq!(vanilla.y, 1.0);
        assert_eq!(vanilla.z, 1.0);
        assert_ne!(vanilla, old_order);

        // This assertion binds the arithmetic proof to the shader that production compiles.
        assert!(ENTITY_WGSL.contains("diffuse(in.normal) * in.color.rgb"));
        assert!(ENTITY_WGSL.contains("let lit_color = clamp("));
        assert!(ENTITY_WGSL.contains("texel.rgb * lit_color"));
        assert!(!ENTITY_WGSL.contains("return min(light, 1.0)"));
        assert!(!ENTITY_WGSL.contains("texel.rgb * diffuse(in.normal) * in.color.rgb"));
        // The lightmap is texture unit 1, a modulate stage after the clamp; inside it, an upward
        // face under a night lightmap was 37% too bright. The pixel-level proof against
        // fixed-function GL is `tests/l12_entity_lighting.rs`.
        assert!(ENTITY_WGSL.contains("texel.rgb * lit_color * lightmap_color(in.lightmap)"));
        assert!(!ENTITY_WGSL.contains("diffuse(in.normal) * in.color.rgb * lightmap_color"));
    }

    #[test]
    fn clamp_order_is_identical_when_standard_item_light_does_not_overbrighten() {
        let tint = Vec3::new(0.2, 0.8, 1.0);
        for normal in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z, Vec3::NEG_Y] {
            let light = standard_item_light(normal);
            assert!(light <= 1.0, "control normal {normal:?} had light {light}");
            let vanilla = (tint * light).clamp(Vec3::ZERO, Vec3::ONE);
            let scalar_first = tint * light.min(1.0);
            assert!(
                (vanilla - scalar_first).length_squared() < f32::EPSILON,
                "control normal {normal:?}: {vanilla:?} != {scalar_first:?}"
            );
        }
    }

    /// java: `RenderSlime.preRenderCallback` (`:29-35`). The `size` factor is the part that
    /// matters most: a size-4 magma cube drawn with `EntityRenderer::scale`'s static `1.0` is a
    /// **quarter** of its correct width inside its correct hitbox.
    #[test]
    fn the_slime_pre_render_scale_is_size_times_a_side_profile_preserving_squash() {
        // At rest (`squishFactor == 0`) the scale is exactly the slime's size on all three axes.
        for size in 1..=8 {
            let v = slime_pre_render_scale(size, 0.0);
            assert_eq!(v, Vec3::splat(size as f32), "size {size} at rest");
        }
        // An unset watched value must not collapse the model. java: `entityInit` seeds 16 with 1.
        assert_eq!(slime_pre_render_scale(0, 0.0), Vec3::splat(1.0));

        // Squashing (a negative factor — the landing) widens X/Z and shortens Y; stretching does
        // the opposite. X and Y are exact reciprocals about `size`, so the *side profile* area
        // `x * y` is conserved at `size^2` — that is what makes a slime read as jelly. Note the
        // volume is **not** conserved: Z takes the same `f2` as X, so it is `f2 * size^3`. An
        // assertion that the volume is conserved is wrong and this test caught exactly that.
        for (label, squish) in [("landing", -0.165_f32), ("take-off", 0.33)] {
            let v = slime_pre_render_scale(2, squish);
            assert!(
                (v.x * v.y - 4.0).abs() < 1e-4,
                "{label}: the side profile is conserved, got {}",
                v.x * v.y
            );
            assert_eq!(v.x, v.z);
            if squish < 0.0 {
                assert!(v.x > 2.0 && v.y < 2.0, "{label}: {v:?}");
            } else {
                assert!(v.x < 2.0 && v.y > 2.0, "{label}: {v:?}");
            }
        }

        // java, worked through: size 4, squish 0.33 -> f1 = 0.33/3 = 0.11, f2 = 1/1.11 = 0.9009.
        let v = slime_pre_render_scale(4, 0.33);
        assert!((v.x - 0.900_900_9 * 4.0).abs() < 1e-4, "{v:?}");
        assert!((v.y - 4.0 / 0.900_900_9).abs() < 1e-4, "{v:?}");
    }

    /// java: `RendererLivingEntity.prepareScale` (`:230-236`) runs `scale(-1,-1,1)`, then
    /// `preRenderCallback`, then `translate(0, -1.5078125, 0)` — the lift is **inside** the
    /// scale. Composing the scale after the finished matrix instead would leave the model's feet
    /// off the ground by `1.5078125 * (scale - 1)` blocks, which for a size-4 slime is 4.5.
    #[test]
    fn the_pre_render_scale_is_composed_before_the_model_lift() {
        let scaled = entity_matrix_scaled(&EntityTransform::default(), Vec3::splat(4.0));
        let plain = entity_matrix(&EntityTransform::default());
        // The model origin (0,0,0 in model space) is the entity's feet in both, because the lift
        // is scaled along with everything else and the flip puts it back.
        let origin_scaled = scaled.transform_point3(Vec3::ZERO);
        let origin_plain = plain.transform_point3(Vec3::ZERO);
        assert!(
            (origin_scaled.y - origin_plain.y * 4.0).abs() < 1e-4,
            "the lift scales with the model: {origin_scaled:?} vs {origin_plain:?}"
        );
        // And `Vec3::ONE` really is the identity: `entity_matrix` is the same matrix.
        assert_eq!(
            plain,
            entity_matrix_scaled(&EntityTransform::default(), Vec3::ONE)
        );
    }

    /// `load_texture` is what closed the gap where **no** entity texture was ever uploaded in
    /// production — the descriptor table's paths were computed and dropped, so every mob drew on
    /// the 1x1 white fallback. A resource pack that does not have the file must not panic, must
    /// not register a key, and must leave the caller free to fall back.
    #[test]
    fn a_missing_entity_texture_is_a_false_not_a_panic() {
        use crate::context::GpuOptions;
        let gpu = match crate::context::Gpu::headless(
            8,
            8,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => g,
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        };
        let mut phase = EntityPhase::new(&gpu);
        let resources = mc_assets::ResourceManager::new();

        assert!(!phase.load_texture(&gpu, &resources, "textures/entity/creeper/creeper.png"));
        assert!(!phase.has_texture("textures/entity/creeper/creeper.png"));
        // A syntactically invalid location is refused the same way, not unwrapped.
        assert!(!phase.load_texture(&gpu, &resources, "Not A Location"));
        // The fallback the caller is meant to use instead is always resident.
        assert!(phase.has_texture(EntityPhase::FALLBACK_TEXTURE));
        assert_eq!(
            phase.texture_size(EntityPhase::FALLBACK_TEXTURE),
            Some((MISSING_TEXTURE_SIZE, MISSING_TEXTURE_SIZE))
        );

        // An already-resident key short-circuits without touching the resource manager, which is
        // what keeps this off the per-frame cost after the first use.
        phase.upload_texture(&gpu, "already/there.png", &[1, 2, 3, 4], 1, 1);
        assert!(phase.load_texture(&gpu, &resources, "already/there.png"));
        assert_eq!(phase.texture_size("already/there.png"), Some((1, 1)));
    }

    #[test]
    fn entity_diagnostics_are_opt_in_idempotent_and_reset_on_begin() {
        use crate::context::GpuOptions;

        let gpu = match crate::context::Gpu::headless(
            8,
            8,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => g,
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        };
        let mut phase = EntityPhase::new(&gpu);

        assert_eq!(phase.diagnostic_stats(), None);
        phase.set_diagnostics_enabled(true);
        phase.set_diagnostics_enabled(true);
        assert_eq!(phase.diagnostic_stats(), Some(EntityFrameStats::default()));

        // An empty finish is still a real finish call, with no submitted payload.
        phase.begin();
        phase.finish(&gpu);
        let empty = phase.diagnostic_stats().expect("diagnostics enabled");
        assert_eq!(empty.finish_calls, 1);
        assert_eq!(empty.upload_bytes, 0);
        assert_eq!((empty.vertex_count, empty.batch_count), (0, 0));

        // Repeated finishes retain the accumulated geometry/batch identity while counting the
        // duplicate full-buffer writes for attribution.
        let vertices = [EntityVertex::default(); 3];
        phase.push(EntityPhase::FALLBACK_TEXTURE, &vertices);
        phase.finish(&gpu);
        let first = phase.diagnostic_stats().expect("diagnostics enabled");
        assert_eq!(first.finish_calls, 2);
        assert_eq!((first.vertex_count, first.batch_count), (3, 1));
        assert!(first.upload_bytes > 0);
        let geometry_after_first = (phase.vertex_count(), phase.batch_count());

        phase.finish(&gpu);
        let repeated = phase.diagnostic_stats().expect("diagnostics enabled");
        assert_eq!(repeated.finish_calls, 3);
        assert_eq!(
            (repeated.vertex_count, repeated.batch_count),
            geometry_after_first
        );
        assert_eq!(
            (phase.vertex_count(), phase.batch_count()),
            geometry_after_first
        );
        assert_eq!(repeated.upload_bytes, first.upload_bytes * 2);

        phase.begin();
        let reset = phase.diagnostic_stats().expect("diagnostics enabled");
        assert_eq!(reset, EntityFrameStats::default());
        assert_eq!((phase.vertex_count(), phase.batch_count()), (0, 0));

        phase.set_diagnostics_enabled(false);
        assert_eq!(phase.diagnostic_stats(), None);
    }

    /// The live horse path must decode every selected sheet, compose transparent pixels, cache
    /// the generated binding, and permit a same-key replacement after a resource reload. This is
    /// intentionally headless: the GPU upload itself is the lifecycle boundary under test.
    #[test]
    fn horse_resource_layers_upload_cache_and_replace_transactionally() {
        use crate::context::GpuOptions;
        use image::ImageEncoder;
        use std::io::Cursor;

        let gpu = match crate::context::Gpu::headless(
            8,
            8,
            GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: false,
            },
        ) {
            Ok(g) => g,
            Err(crate::RenderError::NoAdapter) => {
                eprintln!("SKIP: no wgpu adapter on this machine");
                return;
            }
            Err(e) => panic!("headless context creation failed: {e}"),
        };

        fn png(rgba: [u8; 4]) -> Vec<u8> {
            let mut bytes = Cursor::new(Vec::new());
            image::codecs::png::PngEncoder::new(&mut bytes)
                .write_image(&rgba, 1, 1, image::ExtendedColorType::Rgba8)
                .expect("encode 1x1 png");
            bytes.into_inner()
        }

        let temp = tempfile::tempdir().expect("temp resource pack");
        let horse_root = temp.path().join("assets/minecraft/textures/entity/horse");
        let armour_root = horse_root.join("armor");
        std::fs::create_dir_all(&armour_root).expect("horse resource directories");
        let base_path = horse_root.join("horse_white.png");
        let armour_path = armour_root.join("horse_armor_diamond.png");
        std::fs::write(&base_path, png([10, 20, 30, 255])).expect("base png");
        std::fs::write(&armour_path, png([200, 150, 100, 128])).expect("armour png");

        let mut resources = mc_assets::ResourceManager::new();
        resources.push_pack(Box::new(
            mc_assets::FolderResourcePack::open(temp.path()).expect("folder pack"),
        ));
        let selection = super::super::manager::horse_texture_selection(0, 0, 3);
        let key = selection.upload_key().to_owned();
        let mut phase = EntityPhase::new(&gpu);
        assert!(phase.load_horse_texture(&gpu, &resources, &selection));
        assert_eq!(phase.texture_size(&key), Some((1, 1)));
        assert!(phase.load_horse_texture(&gpu, &resources, &selection));

        // A malformed replacement is transactional: the old cache is explicitly cleared by the
        // reload boundary, and a failed decode must not recreate a half-composed generated key.
        std::fs::write(&armour_path, b"not a png").expect("malformed replacement");
        phase.clear_textures(&gpu);
        assert!(!phase.has_texture(&key));
        assert!(!phase.load_horse_texture(&gpu, &resources, &selection));
        assert!(!phase.has_texture(&key));
        assert!(phase.has_texture(EntityPhase::FALLBACK_TEXTURE));

        // Once the replacement sheet is valid again, the same generated key is uploaded fresh.
        std::fs::write(&armour_path, png([1, 2, 3, 255])).expect("replacement armour png");
        assert!(phase.load_horse_texture(&gpu, &resources, &selection));
        assert!(phase.has_texture(&key));
    }

    #[test]
    fn model_offset_is_not_a_rounded_one_and_a_half() {
        assert_eq!(MODEL_Y_OFFSET, -24.125 / 16.0);
        assert_ne!(MODEL_Y_OFFSET, -1.5);
    }

    #[test]
    fn a_model_units_feet_land_an_eighth_of_a_pixel_above_the_entity() {
        // A living model's feet are at y = 24 in model space (12 body + 12 leg), which is
        // 1.5 blocks. The lift is 1.5078125 = 24.125/16, an eighth of a *pixel* more, so
        // vanilla's mobs stand a hair above their own position rather than exactly on it.
        // That gap is real and load-bearing: it is what keeps a mob's soles out of the top
        // face of the block they are standing on.
        let m = entity_matrix(&EntityTransform::default());
        let feet = m.transform_point3(Vec3::new(0.0, 24.0 * MODEL_SCALE, 0.0));
        assert!(
            (feet.y - 0.0078125).abs() < 1e-7,
            "feet at {}, expected 0.125/16",
            feet.y
        );
        assert_eq!(0.0078125, 0.125 / 16.0);
        // The head top (y = 0 in model space) is the full lift up.
        let head = m.transform_point3(Vec3::ZERO);
        assert!((head.y - 1.5078125).abs() < 1e-6);
    }

    #[test]
    fn the_net_flip_is_y_and_z_not_x() {
        // `rotate(180 - yaw, Y)` and `scale(-1, -1, 1)` compose: the 180 turn negates X and Z,
        // the flip negates X and Y again, so X survives and Y and Z invert. That is the same
        // `scale(1, -1, -1)` the chest renderer writes out directly, which is a useful check
        // that the two paths agree.
        let m = entity_matrix(&EntityTransform::default());
        let p = m.transform_point3(Vec3::new(1.0, 0.0, 1.0));
        assert!((p.x - 1.0).abs() < 1e-6, "x survives, got {}", p.x);
        assert!((p.z + 1.0).abs() < 1e-6, "z inverts, got {}", p.z);
        let up = m.transform_vector3(Vec3::Y);
        assert!(up.y < 0.0, "model +y is world down");
    }

    #[test]
    fn body_yaw_zero_faces_south() {
        // Vanilla yaw 0 means facing +Z. The model's -Z face is its front, and the 180 - yaw
        // rotation plus the X flip is what turns that into +Z.
        let m = entity_matrix(&EntityTransform::default());
        let front = m.transform_vector3(Vec3::new(0.0, 0.0, -1.0)).normalize();
        assert!((front.z - 1.0).abs() < 1e-5, "front points {front}");
    }

    #[test]
    fn body_yaw_ninety_faces_west() {
        let m = entity_matrix(&EntityTransform {
            body_yaw: 90.0,
            ..Default::default()
        });
        let front = m.transform_vector3(Vec3::new(0.0, 0.0, -1.0)).normalize();
        assert!((front.x + 1.0).abs() < 1e-5, "front points {front}");
    }

    #[test]
    fn the_transform_is_applied_at_the_camera_relative_position() {
        let m = entity_matrix(&EntityTransform {
            rel: Vec3::new(3.0, -2.0, 7.0),
            ..Default::default()
        });
        let feet = m.transform_point3(Vec3::new(0.0, 24.0 * MODEL_SCALE, 0.0));
        assert!(
            (feet - Vec3::new(3.0, -2.0 + 0.0078125, 7.0)).length() < 1e-5,
            "{feet}"
        );
    }

    #[test]
    fn pre_render_scale_multiplies_about_the_feet_lift() {
        // java: preRenderCallback's scale happens *before* the -1.5078125 translate, so a
        // half-size mob's feet still land on the ground.
        let m = entity_matrix(&EntityTransform {
            scale: 0.5,
            ..Default::default()
        });
        // Everything below the flip scales together, so the whole model — including the
        // eighth-of-a-pixel float — halves.
        let feet = m.transform_point3(Vec3::new(0.0, 24.0 * MODEL_SCALE, 0.0));
        assert!(
            (feet.y - 0.0078125 * 0.5).abs() < 1e-7,
            "scaled feet at {}",
            feet.y
        );
        let head = m.transform_point3(Vec3::ZERO);
        assert!((head.y - 1.5078125 * 0.5).abs() < 1e-6);
    }

    #[test]
    fn death_rotation_ramps_to_ninety_and_stops() {
        assert_eq!(death_rotation(0, 0.5, 90.0), 0.0);
        // deathTime 1, partial 0 -> (1 + 0 - 1) / 20 * 1.6 = 0 -> sqrt 0.
        assert_eq!(death_rotation(1, 0.0, 90.0), 0.0);
        // deathTime 20 -> 19/20*1.6 = 1.52 -> sqrt 1.232 -> clamped to 1 -> 90.
        assert_eq!(death_rotation(20, 0.0, 90.0), 90.0);
        // Half way: deathTime 5 -> 4/20*1.6 = 0.32 -> sqrt = 0.5657.
        let d = death_rotation(5, 0.0, 90.0);
        assert!((d - mc_math::sqrt_f(0.32) * 90.0).abs() < 1e-4, "{d}");
        assert!(d > 0.0 && d < 90.0);
    }

    #[test]
    fn death_rotation_spins_about_z_in_the_flipped_frame() {
        let m = entity_matrix(&EntityTransform {
            death_rotation: 90.0,
            ..Default::default()
        });
        // A point straight above the feet ends up horizontal once the corpse has fallen.
        let p = m.transform_point3(Vec3::new(0.0, 0.0, 0.0));
        assert!(
            p.y.abs() < 1.0,
            "a fallen corpse is not standing: y = {}",
            p.y
        );
    }

    #[test]
    fn dinnerbone_transform_has_a_distinct_height_translation_and_default_is_upright() {
        let upright = EntityTransform::default();
        let upside_down = EntityTransform {
            rel: Vec3::new(3.0, 5.0, 7.0),
            body_yaw: 37.0,
            upside_down: true,
            upside_down_height: 1.8,
            ..upright
        };
        assert!(!upright.upside_down);
        assert_eq!(upright.upside_down_height, 0.0);
        assert_ne!(entity_matrix(&upright), entity_matrix(&upside_down));

        let expected = Mat4::from_translation(upside_down.rel)
            * Mat4::from_rotation_y((180.0 - upside_down.body_yaw).to_radians())
            * Mat4::from_translation(Vec3::new(0.0, 1.9, 0.0))
            * Mat4::from_rotation_z(core::f32::consts::PI)
            * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0))
            * Mat4::from_translation(Vec3::new(0.0, MODEL_Y_OFFSET, 0.0));
        assert_eq!(entity_matrix(&upside_down), expected);
    }

    #[test]
    fn the_vertex_layout_matches_the_struct() {
        // A02-07 widened this from 32 to 48: pos at 0, uv at 12, normal at 20, colour at 32.
        assert_eq!(ENTITY_VERTEX_LAYOUT.array_stride, 48);
        assert_eq!(core::mem::size_of::<EntityVertex>(), 48);
        assert_eq!(ENTITY_VERTEX_LAYOUT.attributes[1].offset, 12);
        assert_eq!(ENTITY_VERTEX_LAYOUT.attributes[2].offset, 20);
        assert_eq!(ENTITY_VERTEX_LAYOUT.attributes[3].offset, 32);
        assert_eq!(
            ENTITY_VERTEX_LAYOUT.attributes[3].format,
            wgpu::VertexFormat::Float32x4
        );
        // The offsets have to be exactly the field offsets or the GPU reads garbage; `Pod`
        // guarantees there is no padding to make them drift, but nothing guarantees the
        // hand-written table agrees with the declaration order.
        let v = EntityVertex::default();
        let base = std::ptr::from_ref(&v) as usize;
        assert_eq!(std::ptr::from_ref(&v.uv) as usize - base, 12);
        assert_eq!(std::ptr::from_ref(&v.normal) as usize - base, 20);
        assert_eq!(std::ptr::from_ref(&v.color) as usize - base, 32);
    }

    /// java: `TextureUtil.uploadTextureImageAllocate` uses plain `GL_RGBA` for ordinary
    /// `SimpleTexture` uploads. A mid-gray byte is therefore sampled as its normalized byte
    /// value, not as an sRGB-decoded linear value. The explicit sRGB branch is a negative
    /// control: it is intentionally observably different, while alpha remains byte-normalized
    /// in either format.
    #[test]
    fn ordinary_entity_texture_format_preserves_java_plain_rgba_mid_gray() {
        let encoded = 128.0_f32 / 255.0;
        let srgb_decoded = if encoded <= 0.04045 {
            encoded / 12.92
        } else {
            ((encoded + 0.055) / 1.055).powf(2.4)
        };
        let sampled_mid_gray = match ENTITY_TEXTURE_FORMAT {
            wgpu::TextureFormat::Rgba8Unorm => encoded,
            wgpu::TextureFormat::Rgba8UnormSrgb => srgb_decoded,
            other => panic!("unexpected entity texture format: {other:?}"),
        };

        assert_eq!(ENTITY_TEXTURE_FORMAT, wgpu::TextureFormat::Rgba8Unorm);
        assert_ne!(
            ENTITY_TEXTURE_FORMAT,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            "ordinary Java GL_RGBA must not acquire an automatic sRGB decode"
        );
        assert_eq!(sampled_mid_gray, encoded);
        assert!(
            (encoded - srgb_decoded).abs() > 0.2,
            "the negative sRGB control must be materially different at mid-gray"
        );

        // sRGB transfer applies only to RGB; the alpha cutout path must continue to see the raw
        // normalized alpha byte, including for the transparent-texture controls.
        let alpha = 26.0_f32 / 255.0;
        assert_eq!(alpha, 26.0 / 255.0);
        assert!(ENTITY_WGSL.contains("if (texel.a <= 0.1)"));
    }

    /// An emitter that never mentions colour must produce untinted geometry, or every
    /// existing model would go black the moment the channel was added.
    #[test]
    fn the_default_vertex_is_untinted_not_transparent_black() {
        assert_eq!(EntityVertex::default().color, EntityVertex::WHITE);
        assert_eq!(EntityVertex::WHITE, [1.0, 1.0, 1.0, 1.0]);
    }

    /// `tint` is `GlStateManager.color`, and two of them compose by multiplying — a dyed
    /// leather chestplate on a hurt-flashing mob is the case that needs it.
    #[test]
    fn tint_multiplies_and_composes() {
        let mut verts = [EntityVertex::default(); 2];
        super::super::tint(&mut verts, [1.0, 0.0, 0.0, 0.3]);
        assert_eq!(verts[0].color, [1.0, 0.0, 0.0, 0.3]);
        super::super::tint(&mut verts, [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(verts[1].color, [0.5, 0.0, 0.0, 0.3]);
        // The identity is a no-op, and cheap enough to be worth the early return.
        super::super::tint(&mut verts, EntityVertex::WHITE);
        assert_eq!(verts[0].color, [0.5, 0.0, 0.0, 0.3]);
    }

    /// The texture-matrix translate `LayerCreeperCharge` uses to make the aura crawl.
    #[test]
    fn offset_uv_translates_every_coordinate() {
        let mut verts = [EntityVertex {
            uv: [0.25, 0.75],
            ..EntityVertex::default()
        }];
        super::super::offset_uv(&mut verts, [0.01, 0.02]);
        assert!((verts[0].uv[0] - 0.26).abs() < 1e-6);
        assert!((verts[0].uv[1] - 0.77).abs() < 1e-6);
    }

    #[test]
    fn diagnostics_accumulate_repeated_finish_payloads_but_keep_latest_counts() {
        let vertices = [EntityVertex::default(); 2];
        let lights = [NO_VERTEX_LIGHT; 2];
        let vertex_bytes = std::mem::size_of_val(&vertices) as u64;
        let light_bytes = std::mem::size_of_val(&lights) as u64;
        let dynamic_bytes = LIGHTMAP_UNIFORM_STRIDE;
        let mut stats = EntityFrameStats::default();

        stats.record_finish(EntityFrameStats {
            finish_calls: 1,
            upload_bytes: vertex_bytes + light_bytes + dynamic_bytes,
            vertex_upload_bytes: vertex_bytes,
            light_upload_bytes: light_bytes,
            dynamic_uniform_upload_bytes: dynamic_bytes,
            vertex_count: 2,
            batch_count: 1,
            gpu_capacity_bytes_total: 4096,
            ..EntityFrameStats::default()
        });
        stats.record_finish(EntityFrameStats {
            finish_calls: 1,
            upload_bytes: vertex_bytes + light_bytes + dynamic_bytes,
            vertex_upload_bytes: vertex_bytes,
            light_upload_bytes: light_bytes,
            dynamic_uniform_upload_bytes: dynamic_bytes,
            vertex_count: 4,
            batch_count: 2,
            gpu_capacity_bytes_total: 8192,
            ..EntityFrameStats::default()
        });

        assert_eq!(stats.finish_calls, 2);
        assert_eq!(stats.vertex_upload_bytes, vertex_bytes * 2);
        assert_eq!(stats.light_upload_bytes, light_bytes * 2);
        assert_eq!(stats.dynamic_uniform_upload_bytes, dynamic_bytes * 2);
        assert_eq!(
            stats.upload_bytes,
            (vertex_bytes + light_bytes + dynamic_bytes) * 2
        );
        assert_eq!((stats.vertex_count, stats.batch_count), (4, 2));
        assert_eq!(stats.gpu_capacity_bytes_total, 8192);
    }

    #[test]
    fn diagnostics_empty_payload_is_zero_and_default_reset_is_off_shape() {
        let empty: [EntityVertex; 0] = [];
        assert_eq!(uploaded_payload_bytes(&empty), 0);
        let mut stats = EntityFrameStats::default();
        stats.record_finish(EntityFrameStats {
            finish_calls: 1,
            vertex_count: 3,
            batch_count: 1,
            gpu_capacity_bytes_total: 4096,
            ..EntityFrameStats::default()
        });
        stats = EntityFrameStats::default();
        assert_eq!(stats, EntityFrameStats::default());
        assert_eq!(stats.finish_calls, 0);
        assert_eq!(stats.upload_bytes, 0);
    }
}

/// [`append_draw_fused`] plus
/// [`compact_after_fusion`] must leave exactly what [`append_draw`] plus
/// `indexed_experiment::compact` leave: the same vertices, lights and batches, bit for bit.
#[cfg(test)]
mod nw_f1_fused_compact_tests {
    use super::*;

    /// A tiny deterministic generator (xorshift64*), so the sweep needs no dependency.
    struct Gen(u64);

    impl Gen {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn vertex(tag: f32) -> EntityVertex {
        EntityVertex {
            pos: [tag, tag * 0.5, -tag],
            uv: [tag * 0.25, 1.0 - tag * 0.125],
            ..EntityVertex::default()
        }
    }

    /// One push's geometry: `quads` GL_QUADS faces as the model emitters write them, optionally
    /// with one group broken (so the push is not quad-shaped), or an odd-length tail.
    fn push_verts(g: &mut Gen, quads: usize, broken: bool, tail: usize) -> Vec<EntityVertex> {
        let mut out = Vec::new();
        for _ in 0..quads {
            let base = (g.below(1000) as f32) * 0.01;
            let face: [EntityVertex; 4] = core::array::from_fn(|i| vertex(base + i as f32));
            out.extend_from_slice(&[face[0], face[1], face[2], face[0], face[2], face[3]]);
        }
        if broken && !out.is_empty() {
            let at = (g.below(quads as u64) as usize) * 6 + 3;
            out[at].pos[0] += 1.0;
        }
        for _ in 0..tail {
            out.push(vertex(g.below(1000) as f32));
        }
        out
    }

    struct Push {
        texture: TextureId,
        blend: LayerBlend,
        light: u32,
        verts: Vec<EntityVertex>,
        lights: Option<Vec<u32>>,
        separate: bool,
    }

    fn random_pushes(seed: u64, count: usize) -> Vec<Push> {
        let mut g = Gen(seed | 1);
        let mut interner = TextureInterner::default();
        let textures = [interner.intern("a.png"), interner.intern("b.png")];
        let blends = [
            LayerBlend::Opaque,
            LayerBlend::Glint,
            LayerBlend::Translucent,
            LayerBlend::FishingLine,
            LayerBlend::Leash,
        ];
        (0..count)
            .map(|_| {
                let quads = g.below(4) as usize;
                let broken = g.below(8) == 0;
                let tail = if g.below(10) == 0 {
                    1 + g.below(5) as usize
                } else {
                    0
                };
                let verts = push_verts(&mut g, quads, broken, tail);
                let lights = match g.below(6) {
                    0 => Some(verts.iter().map(|_| NO_VERTEX_LIGHT).collect()),
                    1 => Some((0..verts.len()).map(|i| (i as u32 / 6) << 4).collect()),
                    2 => Some(verts.iter().map(|_| g.below(3) as u32).collect()),
                    _ => None,
                };
                // Mostly opaque, so long merged runs (the real case) are common.
                let blend = if g.below(3) == 0 {
                    blends[g.below(5) as usize]
                } else {
                    LayerBlend::Opaque
                };
                Push {
                    texture: textures[g.below(2) as usize],
                    blend,
                    light: [0x00F0_00F0, 0x0070_0050][g.below(2) as usize],
                    verts,
                    lights,
                    separate: g.below(12) == 0,
                }
            })
            .collect()
    }

    type Streams = (Vec<EntityVertex>, Vec<u32>, Vec<DrawBatch>);

    fn reference(pushes: &[Push], finish_after: Option<usize>) -> Streams {
        let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
        for (i, p) in pushes.iter().enumerate() {
            append_draw(
                &mut v,
                &mut l,
                &mut b,
                p.texture,
                p.blend,
                p.light,
                &p.verts,
                p.lights.as_deref(),
                p.separate,
            );
            if finish_after == Some(i) {
                indexed_experiment::compact(&mut v, &mut l, &mut b);
            }
        }
        indexed_experiment::compact(&mut v, &mut l, &mut b);
        (v, l, b)
    }

    fn fused(pushes: &[Push], finish_after: Option<usize>, finish_pass: bool) -> Streams {
        let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
        let mut open = false;
        for (i, p) in pushes.iter().enumerate() {
            append_draw_fused(
                &mut v,
                &mut l,
                &mut b,
                &mut open,
                p.texture,
                p.blend,
                p.light,
                &p.verts,
                p.lights.as_deref(),
                p.separate,
            );
            if finish_after == Some(i) {
                compact_after_fusion(&mut v, &mut l, &mut b);
                open = false;
            }
        }
        if finish_pass {
            compact_after_fusion(&mut v, &mut l, &mut b);
        }
        (v, l, b)
    }

    fn same_streams(a: &Streams, b: &Streams) -> bool {
        let bytes = |v: &[EntityVertex]| bytemuck::cast_slice::<EntityVertex, u8>(v).to_vec();
        bytes(&a.0) == bytes(&b.0) && a.1 == b.1 && a.2 == b.2
    }

    #[test]
    fn fused_compaction_matches_append_then_compact_bit_for_bit() {
        let mut compacted_batches = 0;
        let mut raw_batches = 0;
        for seed in 1..=400_u64 {
            let pushes = random_pushes(seed * 0x9E37_79B9, 1 + (seed % 40) as usize);
            let want = reference(&pushes, None);
            let got = fused(&pushes, None, true);
            assert!(
                same_streams(&want, &got),
                "seed {seed}: fused streams differ"
            );
            compacted_batches += want.2.iter().filter(|b| b.indexed).count();
            raw_batches += want.2.iter().filter(|b| !b.indexed).count();
        }
        // The sweep must exercise both outcomes, or it proves nothing.
        assert!(
            compacted_batches > 1000,
            "{compacted_batches} compacted batches"
        );
        assert!(raw_batches > 100, "{raw_batches} raw batches");
    }

    /// A second `finish` in one frame: pushes after it never merge into finished batches.
    #[test]
    fn fused_compaction_matches_across_a_mid_frame_finish() {
        for seed in 1..=200_u64 {
            let pushes = random_pushes(seed * 0xD1B5_4A32, 30);
            let at = (seed % 30) as usize;
            let want = reference(&pushes, Some(at));
            let got = fused(&pushes, Some(at), true);
            assert!(
                same_streams(&want, &got),
                "seed {seed}: differ after finish at {at}"
            );
        }
    }

    /// Negative controls: the comparison sees compaction (streams that were never compacted
    /// differ), and `finish`'s pass over raw batches matters: two pushes that are not
    /// quad-shaped on their own (4 + 2 vertices) can still form one quad between them, which
    /// only the whole-batch test compacts.
    #[test]
    fn the_comparison_notices_missing_compaction() {
        let differs = (1..=50_u64)
            .filter(|&seed| {
                let pushes = random_pushes(seed * 0x9E37_79B9, 1 + (seed % 40) as usize);
                let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
                for p in &pushes {
                    append_draw(
                        &mut v,
                        &mut l,
                        &mut b,
                        p.texture,
                        p.blend,
                        p.light,
                        &p.verts,
                        p.lights.as_deref(),
                        p.separate,
                    );
                }
                !same_streams(&(v, l, b), &fused(&pushes, None, true))
            })
            .count();
        assert!(
            differs > 40,
            "only {differs} of 50 uncompacted sweeps differed"
        );

        let mut interner = TextureInterner::default();
        let texture = interner.intern("a.png");
        let face: [EntityVertex; 4] = core::array::from_fn(|i| vertex(i as f32));
        let quad = [face[0], face[1], face[2], face[0], face[2], face[3]];
        let push = |verts: &[EntityVertex]| Push {
            texture,
            blend: LayerBlend::Opaque,
            light: 0,
            verts: verts.to_vec(),
            lights: None,
            separate: false,
        };
        let pushes = [push(&quad[..4]), push(&quad[4..])];
        let want = reference(&pushes, None);
        assert!(
            want.2[0].indexed,
            "the straddling pair is one quad after compact"
        );
        assert!(same_streams(&want, &fused(&pushes, None, true)));
        assert!(
            !same_streams(&want, &fused(&pushes, None, false)),
            "without finish's pass the straddling quad stays raw"
        );
    }

    /// The usual entity frame: model quads only, one merged run per texture switch, all
    /// compacted on arrival, with nothing left for `finish` to move.
    #[test]
    fn model_quads_compact_on_arrival_and_finish_moves_nothing() {
        let mut g = Gen(7);
        let mut interner = TextureInterner::default();
        let skin = interner.intern("skin.png");
        let armour = interner.intern("armour.png");
        let (mut v, mut l, mut b) = (Vec::new(), Vec::new(), Vec::new());
        let mut open = false;
        for texture in [skin, skin, armour, skin] {
            let verts = push_verts(&mut g, 12, false, 0);
            append_draw_fused(
                &mut v,
                &mut l,
                &mut b,
                &mut open,
                texture,
                LayerBlend::Opaque,
                0x00F0_00F0,
                &verts,
                None,
                false,
            );
        }
        assert_eq!(b.len(), 3, "the two leading skin pushes share a batch");
        assert!(b.iter().all(|batch| batch.indexed));
        assert_eq!(
            v.len(),
            4 * 12 * 4,
            "four vertices per quad, stored as they arrived"
        );
        let before = (v.clone(), b.clone());
        compact_after_fusion(&mut v, &mut l, &mut b);
        assert_eq!(
            bytemuck::cast_slice::<EntityVertex, u8>(&before.0),
            bytemuck::cast_slice::<EntityVertex, u8>(&v)
        );
        assert_eq!(before.1, b);
    }
}

/// The spectator outline's screen-sized
/// targets are made the first frame the outline is drawn, at the latest window size.
#[cfg(test)]
mod w21_mr_outline_tests {
    use super::*;
    use crate::w21_mr::test_support::{allocated_bytes, gpu, vanilla_resources};

    const MASK: &str = "mc-render spectator outline mask";

    /// Install lazily, resize while dormant, draw a frame without the outline (still dormant),
    /// then the first frame with it: the mask and the chain's `swap`/`final` targets are then
    /// exactly the latest window size, as an eager install at that size makes them. The
    /// allocator report shows the memory is really not held before that frame.
    #[test]
    fn w21_mr_outline_targets_are_made_at_first_draw_at_the_latest_size() {
        let (Some(gpu), Some(resources)) = (gpu(), vanilla_resources()) else {
            return;
        };
        let (width, height) = (640, 480);
        let full = u64::from(width * height * 4);
        let mut phase = EntityPhase::new(&gpu);
        assert!(phase.install_outline_with(&gpu, &resources, 320, 240, true));
        phase.resize(&gpu, width, height);
        {
            let mask = phase.outline_mask.as_ref().expect("installed");
            assert_eq!(mask.dormant, Some((width, height)));
            assert_eq!((mask.texture.width(), mask.texture.height()), (1, 1));
        }
        let report = allocated_bytes(&gpu.device, MASK);
        if let Some(bytes) = report {
            assert!(bytes < full, "a dormant mask holds {bytes} bytes");
            for target in ["swap", "final"] {
                let held = allocated_bytes(&gpu.device, target).unwrap();
                assert!(held < full, "dormant `{target}` holds {held} bytes");
            }
        }

        phase.begin();
        phase.set_outline_enabled(false);
        phase.finish(&gpu);
        assert!(phase.outline_mask.as_ref().unwrap().dormant.is_some());
        assert!(!outline_records(&phase));

        phase.begin();
        phase.set_outline_enabled(true);
        phase.push_outline([1.0, 0.0, 0.0, 1.0], &[EntityVertex::default(); 3]);
        phase.finish(&gpu);
        let mask = phase.outline_mask.as_ref().unwrap();
        assert_eq!(mask.dormant, None);
        assert_eq!((mask.width, mask.height), (width, height));
        assert_eq!(
            (mask.texture.width(), mask.texture.height()),
            (width, height)
        );
        assert!(
            outline_records(&phase),
            "the woken outline records this frame"
        );
        if report.is_some() {
            assert!(allocated_bytes(&gpu.device, MASK).unwrap() >= full);
            // `PostChain::resize` replaced the 1x1 `swap` and `final` with full-size targets.
            for target in ["swap", "final"] {
                assert_eq!(allocated_bytes(&gpu.device, target), Some(0), "{target}");
            }
            let held = allocated_bytes(&gpu.device, "post chain resized target").unwrap();
            assert!(held >= 2 * full, "the woken chain holds only {held} bytes");
        }

        // The switch off: an eager install holds the full size at once, as before.
        let mut eager = EntityPhase::new(&gpu);
        assert!(eager.install_outline_with(&gpu, &resources, width, height, false));
        let eager_mask = eager.outline_mask.as_ref().unwrap();
        assert_eq!(eager_mask.dormant, None);
        assert_eq!(
            (eager_mask.texture.width(), eager_mask.texture.height()),
            (width, height)
        );
        // A resize of the woken mask follows the window as the eager path always did.
        phase.resize(&gpu, 800, 600);
        let mask = phase.outline_mask.as_ref().unwrap();
        assert_eq!((mask.texture.width(), mask.texture.height()), (800, 600));
    }
}
