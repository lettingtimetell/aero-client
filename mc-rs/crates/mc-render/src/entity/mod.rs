//! Entity models, animation, and rendering.
//!
//! S11 owns [`crate::terrain`]; do not edit
//! across the boundary.
//!
//! java: `client/model/` (52 files), `client/renderer/entity/` including `layers/`, and
//! `client/renderer/tileentity/`.
//!
//! # Layout
//!
//! | Module | Java counterpart |
//! |---|---|
//! | [`model`] | `ModelRenderer`, `ModelBox`, `TexturedQuad`, `ModelBase` — box geometry and pivots |
//! | [`anim`] | `ModelBase.setRotationAngles` arguments; `RendererLivingEntity`'s interpolation |
//! | [`models`] | the ~40 concrete `Model*` classes |
//! | [`manager`] | `RenderManager` — entity type to renderer, and each renderer's texture/scale |
//! | [`layers`] | `renderer/entity/layers/**` — armour, held items, capes, eyes, saddles |
//! | [`skin`] | `ImageBufferDownload`, `DefaultPlayerSkin`, `SkinManager` — async, never blocking |
//! | [`nameplate`] | `Render.renderLivingLabel` — the two-pass depth behaviour |
//! | [`item`] | `RenderEntityItem` — dropped items, 2D and 3D |
//! | [`tile`] | `renderer/tileentity/**` — chest, sign, banner, skull, beacon, ender chest |
//! | [`interp`] | `EntityLivingBase`'s server-position lerp (`setPositionAndRotation2`) |
//! | [`render`] | the outer entity transform and the [`EntityPhase`] render phase |
//!
//! # Conventions this module inherits
//!
//! * Geometry is **camera-relative** — see [`crate::Camera`]. Entity positions arrive as `DVec3`
//!   world coordinates and go through [`crate::Camera::relative`] before they touch a matrix.
//! * Model space is Y-down and one unit is a sixteenth of a block; the flip lives in
//!   [`render::entity_matrix`].
//! * All model trigonometry uses [`mc_math::sin`]/[`mc_math::cos`], never `f32::sin` (hazard H1).
//! * Skin downloads are `tokio`-side only and reach the synchronous game loop over a channel
//!   — [`skin::SkinCache`] never blocks a frame.

use bytemuck::{Pod, Zeroable};

/// Side length, in texels, of vanilla's `TextureUtil.missingTexture` image.
pub(crate) const MISSING_TEXTURE_SIZE: u32 = 16;

/// The opaque magenta-and-black placeholder used when a requested entity texture cannot load.
///
/// This is byte-for-byte the `TextureUtil.missingTexture` checkerboard: diagonal 8×8 quadrants
/// are `0xF800F8`, and the other quadrants are black. Keep this independent of the block atlas:
/// entity and tile textures are individually bound rather than atlas sprites.
pub(crate) fn missing_texture_pixels() -> Vec<u8> {
    let mut pixels = Vec::with_capacity((MISSING_TEXTURE_SIZE * MISSING_TEXTURE_SIZE * 4) as usize);
    for y in 0..MISSING_TEXTURE_SIZE {
        for x in 0..MISSING_TEXTURE_SIZE {
            let magenta = (x < MISSING_TEXTURE_SIZE / 2) == (y < MISSING_TEXTURE_SIZE / 2);
            if magenta {
                pixels.extend_from_slice(&[0xF8, 0x00, 0xF8, 0xFF]);
            } else {
                pixels.extend_from_slice(&[0x00, 0x00, 0x00, 0xFF]);
            }
        }
    }
    pixels
}

pub mod anim;
pub mod cape_cloth;
pub mod dragon;
// The Hitboxes mod's wide-line renderer.
pub mod hitbox;
pub mod item;
pub mod layers;
pub mod manager;
// One atlas for every framed map's image.
pub mod map_atlas;
pub mod model;
pub mod models;
pub mod nameplate;
pub mod objects;
pub mod render;
pub mod skin;
pub mod skin3d;
pub mod tile;
pub mod tilepass;

pub use layers::{LayerBlend, LayerKind, FIRE_OVERLAY_BLEND};
pub use objects::{
    arrow_matrix, arrow_matrix_from, boat_matrix, camera_facing_matrix, emit_arrow,
    emit_arrow_from, emit_fireball, emit_fishing_bobber, emit_fishing_line, emit_guardian_beam,
    emit_leash, emit_painting, emit_painting_with_lightmap, emit_thrown_item, emit_xp_orb,
    ender_crystal_matrix, falling_block_matrix, hanging_position, hanging_yaw, item_facing_matrix,
    item_model_pre_transform, item_renderer_display_transform, item_renderer_model_transform,
    item_transform_matrix, leash_knot_matrix, minecart_display_matrix,
    minecart_display_matrix_scaled, minecart_matrix, painting_cell_light_position, tnt_flash_alpha,
    tnt_flashing, tnt_matrix, tnt_scale, wither_skull_matrix, xp_orb_color, xp_orb_texture_index,
    xp_orb_uv, Art, FishingLineAngler, PaintingLightmap, PaintingVertex, ARROW_TEXTURE, ART,
    FISH_TEXTURE, GUARDIAN_BEAM_TEXTURE, LARGE_FIREBALL_SCALE, PAINTING_TEXTURE, RENDER_ITEM_SCALE,
    SMALL_FIREBALL_SCALE, TNT_FUSE_TICKS, WITHER_SKULL_INVULNERABLE_TEXTURE, WITHER_SKULL_TEXTURE,
    XP_ORB_TEXTURE,
};
pub use tilepass::{light_brightness, lit, TileModels, TilePhase, TileVertex, TILE_VERTEX_LAYOUT};

/// Re-exported so a crate downstream of this one can name the matrices this module's public API
/// takes and returns — [`render::entity_matrix`], [`manager::RenderManager::emit_with`] and
/// every `*_matrix` helper in [`layers`] all speak `Mat4` — without taking its own `glam`
/// dependency purely to spell a function signature. `mc-client`'s entity submission path is the
/// caller this exists for.
pub use glam::Mat4;

pub use anim::{
    AnimParams, DragonRenderState, EntityAnimState, EntityModel, LivingRenderState,
    RemotePlayerCameraState, DEG_TO_RAD,
};
pub use dragon::{
    emit_dragon_beam, emit_dragon_death_shards, DRAGON_BEAM_TEXTURE, DRAGON_EXPLODING_TEXTURE,
};
pub use manager::{
    renderer_for, renderer_for_kind, renderer_for_player, renderer_for_variant, EntityRenderer,
    ModelKind, RenderManager,
};
pub use model::{Model, ModelBox, ModelPart, PartId, QuadVertex, TexturedQuad};
pub use nameplate::{
    argb_to_rgba, Nameplate, NameplateGlyphVertex, NameplatePass, NameplatePhase, NameplateVertex,
    NAMEPLATE_GLYPH_VERTEX_LAYOUT, NAMEPLATE_VERTEX_LAYOUT,
};
pub use render::{entity_matrix, wither_pre_render_scale, EntityPhase, ENTITY_VERTEX_LAYOUT};
pub use skin::{
    CapeCache, CapeRequest, CapeTexture, SkinCache, SkinLayout, SkinModel, SkinRequest, SkinTexture,
};

#[cfg(test)]
mod missing_texture_tests {
    use super::{missing_texture_pixels, MISSING_TEXTURE_SIZE};

    #[test]
    fn the_missing_texture_is_vanillas_opaque_sixteen_texel_checkerboard() {
        let pixels = missing_texture_pixels();
        assert_eq!(
            pixels.len(),
            (MISSING_TEXTURE_SIZE * MISSING_TEXTURE_SIZE * 4) as usize
        );

        let texel = |x: u32, y: u32| {
            let i = ((y * MISSING_TEXTURE_SIZE + x) * 4) as usize;
            &pixels[i..i + 4]
        };
        assert_eq!(texel(0, 0), [0xF8, 0x00, 0xF8, 0xFF]);
        assert_eq!(texel(7, 7), [0xF8, 0x00, 0xF8, 0xFF]);
        assert_eq!(texel(8, 0), [0x00, 0x00, 0x00, 0xFF]);
        assert_eq!(texel(0, 8), [0x00, 0x00, 0x00, 0xFF]);
        assert_eq!(texel(8, 8), [0xF8, 0x00, 0xF8, 0xFF]);
    }
}

/// One vertex of entity geometry, already transformed into camera-relative space on the CPU.
///
/// Vanilla compiles each `ModelRenderer` into a GL display list and replays it under a matrix
/// stack. There is no display list in wgpu and no fixed-function matrix stack, so
/// the equivalent is to walk the part tree once per entity per frame and write transformed
/// vertices into a per-frame buffer. Entity vertex counts are tiny — a player is 6 parts x 6
/// faces x 6 vertices = 216 vertices — so this is cheaper than the per-part draw calls and
/// uniform updates the alternative would need.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct EntityVertex {
    /// Camera-relative position, in blocks.
    pub pos: [f32; 3],
    /// Texture coordinate in `0..1` of the entity's own texture.
    pub uv: [f32; 2],
    /// Normal, transformed by the same matrix as the position.
    pub normal: [f32; 3],
    /// Linear RGBA multiplier applied to the sampled texel.
    ///
    /// java: `GlStateManager.color(r, g, b, a)`, the fixed-function *primary colour*, which the
    /// default texture environment multiplies into the sampled texel. Every tint in the entity
    /// pipeline is one of those calls and nothing else: `RendererLivingEntity.setBrightness`'s
    /// hurt flash (`1, 0, 0, 0.3`), `getColorMultiplier`'s creeper ignition white
    /// (`RenderCreeper:44`), `LayerSheepWool:31` and `LayerWolfCollar:30`'s
    /// `EntitySheep.getDyeRgb`, `LayerArmorBase`'s leather overlay pass,
    /// `LayerCreeperCharge:36`'s `0.5` grey, `RenderWolf:31`'s wet shading, and
    /// `RendererLivingEntity.renderModel`'s `1, 1, 1, 0.15` invisible ghost.
    ///
    /// [`EntityVertex::WHITE`] is the no-tint identity and is what [`Default`] produces, so an
    /// emitter that does not care about colour never has to think about this field.
    pub color: [f32; 4],
}

impl Default for EntityVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            uv: [0.0; 2],
            normal: [0.0; 3],
            color: Self::WHITE,
        }
    }
}

impl EntityVertex {
    /// The identity tint: `GlStateManager.color(1, 1, 1, 1)`, vanilla's state between draws.
    pub const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
}

/// Multiplies `color` into every vertex of an already-emitted batch.
///
/// java: this *is* `GlStateManager.color`. Vanilla sets the primary colour and then replays a
/// display list; the display list's vertices carry no colour of their own, so the state applies
/// to the whole draw. The equivalent here is to walk the batch the emitter just wrote, because
/// entity geometry is built on the CPU one frame at a time anyway (see [`EntityVertex`]) — which
/// means **no emitter signature has to change to gain a tint**, and an emitter that never thinks
/// about colour still composes correctly with one that does.
///
/// Multiplicative rather than assignment so two tints compose the way two nested
/// `GlStateManager.color` calls would not — vanilla's would clobber, but vanilla never nests
/// them; the cases that *look* nested (a dyed leather chestplate on a hurt-flashing mob) are the
/// ones where multiplying is the visually right answer.
pub fn tint(verts: &mut [EntityVertex], color: [f32; 4]) {
    if color == EntityVertex::WHITE {
        return;
    }
    for v in verts {
        for (channel, by) in v.color.iter_mut().zip(color) {
            *channel *= by;
        }
    }
}

/// Shifts every vertex's texture coordinate.
///
/// java: `GlStateManager.matrixMode(GL_TEXTURE); translate(f2, f2, 0)` in
/// `LayerCreeperCharge.doRenderLayer` (`:34-40`), which is how the charged-creeper aura crawls
/// across the model. A texture matrix is exactly a UV translation, and the UVs here are written
/// per frame on the CPU, so applying it to the batch is equivalent to (and cheaper than) a
/// per-draw uniform.
pub fn offset_uv(verts: &mut [EntityVertex], offset: [f32; 2]) {
    for v in verts {
        v.uv[0] += offset[0];
        v.uv[1] += offset[1];
    }
}
