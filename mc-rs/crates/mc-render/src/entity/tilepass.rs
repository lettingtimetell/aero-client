//! The tile-entity render pass (TESR).
//!
//! [`super::tile`] and [`super::models::tileent`] already hold the placement math and the
//! models; what is missing is a phase that draws them. Keep vanilla's own split in mind:
//! `RenderGlobal.renderEntities` runs the TESR loop in the same pass as entities, after the
//! opaque terrain and before the translucent one.
//!
//! # Why this is not [`super::render::EntityPhase`]
//!
//! It very nearly is — same texture-keyed batching, same alpha test, same disabled cull — and
//! reusing it was the first plan. One vanilla behaviour makes it the wrong home:
//! `TileEntityRendererDispatcher.renderTileEntity` (`:107-116`) sets
//! `OpenGlHelper.setLightmapTextureCoords` from `world.getCombinedLight(pos, 0)` **per tile
//! entity**, so a chest in a dark room is dark. `EntityPhase` has no lightmap input at all
//! (its own shader comment says so), and adding a per-vertex light channel to `EntityVertex`
//! would collide with the concurrent entity-appearance author, who owns that struct this wave.
//! So [`TileVertex`] carries the brightness and this phase has its own two-line shader
//! difference. That closes the first third of `A05-21` as a side effect.
//!
//! # What draws what
//!
//! | vanilla TESR | here |
//! |---|---|
//! | `TileEntityChestRenderer` / `TileEntityEnderChestRenderer` | [`TileModels::emit_chest`] |
//! | `TileEntitySignRenderer` (board only; text is `mc-client`'s, over the nameplate seam) | [`TileModels::emit_sign`] |
//! | `TileEntityBannerRenderer` (layered base cloth and patterns) | [`TileModels::emit_banner`] + [`TilePhase::load_banner_texture`] |
//! | `TileEntitySkullRenderer` | [`TileModels::emit_skull`] |
//! | `TileEntityEnchantmentTableRenderer` | [`TileModels::emit_book`] |
//! | `TileEntityEndPortalRenderer` | [`TilePhase::push_end_portal`] |
//! | `TileEntityBeaconRenderer` | [`super::tile::BeaconPhase`], registered alongside this one |
//! | `TileEntityMobSpawnerRenderer` | `mc-client`, which owns the `RenderManager` the miniature mob needs |
//! | `TileEntityPistonRenderer` | [`emit_baked_block_model`] from `mc-client`'s piston branch |

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};
use mc_data::{block_props, BlockState};
use mc_math::{BlockPos, Facing, JavaRandom};

use crate::context::{Gpu, DEPTH_FORMAT};
use crate::frame::Camera;
use crate::pipeline::{PassTarget, RenderPhase};
use crate::sky::lightmap::{Lightmap, LightmapParams};

use super::models::{passive::HorseModel, passive::HorseRenderState, tileent, ModelKind};
use super::tile::{
    banner_slate_angle, chest_lid_angle, chest_matrix, chest_texture, sign_matrix,
    sign_model_matrix, skull_rotation_degrees, skull_transform, SkullMount, SkullVariant, TileKind,
};
use super::{missing_texture_pixels, AnimParams, EntityModel, EntityVertex, MISSING_TEXTURE_SIZE};

/// One vertex of tile-entity geometry.
///
/// [`super::EntityVertex`] plus the per-tile-entity lightmap brightness vanilla pushes into
/// `OpenGlHelper.setLightmapTextureCoords` before each TESR draw
/// (`TileEntityRendererDispatcher.java:109-112`). `0..1`, already collapsed from the packed
/// `skyLight << 20 | blockLight << 4` the Java carries — nothing downstream of the fragment
/// shader needs the two channels apart for ordinary TESRs; piston vertices retain both channels
/// below so BlockModelRenderer's packed-light/AO result survives the tile pipeline.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct TileVertex {
    /// Camera-relative position, in blocks.
    pub pos: [f32; 3],
    /// Texture coordinate in `0..1`.
    pub uv: [f32; 2],
    /// Normal, transformed by the same matrix as the position.
    pub normal: [f32; 3],
    /// World lightmap brightness, `0..1`.
    pub light: f32,
    /// Per-vertex RGB multiplier from BlockModelRenderer's AO/face-shade/tint path.
    pub color: [f32; 4],
    /// Java packed sky/block light. [`NO_PACKED_LIGHT`] selects [`Self::light`].
    pub packed_light: u32,
}

/// Marker used for ordinary TESR geometry that still uses the scalar light channel.
pub const NO_PACKED_LIGHT: u32 = u32::MAX;

impl Default for TileVertex {
    fn default() -> Self {
        Self {
            pos: [0.0; 3],
            uv: [0.0; 2],
            normal: [0.0; 3],
            light: 0.0,
            color: [1.0, 1.0, 1.0, 1.0],
            packed_light: NO_PACKED_LIGHT,
        }
    }
}

/// The vertex layout [`TileVertex`] is fed to [`TilePhase`]'s pipeline with.
pub const TILE_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<TileVertex>() as wgpu::BufferAddress,
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
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32,
            offset: 32,
            shader_location: 3,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Float32x4,
            offset: 36,
            shader_location: 4,
        },
        wgpu::VertexAttribute {
            format: wgpu::VertexFormat::Uint32,
            offset: 52,
            shader_location: 5,
        },
    ],
};

/// The number of stacked quads emitted by Java's `TileEntityEndPortalRenderer`.
pub const END_PORTAL_LAYER_COUNT: usize = 16;
/// The portal surface is inset three quarters of a block above the tile entity's base.
pub const END_PORTAL_TOP_OFFSET: f32 = 0.75;
/// Java's fixed scroll period in milliseconds (`Minecraft.getSystemTime() % 700000L`).
pub const END_PORTAL_SCROLL_PERIOD_MS: u64 = 700_000;
/// Seed used by `TileEntityEndPortalRenderer.field_147527_e` every frame.
pub const END_PORTAL_RANDOM_SEED: i64 = 31_100;
/// `TileEntityEndPortalRenderer`'s first (opaque-ish) environment layer.
pub const END_PORTAL_SKY_TEXTURE: &str = "textures/environment/end_sky.png";
/// `TileEntityEndPortalRenderer`'s fifteen additive starfield layers.
pub const END_PORTAL_TEXTURE: &str = "textures/entity/end_portal.png";

/// One layer of the Java end-portal starfield.
///
/// The fixed-function renderer mutates blend state, texture scale, rotation and per-layer
/// colour before drawing each quad.  Keeping those values as an explicit record lets the CPU
/// seam reproduce the same state transitions while the GPU phase remains a normal batched
/// vertex pass.  `index == 0` is the `end_sky.png` layer; the remaining fifteen layers use
/// additive blending and `end_portal.png`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EndPortalLayer {
    /// The Java loop index (`0..16`).
    pub index: usize,
    /// Texture path bound for this layer.
    pub texture: &'static str,
    /// Whether Java's `blendFunc(1, 1)` is active for this layer.
    pub additive: bool,
    /// Java's `f4`, the projected depth extent (`65` for sky, `16 - index` otherwise).
    pub extent: f32,
    /// Java's `f5`, the texture-matrix scale.
    pub texture_scale: f32,
    /// Java's `(index * index * 4321 + index * 9) * 2` degree twist.
    pub rotation_degrees: f32,
    /// The shared `systemTime % 700000 / 700000` texture scroll.
    pub scroll: f32,
    /// Java's per-layer `WorldRenderer.color` RGB; alpha is always one.
    pub color: [f32; 3],
}

/// One vertex for an end-portal layer.
///
/// End portals deliberately do not use [`TileVertex`]: vanilla disables lighting and supplies a
/// per-layer colour, while ordinary tile models need normals and the per-TE lightmap value.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct EndPortalVertex {
    /// Camera-relative position in blocks.
    pub pos: [f32; 3],
    /// Explicitly generated projective texture coordinate.
    pub uv: [f32; 2],
    /// Java's per-layer colour, with alpha in the fourth component.
    pub color: [f32; 4],
}

/// Vertex layout for [`EndPortalVertex`].
pub const END_PORTAL_VERTEX_LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
    array_stride: core::mem::size_of::<EndPortalVertex>() as wgpu::BufferAddress,
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
            format: wgpu::VertexFormat::Float32x4,
            offset: 20,
            shader_location: 2,
        },
    ],
};

/// Build all sixteen end-portal layer states for one rendered frame.
///
/// java: `TileEntityEndPortalRenderer.renderTileEntityAt:27-56`.  The random stream is reset
/// for every frame and still consumes three `nextFloat()` calls for layer zero before that layer
/// is overwritten with white (`f6`), which is why the later colours must not be generated by a
/// special-cased nine-line shortcut.
#[must_use]
pub fn end_portal_layers(system_time_millis: u64) -> [EndPortalLayer; END_PORTAL_LAYER_COUNT] {
    let scroll = (system_time_millis % END_PORTAL_SCROLL_PERIOD_MS) as f32
        / END_PORTAL_SCROLL_PERIOD_MS as f32;
    let mut random = JavaRandom::new(END_PORTAL_RANDOM_SEED);
    core::array::from_fn(|index| {
        let mut extent = (END_PORTAL_LAYER_COUNT - index) as f32;
        let mut texture_scale = 0.0625_f32;
        let mut depth_factor = 1.0 / (extent + 1.0);
        let (texture, additive) = if index == 0 {
            extent = 65.0;
            texture_scale = 0.125;
            depth_factor = 0.1;
            (END_PORTAL_SKY_TEXTURE, false)
        } else {
            (END_PORTAL_TEXTURE, true)
        };
        if index == 1 {
            texture_scale = 0.5;
        }

        // Keep the three draws even for layer zero: Java's Random state advances before the
        // white override, and those values determine every subsequent layer's colour.
        let mut color = [
            (random.next_f32() * 0.5 + 0.1) * depth_factor,
            (random.next_f32() * 0.5 + 0.4) * depth_factor,
            (random.next_f32() * 0.5 + 0.5) * depth_factor,
        ];
        if index == 0 {
            color = [depth_factor; 3];
        }
        EndPortalLayer {
            index,
            texture,
            additive,
            extent,
            texture_scale,
            rotation_degrees: ((index * index * 4321 + index * 9) as f32) * 2.0,
            scroll,
            color,
        }
    })
}

/// The Java `systemTime % 700000 / 700000` scroll value, exposed for CPU controls.
#[must_use]
pub fn end_portal_scroll(system_time_millis: u64) -> f32 {
    (system_time_millis % END_PORTAL_SCROLL_PERIOD_MS) as f32 / END_PORTAL_SCROLL_PERIOD_MS as f32
}

/// Generate one corner's projective UV for an end-portal layer.
///
/// Java obtains these coordinates using object/eye-linear texture generation and a texture
/// matrix.  `mc-rs` phases use explicit vertex attributes, so this is the CPU form of the same
/// operation: project the camera-relative X/Z position to the layer's extent, then apply the
/// texture-matrix scale, centre rotation, and scroll in the order Java specifies.  The eye-height
/// denominator is the camera-relative distance from the portal surface; at the exact plane (an
/// impossible normal player eye position, but possible for a synthetic camera) we return the
/// centre coordinate instead of introducing an infinity into the vertex buffer.
#[must_use]
pub fn end_portal_uv(layer: EndPortalLayer, rel: Vec3, local_x: f32, local_z: f32) -> [f32; 2] {
    let eye_to_surface = -(rel.y + END_PORTAL_TOP_OFFSET);
    let projection = if eye_to_surface.abs() > f32::EPSILON {
        layer.extent / eye_to_surface
    } else {
        0.0
    };
    let mut u = (rel.x + local_x) * projection;
    let mut v = (rel.z + local_z) * projection;
    u *= layer.texture_scale;
    v *= layer.texture_scale;

    let angle = layer.rotation_degrees.to_radians();
    let (sin, cos) = angle.sin_cos();
    let centred_u = u - 0.5;
    let centred_v = v - 0.5;
    [
        centred_u * cos - centred_v * sin + 0.5,
        centred_u * sin + centred_v * cos + 0.5 + layer.scroll,
    ]
}

/// Emit all sixteen end-portal quads into `out` (96 vertices per block).
///
/// java: `TileEntityEndPortalRenderer.renderTileEntityAt:22-110`.  Each layer is kept as six
/// vertices in Java draw order (`0,1,2,0,2,3`) so [`TilePhase::push_end_portal`] can preserve the
/// sky-then-additive state transitions instead of collapsing the stack into one ordinary tile
/// batch.
pub fn emit_end_portal(rel: Vec3, system_time_millis: u64, out: &mut Vec<EndPortalVertex>) {
    let layers = end_portal_layers(system_time_millis);
    for layer in layers {
        let corners = [(0.0_f32, 0.0_f32), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)];
        let quad: [EndPortalVertex; 4] = core::array::from_fn(|corner| {
            let (local_x, local_z) = corners[corner];
            let uv = end_portal_uv(layer, rel, local_x, local_z);
            EndPortalVertex {
                pos: [
                    rel.x + local_x,
                    rel.y + END_PORTAL_TOP_OFFSET,
                    rel.z + local_z,
                ],
                uv,
                color: [layer.color[0], layer.color[1], layer.color[2], 1.0],
            }
        });
        for &index in &[0_usize, 1, 2, 0, 2, 3] {
            out.push(quad[index]);
        }
    }
}

/// Vanilla's minimum lightmap contribution.
///
/// java: `EntityRenderer.updateLightmap` never writes a fully black texel — the ambient floor
/// (`f1 = world.provider.getLightBrightnessTable()...`, plus the `0.05` night-vision-free floor
/// at `:1152`) bottoms out around here. Without a floor a chest in an unlit cave is a black
/// silhouette rather than a dim chest, which reads as a rendering bug.
pub const MIN_TILE_BRIGHTNESS: f32 = 0.05;

/// The `0..1` brightness a `0..=15` combined light level maps to.
///
/// java: `World.getCombinedLight(pos, 0)` feeds `OpenGlHelper.setLightmapTextureCoords`, which
/// indexes `EntityRenderer.lightmapColors`. That table is roughly `l / (4 - 3l)` with
/// `l = level / 15` — the curve that makes level 7 look like half brightness rather than
/// 47% — reproduced here rather than sampling the real lightmap texture, which this phase has no
/// binding for.
#[must_use]
pub fn light_brightness(level: u8) -> f32 {
    let l = f32::from(level.min(15)) / 15.0;
    let brightness = l / (4.0 - 3.0 * l);
    brightness.clamp(MIN_TILE_BRIGHTNESS, 1.0)
}

/// The Java lightmap is a 16x16 texture, addressed at the centre of one texel for each packed
/// sky/block nibble.  `EntityRenderer.enableLightmap` applies a texture matrix of
/// `scale(1 / 256)` followed by `translate(8, 8)`, so a byte value of `level * 16` becomes
/// `(level * 16 + 8) / 256` in normalized texture coordinates.  Java puts the block byte on
/// texture X and the sky byte on texture Y (`i % 65536`, then `i / 65536`).
pub const TILE_LIGHTMAP_SIDE: u32 = 16;

/// Convert a Java packed light value into the normalized coordinate consumed by the tile shader.
///
/// The two components intentionally stay independent: Java samples the generated
/// `EntityRenderer.lightmapTexture` at `(block * 16, sky * 16)`, rather than choosing whichever
/// raw channel is brighter.  Keeping this helper next to the shader's binding seam gives the CPU
/// controls and focused tests one canonical representation of that texture-matrix transform.
#[must_use]
pub fn packed_lightmap_uv(packed_light: u32) -> [f32; 2] {
    let sky = f32::from(crate::terrain::light::sky_byte(packed_light));
    let block = f32::from(crate::terrain::light::block_byte(packed_light));
    [
        (block + TILE_LIGHTMAP_SIDE as f32 / 2.0)
            / (TILE_LIGHTMAP_SIDE * TILE_LIGHTMAP_SIDE) as f32,
        (sky + TILE_LIGHTMAP_SIDE as f32 / 2.0) / (TILE_LIGHTMAP_SIDE * TILE_LIGHTMAP_SIDE) as f32,
    ]
}

fn upload_tile_lightmap(gpu: &Gpu, texture: &wgpu::Texture, rgba: &[u8]) {
    assert_eq!(
        rgba.len(),
        (TILE_LIGHTMAP_SIDE * TILE_LIGHTMAP_SIDE * 4) as usize,
        "invalid tile lightmap payload"
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
            bytes_per_row: Some(TILE_LIGHTMAP_SIDE * 4),
            rows_per_image: Some(TILE_LIGHTMAP_SIDE),
        },
        wgpu::Extent3d {
            width: TILE_LIGHTMAP_SIDE,
            height: TILE_LIGHTMAP_SIDE,
            depth_or_array_layers: 1,
        },
    );
}

/// Emits a baked block model in its native `0..1` block-local coordinates.
///
/// [`super::item::emit_baked_model`] uses the item renderer's `(-0.5, -0.5, -0.5)` recentering
/// convention. Tile entities instead receive the model translation that
/// `BlockModelRenderer.renderModel` applies to a world block, so their quads must remain in the
/// block's original coordinate cube. Composing the inverse recenter here keeps the shared quad
/// traversal (including normals and atlas UVs) in one place while preserving the tile convention.
///
/// java: `BlockModelRenderer.renderModel` as used by
/// `TileEntityPistonRenderer.renderTileEntityAt:50-71`.
pub fn emit_baked_block_model(
    model: &mc_assets::BakedModel,
    transform: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    super::item::emit_baked_model(
        model,
        transform * Mat4::from_translation(Vec3::splat(0.5)),
        out,
    );
}

/// Emit one piston-carried block with BlockModelRenderer's world-aware lighting.
///
/// The ordinary tile models are TESRs and intentionally keep their historical per-entity light
/// scalar.  A piston is different: Java calls `BlockModelRenderer.renderModel` directly, so its
/// face culling, AO/tint colour and packed sky/block light are all evaluated at the carried
/// block's world position.  `check_sides` is the renderer's `renderModel(..., checkSides)` flag:
/// retraction/head branches cull against neighbours, while an extending carrier draws its face
/// quads unconditionally.
#[allow(clippy::too_many_arguments)]
pub fn emit_baked_block_model_lit(
    world: &mc_world::World,
    state: BlockState,
    pos: BlockPos,
    model: &mc_assets::BakedModel,
    transform: Mat4,
    check_sides: bool,
    settings: crate::terrain::MeshSettings,
    anaglyph: bool,
    out: &mut Vec<TileVertex>,
) {
    let props = block_props(state);
    let smooth = settings.ambient_occlusion && props.light_emission == 0 && model.ambient_occlusion;

    for facing in Facing::VALUES {
        let quads = &model.face_quads[facing.index() as usize];
        if quads.is_empty()
            || (check_sides
                && !crate::terrain::cull_face::should_side_be_rendered(
                    world, state, pos, facing, settings,
                ))
        {
            continue;
        }
        if smooth {
            for quad in quads {
                emit_lit_quad(
                    world, state, pos, quad, transform, settings, anaglyph, None, out,
                );
            }
        } else {
            // Java's standard renderer uses the neighbour brightness for every culled face.
            let brightness =
                crate::terrain::light::mixed_brightness(world, state, pos.relative(facing));
            for quad in quads {
                emit_lit_quad(
                    world,
                    state,
                    pos,
                    quad,
                    transform,
                    settings,
                    anaglyph,
                    Some(brightness),
                    out,
                );
            }
        }
    }

    for quad in &model.general_quads {
        if smooth {
            emit_lit_quad(
                world, state, pos, quad, transform, settings, anaglyph, None, out,
            );
        } else {
            // Java's `renderModelStandardQuads` uses `ownBrightness == true` for general quads:
            // only a face-flush quad samples through its own face.
            let brightness = match quad.face {
                Some(face) => {
                    let bounds = crate::terrain::ao::fill_quad_bounds(quad, face, props.full_cube);
                    if bounds.flush {
                        crate::terrain::light::mixed_brightness(world, state, pos.relative(face))
                    } else {
                        crate::terrain::light::mixed_brightness(world, state, pos)
                    }
                }
                None => crate::terrain::light::mixed_brightness(world, state, pos),
            };
            emit_lit_quad(
                world,
                state,
                pos,
                quad,
                transform,
                settings,
                anaglyph,
                Some(brightness),
                out,
            );
        }
    }
}

/// Emit one four-vertex baked quad as Java's `GL_QUADS` fan (`0,1,2,0,2,3`).
#[allow(clippy::too_many_arguments)]
fn emit_lit_quad(
    world: &mc_world::World,
    state: BlockState,
    pos: BlockPos,
    quad: &mc_assets::model::BakedQuad,
    transform: Mat4,
    settings: crate::terrain::MeshSettings,
    anaglyph: bool,
    flat_brightness: Option<u32>,
    out: &mut Vec<TileVertex>,
) {
    let face = quad.face;
    let ao = if let (None, Some(face)) = (flat_brightness, face) {
        let bounds = crate::terrain::ao::fill_quad_bounds(quad, face, block_props(state).full_cube);
        Some(crate::terrain::ao::update_vertex_brightness(
            world,
            state,
            pos,
            face,
            &bounds,
            Some(&quad.verts),
            settings,
        ))
    } else {
        None
    };
    let tint = if quad.tint_index.is_some() {
        piston_tint_color(
            crate::terrain::tint::color_multiplier(world, state, pos),
            anaglyph,
        )
    } else {
        crate::terrain::tint::NO_TINT
    };
    let shade = ((crate::terrain::ao::quad_shade(quad) * 255.0) as i32).clamp(0, 255) as f32;
    let tint_channel = |shift: u32| ((tint >> shift) & 255) as f32 / 255.0;
    let color_for = |ao_multiplier: f32| {
        [
            (shade * (ao_multiplier * tint_channel(16))).clamp(0.0, 255.0) / 255.0,
            (shade * (ao_multiplier * tint_channel(8))).clamp(0.0, 255.0) / 255.0,
            (shade * (ao_multiplier * tint_channel(0))).clamp(0.0, 255.0) / 255.0,
            1.0,
        ]
    };
    let normal = transform
        .transform_vector3(tile_quad_normal(quad))
        .normalize_or_zero();
    let points: [Vec3; 4] = core::array::from_fn(|i| {
        let v = quad.verts[i].pos;
        transform.transform_point3(Vec3::new(v[0], v[1], v[2]))
    });
    for &i in &[0usize, 1, 2, 0, 2, 3] {
        let (color, packed_light) = if let Some(ao) = ao {
            (color_for(ao.color_multiplier[i]), ao.brightness[i])
        } else {
            (
                color_for(1.0),
                flat_brightness
                    .unwrap_or_else(|| crate::terrain::light::mixed_brightness(world, state, pos)),
            )
        };
        out.push(TileVertex {
            pos: points[i].into(),
            uv: quad.verts[i].uv,
            normal: normal.into(),
            light: 1.0,
            color,
            packed_light,
        });
    }
}

/// Apply Java's renderer-side anaglyph conversion to a tinted block colour.  The atlas upload
/// conversion handles texture RGB, but `BlockModelRenderer` also converts each tint multiplier
/// after the biome/world lookup and before baked shade/AO are applied.
fn piston_tint_color(tint: u32, anaglyph: bool) -> u32 {
    if !anaglyph {
        return tint;
    }
    let transformed = mc_assets::sprite::anaglyph_color([
        ((tint >> 16) & 255) as u8,
        ((tint >> 8) & 255) as u8,
        (tint & 255) as u8,
        255,
    ]);
    (u32::from(transformed[0]) << 16) | (u32::from(transformed[1]) << 8) | u32::from(transformed[2])
}

/// The outward normal for a baked block quad, including the rare oblique local rotation.
fn tile_quad_normal(quad: &mc_assets::model::BakedQuad) -> Vec3 {
    if let Some(facing) = quad.face {
        let (dx, dy, dz) = facing.offset();
        return Vec3::new(dx as f32, dy as f32, dz as f32);
    }
    let point = |index: usize| Vec3::from(quad.verts[index].pos);
    (point(2) - point(1))
        .cross(point(0) - point(1))
        .normalize_or_zero()
}

// ---------------------------------------------------------------------------------------------
// builtin/entity item models
// ---------------------------------------------------------------------------------------------

/// The item ids whose vanilla model resolves to `builtin/entity` and therefore delegates to
/// [`TileEntityItemStackRenderer`] instead of providing baked quads.
///
/// java: `RenderItem.renderItem:145-153` and
/// `TileEntityItemStackRenderer.renderByItem:27-88`.  The three chest blocks reuse their block
/// item ids, while banner and skull are ordinary item ids.  Keeping this classification next to
/// the renderer is useful even though `BakedModel` deliberately represents `builtin/entity` as an
/// empty sentinel: an empty model must not be mistaken for an absent item when this route is
/// selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuiltinTileItemKind {
    /// `Blocks.chest` / `TileEntityChest(0)`.
    Chest,
    /// `Blocks.trapped_chest` / `TileEntityChest(1)`.
    TrappedChest,
    /// `Blocks.ender_chest` / `TileEntityEnderChest`.
    EnderChest,
    /// `Items.banner` / `TileEntityBanner.setItemValues`.
    Banner,
    /// `Items.skull` / `TileEntitySkullRenderer.renderSkull`.
    Skull,
}

/// Vanilla 1.8.9 item ids admitted by [`builtin_tile_item_kind`].
pub const BANNER_ITEM_ID: u16 = 425;
/// Vanilla 1.8.9 skull item id.
pub const SKULL_ITEM_ID: u16 = 397;

/// Whether the process-local date is one of the three days on which vanilla swaps chest sheets.
///
/// java: `TileEntityChestRenderer` samples `Calendar` once in its constructor.  The client
/// already uses UTC civil-date arithmetic for the same startup decision; keeping the small
/// helper here lets GUI builtin/entity items make the identical choice without depending on
/// `mc-client` (and without changing the existing explicit [`TileModels::new`] API).
#[must_use]
pub fn christmas_today() -> bool {
    static TODAY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TODAY.get_or_init(|| {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs()) as i64;
        let days = secs.div_euclid(86_400);
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let month_part = (5 * doy + 2) / 153;
        let day = (doy - (153 * month_part + 2) / 5 + 1) as u32;
        let month = if month_part < 10 {
            month_part + 3
        } else {
            month_part - 9
        } as u32;
        super::tile::is_christmas(month, day)
    })
}

/// Resolve the `builtin/entity` item renderer for a vanilla item id.
///
/// The caller still needs the model's `isBuiltInRenderer` bit to decide whether to take this
/// branch; this function is intentionally only the exact item-family table, not a guess that
/// every empty model is an entity renderer. java: `Items.banner`, `Items.skull`, then
/// `Block.getBlockFromItem` against chest/trapped/ender chest in
/// `TileEntityItemStackRenderer.renderByItem`.
#[must_use]
pub fn builtin_tile_item_kind(item_id: u16) -> Option<BuiltinTileItemKind> {
    match item_id {
        mc_world::tileent::CHEST_ID => Some(BuiltinTileItemKind::Chest),
        mc_world::tileent::TRAPPED_CHEST_ID => Some(BuiltinTileItemKind::TrappedChest),
        mc_world::tileent::ENDER_CHEST_ID => Some(BuiltinTileItemKind::EnderChest),
        BANNER_ITEM_ID => Some(BuiltinTileItemKind::Banner),
        SKULL_ITEM_ID => Some(BuiltinTileItemKind::Skull),
        _ => None,
    }
}

/// Whether a baked item model is the empty `builtin/entity` sentinel.
///
/// `BakedModel` intentionally does not expose Java's `isBuiltInRenderer` bit.  The item baker
/// preserves the observable sentinel shape instead: `BuiltInModel` has GUI-3D enabled, no
/// quads, and no particle sprite.  Keeping this predicate here gives callers that already
/// selected the custom-renderer branch a safe way to reproduce `TileEntityItemStackRenderer`'s
/// final `else` arm without treating an ordinary missing/default model as a chest.
///
/// java: `ModelBakery.bakeBlockModels`'s `builtin/entity` -> `BuiltInModel` branch and
/// `RenderItem.renderItem`'s `model.isBuiltInRenderer()` gate.
#[must_use]
pub fn is_builtin_entity_model(model: &mc_assets::BakedModel) -> bool {
    model.gui_3d && model.is_empty() && model.particle_uv.is_none()
}

/// Resolve a builtin family after the caller has proved that the selected model is
/// `builtin/entity`.
///
/// Vanilla's `TileEntityItemStackRenderer.renderByItem` has a deliberately broad fallback: after
/// banner/skull, ender-chest, and trapped-chest checks, every other item sent to that renderer
/// uses the ordinary chest tile entity.  The exact [`builtin_tile_item_kind`] table remains
/// useful for normal item dispatch; this helper is the opt-in branch for a resource-pack or
/// future item family whose model selected the custom renderer but whose id is not one of the
/// five 1.8.9 registrations.
///
/// `None` means the model was not the custom-renderer sentinel and the caller must keep the
/// ordinary baked-item path.
#[must_use]
pub fn builtin_tile_item_kind_for_model(
    item_id: u16,
    model: &mc_assets::BakedModel,
) -> Option<BuiltinTileItemKind> {
    builtin_tile_item_kind(item_id)
        .or_else(|| is_builtin_entity_model(model).then_some(BuiltinTileItemKind::Chest))
}

/// A player profile resolved from a skull item's `SkullOwner` compound.
///
/// The item adapters live above `mc-render` and fill this small, wire-independent value after
/// decoding the profile.  `cache_key` names the existing normalized [`super::skin::SkinCache`]
/// entry; until that entry is resident, [`Self::texture_key`] returns the UUID-derived Alex/Steve
/// fallback that `TileEntitySkullRenderer` uses for a profile with no loaded skin.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BuiltinSkullOwner {
    /// The process-local key under which the appearance cache publishes normalized pixels.
    pub cache_key: String,
    /// The profile-declared arm model, retained for the skin request/cache handoff.
    pub model: super::skin::SkinModel,
    /// The UUID-derived default model used while the asynchronous skin is unavailable.
    pub fallback_model: super::skin::SkinModel,
    /// The name from a legacy string `SkullOwner`, when no complete profile was available.
    ///
    /// Vanilla's client-side `TileEntitySkull.updateGameprofile` leaves this profile incomplete
    /// on a multiplayer client (`MinecraftServer.getServer() == null`). Retaining the name keeps
    /// that distinction observable without fabricating a UUID or queueing a URL-less request.
    pub legacy_name: Option<String>,
}

impl BuiltinSkullOwner {
    /// Construct an owner descriptor from an appearance-cache key and model decisions.
    #[must_use]
    pub fn new(
        cache_key: impl Into<String>,
        model: super::skin::SkinModel,
        fallback_model: super::skin::SkinModel,
    ) -> Self {
        Self {
            cache_key: cache_key.into(),
            model,
            fallback_model,
            legacy_name: None,
        }
    }

    /// Construct the incomplete profile state produced by vanilla's legacy string branch.
    ///
    /// The profile itself has no UUID, but `TileEntitySkullRenderer` calls
    /// `EntityPlayer.getUUID(profile)`, which derives the vanilla offline UUID from its name and
    /// therefore may choose either Steve or Alex. The escaped name key is intentionally
    /// non-fetchable and only exists so this owner remains a stable, comparable descriptor.
    #[must_use]
    pub fn legacy(name: impl Into<String>, fallback_model: super::skin::SkinModel) -> Self {
        let name = name.into();
        let mut cache_key = String::from("mc-render/entity/skin/legacy/");
        for byte in name.bytes() {
            use core::fmt::Write as _;
            let _ = write!(cache_key, "{byte:02x}");
        }
        Self {
            cache_key,
            model: fallback_model,
            fallback_model,
            legacy_name: Some(name),
        }
    }

    /// Resolve the texture key for this frame, preserving a safe default before download.
    #[must_use]
    pub fn texture_key(&self) -> String {
        if super::skin::has_shared_skin(&self.cache_key) {
            self.cache_key.clone()
        } else {
            self.fallback_model.default_texture().to_owned()
        }
    }
}

/// The typed state a `builtin/entity` item renderer needs after the item stack has been decoded.
///
/// `TileEntityItemStackRenderer` keeps one mutable tile entity instance per family and feeds it
/// to the ordinary TESR.  This value is the allocation-free equivalent for chests/skulls and the
/// stateful equivalent for banners.  A player skull carries only its wire-independent profile
/// key/model decision; normalized pixels remain owned by the existing skin cache and are selected
/// at texture-key time without blocking the frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuiltinTileItem {
    /// A single normal or trapped chest.  Item rendering never has adjacent halves.
    Chest {
        /// `TileEntityChest.getChestType() == 1` selects the trapped sheet.
        trapped: bool,
    },
    /// A single ender chest.
    EnderChest,
    /// An item-form banner, already decoded from `BlockEntityTag`.
    Banner {
        /// The state established by `TileEntityBanner.setItemValues`.
        state: mc_world::BannerState,
    },
    /// An item-form skull.  The type is the item damage byte; its mount is always `UP`.
    Skull {
        /// `ItemStack.getMetadata()` passed to `TileEntitySkullRenderer.renderSkull`.
        skull_type: u8,
    },
    /// A player skull with a valid `SkullOwner` profile and an appearance-cache handoff.
    PlayerSkull {
        /// `ItemStack.getMetadata()` passed to `TileEntitySkullRenderer.renderSkull`.
        skull_type: u8,
        /// The profile-derived skin key and fallback model.
        owner: BuiltinSkullOwner,
    },
}

impl BuiltinTileItem {
    /// Which builtin family this value dispatches to.
    #[must_use]
    pub fn kind(&self) -> BuiltinTileItemKind {
        match self {
            Self::Chest { trapped: false } => BuiltinTileItemKind::Chest,
            Self::Chest { trapped: true } => BuiltinTileItemKind::TrappedChest,
            Self::EnderChest => BuiltinTileItemKind::EnderChest,
            Self::Banner { .. } => BuiltinTileItemKind::Banner,
            Self::Skull { .. } | Self::PlayerSkull { .. } => BuiltinTileItemKind::Skull,
        }
    }

    /// The texture key that the emitted geometry expects.
    ///
    /// Chest/skull paths use the fixed entity sheets.  A banner key is the same deterministic
    /// internal key accepted by [`TilePhase::load_banner_texture`], so a caller can upload the
    /// composite and then bind this result.  An uninitialised banner is the Java null-texture
    /// path and therefore returns `None`.
    #[must_use]
    pub fn texture_key(&self, christmas: bool) -> Option<String> {
        match self {
            Self::Chest { trapped } => {
                Some(chest_texture(TileKind::chest(*trapped, false), christmas).to_owned())
            }
            Self::EnderChest => Some(
                TileKind::EnderChest
                    .texture()
                    .expect("the ender chest has a fixed texture")
                    .to_owned(),
            ),
            Self::Banner { state } => state.initialized.then(|| banner_texture_key(state)),
            Self::Skull { skull_type } => Some(
                SkullVariant::from_id(*skull_type)
                    .texture()
                    .unwrap_or("textures/entity/steve.png")
                    .to_owned(),
            ),
            Self::PlayerSkull { owner, .. } => Some(owner.texture_key()),
        }
    }
}

/// The matrix `RenderItem.renderItem` has on the stack immediately before it calls
/// `TileEntityItemStackRenderer.renderByItem`.
///
/// java: `RenderItem.renderItem:141-153`: `scale(0.5)`, `rotate(180, 0, 1, 0)`, then
/// `translate(-0.5, -0.5, -0.5)`.  `parent` already contains the display-context transform and
/// GUI/entity setup performed by the caller; this helper only owns the builtin branch.
#[must_use]
pub fn builtin_item_matrix(parent: Mat4) -> Mat4 {
    parent
        * Mat4::from_scale(Vec3::splat(0.5))
        * Mat4::from_rotation_y(core::f32::consts::PI)
        * Mat4::from_translation(Vec3::splat(-0.5))
}

/// The additional matrix around `TileEntitySkullRenderer.renderSkull` for a skull item.
///
/// java: `TileEntityItemStackRenderer.renderByItem:57-61` translates by `(-.5, 0, -.5)` and
/// scales by `2` after the common [`builtin_item_matrix`] and before calling
/// `renderSkull(..., EnumFacing.UP, ...)`.  The factor of two is easy to lose because it exactly
/// cancels the common item half-scale for skulls.
#[must_use]
pub fn builtin_skull_item_matrix(parent: Mat4) -> Mat4 {
    builtin_item_matrix(parent)
        * Mat4::from_translation(Vec3::new(-0.5, 0.0, -0.5))
        * Mat4::from_scale(Vec3::splat(2.0))
}

/// Copies model geometry into [`TileVertex`]es at one brightness.
///
/// The models emit [`super::EntityVertex`] because that is what every `ModelRenderer` in the
/// tree produces; a tile entity's only extra input is the per-TE lightmap, which is constant
/// across the whole model.
#[must_use]
pub fn lit(verts: &[EntityVertex], light: f32) -> Vec<TileVertex> {
    let mut out = Vec::with_capacity(verts.len());
    lit_into(verts, light, &mut out);
    out
}

/// [`lit`] into a caller-owned buffer, which is cleared first.
///
/// 06-entity-batch U3: the TESR loop reuses one buffer for every tile entity of a frame instead
/// of allocating a fresh `Vec<TileVertex>` (56 B per vertex) per chest, sign, skull and book.
pub fn lit_into(verts: &[EntityVertex], light: f32, out: &mut Vec<TileVertex>) {
    out.clear();
    out.extend(verts.iter().map(|v| TileVertex {
        pos: v.pos,
        uv: v.uv,
        normal: v.normal,
        light,
        color: v.color,
        packed_light: NO_PACKED_LIGHT,
    }));
}

/// Replay TESR geometry through vanilla's destroy-stage texture matrix.
///
/// `TileEntityChestRenderer`, `TileEntitySignRenderer`, and `TileEntitySkullRenderer` bind a
/// standalone `destroy_stage_N` texture and then apply `scale(u, v, 1)` followed by
/// `translate(1/16, 1/16, 1/16)` on the texture matrix.  The model's ordinary `0..1` UVs must
/// therefore become `scale * (uv + 1/16)`; simply swapping the texture key leaves the crack
/// sprite at the wrong scale and is especially visible on the large-chest path (`8,4`).
///
/// The position, normal, and lightmap channels remain byte-for-byte unchanged.  Callers choose
/// the scale from the concrete TESR (single chest/ender `4,4`, large chest `8,4`, sign/skull
/// `4,2`), keeping this render crate independent of the client packet/state owner.
#[must_use]
pub fn remap_destroy_stage_uvs(
    verts: &[TileVertex],
    scale_u: f32,
    scale_v: f32,
) -> Vec<TileVertex> {
    const JAVA_TEXTURE_TRANSLATION: f32 = 1.0 / 16.0;
    verts
        .iter()
        .map(|vertex| {
            let mut remapped = *vertex;
            remapped.uv = [
                scale_u * (vertex.uv[0] + JAVA_TEXTURE_TRANSLATION),
                scale_v * (vertex.uv[1] + JAVA_TEXTURE_TRANSLATION),
            ];
            remapped
        })
        .collect()
}

const TILE_WGSL: &str = r#"
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
    @location(1) normal: vec3<f32>,
    @location(2) light: f32,
    @location(3) color: vec4<f32>,
    @location(4) lightmap: vec2<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) normal: vec3<f32>,
    @location(3) light: f32,
    @location(4) color: vec4<f32>,
    @location(5) packed_light: u32,
) -> VsOut {
    var out: VsOut;
    // `pos` is already CAMERA-RELATIVE: the CPU walked the part tree under a matrix whose
    // translation came from Camera::relative.
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.normal = normal;
    out.light = light;
    out.color = color;
    if packed_light == 0xffffffffu {
        out.lightmap = vec2<f32>(-1.0, -1.0);
    } else {
        // java: TileEntityRendererDispatcher splits the packed value as
        // `j = combined % 65536` (block/X), `k = combined / 65536` (sky/Y).
        out.lightmap = vec2<f32>(
            f32(packed_light & 255u),
            f32((packed_light >> 16u) & 255u),
        );
    }
    return out;
}

// Java's `EntityRenderer.enableLightmap` scales texture coordinates by 1/256 and translates
// them by 8/256, landing on the centre of the texel for each byte value (0, 16, ..., 240).
// Keep sky and block in separate coordinates: the generated 16x16 texture intentionally has
// mixed RGB channels, so `max(sky, block)` is not equivalent to this sample.
fn packed_lightmap_uv(lightmap: vec2<f32>) -> vec2<f32> {
    return (lightmap + vec2<f32>(8.0, 8.0)) / 256.0;
}

fn tile_lightmap(in: VsOut) -> vec3<f32> {
    if in.lightmap.x < 0.0 {
        return vec3<f32>(in.light);
    }
    return textureSample(lightmap_tex, lightmap_samp, packed_lightmap_uv(in.lightmap)).rgb;
}

// java: RenderHelper.enableStandardItemLighting, the same two directional lights the entity
// pass uses — `RenderGlobal.renderEntities` enables it once and the TESR loop runs inside it.
const LIGHT0 = vec3<f32>(0.2, 1.0, -0.7);
const LIGHT1 = vec3<f32>(-0.2, 1.0, 0.7);

fn diffuse(normal: vec3<f32>) -> f32 {
    let n = normalize(normal);
    var light = 0.4;
    light = light + 0.6 * max(dot(n, normalize(LIGHT0)), 0.0);
    light = light + 0.6 * max(dot(n, normalize(LIGHT1)), 0.0);
    return min(light, 1.0);
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // java: GlStateManager.alphaFunc(516, 0.1) — GL_GREATER, ref 0.1.
    if (texel.a <= 0.1) {
        discard;
    }
    // The one line that makes this phase not the entity phase: the world lightmap, sampled
    // per tile entity rather than per vertex. java:
    // `TileEntityRendererDispatcher.renderTileEntity:109-112`.
    return vec4<f32>(
        texel.rgb * diffuse(in.normal) * tile_lightmap(in) * in.color.rgb,
        texel.a * in.color.a,
    );
}

// java: TileEntityPistonRenderer disables standard item lighting before BlockModelRenderer.
// Face shade/AO/tint are already folded into `color`, while packed world light remains per vertex.
@fragment
fn fs_piston(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    if (texel.a <= 0.1) {
        discard;
    }
    return vec4<f32>(
        texel.rgb * tile_lightmap(in) * in.color.rgb,
        texel.a * in.color.a,
    );
}
"#;

/// End-portal shader.  The Java renderer disables lighting and supplies a colour per layer, so
/// this deliberately has no normal/light inputs and no alpha test beyond the sampled texture.
const END_PORTAL_WGSL: &str = r#"
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

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    out.clip = camera.view_proj * vec4<f32>(pos, 1.0);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv) * in.color;
    // java: `GlStateManager.alphaFunc(516, 0.1F)` is inherited from the render-world setup.
    if (texel.a < 0.1) {
        discard;
    }
    return texel;
}
"#;

/// One uploaded tile-entity texture.
struct TextureEntry {
    /// The texture allocation is retained so a live `GameSettings.anaglyph` toggle can replay
    /// Java's upload-time transform without rebuilding the bind group or trying to invert bytes.
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// Decoded/generated RGBA8 source bytes, before `TextureUtil.anaglyphColor`.
    source_rgba: Vec<u8>,
    width: u32,
    height: u32,
}

impl TextureEntry {
    /// Rewrite the resident allocation from the unmodified source bytes.
    ///
    /// Java converts a temporary upload array, not the resource-pack pixels.  Replaying from
    /// `source_rgba` therefore makes `false -> true -> false` exact and prevents double-toggle
    /// drift from the lossy integer transform.
    fn set_anaglyph(&self, gpu: &Gpu, enabled: bool) {
        let upload_rgba = tile_upload_rgba(&self.source_rgba, enabled);
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
}

/// Build the temporary RGBA8 upload array used by Java's `TextureUtil.copyToBufferPos`.
///
/// Tile entities are standalone textures from the stitched block atlas and the same boundary is
/// crossed by every TESR/builtin upload: chest sheets, signs, banners, skull skins, destroy-stage
/// cracks, portal sheets, and the piston atlas.  Keep the conversion here so all of those paths
/// share `TextureUtil.anaglyphColor`'s integer arithmetic and alpha-preserving behavior.
fn tile_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    assert!(
        source_rgba.len().is_multiple_of(4),
        "tile texture upload requires RGBA8 pixels, got {} bytes",
        source_rgba.len()
    );
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

const BANNER_BASE_TEXTURE: &str = "textures/entity/banner_base.png";
const BANNER_BASE_MASK: &str = "textures/entity/banner/base.png";
/// `LayeredColorMaskTexture` applies the base mask plus at most sixteen additional patterns.
const BANNER_MAX_PATTERNS: usize = 16;
/// Java's `TileEntityBannerRenderer.DESIGNS` keeps at most this many composites resident.
const BANNER_CACHE_CAPACITY: usize = 256;
/// Java's banner-design cleanup removes entries idle for more than one minute.
const BANNER_CACHE_IDLE_MILLIS: i64 = 60_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BannerCacheEntry {
    /// The last access time used by Java's `TimedBannerTexture.systemTime`.
    last_used_millis: i64,
}

/// The CPU side of Java's `TileEntityBannerRenderer.DESIGNS` lifetime policy.
///
/// The GPU bind group remains in [`TilePhase::textures`] while this map owns the design's
/// timestamp.  Separating the timestamp map from the generic texture table is important:
/// ordinary tile textures are not subject to the banner renderer's 256-entry limit, while an
/// evicted banner must release both records together.  The clock is injected so cache controls
/// can exercise the exact 60,000 ms edge without sleeping in a GPU test.
struct BannerTextureCache {
    entries: HashMap<String, BannerCacheEntry>,
    clock: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl BannerTextureCache {
    fn new() -> Self {
        Self::with_clock(Arc::new(current_time_millis))
    }

    fn with_clock(clock: Arc<dyn Fn() -> i64 + Send + Sync>) -> Self {
        Self {
            entries: HashMap::new(),
            clock,
        }
    }

    fn now(&self) -> i64 {
        (self.clock)()
    }

    /// Touch an existing design, mirroring Java's `systemTime = currentTimeMillis()` on a hit.
    fn touch(&mut self, key: &str) -> bool {
        let now = self.now();
        let Some(entry) = self.entries.get_mut(key) else {
            return false;
        };
        entry.last_used_millis = now;
        true
    }

    /// Prepare a cache miss for insertion.
    ///
    /// Java only performs cleanup on a miss once the map is already at capacity.  It removes
    /// every entry older than 60 seconds, then returns `None` if the map is still full.  The
    /// returned keys let [`TilePhase`] drop the corresponding GPU bind groups in the same pass.
    fn make_room(&mut self) -> Option<Vec<String>> {
        if self.entries.len() < BANNER_CACHE_CAPACITY {
            return Some(Vec::new());
        }

        let now = self.now();
        let mut evicted = Vec::new();
        self.entries.retain(|key, entry| {
            let stale = now
                .checked_sub(entry.last_used_millis)
                .is_some_and(|elapsed| elapsed > BANNER_CACHE_IDLE_MILLIS);
            if stale {
                evicted.push(key.clone());
            }
            !stale
        });

        (self.entries.len() < BANNER_CACHE_CAPACITY).then_some(evicted)
    }

    fn insert(&mut self, key: String) {
        debug_assert!(self.entries.len() < BANNER_CACHE_CAPACITY);
        self.entries.insert(
            key,
            BannerCacheEntry {
                last_used_millis: self.now(),
            },
        );
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn contains(&self, key: &str) -> bool {
        self.entries.contains_key(key)
    }
}

fn current_time_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |duration| {
            i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)
        })
}

/// `MapColor.colorValue` for `EnumDyeColor.byDyeDamage(0..15)` in 1.8.9.
///
/// The NBT field is dye damage, not the metadata order shown in the creative menu: zero is
/// black and fifteen is white.  Keeping the integer values here mirrors Java's
/// `MathHelper.func_180188_d` input exactly; using the softer item/particle dye RGB palette would
/// make the banner visibly too bright and is not the vanilla compositor.
const BANNER_MAP_COLORS_BY_DYE_DAMAGE: [u32; 16] = [
    1_644_825,  // black
    10_040_115, // red
    6_717_235,  // green
    6_704_179,  // brown
    3_361_970,  // blue
    8_339_378,  // purple
    5_013_401,  // cyan
    10_066_329, // silver
    5_000_268,  // gray
    15_892_389, // pink
    8_375_321,  // lime
    15_066_419, // yellow
    6_724_056,  // light blue
    11_685_080, // magenta
    14_188_339, // orange
    16_777_215, // white
];

/// The cache key Java calls `patternResourceLocation` (`"b" + base + Σ(patternID + colour)`).
///
/// Unknown pattern ids are ignored just like `EnumBannerPattern.getPatternByID`; the client
/// decoder filters them already, but keeping this guard makes the renderer safe for a manually
/// constructed `BannerState` too.  The texture map itself supplies the lifetime cache, so an
/// equal design always reuses one GPU texture.
#[must_use]
pub fn banner_texture_key(state: &mc_world::BannerState) -> String {
    let mut key = format!("mc-render/banner/b{}", state.base_color);
    for (pattern_id, colour) in state.patterns.iter().take(BANNER_MAX_PATTERNS) {
        if mc_world::tileent::banner_pattern_name(pattern_id).is_some() {
            key.push_str(pattern_id);
            key.push_str(&colour.to_string());
        }
    }
    key
}

fn banner_map_color(dye_damage: i32) -> [u8; 3] {
    // java: EnumDyeColor.byDyeDamage clamps invalid values to lookup index 0 (BLACK).
    let index = usize::try_from(dye_damage)
        .ok()
        .filter(|&value| value < BANNER_MAP_COLORS_BY_DYE_DAMAGE.len())
        .unwrap_or(0);
    let value = BANNER_MAP_COLORS_BY_DYE_DAMAGE[index];
    [(value >> 16) as u8, (value >> 8) as u8, value as u8]
}

fn decode_banner_png(
    resources: &mc_assets::ResourceManager,
    path: &str,
) -> Option<(Vec<u8>, u32, u32)> {
    let location = mc_assets::ResourceLocation::parse(path).ok()?;
    let bytes = resources.load(&location).ok()?;
    // Refused (missing) when the device cannot hold it.
    let image = mc_assets::decode_texture_png(&bytes).ok()?;
    let image = image.into_rgba8();
    let (width, height) = image.dimensions();
    (width != 0 && height != 0).then(|| (image.into_raw(), width, height))
}

/// Build the pixels for one item/placed-banner design without touching a GPU.
///
/// `TilePhase` and the GUI renderer both need the exact `LayeredColorMaskTexture` result, but
/// neither should borrow the other's GPU texture cache.  Keeping this small CPU seam beside the
/// tile-item descriptor lets both callers use the same source/mask decoding, Java MapColor
/// arithmetic and sixteen-pattern cap.  The returned key is the one from
/// [`banner_texture_key`] and is suitable for either renderer's texture table.
#[must_use]
pub fn banner_texture_pixels(
    resources: &mc_assets::ResourceManager,
    state: &mc_world::BannerState,
) -> Option<(String, Vec<u8>, u32, u32)> {
    if !state.initialized {
        return None;
    }

    let key = banner_texture_key(state);
    let (source, width, height) = decode_banner_png(resources, BANNER_BASE_TEXTURE)?;
    let (base_mask, base_width, base_height) = decode_banner_png(resources, BANNER_BASE_MASK)?;
    if (base_width, base_height) != (width, height) {
        return None;
    }
    let mut composite = source.clone();
    apply_banner_layer(&mut composite, &source, &base_mask, state.base_color);

    for (pattern_id, colour) in state.patterns.iter().take(BANNER_MAX_PATTERNS) {
        let Some(pattern_name) = mc_world::tileent::banner_pattern_name(pattern_id) else {
            continue;
        };
        let path = format!("textures/entity/banner/{pattern_name}.png");
        let (mask, mask_width, mask_height) = decode_banner_png(resources, &path)?;
        if (mask_width, mask_height) != (width, height) {
            return None;
        }
        apply_banner_layer(&mut composite, &source, &mask, *colour);
    }
    Some((key, composite, width, height))
}

/// Overlays one Java `LayeredColorMaskTexture` mask onto a composite image.
///
/// `source` is always the original greyscale `banner_base.png`, not the previous composite: Java
/// reads `bufferedimage1` for every layer, colourizes that source with the layer's MapColor, then
/// draws the resulting image over the accumulator.  The mask's **red** channel becomes alpha;
/// its own alpha merely decides whether the pixel is visited (`i1 & -16777216 != 0`).
fn apply_banner_layer(composite: &mut [u8], source: &[u8], mask: &[u8], dye_damage: i32) {
    let tint = banner_map_color(dye_damage);
    for ((dst, src), layer) in composite
        .chunks_exact_mut(4)
        .zip(source.chunks_exact(4))
        .zip(mask.chunks_exact(4))
    {
        if layer[3] == 0 || layer[0] == 0 {
            continue;
        }
        // java: `MathHelper.func_180188_d(k1, mapcolor.colorValue)` — integer truncation after
        // float multiplication by /255 — followed by `j1 = mask.red` as the output alpha.
        let rgb = [
            (u16::from(src[0]) * u16::from(tint[0]) / 255) as u8,
            (u16::from(src[1]) * u16::from(tint[1]) / 255) as u8,
            (u16::from(src[2]) * u16::from(tint[2]) / 255) as u8,
        ];
        let source_alpha = u32::from(layer[0]);
        let destination_alpha = u32::from(dst[3]);
        let inverse = 255 - source_alpha;
        let out_alpha = source_alpha + (destination_alpha * inverse + 127) / 255;
        if out_alpha == 0 {
            continue;
        }
        for channel in 0..3 {
            let source_premultiplied = u32::from(rgb[channel]) * source_alpha;
            let destination_premultiplied =
                u32::from(dst[channel]) * destination_alpha * inverse / 255;
            dst[channel] = ((source_premultiplied + destination_premultiplied + out_alpha / 2)
                / out_alpha)
                .min(255) as u8;
        }
        dst[3] = out_alpha.min(255) as u8;
    }
}

/// The two immutable blend states used by the ordinary TESR and piston TESR paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileBlend {
    /// Chest/sign/banner/skull geometry: Java's surrounding opaque TESR state.
    Opaque,
    /// The enchantment-table book: opaque, plus the back-face cull its renderer turns on.
    ///
    /// java: `TileEntityEnchantmentTableRenderer.renderTileEntityAt:64` calls
    /// `GlStateManager.enableCull()` right before `ModelBook.render`. The covers, the spine and
    /// the flipping pages are zero-depth `ModelBox`es whose two Z quads are coplanar with
    /// opposite windings: culled, only the camera-facing side draws; unculled, both rasterise at
    /// one depth and z-fight.
    OpaqueCulled,
    /// Moving piston geometry: Java enables source-alpha blending for the carried block.
    Translucent,
}

/// Java piston TESRs use source-alpha-over blending (`770, 771`) for every carried block.
#[must_use]
pub fn piston_blend_state() -> wgpu::BlendState {
    // Java's glBlendFunc applies the same `(770, 771)` pair to colour *and* alpha. wgpu's
    // convenience `ALPHA_BLENDING` uses `One` for the source alpha component, so spell the
    // fixed-function state out rather than accepting that subtle alpha-channel mismatch.
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// A contiguous run of vertices sharing one texture and one blend state.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DrawBatch {
    texture: String,
    blend: TileBlend,
    start: u32,
    end: u32,
}

/// The TESR pass: chests, signs, banners, skulls, the enchantment-table book and end portals.
///
/// java: the `TileEntityRendererDispatcher` half of `RenderGlobal.renderEntities` (`:695`,
/// `:704`, `:735`). Registered at [`crate::PhaseOrder::ENTITIES`] — which the enum's own doc
/// comment already reserves for "entities **and tile entities**" — immediately after
/// [`super::render::EntityPhase`], so the ordering against opaque and translucent terrain is
/// vanilla's for free.
pub struct TilePhase {
    pipeline: wgpu::RenderPipeline,
    /// [`Self::pipeline`] with back-face culling, for [`TileBlend::OpaqueCulled`].
    culled_pipeline: wgpu::RenderPipeline,
    piston_pipeline: wgpu::RenderPipeline,
    portal_pipeline_sky: wgpu::RenderPipeline,
    portal_pipeline_additive: wgpu::RenderPipeline,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    lightmap_texture: wgpu::Texture,
    /// What the lightmap texture currently holds, when it came from parameters.
    uploaded_lightmap_params: Option<LightmapParams>,
    lightmap_bind_group: wgpu::BindGroup,
    textures: HashMap<String, TextureEntry>,
    banner_cache: BannerTextureCache,
    fallback: String,
    verts: Vec<TileVertex>,
    batches: Vec<DrawBatch>,
    buffer: Option<wgpu::Buffer>,
    capacity: u64,
    uploaded: u32,
    portal_verts: Vec<EndPortalVertex>,
    portal_batches: Vec<PortalBatch>,
    portal_buffer: Option<wgpu::Buffer>,
    portal_capacity: u64,
    portal_uploaded: u32,
    /// Java's global `GameSettings.anaglyph` upload switch for resident tile textures.
    anaglyph: bool,
    enabled: bool,
}

/// A contiguous six-vertex end-portal layer and its blend mode.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PortalBatch {
    texture: &'static str,
    start: u32,
    end: u32,
    additive: bool,
}

impl TilePhase {
    /// The texture key used when a tile entity's real texture is not resident yet.
    pub const FALLBACK_TEXTURE: &'static str = "mc-render/tile/fallback";

    /// Builds the pipeline with Java's global upload-time anaglyph switch selected.
    ///
    /// The switch is stored before the fallback texture is uploaded, so the first resident tile
    /// texture already uses the persisted `GameSettings.anaglyph` value.  Call
    /// [`Self::set_anaglyph_with_gpu`] when changing the setting on an installed phase.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render tile"),
                source: wgpu::ShaderSource::Wgsl(TILE_WGSL.into()),
            });
        let portal_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render end portal"),
                source: wgpu::ShaderSource::Wgsl(END_PORTAL_WGSL.into()),
            });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("mc-render tile texture"),
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
                    label: Some("mc-render tile lightmap"),
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
        let layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render tile layout"),
                bind_group_layouts: &[
                    Some(gpu.camera_layout()),
                    Some(&texture_layout),
                    Some(&lightmap_layout),
                ],
                immediate_size: 0,
            });
        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render tile pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(TILE_VERTEX_LAYOUT)],
                },
                // java: `TileEntitySkullRenderer.renderSkull:101` calls `disableCull()`, and the
                // sign/banner boards are single-sided quads that must be readable from behind.
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    // java: `TileEntityChestRenderer:35` — `depthMask(true)`, and the whole TESR
                    // loop runs opaque.
                    depth_write_enabled: Some(true),
                    // java: `depthFunc(515)` — GL_LEQUAL.
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
        // java: `TileEntityEnchantmentTableRenderer.renderTileEntityAt:64` — `enableCull()`
        // before `ModelBook.render`, with vanilla's `cullFace(GL_BACK)` and `GL_CCW` front faces.
        // `wgpu::FrontFace::Ccw` (the default) is the same convention the terrain pipeline culls
        // `FaceBakery`'s quads with. Everything else is the ordinary opaque TESR state.
        let culled_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render tile pipeline (culled)"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(TILE_VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
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
        // Java's piston renderer enables `blendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)` for
        // every carried block, including opaque states. Keep its depth writes and disabled cull
        // state, but use a separate immutable pipeline and the lighting-free piston fragment.
        let piston_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render piston tile pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(TILE_VERTEX_LAYOUT)],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
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
                    entry_point: Some("fs_piston"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.config.format,
                        blend: Some(piston_blend_state()),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        // Java's first layer uses `blendFunc(770, 771)` (source alpha over destination), while
        // every starfield layer uses `blendFunc(1, 1)` (additive for both colour and alpha).
        // Keep those states in separate pipelines because wgpu makes blend state immutable.
        let portal_sky_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let portal_additive_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
        };
        let build_portal_pipeline = |label: &str, blend: wgpu::BlendState| {
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &portal_shader,
                        entry_point: Some("vs"),
                        compilation_options: Default::default(),
                        buffers: &[Some(END_PORTAL_VERTEX_LAYOUT)],
                    },
                    // java: `renderWorldPass` enables back-face culling
                    // (`EntityRenderer:1328`) and `TileEntityEndPortalRenderer` never touches it,
                    // so its one quad, wound (x,z) (x,z+1) (x+1,z+1) (x+1,z) and so
                    // counter-clockwise seen from above, is visible from above only. From below
                    // there is no starfield "void". Vanilla
                    // only draws it double-sided when an earlier skull/beacon/piston renderer
                    // leaked `disableCull` into the same frame, a state leak, not the design.
                    primitive: wgpu::PrimitiveState {
                        cull_mode: Some(wgpu::Face::Back),
                        ..Default::default()
                    },
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: DEPTH_FORMAT,
                        // The renderer does not alter depth state. RenderGlobal enters this loop
                        // with depth testing and writes enabled, and later layers use LEQUAL so
                        // they can land on the sky layer's equal-depth surface.
                        depth_write_enabled: Some(true),
                        depth_compare: Some(wgpu::CompareFunction::LessEqual),
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState {
                        count: gpu.sample_count(),
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &portal_shader,
                        entry_point: Some("fs"),
                        compilation_options: Default::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.config.format,
                            blend: Some(blend),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                    }),
                    multiview_mask: None,
                    cache: None,
                })
        };
        let portal_pipeline_sky =
            build_portal_pipeline("mc-render end portal/sky", portal_sky_blend);
        let portal_pipeline_additive =
            build_portal_pipeline("mc-render end portal/additive", portal_additive_blend);
        let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render tile sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let lightmap = Lightmap::generate(LightmapParams::default());
        let lightmap_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render tile lightmap"),
            size: wgpu::Extent3d {
                width: TILE_LIGHTMAP_SIDE,
                height: TILE_LIGHTMAP_SIDE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        upload_tile_lightmap(gpu, &lightmap_texture, &lightmap.rgba);
        let lightmap_view = lightmap_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let lightmap_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("mc-render tile lightmap sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let lightmap_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render tile lightmap bind group"),
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
        let mut phase = Self {
            pipeline,
            culled_pipeline,
            piston_pipeline,
            portal_pipeline_sky,
            portal_pipeline_additive,
            texture_layout,
            sampler,
            lightmap_texture,
            uploaded_lightmap_params: None,
            lightmap_bind_group,
            textures: HashMap::new(),
            banner_cache: BannerTextureCache::new(),
            fallback: Self::FALLBACK_TEXTURE.to_owned(),
            verts: Vec::new(),
            batches: Vec::new(),
            buffer: None,
            capacity: 0,
            uploaded: 0,
            portal_verts: Vec::new(),
            portal_batches: Vec::new(),
            portal_buffer: None,
            portal_capacity: 0,
            portal_uploaded: 0,
            anaglyph,
            enabled: true,
        };
        // Vanilla's 16x16 magenta/black missing texture, so a tile entity whose texture has not
        // loaded still draws as a visible diagnostic rather than stalling the frame or vanishing.
        phase.upload_texture(
            gpu,
            Self::FALLBACK_TEXTURE,
            &missing_texture_pixels(),
            MISSING_TEXTURE_SIZE,
            MISSING_TEXTURE_SIZE,
        );
        phase
    }

    /// Builds a non-anaglyph tile phase, retaining the historical constructor for tests and
    /// callers that do not own `GameSettings`.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Uploads (or replaces) one texture under `key`.
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
            &mc_assets::TextureMetadata::default(),
        );
    }

    /// Upload one tile-entity sheet with the sampler state requested by its texture sidecar.
    pub fn upload_texture_with_options(
        &mut self,
        gpu: &Gpu,
        key: &str,
        rgba: &[u8],
        width: u32,
        height: u32,
        options: &mc_assets::TextureMetadata,
    ) {
        assert_eq!(
            rgba.len(),
            (width as usize) * (height as usize) * 4,
            "tile texture {key} is {} bytes, expected {width}x{height}x4",
            rgba.len()
        );
        // `TextureUtil.copyToBufferPos` transforms a temporary upload array. Keep the caller's
        // decoded/generated bytes untouched because a later live toggle must replay from the same
        // source rather than applying the lossy integer transform to already-converted pixels.
        let source_rgba = rgba.to_vec();
        let upload_rgba = tile_upload_rgba(&source_rgba, self.anaglyph);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("mc-render tile texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Java's `TextureUtil.allocateTextureImpl` requests plain GL_RGBA, not an sRGB
            // internal format.  Keep this upload in the byte-domain format so the integer
            // anaglyph conversion is not followed by an implicit transfer-function conversion.
            format: wgpu::TextureFormat::Rgba8Unorm,
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
        let option_sampler = if *options == mc_assets::TextureMetadata::default() {
            None
        } else {
            Some(gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("mc-render tile metadata sampler"),
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
            label: Some("mc-render tile texture"),
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
        self.textures.insert(
            key.to_owned(),
            TextureEntry {
                texture,
                bind_group,
                source_rgba,
                width,
                height,
            },
        );
    }

    /// Whether `key` is resident.
    #[must_use]
    pub fn has_texture(&self, key: &str) -> bool {
        self.textures.contains_key(key)
    }

    /// Whether Java's upload-time anaglyph transform is selected for tile/TESR textures.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Rewrite every resident tile texture from its retained raw source bytes.
    ///
    /// This is the public hook for the client frame owner: call it before any lazy TESR/builtin
    /// upload after `GameSettings.anaglyph` changes.  The method is idempotent and safe on an
    /// empty texture table, which covers startup before the first visible tile entity.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        for entry in self.textures.values() {
            entry.set_anaglyph(gpu, enabled);
        }
    }

    /// Change the upload-time setting for future resource uploads without touching resident GPU
    /// allocations.  Use [`Self::set_anaglyph_with_gpu`] for a live setting change.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Replace the banner-design clock.
    ///
    /// Production uses `SystemTime`, while deterministic renderer controls can inject a clock
    /// that advances without sleeping.  Existing entries retain their last-use timestamps; this
    /// matches changing the sampled clock source rather than resetting the cache.
    pub fn set_banner_clock(&mut self, clock: impl Fn() -> i64 + Send + Sync + 'static) {
        self.banner_cache.clock = Arc::new(clock);
    }

    /// Builds and caches one banner's layered texture, returning the key to bind for its model.
    ///
    /// java: `TileEntityBannerRenderer.func_178463_a` and
    /// `LayeredColorMaskTexture.loadTexture`.  The first layer is the base dye and the next at
    /// most sixteen are the NBT patterns (Java's loop is `j < 17`, counting the base).  A
    /// missing/corrupt source or mask makes the design unavailable, so the caller can skip the
    /// entire banner just as vanilla does when `func_178463_a` returns `null`.
    pub fn load_banner_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        state: &mc_world::BannerState,
    ) -> Option<String> {
        if !state.initialized {
            return None;
        }
        let key = banner_texture_key(state);
        if self.banner_cache.touch(&key) || self.has_texture(&key) {
            return Some(key);
        }

        let evicted = self.banner_cache.make_room()?;
        for evicted_key in evicted {
            // Dropping the bind group releases the wgpu texture/view resources.  Keep this
            // paired with the timestamp removal above so a stale design cannot consume a slot
            // forever or leave an untracked GPU allocation behind.
            self.textures.remove(&evicted_key);
        }

        let (key, composite, width, height) = banner_texture_pixels(resources, state)?;
        self.upload_texture(gpu, &key, &composite, width, height);
        self.banner_cache.insert(key.clone());
        Some(key)
    }

    /// Loads `path` from the pack if it is not resident yet. `false` means it could not be read,
    /// in which case the caller should fall back to [`TilePhase::FALLBACK_TEXTURE`].
    pub fn load_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        path: &str,
    ) -> bool {
        if self.has_texture(path) {
            return true;
        }
        let Ok(location) = mc_assets::ResourceLocation::parse(path) else {
            return false;
        };
        let Ok(resource) = resources.resource(&location) else {
            return false;
        };
        // Refused (missing) when the device cannot hold it.
        let Ok(image) = mc_assets::decode_texture_png(&resource.data) else {
            return false;
        };
        let image = image.into_rgba8();
        let (width, height) = image.dimensions();
        if width == 0 || height == 0 {
            return false;
        }
        let Ok(options) =
            mc_assets::parse_texture_metadata(&location, resource.metadata.as_deref())
        else {
            return false;
        };
        self.upload_texture_with_options(gpu, path, &image.into_raw(), width, height, &options);
        true
    }

    /// Loads the two textures used by [`TileEntityEndPortalRenderer`].
    ///
    /// Java's `TextureManager.bindTexture` does not make a missing resource disappear: its
    /// failed `SimpleTexture` load binds the shared magenta/black missing-texture image instead.
    /// Install that same image under the requested path when a resource is absent, so a portal
    /// still emits all sixteen layers and the sky/portal halves fail independently.  Returning
    /// `true` means both binding keys are resident, whether they came from the pack or from the
    /// vanilla fallback.
    pub fn load_end_portal_textures(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
    ) -> bool {
        self.load_end_portal_texture(gpu, resources, END_PORTAL_SKY_TEXTURE);
        self.load_end_portal_texture(gpu, resources, END_PORTAL_TEXTURE);
        true
    }

    /// Load one end-portal binding, retaining Java's missing-texture alias on failure.
    fn load_end_portal_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        path: &'static str,
    ) {
        if self.load_texture(gpu, resources, path) {
            return;
        }
        self.upload_texture(
            gpu,
            path,
            &missing_texture_pixels(),
            MISSING_TEXTURE_SIZE,
            MISSING_TEXTURE_SIZE,
        );
    }

    /// Drops all per-frame geometry. Call once at the top of the frame.
    pub fn begin(&mut self) {
        self.verts.clear();
        self.batches.clear();
        self.portal_verts.clear();
        self.portal_batches.clear();
    }

    /// Queues one tile entity's geometry under `texture`, merging with the previous batch when
    /// they share it — which for a wall of chests means one draw call.
    pub fn push(&mut self, texture: &str, verts: &[TileVertex]) {
        self.push_blended(texture, TileBlend::Opaque, verts);
    }

    /// Queues one tile entity under an explicit Java-compatible blend state.
    pub fn push_blended(&mut self, texture: &str, blend: TileBlend, verts: &[TileVertex]) {
        if verts.is_empty() {
            return;
        }
        let key = if self.textures.contains_key(texture) {
            texture
        } else {
            &self.fallback
        };
        let start = self.verts.len() as u32;
        self.verts.extend_from_slice(verts);
        let end = self.verts.len() as u32;
        match self.batches.last_mut() {
            Some(batch) if batch.texture == key && batch.blend == blend => batch.end = end,
            _ => self.batches.push(DrawBatch {
                texture: key.to_owned(),
                blend,
                start,
                end,
            }),
        }
    }

    /// Queues one end portal's sixteen layered quads.
    ///
    /// This is the state-to-render seam for `TileEntityEndPortalRenderer`: the CPU expands Java's
    /// texgen/texture-matrix work into explicit UVs, while [`record`](RenderPhase::record) keeps
    /// the sky layer and fifteen additive layers on their corresponding immutable pipelines.
    /// The two texture keys are guaranteed resident by
    /// [`Self::load_end_portal_textures`], including the Java missing-texture fallback.
    pub fn push_end_portal(&mut self, rel: Vec3, system_time_millis: u64) -> bool {
        if !self.has_texture(END_PORTAL_SKY_TEXTURE) || !self.has_texture(END_PORTAL_TEXTURE) {
            return false;
        }
        let start = self.portal_verts.len() as u32;
        emit_end_portal(rel, system_time_millis, &mut self.portal_verts);
        let layer_vertices = 6_u32;
        for index in 0..END_PORTAL_LAYER_COUNT {
            let layer_start = start + index as u32 * layer_vertices;
            self.portal_batches.push(PortalBatch {
                texture: if index == 0 {
                    END_PORTAL_SKY_TEXTURE
                } else {
                    END_PORTAL_TEXTURE
                },
                start: layer_start,
                end: layer_start + layer_vertices,
                additive: index != 0,
            });
        }
        true
    }

    /// Uploads this frame's geometry. Call once after the last [`TilePhase::push`].
    pub fn finish(&mut self, gpu: &Gpu) {
        self.uploaded = self.verts.len() as u32;
        if !self.verts.is_empty() {
            let needed = std::mem::size_of_val(&self.verts[..]) as u64;
            if self.capacity < needed || self.buffer.is_none() {
                let cap = needed.next_power_of_two().max(4096);
                self.buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc-render tile vertices"),
                    size: cap,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.capacity = cap;
            }
            if let Some(buf) = &self.buffer {
                gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(&self.verts));
            }
        }

        self.portal_uploaded = self.portal_verts.len() as u32;
        if !self.portal_verts.is_empty() {
            let needed = std::mem::size_of_val(&self.portal_verts[..]) as u64;
            if self.portal_capacity < needed || self.portal_buffer.is_none() {
                let cap = needed.next_power_of_two().max(4096);
                self.portal_buffer = Some(gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc-render end portal vertices"),
                    size: cap,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.portal_capacity = cap;
            }
            if let Some(buf) = &self.portal_buffer {
                gpu.write_buffer_staged(buf, 0, bytemuck::cast_slice(&self.portal_verts));
            }
        }
    }

    /// Vertices queued this frame.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.verts.len() + self.portal_verts.len()
    }

    /// Draw calls queued this frame.
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len() + self.portal_batches.len()
    }

    /// Turns the phase off without unregistering it.
    pub fn set_enabled(&mut self, on: bool) {
        self.enabled = on;
    }

    /// Upload the frame's shared Java 16x16 lightmap for ordinary tiles and moving pistons.
    ///
    /// Vanilla keeps one `EntityRenderer.lightmapTexture` active while terrain, entities and
    /// tile entities render.  The phase starts with the default table so it remains drawable in
    /// isolation; the frame owner should hand the regenerated [`Lightmap`] here each frame to
    /// keep gamma, daylight, torch flicker and potion effects in lockstep with the other passes.
    pub fn set_lightmap(&mut self, gpu: &Gpu, lightmap: &Lightmap) {
        self.uploaded_lightmap_params = None;
        upload_tile_lightmap(gpu, &self.lightmap_texture, &lightmap.rgba);
    }

    /// Regenerate and upload the shared Java lightmap from frame parameters.
    pub fn set_lightmap_params(&mut self, gpu: &Gpu, params: LightmapParams) {
        if self.uploaded_lightmap_params == Some(params) {
            return;
        }
        self.set_lightmap(gpu, &Lightmap::generate_cached(params));
        self.uploaded_lightmap_params = Some(params);
    }
}

impl RenderPhase for TilePhase {
    fn name(&self) -> &'static str {
        "tile entities"
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

    // One pass, nothing else on the encoder: it joins the world pass.
    fn shares_world_pass(&self) -> bool {
        true
    }

    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, _cam: &Camera) {
        self.record_to(gpu, PassTarget::Shared(pass));
    }
}

impl TilePhase {
    /// The phase's draws, into a pass of its own or the shared world pass.
    fn record_to(&self, gpu: &Gpu, target: PassTarget<'_>) {
        if self.uploaded == 0 && self.portal_uploaded == 0 {
            return;
        }
        let mut own = None;
        let pass = target.begin(gpu, "tile entities", &mut own);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        pass.set_bind_group(2, &self.lightmap_bind_group, &[]);

        if self.uploaded > 0 {
            if let Some(buf) = &self.buffer {
                pass.set_vertex_buffer(0, buf.slice(..));
                // 06-entity-batch U3: all three pipelines share one layout, so a binding survives
                // `set_pipeline`; issue each state change only when it actually changes.
                let mut bound_blend: Option<TileBlend> = None;
                let mut bound_texture: Option<&str> = None;
                for batch in &self.batches {
                    let Some(tex) = self.textures.get(&batch.texture) else {
                        continue;
                    };
                    if bound_blend != Some(batch.blend) {
                        pass.set_pipeline(match batch.blend {
                            TileBlend::Opaque => &self.pipeline,
                            TileBlend::OpaqueCulled => &self.culled_pipeline,
                            TileBlend::Translucent => &self.piston_pipeline,
                        });
                        bound_blend = Some(batch.blend);
                    }
                    if bound_texture != Some(batch.texture.as_str()) {
                        pass.set_bind_group(1, &tex.bind_group, &[]);
                        bound_texture = Some(batch.texture.as_str());
                    }
                    pass.draw(batch.start..batch.end, 0..1);
                }
            }
        }

        // End portals use their own vertex layout and immutable blend pipelines.  The sky layer
        // is first, followed by each of the fifteen additive layers, preserving Java's exact
        // state order even when ordinary tile models were queued in the same frame.
        if self.portal_uploaded > 0 {
            if let Some(buf) = &self.portal_buffer {
                pass.set_vertex_buffer(0, buf.slice(..));
                for batch in &self.portal_batches {
                    let Some(tex) = self.textures.get(batch.texture) else {
                        continue;
                    };
                    pass.set_pipeline(if batch.additive {
                        &self.portal_pipeline_additive
                    } else {
                        &self.portal_pipeline_sky
                    });
                    pass.set_bind_group(1, &tex.bind_group, &[]);
                    pass.draw(batch.start..batch.end, 0..1);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------------------------

/// The concrete tile-entity models, kept alive between frames.
///
/// Vanilla's TESRs each hold their own `ModelChest`/`ModelSign`/`ModelBanner` instance as a
/// field and mutate its part angles in place before every draw
/// (`TileEntityChestRenderer.java:20-21`). These are those fields.
///
/// Deliberately **not** [`super::manager::RenderManager`]: that hands back a `&mut dyn
/// EntityModel`, and every one of these renderers has to write a *named* part
/// (`chestLid.rotateAngleX`, `signStick.showModel`, `bannerSlate.rotateAngleX`) that the trait
/// object cannot expose without a downcast.
#[derive(Clone, Debug)]
pub struct TileModels {
    /// java: `TileEntityChestRenderer.simpleChest`
    pub chest: tileent::ChestModel,
    /// java: `TileEntityChestRenderer.largeChest`
    pub large_chest: tileent::LargeChestModel,
    /// java: `TileEntitySignRenderer.model`
    pub sign: tileent::SignModel,
    /// java: `TileEntityBannerRenderer.bannerModel`
    pub banner: tileent::BannerModel,
    /// java: `TileEntityEnchantmentTableRenderer.field_147541_c`
    pub book: tileent::BookModel,
    /// java: `TileEntitySkullRenderer.skeletonHead`
    pub skeleton_head: tileent::SkeletonHeadModel,
    /// java: `TileEntitySkullRenderer.humanoidHead`
    pub humanoid_head: tileent::HumanoidHeadModel,
    /// The stateful `ModelHorse` used by the mob-spawner miniature.
    ///
    /// `RenderManager::emit_horse_with_state` owns the same model for world entities, but the
    /// spawner has an extra TESR placement matrix (`translate/rotate/scale`) that cannot be
    /// reconstructed from an `EntityTransform`. Keeping one horse model in this tile-owned cache
    /// lets the caller preserve that matrix while still submitting saddle/chest/rein parts.
    pub horse: HorseModel,
    /// java: `TileEntityChestRenderer.isChristmas`, sampled once at construction (`:24-31`).
    pub christmas: bool,
}

impl Default for TileModels {
    fn default() -> Self {
        Self::new(false)
    }
}

impl TileModels {
    /// Builds every model once. `christmas` comes from [`super::tile::is_christmas`].
    #[must_use]
    pub fn new(christmas: bool) -> TileModels {
        TileModels {
            chest: tileent::ChestModel::new(),
            large_chest: tileent::LargeChestModel::new(),
            sign: tileent::SignModel::new(),
            banner: tileent::BannerModel::new(),
            book: tileent::BookModel::new(),
            skeleton_head: tileent::SkeletonHeadModel::new(0, 0, 64, 32),
            humanoid_head: tileent::HumanoidHeadModel::new(),
            horse: HorseModel::new(),
            christmas,
        }
    }

    /// Pose and emit a horse under a caller-owned placement matrix.
    ///
    /// java: `RenderHorse` delegates to the shared `ModelHorse`, while
    /// `TileEntityMobSpawnerRenderer.renderMob` supplies its own miniature matrix before calling
    /// `RenderManager.renderEntityWithPosYaw`. The explicit state carries the NBT-backed
    /// `Type`/adult/saddle/chest facts that the generic entity model contract does not include.
    pub fn emit_horse_with_state(
        &mut self,
        base: Mat4,
        params: &AnimParams,
        state: &HorseRenderState,
        out: &mut Vec<EntityVertex>,
    ) {
        self.horse
            .pose_and_render_with_state(params, state, base, out);
    }

    /// A chest (single, double, trapped or ender), and the texture to bind it with.
    ///
    /// java: `TileEntityChestRenderer.renderTileEntityAt` in full — the metadata-to-yaw chain
    /// (`:126-141`), the double-chest `translate(1, 0, 0)` / `translate(0, 0, -1)` insertion
    /// (`:143-151`), and the lid easing (`:153-176`). The caller has already decided that this
    /// half is the one that draws (`ChestNeighbours::draws`).
    ///
    /// `neighbour_lids` are the adjacent halves' already-interpolated lid angles, whose larger
    /// value wins — `chest_lid_angle`'s long-unused `neighbour_lid` slice.
    ///
    /// The argument count mirrors what `TileEntityChestRenderer.renderTileEntityAt` reads off its
    /// `TileEntityChest`; collapsing them into a parameter struct would only move the same eleven
    /// values one line up at every call site.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_chest(
        &mut self,
        rel: Vec3,
        metadata: u8,
        trapped: bool,
        double: bool,
        double_offset: Vec3,
        prev_lid: f32,
        lid: f32,
        neighbour_lids: &[f32],
        partial_ticks: f32,
        out: &mut Vec<EntityVertex>,
    ) -> &'static str {
        let kind = TileKind::chest(trapped, double);
        // java: `if (i == 2) j = 180; if (i == 3) j = 0; if (i == 4) j = 90; if (i == 5) j = -90;`
        // — the identical chain `wall_facing_degrees` already ports for signs and banners.
        let yaw = super::tile::wall_facing_degrees(i32::from(metadata));
        let base = chest_matrix(rel, yaw, double_offset);
        let angle = chest_lid_angle(prev_lid, lid, partial_ticks, neighbour_lids);
        let params = AnimParams::still();
        if double {
            let id = self.large_chest.chest_lid;
            self.large_chest.model.get_mut(id).rotate_angle[0] = angle;
            self.large_chest.pose_and_render(&params, base, out);
        } else {
            let id = self.chest.chest_lid;
            self.chest.model.get_mut(id).rotate_angle[0] = angle;
            self.chest.pose_and_render(&params, base, out);
        }
        chest_texture(kind, self.christmas)
    }

    /// Emit `BlockRendererDispatcher.renderBlockBrightness`'s render-type-2 chest branch.
    ///
    /// java: `ChestRenderer.renderChestBrightness` colours by the supplied entity brightness,
    /// rotates 90 degrees around Y, then calls `TileEntityItemStackRenderer.renderByItem` without
    /// entering `RenderItem`'s common builtin-item half-scale. `RenderMinecart` is the observable
    /// caller: a chest display therefore needs this direct path rather than
    /// [`Self::emit_builtin_item`]. `state` must be one of the three render-type-2 chest blocks.
    #[must_use]
    pub fn emit_chest_block_brightness(
        &mut self,
        state: mc_data::BlockState,
        parent: Mat4,
        brightness: f32,
        out: &mut Vec<EntityVertex>,
    ) -> Option<&'static str> {
        let kind = match state.block().0 {
            54 => TileKind::chest(false, false),
            146 => TileKind::chest(true, false),
            130 => TileKind::EnderChest,
            _ => return None,
        };
        let id = self.chest.chest_lid;
        self.chest.model.get_mut(id).rotate_angle[0] = 0.0;
        let start = out.len();
        self.chest.pose_and_render(
            &AnimParams::still(),
            parent
                * Mat4::from_rotation_y(core::f32::consts::FRAC_PI_2)
                * chest_matrix(Vec3::ZERO, 0.0, Vec3::ZERO),
            out,
        );
        super::tint(&mut out[start..], [brightness, brightness, brightness, 1.0]);
        match kind {
            TileKind::EnderChest => kind.texture(),
            _ => Some(chest_texture(kind, self.christmas)),
        }
    }

    /// Emit one `builtin/entity` item under the matrix supplied by its caller.
    ///
    /// This is the CPU half of `TileEntityItemStackRenderer.renderByItem`.  `parent` is the
    /// matrix immediately before `RenderItem.renderItem`'s common builtin branch; the helper
    /// applies the branch's half-scale/180-degree turn/recentre, then invokes the same concrete
    /// tile models used by world TESRs.  The returned key is suitable for the entity/tile texture
    /// table; banners return their deterministic layered-design key and still require
    /// [`TilePhase::load_banner_texture`] before a GPU draw.
    ///
    /// java: `RenderItem.renderItem:141-153` and
    /// `TileEntityItemStackRenderer.renderByItem:27-88`.  Item tile entities have no world,
    /// neighbours, or animation state, so chests and banners use zero lid/wave inputs and skulls
    /// always use the `UP` mount with zero rotation.
    #[must_use]
    pub fn emit_builtin_item(
        &mut self,
        item: &BuiltinTileItem,
        parent: Mat4,
        out: &mut Vec<EntityVertex>,
    ) -> Option<String> {
        match item {
            BuiltinTileItem::Chest { trapped } => {
                let kind = TileKind::chest(*trapped, false);
                let id = self.chest.chest_lid;
                self.chest.model.get_mut(id).rotate_angle[0] = 0.0;
                self.chest.pose_and_render(
                    &AnimParams::still(),
                    builtin_item_matrix(parent) * chest_matrix(Vec3::ZERO, 0.0, Vec3::ZERO),
                    out,
                );
                Some(chest_texture(kind, self.christmas).to_owned())
            }
            BuiltinTileItem::EnderChest => {
                let id = self.chest.chest_lid;
                self.chest.model.get_mut(id).rotate_angle[0] = 0.0;
                self.chest.pose_and_render(
                    &AnimParams::still(),
                    builtin_item_matrix(parent) * chest_matrix(Vec3::ZERO, 0.0, Vec3::ZERO),
                    out,
                );
                Some(
                    TileKind::EnderChest
                        .texture()
                        .expect("the ender chest has a fixed texture")
                        .to_owned(),
                )
            }
            BuiltinTileItem::Banner { state } => {
                if !state.initialized {
                    // java: `TileEntityBannerRenderer.func_178463_a` returns null until
                    // `setItemValues` has marked the item tile entity ready.
                    return None;
                }
                let stand = self.banner.banner_stand;
                self.banner.model.get_mut(stand).show_model = true;
                let slate = self.banner.banner_slate;
                // Item banners have no world, and TileEntityItemStackRenderer passes zero for
                // partialTicks.  The default BlockPos is also the origin in vanilla.
                self.banner.model.get_mut(slate).rotate_angle[0] =
                    banner_slate_angle(0, 0, 0, 0, 0.0);
                let base = super::tile::banner_matrix(Vec3::ZERO, true, 0);
                self.banner.pose_and_render(
                    &AnimParams::still(),
                    builtin_item_matrix(parent) * sign_model_matrix(base),
                    out,
                );
                Some(banner_texture_key(state))
            }
            BuiltinTileItem::Skull { skull_type } => {
                let variant = SkullVariant::from_id(*skull_type);
                // TileEntityItemStackRenderer calls renderSkull(..., UP, ..., rotation 0),
                // unlike a placed skull where metadata selects a wall mount.
                let (skull, yaw) = skull_transform(Vec3::ZERO, SkullMount::Floor, 0.0);
                let params = AnimParams {
                    net_head_yaw: yaw,
                    ..AnimParams::still()
                };
                let matrix = builtin_skull_item_matrix(parent) * skull;
                match variant.model() {
                    ModelKind::HumanoidHead => {
                        self.humanoid_head.pose_and_render(&params, matrix, out)
                    }
                    _ => self.skeleton_head.pose_and_render(&params, matrix, out),
                }
                Some(
                    variant
                        .texture()
                        .unwrap_or("textures/entity/steve.png")
                        .to_owned(),
                )
            }
            BuiltinTileItem::PlayerSkull { skull_type, owner } => {
                let variant = SkullVariant::from_id(*skull_type);
                // A valid SkullOwner still uses the ordinary player head model; only the
                // texture binding differs from the owner-less Steve fallback.
                let (skull, yaw) = skull_transform(Vec3::ZERO, SkullMount::Floor, 0.0);
                let params = AnimParams {
                    net_head_yaw: yaw,
                    ..AnimParams::still()
                };
                let matrix = builtin_skull_item_matrix(parent) * skull;
                match variant.model() {
                    ModelKind::HumanoidHead => {
                        self.humanoid_head.pose_and_render(&params, matrix, out)
                    }
                    _ => self.skeleton_head.pose_and_render(&params, matrix, out),
                }
                Some(owner.texture_key())
            }
        }
    }

    /// An ender chest: the single-chest model under `ender.png`.
    ///
    /// java: `TileEntityEnderChestRenderer` — a near-copy of the chest renderer with no
    /// adjacency, no chest type and no Christmas fork.
    pub fn emit_ender_chest(
        &mut self,
        rel: Vec3,
        metadata: u8,
        prev_lid: f32,
        lid: f32,
        partial_ticks: f32,
        out: &mut Vec<EntityVertex>,
    ) -> &'static str {
        let yaw = super::tile::wall_facing_degrees(i32::from(metadata));
        let base = chest_matrix(rel, yaw, Vec3::ZERO);
        let id = self.chest.chest_lid;
        self.chest.model.get_mut(id).rotate_angle[0] =
            chest_lid_angle(prev_lid, lid, partial_ticks, &[]);
        self.chest.pose_and_render(&AnimParams::still(), base, out);
        TileKind::EnderChest
            .texture()
            .expect("the ender chest has a fixed texture")
    }

    /// A sign board. Returns the *board* matrix so the caller can hang text off it with
    /// [`super::tile::sign_text_matrix`].
    ///
    /// java: `TileEntitySignRenderer.renderTileEntityAt:26-101`, minus the text — which needs a
    /// font renderer this crate deliberately does not have (the `mc-ui` seam
    /// [`super::nameplate`] documents).
    pub fn emit_sign(
        &mut self,
        rel: Vec3,
        standing: bool,
        metadata: u8,
        out: &mut Vec<EntityVertex>,
    ) -> (&'static str, Mat4) {
        let base = sign_matrix(rel, standing, i32::from(metadata));
        // java: `this.model.signStick.showModel = <standing>`
        let stick = self.sign.sign_stick;
        self.sign.model.get_mut(stick).show_model = standing;
        self.sign
            .pose_and_render(&AnimParams::still(), sign_model_matrix(base), out);
        (
            TileKind::Sign.texture().expect("the sign texture is fixed"),
            base,
        )
    }

    /// A banner, waving. The caller binds the state-aware layered texture from
    /// [`TilePhase::load_banner_texture`].
    ///
    /// java: `TileEntityBannerRenderer.renderTileEntityAt:30-82`; texture selection is
    /// `func_178463_a` and `LayeredColorMaskTexture`, while this method is the model/pose half.
    #[allow(clippy::too_many_arguments)]
    pub fn emit_banner(
        &mut self,
        rel: Vec3,
        standing: bool,
        metadata: u8,
        block_x: i32,
        block_y: i32,
        block_z: i32,
        world_time: i64,
        partial_ticks: f32,
        out: &mut Vec<EntityVertex>,
    ) -> &'static str {
        let base = super::tile::banner_matrix(rel, standing, i32::from(metadata));
        // java: `this.bannerModel.bannerStand.showModel = <standing>`
        let stand = self.banner.banner_stand;
        self.banner.model.get_mut(stand).show_model = standing;
        let slate = self.banner.banner_slate;
        self.banner.model.get_mut(slate).rotate_angle[0] =
            banner_slate_angle(block_x, block_y, block_z, world_time, partial_ticks);
        self.banner
            .pose_and_render(&AnimParams::still(), sign_model_matrix(base), out);
        TileKind::Banner
            .texture()
            .expect("the banner base texture is fixed")
    }

    /// A skull. `None` from the texture means a player head, whose skin the caller supplies.
    ///
    /// java: `TileEntitySkullRenderer.renderSkull` — the facing switch, the `scale(-1, -1, 1)`
    /// flip, and the yaw the wall facings overwrite.
    pub fn emit_skull(
        &mut self,
        rel: Vec3,
        metadata: u8,
        skull_type: u8,
        rotation: u8,
        out: &mut Vec<EntityVertex>,
    ) -> Option<&'static str> {
        let variant = SkullVariant::from_id(skull_type);
        let mount = SkullMount::from_meta(metadata);
        let (base, yaw) = skull_transform(rel, mount, skull_rotation_degrees(rotation));
        // java: `modelbase.render(null, 0, 0, 0, p_180543_5_, 0, f)` — the yaw goes in the
        // `netHeadYaw` slot, which is what `ModelSkeletonHead.setRotationAngles` reads.
        let params = AnimParams {
            net_head_yaw: yaw,
            ..AnimParams::still()
        };
        match variant.model() {
            ModelKind::HumanoidHead => self.humanoid_head.pose_and_render(&params, base, out),
            _ => self.skeleton_head.pose_and_render(&params, base, out),
        }
        variant.texture()
    }

    /// Emit a placed skull and resolve its texture binding in the same render-side seam.  A
    /// player skull's `BuiltinSkullOwner` is optional because Java uses the legacy Steve texture
    /// when `getPlayerProfile()` is absent; non-player variants ignore the owner and retain their
    /// fixed entity sheet.
    #[must_use]
    pub fn emit_skull_with_owner(
        &mut self,
        rel: Vec3,
        metadata: u8,
        skull_type: u8,
        rotation: u8,
        owner: Option<&BuiltinSkullOwner>,
        out: &mut Vec<EntityVertex>,
    ) -> String {
        self.emit_skull(rel, metadata, skull_type, rotation, out);
        Self::skull_texture_key(skull_type, owner)
    }

    /// Resolve the texture binding for one placed skull after its model geometry has been
    /// emitted.  Vanilla's player-head arm has no fixed resource: it asks the skin manager for
    /// the profile texture and only falls back to Steve/Alex when that profile is absent or not
    /// resident yet. Keeping this selector beside [`Self::emit_skull`] prevents the live tile
    /// caller from treating every type-3 head as the ordinary Steve texture.
    #[must_use]
    pub fn skull_texture_key(skull_type: u8, owner: Option<&BuiltinSkullOwner>) -> String {
        SkullVariant::from_id(skull_type).texture().map_or_else(
            || {
                owner.map_or_else(
                    || "textures/entity/steve.png".to_owned(),
                    BuiltinSkullOwner::texture_key,
                )
            },
            str::to_owned,
        )
    }

    /// The enchanting table's floating book.
    ///
    /// java: `TileEntityEnchantmentTableRenderer.renderTileEntityAt:16-66`. The three animation
    /// numbers go in through the argument slots `ModelBook` abuses — open factor in
    /// `netHeadYaw`, the two page-flip phases in `limbSwingAmount` and `ageInTicks` — which is
    /// exactly how the Java calls `render(null, f, f3, f4, f5, 0, 0.0625)`.
    pub fn emit_book(
        &mut self,
        base: Mat4,
        tick_count: i32,
        partial_ticks: f32,
        flips: (f32, f32),
        spread: f32,
        out: &mut Vec<EntityVertex>,
    ) -> &'static str {
        self.emit_book_with_age(base, tick_count as f32 + partial_ticks, flips, spread, out)
    }

    /// Emit the enchanting book with an already-composed Java `limbSwing` value.
    ///
    /// The normal tile-entity caller has an integer `tickCount` and a partial tick, so
    /// [`Self::emit_book`] keeps that convenient shape.  `GuiEnchantment` passes a literal
    /// `0.0F` (`GuiEnchantment.java:162`).  `age_in_ticks` is whatever Java passes as the first
    /// argument to `ModelBook.render`.
    pub fn emit_book_with_age(
        &mut self,
        base: Mat4,
        age_in_ticks: f32,
        flips: (f32, f32),
        spread: f32,
        out: &mut Vec<EntityVertex>,
    ) -> &'static str {
        let params = AnimParams {
            // java: `f = (float) te.tickCount + partialTicks` is the `limbSwing` slot, which
            // `ModelBook.setRotationAngles` uses for the `sin(limbSwing * 0.02)` breathing.
            limb_swing: age_in_ticks,
            limb_swing_amount: flips.0,
            age_in_ticks: flips.1,
            net_head_yaw: spread,
            ..AnimParams::still()
        };
        self.book.pose_and_render(&params, base, out);
        "textures/entity/enchanting_table_book.png"
    }
}

/// The fixed GUI viewport Java uses for `GuiEnchantment`'s second projection.
///
/// java: `GuiEnchantment.drawGuiContainerBackgroundLayer` calls `viewport` with a 320x240
/// rectangle, centred in the current scaled surface.  The physical scale factor cancels when
/// converting the result back into `mc_ui`'s logical GUI pixels.
pub const ENCHANT_BOOK_SCREEN_VIEWPORT: [f32; 2] = [320.0, 240.0];

/// The model-view matrix for the enchanting-screen book, before `ModelBook`'s own part matrices.
///
/// java: `GuiEnchantment.drawGuiContainerBackgroundLayer:125-136`.  Keeping this in the render
/// crate makes the order of the translation/scale/rotation chain independently testable, while
/// the UI crate only has to feed the screen-owned spread and consume the resulting geometry.
#[must_use]
pub fn enchant_book_screen_model_matrix(spread: f32) -> Mat4 {
    let closed = 1.0 - spread;
    Mat4::from_translation(Vec3::new(0.0, 3.3, -16.0))
        * Mat4::from_scale(Vec3::splat(5.0))
        * Mat4::from_rotation_z(180.0f32.to_radians())
        * Mat4::from_rotation_x(20.0f32.to_radians())
        * Mat4::from_translation(Vec3::new(closed * 0.2, closed * 0.1, closed * 0.25))
        * Mat4::from_rotation_y((-(closed * 90.0) - 90.0).to_radians())
        * Mat4::from_rotation_x(180.0f32.to_radians())
}

/// The OpenGL-style perspective matrix used by the enchanting-screen book projection.
///
/// java: `Project.gluPerspective(90.0F, 1.3333334F, 9.0F, 80.0F)` after the projection-space
/// `translate(-0.34F, 0.23F, 0.0F)`.  The matrix is written in the same OpenGL right-handed clip
/// convention as LWJGL's `gluPerspective`; spelling it out also avoids glam's deprecated helper.
#[must_use]
pub fn enchant_book_screen_projection_matrix() -> Mat4 {
    let f = 1.0 / (90.0f32.to_radians() * 0.5).tan();
    let aspect = 1.3333334f32;
    let near = 9.0f32;
    let far = 80.0f32;
    Mat4::from_translation(Vec3::new(-0.34, 0.23, 0.0))
        * Mat4::from_cols_array_2d(&[
            [f / aspect, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), -1.0],
            [0.0, 0.0, 2.0 * far * near / (near - far), 0.0],
        ])
}

/// Project one model-view-space book vertex into logical GUI pixels.
///
/// The returned triplet is `[screen_x, screen_y, depth]`, where `depth` is OpenGL NDC `z` (`-1`
/// is near, `1` is far).  The viewport is the centred 320x240 rectangle from
/// [`ENCHANT_BOOK_SCREEN_VIEWPORT`].  `None` is returned for an invalid/behind-camera point
/// instead of allowing an infinity into a UI vertex buffer.
#[must_use]
pub fn project_enchant_book_vertex(
    position: Vec3,
    scaled_width: i32,
    scaled_height: i32,
) -> Option<[f32; 3]> {
    if scaled_width <= 0 || scaled_height <= 0 {
        return None;
    }
    let clip = enchant_book_screen_projection_matrix() * position.extend(1.0);
    if !clip.w.is_finite() || clip.w <= 0.0 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if !ndc.is_finite() {
        return None;
    }
    // java: `viewport((sw - 320) / 2 * scale, (sh - 240) / 2 * scale, ...)` — Java `int`
    // division, which truncates toward zero exactly as Rust's does, so an odd margin puts the
    // extra pixel on the right. GL's viewport `y` is the bottom edge; the top edge in GUI
    // coordinates is the remaining margin above it.
    let margin_x = scaled_width - ENCHANT_BOOK_SCREEN_VIEWPORT[0] as i32;
    let margin_y = scaled_height - ENCHANT_BOOK_SCREEN_VIEWPORT[1] as i32;
    let viewport_left = (margin_x / 2) as f32;
    let viewport_top = (margin_y - margin_y / 2) as f32;
    Some([
        viewport_left + (ndc.x + 1.0) * ENCHANT_BOOK_SCREEN_VIEWPORT[0] * 0.5,
        viewport_top + (1.0 - ndc.y) * ENCHANT_BOOK_SCREEN_VIEWPORT[1] * 0.5,
        ndc.z,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 06-entity-batch U2: the object pass now reuses the client's retained `TileModels` (shared
    /// with the TESR pass) instead of building a fresh set every frame. That is only sound if a
    /// builtin item emit never depends on part state a previous TESR pose left behind: dirty a
    /// shared instance the way world chests and wall banners do, then require byte-identical
    /// geometry and texture keys against a fresh instance.
    #[test]
    fn builtin_item_emits_do_not_depend_on_state_left_by_tesr_poses() {
        let mut fresh = TileModels::new(false);
        let mut shared = TileModels::new(false);
        let lid = shared.chest.chest_lid;
        shared.chest.model.get_mut(lid).rotate_angle[0] = -1.1;
        let stand = shared.banner.banner_stand;
        shared.banner.model.get_mut(stand).show_model = false;
        let slate = shared.banner.banner_slate;
        shared.banner.model.get_mut(slate).rotate_angle[0] = 0.4;
        let mut items = vec![
            BuiltinTileItem::Chest { trapped: false },
            BuiltinTileItem::Chest { trapped: true },
            BuiltinTileItem::EnderChest,
            BuiltinTileItem::Banner {
                state: mc_world::BannerState {
                    initialized: true,
                    ..mc_world::BannerState::default()
                },
            },
        ];
        items.extend((0..=4).map(|skull_type| BuiltinTileItem::Skull { skull_type }));
        let parent = Mat4::from_translation(Vec3::new(0.25, 1.5, -2.0));
        for item in &items {
            let (mut expected, mut actual) = (Vec::new(), Vec::new());
            let expected_key = fresh.emit_builtin_item(item, parent, &mut expected);
            let actual_key = shared.emit_builtin_item(item, parent, &mut actual);
            assert_eq!(actual_key, expected_key, "{item:?}");
            assert!(!actual.is_empty(), "{item:?} emitted geometry");
            assert_eq!(actual, expected, "{item:?} depends on leftover pose state");
            // The fresh instance is now also "used": emitting again must not drift either.
            let mut again = Vec::new();
            let again_key = fresh.emit_builtin_item(item, parent, &mut again);
            assert_eq!(again_key, expected_key, "{item:?} key is not idempotent");
            assert_eq!(again, expected, "{item:?} is not idempotent");
        }
    }

    #[test]
    fn tile_upload_uses_java_integer_channels_and_preserves_alpha() {
        assert_eq!(
            tile_upload_rgba(&[255, 0, 0, 7, 0, 255, 0, 19], true),
            [76, 76, 76, 7, 178, 150, 0, 19],
            "TextureUtil.anaglyphColor changes RGB only"
        );
    }

    #[test]
    fn piston_tint_uses_java_anaglyph_colour_before_shade() {
        assert_eq!(piston_tint_color(0xFF0000, false), 0xFF0000);
        assert_eq!(
            piston_tint_color(0xFF0000, true),
            0x4C4C4C,
            "BlockModelRenderer transforms a tint multiplier, independently of atlas upload"
        );
    }

    #[test]
    fn tile_upload_false_true_false_replays_the_retained_raw_bytes() {
        let source = [17, 93, 201, 3, 240, 12, 88, 127];
        let enabled = tile_upload_rgba(&source, true);
        assert_eq!(tile_upload_rgba(&source, false), source);
        assert_ne!(enabled, source);
        assert_eq!(
            tile_upload_rgba(&source, false),
            source,
            "disabling must not leave a previously converted upload resident"
        );
        assert_eq!(
            tile_upload_rgba(&source, true),
            enabled,
            "re-enabling must replay raw source bytes, not convert the prior upload"
        );
    }

    #[test]
    fn tile_upload_rejects_malformed_rgba_before_conversion() {
        let malformed = std::panic::catch_unwind(|| tile_upload_rgba(&[1, 2, 3], true));
        assert!(
            malformed.is_err(),
            "partial pixels must never cross the upload boundary"
        );
        let malformed_disabled = std::panic::catch_unwind(|| tile_upload_rgba(&[1, 2, 3], false));
        assert!(
            malformed_disabled.is_err(),
            "disabled uploads still validate the raw RGBA shape"
        );
    }

    #[test]
    fn tile_phase_startup_seeds_anaglyph_before_the_fallback_upload() {
        let Some(gpu) = (match crate::Gpu::headless(
            4,
            4,
            crate::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: true,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(crate::RenderError::NoAdapter) => None,
            Err(error) => panic!("headless context creation failed: {error}"),
        }) else {
            eprintln!("skip: no wgpu adapter available on this machine");
            return;
        };

        let phase = TilePhase::new_with_anaglyph(&gpu, true);
        assert!(phase.anaglyph_enabled());
        let fallback = phase
            .textures
            .get(TilePhase::FALLBACK_TEXTURE)
            .expect("startup always installs the missing-texture fallback");
        assert_eq!(
            tile_upload_rgba(&fallback.source_rgba, true),
            tile_upload_rgba(&fallback.source_rgba, phase.anaglyph_enabled()),
            "the first fallback upload uses the constructor's persisted setting"
        );
    }

    #[test]
    fn builtin_item_table_matches_the_five_vanilla_entity_models() {
        assert_eq!(
            builtin_tile_item_kind(mc_world::tileent::CHEST_ID),
            Some(BuiltinTileItemKind::Chest)
        );
        assert_eq!(
            builtin_tile_item_kind(mc_world::tileent::TRAPPED_CHEST_ID),
            Some(BuiltinTileItemKind::TrappedChest)
        );
        assert_eq!(
            builtin_tile_item_kind(mc_world::tileent::ENDER_CHEST_ID),
            Some(BuiltinTileItemKind::EnderChest)
        );
        assert_eq!(
            builtin_tile_item_kind(BANNER_ITEM_ID),
            Some(BuiltinTileItemKind::Banner)
        );
        assert_eq!(
            builtin_tile_item_kind(SKULL_ITEM_ID),
            Some(BuiltinTileItemKind::Skull)
        );

        // Negative control: ordinary block items use their baked JSON models.  Treating every
        // ItemBlock as a TileEntityItemStackRenderer input would turn stone into a chest.
        assert_eq!(builtin_tile_item_kind(1), None);
        assert_eq!(
            builtin_tile_item_kind(342),
            None,
            "chest minecart is not builtin/entity"
        );
    }

    #[test]
    fn builtin_entity_model_uses_java_default_chest_only_after_the_custom_renderer_gate() {
        let sentinel = mc_assets::BakedModel {
            ambient_occlusion: false,
            gui_3d: true,
            particle_uv: None,
            ..mc_assets::BakedModel::default()
        };
        assert!(is_builtin_entity_model(&sentinel));
        assert_eq!(
            builtin_tile_item_kind_for_model(1, &sentinel),
            Some(BuiltinTileItemKind::Chest),
            "an unexpected family reaching TEISR takes its ordinary-chest else arm"
        );
        assert_eq!(
            builtin_tile_item_kind_for_model(342, &sentinel),
            Some(BuiltinTileItemKind::Chest),
            "the fallback is about the selected builtin model, not the numeric id"
        );

        // Ordinary empty/default models are not Java's BuiltInModel.  This negative control
        // prevents the fallback from turning an item whose model is merely unavailable into a
        // chest, and keeps the exact five-family table's old answer intact.
        let ordinary = mc_assets::BakedModel::default();
        assert!(!is_builtin_entity_model(&ordinary));
        assert_eq!(builtin_tile_item_kind_for_model(1, &ordinary), None);

        let mut hostile = sentinel.clone();
        hostile.particle_uv = Some(mc_assets::sprite::UvRect::FULL);
        assert!(!is_builtin_entity_model(&hostile));
        assert_eq!(builtin_tile_item_kind_for_model(1, &hostile), None);
    }

    #[test]
    fn builtin_item_matrix_preserves_render_item_call_order() {
        let origin = builtin_item_matrix(Mat4::IDENTITY).transform_point3(Vec3::ZERO);
        assert!((origin.x - 0.25).abs() < 1e-6);
        assert!((origin.y + 0.25).abs() < 1e-6);
        assert!((origin.z - 0.25).abs() < 1e-6);

        let skull_origin = builtin_skull_item_matrix(Mat4::IDENTITY).transform_point3(Vec3::ZERO);
        assert!((skull_origin.x - 0.5).abs() < 1e-6);
        assert!((skull_origin.y + 0.25).abs() < 1e-6);
        assert!((skull_origin.z - 0.5).abs() < 1e-6);

        // Negative control: moving the recenter after the half-scale would put the origin at
        // (-.25, -.25, -.25), so this catches a plausible but visibly wrong matrix order.
        assert_ne!(origin, Vec3::splat(-0.25));
    }

    #[test]
    fn destroy_stage_uv_replays_the_java_texture_matrix_without_moving_tile_geometry() {
        let source = [
            TileVertex {
                pos: [1.0, 2.0, 3.0],
                uv: [0.0, 0.0],
                normal: [0.0, 1.0, 0.0],
                light: 0.375,
                color: [1.0, 1.0, 1.0, 1.0],
                packed_light: NO_PACKED_LIGHT,
            },
            TileVertex {
                pos: [-1.0, 0.25, 0.5],
                uv: [1.0, 1.0],
                normal: [1.0, 0.0, 0.0],
                light: 0.875,
                color: [1.0, 1.0, 1.0, 1.0],
                packed_light: NO_PACKED_LIGHT,
            },
        ];
        let single = remap_destroy_stage_uvs(&source, 4.0, 4.0);
        assert_eq!(single[0].uv, [0.25, 0.25]);
        assert_eq!(single[1].uv, [4.25, 4.25]);
        assert_eq!(single[0].pos, source[0].pos);
        assert_eq!(single[0].normal, source[0].normal);
        assert_eq!(single[0].light, source[0].light);

        // Negative control for the large chest: its Java matrix is 8x4, not the ordinary
        // single-chest 4x4.  This catches a texture-key-only replay and a one-size-fits-all UV
        // implementation while keeping the same source geometry.
        let large = remap_destroy_stage_uvs(&source, 8.0, 4.0);
        assert_eq!(large[0].uv, [0.5, 0.25]);
        assert_eq!(large[1].uv, [8.5, 4.25]);
        assert_ne!(large[1].uv, single[1].uv);
    }

    #[test]
    fn builtin_chest_families_emit_single_tesr_models_with_exact_sheets() {
        let mut models = TileModels::new(false);
        let mut normal = Vec::new();
        let normal_texture = models.emit_builtin_item(
            &BuiltinTileItem::Chest { trapped: false },
            Mat4::IDENTITY,
            &mut normal,
        );
        assert_eq!(
            normal_texture.as_deref(),
            Some("textures/entity/chest/normal.png")
        );
        assert_eq!(normal.len(), 3 * 6 * 6);

        let mut trapped = Vec::new();
        let trapped_texture = models.emit_builtin_item(
            &BuiltinTileItem::Chest { trapped: true },
            Mat4::IDENTITY,
            &mut trapped,
        );
        assert_eq!(
            trapped_texture.as_deref(),
            Some("textures/entity/chest/trapped.png")
        );
        assert_eq!(trapped.len(), normal.len());

        let mut ender = Vec::new();
        let ender_texture =
            models.emit_builtin_item(&BuiltinTileItem::EnderChest, Mat4::IDENTITY, &mut ender);
        assert_eq!(
            ender_texture.as_deref(),
            Some("textures/entity/chest/ender.png")
        );
        assert_eq!(ender.len(), normal.len());
        assert_eq!(
            normal[0].uv, ender[0].uv,
            "ender reuses ModelChest geometry; only the bound sheet differs"
        );
    }

    #[test]
    fn builtin_skull_uses_item_type_and_the_up_mount_with_skull_item_scale() {
        let mut models = TileModels::new(false);
        let mut skeleton = Vec::new();
        let skeleton_texture = models.emit_builtin_item(
            &BuiltinTileItem::Skull { skull_type: 0 },
            Mat4::IDENTITY,
            &mut skeleton,
        );
        assert_eq!(
            skeleton_texture.as_deref(),
            Some("textures/entity/skeleton/skeleton.png")
        );

        let mut zombie = Vec::new();
        let zombie_texture = models.emit_builtin_item(
            &BuiltinTileItem::Skull { skull_type: 2 },
            Mat4::IDENTITY,
            &mut zombie,
        );
        assert_eq!(
            zombie_texture.as_deref(),
            Some("textures/entity/zombie/zombie.png")
        );
        assert!(
            zombie.len() > skeleton.len(),
            "zombie uses the humanoid hat layer"
        );

        // The explicit (-.5, 0, -.5) + scale(2) around renderSkull means the output remains
        // centred around the item origin.  A direct wall-mount call would introduce a .25 lift;
        // this check bites that accidental reuse of the placed-skull path.
        let min_y = skeleton
            .iter()
            .map(|vertex| vertex.pos[1])
            .fold(f32::INFINITY, f32::min);
        assert!(
            min_y < 0.0,
            "item skull has no placed-skull wall lift: {min_y}"
        );
    }

    #[test]
    fn player_builtin_skull_uses_safe_default_until_shared_skin_is_ready() {
        // The key is intentionally unique because the shared handoff persists for the process;
        // this test must not depend on another cache/network worker.
        let key = "mc-render/entity/skin/player-skull-tilepass-test";
        let owner = BuiltinSkullOwner::new(
            key,
            crate::entity::SkinModel::Slim,
            crate::entity::SkinModel::Classic,
        );
        assert_eq!(owner.texture_key(), "textures/entity/steve.png");

        let (rgba, layout) = crate::entity::skin::normalise_skin(&vec![255; 64 * 64 * 4], 64, 64)
            .expect("test skin dimensions");
        let (mut cache, _channel) = crate::entity::skin::SkinCache::new();
        cache.insert(
            key,
            crate::entity::skin::SkinTexture {
                rgba: rgba.into(),
                layout,
                model: crate::entity::SkinModel::Slim,
            },
        );
        assert_eq!(owner.texture_key(), key);

        let item = BuiltinTileItem::PlayerSkull {
            skull_type: 3,
            owner,
        };
        assert_eq!(item.kind(), BuiltinTileItemKind::Skull);
        assert_eq!(item.texture_key(false).as_deref(), Some(key));
    }

    #[test]
    fn builtin_banner_uses_the_item_state_and_skips_uninitialised_state() {
        let state = mc_world::BannerState {
            base_color: 14,
            patterns: vec![("cr".to_owned(), 1)],
            initialized: true,
        };
        let item = BuiltinTileItem::Banner {
            state: state.clone(),
        };
        assert_eq!(
            item.texture_key(false).as_deref(),
            Some("mc-render/banner/b14cr1")
        );

        let mut models = TileModels::new(false);
        let mut vertices = Vec::new();
        let key = models.emit_builtin_item(&item, Mat4::IDENTITY, &mut vertices);
        assert_eq!(key.as_deref(), Some("mc-render/banner/b14cr1"));
        assert!(!vertices.is_empty());

        let uninitialised = BuiltinTileItem::Banner {
            state: mc_world::BannerState {
                initialized: false,
                ..state
            },
        };
        let mut skipped = Vec::new();
        assert!(models
            .emit_builtin_item(&uninitialised, Mat4::IDENTITY, &mut skipped)
            .is_none());
        assert!(skipped.is_empty());
    }

    #[test]
    fn plain_and_patterned_banner_items_keep_cloth_geometry_but_change_layered_color() {
        // Java's TileEntityItemStackRenderer reuses the same standing ModelBanner for both
        // stacks; only TileEntityBanner.setItemValues changes the layered texture design.
        let plain = mc_world::BannerState {
            base_color: 14,
            patterns: Vec::new(),
            initialized: true,
        };
        let patterned = mc_world::BannerState {
            patterns: vec![("cr".to_owned(), 1)],
            ..plain.clone()
        };

        let mut models = TileModels::new(false);
        let mut plain_vertices = Vec::new();
        let plain_key = models.emit_builtin_item(
            &BuiltinTileItem::Banner {
                state: plain.clone(),
            },
            Mat4::IDENTITY,
            &mut plain_vertices,
        );
        let mut patterned_vertices = Vec::new();
        let patterned_key = models.emit_builtin_item(
            &BuiltinTileItem::Banner {
                state: patterned.clone(),
            },
            Mat4::IDENTITY,
            &mut patterned_vertices,
        );

        assert_eq!(plain_key.as_deref(), Some("mc-render/banner/b14"));
        assert_eq!(patterned_key.as_deref(), Some("mc-render/banner/b14cr1"));
        assert!(!plain_vertices.is_empty());
        assert_eq!(
            plain_vertices, patterned_vertices,
            "patterns belong to LayeredColorMaskTexture; they must not change ModelBanner cloth"
        );

        // A tiny one-pixel oracle for the same Java compositor: start with the base layer and
        // replay the ordered pattern list.  The pattern must change the sampled colour while
        // retaining the exact MapColor arithmetic exercised by the mask tests below.
        let source = [100, 200, 50, 255];
        let mask = [255, 0, 0, 255];
        let composite = |state: &mc_world::BannerState| {
            let mut pixels = source;
            apply_banner_layer(&mut pixels, &source, &mask, state.base_color);
            for (_, colour) in &state.patterns {
                apply_banner_layer(&mut pixels, &source, &mask, *colour);
            }
            pixels
        };
        assert_ne!(
            composite(&plain),
            composite(&patterned),
            "a patterned item must select a visibly different layered texture"
        );

        // Negative control: a non-banner item is still absent from this renderer-family path;
        // the empty builtin model sentinel alone must never make ordinary items tile entities.
        assert_eq!(builtin_tile_item_kind(1), None);
    }

    #[test]
    fn brightness_is_monotonic_and_floored() {
        // java: `EntityRenderer.updateLightmap`'s `l / (4 - 3l)` curve, plus the ambient floor.
        assert_eq!(light_brightness(15), 1.0);
        assert_eq!(light_brightness(0), MIN_TILE_BRIGHTNESS);
        let mut last = 0.0;
        for level in 0..=15u8 {
            let b = light_brightness(level);
            assert!(b >= last, "level {level} went backwards: {b} < {last}");
            last = b;
        }
        // Half-ish light must read as clearly less than half brightness, which is the whole
        // point of the curve.
        assert!(light_brightness(7) < 0.4, "{}", light_brightness(7));
    }

    #[test]
    fn a_chest_emits_geometry_and_binds_the_right_texture() {
        let mut models = TileModels::new(false);
        let mut verts = Vec::new();
        let texture = models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            false,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut verts,
        );
        assert_eq!(texture, "textures/entity/chest/normal.png");
        assert!(!verts.is_empty(), "a chest must produce triangles");
        // Three parts x six faces x six vertices.
        assert_eq!(verts.len(), 3 * 6 * 6);
        // The whole chest sits inside its own block, give or take the lid's overhang.
        for v in &verts {
            assert!(
                (-0.01..=1.01).contains(&v.pos[0]) && (-0.01..=1.01).contains(&v.pos[2]),
                "chest vertex escaped its block: {:?}",
                v.pos
            );
        }
    }

    #[test]
    fn an_open_chest_lid_moves_the_geometry() {
        let mut models = TileModels::new(false);
        let mut shut = Vec::new();
        let mut open = Vec::new();
        models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            false,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            1.0,
            &mut shut,
        );
        models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            false,
            Vec3::ZERO,
            1.0,
            1.0,
            &[],
            1.0,
            &mut open,
        );
        assert_eq!(shut.len(), open.len());
        assert!(
            shut.iter().zip(&open).any(|(a, b)| a.pos != b.pos),
            "the lid must actually rotate"
        );
    }

    #[test]
    fn a_double_chest_is_wider_than_a_single_one() {
        let mut models = TileModels::new(false);
        let mut single = Vec::new();
        let mut double = Vec::new();
        models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            false,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut single,
        );
        let texture = models.emit_chest(
            Vec3::ZERO,
            2,
            false,
            true,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut double,
        );
        assert_eq!(texture, "textures/entity/chest/normal_double.png");
        let span = |v: &[EntityVertex]| {
            let min = v.iter().map(|x| x.pos[0]).fold(f32::MAX, f32::min);
            let max = v.iter().map(|x| x.pos[0]).fold(f32::MIN, f32::max);
            max - min
        };
        assert!(
            span(&double) > span(&single) + 0.5,
            "{} vs {}",
            span(&double),
            span(&single)
        );
    }

    #[test]
    fn a_double_trapped_chest_gets_its_own_texture() {
        // Before `LargeTrappedChest` existed, this bound `normal_double.png`.
        let mut models = TileModels::new(false);
        let mut verts = Vec::new();
        let texture = models.emit_chest(
            Vec3::ZERO,
            2,
            true,
            true,
            Vec3::ZERO,
            0.0,
            0.0,
            &[],
            0.0,
            &mut verts,
        );
        assert_eq!(texture, "textures/entity/chest/trapped_double.png");
    }

    #[test]
    fn christmas_overrides_both_chest_textures() {
        let mut models = TileModels::new(true);
        let mut verts = Vec::new();
        assert_eq!(
            models.emit_chest(
                Vec3::ZERO,
                2,
                true,
                false,
                Vec3::ZERO,
                0.0,
                0.0,
                &[],
                0.0,
                &mut verts
            ),
            "textures/entity/chest/christmas.png"
        );
        assert_eq!(
            models.emit_chest(
                Vec3::ZERO,
                2,
                false,
                true,
                Vec3::ZERO,
                0.0,
                0.0,
                &[],
                0.0,
                &mut verts
            ),
            "textures/entity/chest/christmas_double.png"
        );
    }

    #[test]
    fn a_wall_sign_hides_its_stick_and_a_standing_one_shows_it() {
        let mut models = TileModels::new(false);
        let mut standing = Vec::new();
        let mut wall = Vec::new();
        models.emit_sign(Vec3::ZERO, true, 0, &mut standing);
        models.emit_sign(Vec3::ZERO, false, 2, &mut wall);
        assert!(
            standing.len() > wall.len(),
            "the stick is one more box: {} vs {}",
            standing.len(),
            wall.len()
        );
        assert_eq!(standing.len() - wall.len(), 36, "one six-faced box");
    }

    #[test]
    fn a_banner_waves_and_neighbouring_banners_wave_out_of_phase() {
        // java: the `posX*7 + posY*9 + posZ*13` hash is what desynchronises a row of banners.
        let mut models = TileModels::new(false);
        let mut a = Vec::new();
        let mut b = Vec::new();
        models.emit_banner(Vec3::ZERO, true, 0, 0, 64, 0, 100, 0.0, &mut a);
        models.emit_banner(Vec3::ZERO, true, 0, 1, 64, 0, 100, 0.0, &mut b);
        assert_eq!(a.len(), b.len());
        assert!(
            a.iter().zip(&b).any(|(x, y)| x.pos != y.pos),
            "adjacent banners must not move in lockstep"
        );
    }

    #[test]
    fn banner_layers_use_java_map_colors_and_the_mask_red_channel() {
        // A red-dye layer (dye damage 1) over a greyscale source uses MapColor.redColor
        // (0x993333), not the softer item/particle tint table.  The second pixel has a green
        // mask but no red, which Java treats as transparent even though the mask alpha is 255.
        let source = [100, 200, 50, 255, 100, 200, 50, 255];
        let mask = [255, 0, 0, 255, 0, 255, 0, 255];
        let mut composite = source;
        apply_banner_layer(&mut composite, &source, &mask, 1);
        assert_eq!(&composite[..4], &[60, 40, 10, 255]);
        assert_eq!(&composite[4..], &[100, 200, 50, 255]);

        // Negative control: a nonzero red mask with zero alpha is skipped by Java's
        // `(i1 & -16777216) != 0` guard, so the layer cannot be made opaque by red alone.
        let transparent_mask = [255, 0, 0, 0, 255, 0, 0, 0];
        let mut untouched = source;
        apply_banner_layer(&mut untouched, &source, &transparent_mask, 1);
        assert_eq!(untouched, source);
    }

    #[test]
    fn banner_texture_keys_match_java_order_and_cap_at_sixteen_patterns() {
        let state = mc_world::BannerState {
            base_color: 14,
            patterns: vec![
                ("cr".to_owned(), 1),
                ("does_not_exist".to_owned(), 2),
                ("bo".to_owned(), 3),
            ],
            initialized: true,
        };
        assert_eq!(
            banner_texture_key(&state),
            "mc-render/banner/b14cr1bo3",
            "unknown ids do not become texture paths"
        );

        let capped = mc_world::BannerState {
            base_color: 0,
            patterns: (0..17).map(|i| ("cr".to_owned(), i)).collect(),
            initialized: true,
        };
        let key = banner_texture_key(&capped);
        assert!(key.ends_with("cr0cr1cr2cr3cr4cr5cr6cr7cr8cr9cr10cr11cr12cr13cr14cr15"));
        assert!(
            !key.ends_with("cr16"),
            "Java's j < 17 includes base + 16 patterns"
        );
    }

    #[test]
    fn banner_cache_caps_at_256_and_rejects_a_fresh_257th_design() {
        let clock = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
        let source = std::sync::Arc::clone(&clock);
        let mut cache = BannerTextureCache::with_clock(std::sync::Arc::new(move || {
            source.load(std::sync::atomic::Ordering::Relaxed)
        }));
        for index in 0..BANNER_CACHE_CAPACITY {
            cache.insert(format!("mc-render/banner/{index}"));
        }
        assert_eq!(cache.len(), BANNER_CACHE_CAPACITY);
        assert!(
            cache.make_room().is_none(),
            "Java returns null when all 256 designs are still fresh"
        );
        assert_eq!(cache.len(), BANNER_CACHE_CAPACITY);
    }

    #[test]
    fn banner_cache_removes_stale_entries_but_keeps_fresh_entries() {
        let clock = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
        let source = std::sync::Arc::clone(&clock);
        let mut cache = BannerTextureCache::with_clock(std::sync::Arc::new(move || {
            source.load(std::sync::atomic::Ordering::Relaxed)
        }));
        for index in 0..BANNER_CACHE_CAPACITY {
            cache.insert(format!("mc-render/banner/{index}"));
        }
        clock.store(
            BANNER_CACHE_IDLE_MILLIS + 1,
            std::sync::atomic::Ordering::Relaxed,
        );
        cache.touch("mc-render/banner/0");

        let evicted = cache
            .make_room()
            .expect("the 255 stale designs make room for a new one");
        assert_eq!(evicted.len(), BANNER_CACHE_CAPACITY - 1);
        assert!(
            cache.contains("mc-render/banner/0"),
            "the accessed entry is fresh"
        );
        assert!(
            !cache.contains("mc-render/banner/1"),
            "the idle entry is stale"
        );
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn banner_cache_refreshes_access_and_uses_strictly_more_than_60_seconds() {
        let clock = std::sync::Arc::new(std::sync::atomic::AtomicI64::new(0));
        let source = std::sync::Arc::clone(&clock);
        let mut cache = BannerTextureCache::with_clock(std::sync::Arc::new(move || {
            source.load(std::sync::atomic::Ordering::Relaxed)
        }));
        for index in 0..BANNER_CACHE_CAPACITY {
            cache.insert(format!("mc-render/banner/{index}"));
        }

        // Java's `> 60000L` leaves the exact boundary resident.  Refreshing entry zero at the
        // same time also proves a hit moves its idle deadline forward.
        clock.store(
            BANNER_CACHE_IDLE_MILLIS,
            std::sync::atomic::Ordering::Relaxed,
        );
        assert!(cache.touch("mc-render/banner/0"));
        assert!(
            cache.make_room().is_none(),
            "exactly 60 seconds is not stale, including the refreshed entry"
        );

        clock.store(
            BANNER_CACHE_IDLE_MILLIS + 1,
            std::sync::atomic::Ordering::Relaxed,
        );
        let evicted = cache
            .make_room()
            .expect("one millisecond past the edge makes the untouched entries stale");
        assert_eq!(evicted.len(), BANNER_CACHE_CAPACITY - 1);
        assert!(cache.contains("mc-render/banner/0"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn end_portal_layers_preserve_java_count_blend_textures_and_state() {
        let layers = end_portal_layers(END_PORTAL_SCROLL_PERIOD_MS + 1234);
        assert_eq!(layers.len(), END_PORTAL_LAYER_COUNT);
        assert_eq!(layers[0].texture, END_PORTAL_SKY_TEXTURE);
        assert!(!layers[0].additive, "the sky layer is source-over");
        assert_eq!(layers[0].extent, 65.0);
        assert_eq!(layers[0].texture_scale, 0.125);
        assert_eq!(layers[0].color, [0.1, 0.1, 0.1]);
        assert!(layers[1..]
            .iter()
            .all(|layer| { layer.texture == END_PORTAL_TEXTURE && layer.additive }));
        assert_eq!(layers[1].extent, 15.0);
        assert_eq!(layers[1].texture_scale, 0.5);
        assert_eq!(layers[2].extent, 14.0);
        assert_eq!(layers[2].texture_scale, 0.0625);
        assert_eq!(layers[0].scroll, end_portal_scroll(1234));
        assert_eq!(end_portal_scroll(END_PORTAL_SCROLL_PERIOD_MS), 0.0);
        // The three random draws consumed by layer zero make the first starfield colour a
        // useful negative control: dropping those draws would shift every layer after it.
        assert_ne!(layers[1].color, layers[2].color);
    }

    #[test]
    fn end_portal_emits_six_vertices_per_layer_with_inset_and_projective_uvs() {
        let rel = Vec3::new(-0.5, -1.6, 3.0);
        let mut vertices = Vec::new();
        emit_end_portal(rel, 42_000, &mut vertices);
        assert_eq!(
            vertices.len(),
            END_PORTAL_LAYER_COUNT * 6,
            "one quad (two triangles) per Java layer"
        );
        assert!(vertices.iter().all(|vertex| {
            (vertex.pos[1] - (rel.y + END_PORTAL_TOP_OFFSET)).abs() < f32::EPSILON
        }));
        for (index, layer_vertices) in vertices.chunks_exact(6).enumerate() {
            assert!(layer_vertices.iter().all(|vertex| vertex.color[3] == 1.0));
            assert!(layer_vertices
                .iter()
                .all(|vertex| vertex.color[..3] == end_portal_layers(42_000)[index].color));
        }
        // A non-horizon portal must carry real (non-NaN) coordinates, and the animated scroll
        // must move them. This catches accidentally binding plain 0..1 model UVs for every layer.
        assert!(vertices
            .iter()
            .all(|vertex| vertex.uv.iter().all(|value| value.is_finite())));
        let mut next = Vec::new();
        emit_end_portal(rel, 42_001, &mut next);
        assert!(vertices
            .iter()
            .zip(next.iter())
            .any(|(before, after)| before.uv != after.uv));

        // Negative control: the synthetic camera exactly on the surface must stay finite rather
        // than injecting an infinity into the GPU buffer.
        let mut horizon = Vec::new();
        emit_end_portal(Vec3::new(0.0, -END_PORTAL_TOP_OFFSET, 0.0), 0, &mut horizon);
        assert!(horizon
            .iter()
            .all(|vertex| vertex.uv.iter().all(|value| value.is_finite())));
    }

    #[test]
    fn missing_end_portal_resources_bind_vanillas_checkerboard_and_keep_all_layers() {
        let Some(gpu) = (match crate::Gpu::headless(
            4,
            4,
            crate::GpuOptions {
                vsync: false,
                sample_count: 1,
                low_power: true,
            },
        ) {
            Ok(gpu) => Some(gpu),
            Err(crate::RenderError::NoAdapter) => None,
            Err(error) => panic!("headless context creation failed: {error}"),
        }) else {
            eprintln!("skip: no wgpu adapter available on this machine");
            return;
        };
        let mut phase = TilePhase::new(&gpu);
        let resources = mc_assets::ResourceManager::new();

        assert!(phase.load_end_portal_textures(&gpu, &resources));
        assert!(phase.has_texture(END_PORTAL_SKY_TEXTURE));
        assert!(phase.has_texture(END_PORTAL_TEXTURE));
        assert!(
            phase.push_end_portal(Vec3::ZERO, 0),
            "a missing pack file must bind the Java missing texture, not suppress the TESR"
        );
        assert_eq!(phase.portal_batches.len(), END_PORTAL_LAYER_COUNT);
        assert_eq!(phase.portal_verts.len(), END_PORTAL_LAYER_COUNT * 6);
    }

    #[test]
    fn a_skull_picks_its_model_and_texture_from_the_type_byte() {
        let mut models = TileModels::new(false);
        let mut skeleton = Vec::new();
        let mut zombie = Vec::new();
        let mut player = Vec::new();
        assert_eq!(
            models.emit_skull(Vec3::ZERO, 1, 0, 0, &mut skeleton),
            Some("textures/entity/skeleton/skeleton.png")
        );
        assert_eq!(
            models.emit_skull(Vec3::ZERO, 1, 2, 0, &mut zombie),
            Some("textures/entity/zombie/zombie.png")
        );
        // java: a player head has no fixed texture — it wears the profile's skin.
        assert_eq!(models.emit_skull(Vec3::ZERO, 1, 3, 0, &mut player), None);
        assert!(!skeleton.is_empty() && !zombie.is_empty());
        assert!(
            zombie.len() > skeleton.len(),
            "the humanoid head adds a hat layer"
        );
    }

    #[test]
    fn a_placed_player_skull_uses_the_owner_skin_then_the_java_default() {
        let key = "mc-render/entity/skin/placed-skull-tilepass-test";
        let owner = BuiltinSkullOwner::new(
            key,
            crate::entity::SkinModel::Classic,
            crate::entity::SkinModel::Classic,
        );
        let mut models = TileModels::new(false);
        let mut missing = Vec::new();
        assert_eq!(
            models.emit_skull_with_owner(Vec3::ZERO, 1, 3, 0, Some(&owner), &mut missing),
            "textures/entity/steve.png",
            "an unready profile uses Java's legacy default"
        );
        assert!(!missing.is_empty());

        let (rgba, layout) = crate::entity::skin::normalise_skin(&vec![255; 64 * 64 * 4], 64, 64)
            .expect("test skin dimensions");
        let (mut cache, _channel) = crate::entity::skin::SkinCache::new();
        cache.insert(
            key,
            crate::entity::skin::SkinTexture {
                rgba: rgba.into(),
                layout,
                model: crate::entity::SkinModel::Classic,
            },
        );
        let mut ready = Vec::new();
        assert_eq!(
            models.emit_skull_with_owner(Vec3::ZERO, 1, 3, 0, Some(&owner), &mut ready),
            key,
            "the live owner key is selected once normalized pixels are resident"
        );

        let mut ownerless = Vec::new();
        assert_eq!(
            models.emit_skull_with_owner(Vec3::ZERO, 1, 3, 0, None, &mut ownerless),
            "textures/entity/steve.png"
        );
        let mut skeleton = Vec::new();
        assert_eq!(
            models.emit_skull_with_owner(Vec3::ZERO, 1, 0, 0, Some(&owner), &mut skeleton),
            "textures/entity/skeleton/skeleton.png",
            "a profile cannot override a non-player skull"
        );
    }

    #[test]
    fn a_wall_skull_is_pushed_against_its_block() {
        let mut models = TileModels::new(false);
        let mut floor = Vec::new();
        let mut north = Vec::new();
        models.emit_skull(Vec3::ZERO, 1, 0, 0, &mut floor);
        models.emit_skull(Vec3::ZERO, 2, 0, 0, &mut north);
        let mid = |v: &[EntityVertex]| v.iter().map(|x| x.pos[2]).sum::<f32>() / v.len() as f32;
        assert!(
            mid(&north) > mid(&floor),
            "{} vs {}",
            mid(&north),
            mid(&floor)
        );
    }

    #[test]
    fn the_enchantment_book_opens_by_its_spread() {
        let mut models = TileModels::new(false);
        let mut shut = Vec::new();
        let mut open = Vec::new();
        models.emit_book(Mat4::IDENTITY, 0, 0.0, (0.0, 0.0), 0.0, &mut shut);
        models.emit_book(Mat4::IDENTITY, 0, 0.0, (0.0, 0.0), 1.0, &mut open);
        assert_eq!(shut.len(), open.len());
        assert!(shut.iter().zip(&open).any(|(a, b)| a.pos != b.pos));
    }

    #[test]
    fn enchanting_book_screen_projection_matches_java_viewport_offset() {
        // Java's projection-space translate happens after the perspective matrix, so a point on
        // the model-view camera axis lands at NDC (-0.34, 0.23), not at the viewport centre.
        let projected = project_enchant_book_vertex(Vec3::new(0.0, 0.0, -16.0), 320, 240)
            .expect("camera-axis point is in the 9..80 clip range");
        assert!((projected[0] - 105.6).abs() < 5e-5, "x = {}", projected[0]);
        assert!((projected[1] - 92.4).abs() < 5e-5, "y = {}", projected[1]);
        assert!(projected[2].is_finite());

        // The viewport is centred in the logical surface, independently of the physical scale.
        let wider = project_enchant_book_vertex(Vec3::new(0.0, 0.0, -16.0), 854, 480)
            .expect("wider surface keeps the camera-axis point visible");
        assert!((wider[0] - 372.6).abs() < 5e-5, "x = {}", wider[0]);
        assert!((wider[1] - 212.4).abs() < 5e-5, "y = {}", wider[1]);

        // java: `(sw - 320) / 2` and `(sh - 240) / 2` are `int` divisions. 854x480 at GUI scale
        // 2 is a 427-wide surface: the viewport starts at 53, not 53.5. Its GL `y` is the bottom
        // margin, so an odd vertical margin leaves the extra pixel above the viewport.
        let odd = project_enchant_book_vertex(Vec3::new(0.0, 0.0, -16.0), 427, 241)
            .expect("odd surface keeps the camera-axis point visible");
        assert!((odd[0] - 158.6).abs() < 5e-5, "x = {}", odd[0]);
        assert!((odd[1] - 93.4).abs() < 5e-5, "y = {}", odd[1]);
    }

    #[test]
    fn enchanting_book_screen_model_emits_finite_posed_geometry() {
        let mut models = TileModels::new(false);
        let mut rest = Vec::new();
        let mut breathed = Vec::new();
        let texture = models.emit_book_with_age(
            enchant_book_screen_model_matrix(0.75),
            0.0,
            (0.2, 0.8),
            0.75,
            &mut rest,
        );
        let texture_again = models.emit_book_with_age(
            enchant_book_screen_model_matrix(0.75),
            31.0,
            (0.2, 0.8),
            0.75,
            &mut breathed,
        );
        assert_eq!(texture, "textures/entity/enchanting_table_book.png");
        assert_eq!(texture_again, texture);
        assert_eq!(rest.len(), breathed.len());
        assert_eq!(
            rest.len(),
            252,
            "seven ModelBook parts, six vertices per face"
        );
        assert!(rest.iter().all(|vertex| {
            vertex.pos.iter().all(|value| value.is_finite())
                && vertex.uv.iter().all(|value| value.is_finite())
        }));
        assert!(rest
            .iter()
            .zip(&breathed)
            .any(|(before, after)| before.pos != after.pos));

        let projected = rest
            .iter()
            .map(|vertex| project_enchant_book_vertex(Vec3::from(vertex.pos), 854, 480))
            .collect::<Option<Vec<_>>>()
            .expect("the full screen book remains in front of the camera");
        assert!(projected
            .iter()
            .all(|vertex| vertex.iter().all(|value| value.is_finite())));
    }

    #[test]
    fn lit_vertices_carry_the_brightness_through() {
        let verts = vec![EntityVertex::default(); 3];
        let out = lit(&verts, 0.25);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|v| v.light == 0.25));
        assert!(out.iter().all(|v| v.packed_light == NO_PACKED_LIGHT));
        assert!(out.iter().all(|v| v.color == [1.0, 1.0, 1.0, 1.0]));
    }

    #[test]
    fn piston_blend_is_java_source_alpha_over_and_alpha_test_is_strict() {
        let blend = piston_blend_state();
        assert_eq!(blend.color.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.color.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert_eq!(blend.alpha.src_factor, wgpu::BlendFactor::SrcAlpha);
        assert_eq!(blend.alpha.dst_factor, wgpu::BlendFactor::OneMinusSrcAlpha);
        assert!(TILE_WGSL.contains("fs_piston"));
        assert!(TILE_WGSL.contains("texel.a <= 0.1"));
        assert!(TILE_WGSL.contains("packed_light: u32"));
        assert!(TILE_WGSL.contains("@group(2) @binding(0) var lightmap_tex"));
        assert!(TILE_WGSL.contains("textureSample(lightmap_tex, lightmap_samp"));
        assert!(!TILE_WGSL.contains("max(lightmap.x, lightmap.y)"));
    }

    #[test]
    fn piston_lightmap_keeps_mixed_sky_and_block_coordinates() {
        let packed = crate::terrain::light::pack(4, 11);
        let uv = packed_lightmap_uv(packed);

        // Java's lightmap matrix maps the 4 and 11 nibbles to the centres of texels 4 and 11;
        // collapsing to max(4, 11) would make both coordinates point at texel 11 instead.
        assert_eq!(uv, [184.0 / 256.0, 72.0 / 256.0]);
        assert_ne!(uv[0], uv[1]);

        let lightmap = Lightmap::generate(LightmapParams::default());
        let mixed = lightmap.sample(4, 11);
        let max_channel_control = lightmap.sample(11, 11);
        assert_ne!(
            mixed, max_channel_control,
            "the generated Java table must retain the mixed sky/block sample"
        );
    }

    #[test]
    fn piston_model_lighting_keeps_mixed_packed_light_per_vertex() {
        let pos = BlockPos::new(0, 64, 0);
        let state = mc_data::block_by_name("minecraft:stone")
            .expect("stone is registered")
            .default_state();
        let mut world = crate::terrain::fixture::blank_world(1);
        world.set_block(pos, state);
        world.set_light_for(mc_world::SkyBlock::Sky, pos, 4);
        world.set_light_for(mc_world::SkyBlock::Block, pos, 11);
        world.set_light_for(mc_world::SkyBlock::Block, pos.north(), 11);

        let mut model = mc_assets::BakedModel::default();
        model.face_quads[Facing::North.index() as usize].push(mc_assets::BakedQuad {
            verts: [
                mc_assets::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                },
                mc_assets::Vertex {
                    pos: [1.0, 0.0, 0.0],
                    uv: [1.0, 0.0],
                },
                mc_assets::Vertex {
                    pos: [1.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                },
                mc_assets::Vertex {
                    pos: [0.0, 1.0, 0.0],
                    uv: [0.0, 1.0],
                },
            ],
            face: Some(Facing::North),
            shade_face: Some(Facing::North),
            tint_index: None,
            shade: true,
        });

        let mut out = Vec::new();
        emit_baked_block_model_lit(
            &world,
            state,
            pos,
            &model,
            Mat4::IDENTITY,
            true,
            crate::terrain::MeshSettings::default(),
            false,
            &mut out,
        );
        assert_eq!(out.len(), 6);
        assert!(out.iter().all(|v| v.packed_light != NO_PACKED_LIGHT));
        assert!(out
            .iter()
            .all(|v| crate::terrain::light::sky_byte(v.packed_light) > 0));
        assert!(out
            .iter()
            .any(|v| crate::terrain::light::block_byte(v.packed_light) > 0));
        assert!(out.iter().any(|v| v.color[0] < 1.0));
    }

    #[test]
    fn piston_carrier_cull_oracle_keeps_stained_glass_seams_and_slab_halves() {
        let pos = BlockPos::new(0, 64, 0);
        let mut world = crate::terrain::fixture::blank_world(1);
        let glass = crate::terrain::fixture::state_of("minecraft:stained_glass");
        world.set_block(pos, glass);
        world.set_block(pos.east(), glass);
        let settings = crate::terrain::MeshSettings::default();
        assert!(!crate::terrain::cull_face::should_side_be_rendered(
            &world,
            glass,
            pos,
            Facing::East,
            settings,
        ));
        let other_colour = mc_data::BlockState::new(glass.block(), glass.meta() ^ 1);
        world.set_block(pos.east(), other_colour);
        assert!(crate::terrain::cull_face::should_side_be_rendered(
            &world,
            glass,
            pos,
            Facing::East,
            settings,
        ));

        let bottom = crate::terrain::fixture::state_of("minecraft:stone_slab");
        let top = mc_data::BlockState::new(bottom.block(), bottom.meta() | 8);
        world.set_block(pos, bottom);
        world.set_block(pos.east(), bottom);
        assert!(!crate::terrain::cull_face::should_side_be_rendered(
            &world,
            bottom,
            pos,
            Facing::East,
            settings,
        ));
        world.set_block(pos.east(), top);
        assert!(crate::terrain::cull_face::should_side_be_rendered(
            &world,
            bottom,
            pos,
            Facing::East,
            settings,
        ));
    }

    #[test]
    fn baked_block_emitter_keeps_native_block_coordinates() {
        let mut model = mc_assets::BakedModel::default();
        model.general_quads.push(mc_assets::BakedQuad {
            verts: [
                mc_assets::Vertex {
                    pos: [0.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                },
                mc_assets::Vertex {
                    pos: [1.0, 0.0, 0.0],
                    uv: [1.0, 0.0],
                },
                mc_assets::Vertex {
                    pos: [1.0, 1.0, 0.0],
                    uv: [1.0, 1.0],
                },
                mc_assets::Vertex {
                    pos: [0.0, 1.0, 0.0],
                    uv: [0.0, 1.0],
                },
            ],
            face: Some(mc_math::Facing::North),
            shade_face: Some(mc_math::Facing::North),
            tint_index: None,
            shade: true,
        });

        let mut out = Vec::new();
        emit_baked_block_model(
            &model,
            Mat4::from_translation(Vec3::new(2.0, 3.0, 4.0)),
            &mut out,
        );
        assert_eq!(out.len(), 6, "one baked quad becomes two triangles");
        assert_eq!(out[0].pos, [2.0, 3.0, 4.0]);
        assert_eq!(out[1].pos, [3.0, 3.0, 4.0]);
        assert_eq!(out[2].pos, [3.0, 4.0, 4.0]);
    }
}
