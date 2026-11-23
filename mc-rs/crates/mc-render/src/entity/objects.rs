//! Non-living entity renderers.
//!
//! java: the `Render*` subclasses that extend `Render` directly rather than
//! `RendererLivingEntity` — `RenderEntityItem`, `RenderXPOrb`, `RenderArrow`, `RenderSnowball`,
//! `RenderPotion`, `RenderFireball`, `RenderTNTPrimed`, `RenderFallingBlock`, `RenderPainting`,
//! `RenderFish`, `RenderLightningBolt`, plus `RenderBoat`, `RenderMinecart`, `RenderLeashKnot`
//! and the two `tileentity`-package entity renderers `RenderEnderCrystal` and
//! `RenderWitherSkull`.
//!
//! # Why these cannot go through [`super::render::entity_matrix`]
//!
//! `RendererLivingEntity.doRender` builds a fixed transform — `translate`, `rotate(180 - yaw)`,
//! the death spin, `scale(-1, -1, 1)`, `preRenderCallback`, then the `-1.5078125` model lift —
//! and every mob shares it. A plain `Render` subclass builds its *own* matrix from scratch and
//! shares none of that: `RenderLeashKnot` never rotates by yaw at all, `RenderEnderCrystal` only
//! translates, `RenderBoat` lifts by `+0.25` rather than dropping by `1.5078125`, and the
//! billboards rotate to face the camera instead of the world. Pushing them through the living
//! matrix is A02-12: it puts boats and minecarts a block and a half above the water and the
//! rails. The matrix builders here are that fork, one function per vanilla renderer, so each one
//! can be diffed against its `doRender` line for line.
//!
//! # Three shapes of geometry
//!
//! * **Billboards** ([`camera_facing_matrix`], [`item_facing_matrix`]) — a single quad turned to
//!   face the view. Vanilla spells this `rotate(180 - playerViewY, Y); rotate(-playerViewX, X)`
//!   for the ones it draws itself (XP orb, fireball, fishing bobber) and the *un*-negated
//!   `rotate(-playerViewY, Y); rotate(playerViewX, X)` for the ones that hand a baked item model
//!   to `RenderItem` (every thrown item), because an item model is already drawn facing the
//!   viewer's `-Z`.
//! * **Baked block/item models** — TNT, falling blocks and the thrown items reuse
//!   [`super::item::emit_baked_model`], which is the same conversion the dropped-item renderer
//!   uses. Their UVs are block-atlas coordinates, so the caller must bind the atlas rather than
//!   an entity texture.
//! * **Hand-rolled quads** — the arrow's six faces and the painting's tiled cube are written out
//!   in `doRender` as raw tessellator calls with no model behind them, and are transcribed the
//!   same way here.
//!
//! Everything in this module is pure arithmetic over [`Mat4`] and [`EntityVertex`]: no GPU
//! state, no asset lookups beyond the [`mc_assets::BakedModel`]/[`mc_assets::Sprite`] a caller
//! passes in, and therefore all of it is unit-testable against the Java numbers.

use glam::{DVec3, Mat4, Vec3};
use mc_assets::{BakedModel, Sprite};
use mc_math::{Aabb, BlockPos, Facing, JavaRandom};
use mc_world::World;

use super::anim::DEG_TO_RAD;
use super::item::emit_baked_model_tinted;
use super::layers::LayerBlend;
use super::EntityVertex;

// ---------------------------------------------------------------------------------------------
// Vertex plumbing
// ---------------------------------------------------------------------------------------------

/// One transformed vertex.
///
/// Every vertex this module emits goes through here so the [`EntityVertex`] literal exists in
/// exactly one place — the struct grew a colour channel mid-flight and will grow a
/// lightmap coordinate eventually.
fn vertex(m: &Mat4, pos: Vec3, uv: [f32; 2], normal: Vec3, color: [f32; 4]) -> EntityVertex {
    EntityVertex {
        pos: m.transform_point3(pos).into(),
        uv,
        normal: m.transform_vector3(normal).normalize_or_zero().into(),
        color,
    }
}

/// Emits one `GL_QUADS` quad as two triangles, in vanilla's vertex order.
///
/// The `0-1-2 / 0-2-3` fan is the same split [`super::model::Model::emit_part`] and
/// [`super::item::emit_baked_model`] use, so all three agree on winding.
fn emit_quad(
    m: &Mat4,
    pos: [Vec3; 4],
    uv: [[f32; 2]; 4],
    normal: Vec3,
    color: [f32; 4],
    out: &mut Vec<EntityVertex>,
) {
    for &i in &[0usize, 1, 2, 0, 2, 3] {
        out.push(vertex(m, pos[i], uv[i], normal, color));
    }
}

/// The no-tint colour. java: the implicit `GlStateManager.color(1, 1, 1, 1)`. Only the XP orb
/// (`RenderXPOrb.java:56-59`) sets a real one; everything else in this module draws white.
const WHITE: [f32; 4] = EntityVertex::WHITE;

// ---------------------------------------------------------------------------------------------
// Billboards
// ---------------------------------------------------------------------------------------------

/// A quad turned to face the camera, for the renderers that build their own geometry.
///
/// java: `RenderXPOrb.java:49-50`, `RenderFireball.java:44-45`, `RenderFish.java:43-44` — all
/// three spell it `rotate(180 - playerViewY, 0, 1, 0)` then `rotate(-playerViewX, 1, 0, 0)`.
/// `renderManager.playerViewY`/`playerViewX` are the *camera's* interpolated yaw and pitch
/// (`RenderManager.cacheActiveRenderInfo`), i.e. [`crate::Camera::yaw`]/[`crate::Camera::pitch`].
///
/// `rel` is camera-relative — see [`crate::Camera::relative`].
#[must_use]
pub fn camera_facing_matrix(rel: Vec3, view_yaw: f32, view_pitch: f32) -> Mat4 {
    Mat4::from_translation(rel)
        * Mat4::from_rotation_y((180.0 - view_yaw) * DEG_TO_RAD)
        * Mat4::from_rotation_x(-view_pitch * DEG_TO_RAD)
}

/// A camera-facing frame for a *baked item model*, which vanilla orients the other way round.
///
/// java: `RenderSnowball.java:32-33` — `rotate(-playerViewY, 0, 1, 0)` then
/// `rotate(playerViewX, 1, 0, 0)`, with **no** `180 -` and the pitch **not** negated. The
/// difference from [`camera_facing_matrix`] is not a decompiler artefact: `RenderItem` draws a
/// flat item icon in the model's own `+Z`-facing orientation, so the billboard needs the inverse
/// of the camera rotation rather than the mirrored one a hand-built quad needs.
#[must_use]
pub fn item_facing_matrix(rel: Vec3, view_yaw: f32, view_pitch: f32) -> Mat4 {
    Mat4::from_translation(rel)
        * Mat4::from_rotation_y(-view_yaw * DEG_TO_RAD)
        * Mat4::from_rotation_x(view_pitch * DEG_TO_RAD)
}

// ---------------------------------------------------------------------------------------------
// RenderXPOrb
// ---------------------------------------------------------------------------------------------

/// `textures/entity/experience_orb.png`. java: `RenderXPOrb.experienceOrbTextures`
pub const XP_ORB_TEXTURE: &str = "textures/entity/experience_orb.png";

/// Which of the eleven 16x16 cells of `experience_orb.png` an orb of this value uses.
///
/// java: `EntityXPOrb.getTextureByXP` (`:259`) — a descending ladder of ten thresholds. The
/// value is the orb's *whole* XP, which `S11PacketSpawnExperienceOrb` carries and which nothing
/// else on the wire repeats, so it has to be kept from the spawn packet.
#[must_use]
pub fn xp_orb_texture_index(xp_value: i32) -> u32 {
    match xp_value {
        v if v >= 2477 => 10,
        v if v >= 1237 => 9,
        v if v >= 617 => 8,
        v if v >= 307 => 7,
        v if v >= 149 => 6,
        v if v >= 73 => 5,
        v if v >= 37 => 4,
        v if v >= 17 => 3,
        v if v >= 7 => 2,
        v if v >= 3 => 1,
        _ => 0,
    }
}

/// The `[min_u, max_u, min_v, max_v]` of cell `i` on the 64x64 orb sheet.
///
/// java: `RenderXPOrb.java:32-35` — `i % 4` picks the column and `i / 4` the row, each cell 16
/// texels of a 64-pixel sheet.
#[must_use]
pub fn xp_orb_uv(index: u32) -> [f32; 4] {
    let col = (index % 4) as f32;
    let row = (index / 4) as f32;
    [
        col * 16.0 / 64.0,
        (col * 16.0 + 16.0) / 64.0,
        row * 16.0 / 64.0,
        (row * 16.0 + 16.0) / 64.0,
    ]
}

/// The orb's animated green/yellow tint, as linear RGBA.
///
/// java: `RenderXPOrb.java:45-48` — `f9 = (xpColor + partialTicks) / 2`, red
/// `(sin(f9) + 1) * 0.5 * 255`, green a flat `255`, blue `(sin(f9 + 4.1887903) + 1) * 0.1 * 255`,
/// alpha a flat `128`. The `4.1887903` is `2*PI/3 * 2`, a 240-degree phase offset. `xpColor` is
/// a per-orb tick counter (`EntityXPOrb.xpColor`, incremented once per `onUpdate`), so
/// `ticks_existed` is the same number.
///
/// Vanilla writes these as *bytes* through `WorldRenderer.color(int, int, int, int)`, so the
/// truncation to an integer is reproduced rather than smoothed: it is visible as a slight
/// stepping in the colour cycle.
#[must_use]
pub fn xp_orb_color(xp_color: u32, partial_ticks: f32) -> [f32; 4] {
    let f9 = (xp_color as f32 + partial_ticks) / 2.0;
    let r = ((mc_math::sin(f9) + 1.0) * 0.5 * 255.0) as i32;
    let b = ((mc_math::sin(f9 + 4.188_790_3) + 1.0) * 0.1 * 255.0) as i32;
    [r as f32 / 255.0, 1.0, b as f32 / 255.0, 128.0 / 255.0]
}

/// One XP orb. java: `RenderXPOrb.doRender`.
///
/// The quad is `1.0` wide and `1.0` tall in local space, hung `0.5` left and `0.25` down of the
/// origin (`f4 = 1, f5 = 0.5, f6 = 0.25`), then scaled by `0.3`.
pub fn emit_xp_orb(
    rel: Vec3,
    view_yaw: f32,
    view_pitch: f32,
    xp_value: i32,
    xp_color: u32,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    // java: `GlStateManager.scale(0.3F, 0.3F, 0.3F)` — after the two camera rotations.
    let m = camera_facing_matrix(rel, view_yaw, view_pitch) * Mat4::from_scale(Vec3::splat(0.3));
    let [u0, u1, v0, v1] = xp_orb_uv(xp_orb_texture_index(xp_value));
    let color = xp_orb_color(xp_color, partial_ticks);
    emit_quad(
        &m,
        [
            Vec3::new(-0.5, -0.25, 0.0),
            Vec3::new(0.5, -0.25, 0.0),
            Vec3::new(0.5, 0.75, 0.0),
            Vec3::new(-0.5, 0.75, 0.0),
        ],
        // java: the first two vertices take `f3` (the *bottom* of the cell) and the last two
        // `f2` — the sheet is drawn V-flipped relative to the quad.
        [[u0, v1], [u1, v1], [u1, v0], [u0, v0]],
        Vec3::Y,
        color,
        out,
    );
}

// ---------------------------------------------------------------------------------------------
// RenderArrow
// ---------------------------------------------------------------------------------------------

/// `textures/entity/arrow.png`. java: `RenderArrow.arrowTextures`
pub const ARROW_TEXTURE: &str = "textures/entity/arrow.png";

/// The arrow's outer transform, before the `0.05625` model scale.
///
/// java: `RenderArrow.java:29-55` — `translate(x, y, z)`, `rotate(yaw - 90, Y)`,
/// `rotate(pitch, Z)`, the wobble, `rotate(45, X)`, `scale(0.05625)`, `translate(-4, 0, 0)`.
///
/// `shake` is `EntityArrow.arrowShake - partialTicks`; while positive the arrow rattles about Z
/// by `-sin(shake * 3) * shake` degrees, which is what an arrow that has just struck a mob does.
#[must_use]
pub fn arrow_matrix(rel: Vec3, yaw: f32, pitch: f32, shake: f32) -> Mat4 {
    arrow_matrix_from(Mat4::from_translation(rel), yaw, pitch, shake)
}

/// [`arrow_matrix`] composed after an existing attachment matrix.
///
/// `LayerArrow` calls `ModelRenderer.postRender`, translates to a point inside that part's
/// selected cube, and only then invokes `RenderArrow` at local `(0, 0, 0)`. Supplying that posed
/// matrix here keeps the standalone projectile path and the stuck-arrow layer on one renderer.
#[must_use]
pub fn arrow_matrix_from(base: Mat4, yaw: f32, pitch: f32, shake: f32) -> Mat4 {
    let mut m = base
        * Mat4::from_rotation_y((yaw - 90.0) * DEG_TO_RAD)
        * Mat4::from_rotation_z(pitch * DEG_TO_RAD);
    if shake > 0.0 {
        m *= Mat4::from_rotation_z(-mc_math::sin(shake * 3.0) * shake * DEG_TO_RAD);
    }
    m * Mat4::from_rotation_x(45.0 * DEG_TO_RAD)
        * Mat4::from_scale(Vec3::splat(ARROW_SCALE))
        * Mat4::from_translation(Vec3::new(-4.0, 0.0, 0.0))
}

/// java: `RenderArrow`'s `f8` — `scale(0.05625, 0.05625, 0.05625)`, which is `0.9 / 16`, not the
/// `0.0625` every `ModelBase` uses.
pub const ARROW_SCALE: f32 = 0.056_25;

/// The arrow's six quads. java: `RenderArrow.doRender`'s two end caps and four fletching vanes.
pub fn emit_arrow(rel: Vec3, yaw: f32, pitch: f32, shake: f32, out: &mut Vec<EntityVertex>) {
    emit_arrow_from(Mat4::from_translation(rel), yaw, pitch, shake, out);
}

/// Emits an arrow after a caller-supplied posed attachment matrix.
pub fn emit_arrow_from(base: Mat4, yaw: f32, pitch: f32, shake: f32, out: &mut Vec<EntityVertex>) {
    let m = arrow_matrix_from(base, yaw, pitch, shake);
    // java: `int i = 0` — the arrow's damage-tier row, always 0 in 1.8.9.
    let (f, f1) = (0.0f32, 0.5f32);
    let (f2, f3) = (0.0 / 32.0, 5.0 / 32.0);
    let (f4, f5) = (0.0f32, 0.156_25f32);
    let (f6, f7) = (5.0 / 32.0, 10.0 / 32.0);

    // java: `GL11.glNormal3f(f8, 0, 0)` then the +X cap.
    emit_quad(
        &m,
        [
            Vec3::new(-7.0, -2.0, -2.0),
            Vec3::new(-7.0, -2.0, 2.0),
            Vec3::new(-7.0, 2.0, 2.0),
            Vec3::new(-7.0, 2.0, -2.0),
        ],
        [[f4, f6], [f5, f6], [f5, f7], [f4, f7]],
        Vec3::X,
        WHITE,
        out,
    );
    // java: `GL11.glNormal3f(-f8, 0, 0)` then the same cap wound the other way.
    emit_quad(
        &m,
        [
            Vec3::new(-7.0, 2.0, -2.0),
            Vec3::new(-7.0, 2.0, 2.0),
            Vec3::new(-7.0, -2.0, 2.0),
            Vec3::new(-7.0, -2.0, -2.0),
        ],
        [[f4, f6], [f5, f6], [f5, f7], [f4, f7]],
        -Vec3::X,
        WHITE,
        out,
    );
    // java: four vanes, each preceded by `rotate(90, 1, 0, 0)` — so the first is already turned
    // 90 degrees, not flat, and the fourth returns to the start.
    let mut vane = m;
    for _ in 0..4 {
        vane *= Mat4::from_rotation_x(90.0 * DEG_TO_RAD);
        emit_quad(
            &vane,
            [
                Vec3::new(-8.0, -2.0, 0.0),
                Vec3::new(8.0, -2.0, 0.0),
                Vec3::new(8.0, 2.0, 0.0),
                Vec3::new(-8.0, 2.0, 0.0),
            ],
            [[f, f2], [f1, f2], [f1, f3], [f, f3]],
            Vec3::Z,
            WHITE,
            out,
        );
    }
}

/// The entity-pass state a free arrow is drawn with.
///
/// java: `RenderArrow.doRender` touches neither `GL_CULL_FACE` nor blending, so it draws under
/// the world pass's state (`EntityRenderer.renderWorldPass:1328-1402`): back faces culled,
/// lighting on. Each of its six quads is one of a coplanar pair wound opposite ways (the two end
/// caps share `x = -7`; vanes 1 and 3, and 2 and 4, are 180 degrees apart), so culling leaves
/// exactly the camera-facing quad of each pair. Drawn two-sided, the pairs z-fight, which is what
/// "z-fighting on arrows stuck in the ground" was.
///
/// The fixed-function lighting is folded into the vertex colour by [`emit_arrow_lit`] (see
/// [`arrow_light`]), so the pipeline is the unlit cutout one. A picked-up arrow
/// (`EntityPickupFX`, drawn in `renderLitParticles` with lighting off and culling still on) uses
/// the same state with white vertices from [`emit_arrow`]; an arrow stuck in a mob does not
/// (`LayerArrow` runs inside `RendererLivingEntity`'s `disableCull()`, see
/// [`super::layers::LayerKind::blend`]).
pub const ARROW_BLEND: LayerBlend = LayerBlend::OpaqueUnlitCulled;

/// `RenderArrow`'s fixed-function brightness for one face, from its world-space unit normal.
///
/// java: `GL11.glNormal3f(f8, 0, 0)` for the caps and `(0, 0, f8)` for the vanes, `f8 = 0.05625`,
/// under `GlStateManager.enableRescaleNormal()`. `GL_RESCALE_NORMAL` divides out only the
/// modelview's own uniform scale, which is that same `scale(f8, f8, f8)`, so the eye-space normal
/// keeps the length `0.05625`; `GL_NORMALIZE` stays off (only `LayerSlimeGel` turns it on). The two
/// `RenderHelper.enableStandardItemLighting` lights therefore add `0.6 * 0.05625 * max(n.L, 0)`
/// each over the `0.4` ambient floor: a free arrow is lit between 0.40 and 0.455, nearly ambient
/// alone, where a unit normal (the entity shader's) gives 0.4 up to a clamped 1.0. Measured on the
/// real GL by `ArrowRenderRef` (`GL_CURRENT_NORMAL` `(0.05625, 0, 0)`, rescale on, normalize off).
#[must_use]
pub fn arrow_light(normal: Vec3) -> f32 {
    // java: `RenderHelper.LIGHT0_POS`/`LIGHT1_POS`, normalised.
    let light0 = Vec3::new(0.2, 1.0, -0.7).normalize();
    let light1 = Vec3::new(-0.2, 1.0, 0.7).normalize();
    let n = normal * ARROW_SCALE;
    (0.4 + 0.6 * n.dot(light0).max(0.0) + 0.6 * n.dot(light1).max(0.0)).min(1.0)
}

/// Multiplies [`arrow_light`] into each vertex colour, from the vertex's own normal.
///
/// java: `COLOR_MATERIAL` makes the current colour both the ambient and the diffuse material, so
/// the lit colour is the light factor times that colour, clamped, before `GL_MODULATE` applies
/// the texel (and the lightmap stage after it).
pub fn light_arrow(verts: &mut [EntityVertex]) {
    for vertex in verts {
        let light = arrow_light(Vec3::from(vertex.normal));
        for channel in &mut vertex.color[..3] {
            *channel = (*channel * light).clamp(0.0, 1.0);
        }
    }
}

/// A free arrow, as `RenderArrow` draws it in the world: [`emit_arrow`]'s quads lit by
/// [`light_arrow`], for [`ARROW_BLEND`].
pub fn emit_arrow_lit(rel: Vec3, yaw: f32, pitch: f32, shake: f32, out: &mut Vec<EntityVertex>) {
    let start = out.len();
    emit_arrow(rel, yaw, pitch, shake, out);
    light_arrow(&mut out[start..]);
}

// ---------------------------------------------------------------------------------------------
// RenderSnowball / RenderPotion — thrown items
// ---------------------------------------------------------------------------------------------

/// java: `RenderItem.renderItem(ItemStack, IBakedModel)` (`:145`) — the `scale(0.5)` **every**
/// item draw applies before the `translate(-0.5, -0.5, -0.5)` recentring.
///
/// This is the half of the item pipeline that is easy to lose: `RenderEntityItem.doRender`'s
/// gui-3d branch scales by `0.5` *as well* (`:125`), so a dropped block ends up a quarter of a
/// block across, not a half.
pub const RENDER_ITEM_SCALE: f32 = 0.5;

/// The `display.<context>` transform as a matrix.
///
/// java: `ItemCameraTransforms.applyTransform` — `translate`, then `rotate(y, Y)`,
/// `rotate(x, X)`, `rotate(z, Z)` in that order, then `scale`. Vanilla skips the whole block
/// when the transform is the default, which is exactly the identity this produces anyway.
#[must_use]
pub fn item_transform_matrix(t: &mc_assets::model::ItemTransform) -> Mat4 {
    Mat4::from_translation(Vec3::from(t.translation))
        * Mat4::from_rotation_y(t.rotation[1] * DEG_TO_RAD)
        * Mat4::from_rotation_x(t.rotation[0] * DEG_TO_RAD)
        * Mat4::from_rotation_z(t.rotation[2] * DEG_TO_RAD)
        * Mat4::from_scale(Vec3::from(t.scale))
}

/// The `RenderItem.preTransform` branch for an item model.
///
/// java: `RenderItem.renderItemModelTransform` scales flat (`!model.isGui3d()`) models by `2`
/// before applying the selected `display` transform.  GUI-3D models take their `2` from the
/// caller's `ItemRenderer.renderItem` wrapper instead, so this helper intentionally returns the
/// identity for them.  Keeping this branch explicit is important for held and fixed/frame
/// callers: the later, unconditional `RenderItem.renderItem` half-scale does not replace it.
#[must_use]
pub fn item_model_pre_transform(gui_3d: bool) -> Mat4 {
    if gui_3d {
        Mat4::IDENTITY
    } else {
        Mat4::from_scale(Vec3::splat(2.0))
    }
}

/// Compose the display transform around an `ItemRenderer.renderItem` call.
///
/// `ItemRenderer` supplies an outer `scale(2)` for GUI-3D held/head items; `RenderItem` supplies
/// the same scale in [`item_model_pre_transform`] for flat models.  The selected display context
/// is applied after that branch, exactly as `ItemCameraTransforms.applyTransform` does.
#[must_use]
pub fn item_renderer_display_transform(
    gui_3d: bool,
    display: &mc_assets::model::ItemTransform,
) -> Mat4 {
    let outer = if gui_3d {
        Mat4::from_scale(Vec3::splat(2.0))
    } else {
        Mat4::IDENTITY
    };
    outer * item_model_pre_transform(gui_3d) * item_transform_matrix(display)
}

/// Compose an ordinary baked model's complete `ItemRenderer`/`RenderItem` item transform.
///
/// This includes the unconditional common `scale(0.5)` immediately before the baked model's
/// `(-.5, -.5, -.5)` recenter (the recenter itself remains in [`super::item::
/// emit_baked_model_tinted`]).  The GUI-3D outer/pre scales and this common half cancel for the
/// default display transform, but retaining all three in order keeps custom display translation
/// and scale values faithful.
#[must_use]
pub fn item_renderer_model_transform(
    gui_3d: bool,
    display: &mc_assets::model::ItemTransform,
) -> Mat4 {
    item_renderer_display_transform(gui_3d, display)
        * Mat4::from_scale(Vec3::splat(RENDER_ITEM_SCALE))
}

/// A thrown item — snowball, egg, ender pearl, eye of ender, XP bottle, firework, splash potion.
///
/// java: `RenderSnowball.doRender` (`:26-39`) — `translate`, `scale(0.5)`, the two camera
/// rotations, then `renderItem(stack, GROUND)`. `RenderPotion` is the same renderer with a
/// damage-valued stack, and `RenderManager`'s constructor registers six more item types against
/// this one class.
///
/// The emitted UVs are block-atlas coordinates (the item model is baked against the stitched
/// atlas), so the caller must push these under the atlas texture, not an entity texture.
pub fn emit_thrown_item(
    rel: Vec3,
    view_yaw: f32,
    view_pitch: f32,
    model: &BakedModel,
    out: &mut Vec<EntityVertex>,
) {
    emit_thrown_item_tinted(rel, view_yaw, view_pitch, model, out, |_| 0xFF_FF_FF);
}

/// A thrown item with vanilla's per-quad item tint hook.
///
/// java: `RenderSnowball.doRender` supplies the item-facing transform, then
/// `RenderItem.renderItem` applies the ground transform, recentres the model and calls
/// `Item.getColorFromItemStack` only for quads carrying a tint index. Keeping the transform here
/// (rather than in the client dispatcher) makes `RenderPotion` and every other
/// `RenderSnowball` consumer share one geometry/tint seam. The callback is not consulted for
/// untagged bottle layers, matching `RenderItem.renderQuads`'s `hasTintIndex()` guard.
pub fn emit_thrown_item_tinted(
    rel: Vec3,
    view_yaw: f32,
    view_pitch: f32,
    model: &BakedModel,
    out: &mut Vec<EntityVertex>,
    tint: impl FnMut(u8) -> u32,
) {
    let m = thrown_item_matrix(rel, view_yaw, view_pitch, model);
    emit_baked_model_tinted(model, m, out, tint);
}

/// `RenderSnowball`'s complete item transform, up to (not including) the baked model's
/// `(-0.5, -0.5, -0.5)` recentre. The glint pass of a foiled thrown item reuses it.
///
/// java: `RenderSnowball.doRender` (`:29-35`) — `translate`, `scale(0.5)`, the two camera
/// rotations — then `RenderItem.renderItem(stack, GROUND)` (`:263-270`) goes through
/// `renderItemModelTransform` (`:316-334`), whose `preTransform` (`:245-261`) first scales a flat
/// (`!isGui3d()`) model by `2`, then `applyTransform(GROUND)`, then `renderItem(stack, model)`'s
/// `scale(0.5)` (`:145`). Every 1.8.9 thrown item is `builtin/generated` with no `ground` display,
/// so its sprite spans `0.5 * 2 * 0.5` = half a block, like the dropped item. The bytecode agrees:
/// `bkc.a` calls `bjh.a(zx, bgr$b)`, which reaches `bjh.b(zx)`'s `if (!boq.c()) scale(2, 2, 2)`.
/// Leaving out that `2` drew pearls, snowballs, eggs, potions and XP bottles at half size.
#[must_use]
pub fn thrown_item_matrix(rel: Vec3, view_yaw: f32, view_pitch: f32, model: &BakedModel) -> Mat4 {
    // java: `scale(0.5)` comes *before* the two rotations; a uniform scale commutes with them.
    item_facing_matrix(rel, view_yaw, view_pitch)
        * Mat4::from_scale(Vec3::splat(0.5))
        * item_model_pre_transform(model.gui_3d)
        * item_transform_matrix(&model.transforms.ground)
        * Mat4::from_scale(Vec3::splat(RENDER_ITEM_SCALE))
}

// ---------------------------------------------------------------------------------------------
// RenderFireball
// ---------------------------------------------------------------------------------------------

/// java: `RenderManager` registers `new RenderFireball(this, 2.0F)` for `EntityLargeFireball`.
pub const LARGE_FIREBALL_SCALE: f32 = 2.0;

/// java: `RenderManager` registers `new RenderFireball(this, 0.5F)` for `EntitySmallFireball`.
pub const SMALL_FIREBALL_SCALE: f32 = 0.5;

/// A ghast/blaze fireball. java: `RenderFireball.doRender`.
///
/// The sprite is the *particle icon* of `Items.fire_charge` — `blocks/fire_charge` on the block
/// atlas — so like [`emit_thrown_item`] this must be pushed under the atlas key.
pub fn emit_fireball(
    rel: Vec3,
    view_yaw: f32,
    view_pitch: f32,
    scale: f32,
    sprite: &Sprite,
    out: &mut Vec<EntityVertex>,
) {
    let m = Mat4::from_translation(rel)
        * Mat4::from_scale(Vec3::splat(scale))
        * Mat4::from_rotation_y((180.0 - view_yaw) * DEG_TO_RAD)
        * Mat4::from_rotation_x(-view_pitch * DEG_TO_RAD);
    let uv = sprite.uv;
    emit_quad(
        &m,
        [
            Vec3::new(-0.5, -0.25, 0.0),
            Vec3::new(0.5, -0.25, 0.0),
            Vec3::new(0.5, 0.75, 0.0),
            Vec3::new(-0.5, 0.75, 0.0),
        ],
        [
            [uv.min_u, uv.max_v],
            [uv.max_u, uv.max_v],
            [uv.max_u, uv.min_v],
            [uv.min_u, uv.min_v],
        ],
        Vec3::Y,
        WHITE,
        out,
    );
}

// ---------------------------------------------------------------------------------------------
// RenderFish — the fishing bobber
// ---------------------------------------------------------------------------------------------

/// java: `RenderFish.FISH_PARTICLES` — the bobber is one 8x8 cell of the particle sheet, not an
/// entity texture of its own.
pub const FISH_TEXTURE: &str = "textures/particle/particles.png";

/// java: `RenderGuardian.GUARDIAN_BEAM_TEXTURE`.
pub const GUARDIAN_BEAM_TEXTURE: &str = "textures/entity/guardian_beam.png";

/// The state `RenderFish` reads from its angler while building the untextured line.  Keeping the
/// tick samples here lets the client supply either the session-owned local player or a
/// generation-checked remote player without making this pure renderer know about ECS.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FishingLineAngler {
    /// `Entity.prevPosX/Y/Z`.
    pub prev_pos: DVec3,
    /// `Entity.posX/Y/Z`.
    pub pos: DVec3,
    /// `Entity.prevRotationYaw` / `rotationYaw`.
    pub prev_yaw: f32,
    /// Current `Entity.rotationYaw`.
    pub yaw: f32,
    /// `Entity.prevRotationPitch` / `rotationPitch`.
    pub prev_pitch: f32,
    /// Current `Entity.rotationPitch`.
    pub pitch: f32,
    /// `Entity.prevRenderYawOffset` / `renderYawOffset` for the third-person line branch.
    pub prev_body_yaw: f32,
    /// Current `Entity.renderYawOffset`.
    pub body_yaw: f32,
    /// `EntityLivingBase.getEyeHeight()` at the current tick.
    pub eye_height: f32,
    /// `EntityLivingBase.isSneaking()`.
    pub sneaking: bool,
    /// `Entity.prevSwingProgress` / `swingProgress`.
    pub prev_swing_progress: f32,
    /// Current `Entity.swingProgress`.
    pub swing_progress: f32,
    /// Whether this is the session-owned local player. Remote owners always take Java's
    /// third-person branch because `angler != Minecraft.thePlayer`.
    pub is_local: bool,
    /// `GameSettings.thirdPersonView > 0` for the local player.
    pub third_person: bool,
}

/// Emit the 17-vertex `RenderFish` line (`GL_LINE_STRIP`) into camera-relative space.
///
/// java: `RenderFish.doRender` (`:54-98`), checked vertex for vertex against the real renderer
/// by `tests/fishing_line_parity.rs`. `bobber_pos` is the hook's interpolated position; Java's
/// `d5` adds `0.25` to its Y before the endpoint difference is taken, so the rod end of the line
/// lands exactly on the rod tip. The three endpoint differences are narrowed to `float` where
/// Java narrows them, and the first-person eye height (or the third-person sneak offset) is added
/// after that narrowing, as `d11 = (float)(d1 - d5) + d3` does.
pub fn emit_fishing_line(
    bobber_pos: DVec3,
    camera_pos: DVec3,
    angler: FishingLineAngler,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    let partial = f64::from(partial_ticks);
    let bobber = bobber_pos - camera_pos;
    let angler_pos = angler.prev_pos + (angler.pos - angler.prev_pos) * partial;
    // java: `EntityLivingBase.getSwingProgress` wraps the per-tick delta, not the result, so the
    // last frames of a swing keep rising towards 1.0 instead of running back to 0.
    let mut swing_delta = angler.swing_progress - angler.prev_swing_progress;
    if swing_delta < 0.0 {
        swing_delta += 1.0;
    }
    let swing = angler.prev_swing_progress + swing_delta * partial_ticks;
    let swing_sine = mc_math::sin(mc_math::sqrt_f(swing) * core::f32::consts::PI);

    // `(tip, d3)`: the rod tip, and what Java adds to the narrowed Y difference.
    let (tip, d3) = if !angler.is_local || angler.third_person {
        // java: `RenderFish`'s `angler != Minecraft.thePlayer` / third-person branch.
        let body = (angler.prev_body_yaw
            + (angler.body_yaw - angler.prev_body_yaw) * partial_ticks)
            * core::f32::consts::PI
            / 180.0;
        let sin_body = f64::from(mc_math::sin(body));
        let cos_body = f64::from(mc_math::cos(body));
        (
            DVec3::new(
                angler_pos.x - cos_body * 0.35 - sin_body * 0.8,
                angler.prev_pos.y
                    + f64::from(angler.eye_height)
                    + (angler.pos.y - angler.prev_pos.y) * partial
                    - 0.45,
                angler_pos.z - sin_body * 0.35 + cos_body * 0.8,
            ),
            if angler.sneaking { -0.1875 } else { 0.0 },
        )
    } else {
        // java: the first-person hand offset, a double `Vec3` turned with float trig.
        let pitch = angler.prev_pitch + (angler.pitch - angler.prev_pitch) * partial_ticks;
        let yaw = angler.prev_yaw + (angler.yaw - angler.prev_yaw) * partial_ticks;
        let mut hand = DVec3::new(-0.36, 0.03, 0.35);
        hand = rotate_pitch(hand, -pitch * core::f32::consts::PI / 180.0);
        hand = rotate_yaw(hand, -yaw * core::f32::consts::PI / 180.0);
        hand = rotate_yaw(hand, swing_sine * 0.5);
        hand = rotate_pitch(hand, -swing_sine * 0.7);
        (angler_pos + hand, f64::from(angler.eye_height))
    };

    // java: `d9`/`d12` and `d11 = (float)(d1 - d5) + d3`, where `d5` is the bobber Y + 0.25.
    let dx = f64::from((tip.x - bobber_pos.x) as f32);
    let dy = f64::from((tip.y - (bobber_pos.y + 0.25)) as f32) + d3;
    let dz = f64::from((tip.z - bobber_pos.z) as f32);
    for index in 0..=16 {
        let f = index as f32 / 16.0;
        out.push(EntityVertex {
            pos: [
                (bobber.x + dx * f64::from(f)) as f32,
                (bobber.y + dy * f64::from(f * f + f) * 0.5 + 0.25) as f32,
                (bobber.z + dz * f64::from(f)) as f32,
            ],
            uv: [0.0, 0.0],
            normal: [0.0, 1.0, 0.0],
            color: [0.0, 0.0, 0.0, 1.0],
        });
    }
}

/// java: `Vec3.rotatePitch` — float `MathHelper` trig, double products. Note the sign: a
/// positive angle turns +Z towards +Y (`y' = y cos + z sin`), so `rotatePitch(-pitch)` lowers
/// the hand when the angler looks down.
fn rotate_pitch(vector: DVec3, angle: f32) -> DVec3 {
    let cos = f64::from(mc_math::cos(angle));
    let sin = f64::from(mc_math::sin(angle));
    DVec3::new(
        vector.x,
        vector.y * cos + vector.z * sin,
        vector.z * cos - vector.y * sin,
    )
}

/// java: `Vec3.rotateYaw` — float `MathHelper` trig, double products.
fn rotate_yaw(vector: DVec3, angle: f32) -> DVec3 {
    let cos = f64::from(mc_math::cos(angle));
    let sin = f64::from(mc_math::sin(angle));
    DVec3::new(
        vector.x * cos + vector.z * sin,
        vector.y,
        vector.z * cos - vector.x * sin,
    )
}

/// Emit both `RenderLiving.renderLeash` triangle strips.  The caller submits the first 50 and
/// second 50 vertices as separate batches: Java calls `tessellator.draw()` between strips, so a
/// single GPU triangle-strip draw would incorrectly join the two strips with a diagonal.
pub fn emit_leash(start: DVec3, holder_delta: DVec3, out: &mut Vec<EntityVertex>) {
    for strip in 0..2 {
        for index in 0..=24 {
            let f = index as f32 / 24.0;
            let base = if index % 2 == 0 {
                [0.35, 0.28, 0.21, 1.0]
            } else {
                [0.5, 0.4, 0.3, 1.0]
            };
            let y = (24 - index) as f32 / 18.0 + 0.125;
            let point = Vec3::new(
                (start.x as f32) + holder_delta.x as f32 * f,
                (start.y as f32) + holder_delta.y as f32 * (f * f + f) * 0.5 + y,
                (start.z as f32) + holder_delta.z as f32 * f,
            );
            let (a, b) = if strip == 0 {
                (point, point + Vec3::new(0.025, 0.025, 0.0))
            } else {
                (
                    point + Vec3::new(0.0, 0.025, 0.0),
                    point + Vec3::new(0.025, 0.0, 0.025),
                )
            };
            for position in [a, b] {
                out.push(EntityVertex {
                    pos: position.into(),
                    uv: [0.0, 0.0],
                    normal: [0.0, 1.0, 0.0],
                    color: base,
                });
            }
        }
    }
}

/// Emit `RenderGuardian.doRender`'s beam (`:78-158`) in camera-relative space: Java's three
/// `GL_QUADS` (two crossed planes along the beam, then the end cap) as 18 triangle-list
/// vertices, the `0-1-2 / 0-2-3` split `emit_quad` uses. The entity phase draws a plain
/// triangle list, so Java's four-corner quads must not reach it as they are.
///
/// The inputs are Java's: `guardian_eye` is `func_177110_a(entity, getEyeHeight(), partial)`,
/// the guardian's interpolated feet plus its eye height; `target_centre` is the target's
/// interpolated feet plus half its height; `progress` is `func_175477_p(partial)`, which runs
/// past `1.0` once the wind-up counter is full; `world_time` is `getTotalWorldTime()`. The
/// arithmetic keeps Java's precision: a float world time and texture scroll, a double phase,
/// `MathHelper.sqrt_double`'s float length, `Math.acos`/`Math.atan2`, the exact rotations
/// `glRotatef` builds from float degrees, and colour bytes wrapped by `WorldRenderer.color`'s
/// `(byte)` casts. The texture wrap, alpha test, lightmap and depth state belong to
/// [`LayerBlend::GuardianBeam`]. Checked vertex for vertex against the real renderer by
/// `tests/guardian_beam_java_parity.rs`.
// Keep Java's explicit beam-input order visible at this seam; a context object would obscure the
// renderer's one-to-one argument mapping.
#[allow(clippy::too_many_arguments)]
pub fn emit_guardian_beam(
    guardian_eye: DVec3,
    target_centre: DVec3,
    camera_pos: DVec3,
    world_time: i64,
    progress: f32,
    guardian_ticks_existed: u32,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    // java: `vec32 = vec3.subtract(vec31)`; `lengthVector()` and `normalize()` both take the
    // float `MathHelper.sqrt_double`, and `normalize()` returns zero under 1e-4.
    let delta = target_centre - guardian_eye;
    let distance = f64::from(mc_math::sqrt_d(delta.length_squared()));
    let length = distance + 1.0;
    let direction = if distance < 1.0e-4 {
        DVec3::ZERO
    } else {
        delta / distance
    };
    // `Math.acos` of a float-normalised component a rounding past 1 is NaN in Java, and a NaN
    // rotation leaves nothing on screen: draw nothing rather than NaN vertices.
    if !length.is_finite() || direction.y.abs() > 1.0 {
        return;
    }
    let pitch = direction.y.acos() as f32;
    let yaw = direction.z.atan2(direction.x) as f32;
    // java: `rotate((PI / 2F + -f6) * (180F / PI), 0, 1, 0)` then `rotate(f5 * (180F / PI), 1, 0,
    // 0)`. The degrees are float; `glRotatef` turns them into an exact rotation.
    let to_degrees = 180.0_f32 / core::f32::consts::PI;
    let (sin_yaw, cos_yaw) = f64::from((core::f32::consts::FRAC_PI_2 + -yaw) * to_degrees)
        .to_radians()
        .sin_cos();
    let (sin_pitch, cos_pitch) = f64::from(pitch * to_degrees).to_radians().sin_cos();
    let start = guardian_eye - camera_pos;
    // `translate(eye) * rotate(Y) * rotate(X)`: the local point is turned about X first.
    let place = |local: DVec3| {
        let y = local.y * cos_pitch - local.z * sin_pitch;
        let z = local.y * sin_pitch + local.z * cos_pitch;
        start
            + DVec3::new(
                local.x * cos_yaw + z * sin_yaw,
                y,
                z * cos_yaw - local.x * sin_yaw,
            )
    };

    // java: `f2 = (float)getTotalWorldTime() + partialTicks`, `f3 = f2 * 0.5F % 1.0F`, and the
    // double phase `d1 = (double)f2 * 0.05D * (1.0D - (double)(i & 1) * 2.5D)` with `i = 1`. In a
    // long-running world the float time has no fraction left; the double phase still matters.
    let time = world_time as f32 + partial_ticks;
    let scroll = time * 0.5 % 1.0;
    let phase = f64::from(time) * 0.05 * (1.0 - 2.5);
    // java: `j = 64 + (int)(f7 * 240.0F)` and so on, with `f7 = f * f`. `WorldRenderer.color`
    // stores each channel with a `(byte)` cast, so from progress 0.894 red passes 255 and wraps:
    // the beam turns green for the last tenth of the wind-up.
    let f7 = progress * progress;
    let red = 64 + (f7 * 240.0) as i32;
    let green = 32 + (f7 * 192.0) as i32;
    let blue = 128 - (f7 * 64.0) as i32;
    let byte = |channel: i32| f32::from(channel as u8) / 255.0;
    let color = [byte(red), byte(green), byte(blue), 1.0];

    let radius = 0.2_f64;
    let corner = radius * 1.41;
    let at = |angle: f64, r: f64| (angle.cos() * r, angle.sin() * r);
    let (d4, d5) = at(phase + 2.356_194_490_192_345, corner);
    let (d6, d7) = at(phase + core::f64::consts::FRAC_PI_4, corner);
    let (d8, d9) = at(phase + 3.926_990_816_987_241_4, corner);
    let (d10, d11) = at(phase + 5.497_787_143_782_138, corner);
    let (d12, d13) = at(phase + core::f64::consts::PI, radius);
    let (d14, d15) = at(phase, radius);
    let (d16, d17) = at(phase + core::f64::consts::FRAC_PI_2, radius);
    let (d18, d19) = at(phase + core::f64::consts::PI * 3.0 / 2.0, radius);
    // java: `d22 = (double)(-1.0F + f3)`, `d23 = d0 * (0.5D / d2) + d22`.
    let bottom = f64::from(-1.0_f32 + scroll);
    let top = (length * (0.5 / radius) + bottom) as f32;
    let bottom = bottom as f32;
    // java: `d24 = ticksExisted % 2 == 0 ? 0.5D : 0.0D`.
    let cap = if guardian_ticks_existed.is_multiple_of(2) {
        0.5
    } else {
        0.0
    };
    let quads: [[(DVec3, [f32; 2]); 4]; 3] = [
        [
            (DVec3::new(d12, length, d13), [0.4999, top]),
            (DVec3::new(d12, 0.0, d13), [0.4999, bottom]),
            (DVec3::new(d14, 0.0, d15), [0.0, bottom]),
            (DVec3::new(d14, length, d15), [0.0, top]),
        ],
        [
            (DVec3::new(d16, length, d17), [0.4999, top]),
            (DVec3::new(d16, 0.0, d17), [0.4999, bottom]),
            (DVec3::new(d18, 0.0, d19), [0.0, bottom]),
            (DVec3::new(d18, length, d19), [0.0, top]),
        ],
        [
            (DVec3::new(d4, length, d5), [0.5, cap + 0.5]),
            (DVec3::new(d6, length, d7), [1.0, cap + 0.5]),
            (DVec3::new(d10, length, d11), [1.0, cap]),
            (DVec3::new(d8, length, d9), [0.5, cap]),
        ],
    ];
    for quad in quads {
        let corners = quad.map(|(local, uv)| EntityVertex {
            pos: place(local).as_vec3().into(),
            uv,
            normal: [0.0, 1.0, 0.0],
            color,
        });
        for i in [0, 1, 2, 0, 2, 3] {
            out.push(corners[i]);
        }
    }
}

/// The fishing bobber's billboard. java: `RenderFish.doRender` (`:27-50`).
///
/// The line back to the angler (`:54-98`) is **not** drawn here: it needs the holder's
/// interpolated position, eye height, swing progress and sneak state, none of which this crate
/// has an entity for. See the section report.
pub fn emit_fishing_bobber(rel: Vec3, view_yaw: f32, view_pitch: f32, out: &mut Vec<EntityVertex>) {
    let m = Mat4::from_translation(rel)
        * Mat4::from_scale(Vec3::splat(0.5))
        * Mat4::from_rotation_y((180.0 - view_yaw) * DEG_TO_RAD)
        * Mat4::from_rotation_x(-view_pitch * DEG_TO_RAD);
    emit_quad(
        &m,
        [
            Vec3::new(-0.5, -0.5, 0.0),
            Vec3::new(0.5, -0.5, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
            Vec3::new(-0.5, 0.5, 0.0),
        ],
        // java: `tex(0.0625, 0.1875)`, `(0.125, 0.1875)`, `(0.125, 0.125)`, `(0.0625, 0.125)`.
        [
            [0.0625, 0.1875],
            [0.125, 0.1875],
            [0.125, 0.125],
            [0.0625, 0.125],
        ],
        Vec3::Y,
        WHITE,
        out,
    );
}

// ---------------------------------------------------------------------------------------------
// RenderTNTPrimed
// ---------------------------------------------------------------------------------------------

/// java: `EntityTNTPrimed`'s constructor — `this.fuse = 80`. The server never sends the fuse, so
/// the client counts it down itself from the spawn tick.
pub const TNT_FUSE_TICKS: i32 = 80;

/// The vertex colour of primed TNT's (and the TNT minecart's) white overlay pass.
///
/// java: `BlockModelRenderer.renderModelBrightnessColorQuads(1.0F, ...)` writes
/// `putColorRGB_F4(1, 1, 1)`, alpha 255, over every vertex, which overrides the
/// `color(1, 1, 1, f2)` the renderer set just before (see [`tnt_flash_alpha`]). The entity
/// phase's `TntFlash` pipeline multiplies it by the entity's lightmap texel and adds it to the
/// body (`GL_SRC_ALPHA, GL_DST_ALPHA`). X6.
pub const TNT_FLASH_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

/// The swell as the fuse runs out.
///
/// java: `RenderTNTPrimed.java:29-37` — inside the last ten ticks,
/// `f = clamp(1 - (fuse - partial + 1) / 10, 0, 1)`, squared **twice**, and the scale is
/// `1 + f * 0.3`. Outside that window it is exactly `1.0`.
#[must_use]
pub fn tnt_scale(fuse: i32, partial_ticks: f32) -> f32 {
    let remaining = fuse as f32 - partial_ticks + 1.0;
    if remaining >= 10.0 {
        return 1.0;
    }
    let f = (1.0 - remaining / 10.0).clamp(0.0, 1.0);
    let f = f * f;
    let f = f * f;
    1.0 + f * 0.3
}

/// Whether the white overlay pass is drawn this tick.
///
/// java: `RenderTNTPrimed.java:45` — `entity.fuse / 5 % 2 == 0`, so TNT strobes with a five-tick
/// duty cycle for its whole life, not only at the end.
#[must_use]
pub fn tnt_flashing(fuse: i32) -> bool {
    (fuse / 5) % 2 == 0
}

/// The `GlStateManager.color` alpha `RenderTNTPrimed` sets before its white overlay. java:
/// `RenderTNTPrimed.java:39` — `(1 - (fuse - partial + 1) / 100) * 0.8`.
///
/// **It never reaches the screen in 1.8.9.** The overlay is drawn by `renderBlockBrightness`,
/// i.e. `DefaultVertexFormats.ITEM` quads followed by `putColorRGB_F4(1, 1, 1)`, and that
/// per-vertex colour (alpha 255) replaces the current colour, so the flash is always at full
/// strength: the vertex colour is [`TNT_FLASH_COLOR`]. Measured on fixed-function GL by
/// `crates/mc-client/javaref/TntFlashRef.java` (alpha 0, 0.4 and 0.8 give identical pixels).
/// Kept as documentation of the dead Java value; nothing in the draw path reads it.
#[must_use]
pub fn tnt_flash_alpha(fuse: i32, partial_ticks: f32) -> f32 {
    (1.0 - (fuse as f32 - partial_ticks + 1.0) / 100.0) * 0.8
}

/// The matrix primed TNT's block model is drawn under.
///
/// java: `RenderTNTPrimed.doRender` — `translate(x, y + 0.5, z)`, the swell scale,
/// `translate(-0.5, -0.5, 0.5)`, then `renderBlockBrightness`, whose *first* act is
/// `GlStateManager.rotate(90, 0, 1, 0)` (`BlockModelRenderer.java:314`) before drawing the model
/// in raw `0..1` cube space.
///
/// That `translate(-0.5, -0.5, 0.5)` looks wrong on its own — it shifts `+0.5` in Z — and it is
/// not: the `rotate(90, Y)` that follows maps the cube's `z in 0..1` onto `x in 0..1` and its
/// `x in 0..1` onto `z in -1..0`, and the odd-looking translate is what re-centres *that*. Since
/// [`super::item::emit_block_model_brightness`] owns the following quarter turn and draws raw
/// `0..1` block coordinates, this helper deliberately stops at the dispatcher boundary.
#[must_use]
pub fn tnt_matrix(rel: Vec3, fuse: i32, partial_ticks: f32) -> Mat4 {
    Mat4::from_translation(rel + Vec3::new(0.0, 0.5, 0.0))
        * Mat4::from_scale(Vec3::splat(tnt_scale(fuse, partial_ticks)))
        * Mat4::from_translation(Vec3::new(-0.5, -0.5, 0.5))
}

// ---------------------------------------------------------------------------------------------
// RenderFallingBlock
// ---------------------------------------------------------------------------------------------

/// The matrix a falling sand/gravel/anvil block's model is drawn under.
///
/// java: `RenderFallingBlock.doRender` (`:43-56`) — `translate(x, y, z)` and then
/// `worldrenderer.setTranslation(-i - 0.5, -j, -k - 0.5)` against a model emitted at the *block*
/// position `(i, j, k) = floor(entity)`, which nets out to a `(-0.5, 0, -0.5)` shift of the
/// `0..1` model. Composed with [`super::item::emit_baked_model`]'s own `(-0.5, -0.5, -0.5)` this
/// is a plain `+0.5` lift: a falling block's origin is its *bottom* face, unlike TNT's centre.
///
/// Vanilla additionally skips the draw when the block's own state is already in the world at
/// that position (`:39`), which stops the first tick of a falling block double-drawing over the
/// block it came from. That test needs a world read and is the caller's.
#[must_use]
pub fn falling_block_matrix(rel: Vec3) -> Mat4 {
    Mat4::from_translation(rel + Vec3::new(0.0, 0.5, 0.0))
}

// ---------------------------------------------------------------------------------------------
// RenderPainting
// ---------------------------------------------------------------------------------------------

/// `textures/painting/paintings_kristoffer_zetterstrand.png` — every painting on one 256x256
/// sheet. java: `RenderPainting.KRISTOFFER_PAINTING_TEXTURE`
pub const PAINTING_TEXTURE: &str = "textures/painting/paintings_kristoffer_zetterstrand.png";

/// One entry of vanilla's painting table.
///
/// java: `EntityPainting.EnumArt` — `(title, sizeX, sizeY, offsetX, offsetY)`, all in texels of
/// the 256x256 sheet. The wire carries `title` as a string (`S10PacketSpawnPainting`), so the
/// lookup is by name and an unknown name retains the no-valid-candidate state from
/// `EntityPainting`'s constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Art {
    /// java: `EnumArt.title` — the exact string the spawn packet carries.
    pub title: &'static str,
    /// java: `EnumArt.sizeX`, in texels (a multiple of 16).
    pub size_x: i32,
    /// java: `EnumArt.sizeY`, in texels.
    pub size_y: i32,
    /// java: `EnumArt.offsetX` — U of the art's top-left on the sheet.
    pub offset_x: i32,
    /// java: `EnumArt.offsetY` — V of the art's top-left on the sheet.
    pub offset_y: i32,
}

/// Every `EnumArt` constant, in declaration order.
///
/// java: `EntityPainting.java:144-169`. Order matters: `EntityPainting`'s no-title constructor
/// leaves the last `EnumArt` candidate assigned when no candidate fits the wall. The titled
/// constructor calls that constructor first, then only replaces the art on an exact
/// case-sensitive title match. The client-facing title-only helper uses that deterministic
/// no-valid-candidate result; a world-aware caller that can build the valid-candidate list must
/// apply Java's `rand.nextInt(list.size())` before this title pass.
pub const ART: [Art; 26] = [
    art("Kebab", 16, 16, 0, 0),
    art("Aztec", 16, 16, 16, 0),
    art("Alban", 16, 16, 32, 0),
    art("Aztec2", 16, 16, 48, 0),
    art("Bomb", 16, 16, 64, 0),
    art("Plant", 16, 16, 80, 0),
    art("Wasteland", 16, 16, 96, 0),
    art("Pool", 32, 16, 0, 32),
    art("Courbet", 32, 16, 32, 32),
    art("Sea", 32, 16, 64, 32),
    art("Sunset", 32, 16, 96, 32),
    art("Creebet", 32, 16, 128, 32),
    art("Wanderer", 16, 32, 0, 64),
    art("Graham", 16, 32, 16, 64),
    art("Match", 32, 32, 0, 128),
    art("Bust", 32, 32, 32, 128),
    art("Stage", 32, 32, 64, 128),
    art("Void", 32, 32, 96, 128),
    art("SkullAndRoses", 32, 32, 128, 128),
    art("Wither", 32, 32, 160, 128),
    art("Fighters", 64, 32, 0, 96),
    art("Pointer", 64, 64, 0, 192),
    art("Pigscene", 64, 64, 64, 192),
    art("BurningSkull", 64, 64, 128, 192),
    art("Skeleton", 64, 48, 192, 64),
    art("DonkeyKong", 64, 48, 192, 112),
];

const fn art(title: &'static str, size_x: i32, size_y: i32, offset_x: i32, offset_y: i32) -> Art {
    Art {
        title,
        size_x,
        size_y,
        offset_x,
        offset_y,
    }
}

impl Art {
    /// The art with this title, or the no-valid-candidate constructor state.
    ///
    /// java: `EntityPainting(World, BlockPos, EnumFacing, String)` first invokes the no-title
    /// constructor. That constructor assigns every candidate while checking the wall, so when
    /// no candidate is valid the final loop value is `EnumArt.DONKEY_KONG`; the title constructor
    /// then leaves that value unchanged if `String.equals` finds no exact (case-sensitive) match.
    /// This helper has no world/candidate list, so it exposes that source-faithful deterministic
    /// fallback rather than pretending that invalid network text means `EnumArt.KEBAB`.
    #[must_use]
    pub fn by_title(title: &str) -> &'static Art {
        ART.iter()
            .find(|a| a.title == title)
            .unwrap_or(&ART[ART.len() - 1])
    }

    /// Resolve a network painting using `EntityPainting`'s titled-constructor order.
    ///
    /// java: `EntityPainting(World, BlockPos, EnumFacing, String)` first runs the no-title
    /// constructor (`EntityPainting.java:24-46`), which tests every art in declaration order,
    /// selects one valid candidate with `rand.nextInt(list.size())`, and leaves the final enum
    /// value (`DONKEY_KONG`) in place when no candidate is valid. The titled constructor then
    /// scans the same table and replaces that candidate only for an exact `String.equals`
    /// match (`:48-62`). `hanging_entities` is the caller's snapshot of existing
    /// `EntityHanging` boxes; the painting being constructed is not in the world yet.
    ///
    /// The random stream is deliberately injected. A client boundary that does not expose the
    /// JVM-created `Entity.rand` can still preserve candidate ordering, bounded selection, and
    /// random-state consumption without hiding a second deterministic title fallback here.
    #[must_use]
    pub fn resolve_for_spawn(
        world: &World,
        anchor: BlockPos,
        facing: Facing,
        title: &str,
        random: &mut JavaRandom,
        hanging_entities: &[Aabb],
    ) -> &'static Art {
        let mut valid = Vec::new();
        for art in &ART {
            if painting_surface_valid(world, anchor, facing, art, hanging_entities) {
                valid.push(art);
            }
        }

        // `this.art` is assigned on every loop iteration in the Java constructor. If the list
        // is empty, the last iteration therefore leaves DONKEY_KONG, not KEBAB, in the field.
        let mut resolved = ART.last().expect("vanilla painting table is non-empty");
        if !valid.is_empty() {
            let index = usize::try_from(random.next_i32_bound(valid.len() as i32))
                .expect("JavaRandom.nextInt bound must be non-negative");
            resolved = valid[index];
        }

        // The title pass is intentionally after random candidate selection. In particular, a
        // known title still consumes the constructor's random draw whenever any art fits.
        ART.iter()
            .find(|art| art.title == title)
            .unwrap_or(resolved)
    }
}

/// Whether one `EntityPainting` candidate survives `EntityHanging.onValidSurface`.
///
/// This is kept beside the art table because the candidate's dimensions are part of the check.
/// The world collision scan is the same shared physics oracle used by the client interaction
/// path and preserves Java's loaded-column guard. Support blocks deliberately use `Material`'s
/// `isSolid` predicate, with the four repeater ids admitted by `EntityHanging.java:141-143`.
fn painting_surface_valid(
    world: &World,
    anchor: BlockPos,
    facing: Facing,
    art: &Art,
    hanging_entities: &[Aabb],
) -> bool {
    let position = hanging_position(
        [anchor.x, anchor.y, anchor.z],
        facing,
        art.size_x,
        art.size_y,
    );
    let bounds = painting_bounding_box(position, facing, art.size_x, art.size_y);

    let mut collisions = Vec::new();
    mc_physics::collision::collision_boxes_in(world, &bounds, &mut collisions);
    if !collisions.is_empty() {
        return false;
    }

    let support = anchor.relative(facing.opposite());
    let across = facing.rotate_y_ccw();
    let (across_x, _, across_z) = across.offset();
    for horizontal in 0..(art.size_x / 16).max(1) {
        for vertical in 0..(art.size_y / 16).max(1) {
            let position = BlockPos::new(
                support.x + across_x * horizontal,
                support.y + vertical,
                support.z + across_z * horizontal,
            );
            let state = world.block_at(position);
            if !state.props().material.is_solid() && !matches!(state.block().0, 93 | 94 | 149 | 150)
            {
                return false;
            }
        }
    }

    // `World.getEntitiesWithinAABBExcludingEntity` uses the prospective box itself when testing
    // resident entity AABBs (the extra two blocks only widen the chunk lookup). Any existing
    // EntityHanging makes the surface invalid regardless of whether that entity has a collision
    // box of its own.
    !hanging_entities
        .iter()
        .any(|existing| existing.intersects_with(&bounds))
}

/// `EntityHanging.updateBoundingBox`'s AABB for one art at an already-resolved position.
fn painting_bounding_box(
    position: [f64; 3],
    facing: Facing,
    width_pixels: i32,
    height_pixels: i32,
) -> Aabb {
    let mut width = f64::from(width_pixels);
    let height = f64::from(height_pixels);
    let mut depth = f64::from(width_pixels);
    if matches!(facing, Facing::North | Facing::South) {
        depth = 1.0;
    } else {
        width = 1.0;
    }
    let half = DVec3::new(width / 32.0, height / 32.0, depth / 32.0);
    Aabb::from_bounds(
        position[0] - half.x,
        position[1] - half.y,
        position[2] - half.z,
        position[0] + half.x,
        position[1] + half.y,
        position[2] + half.z,
    )
}

/// A painting's quads. java: `RenderPainting.doRender` + `renderPainting`.
///
/// `yaw` is `EntityHanging.rotationYaw`, i.e. `facing.horizontal_index() * 90`. Each 16x16 cell
/// of the art is a whole cube — front, back and four edge strips — so a 4x3 painting emits 12
/// cubes and 72 quads. The edge strips all sample the same small patch of the sheet
/// (`0.75..0.8125` U), which is the wooden frame border baked into the top-left of the texture.
///
/// The unlit compatibility entry point.  For the per-cell lightmap data that vanilla feeds to
/// `OpenGlHelper.setLightmapTextureCoords` before each group of six quads, use
/// [`emit_painting_with_lightmap`].  The ordinary entity phase does not consume that data yet,
/// so keeping this entry point also keeps the existing unlit submission path allocation-free.
pub fn emit_painting(rel: Vec3, yaw: f32, art: &Art, out: &mut Vec<EntityVertex>) {
    // java: `rotate(180 - entityYaw, 0, 1, 0)` then `scale(0.0625)` — the painting is built in
    // texel space like a model, but with no `(-1, -1, 1)` flip, so +Y is up.
    let m = Mat4::from_translation(rel)
        * Mat4::from_rotation_y((180.0 - yaw) * DEG_TO_RAD)
        * Mat4::from_scale(Vec3::splat(0.0625));

    let width = art.size_x;
    let height = art.size_y;
    let f = -(width as f32) / 2.0;
    let f1 = -(height as f32) / 2.0;
    // java: the eight literals `f2..f14`, kept under their Java names so the quad list below can
    // be diffed against `renderPainting` a line at a time.
    let f2 = 0.5f32;
    let f3 = 0.75f32;
    let f4 = 0.8125f32;
    let f5 = 0.0f32;
    let f6 = 0.0625f32;
    let f7 = 0.75f32;
    let f8 = 0.8125f32;
    let f9 = 0.001_953_125f32;
    let f10 = 0.001_953_125f32;
    let f11 = 0.751_953_1f32;
    let f12 = 0.751_953_1f32;
    let f13 = 0.0f32;
    let f14 = 0.0625f32;

    for i in 0..width / 16 {
        for j in 0..height / 16 {
            let f15 = f + ((i + 1) * 16) as f32;
            let f16 = f + (i * 16) as f32;
            let f17 = f1 + ((j + 1) * 16) as f32;
            let f18 = f1 + (j * 16) as f32;
            let f19 = (art.offset_x + width - i * 16) as f32 / 256.0;
            let f20 = (art.offset_x + width - (i + 1) * 16) as f32 / 256.0;
            let f21 = (art.offset_y + height - j * 16) as f32 / 256.0;
            let f22 = (art.offset_y + height - (j + 1) * 16) as f32 / 256.0;

            // Front (-Z), the art itself.
            emit_quad(
                &m,
                [
                    Vec3::new(f15, f18, -f2),
                    Vec3::new(f16, f18, -f2),
                    Vec3::new(f16, f17, -f2),
                    Vec3::new(f15, f17, -f2),
                ],
                [[f20, f21], [f19, f21], [f19, f22], [f20, f22]],
                -Vec3::Z,
                WHITE,
                out,
            );
            // Back (+Z), the plank texture.
            emit_quad(
                &m,
                [
                    Vec3::new(f15, f17, f2),
                    Vec3::new(f16, f17, f2),
                    Vec3::new(f16, f18, f2),
                    Vec3::new(f15, f18, f2),
                ],
                [[f3, f5], [f4, f5], [f4, f6], [f3, f6]],
                Vec3::Z,
                WHITE,
                out,
            );
            // Top (+Y).
            emit_quad(
                &m,
                [
                    Vec3::new(f15, f17, -f2),
                    Vec3::new(f16, f17, -f2),
                    Vec3::new(f16, f17, f2),
                    Vec3::new(f15, f17, f2),
                ],
                [[f7, f9], [f8, f9], [f8, f10], [f7, f10]],
                Vec3::Y,
                WHITE,
                out,
            );
            // Bottom (-Y).
            emit_quad(
                &m,
                [
                    Vec3::new(f15, f18, f2),
                    Vec3::new(f16, f18, f2),
                    Vec3::new(f16, f18, -f2),
                    Vec3::new(f15, f18, -f2),
                ],
                [[f7, f9], [f8, f9], [f8, f10], [f7, f10]],
                -Vec3::Y,
                WHITE,
                out,
            );
            // Left (-X).
            emit_quad(
                &m,
                [
                    Vec3::new(f15, f17, f2),
                    Vec3::new(f15, f18, f2),
                    Vec3::new(f15, f18, -f2),
                    Vec3::new(f15, f17, -f2),
                ],
                [[f12, f13], [f12, f14], [f11, f14], [f11, f13]],
                -Vec3::X,
                WHITE,
                out,
            );
            // Right (+X).
            emit_quad(
                &m,
                [
                    Vec3::new(f16, f17, -f2),
                    Vec3::new(f16, f18, -f2),
                    Vec3::new(f16, f18, f2),
                    Vec3::new(f16, f17, f2),
                ],
                [[f12, f13], [f12, f14], [f11, f14], [f11, f13]],
                Vec3::X,
                WHITE,
                out,
            );
        }
    }
}

/// One cell's lightmap coordinate as passed to
/// `OpenGlHelper.setLightmapTextureCoords(OpenGlHelper.lightmapTexUnit, block, sky)`.
///
/// Java represents this as the low and high 16-bit halves of
/// `World.getCombinedLight(pos, 0)`.  They deliberately remain raw lightmap texture coordinates
/// (`0, 16, ..., 240`) rather than being normalised brightness: an eventual entity-lightmap
/// shader needs the two axes separately, just as the fixed-function texture matrix did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PaintingLightmap {
    /// Low 16 bits of `getCombinedLight`: block-light texture coordinate.
    pub block: u16,
    /// High 16 bits of `getCombinedLight`: sky-light texture coordinate.
    pub sky: u16,
}

impl PaintingLightmap {
    /// Splits the packed value returned by `World.getCombinedLight` exactly as
    /// `RenderPainting.setLightmap` does with `% 65536` and `/ 65536`.
    #[must_use]
    pub const fn from_combined_light(combined_light: u32) -> Self {
        Self {
            block: combined_light as u16,
            sky: (combined_light >> 16) as u16,
        }
    }
}

/// An emitted painting vertex together with the lightmap state active when vanilla emitted it.
///
/// `RenderPainting.renderPainting` changes lightmap state once per 16x16 cell, then writes that
/// cell's six quads.  Consequently every consecutive group of 36 vertices belonging to a cell
/// has one [`PaintingLightmap`], while adjacent cells can differ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PaintingVertex {
    /// The painting geometry vertex.
    pub vertex: EntityVertex,
    /// The cell-local world-light sample.
    pub lightmap: PaintingLightmap,
}

/// The block position sampled by `RenderPainting.setLightmap` for one painting cell.
///
/// `cell_x_texels` and `cell_y_texels` are the centre of the cell in the local texel coordinate
/// system used by `renderPainting` (the averages of `f15/f16` and `f17/f18`).  The entity
/// position is its *world* `posX/Y/Z`, not the camera-relative render translation.  Using the
/// latter makes a painting's lighting visibly swim when the player crosses a block boundary.
///
/// java: `RenderPainting.setLightmap(EntityPainting, float, float)`.
#[must_use]
pub fn painting_cell_light_position(
    entity_world: [f64; 3],
    facing: Facing,
    cell_x_texels: f32,
    cell_y_texels: f32,
) -> [i32; 3] {
    let mut x = entity_world[0].floor() as i32;
    let y = (entity_world[1] + f64::from(cell_y_texels) / 16.0).floor() as i32;
    let mut z = entity_world[2].floor() as i32;
    let cell_x = f64::from(cell_x_texels) / 16.0;

    // Keep the four independent Java `if`s in this order conceptually: only the coordinate
    // along the painting changes, and vertical facings (which EntityPainting never permits)
    // retain the base position rather than inventing a projection.
    match facing {
        Facing::North => x = (entity_world[0] + cell_x).floor() as i32,
        Facing::West => z = (entity_world[2] - cell_x).floor() as i32,
        Facing::South => x = (entity_world[0] - cell_x).floor() as i32,
        Facing::East => z = (entity_world[2] + cell_x).floor() as i32,
        Facing::Up | Facing::Down => {}
    }

    [x, y, z]
}

/// Emits a painting with the exact per-cell `RenderPainting.setLightmap` state attached.
///
/// `combined_light` is called once per 16x16 art cell at the Java sample position, in the same
/// column-major `(i, j)` order as `renderPainting`; it supplies
/// `World.getCombinedLight(BlockPos, 0)`.  The callback boundary makes this pure renderer code:
/// world ownership and the later GPU lightmap handoff can stay with their respective callers.
///
/// The geometry and vertex ordering are byte-for-byte the same as [`emit_painting`].
pub fn emit_painting_with_lightmap<F>(
    rel: Vec3,
    yaw: f32,
    entity_world: [f64; 3],
    facing: Facing,
    art: &Art,
    mut combined_light: F,
    out: &mut Vec<PaintingVertex>,
) where
    F: FnMut([i32; 3]) -> u32,
{
    // `emit_painting` deliberately groups one cell's six quads contiguously.  Retaining that
    // single geometry transcription avoids a second, subtly divergent copy of the 144-vertex
    // painting cube purely to carry the GL state associated with it.
    const VERTICES_PER_CELL: usize = 6 * 6;
    let mut geometry = Vec::with_capacity(
        usize::try_from((art.size_x / 16) * (art.size_y / 16)).unwrap_or(0) * VERTICES_PER_CELL,
    );
    emit_painting(rel, yaw, art, &mut geometry);

    let f = -(art.size_x as f32) / 2.0;
    let f1 = -(art.size_y as f32) / 2.0;
    let mut cell = 0usize;
    for i in 0..art.size_x / 16 {
        for j in 0..art.size_y / 16 {
            // Java passes `(f15 + f16) / 2` and `(f17 + f18) / 2`; spelling the average in
            // expanded form preserves the half-cell location without relying on an art centre.
            let cell_x = (f + ((i + 1) * 16) as f32 + f + (i * 16) as f32) / 2.0;
            let cell_y = (f1 + ((j + 1) * 16) as f32 + f1 + (j * 16) as f32) / 2.0;
            let sample = painting_cell_light_position(entity_world, facing, cell_x, cell_y);
            let lightmap = PaintingLightmap::from_combined_light(combined_light(sample));
            let start = cell * VERTICES_PER_CELL;
            out.extend(
                geometry[start..start + VERTICES_PER_CELL]
                    .iter()
                    .copied()
                    .map(|vertex| PaintingVertex { vertex, lightmap }),
            );
            cell += 1;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// EntityHanging placement
// ---------------------------------------------------------------------------------------------

/// Where a hanging entity actually sits, given the block it is mounted on.
///
/// java: `EntityHanging.updateBoundingBox` (`:53-90`) — start at the block centre, back off
/// `0.46875` along the facing so the art sits *on* the wall rather than inside it, lift by
/// `func_174858_a(heightPixels)` and slide by `func_174858_a(widthPixels)` along
/// `facing.rotateYCCW()`. `func_174858_a` returns `0.5` when the pixel size is a multiple of 32
/// and `0` otherwise, which is what centres even-sized art (2x1, 4x3, ...) on the *edge* between
/// two blocks instead of on the anchor block.
///
/// The spawn packet gives only the anchor `BlockPos`, so without this a painting sits embedded
/// in the wall at the wrong height and, for even sizes, half a block off sideways.
#[must_use]
pub fn hanging_position(
    block: [i32; 3],
    facing: Facing,
    width_pixels: i32,
    height_pixels: i32,
) -> [f64; 3] {
    let offset = |pixels: i32| if pixels % 32 == 0 { 0.5 } else { 0.0 };
    let (fx, _, fz) = facing.offset();
    let (cx, _, cz) = facing.rotate_y_ccw().offset();
    let d4 = offset(width_pixels);
    let d5 = offset(height_pixels);
    [
        f64::from(block[0]) + 0.5 - f64::from(fx) * 0.468_75 + d4 * f64::from(cx),
        f64::from(block[1]) + 0.5 + d5,
        f64::from(block[2]) + 0.5 - f64::from(fz) * 0.468_75 + d4 * f64::from(cz),
    ]
}

/// A hanging entity's yaw. java: `EntityHanging.updateFacingWithBoundingBox` (`:46`) —
/// `rotationYaw = facing.getHorizontalIndex() * 90`.
///
/// A vertical facing is not legal for a hanging entity (vanilla `Validate.isTrue(...isHorizontal
/// ())`). The `unwrap_or(0)` is only a defensive fallback after that validation: Java's
/// `getHorizontalIndex()` would return `-1`, and its unguarded multiplication would produce
/// `-90.0`, not south's `0`.
#[must_use]
pub fn hanging_yaw(facing: Facing) -> f32 {
    f32::from(facing.horizontal_index().unwrap_or(0)) * 90.0
}

// ---------------------------------------------------------------------------------------------
// Object placement
// ---------------------------------------------------------------------------------------------

/// The boat's transform. java: `RenderBoat.doRender` (`:26-49`).
///
/// `translate(x, y + 0.25, z)`, `rotate(180 - yaw, Y)`, the damage rock about X, then
/// `scale(-1, -1, 1)`. There is **no** `-1.5078125` lift — that belongs to
/// `RendererLivingEntity` — and the model is then drawn with `render(..., -0.1F, ...)`, a
/// `limbSwing`-slot argument `ModelBoat` ignores.
///
/// The `scale(0.75)` immediately followed by `scale(1/0.75)` at `RenderBoat.java:45-46` is a
/// vanilla no-op and is deliberately not reproduced.
#[must_use]
pub fn boat_matrix(
    rel: Vec3,
    yaw: f32,
    time_since_hit: f32,
    damage_taken: f32,
    forward_direction: i32,
    partial_ticks: f32,
) -> Mat4 {
    let mut m = Mat4::from_translation(rel + Vec3::new(0.0, 0.25, 0.0))
        * Mat4::from_rotation_y((180.0 - yaw) * DEG_TO_RAD);
    let f = time_since_hit - partial_ticks;
    let f1 = (damage_taken - partial_ticks).max(0.0);
    if f > 0.0 {
        let angle = mc_math::sin(f) * f * f1 / 10.0 * forward_direction as f32;
        m *= Mat4::from_rotation_x(angle * DEG_TO_RAD);
    }
    m * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0))
}

/// The minecart's transform. java: `RenderMinecart.doRender` (`:30-90`).
///
/// `translate(x, y + 0.375, z)`, `rotate(180 - yaw, Y)`, `rotate(-pitch, Z)`, the damage roll,
/// then `scale(-1, -1, 1)`. `yaw`/`pitch` come from `func_70489_a`/`func_70495_a`'s rail sample
/// rather than the entity's own rotation when the cart is on a rail; the sample needs the world
/// and is the caller's.
#[must_use]
pub fn minecart_matrix(
    rel: Vec3,
    yaw: f32,
    pitch: f32,
    time_since_hit: f32,
    damage_taken: f32,
    rock_direction: i32,
    partial_ticks: f32,
) -> Mat4 {
    minecart_base_matrix(
        rel,
        yaw,
        pitch,
        time_since_hit,
        damage_taken,
        rock_direction,
        partial_ticks,
    ) * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0))
}

/// The transform active when `RenderMinecart` submits its optional display block.
///
/// Java draws the display before the final `scale(-1, -1, 1)` used by `ModelMinecart`: it scales
/// the block to `0.75`, then translates by `(-0.5, (offset - 8) / 16, 0.5)` before dispatching
/// either the baked block model or the chest item renderer. Keeping this separate from
/// [`minecart_matrix`] prevents cargo from inheriting the shell-only flip.
#[must_use]
// Keep the public wrapper's Java renderer argument order; the scaled helper below has the same
// intentional shape for the TNT minecart variant.
#[allow(clippy::too_many_arguments)]
pub fn minecart_display_matrix(
    rel: Vec3,
    yaw: f32,
    pitch: f32,
    time_since_hit: f32,
    damage_taken: f32,
    rock_direction: i32,
    partial_ticks: f32,
    display_offset: i32,
) -> Mat4 {
    minecart_display_matrix_scaled(
        rel,
        yaw,
        pitch,
        time_since_hit,
        damage_taken,
        rock_direction,
        partial_ticks,
        display_offset,
        1.0,
    )
}

/// The display transform with `RenderTntMinecart`'s optional pre-cargo swell inserted before
/// `RenderMinecart`'s fixed block scale and offset.
///
/// Keeping the extra scale ahead of the cargo translation matters: Java scales the matrix in
/// `renderCartContents` before delegating to `RenderMinecart.renderCartContents`, so the TNT
/// grows about the cart origin rather than each baked vertex's already-translated origin.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn minecart_display_matrix_scaled(
    rel: Vec3,
    yaw: f32,
    pitch: f32,
    time_since_hit: f32,
    damage_taken: f32,
    rock_direction: i32,
    partial_ticks: f32,
    display_offset: i32,
    cargo_scale: f32,
) -> Mat4 {
    minecart_base_matrix(
        rel,
        yaw,
        pitch,
        time_since_hit,
        damage_taken,
        rock_direction,
        partial_ticks,
    ) * Mat4::from_scale(Vec3::splat(cargo_scale))
        * Mat4::from_scale(Vec3::splat(0.75))
        * Mat4::from_translation(Vec3::new(-0.5, (display_offset - 8) as f32 / 16.0, 0.5))
}

fn minecart_base_matrix(
    rel: Vec3,
    yaw: f32,
    pitch: f32,
    time_since_hit: f32,
    damage_taken: f32,
    rock_direction: i32,
    partial_ticks: f32,
) -> Mat4 {
    let mut m = Mat4::from_translation(rel + Vec3::new(0.0, 0.375, 0.0))
        * Mat4::from_rotation_y((180.0 - yaw) * DEG_TO_RAD)
        * Mat4::from_rotation_z(-pitch * DEG_TO_RAD);
    let f = time_since_hit - partial_ticks;
    let f1 = (damage_taken - partial_ticks).max(0.0);
    if f > 0.0 {
        let angle = mc_math::sin(f) * f * f1 / 10.0 * rock_direction as f32;
        m *= Mat4::from_rotation_x(angle * DEG_TO_RAD);
    }
    m
}

/// The leash knot's transform. java: `RenderLeashKnot.doRender` (`:22-31`) — `translate(x, y, z)`
/// then `scale(-1, -1, 1)`, and notably **no yaw rotation at all**; a knot always faces the same
/// way regardless of which side of the fence the lead came from.
#[must_use]
pub fn leash_knot_matrix(rel: Vec3) -> Mat4 {
    Mat4::from_translation(rel) * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0))
}

/// The ender crystal's transform. java: `RenderEnderCrystal.doRender` (`:29-30`) — a bare
/// `translate(x, y, z)`. No yaw, no flip, no lift: `ModelEnderCrystal.render` does all of its own
/// orientation internally (see [`super::models::object::EnderCrystalModel`]).
#[must_use]
pub fn ender_crystal_matrix(rel: Vec3) -> Mat4 {
    Mat4::from_translation(rel)
}

/// The wither skull's transform. java: `RenderWitherSkull.doRender` (`:49-52`) —
/// `translate(x, y, z)` then `scale(-1, -1, 1)`; the yaw and pitch go to
/// `ModelSkeletonHead.setRotationAngles` as `netHeadYaw`/`headPitch`, not into this matrix.
#[must_use]
pub fn wither_skull_matrix(rel: Vec3) -> Mat4 {
    Mat4::from_translation(rel) * Mat4::from_scale(Vec3::new(-1.0, -1.0, 1.0))
}

/// `textures/entity/wither/wither_skull.png` — the ordinary blue-black skull.
/// java: `RenderWitherSkull.witherTextures`
pub const WITHER_SKULL_TEXTURE: &str = "textures/entity/wither/wither.png";

/// `textures/entity/wither/wither_invulnerable.png` — the charged (blue) skull.
/// java: `RenderWitherSkull.invulnerableWitherTextures`
pub const WITHER_SKULL_INVULNERABLE_TEXTURE: &str =
    "textures/entity/wither/wither_invulnerable.png";

#[cfg(test)]
mod tests {
    use super::super::item::emit_baked_model;
    use super::*;
    use mc_assets::model::{ItemCameraTransforms, ItemTransform};
    use mc_assets::{BakedQuad, Vertex as BakedVertex};

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // -- billboards ---------------------------------------------------------------------------

    /// java: `RenderXPOrb.java:49-50` vs `RenderSnowball.java:32-33` — the two billboard frames
    /// are inverses of each other about Y, not the same rotation written twice.
    #[test]
    fn the_two_billboard_frames_face_opposite_ways() {
        let a = camera_facing_matrix(Vec3::ZERO, 90.0, 0.0);
        let b = item_facing_matrix(Vec3::ZERO, 90.0, 0.0);
        let n = Vec3::Z;
        let fa = a.transform_vector3(n);
        let fb = b.transform_vector3(n);
        // 180 - 90 = 90 vs -90: the two normals are 180 degrees apart.
        assert!(fa.dot(fb) < -0.99, "{fa:?} {fb:?}");
    }

    /// The billboard's translation is the entity position and nothing else.
    #[test]
    fn a_billboard_is_anchored_at_the_entity() {
        let rel = Vec3::new(3.0, -2.0, 7.5);
        let m = camera_facing_matrix(rel, 33.0, -12.0);
        let o = m.transform_point3(Vec3::ZERO);
        assert!(
            approx(o.x, rel.x) && approx(o.y, rel.y) && approx(o.z, rel.z),
            "{o:?}"
        );
    }

    // -- XP orb -------------------------------------------------------------------------------

    /// java: `EntityXPOrb.getTextureByXP` (`:259`) — every threshold, on both sides.
    #[test]
    fn xp_orb_texture_ladder_matches_the_java_thresholds() {
        let expected = [
            (0, 0),
            (2, 0),
            (3, 1),
            (6, 1),
            (7, 2),
            (16, 2),
            (17, 3),
            (36, 3),
            (37, 4),
            (72, 4),
            (73, 5),
            (148, 5),
            (149, 6),
            (306, 6),
            (307, 7),
            (616, 7),
            (617, 8),
            (1236, 8),
            (1237, 9),
            (2476, 9),
            (2477, 10),
            (100_000, 10),
        ];
        for (xp, index) in expected {
            assert_eq!(xp_orb_texture_index(xp), index, "xp = {xp}");
        }
    }

    /// java: `RenderXPOrb.java:32-35` — four columns of 16 texels on a 64-pixel sheet.
    #[test]
    fn xp_orb_cells_tile_a_four_by_three_sheet() {
        assert_eq!(xp_orb_uv(0), [0.0, 0.25, 0.0, 0.25]);
        assert_eq!(xp_orb_uv(3), [0.75, 1.0, 0.0, 0.25]);
        assert_eq!(xp_orb_uv(4), [0.0, 0.25, 0.25, 0.5]);
        assert_eq!(xp_orb_uv(10), [0.5, 0.75, 0.5, 0.75]);
    }

    /// java: `RenderXPOrb.java:45-48` — green is pinned at full, alpha at `128`, and red/blue
    /// swing out of phase with each other.
    #[test]
    fn xp_orb_colour_cycles_green_against_a_phase_shifted_blue() {
        let c = xp_orb_color(0, 0.0);
        assert!(approx(c[1], 1.0), "{c:?}");
        assert!(approx(c[3], 128.0 / 255.0), "{c:?}");
        // sin(0) = 0 -> red = (0 + 1) * 0.5 * 255 = 127; blue = (sin(4.1888) + 1) * 0.1 * 255.
        assert_eq!((c[0] * 255.0).round() as i32, 127);
        let blue = ((mc_math::sin(4.188_790_3) + 1.0) * 0.1 * 255.0) as i32;
        assert_eq!((c[2] * 255.0).round() as i32, blue);
        // The two never coincide: red peaks where blue is near its floor.
        let mut min_blue = f32::MAX;
        let mut max_blue: f32 = 0.0;
        for t in 0..64 {
            let c = xp_orb_color(t, 0.0);
            min_blue = min_blue.min(c[2]);
            max_blue = max_blue.max(c[2]);
        }
        assert!(
            max_blue <= 0.2 + 1e-3,
            "blue is scaled by 0.1, not 0.5: {max_blue}"
        );
        assert!(min_blue < 0.01);
    }

    /// One quad, six vertices, `0.3` blocks across after the scale.
    #[test]
    fn an_xp_orb_is_one_quad_three_tenths_of_a_block_wide() {
        let mut out = Vec::new();
        emit_xp_orb(Vec3::ZERO, 0.0, 0.0, 1, 0, 0.0, &mut out);
        assert_eq!(out.len(), 6);
        let min_x = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[0]));
        let max_x = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[0]));
        assert!(approx(max_x - min_x, 0.3), "{}", max_x - min_x);
        // f5 = 0.5 and f6 = 0.25: the quad hangs a quarter of its height below the origin.
        let min_y = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[1]));
        assert!(approx(min_y, -0.25 * 0.3), "{min_y}");
    }

    /// `RenderFish.doRender` renders one half-block particle-sheet quad. This is deliberately
    /// separate from the angler line, which is a different untextured render pass.
    #[test]
    fn a_fishing_bobber_is_one_half_block_particle_quad() {
        let mut out = Vec::new();
        emit_fishing_bobber(Vec3::ZERO, 0.0, 0.0, &mut out);
        assert_eq!(out.len(), 6);
        let span = |axis: usize| {
            let min = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[axis]));
            let max = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[axis]));
            max - min
        };
        assert!(approx(span(0), 0.5), "{}", span(0));
        assert!(approx(span(1), 0.5), "{}", span(1));
        assert_eq!(out[0].uv, [0.0625, 0.1875]);
        assert_eq!(out[2].uv, [0.125, 0.125]);
    }

    #[test]
    fn fishing_line_has_exactly_seventeen_sagged_vertices() {
        let mut out = Vec::new();
        emit_fishing_line(
            DVec3::new(4.0, 65.0, -2.0),
            DVec3::ZERO,
            FishingLineAngler {
                prev_pos: DVec3::new(0.0, 64.0, 0.0),
                pos: DVec3::new(0.0, 64.0, 0.0),
                prev_yaw: 0.0,
                yaw: 0.0,
                prev_pitch: 0.0,
                pitch: 0.0,
                prev_body_yaw: 0.0,
                body_yaw: 0.0,
                eye_height: 1.62,
                sneaking: false,
                prev_swing_progress: 0.0,
                swing_progress: 0.0,
                is_local: true,
                third_person: false,
            },
            0.5,
            &mut out,
        );
        assert_eq!(out.len(), 17);
        assert_eq!(out.first().unwrap().color, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(out.first().unwrap().pos, [4.0, 65.25, -2.0]);
        assert!(
            out[8].pos[1] > out[0].pos[1],
            "the line keeps Java's quadratic rise"
        );
    }

    #[test]
    fn leash_emits_two_disjoint_twenty_four_segment_strips() {
        let mut out = Vec::new();
        emit_leash(
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(2.0, 1.0, 3.0),
            &mut out,
        );
        assert_eq!(out.len(), 2 * 25 * 2);
        assert_eq!(out[0].color, [0.35, 0.28, 0.21, 1.0]);
        assert_eq!(out[1].color, [0.35, 0.28, 0.21, 1.0]);
        assert!(approx(out[50].pos[1] - out[0].pos[1], 0.025));
    }

    fn guardian_beam(progress: f32, world_time: i64, partial: f32) -> Vec<EntityVertex> {
        let mut out = Vec::new();
        emit_guardian_beam(
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, 4.0),
            DVec3::ZERO,
            world_time,
            progress,
            2,
            partial,
            &mut out,
        );
        out
    }

    /// java: `RenderGuardian.doRender` draws three `GL_QUADS`. The entity phase draws a plain
    /// triangle list, so each quad must arrive as `0-1-2 / 0-2-3`: Java's twelve corners as they
    /// stand made four unrelated triangles (half a plane, two slivers and half the cap).
    #[test]
    fn guardian_beam_is_three_triangulated_quads() {
        let out = guardian_beam(0.5, 100, 0.0);
        assert_eq!(out.len(), 18);
        for quad in out.chunks_exact(6) {
            assert_eq!(quad[3].pos, quad[0].pos);
            assert_eq!(quad[4].pos, quad[2].pos);
            assert_eq!(quad[3].uv, quad[0].uv);
            assert_eq!(quad[4].uv, quad[2].uv);
        }
        // The two planes run the whole beam, the target distance plus one block.
        let along = |v: &EntityVertex| v.pos[2];
        let (near, far) = out[..12].iter().fold((f32::MAX, f32::MIN), |(a, b), v| {
            (a.min(along(v)), b.max(along(v)))
        });
        assert!(approx(near, 0.0) && approx(far, 5.0), "{near}..{far}");
        // The cap sits at the far end, textured from the right half; even ticks use its top row.
        assert!(out[12..].iter().all(|v| approx(v.pos[2], 5.0)));
        assert_eq!(out[12].uv, [0.5, 1.0]);
        assert_eq!(out[0].uv[0], 0.4999);
    }

    /// java: `j = 64 + (int)(f7 * 240.0F)`, `k = 32 + (int)(f7 * 192.0F)`,
    /// `l = 128 - (int)(f7 * 64.0F)` with `f7 = f * f`, each stored by `WorldRenderer.color`
    /// through a `(byte)` cast. Truncated, not rounded, and red wraps past 255.
    #[test]
    fn guardian_beam_colour_bytes_truncate_and_wrap_like_java() {
        let bytes = |progress: f32| {
            let colour = guardian_beam(progress, 100, 0.0)[0].color;
            colour.map(|c| {
                let byte = c * 255.0;
                assert!((byte - byte.round()).abs() < 1e-3, "{c} is not a byte");
                byte.round() as u8
            })
        };
        assert_eq!(bytes(0.0), [64, 32, 128, 255]);
        assert_eq!(bytes(0.5), [124, 80, 112, 255]);
        // f7 = 0.2916 (0.54 squared): 64 + (int)69.98 = 133, not a rounded 134.
        assert_eq!(bytes(0.54), [133, 87, 110, 255]);
        // 0.8936 squared * 240 = 191.6: the last progress before the red byte wraps.
        assert_eq!(bytes(0.8936), [255, 185, 77, 255]);
        assert_eq!(bytes(0.9), [2, 187, 77, 255]);
        assert_eq!(bytes(0.95), [24, 205, 71, 255]);
        assert_eq!(bytes(1.0), [48, 224, 64, 255]);
    }

    /// java: `d1 = (double)f2 * 0.05D * -1.5D` with the float `f2 = (float)worldTime + partial`.
    /// In a long-running world that product is millions of radians: computed in float it lands
    /// a large fraction of a radian away and turns the beam's cross section.
    #[test]
    fn guardian_beam_phase_is_double_even_in_a_long_running_world() {
        let world_time = 123_456_789_i64;
        let partial = 0.6_f32;
        let out = guardian_beam(0.5, world_time, partial);
        let time = world_time as f32 + partial;
        let java = f64::from(time) * 0.05 * -1.5;
        let float_phase = f64::from(time * 0.05 * -1.5);
        // The beam runs along +Z here: the pitch of 90 degrees takes the first plane's far
        // corner, local (d12, length, d13), to (d12, -d13, length).
        let corner = |phase: f64| {
            let d12 = (phase + core::f64::consts::PI).cos() * 0.2;
            let d13 = (phase + core::f64::consts::PI).sin() * 0.2;
            (d12, d13)
        };
        let (java_x, java_y) = corner(java);
        let (float_x, float_y) = corner(float_phase);
        let ours = out[0].pos;
        let matches = |x: f64, y: f64| {
            (f64::from(ours[0]).abs() - x.abs()).abs() < 1e-4
                && (f64::from(ours[1]).abs() - y.abs()).abs() < 1e-4
        };
        assert!(
            matches(java_x, java_y),
            "{ours:?} vs Java ({java_x}, {java_y})"
        );
        assert!(
            !matches(float_x, float_y),
            "the float phase must be distinguishable here"
        );
    }

    // -- arrow --------------------------------------------------------------------------------

    /// java: `RenderArrow.doRender` — two end caps plus four vanes.
    #[test]
    fn an_arrow_is_six_quads() {
        let mut out = Vec::new();
        emit_arrow(Vec3::ZERO, 0.0, 0.0, 0.0, &mut out);
        assert_eq!(out.len(), 6 * 6);
    }

    /// java: `scale(f8)` with `f8 = 0.05625`, and `translate(-4, 0, 0)` afterwards — so the
    /// arrow's 16-texel shaft is `0.9` blocks long and hangs asymmetrically about the entity.
    #[test]
    fn the_arrow_shaft_is_nine_tenths_of_a_block() {
        assert!(approx(ARROW_SCALE, 0.9 / 16.0));
        let mut out = Vec::new();
        // Yaw 90 turns the shaft (built along local X) onto the world -Z... whatever the axis,
        // the extent is what matters here, so measure the longest axis.
        emit_arrow(Vec3::ZERO, 0.0, 0.0, 0.0, &mut out);
        let span = |axis: usize| {
            let mn = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[axis]));
            let mx = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[axis]));
            mx - mn
        };
        let longest = span(0).max(span(1)).max(span(2));
        assert!(approx(longest, 16.0 * ARROW_SCALE), "{longest}");
    }

    /// java: `float f9 = arrowShake - partialTicks; if (f9 > 0) rotate(-sin(f9 * 3) * f9, Z)`.
    #[test]
    fn a_settled_arrow_does_not_wobble() {
        assert_eq!(
            arrow_matrix(Vec3::ZERO, 10.0, 20.0, 0.0),
            arrow_matrix(Vec3::ZERO, 10.0, 20.0, -1.0)
        );
        assert_ne!(
            arrow_matrix(Vec3::ZERO, 10.0, 20.0, 0.0),
            arrow_matrix(Vec3::ZERO, 10.0, 20.0, 3.0)
        );
    }

    /// The geometric (winding) normal of quad `q` of an emitted arrow, and its plane offset.
    fn arrow_quad_plane(out: &[EntityVertex], q: usize) -> (Vec3, f32) {
        let p = |i: usize| Vec3::from(out[q * 6 + i].pos);
        let n = (p(1) - p(0)).cross(p(2) - p(0)).normalize();
        (n, n.dot(p(0)))
    }

    /// Why a free arrow must be culled: every one of its six quads has a partner in the same
    /// plane wound the other way (the two caps at `x = -7`, vanes 1/3 and 2/4). With
    /// `GL_CULL_FACE` exactly one of each pair faces the camera; two-sided they z-fight.
    #[test]
    fn every_arrow_quad_has_a_coplanar_partner_wound_the_other_way() {
        let mut out = Vec::new();
        emit_arrow(Vec3::new(0.3, -1.2, 2.0), 37.0, -55.0, 0.0, &mut out);
        for q in 0..6 {
            let (n, d) = arrow_quad_plane(&out, q);
            let partners = (0..6)
                .filter(|&other| other != q)
                .filter(|&other| {
                    let (m, e) = arrow_quad_plane(&out, other);
                    (n + m).length() < 1e-4 && (d + e).abs() < 1e-4
                })
                .count();
            assert_eq!(
                partners, 1,
                "quad {q} must have exactly one back-to-back twin"
            );
            // The vanes' `glNormal3f(0, 0, f8)` is their winding's front. The caps' normals point
            // the other way (`(f8, 0, 0)` on the quad whose winding faces -X), a vanilla quirk
            // kept as is: culling follows the winding, the light follows the normal.
            let along = Vec3::from(out[q * 6].normal).dot(n);
            let expected = if q < 2 { -1.0 } else { 1.0 };
            assert!((along - expected).abs() < 1e-3, "quad {q}: {along}");
        }
    }

    /// java: `glNormal3f(f8, ...)` under `GL_RESCALE_NORMAL` keeps the normal `0.05625` long, so
    /// the two item lights barely reach a free arrow: 0.4 ambient plus at most `0.6 * 0.05625`
    /// of each.
    #[test]
    fn a_free_arrow_is_lit_almost_by_ambient_alone() {
        let light0 = Vec3::new(0.2, 1.0, -0.7).normalize();
        assert!(approx(
            arrow_light(Vec3::Y),
            0.4 + 0.6 * ARROW_SCALE * 2.0 * light0.y
        ));
        assert!(approx(arrow_light(Vec3::NEG_Y), 0.4));
        assert!(approx(
            arrow_light(Vec3::NEG_Z),
            0.4 + 0.6 * ARROW_SCALE * -light0.z
        ));
        for normal in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::new(0.6, 0.8, 0.0)] {
            let light = arrow_light(normal);
            assert!((0.4..0.455).contains(&light), "{normal:?}: {light}");
        }
    }

    /// The lit emitter changes nothing but the colour, which is [`arrow_light`] of each quad.
    #[test]
    fn emit_arrow_lit_folds_the_light_into_the_colour_only() {
        let (rel, yaw, pitch, shake) = (Vec3::new(0.25, -1.125, 1.25), 200.0, -55.0, 2.4);
        let mut plain = Vec::new();
        emit_arrow(rel, yaw, pitch, shake, &mut plain);
        let sentinel = EntityVertex {
            pos: [9.0; 3],
            uv: [0.5; 2],
            normal: [0.0, 1.0, 0.0],
            color: [0.5; 4],
        };
        let mut lit = vec![sentinel];
        emit_arrow_lit(rel, yaw, pitch, shake, &mut lit);
        assert_eq!(lit.len(), plain.len() + 1);
        assert_eq!(lit[0], sentinel, "earlier vertices are untouched");
        for (a, b) in plain.iter().zip(&lit[1..]) {
            assert_eq!((a.pos, a.uv, a.normal), (b.pos, b.uv, b.normal));
            let light = arrow_light(Vec3::from(a.normal));
            assert_eq!(b.color, [light, light, light, 1.0]);
        }
    }

    /// A free arrow keeps the world pass's culling; one stuck in a mob is drawn inside
    /// `RendererLivingEntity`'s `disableCull()` with the item lights off.
    #[test]
    fn free_arrows_cull_and_stuck_arrows_are_unlit_and_two_sided() {
        use super::super::layers::LayerKind;
        assert_eq!(ARROW_BLEND, LayerBlend::OpaqueUnlitCulled);
        assert_eq!(LayerKind::Arrow.blend(), LayerBlend::OpaqueUnlit);
    }

    // -- thrown items -------------------------------------------------------------------------

    /// A one-quad "model" spanning the full `0..1` cube in X and Z — enough to measure the
    /// scale the item pipeline applies without needing a resource pack.
    fn quad_at(corners: [[f32; 3]; 4]) -> BakedQuad {
        BakedQuad {
            verts: core::array::from_fn(|i| BakedVertex {
                pos: corners[i],
                uv: [0.0, 0.0],
            }),
            face: None,
            shade_face: None,
            tint_index: None,
            shade: true,
        }
    }

    fn unit_cube_model(transforms: ItemCameraTransforms) -> BakedModel {
        BakedModel {
            general_quads: vec![quad_at([
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ])],
            transforms,
            ..Default::default()
        }
    }

    /// java: `RenderSnowball.java:31` `scale(0.5)`, `RenderItem.preTransform`'s `scale(2)` for a
    /// flat model (`RenderItem.java:254-257`, reached through `renderItem(stack, GROUND)` ->
    /// `renderItemModelTransform`), and `RenderItem.java:145`'s `scale(0.5)`. Every thrown item is
    /// `builtin/generated` (flat), so a pearl or snowball sprite is half a block across, the size
    /// of the dropped item. This test used to assert a quarter block for the flat model, which was
    /// the missing-`preTransform` bug; only a GUI-3D (block) model lands on a quarter.
    #[test]
    fn a_flat_thrown_item_is_half_a_block_and_a_gui_3d_one_a_quarter() {
        let span = |model: &BakedModel| {
            let mut out = Vec::new();
            emit_thrown_item(Vec3::ZERO, 0.0, 0.0, model, &mut out);
            let mn = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[0]));
            let mx = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[0]));
            mx - mn
        };
        let flat = unit_cube_model(ItemCameraTransforms::DEFAULT);
        assert!(!flat.gui_3d);
        assert!(approx(span(&flat), 0.5), "{}", span(&flat));
        let block = BakedModel {
            gui_3d: true,
            ..unit_cube_model(ItemCameraTransforms::DEFAULT)
        };
        assert!(approx(span(&block), 0.25), "{}", span(&block));
    }

    /// The whole `RenderSnowball` chain in Java's own GL call order (`bkc.a` then `bjh.a(zx,
    /// bgr$b)` -> `bjh.b(zx)` -> `ItemCameraTransforms.applyTransform` -> `bjh.a(zx, boq)`), with a
    /// non-trivial ground transform and camera so a misplaced factor cannot hide.
    #[test]
    fn thrown_item_matrix_is_render_snowball_then_render_item_in_java_order() {
        let ground = ItemTransform {
            rotation: [10.0, 20.0, 30.0],
            translation: [0.125, -0.25, 0.0625],
            scale: [1.5, 0.75, 1.25],
        };
        let transforms = ItemCameraTransforms {
            ground,
            ..ItemCameraTransforms::DEFAULT
        };
        let (rel, yaw, pitch) = (Vec3::new(1.5, -2.0, 3.25), 37.0_f32, -21.0_f32);
        for gui_3d in [false, true] {
            let model = BakedModel {
                gui_3d,
                ..unit_cube_model(transforms)
            };
            let java = Mat4::from_translation(rel)
                * Mat4::from_scale(Vec3::splat(0.5))
                * Mat4::from_rotation_y(-yaw * DEG_TO_RAD)
                * Mat4::from_rotation_x(pitch * DEG_TO_RAD)
                * if gui_3d {
                    Mat4::IDENTITY
                } else {
                    Mat4::from_scale(Vec3::splat(2.0))
                }
                * Mat4::from_translation(Vec3::from(ground.translation))
                * Mat4::from_rotation_y(ground.rotation[1] * DEG_TO_RAD)
                * Mat4::from_rotation_x(ground.rotation[0] * DEG_TO_RAD)
                * Mat4::from_rotation_z(ground.rotation[2] * DEG_TO_RAD)
                * Mat4::from_scale(Vec3::from(ground.scale))
                * Mat4::from_scale(Vec3::splat(0.5));
            let ours = thrown_item_matrix(rel, yaw, pitch, &model);
            for point in [
                Vec3::ZERO,
                Vec3::new(1.0, 0.5, -0.25),
                Vec3::new(-0.5, 1.0, 1.0),
            ] {
                let (a, b) = (ours.transform_point3(point), java.transform_point3(point));
                assert!((a - b).length() < 1e-5, "gui_3d {gui_3d}: {a:?} != {b:?}");
            }
        }
    }

    #[test]
    fn held_item_transform_keeps_java_common_half_and_context_branch() {
        let display = ItemTransform {
            rotation: [0.0, 31.0, 0.0],
            translation: [0.25, -0.125, 0.0625],
            scale: [1.5, 0.75, 2.0],
        };
        let flat = item_renderer_model_transform(false, &display);
        let gui_3d = item_renderer_model_transform(true, &display);

        // Flat held items get RenderItem.preTransform's `2`; GUI-3D held items get
        // ItemRenderer's outer `2`. Both then pass through display and RenderItem's common `0.5`.
        // The two complete chains must therefore agree for every point, including translations.
        for point in [Vec3::ZERO, Vec3::new(0.25, -0.5, 0.75)] {
            let a = flat.transform_point3(point);
            let b = gui_3d.transform_point3(point);
            for (got, want) in a.to_array().into_iter().zip(b.to_array()) {
                assert!((got - want).abs() < 1e-6, "{a:?} != {b:?}");
            }
        }
        assert_eq!(flat, gui_3d);

        // A common-scale omission is caught independently: a caller that only applies display
        // would leave this translation at a different point than Java's full chain.
        let display_only = item_transform_matrix(&display);
        assert_ne!(
            flat.transform_point3(Vec3::ZERO),
            display_only.transform_point3(Vec3::ZERO)
        );
    }

    #[test]
    fn fixed_flat_items_retain_render_item_pre_transform_control() {
        let point = Vec3::new(0.25, 0.5, -0.25);
        let flat = item_model_pre_transform(false).transform_point3(point);
        let gui_3d = item_model_pre_transform(true).transform_point3(point);
        assert_eq!(flat, point * 2.0);
        assert_eq!(gui_3d, point);
    }

    /// java: `ItemCameraTransforms.applyTransform` runs *between* the two `scale(0.5)`s (after
    /// a flat model's `preTransform` `2`), so a model whose `display.ground` scales is scaled by
    /// all four: `0.5 * 2 * 2 * 0.5` = one block for this flat unit model.
    #[test]
    fn the_ground_display_transform_is_applied() {
        let transforms = ItemCameraTransforms {
            ground: ItemTransform {
                rotation: [0.0; 3],
                translation: [0.0; 3],
                scale: [2.0; 3],
            },
            ..ItemCameraTransforms::DEFAULT
        };
        let model = unit_cube_model(transforms);
        let mut out = Vec::new();
        emit_thrown_item(Vec3::ZERO, 0.0, 0.0, &model, &mut out);
        let mn = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[0]));
        let mx = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[0]));
        assert!(approx(mx - mn, 1.0), "{}", mx - mn);
    }

    #[test]
    fn a_thrown_item_tints_only_tagged_quads_and_keeps_the_item_transform() {
        // `bottle_splash.json` is a generated two-layer model: layer 0 is the liquid overlay
        // and layer 1 is the white glass/bottle art. This synthetic pair exercises the same
        // contract without needing a GPU or a resource-pack global.
        let mut model = unit_cube_model(ItemCameraTransforms::DEFAULT);
        let untagged = model.general_quads[0].clone();
        model.general_quads[0].tint_index = Some(0);
        model.general_quads.push(untagged);

        let mut tinted = Vec::new();
        let mut calls = Vec::new();
        emit_thrown_item_tinted(
            Vec3::new(2.0, 3.0, 4.0),
            17.0,
            -11.0,
            &model,
            &mut tinted,
            |i| {
                calls.push(i);
                0x12_3456
            },
        );

        assert_eq!(
            calls,
            [0],
            "RenderItem asks for tint only on tagged bottle layers"
        );
        let expected = [
            0x12 as f32 / 255.0,
            0x34 as f32 / 255.0,
            0x56 as f32 / 255.0,
            1.0,
        ];
        assert!(tinted[..6].iter().all(|v| v.color == expected));
        assert!(tinted[6..].iter().all(|v| v.color == EntityVertex::WHITE));

        // The tint-aware entry point must not fork RenderSnowball's transform. Its white
        // callback is therefore geometrically identical to the legacy entry point.
        let mut plain = Vec::new();
        emit_thrown_item(Vec3::new(2.0, 3.0, 4.0), 17.0, -11.0, &model, &mut plain);
        assert_eq!(plain.len(), tinted.len());
        assert!(plain.iter().zip(&tinted).all(|(a, b)| {
            a.pos == b.pos && a.uv == b.uv && a.normal == b.normal && a.color == EntityVertex::WHITE
        }));
    }

    /// java: `applyTransform` is `translate`, `rotate(y, Y)`, `rotate(x, X)`, `rotate(z, Z)`,
    /// `scale` — the rotation order is Y, X, Z and it is not the glam Euler default.
    #[test]
    fn the_display_transform_rotates_y_then_x_then_z() {
        let t = ItemTransform {
            rotation: [0.0, 90.0, 0.0],
            translation: [1.0, 0.0, 0.0],
            scale: [1.0; 3],
        };
        let m = item_transform_matrix(&t);
        // Translation happens first, in the *outer* frame, so the origin lands at +X.
        let o = m.transform_point3(Vec3::ZERO);
        assert!(
            approx(o.x, 1.0) && approx(o.y, 0.0) && approx(o.z, 0.0),
            "{o:?}"
        );
        // rotate(90, Y) maps +X onto -Z.
        let d = m.transform_vector3(Vec3::X);
        assert!(approx(d.z, -1.0), "{d:?}");
    }

    // -- TNT ----------------------------------------------------------------------------------

    /// java: `RenderTNTPrimed.java:29-37` — nothing happens until the last ten ticks.
    #[test]
    fn tnt_only_swells_in_the_last_ten_ticks() {
        assert!(approx(tnt_scale(80, 0.0), 1.0));
        assert!(
            approx(tnt_scale(10, 0.0), 1.0),
            "fuse 10 gives remaining 11"
        );
        assert!(tnt_scale(5, 0.0) > 1.0);
        // f is squared twice, so the swell is very back-loaded: at fuse 5 (remaining 6) the raw
        // f is 0.4 and f^4 = 0.0256, a 0.77% swell.
        assert!(
            approx(tnt_scale(5, 0.0), 1.0 + 0.4f32.powi(4) * 0.3),
            "{}",
            tnt_scale(5, 0.0)
        );
        // At fuse 0 (remaining 1) f = 0.9, f^4 = 0.6561, a 19.7% swell — the maximum.
        assert!(
            approx(tnt_scale(0, 0.0), 1.0 + 0.9f32.powi(4) * 0.3),
            "{}",
            tnt_scale(0, 0.0)
        );
    }

    /// java: `entity.fuse / 5 % 2 == 0` — five ticks lit, five ticks dark, from the moment it is
    /// primed.
    #[test]
    fn tnt_strobes_on_a_five_tick_cycle() {
        let lit: Vec<bool> = (0..20).map(tnt_flashing).collect();
        assert_eq!(
            lit,
            [
                true, true, true, true, true, false, false, false, false, false, true, true, true,
                true, true, false, false, false, false, false
            ]
        );
        assert!(tnt_flashing(80));
    }

    /// Java integer division truncates toward zero.  This bites the old `div_euclid` mutant
    /// even though live primed TNT normally despawns before its fuse becomes negative.
    #[test]
    fn tnt_flash_uses_java_division_for_negative_helper_inputs() {
        let actual: Vec<bool> = (-6..=6).map(tnt_flashing).collect();
        assert_eq!(
            actual,
            [false, false, true, true, true, true, true, true, true, true, true, false, false,]
        );
        assert!(tnt_flashing(-4));
        assert!(!tnt_flashing(-6));
    }

    /// java: `f2 = (1 - (fuse - partial + 1) / 100) * 0.8`.
    #[test]
    fn the_tnt_flash_brightens_as_the_fuse_runs_out() {
        let early = tnt_flash_alpha(80, 0.0);
        let late = tnt_flash_alpha(0, 0.0);
        assert!(late > early);
        assert!(approx(early, (1.0 - 81.0 / 100.0) * 0.8), "{early}");
        assert!(approx(late, (1.0 - 1.0 / 100.0) * 0.8), "{late}");
    }

    /// The composite of vanilla's pre-dispatch translation and the block-brightness emitter's
    /// `rotate(90, Y)` must leave a unit cube centred on the entity's mid-height, i.e. spanning
    /// `y in 0..1` above its feet.
    #[test]
    fn primed_tnt_is_a_unit_cube_sitting_on_the_entity_position() {
        let model = full_cube_model();
        let mut out = Vec::new();
        super::super::item::emit_block_model_brightness(
            &model,
            tnt_matrix(Vec3::ZERO, 80, 0.0),
            1.0,
            0xFF_FF_FF,
            &mut out,
        );
        for axis in 0..3 {
            let mn = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[axis]));
            let mx = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[axis]));
            assert!(approx(mx - mn, 1.0), "axis {axis}: {mn}..{mx}");
            let expected = if axis == 1 { (0.0, 1.0) } else { (-0.5, 0.5) };
            assert!(
                approx(mn, expected.0) && approx(mx, expected.1),
                "axis {axis}: {mn}..{mx}"
            );
        }
    }

    /// java: `RenderFallingBlock` shifts the model by `(-0.5, 0, -0.5)` off a *block*-space
    /// draw, so like TNT the cube sits on the entity position — but with no quarter turn.
    #[test]
    fn a_falling_block_sits_on_the_entity_position() {
        let model = full_cube_model();
        let mut out = Vec::new();
        emit_baked_model(&model, falling_block_matrix(Vec3::ZERO), &mut out);
        let mn_y = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[1]));
        let mx_y = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[1]));
        assert!(approx(mn_y, 0.0) && approx(mx_y, 1.0), "{mn_y}..{mx_y}");
        // No rotation: the +X face stays on +X.
        assert_ne!(
            tnt_matrix(Vec3::ZERO, 80, 0.0),
            falling_block_matrix(Vec3::ZERO)
        );
    }

    fn full_cube_model() -> BakedModel {
        let mut model = BakedModel::default();
        // Two opposite faces are enough to bound the cube in all three axes.
        model.general_quads.push(quad_at([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]));
        model.general_quads.push(quad_at([
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]));
        model
    }

    // -- paintings ----------------------------------------------------------------------------

    /// java: `EntityPainting.EnumArt` — 26 constants, and the sizes/offsets they carry.
    #[test]
    fn the_art_table_matches_enum_art() {
        assert_eq!(ART.len(), 26);
        assert_eq!(ART[0].title, "Kebab");
        assert_eq!(ART[25].title, "DonkeyKong");
        let kong = Art::by_title("DonkeyKong");
        assert_eq!(
            (kong.size_x, kong.size_y, kong.offset_x, kong.offset_y),
            (64, 48, 192, 112)
        );
        let sar = Art::by_title("SkullAndRoses");
        assert_eq!(
            (sar.size_x, sar.size_y, sar.offset_x, sar.offset_y),
            (32, 32, 128, 128)
        );
        // Every size is a whole number of 16-texel cells — `renderPainting` divides by 16.
        for a in ART {
            assert_eq!(a.size_x % 16, 0, "{}", a.title);
            assert_eq!(a.size_y % 16, 0, "{}", a.title);
        }
        // Titles are unique, so `by_title` is unambiguous.
        for (i, a) in ART.iter().enumerate() {
            assert!(
                ART.iter().skip(i + 1).all(|b| b.title != a.title),
                "duplicate {}",
                a.title
            );
        }
    }

    /// java: the titled constructor first runs the no-title wall candidate scan. When this
    /// title-only adapter has no candidate list, the source's no-valid-candidate state is the
    /// final enum value, not `EnumArt.KEBAB` and not the first table entry.
    #[test]
    fn an_unknown_painting_title_keeps_the_no_valid_candidate_state() {
        let invalid = Art::by_title("NotAPainting");
        assert_eq!(invalid.title, "DonkeyKong");
        assert_eq!(invalid, &ART[ART.len() - 1]);
        assert_ne!(invalid, &ART[0], "the first-art fallback mutant must bite");
    }

    #[test]
    fn painting_title_lookup_is_exact_and_case_sensitive() {
        assert_eq!(Art::by_title("Kebab").title, "Kebab");
        assert_eq!(Art::by_title("SkullAndRoses").title, "SkullAndRoses");
        assert_eq!(Art::by_title("kebab").title, "DonkeyKong");
        assert_eq!(Art::by_title("DONKEYKONG").title, "DonkeyKong");
    }

    fn one_wall_block(anchor: BlockPos, facing: Facing) -> mc_world::World {
        let mut world = mc_world::World::new();
        world.load_empty_chunk(anchor.chunk_pos());
        let stone = mc_data::block_by_name("stone")
            .expect("vanilla stone is registered")
            .default_state();
        world.set_block(anchor.relative(facing.opposite()), stone);
        world
    }

    /// java: `EntityPainting`'s first constructor chooses only from arts whose complete support
    /// footprint is solid. A one-block wall admits the seven 16x16 entries, but not a 2-wide or
    /// 2-tall candidate; adding the missing support blocks changes those answers independently.
    #[test]
    fn painting_wall_validity_uses_the_art_footprint() {
        let anchor = BlockPos::new(4, 64, 4);
        let facing = Facing::North;
        let mut world = one_wall_block(anchor, facing);

        assert!(painting_surface_valid(&world, anchor, facing, &ART[0], &[]));
        assert!(!painting_surface_valid(
            &world,
            anchor,
            facing,
            &ART[7],
            &[],
        ));
        assert!(!painting_surface_valid(
            &world,
            anchor,
            facing,
            &ART[12],
            &[],
        ));

        let stone = mc_data::block_by_name("stone")
            .expect("vanilla stone is registered")
            .default_state();
        let across = facing.rotate_y_ccw();
        world.set_block(anchor.relative(facing.opposite()).relative(across), stone);
        world.set_block(anchor.relative(facing.opposite()).up(), stone);
        assert!(painting_surface_valid(&world, anchor, facing, &ART[7], &[]));
        assert!(painting_surface_valid(
            &world,
            anchor,
            facing,
            &ART[12],
            &[]
        ));
    }

    /// java: the titled constructor's exact-title pass runs after the random wall candidate. An
    /// unknown title on a one-block wall must therefore resolve to one of the seven valid small
    /// arts, while a known oversized title still wins even though its footprint is invalid.
    #[test]
    fn unknown_painting_title_uses_valid_candidates_before_no_candidate_fallback() {
        let anchor = BlockPos::new(4, 64, 4);
        let facing = Facing::North;
        let world = one_wall_block(anchor, facing);
        assert_eq!(world.block_at(anchor.south()).block().0, 1);
        let mut random = JavaRandom::new(0x51_0A);
        let unknown =
            Art::resolve_for_spawn(&world, anchor, facing, "NotAPainting", &mut random, &[]);
        assert!((0..7).any(|index| *unknown == ART[index]));

        let mut known_random = JavaRandom::new(0x51_0A);
        let known =
            Art::resolve_for_spawn(&world, anchor, facing, "DonkeyKong", &mut known_random, &[]);
        assert_eq!(known.title, "DonkeyKong");
        assert_ne!(random.raw_seed(), JavaRandom::new(0x51_0A).raw_seed());
        assert_ne!(known_random.raw_seed(), JavaRandom::new(0x51_0A).raw_seed());
    }

    /// java: when no candidate passes `onValidSurface`, the no-title loop leaves its final enum
    /// value in `this.art`, and the unknown title does not replace it.
    #[test]
    fn unknown_painting_title_keeps_donkey_kong_when_the_world_has_no_wall() {
        let world = mc_world::World::new();
        let anchor = BlockPos::new(4, 64, 4);
        let mut random = JavaRandom::new(0xD0_0B);
        let before = random.raw_seed();
        let art = Art::resolve_for_spawn(
            &world,
            anchor,
            Facing::North,
            "NotAPainting",
            &mut random,
            &[],
        );
        assert_eq!(art.title, "DonkeyKong");
        assert_eq!(
            random.raw_seed(),
            before,
            "empty candidate list must not draw RNG"
        );
    }

    /// java: `renderPainting`'s `width / 16` by `height / 16` loop, six quads per cell.
    #[test]
    fn a_painting_emits_six_quads_per_sixteen_texel_cell() {
        for a in ART {
            let mut out = Vec::new();
            emit_painting(Vec3::ZERO, 0.0, &a, &mut out);
            let cells = (a.size_x / 16) * (a.size_y / 16);
            assert_eq!(out.len() as i32, cells * 6 * 6, "{}", a.title);
        }
    }

    /// java: `scale(0.0625)` with the art centred on `(-width/2, -height/2)` — a 64x48
    /// `DonkeyKong` is four blocks by three, centred on the entity.
    #[test]
    fn a_painting_is_its_texel_size_in_sixteenths_of_a_block() {
        let kong = Art::by_title("DonkeyKong");
        let mut out = Vec::new();
        emit_painting(Vec3::ZERO, 0.0, kong, &mut out);
        let span = |axis: usize| {
            let mn = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[axis]));
            let mx = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[axis]));
            (mn, mx)
        };
        let (x0, x1) = span(0);
        let (y0, y1) = span(1);
        let (z0, z1) = span(2);
        assert!(approx(x1 - x0, 4.0), "{x0}..{x1}");
        assert!(approx(y1 - y0, 3.0), "{y0}..{y1}");
        // Depth is the `+-f2 = +-0.5` texel pair, one sixteenth of a block thick.
        assert!(approx(z1 - z0, 1.0 / 16.0), "{z0}..{z1}");
        // Centred on the entity in both in-plane axes.
        assert!(approx(x0 + x1, 0.0) && approx(y0 + y1, 0.0));
    }

    /// java: `renderPainting`'s U/V arithmetic — the art's own cell, on a 256-pixel sheet.
    #[test]
    fn the_painting_front_face_samples_its_own_patch_of_the_sheet() {
        let sar = Art::by_title("SkullAndRoses");
        let mut out = Vec::new();
        emit_painting(Vec3::ZERO, 0.0, sar, &mut out);
        // The front face is the first quad of the first cell: six vertices.
        let front = &out[..6];
        let us: Vec<f32> = front.iter().map(|v| v.uv[0]).collect();
        let vs: Vec<f32> = front.iter().map(|v| v.uv[1]).collect();
        let u_lo = us.iter().cloned().fold(f32::MAX, f32::min);
        let u_hi = us.iter().cloned().fold(f32::MIN, f32::max);
        let v_lo = vs.iter().cloned().fold(f32::MAX, f32::min);
        let v_hi = vs.iter().cloned().fold(f32::MIN, f32::max);
        // offsetX 128, width 32, i = 0: u runs (128 + 32 - 16)/256 .. (128 + 32)/256.
        assert!(
            approx(u_lo, 144.0 / 256.0) && approx(u_hi, 160.0 / 256.0),
            "{u_lo}..{u_hi}"
        );
        assert!(
            approx(v_lo, 144.0 / 256.0) && approx(v_hi, 160.0 / 256.0),
            "{v_lo}..{v_hi}"
        );
    }

    /// java: `RenderPainting.setLightmap` samples the *centre* of each 16x16 cell, and rotates
    /// that X sample into the world axis along the painting.  Fractional and negative positions
    /// make the `MathHelper.floor_double` calls observable: a cast-to-int mutant fails here.
    #[test]
    fn painting_cell_light_positions_follow_each_facing_and_floor_negative_coordinates() {
        let world = [10.49, 64.2, -4.51];
        // Centres of the left/right and bottom/top cells of a 32x32 painting.
        let left = -8.0;
        let right = 8.0;
        let bottom = -8.0;
        let top = 8.0;

        // y is shared by every facing and comes from the cell's centre, not the entity's floor.
        assert_eq!(
            painting_cell_light_position(world, Facing::North, left, bottom),
            [9, 63, -5]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::North, right, top),
            [10, 64, -5]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::South, left, bottom),
            [10, 63, -5]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::South, right, top),
            [9, 64, -5]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::West, left, bottom),
            [10, 63, -5]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::West, right, top),
            [10, 64, -6]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::East, left, bottom),
            [10, 63, -6]
        );
        assert_eq!(
            painting_cell_light_position(world, Facing::East, right, top),
            [10, 64, -5]
        );
    }

    /// A 2x2 painting is four independently lit tiled cubes.  This checks the state assignment
    /// as well as the sample coordinates: using one entity-light value for the entire painting,
    /// or changing the nested cell order, makes at least one 36-vertex run wrong.
    #[test]
    fn painting_vertices_retain_their_cells_independent_lightmap_gradient() {
        let art = Art::by_title("Match"); // 32x32, four cells.
        let world = [10.49, 64.2, -4.51];
        let mut seen = Vec::new();
        let mut out = Vec::new();
        emit_painting_with_lightmap(
            Vec3::new(1000.0, -50.0, 3.0), // camera-relative placement must not affect samples.
            hanging_yaw(Facing::North),
            world,
            Facing::North,
            art,
            |pos| {
                seen.push(pos);
                // Deliberately encode x in the low half and y in the high half, so both axes
                // have to survive the Java `%65536`/`/65536` split.
                ((pos[1] as u32) << 16) | pos[0] as u32
            },
            &mut out,
        );

        assert_eq!(seen, [[9, 63, -5], [9, 64, -5], [10, 63, -5], [10, 64, -5]]);
        assert_eq!(out.len(), 4 * 36);
        for (cell, sample) in seen.iter().enumerate() {
            let run = &out[cell * 36..(cell + 1) * 36];
            let expected = PaintingLightmap {
                block: sample[0] as u16,
                sky: sample[1] as u16,
            };
            assert!(run.iter().all(|v| v.lightmap == expected), "cell {cell}");
            // Geometry is still emitted once per cell: every vertex on this tile has the one
            // lightmap state Java set before its six quads.
            assert_eq!(run[0].vertex.color, EntityVertex::WHITE);
        }
        assert_ne!(out[0].lightmap, out[36].lightmap, "vertical gradient lost");
        assert_ne!(
            out[0].lightmap, out[72].lightmap,
            "horizontal gradient lost"
        );
    }

    #[test]
    fn painting_lightmap_splits_the_same_two_halves_as_open_gl_helper() {
        assert_eq!(
            PaintingLightmap::from_combined_light(0x00F0_00A0),
            PaintingLightmap {
                block: 0x00A0,
                sky: 0x00F0
            }
        );
    }

    // -- hanging placement --------------------------------------------------------------------

    /// java: `EntityHanging.updateBoundingBox` — `0.46875` off the wall along the facing.
    #[test]
    fn a_hanging_entity_stands_off_its_wall() {
        // A 1x1 painting on a north-facing wall (facing = the direction it looks).
        let p = hanging_position([10, 64, 20], Facing::North, 16, 16);
        assert!((p[0] - 10.5).abs() < 1e-9, "{p:?}");
        assert!((p[1] - 64.5).abs() < 1e-9, "{p:?}");
        // North's front offset is z = -1, so the entity moves +0.46875 in z.
        assert!((p[2] - (20.5 + 0.468_75)).abs() < 1e-9, "{p:?}");
    }

    /// java: `func_174858_a` returns `0.5` only for sizes that are a multiple of 32 — which is
    /// what makes even-celled art straddle the block boundary.
    #[test]
    fn even_sized_art_is_offset_by_half_a_block() {
        let odd = hanging_position([0, 0, 0], Facing::South, 16, 16);
        assert!((odd[1] - 0.5).abs() < 1e-9);
        let even = hanging_position([0, 0, 0], Facing::South, 32, 32);
        assert!((even[1] - 1.0).abs() < 1e-9, "{even:?}");
        // 32 wide slides along rotateYCCW(south) = east, i.e. +x.
        assert!((even[0] - 1.0).abs() < 1e-9, "{even:?}");
        // 64x48: width 64 is a multiple of 32 so it slides; height 48 is not so it does not.
        let kong = hanging_position([0, 0, 0], Facing::South, 64, 48);
        assert!((kong[1] - 0.5).abs() < 1e-9, "{kong:?}");
        assert!((kong[0] - 1.0).abs() < 1e-9, "{kong:?}");
    }

    /// java: `EntityHanging.updateFacingWithBoundingBox` — `horizontalIndex * 90`, over
    /// vanilla's S-W-N-E ordering.
    #[test]
    fn a_hanging_entity_faces_its_wall() {
        assert!(approx(hanging_yaw(Facing::South), 0.0));
        assert!(approx(hanging_yaw(Facing::West), 90.0));
        assert!(approx(hanging_yaw(Facing::North), 180.0));
        assert!(approx(hanging_yaw(Facing::East), 270.0));
    }

    // -- object placement ---------------------------------------------------------------------

    /// None of the object matrices may carry `RendererLivingEntity`'s `-1.5078125` lift.
    /// The regression this pins is a boat floating a block and a half over the water.
    #[test]
    fn no_object_matrix_lifts_the_model_like_a_living_entity() {
        let rel = Vec3::new(1.0, 2.0, 3.0);
        // The living matrix moves the model origin down by 1.5078125 (through the -1 Y flip it
        // comes out as a *lift* of the geometry, but the origin itself lands there).
        let living = super::super::render::entity_matrix(&super::super::render::EntityTransform {
            rel,
            ..Default::default()
        });
        let living_origin = living.transform_point3(Vec3::ZERO);
        assert!(
            approx(living_origin.y, 2.0 + 1.5078125),
            "{living_origin:?}"
        );

        for (name, m, expected_y) in [
            ("boat", boat_matrix(rel, 0.0, 0.0, 0.0, 1, 0.0), 2.25),
            (
                "minecart",
                minecart_matrix(rel, 0.0, 0.0, 0.0, 0.0, 1, 0.0),
                2.375,
            ),
            ("leash knot", leash_knot_matrix(rel), 2.0),
            ("ender crystal", ender_crystal_matrix(rel), 2.0),
            ("wither skull", wither_skull_matrix(rel), 2.0),
        ] {
            let o = m.transform_point3(Vec3::ZERO);
            assert!(approx(o.y, expected_y), "{name}: {o:?}");
        }
    }

    /// F-P1-entity-07-1: cargo is submitted before the shell's negative model flip and uses the
    /// subtype/custom display offset in sixteenths after the fixed 0.75 scale.
    #[test]
    fn minecart_display_uses_java_scale_offset_and_not_the_shell_flip() {
        let display6 = minecart_display_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 6);
        let display8 = minecart_display_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 8);
        let hopper = minecart_display_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 1);

        assert!(approx(
            display6.transform_point3(Vec3::ZERO).y,
            0.375 + 0.75 * -2.0 / 16.0
        ));
        assert!(approx(display8.transform_point3(Vec3::ZERO).y, 0.375));
        assert!(approx(
            hopper.transform_point3(Vec3::ZERO).y,
            0.375 + 0.75 * -7.0 / 16.0
        ));
        assert!(
            display6.transform_vector3(Vec3::Y).y > 0.0,
            "the display block must not inherit ModelMinecart's negative-Y shell flip"
        );
        assert!(
            minecart_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0)
                .transform_vector3(Vec3::Y)
                .y
                < 0.0
        );
    }

    /// `RenderTntMinecart.renderCartContents` inserts its swell before the ordinary cargo
    /// transform; the wrapper remains byte-for-byte equivalent to a scale of one.
    #[test]
    fn tnt_minecart_swell_precedes_the_fixed_cargo_scale_and_offset() {
        let ordinary = minecart_display_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 6);
        let identity =
            minecart_display_matrix_scaled(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 6, 1.0);
        assert_eq!(ordinary, identity);

        let swollen =
            minecart_display_matrix_scaled(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 6, 1.2);
        assert!(approx(
            swollen.transform_vector3(Vec3::Y).length(),
            ordinary.transform_vector3(Vec3::Y).length() * 1.2
        ));
        assert!(approx(
            swollen.transform_point3(Vec3::ZERO).y - 0.375,
            (ordinary.transform_point3(Vec3::ZERO).y - 0.375) * 1.2
        ));
    }

    /// java: `RenderLeashKnot.doRender` has no `rotate` call at all — a knot never turns.
    #[test]
    fn the_leash_knot_ignores_yaw() {
        let a = leash_knot_matrix(Vec3::ZERO);
        // The knot's own +X stays on -X only because of the (-1, -1, 1) flip, never because of
        // a yaw rotation; the matrix is therefore constant for a given position.
        assert_eq!(a, leash_knot_matrix(Vec3::ZERO));
        let d = a.transform_vector3(Vec3::X);
        assert!(approx(d.x, -1.0) && approx(d.z, 0.0), "{d:?}");
    }

    /// java: `RenderEnderCrystal.doRender` translates and nothing else.
    #[test]
    fn the_ender_crystal_matrix_is_a_pure_translation() {
        let m = ender_crystal_matrix(Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(m, Mat4::from_translation(Vec3::new(4.0, 5.0, 6.0)));
    }

    /// java: `RenderBoat.java:41` — the rock is gated on `getTimeSinceHit() - partialTicks > 0`
    /// and scaled by the damage taken, so an undamaged boat sits flat.
    #[test]
    fn an_undamaged_boat_does_not_rock() {
        let flat = boat_matrix(Vec3::ZERO, 90.0, 0.0, 0.0, 1, 0.0);
        let hit = boat_matrix(Vec3::ZERO, 90.0, 8.0, 40.0, 1, 0.0);
        assert_ne!(flat, hit);
        // Hit but undamaged: f1 clamps to 0, so the angle is 0 and the matrix is unchanged.
        assert_eq!(flat, boat_matrix(Vec3::ZERO, 90.0, 8.0, 0.0, 1, 0.0));
    }

    /// java: `RenderMinecart.doRender` rotates the damage rock around X after the rail-pitch Z
    /// rotation. A Z-axis mutant leaves local X fixed and therefore fails this direction oracle.
    #[test]
    fn minecart_damage_rock_uses_java_x_axis() {
        let flat = minecart_matrix(Vec3::ZERO, 0.0, 0.0, 0.0, 0.0, 1, 0.0);
        let hit = minecart_matrix(Vec3::ZERO, 0.0, 0.0, 8.0, 40.0, 1, 0.0);
        let flat_y = flat.transform_vector3(Vec3::Y);
        let hit_y = hit.transform_vector3(Vec3::Y);

        assert!(approx(flat_y.z, 0.0));
        assert!(
            hit_y.z.abs() > 1.0e-4,
            "X rotation must tilt local Y into Z: {hit_y:?}"
        );
        assert!(
            approx(hit_y.x, flat_y.x),
            "X rotation must preserve local X"
        );
    }
}
