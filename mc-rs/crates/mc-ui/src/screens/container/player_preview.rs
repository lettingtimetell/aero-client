//! The rotating 3-D player preview embedded in the survival and creative inventory screens.
//!
//! java: `GuiInventory.drawEntityOnScreen`, called once from `GuiInventory
//! .drawGuiContainerBackgroundLayer` (survival) and once from `GuiContainerCreative
//! .drawGuiContainerBackgroundLayer` (creative, only while the Inventory tab is open) — the
//! *same* static method both call, just with different `posX, posY, scale` and mouse-offset
//! arithmetic.
//!
//! # Why this is CPU geometry, not a second GPU camera
//!
//! [`DrawList`] is a flat list of textured triangles, drawn by `mc-ui`'s own `UiPhase` under one
//! fixed orthographic projection per frame (`@group(0)`'s `ui.proj`; see `crate::widget::gpu`'s
//! module doc for why that phase never touches `mc-render`'s shared camera bind group). Player
//! triangles retain a parallel GUI-depth stream and the renderer's cull-disabled state window.
//! `mc-render`'s entity pipeline is the opposite: one `wgpu::RenderPipeline` reading
//! `gpu.camera_bind_group()`, a *single* uniform [`mc_render::PhaseList::render`] writes once per
//! frame for every 3-D phase that frame. There is no seam to swap in a second camera for one GUI
//! rectangle without a second render target and a second submission — a much bigger change than
//! embedding one panel needs, and not one vanilla itself needed either:
//!
//! `drawEntityOnScreen` runs *inside* the same orthographic GUI projection every other 2-D
//! element uses (`EntityRenderer.setupOverlayRendering`'s `GL11.glOrtho(0, w, h, 0, 1000, 3000)`
//! — genuinely orthographic, no perspective divide). So the "3-D" player in the inventory has
//! never been perspective-correct in vanilla either; it is a real 3-D pose, rotated and lit, with
//! screen position taken from `(x, y)` while its Z remains in the UI depth stream. That is exactly the technique
//! `super::render::draw_iso_block_icon` already uses for block item icons (a different rotation,
//! the same rotate-then-drop-`z` idea) — this module is that idea at bigger scale: pose a real
//! biped model with `mc-render`'s already-proven [`RenderManager::emit_with`] (the same function
//! `mc-client`'s `driver.rs` — `player_body_renderer`/`upload_entities` — uses to draw a remote
//! player in the 3-D world), feed it a `base` matrix that reproduces vanilla's GL matrix stack
//! **in GUI pixel space** instead of camera-relative world space, and hand the resulting
//! triangles straight to [`DrawList::triangle`] with per-vertex Lambertian shading computed on
//! the CPU (there is no fragment shader in this path to share `ENTITY_WGSL`'s copy of the same
//! formula with).
//!
//! # The transform, matching vanilla's GL calls in `drawEntityOnScreen` one for one
//!
//! ```text
//! translate(posX, posY, 0)        // GlStateManager.translate(posX, posY, 50) — z unused, dropped
//! scale(-scale, scale, scale)     // GlStateManager.scale(-scale, scale, scale)
//! rotate(180, Z)                  // GlStateManager.rotate(180, 0, 0, 1)
//! rotate(pitchGl, X)              // GlStateManager.rotate(-atan(mouseY/40)*20, 1, 0, 0)
//! entity_matrix(EntityTransform { rel: 0, body_yaw, y_offset: MODEL_Y_OFFSET, scale: 1 })
//!                                 // renderEntityWithPosYaw -> RendererLivingEntity.doRender
//! ```
//!
//! `rotate(135, Y)` / `rotate(-135, Y)` — vanilla's light-setup bracket, sandwiching
//! `RenderHelper.enableStandardItemLighting()` — cancel exactly for the *geometry* (net identity)
//! and are omitted from the transform above. They do **not** cancel for the two fixed light
//! directions `RenderHelper` bakes in between them (a GL directional light's direction is frozen
//! under whatever modelview is current at the `glLight` call, not re-evaluated afterwards).
//! [`frozen_preview_light_direction`] reproduces that captured frame, including the preceding
//! negative-X GUI scale and 180-degree Z turn, while intentionally leaving out the later pitch.
//!
//! # The "looking at the cursor" formula — vanilla's, verbatim quirk included
//!
//! java, `GuiInventory.drawEntityOnScreen`:
//! ```java
//! ent.renderYawOffset = (float) Math.atan((double) (mouseX / 40.0F)) * 20.0F;
//! ent.rotationYaw      = (float) Math.atan((double) (mouseX / 40.0F)) * 40.0F;
//! ent.rotationPitch    = -((float) Math.atan((double) (mouseY / 40.0F))) * 20.0F;
//! ```
//! `Math.atan` returns **radians**; that radian value — not a degrees conversion of it — is what
//! vanilla multiplies by 20/40 and hands to a "degrees" `GlStateManager.rotate` call. That is not
//! a transcription slip in this file: it is what the decompiled source does (`(float)
//! Math.atan(...)) * 20.0F` used directly as a rotation angle in degrees), so `look_component`
//! reproduces exactly that — `.to_radians()` is applied to the already-mislabelled value inside
//! `outer_transform`/[`entity_matrix`]'s caller, not "fixed" here — which is what makes the look
//! curve match vanilla's rather than a naive real-degrees version.
//!
//! `mouse_x`/`mouse_y` passed to [`PlayerPreviewRenderer::draw`] are **not** the raw cursor
//! position: both vanilla call sites pass `anchor - realMouse` (`GuiInventory`'s survival panel
//! additionally lags this by one frame, matching vanilla's own `oldMouseX`/`oldMouseY`; creative's
//! `GuiContainerCreative` does not lag it, matching vanilla). Computing that offset — and the
//! frame lag, where it applies — is the caller's job; see `screen::ContainerScreen::draw` and
//! `creative_screen::CreativeScreen::draw` for the two call sites.
//!
//! # Snapshot boundary
//!
//! [`PlayerPreviewSnapshot`] carries the body animation inputs and the protocol-visible renderer
//! layers that can be represented without reaching back into client state. The old [`Self::draw`]
//! API remains a deterministic bare/idle negative path, while inventory screens use
//! [`Self::draw_with_snapshot`] for armour, held-item, arrow, custom-head, cape and live pose
//! passes.
//!
//! * **Textures**: the host supplies the player's downloaded body skin and cape. A helmet
//!   skull carries its own resolved owner descriptor, sharing the asynchronous skin cache and
//!   UUID-derived fallback used by item icons and the world renderer.
//! * **Animation and layers**: the snapshot supplies live equipment, model-part visibility,
//!   limb swing, sneaking/riding/invisibility, arrows and cape motion, keeping the body and its
//!   layers on one posed model. Spectator/death/sleeping camera branches and the item-use
//!   countdown remain outside this snapshot boundary.
//! * **The frozen-light-direction bracket** noted above — [`lambertian`] uses the captured eye-space
//!   vectors, and the horse path applies the same GUI normal matrix as the player path.

use glam::{Mat4, Vec3};

use mc_render::entity::item::{
    emit_armor_glint, emit_baked_model_glint, emit_baked_model_tinted, item_glint_system_time_ms,
    ENCHANTED_ITEM_GLINT_TEXTURE,
};
use mc_render::entity::layers::{
    armor_material_name, armor_texture, custom_head_matrix, ArmorSlot, CapeState, CustomHeadItem,
};
use mc_render::entity::models::passive::HorseRenderState;
use mc_render::entity::render::{entity_matrix, EntityTransform, MODEL_Y_OFFSET};
use mc_render::entity::tilepass::{BuiltinSkullOwner, BuiltinTileItem};
use mc_render::entity::{
    emit_arrow_from, renderer_for_player, AnimParams, EntityVertex, ModelKind, RenderManager,
    SkinModel, ARROW_TEXTURE,
};

use crate::widget::draw::{UiBlend, UiCull, UiDepth};
use crate::widget::{DrawList, UiTexture};

use super::stack_ext::ItemStack;

/// The live player facts needed by the inventory preview's renderer layers.
///
/// java: the same `EntityPlayer` instance passed to `GuiInventory.drawEntityOnScreen`.  Keeping
/// the snapshot explicit is important: armour, held items, cape and animation all have to be
/// sampled from one frame, rather than letting a partially-updated screen accidentally combine
/// an old pose with new equipment. `equipment` follows the world renderer's convention:
/// `[held, boots, leggings, chestplate, helmet]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayerPreviewSnapshot {
    /// The live `RendererLivingEntity`/`ModelBiped` animation inputs.
    pub params: AnimParams,
    /// `AbstractClientPlayer.getModelFlags()` used by `RenderPlayer.setModelVisibilities`.
    ///
    /// java: `EnumPlayerModelParts` bits for cape, jacket, sleeves, pants and hat.  The preview
    /// has its own `RenderManager`, so this must be applied immediately before posing the shared
    /// player model instead of relying on the world renderer's last entity.
    pub model_parts: i8,
    /// Held item followed by the four armour slots.
    pub equipment: [Option<ItemStack>; 5],
    /// The helmet stack's resolved `SkullOwner`, independent of the wearer's own skin.
    /// Its descriptor selects the owner-derived fallback until the shared skin cache is ready.
    pub helmet_skull_owner: Option<BuiltinSkullOwner>,
    /// Live cape chase/sneak state, when the profile has a cape.
    pub cape: Option<CapeState>,
    /// The resolved profile cape texture. A state without a texture is intentionally not drawn.
    pub cape_texture: Option<UiTexture>,
    /// java: `!thePlayer.isInvisibleToPlayer(thePlayer)` for an invisible player: true for a
    /// spectator, or when the player's own team sees friendly invisibles
    /// (`EntityPlayer.isInvisibleToPlayer`). `RendererLivingEntity.renderModel` then draws the
    /// invisible body as the 15% ghost instead of skipping it. Ignored while visible.
    pub sees_own_invisible_body: bool,
}

impl Default for PlayerPreviewSnapshot {
    fn default() -> Self {
        Self {
            params: AnimParams::still(),
            // A direct `draw` call has no profile/settings source; keep the historical bare-body
            // default while live producers supply the actual 0x7f/settings mask.
            model_parts: 0,
            equipment: [None, None, None, None, None],
            helmet_skull_owner: None,
            cape: None,
            cape_texture: None,
            sees_own_invisible_body: false,
        }
    }
}

/// The watched/client-local horse state used by `GuiScreenHorseInventory`'s preview.
///
/// `entity_id` is retained with the state so a host cannot accidentally render a stale horse
/// after a network id is reused. `texture` is the host-resolved layered horse texture selected by
/// `RenderManager::emit_horse_with_state` from the packed colour/marking/armour values.
#[derive(Clone, Debug, PartialEq)]
pub struct HorsePreviewSnapshot {
    /// `OpenWindow.entity_id` / the tracked entity id this snapshot belongs to.
    pub entity_id: i32,
    /// Live living-animation inputs for the mount.
    pub params: AnimParams,
    /// Watched saddle/reins/chest/type and interpolated horse springs.
    pub state: HorseRenderState,
    /// `EntityHorse`'s packed colour and markings watcher.
    pub packed_variant: i32,
    /// `EntityHorse`'s armour watcher (`0` none, `1..=3` iron/gold/diamond).
    pub armor: i32,
    /// The resolved base/layered horse texture.
    pub texture: UiTexture,
}

impl Default for HorsePreviewSnapshot {
    fn default() -> Self {
        Self {
            entity_id: 0,
            params: AnimParams::still(),
            state: HorseRenderState::default(),
            packed_variant: 0,
            armor: 0,
            texture: UiTexture::gui("textures/entity/horse/horse_white.png"),
        }
    }
}

/// java: `RenderHelper`'s two fixed directional lights, exactly [`mc_render::entity::render`]'s
/// `ENTITY_WGSL` `LIGHT0`/`LIGHT1` constants (`(0.2, 1.0, -0.7)`/`(-0.2, 1.0, 0.7)`).
const LIGHT0: Vec3 = Vec3::new(0.2, 1.0, -0.7);
const LIGHT1: Vec3 = Vec3::new(-0.2, 1.0, 0.7);

/// The modelview frame in which Java freezes `RenderHelper`'s directional lights.
///
/// `GuiInventory.drawEntityOnScreen` (and the horse screen's copy of the helper) does this in
/// this order:
///
/// ```text
/// scale(-scale, scale, scale)
/// rotate(180, Z)
/// rotate(135, Y)
/// RenderHelper.enableStandardItemLighting() // glLight(..., w = 0) captures this frame
/// rotate(-135, Y)
/// rotate(pitch, X)
/// renderEntityWithPosYaw(...)
/// ```
///
/// The `+135` and `-135` calls cancel for geometry, but not for a directional light: OpenGL
/// transforms the `w = 0` position at the `glLight` call and retains the resulting eye-space
/// vector. The later pitch is intentionally absent here because it happens *after* the light
/// capture. Translation is also absent because it cannot affect a direction. The scale magnitude
/// is immaterial after normalisation; its negative X sign is not.
#[must_use]
fn frozen_preview_light_direction(light: Vec3) -> Vec3 {
    let light_frame = Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0))
        * Mat4::from_rotation_z(180.0f32.to_radians())
        * Mat4::from_rotation_y(135.0f32.to_radians());
    light_frame.transform_vector3(light).normalize_or_zero()
}

/// Vanilla's two-light diffuse formula — `0.4` ambient floor plus up to `0.6` from each light,
/// applied to an already-fully-transformed vertex normal (`base`'s rotation carries through
/// [`RenderManager::emit_with`] into [`EntityVertex::normal`] exactly as it does for the plain
/// 3-D world case; see that struct's own doc).
///
/// This deliberately returns the **unclamped** light scalar. Fixed-function
/// `GL_COLOR_MATERIAL(AMBIENT_AND_DIFFUSE)` multiplies lighting into the primary/material colour
/// and clamps those resulting channels; [`shade`] performs that operation before [`DrawList`]'s
/// texture modulation. Clamping here instead makes tinted upward faces too dark because the two
/// standard item lights add above `1.0` there.
#[must_use]
fn lambertian(normal: Vec3) -> f32 {
    let light0 = frozen_preview_light_direction(LIGHT0);
    let light1 = frozen_preview_light_direction(LIGHT1);
    let n = normal.normalize_or_zero();
    0.4 + 0.6 * n.dot(light0).max(0.0) + 0.6 * n.dot(light1).max(0.0)
}

/// Vanilla's `(float) Math.atan((double) (x / 40.0F)) * k` idiom, shared by all three call sites
/// in `drawEntityOnScreen` (see the module doc's quirk note on the returned value's unit).
///
/// `f32`/`f64` `atan` — not an `mc_math` table: vanilla calls `java.lang.Math.atan` here, not
/// `MathHelper`'s LUT-backed `sin`/`cos`/`atan2` (hazard H1 is about that table, not this).
#[must_use]
fn look_component(x: f32, k: f32) -> f32 {
    (f64::from(x / 40.0)).atan() as f32 * k
}

/// The four vanilla locals `drawEntityOnScreen` derives from `mouseX`/`mouseY`, named after their
/// java fields.
#[derive(Clone, Copy, Debug, PartialEq)]
struct LookAngles {
    /// java: `ent.renderYawOffset`
    body_yaw: f32,
    /// java: `netHeadYaw` as `RendererLivingEntity.doRender` derives it —
    /// `rotationYawHead - renderYawOffset`, and `rotationYawHead` is set to `rotationYaw` two
    /// lines below.
    net_head_yaw: f32,
    /// java: `ent.rotationPitch`, read by the model as head pitch.
    head_pitch: f32,
    /// java: the separate whole-model `GlStateManager.rotate(...)` call — numerically the same
    /// value as `head_pitch` here, but applied to the *outer* transform, not just the head part.
    pitch_gl: f32,
}

#[must_use]
fn look_angles(mouse_x: f32, mouse_y: f32) -> LookAngles {
    let render_yaw_offset = look_component(mouse_x, 20.0);
    let rotation_yaw = look_component(mouse_x, 40.0);
    let rotation_pitch = -look_component(mouse_y, 20.0);
    LookAngles {
        body_yaw: render_yaw_offset,
        net_head_yaw: rotation_yaw - render_yaw_offset,
        head_pitch: rotation_pitch,
        pitch_gl: rotation_pitch,
    }
}

/// Return the extra render-space Y displacement from `RenderPlayer.doRender`.
///
/// `GuiInventory` and `GuiContainerCreative` pass `Minecraft.thePlayer`, which is an
/// `EntityPlayerSP`.  Java's sneaking displacement is explicitly guarded by
/// `!(entity instanceof EntityPlayerSP)`, so the local inventory preview must not receive the
/// usual `-0.125D` remote-entity offset.  Keep the Java condition explicit here in case this
/// helper is later reused for a non-local player preview.
#[must_use]
fn render_player_y_offset(is_local_player: bool, sneaking: bool) -> f32 {
    if sneaking && !is_local_player {
        -0.125
    } else {
        0.0
    }
}

/// The outer GL matrix stack `drawEntityOnScreen` builds before calling
/// `renderEntityWithPosYaw` — see the module doc's transform table.
#[must_use]
fn outer_transform(pos_x: i32, pos_y: i32, scale: i32, pitch_gl_deg: f32) -> glam::Mat4 {
    // java: drawEntityOnScreen translates Z by 50 before RenderManager renders the model. The
    // prior CPU path dropped this only because its painter list had no depth attachment; retain
    // it now so the UI depth value lands in the same 1000..3000 orthographic bracket as vanilla.
    glam::Mat4::from_translation(Vec3::new(pos_x as f32, pos_y as f32, 50.0))
        * glam::Mat4::from_scale(Vec3::new(-(scale as f32), scale as f32, scale as f32))
        * glam::Mat4::from_rotation_z(180f32.to_radians())
        * glam::Mat4::from_rotation_x(pitch_gl_deg.to_radians())
}

/// Map the preview's post-outer-transform eye Z to WGPU's 0..1 GUI depth range. The outer matrix
/// already includes Java's `translate(..., 50)`, so this is the same `glOrtho(..., 1000, 3000)`
/// conversion used by inventory item models.
#[must_use]
fn preview_depth(view_z: f32) -> f32 {
    (0.5 - view_z / 2000.0).clamp(0.0, 1.0)
}

/// Owns the (tiny, lazily-built) player body model, shared across frames so the inventory panel
/// does not rebuild it on every draw. java: no direct equivalent — vanilla shares one
/// `RenderManager` globally; this is that same sharing, scoped to whichever screen embeds a
/// preview (`ContainerScreen`/`CreativeScreen` each own one).
pub struct PlayerPreviewRenderer {
    manager: RenderManager,
}

impl std::fmt::Debug for PlayerPreviewRenderer {
    // `mc_render::entity::RenderManager` does not derive `Debug` (its models are `Box<dyn
    // EntityModel>`), so this can only report that it exists, not what it holds.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlayerPreviewRenderer")
            .finish_non_exhaustive()
    }
}

impl Default for PlayerPreviewRenderer {
    fn default() -> Self {
        Self::new()
    }
}

impl PlayerPreviewRenderer {
    /// An empty renderer; the player body model is built on first [`Self::draw`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            manager: RenderManager::new(),
        }
    }

    /// java: `GuiInventory.drawEntityOnScreen(posX, posY, scale, mouseX, mouseY, ent)`.
    ///
    /// `mouse_x`/`mouse_y` must already be `anchor - realMouse`, matching vanilla's own call
    /// sites — see the module doc. `age_in_ticks` drives idle-only animation (see the module
    /// doc's "what is not reproduced" for what this does *not* drive).
    #[allow(clippy::too_many_arguments)] // near one-for-one with vanilla's own static method,
                                         // plus `skin_model`/`age_in_ticks`, which vanilla reads off the live entity object instead.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        pos_x: i32,
        pos_y: i32,
        scale: i32,
        mouse_x: f32,
        mouse_y: f32,
        skin_model: SkinModel,
        age_in_ticks: f32,
        skin_texture: Option<&UiTexture>,
    ) {
        let snapshot = PlayerPreviewSnapshot {
            params: AnimParams {
                age_in_ticks,
                ..AnimParams::still()
            },
            ..PlayerPreviewSnapshot::default()
        };
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        self.draw_with_snapshot(
            d,
            pos_x,
            pos_y,
            scale,
            mouse_x,
            mouse_y,
            skin_model,
            age_in_ticks,
            skin_texture,
            &snapshot,
            &atlas,
        );
    }

    /// Draw a player using one live snapshot for the body and all renderer layers.
    ///
    /// java: `GuiInventory.drawEntityOnScreen` -> `RenderPlayer.doRender`, where every layer
    /// reads the same posed `EntityPlayer`. `atlas` is the resolved block/item atlas used by the
    /// held-item layer; entity textures remain independent resources.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_with_snapshot(
        &mut self,
        d: &mut DrawList,
        pos_x: i32,
        pos_y: i32,
        scale: i32,
        mouse_x: f32,
        mouse_y: f32,
        skin_model: SkinModel,
        age_in_ticks: f32,
        skin_texture: Option<&UiTexture>,
        snapshot: &PlayerPreviewSnapshot,
        atlas: &UiTexture,
    ) {
        let look = look_angles(mouse_x, mouse_y);
        let outer = outer_transform(pos_x, pos_y, scale, look.pitch_gl);
        let renderer = renderer_for_player(skin_model);

        let mut params = snapshot.params.clone();
        params.age_in_ticks = age_in_ticks;
        params.net_head_yaw = look.net_head_yaw;
        params.head_pitch = look.head_pitch;

        // java: `RenderPlayer.setModelVisibilities` starts with the currently equipped stack,
        // then promotes it to the block/bow pose while that stack is in use.  The UI snapshot
        // carries the same already-decoded pose flags when a live producer has them; filling the
        // ordinary `1` case here keeps the local inventory path faithful even though the screen
        // API does not carry the item-use countdown.  A missing stack clears stale pose bits,
        // exactly as the Java setter does.
        let has_held_item = snapshot.equipment[0].is_some();
        if !has_held_item {
            params.entity.held_item_right = 0;
            params.entity.aimed_bow = false;
        } else if params.entity.aimed_bow {
            params.entity.held_item_right = 1;
        } else {
            params.entity.held_item_right = if params.entity.held_item_right == 3 {
                3
            } else {
                1
            };
        }
        // java: `RenderPlayer.setModelVisibilities` hard-codes the off-hand flag to zero in
        // 1.8.9; this preview has no second equipped slot to override that constant.
        params.entity.held_item_left = 0;

        // Exhaustive on purpose: java's `RenderPlayer` has no `preRenderCallback` override at
        // all, so if `EntityTransform` ever grows another pre-render field this literal should
        // break and be looked at rather than silently inherit a default.
        let mut transform = EntityTransform {
            // java: the inventory call sites pass the local `EntityPlayerSP`, so
            // `RenderPlayer.doRender`'s `!(entity instanceof EntityPlayerSP)` guard suppresses
            // this displacement even while the model still receives its sneaking pose.
            rel: Vec3::new(
                0.0,
                render_player_y_offset(true, params.entity.sneaking),
                0.0,
            ),
            body_yaw: look.body_yaw,
            death_rotation: 0.0,
            post_yaw: 0.0,
            scale: 1.0,
            y_offset: MODEL_Y_OFFSET,
            upside_down: false,
            upside_down_height: 0.0,
        };
        // Matches `RenderManager::emit`'s own "descriptor scale composes with the caller's own
        // dynamic scale" pattern — always `1.0` for a player today, kept for consistency with
        // every other `emit`/`emit_with` call site rather than assuming it always will be.
        transform.scale *= renderer.scale;

        let base = outer * entity_matrix(&transform);

        let mut verts = Vec::new();
        // java: `RenderPlayer.setModelVisibilities` runs before `ModelBiped.setRotationAngles`.
        // The preview manager shares model instances across frames, so always restore the mask
        // even when this frame has no optional layers.
        self.manager
            .set_player_model_visibility(renderer.model, false, snapshot.model_parts);
        self.manager
            .emit_with(renderer.model, base, &params, &mut verts);

        let fallback_texture = UiTexture::gui(renderer.texture);
        let texture = skin_texture.unwrap_or(&fallback_texture);
        // java: `RendererLivingEntity.renderModel` (`:246-278`), the local player viewing itself:
        // an invisible body is skipped unless `isInvisibleToPlayer` keeps it visible, and then
        // it is the `color(1, 1, 1, 0.15)` ghost. The layers below draw either way.
        if !params.entity.invisible {
            draw_sorted_triangles(d, texture, &verts);
        } else if snapshot.sees_own_invisible_body {
            mc_render::entity::tint(&mut verts, [1.0, 1.0, 1.0, 0.15]);
            draw_sorted_triangles(d, texture, &verts);
        }

        // Vanilla replays layers after the body with depth testing enabled. Draw each pass in
        // painter order here is the deterministic CPU fallback; production batches also carry
        // per-vertex GUI depth for the shared UI attachment.
        for (slot, equipment_index) in [
            (ArmorSlot::Helmet, 4usize),
            (ArmorSlot::Chestplate, 3),
            (ArmorSlot::Leggings, 2),
            (ArmorSlot::Boots, 1),
        ] {
            let Some(stack) = snapshot.equipment[equipment_index].as_ref() else {
                continue;
            };
            let Ok(id) = u16::try_from(stack.item_id) else {
                continue;
            };
            let Some(material) = mc_data::item_props(mc_data::ItemId(id)).armor_material else {
                continue;
            };
            let mut armour = Vec::new();
            self.manager
                .emit_armor_slot_for(renderer.model, slot, &params, base, &mut armour);
            if armour.is_empty() {
                continue;
            }
            let material_name = armor_material_name(material);
            if material == mc_data::ArmorMaterial::Leather {
                let dye = crate::widget::draw::unpack_argb(
                    0xFF00_0000 | super::render::item_quad_color_rgb(stack, Some(0)),
                );
                let mut dyed = armour.clone();
                mc_render::entity::tint(&mut dyed, dye);
                draw_sorted_triangles_with_blend(
                    d,
                    &UiTexture::gui(&armor_texture(material_name, slot, None)),
                    UiBlend::Alpha,
                    &dyed,
                );
                draw_sorted_triangles_with_blend(
                    d,
                    &UiTexture::gui(&armor_texture(material_name, slot, Some("overlay"))),
                    UiBlend::Alpha,
                    &armour,
                );
            } else {
                draw_sorted_triangles_with_blend(
                    d,
                    &UiTexture::gui(&armor_texture(material_name, slot, None)),
                    UiBlend::Alpha,
                    &armour,
                );
            }
            if super::tooltip::is_item_enchanted(stack) {
                let mut glint = Vec::new();
                emit_armor_glint(&armour, params.age_in_ticks, &mut glint);
                draw_sorted_triangles_with_blend(
                    d,
                    &UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE),
                    UiBlend::Glint,
                    &glint,
                );
            }
        }

        if let Some(stack) = snapshot.equipment[0].as_ref() {
            if let Ok(id) = u16::try_from(stack.item_id) {
                let item = mc_data::ItemId(id);
                let is_full_block = mc_data::item_props(item).block.is_some_and(|block| {
                    mc_data::block_props(block.default_state()).render_type
                        == mc_data::RenderType::TileEntity
                });
                if let Some(matrix) = self.manager.held_item_matrix_for(
                    renderer.model,
                    params.scale,
                    base,
                    params.entity.child,
                    is_full_block,
                    params.entity.sneaking,
                ) {
                    let model = mc_assets::item_model_for_damage(item, stack.damage);
                    let matrix = matrix
                        // java: `LayerHeldItem` enters `ItemRenderer.renderItem`, which
                        // contributes the context display transform, the GUI-3D/flat branch,
                        // and RenderItem's unconditional common half-scale before the baked
                        // model's `(-.5, -.5, -.5)` recenter.  Using only
                        // `item_transform_matrix` loses the two scale branches and makes
                        // third-person tools/blocks visibly oversized or undersized.
                        * mc_render::entity::objects::item_renderer_model_transform(
                            model.gui_3d,
                            &model.transforms.third_person,
                        );
                    let mut held = Vec::new();
                    emit_baked_model_tinted(model, matrix, &mut held, |tint| {
                        super::render::item_quad_color_rgb(stack, Some(tint))
                    });
                    draw_sorted_triangles_with_blend(d, atlas, UiBlend::Alpha, &held);
                    if super::tooltip::is_item_enchanted(stack) {
                        let mut glint = Vec::new();
                        emit_baked_model_glint(
                            model,
                            matrix,
                            item_glint_system_time_ms(),
                            &mut glint,
                        );
                        draw_sorted_triangles_with_blend(
                            d,
                            &UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE),
                            UiBlend::Glint,
                            &glint,
                        );
                    }
                }
            }
        }

        // java: `LayerArrow.doRenderLayer` — the live arrow count and entity id are both part
        // of the protocol-facing render snapshot.  `RenderManager` owns the exact seeded
        // ModelPlayer box/cube sampling; this pass only supplies the GUI base and replays the
        // ordinary `RenderArrow` geometry under its resource texture.
        if params.entity.arrow_count > 0 {
            let mut arrows = Vec::new();
            for attachment in self.manager.stuck_arrow_attachments(
                renderer.model,
                params.entity.entity_id,
                params.entity.arrow_count,
                params.scale,
                base,
            ) {
                // java: the temporary EntityArrow has `arrowShake = 0`; the negative partial
                // tick therefore takes RenderArrow's no-wobble branch.
                emit_arrow_from(
                    attachment.matrix,
                    attachment.yaw,
                    attachment.pitch,
                    -params.partial_ticks,
                    &mut arrows,
                );
            }
            // java: `LayerArrow`'s `disableStandardItemLighting()`: unlit, like the world's.
            draw_sorted_triangles_lit(
                d,
                &UiTexture::gui(ARROW_TEXTURE),
                UiBlend::Alpha,
                &arrows,
                false,
            );
        }

        // java: `LayerCape.doRenderLayer` requires `!entity.isInvisible()` in addition to a
        // loaded CAPE texture and the CAPE model-part setting.  The latter is already enforced
        // by the producer when it builds `snapshot.cape`; keep the local invisibility gate here
        // because it is the only one available in this UI-owned state boundary.
        if !params.entity.invisible {
            if let (Some(cape), Some(cape_texture)) =
                (snapshot.cape.as_ref(), snapshot.cape_texture.as_ref())
            {
                let mut cape_verts = Vec::new();
                self.manager
                    .emit_cape(renderer.model, cape, base, &mut cape_verts);
                draw_sorted_triangles_with_blend(d, cape_texture, UiBlend::Alpha, &cape_verts);
            }
        }

        // java: `LayerCustomHead` is the final `RenderPlayer` layer.  Keep the atlas-backed
        // ItemBlock branch here (pumpkins, blocks, and ordinary resource-pack block items).
        // Player skulls share the item renderer's asynchronous texture upload descriptor.
        if let Some(stack) = snapshot.equipment[4].as_ref() {
            draw_custom_head(
                &mut self.manager,
                d,
                renderer.model,
                base,
                &params,
                stack,
                snapshot.helmet_skull_owner.as_ref(),
                atlas,
            );
        }
    }

    /// Draw a horse preview with watched state, animation, layered texture selection and armour.
    ///
    /// java: `GuiScreenHorseInventory` passes its retained `EntityHorse` to the same
    /// `drawEntityOnScreen` helper. The snapshot's id is deliberately part of the API so the
    /// caller can reject a state from a recycled network id before it reaches this method.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_horse(
        &mut self,
        d: &mut DrawList,
        pos_x: i32,
        pos_y: i32,
        scale: i32,
        mouse_x: f32,
        mouse_y: f32,
        age_in_ticks: f32,
        snapshot: &HorsePreviewSnapshot,
    ) {
        let look = look_angles(mouse_x, mouse_y);
        let outer = outer_transform(pos_x, pos_y, scale, look.pitch_gl);
        let transform = EntityTransform {
            rel: Vec3::ZERO,
            body_yaw: look.body_yaw,
            death_rotation: 0.0,
            post_yaw: 0.0,
            scale: 1.0,
            y_offset: MODEL_Y_OFFSET,
            upside_down: false,
            upside_down_height: 0.0,
        };
        let mut params = snapshot.params.clone();
        params.age_in_ticks = age_in_ticks;
        params.net_head_yaw = look.net_head_yaw;
        params.head_pitch = look.head_pitch;
        let mut verts = Vec::new();
        let _selection = self.manager.emit_horse_with_state(
            &transform,
            &params,
            &snapshot.state,
            snapshot.packed_variant,
            snapshot.armor,
            &mut verts,
        );
        // `emit_horse_with_state` owns descriptor scaling and placement. Apply the GUI's
        // orthographic pixel-space bracket to its output before sorting into DrawList. Its
        // normals were emitted under the entity matrix only, while player/layer emission receives
        // `outer` as part of `base`; replay the GUI normal matrix here so both preview paths feed
        // the same eye-space normal to the frozen-light shader equivalent below.
        let outer_normal = outer.inverse().transpose();
        for vertex in &mut verts {
            let p = outer.transform_point3(Vec3::from_array(vertex.pos));
            vertex.pos = p.to_array();
            vertex.normal = outer_normal
                .transform_vector3(Vec3::from_array(vertex.normal))
                .normalize_or_zero()
                .to_array();
        }
        draw_sorted_triangles(d, &snapshot.texture, &verts);
    }
}

/// The average `z` of a triangle's three (already fully-transformed) vertices — the sort key for
/// [`draw_sorted_triangles`]'s painter's algorithm.
/// Draw `RenderPlayer`'s helmet-slot `LayerCustomHead` for the subset whose texture source is
/// available to the UI phase.
///
/// java: `LayerCustomHead` first handles an `ItemBlock` with the HEAD display transform, then
/// `Items.skull` through `TileEntitySkullRenderer`. Player skulls use their own profile's cached
/// texture or UUID-derived fallback. Fixed skeleton/zombie/wither/creeper sheets stay unchanged.
#[allow(clippy::too_many_arguments)] // mirrors LayerCustomHead's explicit Java call inputs
fn draw_custom_head(
    manager: &mut RenderManager,
    d: &mut DrawList,
    model_kind: ModelKind,
    base: Mat4,
    params: &AnimParams,
    stack: &ItemStack,
    skull_owner: Option<&BuiltinSkullOwner>,
    atlas: &UiTexture,
) {
    let Ok(id) = u16::try_from(stack.item_id) else {
        return;
    };
    let item = mc_data::ItemId(id);
    let props = mc_data::item_props(item);
    let (kind, skull_variant) = if props.block.is_some() {
        (CustomHeadItem::Block, None)
    } else if props.name == "minecraft:skull" {
        (
            CustomHeadItem::Skull,
            Some(mc_render::entity::tile::SkullVariant::from_id(
                u8::try_from(stack.damage.max(0)).unwrap_or_default(),
            )),
        )
    } else {
        return;
    };

    // Keep the mutable manager borrow scoped to the head lookup.  Geometry emission below needs
    // to borrow the same manager again for a skull model.
    let matrix = {
        let model = manager.model_mut(model_kind);
        let Some(head) = model.model().find("head") else {
            return;
        };
        custom_head_matrix(
            base,
            model.model(),
            head,
            params.scale,
            params.entity.sneaking,
            params.entity.child,
            false, // player previews are not villager-like
            kind,
        )
    };

    if let Some(variant) = skull_variant {
        let mut skull = Vec::new();
        // java: `renderSkull(-0.5, 0, -0.5, UP, 180, ...)` poses the skull with its own fixed
        // yaw and flip, not the player's head angles.
        manager.emit_custom_head_skull(variant.model(), matrix, &mut skull);
        if skull.is_empty() {
            return;
        }
        let texture = match variant.texture() {
            Some(path) => UiTexture::gui(path),
            None => match skull_owner {
                Some(owner) => {
                    let texture = UiTexture::gui(&owner.texture_key());
                    // UiRenderer prepares these descriptors before submitting the draw list,
                    // uploading shared cached pixels through the existing item-icon path.
                    d.register_builtin_tile_item(BuiltinTileItem::PlayerSkull {
                        skull_type: 3,
                        owner: owner.clone(),
                    });
                    texture
                }
                None => UiTexture::gui(SkinModel::Classic.default_texture()),
            },
        };
        draw_sorted_triangles_with_blend(d, &texture, UiBlend::Alpha, &skull);
        return;
    }

    let model = mc_assets::item_model_override_for(item, stack.damage).model;
    if model.is_empty() {
        // Built-in/entity block models need their tile renderer, which this UI snapshot cannot
        // represent without widening the owned draw contract.
        return;
    }
    let matrix = matrix
        * mc_render::entity::objects::item_renderer_model_transform(
            model.gui_3d,
            &model.transforms.head,
        );
    let mut head = Vec::new();
    emit_baked_model_tinted(model, matrix, &mut head, |tint| {
        super::render::item_quad_color_rgb(stack, Some(tint))
    });
    draw_sorted_triangles_with_blend(d, atlas, UiBlend::Alpha, &head);
    if super::tooltip::is_item_enchanted(stack) {
        let mut glint = Vec::new();
        emit_baked_model_glint(model, matrix, item_glint_system_time_ms(), &mut glint);
        draw_sorted_triangles_with_blend(
            d,
            &UiTexture::gui(ENCHANTED_ITEM_GLINT_TEXTURE),
            UiBlend::Glint,
            &glint,
        );
    }
}

#[must_use]
fn avg_z(tri: &[EntityVertex; 3]) -> f32 {
    (tri[0].pos[2] + tri[1].pos[2] + tri[2].pos[2]) / 3.0
}

/// Depth-sorts `verts` (a flat triangle list, three vertices per triangle — 432 of them for a
/// full `ModelKind::Player`, see `a_neutral_pose_produces_a_full_players_worth_of_triangles`
/// below) back-to-front and pushes each as a lit [`DrawList::triangle`].
///
/// Vanilla gets correct part overlap (an arm in front of the torso, say) for free from a real
/// depth buffer under its orthographic GUI projection — `GL11.GL_DEPTH_TEST` stays enabled the
/// whole frame. The sort remains a stable painter fallback for CPU inspection; production batches
/// also submit the retained per-vertex depth and cull-disabled state, so interpenetrating parts
/// and are suitable for the small look-angle range this panel ever rotates through.
/// (`atan(x/40)*20` degrees, well under 90° for any on-screen mouse position) — see
fn draw_sorted_triangles(d: &mut DrawList, texture: &UiTexture, verts: &[EntityVertex]) {
    draw_sorted_triangles_with_blend(d, texture, UiBlend::Alpha, verts);
}

fn draw_sorted_triangles_with_blend(
    d: &mut DrawList,
    texture: &UiTexture,
    blend: UiBlend,
    verts: &[EntityVertex],
) {
    draw_sorted_triangles_lit(d, texture, blend, verts, true);
}

/// [`draw_sorted_triangles_with_blend`] with the two item lights optional. `lit = false` keeps
/// each vertex's own colour: `LayerArrow` brackets its arrows with
/// `RenderHelper.disableStandardItemLighting()`, in this GUI preview as in the world.
fn draw_sorted_triangles_lit(
    d: &mut DrawList,
    texture: &UiTexture,
    blend: UiBlend,
    verts: &[EntityVertex],
    lit: bool,
) {
    let colour = |vertex: &EntityVertex| {
        if lit {
            shade(vertex.normal, vertex.color)
        } else {
            vertex.color.map(|channel| channel.clamp(0.0, 1.0))
        }
    };
    let mut tris: Vec<[EntityVertex; 3]> =
        verts.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    tris.sort_by(|a, b| avg_z(a).total_cmp(&avg_z(b)));
    let depth_mode = if blend == UiBlend::Glint {
        UiDepth::Equal
    } else {
        UiDepth::Item
    };
    // java: `RendererLivingEntity.doRender` calls `enableAlpha()` before the model and its
    // layers, so transparent skin/armour texels write no depth for a later layer or the
    // GL_EQUAL armour/item foil to land on.
    d.with_alpha_test(|d| {
        d.with_render_state(depth_mode, UiCull::None, |d| {
            for tri in &tris {
                let pos = [
                    [tri[0].pos[0], tri[0].pos[1]],
                    [tri[1].pos[0], tri[1].pos[1]],
                    [tri[2].pos[0], tri[2].pos[1]],
                ];
                let depth = [
                    preview_depth(tri[0].pos[2]),
                    preview_depth(tri[1].pos[2]),
                    preview_depth(tri[2].pos[2]),
                ];
                let uv = [tri[0].uv, tri[1].uv, tri[2].uv];
                let color = [colour(&tri[0]), colour(&tri[1]), colour(&tri[2])];
                d.triangle_with_blend_depth(texture, blend, depth, pos, uv, color);
            }
        });
    });
}

#[must_use]
fn shade(normal: [f32; 3], primary: [f32; 4]) -> [f32; 4] {
    let light = lambertian(Vec3::from_array(normal));
    [
        (light * primary[0]).clamp(0.0, 1.0),
        (light * primary[1]).clamp(0.0, 1.0),
        (light * primary[2]).clamp(0.0, 1.0),
        primary[3].clamp(0.0, 1.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(item_id: i16) -> ItemStack {
        ItemStack {
            item_id,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    #[test]
    fn a_centred_mouse_looks_perfectly_straight_ahead() {
        let look = look_angles(0.0, 0.0);
        assert_eq!(look.body_yaw, 0.0);
        assert_eq!(look.net_head_yaw, 0.0);
        assert_eq!(look.head_pitch, 0.0);
        assert_eq!(look.pitch_gl, 0.0);
    }

    #[test]
    fn the_radian_mislabelled_as_degrees_quirk_is_reproduced_verbatim() {
        // A naive (correct-looking) reading of `drawEntityOnScreen` would convert `Math.atan`'s
        // radians to degrees before using them as a rotate angle. Vanilla does not, and this pins
        // that the un-converted value is what actually lands in `LookAngles`.
        let mouse_x = 20.0; // mouseX / 40 == 0.5
        let expected_naive_degrees = (0.5f64.atan().to_degrees()) as f32 * 20.0;
        let expected_vanilla = (0.5f64.atan()) as f32 * 20.0;
        assert_ne!(expected_naive_degrees, expected_vanilla);
        let look = look_angles(mouse_x, 0.0);
        assert!(
            (look.body_yaw - expected_vanilla).abs() < 1e-5,
            "{} vs {expected_vanilla}",
            look.body_yaw
        );
    }

    #[test]
    fn net_head_yaw_is_rotation_yaw_minus_render_yaw_offset() {
        // java: `rotationYaw = atan(mouseX/40)*40`, `renderYawOffset = atan(mouseX/40)*20`, so
        // `netHeadYaw` (`rotationYawHead - renderYawOffset`, and `rotationYawHead ==
        // rotationYaw`) reduces to `atan(mouseX/40)*20` — numerically `renderYawOffset` again,
        // but derived independently here to pin the *reduction*, not just the coincidence.
        let look = look_angles(37.0, 0.0);
        assert!((look.net_head_yaw - look.body_yaw).abs() < 1e-6);
    }

    #[test]
    fn moving_the_mouse_right_turns_the_head_yaw_positive() {
        let centred = look_angles(0.0, 0.0);
        let right = look_angles(40.0, 0.0);
        assert!(right.body_yaw > centred.body_yaw);
    }

    #[test]
    fn moving_the_mouse_down_looks_up_not_down() {
        // java: `rotationPitch = -atan(mouseY/40)*20` — the minus sign is load-bearing.
        let down = look_angles(0.0, 40.0);
        assert!(down.head_pitch < 0.0, "{}", down.head_pitch);
    }

    #[test]
    fn local_inventory_player_keeps_java_sneak_render_height() {
        // GuiInventory/GuiContainerCreative pass EntityPlayerSP. RenderPlayer's Java guard
        // skips the remote-player -1/8 displacement while retaining the crouched ModelPlayer
        // pose.
        assert_eq!(render_player_y_offset(true, true), 0.0);
        assert_eq!(render_player_y_offset(true, false), 0.0);
    }

    #[test]
    fn non_local_sneaking_player_keeps_render_player_displacement_rule() {
        // This branch is not currently reached by the inventory screens, but pins the exact
        // Java condition for a future generic preview.
        assert_eq!(render_player_y_offset(false, true), -0.125);
        assert_eq!(render_player_y_offset(false, false), 0.0);
    }

    #[test]
    fn standard_item_light_is_never_below_ambient_or_above_both_diffuse_lights() {
        for n in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            let l = lambertian(n);
            assert!((0.4..=1.6).contains(&l), "{n:?} -> {l}");
        }
        // The GUI's negative-X scale followed by the 180-degree Z turn flips the model's
        // upward face to eye-space -Y. The captured standard-item lights point down in that
        // same eye frame, so -Y is the overbright fixed-pose direction.
        assert!(lambertian(Vec3::NEG_Y) > 1.0);
    }

    #[test]
    fn lambertian_treats_a_zero_normal_as_pure_ambient() {
        assert_eq!(lambertian(Vec3::ZERO), 0.4);
    }

    #[test]
    fn tinted_overbright_faces_clamp_the_lit_primary_colour_channel_by_channel() {
        let light = lambertian(Vec3::NEG_Y);
        assert!(
            light > 1.0,
            "eye-space upward standard-item light was {light}"
        );

        let primary = [0.2, 0.8, 1.0, 0.35];
        let actual = shade(Vec3::NEG_Y.to_array(), primary);
        let scalar_first = [
            primary[0] * light.min(1.0),
            primary[1] * light.min(1.0),
            primary[2] * light.min(1.0),
            primary[3],
        ];
        assert!((actual[0] - primary[0] * light).abs() < 1e-6);
        assert_eq!(actual[1], 1.0);
        assert_eq!(actual[2], 1.0);
        assert_eq!(actual[3], primary[3]);
        assert_ne!(actual, scalar_first);
    }

    #[test]
    fn clamp_order_matches_scalar_first_for_non_overbright_control_normals() {
        let primary = [0.2, 0.8, 1.0, 0.35];
        for normal in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z, Vec3::Y] {
            let light = lambertian(normal);
            assert!(light <= 1.0, "control normal {normal:?} had light {light}");
            let actual = shade(normal.to_array(), primary);
            let scalar_first = [
                primary[0] * light.min(1.0),
                primary[1] * light.min(1.0),
                primary[2] * light.min(1.0),
                primary[3],
            ];
            assert_eq!(actual, scalar_first, "control normal {normal:?}");
        }
    }

    #[test]
    fn frozen_preview_light_vectors_match_the_java_gui_bracket() {
        let actual0 = frozen_preview_light_direction(LIGHT0);
        let actual1 = frozen_preview_light_direction(LIGHT1);
        let expected0 = Vec3::new(-0.5144958, -0.808_452, 0.28583094);
        let expected1 = Vec3::new(0.5144957, -0.808_452_07, -0.28583094);
        assert!((actual0 - expected0).length() < 3.0e-6, "{actual0:?}");
        assert!((actual1 - expected1).length() < 3.0e-6, "{actual1:?}");
        assert!((actual0.length() - 1.0).abs() < 1.0e-6);
        assert!((actual1.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn fixed_player_pose_has_a_bracketed_per_vertex_lighting_color() {
        let mut renderer = PlayerPreviewRenderer::new();
        let mut player = DrawList::new();
        renderer.draw(
            &mut player,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        let color = player.vertices()[0].color;
        assert!((color[0] - 0.5714985).abs() < 2.0e-6, "{color:?}");
        assert_eq!(color[0], color[1]);
        assert_eq!(color[1], color[2]);
        assert_eq!(color[3], 1.0);
    }

    #[test]
    fn fixed_saddled_armored_chested_horse_has_a_bracketed_per_vertex_lighting_color() {
        let mut renderer = PlayerPreviewRenderer::new();
        let mut horse = HorsePreviewSnapshot::default();
        horse.state.saddled = true;
        horse.state.chested = true;
        horse.state.ridden = true;
        horse.armor = 3;
        let mut horse_draw = DrawList::new();
        renderer.draw_horse(&mut horse_draw, 100, 100, 17, 0.0, 0.0, 0.0, &horse);
        let color = horse_draw.vertices()[0].color;
        assert!((color[0] - 0.5715453).abs() < 2.0e-6, "{color:?}");
        assert_eq!(color[0], color[1]);
        assert_eq!(color[1], color[2]);
        assert_eq!(color[3], 1.0);
    }

    #[test]
    fn a_neutral_pose_produces_a_full_players_worth_of_triangles() {
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw(
            &mut d,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        // `ModelKind::Player` is 12 boxes (7 `BipedModel` parts — head, headwear, body, both
        // arms, both legs — plus 5 skin-overlay "wear" parts: jacket, sleeves, pant legs; see
        // `mc_render::entity::models::biped`'s own
        // `cape_and_deadmau5_are_not_drawn_by_the_model_itself` test), each a 6-face box of 6
        // vertices: `12 * 6 * 6 == 432` with every model part shown. `draw` has no settings and
        // passes `model_parts: 0`, and `RenderPlayer.setModelVisibilities` (`:81-86`) then hides
        // the headwear and the five wear boxes (`isWearing` is false for each part): 6 boxes.
        assert_eq!(d.vertices().len(), 6 * 6 * 6);

        let all_parts = PlayerPreviewSnapshot {
            model_parts: 0x7f,
            ..PlayerPreviewSnapshot::default()
        };
        let mut full = DrawList::new();
        renderer.draw_with_snapshot(
            &mut full,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &all_parts,
            &UiTexture::gui("textures/atlas/blocks.png"),
        );
        assert_eq!(full.vertices().len(), 432);
    }

    #[test]
    fn the_preview_is_anchored_at_the_given_screen_position() {
        // The model's pivot (roughly the hip/waist, given the outer translate) should land near
        // `(pos_x, pos_y)`, not off in some other corner of the screen — a coarse sanity check
        // that `outer_transform`'s translate actually reaches the vertices.
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw(
            &mut d,
            1000,
            2000,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        let xs: Vec<f32> = d.vertices().iter().map(|v| v.pos[0]).collect();
        let ys: Vec<f32> = d.vertices().iter().map(|v| v.pos[1]).collect();
        let min_x = xs.iter().cloned().fold(f32::MAX, f32::min);
        let max_x = xs.iter().cloned().fold(f32::MIN, f32::max);
        let min_y = ys.iter().cloned().fold(f32::MAX, f32::min);
        let max_y = ys.iter().cloned().fold(f32::MIN, f32::max);
        // A 1.8-block player at 30px/block is on the order of 50px tall/wide; generous bounds.
        assert!((800.0..1200.0).contains(&min_x), "{min_x}");
        assert!((800.0..1200.0).contains(&max_x), "{max_x}");
        assert!((1800.0..2200.0).contains(&min_y), "{min_y}");
        assert!((1800.0..2200.0).contains(&max_y), "{max_y}");
    }

    #[test]
    fn a_bigger_scale_argument_produces_a_bigger_on_screen_figure() {
        let mut small_d = DrawList::new();
        PlayerPreviewRenderer::new().draw(
            &mut small_d,
            0,
            0,
            10,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        let mut big_d = DrawList::new();
        PlayerPreviewRenderer::new().draw(
            &mut big_d,
            0,
            0,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        let span = |d: &DrawList| {
            let ys: Vec<f32> = d.vertices().iter().map(|v| v.pos[1]).collect();
            ys.iter().cloned().fold(f32::MIN, f32::max)
                - ys.iter().cloned().fold(f32::MAX, f32::min)
        };
        assert!(span(&big_d) > span(&small_d) * 2.0);
    }

    #[test]
    fn classic_and_slim_skins_use_their_own_default_textures() {
        let mut d = DrawList::new();
        PlayerPreviewRenderer::new().draw(
            &mut d,
            0,
            0,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        assert_eq!(d.batches().len(), 1);
        assert_eq!(
            d.batches()[0].texture,
            UiTexture::gui("textures/entity/steve.png")
        );

        let mut d = DrawList::new();
        PlayerPreviewRenderer::new().draw(&mut d, 0, 0, 30, 0.0, 0.0, SkinModel::Slim, 0.0, None);
        assert_eq!(
            d.batches()[0].texture,
            UiTexture::gui("textures/entity/alex.png")
        );
    }

    #[test]
    fn ready_generation_texture_overrides_the_model_default() {
        let custom = UiTexture::Resource(mc_assets::ResourceLocation::minecraft(
            "textures/entity/skin/player/profile-2.png",
        ));
        let mut d = DrawList::new();
        PlayerPreviewRenderer::new().draw(
            &mut d,
            0,
            0,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            Some(&custom),
        );
        assert_eq!(d.batches().len(), 1);
        assert_eq!(d.batches()[0].texture, custom);
    }

    /// The load-bearing occlusion check for [`draw_sorted_triangles`]'s sort direction: for the
    /// dead-centre look (facing the viewer, per vanilla's own screenshots), the head's front
    /// (south) face must end up **later** in the draw order than its back (north) face, so it is
    /// the one left visible. UV rects verified against this port's own head box —
    /// `BipedModel::with`'s `head` part, `.cube(-4, -8, -4, 8, 8, 8, ..)` at texture offset
    /// `(0, 0)` on a 64-wide sheet — not assumed from memory: `ModelBox::new`'s face table (java:
    /// `ModelBox`'s six `TexturedQuad` constructions) puts south (`+Z`, face index 5) at
    /// `u:8..16, v:8..16` and north (`-Z`, face index 4) at `u:24..32, v:8..16`; normalised over a
    /// 64px sheet that is `u:0.125..0.25` and `u:0.375..0.5` respectively, same `v` range.
    #[test]
    fn front_face_of_the_head_draws_over_the_back_face() {
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw(&mut d, 0, 0, 30, 0.0, 0.0, SkinModel::Classic, 0.0, None);

        let in_rect = |uv: [f32; 2], (u0, u1, v0, v1): (f32, f32, f32, f32)| {
            uv[0] >= u0 && uv[0] <= u1 && uv[1] >= v0 && uv[1] <= v1
        };
        let front_rect = (0.125, 0.25, 0.125, 0.25);
        let back_rect = (0.375, 0.5, 0.125, 0.25);

        let verts = d.vertices();
        let last_front = verts
            .iter()
            .enumerate()
            .filter(|(_, v)| in_rect(v.uv, front_rect))
            .map(|(i, _)| i)
            .max()
            .expect("head has a front face");
        let last_back = verts
            .iter()
            .enumerate()
            .filter(|(_, v)| in_rect(v.uv, back_rect))
            .map(|(i, _)| i)
            .max()
            .expect("head has a back face");
        assert!(
            last_front > last_back,
            "front face (last at {last_front}) must draw after the back face (last at \
             {last_back}) to end up on top"
        );
    }

    #[test]
    fn looking_far_enough_left_or_right_still_produces_a_full_vertex_count() {
        // A regression guard for the sort/emit path under a real (non-zero) rotation, not just
        // the trivial centred case every other test above uses.
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw(&mut d, 50, 50, 20, 35.0, -15.0, SkinModel::Slim, 12.5, None);
        // The bare (`model_parts: 0`) body: six boxes, as in the neutral pose above.
        assert_eq!(d.vertices().len(), 216);
    }

    #[test]
    fn bare_idle_snapshot_is_a_real_negative_control() {
        let snapshot = PlayerPreviewSnapshot::default();
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw_with_snapshot(
            &mut d,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &snapshot,
            &atlas,
        );
        // Six boxes: the default snapshot wears no model parts.
        assert_eq!(d.vertices().len(), 216);
        assert_eq!(d.batches().len(), 1);
    }

    #[test]
    fn armour_held_cape_and_live_pose_all_reach_the_preview() {
        let mut snapshot = PlayerPreviewSnapshot::default();
        snapshot.equipment[0] = Some(stack(276)); // diamond sword
        snapshot.equipment[4] = Some(stack(310)); // diamond helmet
        snapshot.cape = Some(CapeState::default());
        snapshot.cape_texture = Some(UiTexture::gui("textures/entity/cape/test.png"));
        snapshot.params.limb_swing = 5.0;
        snapshot.params.limb_swing_amount = 1.0;
        snapshot.params.entity.sneaking = true;

        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut dressed = DrawList::new();
        renderer.draw_with_snapshot(
            &mut dressed,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &snapshot,
            &atlas,
        );

        // The bare body is 216 vertices (no model parts); armour, sword and cape add to it.
        assert!(dressed.vertices().len() > 216);
        assert!(dressed.batches().iter().any(|batch| {
            batch.texture == UiTexture::gui("textures/models/armor/diamond_layer_1.png")
        }));
        assert!(dressed
            .batches()
            .iter()
            .any(|batch| { batch.texture == UiTexture::gui("textures/entity/cape/test.png") }));

        let mut idle = DrawList::new();
        renderer.draw(
            &mut idle,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
        );
        // Both draw the posed body first: the live pose must change it.
        assert_ne!(dressed.vertices()[..216], idle.vertices()[..216]);
    }

    #[test]
    fn model_parts_mask_is_applied_before_preview_body_and_layers() {
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let masked = PlayerPreviewSnapshot {
            model_parts: 0,
            ..PlayerPreviewSnapshot::default()
        };
        let mut masked_draw = DrawList::new();
        renderer.draw_with_snapshot(
            &mut masked_draw,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &masked,
            &atlas,
        );

        let mut visible = masked;
        visible.model_parts = 0x7f;
        let mut visible_draw = DrawList::new();
        renderer.draw_with_snapshot(
            &mut visible_draw,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &visible,
            &atlas,
        );

        // java: `RenderPlayer.setModelVisibilities` (`:81-86`) — mask 0 hides the headwear and
        // the five wear boxes (6 of the 12), the full 0x7f mask shows all 12.
        assert_eq!(masked_draw.vertices().len(), 216);
        assert_eq!(visible_draw.vertices().len(), 432);
        assert!(
            visible_draw.vertices().len() > masked_draw.vertices().len(),
            "enabled model-part overlays must add preview geometry"
        );
    }

    #[test]
    fn arrows_use_the_snapshot_count_and_entity_seed() {
        let mut snapshot = PlayerPreviewSnapshot::default();
        snapshot.params.entity.arrow_count = 3;
        snapshot.params.entity.entity_id = 47;
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw_with_snapshot(
            &mut d,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &snapshot,
            &atlas,
        );

        let arrow = d
            .batches()
            .iter()
            .find(|batch| batch.texture == UiTexture::gui(ARROW_TEXTURE))
            .expect("LayerArrow batch");
        assert_eq!(
            arrow.range.len(),
            3 * 36,
            "each RenderArrow emits six quads"
        );
        // java: `LayerArrow` turns the item lights off, so no face is shaded.
        assert!(
            d.vertices()[arrow.range.clone()]
                .iter()
                .all(|vertex| vertex.color == [1.0, 1.0, 1.0, 1.0]),
            "stuck arrows are drawn unlit"
        );
    }

    /// java: `RendererLivingEntity.renderModel` (`:246-278`) for the local player viewing itself
    /// while invisible: the body is skipped unless `EntityPlayer.isInvisibleToPlayer` keeps it,
    /// and then it is the `color(1, 1, 1, 0.15)` ghost. `LayerCape` (`!isInvisible()`) draws
    /// nothing either way, and `LayerBipedArmor` has no invisibility gate at all.
    #[test]
    fn invisible_snapshot_skips_body_and_cape_but_keeps_armour() {
        let mut snapshot = PlayerPreviewSnapshot::default();
        snapshot.params.entity.invisible = true;
        snapshot.cape = Some(CapeState::default());
        snapshot.cape_texture = Some(UiTexture::gui("textures/entity/cape/test.png"));
        snapshot.equipment[4] = Some(stack(310)); // diamond helmet
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let skin = UiTexture::gui("textures/entity/steve.png");
        let cape = UiTexture::gui("textures/entity/cape/test.png");
        let helmet = UiTexture::gui("textures/models/armor/diamond_layer_1.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut draw = |snapshot: &PlayerPreviewSnapshot| {
            let mut d = DrawList::new();
            renderer.draw_with_snapshot(
                &mut d,
                100,
                100,
                30,
                0.0,
                0.0,
                SkinModel::Classic,
                0.0,
                None,
                snapshot,
                &atlas,
            );
            d
        };
        let body_alphas = |d: &DrawList| -> Vec<f32> {
            d.batches()
                .iter()
                .filter(|batch| batch.texture == skin)
                .flat_map(|batch| d.vertices()[batch.range.clone()].iter().map(|v| v.color[3]))
                .collect()
        };

        let hidden = draw(&snapshot);
        assert!(
            body_alphas(&hidden).is_empty(),
            "no team sees it: the body is skipped"
        );
        assert!(!hidden.batches().iter().any(|batch| batch.texture == cape));
        assert!(
            hidden.batches().iter().any(|batch| batch.texture == helmet),
            "the armour layer still draws"
        );

        snapshot.sees_own_invisible_body = true;
        let ghost = draw(&snapshot);
        let alphas = body_alphas(&ghost);
        assert_eq!(alphas.len(), 216, "the ghost is the whole (bare) body");
        assert!(
            alphas.iter().all(|alpha| (alpha - 0.15).abs() < 1e-6),
            "{alphas:?}"
        );
        assert!(!ghost.batches().iter().any(|batch| batch.texture == cape));

        // Control: a visible player draws the ordinary opaque body and its cape.
        snapshot.params.entity.invisible = false;
        let visible = draw(&snapshot);
        let alphas = body_alphas(&visible);
        assert_eq!(alphas.len(), 216);
        assert!(alphas.iter().all(|alpha| *alpha == 1.0), "{alphas:?}");
        assert!(visible.batches().iter().any(|batch| batch.texture == cape));
    }

    #[test]
    fn fixed_skull_helmet_uses_the_custom_head_layer_texture() {
        let mut snapshot = PlayerPreviewSnapshot::default();
        snapshot.equipment[4] = Some(stack(397)); // skeleton skull
        snapshot.helmet_skull_owner = Some(BuiltinSkullOwner::new(
            "unused-player-owner",
            SkinModel::Slim,
            SkinModel::Slim,
        ));
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut d = DrawList::new();
        renderer.draw_with_snapshot(
            &mut d,
            100,
            100,
            30,
            0.0,
            0.0,
            SkinModel::Classic,
            0.0,
            None,
            &snapshot,
            &atlas,
        );

        // The bare body is 216 vertices (no model parts); the skull adds its own box.
        assert!(d.vertices().len() > 216);
        assert!(d.batches().iter().any(|batch| {
            batch.texture == UiTexture::gui("textures/entity/skeleton/skeleton.png")
        }));
        assert!(
            d.builtin_tile_items().is_empty(),
            "fixed skull ignores player ownership"
        );
    }

    #[test]
    fn worn_player_skull_uses_owner_cache_and_registers_its_texture_upload() {
        let mut snapshot = PlayerPreviewSnapshot::default();
        let mut helmet = stack(397);
        helmet.damage = 3;
        snapshot.equipment[4] = Some(helmet);
        let owner = BuiltinSkullOwner::new(
            "mc-render/entity/skin/inventory-worn-skull-owner-regression",
            SkinModel::Slim,
            SkinModel::Slim,
        );
        snapshot.helmet_skull_owner = Some(owner.clone());
        let wearer = UiTexture::gui("textures/entity/skin/inventory-wearer.png");
        let draw = |snapshot: &PlayerPreviewSnapshot| {
            let mut d = DrawList::new();
            PlayerPreviewRenderer::new().draw_with_snapshot(
                &mut d,
                100,
                100,
                30,
                0.0,
                0.0,
                SkinModel::Classic,
                0.0,
                Some(&wearer),
                snapshot,
                &UiTexture::gui("textures/atlas/blocks.png"),
            );
            d
        };
        let pending = draw(&snapshot);
        assert!(pending
            .batches()
            .iter()
            .any(|batch| batch.texture == wearer));
        assert!(
            pending.batches().iter().any(|batch| {
                batch.texture == UiTexture::gui(SkinModel::Slim.default_texture())
            }),
            "pending skull uses its owner's fallback, not the wearer's skin"
        );
        assert_eq!(
            pending.builtin_tile_items(),
            &[BuiltinTileItem::PlayerSkull {
                skull_type: 3,
                owner: owner.clone(),
            }]
        );

        mc_render::entity::skin::publish_shared_skin(
            &owner.cache_key,
            &mc_render::entity::SkinTexture {
                rgba: vec![255; 64 * 64 * 4].into(),
                layout: mc_render::entity::SkinLayout::Modern64x64,
                model: SkinModel::Slim,
            },
        );
        let ready = draw(&snapshot);
        assert!(
            ready
                .batches()
                .iter()
                .any(|batch| { batch.texture == UiTexture::gui(&owner.cache_key) }),
            "loaded owner texture reaches the worn skull batch"
        );
        assert_eq!(ready.builtin_tile_items(), pending.builtin_tile_items());
        assert_eq!(
            ready.vertices(),
            pending.vertices(),
            "texture readiness does not change head geometry"
        );

        snapshot.helmet_skull_owner = None;
        let ownerless = draw(&snapshot);
        assert!(
            ownerless.batches().iter().any(|batch| {
                batch.texture == UiTexture::gui(SkinModel::Classic.default_texture())
            }),
            "an ownerless player skull uses Steve rather than the wearer"
        );
        assert!(ownerless.builtin_tile_items().is_empty());

        snapshot.equipment[4] = None;
        snapshot.helmet_skull_owner = Some(owner);
        let unequipped = draw(&snapshot);
        assert!(unequipped.builtin_tile_items().is_empty());
        assert!(unequipped.vertices().len() < ready.vertices().len());
    }

    #[test]
    fn horse_snapshot_reaches_saddle_chest_reins_and_armour_layers() {
        let mut horse = HorsePreviewSnapshot::default();
        horse.state.saddled = true;
        horse.state.chested = true;
        horse.state.ridden = true;
        horse.armor = 3;
        horse.texture = UiTexture::gui("textures/entity/horse/layered-test.png");
        let mut renderer = PlayerPreviewRenderer::new();
        let mut dressed = DrawList::new();
        renderer.draw_horse(&mut dressed, 100, 100, 17, 0.0, 0.0, 0.0, &horse);
        assert!(!dressed.vertices().is_empty());

        let mut bare = DrawList::new();
        renderer.draw_horse(
            &mut bare,
            100,
            100,
            17,
            0.0,
            0.0,
            0.0,
            &HorsePreviewSnapshot::default(),
        );
        assert_ne!(dressed.vertices(), bare.vertices());
    }
}
