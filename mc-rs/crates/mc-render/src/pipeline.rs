//! The render-phase seam.
//!
//! Vanilla's renderer is one enormous ordered sequence of GL state changes spread across
//! `RenderGlobal`, `EntityRenderer`, and `EffectRenderer`. Invariant I2 says we rewrite that
//! rather than transliterate it, so the ordering — the part that actually matters — is made
//! explicit here as a list of [`RenderPhase`] objects sorted by [`PhaseOrder`], and the GL state
//! machine is replaced by each phase owning its own `wgpu::RenderPipeline`.
//!
//! # Implementing a phase
//!
//! ```no_run
//! use mc_render::{Camera, DepthAttachment, Gpu, PhaseOrder, PhaseList, RenderPhase};
//!
//! struct TerrainPhase { /* buffers, pipeline, bind groups */ }
//!
//! impl RenderPhase for TerrainPhase {
//!     fn name(&self) -> &'static str { "terrain/solid" }
//!
//!     fn record(
//!         &mut self,
//!         gpu: &Gpu,
//!         enc: &mut wgpu::CommandEncoder,
//!         view: &wgpu::TextureView,
//!         cam: &Camera,
//!     ) {
//!         let mut pass = gpu.begin_pass(enc, view, "terrain/solid", DepthAttachment::Shared);
//!         pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
//!         // ... set_pipeline / set_vertex_buffer / draw, in camera-relative space ...
//!         let _ = cam;
//!     }
//! }
//!
//! # fn wire(phases: &mut PhaseList) {
//! phases.insert(PhaseOrder::TERRAIN_SOLID, Box::new(TerrainPhase {}));
//! # }
//! ```
//!
//! # What a phase is handed, and what it may assume
//!
//! * `gpu` — device, queue, swap-chain config, the shared depth buffer, the MSAA sample count,
//!   and `@group(0)`'s camera layout and bind group. Build pipelines from
//!   [`Gpu::camera_layout`](crate::Gpu::camera_layout) and declare your own layouts from
//!   `@group(1)` upwards.
//! * `enc` — the frame's single command encoder. Open as many render passes as you need with
//!   [`Gpu::begin_pass`](crate::Gpu::begin_pass); do not call `queue.submit`, the frame loop owns
//!   submission.
//! * `view` — the frame's colour target. Under MSAA this is the resolve target; `begin_pass`
//!   redirects to the multisampled attachment for you.
//! * `cam` — this frame's [`Camera`], already interpolated with `partial_ticks` and updated.
//!
//! **Colour and depth are already cleared** when the first phase records, so `begin_pass` loads
//! rather than clears. Phases run in [`PhaseOrder`] order and see each other's output.
//!
//! # Reading the frame back: scene input
//!
//! A phase that must *sample* what earlier phases drew — screen-space fog, a post-processing
//! chain — cannot do it through `view`, which is a bare `wgpu::TextureView`. Such a phase
//! overrides the two **provided** methods [`RenderPhase::needs_scene_color`] and
//! [`RenderPhase::record_scene`] instead, and is handed a [`SceneTextures`] holding a
//! single-sampled snapshot of the frame so far. See [`SceneTextures`] for the full contract and
//! for what [`PhaseList::render`] does differently once any phase opts in.
//!
//! Depth is *not* part of [`SceneTextures`]: it is always reachable through
//! [`Gpu::depth_texture`](crate::Gpu::depth_texture), needs no snapshot (a phase that reads it
//! opens its pass with [`DepthAttachment::None`](crate::DepthAttachment::None) and therefore
//! never reads and writes it at once), and is available to every phase, opted in or not.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use glam::{Mat4, Vec3};
use mc_assets::AtlasAnimationUpload;
use mc_assets::{Atlas, BakedModel};
use mc_math::{abs_f32, cos, sin, sqrt_f};
use wgpu::util::DeviceExt;

use crate::entity::anim::{AnimParams, EntityModel};
use crate::entity::item::{
    emit_baked_model_glint, emit_baked_model_tinted, item_glint_system_time_ms,
};
use crate::entity::models::biped::PlayerModel;
use crate::entity::objects::item_transform_matrix;
use crate::entity::tilepass::{christmas_today, BuiltinTileItem, TileModels};
use crate::entity::{
    missing_texture_pixels, EntityVertex, ENTITY_VERTEX_LAYOUT, MISSING_TEXTURE_SIZE,
};
use crate::{Camera, DepthAttachment, Gpu, DEPTH_FORMAT};

/// The main menu's panorama blur (S16→S10): a self-referential accumulator following
/// [`SceneTextures`]'s pattern at the scale the panorama actually needs, rather than through this
/// module's own (window-sized, per-frame) `SceneTargets` type. See the [`panorama_blur`] module
/// docs for why the type doesn't fit directly and the pattern does.
pub mod panorama_blur;
pub use panorama_blur::PanoramaBlur;
// "1.7 Animations": the 1.7 held-item maths.
pub mod old_animations;
// "1.7 Animations": 1.7's held-item edges for older packs.
pub mod old_item_edges;

/// Sort key for the phase list. Lower values record first.
///
/// The constants below reserve the vanilla ordering; sections should use them rather than raw
/// numbers, and use `PhaseOrder(TERRAIN_SOLID.0 + 1)` style offsets for sub-phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct PhaseOrder(pub u32);

impl PhaseOrder {
    /// Sky dome, sun, moon, stars, clouds. S20.
    pub const SKY: Self = Self(100);
    /// Opaque terrain. S11.
    pub const TERRAIN_SOLID: Self = Self(200);
    /// Alpha-tested terrain (leaves, grass, glass panes). S11.
    pub const TERRAIN_CUTOUT: Self = Self(250);
    /// Entities and tile entities. S12.
    pub const ENTITIES: Self = Self(300);
    /// Screen-space ambient occlusion over the opaque world.
    /// Reserved; the phase exists only while the mod is on.
    pub const AMBIENT_OCCLUSION: Self = Self(340);
    /// Block-breaking overlay and the selection outline. S11.
    pub const BLOCK_OVERLAY: Self = Self(350);
    /// Volumetric clouds in game, depth-tested against the opaque world. Reserved; the lane may move it and must say why.
    pub const VOLUMETRIC_CLOUDS: Self = Self(360);
    /// Particles. S18.
    pub const PARTICLES: Self = Self(400);
    /// Rain, snow, lightning and the world border. S20.
    ///
    /// java: `EntityRenderer.renderWorldPass` (`:1451-1467`) draws `renderRainSnow` and
    /// `renderWorldBorder` after the particles and *before* the translucent block layer, so water
    /// blends over rain and the border, not the other way round. This used to be 600, after
    /// translucent terrain. Neither pass writes depth, so moving it costs nothing.
    pub const WEATHER: Self = Self(450);
    /// Translucent terrain, sorted back to front. Must come after everything opaque. S11.
    pub const TERRAIN_TRANSLUCENT: Self = Self(500);
    /// Anti-aliasing of the world (velocity, then TAA or FXAA), before the waypoint labels (645),
    /// the hand (650) and the HUD.
    ///
    /// 643, not the 600 first reserved: the screen-space fog (`FogPhase::ORDER` = 640) and the
    /// late volumetric clouds (`sky::volumetric::LATE_ORDER` = 642) read the depth
    /// buffer, which is jittered while TAA runs, so they must be resolved by TAA rather than drawn
    /// over the resolved frame (that flickered at every silhouette). [`PhaseList`] uploads the
    /// unjittered camera again before the first phase at or after this order; see
    /// [`Camera::jitter`]. Shader packs:
    pub const TEMPORAL_AA: Self = Self(643);
    /// Per-pixel motion blur (the `motion_blur` mod's Camera and Camera + objects modes), after
    /// the anti-aliasing and before the waypoint labels and the hand, which move with the camera
    /// and stay sharp. The Frame blend mode still runs in `post/motion-grade` at
    /// [`PhaseOrder::POST`].
    pub const VELOCITY_BLUR: Self = Self(644);
    /// The local player's first-person hand and held item. Java's `renderHand` clears depth and
    /// records this after the world/weather passes, but before screen-space overlays. S20.
    pub const FIRST_PERSON: Self = Self(650);
    /// Screen-space post effects and overlays (underwater, portal, gamma). S20.
    pub const POST: Self = Self(700);
    /// Menu blur behind open screens ([`crate::post::blur::MenuBlurPhase`]). It records *between* the two halves of the UI
    /// draw list: after the HUD half ([`Self::UI`]) and before the open screen's half
    /// ([`Self::UI_SCREEN`]), so the hotbar, chat and scoreboard can blur with the world while the
    /// screen stays sharp. It first sat at 790 (before the UI) and moved here because blurring the
    /// HUD needs the HUD's pixels.
    pub const MENU_BLUR: Self = Self(801);
    /// HUD, screens, chat, F3. S15/S16/S17.
    pub const UI: Self = Self(800);
    /// The open screen's half of the UI draw list while the menu blur splits it
    /// (`mc_ui::widget::UiPhase::set_split`). Without a split the UI phase draws the
    /// whole list at [`Self::UI`] and the phase here is disabled.
    pub const UI_SCREEN: Self = Self(802);
}

/// The render-time visibility facts that gate `EntityRenderer.renderHand` and its arm path.
///
/// java: `EntityRenderer.renderHand` (`thirdPersonView == 0 && !sleeping && !hideGUI &&
/// !isSpectator`) and `ItemRenderer.renderItemInFirstPerson` (`!isInvisible()` for the empty
/// arm). Keeping the two gates together prevents a caller from accidentally drawing an empty
/// arm while hiding the GUI or in spectator mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FirstPersonVisibility {
    /// `GameSettings.thirdPersonView`; zero is first person.
    pub third_person_view: u8,
    /// Whether the render-view entity is sleeping.
    pub sleeping: bool,
    /// `GameSettings.hideGUI`.
    pub hide_gui: bool,
    /// Whether the player controller is in spectator mode.
    pub spectator: bool,
    /// `AbstractClientPlayer.isInvisible()`. This suppresses only the empty arm; held items still
    /// follow Java's item branch when the player itself is invisible.
    pub invisible: bool,
}

impl FirstPersonVisibility {
    /// Whether the first-person hand/item pass may run this frame.
    #[must_use]
    pub fn renders_hand(self) -> bool {
        self.third_person_view == 0 && !self.sleeping && !self.hide_gui && !self.spectator
    }

    /// Whether Java's empty right-arm branch may run this frame.
    #[must_use]
    pub fn renders_empty_arm(self) -> bool {
        self.renders_hand() && !self.invisible
    }
}

/// The four item-use branches in `ItemRenderer.renderItemInFirstPerson`, plus the ordinary
/// no-use swing path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FirstPersonUseAction {
    /// No item is being used; apply `doItemUsedTransformations(swing)` and the swing pose.
    #[default]
    Idle,
    /// `EnumAction.NONE`: suppress the swing while the item is in use.
    None,
    /// `EnumAction.EAT` or `EnumAction.DRINK`.
    EatDrink,
    /// `EnumAction.BLOCK`.
    Block,
    /// `EnumAction.BOW`.
    Bow,
}

/// CPU-side first-person state. It intentionally carries opaque stack identity rather than an
/// item registry type: `update_equipped_item` only needs Java's `getIsItemStackEqual` result, and
/// the caller can supply a stable key covering item/count/damage/NBT without making this renderer
/// depend on client inventory ownership.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FirstPersonRenderState {
    /// Previous tick's `equippedProgress`.
    pub prev_equipped_progress: f32,
    /// Current `equippedProgress`, advanced by at most ±0.4 per update.
    pub equipped_progress: f32,
    /// Opaque identity of Java's `itemToRender`, or `None` for an empty hand.
    pub item_to_render: Option<u64>,
    /// Java's latched `equippedItemSlot` (`-1` before the first update).
    pub equipped_item_slot: i32,
    /// Java's `isSwingInProgress`.
    pub is_swing_in_progress: bool,
    /// Java's integer swing counter.
    pub swing_progress_int: i32,
    /// Current fractional swing progress.
    pub swing_progress: f32,
    /// Previous tick's fractional swing progress.
    pub prev_swing_progress: f32,
    /// Visibility gates consumed by the first-person phase.
    pub visibility: FirstPersonVisibility,
    /// Item-use branch selected by the caller.
    pub use_action: FirstPersonUseAction,
    /// `EntityLivingBase.getItemInUseCount()` for the local player.
    pub item_use_count: i32,
    /// `ItemStack.getMaxItemUseDuration()` for the item in use.
    pub max_item_use_duration: i32,
}

impl Default for FirstPersonRenderState {
    fn default() -> Self {
        Self {
            prev_equipped_progress: 0.0,
            equipped_progress: 0.0,
            item_to_render: None,
            equipped_item_slot: -1,
            is_swing_in_progress: false,
            swing_progress_int: 0,
            swing_progress: 0.0,
            prev_swing_progress: 0.0,
            visibility: FirstPersonVisibility::default(),
            use_action: FirstPersonUseAction::Idle,
            item_use_count: 0,
            max_item_use_duration: 32,
        }
    }
}

/// The animation length used by an ordinary living entity with no haste/fatigue effect.
pub const FIRST_PERSON_DEFAULT_SWING_TICKS: i32 = 6;
/// The maximum per-update equip-progress delta. java: `clamp_float(..., -0.4F, 0.4F)`.
pub const FIRST_PERSON_EQUIP_DELTA: f32 = 0.4;
/// The equip-progress threshold below which Java swaps `itemToRender`.
pub const FIRST_PERSON_EQUIP_SWAP_THRESHOLD: f32 = 0.1;
/// The item model scale in `ItemRenderer.transformFirstPersonItem`.
pub const FIRST_PERSON_ITEM_SCALE: f32 = 0.4;
/// The near plane in `EntityRenderer.renderHand`.
pub const FIRST_PERSON_NEAR: f32 = 0.05;

impl FirstPersonRenderState {
    /// Port `ItemRenderer.updateEquippedItem` and return whether the latched stack changed this
    /// update. The identity key is the caller's exact `getIsItemStackEqual` equivalence class.
    pub fn update_equipped_item(&mut self, current: Option<u64>, current_slot: i32) -> bool {
        self.update_equipped_item_matching(self.item_to_render == current, current, current_slot)
    }

    /// [`Self::update_equipped_item`] with Java's comparison made by the caller:
    /// `matches` is `itemToRender.getIsItemStackEqual(current)` with both-null counting as equal.
    /// Java latches an `ItemStack` *object* and compares it by value, so a stack mutated in place
    /// keeps matching, and an equal-valued replacement matches without being latched.
    pub fn update_equipped_item_matching(
        &mut self,
        matches: bool,
        current: Option<u64>,
        current_slot: i32,
    ) -> bool {
        self.prev_equipped_progress = self.equipped_progress;
        let different = !matches;
        let target = if different { 0.0 } else { 1.0 };
        self.equipped_progress += (target - self.equipped_progress)
            .clamp(-FIRST_PERSON_EQUIP_DELTA, FIRST_PERSON_EQUIP_DELTA);

        let old = self.item_to_render;
        if self.equipped_progress < FIRST_PERSON_EQUIP_SWAP_THRESHOLD {
            self.item_to_render = current;
            self.equipped_item_slot = current_slot;
        }
        old != self.item_to_render
    }

    /// Java's `resetEquippedProgress` entry point.
    pub fn reset_equipped_progress(&mut self) {
        self.equipped_progress = 0.0;
    }

    /// Java's byte-identical `resetEquippedProgress2` entry point.
    pub fn reset_equipped_progress2(&mut self) {
        self.equipped_progress = 0.0;
    }

    /// Interpolate equip progress for a render frame, returning Java's `f = 1 - progress`.
    #[must_use]
    pub fn equip_amount(&self, partial_ticks: f32) -> f32 {
        1.0 - (self.prev_equipped_progress
            + (self.equipped_progress - self.prev_equipped_progress) * partial_ticks)
    }

    /// Start or restart a swing under Java's half-animation guard. Returns `true` when the
    /// counter was accepted, which is useful for keeping a local animation and packet send on
    /// the same decision without making this renderer own networking.
    pub fn swing_item(&mut self, animation_end: i32) -> bool {
        let end = animation_end.max(1);
        if !self.is_swing_in_progress
            || self.swing_progress_int >= end / 2
            || self.swing_progress_int < 0
        {
            self.swing_progress_int = -1;
            self.is_swing_in_progress = true;
            true
        } else {
            false
        }
    }

    /// Advance the local swing exactly once, including Java's `prevSwingProgress` latch.
    pub fn tick_swing(&mut self, animation_end: i32) {
        let end = animation_end.max(1);
        self.prev_swing_progress = self.swing_progress;
        if self.is_swing_in_progress {
            self.swing_progress_int += 1;
            if self.swing_progress_int >= end {
                self.swing_progress_int = 0;
                self.is_swing_in_progress = false;
            }
        } else {
            self.swing_progress_int = 0;
        }
        self.swing_progress = self.swing_progress_int as f32 / end as f32;
    }

    /// Render-interpolate the swing with Java's wrap at the end of an animation.
    #[must_use]
    pub fn swing_amount(&self, partial_ticks: f32) -> f32 {
        let mut delta = self.swing_progress - self.prev_swing_progress;
        if delta < 0.0 {
            delta += 1.0;
        }
        self.prev_swing_progress + delta * partial_ticks
    }

    /// Whether this state permits the first-person phase at all.
    #[must_use]
    pub fn renders_hand(&self) -> bool {
        self.visibility.renders_hand()
    }
}

/// Return Java's `getArmSwingAnimationEnd`, with the two potion amplifiers supplied by the
/// caller. Haste wins when both options are present, matching the source's first branch.
#[must_use]
pub fn arm_swing_animation_end(
    dig_speed_amplifier: Option<i32>,
    dig_slowdown_amplifier: Option<i32>,
) -> i32 {
    if let Some(amplifier) = dig_speed_amplifier {
        FIRST_PERSON_DEFAULT_SWING_TICKS - (1 + amplifier)
    } else if let Some(amplifier) = dig_slowdown_amplifier {
        FIRST_PERSON_DEFAULT_SWING_TICKS + (1 + amplifier) * 2
    } else {
        FIRST_PERSON_DEFAULT_SWING_TICKS
    }
}

fn first_person_radians(degrees: f32) -> f32 {
    degrees * (core::f32::consts::PI / 180.0)
}

/// The camera projection Java rebuilds for `renderHand`: a `0.05` near plane, twice the world
/// far plane, and its own field of view, [`Camera::hand_fov`].
///
/// java: `gluPerspective(getFOVModifier(partialTicks, false), ...)`. With `useFOVSetting` false
/// the base is a fixed 70 degrees: the FOV slider and the sprint/Speed/bow modifier never reach
/// the hand, so it does not shrink at a high FOV or while sprinting. The world FOV
/// ([`Camera::effective_fov`]) must not be used here.
#[must_use]
pub fn first_person_projection(cam: &Camera) -> Mat4 {
    glam::camera::rh::proj::directx::perspective(
        cam.hand_fov.clamp(1.0, 179.0).to_radians(),
        cam.aspect.max(f32::EPSILON),
        FIRST_PERSON_NEAR,
        (cam.far * 2.0).max(FIRST_PERSON_NEAR + f32::EPSILON),
    )
}

/// `ItemRenderer.rotateWithPlayerRotations`, kept as a matrix even though the vanilla
/// `rotateArroundXAndY` helper has a push/pop and therefore no net transform.
#[must_use]
pub fn first_person_view_rotation(
    player_pitch: f32,
    player_yaw: f32,
    prev_render_arm_pitch: f32,
    render_arm_pitch: f32,
    prev_render_arm_yaw: f32,
    render_arm_yaw: f32,
    partial_ticks: f32,
) -> Mat4 {
    let arm_pitch =
        prev_render_arm_pitch + (render_arm_pitch - prev_render_arm_pitch) * partial_ticks;
    let arm_yaw = prev_render_arm_yaw + (render_arm_yaw - prev_render_arm_yaw) * partial_ticks;
    Mat4::from_rotation_x(first_person_radians((player_pitch - arm_pitch) * 0.1))
        * Mat4::from_rotation_y(first_person_radians((player_yaw - arm_yaw) * 0.1))
}

/// Compose the matrix Java installs before drawing the held item or empty arm.
///
/// `EntityRenderer.renderHand` applies the same view-bobbing transform used by the world
/// camera before `ItemRenderer.rotateWithPlayerRotations` adds the damped arm delta.
#[must_use]
pub fn first_person_camera_matrix(
    cam: &Camera,
    view_bobbing: Mat4,
    prev_render_arm_pitch: f32,
    render_arm_pitch: f32,
    prev_render_arm_yaw: f32,
    render_arm_yaw: f32,
    partial_ticks: f32,
) -> Mat4 {
    first_person_projection(cam)
        * view_bobbing
        * first_person_view_rotation(
            cam.pitch,
            cam.yaw,
            prev_render_arm_pitch,
            render_arm_pitch,
            prev_render_arm_yaw,
            render_arm_yaw,
            partial_ticks,
        )
}

/// `RenderHelper.LIGHT0_POS` and `LIGHT1_POS`, as `glLight` receives them.
///
/// java: `new Vec3(0.2F, 1.0, -0.7F).normalize()` and its mirror. `Vec3.normalize` divides the
/// doubles by `MathHelper.sqrt_double`, a *float* square root, and `setColorBuffer` narrows the
/// result to float.
#[must_use]
pub fn standard_item_light_directions() -> [Vec3; 2] {
    let light = |x: f64, y: f64, z: f64| {
        let length = f64::from((x * x + y * y + z * z).sqrt() as f32);
        Vec3::new(
            (x / length) as f32,
            (y / length) as f32,
            (z / length) as f32,
        )
    };
    [
        light(0.200_000_002_980_232_24, 1.0, -0.699_999_988_079_071),
        light(-0.200_000_002_980_232_24, 1.0, 0.699_999_988_079_071),
    ]
}

/// The two standard item lights in the frame the hand's vertices are emitted in.
///
/// java: `renderItemInFirstPerson` calls `rotateArroundXAndY(pitch, yaw)`, which places the
/// lights under `Rx(pitch) * Ry(yaw)` inside a push/pop, and then `rotateWithPlayerRotations`
/// turns the hand by the damped arm delta. `hurtCameraEffect` and `setupViewBobbing` precede
/// both and cancel out of every `N . L`. So each light, seen from the emitted vertices, is
/// `sway^T * Rx(pitch) * Ry(yaw) * L`: the hand's shading changes as the player turns.
#[must_use]
pub fn first_person_light_directions(
    player_pitch: f32,
    player_yaw: f32,
    prev_render_arm_pitch: f32,
    render_arm_pitch: f32,
    prev_render_arm_yaw: f32,
    render_arm_yaw: f32,
    partial_ticks: f32,
) -> [Vec3; 2] {
    let sway = first_person_view_rotation(
        player_pitch,
        player_yaw,
        prev_render_arm_pitch,
        render_arm_pitch,
        prev_render_arm_yaw,
        render_arm_yaw,
        partial_ticks,
    );
    // `sway` is a pure rotation, so its transpose is its inverse.
    let to_vertices = sway.transpose()
        * Mat4::from_rotation_x(first_person_radians(player_pitch))
        * Mat4::from_rotation_y(first_person_radians(player_yaw));
    standard_item_light_directions().map(|light| to_vertices.transform_vector3(light))
}

/// One hand vertex through fixed-function `GL_LIGHTING`, then the lightmap texture stage.
///
/// java: `enableStandardItemLighting` with `GL_COLOR_MATERIAL` (`AMBIENT_AND_DIFFUSE`): scene
/// ambient `0.4`, each light's diffuse `0.6`, no light ambient or specular. GL clamps the lit
/// primary colour to `[0, 1]`; texture unit 0 then multiplies the texel and the lightmap unit
/// (`setLightMapFromPlayer`) multiplies last. The texel multiply is the shader's, so this returns
/// the lit primary colour times the lightmap colour. Alpha stays the vertex alpha.
#[must_use]
pub fn first_person_lit_color(
    color: [f32; 4],
    normal: [f32; 3],
    lights: [Vec3; 2],
    lightmap: [f32; 3],
) -> [f32; 4] {
    let normal = Vec3::from(normal);
    let diffuse = 0.4 + 0.6 * normal.dot(lights[0]).max(0.0) + 0.6 * normal.dot(lights[1]).max(0.0);
    [
        (color[0] * diffuse).clamp(0.0, 1.0) * lightmap[0],
        (color[1] * diffuse).clamp(0.0, 1.0) * lightmap[1],
        (color[2] * diffuse).clamp(0.0, 1.0) * lightmap[2],
        color[3],
    ]
}

/// `ItemRenderer.doItemUsedTransformations`'s three translations.
#[must_use]
pub fn first_person_used_transform(swing: f32) -> Mat4 {
    let root = sqrt_f(swing) * core::f32::consts::PI;
    Mat4::from_translation(Vec3::new(
        -0.4 * sin(root),
        0.2 * sin(root * 2.0),
        -0.2 * sin(swing * core::f32::consts::PI),
    ))
}

/// `ItemRenderer.transformFirstPersonItem`, excluding the item model's `display.firstperson`
/// transform and the model recentering (the latter is performed by `emit_baked_model`).
#[must_use]
pub fn first_person_item_transform(equip: f32, swing: f32) -> Mat4 {
    let swing_sq = sin(swing * swing * core::f32::consts::PI);
    let swing_root = sin(sqrt_f(swing) * core::f32::consts::PI);
    Mat4::from_translation(Vec3::new(0.56, -0.52, -0.71999997))
        * Mat4::from_translation(Vec3::new(0.0, equip * -0.6, 0.0))
        * Mat4::from_rotation_y(first_person_radians(45.0))
        * Mat4::from_rotation_y(first_person_radians(swing_sq * -20.0))
        * Mat4::from_rotation_z(first_person_radians(swing_root * -20.0))
        * Mat4::from_rotation_x(first_person_radians(swing_root * -80.0))
        * Mat4::from_scale(Vec3::splat(FIRST_PERSON_ITEM_SCALE))
}

/// Alias named after Java's public-facing operation, useful to callers that only need the base
/// held-item pose.
#[must_use]
pub fn first_person_item_matrix(equip: f32, swing: f32) -> Mat4 {
    first_person_item_transform(equip, swing)
}

/// `ItemRenderer.performDrinking`, including its double-precision `Math.pow(..., 27)` step.
#[must_use]
pub fn first_person_drinking_transform(
    item_use_count: i32,
    max_item_use_duration: i32,
    partial_ticks: f32,
) -> Mat4 {
    let f = item_use_count as f32 - partial_ticks + 1.0;
    let f1 = f / max_item_use_duration.max(1) as f32;
    let mut f2 = abs_f32(cos(f / 4.0 * core::f32::consts::PI) * 0.1);
    if f1 >= 0.8 {
        f2 = 0.0;
    }
    let f3 = 1.0 - (f64::from(f1).powi(27) as f32);
    Mat4::from_translation(Vec3::new(0.0, f2, 0.0))
        * Mat4::from_translation(Vec3::new(f3 * 0.6, f3 * -0.5, 0.0))
        * Mat4::from_rotation_y(first_person_radians(f3 * 90.0))
        * Mat4::from_rotation_x(first_person_radians(f3 * 10.0))
        * Mat4::from_rotation_z(first_person_radians(f3 * 30.0))
}

/// `ItemRenderer.doBowTransformations`; the non-zero pull term is the Y translation at 0.01.
#[must_use]
pub fn first_person_bow_transform(
    item_use_count: i32,
    max_item_use_duration: i32,
    partial_ticks: f32,
) -> Mat4 {
    let f = max_item_use_duration.max(1) as f32 - (item_use_count as f32 - partial_ticks + 1.0);
    let mut f1 = f / 20.0;
    f1 = (f1 * f1 + f1 * 2.0) / 3.0;
    if f1 > 1.0 {
        f1 = 1.0;
    }
    let mut m = Mat4::from_rotation_z(first_person_radians(-18.0))
        * Mat4::from_rotation_y(first_person_radians(-12.0))
        * Mat4::from_rotation_x(first_person_radians(-8.0))
        * Mat4::from_translation(Vec3::new(-0.9, 0.2, 0.0));
    if f1 > 0.1 {
        let f2 = sin((f - 0.1) * 1.3);
        let f3 = f1 - 0.1;
        let f4 = f2 * f3;
        m *= Mat4::from_translation(Vec3::new(f4 * 0.0, f4 * 0.01, f4 * 0.0));
    }
    m * Mat4::from_translation(Vec3::new(0.0, 0.0, f1 * 0.1))
        * Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0 + f1 * 0.2))
}

/// `ItemRenderer.doBlockTransformations`.
#[must_use]
pub fn first_person_block_transform() -> Mat4 {
    Mat4::from_translation(Vec3::new(-0.5, 0.2, 0.0))
        * Mat4::from_rotation_y(first_person_radians(30.0))
        * Mat4::from_rotation_x(first_person_radians(-80.0))
        * Mat4::from_rotation_y(first_person_radians(60.0))
}

/// Compose the live item-use branch exactly as `renderItemInFirstPerson` does. `Idle` is the
/// ordinary swing path; the other variants are the branches taken while an item is in use.
#[must_use]
pub fn first_person_item_pose(
    equip: f32,
    swing: f32,
    action: FirstPersonUseAction,
    item_use_count: i32,
    max_item_use_duration: i32,
    partial_ticks: f32,
) -> Mat4 {
    match action {
        FirstPersonUseAction::Idle => {
            first_person_used_transform(swing) * first_person_item_transform(equip, swing)
        }
        FirstPersonUseAction::None => first_person_item_transform(equip, 0.0),
        FirstPersonUseAction::EatDrink => {
            first_person_drinking_transform(item_use_count, max_item_use_duration, partial_ticks)
                * first_person_item_transform(equip, 0.0)
        }
        FirstPersonUseAction::Block => {
            first_person_item_transform(equip, 0.0) * first_person_block_transform()
        }
        FirstPersonUseAction::Bow => {
            first_person_item_transform(equip, 0.0)
                * first_person_bow_transform(item_use_count, max_item_use_duration, partial_ticks)
        }
    }
}

/// Compose Java's `RenderItem.renderItemModelTransform` scale seam for a first-person model.
///
/// `ItemRenderer.renderItem` contributes `scale(2)` for GUI-3D models, while
/// `RenderItem.preTransform` contributes the same scale for flat models. Both branches then
/// enter `renderItem`, which contributes its unconditional `scale(0.5)` immediately before the
/// baked-model recenter performed by [`emit_baked_model`]. Keeping the two complementary outer
/// branches visible here matters for a non-identity display translation: whichever Java branch
/// is active scales that translation before `ItemCameraTransforms.applyTransform`.
#[must_use]
pub fn first_person_item_model_transform(
    gui_3d: bool,
    display: &mc_assets::model::ItemTransform,
) -> Mat4 {
    let outer_item_renderer = if gui_3d {
        Mat4::from_scale(Vec3::splat(2.0))
    } else {
        Mat4::IDENTITY
    };
    let pre_transform = if gui_3d {
        Mat4::IDENTITY
    } else {
        Mat4::from_scale(Vec3::splat(2.0))
    };
    outer_item_renderer
        * pre_transform
        * item_transform_matrix(display)
        * Mat4::from_scale(Vec3::splat(0.5))
}

/// Compose the model-side matrix for a first-person `builtin/entity` item.
///
/// java: `ItemRenderer.renderItem` scales GUI-3D models by `2` before calling
/// `RenderItem.renderItemModelForEntity`; `BuiltInModel.isGui3d()` is hard-coded `true`, so
/// `RenderItem.preTransform` contributes no second scale.  The model's `display.firstperson`
/// transform is then applied, followed by the builtin branch's common `0.5` scale and
/// `(-0.5, -0.5, -0.5)` recenter inside [`TileModels::emit_builtin_item`].  Keeping the outer
/// scale here (rather than folding it into the tile model) preserves custom-pack display
/// translations exactly: Java scales the translation before `ItemCameraTransforms.applyTransform`.
#[must_use]
pub fn first_person_builtin_item_transform(display: &mc_assets::model::ItemTransform) -> Mat4 {
    Mat4::from_scale(Vec3::splat(2.0)) * item_transform_matrix(display)
}

/// The complete right-arm matrix through `ItemRenderer.renderPlayerArm`, stopping immediately
/// before `ModelPlayer.renderRightArm`.
///
/// `ItemRenderer.renderRightArm` is deliberately absent here. That helper is the separate
/// `renderItemMap` arm path; the ordinary empty-arm path calls `RenderPlayer.renderRightArm`
/// directly after the `translate(5.6, 0, 0)` below.
#[must_use]
pub fn first_person_arm_matrix(equip: f32, swing: f32) -> Mat4 {
    let root = sqrt_f(swing) * core::f32::consts::PI;
    let swing_sq = sin(swing * swing * core::f32::consts::PI);
    let swing_root = sin(root);
    Mat4::from_translation(Vec3::new(
        -0.3 * swing_root,
        0.4 * sin(root * 2.0),
        -0.4 * sin(swing * core::f32::consts::PI),
    )) * Mat4::from_translation(Vec3::new(0.64000005, -0.6, -0.71999997))
        * Mat4::from_translation(Vec3::new(0.0, equip * -0.6, 0.0))
        * Mat4::from_rotation_y(first_person_radians(45.0))
        * Mat4::from_rotation_y(first_person_radians(swing_root * 70.0))
        * Mat4::from_rotation_z(first_person_radians(swing_sq * -20.0))
        * Mat4::from_translation(Vec3::new(-1.0, 3.6, 3.5))
        * Mat4::from_rotation_z(first_person_radians(120.0))
        * Mat4::from_rotation_x(first_person_radians(200.0))
        * Mat4::from_rotation_y(first_person_radians(-135.0))
        * Mat4::from_translation(Vec3::new(5.6, 0.0, 0.0))
}

/// One icon drawn on a held filled map.
///
/// The wire-side [`mc_world::MapDecoration`] also carries a Java map key, but the first-person
/// renderer only needs the four display fields. Keeping this small copy at the render seam lets
/// the driver hand an owned snapshot across the phase boundary without borrowing world state
/// during command recording.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FirstPersonMapDecoration {
    /// The 4-bit `MapDecoration.Type` sprite index.
    pub icon_type: u8,
    /// Horizontal map coordinate, in the signed 128x128 map space.
    pub x: i8,
    /// Vertical/world-Z-derived map coordinate, in the signed 128x128 map space.
    pub z: i8,
    /// The 4-bit icon rotation.
    pub direction: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FirstPersonMap {
    id: i32,
    rgba: Option<Vec<u8>>,
    decorations: Vec<FirstPersonMapDecoration>,
}

/// The tick-driven part of a [`FirstPersonPhase`]: the equip/swing/use state and everything latched at the last equip swap (the model, a
/// builtin descriptor, a held map, tint, foil and translucency). The phase lives outside the
/// replayed session, so a replay checkpoint copies this beside it and puts it back on a restore
/// ([`FirstPersonPhase::held_snapshot`], [`FirstPersonPhase::restore_held`]). Without it a
/// restore left the hand empty: the model is only latched at a swap edge, and the restored equip
/// latch already matched the held stack, so no swap came.
#[derive(Debug, Clone)]
pub struct FirstPersonHeld {
    render_state: FirstPersonRenderState,
    item_model: Option<BakedModel>,
    item_builtin: Option<BuiltinTileItem>,
    item_map: Option<FirstPersonMap>,
    item_gui_3d: bool,
    item_translucent_block: bool,
    item_tint_snapshot: Vec<u32>,
    item_glint_enabled: bool,
}

impl FirstPersonHeld {
    /// The equip/swing/use state in this copy (tests compare it).
    #[must_use]
    pub fn render_state(&self) -> &FirstPersonRenderState {
        &self.render_state
    }

    /// Whether this copy holds a latched item model.
    #[must_use]
    pub fn has_item_model(&self) -> bool {
        self.item_model.is_some()
    }
}

/// Java's `ItemRenderer.getMapAngleFromPitch`, including its 0..1 clamp.
#[must_use]
pub fn first_person_map_angle_from_pitch(pitch: f32) -> f32 {
    let angle = (1.0 - pitch / 45.0 + 0.1).clamp(0.0, 1.0);
    -cos(angle * core::f32::consts::PI) * 0.5 + 0.5
}

/// The matrix shared by Java's two `renderPlayerArms` calls in `renderItemMap`.
///
/// This stops immediately before the right/left arm model matrices and intentionally excludes
/// the later map-sheet swing rotations. Those rotations apply to the map and its icons, not the
/// arms, in the vanilla matrix stack.
#[must_use]
pub fn first_person_map_arm_matrix(equip: f32, swing: f32, pitch: f32) -> Mat4 {
    let root = sqrt_f(swing) * core::f32::consts::PI;
    let map_angle = first_person_map_angle_from_pitch(pitch);
    Mat4::from_translation(Vec3::new(
        -0.4 * sin(root),
        0.2 * sin(root * 2.0),
        -0.2 * sin(swing * core::f32::consts::PI),
    )) * Mat4::from_translation(Vec3::new(0.0, 0.04, -0.72))
        * Mat4::from_translation(Vec3::new(0.0, equip * -1.2, 0.0))
        * Mat4::from_translation(Vec3::new(0.0, map_angle * -0.5, 0.0))
        * Mat4::from_rotation_y(first_person_radians(90.0))
        * Mat4::from_rotation_z(first_person_radians(map_angle * -85.0))
}

/// The matrix used for the held map background, image, and icon sheet.
#[must_use]
pub fn first_person_map_matrix(equip: f32, swing: f32, pitch: f32) -> Mat4 {
    let root = sqrt_f(swing) * core::f32::consts::PI;
    let swing_sq = sin(swing * swing * core::f32::consts::PI);
    let swing_root = sin(root);
    first_person_map_arm_matrix(equip, swing, pitch)
        * Mat4::from_rotation_y(first_person_radians(swing_sq * -20.0))
        * Mat4::from_rotation_z(first_person_radians(swing_root * -20.0))
        * Mat4::from_rotation_x(first_person_radians(swing_root * -80.0))
        * Mat4::from_scale(Vec3::splat(0.38))
        * Mat4::from_rotation_y(first_person_radians(90.0))
        * Mat4::from_rotation_z(first_person_radians(180.0))
        * Mat4::from_translation(Vec3::new(-1.0, -1.0, 0.0))
        * Mat4::from_scale(Vec3::splat(1.0 / 64.0))
}

/// One ordered step of the frame.
///
/// Implementors are owned by a [`PhaseList`] and recorded once per frame in [`PhaseOrder`] order.
/// See the [module docs](self) for the full contract.
pub trait RenderPhase {
    /// Record this phase's draw commands into the frame encoder.
    ///
    /// Called once per rendered frame — **not** once per game tick. Anything that must advance
    /// at 20 TPS belongs in the game loop, not here; use `cam` and the camera uniform's
    /// `partial_ticks` to interpolate.
    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    );

    /// Short label for debug overlays, `tracing` spans, and GPU captures.
    fn name(&self) -> &'static str {
        "unnamed phase"
    }

    /// The window changed size. Phases holding screen-sized resources (post-effect targets,
    /// shadow maps, UI projection) rebuild them here; the shared depth and MSAA buffers are
    /// already handled by [`Gpu::resize`](crate::Gpu::resize) before this is called.
    fn resize(&mut self, _gpu: &Gpu, _width: u32, _height: u32) {}

    /// Skip this phase entirely for the frame. Cheaper than unregistering and re-registering
    /// when a setting toggles (`hideGUI`, clouds off, particles minimal).
    fn enabled(&self) -> bool {
        true
    }

    /// Opt in to being handed a readable snapshot of the frame so far.
    ///
    /// Returning `true` makes [`PhaseList::render`] call [`RenderPhase::record_scene`] instead
    /// of [`RenderPhase::record`] for this phase, and switches the whole frame onto the
    /// intermediate-target path described on [`SceneTextures`]. Returning `false` — the default —
    /// leaves everything exactly as it was before this method existed.
    ///
    /// Cheap and called every frame; base it on a flag, not on work.
    fn needs_scene_color(&self) -> bool {
        false
    }

    /// Record this phase with access to the frame so far. Only called when
    /// [`RenderPhase::needs_scene_color`] returns `true`.
    ///
    /// `view` is the frame's colour target exactly as [`RenderPhase::record`] receives it (in this
    /// mode the caller's target when it allows copies, else the intermediate target) — draw into
    /// it normally, never sample it. `scene` holds a *separate* snapshot texture to read from, so
    /// sampling and drawing never collide.
    ///
    /// The default implementation forwards to [`RenderPhase::record`] and ignores `scene`, so a
    /// phase that toggles `needs_scene_color` on and off dynamically still works either way.
    fn record_scene(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
        scene: &SceneTextures<'_>,
    ) {
        let _ = scene;
        self.record(gpu, enc, view, cam);
    }

    /// Opt in to recording into one render pass shared with the phases
    /// around it that opt in too.
    ///
    /// Every pass costs the frame its own command buffer plus a barrier buffer when wgpu encodes
    /// it, and with the pipelined submit that encoding is the present thread's largest job. A run
    /// of consecutive enabled phases that each return `true` here is recorded into **one** pass,
    /// configured exactly as [`Gpu::begin_pass`] with [`crate::DepthAttachment::Shared`] does:
    /// colour loaded and stored (resolved at the end with MSAA), the shared depth buffer loaded
    /// and stored. Return `true` only for a frame where [`RenderPhase::record`] would open at most
    /// that one pass and do nothing else with the encoder, so the draws, their order and their
    /// attachments are unchanged. Called every frame; base it on flags, not on work. The default
    /// is `false`: the phase records alone, exactly as before.
    fn shares_world_pass(&self) -> bool {
        false
    }

    /// Record this frame's draws into the shared `pass` instead of opening
    /// one; only called when [`RenderPhase::shares_world_pass`] returned `true` this frame.
    ///
    /// The pass may already hold draws of earlier phases: set every pipeline, bind group, vertex
    /// and index buffer this phase draws with (as it must in a fresh pass anyway), and leave no
    /// pass-level state (scissor, viewport, blend constant, stencil reference) behind. Queue
    /// writes and uploads are fine; the encoder is locked while the pass is open, so it is not
    /// handed over.
    fn record_into_pass(&mut self, gpu: &Gpu, pass: &mut wgpu::RenderPass<'static>, cam: &Camera) {
        let _ = (gpu, pass, cam);
    }

    /// How much of the frame's clear may wait for this phase's own passes when
    /// it records before anything has drawn. The frame's clear then
    /// costs no pass of its own: [`Gpu::begin_pass`] opens the first pass on the frame's colour
    /// target with that colour cleared, and the first pass attaching the shared depth buffer with
    /// it cleared to `1.0`, which leaves every attachment exactly as a clear pass in front would.
    /// Whatever is still pending when a phase that does not allow it records is recorded first,
    /// as the old clear pass. Called every frame; base it on flags, not on work. The default,
    /// [`FrameClearHandoff::None`], keeps the clear pass in front of this phase.
    fn frame_clear_handoff(&self) -> FrameClearHandoff {
        FrameClearHandoff::None
    }

    /// Opt in to recording into one render pass shared with the phases
    /// around it that opt in too, whose shared depth buffer is attached **read-only**: colour loaded and stored (resolved at the end with MSAA) as
    /// [`Gpu::begin_pass`] attaches it, depth loaded and never written
    /// ([`Gpu::begin_depth_read_pass`]). A phase may sample the depth buffer in it (the
    /// screen-space fog) while another depth-tests against it (translucent terrain). Return `true`
    /// only for a frame where [`RenderPhase::record`] would open at most one pass with the frame's
    /// colour target, draw with pipelines that never write depth, and do nothing else with the
    /// encoder. Called every frame; base it on flags, not on work. The default is `false`.
    fn shares_depth_read_pass(&self) -> bool {
        false
    }

    /// Record this frame's draws into the shared read-only-depth `pass`;
    /// only called when [`RenderPhase::shares_depth_read_pass`] returned `true` this frame. Every
    /// pipeline set here needs a depth-stencil state with depth writes off (a phase that drew
    /// without a depth attachment uses a variant with compare `Always`, which draws the same
    /// pixels). The rules of [`RenderPhase::record_into_pass`] apply.
    fn record_into_depth_read_pass(
        &mut self,
        gpu: &Gpu,
        pass: &mut wgpu::RenderPass<'static>,
        cam: &Camera,
    ) {
        let _ = (gpu, pass, cam);
    }
}

/// What a phase lets wait of the frame's clear when it records first (see
/// [`RenderPhase::frame_clear_handoff`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameClearHandoff {
    /// The frame's clear is recorded, as a pass of its own, before this phase records.
    None,
    /// The depth clear may wait: every pass of this phase that attaches the shared depth buffer
    /// opens with [`Gpu::begin_pass`], and nothing else in the phase reads or writes that buffer.
    /// A colour clear still pending is recorded first.
    Depth,
    /// Colour and depth clears may wait: as [`FrameClearHandoff::Depth`], and also the phase's
    /// first pass on the frame's colour target opens with [`Gpu::begin_pass`], with nothing in the
    /// phase reading, copying or drawing into that target before it.
    ColorAndDepth,
}

/// Whether the frame's clear may be folded into the first passes that touch its
/// attachments ([`RenderPhase::frame_clear_handoff`]). Read once; `MC_W18_OB_CLEAR_FOLD=0` keeps
/// the clear pass in front of every frame.
fn clear_fold_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_W18_OB_CLEAR_FOLD").is_none_or(|value| value != "0"))
}

/// Whether consecutive phases that opt in share one read-only-depth pass
/// ([`RenderPhase::shares_depth_read_pass`]). Read once; `MC_W18_OB_DEPTH_READ_PASS=0` gives each
/// its own pass again.
fn depth_read_pass_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("MC_W18_OB_DEPTH_READ_PASS").is_none_or(|value| value != "0")
    })
}

/// Which shared pass is open while `PhaseList` records a run of phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SharedPassKind {
    /// The opaque world's pass: depth loaded and stored.
    World,
    /// The read-only-depth pass (fog and translucent terrain).
    DepthRead,
}

/// Whether consecutive phases that opt in share one render pass. Read once;
/// `MC_NW_F2_SHARED_PASS=0` gives every phase its own pass again. GPU timestamps
/// (`MC_GPU_TIMING`) are written on the encoder, so a shared pass is timed as one scope,
/// `world (shared)`, instead of one row per phase.
fn shared_pass_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_F2_SHARED_PASS").is_none_or(|value| value != "0"))
}

/// Where a phase that can join the shared world pass records this frame:
/// a pass of its own on the frame encoder, or the shared pass. [`PassTarget::begin`] hands back
/// the pass either way, so the phase's drawing code is the same for both.
pub enum PassTarget<'a> {
    /// A pass of the phase's own, opened as [`Gpu::begin_pass`] with [`DepthAttachment::Shared`].
    Own {
        /// The frame encoder the pass is opened on.
        enc: &'a mut wgpu::CommandEncoder,
        /// The colour target, as [`RenderPhase::record`] received it.
        view: &'a wgpu::TextureView,
    },
    /// The frame's shared world pass ([`RenderPhase::record_into_pass`]).
    Shared(&'a mut wgpu::RenderPass<'static>),
}

impl<'a> PassTarget<'a> {
    /// The pass to record into: the shared one, or a new one (labelled `label`) stored in `own`,
    /// which ends it when dropped. `forget_lifetime` changes nothing that is recorded.
    pub fn begin(
        self,
        gpu: &Gpu,
        label: &str,
        own: &'a mut Option<wgpu::RenderPass<'static>>,
    ) -> &'a mut wgpu::RenderPass<'static> {
        match self {
            PassTarget::Shared(pass) => pass,
            PassTarget::Own { enc, view } => own.insert(
                gpu.begin_pass(enc, view, label, DepthAttachment::Shared)
                    .forget_lifetime(),
            ),
        }
    }
}

/// A readable snapshot of the frame so far, handed to a phase that opted in via
/// [`RenderPhase::needs_scene_color`].
///
/// # What `PhaseList::render` does differently
///
/// When **no** enabled phase opts in, nothing changes: the clear pass and every
/// [`RenderPhase::record`] call write straight to the caller's target, exactly as before this
/// type existed.
///
/// When **at least one** enabled phase opts in, [`PhaseList`] allocates two screen-sized
/// textures it owns — an intermediate colour target and a snapshot — and:
///
/// 1. clears and renders the whole frame into the intermediate target instead of the caller's;
/// 2. immediately before each opted-in phase records, copies the intermediate target into the
///    snapshot (`copy_texture_to_texture`, so it is byte-exact, not a resampled blit);
/// 3. hands that snapshot to [`RenderPhase::record_scene`] while the phase still draws into the
///    intermediate target;
/// 4. blits the intermediate target to the caller's target as the frame's last operation.
///
/// Step 4 is a 1:1 `textureLoad` copy at matching format and size — no filtering, no scaling, no
/// mip selection — so an opted-out phase's pixels survive unchanged. `tests/s20_post.rs`'s
/// `an_opted_out_phase_renders_byte_identically_with_and_without_scene_mode` asserts exactly
/// that, byte for byte over a whole frame.
///
/// When the caller's target itself allows `COPY_SRC` at the frame's size
/// and format (the window's swap chain, wherever the surface offers it; headless targets),
/// `PhaseList` skips the intermediate target: it clears and renders into the caller's target,
/// copies *that* into the snapshot before each opted-in phase, and needs no step 4. Same bytes,
/// one full-screen pass and one screen-sized target fewer. `MC_NW_F4_DIRECT_SCENE=0` turns this
/// off.
///
/// MSAA is handled by the intermediate target being the *resolve* target, the same role the
/// caller's target plays without it: [`Gpu::begin_pass`](crate::Gpu::begin_pass) still routes
/// draws to the multisampled attachment, so the snapshot is always single-sampled and directly
/// bindable as `texture_2d<f32>`.
pub struct SceneTextures<'a> {
    /// View of the snapshot: everything drawn this frame up to (not including) the phase being
    /// recorded. Single-sampled, `TEXTURE_BINDING`, same format and size as the frame target.
    pub color: &'a wgpu::TextureView,
    /// The snapshot's format — identical to `gpu.config.format`. Bind a pipeline whose colour
    /// target uses it.
    pub format: wgpu::TextureFormat,
    /// Snapshot width in pixels.
    pub width: u32,
    /// Snapshot height in pixels.
    pub height: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FirstPersonTransformUniform {
    transform: [[f32; 4]; 4],
}

/// A texture and its first-person item/skin bind group. The view is retained by the bind group;
/// keeping the texture here keeps the underlying allocation alive for the phase lifetime.
///
/// Java applies `TextureUtil.anaglyphColor` to a temporary upload array, never to the decoded
/// resource. Keep every resident mip's raw bytes beside the allocation so a live setting toggle
/// can replay the exact source without double-converting the current GPU representation.
struct FirstPersonTexture {
    _texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
    /// Shared by the owners of the shared atlas.
    source_levels: std::rc::Rc<RefCell<Vec<Vec<u8>>>>,
}

const FIRST_PERSON_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct HandUniform {
    transform: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> hand: HandUniform;
@group(2) @binding(0) var hand_tex: texture_2d<f32>;
@group(2) @binding(1) var hand_samp: sampler;

struct VsOut {
    // The base and glint pipelines must reproduce identical depth for GL_EQUAL replay.
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) _normal: vec3<f32>,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    // The CPU matrix is the separate renderHand projection * arm/item pose. `camera` remains
    // group zero by the renderer-wide convention, even though the hand projection deliberately
    // does not use the world camera's view-projection matrix.
    let camera_anchor = camera.frame.x * 0.0;
    out.clip = hand.transform * vec4<f32>(pos, 1.0) + vec4<f32>(0.0, 0.0, 0.0, camera_anchor);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(hand_tex, hand_samp, in.uv);
    // java: RenderItem.renderItemModelTransform -> alphaFunc(GL_GREATER, 0.1F).
    if (texel.a <= 0.1) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.color.rgb, texel.a * in.color.a);
}

// <<NW-RFX:solid>> 1.7 Animations' "1.7 item edges" (`old_item_edges`): 1.7 drew held items
// with the same alpha test but blending off, so a partly transparent texel that passes the test
// is drawn solid.
@fragment
fn fs_solid(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(hand_tex, hand_samp, in.uv);
    if (texel.a <= 0.1) {
        discard;
    }
    return vec4<f32>(texel.rgb * in.color.rgb, 1.0);
}

// <<FIRSTPERSONGLINT-SHADER>>
// java: RenderItem.renderEffect replays the already-depth-tested model with the standalone
// glint texture.  The texture is opaque and replaces the atlas sample, so there is deliberately
// no alpha discard here; transparent model texels have no base depth and therefore fail the
// depth-equal replay naturally.
@fragment
fn fs_glint(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(hand_tex, hand_samp, in.uv);
    return vec4<f32>(texel.rgb * in.color.rgb, texel.a * in.color.a);
}
"#;

// Java temporarily disables the item alpha test while drawing the 128x128 map image. In
// particular, unexplored map texels intentionally use alpha 16/24, so reusing the ordinary
// first-person shader would discard the checkerboard before blending it into the hand pass.
const FIRST_PERSON_MAP_WGSL: &str = r#"
struct CameraUniform {
    view_proj:  mat4x4<f32>,
    view:       mat4x4<f32>,
    proj:       mat4x4<f32>,
    camera_pos: vec4<f32>,
    screen:     vec4<f32>,
    frame:      vec4<f32>,
};
@group(0) @binding(0) var<uniform> camera: CameraUniform;

struct HandUniform {
    transform: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> hand: HandUniform;
@group(2) @binding(0) var hand_tex: texture_2d<f32>;
@group(2) @binding(1) var hand_samp: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

@vertex
fn vs(
    @location(0) pos: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) _normal: vec3<f32>,
    @location(3) color: vec4<f32>,
) -> VsOut {
    var out: VsOut;
    let camera_anchor = camera.frame.x * 0.0;
    out.clip = hand.transform * vec4<f32>(pos, 1.0) + vec4<f32>(0.0, 0.0, 0.0, camera_anchor);
    out.uv = uv;
    out.color = color;
    return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(hand_tex, hand_samp, in.uv);
    return vec4<f32>(texel.rgb * in.color.rgb, texel.a * in.color.a);
}
"#;

// The descriptor fields intentionally follow wgpu's separate Java render-state knobs; keeping
// them explicit makes the depth/blend policy auditable at each call site.
#[allow(clippy::too_many_arguments)]
fn make_first_person_pipeline(
    gpu: &Gpu,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    fragment_entry: &'static str,
    blend: wgpu::BlendState,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    make_first_person_pipeline_with_cull(
        gpu,
        layout,
        shader,
        label,
        fragment_entry,
        blend,
        depth_write_enabled,
        depth_compare,
        None,
    )
}

/// [`make_first_person_pipeline`] with an explicit `GL_CULL_FACE` mode.
///
/// The arms and the held map stay two-sided (`ItemRenderer` brackets the arm draws with
/// `disableCull`/`enableCull`), but a baked held item is drawn with the world pass's
/// `GL_CULL_FACE` still enabled; see [`first_person_item_cull_face`].
#[allow(clippy::too_many_arguments)]
fn make_first_person_pipeline_with_cull(
    gpu: &Gpu,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &'static str,
    fragment_entry: &'static str,
    blend: wgpu::BlendState,
    depth_write_enabled: bool,
    depth_compare: wgpu::CompareFunction,
    cull_mode: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    gpu.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[Some(ENTITY_VERTEX_LAYOUT)],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            // `EntityRenderer.renderHand` clears depth before `ItemRenderer` draws.  The base
            // hand/item replay therefore writes the cleared shared depth, while the foil replay
            // below selects the exact same fragments with `GL_EQUAL` and leaves depth untouched.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write_enabled),
                depth_compare: Some(depth_compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: gpu.sample_count(),
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: shader,
                entry_point: Some(fragment_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: gpu.config.format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        })
}

/// The dimensions of a resident first-person texture mip. Atlas dimensions are powers of two,
/// while resource-pack PNGs normally have one level; clamping keeps validation well-defined for
/// either source without changing the supplied mip chain.
fn first_person_mip_dimensions(width: u32, height: u32, level: u32) -> (u32, u32) {
    (
        width.checked_shr(level).unwrap_or(0).max(1),
        height.checked_shr(level).unwrap_or(0).max(1),
    )
}

/// Build the temporary RGBA8 upload copy used by every first-person texture boundary.
///
/// `TextureUtil.anaglyphColor` mixes RGB with integer arithmetic and carries alpha through
/// unchanged.  The disabled path is intentionally an exact byte-for-byte copy, and callers retain
/// the unmodified source levels beside the GPU allocation for live setting toggles.
fn first_person_upload_rgba(source_rgba: &[u8], anaglyph: bool) -> Vec<u8> {
    mc_assets::sprite::anaglyph_rgba(source_rgba, anaglyph)
}

fn upload_first_person_texture_levels(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    source_levels: &[Vec<u8>],
    anaglyph: bool,
) {
    for (level, source_rgba) in source_levels.iter().enumerate() {
        let level = level as u32;
        let (level_width, level_height) = first_person_mip_dimensions(width, height, level);
        let upload_rgba = first_person_upload_rgba(source_rgba, anaglyph);
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
}

fn upload_first_person_texture_rect(
    gpu: &Gpu,
    texture: &wgpu::Texture,
    upload: &AtlasAnimationUpload,
    anaglyph: bool,
) {
    let upload_rgba = first_person_upload_rgba(&upload.rgba, anaglyph);
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
}

fn reupload_first_person_texture(gpu: &Gpu, texture: &FirstPersonTexture, anaglyph: bool) {
    let source_levels = texture.source_levels.borrow();
    upload_first_person_texture_levels(
        gpu,
        &texture._texture,
        texture.width,
        texture.height,
        &source_levels,
        anaglyph,
    );
}

fn make_first_person_texture(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    width: u32,
    height: u32,
    pixels: &[u8],
    anaglyph: bool,
) -> FirstPersonTexture {
    make_first_person_texture_with_address(
        gpu,
        layout,
        label,
        width,
        height,
        pixels,
        wgpu::AddressMode::ClampToEdge,
        anaglyph,
    )
}

/// Upload a first-person texture with an explicit address mode.  Item/skin/map textures retain
/// vanilla's clamp behaviour, while the animated foil's unbounded texture-matrix UVs require a
/// repeat sampler just like `RenderItem.renderEffect`'s GL texture state.
#[allow(clippy::too_many_arguments)]
fn make_first_person_texture_with_address(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    width: u32,
    height: u32,
    pixels: &[u8],
    address_mode: wgpu::AddressMode,
    anaglyph: bool,
) -> FirstPersonTexture {
    make_first_person_texture_with_levels_and_address(
        gpu,
        layout,
        label,
        width,
        height,
        vec![pixels.to_vec()],
        address_mode,
        anaglyph,
    )
}

#[allow(clippy::too_many_arguments)]
fn make_first_person_texture_with_levels_and_address(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    width: u32,
    height: u32,
    source_levels: Vec<Vec<u8>>,
    address_mode: wgpu::AddressMode,
    anaglyph: bool,
) -> FirstPersonTexture {
    make_first_person_texture_sampled(
        gpu,
        layout,
        label,
        width,
        height,
        source_levels,
        address_mode,
        wgpu::FilterMode::Nearest,
        anaglyph,
    )
}

/// The foil's sampler from its `.mcmeta` (`blur` -> linear, `clamp` -> clamp to edge).
///
/// java: `RenderItem.RES_ITEM_GLINT` is an ordinary `SimpleTexture`, so `enchanted_item_glint.png`'s
/// sidecar applies, and vanilla's says `blur: true`: `RenderItem.renderEffect` samples the foil with
/// `GL_LINEAR`. (`renderItemModelTransform`'s `setBlurMipmap(false, false)` is called on the
/// *blocks atlas* texture object, not on the foil.) The first-person foil was
/// point-sampled, which turned the smooth sweep into hard bands on every enchanted held item.
fn glint_sampler_modes(
    options: &mc_assets::TextureMetadata,
) -> (wgpu::FilterMode, wgpu::AddressMode) {
    (
        if options.blur {
            wgpu::FilterMode::Linear
        } else {
            wgpu::FilterMode::Nearest
        },
        if options.clamp {
            wgpu::AddressMode::ClampToEdge
        } else {
            wgpu::AddressMode::Repeat
        },
    )
}

/// [`make_first_person_texture_with_levels_and_address`] with an explicit min/mag filter (the
/// foil, see [`glint_sampler_modes`]); every other first-person texture is nearest.
#[allow(clippy::too_many_arguments)]
fn make_first_person_texture_sampled(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    label: &str,
    width: u32,
    height: u32,
    source_levels: Vec<Vec<u8>>,
    address_mode: wgpu::AddressMode,
    filter: wgpu::FilterMode,
    anaglyph: bool,
) -> FirstPersonTexture {
    assert!(
        width > 0 && height > 0,
        "{label} must have non-zero dimensions"
    );
    assert!(
        !source_levels.is_empty(),
        "{label} must have a base mip level"
    );
    for (level, pixels) in source_levels.iter().enumerate() {
        let (level_width, level_height) = first_person_mip_dimensions(width, height, level as u32);
        assert_eq!(
            pixels.len(),
            (u64::from(level_width) * u64::from(level_height) * 4) as usize,
            "{label} mip {level} has {} bytes, expected {level_width}x{level_height}x4",
            pixels.len()
        );
    }
    // The block atlas adopts the one shared GPU atlas inside the client's
    // install sweep (`atlas_transfer::AtlasShareScope`); every other texture is a plain create.
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
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        &source_levels,
        anaglyph,
    );
    upload_first_person_texture_levels(gpu, &texture, width, height, &source_levels, anaglyph);
    // java: the held item is drawn by `RenderItem.renderItemModelTransform`, which brackets it
    // with `setBlurMipmap(false, false)` (`RenderItem.java:319`, restored at `:340`): that is
    // `GL_TEXTURE_MIN_FILTER = GL_NEAREST` (`AbstractTexture.setBlurMipmapDirect`), the base
    // level only. The arm, map and foil textures have no mips at all. So the bound view is the
    // base level, and an edge-on or distant held-item face never samples a mip; the atlas's
    // other levels stay allocated only for the animated-sprite uploads that keep them current.
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        mip_level_count: Some(1),
        ..Default::default()
    });
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(label),
        mag_filter: filter,
        min_filter: filter,
        mipmap_filter: wgpu::MipmapFilterMode::Nearest,
        address_mode_u: address_mode,
        address_mode_v: address_mode,
        ..Default::default()
    });
    let bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    // One raw shadow for every owner of the shared atlas (only the
    // block atlas can hold it; every other texture keeps its own levels as before).
    let source_levels = crate::atlas_transfer::share_atlas_shadow(&texture, source_levels);
    FirstPersonTexture {
        _texture: texture,
        bind_group,
        width,
        height,
        source_levels,
    }
}

fn load_first_person_texture(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    resources: &mc_assets::ResourceManager,
    path: &str,
    label: &str,
    anaglyph: bool,
) -> Option<FirstPersonTexture> {
    load_first_person_texture_with_address(
        gpu,
        layout,
        resources,
        path,
        label,
        wgpu::AddressMode::ClampToEdge,
        anaglyph,
    )
}

#[allow(clippy::too_many_arguments)]
fn load_first_person_texture_with_address(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    resources: &mc_assets::ResourceManager,
    path: &str,
    label: &str,
    address_mode: wgpu::AddressMode,
    anaglyph: bool,
) -> Option<FirstPersonTexture> {
    let bytes = resources
        .load(&mc_assets::ResourceLocation::minecraft(path))
        .ok()?;
    let image = image::load_from_memory(&bytes).ok()?.to_rgba8();
    let width = image.width();
    let height = image.height();
    // A pack texture with a side over the device's limit reads as
    // missing (as a PNG that does not decode): creating it would be a validation error, which
    // ends the process on every start while the pack stays selected.
    let max_side = gpu.device.limits().max_texture_dimension_2d;
    (width > 0 && height > 0 && width <= max_side && height <= max_side).then(|| {
        make_first_person_texture_with_address(
            gpu,
            layout,
            label,
            width,
            height,
            image.as_raw(),
            address_mode,
            anaglyph,
        )
    })
}

fn push_first_person_quad(
    out: &mut Vec<EntityVertex>,
    transform: Mat4,
    positions: [[f32; 3]; 4],
    uvs: [[f32; 2]; 4],
) {
    let normal: [f32; 3] = transform
        .transform_vector3(Vec3::Z)
        .normalize_or_zero()
        .into();
    let positions: [Vec3; 4] =
        positions.map(|position| transform.transform_point3(Vec3::from(position)));
    for &index in &[0usize, 1, 2, 0, 2, 3] {
        out.push(EntityVertex {
            pos: positions[index].into(),
            uv: uvs[index],
            normal,
            color: EntityVertex::WHITE,
        });
    }
}

fn push_first_person_map_geometry(
    map: &FirstPersonMap,
    map_matrix: Mat4,
    background: &mut Vec<EntityVertex>,
    image: &mut Vec<EntityVertex>,
    icons: &mut Vec<EntityVertex>,
) {
    push_first_person_quad(
        background,
        map_matrix,
        [
            [-7.0, -7.0, 0.0],
            [-7.0, 135.0, 0.0],
            [135.0, 135.0, 0.0],
            [135.0, -7.0, 0.0],
        ],
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
    );
    if map.rgba.is_some() {
        push_first_person_quad(
            image,
            map_matrix * Mat4::from_translation(Vec3::new(0.0, 0.0, -0.01)),
            [
                [0.0, 0.0, 0.0],
                [0.0, 128.0, 0.0],
                [128.0, 128.0, 0.0],
                [128.0, 0.0, 0.0],
            ],
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        );
    }
    for (index, decoration) in map.decorations.iter().enumerate() {
        let col = f32::from(decoration.icon_type % 4) / 4.0;
        let row = f32::from(decoration.icon_type / 4) / 4.0;
        let icon_matrix = map_matrix
            * Mat4::from_translation(Vec3::new(
                f32::from(decoration.x) / 2.0 + 64.0,
                f32::from(decoration.z) / 2.0 + 64.0,
                -0.02 - index as f32 * 0.001,
            ))
            * Mat4::from_rotation_z(first_person_radians(
                f32::from(decoration.direction) * 360.0 / 16.0,
            ))
            * Mat4::from_scale(Vec3::splat(4.0))
            * Mat4::from_translation(Vec3::new(-0.125, 0.125, 0.0));
        push_first_person_quad(
            icons,
            icon_matrix,
            [
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, -1.0, 0.0],
            ],
            [
                [col, row],
                [col + 0.25, row],
                [col + 0.25, row + 0.25],
                [col, row + 0.25],
            ],
        );
    }
}

fn rgba_len_matches(width: u32, height: u32, pixels: &[u8]) -> bool {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|n| n.checked_mul(4))
        .is_some_and(|n| n == pixels.len() as u64)
}

fn ensure_first_person_vertex_buffer(
    gpu: &Gpu,
    buffer: &mut wgpu::Buffer,
    capacity: &mut usize,
    vertices: &[EntityVertex],
    label: &str,
) {
    if vertices.len() > *capacity {
        let capacity_bytes = core::mem::size_of_val(vertices) as u64;
        *buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *capacity = vertices.len();
    }
    gpu.write_buffer_staged(buffer, 0, bytemuck::cast_slice(vertices));
}

/// Java's `tryBlendFuncSeparate(770, 771, 1, 0)`: source-alpha colour compositing and source
/// replacement for the destination alpha channel. `ALPHA_BLENDING` differs in that alpha uses
/// `OneMinusSrcAlpha` for its destination factor.
fn first_person_blend_state() -> wgpu::BlendState {
    wgpu::BlendState {
        color: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        },
        alpha: wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::Zero,
            operation: wgpu::BlendOperation::Add,
        },
    }
}

/// The exact `RenderItem.renderEffect` blend equation: `blendFunc(GL_SRC_COLOR, GL_ONE)`.
/// Alpha is deliberately the same source-colour-plus-one equation because Java's two-argument
/// `blendFunc` applies to both channels; the glint texture itself supplies the opacity.
fn first_person_glint_blend_state() -> wgpu::BlendState {
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

/// The `GL_CULL_FACE` mode a first-person baked held item (and its foil) is drawn with.
///
/// java: `EntityRenderer.renderWorldPass` calls `enableCull()` (`GL_BACK`) before `renderHand`,
/// and nothing on the held-item path disables it: only `ItemRenderer.renderPlayerArm` and
/// `renderPlayerArms` bracket the *arm* with `disableCull`/`enableCull`.
/// `RenderItem.renderItemModelTransform` then switches to `cullFace(GL_FRONT)` while
/// `isThereOneNegativeScale` (an odd number of negative display-scale components) mirrors the
/// model, and back to `GL_BACK` afterwards.
///
/// The model and its `renderEffect` foil share that state, so of a coplanar pair of
/// opposite-facing `ItemModelGenerator` side quads (one pixel row's UP side and the row above's
/// DOWN side, or the matching LEFT/RIGHT pair) only the one facing the camera is drawn. Drawn
/// two-sided, the `GL_EQUAL` foil lands on such a pixel once or twice depending on how the two
/// quads rasterise: the shimmer along an enchanted bow's diagonal string.
#[must_use]
pub fn first_person_item_cull_face(display: &mc_assets::model::ItemTransform) -> wgpu::Face {
    let [x, y, z] = display.scale;
    if (x < 0.0) ^ (y < 0.0) ^ (z < 0.0) {
        wgpu::Face::Front
    } else {
        wgpu::Face::Back
    }
}

/// One cull face's held baked-item pipelines; see [`first_person_item_cull_face`].
struct FirstPersonItemPipelines {
    /// The ordinary depth-writing model pass.
    base: wgpu::RenderPipeline,
    /// `ItemRenderer.renderItem`'s translucent GUI-3D block pass (`depthMask(false)`).
    translucent: wgpu::RenderPipeline,
    /// `RenderItem.renderEffect`'s `GL_EQUAL` foil replay.
    glint: wgpu::RenderPipeline,
}

/// Whether the base held-item pass writes depth under Java's `ItemRenderer.renderItem` gate.
///
/// `shouldRenderItemIn3D` is the baked model's GUI-3D bit.  A translucent block with a flat
/// override keeps the ordinary depth mask; every non-block and non-translucent block does too.
#[must_use]
pub const fn first_person_item_writes_depth(translucent_block: bool, gui_3d: bool) -> bool {
    !(translucent_block && gui_3d)
}

/// The retained first-person hand/item phase.
///
/// The client supplies the current [`BakedModel`] and atlas/skin bytes through the setters, then
/// updates [`Self::render_state`] once per tick. This phase owns the vanilla `renderHand`
/// projection and CPU-emits the same model quads and right-arm parts as the Java display-list
/// path. It is intentionally independent of the world depth buffer: Java clears depth before
/// `renderItemInFirstPerson`, so the hand must always sit in front of the world.
pub struct FirstPersonPhase {
    /// Equip/swing/action state consumed by the phase.
    pub render_state: FirstPersonRenderState,
    /// Render interpolation supplied by the frame driver.
    pub partial_ticks: f32,
    /// Interpolated player arm pitch input (`renderArmPitch` pair is below).
    pub prev_render_arm_pitch: f32,
    /// Current player arm pitch input.
    pub render_arm_pitch: f32,
    /// Interpolated player arm yaw input (`renderArmYaw` pair is below).
    pub prev_render_arm_yaw: f32,
    /// Current player arm yaw input.
    pub render_arm_yaw: f32,
    /// Whether to draw the empty right arm when no item model is latched.
    pub render_empty_arm: bool,
    /// Clear the shared depth buffer in this phase's slot even when no
    /// hand is drawn, so it holds the hand's depth and nothing else when the suffocation overlay
    /// ([`crate::post::OverlayPhase::opaque_block_hand_depth`]) is depth-tested against it.
    ///
    /// java: `renderWorldPass` clears depth before `renderHand` unconditionally, and
    /// `renderOverlays` runs even when `renderItemInFirstPerson` is skipped (F1, spectator).
    /// `false`, the default, is the old behaviour: no pass at all without a hand to draw.
    pub clear_depth_for_overlay: bool,
    /// The model-view prefix `EntityRenderer.renderHand` installs before
    /// `renderItemInFirstPerson`: `hurtCameraEffect` (death roll, then hurt shake), then
    /// `setupViewBobbing`. The same two effects the world camera applies, without
    /// `orientCamera`.
    pub view_bobbing: Mat4,
    /// `World.getCombinedLight(eye, 0)` from `ItemRenderer.setLightMapFromPlayer`, packed as Java
    /// packs it: sky light in bits 20..24, block light in bits 4..8. Full bright by default.
    pub combined_light: u32,
    /// The "1.7 Animations" switches for the held item, set by the client every
    /// frame. `None` (the default) is vanilla 1.8.9; see [`old_animations::held_item_pose`].
    pub old_animations: Option<old_animations::OldAnimations>,
    /// 3D Skins: the local skin's voxel layers, set by the client while
    /// the mod (and its first-person option) is on. `Some` draws the empty arm's sleeve from the
    /// mesh instead of the flat `bipedRightArmwear` box; `None` (the default) is vanilla.
    pub skin3d_sleeve: Option<std::sync::Arc<crate::entity::skin3d::SkinLayerMesh>>,
    /// `EntityRenderer.lightmapColors` for this frame and the inputs it was generated from; `None`
    /// until the client supplies them, which leaves the hand unmodulated.
    lightmap: Option<(
        crate::sky::lightmap::LightmapParams,
        crate::sky::lightmap::Lightmap,
    )>,

    pipeline: wgpu::RenderPipeline,
    /// `ItemRenderer.renderItem`'s translucent-block variant: identical colour blending and
    /// depth comparison to the ordinary held-item pipeline, but with depth writes disabled.
    translucent_item_pipeline: wgpu::RenderPipeline,
    map_pipeline: wgpu::RenderPipeline,
    /// `RenderItem.renderEffect`'s depth-equal, no-depth-write replay pipeline.
    glint_pipeline: wgpu::RenderPipeline,
    /// Baked held-item pipelines under `GL_BACK` / `GL_FRONT` culling.
    item_back_pipelines: FirstPersonItemPipelines,
    item_front_pipelines: FirstPersonItemPipelines,
    /// The cull face for the current `item_vertices`: set for the baked-model branch, `None`
    /// for the builtin/entity branch, which keeps the two-sided pipelines.
    item_cull_face: Option<wgpu::Face>,
    texture_layout: wgpu::BindGroupLayout,
    item_transform_buffer: wgpu::Buffer,
    item_transform_group: wgpu::BindGroup,
    arm_transform_buffer: wgpu::Buffer,
    item_transform_write: crate::context::ChangedWrite,
    arm_transform_write: crate::context::ChangedWrite,
    arm_transform_group: wgpu::BindGroup,
    item_vertex_buffer: wgpu::Buffer,
    /// Separate storage is required because Java draws the base model, then replays both foil
    /// passes without changing the base vertices' atlas UVs.
    item_glint_vertex_buffer: wgpu::Buffer,
    arm_vertex_buffer: wgpu::Buffer,
    item_vertex_capacity: usize,
    item_glint_vertex_capacity: usize,
    arm_vertex_capacity: usize,
    map_background_vertex_buffer: wgpu::Buffer,
    map_vertex_buffer: wgpu::Buffer,
    map_icon_vertex_buffer: wgpu::Buffer,
    map_background_vertex_capacity: usize,
    map_vertex_capacity: usize,
    map_icon_vertex_capacity: usize,
    /// Java's global upload-time `GameSettings.anaglyph` switch.
    anaglyph: bool,
    atlas: FirstPersonTexture,
    /// Standalone `textures/misc/enchanted_item_glint.png`, sampled with repeat addressing.
    glint: FirstPersonTexture,
    skin: FirstPersonTexture,
    /// Source identity for the installed player skin, avoiding a GPU texture rebuild every frame.
    skin_texture_key: Option<String>,
    map_background: FirstPersonTexture,
    map_texture: FirstPersonTexture,
    map_icons: FirstPersonTexture,
    map_texture_dirty: bool,
    item_model: Option<BakedModel>,
    /// Decoded state for a held model whose baked model delegates to `builtin/entity`.
    item_builtin: Option<BuiltinTileItem>,
    /// The entity-sheet/composited-banner texture bound for [`Self::item_builtin`].
    item_builtin_texture: Option<FirstPersonTexture>,
    /// Internal texture key used to avoid rebuilding the same held entity texture every frame.
    item_builtin_texture_key: Option<String>,
    /// Vanilla TESR models reused by the held builtin/entity branch.
    item_builtin_models: TileModels,
    item_map: Option<FirstPersonMap>,
    item_gui_3d: bool,
    /// Latched `Block.getBlockLayer() == TRANSLUCENT` result for the equipped stack.  This is
    /// deliberately independent of `item_gui_3d`; Java consults the block registry rather than
    /// guessing material from the baked model shape.
    item_translucent_block: bool,
    /// Owned `Item.getColorFromItemStack` results for every baked-quad tint index. The client
    /// latches this beside the equipped stack so an outgoing model cannot borrow the incoming
    /// stack's colour during the equip animation.
    item_tint_snapshot: Vec<u32>,
    /// Latched `ItemStack.hasEffect()` for `item_model`; it changes only at the equip swap edge.
    item_glint_enabled: bool,
    player_model: PlayerModel,
    item_vertices: Vec<EntityVertex>,
    item_glint_vertices: Vec<EntityVertex>,
    arm_vertices: Vec<EntityVertex>,
    map_background_vertices: Vec<EntityVertex>,
    map_vertices: Vec<EntityVertex>,
    map_icon_vertices: Vec<EntityVertex>,
    // 1.7 Animations' "1.7 item edges" (option id `translucent_held`). See [`old_item_edges`].
    /// Draw a flat held item that has partly transparent texels the 1.7 way: 1.7's edges and
    /// no blending. Set by the client every frame; `false`, the default, is vanilla.
    pub old_item_edges: bool,
    /// The 1.7-edged copy of `item_model`: `None` until first needed for the latched model,
    /// `Some(None)` when that model keeps the vanilla path (nothing partly transparent).
    item_model_old: Option<Option<BakedModel>>,
    /// Whether this frame's held item uses [`Self::solid_item_pipelines`].
    item_solid: bool,
    /// The no-blend held-item pipelines under `GL_BACK` and `GL_FRONT` culling, built on first
    /// use and dropped when the option goes off.
    solid_item_pipelines: Option<[wgpu::RenderPipeline; 2]>,
    /// The held-item pipeline layout and shader, kept to build those lazily.
    item_pipeline_layout: wgpu::PipelineLayout,
    item_shader: wgpu::ShaderModule,
}

impl FirstPersonPhase {
    /// Construct a first-person phase with white fallback textures. Resource loading is kept out
    /// of the constructor so a missing pack can be diagnosed and replaced by `set_atlas` or
    /// `set_skin_rgba` without rebuilding the render pipeline.
    #[must_use]
    pub fn new(gpu: &Gpu) -> Self {
        Self::new_with_anaglyph(gpu, false)
    }

    /// Construct the phase with Java's global upload-time anaglyph switch selected.
    ///
    /// The ordinary [`Self::new`] constructor remains disabled for compatibility. A client that
    /// owns `GameSettings` can seed the phase here, or call [`Self::set_anaglyph`] before its first
    /// resource-backed upload.
    #[must_use]
    pub fn new_with_anaglyph(gpu: &Gpu, anaglyph: bool) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("first-person hand/item"),
                source: wgpu::ShaderSource::Wgsl(FIRST_PERSON_WGSL.into()),
            });
        let transform_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("first-person transform"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let texture_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("first-person texture"),
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
                label: Some("first-person layout"),
                bind_group_layouts: &[
                    Some(gpu.camera_layout()),
                    Some(&transform_layout),
                    Some(&texture_layout),
                ],
                immediate_size: 0,
            });
        let pipeline = make_first_person_pipeline(
            gpu,
            &layout,
            &shader,
            "first-person hand/item",
            "fs",
            first_person_blend_state(),
            true,
            wgpu::CompareFunction::LessEqual,
        );
        let translucent_item_pipeline = make_first_person_pipeline(
            gpu,
            &layout,
            &shader,
            "first-person translucent block item",
            "fs",
            first_person_blend_state(),
            false,
            wgpu::CompareFunction::LessEqual,
        );
        let map_shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("first-person map"),
                source: wgpu::ShaderSource::Wgsl(FIRST_PERSON_MAP_WGSL.into()),
            });
        let map_pipeline = make_first_person_pipeline(
            gpu,
            &layout,
            &map_shader,
            "first-person map",
            "fs",
            first_person_blend_state(),
            true,
            wgpu::CompareFunction::LessEqual,
        );
        // Reuse the ordinary first-person vertex shader's camera/hand bindings, but select the
        // no-alpha-discard glint fragment entry point and exact depth/blend state.
        let glint_pipeline = make_first_person_pipeline(
            gpu,
            &layout,
            &shader,
            "first-person glint",
            "fs_glint",
            first_person_glint_blend_state(),
            false,
            wgpu::CompareFunction::Equal,
        );
        let make_item_pipelines = |face: wgpu::Face| FirstPersonItemPipelines {
            base: make_first_person_pipeline_with_cull(
                gpu,
                &layout,
                &shader,
                "first-person held item",
                "fs",
                first_person_blend_state(),
                true,
                wgpu::CompareFunction::LessEqual,
                Some(face),
            ),
            translucent: make_first_person_pipeline_with_cull(
                gpu,
                &layout,
                &shader,
                "first-person held translucent block item",
                "fs",
                first_person_blend_state(),
                false,
                wgpu::CompareFunction::LessEqual,
                Some(face),
            ),
            glint: make_first_person_pipeline_with_cull(
                gpu,
                &layout,
                &shader,
                "first-person held item glint",
                "fs_glint",
                first_person_glint_blend_state(),
                false,
                wgpu::CompareFunction::Equal,
                Some(face),
            ),
        };
        let item_back_pipelines = make_item_pipelines(wgpu::Face::Back);
        let item_front_pipelines = make_item_pipelines(wgpu::Face::Front);

        let transform_value = FirstPersonTransformUniform {
            transform: Mat4::IDENTITY.to_cols_array_2d(),
        };
        let transform_bytes = bytemuck::bytes_of(&transform_value);
        let make_transform = |label: &str| {
            let buffer = gpu
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: transform_bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &transform_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, group)
        };
        let (item_transform_buffer, item_transform_group) = make_transform("first-person item");
        let (arm_transform_buffer, arm_transform_group) = make_transform("first-person arm");
        let initial = [255u8, 255, 255, 255];
        let atlas = make_first_person_texture(
            gpu,
            &texture_layout,
            "first-person atlas fallback",
            1,
            1,
            &initial,
            anaglyph,
        );
        let skin = make_first_person_texture(
            gpu,
            &texture_layout,
            "first-person skin fallback",
            1,
            1,
            &initial,
            anaglyph,
        );
        let make_vertex_buffer = |label: &str| {
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytemuck::bytes_of(&EntityVertex::default()),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                })
        };
        let item_vertex_buffer = make_vertex_buffer("first-person item vertices");
        let item_glint_vertex_buffer = make_vertex_buffer("first-person item glint vertices");
        let arm_vertex_buffer = make_vertex_buffer("first-person arm vertices");
        let map_background_vertex_buffer =
            make_vertex_buffer("first-person map background vertices");
        let map_vertex_buffer = make_vertex_buffer("first-person map vertices");
        let map_icon_vertex_buffer = make_vertex_buffer("first-person map icon vertices");
        let map_background = make_first_person_texture(
            gpu,
            &texture_layout,
            "first-person map background fallback",
            1,
            1,
            &initial,
            anaglyph,
        );
        let map_texture = make_first_person_texture(
            gpu,
            &texture_layout,
            "first-person map fallback",
            1,
            1,
            &initial,
            anaglyph,
        );
        let map_icons = make_first_person_texture(
            gpu,
            &texture_layout,
            "first-person map icons fallback",
            1,
            1,
            &initial,
            anaglyph,
        );
        let glint = make_first_person_texture_with_address(
            gpu,
            &texture_layout,
            "first-person glint fallback",
            1,
            1,
            &initial,
            wgpu::AddressMode::Repeat,
            anaglyph,
        );

        Self {
            render_state: FirstPersonRenderState::default(),
            partial_ticks: 0.0,
            prev_render_arm_pitch: 0.0,
            render_arm_pitch: 0.0,
            prev_render_arm_yaw: 0.0,
            render_arm_yaw: 0.0,
            render_empty_arm: true,
            clear_depth_for_overlay: false,
            view_bobbing: Mat4::IDENTITY,
            combined_light: crate::entity::render::FULL_BRIGHT_COMBINED_LIGHT,
            old_animations: None,
            skin3d_sleeve: None,
            lightmap: None,
            pipeline,
            translucent_item_pipeline,
            map_pipeline,
            glint_pipeline,
            item_back_pipelines,
            item_front_pipelines,
            item_cull_face: None,
            texture_layout,
            item_transform_buffer,
            item_transform_group,
            arm_transform_buffer,
            item_transform_write: crate::context::ChangedWrite::default(),
            arm_transform_write: crate::context::ChangedWrite::default(),
            arm_transform_group,
            item_vertex_buffer,
            item_glint_vertex_buffer,
            arm_vertex_buffer,
            item_vertex_capacity: 1,
            item_glint_vertex_capacity: 1,
            arm_vertex_capacity: 1,
            map_background_vertex_buffer,
            map_vertex_buffer,
            map_icon_vertex_buffer,
            map_background_vertex_capacity: 1,
            map_vertex_capacity: 1,
            map_icon_vertex_capacity: 1,
            atlas,
            glint,
            anaglyph,
            skin,
            skin_texture_key: None,
            map_background,
            map_texture,
            map_icons,
            map_texture_dirty: false,
            item_model: None,
            item_builtin: None,
            item_builtin_texture: None,
            item_builtin_texture_key: None,
            item_builtin_models: TileModels::new(christmas_today()),
            item_map: None,
            item_gui_3d: true,
            item_translucent_block: false,
            item_tint_snapshot: Vec::new(),
            item_glint_enabled: false,
            player_model: PlayerModel::new(0.0, false),
            item_vertices: Vec::new(),
            item_glint_vertices: Vec::new(),
            arm_vertices: Vec::new(),
            map_background_vertices: Vec::new(),
            map_vertices: Vec::new(),
            map_icon_vertices: Vec::new(),
            old_item_edges: false,
            item_model_old: None,
            item_solid: false,
            solid_item_pipelines: None,
            item_pipeline_layout: layout,
            item_shader: shader,
        }
    }

    /// Replace the item/block atlas used by baked model UVs. Invalid dimensions leave the
    /// previous texture installed, keeping a malformed resource pack non-fatal.
    pub fn set_atlas(&mut self, gpu: &Gpu, atlas: &Atlas) {
        if atlas.w == 0 || atlas.h == 0 || !rgba_len_matches(atlas.w, atlas.h, &atlas.texture) {
            return;
        }
        // The 1.7-edged copy reads the old atlas's texels.
        self.item_model_old = None;
        let mut source_levels = Vec::with_capacity(1 + atlas.mipmaps.len());
        source_levels.push(atlas.texture.clone());
        source_levels.extend(atlas.mipmaps.iter().cloned());
        if source_levels.iter().enumerate().any(|(level, pixels)| {
            let (width, height) = first_person_mip_dimensions(atlas.w, atlas.h, level as u32);
            !rgba_len_matches(width, height, pixels)
        }) {
            return;
        }
        self.atlas = make_first_person_texture_with_levels_and_address(
            gpu,
            &self.texture_layout,
            "first-person item atlas",
            atlas.w,
            atlas.h,
            source_levels,
            wgpu::AddressMode::ClampToEdge,
            self.anaglyph,
        );
    }

    /// Apply one S09 animated-atlas dirty rectangle to the retained held-item texture.
    ///
    /// The phase retains the atlas's full sprite-local mip chain, so level-specific animation
    /// records are accepted and written to the corresponding subresource just like Java's
    /// `TextureUtil.uploadTextureMipmap` path.
    pub fn upload_atlas_animation(&self, gpu: &Gpu, upload: &AtlasAnimationUpload) -> bool {
        self.upload_atlas_animation_with_sink(upload, |texture, anaglyph| {
            upload_first_person_texture_rect(gpu, texture, upload, anaglyph);
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
        if upload.width == 0
            || upload.height == 0
            || upload.rgba.len()
                != (u64::from(upload.width) * u64::from(upload.height) * 4) as usize
            || upload.mip_level as usize >= self.atlas.source_levels.borrow().len()
            || upload.x.checked_add(upload.width).is_none_or(|end| {
                end > first_person_mip_dimensions(
                    self.atlas.width,
                    self.atlas.height,
                    upload.mip_level,
                )
                .0
            })
            || upload.y.checked_add(upload.height).is_none_or(|end| {
                end > first_person_mip_dimensions(
                    self.atlas.width,
                    self.atlas.height,
                    upload.mip_level,
                )
                .1
            })
        {
            return false;
        }
        let (level_width, _) =
            first_person_mip_dimensions(self.atlas.width, self.atlas.height, upload.mip_level);
        {
            let mut source_levels = self.atlas.source_levels.borrow_mut();
            let source = &mut source_levels[upload.mip_level as usize];
            for row in 0..upload.height {
                let src_start = (row * upload.width * 4) as usize;
                let dst_start = (((upload.y + row) * level_width + upload.x) * 4) as usize;
                let count = (upload.width * 4) as usize;
                source[dst_start..dst_start + count]
                    .copy_from_slice(&upload.rgba[src_start..src_start + count]);
            }
        }
        sink(&self.atlas._texture, self.anaglyph);
        true
    }

    /// Whether this phase applies Java's global upload-time texture transform.
    #[must_use]
    pub const fn anaglyph_enabled(&self) -> bool {
        self.anaglyph
    }

    /// Toggle Java's global texture-upload transform and replay every resident first-person
    /// texture from its retained raw source levels. Replaying raw pixels avoids double conversion
    /// and includes the latest animated atlas frame at every resident mip.
    pub fn set_anaglyph_with_gpu(&mut self, gpu: &Gpu, enabled: bool) {
        if self.anaglyph == enabled {
            return;
        }
        self.anaglyph = enabled;
        reupload_first_person_texture(gpu, &self.atlas, enabled);
        reupload_first_person_texture(gpu, &self.glint, enabled);
        reupload_first_person_texture(gpu, &self.skin, enabled);
        reupload_first_person_texture(gpu, &self.map_background, enabled);
        reupload_first_person_texture(gpu, &self.map_texture, enabled);
        reupload_first_person_texture(gpu, &self.map_icons, enabled);
        if let Some(texture) = self.item_builtin_texture.as_ref() {
            reupload_first_person_texture(gpu, texture, enabled);
        }
    }

    /// Set the transform for subsequent first-person uploads without touching resident GPU
    /// bytes. Use [`Self::set_anaglyph_with_gpu`] when an already-installed phase must change
    /// immediately.
    pub fn set_anaglyph(&mut self, enabled: bool) {
        self.anaglyph = enabled;
    }

    /// Take this frame's `EntityRenderer.updateLightmap` inputs; the table is regenerated only
    /// when they change, like every other lightmap owner.
    pub fn set_lightmap_params(&mut self, params: crate::sky::lightmap::LightmapParams) {
        if self
            .lightmap
            .as_ref()
            .is_some_and(|(current, _)| *current == params)
        {
            return;
        }
        let table = crate::sky::lightmap::Lightmap::generate_cached(params);
        self.lightmap = Some((params, table));
    }

    /// The lightmap texel `setLightMapFromPlayer` selects: `lightmapColors[sky][block]` at the
    /// eye. Java's coordinates land on texel centres, so linear filtering returns the texel.
    /// The texture is uploaded through `TextureUtil`, so anaglyph mode converts it too.
    fn eye_lightmap_color(&self) -> [f32; 3] {
        let Some((_, table)) = &self.lightmap else {
            return [1.0; 3];
        };
        let sky = ((self.combined_light >> 20) & 0xF) as u8;
        let block = ((self.combined_light >> 4) & 0xF) as u8;
        let mut texel = table.sample(sky, block);
        mc_assets::sprite::apply_anaglyph_rgba(&mut texel, self.anaglyph);
        [
            f32::from(texel[0]) / 255.0,
            f32::from(texel[1]) / 255.0,
            f32::from(texel[2]) / 255.0,
        ]
    }

    /// Bake `RenderHelper.enableStandardItemLighting` and `setLightMapFromPlayer` into the hand's
    /// freshly rebuilt vertex colours. The shader multiplies the texel by the vertex colour, which
    /// is exactly where GL's lit primary colour and lightmap stage land. The vertices are
    /// re-emitted and re-uploaded every frame anyway, so this costs one pass over them.
    fn apply_hand_lighting(&mut self, cam: &Camera) {
        let lights = first_person_light_directions(
            cam.pitch,
            cam.yaw,
            self.prev_render_arm_pitch,
            self.render_arm_pitch,
            self.prev_render_arm_yaw,
            self.render_arm_yaw,
            self.partial_ticks,
        );
        let lightmap = self.eye_lightmap_color();
        for vertex in self.item_vertices.iter_mut().chain(&mut self.arm_vertices) {
            vertex.color = first_person_lit_color(vertex.color, vertex.normal, lights, lightmap);
        }
        // java: `renderItemMap` sets `glNormal3f(0, 0, -1)` and its quads carry no normal, so GL
        // lights the sheet, the map image and the icons with that normal: the opposite of the
        // `+Z` the emitted quads store.
        for vertex in self
            .map_background_vertices
            .iter_mut()
            .chain(&mut self.map_vertices)
            .chain(&mut self.map_icon_vertices)
        {
            let [x, y, z] = vertex.normal;
            vertex.color = first_person_lit_color(vertex.color, [-x, -y, -z], lights, lightmap);
        }
        // java: `RenderItem.renderEffect` disables lighting for the foil, but the lightmap texture
        // stage stays enabled for the whole of `renderItemInFirstPerson`.
        for vertex in &mut self.item_glint_vertices {
            for (channel, light) in vertex.color.iter_mut().zip(lightmap) {
                *channel *= light;
            }
        }
    }

    /// Replace the player skin texture. `width * height * 4` must equal `rgba.len()`.
    pub fn set_skin_rgba(&mut self, gpu: &Gpu, width: u32, height: u32, rgba: &[u8]) {
        if width == 0 || height == 0 || !rgba_len_matches(width, height, rgba) {
            return;
        }
        self.skin = make_first_person_texture(
            gpu,
            &self.texture_layout,
            "first-person player skin",
            width,
            height,
            rgba,
            self.anaglyph,
        );
        self.skin_texture_key = None;
    }

    /// Replace the player skin only when its cache/resource identity changed.
    ///
    /// The live first-person driver checks the asynchronous skin cache every frame. Keeping the
    /// key here makes that readiness poll cheap after the texture is resident.
    pub fn set_skin_rgba_keyed(
        &mut self,
        gpu: &Gpu,
        key: &str,
        width: u32,
        height: u32,
        rgba: &[u8],
    ) {
        if width == 0
            || height == 0
            || !rgba_len_matches(width, height, rgba)
            || self.skin_texture_key.as_deref() == Some(key)
        {
            return;
        }
        self.skin = make_first_person_texture(
            gpu,
            &self.texture_layout,
            "first-person player skin",
            width,
            height,
            rgba,
            self.anaglyph,
        );
        self.skin_texture_key = Some(key.to_owned());
    }

    /// Load an Alex/Steve fallback skin from the active resource stack if it is not installed.
    pub fn load_skin_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
        path: &str,
    ) -> bool {
        if self.skin_texture_key.as_deref() == Some(path) {
            return true;
        }
        let Some(texture) = load_first_person_texture(
            gpu,
            &self.texture_layout,
            resources,
            path,
            "first-person player skin",
            self.anaglyph,
        ) else {
            return false;
        };
        self.skin = texture;
        self.skin_texture_key = Some(path.to_owned());
        true
    }

    /// Load vanilla's held-map backing and icon-sheet textures from the active resource pack.
    /// Missing or malformed optional resources leave the one-pixel fallback installed, while
    /// map pixels themselves still come from the live `MapData` registry.
    pub fn load_map_textures(&mut self, gpu: &Gpu, resources: &mc_assets::ResourceManager) {
        if let Some(texture) = load_first_person_texture(
            gpu,
            &self.texture_layout,
            resources,
            "textures/map/map_background.png",
            "first-person map background",
            self.anaglyph,
        ) {
            self.map_background = texture;
        }
        if let Some(texture) = load_first_person_texture(
            gpu,
            &self.texture_layout,
            resources,
            "textures/map/map_icons.png",
            "first-person map icons",
            self.anaglyph,
        ) {
            self.map_icons = texture;
        }
    }

    /// Load vanilla's standalone enchanted-item foil texture.  Its animated UVs are intentionally
    /// unbounded, so this uses a repeat sampler; a missing/malformed optional resource keeps the
    /// opaque fallback rather than disabling the held item path.
    pub fn load_glint_texture(&mut self, gpu: &Gpu, resources: &mc_assets::ResourceManager) {
        // Read the PNG with its `.mcmeta` and sample it the way `SimpleTexture`
        // would (vanilla's foil is `blur: true`, i.e. linear): see [`glint_sampler_modes`].
        let location =
            mc_assets::ResourceLocation::minecraft("textures/misc/enchanted_item_glint.png");
        let Ok((rgba, width, height, options)) = resources.load_png_rgba(&location) else {
            return;
        };
        let (filter, address) = glint_sampler_modes(&options);
        self.glint = make_first_person_texture_sampled(
            gpu,
            &self.texture_layout,
            "first-person item glint",
            width,
            height,
            vec![rgba],
            address,
            filter,
            self.anaglyph,
        );
    }

    /// Set or clear the latched baked item model. `gui_3d` is Java's `IBakedModel.isGui3d` bit,
    /// which controls `RenderItem.preTransform`: flat models receive an outer `2.0` before the
    /// universal inner `0.5` render scale, while GUI-3D models receive only that inner half.
    pub fn set_item_model(&mut self, model: Option<BakedModel>, gui_3d: bool) {
        // A held bow or rod is re-latched every tick: while the option is on,
        // keep the 1.7-edged copy as long as the model itself is the same.
        let same = self.old_item_edges
            && match (&self.item_model, &model) {
                (Some(old), Some(new)) => {
                    old.general_quads == new.general_quads
                        && old.face_quads == new.face_quads
                        && old.transforms == new.transforms
                }
                _ => false,
            };
        if !same {
            self.item_model_old = None;
        }
        self.item_model = model;
        self.item_gui_3d = gui_3d;
        self.item_translucent_block = false;
        self.item_tint_snapshot.clear();
        self.item_builtin = None;
        self.item_builtin_texture = None;
        self.item_builtin_texture_key = None;
        self.item_glint_enabled = false;
    }

    /// Set the latched model and decoded descriptor for a `builtin/entity` held item.
    ///
    /// `BuiltInModel.isGui3d()` is always true, even though its baked quad list is empty.  Keep
    /// the sentinel model here so its resolved `display.firstperson` transform remains available
    /// to [`Self::rebuild_vertices`], while the descriptor selects the concrete TESR geometry.
    /// Vanilla's `RenderItem.renderItem` does not call `renderEffect` in this branch, so this
    /// setter also clears any stale baked-item foil state.
    pub fn set_builtin_item_model(&mut self, model: BakedModel, item: BuiltinTileItem) {
        self.item_model = Some(model);
        self.item_model_old = None;
        self.item_gui_3d = true;
        self.item_translucent_block = false;
        self.item_tint_snapshot.clear();
        self.item_builtin = Some(item);
        self.item_builtin_texture = None;
        self.item_builtin_texture_key = None;
        self.item_glint_enabled = false;
    }

    /// Return the currently latched builtin/entity descriptor.
    #[must_use]
    pub fn builtin_item(&self) -> Option<&BuiltinTileItem> {
        self.item_builtin.as_ref()
    }

    /// Replace only the builtin descriptor when an asynchronous/local profile becomes complete.
    ///
    /// The sentinel baked model and its first-person transform do not change when a legacy
    /// `SkullOwner` name is upgraded from Steve to a tab-list profile.  Resetting only the
    /// descriptor's texture handoff avoids rebuilding geometry or uploading a GPU resource until
    /// the next normal `ensure_builtin_item_texture` call.
    pub fn update_builtin_item(&mut self, item: BuiltinTileItem) {
        if self.item_builtin.as_ref() == Some(&item) {
            return;
        }
        self.item_builtin = Some(item);
        self.item_builtin_texture = None;
        self.item_builtin_texture_key = None;
        self.item_glint_enabled = false;
    }

    /// Ensure the current builtin/entity item has an entity-sheet or banner-composite texture.
    ///
    /// The render phase cannot borrow the resource manager from `RenderPhase::record`, so the
    /// client calls this once per frame before recording.  Fixed sheets use their vanilla
    /// resource paths; banners use the shared CPU `LayeredColorMaskTexture` compositor.  Missing
    /// or malformed resources retain the same opaque magenta checkerboard fallback as the tile
    /// phase rather than binding the unrelated block atlas.
    pub fn ensure_builtin_item_texture(
        &mut self,
        gpu: &Gpu,
        resources: &mc_assets::ResourceManager,
    ) {
        let Some(item) = self.item_builtin.clone() else {
            self.item_builtin_texture = None;
            self.item_builtin_texture_key = None;
            return;
        };
        // Keep the shared-pixel lookup beside the key
        // decision.  This is the no-GPU seam used by tests and ensures a completed skin can
        // replace the Steve/Alex key on the very next frame.
        let (key, shared_pixels) =
            first_person_builtin_texture_source(&item, self.item_builtin_models.christmas);
        if self.item_builtin_texture_key.as_ref() == key.as_ref() {
            return;
        }
        self.item_builtin_texture_key = key.clone();
        let loaded = match &item {
            BuiltinTileItem::Banner { state } => crate::entity::tilepass::banner_texture_pixels(
                resources, state,
            )
            .map(|(_key, pixels, width, height)| {
                make_first_person_texture(
                    gpu,
                    &self.texture_layout,
                    "first-person builtin banner",
                    width,
                    height,
                    &pixels,
                    self.anaglyph,
                )
            }),
            // A profile skull's resident key is owned by the shared skin cache rather than the
            // resource pack.  Consume the already-normalised pixels here; before they arrive,
            // BuiltinSkullOwner::texture_key selects its ordinary Steve/Alex fallback path.
            BuiltinTileItem::PlayerSkull { .. } => shared_pixels
                .map(|skin| {
                    make_first_person_texture(
                        gpu,
                        &self.texture_layout,
                        "first-person builtin player skull",
                        crate::entity::SkinTexture::WIDTH,
                        crate::entity::SkinTexture::HEIGHT,
                        &skin.rgba,
                        self.anaglyph,
                    )
                })
                .or_else(|| {
                    key.as_deref().and_then(|path| {
                        load_first_person_texture(
                            gpu,
                            &self.texture_layout,
                            resources,
                            path,
                            "first-person builtin item",
                            self.anaglyph,
                        )
                    })
                }),
            _ => key.as_deref().and_then(|path| {
                load_first_person_texture(
                    gpu,
                    &self.texture_layout,
                    resources,
                    path,
                    "first-person builtin item",
                    self.anaglyph,
                )
            }),
        };
        self.item_builtin_texture = key.map(|_| {
            loaded.unwrap_or_else(|| {
                let pixels = missing_texture_pixels();
                make_first_person_texture(
                    gpu,
                    &self.texture_layout,
                    "first-person builtin missing",
                    MISSING_TEXTURE_SIZE,
                    MISSING_TEXTURE_SIZE,
                    &pixels,
                    self.anaglyph,
                )
            })
        });
    }

    /// Latch `ItemStack.hasEffect()` alongside the equipped model.  The driver calls this only
    /// when Java's strict equipped-progress threshold swaps `itemToRender`, so an outgoing model
    /// cannot accidentally borrow the incoming stack's foil predicate during its lower animation.
    pub fn set_item_glint_enabled(&mut self, enabled: bool) {
        self.item_glint_enabled = enabled;
    }

    /// Latch the held stack's actual block render layer beside its equipped model.
    ///
    /// Java disables depth writes only when the stack maps to a translucent block *and* the
    /// selected baked model is GUI-3D.  Keeping the registry answer here lets the record path
    /// apply that second condition without conflating material with model shape.
    pub fn set_item_translucent_block(&mut self, translucent: bool) {
        self.item_translucent_block = translucent;
    }

    /// Latch the stack-aware RGB result for each possible `BakedQuad.tint_index`.
    /// Missing indices intentionally resolve to white in [`Self::rebuild_vertices`].
    pub fn set_item_tint_snapshot(&mut self, colors: Vec<u32>) {
        self.item_tint_snapshot = colors;
    }

    /// Set the live filled-map snapshot consumed by the held-item branch.
    ///
    /// `rgba` is the already palette-resolved 128x128 image from `mc-ui`; `None` represents a
    /// filled map whose `map_<id>` record has not arrived yet. Java still draws the map backing in
    /// that case, but `MapItemRenderer.renderMap` has no image to bind. Invalid pixel lengths are
    /// rejected without changing the previous snapshot.
    pub fn set_item_map(
        &mut self,
        map_id: i32,
        rgba: Option<&[u8]>,
        decorations: &[FirstPersonMapDecoration],
    ) {
        const MAP_BYTES: usize = 128 * 128 * 4;
        if rgba.is_some_and(|pixels| pixels.len() != MAP_BYTES) {
            return;
        }
        // Compare first: the 64 KiB image and the decorations are copied
        // only when this tick's snapshot differs from the latched one (the same test as before,
        // `item_map != Some(map)`, without building `map` every tick).
        let unchanged = self.item_map.as_ref().is_some_and(|map| {
            map.id == map_id
                && map.rgba.as_deref() == rgba
                && map.decorations.as_slice() == decorations
        });
        if unchanged {
            return;
        }
        let map = FirstPersonMap {
            id: map_id,
            rgba: rgba.map(ToOwned::to_owned),
            decorations: decorations.to_vec(),
        };
        self.map_texture_dirty = map.rgba.is_some();
        self.item_map = Some(map);
    }

    /// Clear the held filled-map branch and return to either a baked item or the empty arm.
    pub fn clear_item_map(&mut self) {
        self.item_map = None;
        self.map_texture_dirty = false;
    }

    /// Clear the latched held item and return to the empty-arm path.
    pub fn clear_item_model(&mut self) {
        self.item_model = None;
        self.item_model_old = None;
        self.item_translucent_block = false;
        self.item_tint_snapshot.clear();
        self.item_builtin = None;
        self.item_builtin_texture = None;
        self.item_builtin_texture_key = None;
        self.item_glint_enabled = false;
    }

    /// Borrow the currently latched baked model for debug/HUD inspection.
    #[must_use]
    pub fn item_model(&self) -> Option<&BakedModel> {
        self.item_model.as_ref()
    }

    /// Copy the tick-driven held-item state ([`FirstPersonHeld`]) for a replay
    /// checkpoint. Called once per checkpoint, never per frame.
    #[must_use]
    pub fn held_snapshot(&self) -> FirstPersonHeld {
        FirstPersonHeld {
            render_state: self.render_state,
            item_model: self.item_model.clone(),
            item_builtin: self.item_builtin.clone(),
            item_map: self.item_map.clone(),
            item_gui_3d: self.item_gui_3d,
            item_translucent_block: self.item_translucent_block,
            item_tint_snapshot: self.item_tint_snapshot.clone(),
            item_glint_enabled: self.item_glint_enabled,
        }
    }

    /// Put a [`Self::held_snapshot`] back. The GPU-side copies follow on the next
    /// frame as after any equip swap: the vertices are rebuilt every frame, the builtin texture
    /// by [`Self::ensure_builtin_item_texture`], the 1.7-edged copy on first use, and a held map's
    /// image is uploaded again.
    pub fn restore_held(&mut self, held: FirstPersonHeld) {
        self.render_state = held.render_state;
        self.item_model = held.item_model;
        self.item_model_old = None;
        self.item_builtin = held.item_builtin;
        self.item_builtin_texture = None;
        self.item_builtin_texture_key = None;
        self.map_texture_dirty = held.item_map.as_ref().is_some_and(|map| map.rgba.is_some());
        self.item_map = held.item_map;
        self.item_gui_3d = held.item_gui_3d;
        self.item_translucent_block = held.item_translucent_block;
        self.item_tint_snapshot = held.item_tint_snapshot;
        self.item_glint_enabled = held.item_glint_enabled;
    }

    /// Switch between classic (4-pixel) and Alex/slim (3-pixel) first-person arms.
    pub fn set_small_arms(&mut self, small_arms: bool) {
        if self.player_model.small_arms != small_arms {
            self.player_model = PlayerModel::new(0.0, small_arms);
        }
    }

    /// Apply the local player's `EnumPlayerModelParts.RIGHT_SLEEVE` visibility to the empty-arm
    /// model. Java's `RenderPlayer.renderRightArm` copies this model-part preference before the
    /// arm display list is replayed.
    pub fn set_right_sleeve_visible(&mut self, visible: bool) {
        let sleeve = self.player_model.wear.right_arm;
        self.player_model.biped.model.get_mut(sleeve).show_model = visible;
    }

    /// Whether this phase currently has a latched item model.
    #[must_use]
    pub fn has_item_model(&self) -> bool {
        self.item_model.is_some()
    }

    /// Whether the latched model was identified as a vanilla GUI-3D item/block model.
    #[must_use]
    pub fn item_is_gui_3d(&self) -> bool {
        self.item_gui_3d
    }

    fn rebuild_vertices(&mut self, pitch: f32, system_time_ms: u64) {
        let equip = self.render_state.equip_amount(self.partial_ticks);
        let swing = self.render_state.swing_amount(self.partial_ticks);

        // Java's RenderPlayer.renderRightArm/renderLeftArm resets the shared ModelPlayer with
        // setRotationAngles(0, 0, 0, 0, 0, 0.0625, entity) before replaying either arm. In
        // particular this supplies the idle roll and copies each base arm's pivot/angles to its
        // wear part (the constructors intentionally start wear at z=10 for normal entity draws).
        // The first-person phase emits those parts directly, so perform that same model-side
        // reset before both the ordinary and map arm branches.
        self.player_model.set_rotation_angles(&AnimParams::still());

        self.item_vertices.clear();
        self.item_glint_vertices.clear();
        self.item_cull_face = None;
        // 1.7 item edges (off: one `bool`).
        self.item_solid = false;
        self.prepare_old_item_edges();
        self.arm_vertices.clear();
        self.map_background_vertices.clear();
        self.map_vertices.clear();
        self.map_icon_vertices.clear();
        if let Some(map) = &self.item_map {
            // java: `ItemRenderer.renderItemMap` calls `renderPlayerArms` before the final map
            // swing/scale transforms, so both arm meshes use the first matrix while the sheet
            // geometry uses the second one.
            let arm_matrix = first_person_map_arm_matrix(equip, swing, pitch);
            let player = &self.player_model;
            let right_arm = player.biped.parts.right_arm;
            let left_arm = player.biped.parts.left_arm;
            let right_wear = player.wear.right_arm;
            let left_wear = player.wear.left_arm;
            // 3D Skins: both sleeves from the voxel mesh.
            let sleeves = self
                .skin3d_sleeve
                .as_deref()
                .filter(|mesh| mesh.slim == player.small_arms);
            player
                .biped
                .model
                .emit_part(right_arm, 0.0625, arm_matrix, &mut self.arm_vertices);
            emit_first_person_sleeve(
                player,
                right_wear,
                sleeves.map(|mesh| mesh.part(crate::entity::skin3d::WearPart::RightSleeve)),
                arm_matrix,
                &mut self.arm_vertices,
            );
            player
                .biped
                .model
                .emit_part(left_arm, 0.0625, arm_matrix, &mut self.arm_vertices);
            emit_first_person_sleeve(
                player,
                left_wear,
                sleeves.map(|mesh| mesh.part(crate::entity::skin3d::WearPart::LeftSleeve)),
                arm_matrix,
                &mut self.arm_vertices,
            );
            push_first_person_map_geometry(
                map,
                first_person_map_matrix(equip, swing, pitch),
                &mut self.map_background_vertices,
                &mut self.map_vertices,
                &mut self.map_icon_vertices,
            );
        } else if let (Some(item), Some(model), Some(_texture)) = (
            &self.item_builtin,
            &self.item_model,
            &self.item_builtin_texture,
        ) {
            let pose = first_person_item_pose(
                equip,
                swing,
                self.render_state.use_action,
                self.render_state.item_use_count,
                self.render_state.max_item_use_duration,
                self.partial_ticks,
            );
            let model_transform =
                pose * first_person_builtin_item_transform(&model.transforms.first_person);
            let _ = self.item_builtin_models.emit_builtin_item(
                item,
                model_transform,
                &mut self.item_vertices,
            );
        } else if let Some(model) = &self.item_model {
            // 1.7 item edges: the 1.7-edged copy, drawn without blending.
            let model = match &self.item_model_old {
                Some(Some(old)) if self.old_item_edges_apply() => {
                    self.item_solid = true;
                    old
                }
                _ => model,
            };
            let pose = first_person_item_pose(
                equip,
                swing,
                self.render_state.use_action,
                self.render_state.item_use_count,
                self.render_state.max_item_use_duration,
                self.partial_ticks,
            );
            // "1.7 Animations". `None` hands `pose` back untouched (vanilla).
            let pose = old_animations::held_item_pose(
                self.old_animations,
                pose,
                equip,
                swing,
                &self.render_state,
                self.partial_ticks,
                self.item_gui_3d,
            );
            // Java's ItemRenderer wrapper or RenderItem preTransform supplies the outer scale
            // (the branch depends on gui_3d), then the display transform and RenderItem's
            // unconditional scale(0.5) run immediately before the -0.5 recenter. Keep that
            // ordering so a display translation/rotation observes the same matrix stack.
            let model_transform = pose
                * first_person_item_model_transform(
                    self.item_gui_3d,
                    &model.transforms.first_person,
                );
            emit_first_person_item_model(
                model,
                model_transform,
                &self.item_tint_snapshot,
                &mut self.item_vertices,
            );
            // Only the drawn bow's pose is not a similarity.
            apply_gl_rescale_normals(model_transform, &mut self.item_vertices);
            self.item_cull_face = Some(first_person_item_cull_face(&model.transforms.first_person));
            if self.item_glint_enabled {
                emit_baked_model_glint(
                    model,
                    model_transform,
                    system_time_ms,
                    &mut self.item_glint_vertices,
                );
            }
        } else if self.render_empty_arm {
            let arm_matrix = first_person_arm_matrix(equip, swing);
            let player = &self.player_model;
            let arm = player.biped.parts.right_arm;
            let wear = player.wear.right_arm;
            player
                .biped
                .model
                .emit_part(arm, 0.0625, arm_matrix, &mut self.arm_vertices);
            // 3D Skins: the sleeve from the voxel mesh when set.
            let sleeve = self
                .skin3d_sleeve
                .as_deref()
                .filter(|mesh| mesh.slim == player.small_arms)
                .map(|mesh| mesh.part(crate::entity::skin3d::WearPart::RightSleeve));
            emit_first_person_sleeve(player, wear, sleeve, arm_matrix, &mut self.arm_vertices);
        }
    }
}

/// One first-person sleeve: the flat `*Armwear` box (vanilla), or the
/// 3D Skins voxel layer under the same part matrix. Unculled: the hand's vertex space is not
/// camera-centred (the view-bobbing transform follows on the GPU), and it is one part.
fn emit_first_person_sleeve(
    player: &PlayerModel,
    wear: crate::entity::PartId,
    voxels: Option<&[crate::entity::skin3d::VoxelQuad]>,
    arm_matrix: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    match voxels {
        None => player.biped.model.emit_part(wear, 0.0625, arm_matrix, out),
        Some(quads) => {
            let part = player.biped.model.get(wear);
            if !part.is_hidden && part.show_model {
                crate::entity::skin3d::emit_quads_unculled(
                    quads,
                    arm_matrix * part.local_matrix(0.0625),
                    0.0625,
                    out,
                );
            }
        }
    }
}

/// Fixed-function GL's normal transform under `GL_RESCALE_NORMAL`, or
/// `None` when `model` is a similarity (rotation, reflection and uniform scale only).
///
/// java: `ItemRenderer.renderItemInFirstPerson` and `RenderItem.renderItemModelTransform` call
/// `enableRescaleNormal`; GL then lights each quad's face normal (`RenderItem.putQuadNormal`) as
/// `n' = f (M^-1)^T n` with `f = 1 / |row 3 of M^-1|` (OpenGL 2.1, 2.11.3), and `GL_NORMALIZE` is
/// off. For a similarity that is exactly the normalised forward transform the shared emitter
/// writes. Under the drawn bow's `scale(1, 1, 1 + f1 * 0.2)` (`doBowTransformations`) it is
/// neither the forward direction nor unit length, and the bow's faces light accordingly. Only
/// the item's own matrix matters: the hurt, bob and arm-sway rotations in front of it leave
/// `|row 3 of M^-1|` unchanged.
#[must_use]
pub fn gl_rescale_normal_matrix(model: Mat4) -> Option<glam::Mat3> {
    let m = glam::Mat3::from_mat4(model);
    let lengths = [m.x_axis.length(), m.y_axis.length(), m.z_axis.length()];
    let scale = lengths[0].max(lengths[1]).max(lengths[2]);
    if !scale.is_finite() || scale <= f32::EPSILON {
        return None;
    }
    let tolerance = scale * 1e-5;
    let similar = lengths
        .iter()
        .all(|length| (length - lengths[0]).abs() <= tolerance)
        && m.x_axis.dot(m.y_axis).abs() <= tolerance * scale
        && m.y_axis.dot(m.z_axis).abs() <= tolerance * scale
        && m.z_axis.dot(m.x_axis).abs() <= tolerance * scale;
    if similar || m.determinant().abs() <= f32::EPSILON * scale * scale * scale {
        return None;
    }
    let inverse = m.inverse();
    let factor = 1.0 / inverse.row(2).length();
    Some(inverse.transpose() * factor)
}

/// Re-derive emitted held-item normals with [`gl_rescale_normal_matrix`]. The emitter stored
/// `normalize(M n)`; for a face normal `n` that is recovered as `normalize(M^-1 stored)`.
fn apply_gl_rescale_normals(model: Mat4, vertices: &mut [EntityVertex]) {
    let Some(normal_matrix) = gl_rescale_normal_matrix(model) else {
        return;
    };
    let forward_inverse = glam::Mat3::from_mat4(model).inverse();
    for vertex in vertices {
        let face = (forward_inverse * Vec3::from(vertex.normal)).normalize_or_zero();
        vertex.normal = (normal_matrix * face).into();
    }
}

/// CPU half of the first-person baked-item path. Keeping the palette lookup here makes the
/// retained phase's final vertex colour testable without constructing a GPU.
fn emit_first_person_item_model(
    model: &BakedModel,
    transform: Mat4,
    tint_snapshot: &[u32],
    out: &mut Vec<EntityVertex>,
) {
    emit_baked_model_tinted(model, transform, out, |index| {
        tint_snapshot
            .get(usize::from(index))
            .copied()
            .unwrap_or(0xFF_FF_FF)
    });
}

/// Select the first-person builtin texture key and the already-normalized player-skull pixels.
///
/// This pure handoff keeps resource/GPU work in [`FirstPersonPhase::ensure_builtin_item_texture`]
/// while making the important readiness boundary testable: before `SkinCache` publishes a skin,
/// [`BuiltinSkullOwner::texture_key`] is the UUID/legacy Steve-Alex fallback; after publication,
/// the owner cache key and its 64x64 pixels are selected together.
#[must_use]
fn first_person_builtin_texture_source(
    item: &BuiltinTileItem,
    christmas: bool,
) -> (Option<String>, Option<crate::entity::SkinTexture>) {
    let key = item.texture_key(christmas);
    let shared_pixels = match item {
        BuiltinTileItem::PlayerSkull { owner, .. } => {
            crate::entity::skin::shared_skin(&owner.cache_key)
        }
        _ => None,
    };
    (key, shared_pixels)
}

impl FirstPersonPhase {
    /// Whether `renderHand` draws a hand, item or map this frame (the phase's whole job before
    /// [`Self::clear_depth_for_overlay`] existed).
    fn hand_enabled(&self) -> bool {
        self.render_state.visibility.renders_hand()
            && (self.item_map.is_some()
                || self.item_model.is_some()
                || (self.render_empty_arm && self.render_state.visibility.renders_empty_arm()))
    }
}

// 1.7 Animations' "1.7 item edges".
impl FirstPersonPhase {
    /// Whether the option applies to the latched item: on, and a flat `builtin/generated` model
    /// that is not a translucent block (stained glass panes keep vanilla's blending, as in 1.7).
    fn old_item_edges_apply(&self) -> bool {
        self.old_item_edges
            && !self.item_gui_3d
            && !self.item_translucent_block
            && self.item_builtin.is_none()
            && self.item_map.is_none()
    }

    /// Build the 1.7-edged copy of the latched item once, when the option needs it; drop it and
    /// the solid pipelines when the option is off. Off, this is one `bool` and two `Option` tests.
    fn prepare_old_item_edges(&mut self) {
        if !self.old_item_edges {
            if self.item_model_old.is_some() || self.solid_item_pipelines.is_some() {
                self.item_model_old = None;
                self.solid_item_pipelines = None;
            }
            return;
        }
        if self.item_model_old.is_some() || !self.old_item_edges_apply() {
            return;
        }
        let Some(model) = &self.item_model else {
            return;
        };
        let edged = {
            let levels = self.atlas.source_levels.borrow();
            levels.first().and_then(|atlas| {
                old_item_edges::old_edged_model(model, atlas, self.atlas.width, self.atlas.height)
            })
        };
        self.item_model_old = Some(edged);
    }

    /// The held-item pipelines without blending, `[GL_BACK, GL_FRONT]` culled.
    fn make_solid_item_pipelines(&self, gpu: &Gpu) -> [wgpu::RenderPipeline; 2] {
        [wgpu::Face::Back, wgpu::Face::Front].map(|face| {
            make_first_person_pipeline_with_cull(
                gpu,
                &self.item_pipeline_layout,
                &self.item_shader,
                "first-person held item (1.7 edges)",
                "fs_solid",
                first_person_blend_state(),
                true,
                wgpu::CompareFunction::LessEqual,
                Some(face),
            )
        })
    }

    /// Whether the last recorded frame drew the held item with 1.7's edges, without blending.
    #[must_use]
    pub fn item_drawn_with_old_edges(&self) -> bool {
        self.item_solid
    }

    /// Whether the no-blend pipelines exist (from the first 1.7-edged frame until the option
    /// goes off).
    #[must_use]
    pub fn old_edge_pipelines_built(&self) -> bool {
        self.solid_item_pipelines.is_some()
    }
}

impl RenderPhase for FirstPersonPhase {
    fn name(&self) -> &'static str {
        "first-person"
    }

    fn enabled(&self) -> bool {
        self.hand_enabled() || self.clear_depth_for_overlay
    }

    fn record(
        &mut self,
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        cam: &Camera,
    ) {
        if !self.hand_enabled() {
            // java: the depth clear before `renderHand` still happens.
            if self.clear_depth_for_overlay {
                drop(gpu.begin_pass(enc, view, "first-person", DepthAttachment::Clear));
            }
            return;
        }
        // Java's `RenderItem.renderEffect` samples `Minecraft.getSystemTime()` while the held
        // model is being replayed.  One frame-domain sample keeps both passes coherent and matches
        // the entity foil seam's clock policy.
        let glint_time_ms = item_glint_system_time_ms();
        self.rebuild_vertices(cam.pitch, glint_time_ms);
        self.apply_hand_lighting(cam);
        if self.item_vertices.is_empty()
            && self.arm_vertices.is_empty()
            && self.map_background_vertices.is_empty()
        {
            // Nothing to draw, but the overlay still needs the clear.
            if self.clear_depth_for_overlay {
                drop(gpu.begin_pass(enc, view, "first-person", DepthAttachment::Clear));
            }
            return;
        }

        if !self.item_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.item_vertex_buffer,
                &mut self.item_vertex_capacity,
                &self.item_vertices,
                "first-person item vertices",
            );
        }
        if !self.item_glint_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.item_glint_vertex_buffer,
                &mut self.item_glint_vertex_capacity,
                &self.item_glint_vertices,
                "first-person item glint vertices",
            );
        }
        if !self.arm_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.arm_vertex_buffer,
                &mut self.arm_vertex_capacity,
                &self.arm_vertices,
                "first-person arm vertices",
            );
        }
        if !self.map_background_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.map_background_vertex_buffer,
                &mut self.map_background_vertex_capacity,
                &self.map_background_vertices,
                "first-person map background vertices",
            );
        }
        if !self.map_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.map_vertex_buffer,
                &mut self.map_vertex_capacity,
                &self.map_vertices,
                "first-person map vertices",
            );
        }
        if !self.map_icon_vertices.is_empty() {
            ensure_first_person_vertex_buffer(
                gpu,
                &mut self.map_icon_vertex_buffer,
                &mut self.map_icon_vertex_capacity,
                &self.map_icon_vertices,
                "first-person map icon vertices",
            );
        }
        if self.map_texture_dirty {
            if let Some(pixels) = self.item_map.as_ref().and_then(|map| map.rgba.as_deref()) {
                self.map_texture = make_first_person_texture(
                    gpu,
                    &self.texture_layout,
                    "first-person map",
                    128,
                    128,
                    pixels,
                    self.anaglyph,
                );
            }
            self.map_texture_dirty = false;
        }

        // Model/arm poses are baked into their geometry. The camera's separate renderHand
        // projection and arm-rotation delta stay in this transform uniform, retaining proper
        // perspective interpolation in the vertex stage.
        let camera_matrix = first_person_camera_matrix(
            cam,
            self.view_bobbing,
            self.prev_render_arm_pitch,
            self.render_arm_pitch,
            self.prev_render_arm_yaw,
            self.render_arm_yaw,
            self.partial_ticks,
        );
        let transform = FirstPersonTransformUniform {
            transform: camera_matrix.to_cols_array_2d(),
        };
        // Both buffers are read only by this phase's passes.
        self.item_transform_write.write_for_frame(
            gpu,
            &self.item_transform_buffer,
            bytemuck::bytes_of(&transform),
        );
        self.arm_transform_write.write_for_frame(
            gpu,
            &self.arm_transform_buffer,
            bytemuck::bytes_of(&transform),
        );

        // `renderHand` begins with `glClear(GL_DEPTH_BUFFER_BIT)`; Clear retains the world colour
        // while resetting the shared depth so the base item can write its own surface for the
        // exact-equal foil replay.
        // 1.7 item edges: the no-blend pipelines, built on first use.
        if self.item_solid && self.solid_item_pipelines.is_none() {
            self.solid_item_pipelines = Some(self.make_solid_item_pipelines(gpu));
        }
        let mut pass = gpu.begin_pass(enc, view, "first-person", DepthAttachment::Clear);
        pass.set_bind_group(0, gpu.camera_bind_group(), &[]);
        let map_active = !self.map_background_vertices.is_empty();
        if map_active {
            // `renderPlayerArms` is emitted before the map sheet in Java's matrix stack.
            if !self.arm_vertices.is_empty() {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(1, &self.arm_transform_group, &[]);
                pass.set_bind_group(2, &self.skin.bind_group, &[]);
                pass.set_vertex_buffer(0, self.arm_vertex_buffer.slice(..));
                pass.draw(0..self.arm_vertices.len() as u32, 0..1);
            }
            pass.set_pipeline(&self.map_pipeline);
            pass.set_bind_group(1, &self.item_transform_group, &[]);
            pass.set_bind_group(2, &self.map_background.bind_group, &[]);
            pass.set_vertex_buffer(0, self.map_background_vertex_buffer.slice(..));
            pass.draw(0..self.map_background_vertices.len() as u32, 0..1);
            if !self.map_vertices.is_empty() {
                pass.set_bind_group(2, &self.map_texture.bind_group, &[]);
                pass.set_vertex_buffer(0, self.map_vertex_buffer.slice(..));
                pass.draw(0..self.map_vertices.len() as u32, 0..1);
            }
            if !self.map_icon_vertices.is_empty() {
                pass.set_bind_group(2, &self.map_icons.bind_group, &[]);
                pass.set_vertex_buffer(0, self.map_icon_vertex_buffer.slice(..));
                pass.draw(0..self.map_icon_vertices.len() as u32, 0..1);
            }
        } else {
            if !self.item_vertices.is_empty() {
                let culled = match self.item_cull_face {
                    Some(wgpu::Face::Back) => Some(&self.item_back_pipelines),
                    Some(wgpu::Face::Front) => Some(&self.item_front_pipelines),
                    None => None,
                };
                let writes_depth =
                    first_person_item_writes_depth(self.item_translucent_block, self.item_gui_3d);
                let item_pipeline = match (culled, writes_depth) {
                    (Some(culled), true) => &culled.base,
                    (Some(culled), false) => &culled.translucent,
                    (None, true) => &self.pipeline,
                    (None, false) => &self.translucent_item_pipeline,
                };
                // 1.7 item edges: the same state, without blending.
                let solid = self.item_solid && writes_depth;
                let item_pipeline = match (&self.solid_item_pipelines, self.item_cull_face) {
                    (Some([_, front]), Some(wgpu::Face::Front)) if solid => front,
                    (Some([back, _]), Some(_)) if solid => back,
                    _ => item_pipeline,
                };
                pass.set_pipeline(item_pipeline);
                pass.set_bind_group(1, &self.item_transform_group, &[]);
                let item_texture = self.item_builtin_texture.as_ref().unwrap_or(&self.atlas);
                pass.set_bind_group(2, &item_texture.bind_group, &[]);
                pass.set_vertex_buffer(0, self.item_vertex_buffer.slice(..));
                pass.draw(0..self.item_vertices.len() as u32, 0..1);
                if !self.item_glint_vertices.is_empty() {
                    pass.set_pipeline(culled.map_or(&self.glint_pipeline, |culled| &culled.glint));
                    pass.set_bind_group(1, &self.item_transform_group, &[]);
                    pass.set_bind_group(2, &self.glint.bind_group, &[]);
                    pass.set_vertex_buffer(0, self.item_glint_vertex_buffer.slice(..));
                    pass.draw(0..self.item_glint_vertices.len() as u32, 0..1);
                }
            }
            if !self.arm_vertices.is_empty() {
                // The foil and translucent-block paths both change pipeline state.  Empty-arm
                // geometry is ordinary opaque skin and must restore the depth-writing pipeline.
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(1, &self.arm_transform_group, &[]);
                pass.set_bind_group(2, &self.skin.bind_group, &[]);
                pass.set_vertex_buffer(0, self.arm_vertex_buffer.slice(..));
                pass.draw(0..self.arm_vertices.len() as u32, 0..1);
            }
        }
    }
}

/// The shared `@group(0) @binding(0)` uniform, refreshed once per frame before phases record.
///
/// WGSL declaration to paste into a phase's shader:
///
/// ```wgsl
/// struct CameraUniform {
///     view_proj:  mat4x4<f32>,
///     view:       mat4x4<f32>,
///     proj:       mat4x4<f32>,
///     camera_pos: vec4<f32>,   // xyz = eye position demoted to f32, w = effective FOV degrees
///     screen:     vec4<f32>,   // width, height, near, far
///     frame:      vec4<f32>,   // partial_ticks, seconds since start, aspect, sample count
/// };
/// @group(0) @binding(0) var<uniform> camera: CameraUniform;
/// ```
///
/// `view_proj` is **camera-relative**: feed it positions produced by
/// [`Camera::relative`](crate::Camera::relative), never raw world coordinates.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// `proj * view`, rotation-only view — apply to camera-relative positions.
    pub view_proj: [[f32; 4]; 4],
    /// Rotation-only view matrix.
    pub view: [[f32; 4]; 4],
    /// Perspective projection.
    pub proj: [[f32; 4]; 4],
    /// `xyz` = eye position demoted to `f32` (for sky/fog maths only, never for geometry),
    /// `w` = effective FOV in degrees.
    pub camera_pos: [f32; 4],
    /// Viewport width, height, near plane, far plane.
    pub screen: [f32; 4],
    /// `partial_ticks`, seconds since the context was created, aspect ratio, MSAA sample count.
    pub frame: [f32; 4],
}

impl CameraUniform {
    /// Snapshot a camera for upload. `view_proj` and `proj` carry [`Camera::jitter`] (exactly
    /// [`Camera::view_proj`] and [`Camera::proj`] when it is zero); see
    /// [`CameraUniform::new_unjittered`] for the copy the post-world phases get.
    #[must_use]
    pub fn new(cam: &Camera, gpu: &Gpu, partial_ticks: f32, seconds: f32) -> Self {
        let mut uniform = Self::new_unjittered(cam, gpu, partial_ticks, seconds);
        // Only a non-zero jitter touches the matrices: the zero case keeps
        // today's bits (an identity translation could still flip the sign of a zero).
        if cam.jitter != glam::Vec2::ZERO {
            let proj = cam.jittered_proj();
            uniform.view_proj = (proj * cam.view).to_cols_array_2d();
            uniform.proj = proj.to_cols_array_2d();
        }
        uniform
    }

    /// [`CameraUniform::new`] without [`Camera::jitter`]: what the phases at and after
    /// [`PhaseOrder::TEMPORAL_AA`] read while TAA jitters the world.
    #[must_use]
    pub fn new_unjittered(cam: &Camera, gpu: &Gpu, partial_ticks: f32, seconds: f32) -> Self {
        Self {
            view_proj: cam.view_proj().to_cols_array_2d(),
            view: cam.view.to_cols_array_2d(),
            proj: cam.proj.to_cols_array_2d(),
            camera_pos: [
                cam.pos.x as f32,
                cam.pos.y as f32,
                cam.pos.z as f32,
                cam.effective_fov(),
            ],
            screen: [
                gpu.config.width as f32,
                gpu.config.height as f32,
                cam.near,
                cam.far,
            ],
            frame: [
                partial_ticks,
                seconds,
                cam.aspect,
                gpu.sample_count() as f32,
            ],
        }
    }
}

/// Most scopes one frame can time on the GPU. Two query slots each.
///
/// A frame with more enabled phases than this still renders correctly; the extra phases are
/// simply not timed, and [`GpuTimings::untimed_scopes`] counts them so the omission is visible
/// rather than silent.
const MAX_TIMED_SCOPES: usize = 48;
/// How many frames of timestamp results are in flight before one is read back. Three is enough
/// that the CPU never waits on the GPU for a measurement — which would change what is measured.
const READBACK_DEPTH: usize = 3;
/// The synthetic scope wrapping the whole frame's command stream. See [`GpuTimings`].
const GPU_FRAME_SCOPE: &str = "gpu.frame";
/// The synthetic scope for the frame's clear pass, which belongs to no phase.
const GPU_CLEAR_SCOPE: &str = "gpu.clear";
/// The synthetic scope for the end-of-frame blit in scene mode.
const GPU_BLIT_SCOPE: &str = "gpu.scene_blit";
/// The scope of one shared world pass: timestamps are written on the
/// encoder, so the phases inside a shared pass are timed together, under this name.
const GPU_SHARED_SCOPE: &str = "world (shared)";
/// The scope (and pass label) of one shared read-only-depth pass.
const GPU_DEPTH_READ_SCOPE: &str = "depth-read (shared)";

/// What one scope cost **on the GPU**, from timestamp queries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GpuScopeStats {
    /// Frames in which this scope was timed.
    pub calls: u64,
    /// Nanoseconds of GPU time, summed.
    pub total_ns: u64,
    /// The most expensive single frame, in nanoseconds.
    pub max_ns: u64,
}

impl GpuScopeStats {
    /// Mean nanoseconds per frame this scope was timed in.
    #[must_use]
    pub fn mean_ns(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.total_ns as f64 / self.calls as f64
        }
    }
}

/// Per-phase GPU timings for a run, plus the reconciliation.
///
/// # What these numbers are, and what the old ones were
///
/// Before T0.4 the only "GPU time" in this project was `device.poll` wall-clock after a
/// submission (`benches/frame_loop.rs`). That number is **queue latency plus GPU execution plus
/// driver bookkeeping**, measured on the CPU, and it cannot separate them. These are timestamps
/// written into the command stream by the GPU itself, so they are execution only.
///
/// # The named difference
///
/// Three quantities, all reported, and the gaps between them are the deliverable:
///
/// | quantity | what it is |
/// |---|---|
/// | Σ per-phase | GPU nanoseconds attributable to a named phase |
/// | [`GpuTimings::frame_ns`] (`gpu.frame`) | first timestamp to last, i.e. the whole frame's GPU span |
/// | CPU `shell.phases_render` | wall time the CPU spent *recording* those phases |
///
/// `frame_ns − Σ per-phase` is **inter-pass GPU time**: render-pass begin/end (load, store and
/// MSAA resolve), the query resolve itself, and any barriers the driver inserts between passes.
/// It is not attributable to a phase and pretending otherwise would be the exact error the old
/// poll number made.
///
/// `CPU shell.phases_render − frame_ns` is **not** a GPU number at all: it is command recording,
/// and on a client that submits one encoder per frame it is expected to be the larger of the two
/// at low geometry counts. A negative value there means the GPU is the bottleneck.
///
/// # Caveats that must travel with the numbers
///
/// * `CommandEncoder::write_timestamp` is documented by wgpu as **not** strictly ordered against
///   surrounding commands — "commands within a command recorder may be reordered". On a
///   tile-based or heavily-reordering driver a per-phase figure is therefore an attribution, not
///   a guarantee. `gpu.frame` (outermost pair) is the robust number; the per-phase split is the
///   useful one. Both are reported so a reader can see if they disagree.
/// * Timestamps come from a GPU clock that can be reset by the driver. A pair whose end precedes
///   its beginning is discarded rather than wrapped, and counted in
///   [`GpuTimings::discarded_pairs`].
#[derive(Debug, Clone, Default)]
pub struct GpuTimings {
    /// Frames that produced a complete set of timestamps.
    pub frames: u64,
    /// Per-scope stats, sorted by total GPU time descending. Phase names, plus the `gpu.*`
    /// synthetic scopes.
    pub scopes: Vec<(&'static str, GpuScopeStats)>,
    /// The `gpu.frame` scope's stats — the whole frame's GPU span.
    pub frame_ns: GpuScopeStats,
    /// Phases that were enabled but not timed because [`MAX_TIMED_SCOPES`] was reached.
    pub untimed_scopes: u64,
    /// Timestamp pairs thrown away because the end preceded the beginning.
    pub discarded_pairs: u64,
}

impl GpuTimings {
    /// Whether anything was recorded at all. `false` for every run without `MC_GPU_TIMING`.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// Mean nanoseconds per frame summed over every scope **except** the synthetic `gpu.frame`
    /// wrapper, which would double-count.
    #[must_use]
    pub fn attributed_mean_ns(&self) -> f64 {
        self.scopes
            .iter()
            .filter(|(name, _)| *name != GPU_FRAME_SCOPE)
            .map(|(_, s)| s.mean_ns())
            .sum()
    }

    /// `gpu.frame` minus the sum of everything attributed to a scope inside it: the inter-pass
    /// GPU time described on this type. Can be negative if the driver reordered the timestamps,
    /// which is itself worth seeing.
    #[must_use]
    pub fn unattributed_mean_ns(&self) -> f64 {
        self.frame_ns.mean_ns() - self.attributed_mean_ns()
    }

    /// A one-table report, in milliseconds per frame.
    #[must_use]
    pub fn format_table(&self) -> String {
        if self.is_empty() {
            return "no GPU timestamps recorded (set MC_GPU_TIMING=1)\n".to_owned();
        }
        let ms = |ns: f64| ns / 1_000_000.0;
        let mut out = format!(
            "\n===== mc-rs GPU timestamps — {} timed frames =====\n{:<34} {:>9} {:>12} {:>12}\n",
            self.frames, "scope", "frames", "ms/frame", "max ms"
        );
        for (name, stats) in &self.scopes {
            out.push_str(&format!(
                "{name:<34} {:>9} {:>12.4} {:>12.4}\n",
                stats.calls,
                ms(stats.mean_ns()),
                ms(stats.max_ns as f64),
            ));
        }
        out.push_str(&format!(
            "{:<34} {:>9} {:>12.4}\n{:<34} {:>9} {:>12.4}\n",
            "(sum of phases + synthetics)",
            "",
            ms(self.attributed_mean_ns()),
            "(gpu.frame - the above: inter-pass)",
            "",
            ms(self.unattributed_mean_ns()),
        ));
        if self.untimed_scopes > 0 || self.discarded_pairs > 0 {
            out.push_str(&format!(
                "untimed scopes: {}, discarded pairs: {}\n",
                self.untimed_scopes, self.discarded_pairs
            ));
        }
        out
    }
}

/// State of one readback buffer in the ring.
mod readback_state {
    /// Not in use; safe to write into.
    pub const FREE: u32 = 0;
    /// Copy submitted, `map_async` in flight.
    pub const PENDING: u32 = 1;
    /// Mapped and waiting to be read.
    pub const READY: u32 = 2;
    /// Mapping failed; recycle without reading.
    pub const FAILED: u32 = 3;
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU32>,
    labels: Vec<&'static str>,
}

/// Writes a begin/end timestamp pair around each phase and reads the results back a few frames
/// later, without ever blocking the frame loop on the GPU.
///
/// Only constructed when [`Gpu::supports_gpu_timing`](crate::Gpu::supports_gpu_timing) is true,
/// which requires `MC_GPU_TIMING` — see [`GPU_TIMING_ENV`](crate::context::GPU_TIMING_ENV).
struct GpuFrameTimer {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Readback>,
    cursor: usize,
    period_ns: f64,
    /// This frame's scope names; index `k` owns queries `2k` and `2k+1`.
    labels: Vec<&'static str>,
    totals: BTreeMap<&'static str, GpuScopeStats>,
    frames: u64,
    untimed: u64,
    discarded: u64,
}

impl GpuFrameTimer {
    fn new(gpu: &Gpu) -> Option<Self> {
        if !gpu.supports_gpu_timing() {
            return None;
        }
        let queries = (MAX_TIMED_SCOPES * 2) as u32;
        let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
        let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("mc-render phase timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mc-render timestamp resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..READBACK_DEPTH)
            .map(|_| Readback {
                buffer: gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc-render timestamp readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU32::new(readback_state::FREE)),
                labels: Vec::new(),
            })
            .collect::<Vec<_>>();
        Some(Self {
            query_set,
            resolve,
            slots,
            cursor: 0,
            period_ns: f64::from(gpu.timestamp_period_ns()),
            labels: Vec::new(),
            totals: BTreeMap::new(),
            frames: 0,
            untimed: 0,
            discarded: 0,
        })
    }

    /// Harvest whatever the GPU has finished, and start a fresh frame.
    fn begin_frame(&mut self, gpu: &Gpu) {
        // Non-blocking. `map_async` callbacks only run inside a poll, and the windowed loop never
        // otherwise polls, so without this the ring fills up and no result is ever read.
        let _ = gpu.device.poll(wgpu::PollType::Poll);
        for index in 0..self.slots.len() {
            match self.slots[index].state.load(Ordering::Acquire) {
                readback_state::READY => self.harvest(index),
                readback_state::FAILED => {
                    self.slots[index].buffer.unmap();
                    self.slots[index]
                        .state
                        .store(readback_state::FREE, Ordering::Release);
                }
                _ => {}
            }
        }
        self.labels.clear();
    }

    fn harvest(&mut self, index: usize) {
        let slot = &self.slots[index];
        let pairs = slot.labels.len();
        let mut samples: Vec<(&'static str, u64)> = Vec::with_capacity(pairs);
        let mut discarded = 0u64;
        if let Ok(view) = slot.buffer.get_mapped_range(..) {
            let words: &[u64] = bytemuck::cast_slice(&view[..pairs * 2 * 8]);
            for (k, name) in slot.labels.iter().enumerate() {
                let (begin, end) = (words[k * 2], words[k * 2 + 1]);
                if end >= begin {
                    samples.push((name, end - begin));
                } else {
                    discarded += 1;
                }
            }
        }
        self.slots[index].buffer.unmap();
        self.slots[index]
            .state
            .store(readback_state::FREE, Ordering::Release);

        self.discarded += discarded;
        if samples.is_empty() {
            return;
        }
        self.frames += 1;
        for (name, ticks) in samples {
            let ns = (ticks as f64 * self.period_ns) as u64;
            let entry = self.totals.entry(name).or_default();
            entry.calls += 1;
            entry.total_ns += ns;
            entry.max_ns = entry.max_ns.max(ns);
        }
    }

    /// Open a scope. `None` when the frame has run out of query slots.
    fn open(&mut self, enc: &mut wgpu::CommandEncoder, name: &'static str) -> Option<usize> {
        if self.labels.len() >= MAX_TIMED_SCOPES {
            self.untimed += 1;
            return None;
        }
        let slot = self.labels.len();
        self.labels.push(name);
        enc.write_timestamp(&self.query_set, (slot * 2) as u32);
        Some(slot)
    }

    fn close(&mut self, enc: &mut wgpu::CommandEncoder, slot: Option<usize>) {
        if let Some(slot) = slot {
            enc.write_timestamp(&self.query_set, (slot * 2 + 1) as u32);
        }
    }

    /// Resolve this frame's queries into a free readback buffer, if there is one. When the ring
    /// is full the frame is simply not measured — dropping a sample is always better than
    /// stalling the loop to take one.
    fn finish(&mut self, enc: &mut wgpu::CommandEncoder) -> Option<usize> {
        if self.labels.is_empty() {
            return None;
        }
        let index = (0..self.slots.len())
            .map(|offset| (self.cursor + offset) % self.slots.len())
            .find(|i| self.slots[*i].state.load(Ordering::Acquire) == readback_state::FREE)?;
        self.cursor = (index + 1) % self.slots.len();

        let queries = (self.labels.len() * 2) as u32;
        enc.resolve_query_set(&self.query_set, 0..queries, &self.resolve, 0);
        let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);
        enc.copy_buffer_to_buffer(&self.resolve, 0, &self.slots[index].buffer, 0, bytes);
        self.slots[index].labels.clear();
        self.slots[index].labels.extend_from_slice(&self.labels);
        Some(index)
    }

    /// Kick off the mapping for the buffer [`GpuFrameTimer::finish`] chose. Must run after the
    /// encoder has been submitted, or the copy has not been queued yet.
    fn after_submit(&mut self, index: Option<usize>) {
        let Some(index) = index else { return };
        let slot = &self.slots[index];
        slot.state.store(readback_state::PENDING, Ordering::Release);
        let state = Arc::clone(&slot.state);
        slot.buffer
            .map_async(wgpu::MapMode::Read, .., move |result| {
                state.store(
                    if result.is_ok() {
                        readback_state::READY
                    } else {
                        readback_state::FAILED
                    },
                    Ordering::Release,
                );
            });
    }

    fn timings(&self) -> GpuTimings {
        let mut scopes: Vec<(&'static str, GpuScopeStats)> =
            self.totals.iter().map(|(k, v)| (*k, *v)).collect();
        scopes.sort_by_key(|(_, s)| std::cmp::Reverse(s.total_ns));
        GpuTimings {
            frames: self.frames,
            frame_ns: self
                .totals
                .get(GPU_FRAME_SCOPE)
                .copied()
                .unwrap_or_default(),
            scopes,
            untimed_scopes: self.untimed,
            discarded_pairs: self.discarded,
        }
    }
}

/// Ordered list of phases, plus the frame's clear colour.
///
/// Insertion order between two phases with the same [`PhaseOrder`] is preserved, so a section
/// that registers several sub-phases at one order keeps them in the order it added them.
pub struct PhaseList {
    phases: Vec<(PhaseOrder, Box<dyn RenderPhase>)>,
    /// Colour the frame starts from. Overwritten by the sky phase (S20) once one exists; until
    /// then it is what you actually see. Vanilla's void/fog blue is roughly `0.02, 0.05, 0.10`.
    pub clear_color: wgpu::Color,
    created: std::time::Instant,
    /// Allocated lazily, and only once some enabled phase returns `true` from
    /// [`RenderPhase::needs_scene_color`]. `None` means the frame path is byte-for-byte the one
    /// that existed before [`SceneTextures`] did.
    scene: Option<SceneTargets>,
    /// Frames in a row without scene mode while `scene` is held; at
    /// `w21_mr::SCENE_RELEASE_IDLE_FRAMES` the snapshot is dropped (made again on demand).
    scene_idle_frames: u32,
    /// `None` unless `MC_GPU_TIMING` asked for T0.4's timestamps *and* the device granted them.
    gpu_timer: Option<GpuFrameTimer>,
    /// Whether the one-shot attempt to build [`PhaseList::gpu_timer`] has been made. Without it
    /// an unsupported device would re-query its features on every frame forever.
    gpu_timer_built: bool,
    /// The unjittered camera uniform, copied over the camera buffer before the
    /// first phase at or after [`PhaseOrder::TEMPORAL_AA`] while [`Camera::jitter`] is non-zero.
    /// Built the first time a frame is jittered; never touched otherwise.
    camera_restore: Option<wgpu::Buffer>,
    /// Whether scene mode may render straight into a copyable frame
    /// target (see [`direct_scene_target`]). From `MC_NW_F4_DIRECT_SCENE` (default on); tests
    /// and A/B tools set it with [`PhaseList::set_direct_scene`].
    direct_scene: bool,
    /// Whether consecutive phases that opt in share one render pass
    /// ([`RenderPhase::shares_world_pass`]). From `MC_NW_F2_SHARED_PASS` (default on); tests and
    /// A/B tools set it with [`PhaseList::set_shared_world_pass`].
    shared_world_pass: bool,
    /// Whether the frame's clear may fold into the first passes that touch its
    /// attachments ([`RenderPhase::frame_clear_handoff`]). From `MC_W18_OB_CLEAR_FOLD` (default
    /// on); tests and A/B tools set it with [`PhaseList::set_clear_fold`].
    clear_fold: bool,
    /// Whether consecutive phases that opt in share one read-only-depth
    /// pass ([`RenderPhase::shares_depth_read_pass`]). From `MC_W18_OB_DEPTH_READ_PASS` (default
    /// on); tests and A/B tools set it with [`PhaseList::set_depth_read_pass`].
    depth_read_pass: bool,
}

/// The snapshot backing [`SceneTextures`], plus (only once a frame's target cannot be copied
/// from, see [`direct_scene_target`]) the intermediate colour target and the 1:1 blit that puts
/// it back on the caller's target at end of frame.
struct SceneTargets {
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    snapshot: wgpu::Texture,
    snapshot_view: wgpu::TextureView,
    /// `None` while every scene-mode frame has drawn straight into its
    /// target; built on the first frame whose target is not copyable, then kept.
    intermediate: Option<SceneIntermediate>,
}

/// The intermediate colour target and its end-of-frame blit (the scene path from before this change, still used where the frame target lacks `COPY_SRC` or with
/// `MC_NW_F4_DIRECT_SCENE=0`).
struct SceneIntermediate {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    blit_group: wgpu::BindGroup,
}

/// Whether scene mode may render straight into the frame target.
/// `MC_NW_F4_DIRECT_SCENE=0` restores the intermediate target
/// and the end-of-frame blit on every frame. Read once.
fn direct_scene_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("MC_NW_F4_DIRECT_SCENE").is_none_or(|v| v != "0"))
}

/// Whether scene mode can draw into `target` itself and take each
/// snapshot by copying it: a whole, single-sampled 2D texture with the frame's size and format
/// that allows `COPY_SRC` (the window's swap chain does wherever the surface offers it, which is
/// also what screenshots need; headless targets always do).
///
/// Every phase then writes the same bytes into the same format it wrote into the intermediate
/// target, and the end-of-frame blit (a 1:1 `textureLoad` copy of that target) is not needed:
/// the image is identical and the frame saves one full-screen pass and one screen-sized target.
fn direct_scene_target(gpu: &Gpu, target: &wgpu::TextureView) -> bool {
    let texture = target.texture();
    texture.usage().contains(wgpu::TextureUsages::COPY_SRC)
        && texture.format() == gpu.config.format
        && texture.width() == gpu.config.width.max(1)
        && texture.height() == gpu.config.height.max(1)
        && texture.sample_count() == 1
        && texture.mip_level_count() == 1
        && texture.depth_or_array_layers() == 1
        && texture.dimension() == wgpu::TextureDimension::D2
}

/// A 1:1, unfiltered copy. `textureLoad` with the fragment's own integer pixel coordinates is
/// what makes the final blit exact — a sampler would introduce filtering and a half-texel
/// question this must not have.
const SCENE_BLIT_WGSL: &str = r#"
@group(0) @binding(0) var scene: texture_2d<f32>;

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
    return textureLoad(scene, vec2<i32>(pos.xy), 0);
}
"#;

/// One screen-sized, single-sampled colour texture in the frame's format, for scene mode.
fn scene_texture(
    gpu: &Gpu,
    label: &str,
    extra: wgpu::TextureUsages,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: gpu.config.width.max(1),
            height: gpu.config.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        // Always 1: with MSAA on this is the *resolve* target, which is where the
        // caller's own target sat before. That is what keeps the snapshot bindable.
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: gpu.config.format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::TEXTURE_BINDING
            | extra,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

impl SceneTargets {
    fn new(gpu: &Gpu) -> Self {
        let (snapshot, snapshot_view) = scene_texture(
            gpu,
            "mc-render scene snapshot",
            wgpu::TextureUsages::COPY_DST,
        );
        Self {
            width: gpu.config.width.max(1),
            height: gpu.config.height.max(1),
            format: gpu.config.format,
            snapshot,
            snapshot_view,
            intermediate: None,
        }
    }
}

impl SceneIntermediate {
    fn new(gpu: &Gpu) -> Self {
        let format = gpu.config.format;
        let (color, color_view) =
            scene_texture(gpu, "mc-render scene colour", wgpu::TextureUsages::COPY_SRC);

        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("mc-render scene blit"),
                source: wgpu::ShaderSource::Wgsl(SCENE_BLIT_WGSL.into()),
            });
        let blit_layout = gpu
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("mc-render scene blit group0"),
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
        let pipeline_layout = gpu
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("mc-render scene blit layout"),
                bind_group_layouts: &[Some(&blit_layout)],
                immediate_size: 0,
            });
        let blit_pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("mc-render scene blit"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                // The caller's target is single-sampled even when MSAA is on, so the final blit
                // never goes through the multisampled attachment.
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
        let blit_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mc-render scene blit group0"),
            layout: &blit_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&color_view),
            }],
        });

        Self {
            color,
            color_view,
            blit_pipeline,
            blit_group,
        }
    }

    fn blit_to(&self, enc: &mut wgpu::CommandEncoder, target: &wgpu::TextureView) {
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mc-render scene blit"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
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
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &self.blit_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

impl SceneTargets {
    fn matches(&self, gpu: &Gpu) -> bool {
        self.width == gpu.config.width.max(1)
            && self.height == gpu.config.height.max(1)
            && self.format == gpu.config.format
    }

    /// Copy `source` (the frame target, or the intermediate target) into the snapshot.
    fn snapshot_from(&self, enc: &mut wgpu::CommandEncoder, source: &wgpu::Texture) {
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: source,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &self.snapshot,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn textures(&self) -> SceneTextures<'_> {
        SceneTextures {
            color: &self.snapshot_view,
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }
}

impl Default for PhaseList {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseList {
    /// An empty list with vanilla's fog-blue clear colour.
    #[must_use]
    pub fn new() -> Self {
        Self {
            phases: Vec::new(),
            clear_color: wgpu::Color {
                r: 0.043,
                g: 0.070,
                b: 0.145,
                a: 1.0,
            },
            created: std::time::Instant::now(),
            scene: None,
            scene_idle_frames: 0,
            gpu_timer: None,
            gpu_timer_built: false,
            camera_restore: None,
            direct_scene: direct_scene_enabled(),
            shared_world_pass: shared_pass_enabled(),
            clear_fold: clear_fold_enabled(),
            depth_read_pass: depth_read_pass_enabled(),
        }
    }

    /// Fold the frame's clear into the first passes that touch its attachments
    /// (the default, unless `MC_W18_OB_CLEAR_FOLD=0`) or record it as a pass of its own. The frame
    /// is the same bytes either way. For tests and A/B tools.
    pub fn set_clear_fold(&mut self, fold: bool) {
        self.clear_fold = fold;
    }

    /// Let consecutive phases that opt in share one read-only-depth pass
    /// (the default, unless `MC_W18_OB_DEPTH_READ_PASS=0`) or give each its own pass. The frame is
    /// the same bytes either way. For tests and A/B tools.
    pub fn set_depth_read_pass(&mut self, share: bool) {
        self.depth_read_pass = share;
    }

    /// Let consecutive phases that opt in share one render pass (the
    /// default, unless `MC_NW_F2_SHARED_PASS=0`) or give each its own pass. The frame is the same
    /// bytes either way. For tests and A/B tools.
    pub fn set_shared_world_pass(&mut self, share: bool) {
        self.shared_world_pass = share;
    }

    /// Allow (the default, unless `MC_NW_F4_DIRECT_SCENE=0`) or forbid
    /// scene mode drawing straight into a copyable frame target. Either way the frame is the
    /// same bytes; forbidding it brings back the intermediate target and its end-of-frame blit.
    /// For tests and A/B tools.
    pub fn set_direct_scene(&mut self, allow: bool) {
        self.direct_scene = allow;
    }

    /// Whether scene mode's intermediate colour target exists (it is only
    /// built for a frame target that cannot be copied from, or with direct scene mode off). For
    /// tests.
    #[must_use]
    pub fn scene_intermediate_allocated(&self) -> bool {
        self.scene
            .as_ref()
            .is_some_and(|scene| scene.intermediate.is_some())
    }

    /// Register a phase at `order`. Stable with respect to earlier insertions at the same order.
    pub fn insert(&mut self, order: PhaseOrder, phase: Box<dyn RenderPhase>) {
        let at = self.phases.partition_point(|(o, _)| *o <= order);
        self.phases.insert(at, (order, phase));
    }

    /// Remove every phase whose [`RenderPhase::name`] matches, returning how many went.
    /// Screens that toggle a whole subsystem off use this; most callers want
    /// [`RenderPhase::enabled`] instead.
    pub fn remove_named(&mut self, name: &str) -> usize {
        let before = self.phases.len();
        self.phases.retain(|(_, p)| p.name() != name);
        before - self.phases.len()
    }

    /// Drop every registered phase, returning how many went.
    ///
    /// The counterpart to registering phases from [`ShellApp::frame`](crate::ShellApp::frame)
    /// rather than only from `init`: a resource-pack reload re-stitches the block atlas, and a
    /// phase that baked the old atlas into a bind group at construction cannot be *updated*, only
    /// rebuilt. java: `Minecraft.refreshResources` ends with `RenderGlobal.loadRenderers()`, which
    /// throws away and rebuilds exactly this much.
    ///
    /// Every phase's GPU resources are released as it drops, so the caller must re-register before
    /// the next [`PhaseList::render`] — an empty list renders a cleared frame, not a stale one.
    pub fn clear(&mut self) -> usize {
        let count = self.phases.len();
        self.phases.clear();
        // The intermediate scene targets belong to whoever opted in; with no phases registered
        // nothing can, so dropping them keeps `scene_mode()` and the allocation in step.
        self.scene = None;
        count
    }

    /// Number of registered phases.
    #[must_use]
    pub fn len(&self) -> usize {
        self.phases.len()
    }

    /// Whether no phases are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }

    /// Registered phase names in record order — for the F3 overlay and for tests.
    pub fn names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.phases.iter().map(|(_, p)| p.name())
    }

    /// Forward a window resize to every phase. The caller must have already called
    /// [`Gpu::resize`](crate::Gpu::resize).
    pub fn resize(&mut self, gpu: &Gpu, width: u32, height: u32) {
        for (_, phase) in &mut self.phases {
            phase.resize(gpu, width, height);
        }
    }

    /// Whether any enabled phase asked for [`SceneTextures`] this frame. Exposed for tests and
    /// for the F3 overlay, which wants to say whether the extra intermediate target is live.
    #[must_use]
    pub fn scene_mode(&self) -> bool {
        self.phases
            .iter()
            .any(|(_, p)| p.enabled() && p.needs_scene_color())
    }

    /// Record and submit one frame: upload the camera uniform, clear colour and depth, run every
    /// enabled phase in order, submit.
    ///
    /// Returns the submission index, so a caller that needs to block on GPU completion (the
    /// headless readback path, a screenshot) can wait on exactly this frame.
    ///
    /// If any enabled phase returns `true` from [`RenderPhase::needs_scene_color`], the frame
    /// takes the intermediate-target path documented on [`SceneTextures`]. Otherwise this is
    /// exactly the code it always was.
    pub fn render(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        cam: &Camera,
        partial_ticks: f32,
    ) -> wgpu::SubmissionIndex {
        let (enc, frame_scope) = self.record_frame(gpu, target, cam, partial_ticks);
        mc_core::profile_scope!("gpu.queue.submit");
        finish_frame(gpu, enc, self.gpu_timer.as_mut(), frame_scope)
    }

    /// [`Self::render`] for the opt-in pipelined submit (#14 step 1b, `MC_PIPELINED_SUBMIT`):
    /// records the same frame, then leaves it pending on `gpu` ([`Gpu::defer_frame`]) for the
    /// shell to hand to the present thread, which finishes and submits it. Any main-thread queue
    /// call before that hand-off submits it first, so the order is the serial loop's.
    ///
    /// With GPU timestamps on (`MC_GPU_TIMING`) the frame is submitted here, as in
    /// [`Self::render`]: the timer resolves its queries around the submit on this thread.
    pub fn render_deferred(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        cam: &Camera,
        partial_ticks: f32,
    ) {
        let (enc, frame_scope) = self.record_frame(gpu, target, cam, partial_ticks);
        if self.gpu_timer.is_some() {
            mc_core::profile_scope!("gpu.queue.submit");
            finish_frame(gpu, enc, self.gpu_timer.as_mut(), frame_scope);
            return;
        }
        mc_core::profile_scope!("gpu.defer_frame");
        gpu.defer_frame(enc);
    }

    /// Record the frame [`Self::render`] records, into `target`, and hand back
    /// its encoder unfinished and unsubmitted, so the caller can append commands after the last
    /// phase (the shell copies `target` into the swap-chain image) before it submits or defers the
    /// encoder itself. The staged uploads stay in the upload ring for that submission.
    ///
    /// Returns `None` while GPU timestamps are on (`MC_GPU_TIMING`): their queries resolve around
    /// a submit on this thread, so the frame has then been finished and submitted here, as in
    /// [`Self::render_deferred`].
    pub fn record_unsubmitted(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        cam: &Camera,
        partial_ticks: f32,
    ) -> Option<wgpu::CommandEncoder> {
        let (enc, frame_scope) = self.record_frame(gpu, target, cam, partial_ticks);
        if self.gpu_timer.is_some() {
            mc_core::profile_scope!("gpu.queue.submit");
            finish_frame(gpu, enc, self.gpu_timer.as_mut(), frame_scope);
            return None;
        }
        Some(enc)
    }

    /// Everything [`Self::render`] does before `finish()`: the camera uniform, the clear, every
    /// enabled phase in order (and the scene blit on the intermediate-target path). Returns the
    /// frame encoder and the GPU timer's open frame scope.
    fn record_frame(
        &mut self,
        gpu: &Gpu,
        target: &wgpu::TextureView,
        cam: &Camera,
        partial_ticks: f32,
    ) -> (wgpu::CommandEncoder, Option<usize>) {
        let seconds = self.created.elapsed().as_secs_f32();
        gpu.write_camera(&CameraUniform::new(cam, gpu, partial_ticks, seconds));
        // While TAA jitters the world, stage the unjittered camera too; it is
        // copied over the camera buffer before the first phase at `PhaseOrder::TEMPORAL_AA`.
        // With no jitter (always, unless TAA is on) nothing here runs.
        let mut jittered = cam.jitter != glam::Vec2::ZERO;
        if jittered {
            let restore = self.camera_restore.get_or_insert_with(|| {
                gpu.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("mc-render camera uniform (unjittered)"),
                    size: std::mem::size_of::<CameraUniform>() as u64,
                    usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            let unjittered = CameraUniform::new_unjittered(cam, gpu, partial_ticks, seconds);
            // Staged like the camera itself: the frame encoder's restore
            // copy is this buffer's only reader.
            gpu.write_camera_uniform(restore, &unjittered);
        }

        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mc-render frame"),
            });

        let scene_mode = self.scene_mode();
        if scene_mode && !self.scene.as_ref().is_some_and(|s| s.matches(gpu)) {
            self.scene = Some(SceneTargets::new(gpu));
        }
        // A snapshot no frame has read for a while is dropped; the
        // line above makes it again, exactly as the first time, when a phase reads the scene.
        // Every phase is handed the snapshot only after this frame's copy into it, and every
        // phase that caches a bind group over it compares the view, as after a resize. Kept
        // when the intermediate target exists (rare; it owns a pipeline).
        crate::w21_mr::release_when_idle(
            scene_mode,
            &mut self.scene,
            &mut self.scene_idle_frames,
            crate::w21_mr::SCENE_RELEASE_IDLE_FRAMES,
            |scene| scene.intermediate.is_none(),
        );
        if !self.gpu_timer_built {
            self.gpu_timer_built = true;
            self.gpu_timer = GpuFrameTimer::new(gpu);
        }
        // Split the borrow so `phases` can be mutable while `scene` is read and the timer is
        // written. Every `gpu_timer` call below is a no-op `None` on the default path.
        let clear_color = self.clear_color;
        let Self {
            phases,
            scene,
            gpu_timer,
            camera_restore,
            direct_scene,
            shared_world_pass,
            clear_fold,
            depth_read_pass,
            ..
        } = self;
        let camera_restore = camera_restore.as_ref();
        // The measurement aid may turn either switch off for this frame.
        let clear_fold = *clear_fold && !crate::w18_ob_ab::off(crate::w18_ob_ab::CLEAR_FOLD);
        let depth_read_pass =
            *depth_read_pass && !crate::w18_ob_ab::off(crate::w18_ob_ab::DEPTH_READ_PASS);
        if let Some(timer) = gpu_timer.as_mut() {
            timer.begin_frame(gpu);
        }
        let frame_scope = gpu_timer
            .as_mut()
            .and_then(|t| t.open(&mut enc, GPU_FRAME_SCOPE));

        if !scene_mode {
            // One clear pass up front means every phase can use LoadOp::Load and none of them has
            // to know whether it happens to be first this frame. With the fold
            // on, the first passes that touch colour and depth clear them instead.
            if clear_fold {
                gpu.arm_frame_clear(target, clear_color);
            } else {
                let clear_scope = gpu_timer
                    .as_mut()
                    .and_then(|t| t.open(&mut enc, GPU_CLEAR_SCOPE));
                drop(gpu.begin_clear_pass(&mut enc, target, clear_color));
                if let Some(timer) = gpu_timer.as_mut() {
                    timer.close(&mut enc, clear_scope);
                }
            }

            // A run of phases that opt in records into one pass, opened by
            // the first of them and ended (dropped) before anything else touches the encoder.
            // With GPU timestamps on, the run is timed as one `GPU_SHARED_SCOPE`.
            // The same for the read-only-depth pass (`GPU_DEPTH_READ_SCOPE`).
            let share = *shared_world_pass;
            let share_depth_read = depth_read_pass;
            let mut shared: Option<SharedPass> = None;
            for (order, phase) in phases {
                if !phase.enabled() {
                    continue;
                }
                if jittered && *order >= PhaseOrder::TEMPORAL_AA {
                    end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
                    jittered = false;
                    restore_unjittered_camera(gpu, &mut enc, camera_restore);
                }
                let kind = shared_pass_kind(&**phase, share, share_depth_read);
                if clear_fold {
                    settle_frame_clear(gpu, &mut enc, &**phase, kind, false);
                }
                if let Some(kind) = kind {
                    let _record = phase_record_span(&**phase);
                    if shared.as_ref().is_some_and(|open| open.kind != kind) {
                        end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
                    }
                    let open = shared.get_or_insert_with(|| {
                        SharedPass::open(gpu, &mut enc, target, kind, gpu_timer.as_mut())
                    });
                    open.record(gpu, &mut **phase, cam);
                    continue;
                }
                end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
                let scope = gpu_timer
                    .as_mut()
                    .and_then(|t| t.open(&mut enc, phase.name()));
                let _record = phase_record_span(&**phase);
                phase.record(gpu, &mut enc, target, cam);
                if let Some(timer) = gpu_timer.as_mut() {
                    timer.close(&mut enc, scope);
                }
            }
            end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
            // An attachment no pass touched is still cleared, as before.
            gpu.flush_frame_clear(&mut enc);

            return (enc, frame_scope);
        }

        // Draw into the target itself when it can be copied from; the
        // intermediate target and its end-of-frame blit exist only for targets that cannot.
        let direct = *direct_scene && direct_scene_target(gpu, target);
        let scene = scene.as_mut().expect("just built");
        if !direct && scene.intermediate.is_none() {
            scene.intermediate = Some(SceneIntermediate::new(gpu));
        }
        let scene = &*scene;
        let (frame_view, frame_texture) = match &scene.intermediate {
            Some(inter) if !direct => (&inter.color_view, &inter.color),
            _ => (target, target.texture()),
        };

        if clear_fold {
            // As in the direct path above.
            gpu.arm_frame_clear(frame_view, clear_color);
        } else {
            let clear_scope = gpu_timer
                .as_mut()
                .and_then(|t| t.open(&mut enc, GPU_CLEAR_SCOPE));
            drop(gpu.begin_clear_pass(&mut enc, frame_view, clear_color));
            if let Some(timer) = gpu_timer.as_mut() {
                timer.close(&mut enc, clear_scope);
            }
        }

        // As in the direct path above; a phase reading the scene ends the
        // run, because its snapshot is an encoder copy.
        let share = *shared_world_pass;
        let share_depth_read = depth_read_pass;
        let mut shared: Option<SharedPass> = None;
        for (order, phase) in phases {
            if !phase.enabled() {
                continue;
            }
            if jittered && *order >= PhaseOrder::TEMPORAL_AA {
                end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
                jittered = false;
                restore_unjittered_camera(gpu, &mut enc, camera_restore);
            }
            let reads_scene = phase.needs_scene_color();
            let kind = if reads_scene {
                None
            } else {
                shared_pass_kind(&**phase, share, share_depth_read)
            };
            if clear_fold {
                settle_frame_clear(gpu, &mut enc, &**phase, kind, reads_scene);
            }
            if let Some(kind) = kind {
                let _record = phase_record_span(&**phase);
                if shared.as_ref().is_some_and(|open| open.kind != kind) {
                    end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
                }
                let open = shared.get_or_insert_with(|| {
                    SharedPass::open(gpu, &mut enc, frame_view, kind, gpu_timer.as_mut())
                });
                open.record(gpu, &mut **phase, cam);
                continue;
            }
            end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
            let scope = gpu_timer
                .as_mut()
                .and_then(|t| t.open(&mut enc, phase.name()));
            let _record = phase_record_span(&**phase);
            if phase.needs_scene_color() {
                scene.snapshot_from(&mut enc, frame_texture);
                phase.record_scene(gpu, &mut enc, frame_view, cam, &scene.textures());
            } else {
                phase.record(gpu, &mut enc, frame_view, cam);
            }
            if let Some(timer) = gpu_timer.as_mut() {
                timer.close(&mut enc, scope);
            }
        }
        end_shared_pass(&mut enc, gpu_timer.as_mut(), shared.take());
        // An attachment no pass touched is still cleared, as before.
        gpu.flush_frame_clear(&mut enc);

        if let Some(inter) = scene.intermediate.as_ref().filter(|_| !direct) {
            let blit_scope = gpu_timer
                .as_mut()
                .and_then(|t| t.open(&mut enc, GPU_BLIT_SCOPE));
            inter.blit_to(&mut enc, target);
            if let Some(timer) = gpu_timer.as_mut() {
                timer.close(&mut enc, blit_scope);
            }
        }

        (enc, frame_scope)
    }

    /// Per-phase GPU timings accumulated so far.
    ///
    /// Empty unless `MC_GPU_TIMING` was set *and* the adapter granted the timestamp features —
    /// see [`GpuTimings`] for what the numbers are and, more importantly, what they are not.
    #[must_use]
    pub fn gpu_timings(&self) -> GpuTimings {
        self.gpu_timer
            .as_ref()
            .map(GpuFrameTimer::timings)
            .unwrap_or_default()
    }
}

/// Copy the unjittered camera uniform over the shared camera buffer inside the
/// frame encoder, so every pass recorded after this point (labels, hand, overlays, UI) reads the
/// unjittered matrices while the world passes before it read the jittered ones. Only called on a
/// jittered frame; see [`Camera::jitter`].
/// End an open shared world pass (dropping it ends it), then close its GPU
/// timer scope: the end timestamp goes on the encoder, after the pass.
fn end_shared_pass(
    enc: &mut wgpu::CommandEncoder,
    timer: Option<&mut GpuFrameTimer>,
    shared: Option<SharedPass>,
) {
    let Some(SharedPass { pass, scope, .. }) = shared else {
        return;
    };
    drop(pass);
    if let Some(timer) = timer {
        timer.close(enc, scope);
    }
}

/// The shared pass open while `PhaseList` records a
/// run of phases that opt in, with its kind and its GPU timer scope.
struct SharedPass {
    kind: SharedPassKind,
    pass: wgpu::RenderPass<'static>,
    scope: Option<usize>,
}

impl SharedPass {
    /// Open the shared pass of `kind` on `target`, timed as one scope. `forget_lifetime` changes
    /// nothing that is recorded.
    fn open(
        gpu: &Gpu,
        enc: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        kind: SharedPassKind,
        timer: Option<&mut GpuFrameTimer>,
    ) -> Self {
        let label = match kind {
            SharedPassKind::World => GPU_SHARED_SCOPE,
            SharedPassKind::DepthRead => GPU_DEPTH_READ_SCOPE,
        };
        let scope = timer.and_then(|t| t.open(enc, label));
        let pass = match kind {
            SharedPassKind::World => gpu.begin_pass(enc, target, label, DepthAttachment::Shared),
            SharedPassKind::DepthRead => gpu.begin_depth_read_pass(enc, target, label),
        }
        .forget_lifetime();
        Self { kind, pass, scope }
    }

    fn record(&mut self, gpu: &Gpu, phase: &mut dyn RenderPhase, cam: &Camera) {
        match self.kind {
            SharedPassKind::World => phase.record_into_pass(gpu, &mut self.pass, cam),
            SharedPassKind::DepthRead => {
                phase.record_into_depth_read_pass(gpu, &mut self.pass, cam);
            }
        }
    }
}

/// The shared pass `phase` records into this frame, if any: the world pass
/// first, then the read-only-depth pass.
fn shared_pass_kind(
    phase: &dyn RenderPhase,
    share_world: bool,
    share_depth_read: bool,
) -> Option<SharedPassKind> {
    if share_world && phase.shares_world_pass() {
        Some(SharedPassKind::World)
    } else if share_depth_read && phase.shares_depth_read_pass() {
        Some(SharedPassKind::DepthRead)
    } else {
        None
    }
}

/// Before `phase` records: record whatever of the frame's clear is still
/// pending and will not be done by the passes `phase` opens. The shared world pass opens with
/// [`Gpu::begin_pass`] on the frame's target with the shared depth buffer, so it takes both; the
/// read-only-depth pass and a phase reading the scene take nothing; any other phase takes what
/// its [`RenderPhase::frame_clear_handoff`] allows.
fn settle_frame_clear(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    phase: &dyn RenderPhase,
    kind: Option<SharedPassKind>,
    reads_scene: bool,
) {
    let (color, depth) = gpu.frame_clear_pending();
    if !color && !depth {
        return;
    }
    let waits = !reads_scene
        && match kind {
            Some(SharedPassKind::World) => true,
            Some(SharedPassKind::DepthRead) => false,
            None => match phase.frame_clear_handoff() {
                FrameClearHandoff::ColorAndDepth => true,
                FrameClearHandoff::Depth => !color,
                FrameClearHandoff::None => false,
            },
        };
    if !waits {
        gpu.flush_frame_clear(enc);
    }
}

fn restore_unjittered_camera(
    gpu: &Gpu,
    enc: &mut wgpu::CommandEncoder,
    restore: Option<&wgpu::Buffer>,
) {
    if let Some(restore) = restore {
        let size = std::mem::size_of::<CameraUniform>() as u64;
        enc.copy_buffer_to_buffer(restore, 0, gpu.camera_buffer(), 0, size);
    }
}

/// The CPU profiling span for one phase's `record`/`record_scene`, named per phase so
/// `MC_PROFILE` can split `shell.phases_render`.
///
/// `profile_scope!` needs a literal name and the list only knows [`RenderPhase::name`] at run
/// time, so the mapping lives here beside the loop that records every phase. Names use their own
/// `phase.` prefix so that a phase that also opens a scope inside its own `record` (`ui.record`,
/// or a `*.record` added later) nests under a different name instead of double-counting. With no
/// `TRACE` subscriber (the default: the profiling layer is not installed unless `MC_PROFILE` is
/// set) this is `level_enabled!`'s check, spelled with public items: one relaxed atomic load, and
/// `name()` is never called.
fn phase_record_span(phase: &dyn RenderPhase) -> tracing::span::EnteredSpan {
    use tracing::level_filters::{LevelFilter, STATIC_MAX_LEVEL};
    let level = tracing::Level::TRACE;
    if !(level <= STATIC_MAX_LEVEL && level <= LevelFilter::current()) {
        return tracing::Span::none().entered();
    }
    let span = match phase.name() {
        "terrain/solid" | "terrain/cutout" | "terrain/translucent" => {
            tracing::trace_span!("phase.terrain")
        }
        "sky/dome" => tracing::trace_span!("phase.sky.dome"),
        "sky/clouds" => tracing::trace_span!("phase.sky.clouds"),
        "sky/weather" => tracing::trace_span!("phase.sky.weather"),
        "sky/fog" => tracing::trace_span!("phase.sky.fog"),
        "entity" => tracing::trace_span!("phase.entity"),
        "tile entities" => tracing::trace_span!("phase.tiles"),
        "beacon" => tracing::trace_span!("phase.beacon"),
        "nameplate" => tracing::trace_span!("phase.nameplate"),
        "particles" => tracing::trace_span!("phase.particles"),
        "first-person" => tracing::trace_span!("phase.first_person"),
        "terrain/block-overlay" => tracing::trace_span!("phase.block_overlay"),
        "post/overlay" => tracing::trace_span!("phase.post.overlay"),
        "post/chain" => tracing::trace_span!("phase.post.chain"),
        "ui" => tracing::trace_span!("phase.ui"),
        // The menu blur and the UI's screen half.
        "post/menu-blur" => tracing::trace_span!("phase.post.menu_blur"),
        "ui/screen" => tracing::trace_span!("phase.ui.screen"),
        _ => tracing::trace_span!("phase.other"),
    };
    span.entered()
}

/// Close the frame scope, queue the timestamp resolve, submit, and start the readback.
///
/// Split out because the two paths through [`PhaseList::render`] end identically and the ordering
/// matters: the resolve must be recorded before `finish()`, and `map_async` must be called after
/// `submit()` or the copy it maps has not been queued.
fn finish_frame(
    gpu: &Gpu,
    mut enc: wgpu::CommandEncoder,
    timer: Option<&mut GpuFrameTimer>,
    frame_scope: Option<usize>,
) -> wgpu::SubmissionIndex {
    // wgpu 30 records passes as commands and encodes them into the backend's command buffer in
    // `finish()`, so this scope is the frame's whole command encoding; `gpu.submit_call` inside
    // `submit_frame` is the submission alone.
    let finish = |enc: wgpu::CommandEncoder| {
        mc_core::profile_scope!("gpu.encoder_finish");
        enc.finish()
    };
    let Some(timer) = timer else {
        return gpu.submit_frame(finish(enc));
    };
    timer.close(&mut enc, frame_scope);
    let pending = timer.finish(&mut enc);
    let index = gpu.submit_frame(finish(enc));
    timer.after_submit(pending);
    index
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Noop(&'static str);
    impl RenderPhase for Noop {
        fn record(
            &mut self,
            _: &Gpu,
            _: &mut wgpu::CommandEncoder,
            _: &wgpu::TextureView,
            _: &Camera,
        ) {
        }
        fn name(&self) -> &'static str {
            self.0
        }
    }

    #[test]
    fn phases_record_in_order_regardless_of_insertion_order() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::UI, Box::new(Noop("ui")));
        list.insert(PhaseOrder::TERRAIN_SOLID, Box::new(Noop("terrain")));
        list.insert(PhaseOrder::SKY, Box::new(Noop("sky")));
        list.insert(PhaseOrder::PARTICLES, Box::new(Noop("particles")));
        assert_eq!(
            list.names().collect::<Vec<_>>(),
            ["sky", "terrain", "particles", "ui"]
        );
    }

    #[test]
    fn equal_orders_keep_insertion_order() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::ENTITIES, Box::new(Noop("first")));
        list.insert(PhaseOrder::ENTITIES, Box::new(Noop("second")));
        list.insert(PhaseOrder::ENTITIES, Box::new(Noop("third")));
        assert_eq!(
            list.names().collect::<Vec<_>>(),
            ["first", "second", "third"]
        );
    }

    #[test]
    fn vanilla_ordering_constants_are_monotonic() {
        let order = [
            PhaseOrder::SKY,
            PhaseOrder::TERRAIN_SOLID,
            PhaseOrder::TERRAIN_CUTOUT,
            PhaseOrder::ENTITIES,
            PhaseOrder::BLOCK_OVERLAY,
            PhaseOrder::PARTICLES,
            // java: `EntityRenderer.renderWorldPass:1451-1467`, weather before translucent.
            PhaseOrder::WEATHER,
            PhaseOrder::TERRAIN_TRANSLUCENT,
            PhaseOrder::FIRST_PERSON,
            PhaseOrder::POST,
            PhaseOrder::UI,
        ];
        assert!(order.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn first_person_visibility_matches_entity_renderer_hand_gate() {
        let mut visibility = FirstPersonVisibility::default();
        assert!(visibility.renders_hand());
        assert!(visibility.renders_empty_arm());

        visibility.third_person_view = 1;
        assert!(!visibility.renders_hand());
        visibility = FirstPersonVisibility::default();
        visibility.sleeping = true;
        assert!(!visibility.renders_hand());
        visibility = FirstPersonVisibility::default();
        visibility.hide_gui = true;
        assert!(!visibility.renders_hand());
        visibility = FirstPersonVisibility::default();
        visibility.spectator = true;
        assert!(!visibility.renders_hand());

        visibility = FirstPersonVisibility::default();
        visibility.invisible = true;
        assert!(visibility.renders_hand());
        assert!(!visibility.renders_empty_arm());
    }

    #[test]
    fn first_person_camera_matrix_inherits_world_view_bobbing_before_arm_rotation() {
        let cam = Camera {
            pitch: 18.0,
            yaw: -37.0,
            ..Default::default()
        };
        let bob = Mat4::from_translation(Vec3::new(0.03125, -0.0625, 0.0))
            * Mat4::from_rotation_z(first_person_radians(2.75))
            * Mat4::from_rotation_x(first_person_radians(4.5));
        let arm = first_person_view_rotation(cam.pitch, cam.yaw, 10.0, 14.0, -31.0, -35.0, 0.5);

        let inherited = first_person_camera_matrix(&cam, bob, 10.0, 14.0, -31.0, -35.0, 0.5);
        let expected = first_person_projection(&cam) * bob * arm;
        let omitted = first_person_projection(&cam) * arm;

        assert_eq!(inherited, expected);
        assert_ne!(inherited, omitted, "walking bob must reach the held hand");
        assert_eq!(
            first_person_camera_matrix(&cam, Mat4::IDENTITY, 10.0, 14.0, -31.0, -35.0, 0.5,),
            omitted,
            "disabled, riding, and stationary gates supply identity"
        );
    }

    /// Bug report: the held item "looks too far on screen ... worse while sprinting".
    /// java: `renderHand` projects with `getFOVModifier(partialTicks, false)`, a fixed 70 base.
    #[test]
    fn hand_projection_uses_the_hand_fov_not_the_world_fov() {
        let player = Camera {
            fov: 110.0,
            fov_modifier: 1.15,
            ..Default::default()
        };
        let vanilla = Camera::default();
        assert_eq!(player.hand_fov, crate::frame::HAND_FOV);
        assert_eq!(
            first_person_projection(&player),
            first_person_projection(&vanilla),
            "the FOV slider and the sprint modifier must not reach the hand"
        );
        // gluPerspective(70, ...): the y scale is cot(35 degrees).
        let y_scale = first_person_projection(&player).y_axis.y;
        let expected = 1.0 / (35.0_f32).to_radians().tan();
        assert!((y_scale - expected).abs() < 1e-5, "{y_scale} vs {expected}");
        // The death zoom and the underwater squeeze arrive through `hand_fov` itself.
        let underwater = Camera {
            hand_fov: 60.0,
            ..Default::default()
        };
        assert!(first_person_projection(&underwater).y_axis.y > y_scale);
    }

    /// GL_RESCALE_NORMAL is the identity on directions for every
    /// similarity, and for `diag(1, 1, 2)` scales `f (M^-1)^T = 2 diag(1, 1, 1/2)`: the X and Y
    /// face normals come out twice as long (brighter), the Z one unit length.
    #[test]
    fn gl_rescale_normal_only_changes_non_uniform_item_matrices() {
        let similar = Mat4::from_rotation_y(0.7)
            * Mat4::from_scale(Vec3::splat(0.4))
            * Mat4::from_rotation_z(-0.3)
            * Mat4::from_translation(Vec3::new(0.5, -0.2, 0.1));
        assert_eq!(gl_rescale_normal_matrix(similar), None);
        assert_eq!(gl_rescale_normal_matrix(Mat4::IDENTITY), None);

        let stretched = Mat4::from_scale(Vec3::new(1.0, 1.0, 2.0));
        let normals = gl_rescale_normal_matrix(stretched).expect("a non-uniform scale");
        assert!((normals * Vec3::X - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
        assert!((normals * Vec3::Y - Vec3::new(0.0, 2.0, 0.0)).length() < 1e-6);
        assert!((normals * Vec3::Z - Vec3::Z).length() < 1e-6);

        // The emitter's stored `normalize(M n)` is mapped back onto `n` before rescaling.
        let mut vertex = EntityVertex {
            normal: stretched.transform_vector3(Vec3::X).normalize().into(),
            ..EntityVertex::default()
        };
        apply_gl_rescale_normals(stretched, core::slice::from_mut(&mut vertex));
        assert!((Vec3::from(vertex.normal) - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn standard_item_lights_are_javas_normalized_vectors() {
        let [light0, light1] = standard_item_light_directions();
        // java: (0.2F, 1, -0.7F) / (float) sqrt(1.53...), narrowed to float.
        let length = f64::from((0.04_f64 + 1.0 + 0.49).sqrt() as f32);
        assert!((light0 - Vec3::new(0.161_690, 0.808_452, -0.565_916)).length() < 2e-6);
        assert!((f64::from(light0.y) - 1.0 / length).abs() < 1e-7);
        assert_eq!(light1, Vec3::new(-light0.x, light0.y, -light0.z));
    }

    /// GL evaluates `N . L` in eye space: lights under `Rx(pitch) * Ry(yaw)` and normals under
    /// the arm sway, both behind the hurt/bob prefix. The vertex-frame lights must give the same
    /// dot products for every normal.
    #[test]
    fn hand_lights_reproduce_gl_eye_space_dot_products() {
        let (pitch, yaw) = (23.0_f32, 131.0_f32);
        let (prev_arm_pitch, arm_pitch, prev_arm_yaw, arm_yaw, pt) =
            (10.0, 14.0, 120.0, 125.0, 0.5);
        let prefix = Mat4::from_rotation_z(0.3) * Mat4::from_rotation_x(-0.2);
        let sway = first_person_view_rotation(
            pitch,
            yaw,
            prev_arm_pitch,
            arm_pitch,
            prev_arm_yaw,
            arm_yaw,
            pt,
        );
        let light_matrix = prefix
            * Mat4::from_rotation_x(first_person_radians(pitch))
            * Mat4::from_rotation_y(first_person_radians(yaw));
        let gl_lights = standard_item_light_directions().map(|l| light_matrix.transform_vector3(l));
        let ours = first_person_light_directions(
            pitch,
            yaw,
            prev_arm_pitch,
            arm_pitch,
            prev_arm_yaw,
            arm_yaw,
            pt,
        );
        for normal in [
            Vec3::X,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::new(0.3, -0.5, 0.8).normalize(),
        ] {
            let eye_normal = (prefix * sway).transform_vector3(normal);
            for (gl, light) in gl_lights.iter().zip(ours) {
                assert!(
                    (eye_normal.dot(*gl) - normal.dot(light)).abs() < 1e-5,
                    "{normal:?}: GL {} vs vertex frame {}",
                    eye_normal.dot(*gl),
                    normal.dot(light)
                );
            }
        }
        // Facing south with no sway the lights are Java's own vectors; facing north they swap.
        assert_eq!(
            first_person_light_directions(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            standard_item_light_directions()
        );
        let [north0, north1] =
            first_person_light_directions(0.0, 180.0, 0.0, 0.0, 180.0, 180.0, 0.0);
        let [light0, light1] = standard_item_light_directions();
        assert!((north0 - light1).length() < 1e-5 && (north1 - light0).length() < 1e-5);
    }

    #[test]
    fn hand_lighting_is_fixed_function_then_the_lightmap_stage() {
        let lights = standard_item_light_directions();
        let white = EntityVertex::WHITE;
        // A held block's top face takes both lights and clamps to the vertex colour.
        let top = first_person_lit_color(white, [0.0, 1.0, 0.0], lights, [1.0; 3]);
        assert_eq!(top, [1.0, 1.0, 1.0, 1.0]);
        // A side face facing south-east: only LIGHT1 reaches it.
        let side = Vec3::new(0.5_f32.sqrt(), 0.0, 0.5_f32.sqrt());
        let expected = 0.4 + 0.6 * side.dot(lights[1]);
        let lit = first_person_lit_color(white, side.into(), lights, [1.0; 3]);
        assert!((lit[0] - expected).abs() < 1e-6 && lit[0] < 0.6, "{lit:?}");
        // Facing away from both lights leaves the 0.4 ambient.
        let under = first_person_lit_color(white, [0.0, -1.0, 0.0], lights, [1.0; 3]);
        assert!((under[0] - 0.4).abs() < 1e-6);
        // GL clamps the lit colour before the lightmap stage multiplies: a dark lightmap scales
        // the clamped 1.0, not the unclamped 1.37.
        let dark = first_person_lit_color(white, [0.0, 1.0, 0.0], lights, [0.25, 0.5, 0.75]);
        assert_eq!(dark, [0.25, 0.5, 0.75, 1.0]);
        // Alpha is the vertex alpha, untouched by lighting.
        let ghost =
            first_person_lit_color([1.0, 0.5, 0.25, 0.3], [0.0, -1.0, 0.0], lights, [1.0; 3]);
        assert_eq!(ghost[3], 0.3);
    }

    /// The empty arm used to be drawn full bright on every face. With the bake, each face gets
    /// its own standard-item-lighting shade, and the eye's lightmap texel scales all of them.
    #[test]
    fn rebuilt_hand_vertices_carry_the_standard_item_lighting_and_the_eye_lightmap() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let mut phase = FirstPersonPhase::new(&gpu);
        phase.render_empty_arm = true;
        let cam = Camera {
            yaw: 37.0,
            pitch: 12.0,
            ..Default::default()
        };
        phase.rebuild_vertices(cam.pitch, 0);
        assert!(
            !phase.arm_vertices.is_empty(),
            "the empty arm emits geometry"
        );
        let unlit = phase.arm_vertices.clone();
        phase.apply_hand_lighting(&cam);
        let lights = first_person_light_directions(cam.pitch, cam.yaw, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (lit, raw) in phase.arm_vertices.iter().zip(&unlit) {
            assert_eq!(
                lit.color,
                first_person_lit_color(raw.color, raw.normal, lights, [1.0; 3])
            );
        }
        let shades: std::collections::BTreeSet<u32> = phase
            .arm_vertices
            .iter()
            .map(|v| (v.color[0] * 1000.0) as u32)
            .collect();
        assert!(
            shades.len() > 1,
            "the arm's faces must not all share one shade"
        );

        // Night at block light 0: the lightmap texel darkens everything.
        phase.set_lightmap_params(crate::sky::lightmap::LightmapParams {
            sun_brightness: 0.2,
            ..Default::default()
        });
        phase.combined_light = 0;
        let texel = phase
            .lightmap
            .as_ref()
            .map(|(_, table)| table.sample(0, 0))
            .expect("params were set");
        phase.rebuild_vertices(cam.pitch, 0);
        phase.apply_hand_lighting(&cam);
        let lightmap = [
            f32::from(texel[0]) / 255.0,
            f32::from(texel[1]) / 255.0,
            f32::from(texel[2]) / 255.0,
        ];
        assert!(lightmap[0] < 0.5, "a dark texel: {texel:?}");
        for (lit, raw) in phase.arm_vertices.iter().zip(&unlit) {
            assert_eq!(
                lit.color,
                first_person_lit_color(raw.color, raw.normal, lights, lightmap)
            );
        }
    }

    #[test]
    fn equipped_item_latch_waits_for_vanillas_strict_threshold() {
        let mut state = FirstPersonRenderState {
            equipped_progress: 0.6,
            item_to_render: Some(1),
            ..FirstPersonRenderState::default()
        };
        // Java clamps the delta, so the first step still sits above the threshold.
        assert!(!state.update_equipped_item(Some(2), 4));
        assert!(state.equipped_progress > FIRST_PERSON_EQUIP_SWAP_THRESHOLD);
        assert_eq!(state.item_to_render, Some(1));
        // The next step falls below .1 and latches the new stack/slot.
        assert!(state.update_equipped_item(Some(2), 4));
        assert!(state.equipped_progress < FIRST_PERSON_EQUIP_SWAP_THRESHOLD);
        assert_eq!(state.item_to_render, Some(2));
        assert_eq!(state.equipped_item_slot, 4);
    }

    #[test]
    fn swing_guard_and_wrap_match_entity_living_base() {
        let mut state = FirstPersonRenderState::default();
        assert!(state.swing_item(6));
        assert_eq!(state.swing_progress_int, -1);
        // The negative sentinel is explicitly restartable, as in Java.
        assert!(state.swing_item(6));
        state.tick_swing(6);
        assert_eq!(state.swing_progress_int, 0);
        assert!(!state.swing_item(6));
        state.tick_swing(6);
        assert_eq!(state.swing_progress_int, 1);

        state.swing_progress = 0.0;
        state.prev_swing_progress = 5.0 / 6.0;
        // A completed swing wraps the delta through +1 instead of snapping backwards.
        let amount = state.swing_amount(0.5);
        assert!((amount - (5.0 / 6.0 + (1.0 - 5.0 / 6.0) * 0.5)).abs() < 1e-6);
    }

    #[test]
    fn first_person_item_and_arm_poses_keep_java_anchor_constants() {
        let item_origin = first_person_item_matrix(0.0, 0.0).transform_point3(Vec3::ZERO);
        assert!((item_origin.x - 0.56).abs() < 1e-6);
        assert!((item_origin.y + 0.52).abs() < 1e-6);
        assert!((item_origin.z + 0.71999997).abs() < 1e-6);

        let arm_origin = first_person_arm_matrix(0.0, 0.0).transform_point3(Vec3::ZERO);
        // The arm path must not collapse onto the held-item anchor: it includes the nested
        // RenderPlayer 54/64/-62 degree transform and its .25/-.85/.75 translation.
        assert!((arm_origin - item_origin).length() > 0.1);

        let idle = first_person_item_pose(0.0, 0.5, FirstPersonUseAction::Idle, 0, 32, 0.0);
        let block = first_person_item_pose(0.0, 0.5, FirstPersonUseAction::Block, 0, 32, 0.0);
        assert_ne!(idle, block);
    }

    #[test]
    fn empty_arm_emission_intersects_first_person_clip_frustum() {
        // This is the ordinary ItemRenderer.renderPlayerArm path. The map-only
        // renderRightArm helper must not be appended to this matrix: with that
        // suffix the complete classic arm is behind/outside the near/frustum planes.
        let player = PlayerModel::new(0.0, false);
        let mut vertices = Vec::new();
        player.biped.model.emit_part(
            player.biped.parts.right_arm,
            0.0625,
            first_person_arm_matrix(0.0, 0.0), // settled equip amount
            &mut vertices,
        );
        assert!(
            !vertices.is_empty(),
            "the right arm must emit model vertices"
        );

        // Keep the initial/equipping pose in the same contract: it is a distinct, finite
        // transform even though the settled pose is the one expected to reach the viewport.
        let settled_origin = first_person_arm_matrix(0.0, 0.0).transform_point3(Vec3::ZERO);
        let incoming_origin = first_person_arm_matrix(1.0, 0.0).transform_point3(Vec3::ZERO);
        assert!((incoming_origin.y - settled_origin.y + 0.6).abs() < 1e-6);
        assert!(incoming_origin.is_finite());

        let projection = first_person_projection(&Camera::default());
        let inside = vertices
            .iter()
            .filter(|vertex| {
                let clip = projection * Vec3::from(vertex.pos).extend(1.0);
                clip.w > 0.0
                    && clip.x >= -clip.w
                    && clip.x <= clip.w
                    && clip.y >= -clip.w
                    && clip.y <= clip.w
                    && clip.z >= 0.0
                    && clip.z <= clip.w
            })
            .count();
        assert!(
            inside > 0,
            "ordinary empty-arm geometry must reach the first-person clip volume"
        );
    }

    #[test]
    fn first_person_arm_pose_copies_idle_roll_and_wear_pivot_for_both_models() {
        let arm_matrix = first_person_arm_matrix(0.0, 0.0);
        for small_arms in [false, true] {
            let mut player = PlayerModel::new(0.0, small_arms);
            player.set_rotation_angles(&AnimParams::still());
            let base = player.biped.parts.right_arm;
            let wear = player.wear.right_arm;
            let base_part = player.biped.model.get(base);
            let wear_part = player.biped.model.get(wear);

            assert!((base_part.rotate_angle[2] - 0.1).abs() < 1e-6);
            assert_eq!(base_part.rotation_point, wear_part.rotation_point);
            assert_eq!(base_part.rotate_angle, wear_part.rotate_angle);

            let mut base_vertices = Vec::new();
            let mut wear_vertices = Vec::new();
            player
                .biped
                .model
                .emit_part(base, 0.0625, arm_matrix, &mut base_vertices);
            player
                .biped
                .model
                .emit_part(wear, 0.0625, arm_matrix, &mut wear_vertices);
            assert!(!base_vertices.is_empty());
            assert!(!wear_vertices.is_empty());

            let centroid = |vertices: &[EntityVertex]| {
                vertices
                    .iter()
                    .map(|vertex| Vec3::from(vertex.pos))
                    .sum::<Vec3>()
                    / vertices.len() as f32
            };
            assert!(
                (centroid(&base_vertices) - centroid(&wear_vertices)).length() < 1e-6,
                "wear must share the posed arm pivot for small_arms={small_arms}"
            );

            player.biped.model.get_mut(wear).show_model = false;
            let mut hidden_wear = Vec::new();
            player
                .biped
                .model
                .emit_part(wear, 0.0625, arm_matrix, &mut hidden_wear);
            assert!(hidden_wear.is_empty());
        }
    }

    /// 3D Skins' first-person sleeve: without a mesh it is exactly the
    /// flat `emit_part`; with one, every voxel quad (unculled: the hand's space is not
    /// camera-centred) under the same posed matrix; a hidden sleeve emits nothing either way.
    #[test]
    fn the_first_person_sleeve_is_flat_without_a_mesh_and_voxels_with_one() {
        use crate::entity::skin3d::{SkinLayerMesh, WearPart};
        let arm_matrix = first_person_arm_matrix(0.0, 0.0);
        let mut player = PlayerModel::new(0.0, false);
        player.set_rotation_angles(&AnimParams::still());
        let wear = player.wear.right_arm;
        let (mut flat, mut via) = (Vec::new(), Vec::new());
        player
            .biped
            .model
            .emit_part(wear, 0.0625, arm_matrix, &mut flat);
        emit_first_person_sleeve(&player, wear, None, arm_matrix, &mut via);
        assert_eq!(
            bytemuck::cast_slice::<_, u32>(&flat[..]),
            bytemuck::cast_slice::<_, u32>(&via[..])
        );
        let mesh = SkinLayerMesh::build(&[255u8; 64 * 64 * 4], 64, 64, false).unwrap();
        let quads = mesh.part(WearPart::RightSleeve);
        let mut voxels = Vec::new();
        emit_first_person_sleeve(&player, wear, Some(quads), arm_matrix, &mut voxels);
        assert_eq!(voxels.len(), quads.len() * 6);
        player.biped.model.get_mut(wear).show_model = false;
        let mut hidden = Vec::new();
        emit_first_person_sleeve(&player, wear, Some(quads), arm_matrix, &mut hidden);
        assert!(hidden.is_empty());
    }

    #[test]
    fn held_map_pitch_and_geometry_follow_vanilla_layering() {
        assert!((first_person_map_angle_from_pitch(-90.0) - 1.0).abs() < 1e-6);
        assert!((first_person_map_angle_from_pitch(90.0) - 0.0).abs() < 1e-6);
        assert!((first_person_map_angle_from_pitch(0.0) - 1.0).abs() < 1e-6);

        let map = FirstPersonMap {
            id: 17,
            rgba: Some(vec![255; 128 * 128 * 4]),
            decorations: vec![FirstPersonMapDecoration {
                icon_type: 6,
                x: 4,
                z: -8,
                direction: 0,
            }],
        };
        let mut background = Vec::new();
        let mut image = Vec::new();
        let mut icons = Vec::new();
        push_first_person_map_geometry(
            &map,
            Mat4::IDENTITY,
            &mut background,
            &mut image,
            &mut icons,
        );
        assert_eq!(background.len(), 6, "the 142x142 map backing is one quad");
        assert_eq!(image.len(), 6, "a received map record adds one image quad");
        assert_eq!(icons.len(), 6, "each decoration is one 4x4-sheet quad");
        assert_eq!(background[0].pos, [-7.0, -7.0, 0.0]);
        assert_eq!(image[0].uv, [0.0, 0.0]);
        assert_eq!(icons[0].uv, [0.5, 0.25]);
        assert_eq!(icons[0].pos, [61.5, 64.5, -0.02]);
        assert!(icons[0].pos[2] < image[0].pos[2]);

        let blank = FirstPersonMap {
            id: 17,
            rgba: None,
            decorations: Vec::new(),
        };
        background.clear();
        image.clear();
        icons.clear();
        push_first_person_map_geometry(
            &blank,
            Mat4::IDENTITY,
            &mut background,
            &mut image,
            &mut icons,
        );
        assert_eq!(background.len(), 6);
        assert!(
            image.is_empty(),
            "unknown map ids still draw only the backing"
        );
    }

    #[test]
    fn held_map_shader_keeps_low_alpha_unexplored_texels() {
        assert!(FIRST_PERSON_MAP_WGSL.contains("return vec4<f32>"));
        assert!(!FIRST_PERSON_MAP_WGSL.contains("discard"));
        assert!(FIRST_PERSON_WGSL.contains("discard"));
    }

    /// `set_item_map` re-latches (and marks the texture dirty) exactly when
    /// the id, pixels or decorations change, as the old build-then-compare did, and an unchanged
    /// tick no longer copies the 64 KiB image.
    #[test]
    fn held_map_snapshot_relatches_only_when_it_changes() {
        let gpu = crate::Gpu::headless(4, 4, crate::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let mut phase = FirstPersonPhase::new_with_anaglyph(&gpu, false);
        let pixels = vec![9_u8; 128 * 128 * 4];
        let mut changed = pixels.clone();
        changed[400] = 10;
        let icon = FirstPersonMapDecoration {
            icon_type: 1,
            x: 2,
            z: 3,
            direction: 4,
        };
        phase.set_item_map(5, Some(&pixels), &[icon]);
        assert!(phase.map_texture_dirty);
        phase.map_texture_dirty = false;
        let image_ptr = |phase: &FirstPersonPhase| {
            phase
                .item_map
                .as_ref()
                .and_then(|map| map.rgba.as_ref().map(Vec::as_ptr))
        };
        let latched = image_ptr(&phase);
        phase.set_item_map(5, Some(&pixels), &[icon]);
        assert!(
            !phase.map_texture_dirty,
            "an unchanged snapshot is not re-latched"
        );
        assert_eq!(image_ptr(&phase), latched, "nor is its image copied");
        let changed: &[u8] = &changed;
        let cases = [
            (5, Some(changed), vec![icon]),
            (6, Some(changed), vec![icon]),
            (6, Some(changed), Vec::new()),
            (6, None, Vec::new()),
        ];
        for (id, rgba, decorations) in cases {
            phase.map_texture_dirty = false;
            phase.set_item_map(id, rgba, &decorations);
            let map = phase.item_map.as_ref().expect("latched");
            assert_eq!(
                (map.id, map.rgba.as_deref(), map.decorations.as_slice()),
                (id, rgba, decorations.as_slice())
            );
            assert_eq!(phase.map_texture_dirty, rgba.is_some());
        }
        phase.set_item_map(6, Some(&pixels[..10]), &[]);
        assert!(
            phase
                .item_map
                .as_ref()
                .is_some_and(|map| map.rgba.is_none()),
            "a malformed image leaves the previous snapshot"
        );
    }

    #[test]
    fn first_person_anaglyph_upload_matches_java_channels_and_preserves_alpha() {
        let source = [
            255, 0, 0, 7, // red, translucent
            0, 255, 0, 19, // green
            0, 0, 255, 31, // blue
            128, 128, 128, 64,
        ];
        assert_eq!(first_person_upload_rgba(&source, false), source);
        assert_eq!(
            first_person_upload_rgba(&source, true),
            [76, 76, 76, 7, 178, 150, 0, 19, 0, 28, 178, 31, 128, 128, 128, 64,],
            "Java's integer anaglyph transform mixes RGB only"
        );
    }

    #[test]
    fn first_person_atlas_accepts_mips_and_replays_raw_animation_across_toggles() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let base = [255u8, 0, 0, 7].repeat(4 * 4);
        let mip = [0u8, 0, 255, 19].repeat(2 * 2);
        let atlas = Atlas {
            texture: base.clone(),
            mipmaps: vec![mip.clone()],
            w: 4,
            h: 4,
        };
        let mut phase = FirstPersonPhase::new(&gpu);
        phase.set_atlas(&gpu, &atlas);

        let animated = AtlasAnimationUpload {
            mip_level: 1,
            x: 1,
            y: 0,
            width: 1,
            height: 2,
            rgba: [0u8, 255, 0, 23].repeat(2),
        };
        assert!(phase.upload_atlas_animation(&gpu, &animated));

        let mut expected_mip = mip;
        expected_mip[4..8].copy_from_slice(&animated.rgba[..4]);
        expected_mip[12..16].copy_from_slice(&animated.rgba[4..8]);
        let raw_levels = vec![base, expected_mip];
        assert_eq!(
            phase.atlas.source_levels.borrow().as_slice(),
            raw_levels.as_slice(),
            "animated mip writes must retain raw source bytes"
        );
        assert_eq!(phase.atlas.source_levels.borrow().len(), 2);

        phase.set_anaglyph_with_gpu(&gpu, true);
        assert!(phase.anaglyph_enabled());
        assert_eq!(phase.atlas.source_levels.borrow().as_slice(), raw_levels);
        phase.set_anaglyph_with_gpu(&gpu, false);
        assert!(!phase.anaglyph_enabled());
        assert_eq!(phase.atlas.source_levels.borrow().as_slice(), raw_levels);
    }

    #[test]
    fn first_person_item_model_scale_matches_render_item_pre_transform() {
        let display = mc_assets::model::ItemTransform {
            rotation: [0.0; 3],
            translation: [1.0, 0.0, 0.0],
            scale: [1.0; 3],
        };
        let gui_3d = first_person_item_model_transform(true, &display);
        let flat = first_person_item_model_transform(false, &display);

        // The ItemRenderer wrapper supplies ×2 for GUI-3D, RenderItem.preTransform supplies ×2
        // for flat models, and RenderItem always scales by .5. Both paths therefore span 1.0.
        assert!((gui_3d.transform_vector3(Vec3::X).length() - 1.0).abs() < 1e-6);
        assert!((flat.transform_vector3(Vec3::X).length() - 1.0).abs() < 1e-6);
        // Whichever branch supplies ×2, it runs before ItemCameraTransforms.applyTransform and
        // therefore scales the display translation, matching Java's matrix-stack order.
        assert!((gui_3d.transform_point3(Vec3::ZERO).x - 2.0).abs() < 1e-6);
        assert!((flat.transform_point3(Vec3::ZERO).x - 2.0).abs() < 1e-6);
    }

    #[test]
    fn first_person_translucent_block_disables_depth_writes_only_for_gui_3d_models() {
        assert!(
            !first_person_item_writes_depth(true, true),
            "a GUI-3D ice/glass stack follows ItemRenderer.depthMask(false)"
        );
        assert!(
            first_person_item_writes_depth(false, true),
            "opaque GUI-3D blocks keep ordinary depth writes"
        );
        assert!(
            first_person_item_writes_depth(true, false),
            "Java's shouldRenderItemIn3D guard keeps flat overrides depth-writing"
        );
    }

    #[test]
    fn first_person_builtin_transform_keeps_outer_scale_before_display_translation() {
        let display = mc_assets::model::ItemTransform {
            rotation: [0.0, -180.0, 0.0],
            translation: [1.0, 0.0, 0.0],
            scale: [0.55, 0.55, 0.55],
        };
        let transform = first_person_builtin_item_transform(&display);

        // `ItemRenderer.renderItem`'s GUI-3D scale is before the display transform, so the
        // one-block display translation becomes two blocks.  The skull JSON's .55 scale still
        // remains visible here; the builtin skull helper adds its own ×2 after common .5.
        assert!((transform.transform_point3(Vec3::ZERO).x - 2.0).abs() < 1e-6);
        assert!((transform.transform_vector3(Vec3::X).length() - 1.1).abs() < 1e-6);
    }

    #[test]
    fn first_person_builtin_models_emit_entity_geometry_and_fixed_texture_keys() {
        let mut models = TileModels::new(false);
        let mut vertices = Vec::new();
        let chest_key = models.emit_builtin_item(
            &BuiltinTileItem::Chest { trapped: false },
            first_person_builtin_item_transform(&mc_assets::model::ItemTransform::DEFAULT),
            &mut vertices,
        );
        assert_eq!(
            chest_key.as_deref(),
            Some("textures/entity/chest/normal.png")
        );
        assert_eq!(vertices.len(), 3 * 6 * 6, "held chest uses the TESR model");

        vertices.clear();
        let skull_key = models.emit_builtin_item(
            &BuiltinTileItem::Skull { skull_type: 2 },
            first_person_builtin_item_transform(&mc_assets::model::ItemTransform {
                rotation: [0.0, -180.0, 0.0],
                translation: [0.0; 3],
                scale: [0.55; 3],
            }),
            &mut vertices,
        );
        assert_eq!(
            skull_key.as_deref(),
            Some("textures/entity/zombie/zombie.png")
        );
        assert!(
            !vertices.is_empty(),
            "held skull uses the concrete head model"
        );

        // Keep a static control at the caller seam: a builtin descriptor must bind the entity
        // texture, and must never fall through to the baked atlas/glint replay.
        let source = include_str!("pipeline.rs");
        assert!(source.contains("item_builtin_models.emit_builtin_item"));
        assert!(source.contains("item_builtin_texture"));
        assert!(source.contains("item_builtin_texture.as_ref().unwrap_or(&self.atlas)"));
        assert!(source.contains("set_builtin_item_model"));
        assert!(source.contains("skin::shared_skin"));
    }

    #[test]
    fn first_person_player_skull_switches_from_default_to_shared_normalized_pixels_without_gpu() {
        let owner = crate::entity::tilepass::BuiltinSkullOwner::new(
            "mc-render/entity/skin/13572468-9abc-def0-0123-456789abcdef",
            crate::entity::SkinModel::Slim,
            crate::entity::SkinModel::Slim,
        );
        let item = BuiltinTileItem::PlayerSkull {
            skull_type: 3,
            owner: owner.clone(),
        };

        let (fallback_key, fallback_pixels) = first_person_builtin_texture_source(&item, false);
        assert_eq!(fallback_key.as_deref(), Some("textures/entity/alex.png"));
        assert!(
            fallback_pixels.is_none(),
            "unpublished skins use Alex/Steve first"
        );

        let texture = crate::entity::SkinTexture {
            rgba: vec![
                173;
                (crate::entity::SkinTexture::WIDTH * crate::entity::SkinTexture::HEIGHT * 4)
                    as usize
            ]
            .into(),
            layout: crate::entity::SkinLayout::Modern64x64,
            model: crate::entity::SkinModel::Slim,
        };
        let (mut cache, _channel) = crate::entity::SkinCache::new();
        cache.insert(&owner.cache_key, texture.clone());

        let (ready_key, ready_pixels) = first_person_builtin_texture_source(&item, false);
        assert_eq!(ready_key.as_deref(), Some(owner.cache_key.as_str()));
        let ready_pixels = ready_pixels.expect("shared skin is selected after publication");
        assert_eq!(ready_pixels, texture);
        assert_eq!(ready_pixels.rgba.len(), 64 * 64 * 4);

        let source = include_str!("pipeline.rs");
        assert!(source.contains("first_person_builtin_texture_source"));
        assert!(source.contains("first-person builtin player skull"));
        assert!(source.contains("SkinTexture::WIDTH"));
    }

    #[test]
    fn first_person_baked_item_applies_only_latched_tagged_quad_tints() {
        let vertex = |x: f32, y: f32| mc_assets::Vertex {
            pos: [x, y, 0.0],
            uv: [x, y],
        };
        let quad = |tint_index| mc_assets::BakedQuad {
            verts: [
                vertex(0.0, 0.0),
                vertex(1.0, 0.0),
                vertex(1.0, 1.0),
                vertex(0.0, 1.0),
            ],
            face: None,
            shade_face: None,
            tint_index,
            shade: false,
        };
        let mut model = BakedModel::default();
        model
            .general_quads
            .extend([quad(Some(0)), quad(None), quad(Some(7))]);

        let mut out = Vec::new();
        emit_first_person_item_model(&model, Mat4::IDENTITY, &[0x48_B5_18], &mut out);

        assert_eq!(out.len(), 18);
        assert!(out[..6]
            .iter()
            .all(|vertex| vertex.color == [72.0 / 255.0, 181.0 / 255.0, 24.0 / 255.0, 1.0]));
        assert!(out[6..]
            .iter()
            .all(|vertex| vertex.color == EntityVertex::WHITE));
    }

    #[test]
    fn first_person_glint_replays_two_exact_render_item_uv_passes() {
        let vertex = |x: f32, y: f32, u: f32, v: f32| mc_assets::Vertex {
            pos: [x, y, 0.0],
            uv: [u, v],
        };
        let mut model = BakedModel::default();
        model.general_quads.push(mc_assets::BakedQuad {
            verts: [
                vertex(0.0, 0.0, 0.125, 0.25),
                vertex(1.0, 0.0, 0.875, 0.25),
                vertex(1.0, 1.0, 0.875, 0.75),
                vertex(0.0, 1.0, 0.125, 0.75),
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: false,
        });

        let mut base = Vec::new();
        crate::entity::item::emit_baked_model(&model, Mat4::IDENTITY, &mut base);
        let mut glint = Vec::new();
        emit_baked_model_glint(&model, Mat4::IDENTITY, 1_500, &mut glint);

        assert_eq!(base.len(), 6, "the fixture is one quad / two triangles");
        assert_eq!(
            glint.len(),
            base.len() * 2,
            "RenderItem emits two foil passes"
        );
        for (base_vertex, glint_vertex) in base.iter().zip(&glint[..base.len()]) {
            assert_eq!(base_vertex.pos, glint_vertex.pos);
            assert_eq!(base_vertex.normal, glint_vertex.normal);
            assert_eq!(glint_vertex.color, crate::entity::item::ITEM_GLINT_COLOR);
        }
        for (base_vertex, glint_vertex) in base.iter().zip(&glint[base.len()..]) {
            assert_eq!(base_vertex.pos, glint_vertex.pos);
            assert_eq!(base_vertex.normal, glint_vertex.normal);
            assert_eq!(glint_vertex.color, crate::entity::item::ITEM_GLINT_COLOR);
        }

        let uv = base[0].uv;
        assert_eq!(
            glint[0].uv,
            crate::entity::item::item_glint_uv(uv, 1_500, 0)
        );
        assert_eq!(
            glint[base.len()].uv,
            crate::entity::item::item_glint_uv(uv, 1_500, 1)
        );
        assert_ne!(
            glint[0].uv,
            glint[base.len()].uv,
            "the independent 3000/4873 ms matrices must not collapse"
        );
    }

    #[test]
    fn first_person_glint_pipeline_keeps_java_blend_depth_and_repeat_contract() {
        let glint = first_person_glint_blend_state();
        let expected = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Src,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        assert_eq!(glint.color, expected, "GL_SRC_COLOR + GL_ONE colour blend");
        assert_eq!(
            glint.alpha, expected,
            "two-argument Java blend applies to alpha"
        );

        let source = include_str!("pipeline.rs");
        assert!(source.contains("fn fs_glint"));
        let (_, glint_shader) = source
            .split_once("fn fs_glint")
            .expect("the dedicated glint fragment entry point remains present");
        let (glint_shader, _) = glint_shader
            .split_once("// Java temporarily disables")
            .expect("the first-person shader literal remains closed after fs_glint");
        assert!(
            !glint_shader.contains("discard"),
            "depth-equal foil replay must not alpha-discard against the standalone texture"
        );
        assert!(source.contains("DepthAttachment::Clear"));
        assert!(source.contains("CompareFunction::Equal"));
        assert!(source.contains("depth_write_enabled: Some(depth_write_enabled)"));
        assert!(source.contains("address_mode_u: address_mode"));
        assert!(source.contains("AddressMode::Repeat"));
    }

    /// The first-person foil follows its `.mcmeta` like `SimpleTexture`:
    /// vanilla's `{"texture": {"blur": true}}` is linear and repeating.
    #[test]
    fn first_person_glint_sampler_follows_the_texture_metadata() {
        let vanilla = mc_assets::TextureMetadata {
            blur: true,
            ..Default::default()
        };
        assert_eq!(
            glint_sampler_modes(&vanilla),
            (wgpu::FilterMode::Linear, wgpu::AddressMode::Repeat)
        );
        // A pack whose foil has no sidecar gets vanilla's GL_NEAREST, as SimpleTexture would.
        assert_eq!(
            glint_sampler_modes(&mc_assets::TextureMetadata::default()),
            (wgpu::FilterMode::Nearest, wgpu::AddressMode::Repeat)
        );
        let clamped = mc_assets::TextureMetadata {
            clamp: true,
            ..Default::default()
        };
        assert_eq!(
            glint_sampler_modes(&clamped).1,
            wgpu::AddressMode::ClampToEdge
        );
        let source = include_str!("pipeline.rs");
        let (_, loader) = source
            .split_once("pub fn load_glint_texture")
            .expect("the first-person foil loader");
        let loader = loader.split("\n    }").next().unwrap_or(loader);
        assert!(loader.contains("load_png_rgba") && loader.contains("glint_sampler_modes"));
    }

    /// The seam a resource-pack reload rebuilds through: everything goes, and the list is
    /// immediately reusable rather than poisoned.
    #[test]
    fn clear_drops_every_phase_and_leaves_the_list_reusable() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::SKY, Box::new(Noop("sky")));
        list.insert(PhaseOrder::TERRAIN_SOLID, Box::new(Noop("terrain")));
        list.insert(PhaseOrder::UI, Box::new(Noop("ui")));
        assert_eq!(list.clear(), 3);
        assert!(list.is_empty());
        assert_eq!(list.names().count(), 0);
        assert_eq!(list.clear(), 0, "clearing an empty list is a no-op");

        list.insert(PhaseOrder::UI, Box::new(Noop("ui")));
        assert_eq!(list.names().collect::<Vec<_>>(), ["ui"]);
    }

    /// Clearing must also drop the scene-mode intermediates, or a rebuilt list would keep an
    /// allocation nothing can opt into any more.
    #[test]
    fn clear_takes_the_scene_targets_with_it() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::POST, Box::new(SceneHungry(true)));
        assert!(list.scene_mode());
        list.clear();
        assert!(!list.scene_mode());
    }

    #[test]
    fn remove_named_drops_matching_phases() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::PARTICLES, Box::new(Noop("particles")));
        list.insert(PhaseOrder::UI, Box::new(Noop("ui")));
        assert_eq!(list.remove_named("particles"), 1);
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }

    /// #14 step 1b: `render_deferred` records the same frame as `render`, and a readback right
    /// after it (the F2 screenshot in `after_render`) submits it first, so the readback sees this
    /// frame and not the previous one.
    #[test]
    fn a_deferred_frame_reads_back_like_a_submitted_one() {
        let mut gpu = Gpu::headless(4, 4, crate::context::GpuOptions::default())
            .expect("this qualification requires a working GPU adapter");
        let mut list = PhaseList::new();
        let cam = Camera::default();
        let colours = [
            wgpu::Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            wgpu::Color {
                r: 0.0,
                g: 1.0,
                b: 0.0,
                a: 1.0,
            },
        ];
        let mut serial = Vec::new();
        for colour in colours {
            list.clear_color = colour;
            let frame = gpu.acquire().expect("headless frame");
            list.render(&gpu, frame.view(), &cam, 0.0);
            serial.push(gpu.read_pixels().expect("serial pixels"));
        }
        assert_ne!(serial[0], serial[1], "the two frames must be told apart");
        for (colour, expected) in colours.into_iter().zip(&serial) {
            list.clear_color = colour;
            let frame = gpu.acquire().expect("headless frame");
            list.render_deferred(&gpu, frame.view(), &cam, 0.0);
            assert_eq!(&gpu.read_pixels().expect("deferred pixels"), expected);
        }
    }

    struct SceneHungry(bool);
    impl RenderPhase for SceneHungry {
        fn record(
            &mut self,
            _: &Gpu,
            _: &mut wgpu::CommandEncoder,
            _: &wgpu::TextureView,
            _: &Camera,
        ) {
        }
        fn name(&self) -> &'static str {
            "scene-hungry"
        }
        fn enabled(&self) -> bool {
            self.0
        }
        fn needs_scene_color(&self) -> bool {
            true
        }
    }

    #[test]
    fn scene_mode_is_off_unless_an_enabled_phase_opts_in() {
        let mut list = PhaseList::new();
        list.insert(PhaseOrder::SKY, Box::new(Noop("sky")));
        list.insert(PhaseOrder::UI, Box::new(Noop("ui")));
        assert!(
            !list.scene_mode(),
            "phases that never opt in must leave the frame path alone"
        );

        // A phase that opts in but is disabled must not drag the frame onto the scene path.
        list.insert(PhaseOrder::POST, Box::new(SceneHungry(false)));
        assert!(!list.scene_mode());

        list.insert(PhaseOrder::POST, Box::new(SceneHungry(true)));
        assert!(list.scene_mode());
    }

    #[test]
    fn a_default_phase_does_not_opt_into_scene_input() {
        // The provided default is what keeps this addition backwards-compatible for the four
        // in-crate `RenderPhase` implementors and `mc_ui::widget::UiPhase`.
        let phase = Noop("plain");
        assert!(!RenderPhase::needs_scene_color(&phase));
    }

    #[test]
    fn camera_uniform_is_std140_sized() {
        // 3 mat4 + 3 vec4 = 192 + 48. A uniform buffer binding must be 16-byte aligned.
        assert_eq!(std::mem::size_of::<CameraUniform>(), 240);
        assert_eq!(std::mem::size_of::<CameraUniform>() % 16, 0);
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
        let mut direct = FirstPersonPhase::new(&gpu);
        let mut captured = FirstPersonPhase::new(&gpu);
        direct.set_atlas(&gpu, &atlas);
        captured.set_atlas(&gpu, &atlas);
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
                *direct.atlas.source_levels.borrow(),
                expected,
                "toggle must retain raw bytes"
            );
            assert_eq!(
                *captured.atlas.source_levels.borrow(),
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
                            std::ptr::eq(texture, &captured.atlas._texture),
                            "sink must receive this owner's texture"
                        );
                        assert_eq!(
                            setting, enabled,
                            "sink must receive the owner's current setting"
                        );
                        // This borrow also proves the mutation borrow ended before the callback.
                        assert_eq!(
                            *captured.atlas.source_levels.borrow(),
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
                    *captured.atlas.source_levels.borrow(),
                    expected,
                    "capture raw bytes at record {index}"
                );
                assert_eq!(
                    *direct.atlas.source_levels.borrow(),
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
    }
}
