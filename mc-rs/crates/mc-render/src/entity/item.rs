//! Dropped items: the bob, the spin, the stack, and the 2D/3D split.
//!
//! java: `client/renderer/entity/RenderEntityItem.java` (`doRender`, `func_177077_a`,
//! `func_177078_a`), `client/renderer/entity/RenderItem.java`,
//! `client/renderer/block/model/ItemCameraTransforms.java`.
//!
//! # Two kinds of dropped item
//!
//! Vanilla asks the baked model whether it is `isGui3d()`. That flag is what separates a
//! *block* — a real cube, drawn once, spinning — from a flat *item* icon, which is drawn as a
//! stack of thin slices `0.046875` blocks apart so a pile of arrows looks like a pile rather
//! than a single card. Both come from [`mc_assets::item_model_for`], so the geometry is S09's;
//! what lives here is the placement, which is pure arithmetic and fully testable.
//!
//! # The count rule is a step function, not a division
//!
//! `func_177078_a` maps stack size to a *render* count of 1-5 in fixed steps
//! ([`render_count`]). Extra copies after the first are jittered by a
//! [`mc_math::JavaRandom`] **reseeded to 187 every frame** (`field_177079_e.setSeed(187L)` at
//! the top of `doRender`), so the jitter is stable frame to frame instead of shimmering. That
//! reseed is easy to miss and the result looks obviously wrong without it.

use glam::{Mat4, Vec3};
use mc_assets::{BakedModel, BakedQuad};
use mc_math::JavaRandom;

use super::EntityVertex;

/// `RenderItem.RES_ITEM_GLINT` / `LayerArmorBase.ENCHANTED_ITEM_GLINT_RES`.
///
/// Both the item renderer and armour layer bind this same standalone texture.  It is kept as a
/// public entity-side key so callers can load it through [`super::render::EntityPhase`] rather
/// than accidentally treating it as an atlas sprite.
pub const ENCHANTED_ITEM_GLINT_TEXTURE: &str = "textures/misc/enchanted_item_glint.png";

/// The two `RenderItem.renderEffect` texture-matrix periods, in milliseconds.
pub const ITEM_GLINT_FIRST_PERIOD_MS: u64 = 3_000;
/// The second `RenderItem.renderEffect` texture-matrix period, in milliseconds.
pub const ITEM_GLINT_SECOND_PERIOD_MS: u64 = 4_873;
/// `RenderItem.renderEffect`'s texture-matrix scale.
pub const ITEM_GLINT_TEXTURE_SCALE: f32 = 8.0;

/// `WorldRenderer.putColor4(-8372020)` as linear RGBA bytes.  The alpha is opaque: the glint's
/// strength comes from `GL_SRC_COLOR, GL_ONE`, not from an invented translucent tint.
pub const ITEM_GLINT_COLOR: [f32; 4] = [128.0 / 255.0, 64.0 / 255.0, 204.0 / 255.0, 1.0];

/// `LayerArmorBase.renderGlint`'s per-pass primary colour after its `0.5 * 0.76` factors.
pub const ARMOR_GLINT_COLOR: [f32; 4] = [0.5 * 0.76, 0.25 * 0.76, 0.8 * 0.76, 1.0];

/// One animated `RenderItem.renderEffect` pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemGlintPass {
    /// The clock period used for this pass, in milliseconds.
    pub period_ms: u64,
    /// Java's post-scale translation argument (`fraction / 8`).
    pub translation: f32,
    /// The texture-matrix rotation in degrees.
    pub rotation_degrees: f32,
}

/// Build the two `RenderItem.renderEffect` states at a Java-compatible system-clock value.
///
/// Java computes `(Minecraft.getSystemTime() % period) / period / 8` in float arithmetic, then
/// calls `scale(8)`, `translate(±f, 0, 0)`, and `rotate(angle)`.  Keeping this policy here lets
/// dropped items, item-frame contents, and held item models share one implementation.
#[must_use]
pub fn item_glint_passes(system_time_ms: u64) -> [ItemGlintPass; 2] {
    let first = (system_time_ms % ITEM_GLINT_FIRST_PERIOD_MS) as f32
        / ITEM_GLINT_FIRST_PERIOD_MS as f32
        / ITEM_GLINT_TEXTURE_SCALE;
    let second = (system_time_ms % ITEM_GLINT_SECOND_PERIOD_MS) as f32
        / ITEM_GLINT_SECOND_PERIOD_MS as f32
        / ITEM_GLINT_TEXTURE_SCALE;
    [
        ItemGlintPass {
            period_ms: ITEM_GLINT_FIRST_PERIOD_MS,
            translation: first,
            rotation_degrees: -50.0,
        },
        ItemGlintPass {
            period_ms: ITEM_GLINT_SECOND_PERIOD_MS,
            translation: -second,
            rotation_degrees: 10.0,
        },
    ]
}

/// Apply one `RenderItem.renderEffect` texture-matrix pass to a model UV.
///
/// OpenGL post-multiplies the calls, so the matrix is `scale(8) * translate(±f) * rotate(angle)`.
/// The resulting coordinates intentionally remain unbounded; the entity sampler repeats the
/// glint texture just as vanilla's texture state does.
#[must_use]
pub fn item_glint_uv(uv: [f32; 2], system_time_ms: u64, pass: usize) -> [f32; 2] {
    ItemGlintUv::new(system_time_ms, pass).apply(uv)
}

/// One [`item_glint_uv`] pass with its clock, divide and trigonometry resolved once.
///
/// R2-5/#22: replays used to rebuild both [`item_glint_passes`] (two `u64 %`, divides,
/// `to_radians`, two table lookups) for every vertex. [`item_glint_uv`] is now exactly
/// `new(..).apply(uv)`, so hoisting `new` out of a vertex loop is bit-identical by construction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemGlintUv {
    sin: f32,
    cos: f32,
    translation: f32,
}

impl ItemGlintUv {
    /// Resolves pass `pass` (0 or 1) at `system_time_ms`.
    #[must_use]
    pub fn new(system_time_ms: u64, pass: usize) -> Self {
        let state = item_glint_passes(system_time_ms)
            .get(pass)
            .copied()
            .unwrap_or_else(|| panic!("item glint pass must be 0 or 1, got {pass}"));
        let angle = state.rotation_degrees.to_radians();
        Self {
            sin: mc_math::sin(angle),
            cos: mc_math::cos(angle),
            translation: state.translation,
        }
    }

    /// `scale(8) * translate(±f) * rotate(angle)` applied to one model UV.
    #[inline]
    #[must_use]
    pub fn apply(self, uv: [f32; 2]) -> [f32; 2] {
        let rotated_u = uv[0] * self.cos - uv[1] * self.sin;
        let rotated_v = uv[0] * self.sin + uv[1] * self.cos;
        [
            ITEM_GLINT_TEXTURE_SCALE * (rotated_u + self.translation),
            ITEM_GLINT_TEXTURE_SCALE * rotated_v,
        ]
    }
}

/// One `LayerArmorBase.renderGlint` texture-matrix pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmorGlintPass {
    /// The texture-matrix scale (`1 / 3`).
    pub scale: f32,
    /// The texture-matrix rotation in degrees.
    pub rotation_degrees: f32,
    /// The Y translation applied before the scale/rotation matrix.
    pub translation_y: f32,
}

/// Build the two `LayerArmorBase.renderGlint` states for an interpolated entity age.
///
/// java: `f = entity.ticksExisted + partialTicks`, then for pass `i`:
/// `scale(1/3)`, `rotate(30 - i*60)`, `translate(0, f*(0.001 + i*0.003)*20, 0)`.
#[must_use]
pub fn armor_glint_passes(age_in_ticks: f32) -> [ArmorGlintPass; 2] {
    [0usize, 1].map(|index| ArmorGlintPass {
        scale: 1.0 / 3.0,
        rotation_degrees: 30.0 - index as f32 * 60.0,
        translation_y: age_in_ticks * (0.001 + index as f32 * 0.003) * 20.0,
    })
}

/// Apply one `LayerArmorBase.renderGlint` texture-matrix pass to an armour-model UV.
///
/// Its call order is `scale * rotate * translate`, unlike `RenderItem`'s
/// `scale * translate * rotate`; therefore the Y translation is applied to the UV before the
/// rotation.  Keeping these operations separate prevents the two vanilla glints from silently
/// sharing the item pass's timing or matrix order.
#[must_use]
pub fn armor_glint_uv(uv: [f32; 2], age_in_ticks: f32, pass: usize) -> [f32; 2] {
    ArmorGlintUv::new(age_in_ticks, pass).apply(uv)
}

/// One [`armor_glint_uv`] pass with its trigonometry resolved once — see [`ItemGlintUv`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmorGlintUv {
    scale: f32,
    sin: f32,
    cos: f32,
    translation_y: f32,
}

impl ArmorGlintUv {
    /// Resolves pass `pass` (0 or 1) at the interpolated entity age.
    #[must_use]
    pub fn new(age_in_ticks: f32, pass: usize) -> Self {
        let state = armor_glint_passes(age_in_ticks)
            .get(pass)
            .copied()
            .unwrap_or_else(|| panic!("armor glint pass must be 0 or 1, got {pass}"));
        let angle = state.rotation_degrees.to_radians();
        Self {
            scale: state.scale,
            sin: mc_math::sin(angle),
            cos: mc_math::cos(angle),
            translation_y: state.translation_y,
        }
    }

    /// `scale * rotate * translate` applied to one armour-model UV.
    #[inline]
    #[must_use]
    pub fn apply(self, uv: [f32; 2]) -> [f32; 2] {
        let translated_v = uv[1] + self.translation_y;
        [
            self.scale * (uv[0] * self.cos - translated_v * self.sin),
            self.scale * (uv[0] * self.sin + translated_v * self.cos),
        ]
    }
}

/// Read the live host millisecond source used by non-GUI entity glint wrappers.
///
/// Explicit-time callers should use [`item_glint_passes`] / [`item_glint_uv`] for deterministic
/// tests.  The renderer has no shared `Minecraft.getSystemTime` object yet, so this is the same
/// narrow clock boundary used by the existing GUI seam.
#[must_use]
pub fn item_glint_system_time_ms() -> u64 {
    // One relaxed load; `u64::MAX` = no override.
    let pinned = GLINT_CLOCK.load(std::sync::atomic::Ordering::Relaxed);
    if pinned != u64::MAX {
        return pinned;
    }
    item_glint_wall_clock_ms()
}

/// The wall clock [`item_glint_system_time_ms`] reads when nothing is pinned.
#[must_use]
pub fn item_glint_wall_clock_ms() -> u64 {
    // A benchmark's pinned animation clock stands in for the wall
    // clock, also while a replay records (one relaxed load; `u64::MAX` = none).
    let animation = ANIMATION_CLOCK.load(std::sync::atomic::Ordering::Relaxed);
    if animation != u64::MAX {
        return animation;
    }
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u128::from(u64::MAX - 1)) as u64
}

/// The pinned glint clock (`u64::MAX` = none). See [`pin_item_glint_clock`].
static GLINT_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Pin the enchantment-glint clock to `ms` (`None`
/// unpins it). While a replay records, the client pins it once per frame, so every glint drawn
/// in that frame (hand, hotbar, entities, item frames) uses one recorded value; while a replay
/// plays back or exports, it pins the recorded value, so a replayed frame's glint matches the
/// recorded one exactly. Unpinned (the default) is vanilla's live `getSystemTime()`.
pub fn pin_item_glint_clock(ms: Option<u64>) {
    GLINT_CLOCK.store(
        ms.map_or(u64::MAX, |ms| ms.min(u64::MAX - 1)),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// The pinned benchmark animation clock (`u64::MAX` = the live
/// clocks). See [`pin_animation_clock`].
static ANIMATION_CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

/// Pin every wall-clock animation to `ms` (`None`, the default, keeps
/// the live clocks). Only benchmark harnesses call this: `skywars_bench` pins its lockstep time
/// on every frame, so two runs of a lockstep scene draw identical frames. Nothing a player runs
/// calls it. While pinned it stands in for the wall clock of the enchantment glint (also while a
/// replay records), the HUD's millisecond clocks (heart blink, tab list hearts, achievement
/// toast, mod HUD fades), the world border's forcefield scroll and the end portal's scroll. A
/// glint clock pinned by [`pin_item_glint_clock`] (replay playback) still wins for the glint.
pub fn pin_animation_clock(ms: Option<u64>) {
    ANIMATION_CLOCK.store(
        ms.map_or(u64::MAX, |ms| ms.min(u64::MAX - 1)),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// `live_ms` unless a benchmark pinned the animation clock
/// ([`pin_animation_clock`]), then the pinned value. One relaxed load.
#[inline]
#[must_use]
pub fn animation_clock_ms(live_ms: u64) -> u64 {
    let pinned = ANIMATION_CLOCK.load(std::sync::atomic::Ordering::Relaxed);
    if pinned != u64::MAX {
        pinned
    } else {
        live_ms
    }
}

/// The seed `RenderEntityItem` resets its RNG to at the top of every `doRender`.
///
/// java: `this.field_177079_e.setSeed(187L)`
pub const ITEM_JITTER_SEED: i64 = 187;

/// Vertical gap between the slices of a flat item stack, in blocks.
///
/// java: `translate(0, 0, 0.046875 * f5)` after each slice.
pub const FLAT_STACK_SPACING: f32 = 0.046_875;

/// How many copies of the model a stack of this size is drawn with.
///
/// java: `RenderEntityItem.func_177078_a` — a step function, `1 / 2 / 3 / 4 / 5` at
/// `1 / >1 / >16 / >32 / >48`.
#[must_use]
pub fn render_count(stack_size: i32) -> u32 {
    if stack_size > 48 {
        5
    } else if stack_size > 32 {
        4
    } else if stack_size > 16 {
        3
    } else if stack_size > 1 {
        2
    } else {
        1
    }
}

/// The hover offset, in blocks.
///
/// java: `MathHelper.sin((age + partialTicks) / 10 + hoverStart) * 0.1 + 0.1`. `hover_start` is
/// the entity's own random phase (`EntityItem.hoverStart`), which is what stops a field of
/// dropped items bobbing in lockstep.
#[must_use]
pub fn hover_offset(age: i32, partial_ticks: f32, hover_start: f32) -> f32 {
    mc_math::sin((age as f32 + partial_ticks) / 10.0 + hover_start) * 0.1 + 0.1
}

/// The spin, in degrees.
///
/// java: `((age + partialTicks) / 20 + hoverStart) * (180 / PI)` — note this multiplies by the
/// radians-to-degrees factor without ever having been in radians, so the item completes a turn
/// roughly every 2 seconds. It is a vanilla quirk, not a bug to fix.
#[must_use]
pub fn spin_degrees(age: i32, partial_ticks: f32, hover_start: f32) -> f32 {
    ((age as f32 + partial_ticks) / 20.0 + hover_start) * (180.0 / core::f32::consts::PI)
}

/// One copy of the model in a dropped stack.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ItemCopy {
    /// The matrix this copy is drawn under, relative to the entity transform.
    pub matrix: Mat4,
    /// Index within the stack, `0` being the base copy.
    pub index: u32,
}

/// Everything needed to place a dropped item's copies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DroppedItem {
    /// Camera-relative position of the entity.
    pub rel: Vec3,
    /// java: `EntityItem.getAge`
    pub age: i32,
    /// java: `EntityItem.hoverStart`
    pub hover_start: f32,
    /// java: `ItemStack.stackSize`
    pub stack_size: i32,
    /// java: `IBakedModel.isGui3d` — true for blocks, false for flat item icons.
    pub gui_3d: bool,
    /// java: `ItemCameraTransforms.ground.scale` — the `display.ground` scale of the model.
    pub ground_scale: [f32; 3],
}

impl Default for DroppedItem {
    fn default() -> Self {
        Self {
            rel: Vec3::ZERO,
            age: 0,
            hover_start: 0.0,
            stack_size: 1,
            gui_3d: true,
            ground_scale: [1.0; 3],
        }
    }
}

impl DroppedItem {
    /// The matrix shared by every copy, before the per-copy jitter or slice offset.
    ///
    /// java: `func_177077_a` — translate to `(x, y + hover + 0.25 * groundScaleY, z)`, then
    /// spin about Y, then (for the flat case only) shift the whole stack back by half its
    /// depth so it stays centred.
    #[must_use]
    pub fn base_matrix(&self, partial_ticks: f32) -> Mat4 {
        let bob = hover_offset(self.age, partial_ticks, self.hover_start);
        let mut m = Mat4::from_translation(Vec3::new(
            self.rel.x,
            self.rel.y + bob + 0.25 * self.ground_scale[1],
            self.rel.z,
        ));
        // java: the spin is applied when the model is 3d *or* `renderManager.options != null`,
        // which is always true in a real client, so a flat icon spins too.
        m *= Mat4::from_rotation_y(
            spin_degrees(self.age, partial_ticks, self.hover_start).to_radians(),
        );
        if !self.gui_3d {
            let n = self.render_count() as f32;
            // java: f6 and f4 are `-0.0F * ...`, i.e. exactly zero; only Z shifts.
            m *= Mat4::from_translation(Vec3::new(0.0, 0.0, -FLAT_STACK_SPACING * (n - 1.0) * 0.5));
        }
        m
    }

    /// How many copies this stack draws.
    #[must_use]
    pub fn render_count(&self) -> u32 {
        render_count(self.stack_size)
    }

    /// Every copy's matrix, in draw order.
    ///
    /// A 3D model gets `0.5` scale plus a jitter from the 187-seeded RNG on copies after the
    /// first; a flat model gets no jitter and steps `0.046875 * groundScaleZ` along Z per
    /// slice. java: `RenderEntityItem.doRender`'s loop, both branches.
    #[must_use]
    pub fn copies(&self, partial_ticks: f32) -> Vec<ItemCopy> {
        let base = self.base_matrix(partial_ticks);
        let n = self.render_count();
        let mut rng = JavaRandom::new(ITEM_JITTER_SEED);
        let mut out = Vec::with_capacity(n as usize);
        let mut flat_z = 0.0f32;
        for index in 0..n {
            let m = if self.gui_3d {
                let jitter = if index > 0 {
                    // java: three nextFloat calls per copy, in x, y, z order.
                    let f = (rng.next_f32() * 2.0 - 1.0) * 0.15;
                    let f1 = (rng.next_f32() * 2.0 - 1.0) * 0.15;
                    let f2 = (rng.next_f32() * 2.0 - 1.0) * 0.15;
                    Vec3::new(f, f1, f2)
                } else {
                    Vec3::ZERO
                };
                base * Mat4::from_translation(jitter) * Mat4::from_scale(Vec3::splat(0.5))
            } else {
                let m = base * Mat4::from_translation(Vec3::new(0.0, 0.0, flat_z));
                flat_z += FLAT_STACK_SPACING * self.ground_scale[2];
                m
            };
            out.push(ItemCopy { matrix: m, index });
        }
        out
    }
}

// ---------------------------------------------------------------------------------------------
// BakedModel -> EntityVertex
// ---------------------------------------------------------------------------------------------

/// Converts one baked block/item model (S09's [`mc_assets::BakedModel`]) into camera-ready
/// [`EntityVertex`]es.
///
/// java: `RenderItem.renderModel`/`renderQuads` walk `IBakedModel.getFaceQuads(facing)` for
/// every facing plus `getGeneralQuads()` and feed each quad straight to the tessellator with no
/// face culling — a dropped item floats free of any neighbouring block, so every face is always
/// visible, and `face_quads` (culled *in the terrain mesher*, S11, against an opaque same-side
/// neighbour) is drawn unconditionally here.
///
/// `transform` is applied **after** the `(-0.5, -0.5, -0.5)` recentring `RenderItem.renderItem`
/// always does first (java: `GlStateManager.translate(-0.5F, -0.5F, -0.5F)`, `RenderItem.java`)
/// — a baked model's quads live in `0..1` cube space, but an item is drawn centred on its own
/// origin, so every caller gets that recentring for free instead of having to remember it.
/// [`emit_dropped_item`] is the usual caller; `transform` there is the outer entity transform
/// composed with one [`ItemCopy::matrix`].
///
/// UVs are copied through unchanged: they are atlas coordinates (S09 bakes item and block models
/// against the shared [`mc_assets::Atlas`]), so whatever draws the returned vertices must bind
/// that atlas texture, not a per-entity one — unlike every other geometry this section emits.
///
/// # Ground display transform
/// `mc_assets::BakedModel` carries the model JSON's `display` block. For the ground context,
/// ground display applies rotation, translation, and scale in the context-specific matrix.
/// The dropped-item path applies that complete transform here. [`DroppedItem::ground_scale`]
/// remains separate because vanilla uses its Y/Z scale for the bob lift and flat-stack spacing
/// *outside* the model transform.
pub fn emit_baked_model(model: &BakedModel, transform: Mat4, out: &mut Vec<EntityVertex>) {
    emit_baked_model_tinted(model, transform, out, |_| 0xFF_FF_FF);
}

/// [`emit_baked_model`] with vanilla's per-quad item tint hook.
///
/// The callback is invoked only for quads carrying `BakedQuad.tint_index`; its packed RGB result
/// becomes opaque vertex colour. Untagged layers stay white without consulting the callback.
/// java: `RenderItem.renderQuads` -> `Item.getColorFromItemStack(stack, tintIndex)`.
pub fn emit_baked_model_tinted(
    model: &BakedModel,
    transform: Mat4,
    out: &mut Vec<EntityVertex>,
    mut tint: impl FnMut(u8) -> u32,
) {
    let m = transform * Mat4::from_translation(Vec3::new(-0.5, -0.5, -0.5));
    emit_model_quads(model, m, out, |quad| {
        quad.tint_index
            .map_or(EntityVertex::WHITE, |index| rgb_to_color(tint(index)))
    });
}

/// [`emit_baked_model_tinted`] with the renderer's back-face cull state applied on the CPU.
///
/// Most item callers intentionally replay every quad: dropped items are free-standing and the
/// GUI's flat cards are painter ordered. `RenderItemFrame`, however, runs while the world
/// renderer's ordinary `GL_CULL_FACE/GL_BACK` state is active. Supplying the camera view here
/// reproduces that state without changing the shared entity pipeline (which must stay two-sided
/// for living models). `view` is rotation-only, as provided by [`crate::Camera`].
pub fn emit_baked_model_culled_tinted(
    model: &BakedModel,
    transform: Mat4,
    view: Mat4,
    out: &mut Vec<EntityVertex>,
    mut tint: impl FnMut(u8) -> u32,
) {
    let m = transform * Mat4::from_translation(Vec3::new(-0.5, -0.5, -0.5));
    emit_model_quads_with_uv_and_cull(
        model,
        m,
        out,
        |quad| {
            quad.tint_index
                .map_or(EntityVertex::WHITE, |index| rgb_to_color(tint(index)))
        },
        |uv| uv,
        Some(view),
    );
}

/// [`emit_baked_model_culled_tinted`] with an opaque white primary colour.
pub fn emit_baked_model_culled(
    model: &BakedModel,
    transform: Mat4,
    view: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    emit_baked_model_culled_tinted(model, transform, view, out, |_| 0xFF_FF_FF);
}

/// Replay an item's glint passes while retaining the same world back-face cull decision as its
/// ordinary item-frame content. The glint must not resurrect a quad the base pass discarded.
pub fn emit_baked_model_culled_glint(
    model: &BakedModel,
    transform: Mat4,
    view: Mat4,
    system_time_ms: u64,
    out: &mut Vec<EntityVertex>,
) {
    let m = transform * Mat4::from_translation(Vec3::new(-0.5, -0.5, -0.5));
    for pass in 0..2 {
        let glint = ItemGlintUv::new(system_time_ms, pass);
        emit_model_quads_with_uv_and_cull(
            model,
            m,
            out,
            |_| ITEM_GLINT_COLOR,
            |uv| glint.apply(uv),
            Some(view),
        );
    }
}

/// Replays one baked item model under both `RenderItem.renderEffect` texture matrices.
///
/// `transform` follows the same convention as [`emit_baked_model_tinted`], including the
/// renderer's mandatory `(-0.5, -0.5, -0.5)` recentering.  The returned vertices are intended
/// for [`LayerBlend::Glint`](super::layers::LayerBlend::Glint), whose shader supplies the
/// standalone glint texture and disables entity lighting.
pub fn emit_baked_model_glint(
    model: &BakedModel,
    transform: Mat4,
    system_time_ms: u64,
    out: &mut Vec<EntityVertex>,
) {
    let m = transform * Mat4::from_translation(Vec3::new(-0.5, -0.5, -0.5));
    for pass in 0..2 {
        let glint = ItemGlintUv::new(system_time_ms, pass);
        emit_model_quads_with_uv(model, m, out, |_| ITEM_GLINT_COLOR, |uv| glint.apply(uv));
    }
}

/// [`emit_baked_model_glint`] for an item whose base pass is already emitted: appends both
/// `RenderItem.renderEffect` passes of `base` to `out`.
///
/// `base` must hold `copies` consecutive replays of one baked model made by
/// [`emit_baked_model_tinted`] under the very matrices the glint replay would use: a held or worn
/// item is one copy, a dropped stack is [`DroppedItem::render_count`] copies from
/// [`emit_dropped_item_tinted`]. Output order is the re-walk's: per copy, pass 0 then pass 1.
///
/// Only UV and primary colour differ between the base and the glint, so positions and normals
/// are copied rather than re-transformed. The re-walk computed them from the same quads under the
/// same composed matrix with the same operations, so the copy is bit-identical — and it makes the
/// base/glint position identity the `Equal` depth test needs (`@invariant` on the entity vertex
/// shader) structural rather than incidental. The base pass writes each baked UV through
/// unchanged, so the base vertex's UV is exactly the glint's input UV.
pub fn emit_item_glint_from_base(
    base: &[EntityVertex],
    copies: usize,
    system_time_ms: u64,
    out: &mut Vec<EntityVertex>,
) {
    let per_copy = base.len().checked_div(copies).unwrap_or(0);
    if per_copy == 0 {
        return;
    }
    debug_assert_eq!(base.len() % copies, 0, "every copy replays the whole model");
    let passes = [
        ItemGlintUv::new(system_time_ms, 0),
        ItemGlintUv::new(system_time_ms, 1),
    ];
    out.reserve(base.len() * 2);
    for copy in base.chunks_exact(per_copy) {
        for glint in passes {
            out.extend(copy.iter().map(|&vertex| EntityVertex {
                uv: glint.apply(vertex.uv),
                color: ITEM_GLINT_COLOR,
                ..vertex
            }));
        }
    }
}

/// [`emit_item_glint_from_base`] for one copy that sits at the front of `verts` itself — the
/// layer scratch its base pass was just submitted from. Appends both passes of `verts[..base_len]`
/// after it, so the caller submits `verts[base_len..]`.
pub fn append_item_glint(verts: &mut Vec<EntityVertex>, base_len: usize, system_time_ms: u64) {
    for pass in 0..2 {
        let glint = ItemGlintUv::new(system_time_ms, pass);
        let start = verts.len();
        verts.extend_from_within(..base_len);
        for vertex in &mut verts[start..] {
            vertex.uv = glint.apply(vertex.uv);
            vertex.color = ITEM_GLINT_COLOR;
        }
    }
}

/// Draws a block through `BlockModelRenderer.renderModelBrightness`.
///
/// This is deliberately distinct from [`emit_baked_model_tinted`]. `RenderItem` recentres an
/// item around its origin; `renderModelBrightness` leaves the baked `0..1` block coordinates
/// alone and begins with `rotate(90, 0, 1, 0)`. Its supplied brightness reaches **every** quad,
/// while `Block.getRenderColor` reaches only the baked quads carrying a tint index.
///
/// java: `BlockModelRenderer.renderModelBrightness` / `renderModelBrightnessColorQuads`.
/// `transform` is the caller's matrix immediately before the dispatcher call, so the quarter
/// turn belongs here once for every entity-side block route (held blocks, TNT, minecart cargo,
/// and the decorative block layers), never in an individual layer pose.
pub fn emit_block_model_brightness(
    model: &BakedModel,
    transform: Mat4,
    brightness: f32,
    render_color: u32,
    out: &mut Vec<EntityVertex>,
) {
    let m = transform * Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let white = [brightness, brightness, brightness, 1.0];
    let tinted = {
        let [r, g, b, _] = rgb_to_color(render_color);
        [r * brightness, g * brightness, b * brightness, 1.0]
    };
    emit_model_quads(model, m, out, |quad| {
        if quad.tint_index.is_some() {
            tinted
        } else {
            white
        }
    });
}

fn rgb_to_color(rgb: u32) -> [f32; 4] {
    [
        ((rgb >> 16) & 255) as f32 / 255.0,
        ((rgb >> 8) & 255) as f32 / 255.0,
        (rgb & 255) as f32 / 255.0,
        1.0,
    ]
}

/// Converts all baked quads under an already-composed model-space matrix.
///
/// The two public entry points differ only in that matrix convention and in how they choose a
/// primary colour. Keeping the fan, normal, and face/general traversal here prevents an entity
/// block draw from silently diverging from an entity item draw.
fn emit_model_quads(
    model: &BakedModel,
    m: Mat4,
    out: &mut Vec<EntityVertex>,
    color_for: impl FnMut(&BakedQuad) -> [f32; 4],
) {
    emit_model_quads_with_uv(model, m, out, color_for, |uv| uv);
}

/// Converts all baked quads while also applying a caller-supplied texture-coordinate transform.
///
/// The normal/position path is shared with ordinary item geometry.  Only the UV and primary
/// colour differ for glint, which keeps the depth-equal replay tied to exactly the same model
/// quads rather than rebuilding an approximate shell.
fn emit_model_quads_with_uv(
    model: &BakedModel,
    m: Mat4,
    out: &mut Vec<EntityVertex>,
    color_for: impl FnMut(&BakedQuad) -> [f32; 4],
    uv_for: impl FnMut([f32; 2]) -> [f32; 2],
) {
    emit_model_quads_with_uv_and_cull(model, m, out, color_for, uv_for, None);
}

/// The shared baked-quad replay, with an optional camera-space back-face test.
fn emit_model_quads_with_uv_and_cull(
    model: &BakedModel,
    m: Mat4,
    out: &mut Vec<EntityVertex>,
    mut color_for: impl FnMut(&BakedQuad) -> [f32; 4],
    mut uv_for: impl FnMut([f32; 2]) -> [f32; 2],
    cull_view: Option<Mat4>,
) {
    let cull_matrix = cull_view.map(|view| view * m);
    let mut push = |quad: &BakedQuad| {
        if let Some(cull_matrix) = cull_matrix {
            let points: [Vec3; 4] = core::array::from_fn(|i| {
                let v = quad.verts[i].pos;
                cull_matrix.transform_point3(Vec3::new(v[0], v[1], v[2]))
            });
            if !quad_is_front_facing(points) {
                return;
            }
        }
        let n = m.transform_vector3(quad_normal(quad)).normalize_or_zero();
        let p: [Vec3; 4] = core::array::from_fn(|i| {
            let v = quad.verts[i].pos;
            m.transform_point3(Vec3::new(v[0], v[1], v[2]))
        });
        let color = color_for(quad);
        // GL_QUADS -> two triangles, the same 0-1-2 / 0-2-3 fan `model::Model::emit_part` uses.
        for &i in &[0usize, 1, 2, 0, 2, 3] {
            out.push(EntityVertex {
                pos: p[i].into(),
                uv: uv_for(quad.verts[i].uv),
                normal: n.into(),
                color,
            });
        }
    };
    for face in &model.face_quads {
        for q in face {
            push(q);
        }
    }
    for q in &model.general_quads {
        push(q);
    }
}

/// Java's default `GL_CCW`/`GL_BACK` test expressed in camera space (eye at the origin). GL
/// takes a polygon's facing from the winding of its *projected* shape; under a perspective
/// projection that is the side of the polygon's plane the eye is on, `normal · (eye - p) > 0`.
/// The world matrices use the same right-handed convention as OpenGL.
///
/// Testing the camera-space `normal.z` alone is the orthographic version of this: it agrees
/// only on the view axis, and off it drops faces perspective shows (a frame's side faces near
/// the edge of the screen) or keeps ones it hides.
fn quad_is_front_facing(points: [Vec3; 4]) -> bool {
    let normal = (points[1] - points[0]).cross(points[2] - points[0]);
    let length_squared = normal.length_squared();
    length_squared.is_finite() && length_squared > f32::EPSILON && normal.dot(points[0]) < 0.0
}

/// Filter an already-emitted triangle list with the world renderer's back-face state.
///
/// This is used for builtin tile-item geometry in an item frame, where the model is emitted by
/// [`super::tilepass::TileModels`] rather than by the baked-item helper above. Positions are
/// already camera-relative; `view` supplies the camera's rotation-only view matrix. The function
/// preserves triangle order and drops no partial trailing vertex, making it safe at a submission
/// boundary even when a future model emitter writes a non-multiple-of-three tail.
pub fn retain_front_facing_triangles(view: Mat4, verts: &mut Vec<EntityVertex>) {
    let complete_len = verts.len() - verts.len() % 3;
    let mut write = 0;
    for start in (0..complete_len).step_by(3) {
        let triangle = [verts[start], verts[start + 1], verts[start + 2]];
        let points: [Vec3; 3] =
            core::array::from_fn(|index| view.transform_point3(Vec3::from(triangle[index].pos)));
        // The perspective facing test of `quad_is_front_facing`: the eye (the origin) must be on
        // the side of the triangle's plane its winding normal points to.
        let normal = (points[1] - points[0]).cross(points[2] - points[0]);
        let length_squared = normal.length_squared();
        if length_squared.is_finite()
            && length_squared > f32::EPSILON
            && normal.dot(points[0]) < 0.0
        {
            for vertex in triangle {
                verts[write] = vertex;
                write += 1;
            }
        }
    }
    let tail = verts[complete_len..].to_vec();
    for vertex in tail {
        verts[write] = vertex;
        write += 1;
    }
    verts.truncate(write);
}

/// The outward normal for one baked quad.
///
/// Uses the quad's own baked [`BakedQuad::face`] when it has one — every axis-aligned face does
/// — and falls back to the geometric cross product (the same `b x a` vanilla's `TexturedQuad`
/// and this crate's [`super::model::TexturedQuad::normal`] use) for the rare oblique quad
/// `FaceBakery` could not assign an axis to.
fn quad_normal(quad: &BakedQuad) -> Vec3 {
    if let Some(facing) = quad.face {
        let (dx, dy, dz) = facing.offset();
        return Vec3::new(dx as f32, dy as f32, dz as f32);
    }
    let p = |i: usize| Vec3::from(quad.verts[i].pos);
    let a = p(0) - p(1);
    let b = p(2) - p(1);
    b.cross(a).normalize_or_zero()
}

/// Emits every copy of a dropped-item stack, baked model and placement math combined.
///
/// `entity_base` is the outer entity transform (camera-relative — see [`crate::entity::render::
/// entity_matrix`], though a dropped item has no yaw/scale of its own, so callers typically pass
/// just the camera-relative translation). Each of [`DroppedItem::copies`]'s per-copy matrices is
/// composed onto it and handed to [`emit_baked_model`], so the jitter/bob/spin/flat-stack
/// placement and the geometry conversion agree on one coordinate frame without the caller having
/// to interleave them by hand.
///
/// # The two halves `doRender` splits between
///
/// [`DroppedItem::copies`] is `RenderEntityItem.doRender`'s own loop and stops where vanilla
/// hands the stack to `RenderItem`. The rest of the pipeline lives here, because it belongs to
/// *every* item draw rather than to the dropped-item placement:
///
/// * `ibakedmodel.getItemCameraTransforms().applyTransform(GROUND)` (`RenderEntityItem.java:126`
///   and `:133`) — the model's own `display.ground` rotation/translation/scale, applied through
///   [`super::objects::item_transform_matrix`]. Without it a dropped sword lies flat instead of
///   standing at vanilla's angle.
/// * `RenderItem.renderItem(stack, model)`'s `GlStateManager.scale(0.5F, 0.5F, 0.5F)`
///   (`RenderItem.java:145`), which is a **second** halving on top of the gui-3d branch's own
///   `scale(0.5)` at `RenderEntityItem.java:125`. A dropped block is a *quarter* of a block
///   across in vanilla; emitting only one of the two scales drew it at half a block — twice the
///   right size, and the audit's "present and correct" verdict on this file missed it.
pub fn emit_dropped_item(
    item: &DroppedItem,
    model: &BakedModel,
    entity_base: Mat4,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
) {
    emit_dropped_item_tinted(item, model, entity_base, partial_ticks, out, |_| 0xFF_FF_FF);
}

/// [`emit_dropped_item`] with one item tint callback shared by every emitted stack copy.
pub fn emit_dropped_item_tinted(
    item: &DroppedItem,
    model: &BakedModel,
    entity_base: Mat4,
    partial_ticks: f32,
    out: &mut Vec<EntityVertex>,
    mut tint: impl FnMut(u8) -> u32,
) {
    let inner = super::objects::item_transform_matrix(&model.transforms.ground)
        * Mat4::from_scale(Vec3::splat(super::objects::RENDER_ITEM_SCALE));
    for copy in item.copies(partial_ticks) {
        emit_baked_model_tinted(model, entity_base * copy.matrix * inner, out, &mut tint);
    }
}

/// Emits every copy of an enchanted dropped-item stack under the two `RenderItem.renderEffect`
/// passes.  The placement and model recentering are byte-for-byte the ordinary dropped path;
/// only the UV matrix and glint primary colour differ.
pub fn emit_dropped_item_glint(
    item: &DroppedItem,
    model: &BakedModel,
    entity_base: Mat4,
    partial_ticks: f32,
    system_time_ms: u64,
    out: &mut Vec<EntityVertex>,
) {
    let inner = super::objects::item_transform_matrix(&model.transforms.ground)
        * Mat4::from_scale(Vec3::splat(super::objects::RENDER_ITEM_SCALE));
    for copy in item.copies(partial_ticks) {
        emit_baked_model_glint(
            model,
            entity_base * copy.matrix * inner,
            system_time_ms,
            out,
        );
    }
}

/// Replays already-emitted armour geometry under both `LayerArmorBase.renderGlint` passes.
///
/// Armour's model pose and visibility mask are established by [`super::layers::emit_armor`];
/// this function deliberately copies only positions/normals and replaces each vertex's UV and
/// primary colour, matching Java's separate model replay under `ENCHANTED_ITEM_GLINT_RES`.
pub fn emit_armor_glint(source: &[EntityVertex], age_in_ticks: f32, out: &mut Vec<EntityVertex>) {
    for pass in 0..2 {
        let glint = ArmorGlintUv::new(age_in_ticks, pass);
        out.extend(source.iter().copied().map(|mut vertex| {
            vertex.uv = glint.apply(vertex.uv);
            vertex.color = ARMOR_GLINT_COLOR;
            vertex
        }));
    }
}

/// [`emit_armor_glint`] for a slot whose posed base sits at the front of `verts` (the layer
/// scratch it was just submitted from): appends both passes of `verts[..base_len]` after it,
/// without a per-slot allocation, so the caller submits `verts[base_len..]`.
pub fn append_armor_glint(verts: &mut Vec<EntityVertex>, base_len: usize, age_in_ticks: f32) {
    for pass in 0..2 {
        let glint = ArmorGlintUv::new(age_in_ticks, pass);
        let start = verts.len();
        verts.extend_from_within(..base_len);
        for vertex in &mut verts[start..] {
            vertex.uv = glint.apply(vertex.uv);
            vertex.color = ARMOR_GLINT_COLOR;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_assets::Vertex as BakedVertex;
    use mc_math::Facing;

    /// A synthetic two-quad "model": the top (`+Y`) and bottom (`-Y`) faces of a unit cube in
    /// `0..1` space, matching how S09 bakes a full block. Enough to exercise the conversion
    /// without needing a real resource pack (item.rs stays inside `entity/**`, so it cannot pull
    /// in `tests/s12_entity.rs`'s vanilla-jar fixtures).
    fn unit_quad(y: f32, face: Facing) -> BakedQuad {
        let v = |x: f32, z: f32, u: f32, w: f32| BakedVertex {
            pos: [x, y, z],
            uv: [u, w],
        };
        BakedQuad {
            verts: [
                v(0.0, 0.0, 0.0, 0.0),
                v(1.0, 0.0, 1.0, 0.0),
                v(1.0, 1.0, 1.0, 1.0),
                v(0.0, 1.0, 0.0, 1.0),
            ],
            face: Some(face),
            shade_face: Some(face),
            tint_index: None,
            shade: false,
        }
    }

    fn two_face_model() -> BakedModel {
        let mut m = BakedModel::default();
        m.general_quads.push(unit_quad(1.0, Facing::Up));
        m.general_quads.push(unit_quad(0.0, Facing::Down));
        m
    }

    #[test]
    fn each_quad_becomes_two_triangles_with_the_uv_carried_through() {
        let model = two_face_model();
        let mut out = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut out);
        assert_eq!(
            out.len(),
            2 * 6,
            "2 quads, 6 verts each (a 0-1-2/0-2-3 fan)"
        );
        // Every UV the fixture wrote survives unchanged (atlas coordinates pass straight through).
        for v in &out {
            assert!((0.0..=1.0).contains(&v.uv[0]) && (0.0..=1.0).contains(&v.uv[1]));
        }
    }

    #[test]
    fn tinted_emitter_colors_only_the_quad_with_a_tint_index() {
        let mut model = two_face_model();
        model.general_quads[0].tint_index = Some(0);
        let mut out = Vec::new();
        let mut calls = Vec::new();
        emit_baked_model_tinted(&model, Mat4::IDENTITY, &mut out, |index| {
            calls.push(index);
            0x7C_AF_C6
        });

        assert_eq!(calls, [0], "untagged bottle layers do not ask for tint");
        let liquid = [124.0 / 255.0, 175.0 / 255.0, 198.0 / 255.0, 1.0];
        assert!(out[..6].iter().all(|vertex| vertex.color == liquid));
        assert!(out[6..]
            .iter()
            .all(|vertex| vertex.color == EntityVertex::WHITE));

        let mut legacy = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut legacy);
        assert!(legacy
            .iter()
            .all(|vertex| vertex.color == EntityVertex::WHITE));
    }

    #[test]
    fn item_glint_uses_vanillas_two_periods_and_opposite_scroll() {
        let passes = item_glint_passes(1_500);
        assert_eq!(passes[0].period_ms, 3_000);
        assert_eq!(passes[1].period_ms, 4_873);
        assert_eq!(passes[0].rotation_degrees, -50.0);
        assert_eq!(passes[1].rotation_degrees, 10.0);
        assert!((passes[0].translation - 0.0625).abs() < 1e-7);
        assert!((passes[1].translation + 1_500.0 / 4_873.0 / 8.0).abs() < 1e-7);

        // The first pass wraps exactly at 3000 ms; a static duplicate or one shared period cannot
        // satisfy this control while preserving the second pass's independent cadence.
        assert_eq!(
            item_glint_uv([0.0, 0.0], 3_000, 0),
            item_glint_uv([0.0, 0.0], 0, 0)
        );
        assert_ne!(
            item_glint_uv([0.25, 0.75], 1_500, 0),
            item_glint_uv([0.25, 0.75], 1_500, 1)
        );
    }

    #[test]
    fn armor_glint_uses_layerarmorbase_order_and_tick_scroll() {
        let passes = armor_glint_passes(100.0);
        assert_eq!(passes[0].scale, 1.0 / 3.0);
        assert_eq!(passes[1].scale, 1.0 / 3.0);
        assert_eq!(passes[0].rotation_degrees, 30.0);
        assert_eq!(passes[1].rotation_degrees, -30.0);
        assert!((passes[0].translation_y - 2.0).abs() < 1e-6);
        assert!((passes[1].translation_y - 8.0).abs() < 1e-6);

        let uv = [0.25, 0.75];
        let first = armor_glint_uv(uv, 100.0, 0);
        let second = armor_glint_uv(uv, 100.0, 1);
        assert_ne!(first, second);
        // The Y translation is inside the rotation (`scale * rotate * translate`), not a post-
        // rotation offset.  This is a biting control against reusing RenderItem's matrix order.
        let angle = 30.0_f32.to_radians();
        let expected = [
            (uv[0] * mc_math::cos(angle) - (uv[1] + 2.0) * mc_math::sin(angle)) / 3.0,
            (uv[0] * mc_math::sin(angle) + (uv[1] + 2.0) * mc_math::cos(angle)) / 3.0,
        ];
        assert!(
            (first[0] - expected[0]).abs() < 1e-6,
            "{first:?} vs {expected:?}"
        );
        assert!(
            (first[1] - expected[1]).abs() < 1e-6,
            "{first:?} vs {expected:?}"
        );
    }

    #[test]
    fn item_glint_replays_dropped_geometry_with_opaque_primary_colour() {
        let model = two_face_model();
        let item = DroppedItem {
            stack_size: 2,
            gui_3d: false,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item_glint(&item, &model, Mat4::IDENTITY, 0.0, 1_500, &mut out);
        assert_eq!(out.len(), item.render_count() as usize * 2 * 2 * 6);
        assert!(out.iter().all(|vertex| vertex.color == ITEM_GLINT_COLOR));
        let per_pass = 2 * 6;
        for copy in 0..item.render_count() as usize {
            let start = copy * per_pass * 2;
            assert_ne!(
                out[start..start + per_pass]
                    .iter()
                    .map(|v| v.uv)
                    .collect::<Vec<_>>(),
                out[start + per_pass..start + per_pass * 2]
                    .iter()
                    .map(|v| v.uv)
                    .collect::<Vec<_>>(),
                "the two dropped-item foil passes must not be a static duplicate"
            );
        }
    }

    #[test]
    fn armor_glint_preserves_pose_and_replaces_source_tint() {
        let source = [EntityVertex {
            pos: [1.0, 2.0, 3.0],
            uv: [0.25, 0.75],
            normal: [0.0, 1.0, 0.0],
            color: [0.1, 0.2, 0.3, 0.4],
        }];
        let mut out = Vec::new();
        emit_armor_glint(&source, 20.5, &mut out);
        assert_eq!(out.len(), 2);
        assert!(out.iter().all(|vertex| vertex.pos == source[0].pos));
        assert!(out.iter().all(|vertex| vertex.normal == source[0].normal));
        assert!(out.iter().all(|vertex| vertex.color == ARMOR_GLINT_COLOR));
        assert_ne!(out[0].uv, out[1].uv);
        assert_ne!(out[0].color, source[0].color);
    }

    fn bits(verts: &[EntityVertex]) -> &[u32] {
        bytemuck::cast_slice(verts)
    }

    /// The pre-hoist per-vertex formulas, verbatim: the oracle for the hoisted pass state.
    fn item_glint_uv_per_vertex(uv: [f32; 2], system_time_ms: u64, pass: usize) -> [f32; 2] {
        let state = item_glint_passes(system_time_ms)[pass];
        let angle = state.rotation_degrees.to_radians();
        let sin = mc_math::sin(angle);
        let cos = mc_math::cos(angle);
        let rotated_u = uv[0] * cos - uv[1] * sin;
        let rotated_v = uv[0] * sin + uv[1] * cos;
        [
            ITEM_GLINT_TEXTURE_SCALE * (rotated_u + state.translation),
            ITEM_GLINT_TEXTURE_SCALE * rotated_v,
        ]
    }

    fn armor_glint_uv_per_vertex(uv: [f32; 2], age_in_ticks: f32, pass: usize) -> [f32; 2] {
        let state = armor_glint_passes(age_in_ticks)[pass];
        let angle = state.rotation_degrees.to_radians();
        let sin = mc_math::sin(angle);
        let cos = mc_math::cos(angle);
        let translated_v = uv[1] + state.translation_y;
        [
            state.scale * (uv[0] * cos - translated_v * sin),
            state.scale * (uv[0] * sin + translated_v * cos),
        ]
    }

    #[test]
    fn hoisted_glint_pass_state_is_bit_identical_to_the_per_vertex_formula() {
        let uvs = [
            [0.0, 0.0],
            [0.25, 0.75],
            [0.999_9, 0.000_1],
            [-3.5, 17.25],
            [0.123_456_7, 0.765_432_1],
        ];
        for time in [
            0_u64,
            1,
            1_500,
            2_999,
            3_000,
            4_873,
            1_730_000_123_456,
            u64::MAX,
        ] {
            for pass in 0..2 {
                let hoisted = ItemGlintUv::new(time, pass);
                for uv in uvs {
                    let want = item_glint_uv_per_vertex(uv, time, pass);
                    assert_eq!(hoisted.apply(uv).map(f32::to_bits), want.map(f32::to_bits));
                    assert_eq!(
                        item_glint_uv(uv, time, pass).map(f32::to_bits),
                        want.map(f32::to_bits)
                    );
                }
            }
        }
        for age in [0.0_f32, 0.5, 20.5, 100.0, 12_345.678, 1.0e7] {
            for pass in 0..2 {
                let hoisted = ArmorGlintUv::new(age, pass);
                for uv in uvs {
                    let want = armor_glint_uv_per_vertex(uv, age, pass);
                    assert_eq!(hoisted.apply(uv).map(f32::to_bits), want.map(f32::to_bits));
                    assert_eq!(
                        armor_glint_uv(uv, age, pass).map(f32::to_bits),
                        want.map(f32::to_bits)
                    );
                }
            }
        }
    }

    /// An "enchanted sword" stand-in: quads in both the face and general lists, one tinted
    /// layer, distinct UVs per corner, and an oblique quad with no baked face (cross normal).
    fn sword_like_model() -> BakedModel {
        let mut m = BakedModel::default();
        let cards = [
            (0.468_75_f32, Facing::South),
            (0.531_25, Facing::North),
            (0.1, Facing::Up),
            (0.9, Facing::Down),
        ];
        for (i, (y, face)) in cards.into_iter().enumerate() {
            let mut quad = unit_quad(y, face);
            quad.tint_index = (i == 1).then_some(0);
            for (k, vertex) in quad.verts.iter_mut().enumerate() {
                vertex.uv = [
                    0.312_5 + 0.023 * k as f32,
                    0.5 - 0.0625 * i as f32 + 0.011 * k as f32,
                ];
            }
            if i < 2 {
                m.face_quads[face.index() as usize].push(quad);
            } else {
                m.general_quads.push(quad);
            }
        }
        let mut oblique = unit_quad(0.25, Facing::Up);
        oblique.face = None;
        oblique.verts[2].pos[1] = 0.75;
        m.general_quads.push(oblique);
        m
    }

    /// #22 / R2-5: glint derived from the base pass reproduces the full re-walk bit for bit — a
    /// held item (one copy, in the layer scratch and from a slice) and both dropped-stack
    /// branches (five jittered 3D copies, five flat slices).
    #[test]
    fn glint_from_base_matches_the_rewalk_bit_for_bit() {
        let model = sword_like_model();
        let held = Mat4::from_translation(Vec3::new(-0.4, 1.3, 2.2))
            * Mat4::from_rotation_x(-1.1)
            * Mat4::from_rotation_y(0.3)
            * Mat4::from_scale(Vec3::new(0.375, -0.375, -0.375));
        let tint = |index: u8| 0x3F_7F_BF ^ u32::from(index);
        for time in [0_u64, 1_500, 1_730_000_123_456] {
            let mut rewalk = Vec::new();
            emit_baked_model_glint(&model, held, time, &mut rewalk);
            assert_eq!(rewalk.len(), 2 * 5 * 6);

            let mut scratch = Vec::new();
            emit_baked_model_tinted(&model, held, &mut scratch, tint);
            let base_len = scratch.len();
            append_item_glint(&mut scratch, base_len, time);
            assert_eq!(
                bits(&scratch[base_len..]),
                bits(&rewalk),
                "in place at {time}"
            );
            let mut from_slice = Vec::new();
            emit_item_glint_from_base(&scratch[..base_len], 1, time, &mut from_slice);
            assert_eq!(bits(&from_slice), bits(&rewalk), "from slice at {time}");

            for gui_3d in [true, false] {
                let item = DroppedItem {
                    rel: Vec3::new(1.5, -0.25, 3.0),
                    age: 37,
                    hover_start: 1.25,
                    stack_size: 64,
                    gui_3d,
                    ground_scale: [0.5, 0.5, 0.5],
                };
                assert_eq!(item.render_count(), 5);
                let mut base = Vec::new();
                emit_dropped_item_tinted(&item, &model, held, 0.375, &mut base, tint);
                let mut derived = Vec::new();
                emit_item_glint_from_base(&base, item.render_count() as usize, time, &mut derived);
                let mut rewalk = Vec::new();
                emit_dropped_item_glint(&item, &model, held, 0.375, time, &mut rewalk);
                assert_eq!(bits(&derived), bits(&rewalk), "gui_3d {gui_3d} at {time}");
            }
        }
    }

    #[test]
    fn appended_armor_glint_matches_the_copying_replay_bit_for_bit() {
        let source: Vec<EntityVertex> = (0..36)
            .map(|i| EntityVertex {
                pos: [i as f32 * 0.1, -0.7, 1.0 / (i as f32 + 3.0)],
                uv: [0.015_625 * i as f32, 1.0 - 0.031_25 * i as f32],
                normal: [0.0, 0.6, 0.8],
                color: [0.4, 0.5, 0.6, 1.0],
            })
            .collect();
        for age in [0.0_f32, 20.5, 98_765.43] {
            let mut want = Vec::new();
            emit_armor_glint(&source, age, &mut want);
            let mut scratch = source.clone();
            append_armor_glint(&mut scratch, source.len(), age);
            assert_eq!(
                bits(&scratch[..source.len()]),
                bits(&source),
                "base untouched"
            );
            assert_eq!(bits(&scratch[source.len()..]), bits(&want), "age {age}");
        }
    }

    #[test]
    fn block_brightness_rotates_the_baked_block_without_item_recentring() {
        let mut model = BakedModel::default();
        // An east-facing directional face makes a missing dispatcher rotation observable. The
        // item path would also re-centre it, so check both the normal and the raw 0..1 bounds.
        model.general_quads.push(unit_quad(0.25, Facing::East));
        let mut out = Vec::new();
        emit_block_model_brightness(&model, Mat4::IDENTITY, 1.0, 0xFF_FF_FF, &mut out);

        assert!(
            out.iter()
                .all(|vertex| (vertex.normal[2] + 1.0).abs() < 1e-6),
            "{:?}",
            out[0].normal
        );
        assert!(out
            .iter()
            .all(|vertex| (-1e-6..=1.0 + 1e-6).contains(&vertex.pos[0])));
        assert!(out
            .iter()
            .all(|vertex| (-1.0 - 1e-6..=1e-6).contains(&vertex.pos[2])));
        assert!(
            out.iter().any(|vertex| (vertex.pos[2] + 1.0).abs() < 1e-6),
            "the rotated block must not be recentred"
        );
    }

    #[test]
    fn block_brightness_applies_light_to_every_quad_and_render_color_only_to_tinted_ones() {
        let mut model = two_face_model();
        model.general_quads[0].tint_index = Some(7);
        let mut out = Vec::new();
        emit_block_model_brightness(&model, Mat4::IDENTITY, 0.4, 0x80_C0_20, &mut out);

        let tinted = [
            128.0 / 255.0 * 0.4,
            192.0 / 255.0 * 0.4,
            32.0 / 255.0 * 0.4,
            1.0,
        ];
        assert!(out[..6].iter().all(|vertex| vertex.color == tinted));
        assert!(out[6..]
            .iter()
            .all(|vertex| vertex.color == [0.4, 0.4, 0.4, 1.0]));

        // Mutation control: multiplying the packed block colour into every quad turns the plain
        // face green; omitting brightness leaves it white. Both outcomes must stay impossible.
        assert_ne!(out[6].color, tinted);
        assert_ne!(
            out[0].color,
            [128.0 / 255.0, 192.0 / 255.0, 32.0 / 255.0, 1.0]
        );
    }

    #[test]
    fn the_identity_transform_recentres_the_unit_cube_on_the_origin() {
        // java: `RenderItem.renderItem` always translates (-0.5, -0.5, -0.5) before drawing —
        // a baked model's 0..1 cube must land on -0.5..0.5 once emitted.
        let model = two_face_model();
        let mut out = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut out);
        for v in &out {
            assert!((-0.5..=0.5).contains(&v.pos[0]), "{:?}", v.pos);
            assert!((-0.5..=0.5).contains(&v.pos[1]), "{:?}", v.pos);
            assert!((-0.5..=0.5).contains(&v.pos[2]), "{:?}", v.pos);
        }
        // The top face (y = 1 in model space) recentres to +0.5; the bottom to -0.5.
        assert!(out[..6].iter().all(|v| (v.pos[1] - 0.5).abs() < 1e-6));
        assert!(out[6..].iter().all(|v| (v.pos[1] + 0.5).abs() < 1e-6));
    }

    #[test]
    fn a_caller_supplied_transform_composes_after_the_recentring() {
        let model = two_face_model();
        let mut out = Vec::new();
        emit_baked_model(
            &model,
            Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0)),
            &mut out,
        );
        // The recentred cube still spans a full unit in x (-0.5..0.5 becomes 9.5..10.5); only
        // the translate's *centre* is what the caller controls.
        assert!(out.iter().all(|v| (9.5..=10.5).contains(&v.pos[0])));
    }

    #[test]
    fn face_normals_come_from_the_baked_facing_not_a_recomputation() {
        let model = two_face_model();
        let mut out = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut out);
        assert!(out[..6].iter().all(|v| v.normal == [0.0, 1.0, 0.0]), "up");
        assert!(
            out[6..].iter().all(|v| v.normal == [0.0, -1.0, 0.0]),
            "down"
        );
    }

    #[test]
    fn an_oblique_quad_with_no_baked_facing_falls_back_to_the_geometric_normal() {
        let v = |x: f32, y: f32, z: f32| BakedVertex {
            pos: [x, y, z],
            uv: [0.0, 0.0],
        };
        // A plain +Z-facing quad, but with `face: None` so the fallback path is exercised.
        let quad = BakedQuad {
            verts: [
                v(0.0, 0.0, 1.0),
                v(1.0, 0.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(0.0, 1.0, 1.0),
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: false,
        };
        let mut model = BakedModel::default();
        model.general_quads.push(quad);
        let mut out = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut out);
        for v in &out {
            assert!((v.normal[2] - 1.0).abs() < 1e-5, "{:?}", v.normal);
        }
    }

    /// `face_quads` are drawn unconditionally — a floating item has no neighbour to cull
    /// against, unlike the terrain mesher (S11) which is the only other `BakedModel` consumer.
    #[test]
    fn face_quads_are_drawn_without_any_culling() {
        let mut model = BakedModel::default();
        model.face_quads[Facing::North.index() as usize].push(unit_quad(0.5, Facing::North));
        let mut out = Vec::new();
        emit_baked_model(&model, Mat4::IDENTITY, &mut out);
        assert_eq!(out.len(), 6, "the face_quads entry must reach the output");
    }

    #[test]
    fn item_frame_cull_replays_front_winding_and_keeps_glint_in_lockstep() {
        let v = |x: f32, y: f32, z: f32| BakedVertex {
            pos: [x, y, z],
            uv: [x, y],
        };
        let front = BakedQuad {
            verts: [
                v(0.0, 0.0, 1.0),
                v(1.0, 0.0, 1.0),
                v(1.0, 1.0, 1.0),
                v(0.0, 1.0, 1.0),
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: false,
        };
        let back = BakedQuad {
            verts: [
                v(0.0, 0.0, 0.0),
                v(0.0, 1.0, 0.0),
                v(1.0, 1.0, 0.0),
                v(1.0, 0.0, 0.0),
            ],
            ..front.clone()
        };
        let mut model = BakedModel::default();
        model.general_quads.extend([front, back]);
        // Three blocks in front of the eye (camera space looks down -Z), so the +Z quad faces it
        // and the -Z quad faces away under the perspective facing rule.
        let in_front = Mat4::from_translation(Vec3::new(0.0, 0.0, -3.0));

        let mut ordinary = Vec::new();
        emit_baked_model(&model, in_front, &mut ordinary);
        assert_eq!(
            ordinary.len(),
            12,
            "the ordinary item path remains two-sided"
        );

        let mut culled = Vec::new();
        emit_baked_model_culled(&model, in_front, Mat4::IDENTITY, &mut culled);
        assert_eq!(
            culled.len(),
            6,
            "GL_BACK removes the reversed frame-facing quad"
        );

        let mut glint = Vec::new();
        emit_baked_model_culled_glint(&model, in_front, Mat4::IDENTITY, 1500, &mut glint);
        assert_eq!(
            glint.len(),
            12,
            "both glint passes replay only the retained front quad"
        );

        // Negative control: filtering the emitted triangle list must also remove the same back
        // triangle when builtin tile geometry bypasses the baked-model helper.
        retain_front_facing_triangles(Mat4::IDENTITY, &mut ordinary);
        assert_eq!(ordinary.len(), 6);
    }

    /// GL decides facing from the winding of the projected polygon. Off the view axis that is
    /// not the camera-space `normal.z` test these helpers used to apply: a face to the left of
    /// the view that is turned toward the axis (a frame's inner side) is on screen.
    #[test]
    fn cpu_cull_matches_the_projected_winding_off_the_view_axis() {
        // The camera's own projection (`Camera::update`).
        let proj =
            glam::camera::rh::proj::directx::perspective(70.0_f32.to_radians(), 1.0, 0.05, 100.0);
        let projected_front = |p: [Vec3; 3]| {
            let ndc = p.map(|v| {
                let clip = proj * v.extend(1.0);
                glam::Vec2::new(clip.x, clip.y) / clip.w
            });
            (ndc[1] - ndc[0]).perp_dot(ndc[2] - ndc[0]) > 0.0
        };
        let normals = [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
            Vec3::new(0.6, 0.0, 0.8),
            Vec3::new(-0.6, 0.48, 0.64),
        ];
        let (mut checked, mut orthographic_wrong) = (0, 0);
        for x in [-4.0_f32, -1.5, 0.0, 1.5, 4.0] {
            for y in [-2.0_f32, 0.0, 2.0] {
                for z in [-3.0_f32, -7.0] {
                    for n in normals {
                        let centre = Vec3::new(x, y, z);
                        let t = n.any_orthonormal_vector();
                        let b = n.cross(t);
                        let h = 0.25;
                        // Counter-clockwise around `n`: (p1 - p0) x (p2 - p0) is along `n`.
                        let quad = [
                            centre + (-t - b) * h,
                            centre + (t - b) * h,
                            centre + (t + b) * h,
                            centre + (-t + b) * h,
                        ];
                        let gl = projected_front([quad[0], quad[1], quad[2]]);
                        assert_eq!(
                            quad_is_front_facing(quad),
                            gl,
                            "quad at {centre:?} facing {n:?}"
                        );
                        let mut tris: Vec<EntityVertex> = [0, 1, 2, 0, 2, 3]
                            .iter()
                            .map(|&i| EntityVertex {
                                pos: quad[i].into(),
                                uv: [0.0, 0.0],
                                normal: n.into(),
                                color: EntityVertex::WHITE,
                            })
                            .collect();
                        retain_front_facing_triangles(Mat4::IDENTITY, &mut tris);
                        assert_eq!(tris.len(), if gl { 6 } else { 0 }, "{centre:?} {n:?}");
                        let winding = (quad[1] - quad[0]).cross(quad[2] - quad[0]);
                        if (winding.z > 0.0) != gl {
                            orthographic_wrong += 1;
                        }
                        checked += 1;
                    }
                }
            }
        }
        assert_eq!(checked, 5 * 3 * 2 * 8);
        eprintln!("the orthographic rule is wrong on {orthographic_wrong} of {checked} quads");
        // The side faces off the axis: the old `normal.z > 0` rule gets these wrong.
        assert!(orthographic_wrong > 20, "{orthographic_wrong} of {checked}");
    }

    #[test]
    fn dropped_item_emission_matches_the_placement_math_copy_count() {
        let model = two_face_model();
        let item = DroppedItem {
            stack_size: 64, // render_count == 5
            gui_3d: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);
        assert_eq!(out.len(), item.render_count() as usize * 2 * 6);
    }

    #[test]
    fn dropped_item_tint_callback_reaches_each_tinted_layer_of_each_copy() {
        let mut model = two_face_model();
        model.general_quads[0].tint_index = Some(0);
        model.general_quads[1].tint_index = Some(1);
        let item = DroppedItem {
            stack_size: 2, // two copies make a callback accidentally used only once observable.
            gui_3d: false,
            ..Default::default()
        };
        let mut out = Vec::new();
        let mut calls = Vec::new();
        emit_dropped_item_tinted(&item, &model, Mat4::IDENTITY, 0.0, &mut out, |index| {
            calls.push(index);
            match index {
                0 => 0x7C_AF_C6,
                1 => 0x55_55_55,
                _ => unreachable!("fixture has only layers zero and one"),
            }
        });

        assert_eq!(calls, [0, 1, 0, 1]);
        let liquid = [124.0 / 255.0, 175.0 / 255.0, 198.0 / 255.0, 1.0];
        let overlay = [85.0 / 255.0, 85.0 / 255.0, 85.0 / 255.0, 1.0];
        for copy in 0..2 {
            let start = copy * 12;
            assert!(out[start..start + 6]
                .iter()
                .all(|vertex| vertex.color == liquid));
            assert!(out[start + 6..start + 12]
                .iter()
                .all(|vertex| vertex.color == overlay));
        }

        // Biting control: a missing tint index must bypass the caller completely, even though
        // this is the same multi-copy dropped-item route.
        model.general_quads[1].tint_index = None;
        let mut untagged = Vec::new();
        let mut untagged_calls = 0;
        emit_dropped_item_tinted(&item, &model, Mat4::IDENTITY, 0.0, &mut untagged, |_| {
            untagged_calls += 1;
            0x7C_AF_C6
        });
        assert_eq!(untagged_calls, 2, "only layer zero of each copy is tinted");
        assert!(untagged[6..12]
            .iter()
            .all(|vertex| vertex.color == EntityVertex::WHITE));
    }

    /// java: `RenderEntityItem.java:125` `scale(0.5)` **times** `RenderItem.java:145`
    /// `scale(0.5)` — a dropped full block is a quarter of a block across, not a half. This is
    /// the whole point of composing `RenderItem`'s own transform into [`emit_dropped_item`]
    /// rather than stopping at `doRender`'s loop.
    #[test]
    fn a_dropped_block_is_a_quarter_of_a_block_across() {
        let model = two_face_model();
        let item = DroppedItem {
            stack_size: 1, // one copy, no jitter
            gui_3d: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);
        let span = |axis: usize| {
            let lo = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[axis]));
            let hi = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[axis]));
            hi - lo
        };
        // The spin is about Y, so X and Z sweep with it; Y is the axis the rotation cannot
        // change and is therefore the one to measure.
        assert!((span(1) - 0.25).abs() < 1e-6, "{}", span(1));
    }

    /// java: `applyTransform(GROUND)` sits *between* the two halvings, so a model whose
    /// `display.ground` scales is scaled by all three.
    #[test]
    fn the_ground_display_transform_reaches_the_dropped_item() {
        let mut model = two_face_model();
        model.transforms.ground = mc_assets::model::ItemTransform {
            rotation: [0.0; 3],
            translation: [0.0; 3],
            scale: [2.0; 3],
        };
        let item = DroppedItem {
            stack_size: 1,
            gui_3d: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);
        let lo = out.iter().fold(f32::MAX, |a, v| a.min(v.pos[1]));
        let hi = out.iter().fold(f32::MIN, |a, v| a.max(v.pos[1]));
        assert!((hi - lo - 0.5).abs() < 1e-6, "{}", hi - lo);
    }

    #[test]
    fn the_ground_display_rotation_translation_and_scale_use_vanillas_order() {
        let mut model = two_face_model();
        model.transforms.ground = mc_assets::model::ItemTransform {
            rotation: [0.0, 0.0, 90.0],
            translation: [0.25, -0.125, 0.375],
            scale: [2.0, 3.0, 4.0],
        };
        let item = DroppedItem {
            stack_size: 1,
            gui_3d: true,
            ground_scale: model.transforms.ground.scale,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);

        // First vertex of the top quad is (0, 1, 0).  The expected point spells out the
        // Java order: recenter, RenderItem's half-scale, display.ground scale/rotation/
        // translation, then the dropped 3D copy's half-scale and bob/lift.
        let expected = Vec3::new(-0.25, 0.5375, -0.3125);
        let actual = Vec3::from(out[0].pos);
        assert!(
            (actual - expected).length() < 1e-6,
            "{actual:?} vs {expected:?}"
        );
    }

    #[test]
    fn flat_stack_spacing_stays_outside_the_ground_display_transform() {
        let mut model = two_face_model();
        model.transforms.ground = mc_assets::model::ItemTransform {
            rotation: [17.0, 23.0, 31.0],
            translation: [0.25, -0.125, 0.375],
            scale: [2.0, 3.0, 4.0],
        };
        let item = DroppedItem {
            stack_size: 20, // three flat copies
            gui_3d: false,
            ground_scale: model.transforms.ground.scale,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);

        let vertices_per_copy = 2 * 6;
        let p0 = Vec3::from(out[0].pos);
        let p1 = Vec3::from(out[vertices_per_copy].pos);
        let p2 = Vec3::from(out[vertices_per_copy * 2].pos);
        let expected_step = Vec3::new(0.0, 0.0, FLAT_STACK_SPACING * 4.0);
        assert!(
            (p1 - p0 - expected_step).length() < 1e-6,
            "{p0:?} -> {p1:?}"
        );
        assert!(
            (p2 - p1 - expected_step).length() < 1e-6,
            "{p1:?} -> {p2:?}"
        );
    }

    #[test]
    fn dropped_item_copies_are_jittered_apart_in_the_final_geometry() {
        // The jitter [`DroppedItem::copies`] already proves is not lost by the conversion: two
        // copies of a 3D stack must not land on identical vertices.
        let model = two_face_model();
        let item = DroppedItem {
            stack_size: 64,
            gui_3d: true,
            ..Default::default()
        };
        let mut out = Vec::new();
        emit_dropped_item(&item, &model, Mat4::IDENTITY, 0.0, &mut out);
        let per_copy = 2 * 6;
        let first = &out[..per_copy];
        let second = &out[per_copy..per_copy * 2];
        assert_ne!(
            first.iter().map(|v| v.pos).collect::<Vec<_>>(),
            second.iter().map(|v| v.pos).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn the_count_rule_is_the_five_step_function() {
        assert_eq!(render_count(0), 1);
        assert_eq!(
            render_count(-1),
            1,
            "negative Java stackSize is not an unsigned pile"
        );
        assert_eq!(render_count(1), 1);
        assert_eq!(render_count(2), 2);
        assert_eq!(render_count(16), 2);
        assert_eq!(render_count(17), 3);
        assert_eq!(render_count(32), 3);
        assert_eq!(render_count(33), 4);
        assert_eq!(render_count(48), 4);
        assert_eq!(render_count(49), 5);
        assert_eq!(render_count(64), 5);
    }

    #[test]
    fn the_bob_oscillates_between_zero_and_a_fifth_of_a_block() {
        for age in 0..200 {
            let h = hover_offset(age, 0.0, 0.0);
            assert!((0.0..=0.2001).contains(&h), "age {age} gave {h}");
        }
        // sin(0) * 0.1 + 0.1 = 0.1 exactly at age 0, phase 0.
        assert!((hover_offset(0, 0.0, 0.0) - 0.1).abs() < 1e-6);
        // Peak a quarter period later: (age + p)/10 = PI/2 -> age = 15.7.
        let peak = hover_offset(15, 0.708, 0.0);
        assert!((peak - 0.2).abs() < 1e-4, "{peak}");
    }

    #[test]
    fn the_hover_phase_desynchronises_two_items() {
        let a = hover_offset(10, 0.0, 0.0);
        let b = hover_offset(10, 0.0, 3.0);
        assert!((a - b).abs() > 0.01, "hoverStart must shift the phase");
    }

    #[test]
    fn the_spin_uses_vanillas_radians_to_degrees_quirk() {
        // java: ((age + partial) / 20 + hoverStart) * (180 / PI) — no radians anywhere.
        assert_eq!(spin_degrees(0, 0.0, 0.0), 0.0);
        let d = spin_degrees(20, 0.0, 0.0);
        assert!((d - 180.0 / core::f32::consts::PI).abs() < 1e-4, "{d}");
        // A full 360 takes about 125 ticks, not 20.
        let full = 360.0 / (180.0 / core::f32::consts::PI) * 20.0;
        assert!((full - 125.66).abs() < 0.1, "{full}");
    }

    #[test]
    fn a_single_block_gets_one_half_scale_copy() {
        let d = DroppedItem::default();
        let c = d.copies(0.0);
        assert_eq!(c.len(), 1);
        // The 0.5 scale is in the matrix.
        let s = c[0].matrix.transform_vector3(Vec3::X).length();
        assert!((s - 0.5).abs() < 1e-6, "{s}");
    }

    #[test]
    fn extra_block_copies_are_jittered_within_a_fifteenth_of_a_block() {
        let d = DroppedItem {
            stack_size: 64,
            ..Default::default()
        };
        let c = d.copies(0.0);
        assert_eq!(c.len(), 5);
        let base = c[0].matrix.transform_point3(Vec3::ZERO);
        for copy in &c[1..] {
            let p = copy.matrix.transform_point3(Vec3::ZERO);
            let off = p - base;
            assert!(off.length() > 1e-6, "copy {} was not jittered", copy.index);
            assert!(
                off.x.abs() <= 0.15 + 1e-5
                    && off.y.abs() <= 0.15 + 1e-5
                    && off.z.abs() <= 0.15 + 1e-5,
                "copy {} moved too far: {off}",
                copy.index
            );
        }
    }

    #[test]
    fn the_jitter_is_stable_across_frames_because_the_seed_is_reset() {
        // java: setSeed(187) at the top of doRender — otherwise the pile shimmers.
        let d = DroppedItem {
            stack_size: 64,
            ..Default::default()
        };
        let a = d.copies(0.0);
        let b = d.copies(0.0);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(
                x.matrix.transform_point3(Vec3::ZERO),
                y.matrix.transform_point3(Vec3::ZERO)
            );
        }
        // And two stacks in different places jitter by the same *local* offsets, because the
        // sequence restarts from 187 for each. (Two stacks with different `hoverStart` spin by
        // different amounts, so their world-space offsets differ even though the local jitter
        // is identical — the jitter is applied inside the spin.)
        let d2 = DroppedItem {
            stack_size: 64,
            rel: Vec3::new(10.0, -3.0, 4.0),
            ..Default::default()
        };
        let c = d2.copies(0.0);
        let da =
            a[1].matrix.transform_point3(Vec3::ZERO) - a[0].matrix.transform_point3(Vec3::ZERO);
        let dc =
            c[1].matrix.transform_point3(Vec3::ZERO) - c[0].matrix.transform_point3(Vec3::ZERO);
        assert!((da - dc).length() < 1e-5, "{da} vs {dc}");
    }

    #[test]
    fn a_flat_stack_is_a_row_of_slices_not_a_jittered_pile() {
        let d = DroppedItem {
            stack_size: 20, // 3 copies
            gui_3d: false,
            ..Default::default()
        };
        let c = d.copies(0.0);
        assert_eq!(c.len(), 3);
        // The slices are exactly FLAT_STACK_SPACING apart along the model's own Z.
        let p: Vec<_> = c
            .iter()
            .map(|x| x.matrix.transform_point3(Vec3::ZERO))
            .collect();
        let step1 = (p[1] - p[0]).length();
        let step2 = (p[2] - p[1]).length();
        assert!((step1 - FLAT_STACK_SPACING).abs() < 1e-6, "{step1}");
        assert!((step2 - FLAT_STACK_SPACING).abs() < 1e-6, "{step2}");
        // No 0.5 scale on the flat path.
        let s = c[0].matrix.transform_vector3(Vec3::X).length();
        assert!((s - 1.0).abs() < 1e-6, "{s}");
    }

    #[test]
    fn a_flat_stack_stays_centred_on_the_entity() {
        // java: the -0.046875 * (n - 1) * 0.5 pre-shift in func_177077_a.
        let d = DroppedItem {
            stack_size: 64, // 5 copies
            gui_3d: false,
            ..Default::default()
        };
        let c = d.copies(0.0);
        let first = c[0].matrix.transform_point3(Vec3::ZERO);
        let last = c[4].matrix.transform_point3(Vec3::ZERO);
        let mid = (first + last) * 0.5;
        let base_no_shift = Vec3::new(d.rel.x, d.rel.y + hover_offset(0, 0.0, 0.0) + 0.25, d.rel.z);
        assert!(
            (mid - base_no_shift).length() < 1e-5,
            "stack midpoint {mid} should sit on {base_no_shift}"
        );
    }

    #[test]
    fn the_ground_scale_lifts_the_item_off_the_floor() {
        // java: translate(x, y + f1 + 0.25 * groundScale.y, z)
        let a = DroppedItem::default();
        let b = DroppedItem {
            ground_scale: [1.0, 2.0, 1.0],
            ..Default::default()
        };
        let ya = a.base_matrix(0.0).transform_point3(Vec3::ZERO).y;
        let yb = b.base_matrix(0.0).transform_point3(Vec3::ZERO).y;
        assert!((yb - ya - 0.25).abs() < 1e-6, "{ya} vs {yb}");
    }
}
