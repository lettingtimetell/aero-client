//! "1.7 Animations": the 1.7 held-item maths on top of the
//! vanilla 1.8.9 chain in [`super`].
//!
//! Everything here is render-side and packet-free. With the mod off the callers never reach
//! the 1.7 branches: [`held_item_pose`] hands back the vanilla pose it was given, the
//! third-person helpers return their input, and the thread-local switches stay `false`.
//!
//! The 1.7 numbers come from 1.7.x `ItemRenderer.renderItemInFirstPerson` / `renderItem` and
//! `RenderPlayer.renderEquippedItems`, read for their constants only. Relative to 1.8.9 the first
//! person differs in exactly four places (equip drop, swing translation, block pose,
//! eat/drink, empty arm and map are numerically identical):
//!
//! 1. **Swing while using.** 1.7 applies the swing's three *rotations* with the live swing
//!    progress in every in-use branch; 1.8.9 calls `transformFirstPersonItem(equip, 0)` there.
//!    The swing *translation* stays absent while using, in both versions.
//! 2. **Bow.** 1.7 wraps the pull stretch in a pivot so the bow lengthens along its own axis.
//! 3. **Rod.** `Item.shouldRotateAroundWhenRendering` (fishing rod, carrot on a stick) added
//!    `Ry(180)` right before `renderItem`.
//! 4. **Flat items.** 1.7 drew every non-3D item with `renderItemIn2D` under
//!    `T(0,-0.3,0)·S(1.5)·Ry(50)·Rz(335)·T(-0.9375,-0.0625,0)`, where 1.8.9 uses the model's
//!    `display.firstperson` (rotation `[0,-135,25]`, translation `[0,4,2]`, scale `1.7`).
//!
//! The flat-item correction is applied *relative to* the 1.8.9 default display, between the
//! pose and the model's own display transform. A vanilla model therefore lands exactly on the
//! 1.7 placement, and a resource pack's custom display transform keeps its offset on top of it.

use std::cell::Cell;
use std::sync::OnceLock;

use glam::{Mat4, Vec3};
use mc_math::sin;

use super::{
    first_person_block_transform, first_person_bow_transform, first_person_drinking_transform,
    first_person_item_model_transform, first_person_item_transform, first_person_radians,
    FirstPersonRenderState, FirstPersonUseAction,
};

/// The 1.7 first-person switches for the held item.
///
/// [`super::FirstPersonPhase::old_animations`] holds `None` for vanilla. `Some` with every
/// switch off is also exactly vanilla.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OldAnimations {
    /// Keep the live swing's rotations while blocking, eating, drinking or drawing a bow.
    pub blockhit: bool,
    /// 1.7's flat-item placement for every non-GUI-3D model.
    pub item_positions: bool,
    /// 1.7's bow pull pivot.
    pub bow: bool,
    /// 1.7's fishing-rod placement (applies only while [`Self::rotate_around`] is set).
    pub rod: bool,
    /// The latched held stack is a fishing rod or a carrot on a stick, the two items whose
    /// `Item.shouldRotateAroundWhenRendering()` is true. The client latches it with the stack.
    pub rotate_around: bool,
}

/// 1.8.9's `display.firstperson` for every `builtin/generated` item but the two rods.
const VANILLA_FLAT_DISPLAY: mc_assets::model::ItemTransform = mc_assets::model::ItemTransform {
    rotation: [0.0, -135.0, 25.0],
    translation: [0.0, 4.0 * 0.0625, 2.0 * 0.0625],
    scale: [1.7, 1.7, 1.7],
};

/// 1.8.9's `item/fishing_rod`, `fishing_rod_cast` and `carrot_on_a_stick` first-person display.
const VANILLA_ROD_DISPLAY: mc_assets::model::ItemTransform = mc_assets::model::ItemTransform {
    rotation: [0.0, 45.0, 25.0],
    translation: [0.0, 4.0 * 0.0625, 2.0 * 0.0625],
    scale: [1.7, 1.7, 1.7],
};

/// 1.7 `ItemRenderer.renderItem`'s 2D branch, expressed on a 1.8 `builtin/generated` model
/// *after* `emit_baked_model`'s `(-0.5, -0.5, -0.5)` recentre.
///
/// The last two factors map the 1.8 slab onto 1.7's `renderItemIn2D` slab: `S(-1, 1, -1)` (an
/// exact `Ry(180)`) and `T(0.5, 0.5, -1/32)` send 1.8's south face (`z = +1/32`) to 1.7's back
/// face (`z = -1/16`) and its north face to the front face (`z = 0`), with `x = 1 - u` as
/// `renderItemIn2D(maxU, minV, minU, maxV)` lays the sprite out.
#[must_use]
pub fn flat_item_transform_17() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.0, -0.3, 0.0))
        * Mat4::from_scale(Vec3::splat(1.5))
        * Mat4::from_rotation_y(first_person_radians(50.0))
        * Mat4::from_rotation_z(first_person_radians(335.0))
        * Mat4::from_translation(Vec3::new(-0.9375, -0.0625, 0.0))
        * Mat4::from_translation(Vec3::new(0.5, 0.5, -0.03125))
        * Mat4::from_scale(Vec3::new(-1.0, 1.0, -1.0))
}

/// The correction inserted between the pose and a flat model's display transform:
/// `flat_item_transform_17() * inverse(1.8.9 default flat display chain)`.
///
/// It is `S(1.5/1.7)·Ry(5)·T(≈-0.292, 0.148, -0.033)`: a rotation, translation and positive
/// scale, so the cull face selected from the model's own display stays correct.
#[must_use]
pub fn flat_item_correction() -> Mat4 {
    static CORRECTION: OnceLock<Mat4> = OnceLock::new();
    *CORRECTION.get_or_init(|| {
        flat_item_transform_17()
            * first_person_item_model_transform(false, &VANILLA_FLAT_DISPLAY).inverse()
    })
}

/// The rod's correction: 1.7's `Ry(180)` before the same 2D branch, relative to 1.8.9's rod
/// display (whose own `[0, 45, 25]` already turns the rod round once).
#[must_use]
pub fn rod_correction() -> Mat4 {
    static CORRECTION: OnceLock<Mat4> = OnceLock::new();
    *CORRECTION.get_or_init(|| {
        Mat4::from_rotation_y(first_person_radians(180.0))
            * flat_item_transform_17()
            * first_person_item_model_transform(false, &VANILLA_ROD_DISPLAY).inverse()
    })
}

/// The correction for the current held model, or `None` to leave the vanilla chain untouched.
#[must_use]
pub fn item_correction(old: OldAnimations, gui_3d: bool) -> Option<Mat4> {
    if gui_3d {
        None
    } else if old.rod && old.rotate_around {
        Some(rod_correction())
    } else if old.item_positions {
        Some(flat_item_correction())
    } else {
        None
    }
}

/// `doBowTransformations`' draw progress: Java's `f` (ticks drawn) and `f1` (0..1).
fn bow_draw(item_use_count: i32, max_item_use_duration: i32, partial_ticks: f32) -> (f32, f32) {
    let f = max_item_use_duration.max(1) as f32 - (item_use_count as f32 - partial_ticks + 1.0);
    let mut f1 = f / 20.0;
    f1 = (f1 * f1 + f1 * 2.0) / 3.0;
    if f1 > 1.0 {
        f1 = 1.0;
    }
    (f, f1)
}

/// 1.7's bow pull: vanilla 1.8.9's [`first_person_bow_transform`] with the stretch wrapped in
/// `Rz(-335)·Ry(-50)·T(0,0.5,0)` … `T(0,-0.5,0)·Ry(50)·Rz(335)`, so it lengthens the bow along
/// its own axis about `y = 0.5` instead of along the camera's Z.
#[must_use]
pub fn bow_transform_17(
    item_use_count: i32,
    max_item_use_duration: i32,
    partial_ticks: f32,
) -> Mat4 {
    let (f, f1) = bow_draw(item_use_count, max_item_use_duration, partial_ticks);
    let mut m = Mat4::from_rotation_z(first_person_radians(-18.0))
        * Mat4::from_rotation_y(first_person_radians(-12.0))
        * Mat4::from_rotation_x(first_person_radians(-8.0))
        * Mat4::from_translation(Vec3::new(-0.9, 0.2, 0.0));
    if f1 > 0.1 {
        m *= Mat4::from_translation(Vec3::new(
            0.0,
            sin((f - 0.1) * 1.3) * 0.01 * (f1 - 0.1),
            0.0,
        ));
    }
    m * Mat4::from_translation(Vec3::new(0.0, 0.0, f1 * 0.1))
        * Mat4::from_rotation_z(first_person_radians(-335.0))
        * Mat4::from_rotation_y(first_person_radians(-50.0))
        * Mat4::from_translation(Vec3::new(0.0, 0.5, 0.0))
        * Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0 + f1 * 0.2))
        * Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0))
        * Mat4::from_rotation_y(first_person_radians(50.0))
        * Mat4::from_rotation_z(first_person_radians(335.0))
}

/// The held baked item's pose for this frame, with the 1.7 switches applied.
///
/// `vanilla_pose` is the caller's [`super::first_person_item_pose`] result for the same
/// inputs. It is returned untouched (moved, bit for bit) when `old` is `None` and whenever no
/// switch applies to this frame's branch, so the off path is exactly the vanilla matrix.
/// Otherwise the in-use branch is recomposed through the same vanilla helpers, and the flat-item
/// correction is appended so the caller's `pose * first_person_item_model_transform(..)` sees
/// it between the pose and the display transform.
#[must_use]
pub fn held_item_pose(
    old: Option<OldAnimations>,
    vanilla_pose: Mat4,
    equip: f32,
    swing: f32,
    state: &FirstPersonRenderState,
    partial_ticks: f32,
    gui_3d: bool,
) -> Mat4 {
    let Some(old) = old else {
        return vanilla_pose;
    };
    let count = state.item_use_count;
    let max = state.max_item_use_duration;
    // java (1.7): the three swing rotations always use `getSwingProgress`; 1.8.9 passes 0.
    let s = if old.blockhit { swing } else { 0.0 };
    let pose = match state.use_action {
        FirstPersonUseAction::Idle => vanilla_pose,
        _ if !old.blockhit && !(old.bow && state.use_action == FirstPersonUseAction::Bow) => {
            vanilla_pose
        }
        FirstPersonUseAction::None => first_person_item_transform(equip, s),
        FirstPersonUseAction::EatDrink => {
            first_person_drinking_transform(count, max, partial_ticks)
                * first_person_item_transform(equip, s)
        }
        FirstPersonUseAction::Block => {
            first_person_item_transform(equip, s) * first_person_block_transform()
        }
        FirstPersonUseAction::Bow => {
            let bow = if old.bow {
                bow_transform_17(count, max, partial_ticks)
            } else {
                first_person_bow_transform(count, max, partial_ticks)
            };
            first_person_item_transform(equip, s) * bow
        }
    };
    match item_correction(old, gui_3d) {
        Some(correction) => pose * correction,
        None => pose,
    }
}

// ---- third person ---------------------------------------------------------------------------

thread_local! {
    /// `sword_block_3p`: read by the biped pose and the held-item layer on the render thread.
    static SWORD_BLOCK_3P: Cell<bool> = const { Cell::new(false) };
    /// `armor_red`: read where the hurt flash picks each layer's blend.
    static ARMOR_RED: Cell<bool> = const { Cell::new(false) };
}

/// Set the third-person switches for this thread's entity renderer. The client calls it only
/// when the mod settings change; with the mod off both are `false`, which is vanilla.
///
/// Thread-local rather than global so tests running in parallel threads never see each other's
/// switches.
pub fn set_third_person(sword_block: bool, armor_red: bool) {
    SWORD_BLOCK_3P.with(|cell| cell.set(sword_block));
    ARMOR_RED.with(|cell| cell.set(armor_red));
}

/// Whether the 1.7 third-person sword block is on for this thread.
#[must_use]
pub fn third_person_sword_block() -> bool {
    SWORD_BLOCK_3P.with(Cell::get)
}

/// Whether armour joins the hurt/death flash on this thread.
#[must_use]
pub fn armor_red() -> bool {
    ARMOR_RED.with(Cell::get)
}

/// `ModelBiped.setRotationAngles`' `heldItemRight == 3` arm yaw: 1.8.9's `-30°` (`vanilla`), or
/// 0 with the 1.7 sword block (1.7's `ModelBiped` had no block yaw).
#[must_use]
pub fn blocking_arm_yaw(vanilla: f32) -> f32 {
    if third_person_sword_block() {
        0.0
    } else {
        vanilla
    }
}

/// 1.7 `RenderPlayer.renderEquippedItems`' extra transform for a blocking player's sword:
/// `T(0.05, 0, -0.1)·Ry(-50)·Rx(-10)·Rz(-60)`.
#[must_use]
pub fn third_person_block_transform_17() -> Mat4 {
    Mat4::from_translation(Vec3::new(0.05, 0.0, -0.1))
        * Mat4::from_rotation_y(first_person_radians(-50.0))
        * Mat4::from_rotation_x(first_person_radians(-10.0))
        * Mat4::from_rotation_z(first_person_radians(-60.0))
}

/// `LayerHeldItem`'s matrix just before `renderItem(THIRD_PERSON)`, with the 1.7 block
/// transform appended for a blocking player (`held_item_right == 3`, which only a player whose
/// in-use item has the BLOCK action gets). Anything else returns `matrix` unchanged.
#[must_use]
pub fn third_person_held_item_matrix(matrix: Mat4, held_item_right: i32) -> Mat4 {
    if held_item_right == 3 && third_person_sword_block() {
        matrix * third_person_block_transform_17()
    } else {
        matrix
    }
}

/// Whether a living layer takes `RendererLivingEntity.setBrightness`' hurt combiner: vanilla's
/// `shouldCombineTextures()`, plus armour while `armor_red` is on (1.7 tinted armour too).
#[must_use]
pub fn hurt_flash_combines(layer: crate::entity::layers::LayerKind) -> bool {
    layer.combines_brightness() || (layer == crate::entity::layers::LayerKind::Armor && armor_red())
}

#[cfg(test)]
mod tests {
    use super::super::{first_person_item_pose, FirstPersonPhase};
    use super::*;
    use crate::Gpu;

    const ACTIONS: [FirstPersonUseAction; 5] = [
        FirstPersonUseAction::Idle,
        FirstPersonUseAction::None,
        FirstPersonUseAction::EatDrink,
        FirstPersonUseAction::Block,
        FirstPersonUseAction::Bow,
    ];

    fn state(action: FirstPersonUseAction, count: i32) -> FirstPersonRenderState {
        FirstPersonRenderState {
            use_action: action,
            item_use_count: count,
            max_item_use_duration: if action == FirstPersonUseAction::Bow {
                72_000
            } else {
                32
            },
            ..FirstPersonRenderState::default()
        }
    }

    fn vanilla(equip: f32, swing: f32, st: &FirstPersonRenderState, pt: f32) -> Mat4 {
        first_person_item_pose(
            equip,
            swing,
            st.use_action,
            st.item_use_count,
            st.max_item_use_duration,
            pt,
        )
    }

    fn bits(m: Mat4) -> [u32; 16] {
        m.to_cols_array().map(f32::to_bits)
    }

    /// Off means vanilla: `None`, and `Some` with every switch off, hand back the exact bits of
    /// the vanilla pose for every branch, swing, use count and either model kind.
    #[test]
    fn off_and_all_switches_off_are_bitwise_vanilla() {
        let all_off = [
            None,
            Some(OldAnimations::default()),
            Some(OldAnimations {
                rotate_around: true,
                ..OldAnimations::default()
            }),
        ];
        for action in ACTIONS {
            for count in [1, 7, 20, 31, 71_990] {
                let st = state(action, count);
                for (equip, swing, pt) in [(0.0, 0.0, 0.0), (0.3, 0.42, 0.25), (1.0, 0.9, 0.99)] {
                    let pose = vanilla(equip, swing, &st, pt);
                    for old in all_off {
                        for gui_3d in [false, true] {
                            let got = held_item_pose(old, pose, equip, swing, &st, pt, gui_3d);
                            assert_eq!(bits(got), bits(pose), "{action:?} {old:?} {gui_3d}");
                        }
                    }
                }
            }
        }
        // The two switches that only touch flat models leave GUI-3D (block) models alone.
        let flat_only = OldAnimations {
            item_positions: true,
            rod: true,
            rotate_around: true,
            ..OldAnimations::default()
        };
        let st = state(FirstPersonUseAction::Idle, 0);
        let pose = vanilla(0.2, 0.3, &st, 0.5);
        let got = held_item_pose(Some(flat_only), pose, 0.2, 0.3, &st, 0.5, true);
        assert_eq!(bits(got), bits(pose));
    }

    // ---- an independent oracle: Java's GL matrix stack in f64, axis-angle rotations ----

    /// `glRotatef(angle, x, y, z)` with a unit axis, as the GL spec defines it, in f64.
    fn gl_rotate(angle_deg: f64, axis: [f64; 3]) -> [[f64; 4]; 4] {
        let (s, c) = angle_deg.to_radians().sin_cos();
        let [x, y, z] = axis;
        let t = 1.0 - c;
        // Row-major: m[row][col].
        [
            [x * x * t + c, x * y * t - z * s, x * z * t + y * s, 0.0],
            [y * x * t + z * s, y * y * t + c, y * z * t - x * s, 0.0],
            [z * x * t - y * s, z * y * t + x * s, z * z * t + c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn gl_translate(x: f64, y: f64, z: f64) -> [[f64; 4]; 4] {
        [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    fn gl_scale(x: f64, y: f64, z: f64) -> [[f64; 4]; 4] {
        [
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// The GL stack: every call post-multiplies the current matrix.
    struct GlStack([[f64; 4]; 4]);

    impl GlStack {
        fn new() -> Self {
            Self(gl_scale(1.0, 1.0, 1.0))
        }
        fn mul(&mut self, m: [[f64; 4]; 4]) {
            let a = self.0;
            let mut out = [[0.0; 4]; 4];
            for (r, row) in out.iter_mut().enumerate() {
                for (c, cell) in row.iter_mut().enumerate() {
                    *cell = (0..4).map(|k| a[r][k] * m[k][c]).sum();
                }
            }
            self.0 = out;
        }
        fn rotate(&mut self, angle: f64, x: f64, y: f64, z: f64) {
            self.mul(gl_rotate(angle, [x, y, z]));
        }
        fn translate(&mut self, x: f64, y: f64, z: f64) {
            self.mul(gl_translate(x, y, z));
        }
        fn scale(&mut self, x: f64, y: f64, z: f64) {
            self.mul(gl_scale(x, y, z));
        }
        fn apply(&self, p: [f64; 3]) -> [f64; 3] {
            let m = self.0;
            let mut out = [0.0; 3];
            for (r, cell) in out.iter_mut().enumerate() {
                *cell = m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3];
            }
            out
        }
    }

    /// 1.7.x `renderItemInFirstPerson` for a flat item, transcribed call by call from the Java
    /// (`renderItem`'s 2D branch at the end), then a `renderItemIn2D`-space point.
    ///
    /// The scalar arguments are computed the way the Java computes them (`float` arithmetic and
    /// `MathHelper`'s table sine); only the GL matrix stack itself runs in `f64`.
    #[allow(clippy::too_many_arguments)]
    fn java17_flat_item(
        equip: f32,
        swing: f32,
        action: FirstPersonUseAction,
        count: i32,
        max: i32,
        pt: f32,
        rotate_around: bool,
        point17: [f64; 3],
    ) -> [f64; 3] {
        use mc_math::{cos, sqrt_f};
        let pi = core::f32::consts::PI;
        let w = f64::from;
        let mut gl = GlStack::new();
        let using = action != FirstPersonUseAction::Idle;
        if using {
            if action == FirstPersonUseAction::EatDrink {
                let f14 = count as f32 - pt + 1.0;
                let f15 = 1.0 - f14 / max as f32;
                let mut f16 = 1.0 - f15;
                f16 = f16 * f16 * f16;
                f16 = f16 * f16 * f16;
                f16 = f16 * f16 * f16;
                let f17 = 1.0 - f16;
                let bob = if f64::from(f15) > 0.2 { 1.0 } else { 0.0 };
                gl.translate(0.0, w((cos(f14 / 4.0 * pi) * 0.1).abs() * bob), 0.0);
                gl.translate(w(f17 * 0.6), w(-f17 * 0.5), 0.0);
                gl.rotate(w(f17 * 90.0), 0.0, 1.0, 0.0);
                gl.rotate(w(f17 * 10.0), 1.0, 0.0, 0.0);
                gl.rotate(w(f17 * 30.0), 0.0, 0.0, 1.0);
            }
        } else {
            let f14 = sin(swing * pi);
            let f15 = sin(sqrt_f(swing) * pi);
            gl.translate(
                w(-f15 * 0.4),
                w(sin(sqrt_f(swing) * pi * 2.0) * 0.2),
                w(-f14 * 0.2),
            );
        }
        let f13 = 0.8_f32;
        gl.translate(w(0.7 * f13), w(-0.65 * f13 - equip * 0.6), w(-0.9 * f13));
        gl.rotate(45.0, 0.0, 1.0, 0.0);
        let f14 = sin(swing * swing * pi);
        let f15 = sin(sqrt_f(swing) * pi);
        gl.rotate(w(-f14 * 20.0), 0.0, 1.0, 0.0);
        gl.rotate(w(-f15 * 20.0), 0.0, 0.0, 1.0);
        gl.rotate(w(-f15 * 80.0), 1.0, 0.0, 0.0);
        gl.scale(0.4, 0.4, 0.4);
        if action == FirstPersonUseAction::Block {
            gl.translate(-0.5, w(0.2_f32), 0.0);
            gl.rotate(30.0, 0.0, 1.0, 0.0);
            gl.rotate(-80.0, 1.0, 0.0, 0.0);
            gl.rotate(60.0, 0.0, 1.0, 0.0);
        } else if action == FirstPersonUseAction::Bow {
            gl.rotate(-18.0, 0.0, 0.0, 1.0);
            gl.rotate(-12.0, 0.0, 1.0, 0.0);
            gl.rotate(-8.0, 1.0, 0.0, 0.0);
            gl.translate(w(-0.9_f32), w(0.2_f32), 0.0);
            let f18 = max as f32 - (count as f32 - pt + 1.0);
            let mut f19 = f18 / 20.0;
            f19 = (f19 * f19 + f19 * 2.0) / 3.0;
            if f19 > 1.0 {
                f19 = 1.0;
            }
            if f19 > 0.1 {
                gl.translate(0.0, w(sin((f18 - 0.1) * 1.3) * 0.01 * (f19 - 0.1)), 0.0);
            }
            gl.translate(0.0, 0.0, w(f19 * 0.1));
            gl.rotate(-335.0, 0.0, 0.0, 1.0);
            gl.rotate(-50.0, 0.0, 1.0, 0.0);
            gl.translate(0.0, 0.5, 0.0);
            gl.scale(1.0, 1.0, w(1.0 + f19 * 0.2));
            gl.translate(0.0, -0.5, 0.0);
            gl.rotate(50.0, 0.0, 1.0, 0.0);
            gl.rotate(335.0, 0.0, 0.0, 1.0);
        }
        if rotate_around {
            gl.rotate(180.0, 0.0, 1.0, 0.0);
        }
        // `renderItem`, 2D branch.
        gl.translate(-0.0, w(-0.3_f32), 0.0);
        gl.scale(1.5, 1.5, 1.5);
        gl.rotate(50.0, 0.0, 1.0, 0.0);
        gl.rotate(335.0, 0.0, 0.0, 1.0);
        gl.translate(-0.9375, -0.0625, 0.0);
        gl.apply(point17)
    }

    /// A 1.8 generated-model point (in the model's `0..1` space, before the recentre) and the
    /// same texel's point in 1.7's `renderItemIn2D` space.
    fn model_points() -> Vec<([f32; 3], [f64; 3])> {
        let mut out = Vec::new();
        for &(u, v) in &[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.25, 0.75)] {
            for &(z18, z17) in &[(8.5 / 16.0, -0.0625), (7.5 / 16.0, 0.0)] {
                // 1.8: x = u, y = 1 - v. 1.7: x = 1 - u, y = 1 - v.
                out.push((
                    [u as f32, (1.0 - v) as f32, z18 as f32],
                    [1.0 - u, 1.0 - v, z17],
                ));
            }
        }
        out
    }

    fn assert_close(got: Vec3, want: [f64; 3], what: &str) {
        for (axis, (g, w)) in [got.x, got.y, got.z].iter().zip(want).enumerate() {
            assert!(
                (f64::from(*g) - w).abs() < 2e-5,
                "{what}: axis {axis}: got {got:?}, want {want:?}"
            );
        }
    }

    /// The 1.7 path, run through the exact matrices the phase emits with (pose, the model's
    /// display chain, then `emit_baked_model`'s recentre), lands every texel where 1.7's GL
    /// stack put it: idle, block-hit, eat with a live swing, the 1.7 bow, and the rod.
    #[test]
    fn all_switches_on_match_the_java_17_gl_stack() {
        let on = OldAnimations {
            blockhit: true,
            item_positions: true,
            bow: true,
            rod: true,
            rotate_around: false,
        };
        let recentre = Mat4::from_translation(Vec3::splat(-0.5));
        let cases = [
            (FirstPersonUseAction::Idle, 0, 32),
            (FirstPersonUseAction::Block, 71_995, 72_000),
            (FirstPersonUseAction::EatDrink, 30, 32),
            (FirstPersonUseAction::EatDrink, 5, 32),
            (FirstPersonUseAction::Bow, 71_996, 72_000),
            (FirstPersonUseAction::Bow, 71_970, 72_000),
        ];
        for (action, count, max) in cases {
            for rotate_around in [false, true] {
                let display = if rotate_around {
                    VANILLA_ROD_DISPLAY
                } else {
                    VANILLA_FLAT_DISPLAY
                };
                let old = OldAnimations {
                    rotate_around,
                    ..on
                };
                for (equip, swing, pt) in [(0.0, 0.0, 0.0), (0.35, 0.4, 0.5), (0.0, 0.83, 0.9)] {
                    let st = FirstPersonRenderState {
                        use_action: action,
                        item_use_count: count,
                        max_item_use_duration: max,
                        ..FirstPersonRenderState::default()
                    };
                    let pose = held_item_pose(
                        Some(old),
                        vanilla(equip, swing, &st, pt),
                        equip,
                        swing,
                        &st,
                        pt,
                        false,
                    );
                    let full = pose * first_person_item_model_transform(false, &display) * recentre;
                    for (p18, p17) in model_points() {
                        let want = java17_flat_item(
                            equip,
                            swing,
                            action,
                            count,
                            max,
                            pt,
                            rotate_around,
                            p17,
                        );
                        let got = full.transform_point3(Vec3::from(p18));
                        assert_close(
                            got,
                            want,
                            &format!("{action:?} rod={rotate_around} e={equip} s={swing}"),
                        );
                    }
                }
            }
        }
    }

    /// The oracle is not vacuous: 1.8.9's own chain is somewhere else (the bigger 1.7 scale and
    /// lower-right placement), and while blocking it drops the swing the 1.7 path keeps.
    #[test]
    fn vanilla_chain_differs_from_the_17_oracle() {
        let recentre = Mat4::from_translation(Vec3::splat(-0.5));
        let st = state(FirstPersonUseAction::Block, 71_995);
        let full = vanilla(0.0, 0.5, &st, 0.5)
            * first_person_item_model_transform(false, &VANILLA_FLAT_DISPLAY)
            * recentre;
        let (p18, p17) = model_points()[3];
        let want = java17_flat_item(
            0.0,
            0.5,
            FirstPersonUseAction::Block,
            71_995,
            72_000,
            0.5,
            false,
            p17,
        );
        let got = full.transform_point3(Vec3::from(p18));
        let gap = (f64::from(got.x) - want[0]).abs()
            + (f64::from(got.y) - want[1]).abs()
            + (f64::from(got.z) - want[2]).abs();
        assert!(
            gap > 0.05,
            "vanilla and 1.7 must differ while block-hitting: {gap}"
        );

        // Block-hit alone (no position change) changes only the in-use branches.
        let blockhit = OldAnimations {
            blockhit: true,
            ..OldAnimations::default()
        };
        let pose = vanilla(0.0, 0.5, &st, 0.5);
        let hit = held_item_pose(Some(blockhit), pose, 0.0, 0.5, &st, 0.5, false);
        assert_ne!(bits(hit), bits(pose));
        assert_eq!(
            bits(hit),
            bits(first_person_item_transform(0.0, 0.5) * first_person_block_transform())
        );
        let idle = state(FirstPersonUseAction::Idle, 0);
        let pose = vanilla(0.0, 0.5, &idle, 0.5);
        let got = held_item_pose(Some(blockhit), pose, 0.0, 0.5, &idle, 0.5, false);
        assert_eq!(bits(got), bits(pose));
    }

    #[test]
    fn corrections_are_proper_rotations_with_the_rounded_shape() {
        for correction in [flat_item_correction(), rod_correction()] {
            // Positive determinant: the model's own display still decides the cull face.
            assert!(correction.determinant() > 0.0);
            let scale = correction.transform_vector3(Vec3::X).length();
            assert!((scale - 1.5 / 1.7).abs() < 1e-5, "{scale}");
        }
        // The flat correction is ~ S(1.5/1.7)·Ry(5)·T(-0.29, 0.149, -0.0328) once rounded.
        let expected = Mat4::from_scale(Vec3::splat(1.5 / 1.7))
            * Mat4::from_rotation_y(first_person_radians(5.0))
            * Mat4::from_translation(Vec3::new(-0.29, 0.149, -0.0328));
        for (got, want) in flat_item_correction()
            .to_cols_array()
            .iter()
            .zip(expected.to_cols_array())
        {
            assert!((got - want).abs() < 3e-3, "{got} vs {want}");
        }
    }

    #[test]
    fn third_person_switches_are_thread_local_and_default_to_vanilla() {
        // 1.8.9's block yaw, -30°: `ModelBiped`'s `-0.5235988F`.
        let arm = -std::f32::consts::FRAC_PI_6;
        let matrix = Mat4::from_translation(Vec3::new(1.0, 2.0, 3.0));
        assert!(!third_person_sword_block() && !armor_red());
        assert_eq!(blocking_arm_yaw(arm).to_bits(), arm.to_bits());
        assert_eq!(bits(third_person_held_item_matrix(matrix, 3)), bits(matrix));
        assert!(!hurt_flash_combines(
            crate::entity::layers::LayerKind::Armor
        ));

        set_third_person(true, true);
        assert_eq!(blocking_arm_yaw(arm).to_bits(), 0.0_f32.to_bits());
        assert_eq!(
            bits(third_person_held_item_matrix(matrix, 3)),
            bits(matrix * third_person_block_transform_17())
        );
        // Only the blocking pose (3) changes; an ordinary held item (1) does not.
        assert_eq!(bits(third_person_held_item_matrix(matrix, 1)), bits(matrix));
        assert!(hurt_flash_combines(crate::entity::layers::LayerKind::Armor));
        assert!(!hurt_flash_combines(
            crate::entity::layers::LayerKind::HeldItem
        ));
        // Another thread (another test, the asset loader) still sees vanilla.
        let other = std::thread::spawn(|| (third_person_sword_block(), armor_red()))
            .join()
            .expect("thread");
        assert_eq!(other, (false, false));
        set_third_person(false, false);
        assert_eq!(blocking_arm_yaw(arm).to_bits(), arm.to_bits());
    }

    /// The switch reaches the real `ModelBiped` pose: a blocking player's right arm keeps 1.8.9's
    /// -30° yaw while it is off and has none with the 1.7 sword block on.
    #[test]
    fn biped_block_pose_reads_the_switch() {
        use crate::entity::anim::AnimParams;
        use crate::entity::models::biped::PlayerModel;
        let mut params = AnimParams::still();
        params.entity.held_item_right = 3;
        let yaw = |model: &mut PlayerModel| {
            model.biped.pose(&params);
            model
                .biped
                .model
                .get(model.biped.parts.right_arm)
                .rotate_angle[1]
        };
        let mut model = PlayerModel::new(0.0, false);
        let vanilla = yaw(&mut model);
        set_third_person(true, false);
        let old = yaw(&mut model);
        set_third_person(false, false);
        let again = yaw(&mut model);
        assert!(
            (vanilla + std::f32::consts::FRAC_PI_6).abs() < 1e-6,
            "{vanilla}"
        );
        assert_eq!(old.to_bits(), 0.0_f32.to_bits());
        assert_eq!(again.to_bits(), vanilla.to_bits());
    }

    /// Not an assertion: renders vanilla and 1.7 held items side by side into PNGs for a human
    /// look. Opt-in only: runs when `M5_RENDER_DIR` names an output folder (and the vanilla jar
    /// is found through `MC_JARS_DIR`), so the ordinary suite never bakes the jar here.
    #[test]
    fn render_held_items_for_a_look() {
        use crate::{Camera, GpuOptions, PhaseList, PhaseOrder};
        const W: u32 = 480;
        const H: u32 = 270;
        let Some(out) = std::env::var_os("M5_RENDER_DIR").map(std::path::PathBuf::from) else {
            return;
        };
        let Some(jar) = std::env::var_os("MC_JARS_DIR")
            .map(|dir| std::path::PathBuf::from(dir).join("versions/1.8.9/1.8.9.jar"))
            .filter(|jar| jar.is_file())
        else {
            eprintln!("SKIP: set MC_JARS_DIR to the folder holding versions/1.8.9/1.8.9.jar");
            return;
        };
        let mut rm = mc_assets::ResourceManager::new();
        rm.push_pack(Box::new(
            mc_assets::ZipResourcePack::open(&jar).expect("open the vanilla jar"),
        ));
        let (baked, _report) = mc_assets::bake_all(&rm, 4096).expect("bake the vanilla jar");
        let atlas = baked.atlas.clone();
        mc_assets::install(baked);
        let options = GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        };
        let mut gpu = Gpu::headless(W, H, options).expect("a wgpu adapter");
        std::fs::create_dir_all(&out).expect("output dir");

        let on = OldAnimations {
            blockhit: true,
            item_positions: true,
            bow: true,
            rod: true,
            rotate_around: false,
        };
        // (file stem, item, use action, use count, swing progress)
        let shots = [
            (
                "sword_idle",
                "minecraft:diamond_sword",
                FirstPersonUseAction::Idle,
                0,
                0.0,
            ),
            (
                "sword_block",
                "minecraft:diamond_sword",
                FirstPersonUseAction::Block,
                71_990,
                0.0,
            ),
            (
                "sword_blockhit",
                "minecraft:diamond_sword",
                FirstPersonUseAction::Block,
                71_990,
                0.4,
            ),
            (
                "bow_drawn",
                "minecraft:bow",
                FirstPersonUseAction::Bow,
                71_985,
                0.0,
            ),
            (
                "rod_idle",
                "minecraft:fishing_rod",
                FirstPersonUseAction::Idle,
                0,
                0.0,
            ),
            (
                "apple_idle",
                "minecraft:apple",
                FirstPersonUseAction::Idle,
                0,
                0.0,
            ),
        ];
        for (stem, item, action, count, swing) in shots {
            let id = mc_data::item_by_name(item).expect("a 1.8.9 item");
            let model = mc_assets::item_model_for(id).clone();
            let rotate_around = item == "minecraft:fishing_rod";
            for (tag, old) in [
                ("18", None),
                (
                    "17",
                    Some(OldAnimations {
                        rotate_around,
                        ..on
                    }),
                ),
            ] {
                let mut phase = FirstPersonPhase::new(&gpu);
                phase.set_atlas(&gpu, &atlas);
                let gui_3d = model.gui_3d;
                phase.set_item_model(Some(model.clone()), gui_3d);
                phase.render_state.prev_equipped_progress = 1.0;
                phase.render_state.equipped_progress = 1.0;
                phase.render_state.use_action = action;
                phase.render_state.item_use_count = count;
                phase.render_state.max_item_use_duration = 72_000;
                phase.render_state.prev_swing_progress = swing;
                phase.render_state.swing_progress = swing;
                phase.old_animations = old;
                let mut camera = Camera::default();
                camera.set_viewport(W, H);
                let mut phases = PhaseList::new();
                phases.clear_color = wgpu::Color {
                    r: 0.35,
                    g: 0.55,
                    b: 0.85,
                    a: 1.0,
                };
                phases.insert(PhaseOrder::FIRST_PERSON, Box::new(phase));
                let frame = gpu.acquire().expect("acquire offscreen frame");
                phases.render(&gpu, frame.view(), &camera, 0.0);
                let pixels = gpu.read_pixels().expect("readback");
                let path = out.join(format!("{stem}_{tag}.png"));
                image::save_buffer(&path, &pixels, W, H, image::ColorType::Rgba8)
                    .expect("write png");
            }
        }
    }

    /// The phase itself: the emitted held-item vertices are bit-identical with `None` and with
    /// every switch off, and block-hit changes them only while blocking with a live swing.
    #[test]
    fn phase_vertices_follow_the_switches() {
        let Ok(gpu) = Gpu::headless(16, 16, crate::context::GpuOptions::default()) else {
            eprintln!("skip: no GPU adapter in this environment");
            return;
        };
        let vertex = |x: f32, y: f32| mc_assets::Vertex {
            pos: [x, y, 0.5],
            uv: [x, y],
        };
        let mut model = mc_assets::BakedModel::default();
        model.general_quads.push(mc_assets::BakedQuad {
            verts: [
                vertex(0.0, 0.0),
                vertex(1.0, 0.0),
                vertex(1.0, 1.0),
                vertex(0.0, 1.0),
            ],
            face: None,
            shade_face: None,
            tint_index: None,
            shade: false,
        });
        model.transforms.first_person = VANILLA_FLAT_DISPLAY;
        let mut phase = FirstPersonPhase::new(&gpu);
        phase.set_item_model(Some(model), false);
        phase.render_state.use_action = FirstPersonUseAction::Block;
        phase.render_state.item_use_count = 71_990;
        phase.render_state.max_item_use_duration = 72_000;
        phase.render_state.prev_swing_progress = 1.0 / 6.0;
        phase.render_state.swing_progress = 2.0 / 6.0;
        phase.partial_ticks = 0.5;

        let emit = |phase: &mut FirstPersonPhase, old: Option<OldAnimations>| {
            phase.old_animations = old;
            phase.rebuild_vertices(0.0, 0);
            phase
                .item_vertices
                .iter()
                .flat_map(|v| v.pos.map(f32::to_bits))
                .collect::<Vec<_>>()
        };
        let vanilla = emit(&mut phase, None);
        assert!(!vanilla.is_empty());
        assert_eq!(emit(&mut phase, Some(OldAnimations::default())), vanilla);
        let blockhit = Some(OldAnimations {
            blockhit: true,
            ..OldAnimations::default()
        });
        assert_ne!(emit(&mut phase, blockhit), vanilla, "the swing keeps going");
        // Idle: block-hit has nothing to change.
        phase.render_state.use_action = FirstPersonUseAction::Idle;
        let idle = emit(&mut phase, None);
        assert_eq!(emit(&mut phase, blockhit), idle);
        // Item positions move every flat item, idle included.
        let positions = Some(OldAnimations {
            item_positions: true,
            ..OldAnimations::default()
        });
        assert_ne!(emit(&mut phase, positions), idle);
        assert_eq!(phase.item_cull_face, Some(wgpu::Face::Back));
    }
}
