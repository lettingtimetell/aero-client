//! Passive-mob and utility-mob models.
//!
//! java: `ModelChicken.java`, `ModelWolf.java`, `ModelOcelot.java`, `ModelRabbit.java`,
//! `ModelSquid.java`, `ModelBat.java`, `ModelSnowMan.java`, `ModelIronGolem.java`,
//! `ModelVillager.java`, `ModelWitch.java`.
//!
//! Ported here, one Rust type per Java class:
//!
//! | Rust | Java |
//! |---|---|
//! | [`ChickenModel`] | `ModelChicken` |
//! | [`WolfModel`] | `ModelWolf` |
//! | [`OcelotModel`] | `ModelOcelot` |
//! | [`RabbitModel`] | `ModelRabbit` |
//! | [`SquidModel`] | `ModelSquid` |
//! | [`BatModel`] | `ModelBat` |
//! | [`SnowManModel`] | `ModelSnowMan` |
//! | [`IronGolemModel`] | `ModelIronGolem` |
//! | [`HorseModel`] | `ModelHorse` |
//! | [`VillagerModel`] | `ModelVillager` |
//! | [`WitchModel`] | `ModelWitch` (transcribed standalone rather than subclassing) |
//!
//! See [`super::biped`] for the reference shape. Three things recur in this file specifically:
//!
//! * **Baby scaling.** The chicken, wolf, ocelot and rabbit all override `render` to draw the
//!   head group and the body group under different matrices when `isChild`. `GlStateManager`
//!   composes right-to-left onto the current matrix, so `scale(a)` then `translate(t)` becomes
//!   `base * from_scale(a) * from_translation(t)`.
//! * **`renderWithRotation`.** `ModelWolf` draws its head and tail through
//!   `ModelRenderer.renderWithRotation`, which rotates **Y, X, Z** — not the Z, Y, X of
//!   `ModelRenderer.render` — and never walks `childModels`. [`render_with_rotation`] is that
//!   path.
//! * **Entity state with no [`crate::entity::EntityAnimState`] field.** The witch's entity id
//!   lives as a documented public field on the model, the way `ModelWitch.field_82900_g` does in
//!   the Java: the renderer writes it before posing.
//!
//!   `EntityWolf`'s wet-shake timer used to be one of these and no longer is. A field on the
//!   model is only safe for a value that is constant across every entity of that type, because
//!   **one model instance is shared by all of them** — a per-entity value stored there is
//!   whichever entity was posed last. The shake timer is now
//!   [`EntityAnimState::wolf_shake_time`](crate::entity::anim::EntityAnimState::wolf_shake_time)
//!   and the guardian's spike extension is
//!   [`EntityAnimState::spikes_animation`](crate::entity::anim::EntityAnimState::spikes_animation)
//!   for the same reason.

// These decimal angles are Java's source literals. Replacing them with Rust PI fractions can
// change the narrowed f32 bit pattern, so retain the literals used by the model port.
#![allow(clippy::approx_constant)]

use glam::{Mat4, Vec3};
use mc_math::{cos, sin};

use crate::entity::anim::{model_radians, AnimParams, EntityModel};
use crate::entity::model::{Model, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

/// The `0.017453292F` literal several of these models write out longhand instead of dividing by
/// `180F / (float)Math.PI`.
///
/// java: `ModelRabbit.setRotationAngles`. It is *not* bit-identical to [`model_radians`] in general,
/// so the two spellings are kept apart rather than unified.
const DEG_TO_RAD_LITERAL: f32 = 0.017_453_292;

/// `GlStateManager.translate(x, y, z)` as a matrix factor.
fn tr(x: f32, y: f32, z: f32) -> Mat4 {
    Mat4::from_translation(Vec3::new(x, y, z))
}

/// `GlStateManager.scale(s, s, s)` as a matrix factor.
fn sc(s: f32) -> Mat4 {
    Mat4::from_scale(Vec3::splat(s))
}

/// Emits one part the way `ModelRenderer.renderWithRotation` draws it.
///
/// java: `ModelRenderer.renderWithRotation` — pivot translate, then rotate **Y, then X, then Z**
/// (the plain `render` path is Z, then Y, then X), no `offsetX/Y/Z` translate, and no recursion
/// into `childModels`.
///
/// [`Model::emit_part`] always post-multiplies the part's own [`crate::entity::ModelPart::local_matrix`],
/// so the wanted transform is handed in pre-multiplied by that matrix's analytic inverse. The
/// inverse is written out as the reversed chain of primitives rather than a general `Mat4`
/// inversion, which keeps it exact up to the composition rounding.
fn render_with_rotation(
    model: &Model,
    id: PartId,
    scale: f32,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    let part = model.get(id);
    let [px, py, pz] = part.rotation_point;
    let [ax, ay, az] = part.rotate_angle;
    let [ox, oy, oz] = part.offset;

    let want = tr(px * scale, py * scale, pz * scale)
        * Mat4::from_rotation_y(ay)
        * Mat4::from_rotation_x(ax)
        * Mat4::from_rotation_z(az);
    // (T(offset) * T(pivot*scale) * Rz * Ry * Rx)^-1
    let undo = Mat4::from_rotation_x(-ax)
        * Mat4::from_rotation_y(-ay)
        * Mat4::from_rotation_z(-az)
        * tr(-px * scale, -py * scale, -pz * scale)
        * tr(-ox, -oy, -oz);

    model.emit_part(id, scale, base * want * undo, out);
}

// ---------------------------------------------------------------------------------------------
// ModelChicken
// ---------------------------------------------------------------------------------------------

/// The eight parts of the chicken.
#[derive(Clone, Copy, Debug)]
pub struct ChickenParts {
    /// java: `head`
    pub head: PartId,
    /// java: `bill`
    pub bill: PartId,
    /// java: `chin`
    pub chin: PartId,
    /// java: `body`
    pub body: PartId,
    /// java: `rightLeg`
    pub right_leg: PartId,
    /// java: `leftLeg`
    pub left_leg: PartId,
    /// java: `rightWing`
    pub right_wing: PartId,
    /// java: `leftWing`
    pub left_wing: PartId,
}

/// The chicken.
///
/// java: `ModelChicken.java`
#[derive(Clone, Debug)]
pub struct ChickenModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the eight parts.
    pub parts: ChickenParts,
}

impl ChickenModel {
    /// java: `ModelChicken()`
    #[must_use]
    pub fn new() -> Self {
        // java: int i = 16;
        let i = 16.0_f32;
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .cube(-2.0, -6.0, -2.0, 4, 6, 3, 0.0)
            .at(0.0, -1.0 + i, -4.0)
            .id();
        let bill = m
            .part("bill", 14, 0)
            .cube(-2.0, -4.0, -4.0, 4, 2, 2, 0.0)
            .at(0.0, -1.0 + i, -4.0)
            .id();
        let chin = m
            .part("chin", 14, 4)
            .cube(-1.0, -2.0, -3.0, 2, 2, 2, 0.0)
            .at(0.0, -1.0 + i, -4.0)
            .id();
        let body = m
            .part("body", 0, 9)
            .cube(-3.0, -4.0, -3.0, 6, 8, 6, 0.0)
            .at(0.0, i, 0.0)
            .id();
        // java: the leg and wing boxes use the 6-arg `addBox`, i.e. scaleFactor 0.
        let right_leg = m
            .part("right_leg", 26, 0)
            .cube(-1.0, 0.0, -3.0, 3, 5, 3, 0.0)
            .at(-2.0, 3.0 + i, 1.0)
            .id();
        let left_leg = m
            .part("left_leg", 26, 0)
            .cube(-1.0, 0.0, -3.0, 3, 5, 3, 0.0)
            .at(1.0, 3.0 + i, 1.0)
            .id();
        let right_wing = m
            .part("right_wing", 24, 13)
            .cube(0.0, 0.0, -3.0, 1, 4, 6, 0.0)
            .at(-4.0, -3.0 + i, 0.0)
            .id();
        let left_wing = m
            .part("left_wing", 24, 13)
            .cube(-1.0, 0.0, -3.0, 1, 4, 6, 0.0)
            .at(4.0, -3.0 + i, 0.0)
            .id();
        Self {
            model: m,
            parts: ChickenParts {
                head,
                bill,
                chin,
                body,
                right_leg,
                left_leg,
                right_wing,
                left_wing,
            },
        }
    }
}

impl Default for ChickenModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for ChickenModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelChicken.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        let head_x = model_radians(p.head_pitch);
        let head_y = model_radians(p.net_head_yaw);
        m.get_mut(q.head).rotate_angle[0] = head_x;
        m.get_mut(q.head).rotate_angle[1] = head_y;
        m.get_mut(q.bill).rotate_angle[0] = head_x;
        m.get_mut(q.bill).rotate_angle[1] = head_y;
        m.get_mut(q.chin).rotate_angle[0] = head_x;
        m.get_mut(q.chin).rotate_angle[1] = head_y;
        m.get_mut(q.body).rotate_angle[0] = PI / 2.0;
        m.get_mut(q.right_leg).rotate_angle[0] =
            cos(p.limb_swing * 0.6662) * 1.4 * p.limb_swing_amount;
        m.get_mut(q.left_leg).rotate_angle[0] =
            cos(p.limb_swing * 0.6662 + PI) * 1.4 * p.limb_swing_amount;
        m.get_mut(q.right_wing).rotate_angle[2] = p.age_in_ticks;
        m.get_mut(q.left_wing).rotate_angle[2] = -p.age_in_ticks;
    }

    /// java: `ModelChicken.render`
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let s = p.scale;
        if self.model.is_child {
            // java: float f = 2.0F — the head group takes only the translate, no scale.
            let head_m = base * tr(0.0, 5.0 * s, 2.0 * s);
            self.model
                .emit_some(&[q.head, q.bill, q.chin], s, head_m, out);
            let body_m = base * sc(1.0 / 2.0) * tr(0.0, 24.0 * s, 0.0);
            self.model.emit_some(
                &[q.body, q.right_leg, q.left_leg, q.right_wing, q.left_wing],
                s,
                body_m,
                out,
            );
        } else {
            self.model.emit_some(
                &[
                    q.head,
                    q.bill,
                    q.chin,
                    q.body,
                    q.right_leg,
                    q.left_leg,
                    q.right_wing,
                    q.left_wing,
                ],
                s,
                base,
                out,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ModelWolf
// ---------------------------------------------------------------------------------------------

/// The eight parts of the wolf.
#[derive(Clone, Copy, Debug)]
pub struct WolfParts {
    /// java: `wolfHeadMain`
    pub head: PartId,
    /// java: `wolfBody`
    pub body: PartId,
    /// java: `wolfMane`
    pub mane: PartId,
    /// java: `wolfLeg1` — rear left
    pub leg1: PartId,
    /// java: `wolfLeg2` — rear right
    pub leg2: PartId,
    /// java: `wolfLeg3` — front left
    pub leg3: PartId,
    /// java: `wolfLeg4` — front right
    pub leg4: PartId,
    /// java: `wolfTail`
    pub tail: PartId,
}

/// The wolf.
///
/// java: `ModelWolf.java`
#[derive(Clone, Debug)]
pub struct WolfModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the eight parts.
    pub parts: WolfParts,
}

impl WolfModel {
    /// java: `ModelWolf()`
    #[must_use]
    pub fn new() -> Self {
        // java: float f = 0.0F; float f1 = 13.5F;
        let f = 0.0_f32;
        let f1 = 13.5_f32;
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .cube(-3.0, -3.0, -2.0, 6, 6, 4, f)
            .at(-1.0, f1, -7.0)
            .id();
        let body = m
            .part("body", 18, 14)
            .cube(-4.0, -2.0, -3.0, 6, 9, 6, f)
            .at(0.0, 14.0, 2.0)
            .id();
        let mane = m
            .part("mane", 21, 0)
            .cube(-4.0, -3.0, -3.0, 8, 6, 7, f)
            .at(-1.0, 14.0, 2.0)
            .id();
        let leg1 = m
            .part("leg1", 0, 18)
            .cube(-1.0, 0.0, -1.0, 2, 8, 2, f)
            .at(-2.5, 16.0, 7.0)
            .id();
        let leg2 = m
            .part("leg2", 0, 18)
            .cube(-1.0, 0.0, -1.0, 2, 8, 2, f)
            .at(0.5, 16.0, 7.0)
            .id();
        let leg3 = m
            .part("leg3", 0, 18)
            .cube(-1.0, 0.0, -1.0, 2, 8, 2, f)
            .at(-2.5, 16.0, -4.0)
            .id();
        let leg4 = m
            .part("leg4", 0, 18)
            .cube(-1.0, 0.0, -1.0, 2, 8, 2, f)
            .at(0.5, 16.0, -4.0)
            .id();
        let tail = m
            .part("tail", 9, 18)
            .cube(-1.0, 0.0, -1.0, 2, 8, 2, f)
            .at(-1.0, 12.0, 8.0)
            .id();
        // java: the three ear/snout boxes are appended to `wolfHeadMain` at the very end of the
        // constructor, after every other part exists. Order inside `cubeList` is what matters.
        {
            let h = m.get_mut(head);
            h.texture_offset = [16, 14];
            h.add_box(-3.0, -5.0, 0.0, 2, 2, 1, f);
            h.texture_offset = [16, 14];
            h.add_box(1.0, -5.0, 0.0, 2, 2, 1, f);
            h.texture_offset = [0, 10];
            h.add_box(-1.5, 0.0, -5.0, 3, 3, 4, f);
        }
        Self {
            model: m,
            parts: WolfParts {
                head,
                body,
                mane,
                leg1,
                leg2,
                leg3,
                leg4,
                tail,
            },
        }
    }

    /// java: `EntityWolf.getShakeAngle(partialTickTime, offset)`, with the interpolation of
    /// `timeWolfIsShaking` already done into
    /// [`EntityAnimState::wolf_shake_time`](crate::entity::anim::EntityAnimState::wolf_shake_time).
    ///
    /// It is an argument rather than a field on the model because **one `WolfModel` is shared by
    /// every wolf on screen**: a per-entity value stored here would be whichever wolf was posed
    /// last.
    #[must_use]
    pub fn shake_angle(shake_time: f32, offset: f32) -> f32 {
        // java: `if (f < 0) f = 0; else if (f > 1) f = 1;` — `f32::clamp` is that exact chain,
        // NaN handling included.
        let f = ((shake_time + offset) / 1.8).clamp(0.0, 1.0);
        sin(f * PI) * sin(f * PI * 11.0) * 0.15 * PI
    }
}

impl Default for WolfModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for WolfModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelWolf.setLivingAnimations`
    fn set_living_animations(&mut self, p: &AnimParams) {
        let q = self.parts;
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        // java: entitywolf.isAngry() -> p.entity.aggressive
        let angry = p.entity.aggressive;
        // java: entitywolf.isSitting() -> p.entity.sitting
        let sitting = p.entity.sitting;
        // java: entitywolf.getInterestedAngle(partialTickTime) -> p.entity.tail_rotation
        let interested = p.entity.tail_rotation;
        let shake = p.entity.wolf_shake_time;
        let shake0 = Self::shake_angle(shake, 0.0);
        let shake_mane = Self::shake_angle(shake, -0.08);
        let shake_body = Self::shake_angle(shake, -0.16);
        let shake_tail = Self::shake_angle(shake, -0.2);
        let m = &mut self.model;

        if angry {
            m.get_mut(q.tail).rotate_angle[1] = 0.0;
        } else {
            m.get_mut(q.tail).rotate_angle[1] = cos(swing * 0.6662) * 1.4 * amount;
        }

        if sitting {
            m.get_mut(q.mane).set_rotation_point(-1.0, 16.0, -3.0);
            m.get_mut(q.mane).rotate_angle[0] = PI * 2.0 / 5.0;
            m.get_mut(q.mane).rotate_angle[1] = 0.0;
            m.get_mut(q.body).set_rotation_point(0.0, 18.0, 0.0);
            m.get_mut(q.body).rotate_angle[0] = PI / 4.0;
            m.get_mut(q.tail).set_rotation_point(-1.0, 21.0, 6.0);
            m.get_mut(q.leg1).set_rotation_point(-2.5, 22.0, 2.0);
            m.get_mut(q.leg1).rotate_angle[0] = PI * 3.0 / 2.0;
            m.get_mut(q.leg2).set_rotation_point(0.5, 22.0, 2.0);
            m.get_mut(q.leg2).rotate_angle[0] = PI * 3.0 / 2.0;
            m.get_mut(q.leg3).rotate_angle[0] = 5.811_947;
            m.get_mut(q.leg3).set_rotation_point(-2.49, 17.0, -4.0);
            m.get_mut(q.leg4).rotate_angle[0] = 5.811_947;
            m.get_mut(q.leg4).set_rotation_point(0.51, 17.0, -4.0);
        } else {
            m.get_mut(q.body).set_rotation_point(0.0, 14.0, 2.0);
            m.get_mut(q.body).rotate_angle[0] = PI / 2.0;
            m.get_mut(q.mane).set_rotation_point(-1.0, 14.0, -3.0);
            let body_x = m.get(q.body).rotate_angle[0];
            m.get_mut(q.mane).rotate_angle[0] = body_x;
            m.get_mut(q.tail).set_rotation_point(-1.0, 12.0, 8.0);
            m.get_mut(q.leg1).set_rotation_point(-2.5, 16.0, 7.0);
            m.get_mut(q.leg2).set_rotation_point(0.5, 16.0, 7.0);
            m.get_mut(q.leg3).set_rotation_point(-2.5, 16.0, -4.0);
            m.get_mut(q.leg4).set_rotation_point(0.5, 16.0, -4.0);
            m.get_mut(q.leg1).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
            m.get_mut(q.leg2).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
            m.get_mut(q.leg3).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
            m.get_mut(q.leg4).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
        }

        m.get_mut(q.head).rotate_angle[2] = interested + shake0;
        m.get_mut(q.mane).rotate_angle[2] = shake_mane;
        m.get_mut(q.body).rotate_angle[2] = shake_body;
        m.get_mut(q.tail).rotate_angle[2] = shake_tail;
    }

    /// java: `ModelWolf.setRotationAngles` (its `super` call is `ModelBase`'s empty body).
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(q.tail).rotate_angle[0] = p.age_in_ticks;
    }

    /// java: `ModelWolf.render` — head and tail go through `renderWithRotation`.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let s = p.scale;
        let legs = [q.body, q.leg1, q.leg2, q.leg3, q.leg4];
        if self.model.is_child {
            // java: float f = 2.0F — again the head takes only the translate.
            let head_m = base * tr(0.0, 5.0 * s, 2.0 * s);
            render_with_rotation(&self.model, q.head, s, head_m, out);
            let body_m = base * sc(1.0 / 2.0) * tr(0.0, 24.0 * s, 0.0);
            self.model.emit_some(&legs, s, body_m, out);
            render_with_rotation(&self.model, q.tail, s, body_m, out);
            self.model.emit_some(&[q.mane], s, body_m, out);
        } else {
            render_with_rotation(&self.model, q.head, s, base, out);
            self.model.emit_some(&legs, s, base, out);
            render_with_rotation(&self.model, q.tail, s, base, out);
            self.model.emit_some(&[q.mane], s, base, out);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ModelOcelot
// ---------------------------------------------------------------------------------------------

/// The eight parts of the ocelot.
#[derive(Clone, Copy, Debug)]
pub struct OcelotParts {
    /// java: `ocelotHead` — four boxes: `main`, `nose`, `ear1`, `ear2`
    pub head: PartId,
    /// java: `ocelotBody`
    pub body: PartId,
    /// java: `ocelotTail`
    pub tail: PartId,
    /// java: `ocelotTail2`
    pub tail2: PartId,
    /// java: `ocelotBackLeftLeg`
    pub back_left_leg: PartId,
    /// java: `ocelotBackRightLeg`
    pub back_right_leg: PartId,
    /// java: `ocelotFrontLeftLeg`
    pub front_left_leg: PartId,
    /// java: `ocelotFrontRightLeg`
    pub front_right_leg: PartId,
}

/// The ocelot.
///
/// java: `ModelOcelot.java`
#[derive(Clone, Debug)]
pub struct OcelotModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the eight parts.
    pub parts: OcelotParts,
    /// Which pose `setLivingAnimations` last chose: 0 sneaking, 1 normal, 2 sprinting, 3 sitting.
    ///
    /// java: `ModelOcelot.field_78163_i`, which starts at `1` and persists across frames.
    pub pose_state: i32,
}

impl OcelotModel {
    /// java: `ModelOcelot()`
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        // java: setTextureOffset("head.main", 0, 0) and friends, resolved by the named
        // `addBox(String, ...)` overload below.
        let head = m
            .part("head", 0, 0)
            .tex(0, 0)
            .cube(-2.5, -2.0, -3.0, 5, 4, 5, 0.0) // head.main
            .tex(0, 24)
            .cube(-1.5, 0.0, -4.0, 3, 2, 2, 0.0) // head.nose
            .tex(0, 10)
            .cube(-2.0, -3.0, 0.0, 1, 1, 2, 0.0) // head.ear1
            .tex(6, 10)
            .cube(1.0, -3.0, 0.0, 1, 1, 2, 0.0) // head.ear2
            .at(0.0, 15.0, -9.0)
            .id();
        let body = m
            .part("body", 20, 0)
            .cube(-2.0, 3.0, -8.0, 4, 16, 6, 0.0)
            .at(0.0, 12.0, -10.0)
            .id();
        let tail = m
            .part("tail", 0, 15)
            .cube(-0.5, 0.0, 0.0, 1, 8, 1, 0.0)
            .angles(0.9, 0.0, 0.0)
            .at(0.0, 15.0, 8.0)
            .id();
        let tail2 = m
            .part("tail2", 4, 15)
            .cube(-0.5, 0.0, 0.0, 1, 8, 1, 0.0)
            .at(0.0, 20.0, 14.0)
            .id();
        let back_left_leg = m
            .part("back_left_leg", 8, 13)
            .cube(-1.0, 0.0, 1.0, 2, 6, 2, 0.0)
            .at(1.1, 18.0, 5.0)
            .id();
        let back_right_leg = m
            .part("back_right_leg", 8, 13)
            .cube(-1.0, 0.0, 1.0, 2, 6, 2, 0.0)
            .at(-1.1, 18.0, 5.0)
            .id();
        let front_left_leg = m
            .part("front_left_leg", 40, 0)
            .cube(-1.0, 0.0, 0.0, 2, 10, 2, 0.0)
            .at(1.2, 13.8, -5.0)
            .id();
        let front_right_leg = m
            .part("front_right_leg", 40, 0)
            .cube(-1.0, 0.0, 0.0, 2, 10, 2, 0.0)
            .at(-1.2, 13.8, -5.0)
            .id();
        Self {
            model: m,
            parts: OcelotParts {
                head,
                body,
                tail,
                tail2,
                back_left_leg,
                back_right_leg,
                front_left_leg,
                front_right_leg,
            },
            pose_state: 1, // java: `int field_78163_i = 1;`
        }
    }
}

impl Default for OcelotModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for OcelotModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelOcelot.setLivingAnimations`
    fn set_living_animations(&mut self, p: &AnimParams) {
        let q = self.parts;
        // java: entityocelot.isSneaking()/isSprinting() -> p.entity.sneaking/sprinting;
        // EntityOcelot.isSitting() -> p.entity.sitting.
        let (sneaking, sprinting, sitting) =
            (p.entity.sneaking, p.entity.sprinting, p.entity.sitting);
        let m = &mut self.model;

        m.get_mut(q.body).rotation_point[1] = 12.0;
        m.get_mut(q.body).rotation_point[2] = -10.0;
        m.get_mut(q.head).rotation_point[1] = 15.0;
        m.get_mut(q.head).rotation_point[2] = -9.0;
        m.get_mut(q.tail).rotation_point[1] = 15.0;
        m.get_mut(q.tail).rotation_point[2] = 8.0;
        m.get_mut(q.tail2).rotation_point[1] = 20.0;
        m.get_mut(q.tail2).rotation_point[2] = 14.0;
        m.get_mut(q.front_left_leg).rotation_point[1] = 13.8;
        m.get_mut(q.front_right_leg).rotation_point[1] = 13.8;
        m.get_mut(q.front_left_leg).rotation_point[2] = -5.0;
        m.get_mut(q.front_right_leg).rotation_point[2] = -5.0;
        m.get_mut(q.back_left_leg).rotation_point[1] = 18.0;
        m.get_mut(q.back_right_leg).rotation_point[1] = 18.0;
        m.get_mut(q.back_left_leg).rotation_point[2] = 5.0;
        m.get_mut(q.back_right_leg).rotation_point[2] = 5.0;
        m.get_mut(q.tail).rotate_angle[0] = 0.9;

        if sneaking {
            m.get_mut(q.body).rotation_point[1] += 1.0;
            m.get_mut(q.head).rotation_point[1] += 2.0;
            m.get_mut(q.tail).rotation_point[1] += 1.0;
            m.get_mut(q.tail2).rotation_point[1] += -4.0;
            m.get_mut(q.tail2).rotation_point[2] += 2.0;
            m.get_mut(q.tail).rotate_angle[0] = PI / 2.0;
            m.get_mut(q.tail2).rotate_angle[0] = PI / 2.0;
            self.pose_state = 0;
        } else if sprinting {
            let tail_y = m.get(q.tail).rotation_point[1];
            m.get_mut(q.tail2).rotation_point[1] = tail_y;
            m.get_mut(q.tail2).rotation_point[2] += 2.0;
            m.get_mut(q.tail).rotate_angle[0] = PI / 2.0;
            m.get_mut(q.tail2).rotate_angle[0] = PI / 2.0;
            self.pose_state = 2;
        } else if sitting {
            m.get_mut(q.body).rotate_angle[0] = PI / 4.0;
            m.get_mut(q.body).rotation_point[1] += -4.0;
            m.get_mut(q.body).rotation_point[2] += 5.0;
            m.get_mut(q.head).rotation_point[1] += -3.3;
            m.get_mut(q.head).rotation_point[2] += 1.0;
            m.get_mut(q.tail).rotation_point[1] += 8.0;
            m.get_mut(q.tail).rotation_point[2] += -2.0;
            m.get_mut(q.tail2).rotation_point[1] += 2.0;
            m.get_mut(q.tail2).rotation_point[2] += -0.8;
            m.get_mut(q.tail).rotate_angle[0] = 1.727_876_1;
            m.get_mut(q.tail2).rotate_angle[0] = 2.670_354;
            m.get_mut(q.front_left_leg).rotate_angle[0] = -0.157_079_64;
            m.get_mut(q.front_right_leg).rotate_angle[0] = -0.157_079_64;
            m.get_mut(q.front_left_leg).rotation_point[1] = 15.8;
            m.get_mut(q.front_right_leg).rotation_point[1] = 15.8;
            m.get_mut(q.front_left_leg).rotation_point[2] = -7.0;
            m.get_mut(q.front_right_leg).rotation_point[2] = -7.0;
            m.get_mut(q.back_left_leg).rotate_angle[0] = -(PI / 2.0);
            m.get_mut(q.back_right_leg).rotate_angle[0] = -(PI / 2.0);
            m.get_mut(q.back_left_leg).rotation_point[1] = 21.0;
            m.get_mut(q.back_right_leg).rotation_point[1] = 21.0;
            m.get_mut(q.back_left_leg).rotation_point[2] = 1.0;
            m.get_mut(q.back_right_leg).rotation_point[2] = 1.0;
            self.pose_state = 3;
        } else {
            self.pose_state = 1;
        }
    }

    /// java: `ModelOcelot.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        let state = self.pose_state;
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        let m = &mut self.model;

        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);

        if state != 3 {
            m.get_mut(q.body).rotate_angle[0] = PI / 2.0;

            if state == 2 {
                m.get_mut(q.back_left_leg).rotate_angle[0] = cos(swing * 0.6662) * 1.0 * amount;
                m.get_mut(q.back_right_leg).rotate_angle[0] =
                    cos(swing * 0.6662 + 0.3) * 1.0 * amount;
                m.get_mut(q.front_left_leg).rotate_angle[0] =
                    cos(swing * 0.6662 + PI + 0.3) * 1.0 * amount;
                m.get_mut(q.front_right_leg).rotate_angle[0] =
                    cos(swing * 0.6662 + PI) * 1.0 * amount;
                m.get_mut(q.tail2).rotate_angle[0] =
                    1.727_876_1 + (PI / 10.0) * cos(swing) * amount;
            } else {
                m.get_mut(q.back_left_leg).rotate_angle[0] = cos(swing * 0.6662) * 1.0 * amount;
                m.get_mut(q.back_right_leg).rotate_angle[0] =
                    cos(swing * 0.6662 + PI) * 1.0 * amount;
                m.get_mut(q.front_left_leg).rotate_angle[0] =
                    cos(swing * 0.6662 + PI) * 1.0 * amount;
                m.get_mut(q.front_right_leg).rotate_angle[0] = cos(swing * 0.6662) * 1.0 * amount;

                if state == 1 {
                    m.get_mut(q.tail2).rotate_angle[0] =
                        1.727_876_1 + (PI / 4.0) * cos(swing) * amount;
                } else {
                    m.get_mut(q.tail2).rotate_angle[0] =
                        1.727_876_1 + 0.471_238_94 * cos(swing) * amount;
                }
            }
        }
    }

    /// java: `ModelOcelot.render`
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let s = p.scale;
        if self.model.is_child {
            // java: scale(1.5F / 2.0F) then translate(0, 10 * scale, 4 * scale)
            let head_m = base * sc(1.5 / 2.0) * tr(0.0, 10.0 * s, 4.0 * s);
            self.model.emit_some(&[q.head], s, head_m, out);
            let body_m = base * sc(1.0 / 2.0) * tr(0.0, 24.0 * s, 0.0);
            self.model.emit_some(
                &[
                    q.body,
                    q.back_left_leg,
                    q.back_right_leg,
                    q.front_left_leg,
                    q.front_right_leg,
                    q.tail,
                    q.tail2,
                ],
                s,
                body_m,
                out,
            );
        } else {
            self.model.emit_some(
                &[
                    q.head,
                    q.body,
                    q.tail,
                    q.tail2,
                    q.back_left_leg,
                    q.back_right_leg,
                    q.front_left_leg,
                    q.front_right_leg,
                ],
                s,
                base,
                out,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ModelRabbit
// ---------------------------------------------------------------------------------------------

/// The twelve parts of the rabbit.
#[derive(Clone, Copy, Debug)]
pub struct RabbitParts {
    /// java: `rabbitLeftFoot`
    pub left_foot: PartId,
    /// java: `rabbitRightFoot`
    pub right_foot: PartId,
    /// java: `rabbitLeftThigh`
    pub left_thigh: PartId,
    /// java: `rabbitRightThigh`
    pub right_thigh: PartId,
    /// java: `rabbitBody`
    pub body: PartId,
    /// java: `rabbitLeftArm`
    pub left_arm: PartId,
    /// java: `rabbitRightArm`
    pub right_arm: PartId,
    /// java: `rabbitHead`
    pub head: PartId,
    /// java: `rabbitRightEar`
    pub right_ear: PartId,
    /// java: `rabbitLeftEar`
    pub left_ear: PartId,
    /// java: `rabbitTail`
    pub tail: PartId,
    /// java: `rabbitNose`
    pub nose: PartId,
}

/// The rabbit.
///
/// java: `ModelRabbit.java`
#[derive(Clone, Debug)]
pub struct RabbitModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the twelve parts.
    pub parts: RabbitParts,
    /// `sin(jumpProgress * PI)`, cached across the pose so every limb reads the same value.
    ///
    /// java: `ModelRabbit.field_178701_m`. (`field_178699_n` is declared next to it in the Java
    /// and never read or written — there is nothing to port.)
    pub jump_bob: f32,
}

impl RabbitModel {
    /// java: `ModelRabbit()`
    ///
    /// The four `setTextureOffset("head.*", ...)` calls at the top of the Java constructor are
    /// dead: `ModelRabbit` never uses the named `addBox(String, ...)` overload that would read
    /// them back (they are copy-paste from `ModelOcelot`). Nothing to reproduce.
    ///
    /// Every part sets `mirror = true` *after* its `addBox`, so no box is actually mirrored —
    /// the flag only ever affects boxes added later, and none are.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let left_foot = m
            .part("left_foot", 26, 24)
            .cube(-1.0, 5.5, -3.7, 2, 1, 7, 0.0)
            .at(3.0, 17.5, 3.7)
            .mirrored()
            .angles(0.0, 0.0, 0.0)
            .id();
        let right_foot = m
            .part("right_foot", 8, 24)
            .cube(-1.0, 5.5, -3.7, 2, 1, 7, 0.0)
            .at(-3.0, 17.5, 3.7)
            .mirrored()
            .angles(0.0, 0.0, 0.0)
            .id();
        let left_thigh = m
            .part("left_thigh", 30, 15)
            .cube(-1.0, 0.0, 0.0, 2, 4, 5, 0.0)
            .at(3.0, 17.5, 3.7)
            .mirrored()
            .angles(-0.349_065_84, 0.0, 0.0)
            .id();
        let right_thigh = m
            .part("right_thigh", 16, 15)
            .cube(-1.0, 0.0, 0.0, 2, 4, 5, 0.0)
            .at(-3.0, 17.5, 3.7)
            .mirrored()
            .angles(-0.349_065_84, 0.0, 0.0)
            .id();
        let body = m
            .part("body", 0, 0)
            .cube(-3.0, -2.0, -10.0, 6, 5, 10, 0.0)
            .at(0.0, 19.0, 8.0)
            .mirrored()
            .angles(-0.349_065_84, 0.0, 0.0)
            .id();
        let left_arm = m
            .part("left_arm", 8, 15)
            .cube(-1.0, 0.0, -1.0, 2, 7, 2, 0.0)
            .at(3.0, 17.0, -1.0)
            .mirrored()
            .angles(-0.174_532_92, 0.0, 0.0)
            .id();
        let right_arm = m
            .part("right_arm", 0, 15)
            .cube(-1.0, 0.0, -1.0, 2, 7, 2, 0.0)
            .at(-3.0, 17.0, -1.0)
            .mirrored()
            .angles(-0.174_532_92, 0.0, 0.0)
            .id();
        let head = m
            .part("head", 32, 0)
            .cube(-2.5, -4.0, -5.0, 5, 4, 5, 0.0)
            .at(0.0, 16.0, -1.0)
            .mirrored()
            .angles(0.0, 0.0, 0.0)
            .id();
        let right_ear = m
            .part("right_ear", 52, 0)
            .cube(-2.5, -9.0, -1.0, 2, 5, 1, 0.0)
            .at(0.0, 16.0, -1.0)
            .mirrored()
            .angles(0.0, -0.261_799_4, 0.0)
            .id();
        let left_ear = m
            .part("left_ear", 58, 0)
            .cube(0.5, -9.0, -1.0, 2, 5, 1, 0.0)
            .at(0.0, 16.0, -1.0)
            .mirrored()
            .angles(0.0, 0.261_799_4, 0.0)
            .id();
        let tail = m
            .part("tail", 52, 6)
            .cube(-1.5, -1.5, 0.0, 3, 3, 2, 0.0)
            .at(0.0, 20.0, 7.0)
            .mirrored()
            .angles(-0.349_065_9, 0.0, 0.0)
            .id();
        let nose = m
            .part("nose", 32, 9)
            .cube(-0.5, -2.5, -5.5, 1, 1, 1, 0.0)
            .at(0.0, 16.0, -1.0)
            .mirrored()
            .angles(0.0, 0.0, 0.0)
            .id();
        Self {
            model: m,
            parts: RabbitParts {
                left_foot,
                right_foot,
                left_thigh,
                right_thigh,
                body,
                left_arm,
                right_arm,
                head,
                right_ear,
                left_ear,
                tail,
                nose,
            },
            jump_bob: 0.0,
        }
    }
}

impl Default for RabbitModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for RabbitModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelRabbit.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        // java: float f = ageInTicks - (float)entityIn.ticksExisted — the partial-tick remainder,
        // whose only use is as the argument of `EntityRabbit.func_175521_o(f)`. That whole call
        // (the jump progress) arrives pre-computed as `p.entity.tail_rotation`, so `f` has no
        // further consumer here; it is still spelled out to keep the statement visible.
        let _f = p.age_in_ticks - p.entity.ticks_existed as f32;
        // java: entityrabbit.func_175521_o(f) -> p.entity.tail_rotation
        let jump = p.entity.tail_rotation;

        let head_x = p.head_pitch * DEG_TO_RAD_LITERAL;
        let head_y = p.net_head_yaw * DEG_TO_RAD_LITERAL;
        let m = &mut self.model;
        m.get_mut(q.nose).rotate_angle[0] = head_x;
        m.get_mut(q.head).rotate_angle[0] = head_x;
        m.get_mut(q.right_ear).rotate_angle[0] = head_x;
        m.get_mut(q.left_ear).rotate_angle[0] = head_x;
        m.get_mut(q.nose).rotate_angle[1] = head_y;
        m.get_mut(q.head).rotate_angle[1] = head_y;
        m.get_mut(q.right_ear).rotate_angle[1] = head_y - 0.261_799_4;
        m.get_mut(q.left_ear).rotate_angle[1] = head_y + 0.261_799_4;

        self.jump_bob = sin(jump * PI);
        let bob = self.jump_bob;
        let m = &mut self.model;
        let thigh = (bob * 50.0 - 21.0) * DEG_TO_RAD_LITERAL;
        m.get_mut(q.left_thigh).rotate_angle[0] = thigh;
        m.get_mut(q.right_thigh).rotate_angle[0] = thigh;
        let foot = bob * 50.0 * DEG_TO_RAD_LITERAL;
        m.get_mut(q.left_foot).rotate_angle[0] = foot;
        m.get_mut(q.right_foot).rotate_angle[0] = foot;
        let arm = (bob * -40.0 - 11.0) * DEG_TO_RAD_LITERAL;
        m.get_mut(q.left_arm).rotate_angle[0] = arm;
        m.get_mut(q.right_arm).rotate_angle[0] = arm;
    }

    /// java: `ModelRabbit.render`
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let s = p.scale;
        if self.model.is_child {
            let head_m = base * tr(0.0, 5.0 * s, 2.0 * s);
            self.model
                .emit_some(&[q.head, q.left_ear, q.right_ear, q.nose], s, head_m, out);
            let body_m = base * sc(1.0 / 2.0) * tr(0.0, 24.0 * s, 0.0);
            self.model.emit_some(
                &[
                    q.left_foot,
                    q.right_foot,
                    q.left_thigh,
                    q.right_thigh,
                    q.body,
                    q.left_arm,
                    q.right_arm,
                    q.tail,
                ],
                s,
                body_m,
                out,
            );
        } else {
            self.model.emit_some(
                &[
                    q.left_foot,
                    q.right_foot,
                    q.left_thigh,
                    q.right_thigh,
                    q.body,
                    q.left_arm,
                    q.right_arm,
                    q.head,
                    q.right_ear,
                    q.left_ear,
                    q.tail,
                    q.nose,
                ],
                s,
                base,
                out,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ModelSquid
// ---------------------------------------------------------------------------------------------

/// The squid.
///
/// java: `ModelSquid.java`
#[derive(Clone, Debug)]
pub struct SquidModel {
    /// The part tree.
    pub model: Model,
    /// java: `squidBody`
    pub body: PartId,
    /// java: `squidTentacles`, in construction order.
    pub tentacles: [PartId; 8],
}

impl SquidModel {
    /// java: `ModelSquid()`
    ///
    /// The tentacle ring uses `java.lang.Math.cos`/`sin` on a `double`, **not** `MathHelper`'s
    /// LUT — the Java is explicit about it — so this is one of the few places `f64` libm trig is
    /// the faithful choice.
    #[must_use]
    pub fn new() -> Self {
        // java: int i = -16;
        let i = -16.0_f32;
        let mut m = Model::new(64, 32);
        let body = m
            .part("body", 0, 0)
            .cube(-6.0, -8.0, -6.0, 12, 16, 12, 0.0)
            .id();
        // java: this.squidBody.rotationPointY += (float)(24 + i);
        m.get_mut(body).rotation_point[1] += 24.0 + i;

        let n = 8_usize;
        let mut tentacles = [body; 8];
        for (j, slot) in tentacles.iter_mut().enumerate() {
            let id = m
                .part("tentacle", 48, 0)
                .cube(-1.0, 0.0, -1.0, 2, 18, 2, 0.0)
                .id();
            let d0 = j as f64 * core::f64::consts::PI * 2.0 / n as f64;
            let f = d0.cos() as f32 * 5.0;
            let f1 = d0.sin() as f32 * 5.0;
            let part = m.get_mut(id);
            part.rotation_point[0] = f;
            part.rotation_point[2] = f1;
            part.rotation_point[1] = 31.0 + i;
            let d0 =
                j as f64 * core::f64::consts::PI * -2.0 / n as f64 + core::f64::consts::FRAC_PI_2;
            part.rotate_angle[1] = d0 as f32;
            *slot = id;
        }

        Self {
            model: m,
            body,
            tentacles,
        }
    }
}

impl Default for SquidModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SquidModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelSquid.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        for id in self.tentacles {
            self.model.get_mut(id).rotate_angle[0] = p.age_in_ticks;
        }
    }

    /// java: `ModelSquid.render` — body, then the eight tentacles in order, which is exactly the
    /// root order, so the default `emit_all` would do; spelled out to keep the draw list visible.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_part(self.body, p.scale, base, out);
        for id in self.tentacles {
            self.model.emit_part(id, p.scale, base, out);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// ModelBat
// ---------------------------------------------------------------------------------------------

/// The bat's parts. The two ears and the four wing segments are children, not roots.
#[derive(Clone, Copy, Debug)]
pub struct BatParts {
    /// java: `batHead`
    pub head: PartId,
    /// java: the first anonymous `modelrenderer` — the right ear, a child of `batHead`.
    pub right_ear: PartId,
    /// java: the second anonymous `modelrenderer1` — the left ear, a child of `batHead`.
    pub left_ear: PartId,
    /// java: `batBody`
    pub body: PartId,
    /// java: `batRightWing`, a child of `batBody`.
    pub right_wing: PartId,
    /// java: `batLeftWing`, a child of `batBody`.
    pub left_wing: PartId,
    /// java: `batOuterRightWing`, a child of `batRightWing`.
    pub outer_right_wing: PartId,
    /// java: `batOuterLeftWing`, a child of `batLeftWing`.
    pub outer_left_wing: PartId,
}

/// The bat.
///
/// java: `ModelBat.java`
#[derive(Clone, Debug)]
pub struct BatModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the eight parts.
    pub parts: BatParts,
}

impl BatModel {
    /// java: `ModelBat()` — note it raises `textureHeight` to 64 before building anything.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 64);
        let head = m
            .part("head", 0, 0)
            .cube(-3.0, -3.0, -3.0, 6, 6, 6, 0.0)
            .id();
        let right_ear = m
            .child(head, "right_ear", 24, 0)
            .cube(-4.0, -6.0, -2.0, 3, 4, 1, 0.0)
            .id();
        let left_ear = m
            .child(head, "left_ear", 24, 0)
            .mirrored()
            .cube(1.0, -6.0, -2.0, 3, 4, 1, 0.0)
            .id();
        let body = m
            .part("body", 0, 16)
            .cube(-3.0, 4.0, -3.0, 6, 12, 6, 0.0)
            .tex(0, 34)
            .cube(-5.0, 16.0, 0.0, 10, 6, 1, 0.0)
            .id();
        // java: the four wing parts are created as bare `ModelRenderer`s and only parented by the
        // `addChild` block at the end. The resulting tree and draw order are the same either way.
        let right_wing = m
            .child(body, "right_wing", 42, 0)
            .cube(-12.0, 1.0, 1.5, 10, 16, 1, 0.0)
            .id();
        let outer_right_wing = m
            .child(right_wing, "outer_right_wing", 24, 16)
            .at(-12.0, 1.0, 1.5)
            .cube(-8.0, 1.0, 0.0, 8, 12, 1, 0.0)
            .id();
        let left_wing = m
            .child(body, "left_wing", 42, 0)
            .mirrored()
            .cube(2.0, 1.0, 1.5, 10, 16, 1, 0.0)
            .id();
        let outer_left_wing = m
            .child(left_wing, "outer_left_wing", 24, 16)
            .mirrored()
            .at(12.0, 1.0, 1.5)
            .cube(0.0, 1.0, 0.0, 8, 12, 1, 0.0)
            .id();
        Self {
            model: m,
            parts: BatParts {
                head,
                right_ear,
                left_ear,
                body,
                right_wing,
                left_wing,
                outer_right_wing,
                outer_left_wing,
            },
        }
    }
}

impl Default for BatModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for BatModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelBat.setRotationAngles`
    ///
    /// The two `float f = (180F / (float)Math.PI)` locals are dead in both branches of the Java
    /// and have no port.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        // java: ((EntityBat)entityIn).getIsBatHanging() — the roosting flag, DataWatcher index
        // 16 bit 0 on `EntityBat`. This used to read `p.entity.sitting`, which is
        // `EntityTameable.isSitting` — a *different* index on a class hierarchy a bat is not in,
        // and one `decode_entity_state` only ever writes for the wolf and the ocelot. Every bat
        // therefore flew, always. Now on its own slot.
        let hanging = p.entity.bat_hanging;
        let m = &mut self.model;

        if hanging {
            m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
            m.get_mut(q.head).rotate_angle[1] = PI - model_radians(p.net_head_yaw);
            m.get_mut(q.head).rotate_angle[2] = PI;
            m.get_mut(q.head).set_rotation_point(0.0, -2.0, 0.0);
            m.get_mut(q.right_wing).set_rotation_point(-3.0, 0.0, 3.0);
            m.get_mut(q.left_wing).set_rotation_point(3.0, 0.0, 3.0);
            m.get_mut(q.body).rotate_angle[0] = PI;
            m.get_mut(q.right_wing).rotate_angle[0] = -0.157_079_64;
            m.get_mut(q.right_wing).rotate_angle[1] = -(PI * 2.0 / 5.0);
            m.get_mut(q.outer_right_wing).rotate_angle[1] = -1.727_876_1;
            let rw_x = m.get(q.right_wing).rotate_angle[0];
            let rw_y = m.get(q.right_wing).rotate_angle[1];
            m.get_mut(q.left_wing).rotate_angle[0] = rw_x;
            m.get_mut(q.left_wing).rotate_angle[1] = -rw_y;
            let orw_y = m.get(q.outer_right_wing).rotate_angle[1];
            m.get_mut(q.outer_left_wing).rotate_angle[1] = -orw_y;
        } else {
            m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
            m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
            m.get_mut(q.head).rotate_angle[2] = 0.0;
            m.get_mut(q.head).set_rotation_point(0.0, 0.0, 0.0);
            m.get_mut(q.right_wing).set_rotation_point(0.0, 0.0, 0.0);
            m.get_mut(q.left_wing).set_rotation_point(0.0, 0.0, 0.0);
            m.get_mut(q.body).rotate_angle[0] = PI / 4.0 + cos(p.age_in_ticks * 0.1) * 0.15;
            m.get_mut(q.body).rotate_angle[1] = 0.0;
            m.get_mut(q.right_wing).rotate_angle[1] = cos(p.age_in_ticks * 1.3) * PI * 0.25;
            let rw_y = m.get(q.right_wing).rotate_angle[1];
            m.get_mut(q.left_wing).rotate_angle[1] = -rw_y;
            m.get_mut(q.outer_right_wing).rotate_angle[1] = rw_y * 0.5;
            m.get_mut(q.outer_left_wing).rotate_angle[1] = -rw_y * 0.5;
        }
    }

    /// java: `ModelBat.render` — head subtree, then body subtree, which is the root order.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model
            .emit_some(&[self.parts.head, self.parts.body], p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelSnowMan
// ---------------------------------------------------------------------------------------------

/// The five parts of the snow golem.
#[derive(Clone, Copy, Debug)]
pub struct SnowManParts {
    /// java: `head`
    pub head: PartId,
    /// java: `rightHand`
    pub right_hand: PartId,
    /// java: `leftHand`
    pub left_hand: PartId,
    /// java: `body` — the middle snowball
    pub body: PartId,
    /// java: `bottomBody`
    pub bottom_body: PartId,
}

/// The snow golem.
///
/// java: `ModelSnowMan.java`
#[derive(Clone, Debug)]
pub struct SnowManModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the five parts.
    pub parts: SnowManParts,
}

impl SnowManModel {
    /// java: `ModelSnowMan()`
    #[must_use]
    pub fn new() -> Self {
        // java: float f = 4.0F; float f1 = 0.0F;
        let f = 4.0_f32;
        let f1 = 0.0_f32;
        // java: `ModelBase.textureWidth/Height` stay at 64x32; every part calls
        // `setTextureSize(64, 64)` for itself.
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .tex_size(64, 64)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, f1 - 0.5)
            .at(0.0, 0.0 + f, 0.0)
            .id();
        let right_hand = m
            .part("right_hand", 32, 0)
            .tex_size(64, 64)
            .cube(-1.0, 0.0, -1.0, 12, 2, 2, f1 - 0.5)
            .at(0.0, 0.0 + f + 9.0 - 7.0, 0.0)
            .id();
        let left_hand = m
            .part("left_hand", 32, 0)
            .tex_size(64, 64)
            .cube(-1.0, 0.0, -1.0, 12, 2, 2, f1 - 0.5)
            .at(0.0, 0.0 + f + 9.0 - 7.0, 0.0)
            .id();
        let body = m
            .part("body", 0, 16)
            .tex_size(64, 64)
            .cube(-5.0, -10.0, -5.0, 10, 10, 10, f1 - 0.5)
            .at(0.0, 0.0 + f + 9.0, 0.0)
            .id();
        let bottom_body = m
            .part("bottom_body", 0, 36)
            .tex_size(64, 64)
            .cube(-6.0, -12.0, -6.0, 12, 12, 12, f1 - 0.5)
            .at(0.0, 0.0 + f + 20.0, 0.0)
            .id();
        Self {
            model: m,
            parts: SnowManParts {
                head,
                right_hand,
                left_hand,
                body,
                bottom_body,
            },
        }
    }
}

impl Default for SnowManModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SnowManModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelSnowMan.setRotationAngles` (its `super` call is `ModelBase`'s empty body).
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(q.body).rotate_angle[1] = model_radians(p.net_head_yaw) * 0.25;
        let body_y = m.get(q.body).rotate_angle[1];
        let f = sin(body_y);
        let f1 = cos(body_y);
        m.get_mut(q.right_hand).rotate_angle[2] = 1.0;
        m.get_mut(q.left_hand).rotate_angle[2] = -1.0;
        m.get_mut(q.right_hand).rotate_angle[1] = 0.0 + body_y;
        m.get_mut(q.left_hand).rotate_angle[1] = PI + body_y;
        m.get_mut(q.right_hand).rotation_point[0] = f1 * 5.0;
        m.get_mut(q.right_hand).rotation_point[2] = -f * 5.0;
        m.get_mut(q.left_hand).rotation_point[0] = -f1 * 5.0;
        m.get_mut(q.left_hand).rotation_point[2] = f * 5.0;
    }

    /// java: `ModelSnowMan.render` — body, bottomBody, head, rightHand, leftHand, which is *not*
    /// the construction order, so the draw list has to be spelled out.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        self.model.emit_some(
            &[q.body, q.bottom_body, q.head, q.right_hand, q.left_hand],
            p.scale,
            base,
            out,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// ModelIronGolem
// ---------------------------------------------------------------------------------------------

/// java: `ModelIronGolem.func_78172_a` — a normalised triangle wave in `[-1, 1]` with period
/// `p_78172_2_`, used for the golem's stomping gait and its attack swing.
fn golem_wave(t: f32, period: f32) -> f32 {
    ((t % period - period * 0.5).abs() - period * 0.25) / (period * 0.25)
}

/// The six parts of the iron golem.
#[derive(Clone, Copy, Debug)]
pub struct IronGolemParts {
    /// java: `ironGolemHead`
    pub head: PartId,
    /// java: `ironGolemBody`
    pub body: PartId,
    /// java: `ironGolemRightArm`
    pub right_arm: PartId,
    /// java: `ironGolemLeftArm`
    pub left_arm: PartId,
    /// java: `ironGolemLeftLeg`
    pub left_leg: PartId,
    /// java: `ironGolemRightLeg`
    pub right_leg: PartId,
}

/// The iron golem.
///
/// java: `ModelIronGolem.java`
#[derive(Clone, Debug)]
pub struct IronGolemModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the six parts.
    pub parts: IronGolemParts,
}

impl IronGolemModel {
    /// java: `ModelIronGolem()` — `this(0.0F)`, which is `this(0.0F, -7.0F)`.
    #[must_use]
    pub fn new() -> Self {
        Self::with(0.0, -7.0)
    }

    /// java: `ModelIronGolem(float modelSize, float yOffset)`
    #[must_use]
    pub fn with(size: f32, y_off: f32) -> Self {
        // java: int i = 128; int j = 128; — the *part* texture size. `ModelBase`'s own
        // textureWidth/Height are left at the 64x32 default.
        let (i, j) = (128, 128);
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .tex_size(i, j)
            .at(0.0, 0.0 + y_off, -2.0)
            .tex(0, 0)
            .cube(-4.0, -12.0, -5.5, 8, 10, 8, size)
            .tex(24, 0)
            .cube(-1.0, -5.0, -7.5, 2, 4, 2, size)
            .id();
        let body = m
            .part("body", 0, 0)
            .tex_size(i, j)
            .at(0.0, 0.0 + y_off, 0.0)
            .tex(0, 40)
            .cube(-9.0, -2.0, -6.0, 18, 12, 11, size)
            .tex(0, 70)
            .cube(-4.5, 10.0, -3.0, 9, 5, 6, size + 0.5)
            .id();
        let right_arm = m
            .part("right_arm", 0, 0)
            .tex_size(i, j)
            .at(0.0, -7.0, 0.0)
            .tex(60, 21)
            .cube(-13.0, -2.5, -3.0, 4, 30, 6, size)
            .id();
        let left_arm = m
            .part("left_arm", 0, 0)
            .tex_size(i, j)
            .at(0.0, -7.0, 0.0)
            .tex(60, 58)
            .cube(9.0, -2.5, -3.0, 4, 30, 6, size)
            .id();
        let left_leg = m
            .part("left_leg", 0, 22)
            .tex_size(i, j)
            .at(-4.0, 18.0 + y_off, 0.0)
            .tex(37, 0)
            .cube(-3.5, -3.0, -3.0, 6, 16, 5, size)
            .id();
        let right_leg = m
            .part("right_leg", 0, 22)
            .tex_size(i, j)
            .mirrored()
            .tex(60, 0)
            .at(5.0, 18.0 + y_off, 0.0)
            .cube(-3.5, -3.0, -3.0, 6, 16, 5, size)
            .id();
        Self {
            model: m,
            parts: IronGolemParts {
                head,
                body,
                right_arm,
                left_arm,
                left_leg,
                right_leg,
            },
        }
    }
}

impl Default for IronGolemModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for IronGolemModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelIronGolem.setLivingAnimations`
    fn set_living_animations(&mut self, p: &AnimParams) {
        let q = self.parts;
        // java: `int j = entityirongolem.getHoldRoseTick();`, read straight off the entity.
        //
        // This used to be a `pub hold_rose_ticks` field on this struct, which is the
        // shared-model defect class: `RenderManager` keeps exactly **one** `IronGolemModel` for
        // every golem in the world, so a per-entity value parked on it is "whichever golem was
        // posed last". Worse, nothing ever wrote it — the poppy *layer* was wired to
        // `EntityAnimState::hold_rose_ticks` in wave 1 but the *arm pose* still read this dead
        // field, so a golem offering a flower held it out while its arms did the walking swing.
        let rose = p.entity.hold_rose_ticks as f32;
        // java: entityirongolem.getAttackTimer() -> p.entity.attack_timer
        let i = p.entity.attack_timer;
        let m = &mut self.model;

        if i > 0.0 {
            let a = -2.0 + 1.5 * golem_wave(i - p.partial_ticks, 10.0);
            m.get_mut(q.right_arm).rotate_angle[0] = a;
            m.get_mut(q.left_arm).rotate_angle[0] = a;
        } else if rose > 0.0 {
            m.get_mut(q.right_arm).rotate_angle[0] = -0.8 + 0.025 * golem_wave(rose, 70.0);
            m.get_mut(q.left_arm).rotate_angle[0] = 0.0;
        } else {
            let w = golem_wave(p.limb_swing, 13.0);
            m.get_mut(q.right_arm).rotate_angle[0] = (-0.2 + 1.5 * w) * p.limb_swing_amount;
            m.get_mut(q.left_arm).rotate_angle[0] = (-0.2 - 1.5 * w) * p.limb_swing_amount;
        }
    }

    /// java: `ModelIronGolem.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        let w = golem_wave(p.limb_swing, 13.0);
        m.get_mut(q.left_leg).rotate_angle[0] = -1.5 * w * p.limb_swing_amount;
        m.get_mut(q.right_leg).rotate_angle[0] = 1.5 * w * p.limb_swing_amount;
        m.get_mut(q.left_leg).rotate_angle[1] = 0.0;
        m.get_mut(q.right_leg).rotate_angle[1] = 0.0;
    }

    /// java: `ModelIronGolem.render` — head, body, legs, then arms, which is not the construction
    /// order.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        self.model.emit_some(
            &[
                q.head,
                q.body,
                q.left_leg,
                q.right_leg,
                q.right_arm,
                q.left_arm,
            ],
            p.scale,
            base,
            out,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// ModelHorse
// ---------------------------------------------------------------------------------------------

/// The thirty-nine parts in `ModelHorse`, in the constructor's `boxList` order.
///
/// java: `ModelHorse`'s private `ModelRenderer` fields.  The two jaw parts are children of
/// [`HorseParts::head`]; every other part is a root, just as in the Java model.  Keeping the
/// handles public follows the other concrete models in this module and gives the CPU geometry
/// tests a way to inspect the unusually large 128x128 horse sheet without a renderer or a GPU.
#[derive(Clone, Copy, Debug)]
pub struct HorseParts {
    /// java: `body`.
    pub body: PartId,
    /// java: `tailBase`.
    pub tail_base: PartId,
    /// java: `tailMiddle`.
    pub tail_middle: PartId,
    /// java: `tailTip`.
    pub tail_tip: PartId,
    /// java: `backLeftLeg`.
    pub back_left_leg: PartId,
    /// java: `backLeftShin`.
    pub back_left_shin: PartId,
    /// java: `backLeftHoof`.
    pub back_left_hoof: PartId,
    /// java: `backRightLeg`.
    pub back_right_leg: PartId,
    /// java: `backRightShin`.
    pub back_right_shin: PartId,
    /// java: `backRightHoof`.
    pub back_right_hoof: PartId,
    /// java: `frontLeftLeg`.
    pub front_left_leg: PartId,
    /// java: `frontLeftShin`.
    pub front_left_shin: PartId,
    /// java: `frontLeftHoof`.
    pub front_left_hoof: PartId,
    /// java: `frontRightLeg`.
    pub front_right_leg: PartId,
    /// java: `frontRightShin`.
    pub front_right_shin: PartId,
    /// java: `frontRightHoof`.
    pub front_right_hoof: PartId,
    /// java: `head`.
    pub head: PartId,
    /// java: `field_178711_b`, the upper jaw child.
    pub upper_jaw: PartId,
    /// java: `field_178712_c`, the lower jaw child.
    pub lower_jaw: PartId,
    /// java: `horseLeftEar`.
    pub horse_left_ear: PartId,
    /// java: `horseRightEar`.
    pub horse_right_ear: PartId,
    /// java: `muleLeftEar`.
    pub mule_left_ear: PartId,
    /// java: `muleRightEar`.
    pub mule_right_ear: PartId,
    /// java: `neck`.
    pub neck: PartId,
    /// java: `muleLeftChest`.
    pub mule_left_chest: PartId,
    /// java: `muleRightChest`.
    pub mule_right_chest: PartId,
    /// java: `horseSaddleBottom`.
    pub saddle_bottom: PartId,
    /// java: `horseSaddleFront`.
    pub saddle_front: PartId,
    /// java: `horseSaddleBack`.
    pub saddle_back: PartId,
    /// java: `horseLeftSaddleMetal`.
    pub left_saddle_metal: PartId,
    /// java: `horseLeftSaddleRope`.
    pub left_saddle_rope: PartId,
    /// java: `horseRightSaddleMetal`.
    pub right_saddle_metal: PartId,
    /// java: `horseRightSaddleRope`.
    pub right_saddle_rope: PartId,
    /// java: `horseLeftFaceMetal`.
    pub left_face_metal: PartId,
    /// java: `horseRightFaceMetal`.
    pub right_face_metal: PartId,
    /// java: `horseLeftRein`.
    pub left_rein: PartId,
    /// java: `horseRightRein`.
    pub right_rein: PartId,
    /// java: `mane`.
    pub mane: PartId,
    /// java: `horseFaceRopes`.
    pub face_ropes: PartId,
}

/// Client-local horse animation springs.
///
/// java: `EntityHorse.onUpdate`'s `headLean`, `rearingAmount` and `mouthOpenness` fields.  The
/// watched booleans are the only network inputs; the three floats are deliberately kept here as
/// client state and sampled by [`HorseRenderState`] at render time.  This keeps the springs out of
/// the shared [`HorseModel`], which vanilla also reuses for every horse.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HorseAnim {
    /// java: `headLean`.
    pub head_lean: f32,
    /// java: `prevHeadLean`.
    pub prev_head_lean: f32,
    /// java: `rearingAmount`.
    pub rearing_amount: f32,
    /// java: `prevRearingAmount`.
    pub prev_rearing_amount: f32,
    /// java: `mouthOpenness`.
    pub mouth_openness: f32,
    /// java: `prevMouthOpenness`.
    pub prev_mouth_openness: f32,
    /// java: `field_110278_bp != 0`, the short client-local tail swish window.
    pub tail_swishing: bool,
}

impl HorseAnim {
    /// Advances one client tick from the three already-decoded horse watcher booleans.
    ///
    /// java: `EntityHorse.onUpdate` (`headLean`, `rearingAmount`, `mouthOpenness`).  The order is
    /// intentional: rearing clears both head-lean samples after the grazing spring advances.
    pub fn tick(&mut self, eating_haystack: bool, rearing: bool, mouth_open: bool) {
        self.prev_head_lean = self.head_lean;
        if eating_haystack {
            self.head_lean += (1.0 - self.head_lean) * 0.4 + 0.05;
            if self.head_lean > 1.0 {
                self.head_lean = 1.0;
            }
        } else {
            self.head_lean += (0.0 - self.head_lean) * 0.4 - 0.05;
            if self.head_lean < 0.0 {
                self.head_lean = 0.0;
            }
        }

        self.prev_rearing_amount = self.rearing_amount;
        if rearing {
            self.prev_head_lean = 0.0;
            self.head_lean = 0.0;
            self.rearing_amount += (1.0 - self.rearing_amount) * 0.4 + 0.05;
            if self.rearing_amount > 1.0 {
                self.rearing_amount = 1.0;
            }
        } else {
            self.rearing_amount +=
                (0.8 * self.rearing_amount * self.rearing_amount * self.rearing_amount
                    - self.rearing_amount)
                    * 0.6
                    - 0.05;
            if self.rearing_amount < 0.0 {
                self.rearing_amount = 0.0;
            }
        }

        self.prev_mouth_openness = self.mouth_openness;
        if mouth_open {
            self.mouth_openness += (1.0 - self.mouth_openness) * 0.7 + 0.05;
            if self.mouth_openness > 1.0 {
                self.mouth_openness = 1.0;
            }
        } else {
            self.mouth_openness += (0.0 - self.mouth_openness) * 0.7 - 0.05;
            if self.mouth_openness < 0.0 {
                self.mouth_openness = 0.0;
            }
        }
    }

    /// The client-side seat displacement used by `EntityHorse.updateRiderPosition`.
    ///
    /// java: `EntityHorse` reads `prevRearingAmount` (not the current spring sample), then
    /// evaluates the body yaw and the `0.7F`/`0.15F` offsets as floats. Keeping this beside the
    /// spring state lets remote and session-owned riders share one exact interpolation edge.
    #[must_use]
    pub fn rider_offset(&self, body_yaw: f32) -> [f32; 3] {
        let rearing = self.prev_rearing_amount;
        if rearing <= 0.0 {
            return [0.0; 3];
        }
        let yaw = body_yaw * core::f32::consts::PI / 180.0_f32;
        let forward = 0.7_f32 * rearing;
        [forward * sin(yaw), 0.15_f32 * rearing, -forward * cos(yaw)]
    }

    /// Converts tick samples into the interpolated state `ModelHorse` consumes for one frame.
    #[must_use]
    pub fn render_state(
        &self,
        partial: f32,
        horse_type: i32,
        adult: bool,
        saddled: bool,
        chested: bool,
        ridden: bool,
    ) -> HorseRenderState {
        HorseRenderState {
            horse_type,
            adult,
            saddled,
            chested,
            ridden,
            tail_swishing: self.tail_swishing,
            grass_eating_amount: self.prev_head_lean
                + (self.head_lean - self.prev_head_lean) * partial,
            rearing_amount: self.prev_rearing_amount
                + (self.rearing_amount - self.prev_rearing_amount) * partial,
            mouth_openness: self.prev_mouth_openness
                + (self.mouth_openness - self.prev_mouth_openness) * partial,
        }
    }
}

/// Horse facts that `ModelHorse.setLivingAnimations` and `ModelHorse.render` read from the
/// entity.  The booleans/variant are watched state; the three floats are the client-local,
/// interpolated springs carried by [`HorseAnim`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HorseRenderState {
    /// java: `EntityHorse.getHorseType()`.
    pub horse_type: i32,
    /// java: `EntityHorse.isAdultHorse()`.
    pub adult: bool,
    /// java: `EntityHorse.isHorseSaddled()`.
    pub saddled: bool,
    /// java: `EntityHorse.isChested()`.
    pub chested: bool,
    /// java: `EntityHorse.riddenByEntity != null`.
    pub ridden: bool,
    /// java: `field_110278_bp != 0`.
    pub tail_swishing: bool,
    /// java: `getGrassEatingAmount(partialTicks)`.
    pub grass_eating_amount: f32,
    /// java: `getRearingAmount(partialTicks)`.
    pub rearing_amount: f32,
    /// java: `getMouthOpennessAngle(partialTicks)`.
    pub mouth_openness: f32,
}

impl HorseRenderState {
    /// Neutral state for the generic `EntityModel` contract, which has no horse watcher payload.
    #[must_use]
    fn neutral_for(p: &AnimParams) -> Self {
        Self {
            horse_type: p.entity.variant,
            adult: !p.entity.child,
            saddled: false,
            chested: false,
            ridden: false,
            tail_swishing: false,
            grass_eating_amount: 0.0,
            rearing_amount: 0.0,
            mouth_openness: 0.0,
        }
    }
}

impl Default for HorseRenderState {
    fn default() -> Self {
        Self {
            horse_type: 0,
            adult: true,
            saddled: false,
            chested: false,
            ridden: false,
            tail_swishing: false,
            grass_eating_amount: 0.0,
            rearing_amount: 0.0,
            mouth_openness: 0.0,
        }
    }
}

/// The horse model.
///
/// java: `ModelHorse.java`.
///
/// The generic [`EntityModel`] path remains a neutral unsaddled horse because its shared
/// [`AnimParams`] contract has no horse watcher payload.  [`HorseModel::pose_and_render_with_state`]
/// is the explicit horse seam: it carries the already-sampled client springs and watched flags,
/// turns on saddle/rein/chest parts, and keeps the same cached 39-part model.
#[derive(Clone, Debug)]
pub struct HorseModel {
    /// The 128x128 horse part forest.
    pub model: Model,
    /// Handles for the constructor's 39 parts.
    pub parts: HorseParts,
}

impl HorseModel {
    /// java: `ModelHorse()` — a 128x128 sheet with no constructor scale.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(128, 128);

        let body = m
            .part("body", 0, 34)
            .cube(-5.0, -8.0, -19.0, 10, 10, 24, 0.0)
            .at(0.0, 11.0, 9.0)
            .id();
        let tail_base = m
            .part("tail_base", 44, 0)
            .cube(-1.0, -1.0, 0.0, 2, 2, 3, 0.0)
            .at(0.0, 3.0, 14.0)
            .angles(-1.134464, 0.0, 0.0)
            .id();
        let tail_middle = m
            .part("tail_middle", 38, 7)
            .cube(-1.5, -2.0, 3.0, 3, 4, 7, 0.0)
            .at(0.0, 3.0, 14.0)
            .angles(-1.134464, 0.0, 0.0)
            .id();
        let tail_tip = m
            .part("tail_tip", 24, 3)
            .cube(-1.5, -4.5, 9.0, 3, 4, 7, 0.0)
            .at(0.0, 3.0, 14.0)
            .angles(-1.40215, 0.0, 0.0)
            .id();

        let back_left_leg = m
            .part("back_left_leg", 78, 29)
            .cube(-2.5, -2.0, -2.5, 4, 9, 5, 0.0)
            .at(4.0, 9.0, 11.0)
            .id();
        let back_left_shin = m
            .part("back_left_shin", 78, 43)
            .cube(-2.0, 0.0, -1.5, 3, 5, 3, 0.0)
            .at(4.0, 16.0, 11.0)
            .id();
        let back_left_hoof = m
            .part("back_left_hoof", 78, 51)
            .cube(-2.5, 5.1, -2.0, 4, 3, 4, 0.0)
            .at(4.0, 16.0, 11.0)
            .id();
        let back_right_leg = m
            .part("back_right_leg", 96, 29)
            .cube(-1.5, -2.0, -2.5, 4, 9, 5, 0.0)
            .at(-4.0, 9.0, 11.0)
            .id();
        let back_right_shin = m
            .part("back_right_shin", 96, 43)
            .cube(-1.0, 0.0, -1.5, 3, 5, 3, 0.0)
            .at(-4.0, 16.0, 11.0)
            .id();
        let back_right_hoof = m
            .part("back_right_hoof", 96, 51)
            .cube(-1.5, 5.1, -2.0, 4, 3, 4, 0.0)
            .at(-4.0, 16.0, 11.0)
            .id();

        let front_left_leg = m
            .part("front_left_leg", 44, 29)
            .cube(-1.9, -1.0, -2.1, 3, 8, 4, 0.0)
            .at(4.0, 9.0, -8.0)
            .id();
        let front_left_shin = m
            .part("front_left_shin", 44, 41)
            .cube(-1.9, 0.0, -1.6, 3, 5, 3, 0.0)
            .at(4.0, 16.0, -8.0)
            .id();
        let front_left_hoof = m
            .part("front_left_hoof", 44, 51)
            .cube(-2.4, 5.1, -2.1, 4, 3, 4, 0.0)
            .at(4.0, 16.0, -8.0)
            .id();
        let front_right_leg = m
            .part("front_right_leg", 60, 29)
            .cube(-1.1, -1.0, -2.1, 3, 8, 4, 0.0)
            .at(-4.0, 9.0, -8.0)
            .id();
        let front_right_shin = m
            .part("front_right_shin", 60, 41)
            .cube(-1.1, 0.0, -1.6, 3, 5, 3, 0.0)
            .at(-4.0, 16.0, -8.0)
            .id();
        let front_right_hoof = m
            .part("front_right_hoof", 60, 51)
            .cube(-1.6, 5.1, -2.1, 4, 3, 4, 0.0)
            .at(-4.0, 16.0, -8.0)
            .id();

        let head = m
            .part("head", 0, 0)
            .cube(-2.5, -10.0, -1.5, 5, 5, 7, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let upper_jaw = m
            .child(head, "upper_jaw", 24, 18)
            .cube(-2.0, -10.0, -7.0, 4, 3, 6, 0.0)
            .at(0.0, 3.95, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let lower_jaw = m
            .child(head, "lower_jaw", 24, 27)
            .cube(-2.0, -7.0, -6.5, 4, 2, 5, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let horse_left_ear = m
            .part("horse_left_ear", 0, 0)
            .cube(0.45, -12.0, 4.0, 2, 3, 1, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let horse_right_ear = m
            .part("horse_right_ear", 0, 0)
            .cube(-2.45, -12.0, 4.0, 2, 3, 1, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let mule_left_ear = m
            .part("mule_left_ear", 0, 12)
            .cube(-2.0, -16.0, 4.0, 2, 7, 1, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.2617994)
            .id();
        let mule_right_ear = m
            .part("mule_right_ear", 0, 12)
            .cube(0.0, -16.0, 4.0, 2, 7, 1, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, -0.2617994)
            .id();
        let neck = m
            .part("neck", 0, 12)
            .cube(-2.05, -9.8, -2.0, 4, 14, 8, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();

        let mule_left_chest = m
            .part("mule_left_chest", 0, 34)
            .cube(-3.0, 0.0, 0.0, 8, 8, 3, 0.0)
            .at(-7.5, 3.0, 10.0)
            .angles(0.0, PI / 2.0, 0.0)
            .id();
        let mule_right_chest = m
            .part("mule_right_chest", 0, 47)
            .cube(-3.0, 0.0, 0.0, 8, 8, 3, 0.0)
            .at(4.5, 3.0, 10.0)
            .angles(0.0, PI / 2.0, 0.0)
            .id();

        let saddle_bottom = m
            .part("saddle_bottom", 80, 0)
            .cube(-5.0, 0.0, -3.0, 10, 1, 8, 0.0)
            .at(0.0, 2.0, 2.0)
            .id();
        let saddle_front = m
            .part("saddle_front", 106, 9)
            .cube(-1.5, -1.0, -3.0, 3, 1, 2, 0.0)
            .at(0.0, 2.0, 2.0)
            .id();
        let saddle_back = m
            .part("saddle_back", 80, 9)
            .cube(-4.0, -1.0, 3.0, 8, 1, 2, 0.0)
            .at(0.0, 2.0, 2.0)
            .id();
        let left_saddle_metal = m
            .part("left_saddle_metal", 74, 0)
            .cube(-0.5, 6.0, -1.0, 1, 2, 2, 0.0)
            .at(5.0, 3.0, 2.0)
            .id();
        let left_saddle_rope = m
            .part("left_saddle_rope", 70, 0)
            .cube(-0.5, 0.0, -0.5, 1, 6, 1, 0.0)
            .at(5.0, 3.0, 2.0)
            .id();
        let right_saddle_metal = m
            .part("right_saddle_metal", 74, 4)
            .cube(-0.5, 6.0, -1.0, 1, 2, 2, 0.0)
            .at(-5.0, 3.0, 2.0)
            .id();
        let right_saddle_rope = m
            .part("right_saddle_rope", 80, 0)
            .cube(-0.5, 0.0, -0.5, 1, 6, 1, 0.0)
            .at(-5.0, 3.0, 2.0)
            .id();
        let left_face_metal = m
            .part("left_face_metal", 74, 13)
            .cube(1.5, -8.0, -4.0, 1, 2, 2, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let right_face_metal = m
            .part("right_face_metal", 74, 13)
            .cube(-2.5, -8.0, -4.0, 1, 2, 2, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let left_rein = m
            .part("left_rein", 44, 10)
            // java: `addBox(..., 0, 3, 16)` really is zero-width.  Do not turn this into a
            // one-unit strap: its degenerate quads are part of the vanilla display list.
            .cube(2.6, -6.0, -6.0, 0, 3, 16, 0.0)
            .at(0.0, 4.0, -10.0)
            .id();
        let right_rein = m
            .part("right_rein", 44, 5)
            .cube(-2.6, -6.0, -6.0, 0, 3, 16, 0.0)
            .at(0.0, 4.0, -10.0)
            .id();
        let mane = m
            .part("mane", 58, 0)
            .cube(-1.0, -11.5, 5.0, 2, 16, 4, 0.0)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();
        let face_ropes = m
            .part("face_ropes", 80, 12)
            .cube(-2.5, -10.1, -7.0, 5, 5, 12, 0.2)
            .at(0.0, 4.0, -10.0)
            .angles(0.5235988, 0.0, 0.0)
            .id();

        Self {
            model: m,
            parts: HorseParts {
                body,
                tail_base,
                tail_middle,
                tail_tip,
                back_left_leg,
                back_left_shin,
                back_left_hoof,
                back_right_leg,
                back_right_shin,
                back_right_hoof,
                front_left_leg,
                front_left_shin,
                front_left_hoof,
                front_right_leg,
                front_right_shin,
                front_right_hoof,
                head,
                upper_jaw,
                lower_jaw,
                horse_left_ear,
                horse_right_ear,
                mule_left_ear,
                mule_right_ear,
                neck,
                mule_left_chest,
                mule_right_chest,
                saddle_bottom,
                saddle_front,
                saddle_back,
                left_saddle_metal,
                left_saddle_rope,
                right_saddle_metal,
                right_saddle_rope,
                left_face_metal,
                right_face_metal,
                left_rein,
                right_rein,
                mane,
                face_ropes,
            },
        }
    }

    /// Restores optional parts to their constructor pose before applying a stateful frame.
    ///
    /// The Java renderer shares one `ModelHorse` between every horse.  Most of these parts are
    /// only assigned inside the saddle branch, so resetting them here prevents a saddled horse
    /// from leaving stale bridle/chest angles behind for the next entity in the same submission.
    fn reset_optional_pose(&mut self) {
        let q = self.parts;
        let m = &mut self.model;
        m.get_mut(q.mule_left_chest).rotation_point = [-7.5, 3.0, 10.0];
        m.get_mut(q.mule_left_chest).rotate_angle = [0.0, PI / 2.0, 0.0];
        m.get_mut(q.mule_right_chest).rotation_point = [4.5, 3.0, 10.0];
        m.get_mut(q.mule_right_chest).rotate_angle = [0.0, PI / 2.0, 0.0];
        m.get_mut(q.saddle_bottom).rotation_point = [0.0, 2.0, 2.0];
        m.get_mut(q.saddle_bottom).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.saddle_front).rotation_point = [0.0, 2.0, 2.0];
        m.get_mut(q.saddle_front).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.saddle_back).rotation_point = [0.0, 2.0, 2.0];
        m.get_mut(q.saddle_back).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.left_saddle_metal).rotation_point = [5.0, 3.0, 2.0];
        m.get_mut(q.left_saddle_metal).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.left_saddle_rope).rotation_point = [5.0, 3.0, 2.0];
        m.get_mut(q.left_saddle_rope).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.right_saddle_metal).rotation_point = [-5.0, 3.0, 2.0];
        m.get_mut(q.right_saddle_metal).rotate_angle = [0.0, 0.0, 0.0];
        m.get_mut(q.right_saddle_rope).rotation_point = [-5.0, 3.0, 2.0];
        m.get_mut(q.right_saddle_rope).rotate_angle = [0.0, 0.0, 0.0];
        for id in [q.left_face_metal, q.right_face_metal, q.face_ropes] {
            m.get_mut(id).rotation_point = [0.0, 4.0, -10.0];
            m.get_mut(id).rotate_angle = [0.5235988, 0.0, 0.0];
        }
        for id in [q.left_rein, q.right_rein] {
            m.get_mut(id).rotation_point = [0.0, 4.0, -10.0];
            m.get_mut(id).rotate_angle = [0.0, 0.0, 0.0];
        }
    }

    /// java: `ModelHorse.setLivingAnimations`, with the entity-dependent inputs supplied by the
    /// horse-only render state seam.
    // Keep the multiply-by-negative-one spelling from ModelHorse's float expressions; unary
    // negation is mathematically equivalent but needlessly changes the source operation shape.
    #[allow(clippy::neg_multiply)]
    fn set_horse_pose(&mut self, p: &AnimParams, state: &HorseRenderState) {
        self.reset_optional_pose();
        let q = self.parts;
        let grass = state.grass_eating_amount;
        let rearing = state.rearing_amount;
        let mouth = state.mouth_openness;
        let standing = 1.0 - rearing;

        // java: updateHorseRotation -> f3, then `f4 = f2 / (180F / PI)`; clamp the relative
        // head yaw before conversion and add the short walk bob only above amount 0.2.
        let head_yaw_degrees = p.net_head_yaw.clamp(-20.0, 20.0);
        let neutral_head_yaw = model_radians(head_yaw_degrees);
        let mut raw_head_pitch = model_radians(p.head_pitch);
        if p.limb_swing_amount > 0.2 {
            raw_head_pitch += cos(p.limb_swing * 0.4) * 0.15 * p.limb_swing_amount;
        }

        let m = &mut self.model;
        m.get_mut(q.head).rotation_point = [0.0, 4.0, -10.0];
        m.get_mut(q.tail_base).rotation_point[1] = 3.0;
        m.get_mut(q.tail_middle).rotation_point[2] = 14.0;
        m.get_mut(q.mule_right_chest).rotation_point[1] = 3.0;
        m.get_mut(q.mule_right_chest).rotation_point[2] = 10.0;
        m.get_mut(q.body).rotate_angle[0] = 0.0;

        let mut head_pitch = 0.5235988 + raw_head_pitch;
        head_pitch = rearing * (0.2617994 + raw_head_pitch)
            + grass * 2.18166
            + (1.0 - rearing.max(grass)) * head_pitch;
        // java: `f6 * f3 / (180F / PI) + (1 - max(f6, f5)) * head.rotateAngleY`. Keep the
        // multiply-before-divide ordering rather than multiplying a pre-divided angle.
        let head_yaw = rearing * head_yaw_degrees / (180.0 / PI)
            + (1.0 - rearing.max(grass)) * neutral_head_yaw;
        m.get_mut(q.head).rotate_angle[0] = head_pitch;
        m.get_mut(q.head).rotate_angle[1] = head_yaw;
        m.get_mut(q.head).rotation_point[1] =
            rearing * -6.0 + grass * 11.0 + (1.0 - rearing.max(grass)) * 4.0;
        m.get_mut(q.head).rotation_point[2] =
            rearing * -1.0 + grass * -10.0 + (1.0 - rearing.max(grass)) * -10.0;

        // java: `ModelHorse.setLivingAnimations`'s observable rearing offsets.  These are
        // independent of the horse type and are applied to the shared tail/right-chest pivots
        // before the saddle branch copies the right chest pivot to the left chest.  Resetting the
        // neutral values above on every frame is important because vanilla reuses one
        // ModelHorse instance.  Java briefly writes a rearing tail-middle Z here too, but its
        // final shared-tail copy below restores that field from the unchanged tail-base Z.
        m.get_mut(q.tail_base).rotation_point[1] = rearing * 9.0 + standing * 3.0;
        m.get_mut(q.mule_right_chest).rotation_point[1] = rearing * 5.5 + standing * 3.0;
        m.get_mut(q.mule_right_chest).rotation_point[2] = rearing * 15.0 + standing * 10.0;

        // java: all head attachments copy only the head's X/Y rotation and Y/Z pivots.  Mule
        // ear Z rolls are constructor constants and must survive this copy.
        let head_point = m.get(q.head).rotation_point;
        let head_angle = m.get(q.head).rotate_angle;
        for id in [
            q.horse_left_ear,
            q.horse_right_ear,
            q.mule_left_ear,
            q.mule_right_ear,
            q.neck,
            q.mane,
        ] {
            let part = m.get_mut(id);
            part.rotation_point[1] = head_point[1];
            part.rotation_point[2] = head_point[2];
            part.rotate_angle[0] = head_angle[0];
            part.rotate_angle[1] = head_angle[1];
        }
        m.get_mut(q.upper_jaw).rotation_point = [0.0, 0.02, 0.02 - mouth];
        m.get_mut(q.lower_jaw).rotation_point = [0.0, 0.0, mouth];
        m.get_mut(q.upper_jaw).rotate_angle = [-0.09424778 * mouth, 0.0, 0.0];
        m.get_mut(q.lower_jaw).rotate_angle = [0.15707964 * mouth, 0.0, 0.0];
        m.get_mut(q.mane).rotation_point[1] = head_point[1];
        m.get_mut(q.mane).rotation_point[2] = head_point[2];
        m.get_mut(q.mane).rotate_angle[0] = head_angle[0];
        m.get_mut(q.mane).rotate_angle[1] = head_angle[1];

        // The face metal/rope parts are only emitted for a saddle, but they still follow the
        // live head pose just as the saddle branch writes them in Java.
        for id in [q.left_face_metal, q.right_face_metal, q.face_ropes] {
            let part = m.get_mut(id);
            part.rotation_point[1] = head_point[1];
            part.rotation_point[2] = head_point[2];
            part.rotate_angle[0] = head_angle[0];
            part.rotate_angle[1] = head_angle[1];
        }

        let f10 = cos(p.limb_swing * 0.6662 + PI);
        let f11 = f10 * 0.8 * p.limb_swing_amount;
        m.get_mut(q.mule_left_chest).rotate_angle[0] = f11 / 5.0;
        m.get_mut(q.mule_right_chest).rotate_angle[0] = -f11 / 5.0;
        let body_angle = rearing * -45.0 / (180.0 / PI) + standing * m.get(q.body).rotate_angle[0];
        m.get_mut(q.body).rotate_angle[0] = body_angle;

        let f12 = PI / 2.0;
        let f13 = PI * 3.0 / 2.0;
        let f15 = 0.2617994 * rearing;
        let f16 = cos(p.age_in_ticks * 0.6 + PI);
        let front_leg_y = -2.0 * rearing + 9.0 * standing;
        let front_leg_z = -2.0 * rearing + -8.0 * standing;
        m.get_mut(q.front_left_leg).rotation_point[1] = front_leg_y;
        m.get_mut(q.front_left_leg).rotation_point[2] = front_leg_z;
        m.get_mut(q.front_right_leg).rotation_point[1] = front_leg_y;
        m.get_mut(q.front_right_leg).rotation_point[2] = front_leg_z;

        let back_left_point = m.get(q.back_left_leg).rotation_point;
        let back_right_point = m.get(q.back_right_leg).rotation_point;
        m.get_mut(q.back_left_shin).rotation_point[1] =
            back_left_point[1] + sin(f12 + f15 + standing * -f10 * 0.5 * p.limb_swing_amount) * 7.0;
        m.get_mut(q.back_left_shin).rotation_point[2] =
            back_left_point[2] + cos(f13 + f15 + standing * -f10 * 0.5 * p.limb_swing_amount) * 7.0;
        m.get_mut(q.back_right_shin).rotation_point[1] =
            back_right_point[1] + sin(f12 + f15 + standing * f10 * 0.5 * p.limb_swing_amount) * 7.0;
        m.get_mut(q.back_right_shin).rotation_point[2] =
            back_right_point[2] + cos(f13 + f15 + standing * f10 * 0.5 * p.limb_swing_amount) * 7.0;

        let front_left_angle = (-1.0471976 + f16) * rearing + f11 * standing;
        let front_right_angle = (-1.0471976 + -f16) * rearing - f11 * standing;
        let front_left_point = m.get(q.front_left_leg).rotation_point;
        let front_right_point = m.get(q.front_right_leg).rotation_point;
        m.get_mut(q.front_left_shin).rotation_point[1] =
            front_left_point[1] + sin(f12 + front_left_angle) * 7.0;
        m.get_mut(q.front_left_shin).rotation_point[2] =
            front_left_point[2] + cos(f13 + front_left_angle) * 7.0;
        m.get_mut(q.front_right_shin).rotation_point[1] =
            front_right_point[1] + sin(f12 + front_right_angle) * 7.0;
        m.get_mut(q.front_right_shin).rotation_point[2] =
            front_right_point[2] + cos(f13 + front_right_angle) * 7.0;

        let back_left_angle = f15 - f10 * 0.5 * p.limb_swing_amount * standing;
        let back_left_shin_angle = -0.08726646 * rearing
            + (-f10 * 0.5 * p.limb_swing_amount - (f10 * 0.5 * p.limb_swing_amount).max(0.0))
                * standing;
        let back_right_angle = f15 + f10 * 0.5 * p.limb_swing_amount * standing;
        let back_right_shin_angle = -0.08726646 * rearing
            + (f10 * 0.5 * p.limb_swing_amount - (-f10 * 0.5 * p.limb_swing_amount).max(0.0))
                * standing;
        m.get_mut(q.back_left_leg).rotate_angle[0] = back_left_angle;
        m.get_mut(q.back_left_shin).rotate_angle[0] = back_left_shin_angle;
        m.get_mut(q.back_left_hoof).rotate_angle[0] = back_left_shin_angle;
        m.get_mut(q.back_right_leg).rotate_angle[0] = back_right_angle;
        m.get_mut(q.back_right_shin).rotate_angle[0] = back_right_shin_angle;
        m.get_mut(q.back_right_hoof).rotate_angle[0] = back_right_shin_angle;

        let front_left_shin_angle = (front_left_angle + PI * (0.2 + f16 * 0.2).max(0.0)) * rearing
            + (f11 + (f10 * 0.5 * p.limb_swing_amount).max(0.0)) * standing;
        let front_right_shin_angle = (front_right_angle + PI * (0.2 - f16 * 0.2).max(0.0))
            * rearing
            + (-f11 + (-f10 * 0.5 * p.limb_swing_amount).max(0.0)) * standing;
        m.get_mut(q.front_left_leg).rotate_angle[0] = front_left_angle;
        m.get_mut(q.front_left_shin).rotate_angle[0] = front_left_shin_angle;
        m.get_mut(q.front_left_hoof).rotate_angle[0] = front_left_shin_angle;
        m.get_mut(q.front_right_leg).rotate_angle[0] = front_right_angle;
        m.get_mut(q.front_right_shin).rotate_angle[0] = front_right_shin_angle;
        m.get_mut(q.front_right_hoof).rotate_angle[0] = front_right_shin_angle;
        for (hoof, shin) in [
            (q.back_left_hoof, q.back_left_shin),
            (q.back_right_hoof, q.back_right_shin),
            (q.front_left_hoof, q.front_left_shin),
            (q.front_right_hoof, q.front_right_shin),
        ] {
            let point = m.get(shin).rotation_point;
            let hoof = m.get_mut(hoof);
            hoof.rotation_point[1] = point[1];
            hoof.rotation_point[2] = point[2];
        }

        if state.saddled {
            let saddle_y = rearing * 0.5 + standing * 2.0;
            let saddle_z = rearing * 11.0 + standing * 2.0;
            for id in [
                q.saddle_front,
                q.saddle_back,
                q.left_saddle_rope,
                q.right_saddle_rope,
                q.left_saddle_metal,
                q.right_saddle_metal,
            ] {
                m.get_mut(id).rotation_point[1] = saddle_y;
                m.get_mut(id).rotation_point[2] = saddle_z;
            }
            m.get_mut(q.saddle_bottom).rotation_point[1] = saddle_y;
            m.get_mut(q.saddle_bottom).rotation_point[2] = saddle_z;
            m.get_mut(q.mule_left_chest).rotation_point[1] =
                m.get(q.mule_right_chest).rotation_point[1];
            m.get_mut(q.mule_left_chest).rotation_point[2] =
                m.get(q.mule_right_chest).rotation_point[2];
            let body_angle = m.get(q.body).rotate_angle[0];
            for id in [q.saddle_bottom, q.saddle_front, q.saddle_back] {
                m.get_mut(id).rotate_angle[0] = body_angle;
            }
            for id in [q.left_rein, q.right_rein] {
                m.get_mut(id).rotation_point[1] = head_point[1];
                m.get_mut(id).rotation_point[2] = head_point[2];
                m.get_mut(id).rotate_angle[0] = raw_head_pitch;
                m.get_mut(id).rotate_angle[1] = head_angle[1];
            }
            for id in [q.left_face_metal, q.right_face_metal, q.face_ropes] {
                m.get_mut(id).rotation_point[1] = head_point[1];
                m.get_mut(id).rotation_point[2] = head_point[2];
                m.get_mut(id).rotate_angle[0] = head_angle[0];
                m.get_mut(id).rotate_angle[1] = head_angle[1];
            }
            if state.ridden {
                for id in [
                    q.left_saddle_rope,
                    q.left_saddle_metal,
                    q.right_saddle_rope,
                    q.right_saddle_metal,
                ] {
                    m.get_mut(id).rotate_angle[0] = -1.0471976;
                    m.get_mut(id).rotate_angle[2] = 0.0;
                }
            } else {
                for id in [q.left_saddle_rope, q.left_saddle_metal] {
                    m.get_mut(id).rotate_angle[0] = f11 / 3.0;
                    m.get_mut(id).rotate_angle[2] = f11 / 5.0;
                }
                for id in [q.right_saddle_rope, q.right_saddle_metal] {
                    m.get_mut(id).rotate_angle[0] = f11 / 3.0;
                    m.get_mut(id).rotate_angle[2] = -f11 / 5.0;
                }
            }
        }

        let mut tail_angle = -1.3089 + p.limb_swing_amount * 1.5;
        if tail_angle > 0.0 {
            tail_angle = 0.0;
        }
        let tail_yaw = if state.tail_swishing {
            tail_angle = 0.0;
            cos(p.age_in_ticks * 0.7)
        } else {
            0.0
        };
        m.get_mut(q.tail_base).rotate_angle[1] = tail_yaw;
        m.get_mut(q.tail_middle).rotate_angle[1] = tail_yaw;
        m.get_mut(q.tail_tip).rotate_angle[1] = tail_yaw;
        m.get_mut(q.tail_middle).rotation_point[1] = m.get(q.tail_base).rotation_point[1];
        m.get_mut(q.tail_tip).rotation_point[1] = m.get(q.tail_base).rotation_point[1];
        m.get_mut(q.tail_middle).rotation_point[2] = m.get(q.tail_base).rotation_point[2];
        m.get_mut(q.tail_tip).rotation_point[2] = m.get(q.tail_base).rotation_point[2];
        m.get_mut(q.tail_base).rotate_angle[0] = tail_angle;
        m.get_mut(q.tail_middle).rotate_angle[0] = tail_angle;
        m.get_mut(q.tail_tip).rotate_angle[0] = -0.2618 + tail_angle;
    }

    /// Poses and emits a horse with the watched/client-local facts that Java's model reads.
    pub fn pose_and_render_with_state(
        &mut self,
        p: &AnimParams,
        state: &HorseRenderState,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        self.model.swing_progress = p.entity.swing_progress;
        self.model.is_riding = p.entity.riding;
        self.model.is_child = !state.adult;
        self.set_horse_pose(p, state);
        self.render_with_state(p, state, base, out);
    }

    /// java: `ModelHorse.render`, including the adult saddle/rein/chest visibility fork.
    fn render_with_state(
        &self,
        p: &AnimParams,
        state: &HorseRenderState,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        let q = self.parts;
        let legs = [
            q.back_left_leg,
            q.back_left_shin,
            q.back_left_hoof,
            q.back_right_leg,
            q.back_right_shin,
            q.back_right_hoof,
            q.front_left_leg,
            q.front_left_shin,
            q.front_left_hoof,
            q.front_right_leg,
            q.front_right_shin,
            q.front_right_hoof,
        ];
        let body = [
            q.body,
            q.tail_base,
            q.tail_middle,
            q.tail_tip,
            q.neck,
            q.mane,
        ];
        let mule = [q.mule_left_ear, q.mule_right_ear];
        let horse = [q.horse_left_ear, q.horse_right_ear];
        let saddle = [
            q.face_ropes,
            q.saddle_bottom,
            q.saddle_front,
            q.saddle_back,
            q.left_saddle_rope,
            q.left_saddle_metal,
            q.right_saddle_rope,
            q.right_saddle_metal,
            q.left_face_metal,
            q.right_face_metal,
        ];
        let reins = [q.left_rein, q.right_rein];
        let adult_saddle = state.adult && state.saddled;
        let mule_type = matches!(state.horse_type, 1 | 2);
        // java: `isAdultHorse() && isChested()` gates the two mule-chest parts.  The
        // horse-type watcher only selects the ear pair; Java does not add a second
        // type==1/2 condition to chest visibility.
        let adult_chest = state.adult && state.chested;

        if adult_saddle {
            self.model.emit_some(&saddle, p.scale, base, out);
            if state.ridden {
                self.model.emit_some(&reins, p.scale, base, out);
            }
        }

        if state.adult {
            self.model.emit_some(&legs, p.scale, base, out);
            self.model.emit_some(&body, p.scale, base, out);
            if mule_type {
                self.model.emit_some(&mule, p.scale, base, out);
            } else {
                self.model.emit_some(&horse, p.scale, base, out);
            }
            self.model.emit_part(q.head, p.scale, base, out);
        } else {
            let horse_size = 0.5;
            let legs_base = base
                * Mat4::from_scale(Vec3::new(horse_size, 0.5 + horse_size * 0.5, horse_size))
                * Mat4::from_translation(Vec3::new(0.0, 0.95 * (1.0 - horse_size), 0.0));
            self.model.emit_some(&legs, p.scale, legs_base, out);
            let body_base = base
                * Mat4::from_scale(Vec3::splat(horse_size))
                * Mat4::from_translation(Vec3::new(0.0, 1.35 * (1.0 - horse_size), 0.0));
            self.model.emit_some(&body, p.scale, body_base, out);
            let head_scale = 0.5 + horse_size * horse_size * 0.5;
            let head_translation = if state.grass_eating_amount <= 0.0 {
                Vec3::new(0.0, 1.35 * (1.0 - horse_size), 0.0)
            } else {
                Vec3::new(
                    0.0,
                    0.9 * (1.0 - horse_size) * state.grass_eating_amount
                        + 1.35 * (1.0 - horse_size) * (1.0 - state.grass_eating_amount),
                    0.15 * (1.0 - horse_size) * state.grass_eating_amount,
                )
            };
            let head_base = base
                * Mat4::from_scale(Vec3::splat(head_scale))
                * Mat4::from_translation(head_translation);
            if mule_type {
                self.model.emit_some(&mule, p.scale, head_base, out);
            } else {
                self.model.emit_some(&horse, p.scale, head_base, out);
            }
            self.model.emit_part(q.head, p.scale, head_base, out);
        }

        if adult_chest {
            self.model
                .emit_some(&[q.mule_left_chest, q.mule_right_chest], p.scale, base, out);
        }
    }
}

impl Default for HorseModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for HorseModel {
    fn model(&self) -> &Model {
        &self.model
    }

    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// The generic render contract can supply head/body movement and a walking phase.  Horse
    /// type is already carried by `EntityAnimState::variant`, so the mule ear fork is live even
    /// before the horse-only watched booleans have a producer in this crate.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let state = HorseRenderState::neutral_for(p);
        self.set_horse_pose(p, &state);
    }

    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let state = HorseRenderState::neutral_for(p);
        self.render_with_state(p, &state, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelVillager (and the standalone transcription ModelWitch needs)
// ---------------------------------------------------------------------------------------------

/// The six parts of the villager.
#[derive(Clone, Copy, Debug)]
pub struct VillagerParts {
    /// java: `villagerHead`
    pub head: PartId,
    /// java: `villagerNose`, a child of `villagerHead`.
    pub nose: PartId,
    /// java: `villagerBody`
    pub body: PartId,
    /// java: `villagerArms` — both arms and the folded hands in one part.
    pub arms: PartId,
    /// java: `rightVillagerLeg`
    pub right_leg: PartId,
    /// java: `leftVillagerLeg`
    pub left_leg: PartId,
}

/// java: `ModelVillager(float modelSize, float yOffset, int textureWidth, int textureHeight)`.
///
/// Factored out because `ModelWitch` calls it through `super(...)` with a 64x128 texture; this is
/// that constructor, not a subclassing mechanism.
fn build_villager(size: f32, y_off: f32, tw: i32, th: i32) -> (Model, VillagerParts) {
    // java: `ModelBase.textureWidth/Height` are untouched (64x32); each part sets its own.
    let mut m = Model::new(64, 32);
    let head = m
        .part("head", 0, 0)
        .tex_size(tw, th)
        .at(0.0, 0.0 + y_off, 0.0)
        .tex(0, 0)
        .cube(-4.0, -10.0, -4.0, 8, 10, 8, size)
        .id();
    let nose = m
        .child(head, "nose", 0, 0)
        .tex_size(tw, th)
        .at(0.0, y_off - 2.0, 0.0)
        .tex(24, 0)
        .cube(-1.0, -1.0, -6.0, 2, 4, 2, size)
        .id();
    let body = m
        .part("body", 0, 0)
        .tex_size(tw, th)
        .at(0.0, 0.0 + y_off, 0.0)
        .tex(16, 20)
        .cube(-4.0, 0.0, -3.0, 8, 12, 6, size)
        .tex(0, 38)
        .cube(-4.0, 0.0, -3.0, 8, 18, 6, size + 0.5)
        .id();
    let arms = m
        .part("arms", 0, 0)
        .tex_size(tw, th)
        .at(0.0, 0.0 + y_off + 2.0, 0.0)
        .tex(44, 22)
        .cube(-8.0, -2.0, -2.0, 4, 8, 4, size)
        .tex(44, 22)
        .cube(4.0, -2.0, -2.0, 4, 8, 4, size)
        .tex(40, 38)
        .cube(-4.0, 2.0, -2.0, 8, 4, 4, size)
        .id();
    let right_leg = m
        .part("right_leg", 0, 22)
        .tex_size(tw, th)
        .at(-2.0, 12.0 + y_off, 0.0)
        .cube(-2.0, 0.0, -2.0, 4, 12, 4, size)
        .id();
    let left_leg = m
        .part("left_leg", 0, 22)
        .tex_size(tw, th)
        .mirrored()
        .at(2.0, 12.0 + y_off, 0.0)
        .cube(-2.0, 0.0, -2.0, 4, 12, 4, size)
        .id();
    (
        m,
        VillagerParts {
            head,
            nose,
            body,
            arms,
            right_leg,
            left_leg,
        },
    )
}

/// java: `ModelVillager.setRotationAngles` — the body `ModelWitch` reaches through `super`.
fn villager_pose(m: &mut Model, q: &VillagerParts, p: &AnimParams) {
    m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
    m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
    m.get_mut(q.arms).rotation_point[1] = 3.0;
    m.get_mut(q.arms).rotation_point[2] = -1.0;
    m.get_mut(q.arms).rotate_angle[0] = -0.75;
    m.get_mut(q.right_leg).rotate_angle[0] =
        cos(p.limb_swing * 0.6662) * 1.4 * p.limb_swing_amount * 0.5;
    m.get_mut(q.left_leg).rotate_angle[0] =
        cos(p.limb_swing * 0.6662 + PI) * 1.4 * p.limb_swing_amount * 0.5;
    m.get_mut(q.right_leg).rotate_angle[1] = 0.0;
    m.get_mut(q.left_leg).rotate_angle[1] = 0.0;
}

/// java: `ModelVillager.render` — head (with the nose, and for the witch the hat, as children),
/// body, legs, then arms. `ModelWitch` inherits it unchanged.
fn villager_draw(
    m: &Model,
    q: &VillagerParts,
    scale: f32,
    base: Mat4,
    out: &mut Vec<EntityVertex>,
) {
    m.emit_some(
        &[q.head, q.body, q.right_leg, q.left_leg, q.arms],
        scale,
        base,
        out,
    );
}

/// The villager.
///
/// java: `ModelVillager.java`
#[derive(Clone, Debug)]
pub struct VillagerModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the six parts.
    pub parts: VillagerParts,
}

impl VillagerModel {
    /// java: `ModelVillager(float modelSize)` — `this(modelSize, 0.0F, 64, 64)`.
    ///
    /// `EntityVillager.getProfession` (which the porting notes map to `p.entity.variant`) is read
    /// by `RenderVillager` to pick the texture, never by the model, so nothing consumes it here.
    #[must_use]
    pub fn new(size: f32) -> Self {
        Self::with(size, 0.0, 64, 64)
    }

    /// java: `ModelVillager(float modelSize, float yOffset, int textureWidth, int textureHeight)`
    #[must_use]
    pub fn with(size: f32, y_off: f32, tw: i32, th: i32) -> Self {
        let (model, parts) = build_villager(size, y_off, tw, th);
        Self { model, parts }
    }
}

impl EntityModel for VillagerModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        villager_pose(&mut self.model, &q, p);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        villager_draw(&self.model, &self.parts, p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelWitch
// ---------------------------------------------------------------------------------------------

/// The five parts `ModelWitch` adds on top of the villager.
#[derive(Clone, Copy, Debug)]
pub struct WitchParts {
    /// java: `field_82901_h` — the wart on the witch's nose, a child of `villagerNose`.
    pub mole: PartId,
    /// java: `witchHat` — the brim, a child of `villagerHead`.
    pub hat: PartId,
    /// java: the anonymous `modelrenderer` — the hat's first cone segment, a child of `witchHat`.
    pub hat_cone1: PartId,
    /// java: `modelrenderer1` — the second segment, a child of the first.
    pub hat_cone2: PartId,
    /// java: `modelrenderer2` — the tip, a child of the second.
    pub hat_tip: PartId,
}

/// The witch.
///
/// java: `ModelWitch.java`, which `extends ModelVillager`. Transcribed standalone: the villager
/// half is the same [`build_villager`]/[`villager_pose`]/[`villager_draw`] the
/// `ModelVillager(float, float, int, int)` constructor and `super.setRotationAngles` call would
/// have given it.
#[derive(Clone, Debug)]
pub struct WitchModel {
    /// The part tree.
    pub model: Model,
    /// The inherited villager parts.
    pub parts: VillagerParts,
    /// The witch-only parts.
    pub witch: WitchParts,
}

impl WitchModel {
    /// java: `ModelWitch(float modelSize)` — `super(modelSize, 0.0F, 64, 128)` and then the hat.
    #[must_use]
    pub fn new(size: f32) -> Self {
        let (mut m, parts) = build_villager(size, 0.0, 64, 128);
        // java: `field_82901_h`'s `new ModelRenderer(this)` runs as a field initialiser, i.e.
        // after `super(...)` and before the constructor body — so it is the seventh part.
        let mole = m
            .child(parts.nose, "mole", 0, 0)
            .tex_size(64, 128)
            .at(0.0, -2.0, 0.0)
            .tex(0, 0)
            .cube(0.0, 3.0, -6.75, 1, 1, 1, -0.25)
            .id();
        let hat = m
            .child(parts.head, "hat", 0, 0)
            .tex_size(64, 128)
            .at(-5.0, -10.031_25, -5.0)
            .tex(0, 64)
            .cube(0.0, 0.0, 0.0, 10, 2, 10, 0.0)
            .id();
        let hat_cone1 = m
            .child(hat, "hat_cone1", 0, 0)
            .tex_size(64, 128)
            .at(1.75, -4.0, 2.0)
            .tex(0, 76)
            .cube(0.0, 0.0, 0.0, 7, 4, 7, 0.0)
            .angles(-0.052_359_88, 0.0, 0.026_179_94)
            .id();
        let hat_cone2 = m
            .child(hat_cone1, "hat_cone2", 0, 0)
            .tex_size(64, 128)
            .at(1.75, -4.0, 2.0)
            .tex(0, 87)
            .cube(0.0, 0.0, 0.0, 4, 4, 4, 0.0)
            .angles(-0.104_719_76, 0.0, 0.052_359_88)
            .id();
        let hat_tip = m
            .child(hat_cone2, "hat_tip", 0, 0)
            .tex_size(64, 128)
            .at(1.75, -2.0, 2.0)
            .tex(0, 95)
            .cube(0.0, 0.0, 0.0, 1, 2, 1, 0.25)
            .angles(-0.209_439_52, 0.0, 0.104_719_76)
            .id();
        Self {
            model: m,
            parts,
            witch: WitchParts {
                mole,
                hat,
                hat_cone1,
                hat_cone2,
                hat_tip,
            },
        }
    }
}

impl EntityModel for WitchModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelWitch.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let q = self.parts;
        villager_pose(&mut self.model, &q, p);

        let nose = self.model.get_mut(q.nose);
        nose.offset = [0.0, 0.0, 0.0];
        // java: float f = 0.01F * (float)(entityIn.getEntityId() % 10);
        //
        // The id and the held-item flag used to be fields on this struct. `RenderManager` keeps
        // one `WitchModel` for every witch, so both were the shared-model defect class, and
        // neither had a writer: `f` was always 0, which pins `rotateAngleX` at 0 and
        // `rotateAngleZ` at a constant 2.5 degrees, and the drinking dip never fired at all.
        let f = 0.01 * (p.entity.entity_id % 10) as f32;
        // java: entityIn.ticksExisted -> p.entity.ticks_existed
        let ticks = p.entity.ticks_existed as f32;
        nose.rotate_angle[0] = sin(ticks * f) * 4.5 * PI / 180.0;
        nose.rotate_angle[1] = 0.0;
        nose.rotate_angle[2] = cos(ticks * f) * 2.5 * PI / 180.0;

        if p.entity.witch_holding_item {
            nose.rotate_angle[0] = -0.9;
            nose.offset[2] = -0.093_75;
            nose.offset[1] = 0.187_5;
        }
    }

    /// java: inherited `ModelVillager.render`.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        villager_draw(&self.model, &self.parts, p.scale, base, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Poses a model the way `RendererLivingEntity` does — living animations first.
    fn pose<M: EntityModel>(m: &mut M, p: &AnimParams) {
        m.set_living_animations(p);
        m.set_rotation_angles(p);
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    // -- ModelChicken ------------------------------------------------------------------------

    #[test]
    fn chicken_geometry_matches_the_java_constructor() {
        let c = ChickenModel::new();
        let m = &c.model;
        // head: addBox(-2, -6, -2, 4, 6, 3), setRotationPoint(0, -1 + 16, -4)
        assert_eq!(m.get(c.parts.head).cubes[0].pos1, [-2.0, -6.0, -2.0]);
        assert_eq!(m.get(c.parts.head).cubes[0].pos2, [2.0, 0.0, 1.0]);
        assert_eq!(m.get(c.parts.head).rotation_point, [0.0, 15.0, -4.0]);
        // bill and chin share the head's pivot but not its boxes.
        assert_eq!(m.get(c.parts.bill).rotation_point, [0.0, 15.0, -4.0]);
        assert_eq!(m.get(c.parts.chin).rotation_point, [0.0, 15.0, -4.0]);
        assert_eq!(m.get(c.parts.bill).cubes[0].pos1, [-2.0, -4.0, -4.0]);
        assert_eq!(m.get(c.parts.chin).cubes[0].pos2, [1.0, 0.0, -1.0]);
        // body: addBox(-3, -4, -3, 6, 8, 6), setRotationPoint(0, 16, 0)
        assert_eq!(m.get(c.parts.body).cubes[0].pos2, [3.0, 4.0, 3.0]);
        assert_eq!(m.get(c.parts.body).rotation_point, [0.0, 16.0, 0.0]);
        // legs at y = 3 + 16 = 19, wings at y = -3 + 16 = 13.
        assert_eq!(m.get(c.parts.right_leg).rotation_point, [-2.0, 19.0, 1.0]);
        assert_eq!(m.get(c.parts.left_leg).rotation_point, [1.0, 19.0, 1.0]);
        assert_eq!(m.get(c.parts.right_wing).rotation_point, [-4.0, 13.0, 0.0]);
        assert_eq!(m.get(c.parts.left_wing).rotation_point, [4.0, 13.0, 0.0]);
        // The wings are the same 1x4x6 slab mirrored in placement, not in UV.
        assert_eq!(m.get(c.parts.right_wing).cubes[0].pos1, [0.0, 0.0, -3.0]);
        assert_eq!(m.get(c.parts.left_wing).cubes[0].pos1, [-1.0, 0.0, -3.0]);
        assert!(!m.get(c.parts.left_wing).mirror);
    }

    /// java: `rightWing.rotateAngleZ = ageInTicks; leftWing.rotateAngleZ = -ageInTicks;` and the
    /// `cos(limbSwing * 0.6662) * 1.4F * limbSwingAmount` leg swing.
    #[test]
    fn chicken_wings_flap_with_age_and_legs_swing_in_antiphase() {
        let mut c = ChickenModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 7.25;
        p.limb_swing_amount = 0.8;
        p.age_in_ticks = 13.5;
        pose(&mut c, &p);
        let m = &c.model;

        assert!(close(m.get(c.parts.right_wing).rotate_angle[2], 13.5));
        assert!(close(m.get(c.parts.left_wing).rotate_angle[2], -13.5));

        let want_right = cos(7.25 * 0.6662) * 1.4 * 0.8;
        let want_left = cos(7.25 * 0.6662 + PI) * 1.4 * 0.8;
        assert!(close(m.get(c.parts.right_leg).rotate_angle[0], want_right));
        assert!(close(m.get(c.parts.left_leg).rotate_angle[0], want_left));
        assert!(
            want_right * want_left < 0.0,
            "the +PI phase puts the legs in antiphase"
        );
        // The body is always face-down.
        assert!(close(m.get(c.parts.body).rotate_angle[0], PI / 2.0));
        // bill and chin copy the head's two angles exactly.
        let h = m.get(c.parts.head).rotate_angle;
        assert_eq!(m.get(c.parts.bill).rotate_angle[0..2], h[0..2]);
        assert_eq!(m.get(c.parts.chin).rotate_angle[0..2], h[0..2]);
    }

    // -- ModelWolf ---------------------------------------------------------------------------

    #[test]
    fn wolf_head_carries_four_boxes_at_three_texture_offsets() {
        let w = WolfModel::new();
        let head = w.model.get(w.parts.head);
        assert_eq!(head.cubes.len(), 4, "main box + two ears + the snout");
        assert_eq!(head.cubes[0].pos1, [-3.0, -3.0, -2.0]);
        assert_eq!(head.cubes[0].pos2, [3.0, 3.0, 2.0]);
        // ear 1 and ear 2 are 2x2x1 at y = -5, mirrored in placement only.
        assert_eq!(head.cubes[1].pos1, [-3.0, -5.0, 0.0]);
        assert_eq!(head.cubes[2].pos1, [1.0, -5.0, 0.0]);
        // the snout, at texture offset (0, 10)
        assert_eq!(head.cubes[3].pos1, [-1.5, 0.0, -5.0]);
        assert_eq!(head.cubes[3].pos2, [1.5, 3.0, -1.0]);
        assert_eq!(head.texture_offset, [0, 10], "last setTextureOffset wins");
        assert_eq!(head.rotation_point, [-1.0, 13.5, -7.0]);
        // The mane is the only 8-wide box; the four legs are identical 2x8x2 posts.
        assert_eq!(w.model.get(w.parts.mane).cubes[0].pos2, [4.0, 3.0, 4.0]);
        for leg in [w.parts.leg1, w.parts.leg2, w.parts.leg3, w.parts.leg4] {
            assert_eq!(w.model.get(leg).cubes[0].pos1, [-1.0, 0.0, -1.0]);
            assert_eq!(w.model.get(leg).cubes[0].pos2, [1.0, 8.0, 1.0]);
        }
        assert_eq!(w.model.get(w.parts.tail).rotation_point, [-1.0, 12.0, 8.0]);
    }

    /// java: `ModelWolf.setLivingAnimations` — the whole sitting branch moves pivots that the
    /// standing branch puts back.
    #[test]
    fn wolf_sitting_changes_the_pivots_and_folds_the_legs() {
        let mut standing = WolfModel::new();
        let mut sitting = WolfModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 4.0;
        p.limb_swing_amount = 1.0;
        pose(&mut standing, &p);
        p.entity.sitting = true;
        pose(&mut sitting, &p);

        let (sm, tm) = (&standing.model, &sitting.model);
        let q = standing.parts;

        assert_eq!(sm.get(q.body).rotation_point, [0.0, 14.0, 2.0]);
        assert_eq!(tm.get(q.body).rotation_point, [0.0, 18.0, 0.0]);
        assert!(close(sm.get(q.body).rotate_angle[0], PI / 2.0));
        assert!(close(tm.get(q.body).rotate_angle[0], PI / 4.0));

        // The mane's pivot Z is -3 either way; only Y moves. (The Java's `wolfMane` never gets
        // z = 2 back — the standing branch writes -3, so the constructor's 2 is dead.)
        assert_eq!(sm.get(q.mane).rotation_point, [-1.0, 14.0, -3.0]);
        assert_eq!(tm.get(q.mane).rotation_point, [-1.0, 16.0, -3.0]);
        assert!(close(tm.get(q.mane).rotate_angle[0], PI * 2.0 / 5.0));

        assert_eq!(sm.get(q.tail).rotation_point, [-1.0, 12.0, 8.0]);
        assert_eq!(tm.get(q.tail).rotation_point, [-1.0, 21.0, 6.0]);

        assert_eq!(sm.get(q.leg1).rotation_point, [-2.5, 16.0, 7.0]);
        assert_eq!(tm.get(q.leg1).rotation_point, [-2.5, 22.0, 2.0]);
        assert!(close(tm.get(q.leg1).rotate_angle[0], PI * 3.0 / 2.0));
        assert!(close(tm.get(q.leg2).rotate_angle[0], PI * 3.0 / 2.0));
        // The front legs get the 5.811947 literal and the 0.01-nudged X pivots.
        assert!(close(tm.get(q.leg3).rotate_angle[0], 5.811_947));
        assert_eq!(tm.get(q.leg3).rotation_point, [-2.49, 17.0, -4.0]);
        assert_eq!(tm.get(q.leg4).rotation_point, [0.51, 17.0, -4.0]);

        // Standing: legs 1/4 and 2/3 are in antiphase.
        let a = cos(4.0 * 0.6662) * 1.4;
        let b = cos(4.0 * 0.6662 + PI) * 1.4;
        assert!(close(sm.get(q.leg1).rotate_angle[0], a));
        assert!(close(sm.get(q.leg2).rotate_angle[0], b));
        assert!(close(sm.get(q.leg3).rotate_angle[0], b));
        assert!(close(sm.get(q.leg4).rotate_angle[0], a));
    }

    #[test]
    fn wolf_tail_wags_unless_angry_and_always_takes_age_as_pitch() {
        let mut calm = WolfModel::new();
        let mut angry = WolfModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 6.0;
        p.limb_swing_amount = 0.9;
        p.age_in_ticks = 2.25;
        // java: entitywolf.getInterestedAngle(partialTicks) -> p.entity.tail_rotation
        p.entity.tail_rotation = 0.35;
        pose(&mut calm, &p);
        p.entity.aggressive = true;
        pose(&mut angry, &p);

        let want = cos(6.0 * 0.6662) * 1.4 * 0.9;
        assert!(close(calm.model.get(calm.parts.tail).rotate_angle[1], want));
        assert!(close(
            angry.model.get(angry.parts.tail).rotate_angle[1],
            0.0
        ));
        // setRotationAngles runs last, so ageInTicks lands on the tail's X in both cases.
        assert!(close(calm.model.get(calm.parts.tail).rotate_angle[0], 2.25));
        assert!(close(
            angry.model.get(angry.parts.tail).rotate_angle[0],
            2.25
        ));
        // A dry wolf shakes not at all, so the head's Z is purely the interested angle.
        assert!(close(calm.model.get(calm.parts.head).rotate_angle[2], 0.35));
        assert!(close(calm.model.get(calm.parts.body).rotate_angle[2], 0.0));
    }

    #[test]
    fn wolf_wet_shake_offsets_each_part() {
        let mut w = WolfModel::new();
        // Peak of the envelope: (0.9 + 0.0) / 1.8 = 0.5.
        let expect = |off: f32| {
            let f = ((0.9_f32 + off) / 1.8).clamp(0.0, 1.0);
            sin(f * PI) * sin(f * PI * 11.0) * 0.15 * PI
        };
        assert!(close(WolfModel::shake_angle(0.9, 0.0), expect(0.0)));
        assert!(close(WolfModel::shake_angle(0.9, -0.2), expect(-0.2)));
        // The four parts lag by 0, -0.08, -0.16, -0.2 — a travelling wave down the body, and
        // the timer arrives per entity rather than as a field on the shared model.
        let mut p = AnimParams::still();
        p.entity.wolf_shake_time = 0.9;
        pose(&mut w, &p);
        let m = &w.model;
        assert!(close(m.get(w.parts.mane).rotate_angle[2], expect(-0.08)));
        assert!(close(m.get(w.parts.body).rotate_angle[2], expect(-0.16)));
        assert!(close(m.get(w.parts.tail).rotate_angle[2], expect(-0.2)));

        // A second wolf posed straight after with a dry timer must not inherit the first one's
        // shake — the failure mode a field on the shared model has.
        let dry = AnimParams::still();
        pose(&mut w, &dry);
        assert!(close(w.model.get(w.parts.body).rotate_angle[2], 0.0));
    }

    // -- ModelOcelot -------------------------------------------------------------------------

    #[test]
    fn ocelot_head_has_four_boxes_at_the_named_texture_offsets() {
        let o = OcelotModel::new();
        let head = o.model.get(o.parts.head);
        assert_eq!(head.cubes.len(), 4);
        // head.main, head.nose, head.ear1, head.ear2
        assert_eq!(head.cubes[0].pos1, [-2.5, -2.0, -3.0]);
        assert_eq!(head.cubes[0].pos2, [2.5, 2.0, 2.0]);
        assert_eq!(head.cubes[1].pos1, [-1.5, 0.0, -4.0]);
        assert_eq!(head.cubes[2].pos1, [-2.0, -3.0, 0.0]);
        assert_eq!(head.cubes[3].pos1, [1.0, -3.0, 0.0]);
        assert_eq!(head.rotation_point, [0.0, 15.0, -9.0]);
        // head.ear1 sits at (0, 10) and head.ear2 at (6, 10): the -X face U differs by 6/64.
        let ear1_u = head.cubes[2].quads[1].verts[1].uv[0];
        let ear2_u = head.cubes[3].quads[1].verts[1].uv[0];
        assert!(close(ear2_u - ear1_u, 6.0 / 64.0));
        // The body is the tall 4x16x6 slab; the tail starts pitched at 0.9.
        assert_eq!(o.model.get(o.parts.body).cubes[0].pos1, [-2.0, 3.0, -8.0]);
        assert_eq!(o.model.get(o.parts.body).rotation_point, [0.0, 12.0, -10.0]);
        assert!(close(o.model.get(o.parts.tail).rotate_angle[0], 0.9));
        assert_eq!(o.model.get(o.parts.tail2).rotation_point, [0.0, 20.0, 14.0]);
        assert_eq!(o.pose_state, 1, "field_78163_i starts at 1");
    }

    /// java: the sitting branch sets `field_78163_i = 3`, which makes `setRotationAngles` skip
    /// the entire body/leg/tail block — the sitting pose survives untouched.
    #[test]
    fn ocelot_sitting_locks_the_pose_and_skips_the_leg_swing() {
        let mut o = OcelotModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 9.0;
        p.limb_swing_amount = 1.0;
        p.entity.sitting = true;
        pose(&mut o, &p);

        assert_eq!(o.pose_state, 3);
        let m = &o.model;
        assert!(close(m.get(o.parts.body).rotate_angle[0], PI / 4.0));
        assert_eq!(m.get(o.parts.body).rotation_point, [0.0, 8.0, -5.0]);
        // head: 15 - 3.3 = 11.7, -9 + 1 = -8
        assert!(close(m.get(o.parts.head).rotation_point[1], 11.7));
        assert!(close(m.get(o.parts.head).rotation_point[2], -8.0));
        assert!(close(m.get(o.parts.tail).rotate_angle[0], 1.727_876_1));
        assert!(close(m.get(o.parts.tail2).rotate_angle[0], 2.670_354));
        assert!(close(
            m.get(o.parts.back_left_leg).rotate_angle[0],
            -(PI / 2.0)
        ));
        assert!(close(
            m.get(o.parts.front_right_leg).rotate_angle[0],
            -0.157_079_64
        ));
        assert_eq!(
            m.get(o.parts.back_right_leg).rotation_point,
            [-1.1, 21.0, 1.0]
        );
    }

    /// java: the sprinting branch (`field_78163_i == 2`) phases the four legs 0 / +0.3 / PI+0.3 /
    /// PI and narrows the tail sway from PI/4 to PI/10.
    #[test]
    fn ocelot_sprinting_phases_the_legs_and_narrows_the_tail_sway() {
        let mut o = OcelotModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 5.5;
        p.limb_swing_amount = 0.6;
        p.entity.sprinting = true;
        pose(&mut o, &p);
        assert_eq!(o.pose_state, 2);

        let m = &o.model;
        let base = 5.5 * 0.6662;
        assert!(close(
            m.get(o.parts.back_left_leg).rotate_angle[0],
            cos(base) * 0.6
        ));
        assert!(close(
            m.get(o.parts.back_right_leg).rotate_angle[0],
            cos(base + 0.3) * 0.6
        ));
        assert!(close(
            m.get(o.parts.front_left_leg).rotate_angle[0],
            cos(base + PI + 0.3) * 0.6
        ));
        assert!(close(
            m.get(o.parts.front_right_leg).rotate_angle[0],
            cos(base + PI) * 0.6
        ));
        assert!(close(
            m.get(o.parts.tail2).rotate_angle[0],
            1.727_876_1 + (PI / 10.0) * cos(5.5) * 0.6
        ));
        // Sprinting also snaps the tail upright and stacks tail2 onto the tail's pivot.
        assert!(close(m.get(o.parts.tail).rotate_angle[0], PI / 2.0));
        assert_eq!(m.get(o.parts.tail2).rotation_point, [0.0, 15.0, 16.0]);
    }

    // -- ModelRabbit -------------------------------------------------------------------------

    #[test]
    fn rabbit_geometry_and_constructor_rotations() {
        let r = RabbitModel::new();
        let m = &r.model;
        assert_eq!(m.get(r.parts.left_foot).cubes[0].pos1, [-1.0, 5.5, -3.7]);
        assert_eq!(m.get(r.parts.left_foot).rotation_point, [3.0, 17.5, 3.7]);
        assert_eq!(m.get(r.parts.right_foot).rotation_point, [-3.0, 17.5, 3.7]);
        assert_eq!(m.get(r.parts.body).cubes[0].pos1, [-3.0, -2.0, -10.0]);
        assert_eq!(m.get(r.parts.body).cubes[0].pos2, [3.0, 3.0, 0.0]);
        assert_eq!(m.get(r.parts.body).rotation_point, [0.0, 19.0, 8.0]);
        // The constructor's `setRotationOffset` pre-tilts thighs, body and tail by -20 degrees
        // and the arms by -10.
        assert!(close(
            m.get(r.parts.left_thigh).rotate_angle[0],
            -0.349_065_84
        ));
        assert!(close(m.get(r.parts.body).rotate_angle[0], -0.349_065_84));
        assert!(close(m.get(r.parts.tail).rotate_angle[0], -0.349_065_9));
        assert!(close(
            m.get(r.parts.right_arm).rotate_angle[0],
            -0.174_532_92
        ));
        // The two ears splay by +/- 15 degrees and share the head's pivot.
        assert!(close(
            m.get(r.parts.right_ear).rotate_angle[1],
            -0.261_799_4
        ));
        assert!(close(m.get(r.parts.left_ear).rotate_angle[1], 0.261_799_4));
        for part in [
            r.parts.head,
            r.parts.right_ear,
            r.parts.left_ear,
            r.parts.nose,
        ] {
            assert_eq!(m.get(part).rotation_point, [0.0, 16.0, -1.0]);
        }
        // `mirror = true` is set after every `addBox`, so no box is actually mirrored.
        assert!(m.get(r.parts.left_foot).mirror);
        assert_eq!(
            m.get(r.parts.left_foot).cubes[0].quads[4].verts[1].pos[0],
            -1.0
        );
    }

    /// java: `field_178701_m = sin(jumpProgress * PI)`, then the three limb formulas that read
    /// it, all in the `0.017453292F` degree spelling.
    #[test]
    fn rabbit_jump_drives_thighs_feet_and_arms() {
        let mut r = RabbitModel::new();
        let mut p = AnimParams::still();
        // java: entityrabbit.func_175521_o(f) -> p.entity.tail_rotation
        p.entity.tail_rotation = 0.5;
        p.net_head_yaw = 45.0;
        p.head_pitch = -30.0;
        pose(&mut r, &p);

        let bob = sin(0.5 * PI);
        assert!(close(r.jump_bob, bob));
        let m = &r.model;
        let thigh = (bob * 50.0 - 21.0) * 0.017_453_292;
        let foot = bob * 50.0 * 0.017_453_292;
        let arm = (bob * -40.0 - 11.0) * 0.017_453_292;
        assert!(close(m.get(r.parts.left_thigh).rotate_angle[0], thigh));
        assert!(close(m.get(r.parts.right_thigh).rotate_angle[0], thigh));
        assert!(close(m.get(r.parts.left_foot).rotate_angle[0], foot));
        assert!(close(m.get(r.parts.right_foot).rotate_angle[0], foot));
        assert!(close(m.get(r.parts.left_arm).rotate_angle[0], arm));
        assert!(close(m.get(r.parts.right_arm).rotate_angle[0], arm));

        // Ears track the nose's yaw plus/minus 15 degrees; all four head parts share the pitch.
        let yaw = 45.0 * 0.017_453_292;
        let pitch = -30.0 * 0.017_453_292;
        assert!(close(m.get(r.parts.nose).rotate_angle[1], yaw));
        assert!(close(
            m.get(r.parts.right_ear).rotate_angle[1],
            yaw - 0.261_799_4
        ));
        assert!(close(
            m.get(r.parts.left_ear).rotate_angle[1],
            yaw + 0.261_799_4
        ));
        for part in [
            r.parts.nose,
            r.parts.head,
            r.parts.right_ear,
            r.parts.left_ear,
        ] {
            assert!(close(m.get(part).rotate_angle[0], pitch));
        }
    }

    // -- ModelSquid --------------------------------------------------------------------------

    /// java: the eight tentacles sit on a radius-5 circle at `31 + i = 15`, each yawed by
    /// `-j * 2PI / 8 + PI/2`. The trig here is `java.lang.Math` on doubles, not `MathHelper`.
    #[test]
    fn squid_tentacles_ring_the_body_with_matching_yaws() {
        let s = SquidModel::new();
        // java: squidBody.rotationPointY += 24 + (-16)
        assert_eq!(s.model.get(s.body).rotation_point, [0.0, 8.0, 0.0]);
        assert_eq!(s.model.get(s.body).cubes[0].pos1, [-6.0, -8.0, -6.0]);
        assert_eq!(s.model.get(s.body).cubes[0].pos2, [6.0, 8.0, 6.0]);

        for (j, &id) in s.tentacles.iter().enumerate() {
            let t = s.model.get(id);
            let d0 = j as f64 * core::f64::consts::PI * 2.0 / 8.0;
            assert!(close(t.rotation_point[0], d0.cos() as f32 * 5.0), "x {j}");
            assert!(close(t.rotation_point[2], d0.sin() as f32 * 5.0), "z {j}");
            assert!(close(t.rotation_point[1], 15.0), "y {j}");
            let want = (j as f64 * core::f64::consts::PI * -2.0 / 8.0
                + core::f64::consts::FRAC_PI_2) as f32;
            assert!(close(t.rotate_angle[1], want), "yaw {j}");
            // Every arm is the same 2x18x2 post.
            assert_eq!(t.cubes[0].pos1, [-1.0, 0.0, -1.0]);
            assert_eq!(t.cubes[0].pos2, [1.0, 18.0, 1.0]);
        }
        // Spot-check the cardinal ones by hand: j = 0 is +X, j = 2 is +Z.
        assert!(close(s.model.get(s.tentacles[0]).rotation_point[0], 5.0));
        assert!(close(s.model.get(s.tentacles[0]).rotate_angle[1], PI / 2.0));
        assert!(close(s.model.get(s.tentacles[2]).rotation_point[2], 5.0));
        assert!(close(s.model.get(s.tentacles[2]).rotate_angle[1], 0.0));
    }

    #[test]
    fn squid_tentacles_all_take_age_as_pitch() {
        let mut s = SquidModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 1.75;
        pose(&mut s, &p);
        for &id in &s.tentacles {
            assert!(close(s.model.get(id).rotate_angle[0], 1.75));
        }
        // The body is never animated.
        assert_eq!(s.model.get(s.body).rotate_angle, [0.0; 3]);
    }

    // -- ModelBat ----------------------------------------------------------------------------

    #[test]
    fn bat_wings_are_nested_and_the_body_carries_two_boxes() {
        let b = BatModel::new();
        let m = &b.model;
        assert_eq!(m.roots(), &[b.parts.head, b.parts.body]);
        assert_eq!(
            m.get(b.parts.head).children,
            vec![b.parts.right_ear, b.parts.left_ear]
        );
        assert_eq!(
            m.get(b.parts.body).children,
            vec![b.parts.right_wing, b.parts.left_wing]
        );
        assert_eq!(
            m.get(b.parts.right_wing).children,
            vec![b.parts.outer_right_wing]
        );
        assert_eq!(
            m.get(b.parts.left_wing).children,
            vec![b.parts.outer_left_wing]
        );
        // body = the 6x12x6 torso plus the 10x6x1 membrane at texture offset (0, 34)
        assert_eq!(m.get(b.parts.body).cubes.len(), 2);
        assert_eq!(m.get(b.parts.body).cubes[0].pos1, [-3.0, 4.0, -3.0]);
        assert_eq!(m.get(b.parts.body).cubes[1].pos1, [-5.0, 16.0, 0.0]);
        assert_eq!(m.get(b.parts.right_wing).cubes[0].pos1, [-12.0, 1.0, 1.5]);
        assert_eq!(
            m.get(b.parts.outer_right_wing).rotation_point,
            [-12.0, 1.0, 1.5]
        );
        assert_eq!(
            m.get(b.parts.outer_left_wing).rotation_point,
            [12.0, 1.0, 1.5]
        );
        assert!(m.get(b.parts.left_wing).mirror);
        assert!(!m.get(b.parts.right_wing).mirror);
    }

    /// java: `getIsBatHanging()` flips the whole model upside down and folds the wings; the
    /// flying branch drives them off `ageInTicks` instead.
    #[test]
    fn bat_hanging_and_flying_poses_differ_wholesale() {
        let mut hanging = BatModel::new();
        let mut flying = BatModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 11.0;
        p.net_head_yaw = 20.0;
        p.head_pitch = 5.0;
        pose(&mut flying, &p);
        // java: getIsBatHanging() -> p.entity.bat_hanging
        p.entity.bat_hanging = true;
        pose(&mut hanging, &p);

        let (h, q) = (&hanging.model, hanging.parts);
        assert!(close(
            h.get(q.head).rotate_angle[1],
            PI - model_radians(20.0)
        ));
        assert!(close(h.get(q.head).rotate_angle[2], PI));
        assert_eq!(h.get(q.head).rotation_point, [0.0, -2.0, 0.0]);
        assert!(close(h.get(q.body).rotate_angle[0], PI));
        assert_eq!(h.get(q.right_wing).rotation_point, [-3.0, 0.0, 3.0]);
        assert_eq!(h.get(q.left_wing).rotation_point, [3.0, 0.0, 3.0]);
        assert!(close(
            h.get(q.right_wing).rotate_angle[1],
            -(PI * 2.0 / 5.0)
        ));
        assert!(close(h.get(q.left_wing).rotate_angle[1], PI * 2.0 / 5.0));
        assert!(close(
            h.get(q.outer_right_wing).rotate_angle[1],
            -1.727_876_1
        ));
        assert!(close(h.get(q.outer_left_wing).rotate_angle[1], 1.727_876_1));

        let (f, q) = (&flying.model, flying.parts);
        assert!(close(f.get(q.head).rotate_angle[1], model_radians(20.0)));
        assert!(close(f.get(q.head).rotate_angle[2], 0.0));
        assert!(close(
            f.get(q.body).rotate_angle[0],
            PI / 4.0 + cos(11.0 * 0.1) * 0.15
        ));
        let flap = cos(11.0 * 1.3) * PI * 0.25;
        assert!(close(f.get(q.right_wing).rotate_angle[1], flap));
        assert!(close(f.get(q.left_wing).rotate_angle[1], -flap));
        assert!(close(f.get(q.outer_right_wing).rotate_angle[1], flap * 0.5));
        assert!(close(f.get(q.outer_left_wing).rotate_angle[1], -flap * 0.5));
    }

    // -- ModelSnowMan ------------------------------------------------------------------------

    #[test]
    fn snowman_stack_geometry_and_shrunken_boxes() {
        let s = SnowManModel::new();
        let m = &s.model;
        // Every box is inflated by -0.5, so the *recorded* corners are the un-inflated ones.
        assert_eq!(m.get(s.parts.head).cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(m.get(s.parts.head).cubes[0].pos2, [4.0, 0.0, 4.0]);
        // ...but the emitted geometry shrinks by half a pixel on every side.
        assert_eq!(
            m.get(s.parts.head).cubes[0].quads[4].verts[1].pos,
            [-3.5, -7.5, -3.5]
        );
        // f = 4: head at 4, hands at 4 + 9 - 7 = 6, body at 13, bottom at 24.
        assert_eq!(m.get(s.parts.head).rotation_point, [0.0, 4.0, 0.0]);
        assert_eq!(m.get(s.parts.right_hand).rotation_point, [0.0, 6.0, 0.0]);
        assert_eq!(m.get(s.parts.left_hand).rotation_point, [0.0, 6.0, 0.0]);
        assert_eq!(m.get(s.parts.body).rotation_point, [0.0, 13.0, 0.0]);
        assert_eq!(m.get(s.parts.bottom_body).rotation_point, [0.0, 24.0, 0.0]);
        assert_eq!(m.get(s.parts.bottom_body).cubes[0].pos2, [6.0, 0.0, 6.0]);
        // Every part overrides the texture size to 64x64 even though ModelBase says 64x32.
        assert_eq!(m.get(s.parts.bottom_body).texture_height, 64.0);
    }

    /// java: the two arms orbit the body's quarter-yaw on a radius-5 circle, 180 degrees apart.
    #[test]
    fn snowman_hands_orbit_the_quarter_body_yaw() {
        let mut s = SnowManModel::new();
        let mut p = AnimParams::still();
        p.net_head_yaw = 60.0;
        p.head_pitch = -10.0;
        pose(&mut s, &p);
        let m = &s.model;

        let head_y = model_radians(60.0);
        assert!(close(m.get(s.parts.head).rotate_angle[1], head_y));
        assert!(close(
            m.get(s.parts.head).rotate_angle[0],
            model_radians(-10.0)
        ));
        let body_y = head_y * 0.25;
        assert!(close(m.get(s.parts.body).rotate_angle[1], body_y));

        let (f, f1) = (sin(body_y), cos(body_y));
        assert!(close(m.get(s.parts.right_hand).rotation_point[0], f1 * 5.0));
        assert!(close(m.get(s.parts.right_hand).rotation_point[2], -f * 5.0));
        assert!(close(m.get(s.parts.left_hand).rotation_point[0], -f1 * 5.0));
        assert!(close(m.get(s.parts.left_hand).rotation_point[2], f * 5.0));
        // The hands' Y pivot is never touched by the animation.
        assert!(close(m.get(s.parts.right_hand).rotation_point[1], 6.0));
        assert!(close(m.get(s.parts.right_hand).rotate_angle[2], 1.0));
        assert!(close(m.get(s.parts.left_hand).rotate_angle[2], -1.0));
        assert!(close(m.get(s.parts.right_hand).rotate_angle[1], body_y));
        assert!(close(m.get(s.parts.left_hand).rotate_angle[1], PI + body_y));
    }

    // -- ModelIronGolem ----------------------------------------------------------------------

    #[test]
    fn iron_golem_geometry_uses_the_128_texture_and_the_minus_seven_offset() {
        let g = IronGolemModel::new();
        let m = &g.model;
        assert_eq!(m.get(g.parts.head).texture_width, 128.0);
        assert_eq!(m.get(g.parts.head).texture_height, 128.0);
        // head at (0, -7, -2), two boxes: the skull and the nose.
        assert_eq!(m.get(g.parts.head).rotation_point, [0.0, -7.0, -2.0]);
        assert_eq!(m.get(g.parts.head).cubes.len(), 2);
        assert_eq!(m.get(g.parts.head).cubes[0].pos1, [-4.0, -12.0, -5.5]);
        assert_eq!(m.get(g.parts.head).cubes[1].pos1, [-1.0, -5.0, -7.5]);
        // body: the 18x12x11 chest plus the inflated 9x5x6 waist.
        assert_eq!(m.get(g.parts.body).cubes[0].pos2, [9.0, 10.0, 5.0]);
        assert_eq!(m.get(g.parts.body).cubes[1].pos1, [-4.5, 10.0, -3.0]);
        // Arms hang from y = -7 regardless of the model's y offset.
        assert_eq!(m.get(g.parts.right_arm).rotation_point, [0.0, -7.0, 0.0]);
        assert_eq!(m.get(g.parts.right_arm).cubes[0].pos1, [-13.0, -2.5, -3.0]);
        assert_eq!(m.get(g.parts.left_arm).cubes[0].pos1, [9.0, -2.5, -3.0]);
        // Legs: 18 - 7 = 11, and the right one is mirrored at texture offset (60, 0).
        assert_eq!(m.get(g.parts.left_leg).rotation_point, [-4.0, 11.0, 0.0]);
        assert_eq!(m.get(g.parts.right_leg).rotation_point, [5.0, 11.0, 0.0]);
        assert!(m.get(g.parts.right_leg).mirror);
        assert_eq!(m.get(g.parts.right_leg).texture_offset, [60, 0]);
    }

    /// java: `func_78172_a` is a triangle wave; the legs take +/- 1.5 of it and the arms swing
    /// against each other unless the attack timer or the poppy pose takes over.
    #[test]
    fn iron_golem_gait_attack_and_poppy_branches() {
        let want_wave = |t: f32, period: f32| {
            ((t % period - period * 0.5).abs() - period * 0.25) / (period * 0.25)
        };
        // The wave really is a normalised triangle: 0 at the quarter points, +/-1 at the ends.
        assert!(close(want_wave(0.0, 13.0), 1.0));
        assert!(close(want_wave(6.5, 13.0), -1.0));
        assert!(close(want_wave(3.25, 13.0), 0.0));

        let mut g = IronGolemModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 4.0;
        p.limb_swing_amount = 0.5;
        pose(&mut g, &p);
        let w = want_wave(4.0, 13.0);
        let m = &g.model;
        assert!(close(
            m.get(g.parts.left_leg).rotate_angle[0],
            -1.5 * w * 0.5
        ));
        assert!(close(
            m.get(g.parts.right_leg).rotate_angle[0],
            1.5 * w * 0.5
        ));
        assert!(close(
            m.get(g.parts.right_arm).rotate_angle[0],
            (-0.2 + 1.5 * w) * 0.5
        ));
        assert!(close(
            m.get(g.parts.left_arm).rotate_angle[0],
            (-0.2 - 1.5 * w) * 0.5
        ));

        // Attack: both arms rise together off the attack timer.
        let mut g = IronGolemModel::new();
        p.entity.attack_timer = 6.0;
        p.partial_ticks = 0.25;
        pose(&mut g, &p);
        let a = -2.0 + 1.5 * want_wave(6.0 - 0.25, 10.0);
        assert!(close(g.model.get(g.parts.right_arm).rotate_angle[0], a));
        assert!(close(g.model.get(g.parts.left_arm).rotate_angle[0], a));

        // Poppy: only the right arm lifts, and only a little. The timer arrives on
        // `AnimParams`, not on the shared model — java: `entityirongolem.getHoldRoseTick()`.
        let mut g = IronGolemModel::new();
        p.entity.hold_rose_ticks = 40;
        p.entity.attack_timer = 0.0;
        pose(&mut g, &p);
        assert!(close(
            g.model.get(g.parts.right_arm).rotate_angle[0],
            -0.8 + 0.025 * want_wave(40.0, 70.0)
        ));
        assert!(close(g.model.get(g.parts.left_arm).rotate_angle[0], 0.0));
    }

    // -- ModelVillager / ModelWitch ----------------------------------------------------------

    #[test]
    fn villager_geometry_and_head_yaw_pitch_conversion() {
        let mut v = VillagerModel::new(0.0);
        let m = &v.model;
        // The nose is a child of the head, at the head's pivot minus 2.
        assert_eq!(m.get(v.parts.head).children, vec![v.parts.nose]);
        assert_eq!(m.get(v.parts.nose).rotation_point, [0.0, -2.0, 0.0]);
        assert_eq!(m.get(v.parts.nose).cubes[0].pos1, [-1.0, -1.0, -6.0]);
        assert_eq!(m.get(v.parts.head).cubes[0].pos1, [-4.0, -10.0, -4.0]);
        // The body's second box is the robe: same corners, inflated by 0.5.
        assert_eq!(m.get(v.parts.body).cubes.len(), 2);
        assert_eq!(m.get(v.parts.body).cubes[1].pos2, [4.0, 18.0, 3.0]);
        assert_eq!(
            m.get(v.parts.body).cubes[1].quads[4].verts[1].pos,
            [-4.5, -0.5, -3.5]
        );
        // The arms part holds three boxes and starts 2 above the origin.
        assert_eq!(m.get(v.parts.arms).cubes.len(), 3);
        assert_eq!(m.get(v.parts.arms).rotation_point, [0.0, 2.0, 0.0]);
        assert_eq!(m.get(v.parts.right_leg).rotation_point, [-2.0, 12.0, 0.0]);
        assert_eq!(m.get(v.parts.left_leg).rotation_point, [2.0, 12.0, 0.0]);
        assert!(m.get(v.parts.left_leg).mirror);
        assert_eq!(m.get(v.parts.head).texture_height, 64.0);

        // java: rotateAngleY = netHeadYaw / (180F / PI); rotateAngleX = headPitch / (180F / PI)
        let mut p = AnimParams::still();
        p.net_head_yaw = 90.0;
        p.head_pitch = -45.0;
        p.limb_swing = 2.0;
        p.limb_swing_amount = 1.0;
        pose(&mut v, &p);
        let m = &v.model;
        assert!(close(m.get(v.parts.head).rotate_angle[1], PI / 2.0));
        assert!(close(m.get(v.parts.head).rotate_angle[0], -PI / 4.0));
        assert!(close(
            m.get(v.parts.head).rotate_angle[1],
            model_radians(90.0)
        ));
        // The arms are re-pivoted and pitched forward every frame.
        assert_eq!(m.get(v.parts.arms).rotation_point, [0.0, 3.0, -1.0]);
        assert!(close(m.get(v.parts.arms).rotate_angle[0], -0.75));
        // Legs swing at half a biped's amplitude.
        assert!(close(
            m.get(v.parts.right_leg).rotate_angle[0],
            cos(2.0 * 0.6662) * 1.4 * 0.5
        ));
        assert!(close(
            m.get(v.parts.left_leg).rotate_angle[0],
            cos(2.0 * 0.6662 + PI) * 1.4 * 0.5
        ));
    }

    #[test]
    fn witch_hat_stacks_under_the_head_and_the_nose_twitches() {
        let mut w = WitchModel::new(0.0);
        {
            let m = &w.model;
            // The witch's texture is 64x128 all the way down, villager parts included.
            assert_eq!(m.get(w.parts.head).texture_height, 128.0);
            assert_eq!(m.get(w.witch.hat).texture_height, 128.0);
            // head -> [nose, hat]; nose -> [mole]; hat -> cone1 -> cone2 -> tip
            assert_eq!(
                m.get(w.parts.head).children,
                vec![w.parts.nose, w.witch.hat]
            );
            assert_eq!(m.get(w.parts.nose).children, vec![w.witch.mole]);
            assert_eq!(m.get(w.witch.hat).children, vec![w.witch.hat_cone1]);
            assert_eq!(m.get(w.witch.hat_cone1).children, vec![w.witch.hat_cone2]);
            assert_eq!(m.get(w.witch.hat_cone2).children, vec![w.witch.hat_tip]);
            assert_eq!(m.get(w.witch.hat).rotation_point, [-5.0, -10.031_25, -5.0]);
            assert_eq!(m.get(w.witch.hat).cubes[0].pos2, [10.0, 2.0, 10.0]);
            assert_eq!(m.get(w.witch.hat_cone1).rotation_point, [1.75, -4.0, 2.0]);
            assert!(close(
                m.get(w.witch.hat_cone1).rotate_angle[0],
                -0.052_359_88
            ));
            assert!(close(
                m.get(w.witch.hat_cone1).rotate_angle[2],
                0.026_179_94
            ));
            assert!(close(m.get(w.witch.hat_tip).rotate_angle[0], -0.209_439_52));
            assert_eq!(m.get(w.witch.hat_tip).rotation_point, [1.75, -2.0, 2.0]);
            // The mole is a 1x1x1 shrunk by 0.25 and hung off the nose.
            assert_eq!(m.get(w.witch.mole).rotation_point, [0.0, -2.0, 0.0]);
            assert_eq!(m.get(w.witch.mole).cubes[0].pos1, [0.0, 3.0, -6.75]);
        }

        // java: f = 0.01 * (entityId % 10); nose.x = sin(ticks * f) * 4.5 * PI / 180.
        // Both the id and the held-item flag are per-entity and ride on `AnimParams`; the
        // shared `WitchModel` holds neither.
        let mut p = AnimParams::still();
        p.entity.entity_id = 7;
        p.entity.ticks_existed = 25;
        w.set_rotation_angles(&p);
        let f = 0.01 * 7.0;
        let nose = w.model.get(w.parts.nose);
        assert!(close(
            nose.rotate_angle[0],
            sin(25.0 * f) * 4.5 * PI / 180.0
        ));
        assert!(close(nose.rotate_angle[1], 0.0));
        assert!(close(
            nose.rotate_angle[2],
            cos(25.0 * f) * 2.5 * PI / 180.0
        ));
        assert_eq!(nose.offset, [0.0, 0.0, 0.0]);

        // Holding a potion snaps the nose down and nudges it out of the face.
        p.entity.witch_holding_item = true;
        w.set_rotation_angles(&p);
        let nose = w.model.get(w.parts.nose);
        assert!(close(nose.rotate_angle[0], -0.9));
        assert_eq!(nose.offset, [0.0, 0.1875, -0.09375]);
    }

    // -- ModelHorse --------------------------------------------------------------------------

    #[test]
    fn horse_constructor_matches_vanilla_sheet_and_part_tree() {
        let h = HorseModel::new();
        let q = h.parts;
        assert_eq!(h.model.texture_width, 128);
        assert_eq!(h.model.texture_height, 128);
        assert_eq!(h.model.parts().len(), 39);
        // The two jaws are children of head; all other constructor parts are roots.
        assert_eq!(h.model.roots().len(), 37);
        assert_eq!(h.model.get(q.head).children, vec![q.upper_jaw, q.lower_jaw]);

        let names = [
            "body",
            "tail_base",
            "tail_middle",
            "tail_tip",
            "back_left_leg",
            "back_left_shin",
            "back_left_hoof",
            "back_right_leg",
            "back_right_shin",
            "back_right_hoof",
            "front_left_leg",
            "front_left_shin",
            "front_left_hoof",
            "front_right_leg",
            "front_right_shin",
            "front_right_hoof",
            "head",
            "upper_jaw",
            "lower_jaw",
            "horse_left_ear",
            "horse_right_ear",
            "mule_left_ear",
            "mule_right_ear",
            "neck",
            "mule_left_chest",
            "mule_right_chest",
            "saddle_bottom",
            "saddle_front",
            "saddle_back",
            "left_saddle_metal",
            "left_saddle_rope",
            "right_saddle_metal",
            "right_saddle_rope",
            "left_face_metal",
            "right_face_metal",
            "left_rein",
            "right_rein",
            "mane",
            "face_ropes",
        ];
        assert_eq!(
            h.model.parts().iter().map(|p| p.name).collect::<Vec<_>>(),
            names
        );

        let body = h.model.get(q.body);
        assert_eq!(body.cubes[0].pos1, [-5.0, -8.0, -19.0]);
        assert_eq!(body.cubes[0].pos2, [5.0, 2.0, 5.0]);
        assert_eq!(body.rotation_point, [0.0, 11.0, 9.0]);
        let head = h.model.get(q.head);
        assert_eq!(head.cubes[0].pos1, [-2.5, -10.0, -1.5]);
        assert_eq!(head.cubes[0].pos2, [2.5, -5.0, 5.5]);
        assert_eq!(head.rotation_point, [0.0, 4.0, -10.0]);
        assert!(close(head.rotate_angle[0], 0.5235988));

        // Vanilla's mouth parts are relative children with near-zero pivots, not second world
        // heads.  The reins are deliberately zero-width display-list quads.
        assert_eq!(h.model.get(q.upper_jaw).rotation_point, [0.0, 3.95, -10.0]);
        assert_eq!(h.model.get(q.lower_jaw).rotation_point, [0.0, 4.0, -10.0]);
        assert_eq!(h.model.get(q.left_rein).cubes[0].pos1[0], 2.6);
        assert_eq!(h.model.get(q.left_rein).cubes[0].pos2[0], 2.6);
        let ropes = &h.model.get(q.face_ropes).cubes[0];
        let face = ropes.quads[4];
        assert!(close(face.verts[0].pos[0], 2.7));
        assert!(close(face.verts[0].pos[1], -10.3));
        assert!(close(face.verts[0].pos[2], -7.2));
        assert!(close(
            h.model.get(q.mule_left_ear).rotate_angle[2],
            0.2617994
        ));
        assert!(close(
            h.model.get(q.mule_right_ear).rotate_angle[2],
            -0.2617994
        ));
    }

    #[test]
    fn horse_model_emits_visible_adult_foal_and_variant_geometry() {
        let mut h = HorseModel::new();
        let mut adult = AnimParams::still();
        adult.entity.child = false;
        let mut adult_out = Vec::new();
        h.pose_and_render(&adult, Mat4::IDENTITY, &mut adult_out);
        // body/tail/neck/mane (6), legs (12), two horse ears, head + two jaw children (3).
        assert_eq!(adult_out.len(), 23 * 36);
        assert!(adult_out.iter().any(|v| v.pos != [0.0; 3]));

        let mut mule = adult.clone();
        mule.entity.variant = 2;
        let mut mule_out = Vec::new();
        h.pose_and_render(&mule, Mat4::IDENTITY, &mut mule_out);
        assert_eq!(mule_out.len(), adult_out.len());
        assert_ne!(
            mule_out, adult_out,
            "mule ears must be a visible model fork"
        );
        assert!(close(
            h.model.get(h.parts.mule_left_ear).rotate_angle[2],
            0.2617994
        ));

        let mut foal = adult;
        foal.entity.child = true;
        let mut foal_out = Vec::new();
        h.pose_and_render(&foal, Mat4::IDENTITY, &mut foal_out);
        assert_eq!(foal_out.len(), adult_out.len());
        assert_ne!(
            foal_out, adult_out,
            "foal branches must apply the vanilla child scales"
        );
    }

    #[test]
    fn horse_model_walk_and_head_inputs_are_deterministic() {
        let mut h = HorseModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 4.25;
        p.limb_swing_amount = 0.8;
        p.net_head_yaw = 17.0;
        p.head_pitch = -9.0;
        pose(&mut h, &p);
        let q = h.parts;
        let f10 = cos(4.25 * 0.6662 + PI);
        let swing = -f10 * 0.5 * 0.8;
        let head_walk_bob = cos(4.25 * 0.4) * 0.15 * 0.8;
        assert!(close(
            h.model.get(q.head).rotate_angle[1],
            model_radians(17.0)
        ));
        assert!(close(
            h.model.get(q.head).rotate_angle[0],
            0.5235988 + model_radians(-9.0) + head_walk_bob
        ));
        assert!(close(h.model.get(q.back_left_leg).rotate_angle[0], swing));
        assert!(close(
            h.model.get(q.back_right_leg).rotate_angle[0],
            f10 * 0.5 * 0.8
        ));
        assert!(close(
            h.model.get(q.front_left_leg).rotate_angle[0],
            f10 * 0.8 * 0.8
        ));
        assert!(close(
            h.model.get(q.front_right_leg).rotate_angle[0],
            -f10 * 0.8 * 0.8
        ));
        assert!(close(
            h.model.get(q.tail_base).rotate_angle[0],
            (-1.3089_f32 + 0.8 * 1.5).min(0.0)
        ));

        // Vanilla clamps the horse's relative head yaw before converting to radians.  The
        // out-of-range sample is the negative control: a generic biped-style unclamped pose
        // would leave this at 90 degrees rather than the horse's 20-degree limit.
        p.net_head_yaw = 90.0;
        pose(&mut h, &p);
        assert!(close(
            h.model.get(q.head).rotate_angle[1],
            model_radians(20.0)
        ));
        assert_ne!(h.model.get(q.head).rotate_angle[1], model_radians(90.0));
    }

    #[test]
    fn horse_client_springs_match_vanilla_tick_and_frame_sampling() {
        let mut anim = HorseAnim {
            tail_swishing: true,
            ..Default::default()
        };
        anim.tick(true, true, true);

        // Repeated Java updates advance grazing first, then rearing clears both head-lean
        // samples; the mouth spring is independent of that branch.
        assert_eq!(anim.head_lean, 0.0);
        assert_eq!(anim.prev_head_lean, 0.0);
        assert!(close(anim.rearing_amount, 0.45));
        assert!(close(anim.mouth_openness, 0.75));
        let frame = anim.render_state(0.5, 2, true, true, false, true);
        assert!(frame.tail_swishing);
        assert!(close(frame.rearing_amount, 0.225));
        assert!(close(frame.mouth_openness, 0.375));

        anim.tick(false, false, false);
        let want_rearing = 0.45 + (0.8 * 0.45_f32 * 0.45 * 0.45 - 0.45) * 0.6 - 0.05;
        let want_mouth = 0.75 + (0.0 - 0.75_f32) * 0.7 - 0.05;
        assert!(
            close(anim.rearing_amount, want_rearing),
            "got {} want {}",
            anim.rearing_amount,
            want_rearing
        );
        assert!(close(anim.mouth_openness, want_mouth));
        let frame = anim.render_state(0.25, 0, true, false, false, false);
        assert!(close(
            frame.rearing_amount,
            0.45 + (want_rearing - 0.45) * 0.25
        ));
        assert!(close(
            frame.mouth_openness,
            0.75 + (want_mouth - 0.75) * 0.25
        ));
    }

    #[test]
    fn horse_rider_offset_uses_the_previous_rearing_sample_and_body_yaw() {
        let mut anim = HorseAnim::default();
        anim.tick(false, true, false);
        // The first rearing tick advances current=0.45 but Java's rider update still sees
        // prevRearingAmount=0. A following tick consumes that previous sample.
        assert_eq!(anim.rider_offset(90.0), [0.0; 3]);
        anim.tick(false, true, false);
        let [x, y, z] = anim.rider_offset(90.0);
        assert!(close(x, 0.315));
        assert!(close(y, 0.0675));
        assert!(close(z, 0.0));
    }

    #[test]
    fn horse_stateful_render_emits_saddle_reins_and_mule_chest() {
        let mut h = HorseModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 3.5;
        p.limb_swing = 1.25;
        p.limb_swing_amount = 0.7;
        p.net_head_yaw = 18.0;
        p.head_pitch = -7.0;
        let state = HorseRenderState {
            horse_type: 2,
            adult: true,
            saddled: true,
            chested: true,
            ridden: true,
            tail_swishing: true,
            grass_eating_amount: 0.25,
            rearing_amount: 0.4,
            mouth_openness: 0.5,
        };
        let mut out = Vec::new();
        h.pose_and_render_with_state(&p, &state, Mat4::IDENTITY, &mut out);
        // 23 base parts + ten saddle/bridle parts + two reins + two mule chest parts.
        assert_eq!(out.len(), 37 * 36);
        assert!(
            close(
                h.model.get(h.parts.body).rotate_angle[0],
                model_radians(-45.0) * 0.4
            ),
            "got {} want {}",
            h.model.get(h.parts.body).rotate_angle[0],
            model_radians(-45.0) * 0.4
        );
        assert!(close(
            h.model.get(h.parts.upper_jaw).rotate_angle[0],
            -0.09424778 * 0.5
        ));
        assert!(close(
            h.model.get(h.parts.lower_jaw).rotate_angle[0],
            0.15707964 * 0.5
        ));
        assert!(close(
            h.model.get(h.parts.left_rein).rotate_angle[0],
            model_radians(-7.0) + cos(1.25 * 0.4) * 0.15 * 0.7
        ));
        assert!(close(
            h.model.get(h.parts.tail_base).rotate_angle[1],
            cos(3.5 * 0.7)
        ));

        // Render visibility is adult-gated exactly like ModelHorse.render: metadata should not
        // make a foal draw adult saddle/chest attachments.
        let foal = HorseRenderState {
            adult: false,
            ..state
        };
        let mut foal_out = Vec::new();
        h.pose_and_render_with_state(&p, &foal, Mat4::IDENTITY, &mut foal_out);
        assert_eq!(foal_out.len(), 23 * 36);
    }

    #[test]
    fn horse_chest_visibility_is_adult_and_chested_for_every_variant() {
        // ModelHorse's chest branch is not restricted to donkey/mule watcher values: the
        // adult/chested flags alone control the two `mule*Chest` parts.  Keep the complete
        // vanilla variant range here so a future type-specific shortcut cannot regress this.
        for horse_type in [0, 1, 2, 3, 4] {
            let mut horse = HorseModel::new();
            let state = HorseRenderState {
                horse_type,
                adult: true,
                chested: true,
                ..HorseRenderState::default()
            };
            let mut out = Vec::new();
            horse.pose_and_render_with_state(
                &AnimParams::still(),
                &state,
                Mat4::IDENTITY,
                &mut out,
            );
            assert_eq!(out.len(), 25 * 36, "adult chested type {horse_type}");

            let foal_state = HorseRenderState {
                adult: false,
                ..state
            };
            let mut foal_out = Vec::new();
            horse.pose_and_render_with_state(
                &AnimParams::still(),
                &foal_state,
                Mat4::IDENTITY,
                &mut foal_out,
            );
            assert_eq!(foal_out.len(), 23 * 36, "foal chested type {horse_type}");
        }
    }

    #[test]
    fn horse_rearing_offsets_match_vanilla_and_reset_for_neutral_frames() {
        let mut horse = HorseModel::new();
        let mut params = AnimParams::still();
        params.entity.variant = 4;
        let rearing = HorseRenderState {
            horse_type: 4,
            rearing_amount: 0.4,
            ..HorseRenderState::default()
        };
        horse.pose_and_render_with_state(&params, &rearing, Mat4::IDENTITY, &mut Vec::new());

        let q = horse.parts;
        assert!(close(
            horse.model.get(q.tail_base).rotation_point[1],
            0.4 * 9.0 + 0.6 * 3.0
        ));
        assert!(
            close(horse.model.get(q.tail_middle).rotation_point[2], 14.0),
            "Java's later shared-tail copy intentionally restores tail-middle Z"
        );
        assert!(close(
            horse.model.get(q.mule_right_chest).rotation_point[1],
            0.4 * 5.5 + 0.6 * 3.0
        ));
        assert!(close(
            horse.model.get(q.mule_right_chest).rotation_point[2],
            0.4 * 15.0 + 0.6 * 10.0
        ));

        // Java's model is shared between entities; a following non-rearing frame must not retain
        // any interpolated offsets.  The health/attribute stream is intentionally not a model
        // input, so this neutral control also guards against speculative health-driven geometry.
        let neutral = HorseRenderState {
            horse_type: 0,
            ..HorseRenderState::default()
        };
        horse.pose_and_render_with_state(&params, &neutral, Mat4::IDENTITY, &mut Vec::new());
        assert_eq!(horse.model.get(q.tail_base).rotation_point[1], 3.0);
        assert_eq!(horse.model.get(q.tail_middle).rotation_point[2], 14.0);
        assert_eq!(horse.model.get(q.mule_right_chest).rotation_point[1], 3.0);
        assert_eq!(horse.model.get(q.mule_right_chest).rotation_point[2], 10.0);
    }

    // -- shared render behaviour --------------------------------------------------------------

    /// `renderWithRotation` must land the wolf's head where `T(pivot) * Ry * Rx * Rz` puts it —
    /// not where the plain `render` path's `Rz * Ry * Rx` would.
    #[test]
    fn wolf_head_uses_the_render_with_rotation_axis_order() {
        let mut w = WolfModel::new();
        w.model.is_child = false;
        w.model.get_mut(w.parts.head).rotate_angle = [0.4, 0.7, 0.9];

        let mut got = Vec::new();
        render_with_rotation(&w.model, w.parts.head, 0.0625, Mat4::IDENTITY, &mut got);

        let head = w.model.get(w.parts.head);
        let [px, py, pz] = head.rotation_point;
        let pivot = tr(px * 0.0625, py * 0.0625, pz * 0.0625);
        // `emit_part` writes each quad as verts 0, 1, 2, 0, 2, 3 — so out[0] is quad 0 vertex 0.
        let corner = head.cubes[0].quads[0].verts[0].pos;
        let corner = Vec3::new(corner[0] * 0.0625, corner[1] * 0.0625, corner[2] * 0.0625);

        let want = (pivot
            * Mat4::from_rotation_y(0.7)
            * Mat4::from_rotation_x(0.4)
            * Mat4::from_rotation_z(0.9))
        .transform_point3(corner);
        assert!(
            (Vec3::from(got[0].pos) - want).length() < 1e-5,
            "got {:?}, want {want:?}",
            got[0].pos
        );

        // The Z-first order the default `render` path uses is a different point entirely.
        let other = (pivot
            * Mat4::from_rotation_z(0.9)
            * Mat4::from_rotation_y(0.7)
            * Mat4::from_rotation_x(0.4))
        .transform_point3(corner);
        assert!((other - want).length() > 1e-3, "the two orders must differ");
    }

    /// Every model in this file draws exactly its own parts, once each, in both the adult and
    /// (where the Java has one) the baby branch.
    #[test]
    fn baby_and_adult_draw_lists_cover_every_part_once() {
        fn count(m: &mut dyn EntityModel, child: bool) -> usize {
            let mut p = AnimParams::still();
            p.entity.child = child;
            p.entity.ticks_existed = 3;
            let mut out = Vec::new();
            m.pose_and_render(&p, Mat4::IDENTITY, &mut out);
            out.len()
        }
        // 6 vertices per face, 6 faces per box.
        let per_box = 36;
        let cases: [(Box<dyn EntityModel>, usize); 10] = [
            (Box::new(ChickenModel::new()), 8),
            (Box::new(WolfModel::new()), 11),
            (Box::new(OcelotModel::new()), 11),
            (Box::new(RabbitModel::new()), 12),
            (Box::new(SquidModel::new()), 9),
            (Box::new(BatModel::new()), 9),
            (Box::new(SnowManModel::new()), 5),
            (Box::new(IronGolemModel::new()), 8),
            (Box::new(VillagerModel::new(0.0)), 9),
            (Box::new(WitchModel::new(0.0)), 14),
        ];
        for (mut m, boxes) in cases {
            let adult = count(m.as_mut(), false);
            let baby = count(m.as_mut(), true);
            assert_eq!(adult, boxes * per_box, "adult box count");
            assert_eq!(baby, boxes * per_box, "baby draws the same boxes, rescaled");
        }
    }
}
