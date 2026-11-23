//! Humanoid models.
//!
//! java: `ModelBiped.java`, `ModelPlayer.java`, `ModelZombie.java`, `ModelZombieVillager.java`,
//! `ModelSkeleton.java`.
//!
//! This file is the **reference port** for the rest of [`super`]: every other model follows the
//! same shape — a struct holding a [`Model`] plus the [`PartId`]s the Java kept as fields, a
//! constructor that replays the Java constructor in order, and a
//! [`EntityModel::set_rotation_angles`] that replays the Java method statement for statement.
//!
//! Two porting notes that recur everywhere:
//!
//! * Java subclasses *replace* a `ModelRenderer` field (`ModelSkeleton` swaps in thinner arms;
//!   `ModelPlayer` replaces player limbs). The old object stays in `boxList` but is never drawn.
//!   Skeleton can use [`rebuild`] because nothing samples its inherited list; player limbs must
//!   be appended in Java order because `LayerArrow` samples that list by index.
//! * `MathHelper.sin/cos` is the 65536-entry LUT, so [`mc_math::sin`]/[`mc_math::cos`] it is.

use glam::{Mat4, Vec3};
use mc_math::{cos, sin};

use crate::entity::anim::{model_radians, AnimParams, EntityModel};
use crate::entity::model::{Model, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

/// java: `ModelBiped.setRotationAngles`'s literal `-0.5235988F` for the block-held-aloft pose.
///
/// Kept as vanilla's hand-rounded decimal rather than `-FRAC_PI_6`: the two differ in the last
/// `f32` bit, and every angle in this file is a transcription, not a derivation.
#[allow(clippy::approx_constant)]
const HELD_BLOCK_ARM_YAW: f32 = -0.523_598_8;

/// Clears a part and rebuilds it with a new texture offset, box and pivot.
///
/// java: the `this.bipedRightArm = new ModelRenderer(...)` pattern in `ModelSkeleton` and
/// `ModelPlayer`, which leaves the previous part unreferenced.
#[allow(clippy::too_many_arguments)]
fn rebuild(
    model: &mut Model,
    id: PartId,
    u: i32,
    v: i32,
    mirror: bool,
    b: [f32; 3],
    size: [i32; 3],
    delta: f32,
    at: [f32; 3],
) {
    let p = model.get_mut(id);
    p.cubes.clear();
    p.texture_offset = [u, v];
    p.mirror = mirror;
    p.add_box(b[0], b[1], b[2], size[0], size[1], size[2], delta);
    p.set_rotation_point(at[0], at[1], at[2]);
}

/// The seven parts every biped has.
#[derive(Clone, Copy, Debug)]
pub struct BipedParts {
    /// java: `bipedHead`
    pub head: PartId,
    /// java: `bipedHeadwear` — the hat layer, pinned to the head every frame.
    pub headwear: PartId,
    /// java: `bipedBody`
    pub body: PartId,
    /// java: `bipedRightArm`
    pub right_arm: PartId,
    /// java: `bipedLeftArm`
    pub left_arm: PartId,
    /// java: `bipedRightLeg`
    pub right_leg: PartId,
    /// java: `bipedLeftLeg`
    pub left_leg: PartId,
}

/// The humanoid base model.
///
/// java: `ModelBiped`
#[derive(Clone, Debug)]
pub struct BipedModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the seven parts.
    pub parts: BipedParts,
}

impl BipedModel {
    /// java: `ModelBiped(float modelSize)` — 64x32, no vertical offset.
    #[must_use]
    pub fn new(model_size: f32) -> Self {
        Self::with(model_size, 0.0, 64, 32)
    }

    /// java: `ModelBiped(float modelSize, float yOffset, int textureWidth, int textureHeight)`
    #[must_use]
    pub fn with(model_size: f32, y_off: f32, tw: u32, th: u32) -> Self {
        let mut m = Model::new(tw, th);
        let head = m
            .part("head", 0, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, model_size)
            .at(0.0, y_off, 0.0)
            .id();
        let headwear = m
            .part("headwear", 32, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, model_size + 0.5)
            .at(0.0, y_off, 0.0)
            .id();
        let body = m
            .part("body", 16, 16)
            .cube(-4.0, 0.0, -2.0, 8, 12, 4, model_size)
            .at(0.0, y_off, 0.0)
            .id();
        let right_arm = m
            .part("right_arm", 40, 16)
            .cube(-3.0, -2.0, -2.0, 4, 12, 4, model_size)
            .at(-5.0, 2.0 + y_off, 0.0)
            .id();
        let left_arm = m
            .part("left_arm", 40, 16)
            .mirrored()
            .cube(-1.0, -2.0, -2.0, 4, 12, 4, model_size)
            .at(5.0, 2.0 + y_off, 0.0)
            .id();
        let right_leg = m
            .part("right_leg", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size)
            .at(-1.9, 12.0 + y_off, 0.0)
            .id();
        let left_leg = m
            .part("left_leg", 0, 16)
            .mirrored()
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size)
            .at(1.9, 12.0 + y_off, 0.0)
            .id();
        Self {
            model: m,
            parts: BipedParts {
                head,
                headwear,
                body,
                right_arm,
                left_arm,
                right_leg,
                left_leg,
            },
        }
    }

    /// `ModelBiped.setRotationAngles`, statement for statement.
    ///
    /// java: `ModelBiped.java:129-245`
    pub fn pose(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        let age = p.age_in_ticks;
        let e = &p.entity;

        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(q.right_arm).rotate_angle[0] = cos(swing * 0.6662 + PI) * 2.0 * amount * 0.5;
        m.get_mut(q.left_arm).rotate_angle[0] = cos(swing * 0.6662) * 2.0 * amount * 0.5;
        m.get_mut(q.right_arm).rotate_angle[2] = 0.0;
        m.get_mut(q.left_arm).rotate_angle[2] = 0.0;
        m.get_mut(q.right_leg).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
        m.get_mut(q.left_leg).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(q.right_leg).rotate_angle[1] = 0.0;
        m.get_mut(q.left_leg).rotate_angle[1] = 0.0;

        if m.is_riding {
            m.get_mut(q.right_arm).rotate_angle[0] += -(PI / 5.0);
            m.get_mut(q.left_arm).rotate_angle[0] += -(PI / 5.0);
            m.get_mut(q.right_leg).rotate_angle[0] = -(PI * 2.0 / 5.0);
            m.get_mut(q.left_leg).rotate_angle[0] = -(PI * 2.0 / 5.0);
            m.get_mut(q.right_leg).rotate_angle[1] = PI / 10.0;
            m.get_mut(q.left_leg).rotate_angle[1] = -(PI / 10.0);
        }

        if e.held_item_left != 0 {
            let a = m.get(q.left_arm).rotate_angle[0];
            m.get_mut(q.left_arm).rotate_angle[0] = a * 0.5 - (PI / 10.0) * e.held_item_left as f32;
        }

        m.get_mut(q.right_arm).rotate_angle[1] = 0.0;
        m.get_mut(q.right_arm).rotate_angle[2] = 0.0;

        match e.held_item_right {
            1 => {
                let a = m.get(q.right_arm).rotate_angle[0];
                m.get_mut(q.right_arm).rotate_angle[0] = a * 0.5 - (PI / 10.0) * 1.0;
            }
            3 => {
                let a = m.get(q.right_arm).rotate_angle[0];
                m.get_mut(q.right_arm).rotate_angle[0] = a * 0.5 - (PI / 10.0) * 3.0;
                // "1.7 Animations" drops the block yaw (off: vanilla's -30°).
                m.get_mut(q.right_arm).rotate_angle[1] =
                    crate::pipeline::old_animations::blocking_arm_yaw(HELD_BLOCK_ARM_YAW);
            }
            _ => {}
        }

        m.get_mut(q.left_arm).rotate_angle[1] = 0.0;

        if m.swing_progress > -9990.0 {
            let f = m.swing_progress;
            let body_y = sin(mc_math::sqrt_f(f) * PI * 2.0) * 0.2;
            m.get_mut(q.body).rotate_angle[1] = body_y;
            m.get_mut(q.right_arm).rotation_point[2] = sin(body_y) * 5.0;
            m.get_mut(q.right_arm).rotation_point[0] = -cos(body_y) * 5.0;
            m.get_mut(q.left_arm).rotation_point[2] = -sin(body_y) * 5.0;
            m.get_mut(q.left_arm).rotation_point[0] = cos(body_y) * 5.0;
            m.get_mut(q.right_arm).rotate_angle[1] += body_y;
            m.get_mut(q.left_arm).rotate_angle[1] += body_y;
            m.get_mut(q.left_arm).rotate_angle[0] += body_y;

            let mut f = 1.0 - m.swing_progress;
            f *= f;
            f *= f;
            f = 1.0 - f;
            let f1 = sin(f * PI);
            let head_x = m.get(q.head).rotate_angle[0];
            let f2 = sin(m.swing_progress * PI) * -(head_x - 0.7) * 0.75;
            let ra = m.get(q.right_arm).rotate_angle[0];
            m.get_mut(q.right_arm).rotate_angle[0] =
                (f64::from(ra) - (f64::from(f1) * 1.2 + f64::from(f2))) as f32;
            m.get_mut(q.right_arm).rotate_angle[1] += body_y * 2.0;
            m.get_mut(q.right_arm).rotate_angle[2] += sin(m.swing_progress * PI) * -0.4;
        }

        if e.sneaking {
            m.get_mut(q.body).rotate_angle[0] = 0.5;
            m.get_mut(q.right_arm).rotate_angle[0] += 0.4;
            m.get_mut(q.left_arm).rotate_angle[0] += 0.4;
            m.get_mut(q.right_leg).rotation_point[2] = 4.0;
            m.get_mut(q.left_leg).rotation_point[2] = 4.0;
            m.get_mut(q.right_leg).rotation_point[1] = 9.0;
            m.get_mut(q.left_leg).rotation_point[1] = 9.0;
            m.get_mut(q.head).rotation_point[1] = 1.0;
        } else {
            m.get_mut(q.body).rotate_angle[0] = 0.0;
            m.get_mut(q.right_leg).rotation_point[2] = 0.1;
            m.get_mut(q.left_leg).rotation_point[2] = 0.1;
            m.get_mut(q.right_leg).rotation_point[1] = 12.0;
            m.get_mut(q.left_leg).rotation_point[1] = 12.0;
            m.get_mut(q.head).rotation_point[1] = 0.0;
        }

        let idle_z = cos(age * 0.09) * 0.05 + 0.05;
        let idle_x = sin(age * 0.067) * 0.05;
        m.get_mut(q.right_arm).rotate_angle[2] += idle_z;
        m.get_mut(q.left_arm).rotate_angle[2] -= idle_z;
        m.get_mut(q.right_arm).rotate_angle[0] += idle_x;
        m.get_mut(q.left_arm).rotate_angle[0] -= idle_x;

        if e.aimed_bow {
            let head_y = m.get(q.head).rotate_angle[1];
            let head_x = m.get(q.head).rotate_angle[0];
            m.get_mut(q.right_arm).rotate_angle[2] = 0.0;
            m.get_mut(q.left_arm).rotate_angle[2] = 0.0;
            m.get_mut(q.right_arm).rotate_angle[1] = -0.1 + head_y;
            m.get_mut(q.left_arm).rotate_angle[1] = 0.1 + head_y + 0.4;
            m.get_mut(q.right_arm).rotate_angle[0] = -(PI / 2.0) + head_x;
            m.get_mut(q.left_arm).rotate_angle[0] = -(PI / 2.0) + head_x;
            m.get_mut(q.right_arm).rotate_angle[2] += idle_z;
            m.get_mut(q.left_arm).rotate_angle[2] -= idle_z;
            m.get_mut(q.right_arm).rotate_angle[0] += idle_x;
            m.get_mut(q.left_arm).rotate_angle[0] -= idle_x;
        }

        m.copy_angles(q.head, q.headwear);
    }

    /// Poses this biped with `ModelZombie`'s arm-forward tail.
    ///
    /// `LayerBipedArmor` does not copy the main model's already-computed angles: it copies the
    /// shared flags and calls `setRotationAngles` on its own `ModelZombie`/`ModelSkeleton`
    /// instance.  Keeping that tail here lets the armour cache use the same compact biped storage
    /// while still taking the zombie's distinct arm pose.
    ///
    /// java: `ModelZombie.setRotationAngles` (and the identical
    /// `ModelZombieVillager.setRotationAngles`).
    pub fn pose_as_zombie(&mut self, p: &AnimParams) {
        self.pose(p);
        zombie_arms(&mut self.model, &self.parts, p.age_in_ticks);
    }

    /// The `ModelBiped.render` draw order, including the baby split and the sneak nudge.
    ///
    /// java: `ModelBiped.render`
    pub fn draw(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let scale = p.scale;
        if self.model.is_child {
            // java: scale(1.5/2) then translate(0, 16 * scale, 0)
            let head_m = base
                * Mat4::from_scale(Vec3::splat(0.75))
                * Mat4::from_translation(Vec3::new(0.0, 16.0 * scale, 0.0));
            self.model.emit_part(q.head, scale, head_m, out);
            let body_m = base
                * Mat4::from_scale(Vec3::splat(0.5))
                * Mat4::from_translation(Vec3::new(0.0, 24.0 * scale, 0.0));
            self.model.emit_some(
                &[
                    q.body,
                    q.right_arm,
                    q.left_arm,
                    q.right_leg,
                    q.left_leg,
                    q.headwear,
                ],
                scale,
                body_m,
                out,
            );
        } else {
            let m = if p.entity.sneaking {
                base * Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
            } else {
                base
            };
            self.model.emit_some(
                &[
                    q.head,
                    q.body,
                    q.right_arm,
                    q.left_arm,
                    q.right_leg,
                    q.left_leg,
                    q.headwear,
                ],
                scale,
                m,
                out,
            );
        }
    }

    /// The matrix a baby's body parts are drawn under, for layers that must follow them.
    #[must_use]
    pub fn child_body_matrix(base: Mat4, scale: f32) -> Mat4 {
        base * Mat4::from_scale(Vec3::splat(0.5))
            * Mat4::from_translation(Vec3::new(0.0, 24.0 * scale, 0.0))
    }
}

impl EntityModel for BipedModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelBiped.postRenderArm` is `bipedRightArm.postRender(scale)`.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.parts.right_arm)
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.pose(p);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.draw(p, base, out);
    }
}

/// The zombie arm-forward pose shared by `ModelZombie` and `ModelZombieVillager`.
///
/// java: `ModelZombie.setRotationAngles` tail (identical text in `ModelZombieVillager`).
fn zombie_arms(model: &mut Model, q: &BipedParts, age: f32) {
    let sp = model.swing_progress;
    let f = sin(sp * PI);
    let f1 = sin((1.0 - (1.0 - sp) * (1.0 - sp)) * PI);
    model.get_mut(q.right_arm).rotate_angle[2] = 0.0;
    model.get_mut(q.left_arm).rotate_angle[2] = 0.0;
    model.get_mut(q.right_arm).rotate_angle[1] = -(0.1 - f * 0.6);
    model.get_mut(q.left_arm).rotate_angle[1] = 0.1 - f * 0.6;
    model.get_mut(q.right_arm).rotate_angle[0] = -(PI / 2.0);
    model.get_mut(q.left_arm).rotate_angle[0] = -(PI / 2.0);
    model.get_mut(q.right_arm).rotate_angle[0] -= f * 1.2 - f1 * 0.4;
    model.get_mut(q.left_arm).rotate_angle[0] -= f * 1.2 - f1 * 0.4;
    let idle_z = cos(age * 0.09) * 0.05 + 0.05;
    let idle_x = sin(age * 0.067) * 0.05;
    model.get_mut(q.right_arm).rotate_angle[2] += idle_z;
    model.get_mut(q.left_arm).rotate_angle[2] -= idle_z;
    model.get_mut(q.right_arm).rotate_angle[0] += idle_x;
    model.get_mut(q.left_arm).rotate_angle[0] -= idle_x;
}

/// java: `ModelZombie`
#[derive(Clone, Debug)]
pub struct ZombieModel {
    /// The underlying biped.
    pub biped: BipedModel,
}

impl ZombieModel {
    /// java: `ModelZombie()` — `super(0, 0, 64, 64)`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            biped: BipedModel::with(0.0, 0.0, 64, 64),
        }
    }
}

impl Default for ZombieModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for ZombieModel {
    fn model(&self) -> &Model {
        &self.biped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.biped.model
    }
    /// java: inherited from `ModelBiped.postRenderArm`.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.biped.parts.right_arm)
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.biped.pose(p);
        let q = self.biped.parts;
        zombie_arms(&mut self.biped.model, &q, p.age_in_ticks);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.biped.draw(p, base, out);
    }
}

/// java: `ModelZombieVillager`
#[derive(Clone, Debug)]
pub struct ZombieVillagerModel {
    /// The underlying biped, with the villager's brow-and-nose head.
    pub biped: BipedModel,
}

impl ZombieVillagerModel {
    /// java: `ModelZombieVillager()` — `super(0, 0, 64, 64)` and the two-box head.
    #[must_use]
    pub fn new() -> Self {
        let mut biped = BipedModel::with(0.0, 0.0, 64, 64);
        let head = biped.parts.head;
        let p = biped.model.get_mut(head);
        p.cubes.clear();
        p.texture_offset = [0, 32];
        p.add_box(-4.0, -10.0, -4.0, 8, 10, 8, 0.0);
        p.texture_offset = [24, 32];
        p.add_box(-1.0, -3.0, -6.0, 2, 4, 2, 0.0);
        p.set_rotation_point(0.0, 0.0, 0.0);
        Self { biped }
    }

    /// The `LayerVillagerArmor` form of this model.
    ///
    /// Unlike the body model, its replacement head is the ordinary eight-unit armour cube, but
    /// it starts two model units higher so a helmet caps the ten-unit villager head.  The rest is
    /// the normal `ModelBiped` armour tree.
    ///
    /// java: `ModelZombieVillager(float, float, true)`, constructed by
    /// `LayerVillagerArmor.initArmor` at 0.5 and 1.0 inflation.
    #[must_use]
    pub fn armor(model_size: f32) -> BipedModel {
        let mut biped = BipedModel::with(model_size, 0.0, 64, 32);
        rebuild(
            &mut biped.model,
            biped.parts.head,
            0,
            0,
            false,
            [-4.0, -10.0, -4.0],
            [8, 8, 8],
            model_size,
            [0.0, 0.0, 0.0],
        );
        biped
    }
}

impl Default for ZombieVillagerModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for ZombieVillagerModel {
    fn model(&self) -> &Model {
        &self.biped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.biped.model
    }
    /// java: inherited from `ModelBiped.postRenderArm`.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.biped.parts.right_arm)
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.biped.pose(p);
        let q = self.biped.parts;
        zombie_arms(&mut self.biped.model, &q, p.age_in_ticks);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.biped.draw(p, base, out);
    }
}

/// java: `ModelSkeleton` — a `ModelZombie` with 2x2 limbs. Its inherited call chain applies
/// `ModelBiped`'s `aimedBow` pose first, then `ModelZombie`'s final arm-forward tail; Skeleton
/// does not replace that inherited zombie tail.
#[derive(Clone, Debug)]
pub struct SkeletonModel {
    /// The underlying biped.
    pub biped: BipedModel,
}

impl SkeletonModel {
    /// java: `ModelSkeleton(0.0F, false)`
    #[must_use]
    pub fn new() -> Self {
        Self::sized(0.0)
    }

    /// java: `ModelSkeleton(float size, false)`
    #[must_use]
    pub fn sized(size: f32) -> Self {
        let mut biped = BipedModel::with(size, 0.0, 64, 32);
        let q = biped.parts;
        let m = &mut biped.model;
        rebuild(
            m,
            q.right_arm,
            40,
            16,
            false,
            [-1.0, -2.0, -1.0],
            [2, 12, 2],
            size,
            [-5.0, 2.0, 0.0],
        );
        rebuild(
            m,
            q.left_arm,
            40,
            16,
            true,
            [-1.0, -2.0, -1.0],
            [2, 12, 2],
            size,
            [5.0, 2.0, 0.0],
        );
        rebuild(
            m,
            q.right_leg,
            0,
            16,
            false,
            [-1.0, 0.0, -1.0],
            [2, 12, 2],
            size,
            [-2.0, 12.0, 0.0],
        );
        rebuild(
            m,
            q.left_leg,
            0,
            16,
            true,
            [-1.0, 0.0, -1.0],
            [2, 12, 2],
            size,
            [2.0, 12.0, 0.0],
        );
        Self { biped }
    }
}

impl Default for SkeletonModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SkeletonModel {
    fn model(&self) -> &Model {
        &self.biped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.biped.model
    }
    /// java: inherited from `ModelBiped.postRenderArm`.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.biped.parts.right_arm)
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.biped.pose(p);
        let q = self.biped.parts;
        zombie_arms(&mut self.biped.model, &q, p.age_in_ticks);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.biped.draw(p, base, out);
    }
}

/// The five "wear" overlay parts plus the cape.
#[derive(Clone, Copy, Debug)]
pub struct PlayerWear {
    /// java: `bipedLeftArmwear`
    pub left_arm: PartId,
    /// java: `bipedRightArmwear`
    pub right_arm: PartId,
    /// java: `bipedLeftLegwear`
    pub left_leg: PartId,
    /// java: `bipedRightLegwear`
    pub right_leg: PartId,
    /// java: `bipedBodyWear`
    pub body: PartId,
    /// java: `bipedCape` — drawn by [`crate::entity::layers`], not by the model.
    pub cape: PartId,
    /// java: `bipedDeadmau5Head`
    pub deadmau5: PartId,
}

/// java: `ModelPlayer`
#[derive(Clone, Debug)]
pub struct PlayerModel {
    /// The underlying biped.
    pub biped: BipedModel,
    /// The 64x64 overlay parts.
    pub wear: PlayerWear,
    /// java: `ModelPlayer.smallArms` — the Alex/slim arm geometry.
    pub small_arms: bool,
}

impl PlayerModel {
    /// java: `ModelPlayer(float modelSize, boolean smallArms)`
    #[must_use]
    pub fn new(model_size: f32, small_arms: bool) -> Self {
        let mut biped = BipedModel::with(model_size, 0.0, 64, 64);
        let mut q = biped.parts;
        let m = &mut biped.model;

        let deadmau5 = m
            .part("deadmau5", 24, 0)
            .cube(-3.0, -6.0, -1.0, 6, 6, 1, model_size)
            .hidden()
            .id();
        // java: the cape is 64x32 even though the player model is 64x64.
        let cape = m
            .part("cape", 0, 0)
            .tex_size(64, 32)
            .cube(-5.0, 0.0, -1.0, 10, 16, 1, model_size)
            .hidden()
            .id();

        let (left_arm_w, right_arm_w);
        if small_arms {
            // Java assigns fresh ModelRenderers and leaves both inherited arms in boxList.
            // Preserve those orphaned entries because LayerArrow samples that list by index.
            m.get_mut(q.left_arm).name = "left_arm_orphan";
            m.get_mut(q.right_arm).name = "right_arm_orphan";
            q.left_arm = m
                .part("left_arm", 32, 48)
                .cube(-1.0, -2.0, -2.0, 3, 12, 4, model_size)
                .at(5.0, 2.5, 0.0)
                .id();
            q.right_arm = m
                .part("right_arm", 40, 16)
                .cube(-2.0, -2.0, -2.0, 3, 12, 4, model_size)
                .at(-5.0, 2.5, 0.0)
                .id();
            left_arm_w = m
                .part("left_armwear", 48, 48)
                .cube(-1.0, -2.0, -2.0, 3, 12, 4, model_size + 0.25)
                .at(5.0, 2.5, 0.0)
                .id();
            right_arm_w = m
                .part("right_armwear", 40, 32)
                .cube(-2.0, -2.0, -2.0, 3, 12, 4, model_size + 0.25)
                .at(-5.0, 2.5, 10.0)
                .id();
        } else {
            // Classic replaces only the inherited left arm; its inherited right arm stays live.
            m.get_mut(q.left_arm).name = "left_arm_orphan";
            q.left_arm = m
                .part("left_arm", 32, 48)
                .cube(-1.0, -2.0, -2.0, 4, 12, 4, model_size)
                .at(5.0, 2.0, 0.0)
                .id();
            left_arm_w = m
                .part("left_armwear", 48, 48)
                .cube(-1.0, -2.0, -2.0, 4, 12, 4, model_size + 0.25)
                .at(5.0, 2.0, 0.0)
                .id();
            right_arm_w = m
                .part("right_armwear", 40, 32)
                .cube(-3.0, -2.0, -2.0, 4, 12, 4, model_size + 0.25)
                .at(-5.0, 2.0, 10.0)
                .id();
        }

        // Both variants replace the inherited left leg after their arm-wear fields.
        m.get_mut(q.left_leg).name = "left_leg_orphan";
        q.left_leg = m
            .part("left_leg", 16, 48)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size)
            .at(1.9, 12.0, 0.0)
            .id();
        let left_leg_w = m
            .part("left_legwear", 0, 48)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size + 0.25)
            .at(1.9, 12.0, 0.0)
            .id();
        let right_leg_w = m
            .part("right_legwear", 0, 32)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size + 0.25)
            .at(-1.9, 12.0, 0.0)
            .id();
        let body_w = m
            .part("bodywear", 16, 32)
            .cube(-4.0, 0.0, -2.0, 8, 12, 4, model_size + 0.25)
            .at(0.0, 0.0, 0.0)
            .id();

        biped.parts = q;
        Self {
            biped,
            wear: PlayerWear {
                left_arm: left_arm_w,
                right_arm: right_arm_w,
                left_leg: left_leg_w,
                right_leg: right_leg_w,
                body: body_w,
                cape,
                deadmau5,
            },
            small_arms,
        }
    }
}

impl EntityModel for PlayerModel {
    fn model(&self) -> &Model {
        &self.biped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.biped.model
    }
    /// java: inherited from `ModelBiped.postRenderArm`.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.biped.parts.right_arm)
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.biped.pose(p);
        let q = self.biped.parts;
        let w = self.wear;
        let m = &mut self.biped.model;
        m.copy_angles(q.left_leg, w.left_leg);
        m.copy_angles(q.right_leg, w.right_leg);
        m.copy_angles(q.left_arm, w.left_arm);
        m.copy_angles(q.right_arm, w.right_arm);
        m.copy_angles(q.body, w.body);
        m.get_mut(w.cape).rotation_point[1] = if p.entity.sneaking { 2.0 } else { 0.0 };
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.biped.draw(p, base, out);
        let w = self.wear;
        let order = [w.left_leg, w.right_leg, w.left_arm, w.right_arm, w.body];
        if self.biped.model.is_child {
            let m = BipedModel::child_body_matrix(base, p.scale);
            self.biped.model.emit_some(&order, p.scale, m, out);
        } else {
            let m = if p.entity.sneaking {
                base * Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
            } else {
                base
            };
            self.biped.model.emit_some(&order, p.scale, m, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::anim::EntityAnimState;

    #[test]
    fn biped_geometry_matches_the_java_constructor() {
        let b = BipedModel::new(0.0);
        let m = &b.model;
        assert_eq!(m.get(b.parts.head).texture_offset, [0, 0]);
        assert_eq!(m.get(b.parts.head).cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(m.get(b.parts.head).cubes[0].pos2, [4.0, 0.0, 4.0]);
        assert_eq!(m.get(b.parts.body).cubes[0].pos2, [4.0, 12.0, 2.0]);
        assert_eq!(m.get(b.parts.right_arm).rotation_point, [-5.0, 2.0, 0.0]);
        assert_eq!(m.get(b.parts.left_leg).rotation_point, [1.9, 12.0, 0.0]);
        assert!(m.get(b.parts.left_arm).mirror);
        assert!(!m.get(b.parts.right_arm).mirror);
        // The hat layer is the head inflated by 0.5 on every side.
        let hat = &m.get(b.parts.headwear).cubes[0];
        assert_eq!(hat.quads[4].verts[1].pos, [-4.5, -8.5, -4.5]);
    }

    /// Hand-computed against `ModelBiped.setRotationAngles` with the vanilla LUT.
    #[test]
    fn walking_arms_and_legs_swing_in_antiphase() {
        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.limb_swing = 5.0;
        p.limb_swing_amount = 1.0;
        b.model.swing_progress = -9999.0; // java: swingProgress <= -9990 disables the attack pose
        b.pose(&p);

        let want_right_arm = cos(5.0 * 0.6662 + PI) * 2.0 * 1.0 * 0.5;
        let want_left_arm = cos(5.0 * 0.6662) * 2.0 * 1.0 * 0.5;
        let want_right_leg = cos(5.0 * 0.6662) * 1.4;
        // The idle sway is added on top of the swing; age = 0 so sin(0) = 0 for X.
        let got_ra = b.model.get(b.parts.right_arm).rotate_angle[0];
        assert!(
            (got_ra - want_right_arm).abs() < 1e-6,
            "{got_ra} vs {want_right_arm}"
        );
        assert!((b.model.get(b.parts.left_arm).rotate_angle[0] - want_left_arm).abs() < 1e-6);
        assert!((b.model.get(b.parts.right_leg).rotate_angle[0] - want_right_leg).abs() < 1e-6);
        // Antiphase: the right arm and right leg move opposite ways.
        assert!(want_right_arm * want_right_leg <= 0.0);
    }

    /// java: `ModelBiped.setRotationAngles`'s `isRiding` branch (`:149-158`) fixes both legs in
    /// the seated pose and lowers both arms by `PI / 5`. `EntityModel::pose_and_render` copies
    /// the remote S1B-backed `EntityAnimState::riding` flag onto the shared model before posing.
    #[test]
    fn riding_state_reaches_the_biped_seated_pose() {
        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.entity.riding = true;
        let mut out = Vec::new();
        b.pose_and_render(&p, Mat4::IDENTITY, &mut out);

        assert!((b.model.get(b.parts.right_leg).rotate_angle[0] + PI * 2.0 / 5.0).abs() < 1e-6);
        assert!((b.model.get(b.parts.left_leg).rotate_angle[0] + PI * 2.0 / 5.0).abs() < 1e-6);
        assert!((b.model.get(b.parts.right_leg).rotate_angle[1] - PI / 10.0).abs() < 1e-6);
        assert!((b.model.get(b.parts.left_leg).rotate_angle[1] + PI / 10.0).abs() < 1e-6);
        // age=0 has no X idle sway, so both arms expose the exact seated offset.
        assert!((b.model.get(b.parts.right_arm).rotate_angle[0] + PI / 5.0).abs() < 1e-6);
        assert!((b.model.get(b.parts.left_arm).rotate_angle[0] + PI / 5.0).abs() < 1e-6);
    }

    #[test]
    fn head_follows_yaw_and_pitch_in_radians() {
        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.net_head_yaw = 90.0;
        p.head_pitch = -45.0;
        b.model.swing_progress = -9999.0;
        b.pose(&p);
        let h = b.model.get(b.parts.head);
        assert!((h.rotate_angle[1] - PI / 2.0).abs() < 1e-6);
        assert!((h.rotate_angle[0] + PI / 4.0).abs() < 1e-6);
        // The hat layer tracks the head exactly.
        assert_eq!(b.model.get(b.parts.headwear).rotate_angle, h.rotate_angle);
    }

    /// Java bytecode for `ModelBiped.setRotationAngles` loads `57.295776F` and executes `fdiv`.
    /// Find a value where the old reciprocal multiply is observably different, then drive it
    /// through the real model so this test catches a caller that bypasses `model_radians`.
    #[test]
    fn biped_head_uses_java_division_at_a_reciprocal_rounding_boundary() {
        use crate::entity::anim::{DEG_PER_RAD, DEG_TO_RAD};

        let degrees = (-1440_i32..=1440)
            .map(|step| step as f32 * 0.25)
            .find(|&d| d * DEG_TO_RAD != d / DEG_PER_RAD)
            .expect("the reciprocal-multiply mutant must disagree in this range");

        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.net_head_yaw = degrees;
        b.model.swing_progress = -9999.0;
        b.pose(&p);

        let actual = b.model.get(b.parts.head).rotate_angle[1];
        assert_eq!(actual, degrees / DEG_PER_RAD);
        assert_ne!(
            actual,
            degrees * DEG_TO_RAD,
            "negative control did not bite"
        );
    }

    #[test]
    fn idle_sway_is_present_at_rest() {
        // java: rotateAngleZ += cos(age * 0.09) * 0.05 + 0.05, always, even standing still.
        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.age_in_ticks = 0.0;
        b.model.swing_progress = -9999.0;
        b.pose(&p);
        let z = b.model.get(b.parts.right_arm).rotate_angle[2];
        assert!((z - 0.1).abs() < 1e-6, "cos(0)*0.05 + 0.05 = 0.1, got {z}");
        assert!((b.model.get(b.parts.left_arm).rotate_angle[2] + 0.1).abs() < 1e-6);
    }

    #[test]
    fn sneaking_moves_the_legs_forward_and_bends_the_body() {
        let mut b = BipedModel::new(0.0);
        let mut p = AnimParams::still();
        p.entity = EntityAnimState {
            sneaking: true,
            ..Default::default()
        };
        b.model.swing_progress = -9999.0;
        b.pose(&p);
        assert_eq!(b.model.get(b.parts.body).rotate_angle[0], 0.5);
        assert_eq!(
            b.model.get(b.parts.right_leg).rotation_point,
            [-1.9, 9.0, 4.0]
        );
        assert_eq!(b.model.get(b.parts.head).rotation_point[1], 1.0);
    }

    #[test]
    fn skeleton_limbs_are_two_wide() {
        let s = SkeletonModel::new();
        let arm = &s.biped.model.get(s.biped.parts.right_arm).cubes[0];
        assert_eq!(arm.pos1, [-1.0, -2.0, -1.0]);
        assert_eq!(arm.pos2, [1.0, 10.0, 1.0]);
        assert_eq!(
            s.biped.model.get(s.biped.parts.right_leg).rotation_point,
            [-2.0, 12.0, 0.0]
        );
        // Only one cube: the rebuild must not leave the 4x12x4 box behind.
        assert_eq!(s.biped.model.get(s.biped.parts.right_arm).cubes.len(), 1);
    }

    #[test]
    fn slim_player_arms_are_three_wide_and_half_a_pixel_higher() {
        let slim = PlayerModel::new(0.0, true);
        let classic = PlayerModel::new(0.0, false);
        let sa = &slim.biped.model.get(slim.biped.parts.left_arm).cubes[0];
        let ca = &classic.biped.model.get(classic.biped.parts.left_arm).cubes[0];
        assert_eq!(sa.pos2[0] - sa.pos1[0], 3.0);
        assert_eq!(ca.pos2[0] - ca.pos1[0], 4.0);
        assert_eq!(
            slim.biped
                .model
                .get(slim.biped.parts.left_arm)
                .rotation_point[1],
            2.5
        );
        assert_eq!(
            classic
                .biped
                .model
                .get(classic.biped.parts.left_arm)
                .rotation_point[1],
            2.0
        );
        // The classic model leaves the right arm at ModelBiped's own 40,16 offset.
        assert_eq!(
            classic
                .biped
                .model
                .get(classic.biped.parts.right_arm)
                .texture_offset,
            [40, 16]
        );
        // ModelBase.boxList keeps replaced fields in construction order. LayerArrow's seeded
        // random selection depends on these exact arena indices, including the orphaned limbs.
        assert_eq!(classic.biped.model.parts().len(), 16);
        assert_eq!(slim.biped.model.parts().len(), 17);
        assert_eq!(classic.biped.parts.left_arm.index(), 9);
        assert_eq!(classic.biped.parts.left_leg.index(), 12);
        assert_eq!(slim.biped.parts.left_arm.index(), 9);
        assert_eq!(slim.biped.parts.right_arm.index(), 10);
        assert_eq!(slim.biped.parts.left_leg.index(), 13);
    }

    #[test]
    fn wear_layers_track_their_base_parts() {
        let mut pm = PlayerModel::new(0.0, false);
        let mut p = AnimParams::still();
        p.limb_swing = 3.0;
        p.limb_swing_amount = 1.0;
        pm.set_rotation_angles(&p);
        for (base, wear) in [
            (pm.biped.parts.left_leg, pm.wear.left_leg),
            (pm.biped.parts.right_arm, pm.wear.right_arm),
            (pm.biped.parts.body, pm.wear.body),
        ] {
            assert_eq!(
                pm.biped.model.get(base).rotate_angle,
                pm.biped.model.get(wear).rotate_angle
            );
            assert_eq!(
                pm.biped.model.get(base).rotation_point,
                pm.biped.model.get(wear).rotation_point,
                "copyModelAngles also copies the pivot, which is what un-does the z=10 quirk"
            );
        }
    }

    #[test]
    fn cape_and_deadmau5_are_not_drawn_by_the_model_itself() {
        let mut pm = PlayerModel::new(0.0, false);
        let mut p = AnimParams::still();
        p.entity.child = false;
        pm.model_mut().is_child = false;
        let mut out = Vec::new();
        pm.pose_and_render(&p, Mat4::IDENTITY, &mut out);
        // 7 biped parts + 5 wear parts, one box each, 6 faces, 6 verts.
        assert_eq!(out.len(), 12 * 6 * 6);
    }
}
