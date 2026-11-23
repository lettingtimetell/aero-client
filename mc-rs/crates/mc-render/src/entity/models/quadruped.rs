//! The four-legged farm animals.
//!
//! java: `ModelQuadruped.java` (the base), `ModelPig.java`, `ModelCow.java`,
//! `ModelSheep2.java` (the sheared body) and `ModelSheep1.java` (the wool layer).
//!
//! Ported here:
//!
//! | Rust | Java |
//! |---|---|
//! | [`QuadrupedModel`] | `ModelQuadruped` |
//! | [`PigModel`] | `ModelPig` |
//! | [`CowModel`] | `ModelCow` |
//! | [`SheepModel`] | `ModelSheep2` — the body, drawn by `RenderSheep` |
//! | [`SheepWoolModel`] | `ModelSheep1` — the fleece, drawn by `LayerSheepWool` |
//!
//! Everything follows the shape `biped.rs` sets out: one Rust type per Java class holding a
//! [`Model`] plus the [`PartId`]s the Java kept as fields, a constructor that replays the Java
//! constructor's `addBox`/`setRotationPoint` calls in order, and a
//! [`EntityModel::set_rotation_angles`] that replays the Java method statement for statement.
//!
//! Three things recur:
//!
//! * `MathHelper.cos` is vanilla's 65536-entry LUT, so [`mc_math::cos`] it is — never
//!   `f32::cos` (hazard H1).
//! * `ModelCow`, `ModelSheep1` and `ModelSheep2` *replace* inherited `ModelRenderer` fields
//!   (`this.head = new ModelRenderer(this, 0, 0)`), orphaning the superclass's object in
//!   `boxList`. The private `replace` helper empties and re-points the existing part instead,
//!   so the arena keeps no undrawn leftover.
//! * `ModelQuadruped.render` splits into two matrices for babies — the head gets a plain
//!   translate, the body and legs get a half-scale — so [`QuadrupedModel`] overrides
//!   [`EntityModel::render`] rather than using the default "emit every root".

use glam::{Mat4, Vec3};
use mc_math::cos;

use crate::entity::anim::{model_radians, AnimParams, EntityModel};
use crate::entity::model::{Model, ModelPart, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

/// Empties a part and re-points its texture offset, as `this.x = new ModelRenderer(this, u, v)`
/// does in a `ModelQuadruped` subclass.
///
/// The Java allocates a fresh renderer and drops the inherited one on the floor; it stays in
/// `ModelBase.boxList` but nothing ever draws it. Rebuilding in place is the same geometry with
/// no orphan, and it keeps the [`PartId`]s the base constructor handed out valid.
fn replace(model: &mut Model, id: PartId, u: i32, v: i32) -> &mut ModelPart {
    let p = model.get_mut(id);
    p.cubes.clear();
    p.texture_offset = [u, v];
    // A brand-new `ModelRenderer` starts unmirrored, unrotated and pivoted at the origin.
    p.mirror = false;
    p.rotation_point = [0.0; 3];
    p.rotate_angle = [0.0; 3];
    p
}

/// The six parts every quadruped has.
///
/// java: the `head`/`body`/`leg1`..`leg4` fields of `ModelQuadruped`, in declaration order.
#[derive(Clone, Copy, Debug)]
pub struct QuadrupedParts {
    /// java: `head`
    pub head: PartId,
    /// java: `body`
    pub body: PartId,
    /// java: `leg1` — rear left (`+Z`, `-X`).
    pub leg1: PartId,
    /// java: `leg2` — rear right (`+Z`, `+X`).
    pub leg2: PartId,
    /// java: `leg3` — front left (`-Z`, `-X`).
    pub leg3: PartId,
    /// java: `leg4` — front right (`-Z`, `+X`).
    pub leg4: PartId,
}

/// The four-legged base model.
///
/// java: `ModelQuadruped`
#[derive(Clone, Debug)]
pub struct QuadrupedModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the six parts.
    pub parts: QuadrupedParts,
    /// How far a baby's head is pushed down before drawing. java: `childYOffset`
    pub child_y_offset: f32,
    /// How far a baby's head is pushed back before drawing. java: `childZOffset`
    pub child_z_offset: f32,
}

impl QuadrupedModel {
    /// java: `ModelQuadruped(int legYIn, float p_i1153_2_)`
    ///
    /// `leg_y` is both the leg boxes' height and the amount every pivot is raised by, which is
    /// how one constructor serves animals of different heights.
    #[must_use]
    pub fn new(leg_y: i32, size: f32) -> Self {
        let mut m = Model::new(64, 32);

        // java: `public ModelRenderer head = new ModelRenderer(this, 0, 0);` — a field
        // initialiser, so the head is created before the constructor body runs.
        let head = m
            .part("head", 0, 0)
            .cube(-4.0, -4.0, -8.0, 8, 8, 8, size)
            .at(0.0, (18 - leg_y) as f32, -6.0)
            .id();
        let body = m
            .part("body", 28, 8)
            .cube(-5.0, -10.0, -7.0, 10, 16, 8, size)
            .at(0.0, (17 - leg_y) as f32, 2.0)
            .id();
        let leg1 = m
            .part("leg1", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, leg_y, 4, size)
            .at(-3.0, (24 - leg_y) as f32, 7.0)
            .id();
        let leg2 = m
            .part("leg2", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, leg_y, 4, size)
            .at(3.0, (24 - leg_y) as f32, 7.0)
            .id();
        let leg3 = m
            .part("leg3", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, leg_y, 4, size)
            .at(-3.0, (24 - leg_y) as f32, -5.0)
            .id();
        let leg4 = m
            .part("leg4", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, leg_y, 4, size)
            .at(3.0, (24 - leg_y) as f32, -5.0)
            .id();

        Self {
            model: m,
            parts: QuadrupedParts {
                head,
                body,
                leg1,
                leg2,
                leg3,
                leg4,
            },
            // java: `protected float childYOffset = 8.0F;`
            child_y_offset: 8.0,
            // java: `protected float childZOffset = 4.0F;`
            child_z_offset: 4.0,
        }
    }

    /// `ModelQuadruped.setRotationAngles`, statement for statement.
    ///
    /// java: `ModelQuadruped.java:79-89`
    pub fn pose(&mut self, p: &AnimParams) {
        let q = self.parts;
        let m = &mut self.model;
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;

        // java: `float f = (180F / (float)Math.PI);` — assigned and never read. Kept as a note
        // rather than a dead binding.
        m.get_mut(q.head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(q.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(q.body).rotate_angle[0] = PI / 2.0;
        m.get_mut(q.leg1).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
        m.get_mut(q.leg2).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(q.leg3).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(q.leg4).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
    }

    /// The `ModelQuadruped.render` draw order, including the baby split.
    ///
    /// java: `ModelQuadruped.render` — a baby's head is drawn under a plain
    /// `translate(0, childYOffset * scale, childZOffset * scale)` while the body and legs are
    /// drawn at half size under `scale(1/2)` then `translate(0, 24 * scale, 0)`.
    /// `GlStateManager.scale(a)` followed by `.translate(t)` composes as
    /// `base * S(a) * T(t)`.
    pub fn draw(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let scale = p.scale;
        if self.model.is_child {
            // java: `float f = 2.0F;`
            let f = 2.0_f32;
            let head_m = base
                * Mat4::from_translation(Vec3::new(
                    0.0,
                    self.child_y_offset * scale,
                    self.child_z_offset * scale,
                ));
            self.model.emit_part(q.head, scale, head_m, out);
            let body_m = base
                * Mat4::from_scale(Vec3::splat(1.0 / f))
                * Mat4::from_translation(Vec3::new(0.0, 24.0 * scale, 0.0));
            self.model.emit_some(
                &[q.body, q.leg1, q.leg2, q.leg3, q.leg4],
                scale,
                body_m,
                out,
            );
        } else {
            self.model.emit_some(
                &[q.head, q.body, q.leg1, q.leg2, q.leg3, q.leg4],
                scale,
                base,
                out,
            );
        }
    }
}

impl EntityModel for QuadrupedModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.pose(p);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.draw(p, base, out);
    }
}

/// java: `ModelPig`
#[derive(Clone, Debug)]
pub struct PigModel {
    /// The underlying quadruped, with the snout added to the head.
    pub quadruped: QuadrupedModel,
}

impl PigModel {
    /// java: `ModelPig()` — `this(0.0F)`.
    #[must_use]
    pub fn new() -> Self {
        Self::sized(0.0)
    }

    /// java: `ModelPig(float p_i1151_1_)` — `super(6, size)` plus the snout.
    #[must_use]
    pub fn sized(size: f32) -> Self {
        let mut q = QuadrupedModel::new(6, size);
        let head = q.parts.head;
        // java: `this.head.setTextureOffset(16, 16).addBox(-2, 0, -9, 4, 3, 1, size);`
        let h = q.model.get_mut(head);
        h.texture_offset = [16, 16];
        h.add_box(-2.0, 0.0, -9.0, 4, 3, 1, size);
        // java: `this.childYOffset = 4.0F;`
        q.child_y_offset = 4.0;
        Self { quadruped: q }
    }
}

impl Default for PigModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for PigModel {
    fn model(&self) -> &Model {
        &self.quadruped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.quadruped.model
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        // java: `ModelPig` does not override `setRotationAngles`.
        self.quadruped.pose(p);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.quadruped.draw(p, base, out);
    }
}

/// java: `ModelCow`
#[derive(Clone, Debug)]
pub struct CowModel {
    /// The underlying quadruped, with horns, an udder and wider-set legs.
    pub quadruped: QuadrupedModel,
}

impl CowModel {
    /// java: `ModelCow()` — `super(12, 0.0F)` and then a near-total rebuild.
    #[must_use]
    pub fn new() -> Self {
        let mut q = QuadrupedModel::new(12, 0.0);
        let ids = q.parts;

        // java: `this.head = new ModelRenderer(this, 0, 0);` and the three boxes that follow.
        let h = replace(&mut q.model, ids.head, 0, 0);
        h.add_box(-4.0, -4.0, -6.0, 8, 8, 6, 0.0);
        h.set_rotation_point(0.0, 4.0, -8.0);
        h.texture_offset = [22, 0];
        h.add_box(-5.0, -5.0, -4.0, 1, 3, 1, 0.0);
        h.texture_offset = [22, 0];
        h.add_box(4.0, -5.0, -4.0, 1, 3, 1, 0.0);

        // java: `this.body = new ModelRenderer(this, 18, 4);`
        let b = replace(&mut q.model, ids.body, 18, 4);
        b.add_box(-6.0, -10.0, -7.0, 12, 18, 10, 0.0);
        b.set_rotation_point(0.0, 5.0, 2.0);
        // java: `setTextureOffset(52, 0).addBox(-2, 2, -8, 4, 6, 1)` — the six-argument
        // `addBox`, which is the seven-argument one with `scaleFactor = 0`.
        b.texture_offset = [52, 0];
        b.add_box(-2.0, 2.0, -8.0, 4, 6, 1, 0.0);

        // java: `--this.leg1.rotationPointX;` and friends. The two
        // `legN.rotationPointZ += 0.0F;` statements vanilla left in are no-ops and are not
        // reproduced as code.
        q.model.get_mut(ids.leg1).rotation_point[0] -= 1.0;
        q.model.get_mut(ids.leg2).rotation_point[0] += 1.0;
        q.model.get_mut(ids.leg3).rotation_point[0] -= 1.0;
        q.model.get_mut(ids.leg4).rotation_point[0] += 1.0;
        q.model.get_mut(ids.leg3).rotation_point[2] -= 1.0;
        q.model.get_mut(ids.leg4).rotation_point[2] -= 1.0;

        // java: `this.childZOffset += 2.0F;`
        q.child_z_offset += 2.0;

        Self { quadruped: q }
    }
}

impl Default for CowModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for CowModel {
    fn model(&self) -> &Model {
        &self.quadruped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.quadruped.model
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        // java: `ModelCow` does not override `setRotationAngles`.
        self.quadruped.pose(p);
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.quadruped.draw(p, base, out);
    }
}

/// The grazing head angle, in radians.
///
/// java: `EntitySheep.getHeadRotationAngleX(partialTicks)` (`:339`), read by
/// `ModelSheep2.setLivingAnimations` (`:79`). Computed in
/// [`SheepAnim::head_rotation_angle_x`](crate::entity::anim::SheepAnim::head_rotation_angle_x),
/// which is the only place that can see both `sheepTimer` and the entity's raw
/// `rotationPitch`, and delivered on
/// [`EntityAnimState::sheep_head_angle_x`](crate::entity::EntityAnimState::sheep_head_angle_x).
///
/// This used to be a stand-in that read the shared `tail_rotation` slot for the eating
/// branch — with **no writer anywhere**, so a sheep never lowered its head — and
/// `AnimParams::head_pitch` for the resting branch. The old doc called that second half
/// "reproduced exactly"; it is not. Vanilla reads the raw `rotationPitch` field, not the
/// interpolated pitch the renderer hands the model.
fn sheep_head_rotation_angle_x(p: &AnimParams) -> f32 {
    p.entity.sheep_head_angle_x
}

/// The grazing head-lowering ramp, `0..1`.
///
/// java: `EntitySheep.getHeadRotationPointY(partialTicks)` (`:353`), which
/// `ModelSheep2.setLivingAnimations` (`:78`) turns into `head.rotationPointY = 6 + it * 9`.
/// Computed in
/// [`SheepAnim::head_rotation_point_y`](crate::entity::anim::SheepAnim::head_rotation_point_y).
fn sheep_head_rotation_point_y(p: &AnimParams) -> f32 {
    p.entity.sheep_head_point_y
}

/// java: `ModelSheep2` — the sheep's body, drawn by `RenderSheep` whether or not it is sheared.
#[derive(Clone, Debug)]
pub struct SheepModel {
    /// The underlying quadruped, with the sheep's narrower head and body.
    pub quadruped: QuadrupedModel,
    /// java: `private float headRotationAngleX` — written by `setLivingAnimations`, read by
    /// `setRotationAngles`.
    head_rotation_angle_x: f32,
}

impl SheepModel {
    /// java: `ModelSheep2()` — `super(12, 0.0F)`, then a new head and body.
    #[must_use]
    pub fn new() -> Self {
        let mut q = QuadrupedModel::new(12, 0.0);
        let ids = q.parts;

        // java: `this.head = new ModelRenderer(this, 0, 0);`
        let h = replace(&mut q.model, ids.head, 0, 0);
        h.add_box(-3.0, -4.0, -6.0, 6, 6, 8, 0.0);
        h.set_rotation_point(0.0, 6.0, -8.0);

        // java: `this.body = new ModelRenderer(this, 28, 8);`
        let b = replace(&mut q.model, ids.body, 28, 8);
        b.add_box(-4.0, -10.0, -7.0, 8, 16, 6, 0.0);
        b.set_rotation_point(0.0, 5.0, 2.0);

        // The legs are inherited from `ModelQuadruped(12, 0.0F)` untouched.
        Self {
            quadruped: q,
            head_rotation_angle_x: 0.0,
        }
    }
}

impl Default for SheepModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SheepModel {
    fn model(&self) -> &Model {
        &self.quadruped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.quadruped.model
    }
    fn set_living_animations(&mut self, p: &AnimParams) {
        // java: `super.setLivingAnimations(...)` — `ModelBase`'s body is empty.
        let head = self.quadruped.parts.head;
        self.quadruped.model.get_mut(head).rotation_point[1] =
            6.0 + sheep_head_rotation_point_y(p) * 9.0;
        self.head_rotation_angle_x = sheep_head_rotation_angle_x(p);
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.quadruped.pose(p);
        let head = self.quadruped.parts.head;
        self.quadruped.model.get_mut(head).rotate_angle[0] = self.head_rotation_angle_x;
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.quadruped.draw(p, base, out);
    }
}

/// java: `ModelSheep1` — the fleece, an inflated copy of the body drawn by `LayerSheepWool`.
#[derive(Clone, Debug)]
pub struct SheepWoolModel {
    /// The underlying quadruped, with every part inflated into wool.
    pub quadruped: QuadrupedModel,
    /// java: `private float headRotationAngleX`
    head_rotation_angle_x: f32,
}

impl SheepWoolModel {
    /// java: `ModelSheep1()` — `super(12, 0.0F)`, then a fatter head, body and four short legs.
    #[must_use]
    pub fn new() -> Self {
        let mut q = QuadrupedModel::new(12, 0.0);
        let ids = q.parts;

        // java: `this.head = new ModelRenderer(this, 0, 0);`
        let h = replace(&mut q.model, ids.head, 0, 0);
        h.add_box(-3.0, -4.0, -4.0, 6, 6, 6, 0.6);
        h.set_rotation_point(0.0, 6.0, -8.0);

        // java: `this.body = new ModelRenderer(this, 28, 8);`
        let b = replace(&mut q.model, ids.body, 28, 8);
        b.add_box(-4.0, -10.0, -7.0, 8, 16, 6, 1.75);
        b.set_rotation_point(0.0, 5.0, 2.0);

        // java: `float f = 0.5F;` — the inflation on every leg. The wool legs are also six
        // deep rather than twelve: only the top half of the leg is fleecy.
        let f = 0.5;
        for (id, x, z) in [
            (ids.leg1, -3.0, 7.0),
            (ids.leg2, 3.0, 7.0),
            (ids.leg3, -3.0, -5.0),
            (ids.leg4, 3.0, -5.0),
        ] {
            let l = replace(&mut q.model, id, 0, 16);
            l.add_box(-2.0, 0.0, -2.0, 4, 6, 4, f);
            l.set_rotation_point(x, 12.0, z);
        }

        Self {
            quadruped: q,
            head_rotation_angle_x: 0.0,
        }
    }
}

impl Default for SheepWoolModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SheepWoolModel {
    fn model(&self) -> &Model {
        &self.quadruped.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.quadruped.model
    }
    fn set_living_animations(&mut self, p: &AnimParams) {
        // java: identical text to `ModelSheep2.setLivingAnimations`.
        let head = self.quadruped.parts.head;
        self.quadruped.model.get_mut(head).rotation_point[1] =
            6.0 + sheep_head_rotation_point_y(p) * 9.0;
        self.head_rotation_angle_x = sheep_head_rotation_angle_x(p);
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.quadruped.pose(p);
        let head = self.quadruped.parts.head;
        self.quadruped.model.get_mut(head).rotate_angle[0] = self.head_rotation_angle_x;
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        // java: `ModelSheep1` itself always draws; the sheared check lives one level up, in
        // `LayerSheepWool.doRenderLayer` — `if (!entitylivingbaseIn.getSheared() &&
        // !entitylivingbaseIn.isInvisible())`. The fleece is only ever instantiated by that
        // layer, so the `getSheared()` half of the gate rides along with the model here; the
        // invisibility half is a property of every layer and belongs to
        // [`crate::entity::layers`].
        if p.entity.sheared {
            return;
        }
        self.quadruped.draw(p, base, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::anim::EntityAnimState;

    /// Every number below is read straight off `ModelQuadruped`'s constructor with
    /// `legYIn = 6` (the pig): pivots are `18 - 6`, `17 - 6` and `24 - 6`.
    #[test]
    fn quadruped_geometry_matches_the_java_constructor() {
        let q = QuadrupedModel::new(6, 0.0);
        let m = &q.model;

        assert_eq!(m.get(q.parts.head).texture_offset, [0, 0]);
        assert_eq!(m.get(q.parts.head).cubes[0].pos1, [-4.0, -4.0, -8.0]);
        assert_eq!(m.get(q.parts.head).cubes[0].pos2, [4.0, 4.0, 0.0]);
        assert_eq!(m.get(q.parts.head).rotation_point, [0.0, 12.0, -6.0]);

        assert_eq!(m.get(q.parts.body).texture_offset, [28, 8]);
        assert_eq!(m.get(q.parts.body).cubes[0].pos1, [-5.0, -10.0, -7.0]);
        assert_eq!(m.get(q.parts.body).cubes[0].pos2, [5.0, 6.0, 1.0]);
        assert_eq!(m.get(q.parts.body).rotation_point, [0.0, 11.0, 2.0]);

        for id in [q.parts.leg1, q.parts.leg2, q.parts.leg3, q.parts.leg4] {
            assert_eq!(m.get(id).texture_offset, [0, 16]);
            assert_eq!(m.get(id).cubes[0].pos1, [-2.0, 0.0, -2.0]);
            // `legYIn` is the box height as well as the pivot offset.
            assert_eq!(m.get(id).cubes[0].pos2, [2.0, 6.0, 2.0]);
            assert_eq!(m.get(id).rotation_point[1], 18.0);
            assert!(!m.get(id).mirror, "no quadruped leg is mirrored");
        }
        assert_eq!(m.get(q.parts.leg1).rotation_point, [-3.0, 18.0, 7.0]);
        assert_eq!(m.get(q.parts.leg2).rotation_point, [3.0, 18.0, 7.0]);
        assert_eq!(m.get(q.parts.leg3).rotation_point, [-3.0, 18.0, -5.0]);
        assert_eq!(m.get(q.parts.leg4).rotation_point, [3.0, 18.0, -5.0]);

        assert_eq!(q.child_y_offset, 8.0);
        assert_eq!(q.child_z_offset, 4.0);
        assert_eq!(m.parts().len(), 6);
        assert_eq!(
            m.roots().len(),
            6,
            "no quadruped part is parented to another"
        );
    }

    /// `legYIn = 12` (cow and sheep) raises every pivot by six and doubles the leg height.
    #[test]
    fn leg_y_shifts_every_pivot_and_sizes_the_legs() {
        let q = QuadrupedModel::new(12, 0.0);
        let m = &q.model;
        assert_eq!(m.get(q.parts.head).rotation_point, [0.0, 6.0, -6.0]);
        assert_eq!(m.get(q.parts.body).rotation_point, [0.0, 5.0, 2.0]);
        assert_eq!(m.get(q.parts.leg1).rotation_point, [-3.0, 12.0, 7.0]);
        assert_eq!(m.get(q.parts.leg1).cubes[0].pos2, [2.0, 12.0, 2.0]);
    }

    /// java: `ModelPig` adds one snout box at texture (16, 16) and drops `childYOffset` to 4.
    #[test]
    fn pig_adds_a_snout_and_lowers_the_child_head_offset() {
        let p = PigModel::new();
        let m = &p.quadruped.model;
        let head = m.get(p.quadruped.parts.head);
        assert_eq!(head.cubes.len(), 2, "muzzle plus the inherited head box");
        assert_eq!(head.cubes[0].pos1, [-4.0, -4.0, -8.0]);
        assert_eq!(head.cubes[1].pos1, [-2.0, 0.0, -9.0]);
        assert_eq!(head.cubes[1].pos2, [2.0, 3.0, -8.0]);
        assert_eq!(
            head.texture_offset,
            [16, 16],
            "setTextureOffset(16, 16) is left in place after the snout"
        );
        // Inherited from `super(6, 0.0F)`.
        assert_eq!(head.rotation_point, [0.0, 12.0, -6.0]);
        assert_eq!(p.quadruped.child_y_offset, 4.0);
        assert_eq!(p.quadruped.child_z_offset, 4.0);
    }

    /// java: `ModelCow` — a six-deep head with two horns, and a body with an udder.
    #[test]
    fn cow_head_and_body_are_rebuilt_with_horns_and_an_udder() {
        let c = CowModel::new();
        let m = &c.quadruped.model;

        let head = m.get(c.quadruped.parts.head);
        assert_eq!(
            head.cubes.len(),
            3,
            "the rebuild must drop the 8x8x8 base head"
        );
        assert_eq!(head.cubes[0].pos1, [-4.0, -4.0, -6.0]);
        assert_eq!(head.cubes[0].pos2, [4.0, 4.0, 0.0]);
        assert_eq!(head.rotation_point, [0.0, 4.0, -8.0]);
        assert_eq!(head.cubes[1].pos1, [-5.0, -5.0, -4.0]);
        assert_eq!(head.cubes[1].pos2, [-4.0, -2.0, -3.0]);
        assert_eq!(head.cubes[2].pos1, [4.0, -5.0, -4.0]);
        assert_eq!(head.cubes[2].pos2, [5.0, -2.0, -3.0]);

        let body = m.get(c.quadruped.parts.body);
        assert_eq!(body.cubes.len(), 2);
        assert_eq!(body.cubes[0].pos1, [-6.0, -10.0, -7.0]);
        assert_eq!(body.cubes[0].pos2, [6.0, 8.0, 3.0]);
        assert_eq!(body.rotation_point, [0.0, 5.0, 2.0]);
        // The udder: `setTextureOffset(52, 0).addBox(-2, 2, -8, 4, 6, 1)`.
        assert_eq!(body.cubes[1].pos1, [-2.0, 2.0, -8.0]);
        assert_eq!(body.cubes[1].pos2, [2.0, 8.0, -7.0]);
        assert_eq!(body.texture_offset, [52, 0]);
    }

    /// java: the six `++`/`--` statements on the inherited leg pivots, plus `childZOffset += 2`.
    #[test]
    fn cow_legs_are_nudged_out_and_forward() {
        let c = CowModel::new();
        let m = &c.quadruped.model;
        // Base pivots for legY = 12 are (-3, 12, 7), (3, 12, 7), (-3, 12, -5), (3, 12, -5).
        assert_eq!(
            m.get(c.quadruped.parts.leg1).rotation_point,
            [-4.0, 12.0, 7.0]
        );
        assert_eq!(
            m.get(c.quadruped.parts.leg2).rotation_point,
            [4.0, 12.0, 7.0]
        );
        assert_eq!(
            m.get(c.quadruped.parts.leg3).rotation_point,
            [-4.0, 12.0, -6.0]
        );
        assert_eq!(
            m.get(c.quadruped.parts.leg4).rotation_point,
            [4.0, 12.0, -6.0]
        );
        // The rear legs' `rotationPointZ += 0.0F` really is a no-op.
        assert_eq!(m.get(c.quadruped.parts.leg1).rotation_point[2], 7.0);
        assert_eq!(c.quadruped.child_y_offset, 8.0);
        assert_eq!(c.quadruped.child_z_offset, 6.0);
    }

    /// java: `ModelSheep2` — a 6x6x8 head and an 8x16x6 body, legs untouched.
    #[test]
    fn sheep_body_geometry_matches_model_sheep2() {
        let s = SheepModel::new();
        let m = &s.quadruped.model;
        let head = m.get(s.quadruped.parts.head);
        assert_eq!(head.cubes.len(), 1);
        assert_eq!(head.cubes[0].pos1, [-3.0, -4.0, -6.0]);
        assert_eq!(head.cubes[0].pos2, [3.0, 2.0, 2.0]);
        assert_eq!(head.rotation_point, [0.0, 6.0, -8.0]);
        assert_eq!(head.texture_offset, [0, 0]);

        let body = m.get(s.quadruped.parts.body);
        assert_eq!(body.cubes[0].pos1, [-4.0, -10.0, -7.0]);
        assert_eq!(body.cubes[0].pos2, [4.0, 6.0, -1.0]);
        assert_eq!(body.rotation_point, [0.0, 5.0, 2.0]);
        assert_eq!(body.texture_offset, [28, 8]);

        // `ModelSheep2` never touches the legs, so they stay `ModelQuadruped(12, 0)`'s.
        assert_eq!(
            m.get(s.quadruped.parts.leg1).cubes[0].pos2,
            [2.0, 12.0, 2.0]
        );
    }

    /// java: `ModelSheep1` — the same boxes inflated by 0.6 / 1.75 / 0.5, with half-height legs.
    #[test]
    fn sheep_wool_inflates_every_box() {
        let w = SheepWoolModel::new();
        let m = &w.quadruped.model;

        // `delta` never moves pos1/pos2, only the baked vertices. Quad 4 is the -Z face and its
        // vertex 1 is the (x1, y1, z1) corner.
        let head = m.get(w.quadruped.parts.head);
        assert_eq!(head.cubes[0].pos1, [-3.0, -4.0, -4.0]);
        assert_eq!(head.cubes[0].pos2, [3.0, 2.0, 2.0]);
        let c = head.cubes[0].quads[4].verts[1].pos;
        for (got, want) in c.iter().zip([-3.6_f32, -4.6, -4.6]) {
            assert!((got - want).abs() < 1e-5, "head delta 0.6: {got} vs {want}");
        }

        let body = m.get(w.quadruped.parts.body);
        assert_eq!(body.cubes[0].pos1, [-4.0, -10.0, -7.0]);
        let c = body.cubes[0].quads[4].verts[1].pos;
        assert_eq!(c, [-5.75, -11.75, -8.75], "body delta 1.75");

        // The wool legs are 4x6x4, not the body's 4x12x4.
        let leg = m.get(w.quadruped.parts.leg1);
        assert_eq!(leg.cubes.len(), 1, "the 12-deep base leg must be gone");
        assert_eq!(leg.cubes[0].pos2, [2.0, 6.0, 2.0]);
        assert_eq!(leg.cubes[0].quads[4].verts[1].pos, [-2.5, -0.5, -2.5]);
        assert_eq!(leg.texture_offset, [0, 16]);
    }

    /// The fleece sits exactly on the body: same part count, same pivots everywhere.
    ///
    /// java: `ModelSheep1` and `ModelSheep2` both `super(12, 0.0F)` and then set the head to
    /// `(0, 6, -8)` and the body to `(0, 5, 2)`; `ModelSheep1` re-sets the four leg pivots to
    /// the values `ModelQuadruped(12, ...)` already gave them.
    #[test]
    fn sheep_wool_and_body_share_their_pivots() {
        let body = SheepModel::new();
        let wool = SheepWoolModel::new();
        assert_eq!(body.quadruped.model.parts().len(), 6);
        assert_eq!(
            body.quadruped.model.parts().len(),
            wool.quadruped.model.parts().len()
        );
        let bp = body.quadruped.parts;
        let wp = wool.quadruped.parts;
        for (b, w) in [
            (bp.head, wp.head),
            (bp.body, wp.body),
            (bp.leg1, wp.leg1),
            (bp.leg2, wp.leg2),
            (bp.leg3, wp.leg3),
            (bp.leg4, wp.leg4),
        ] {
            assert_eq!(
                body.quadruped.model.get(b).rotation_point,
                wool.quadruped.model.get(w).rotation_point,
                "part {} pivot",
                body.quadruped.model.get(b).name
            );
        }
        assert_eq!(
            body.quadruped.model.get(bp.head).rotation_point,
            [0.0, 6.0, -8.0]
        );
        assert_eq!(
            wool.quadruped.model.get(wp.leg3).rotation_point,
            [-3.0, 12.0, -5.0]
        );
    }

    /// Hand-checked against `ModelQuadruped.setRotationAngles`'s four `cos` lines.
    #[test]
    fn legs_swing_by_the_literal_formula() {
        let mut pig = PigModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 7.0;
        p.limb_swing_amount = 0.8;
        pig.set_rotation_angles(&p);

        let want_a = cos(7.0 * 0.6662) * 1.4 * 0.8;
        let want_b = cos(7.0 * 0.6662 + PI) * 1.4 * 0.8;
        let m = &pig.quadruped.model;
        let q = pig.quadruped.parts;
        assert_eq!(m.get(q.leg1).rotate_angle[0], want_a);
        assert_eq!(m.get(q.leg2).rotate_angle[0], want_b);
        assert_eq!(m.get(q.leg3).rotate_angle[0], want_b);
        assert_eq!(m.get(q.leg4).rotate_angle[0], want_a);
        // Diagonal pairs move together, adjacent pairs oppose.
        assert!(want_a * want_b <= 0.0, "{want_a} and {want_b} must oppose");
        // The body is laid flat every frame regardless of input.
        assert!((m.get(q.body).rotate_angle[0] - PI / 2.0).abs() < 1e-7);
        // Only the X angle is ever written on a leg.
        assert_eq!(m.get(q.leg1).rotate_angle[1], 0.0);
        assert_eq!(m.get(q.leg1).rotate_angle[2], 0.0);
    }

    /// java: `headPitch / (180F / (float)Math.PI)` and the same for `netHeadYaw`.
    #[test]
    fn head_follows_yaw_and_pitch_in_radians() {
        let mut cow = CowModel::new();
        let mut p = AnimParams::still();
        p.net_head_yaw = 90.0;
        p.head_pitch = -30.0;
        cow.set_rotation_angles(&p);
        let h = cow.quadruped.model.get(cow.quadruped.parts.head);
        assert!(
            (h.rotate_angle[1] - PI / 2.0).abs() < 1e-6,
            "{:?}",
            h.rotate_angle
        );
        assert!((h.rotate_angle[0] + PI / 6.0).abs() < 1e-6);
        assert_eq!(h.rotate_angle[2], 0.0);
    }

    /// java: `ModelSheep2.setLivingAnimations` lowers the head pivot to `6 + 1 * 9` and
    /// `setRotationAngles` then overwrites `rotateAngleX` with the grazing angle.
    #[test]
    fn grazing_lowers_the_sheep_head_and_overrides_its_pitch() {
        let graze = PI / 5.0;
        for model in [
            &mut SheepModel::new() as &mut dyn EntityModel,
            &mut SheepWoolModel::new(),
        ] {
            let mut p = AnimParams::still();
            p.head_pitch = -30.0;
            // java: `EntitySheep.getHeadRotationAngleX`/`getHeadRotationPointY`, computed by
            // `SheepAnim` from `sheepTimer` and delivered on these two fields. This used to
            // drive the pose through the shared `tail_rotation` slot, which no producer ever
            // wrote — so no sheep in mc-rs had ever grazed.
            p.entity = EntityAnimState {
                sheep_head_angle_x: graze,
                sheep_head_point_y: 1.0,
                ..Default::default()
            };
            model.set_living_animations(&p);
            model.set_rotation_angles(&p);
            let head = model.model().find("head").expect("sheep has a head part");
            assert_eq!(model.model().get(head).rotation_point[1], 15.0);
            assert_eq!(model.model().get(head).rotate_angle[0], graze);
            // Yaw still tracks the entity while eating; only pitch is taken over.
            assert_eq!(model.model().get(head).rotate_angle[1], 0.0);
        }
    }

    /// With no grazing angle in flight the head sits at its authored pivot and follows pitch,
    /// which is what vanilla's `else` branch (`rotationPitch / (180 / PI)`) does.
    ///
    /// The pitch arrives on `sheep_head_angle_x`, already converted, because vanilla reads the
    /// entity's **raw** `rotationPitch` there rather than the interpolated `head_pitch` the
    /// renderer hands `setRotationAngles`. This test used to feed `head_pitch`, which is a
    /// different value on any tick where the sheep is turning.
    #[test]
    fn an_idle_sheep_keeps_its_head_up_and_follows_pitch() {
        let mut s = SheepModel::new();
        let mut p = AnimParams::still();
        p.entity.sheep_head_angle_x = -30.0 / crate::entity::anim::DEG_PER_RAD;
        p.head_pitch = 60.0;
        s.set_living_animations(&p);
        s.set_rotation_angles(&p);
        let h = s.quadruped.model.get(s.quadruped.parts.head);
        assert_eq!(h.rotation_point[1], 6.0);
        assert!(
            (h.rotate_angle[0] + PI / 6.0).abs() < 1e-6,
            "{:?}",
            h.rotate_angle
        );
    }

    /// java: `LayerSheepWool.doRenderLayer` skips the fleece entirely once the sheep is sheared.
    #[test]
    fn a_sheared_sheep_grows_no_wool_but_keeps_its_body() {
        let mut p = AnimParams::still();
        p.entity.child = false;

        let mut wool = SheepWoolModel::new();
        let mut out = Vec::new();
        wool.pose_and_render(&p, Mat4::IDENTITY, &mut out);
        // 6 parts, one box each, 6 faces, 6 vertices.
        assert_eq!(out.len(), 6 * 6 * 6);

        p.entity.sheared = true;
        out.clear();
        wool.pose_and_render(&p, Mat4::IDENTITY, &mut out);
        assert!(out.is_empty(), "sheared sheep must emit no fleece");

        // The body underneath is drawn either way.
        let mut body = SheepModel::new();
        out.clear();
        body.pose_and_render(&p, Mat4::IDENTITY, &mut out);
        assert_eq!(out.len(), 6 * 6 * 6);
    }

    /// java: `ModelQuadruped.render`'s `isChild` branch — the head is translated by
    /// `(0, childYOffset * scale, childZOffset * scale)` and everything else is halved and
    /// pushed down by `24 * scale`.
    #[test]
    fn a_baby_quadruped_splits_into_two_matrices() {
        let mut pig = PigModel::new();
        let mut p = AnimParams::still();
        p.entity.child = false;
        pig.quadruped.model.is_child = false;
        pig.set_rotation_angles(&p);

        let mut adult = Vec::new();
        pig.render(&p, Mat4::IDENTITY, &mut adult);
        pig.quadruped.model.is_child = true;
        let mut child = Vec::new();
        pig.render(&p, Mat4::IDENTITY, &mut child);

        // 2 head boxes + body + 4 legs = 7 boxes, 6 faces, 6 vertices, both ways.
        assert_eq!(adult.len(), 7 * 6 * 6);
        assert_eq!(child.len(), adult.len());

        let scale = p.scale;
        let head_verts = 2 * 6 * 6;
        // childYOffset = 4 for a pig, childZOffset = 4.
        for i in 0..head_verts {
            let a = adult[i].pos;
            let c = child[i].pos;
            assert!((c[0] - a[0]).abs() < 1e-6);
            assert!((c[1] - (a[1] + 4.0 * scale)).abs() < 1e-6, "head y at {i}");
            assert!((c[2] - (a[2] + 4.0 * scale)).abs() < 1e-6, "head z at {i}");
        }
        // scale(1/2) then translate(0, 24 * scale, 0) == p -> 0.5 * p + (0, 12 * scale, 0).
        for i in head_verts..adult.len() {
            let a = adult[i].pos;
            let c = child[i].pos;
            assert!((c[0] - a[0] * 0.5).abs() < 1e-6);
            assert!(
                (c[1] - (a[1] * 0.5 + 12.0 * scale)).abs() < 1e-6,
                "body y at {i}: {} vs {}",
                c[1],
                a[1] * 0.5 + 12.0 * scale
            );
            assert!((c[2] - a[2] * 0.5).abs() < 1e-6);
        }
    }
}
