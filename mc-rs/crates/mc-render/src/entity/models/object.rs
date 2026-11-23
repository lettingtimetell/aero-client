//! Vehicles, tethers, the ender crystal and armour stands.
//!
//! java: `ModelBoat.java`, `ModelMinecart.java`, `ModelLeashKnot.java`, `ModelEnderCrystal.java`,
//! `ModelArmorStandArmor.java`, `ModelArmorStand.java`.
//!
//! Everything here follows the shape [`super::biped`] establishes: a struct holding a [`Model`]
//! plus the [`PartId`]s the Java kept as fields, a constructor replaying the Java constructor in
//! order, and a [`EntityModel::set_rotation_angles`] replaying `setRotationAngles`.
//!
//! Three things are specific to this file:
//!
//! * None of these are `EntityLivingBase` models. Their renderers reach into the part fields
//!   directly (`ModelChest.chestLid` is the classic; here it is `ModelMinecart.sideModels[5]`
//!   and the armour stand's `showModel` flags), so every part is a `pub` field.
//! * `ModelBoat` and `ModelMinecart` keep their parts in an *array* and both build and draw them
//!   in index order, so the default [`Model::emit_all`] root walk is already the right order.
//! * `ModelArmorStandArmor` extends `ModelBiped` but **does not call `super.setRotationAngles`** —
//!   it replaces the pose wholesale with the stand's stored `Rotations`. Those live on
//!   [`ArmorStandPose`] here, since this crate has no `EntityArmorStand` to downcast to.

use glam::{Mat4, Vec3};

use crate::entity::anim::{model_radians, AnimParams, EntityModel, DEG_TO_RAD};
use crate::entity::model::{Model, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

/// Vanilla's hard-coded degrees-to-radians literal in the two armour-stand models.
///
/// java: `0.017453292F` — `ModelArmorStand`/`ModelArmorStandArmor` multiply their `Rotations` by
/// this literal instead of dividing by `(180F / (float)Math.PI)` like every other model.
const ARMOR_STAND_DEG_TO_RAD: f32 = 0.017_453_292;

/// Writes one `Rotations` triple onto a part, converting degrees to radians vanilla's way.
///
/// java: the `rotateAngleX/Y/Z = 0.017453292F * ...getX()/getY()/getZ()` triple that repeats
/// eleven times across the two armour-stand models.
fn set_rotations(model: &mut Model, id: PartId, deg: [f32; 3]) {
    model.get_mut(id).rotate_angle = [
        ARMOR_STAND_DEG_TO_RAD * deg[0],
        ARMOR_STAND_DEG_TO_RAD * deg[1],
        ARMOR_STAND_DEG_TO_RAD * deg[2],
    ];
}

/// Clears a part and rebuilds it with a new texture offset, box and pivot.
///
/// java: the `this.bipedHead = new ModelRenderer(this, 0, 0)` pattern in `ModelArmorStand`, which
/// leaves the `ModelBiped` part unreferenced in `boxList` and never draws it again.
#[allow(clippy::too_many_arguments)] // one-for-one with the Java `new ModelRenderer` + `addBox` pair
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

// ---------------------------------------------------------------------------------------------
// ModelBoat
// ---------------------------------------------------------------------------------------------

/// The boat: a floor plate and four gunwales, all cut from the same two boxes.
///
/// java: `ModelBoat`
#[derive(Clone, Debug)]
pub struct BoatModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelBoat.boatSides` — index 0 is the floor, 1..4 the four sides.
    pub boat_sides: [PartId; 5],
}

impl BoatModel {
    /// java: `ModelBoat()` — 64x32 (the `ModelBase` default), five parts.
    ///
    /// The Java sets `rotateAngleX`/`rotateAngleY` in a trailing block after every `addBox`;
    /// angles do not affect box baking, so they are folded into each part's chain here.
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        // java: `int i = 24; int j = 6; int k = 20; int l = 4;`
        let i: i32 = 24;
        let j: i32 = 6;
        let k: i32 = 20;
        let l: i32 = 4;

        let floor = m
            .part("boat_side_0", 0, 8)
            .cube((-i / 2) as f32, (-k / 2 + 2) as f32, -3.0, i, k - 4, 4, 0.0)
            .at(0.0, l as f32, 0.0)
            .angles(PI / 2.0, 0.0, 0.0)
            .id();

        // The four sides are the same box four times, differing only in pivot and yaw.
        let side = |m: &mut Model, name: &'static str, at: [f32; 3], yaw: f32| {
            m.part(name, 0, 0)
                .cube((-i / 2 + 2) as f32, (-j - 1) as f32, -1.0, i - 4, j, 2, 0.0)
                .at(at[0], at[1], at[2])
                .angles(0.0, yaw, 0.0)
                .id()
        };
        let left = side(
            &mut m,
            "boat_side_1",
            [(-i / 2 + 1) as f32, l as f32, 0.0],
            PI * 3.0 / 2.0,
        );
        let right = side(
            &mut m,
            "boat_side_2",
            [(i / 2 - 1) as f32, l as f32, 0.0],
            PI / 2.0,
        );
        let back = side(
            &mut m,
            "boat_side_3",
            [0.0, l as f32, (-k / 2 + 1) as f32],
            PI,
        );
        // java: boatSides[4] is the only side left at yaw 0.
        let front = side(
            &mut m,
            "boat_side_4",
            [0.0, l as f32, (k / 2 - 1) as f32],
            0.0,
        );

        Self {
            model: m,
            boat_sides: [floor, left, right, back, front],
        }
    }
}

impl Default for BoatModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for BoatModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelBoat` never overrides `setRotationAngles`; the pose is baked in.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {}
}

// ---------------------------------------------------------------------------------------------
// ModelMinecart
// ---------------------------------------------------------------------------------------------

/// The minecart: the boat's shape with a sixth part for the floor plate that rides the
/// contents' height.
///
/// java: `ModelMinecart`
#[derive(Clone, Debug)]
pub struct MinecartModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelMinecart.sideModels` — the array is `new ModelRenderer[7]` but only six slots
    /// are ever filled, and `render` loops `i < 6`.
    pub side_models: [PartId; 6],
}

/// java: `RenderMinecart.doRender` (`:108`) — the fourth argument of the model's only `render`
/// call, a literal that never varies with the cart, its contents or the frame.
pub const MINECART_FLOOR_ARG: f32 = -0.1;

impl MinecartModel {
    /// java: `ModelMinecart()` — 64x32, six parts.
    ///
    /// The Java creates all six parts first (which fixes the draw order) and only then calls
    /// `addBox` on them, out of index order; boxes are per-part so the two orders are
    /// independent, and the parts are built here in the index order that governs drawing.
    ///
    /// Note `sideModels[5]`'s `+Y` and `+Z` faces run off the right edge of the 64-wide texture
    /// (`u` reaches 81 and 82). That is vanilla — the texture is `GL_REPEAT`, so those two faces
    /// wrap back to the left edge — and it is reproduced rather than "fixed".
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        // java: `int i = 20; int j = 8; int k = 16; int l = 4;`
        let i: i32 = 20;
        let j: i32 = 8;
        let k: i32 = 16;
        let l: i32 = 4;

        let bottom = m
            .part("minecart_side_0", 0, 10)
            .cube((-i / 2) as f32, (-k / 2) as f32, -1.0, i, k, 2, 0.0)
            .at(0.0, l as f32, 0.0)
            .angles(PI / 2.0, 0.0, 0.0)
            .id();

        let side = |m: &mut Model, name: &'static str, at: [f32; 3], yaw: f32| {
            m.part(name, 0, 0)
                .cube((-i / 2 + 2) as f32, (-j - 1) as f32, -1.0, i - 4, j, 2, 0.0)
                .at(at[0], at[1], at[2])
                .angles(0.0, yaw, 0.0)
                .id()
        };
        let left = side(
            &mut m,
            "minecart_side_1",
            [(-i / 2 + 1) as f32, l as f32, 0.0],
            PI * 3.0 / 2.0,
        );
        let right = side(
            &mut m,
            "minecart_side_2",
            [(i / 2 - 1) as f32, l as f32, 0.0],
            PI / 2.0,
        );
        let back = side(
            &mut m,
            "minecart_side_3",
            [0.0, l as f32, (-k / 2 + 1) as f32],
            PI,
        );
        let front = side(
            &mut m,
            "minecart_side_4",
            [0.0, l as f32, (k / 2 - 1) as f32],
            0.0,
        );

        let floor = m
            .part("minecart_side_5", 44, 10)
            .cube(
                (-i / 2 + 1) as f32,
                (-k / 2 + 1) as f32,
                -1.0,
                i - 2,
                k - 2,
                1,
                0.0,
            )
            .at(0.0, l as f32, 0.0)
            .angles(-(PI / 2.0), 0.0, 0.0)
            .id();

        Self {
            model: m,
            side_models: [bottom, left, right, back, front, floor],
        }
    }
}

impl Default for MinecartModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for MinecartModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelMinecart.render` line 1 — `sideModels[5].rotationPointY = 4.0F - p_78088_4_`.
    ///
    /// The Java writes it at the top of `render`, which runs immediately after
    /// `setRotationAngles` in vanilla's call order and immediately after this method in
    /// [`EntityModel::pose_and_render`], so the effect is identical.
    ///
    /// **`p_78088_4_` is not the age here.** `ModelMinecart` has exactly one call site in
    /// 1.8.9 — `RenderMinecart.doRender` (`:108`), `render(entity, 0, 0, -0.1F, 0, 0, 0.0625F)`
    /// — and that slot is the literal `-0.1F` for every cart there is, so the plate sits at a
    /// constant `4.1`. This used to read `p.age_in_ticks`, which the driver fills with
    /// `ticksExisted + partialTicks`: the floor plate slid down out of the cart at one model
    /// unit (1/16 block) per tick and was below the world within a minute of the cart spawning.
    /// The old doc called `-0.1F` "the contents' display height"; it is not — a display tile is
    /// drawn by a separate block pass with its own `getDisplayTileOffset()` (`:70-72`) and never
    /// reaches this argument.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {
        let floor = self.side_models[5];
        self.model.get_mut(floor).rotation_point[1] = 4.0 - MINECART_FLOOR_ARG;
    }
}

// ---------------------------------------------------------------------------------------------
// ModelLeashKnot
// ---------------------------------------------------------------------------------------------

/// The lead knot tied to a fence post.
///
/// java: `ModelLeashKnot`
#[derive(Clone, Debug)]
pub struct LeashKnotModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelLeashKnot.field_110723_a` — the single knot box.
    pub knot: PartId,
}

impl LeashKnotModel {
    /// java: `ModelLeashKnot()` — `this(0, 0, 32, 32)`. Note the 32x32 texture, not 64x32.
    #[must_use]
    pub fn new() -> Self {
        Self::with(0, 0, 32, 32)
    }

    /// java: `ModelLeashKnot(int texOffX, int texOffY, int textureWidth, int textureHeight)`
    #[must_use]
    pub fn with(u: i32, v: i32, tw: i32, th: i32) -> Self {
        let mut m = Model::new(tw as u32, th as u32);
        let knot = m
            .part("knot", u, v)
            .cube(-3.0, -6.0, -3.0, 6, 8, 6, 0.0)
            .at(0.0, 0.0, 0.0)
            .id();
        Self { model: m, knot }
    }
}

impl Default for LeashKnotModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for LeashKnotModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelLeashKnot.setRotationAngles` — the knot simply faces the viewer.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let knot = self.knot;
        self.model.get_mut(knot).rotate_angle[1] = model_radians(p.net_head_yaw);
        self.model.get_mut(knot).rotate_angle[0] = model_radians(p.head_pitch);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelEnderCrystal
// ---------------------------------------------------------------------------------------------

/// The crystal's fixed 60-degree tilt.
///
/// java: `GlStateManager.rotate(60.0F, 0.7071F, 0.0F, 0.7071F)` — `glRotatef` normalises its
/// axis, and `(0.7071, 0, 0.7071)` is a hair short of unit length, so the axis is normalised here
/// too.
#[allow(clippy::approx_constant)] // the Java literal really is `0.7071F`, not `FRAC_1_SQRT_2`
fn crystal_tilt() -> Mat4 {
    Mat4::from_axis_angle(
        Vec3::new(0.7071, 0.0, 0.7071).normalize(),
        60.0 * DEG_TO_RAD,
    )
}

/// The crystal's spin (degrees) and bob (blocks) for a given `innerRotation + partialTicks`
/// phase.
///
/// java: `RenderEnderCrystal.doRender` (`:28`, `:32-34`) —
/// `f = innerRotation + partialTicks`, `f1 = sin(f * 0.2) / 2 + 0.5`, `f1 = f1 * f1 + f1`, and
/// the model is rendered with `f * 3.0F` and `f1 * 0.2F`.
///
/// `f1` before the `* 0.2` runs over `0..2` (`sin` in `-1..1` gives `f1` in `0..1`, and
/// `x^2 + x` maps that onto `0..2`), so the bob is at most `0.4` blocks — the number A02-13 says
/// the driver was replacing with the raw tick counter.
#[must_use]
pub fn ender_crystal_motion(phase: f32) -> (f32, f32) {
    let f1 = mc_math::sin(phase * 0.2) / 2.0 + 0.5;
    let f1 = f1 * f1 + f1;
    (phase * 3.0, f1 * 0.2)
}

/// The end crystal: two nested tilted shells and an optional bedrock base.
///
/// java: `ModelEnderCrystal`
#[derive(Clone, Debug)]
pub struct EnderCrystalModel {
    /// The part tree.
    pub model: Model,
    /// java: `ModelEnderCrystal.glass` — drawn **twice**, at two nesting levels.
    pub glass: PartId,
    /// java: `ModelEnderCrystal.cube` — the innermost shell.
    pub cube: PartId,
    /// java: `ModelEnderCrystal.base` — `null` unless the constructor's boolean is set.
    pub base: Option<PartId>,
}

impl EnderCrystalModel {
    /// java: `ModelEnderCrystal(float p_i1170_1_, boolean p_i1170_2_)` with `0.0F` for the float,
    /// which the Java never reads.
    ///
    /// Part creation order matches the Java's: `glass` is a field initialiser and so is built
    /// before the constructor body creates `cube` and, optionally, `base`.
    #[must_use]
    pub fn new(base: bool) -> Self {
        let mut m = Model::new(64, 32);
        // java: `new ModelRenderer(this, "glass")` then `.setTextureOffset(0, 0)` — the six-arg
        // `addBox` overload, whose `scaleFactor` is a hard 0.
        let glass = m
            .part("glass", 0, 0)
            .cube(-4.0, -4.0, -4.0, 8, 8, 8, 0.0)
            .id();
        let cube = m
            .part("cube", 32, 0)
            .cube(-4.0, -4.0, -4.0, 8, 8, 8, 0.0)
            .id();
        let base = base.then(|| {
            m.part("base", 0, 16)
                .cube(-6.0, 0.0, -6.0, 12, 4, 12, 0.0)
                .id()
        });
        Self {
            model: m,
            glass,
            cube,
            base,
        }
    }
}

impl EntityModel for EnderCrystalModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    /// java: `ModelEnderCrystal` never overrides `setRotationAngles` — all of its motion is in
    /// the matrix stack inside `render`.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {}

    /// java: `ModelEnderCrystal.render`, matrix for matrix — **including**
    /// `RenderEnderCrystal.doRender`'s own two lines of arithmetic.
    ///
    /// # Why the renderer's maths lives in the model
    ///
    /// `RenderEnderCrystal.doRender` (`:28-34`) computes `f = innerRotation + partialTicks`,
    /// `f1 = sin(f * 0.2) / 2 + 0.5; f1 = f1 * f1 + f1`, and calls
    /// `render(entity, 0.0F, f * 3.0F, f1 * 0.2F, 0.0F, 0.0F, 0.0625F)` — passing the spin in
    /// the `limbSwing` **amount** slot and the bob height in the `ageInTicks` slot.
    ///
    /// This model used to read those two slots back out with that punning intact, which meant a
    /// generic caller that filled `AnimParams` the ordinary way (`age_in_ticks = ticksExisted +
    /// partial`, `limb_swing_amount = 0`) got a crystal that never rotated and whose "bob" was
    /// the tick counter *in blocks* — after a minute of uptime its shells were 1200 blocks up in
    /// the sky. Every other model in this crate is driven correctly by the generic caller, so the
    /// fix is to derive the spin and bob from the crystal's own phase instead of generic age.
    ///
    /// `EntityEnderCrystal`'s constructor seeds `innerRotation = rand.nextInt(100000)` and its
    /// tick increments it by one. The tracker carries that current value in `EntityAnimState`;
    /// adding only `partial_ticks` here restores Java's distinct per-crystal
    /// phase; the motion itself — period, amplitude and the `f1 * f1 + f1` easing — is exact.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let scale = p.scale;
        let (spin, bob) = ender_crystal_motion(p.entity.crystal_inner_rotation + p.partial_ticks);

        // java: scale(2, 2, 2); translate(0, -0.5, 0)
        let mut m = base
            * Mat4::from_scale(Vec3::splat(2.0))
            * Mat4::from_translation(Vec3::new(0.0, -0.5, 0.0));

        if let Some(b) = self.base {
            self.model.emit_part(b, scale, m, out);
        }

        // java: rotate(spin, 0, 1, 0); translate(0, 0.8 + bob, 0); rotate(60, 0.7071, 0, 0.7071)
        m = m
            * Mat4::from_rotation_y(spin * DEG_TO_RAD)
            * Mat4::from_translation(Vec3::new(0.0, 0.8 + bob, 0.0))
            * crystal_tilt();
        self.model.emit_part(self.glass, scale, m, out);

        // java: float f = 0.875F; scale(f, f, f); rotate(60, ...); rotate(spin, 0, 1, 0)
        let f = 0.875;
        m = m
            * Mat4::from_scale(Vec3::splat(f))
            * crystal_tilt()
            * Mat4::from_rotation_y(spin * DEG_TO_RAD);
        self.model.emit_part(self.glass, scale, m, out);

        m = m
            * Mat4::from_scale(Vec3::splat(f))
            * crystal_tilt()
            * Mat4::from_rotation_y(spin * DEG_TO_RAD);
        self.model.emit_part(self.cube, scale, m, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelArmorStandArmor
// ---------------------------------------------------------------------------------------------

/// The armour stand's stored limb rotations, in **degrees**, plus the two display flags.
///
/// java: `EntityArmorStand`'s six `Rotations` datawatcher slots, `getShowArms()`,
/// `hasNoBasePlate()` and `rotationYaw`. `ModelArmorStandArmor.setRotationAngles` downcasts its
/// `Entity` to read them; this crate has no entity object to downcast, so the renderer fills
/// these in instead. The defaults are `EntityArmorStand`'s `DEFAULT_*_ROTATION` constants, which
/// is the pose a freshly placed stand holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArmorStandPose {
    /// java: `getHeadRotation()`
    pub head: [f32; 3],
    /// java: `getBodyRotation()` — also drives the two stand sides and the waist.
    pub body: [f32; 3],
    /// java: `getLeftArmRotation()`
    pub left_arm: [f32; 3],
    /// java: `getRightArmRotation()`
    pub right_arm: [f32; 3],
    /// java: `getLeftLegRotation()`
    pub left_leg: [f32; 3],
    /// java: `getRightLegRotation()`
    pub right_leg: [f32; 3],
    /// java: `getShowArms()` — off by default, which is why a plain stand has no arms.
    pub show_arms: bool,
    /// java: `hasNoBasePlate()`
    pub no_base_plate: bool,
    /// java: `entityIn.rotationYaw`, degrees — only the base plate uses it.
    pub rotation_yaw: f32,
}

impl Default for ArmorStandPose {
    /// java: `EntityArmorStand.DEFAULT_HEAD_ROTATION` and friends.
    fn default() -> Self {
        Self {
            head: [0.0, 0.0, 0.0],
            body: [0.0, 0.0, 0.0],
            left_arm: [-10.0, 0.0, -10.0],
            right_arm: [-15.0, 0.0, 10.0],
            left_leg: [-1.0, 0.0, -1.0],
            right_leg: [1.0, 0.0, 1.0],
            show_arms: false,
            no_base_plate: false,
            rotation_yaw: 0.0,
        }
    }
}

/// The seven `ModelBiped` parts, transcribed standalone rather than inherited.
///
/// java: `ModelBiped.bipedHead` ... `bipedLeftLeg`.
#[derive(Clone, Copy, Debug)]
pub struct ArmorStandParts {
    /// java: `bipedHead`
    pub head: PartId,
    /// java: `bipedHeadwear` — `ModelArmorStand` never replaces this one, so the stand really
    /// does draw the biped's inflated hat box.
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

/// The armour worn by an armour stand: `ModelBiped` geometry with the stand's own pose.
///
/// java: `ModelArmorStandArmor`
#[derive(Clone, Debug)]
pub struct ArmorStandArmorModel {
    /// The part tree.
    pub model: Model,
    /// Handles to the seven biped parts.
    pub parts: ArmorStandParts,
    /// The stand's stored limb rotations. See [`ArmorStandPose`].
    pub pose: ArmorStandPose,
}

impl ArmorStandArmorModel {
    /// java: `ModelArmorStandArmor(float modelSize)` — `this(modelSize, 64, 32)`.
    #[must_use]
    pub fn new(size: f32) -> Self {
        Self::with(size, 64, 32)
    }

    /// java: `ModelArmorStandArmor(float modelSize, int textureWidthIn, int textureHeightIn)`,
    /// which forwards to `ModelBiped(modelSize, 0.0F, textureWidthIn, textureHeightIn)`.
    ///
    /// The `ModelBiped` constructor is transcribed inline here — the vertical offset is a hard
    /// `0.0F`, so it is dropped from every pivot.
    #[must_use]
    pub fn with(model_size: f32, tw: u32, th: u32) -> Self {
        let mut m = Model::new(tw, th);
        let head = m
            .part("head", 0, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, model_size)
            .at(0.0, 0.0, 0.0)
            .id();
        let headwear = m
            .part("headwear", 32, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, model_size + 0.5)
            .at(0.0, 0.0, 0.0)
            .id();
        let body = m
            .part("body", 16, 16)
            .cube(-4.0, 0.0, -2.0, 8, 12, 4, model_size)
            .at(0.0, 0.0, 0.0)
            .id();
        let right_arm = m
            .part("right_arm", 40, 16)
            .cube(-3.0, -2.0, -2.0, 4, 12, 4, model_size)
            .at(-5.0, 2.0, 0.0)
            .id();
        let left_arm = m
            .part("left_arm", 40, 16)
            .mirrored()
            .cube(-1.0, -2.0, -2.0, 4, 12, 4, model_size)
            .at(5.0, 2.0, 0.0)
            .id();
        let right_leg = m
            .part("right_leg", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size)
            .at(-1.9, 12.0, 0.0)
            .id();
        let left_leg = m
            .part("left_leg", 0, 16)
            .mirrored()
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, model_size)
            .at(1.9, 12.0, 0.0)
            .id();
        Self {
            model: m,
            parts: ArmorStandParts {
                head,
                headwear,
                body,
                right_arm,
                left_arm,
                right_leg,
                left_leg,
            },
            pose: ArmorStandPose::default(),
        }
    }

    /// `ModelArmorStandArmor.setRotationAngles`, statement for statement.
    ///
    /// The whole Java body sits inside `if (entityIn instanceof EntityArmorStand)`; there is no
    /// `super.setRotationAngles` call, so none of `ModelBiped`'s walk cycle applies. Here the
    /// stand's data always exists ([`ArmorStandPose`]), so the body always runs.
    pub fn pose_stand(&mut self) {
        let q = self.parts;
        let s = self.pose;
        let m = &mut self.model;

        set_rotations(m, q.head, s.head);
        m.get_mut(q.head).set_rotation_point(0.0, 1.0, 0.0);
        set_rotations(m, q.body, s.body);
        set_rotations(m, q.left_arm, s.left_arm);
        set_rotations(m, q.right_arm, s.right_arm);
        set_rotations(m, q.left_leg, s.left_leg);
        m.get_mut(q.left_leg).set_rotation_point(1.9, 11.0, 0.0);
        set_rotations(m, q.right_leg, s.right_leg);
        m.get_mut(q.right_leg).set_rotation_point(-1.9, 11.0, 0.0);
        m.copy_angles(q.head, q.headwear);
    }

    /// The `ModelBiped.render` draw order, including the baby split and the sneak nudge.
    ///
    /// java: `ModelBiped.render` — `ModelArmorStandArmor` does not override it.
    pub fn draw(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let q = self.parts;
        let scale = p.scale;
        if self.model.is_child {
            // java: scale(1.5/2) then translate(0, 16 * scale, 0)
            let head_m = base
                * Mat4::from_scale(Vec3::splat(0.75))
                * Mat4::from_translation(Vec3::new(0.0, 16.0 * scale, 0.0));
            self.model.emit_part(q.head, scale, head_m, out);
            let body_m = Self::child_body_matrix(base, scale);
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

    /// The matrix a baby's body parts are drawn under. java: `scale(1/2); translate(0, 24 * scale, 0)`
    #[must_use]
    pub fn child_body_matrix(base: Mat4, scale: f32) -> Mat4 {
        base * Mat4::from_scale(Vec3::splat(0.5))
            * Mat4::from_translation(Vec3::new(0.0, 24.0 * scale, 0.0))
    }
}

impl EntityModel for ArmorStandArmorModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        // java: `ModelArmorStandArmor.setRotationAngles` opens with
        // `if (entityIn instanceof EntityArmorStand)` and reads the stand's six `Rotations`
        // slots off it. This crate has no entity to downcast, so the caller puts them on
        // `EntityAnimState`; `None` keeps the constructor's `DEFAULT_*_ROTATION` pose, which is
        // what a stand the server has never posed holds.
        //
        // Assigned **unconditionally**. `RenderManager` keeps one of these per `ModelKind`, so
        // an `if let Some(..)` here leaves the *previous* stand's limbs on the shared model
        // whenever `armor_stand` is `None` — the doc above would then be true only of the first
        // stand ever drawn. `unwrap_or_default` is what it already claims to do.
        self.pose = p.entity.armor_stand.unwrap_or_default();
        self.pose_stand();
    }
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.draw(p, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// ModelArmorStand
// ---------------------------------------------------------------------------------------------

/// The armour stand itself: thin sticks in place of the biped limbs, plus the wooden frame.
///
/// java: `ModelArmorStand`
#[derive(Clone, Debug)]
pub struct ArmorStandModel {
    /// The inherited `ModelArmorStandArmor`, with all six limb parts rebuilt in place.
    pub armor: ArmorStandArmorModel,
    /// java: `ModelArmorStand.standRightSide`
    pub stand_right_side: PartId,
    /// java: `ModelArmorStand.standLeftSide`
    pub stand_left_side: PartId,
    /// java: `ModelArmorStand.standWaist`
    pub stand_waist: PartId,
    /// java: `ModelArmorStand.standBase`
    pub stand_base: PartId,
}

impl ArmorStandModel {
    /// java: `ModelArmorStand()` — `this(0.0F)`.
    #[must_use]
    pub fn new() -> Self {
        Self::sized(0.0)
    }

    /// java: `ModelArmorStand(float p_i46306_1_)` — `super(size, 64, 64)`, then six replaced
    /// parts and four new ones.
    ///
    /// `bipedHeadwear` is deliberately *not* replaced: the stand inherits and draws the biped's
    /// inflated hat box, which on `wood.png` is the blank region at (32, 0).
    #[must_use]
    pub fn sized(size: f32) -> Self {
        let mut armor = ArmorStandArmorModel::with(size, 64, 64);
        let q = armor.parts;
        let m = &mut armor.model;

        rebuild(
            m,
            q.head,
            0,
            0,
            false,
            [-1.0, -7.0, -1.0],
            [2, 7, 2],
            size,
            [0.0, 0.0, 0.0],
        );
        rebuild(
            m,
            q.body,
            0,
            26,
            false,
            [-6.0, 0.0, -1.5],
            [12, 3, 3],
            size,
            [0.0, 0.0, 0.0],
        );
        rebuild(
            m,
            q.right_arm,
            24,
            0,
            false,
            [-2.0, -2.0, -1.0],
            [2, 12, 2],
            size,
            [-5.0, 2.0, 0.0],
        );
        rebuild(
            m,
            q.left_arm,
            32,
            16,
            true,
            [0.0, -2.0, -1.0],
            [2, 12, 2],
            size,
            [5.0, 2.0, 0.0],
        );
        rebuild(
            m,
            q.right_leg,
            8,
            0,
            false,
            [-1.0, 0.0, -1.0],
            [2, 11, 2],
            size,
            [-1.9, 12.0, 0.0],
        );
        rebuild(
            m,
            q.left_leg,
            40,
            16,
            true,
            [-1.0, 0.0, -1.0],
            [2, 11, 2],
            size,
            [1.9, 12.0, 0.0],
        );

        let stand_right_side = m
            .part("stand_right_side", 16, 0)
            .cube(-3.0, 3.0, -1.0, 2, 7, 2, size)
            .at(0.0, 0.0, 0.0)
            .id();
        // java: `standRightSide.showModel = true` — already the default.
        let stand_left_side = m
            .part("stand_left_side", 48, 16)
            .cube(1.0, 3.0, -1.0, 2, 7, 2, size)
            .at(0.0, 0.0, 0.0)
            .id();
        let stand_waist = m
            .part("stand_waist", 0, 48)
            .cube(-4.0, 10.0, -1.0, 8, 2, 2, size)
            .at(0.0, 0.0, 0.0)
            .id();
        let stand_base = m
            .part("stand_base", 0, 32)
            .cube(-6.0, 11.0, -6.0, 12, 1, 12, size)
            .at(0.0, 12.0, 0.0)
            .id();

        Self {
            armor,
            stand_right_side,
            stand_left_side,
            stand_waist,
            stand_base,
        }
    }
}

impl Default for ArmorStandModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for ArmorStandModel {
    fn model(&self) -> &Model {
        &self.armor.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.armor.model
    }

    /// java: `ModelArmorStand` inherits `ModelBiped.postRenderArm`, so its held-item layer hangs
    /// the stack from the stand's posed right arm just like any other biped renderer.
    fn held_item_arm(&self) -> Option<PartId> {
        Some(self.armor.parts.right_arm)
    }

    /// java: `ModelArmorStand.setRotationAngles` — `super` first, then the stand's own frame.
    ///
    /// The Java computes three locals `f`/`f1`/`f2` (the average of the two leg rotations) and
    /// then never uses them; they are omitted.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        // java: `if (entityIn instanceof EntityArmorStand)` in the inherited
        // `ModelArmorStandArmor.setRotationAngles`, plus `getShowArms()`/`hasNoBasePlate()` and
        // `entityIn.rotationYaw` below. See `EntityAnimState::armor_stand`, and
        // `ArmorStandArmorModel::set_rotation_angles` for why this is not an `if let`.
        self.armor.pose = p.entity.armor_stand.unwrap_or_default();
        self.armor.pose_stand();

        let q = self.armor.parts;
        let s = self.armor.pose;
        let m = &mut self.armor.model;

        m.get_mut(q.left_arm).show_model = s.show_arms;
        m.get_mut(q.right_arm).show_model = s.show_arms;
        m.get_mut(self.stand_base).show_model = !s.no_base_plate;
        // java: the stand pushes the legs back down to y = 12, undoing the armour's y = 11.
        m.get_mut(q.left_leg).set_rotation_point(1.9, 12.0, 0.0);
        m.get_mut(q.right_leg).set_rotation_point(-1.9, 12.0, 0.0);

        set_rotations(m, self.stand_right_side, s.body);
        set_rotations(m, self.stand_left_side, s.body);
        set_rotations(m, self.stand_waist, s.body);
        // java: `standBase.rotateAngleX = 0.0F` and `rotateAngleZ = 0.0F` bracket the yaw.
        m.get_mut(self.stand_base).rotate_angle =
            [0.0, ARMOR_STAND_DEG_TO_RAD * -s.rotation_yaw, 0.0];
    }

    /// java: `ModelArmorStand.render` — `super.render` (the biped order) and then the four frame
    /// parts under their own child/sneak transform. Note the child branch scales the frame by
    /// `1/2`, not the head's `1.5/2`.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.armor.draw(p, base, out);
        let frame = [
            self.stand_right_side,
            self.stand_left_side,
            self.stand_waist,
            self.stand_base,
        ];
        let m = if self.armor.model.is_child {
            ArmorStandArmorModel::child_body_matrix(base, p.scale)
        } else if p.entity.sneaking {
            base * Mat4::from_translation(Vec3::new(0.0, 0.2, 0.0))
        } else {
            base
        };
        self.armor.model.emit_some(&frame, p.scale, m, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every UV a model emits must land inside its own texture.
    fn assert_uvs_in_range(m: &mut dyn EntityModel, what: &str) {
        let mut out = Vec::new();
        m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
        assert!(!out.is_empty(), "{what} emitted nothing");
        for v in &out {
            assert!(
                (-0.001..=1.001).contains(&v.uv[0]) && (-0.001..=1.001).contains(&v.uv[1]),
                "{what} uv out of range: {:?}",
                v.uv
            );
        }
    }

    /// java: `boatSides[0].addBox(-12, -8, -3, 24, 16, 4, 0)` at pivot `(0, 4, 0)`, `rotX = PI/2`.
    #[test]
    fn boat_floor_is_a_24x16x4_plate_tipped_flat() {
        let b = BoatModel::new();
        let floor = b.model.get(b.boat_sides[0]);
        assert_eq!(floor.texture_offset, [0, 8]);
        assert_eq!(floor.cubes[0].pos1, [-12.0, -8.0, -3.0]);
        assert_eq!(floor.cubes[0].pos2, [12.0, 8.0, 1.0]);
        assert_eq!(floor.rotation_point, [0.0, 4.0, 0.0]);
        assert_eq!(floor.rotate_angle, [PI / 2.0, 0.0, 0.0]);
    }

    /// The four sides share one box and differ only in pivot and yaw.
    ///
    /// java: `addBox(-10, -7, -1, 20, 6, 2, 0)` four times, with pivots `(-11, 4, 0)`,
    /// `(11, 4, 0)`, `(0, 4, -9)`, `(0, 4, 9)` and yaws `3PI/2`, `PI/2`, `PI`, `0`.
    #[test]
    fn boat_sides_have_the_pivots_and_yaws_the_java_sets() {
        let b = BoatModel::new();
        let want = [
            ([-11.0, 4.0, 0.0], PI * 3.0 / 2.0),
            ([11.0, 4.0, 0.0], PI / 2.0),
            ([0.0, 4.0, -9.0], PI),
            ([0.0, 4.0, 9.0], 0.0),
        ];
        for (n, (pivot, yaw)) in want.iter().enumerate() {
            let p = b.model.get(b.boat_sides[n + 1]);
            assert_eq!(p.texture_offset, [0, 0], "side {}", n + 1);
            assert_eq!(p.cubes[0].pos1, [-10.0, -7.0, -1.0], "side {}", n + 1);
            assert_eq!(p.cubes[0].pos2, [10.0, -1.0, 1.0], "side {}", n + 1);
            assert_eq!(p.rotation_point, *pivot, "side {}", n + 1);
            assert_eq!(p.rotate_angle, [0.0, *yaw, 0.0], "side {}", n + 1);
        }
        assert_eq!(b.model.roots().len(), 5, "the boat is exactly five parts");
    }

    /// java: `sideModels[0].addBox(-10, -8, -1, 20, 16, 2, 0)` and
    /// `sideModels[5].addBox(-9, -7, -1, 18, 14, 1, 0)` with `rotX = -PI/2`.
    #[test]
    fn minecart_bottom_and_floor_plate_match_the_java() {
        let c = MinecartModel::new();
        let bottom = c.model.get(c.side_models[0]);
        assert_eq!(bottom.texture_offset, [0, 10]);
        assert_eq!(bottom.cubes[0].pos1, [-10.0, -8.0, -1.0]);
        assert_eq!(bottom.cubes[0].pos2, [10.0, 8.0, 1.0]);
        assert_eq!(bottom.rotate_angle, [PI / 2.0, 0.0, 0.0]);

        let floor = c.model.get(c.side_models[5]);
        assert_eq!(floor.texture_offset, [44, 10]);
        assert_eq!(floor.cubes[0].pos1, [-9.0, -7.0, -1.0]);
        assert_eq!(floor.cubes[0].pos2, [9.0, 7.0, 0.0]);
        assert_eq!(floor.rotation_point, [0.0, 4.0, 0.0]);
        assert_eq!(floor.rotate_angle, [-(PI / 2.0), 0.0, 0.0]);
        // Six parts, not the seven the Java's array is sized for.
        assert_eq!(c.model.roots().len(), 6);
    }

    /// The minecart's sides are the boat's sides with `i = 20` instead of `24`.
    #[test]
    fn minecart_sides_are_narrower_than_the_boats() {
        let c = MinecartModel::new();
        let b = BoatModel::new();
        let cs = &c.model.get(c.side_models[1]).cubes[0];
        let bs = &b.model.get(b.boat_sides[1]).cubes[0];
        assert_eq!(cs.pos2[0] - cs.pos1[0], 16.0, "i - 4 with i = 20");
        assert_eq!(bs.pos2[0] - bs.pos1[0], 20.0, "i - 4 with i = 24");
        assert_eq!(cs.pos2[1] - cs.pos1[1], 8.0, "j = 8");
        assert_eq!(bs.pos2[1] - bs.pos1[1], 6.0, "j = 6");
        assert_eq!(
            c.model.get(c.side_models[1]).rotation_point,
            [-9.0, 4.0, 0.0]
        );
    }

    /// java: `ModelMinecart.render` — `sideModels[5].rotationPointY = 4.0F - p_78088_4_`, and
    /// `RenderMinecart.doRender` (`:108`) passes the literal `-0.1F` in that slot at the model's
    /// only call site. The plate is therefore fixed at `4.1`.
    ///
    /// This test used to feed `age_in_ticks` and assert that at tick 3 the plate had moved to
    /// `1.0` — the bug written into the assertion. The driver passes
    /// `ticksExisted + partialTicks` there, so that is a plate falling out of the cart at 1/16
    /// block per tick, not an animation vanilla has.
    #[test]
    fn minecart_floor_plate_is_fixed_and_does_not_ride_the_clock() {
        let mut c = MinecartModel::new();
        let mut p = AnimParams::still();
        c.set_rotation_angles(&p);
        let y = c.model.get(c.side_models[5]).rotation_point[1];
        assert!((y - 4.1).abs() < 1e-6, "4.0 - (-0.1) = 4.1, got {y}");

        // The negative control that bites: an aged cart's plate must not have moved.
        p.age_in_ticks = 3.0;
        c.set_rotation_angles(&p);
        let aged = c.model.get(c.side_models[5]).rotation_point[1];
        assert!(
            (aged - 4.1).abs() < 1e-6,
            "the plate must not track the clock, got {aged}"
        );
        p.age_in_ticks = 1200.0;
        c.set_rotation_angles(&p);
        assert!((c.model.get(c.side_models[5]).rotation_point[1] - 4.1).abs() < 1e-6);
    }

    /// java: `ModelLeashKnot()` is `this(0, 0, 32, 32)` — a 32x32 texture, not the 64x32 default.
    #[test]
    fn leash_knot_is_one_box_on_a_32x32_texture() {
        let k = LeashKnotModel::new();
        assert_eq!(k.model.texture_width, 32);
        assert_eq!(k.model.texture_height, 32);
        let p = k.model.get(k.knot);
        assert_eq!(p.texture_width, 32.0);
        assert_eq!(p.cubes[0].pos1, [-3.0, -6.0, -3.0]);
        assert_eq!(p.cubes[0].pos2, [3.0, 2.0, 3.0]);
        assert_eq!(p.rotation_point, [0.0, 0.0, 0.0]);
        // -X face: u = 0..6 of 32, v = 6..14 of 32.
        assert_eq!(p.cubes[0].quads[1].verts[1].uv, [0.0, 6.0 / 32.0]);
        assert_eq!(p.cubes[0].quads[1].verts[3].uv, [6.0 / 32.0, 14.0 / 32.0]);
    }

    /// java: `field_110723_a.rotateAngleY = netHeadYaw / (180F / PI)`, same for the pitch on X.
    #[test]
    fn leash_knot_faces_the_head_yaw_and_pitch() {
        let mut k = LeashKnotModel::new();
        let mut p = AnimParams::still();
        p.net_head_yaw = 90.0;
        p.head_pitch = -30.0;
        k.set_rotation_angles(&p);
        let a = k.model.get(k.knot).rotate_angle;
        assert!((a[1] - PI / 2.0).abs() < 1e-6);
        assert!((a[0] + PI / 6.0).abs() < 1e-6);
        assert_eq!(a[2], 0.0);
    }

    /// java: `if (p_i1170_2_) { this.base = new ModelRenderer(this, "base"); ... }`
    #[test]
    fn ender_crystal_base_is_exactly_one_extra_part() {
        let with = EnderCrystalModel::new(true);
        let without = EnderCrystalModel::new(false);
        assert_eq!(with.model.parts().len(), without.model.parts().len() + 1);
        assert_eq!(without.model.parts().len(), 2);
        assert!(without.base.is_none());

        let b = with.model.get(with.base.expect("base part"));
        assert_eq!(b.texture_offset, [0, 16]);
        assert_eq!(b.cubes[0].pos1, [-6.0, 0.0, -6.0]);
        assert_eq!(b.cubes[0].pos2, [6.0, 4.0, 6.0]);
    }

    /// java: the glass is `setTextureOffset(0, 0)`, the cube `setTextureOffset(32, 0)`, both
    /// `addBox(-4, -4, -4, 8, 8, 8)` through the six-arg overload (`scaleFactor = 0`).
    #[test]
    fn ender_crystal_shells_are_two_identical_boxes_at_different_uvs() {
        let c = EnderCrystalModel::new(true);
        for id in [c.glass, c.cube] {
            let p = c.model.get(id);
            assert_eq!(p.cubes[0].pos1, [-4.0, -4.0, -4.0]);
            assert_eq!(p.cubes[0].pos2, [4.0, 4.0, 4.0]);
            assert_eq!(p.rotation_point, [0.0, 0.0, 0.0]);
        }
        assert_eq!(c.model.get(c.glass).texture_offset, [0, 0]);
        assert_eq!(c.model.get(c.cube).texture_offset, [32, 0]);
        // Creation order is glass (a field initialiser), then cube, then base.
        assert_eq!(c.model.parts()[0].name, "glass");
        assert_eq!(c.model.parts()[1].name, "cube");
    }

    /// `render` draws the glass **twice** and the cube once, plus the base — four boxes.
    #[test]
    fn ender_crystal_render_draws_the_glass_twice() {
        let quad_verts = 6 * 6; // six faces, six vertices each
        for (base, boxes) in [(true, 4), (false, 3)] {
            let mut c = EnderCrystalModel::new(base);
            let mut out = Vec::new();
            c.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
            assert_eq!(out.len(), boxes * quad_verts, "base = {base}");
        }
        // The shells float above the base, always — the `0.8` translate, doubled by scale(2).
        let c = EnderCrystalModel::new(true);
        let mut p = AnimParams::still();
        p.age_in_ticks = 1.0;
        let mut out = Vec::new();
        c.render(&p, Mat4::IDENTITY, &mut out);
        let base_top = out[..quad_verts]
            .iter()
            .fold(f32::MIN, |a, v| a.max(v.pos[1]));
        let shell_top = out[quad_verts..]
            .iter()
            .fold(f32::MIN, |a, v| a.max(v.pos[1]));
        assert!(shell_top > base_top, "{shell_top} vs {base_top}");
    }

    #[test]
    fn spawn_owned_inner_rotation_keeps_crystals_out_of_phase_and_ignores_generic_age() {
        let crystal = EnderCrystalModel::new(true);
        let mut first_params = AnimParams::still();
        first_params.age_in_ticks = 1200.0;
        first_params.partial_ticks = 0.25;
        first_params.entity.crystal_inner_rotation = 12.0;
        let mut second_params = first_params.clone();
        second_params.entity.crystal_inner_rotation = 137.0;

        let mut first = Vec::new();
        let mut second = Vec::new();
        crystal.render(&first_params, Mat4::IDENTITY, &mut first);
        crystal.render(&second_params, Mat4::IDENTITY, &mut second);

        assert_ne!(
            first, second,
            "distinct constructor phases must change the shells"
        );
        assert_eq!(
            first,
            {
                let mut age_zero = first_params.clone();
                age_zero.age_in_ticks = 0.0;
                let mut expected = Vec::new();
                crystal.render(&age_zero, Mat4::IDENTITY, &mut expected);
                expected
            },
            "generic entity age must not be added to the crystal's own phase"
        );
    }

    /// java: `RenderEnderCrystal.doRender` (`:28-34`). The bob is
    /// bounded by `0.4` blocks *before* the model's `scale(2)`, and the spin is three degrees a
    /// tick, so a crystal turns once every two seconds.
    #[test]
    fn ender_crystal_motion_is_bounded_and_never_the_raw_tick_counter() {
        // A minute of uptime: the pre-fix bug fed 1200 straight in as a height in blocks.
        for ticks in 0..2400 {
            let (spin, bob) = ender_crystal_motion(ticks as f32);
            assert!((0.0..=0.4 + 1e-6).contains(&bob), "tick {ticks}: bob {bob}");
            assert!((spin - ticks as f32 * 3.0).abs() < 1e-3, "tick {ticks}");
        }
        // The easing: `sin/2 + 0.5` is 0.5 at t = 0, and `0.5^2 + 0.5 = 0.75`, times 0.2.
        let (spin, bob) = ender_crystal_motion(0.0);
        assert_eq!(spin, 0.0);
        assert!((bob - 0.15).abs() < 1e-6, "{bob}");
        // sin(f * 0.2) peaks at f = PI/2 / 0.2; there `f1 = 1` and `bob = 1 * 0.2 * (1 + 1) / ...`
        let peak = core::f32::consts::FRAC_PI_2 / 0.2;
        let (_, bob) = ender_crystal_motion(peak);
        assert!((bob - 0.4).abs() < 1e-4, "{bob}");
    }

    /// The runaway the audit found: with the generic caller's `age_in_ticks`, the shells must
    /// stay within half a block of where they start no matter how long the client has been up.
    #[test]
    fn a_crystal_does_not_fly_off_into_the_sky_after_a_minute() {
        let c = EnderCrystalModel::new(true);
        let top_at = |ticks: f32| {
            let mut p = AnimParams::still();
            p.age_in_ticks = ticks;
            let mut out = Vec::new();
            c.render(&p, Mat4::IDENTITY, &mut out);
            out.iter().fold(f32::MIN, |a, v| a.max(v.pos[1]))
        };
        let early = top_at(1.0);
        let late = top_at(1200.0);
        // The bob's whole range is 0.4 blocks, doubled by the model's own scale(2).
        assert!((late - early).abs() <= 0.8 + 1e-3, "{early} -> {late}");
    }

    /// java: `ModelArmorStandArmor` is a plain `ModelBiped(size, 0, 64, 32)`.
    #[test]
    fn armor_stand_armor_is_an_unmodified_biped() {
        let a = ArmorStandArmorModel::new(0.0);
        assert_eq!(a.model.texture_width, 64);
        assert_eq!(a.model.texture_height, 32);
        let head = a.model.get(a.parts.head);
        assert_eq!(head.cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(head.cubes[0].pos2, [4.0, 0.0, 4.0]);
        assert_eq!(a.model.get(a.parts.body).cubes[0].pos2, [4.0, 12.0, 2.0]);
        assert_eq!(
            a.model.get(a.parts.right_arm).rotation_point,
            [-5.0, 2.0, 0.0]
        );
        assert_eq!(
            a.model.get(a.parts.left_leg).rotation_point,
            [1.9, 12.0, 0.0]
        );
        assert!(a.model.get(a.parts.left_arm).mirror);
        assert!(a.model.get(a.parts.left_leg).mirror);
    }

    /// java: the six replaced parts in `ModelArmorStand`'s constructor. `bipedHeadwear` survives.
    #[test]
    fn armor_stand_replaces_six_limbs_and_keeps_the_hat() {
        let s = ArmorStandModel::sized(0.0);
        let m = &s.armor.model;
        let q = s.armor.parts;

        let head = m.get(q.head);
        assert_eq!(head.texture_offset, [0, 0]);
        assert_eq!(
            head.cubes.len(),
            1,
            "the biped box must be gone, not stacked"
        );
        assert_eq!(head.cubes[0].pos1, [-1.0, -7.0, -1.0]);
        assert_eq!(head.cubes[0].pos2, [1.0, 0.0, 1.0]);

        let body = m.get(q.body);
        assert_eq!(body.texture_offset, [0, 26]);
        assert_eq!(body.cubes[0].pos1, [-6.0, 0.0, -1.5]);
        assert_eq!(body.cubes[0].pos2, [6.0, 3.0, 1.5]);

        assert_eq!(m.get(q.right_arm).texture_offset, [24, 0]);
        assert_eq!(m.get(q.left_arm).texture_offset, [32, 16]);
        assert!(m.get(q.left_arm).mirror);
        assert_eq!(m.get(q.right_leg).cubes[0].pos2, [1.0, 11.0, 1.0]);
        assert_eq!(m.get(q.left_leg).rotation_point, [1.9, 12.0, 0.0]);

        // Untouched by ModelArmorStand: still the biped's inflated 8x8x8 hat at (32, 0).
        let hat = m.get(q.headwear);
        assert_eq!(hat.texture_offset, [32, 0]);
        assert_eq!(hat.cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(hat.cubes[0].quads[4].verts[1].pos, [-4.5, -8.5, -4.5]);

        // The four frame parts.
        assert_eq!(m.get(s.stand_right_side).cubes[0].pos1, [-3.0, 3.0, -1.0]);
        assert_eq!(m.get(s.stand_left_side).cubes[0].pos1, [1.0, 3.0, -1.0]);
        assert_eq!(m.get(s.stand_waist).cubes[0].pos2, [4.0, 12.0, 1.0]);
        assert_eq!(m.get(s.stand_base).cubes[0].pos1, [-6.0, 11.0, -6.0]);
        assert_eq!(m.get(s.stand_base).rotation_point, [0.0, 12.0, 0.0]);
    }

    /// java: `rotateAngleX = 0.017453292F * getRightArmRotation().getX()`, and the leg pivots that
    /// `ModelArmorStand` pushes back to `y = 12` after `super` set them to `11`.
    #[test]
    fn armor_stand_pose_applies_the_default_rotations_in_radians() {
        let mut s = ArmorStandModel::new();
        s.set_rotation_angles(&AnimParams::still());
        let m = &s.armor.model;
        let q = s.armor.parts;
        let r = ARMOR_STAND_DEG_TO_RAD;

        // DEFAULT_RIGHTARM_ROTATION = (-15, 0, 10)
        assert_eq!(m.get(q.right_arm).rotate_angle, [r * -15.0, 0.0, r * 10.0]);
        // DEFAULT_LEFTARM_ROTATION = (-10, 0, -10)
        assert_eq!(m.get(q.left_arm).rotate_angle, [r * -10.0, 0.0, r * -10.0]);
        // DEFAULT_RIGHTLEG_ROTATION = (1, 0, 1)
        assert_eq!(m.get(q.right_leg).rotate_angle, [r, 0.0, r]);
        assert_eq!(m.get(q.head).rotation_point, [0.0, 1.0, 0.0]);
        assert_eq!(m.get(q.left_leg).rotation_point, [1.9, 12.0, 0.0]);
        assert_eq!(m.get(q.right_leg).rotation_point, [-1.9, 12.0, 0.0]);
        // The hat tracks the head, pivot included.
        assert_eq!(m.get(q.headwear).rotation_point, [0.0, 1.0, 0.0]);

        // Without the stand's override the armour leaves the legs at 11.
        let mut a = ArmorStandArmorModel::new(0.0);
        a.set_rotation_angles(&AnimParams::still());
        assert_eq!(
            a.model.get(a.parts.left_leg).rotation_point,
            [1.9, 11.0, 0.0]
        );
    }

    /// java: `bipedLeftArm.showModel = getShowArms()` and
    /// `standBase.showModel = !hasNoBasePlate()`.
    ///
    /// The flags arrive on `AnimParams::entity.armor_stand`, never by poking `armor.pose`: one
    /// `ArmorStandModel` is shared by every stand in the world, so the pose field is scratch
    /// that `set_rotation_angles` overwrites in full on every call. This test used to write it
    /// directly, which is how the shared-state carryover stayed invisible.
    #[test]
    fn armor_stand_arms_and_base_plate_follow_the_entity_flags() {
        let mut s = ArmorStandModel::new();
        s.set_rotation_angles(&AnimParams::still());
        assert!(
            !s.armor.model.get(s.armor.parts.left_arm).show_model,
            "arms off by default"
        );
        assert!(!s.armor.model.get(s.armor.parts.right_arm).show_model);
        assert!(
            s.armor.model.get(s.stand_base).show_model,
            "base plate on by default"
        );

        let mut armed = AnimParams::still();
        armed.entity.armor_stand = Some(ArmorStandPose {
            show_arms: true,
            no_base_plate: true,
            rotation_yaw: 90.0,
            ..ArmorStandPose::default()
        });
        s.set_rotation_angles(&armed);
        assert!(s.armor.model.get(s.armor.parts.left_arm).show_model);
        assert!(!s.armor.model.get(s.stand_base).show_model);
        let base_yaw = s.armor.model.get(s.stand_base).rotate_angle;
        assert_eq!(base_yaw, [0.0, ARMOR_STAND_DEG_TO_RAD * -90.0, 0.0]);

        // Hiding a part removes its geometry from the draw.
        let mut out_hidden = Vec::new();
        s.render(&armed, Mat4::IDENTITY, &mut out_hidden);
        let mut plated = armed;
        plated.entity.armor_stand.as_mut().unwrap().no_base_plate = false;
        s.set_rotation_angles(&plated);
        let mut out_shown = Vec::new();
        s.render(&plated, Mat4::IDENTITY, &mut out_shown);
        assert_eq!(out_shown.len(), out_hidden.len() + 36);

        // The negative control for the whole change: posing a *second* stand that carries no
        // pose of its own must fall back to `DEFAULT_*_ROTATION`, not inherit the armed stand's
        // arms. With the old `if let Some(..)` this assertion fails.
        s.set_rotation_angles(&AnimParams::still());
        assert!(
            !s.armor.model.get(s.armor.parts.left_arm).show_model,
            "a poseless stand must not inherit the previous stand's arms"
        );
        assert!(s.armor.model.get(s.stand_base).show_model);
    }

    /// The four stand-frame parts are drawn on top of the seven biped ones.
    #[test]
    fn armor_stand_draws_the_frame_after_the_biped() {
        let mut s = ArmorStandModel::new();
        let mut p = AnimParams::still();
        p.entity.armor_stand = Some(ArmorStandPose {
            show_arms: true,
            ..ArmorStandPose::default()
        });
        p.entity.child = false;
        let mut out = Vec::new();
        s.pose_and_render(&p, Mat4::IDENTITY, &mut out);
        // 7 biped parts + 4 frame parts, one box each.
        assert_eq!(out.len(), 11 * 6 * 6);
        // The last part drawn is standBase: pivot y = 12 plus a box at y = 11..12, so every one
        // of its vertices sits below the feet at 23..24 model units (1.4375..1.5 blocks).
        for v in &out[out.len() - 36..] {
            assert!(
                v.pos[1] >= 1.43,
                "the base plate must be drawn last, got y = {}",
                v.pos[1]
            );
        }
    }

    #[test]
    fn every_object_models_uvs_land_inside_its_texture() {
        assert_uvs_in_range(&mut BoatModel::new(), "boat");
        assert_uvs_in_range(&mut LeashKnotModel::new(), "leash knot");
        assert_uvs_in_range(&mut EnderCrystalModel::new(true), "ender crystal");
        assert_uvs_in_range(&mut ArmorStandModel::new(), "armor stand");
        assert_uvs_in_range(&mut ArmorStandArmorModel::new(0.0), "armor stand armor");
    }
}
