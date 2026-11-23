//! Hostile-mob models.
//!
//! java: `ModelCreeper.java`, `ModelSpider.java`, `ModelSlime.java`, `ModelMagmaCube.java`,
//! `ModelBlaze.java`, `ModelEnderman.java`, `ModelGhast.java`, `ModelSilverfish.java`,
//! `ModelEnderMite.java`, `ModelGuardian.java`, `ModelDragon.java`, `ModelWither.java`.
//!
//! Ported types, one per Java class: [`CreeperModel`], [`SpiderModel`], [`SlimeModel`],
//! [`MagmaCubeModel`], [`BlazeModel`], [`EndermanModel`], [`GhastModel`], [`SilverfishModel`],
//! [`EndermiteModel`], [`GuardianModel`], [`DragonModel`], [`WitherModel`].
//!
//! [`super::biped`] is the reference port and this file follows its conventions:
//!
//! * A Java `this.foo = new ModelRenderer(...)` that *replaces* an already-built field leaves the
//!   old object orphaned in `boxList`. Here that is [`rebuild`], which refills the part in place.
//!   [`EndermanModel`] (six replaced `ModelBiped` parts) and [`SlimeModel`] (the body box swapped
//!   for the smaller inner one) are the two users.
//! * Trigonometry goes through [`mc_math::sin`]/[`mc_math::cos`] — the 65536-entry vanilla LUT,
//!   never `f32::sin` (hazard H1).
//! * Degrees become radians through [`model_radians`], preserving vanilla's division.
//!
//! # Entity state these models read
//!
//! Vanilla downcasts the `Entity` argument and reads mob-specific fields.
//! [`crate::entity::anim::EntityAnimState`] has no field for most of them, so each site below
//! maps onto the closest one and says so in a `// java:` comment:
//!
//! | Java | Mapped to |
//! |---|---|
//! | `ModelEnderman.isCarrying` | [`EntityAnimState::carrying_block`](crate::entity::anim::EntityAnimState::carrying_block) |
//! | `ModelEnderman.isAttacking` | [`EntityAnimState::aggressive`](crate::entity::anim::EntityAnimState::aggressive) |
//! | `EntityMagmaCube.squishFactor` (interpolated) | [`EntityAnimState::squish_factor`](crate::entity::anim::EntityAnimState::squish_factor) |
//! | `EntityGuardian.func_175471_a` (tail phase) | [`EntityAnimState::tail_rotation`](crate::entity::anim::EntityAnimState::tail_rotation) |
//! | `EntityGuardian.func_175469_o` (spike extension) | [`EntityAnimState::spikes_animation`](crate::entity::anim::EntityAnimState::spikes_animation) |
//!
//! `ModelSpider`, `ModelSlime`, `ModelSilverfish`, `ModelEnderMite`, `ModelGhast` and
//! `ModelBlaze` read nothing off the entity at all; `ModelSlime`'s and `ModelCreeper`'s
//! constructor arguments are pure geometry.
//!
//! # What could not be represented
//!
//! `ModelGuardian.setRotationAngles` asks `Minecraft.getMinecraft().getRenderViewEntity()` for
//! the camera (or the guardian's beam target) and aims the eye at it. That needs world
//! positions, which a model cannot see from here, so the arithmetic lives in
//! [`GuardianModel::look_at`] for the renderer to call; the eye keeps its default pivot until it
//! does.

use glam::{Mat4, Vec3};
use mc_math::{cos, sin, sqrt_f, JavaRandom};

use crate::entity::anim::{model_radians, AnimParams, DragonRenderState, EntityModel};
use crate::entity::model::{Model, PartId};
use crate::entity::EntityVertex;

use core::f32::consts::PI;

/// Clears a part and rebuilds it with a new texture offset, mirror flag, box and pivot.
///
/// java: the `this.bipedBody = new ModelRenderer(...)` pattern in `ModelEnderman`, which drops
/// the `ModelBiped` part on the floor and builds a fresh one in its place.
#[allow(clippy::too_many_arguments)] // one-for-one with the Java statement group it replaces
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
// Creeper
// ---------------------------------------------------------------------------------------------

/// java: `ModelCreeper`
#[derive(Clone, Debug)]
pub struct CreeperModel {
    /// The part tree.
    pub model: Model,
    /// java: `head`
    pub head: PartId,
    /// java: `creeperArmor` — built by the constructor and **never drawn**: `ModelCreeper.render`
    /// lists head, body and the four legs only, and `LayerCreeperCharge` renders a second whole
    /// `ModelCreeper` rather than this part.
    pub creeper_armor: PartId,
    /// java: `body`
    pub body: PartId,
    /// java: `leg1`..`leg4`, 0-based.
    pub legs: [PartId; 4],
}

impl CreeperModel {
    /// java: `ModelCreeper(float p_i46366_1_)`
    #[must_use]
    pub fn new(size: f32) -> Self {
        // java: `int i = 6;`
        let i = 6.0_f32;
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, size)
            .at(0.0, i, 0.0)
            .id();
        let creeper_armor = m
            .part("creeper_armor", 32, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, size + 0.5)
            .at(0.0, i, 0.0)
            .id();
        let body = m
            .part("body", 16, 16)
            .cube(-4.0, 0.0, -2.0, 8, 12, 4, size)
            .at(0.0, i, 0.0)
            .id();
        let leg1 = m
            .part("leg1", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 6, 4, size)
            .at(-2.0, 12.0 + i, 4.0)
            .id();
        let leg2 = m
            .part("leg2", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 6, 4, size)
            .at(2.0, 12.0 + i, 4.0)
            .id();
        let leg3 = m
            .part("leg3", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 6, 4, size)
            .at(-2.0, 12.0 + i, -4.0)
            .id();
        let leg4 = m
            .part("leg4", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 6, 4, size)
            .at(2.0, 12.0 + i, -4.0)
            .id();
        Self {
            model: m,
            head,
            creeper_armor,
            body,
            legs: [leg1, leg2, leg3, leg4],
        }
    }
}

impl EntityModel for CreeperModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelCreeper.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let l = self.legs;
        let m = &mut self.model;
        m.get_mut(self.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(self.head).rotate_angle[0] = model_radians(p.head_pitch);
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        m.get_mut(l[0]).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
        m.get_mut(l[1]).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(l[2]).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(l[3]).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
    }

    /// java: `ModelCreeper.render` — note the absent `creeperArmor`.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let l = self.legs;
        self.model.emit_some(
            &[self.head, self.body, l[0], l[1], l[2], l[3]],
            p.scale,
            base,
            out,
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Spider
// ---------------------------------------------------------------------------------------------

/// java: `ModelSpider`
#[derive(Clone, Debug)]
pub struct SpiderModel {
    /// The part tree.
    pub model: Model,
    /// java: `spiderHead`
    pub head: PartId,
    /// java: `spiderNeck`
    pub neck: PartId,
    /// java: `spiderBody`
    pub body: PartId,
    /// java: `spiderLeg1`..`spiderLeg8`, 0-based — odd Java indices are the left-hand (negative
    /// X) legs, even ones the right.
    pub legs: [PartId; 8],
}

impl SpiderModel {
    /// java: `ModelSpider()`
    #[must_use]
    pub fn new() -> Self {
        // java: `float f = 0.0F; int i = 15;`
        let f = 0.0_f32;
        let i = 15.0_f32;
        let mut m = Model::new(64, 32);
        let head = m
            .part("spider_head", 32, 4)
            .cube(-4.0, -4.0, -8.0, 8, 8, 8, f)
            .at(0.0, i, -3.0)
            .id();
        let neck = m
            .part("spider_neck", 0, 0)
            .cube(-3.0, -3.0, -3.0, 6, 6, 6, f)
            .at(0.0, i, 0.0)
            .id();
        let body = m
            .part("spider_body", 0, 12)
            .cube(-5.0, -4.0, -6.0, 10, 8, 12, f)
            .at(0.0, i, 9.0)
            .id();
        let leg1 = m
            .part("spider_leg1", 18, 0)
            .cube(-15.0, -1.0, -1.0, 16, 2, 2, f)
            .at(-4.0, i, 2.0)
            .id();
        let leg2 = m
            .part("spider_leg2", 18, 0)
            .cube(-1.0, -1.0, -1.0, 16, 2, 2, f)
            .at(4.0, i, 2.0)
            .id();
        let leg3 = m
            .part("spider_leg3", 18, 0)
            .cube(-15.0, -1.0, -1.0, 16, 2, 2, f)
            .at(-4.0, i, 1.0)
            .id();
        let leg4 = m
            .part("spider_leg4", 18, 0)
            .cube(-1.0, -1.0, -1.0, 16, 2, 2, f)
            .at(4.0, i, 1.0)
            .id();
        let leg5 = m
            .part("spider_leg5", 18, 0)
            .cube(-15.0, -1.0, -1.0, 16, 2, 2, f)
            .at(-4.0, i, 0.0)
            .id();
        let leg6 = m
            .part("spider_leg6", 18, 0)
            .cube(-1.0, -1.0, -1.0, 16, 2, 2, f)
            .at(4.0, i, 0.0)
            .id();
        let leg7 = m
            .part("spider_leg7", 18, 0)
            .cube(-15.0, -1.0, -1.0, 16, 2, 2, f)
            .at(-4.0, i, -1.0)
            .id();
        let leg8 = m
            .part("spider_leg8", 18, 0)
            .cube(-1.0, -1.0, -1.0, 16, 2, 2, f)
            .at(4.0, i, -1.0)
            .id();
        Self {
            model: m,
            head,
            neck,
            body,
            legs: [leg1, leg2, leg3, leg4, leg5, leg6, leg7, leg8],
        }
    }
}

impl Default for SpiderModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SpiderModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelSpider.setRotationAngles`
    // `0.3926991F` is vanilla's own rounding of PI/8; using `FRAC_PI_8` would change the f32.
    #[allow(clippy::approx_constant)]
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let l = self.legs;
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        let m = &mut self.model;

        m.get_mut(self.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(self.head).rotate_angle[0] = model_radians(p.head_pitch);

        let f = PI / 4.0;
        m.get_mut(l[0]).rotate_angle[2] = -f;
        m.get_mut(l[1]).rotate_angle[2] = f;
        m.get_mut(l[2]).rotate_angle[2] = -f * 0.74;
        m.get_mut(l[3]).rotate_angle[2] = f * 0.74;
        m.get_mut(l[4]).rotate_angle[2] = -f * 0.74;
        m.get_mut(l[5]).rotate_angle[2] = f * 0.74;
        m.get_mut(l[6]).rotate_angle[2] = -f;
        m.get_mut(l[7]).rotate_angle[2] = f;

        let f1 = -0.0_f32;
        // java: `float f2 = 0.3926991F;` — PI/8, kept as vanilla's literal.
        let f2 = 0.3926991_f32;
        m.get_mut(l[0]).rotate_angle[1] = f2 * 2.0 + f1;
        m.get_mut(l[1]).rotate_angle[1] = -f2 * 2.0 - f1;
        m.get_mut(l[2]).rotate_angle[1] = f2 * 1.0 + f1;
        m.get_mut(l[3]).rotate_angle[1] = -f2 * 1.0 - f1;
        m.get_mut(l[4]).rotate_angle[1] = -f2 * 1.0 + f1;
        m.get_mut(l[5]).rotate_angle[1] = f2 * 1.0 - f1;
        m.get_mut(l[6]).rotate_angle[1] = -f2 * 2.0 + f1;
        m.get_mut(l[7]).rotate_angle[1] = f2 * 2.0 - f1;

        let f3 = -(cos(swing * 0.6662 * 2.0) * 0.4) * amount;
        let f4 = -(cos(swing * 0.6662 * 2.0 + PI) * 0.4) * amount;
        let f5 = -(cos(swing * 0.6662 * 2.0 + PI / 2.0) * 0.4) * amount;
        let f6 = -(cos(swing * 0.6662 * 2.0 + PI * 3.0 / 2.0) * 0.4) * amount;
        let f7 = (sin(swing * 0.6662) * 0.4).abs() * amount;
        let f8 = (sin(swing * 0.6662 + PI) * 0.4).abs() * amount;
        let f9 = (sin(swing * 0.6662 + PI / 2.0) * 0.4).abs() * amount;
        let f10 = (sin(swing * 0.6662 + PI * 3.0 / 2.0) * 0.4).abs() * amount;

        m.get_mut(l[0]).rotate_angle[1] += f3;
        m.get_mut(l[1]).rotate_angle[1] += -f3;
        m.get_mut(l[2]).rotate_angle[1] += f4;
        m.get_mut(l[3]).rotate_angle[1] += -f4;
        m.get_mut(l[4]).rotate_angle[1] += f5;
        m.get_mut(l[5]).rotate_angle[1] += -f5;
        m.get_mut(l[6]).rotate_angle[1] += f6;
        m.get_mut(l[7]).rotate_angle[1] += -f6;
        m.get_mut(l[0]).rotate_angle[2] += f7;
        m.get_mut(l[1]).rotate_angle[2] += -f7;
        m.get_mut(l[2]).rotate_angle[2] += f8;
        m.get_mut(l[3]).rotate_angle[2] += -f8;
        m.get_mut(l[4]).rotate_angle[2] += f9;
        m.get_mut(l[5]).rotate_angle[2] += -f9;
        m.get_mut(l[6]).rotate_angle[2] += f10;
        m.get_mut(l[7]).rotate_angle[2] += -f10;
    }
}

// ---------------------------------------------------------------------------------------------
// Slime
// ---------------------------------------------------------------------------------------------

/// java: `ModelSlime`
///
/// The constructor argument doubles as the texture V offset and as the "does this one have a
/// face" flag: `new ModelSlime(16)` is `RenderSlime`'s **main** model — the six-unit body
/// plus eyes and mouth — and `new ModelSlime(0)` is `LayerSlimeGel`'s **outer** eight-unit
/// shell, drawn translucent on top of it. This comment used to have the two the wrong way
/// round.
#[derive(Clone, Debug)]
pub struct SlimeModel {
    /// The part tree.
    pub model: Model,
    /// java: `slimeBodies`
    pub bodies: PartId,
    /// java: `slimeRightEye` — `null` for the core, so [`None`].
    pub right_eye: Option<PartId>,
    /// java: `slimeLeftEye`
    pub left_eye: Option<PartId>,
    /// java: `slimeMouth`
    pub mouth: Option<PartId>,
}

impl SlimeModel {
    /// java: `ModelSlime(int p_i1157_1_)`
    #[must_use]
    pub fn new(boxes: i32) -> Self {
        let mut m = Model::new(64, 32);
        let bodies = m
            .part("slime_bodies", 0, boxes)
            .cube(-4.0, 16.0, -4.0, 8, 8, 8, 0.0)
            .id();

        if boxes > 0 {
            // java: `this.slimeBodies = new ModelRenderer(this, 0, p_i1157_1_)` a second time —
            // the 8x8x8 renderer above is orphaned, so refill the part with the 6x6x6 box.
            rebuild(
                &mut m,
                bodies,
                0,
                boxes,
                false,
                [-3.0, 17.0, -3.0],
                [6, 6, 6],
                0.0,
                [0.0, 0.0, 0.0],
            );
            let right_eye = m
                .part("slime_right_eye", 32, 0)
                .cube(-3.25, 18.0, -3.5, 2, 2, 2, 0.0)
                .id();
            let left_eye = m
                .part("slime_left_eye", 32, 4)
                .cube(1.25, 18.0, -3.5, 2, 2, 2, 0.0)
                .id();
            let mouth = m
                .part("slime_mouth", 32, 8)
                .cube(0.0, 21.0, -3.5, 1, 1, 1, 0.0)
                .id();
            Self {
                model: m,
                bodies,
                right_eye: Some(right_eye),
                left_eye: Some(left_eye),
                mouth: Some(mouth),
            }
        } else {
            Self {
                model: m,
                bodies,
                right_eye: None,
                left_eye: None,
                mouth: None,
            }
        }
    }
}

impl EntityModel for SlimeModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelSlime` declares no `setRotationAngles`, so `ModelBase`'s empty one runs. The
    /// squash-and-stretch a slime shows is `RenderSlime.preRenderCallback`'s `GlStateManager`
    /// scale, not a model angle.
    fn set_rotation_angles(&mut self, _p: &AnimParams) {}
}

// ---------------------------------------------------------------------------------------------
// Magma cube
// ---------------------------------------------------------------------------------------------

/// Part names for [`MagmaCubeModel::segments`].
const MAGMA_SEGMENTS: [&str; 8] = [
    "segment0", "segment1", "segment2", "segment3", "segment4", "segment5", "segment6", "segment7",
];

/// java: `ModelMagmaCube`
#[derive(Clone, Debug)]
pub struct MagmaCubeModel {
    /// The part tree.
    pub model: Model,
    /// java: `segments` — the eight one-unit slabs the outer crust is sliced into.
    pub segments: [PartId; 8],
    /// java: `core`
    pub core: PartId,
}

impl MagmaCubeModel {
    /// java: `ModelMagmaCube()`
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let segments = core::array::from_fn(|i| {
            // java: `int j = 0; int k = i;` with two special cases.
            let (j, k) = match i {
                2 => (24, 10),
                3 => (24, 19),
                _ => (0, i as i32),
            };
            m.part(MAGMA_SEGMENTS[i], j, k)
                .cube(-4.0, 16.0 + i as f32, -4.0, 8, 1, 8, 0.0)
                .id()
        });
        let core = m
            .part("core", 0, 16)
            .cube(-2.0, 18.0, -2.0, 4, 4, 4, 0.0)
            .id();
        Self {
            model: m,
            segments,
            core,
        }
    }
}

impl Default for MagmaCubeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for MagmaCubeModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelMagmaCube` has no `setRotationAngles`; the crust only moves through
    /// [`EntityModel::set_living_animations`].
    fn set_rotation_angles(&mut self, _p: &AnimParams) {}

    /// java: `ModelMagmaCube.setLivingAnimations`
    fn set_living_animations(&mut self, p: &AnimParams) {
        // java: `float f = prevSquishFactor + (squishFactor - prevSquishFactor) * partialTickTime`
        // — [`EntityAnimState::squish_factor`], which `LivingRenderState::params` fills from
        // `SpeciesState::Slime`. It used to ride on the generic `tail_rotation` knob, which
        // `decode_entity_state` writes for exactly one species (the wolf), so a magma cube's
        // crust never separated.
        let mut f = p.entity.squish_factor;
        if f < 0.0 {
            f = 0.0;
        }
        let segments = self.segments;
        for (i, &id) in segments.iter().enumerate() {
            self.model.get_mut(id).rotation_point[1] = -(4 - i as i32) as f32 * f * 1.7;
        }
    }

    /// java: `ModelMagmaCube.render` — the core first, then the crust slabs.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_part(self.core, p.scale, base, out);
        self.model.emit_some(&self.segments, p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Blaze
// ---------------------------------------------------------------------------------------------

/// Part names for [`BlazeModel::sticks`].
const BLAZE_STICKS: [&str; 12] = [
    "stick0", "stick1", "stick2", "stick3", "stick4", "stick5", "stick6", "stick7", "stick8",
    "stick9", "stick10", "stick11",
];

/// java: `ModelBlaze`
#[derive(Clone, Debug)]
pub struct BlazeModel {
    /// The part tree.
    pub model: Model,
    /// java: `blazeSticks` — three rings of four rods, all identical geometry.
    pub sticks: [PartId; 12],
    /// java: `blazeHead`
    pub head: PartId,
}

impl BlazeModel {
    /// java: `ModelBlaze()`
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let sticks = core::array::from_fn(|i| {
            m.part(BLAZE_STICKS[i], 0, 16)
                .cube(0.0, 0.0, 0.0, 2, 8, 2, 0.0)
                .id()
        });
        let head = m
            .part("blaze_head", 0, 0)
            .cube(-4.0, -4.0, -4.0, 8, 8, 8, 0.0)
            .id();
        Self {
            model: m,
            sticks,
            head,
        }
    }
}

impl Default for BlazeModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for BlazeModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelBlaze.setRotationAngles` — three loops, each advancing the shared angle `f`
    /// by one radian per rod so the ring is fanned out rather than stacked.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let age = p.age_in_ticks;
        let sticks = self.sticks;

        let mut f = age * PI * -0.1;
        for (i, &id) in sticks.iter().enumerate().take(4) {
            let s = self.model.get_mut(id);
            s.rotation_point[1] = -2.0 + cos(((i * 2) as f32 + age) * 0.25);
            s.rotation_point[0] = cos(f) * 9.0;
            s.rotation_point[2] = sin(f) * 9.0;
            f += 1.0;
        }

        f = PI / 4.0 + age * PI * 0.03;
        for (j, &id) in sticks.iter().enumerate().skip(4).take(4) {
            let s = self.model.get_mut(id);
            s.rotation_point[1] = 2.0 + cos(((j * 2) as f32 + age) * 0.25);
            s.rotation_point[0] = cos(f) * 7.0;
            s.rotation_point[2] = sin(f) * 7.0;
            f += 1.0;
        }

        // java: `0.47123894F` — 3*PI/20, vanilla's baked literal.
        f = 0.47123894 + age * PI * -0.05;
        for (k, &id) in sticks.iter().enumerate().skip(8) {
            let s = self.model.get_mut(id);
            s.rotation_point[1] = 11.0 + cos((k as f32 * 1.5 + age) * 0.5);
            s.rotation_point[0] = cos(f) * 5.0;
            s.rotation_point[2] = sin(f) * 5.0;
            f += 1.0;
        }

        self.model.get_mut(self.head).rotate_angle[1] = model_radians(p.net_head_yaw);
        self.model.get_mut(self.head).rotate_angle[0] = model_radians(p.head_pitch);
    }

    /// java: `ModelBlaze.render` — the head first, then the rods.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_part(self.head, p.scale, base, out);
        self.model.emit_some(&self.sticks, p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Enderman
// ---------------------------------------------------------------------------------------------

/// java: `ModelEnderman extends ModelBiped`
///
/// Ported standalone rather than on top of [`super::biped::BipedModel`]: the Java replaces six of
/// the seven biped parts and its `super(0.0F, -14.0F, 64, 32)` shifts every pivot fourteen units
/// up, so the shared code would be one `#[allow]`-riddled special case. The `ModelBiped` pose is
/// transcribed here as [`EndermanModel::biped_pose`].
#[derive(Clone, Debug)]
pub struct EndermanModel {
    /// The part tree.
    pub model: Model,
    /// java: `bipedHead` — the only part `ModelEnderman` inherits unchanged.
    pub head: PartId,
    /// java: `bipedHeadwear` — rebuilt at (0, 16) and *deflated* by half a unit.
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

impl EndermanModel {
    /// java: `ModelEnderman(float p_i46305_1_)`
    #[must_use]
    pub fn new(size: f32) -> Self {
        // java: `super(0.0F, -14.0F, 64, 32)` — note the *hard-coded* 0.0F model size, so the
        // constructor argument only ever reaches the six parts rebuilt below.
        let y_off = -14.0_f32;
        let mut m = Model::new(64, 32);
        let head = m
            .part("head", 0, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, 0.0)
            .at(0.0, y_off, 0.0)
            .id();
        let headwear = m
            .part("headwear", 32, 0)
            .cube(-4.0, -8.0, -4.0, 8, 8, 8, 0.5)
            .at(0.0, y_off, 0.0)
            .id();
        let body = m
            .part("body", 16, 16)
            .cube(-4.0, 0.0, -2.0, 8, 12, 4, 0.0)
            .at(0.0, y_off, 0.0)
            .id();
        let right_arm = m
            .part("right_arm", 40, 16)
            .cube(-3.0, -2.0, -2.0, 4, 12, 4, 0.0)
            .at(-5.0, 2.0 + y_off, 0.0)
            .id();
        let left_arm = m
            .part("left_arm", 40, 16)
            .mirrored()
            .cube(-1.0, -2.0, -2.0, 4, 12, 4, 0.0)
            .at(5.0, 2.0 + y_off, 0.0)
            .id();
        let right_leg = m
            .part("right_leg", 0, 16)
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, 0.0)
            .at(-1.9, 12.0 + y_off, 0.0)
            .id();
        let left_leg = m
            .part("left_leg", 0, 16)
            .mirrored()
            .cube(-2.0, 0.0, -2.0, 4, 12, 4, 0.0)
            .at(1.9, 12.0 + y_off, 0.0)
            .id();

        // java: `float f = -14.0F;` and six `new ModelRenderer(...)` replacements.
        let f = -14.0_f32;
        let arm = [2, 30, 2];
        let leg = [2, 30, 2];
        rebuild(
            &mut m,
            headwear,
            0,
            16,
            false,
            [-4.0, -8.0, -4.0],
            [8, 8, 8],
            size - 0.5,
            [0.0, f, 0.0],
        );
        rebuild(
            &mut m,
            body,
            32,
            16,
            false,
            [-4.0, 0.0, -2.0],
            [8, 12, 4],
            size,
            [0.0, f, 0.0],
        );
        rebuild(
            &mut m,
            right_arm,
            56,
            0,
            false,
            [-1.0, -2.0, -1.0],
            arm,
            size,
            [-3.0, 2.0 + f, 0.0],
        );
        rebuild(
            &mut m,
            left_arm,
            56,
            0,
            true,
            [-1.0, -2.0, -1.0],
            arm,
            size,
            [5.0, 2.0 + f, 0.0],
        );
        rebuild(
            &mut m,
            right_leg,
            56,
            0,
            false,
            [-1.0, 0.0, -1.0],
            leg,
            size,
            [-2.0, 12.0 + f, 0.0],
        );
        rebuild(
            &mut m,
            left_leg,
            56,
            0,
            true,
            [-1.0, 0.0, -1.0],
            leg,
            size,
            [2.0, 12.0 + f, 0.0],
        );

        Self {
            model: m,
            head,
            headwear,
            body,
            right_arm,
            left_arm,
            right_leg,
            left_leg,
        }
    }

    /// `ModelBiped.setRotationAngles`, transcribed statement for statement.
    ///
    /// java: `ModelBiped.java:129-245`. `heldItemLeft`/`heldItemRight`/`aimedBow` are always the
    /// default on an enderman, but the branches are kept so the pose is the same function of the
    /// inputs as vanilla's.
    ///
    /// java: `this.isSneak` -> `p.entity.sneaking`; `this.swingProgress` ->
    /// `self.model.swing_progress`; `this.isRiding` -> `self.model.is_riding`.
    // `-0.5235988F` is vanilla's own rounding of -PI/6; `FRAC_PI_6` would change the f32.
    #[allow(clippy::approx_constant)]
    pub fn biped_pose(&mut self, p: &AnimParams) {
        let (head, body) = (self.head, self.body);
        let (ra, la) = (self.right_arm, self.left_arm);
        let (rl, ll) = (self.right_leg, self.left_leg);
        let swing = p.limb_swing;
        let amount = p.limb_swing_amount;
        let age = p.age_in_ticks;
        let e = &p.entity;
        let m = &mut self.model;

        m.get_mut(head).rotate_angle[1] = model_radians(p.net_head_yaw);
        m.get_mut(head).rotate_angle[0] = model_radians(p.head_pitch);
        m.get_mut(ra).rotate_angle[0] = cos(swing * 0.6662 + PI) * 2.0 * amount * 0.5;
        m.get_mut(la).rotate_angle[0] = cos(swing * 0.6662) * 2.0 * amount * 0.5;
        m.get_mut(ra).rotate_angle[2] = 0.0;
        m.get_mut(la).rotate_angle[2] = 0.0;
        m.get_mut(rl).rotate_angle[0] = cos(swing * 0.6662) * 1.4 * amount;
        m.get_mut(ll).rotate_angle[0] = cos(swing * 0.6662 + PI) * 1.4 * amount;
        m.get_mut(rl).rotate_angle[1] = 0.0;
        m.get_mut(ll).rotate_angle[1] = 0.0;

        if m.is_riding {
            m.get_mut(ra).rotate_angle[0] += -(PI / 5.0);
            m.get_mut(la).rotate_angle[0] += -(PI / 5.0);
            m.get_mut(rl).rotate_angle[0] = -(PI * 2.0 / 5.0);
            m.get_mut(ll).rotate_angle[0] = -(PI * 2.0 / 5.0);
            m.get_mut(rl).rotate_angle[1] = PI / 10.0;
            m.get_mut(ll).rotate_angle[1] = -(PI / 10.0);
        }

        if e.held_item_left != 0 {
            let a = m.get(la).rotate_angle[0];
            m.get_mut(la).rotate_angle[0] = a * 0.5 - (PI / 10.0) * e.held_item_left as f32;
        }

        m.get_mut(ra).rotate_angle[1] = 0.0;
        m.get_mut(ra).rotate_angle[2] = 0.0;

        match e.held_item_right {
            1 => {
                let a = m.get(ra).rotate_angle[0];
                m.get_mut(ra).rotate_angle[0] = a * 0.5 - (PI / 10.0) * 1.0;
            }
            3 => {
                let a = m.get(ra).rotate_angle[0];
                m.get_mut(ra).rotate_angle[0] = a * 0.5 - (PI / 10.0) * 3.0;
                m.get_mut(ra).rotate_angle[1] = -0.5235988;
            }
            _ => {}
        }

        m.get_mut(la).rotate_angle[1] = 0.0;

        if m.swing_progress > -9990.0 {
            let f = m.swing_progress;
            let body_y = sin(sqrt_f(f) * PI * 2.0) * 0.2;
            m.get_mut(body).rotate_angle[1] = body_y;
            m.get_mut(ra).rotation_point[2] = sin(body_y) * 5.0;
            m.get_mut(ra).rotation_point[0] = -cos(body_y) * 5.0;
            m.get_mut(la).rotation_point[2] = -sin(body_y) * 5.0;
            m.get_mut(la).rotation_point[0] = cos(body_y) * 5.0;
            m.get_mut(ra).rotate_angle[1] += body_y;
            m.get_mut(la).rotate_angle[1] += body_y;
            m.get_mut(la).rotate_angle[0] += body_y;

            let mut f = 1.0 - m.swing_progress;
            f *= f;
            f *= f;
            f = 1.0 - f;
            let f1 = sin(f * PI);
            let head_x = m.get(head).rotate_angle[0];
            let f2 = sin(m.swing_progress * PI) * -(head_x - 0.7) * 0.75;
            let a = m.get(ra).rotate_angle[0];
            m.get_mut(ra).rotate_angle[0] =
                (f64::from(a) - (f64::from(f1) * 1.2 + f64::from(f2))) as f32;
            m.get_mut(ra).rotate_angle[1] += body_y * 2.0;
            m.get_mut(ra).rotate_angle[2] += sin(m.swing_progress * PI) * -0.4;
        }

        if e.sneaking {
            m.get_mut(body).rotate_angle[0] = 0.5;
            m.get_mut(ra).rotate_angle[0] += 0.4;
            m.get_mut(la).rotate_angle[0] += 0.4;
            m.get_mut(rl).rotation_point[2] = 4.0;
            m.get_mut(ll).rotation_point[2] = 4.0;
            m.get_mut(rl).rotation_point[1] = 9.0;
            m.get_mut(ll).rotation_point[1] = 9.0;
            m.get_mut(head).rotation_point[1] = 1.0;
        } else {
            m.get_mut(body).rotate_angle[0] = 0.0;
            m.get_mut(rl).rotation_point[2] = 0.1;
            m.get_mut(ll).rotation_point[2] = 0.1;
            m.get_mut(rl).rotation_point[1] = 12.0;
            m.get_mut(ll).rotation_point[1] = 12.0;
            m.get_mut(head).rotation_point[1] = 0.0;
        }

        let idle_z = cos(age * 0.09) * 0.05 + 0.05;
        let idle_x = sin(age * 0.067) * 0.05;
        m.get_mut(ra).rotate_angle[2] += idle_z;
        m.get_mut(la).rotate_angle[2] -= idle_z;
        m.get_mut(ra).rotate_angle[0] += idle_x;
        m.get_mut(la).rotate_angle[0] -= idle_x;

        if e.aimed_bow {
            let head_y = m.get(head).rotate_angle[1];
            let head_x = m.get(head).rotate_angle[0];
            m.get_mut(ra).rotate_angle[2] = 0.0;
            m.get_mut(la).rotate_angle[2] = 0.0;
            m.get_mut(ra).rotate_angle[1] = -0.1 + head_y;
            m.get_mut(la).rotate_angle[1] = 0.1 + head_y + 0.4;
            m.get_mut(ra).rotate_angle[0] = -(PI / 2.0) + head_x;
            m.get_mut(la).rotate_angle[0] = -(PI / 2.0) + head_x;
            m.get_mut(ra).rotate_angle[2] += idle_z;
            m.get_mut(la).rotate_angle[2] -= idle_z;
            m.get_mut(ra).rotate_angle[0] += idle_x;
            m.get_mut(la).rotate_angle[0] -= idle_x;
        }

        m.copy_angles(head, self.headwear);
    }
}

impl EntityModel for EndermanModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelEnderman.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        self.biped_pose(p);

        let (head, body) = (self.head, self.body);
        let (ra, la) = (self.right_arm, self.left_arm);
        let (rl, ll) = (self.right_leg, self.left_leg);
        let m = &mut self.model;

        m.get_mut(head).show_model = true;
        let f = -14.0_f32;
        m.get_mut(body).rotate_angle[0] = 0.0;
        m.get_mut(body).rotation_point[1] = f;
        m.get_mut(body).rotation_point[2] = -0.0;
        m.get_mut(rl).rotate_angle[0] -= 0.0;
        m.get_mut(ll).rotate_angle[0] -= 0.0;
        for id in [ra, la, rl, ll] {
            // java: four separate `(float)((double)x * 0.5D)` statements, in this order.
            let x = m.get(id).rotate_angle[0];
            m.get_mut(id).rotate_angle[0] = (f64::from(x) * 0.5) as f32;
        }

        // java: `float f1 = 0.4F;` then eight one-sided clamps.
        let f1 = 0.4_f32;
        for id in [ra, la, rl, ll] {
            let x = m.get(id).rotate_angle[0];
            if x > f1 {
                m.get_mut(id).rotate_angle[0] = f1;
            }
            if m.get(id).rotate_angle[0] < -f1 {
                m.get_mut(id).rotate_angle[0] = -f1;
            }
        }

        // java: `this.isCarrying` — mapped onto `EntityAnimState::carrying_block`
        // (`EntityEnderman.getHeldBlockState() != null` at the render site).
        if p.entity.carrying_block {
            m.get_mut(ra).rotate_angle[0] = -0.5;
            m.get_mut(la).rotate_angle[0] = -0.5;
            m.get_mut(ra).rotate_angle[2] = 0.05;
            m.get_mut(la).rotate_angle[2] = -0.05;
        }

        m.get_mut(ra).rotation_point[2] = 0.0;
        m.get_mut(la).rotation_point[2] = 0.0;
        m.get_mut(rl).rotation_point[2] = 0.0;
        m.get_mut(ll).rotation_point[2] = 0.0;
        m.get_mut(rl).rotation_point[1] = 9.0 + f;
        m.get_mut(ll).rotation_point[1] = 9.0 + f;
        m.get_mut(head).rotation_point[2] = -0.0;
        m.get_mut(head).rotation_point[1] = f + 1.0;
        let hp = m.get(head).rotation_point;
        let ha = m.get(head).rotate_angle;
        m.get_mut(self.headwear).rotation_point = hp;
        m.get_mut(self.headwear).rotate_angle = ha;

        // java: `this.isAttacking` — mapped onto `EntityAnimState::aggressive`
        // (`EntityEnderman.isScreaming()` at the render site). Note the headwear was pinned
        // *before* this, so only the head itself lifts.
        if p.entity.aggressive {
            let f2 = 1.0_f32;
            m.get_mut(head).rotation_point[1] -= f2 * 5.0;
        }
    }

    /// java: `ModelBiped.render`, inherited unchanged by `ModelEnderman`.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let scale = p.scale;
        if self.model.is_child {
            let head_m = base
                * Mat4::from_scale(Vec3::splat(0.75))
                * Mat4::from_translation(Vec3::new(0.0, 16.0 * scale, 0.0));
            self.model.emit_part(self.head, scale, head_m, out);
            let body_m = base
                * Mat4::from_scale(Vec3::splat(0.5))
                * Mat4::from_translation(Vec3::new(0.0, 24.0 * scale, 0.0));
            self.model.emit_some(
                &[
                    self.body,
                    self.right_arm,
                    self.left_arm,
                    self.right_leg,
                    self.left_leg,
                    self.headwear,
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
                    self.head,
                    self.body,
                    self.right_arm,
                    self.left_arm,
                    self.right_leg,
                    self.left_leg,
                    self.headwear,
                ],
                scale,
                m,
                out,
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Ghast
// ---------------------------------------------------------------------------------------------

/// Part names for [`GhastModel::tentacles`].
const GHAST_TENTACLES: [&str; 9] = [
    "tentacle0",
    "tentacle1",
    "tentacle2",
    "tentacle3",
    "tentacle4",
    "tentacle5",
    "tentacle6",
    "tentacle7",
    "tentacle8",
];

/// java: `ModelGhast`
#[derive(Clone, Debug)]
pub struct GhastModel {
    /// The part tree.
    pub model: Model,
    /// java: `body`
    pub body: PartId,
    /// java: `tentacles` — a 3x3 grid hanging off the underside.
    pub tentacles: [PartId; 9],
    /// The nine tentacle lengths drawn from `new Random(1660L)` in the constructor.
    ///
    /// This is the model's only piece of per-instance state: vanilla bakes it once so every
    /// ghast in the world has identically ragged tentacles.
    tentacle_lengths: [i32; 9],
}

impl GhastModel {
    /// java: `ModelGhast()`
    #[must_use]
    pub fn new() -> Self {
        // java: `int i = -16;`
        let i = -16.0_f32;
        let mut m = Model::new(64, 32);
        let body = m
            .part("body", 0, 0)
            .cube(-8.0, -8.0, -8.0, 16, 16, 16, 0.0)
            // java: `this.body.rotationPointY += (float)(24 + i);` from a zero default.
            .at(0.0, 24.0 + i, 0.0)
            .id();

        let mut random = JavaRandom::new(1660);
        let mut tentacle_lengths = [0_i32; 9];
        let tentacles = core::array::from_fn(|j| {
            let f = ((j % 3) as f32 - (j / 3 % 2) as f32 * 0.5 + 0.25) / 2.0 * 2.0 - 1.0;
            let f = f * 5.0;
            let f1 = ((j / 3) as f32 / 2.0 * 2.0 - 1.0) * 5.0;
            let k = random.next_i32_bound(7) + 8;
            tentacle_lengths[j] = k;
            m.part(GHAST_TENTACLES[j], 0, 0)
                .cube(-1.0, 0.0, -1.0, 2, k, 2, 0.0)
                .at(f, 31.0 + i, f1)
                .id()
        });

        Self {
            model: m,
            body,
            tentacles,
            tentacle_lengths,
        }
    }

    /// The nine baked tentacle lengths, in model units.
    #[must_use]
    pub fn tentacle_lengths(&self) -> [i32; 9] {
        self.tentacle_lengths
    }
}

impl Default for GhastModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for GhastModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelGhast.setRotationAngles` — the tentacle wobble is a pure function of
    /// `ageInTicks`, one radian of phase per tentacle, so nothing is read off the entity.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let tentacles = self.tentacles;
        for (i, &id) in tentacles.iter().enumerate() {
            self.model.get_mut(id).rotate_angle[0] =
                0.2 * sin(p.age_in_ticks * 0.3 + i as f32) + 0.4;
        }
    }

    /// java: `ModelGhast.render` — everything is drawn inside a `translate(0, 0.6, 0)`, in
    /// blocks, *not* scaled by the model scale.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let m = base * Mat4::from_translation(Vec3::new(0.0, 0.6, 0.0));
        self.model.emit_part(self.body, p.scale, m, out);
        self.model.emit_some(&self.tentacles, p.scale, m, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Silverfish
// ---------------------------------------------------------------------------------------------

/// java: `ModelSilverfish.silverfishBoxLength` — width, height, depth of each body segment.
const SILVERFISH_BOX_LENGTH: [[i32; 3]; 7] = [
    [3, 2, 2],
    [4, 3, 2],
    [6, 4, 3],
    [3, 3, 3],
    [2, 2, 3],
    [2, 1, 2],
    [1, 1, 2],
];

/// java: `ModelSilverfish.silverfishTexturePositions`
const SILVERFISH_TEXTURE_POSITIONS: [[i32; 2]; 7] =
    [[0, 0], [0, 4], [0, 9], [0, 16], [0, 22], [11, 0], [13, 4]];

/// Part names for [`SilverfishModel::body_parts`].
const SILVERFISH_PARTS: [&str; 7] = [
    "body0", "body1", "body2", "body3", "body4", "body5", "body6",
];

/// java: `ModelSilverfish`
#[derive(Clone, Debug)]
pub struct SilverfishModel {
    /// The part tree.
    pub model: Model,
    /// java: `silverfishBodyParts` — head to tail.
    pub body_parts: [PartId; 7],
    /// java: `silverfishWings` — the flat "dust" plates standing up off segments 2, 4 and 1.
    pub wings: [PartId; 3],
}

impl SilverfishModel {
    /// java: `ModelSilverfish()`
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let mut f = -3.5_f32;
        // java: `this.field_78170_c` — the Z each segment was placed at, re-read by the wings.
        let mut seg_z = [0.0_f32; 7];
        let body_parts = core::array::from_fn(|i| {
            let len = SILVERFISH_BOX_LENGTH[i];
            let tex = SILVERFISH_TEXTURE_POSITIONS[i];
            let id = m
                .part(SILVERFISH_PARTS[i], tex[0], tex[1])
                .cube(
                    len[0] as f32 * -0.5,
                    0.0,
                    len[2] as f32 * -0.5,
                    len[0],
                    len[1],
                    len[2],
                    0.0,
                )
                .at(0.0, (24 - len[1]) as f32, f)
                .id();
            seg_z[i] = f;
            if i < 6 {
                f += (len[2] + SILVERFISH_BOX_LENGTH[i + 1][2]) as f32 * 0.5;
            }
            id
        });

        let wing0 = m
            .part("wing0", 20, 0)
            .cube(
                -5.0,
                0.0,
                SILVERFISH_BOX_LENGTH[2][2] as f32 * -0.5,
                10,
                8,
                SILVERFISH_BOX_LENGTH[2][2],
                0.0,
            )
            .at(0.0, 16.0, seg_z[2])
            .id();
        let wing1 = m
            .part("wing1", 20, 11)
            .cube(
                -3.0,
                0.0,
                SILVERFISH_BOX_LENGTH[4][2] as f32 * -0.5,
                6,
                4,
                SILVERFISH_BOX_LENGTH[4][2],
                0.0,
            )
            .at(0.0, 20.0, seg_z[4])
            .id();
        // java: the third wing takes its Z *offset* from segment 4 but its depth from segment 1.
        let wing2 = m
            .part("wing2", 20, 18)
            .cube(
                -3.0,
                0.0,
                SILVERFISH_BOX_LENGTH[4][2] as f32 * -0.5,
                6,
                5,
                SILVERFISH_BOX_LENGTH[1][2],
                0.0,
            )
            .at(0.0, 19.0, seg_z[1])
            .id();

        Self {
            model: m,
            body_parts,
            wings: [wing0, wing1, wing2],
        }
    }
}

impl Default for SilverfishModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for SilverfishModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelSilverfish.setRotationAngles` — a travelling wave whose amplitude grows with
    /// the distance from segment 2.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let age = p.age_in_ticks;
        let parts = self.body_parts;
        for (i, &id) in parts.iter().enumerate() {
            let i = i as i32;
            let phase = age * 0.9 + i as f32 * 0.15 * PI;
            let s = self.model.get_mut(id);
            s.rotate_angle[1] = cos(phase) * PI * 0.05 * (1 + (i - 2).abs()) as f32;
            s.rotation_point[0] = sin(phase) * PI * 0.2 * (i - 2).abs() as f32;
        }

        let w = self.wings;
        let m = &mut self.model;
        m.get_mut(w[0]).rotate_angle[1] = m.get(parts[2]).rotate_angle[1];
        m.get_mut(w[1]).rotate_angle[1] = m.get(parts[4]).rotate_angle[1];
        m.get_mut(w[1]).rotation_point[0] = m.get(parts[4]).rotation_point[0];
        m.get_mut(w[2]).rotate_angle[1] = m.get(parts[1]).rotate_angle[1];
        m.get_mut(w[2]).rotation_point[0] = m.get(parts[1]).rotation_point[0];
    }
}

// ---------------------------------------------------------------------------------------------
// Endermite
// ---------------------------------------------------------------------------------------------

/// java: `ModelEnderMite.field_178716_a` — width, height, depth of each segment.
const ENDERMITE_BOX_LENGTH: [[i32; 3]; 4] = [[4, 3, 2], [6, 4, 5], [3, 3, 1], [1, 2, 1]];

/// java: `ModelEnderMite.field_178714_b`
const ENDERMITE_TEXTURE_POSITIONS: [[i32; 2]; 4] = [[0, 0], [0, 5], [0, 14], [0, 18]];

/// Part names for [`EndermiteModel::body_parts`].
const ENDERMITE_PARTS: [&str; 4] = ["body0", "body1", "body2", "body3"];

/// java: `ModelEnderMite`
#[derive(Clone, Debug)]
pub struct EndermiteModel {
    /// The part tree.
    pub model: Model,
    /// java: `field_178713_d` — head to tail.
    pub body_parts: [PartId; 4],
}

impl EndermiteModel {
    /// java: `ModelEnderMite()`
    #[must_use]
    pub fn new() -> Self {
        let mut m = Model::new(64, 32);
        let mut f = -3.5_f32;
        let body_parts = core::array::from_fn(|i| {
            let len = ENDERMITE_BOX_LENGTH[i];
            let tex = ENDERMITE_TEXTURE_POSITIONS[i];
            let id = m
                .part(ENDERMITE_PARTS[i], tex[0], tex[1])
                .cube(
                    len[0] as f32 * -0.5,
                    0.0,
                    len[2] as f32 * -0.5,
                    len[0],
                    len[1],
                    len[2],
                    0.0,
                )
                .at(0.0, (24 - len[1]) as f32, f)
                .id();
            if i < 3 {
                f += (len[2] + ENDERMITE_BOX_LENGTH[i + 1][2]) as f32 * 0.5;
            }
            id
        });
        Self {
            model: m,
            body_parts,
        }
    }
}

impl Default for EndermiteModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for EndermiteModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelEnderMite.setRotationAngles` — the silverfish wave with a fifth of the yaw
    /// and half the sway.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let age = p.age_in_ticks;
        let parts = self.body_parts;
        for (i, &id) in parts.iter().enumerate() {
            let i = i as i32;
            let phase = age * 0.9 + i as f32 * 0.15 * PI;
            let s = self.model.get_mut(id);
            s.rotate_angle[1] = cos(phase) * PI * 0.01 * (1 + (i - 2).abs()) as f32;
            s.rotation_point[0] = sin(phase) * PI * 0.1 * (i - 2).abs() as f32;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Guardian
// ---------------------------------------------------------------------------------------------

/// Part names for [`GuardianModel::spines`].
const GUARDIAN_SPINES: [&str; 12] = [
    "spine0", "spine1", "spine2", "spine3", "spine4", "spine5", "spine6", "spine7", "spine8",
    "spine9", "spine10", "spine11",
];

/// java: `afloat` — each spine's X rotation, in multiples of PI.
const SPINE_ROT_X: [f32; 12] = [
    1.75, 0.25, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 1.25, 0.75, 0.0, 0.0,
];
/// java: `afloat1` — each spine's Y rotation, in multiples of PI.
const SPINE_ROT_Y: [f32; 12] = [
    0.0, 0.0, 0.0, 0.0, 0.25, 1.75, 1.25, 0.75, 0.0, 0.0, 0.0, 0.0,
];
/// java: `afloat2` — each spine's Z rotation, in multiples of PI.
const SPINE_ROT_Z: [f32; 12] = [
    0.0, 0.0, 0.25, 1.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.75, 1.25,
];
/// java: `afloat3` — each spine's fully-extended X position.
const SPINE_POS_X: [f32; 12] = [
    0.0, 0.0, 8.0, -8.0, -8.0, 8.0, 8.0, -8.0, 0.0, 0.0, 8.0, -8.0,
];
/// java: `afloat4` — each spine's fully-extended Y position, relative to 16.
const SPINE_POS_Y: [f32; 12] = [
    -8.0, -8.0, -8.0, -8.0, 0.0, 0.0, 0.0, 0.0, 8.0, 8.0, 8.0, 8.0,
];
/// java: `afloat5` — each spine's fully-extended Z position.
const SPINE_POS_Z: [f32; 12] = [
    8.0, -8.0, 0.0, 0.0, -8.0, -8.0, 8.0, 8.0, 8.0, -8.0, 0.0, 0.0,
];

/// `Math.signum` for `f64`: zero and NaN come back unchanged, unlike [`f64::signum`], which
/// reports `1.0` for `+0.0`.
fn java_signum(d: f64) -> f64 {
    if d > 0.0 {
        1.0
    } else if d < 0.0 {
        -1.0
    } else {
        d
    }
}

/// java: `ModelGuardian`
#[derive(Clone, Debug)]
pub struct GuardianModel {
    /// The part tree. Only [`GuardianModel::body`] is a root; everything else hangs off it.
    pub model: Model,
    /// java: `guardianBody` — five boxes, the only part `ModelGuardian.render` draws directly.
    pub body: PartId,
    /// java: `guardianEye`
    pub eye: PartId,
    /// java: `guardianSpines`
    pub spines: [PartId; 12],
    /// java: `guardianTail` — a three-link chain, each link a child of the previous.
    pub tail: [PartId; 3],
}

impl GuardianModel {
    /// java: `ModelGuardian()`
    #[must_use]
    pub fn new() -> Self {
        // java: `this.textureWidth = 64; this.textureHeight = 64;`
        let mut m = Model::new(64, 64);
        // java: `new ModelRenderer(this)` — no constructor offset, one `setTextureOffset` per box.
        let body = m
            .part("guardian_body", 0, 0)
            .tex(0, 0)
            .cube(-6.0, 10.0, -8.0, 12, 12, 16, 0.0)
            .tex(0, 28)
            .cube(-8.0, 10.0, -6.0, 2, 12, 12, 0.0)
            .tex(0, 28)
            // java: `addBox(..., true)` — the per-box mirror overload, which does not latch.
            .mirrored()
            .cube(6.0, 10.0, -6.0, 2, 12, 12, 0.0)
            .unmirrored()
            .tex(16, 40)
            .cube(-6.0, 8.0, -6.0, 12, 2, 12, 0.0)
            .tex(16, 40)
            .cube(-6.0, 22.0, -6.0, 12, 2, 12, 0.0)
            .id();

        let spines = core::array::from_fn(|i| {
            m.child(body, GUARDIAN_SPINES[i], 0, 0)
                .cube(-1.0, -4.5, -1.0, 2, 9, 2, 0.0)
                .id()
        });

        let eye = m
            .child(body, "guardian_eye", 8, 0)
            .cube(-1.0, 15.0, 0.0, 2, 2, 1, 0.0)
            .id();

        let tail0 = m
            .child(body, "guardian_tail0", 40, 0)
            .cube(-2.0, 14.0, 7.0, 4, 4, 8, 0.0)
            .id();
        let tail1 = m
            .child(tail0, "guardian_tail1", 0, 54)
            .cube(0.0, 14.0, 0.0, 3, 3, 7, 0.0)
            .id();
        let tail2 = m
            .child(tail1, "guardian_tail2", 0, 0)
            .tex(41, 32)
            .cube(0.0, 14.0, 0.0, 2, 2, 6, 0.0)
            .tex(25, 19)
            .cube(1.0, 10.5, 3.0, 1, 9, 9, 0.0)
            .id();

        Self {
            model: m,
            body,
            eye,
            spines,
            tail: [tail0, tail1, tail2],
        }
    }

    /// Feeds this frame's spike extension.
    ///
    /// Aims the eye at a point, exactly as the `if (entity != null)` block of
    /// `ModelGuardian.setRotationAngles` does.
    ///
    /// A model cannot reach `Minecraft.getMinecraft().getRenderViewEntity()`, so the renderer
    /// supplies the three vectors instead: the guardian's own eye position, the watched entity's
    /// eye position (the beam target if there is one, otherwise the camera), and the guardian's
    /// look vector — all `getPositionEyes(0.0F)`/`getLook(0.0F)`, i.e. **not** interpolated.
    ///
    /// Call it after [`EntityModel::set_rotation_angles`], which resets the eye's Z pivot.
    pub fn look_at(&mut self, self_eyes: [f64; 3], target_eyes: [f64; 3], self_look: [f64; 3]) {
        let d0 = target_eyes[1] - self_eyes[1];
        let eye = self.eye;
        self.model.get_mut(eye).rotation_point[1] = if d0 > 0.0 { 0.0 } else { 1.0 };

        // java: `vec32 = entityIn.getLook(0.0F)` flattened to the XZ plane.
        let look = [self_look[0], 0.0, self_look[2]];
        // java: `vec33 = new Vec3(self - target).normalize().rotateYaw(PI / 2)`
        let v = [
            self_eyes[0] - target_eyes[0],
            0.0,
            self_eyes[2] - target_eyes[2],
        ];
        let len = f64::from(mc_math::sqrt_d(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]));
        let n = if len < 1.0e-4 {
            [0.0, 0.0, 0.0]
        } else {
            [v[0] / len, v[1] / len, v[2] / len]
        };
        let (c, s) = (f64::from(cos(PI / 2.0)), f64::from(sin(PI / 2.0)));
        let rot = [n[0] * c + n[2] * s, n[1], n[2] * c - n[0] * s];

        let d1 = look[0] * rot[0] + look[1] * rot[1] + look[2] * rot[2];
        self.model.get_mut(eye).rotation_point[0] =
            sqrt_f(d1.abs() as f32) * 2.0 * java_signum(d1) as f32;
    }
}

impl Default for GuardianModel {
    fn default() -> Self {
        Self::new()
    }
}

impl EntityModel for GuardianModel {
    fn model(&self) -> &Model {
        &self.model
    }
    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelGuardian.setRotationAngles`
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let age = p.age_in_ticks;

        self.model.get_mut(self.body).rotate_angle[1] = model_radians(p.net_head_yaw);
        self.model.get_mut(self.body).rotate_angle[0] = model_radians(p.head_pitch);

        // java: `float f1 = (1.0F - entityguardian.func_175469_o(f)) * 0.55F;`
        //
        // `f` is the frame's partial ticks, and `func_175469_o(f)` interpolates the two tick
        // samples the model cannot see, so the caller does it: `EntityAnimState::spikes_animation`
        // is already `func_175469_o(partialTicks)`. Holding the samples on the *model* the way
        // this used to would also be wrong for a second reason — one `GuardianModel` is shared
        // by every guardian on screen, so the last one posed would overwrite the others.
        let f1 = (1.0 - p.entity.spikes_animation) * 0.55;

        let spines = self.spines;
        for (i, &id) in spines.iter().enumerate() {
            let wobble = 1.0 + cos(age * 1.5 + i as f32) * 0.01 - f1;
            let s = self.model.get_mut(id);
            s.rotate_angle[0] = PI * SPINE_ROT_X[i];
            s.rotate_angle[1] = PI * SPINE_ROT_Y[i];
            s.rotate_angle[2] = PI * SPINE_ROT_Z[i];
            s.rotation_point[0] = SPINE_POS_X[i] * wobble;
            s.rotation_point[1] = 16.0 + SPINE_POS_Y[i] * wobble;
            s.rotation_point[2] = SPINE_POS_Z[i] * wobble;
        }

        self.model.get_mut(self.eye).rotation_point[2] = -8.25;
        if let Some(look) = p.entity.guardian_look_at {
            self.look_at(look.self_eyes, look.target_eyes, look.self_look);
        }
        // java: the `getRenderViewEntity()` look-at block goes here — see
        // [`GuardianModel::look_at`], which the renderer calls with the vectors we cannot see.
        self.model.get_mut(self.eye).show_model = true;

        // java: `float f2 = entityguardian.func_175471_a(f);` — the interpolated tail phase,
        // mapped onto `EntityAnimState::tail_rotation`, which the caller interpolates.
        let f2 = p.entity.tail_rotation;
        let t = self.tail;
        let m = &mut self.model;
        m.get_mut(t[0]).rotate_angle[1] = sin(f2) * PI * 0.05;
        m.get_mut(t[1]).rotate_angle[1] = sin(f2) * PI * 0.1;
        m.get_mut(t[1]).rotation_point[0] = -1.5;
        m.get_mut(t[1]).rotation_point[1] = 0.5;
        m.get_mut(t[1]).rotation_point[2] = 14.0;
        m.get_mut(t[2]).rotate_angle[1] = sin(f2) * PI * 0.15;
        m.get_mut(t[2]).rotation_point[0] = 0.5;
        m.get_mut(t[2]).rotation_point[1] = 0.5;
        m.get_mut(t[2]).rotation_point[2] = 6.0;
    }

    /// java: `ModelGuardian.render` — one call, the body, with everything else as its children.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_part(self.body, p.scale, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Dragon
// ---------------------------------------------------------------------------------------------

/// The twelve named parts in Java's `ModelDragon`.
///
/// `spine` is intentionally one part.  Vanilla renders that same `ModelRenderer` five times
/// for the neck and twelve times for the tail after changing its pivot and angles for each
/// segment; making seventeen copies here would lose that construction fact and make it too easy
/// for the two chains to drift apart.
#[derive(Clone, Copy, Debug)]
pub struct DragonParts {
    /// java: `head`.
    pub head: PartId,
    /// java: `jaw`, a child of `head`.
    pub jaw: PartId,
    /// java: `spine`, reused by the neck and tail loops.
    pub spine: PartId,
    /// java: `body`.
    pub body: PartId,
    /// java: `wing`, parent of `wing_tip`.
    pub wing: PartId,
    /// java: `wingTip`.
    pub wing_tip: PartId,
    /// java: `frontLeg`, parent of `front_leg_tip`.
    pub front_leg: PartId,
    /// java: `frontLegTip`, parent of `front_foot`.
    pub front_leg_tip: PartId,
    /// java: `frontFoot`.
    pub front_foot: PartId,
    /// java: `rearLeg`, parent of `rear_leg_tip`.
    pub rear_leg: PartId,
    /// java: `rearLegTip`, parent of `rear_foot`.
    pub rear_leg_tip: PartId,
    /// java: `rearFoot`.
    pub rear_foot: PartId,
}

/// The ender dragon model.
///
/// java: `ModelDragon`.  The constructor is complete, including the two zero-height wing
/// membranes and the negative U texture offsets used by vanilla's 256x256 sheet.  The
/// entity-side ring/animation values arrive through [`DragonRenderState`] in
/// [`crate::entity::anim::EntityAnimState::dragon`]; a missing state still renders a complete
/// neutral body, which keeps the model safe for generic render-manager callers.
#[derive(Clone, Debug)]
pub struct DragonModel {
    /// The 256x256 dragon part forest.
    pub model: Model,
    /// Handles for the constructor's twelve parts.
    pub parts: DragonParts,
}

/// Java `ModelDragon` uses `Math.sin`/`Math.cos` (double) in its render method rather than
/// `MathHelper`'s float lookup table.  Preserve the float expression's narrowing point before
/// calling the host double operation.
#[inline]
fn dragon_sin(value: f32) -> f32 {
    f64::from(value).sin() as f32
}

#[inline]
fn dragon_cos(value: f32) -> f32 {
    f64::from(value).cos() as f32
}

/// Java's `float degrees * (float)Math.PI / 180.0F` spelling in `ModelDragon`.
#[inline]
fn dragon_radians(degrees: f32) -> f32 {
    degrees * PI / 180.0
}

/// Java's private `ModelDragon.updateRotations(double)` helper.
#[inline]
fn dragon_update_rotations(mut degrees: f64) -> f32 {
    while degrees >= 180.0 {
        degrees -= 360.0;
    }
    while degrees < -180.0 {
        degrees += 360.0;
    }
    degrees as f32
}

/// Emits a part with a temporary pivot/angle without mutating the shared model.
///
/// The dragon's one `spine` node is rendered seventeen times from an immutable trait `render`
/// callback.  Feeding the desired local transform through the inverse of its stored local
/// transform lets the ordinary model walker retain its exact box/child handling while changing
/// only that invocation's pivot and angles.
fn emit_part_with_pose(
    model: &Model,
    id: PartId,
    scale: f32,
    base: Mat4,
    rotation_point: [f32; 3],
    rotate_angle: [f32; 3],
    out: &mut Vec<EntityVertex>,
) {
    let part = model.get(id);
    let desired = Mat4::from_translation(Vec3::new(
        rotation_point[0] * scale,
        rotation_point[1] * scale,
        rotation_point[2] * scale,
    )) * Mat4::from_rotation_z(rotate_angle[2])
        * Mat4::from_rotation_y(rotate_angle[1])
        * Mat4::from_rotation_x(rotate_angle[0]);
    let adjusted_base = base * desired * part.local_matrix(scale).inverse();
    model.emit_part(id, scale, adjusted_base, out);
}

impl DragonModel {
    /// java: `ModelDragon(float)` — the constructor argument is ignored.
    #[must_use]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        let mut m = Model::new(256, 256);
        let f = -16.0;

        let head = m
            .part("head", 0, 0)
            .tex(176, 44)
            .cube(-6.0, -1.0, -8.0 + f, 12, 5, 16, 0.0)
            .tex(112, 30)
            .cube(-8.0, -8.0, 6.0 + f, 16, 16, 16, 0.0)
            .mirrored()
            .tex(0, 0)
            .cube(-5.0, -12.0, 12.0 + f, 2, 4, 6, 0.0)
            .tex(112, 0)
            .cube(-5.0, -3.0, -6.0 + f, 2, 2, 4, 0.0)
            .unmirrored()
            .tex(0, 0)
            .cube(3.0, -12.0, 12.0 + f, 2, 4, 6, 0.0)
            .tex(112, 0)
            .cube(3.0, -3.0, -6.0 + f, 2, 2, 4, 0.0)
            .id();
        let jaw = m
            .child(head, "jaw", 176, 65)
            .at(0.0, 4.0, 8.0 + f)
            .tex(176, 65)
            .cube(-6.0, 0.0, -16.0, 12, 4, 16, 0.0)
            .id();

        let spine = m
            .part("neck", 192, 104)
            .tex(192, 104)
            .cube(-5.0, -5.0, -5.0, 10, 10, 10, 0.0)
            .tex(48, 0)
            .cube(-1.0, -9.0, -3.0, 2, 4, 6, 0.0)
            .id();
        let body = m
            .part("body", 0, 0)
            .tex(0, 0)
            .at(0.0, 4.0, 8.0)
            .cube(-12.0, 0.0, -16.0, 24, 24, 64, 0.0)
            .tex(220, 53)
            .cube(-1.0, -6.0, -10.0, 2, 6, 12, 0.0)
            .tex(220, 53)
            .cube(-1.0, -6.0, 10.0, 2, 6, 12, 0.0)
            .tex(220, 53)
            .cube(-1.0, -6.0, 30.0, 2, 6, 12, 0.0)
            .id();

        let wing = m
            .part("wing", 112, 88)
            .at(-12.0, 5.0, 2.0)
            .tex(112, 88)
            .cube(-56.0, -4.0, -4.0, 56, 8, 8, 0.0)
            .tex(-56, 88)
            .cube(-56.0, 0.0, 2.0, 56, 0, 56, 0.0)
            .id();
        let wing_tip = m
            .child(wing, "wingtip", 112, 136)
            .at(-56.0, 0.0, 0.0)
            .tex(112, 136)
            .cube(-56.0, -2.0, -2.0, 56, 4, 4, 0.0)
            .tex(-56, 144)
            .cube(-56.0, 0.0, 2.0, 56, 0, 56, 0.0)
            .id();

        let front_leg = m
            .part("frontleg", 112, 104)
            .at(-12.0, 20.0, 2.0)
            .tex(112, 104)
            .cube(-4.0, -4.0, -4.0, 8, 24, 8, 0.0)
            .id();
        let front_leg_tip = m
            .child(front_leg, "frontlegtip", 226, 138)
            .at(0.0, 20.0, -1.0)
            .tex(226, 138)
            .cube(-3.0, -1.0, -3.0, 6, 24, 6, 0.0)
            .id();
        let front_foot = m
            .child(front_leg_tip, "frontfoot", 144, 104)
            .at(0.0, 23.0, 0.0)
            .tex(144, 104)
            .cube(-4.0, 0.0, -12.0, 8, 4, 16, 0.0)
            .id();

        let rear_leg = m
            .part("rearleg", 0, 0)
            .at(-16.0, 16.0, 42.0)
            .tex(0, 0)
            .cube(-8.0, -4.0, -8.0, 16, 32, 16, 0.0)
            .id();
        let rear_leg_tip = m
            .child(rear_leg, "rearlegtip", 196, 0)
            .at(0.0, 32.0, -4.0)
            .tex(196, 0)
            .cube(-6.0, -2.0, 0.0, 12, 32, 12, 0.0)
            .id();
        let rear_foot = m
            .child(rear_leg_tip, "rearfoot", 112, 0)
            .at(0.0, 31.0, 4.0)
            .tex(112, 0)
            .cube(-9.0, 0.0, -20.0, 18, 6, 24, 0.0)
            .id();

        Self {
            model: m,
            parts: DragonParts {
                head,
                jaw,
                spine,
                body,
                wing,
                wing_tip,
                front_leg,
                front_leg_tip,
                front_foot,
                rear_leg,
                rear_leg_tip,
                rear_foot,
            },
        }
    }

    fn state_for(p: &AnimParams) -> DragonRenderState {
        p.entity
            .dragon
            .unwrap_or_else(|| DragonRenderState::neutral(p.age_in_ticks))
    }

    /// Applies the one-frame fields that ordinary `ModelRenderer.render` calls consume.
    fn pose_frame(&mut self, p: &AnimParams, state: &DragonRenderState) {
        let q = self.parts;
        let f = state.prev_anim_time + (state.anim_time - state.prev_anim_time) * p.partial_ticks;
        let phase = f * PI * 2.0;
        let jaw_open = (dragon_sin(phase) + 1.0) * 0.2;
        self.model.get_mut(q.jaw).rotate_angle[0] = jaw_open;

        let mut f1 = dragon_sin(phase - 1.0) + 1.0;
        f1 = (f1 * f1 * 1.0 + f1 * 2.0) * 0.05;

        let f6 = dragon_update_rotations(
            state.movement_offset(5, p.partial_ticks)[0]
                - state.movement_offset(10, p.partial_ticks)[0],
        );
        let f7 = dragon_update_rotations(
            state.movement_offset(5, p.partial_ticks)[0] + f64::from(f6 / 2.0),
        );
        let mut point_y = 20.0_f32;
        let mut point_z = -12.0_f32;
        let mut point_x = 0.0_f32;
        for i in 0..5 {
            let sample = state.movement_offset(5 - i, p.partial_ticks);
            let wave = dragon_cos(i as f32 * 0.45 + phase) * 0.15;
            let angle_y =
                dragon_update_rotations(sample[0] - state.movement_offset(6, p.partial_ticks)[0])
                    * PI
                    / 180.0
                    * 1.5;
            let angle_x = wave
                + (sample[1] - state.movement_offset(6, p.partial_ticks)[1]) as f32 * PI / 180.0
                    * 1.5
                    * 5.0;
            point_y += dragon_sin(angle_x) * 10.0;
            point_z -= dragon_cos(angle_y) * dragon_cos(angle_x) * 10.0;
            point_x -= dragon_sin(angle_y) * dragon_cos(angle_x) * 10.0;
        }

        let head_sample = state.movement_offset(0, p.partial_ticks);
        let anchor = state.movement_offset(6, p.partial_ticks);
        self.model.get_mut(q.head).rotation_point = [point_x, point_y, point_z];
        self.model.get_mut(q.head).rotate_angle[0] = 0.0;
        self.model.get_mut(q.head).rotate_angle[1] =
            dragon_update_rotations(head_sample[0] - anchor[0]) * PI / 180.0;
        self.model.get_mut(q.head).rotate_angle[2] =
            -dragon_update_rotations(head_sample[0] - f64::from(f7)) * PI / 180.0;

        self.model.get_mut(q.body).rotate_angle = [0.0; 3];
        let wing_x = 0.125 - dragon_cos(phase) * 0.2;
        let wing_y = 0.25;
        let wing_z = (dragon_sin(phase) + 0.125) * 0.8;
        self.model.get_mut(q.wing).rotate_angle = [wing_x, wing_y, wing_z];
        self.model.get_mut(q.wing_tip).rotate_angle[2] = -(dragon_sin(phase + 2.0) + 0.5) * 0.75;
        self.model.get_mut(q.rear_leg).rotate_angle[0] = 1.0 + f1 * 0.1;
        self.model.get_mut(q.rear_leg_tip).rotate_angle[0] = 0.5 + f1 * 0.1;
        self.model.get_mut(q.rear_foot).rotate_angle[0] = 0.75 + f1 * 0.1;
        self.model.get_mut(q.front_leg).rotate_angle[0] = 1.3 + f1 * 0.1;
        self.model.get_mut(q.front_leg_tip).rotate_angle[0] = -0.5 - f1 * 0.1;
        self.model.get_mut(q.front_foot).rotate_angle[0] = 0.75 + f1 * 0.1;
    }

    /// Renders the complete model using one supplied client-local state.
    ///
    /// This is the model-side half of Java's `ModelDragon.render`.  `pose_and_render` reaches it
    /// through the optional `EntityAnimState::dragon` payload; callers with a dedicated tracker
    /// state should put that state in the frame payload before invoking the ordinary manager API.
    fn render_frame(
        &self,
        p: &AnimParams,
        state: &DragonRenderState,
        base: Mat4,
        out: &mut Vec<EntityVertex>,
    ) {
        let q = self.parts;
        let f = state.prev_anim_time + (state.anim_time - state.prev_anim_time) * p.partial_ticks;
        let phase = f * PI * 2.0;
        let mut f1 = dragon_sin(phase - 1.0) + 1.0;
        f1 = (f1 * f1 * 1.0 + f1 * 2.0) * 0.05;
        let initial = base
            * Mat4::from_translation(Vec3::new(0.0, f1 - 2.0, -3.0))
            * Mat4::from_rotation_x(dragon_radians(f1 * 2.0));

        let anchor = state.movement_offset(6, p.partial_ticks);
        let f6 = dragon_update_rotations(
            state.movement_offset(5, p.partial_ticks)[0]
                - state.movement_offset(10, p.partial_ticks)[0],
        );
        let f7 = dragon_update_rotations(
            state.movement_offset(5, p.partial_ticks)[0] + f64::from(f6 / 2.0),
        );

        let mut point_y = 20.0_f32;
        let mut point_z = -12.0_f32;
        let mut point_x = 0.0_f32;
        for i in 0..5 {
            let sample = state.movement_offset(5 - i, p.partial_ticks);
            let wave = dragon_cos(i as f32 * 0.45 + phase) * 0.15;
            let angle_y = dragon_update_rotations(sample[0] - anchor[0]) * PI / 180.0 * 1.5;
            let angle_x = wave + (sample[1] - anchor[1]) as f32 * PI / 180.0 * 1.5 * 5.0;
            let angle_z = -dragon_update_rotations(sample[0] - f64::from(f7)) * PI / 180.0 * 1.5;
            emit_part_with_pose(
                &self.model,
                q.spine,
                p.scale,
                initial,
                [point_x, point_y, point_z],
                [angle_x, angle_y, angle_z],
                out,
            );
            point_y += dragon_sin(angle_x) * 10.0;
            point_z -= dragon_cos(angle_y) * dragon_cos(angle_x) * 10.0;
            point_x -= dragon_sin(angle_y) * dragon_cos(angle_x) * 10.0;
        }

        self.model.emit_part(q.head, p.scale, initial, out);

        let body_base = initial
            * Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0))
            * Mat4::from_rotation_z(dragon_radians(-f6 * 1.5))
            * Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0));
        self.model.emit_part(q.body, p.scale, body_base, out);

        for side in 0..2 {
            let side_base = if side == 0 {
                body_base
            } else {
                body_base * Mat4::from_scale(Vec3::new(-1.0, 1.0, 1.0))
            };
            self.model.emit_part(q.wing, p.scale, side_base, out);
            self.model.emit_part(q.front_leg, p.scale, side_base, out);
            self.model.emit_part(q.rear_leg, p.scale, side_base, out);
        }

        let mut tail_wave = 0.0_f32;
        let tail_anchor = state.movement_offset(11, p.partial_ticks);
        let mut tail_y = 10.0_f32;
        let mut tail_z = 60.0_f32;
        let mut tail_x = 0.0_f32;
        for k in 0..12 {
            let sample = state.movement_offset(12 + k, p.partial_ticks);
            tail_wave = (f64::from(tail_wave)
                + f64::from(dragon_sin(k as f32 * 0.45 + phase)) * 0.05000000074505806)
                as f32;
            let angle_y =
                (dragon_update_rotations(sample[0] - tail_anchor[0]) * 1.5 + 180.0) * PI / 180.0;
            let angle_x = tail_wave + (sample[1] - tail_anchor[1]) as f32 * PI / 180.0 * 1.5 * 5.0;
            let angle_z = dragon_update_rotations(sample[0] - f64::from(f7)) * PI / 180.0 * 1.5;
            emit_part_with_pose(
                &self.model,
                q.spine,
                p.scale,
                initial,
                [tail_x, tail_y, tail_z],
                [angle_x, angle_y, angle_z],
                out,
            );
            tail_y += dragon_sin(angle_x) * 10.0;
            tail_z -= dragon_cos(angle_y) * dragon_cos(angle_x) * 10.0;
            tail_x -= dragon_sin(angle_y) * dragon_cos(angle_x) * 10.0;
        }
    }
}

impl EntityModel for DragonModel {
    fn model(&self) -> &Model {
        &self.model
    }

    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let state = Self::state_for(p);
        self.pose_frame(p, &state);
    }

    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        let state = Self::state_for(p);
        self.render_frame(p, &state, base, out);
    }
}

// ---------------------------------------------------------------------------------------------
// Wither
// ---------------------------------------------------------------------------------------------

/// java: `ModelWither`.
///
/// The central skull follows the ordinary living head look. The two side skulls instead take
/// their world-derived, **body-relative** look angles from
/// [`EntityAnimState::wither_side_head_yaw`] and [`EntityAnimState::wither_side_head_pitch`],
/// as `EntityWither` does after resolving watched target ids 18 and 19.
#[derive(Clone, Debug)]
pub struct WitherModel {
    /// The part tree, on the Java model's 64 x 64 texture sheet.
    pub model: Model,
    /// java: `field_82905_a` — the three body/neck pieces.
    pub body: [PartId; 3],
    /// java: `field_82904_b` — centre, left, and right skulls.
    pub heads: [PartId; 3],
}

impl WitherModel {
    /// java: `ModelWither(float modelSize)`.
    #[must_use]
    pub fn new(size: f32) -> Self {
        let mut m = Model::new(64, 64);
        let body0 = m
            .part("body0", 0, 16)
            .cube(-10.0, 3.9, -0.5, 20, 3, 3, size)
            .id();
        // Java constructs this part with no initial UV, then changes it before each cube.
        let body1 = m
            .part("body1", 0, 0)
            .tex_size(64, 64)
            .at(-2.0, 6.9, -0.5)
            .tex(0, 22)
            .cube(0.0, 0.0, 0.0, 3, 10, 3, size)
            .tex(24, 22)
            .cube(-4.0, 1.5, 0.5, 11, 2, 2, size)
            .tex(24, 22)
            .cube(-4.0, 4.0, 0.5, 11, 2, 2, size)
            .tex(24, 22)
            .cube(-4.0, 6.5, 0.5, 11, 2, 2, size)
            .id();
        let body2 = m
            .part("body2", 12, 22)
            .cube(0.0, 0.0, 0.0, 3, 6, 3, size)
            .id();

        let head0 = m
            .part("head0", 0, 0)
            .cube(-4.0, -4.0, -4.0, 8, 8, 8, size)
            .id();
        let head1 = m
            .part("head1", 32, 0)
            .cube(-4.0, -4.0, -4.0, 6, 6, 6, size)
            .at(-8.0, 4.0, 0.0)
            .id();
        let head2 = m
            .part("head2", 32, 0)
            .cube(-4.0, -4.0, -4.0, 6, 6, 6, size)
            .at(10.0, 4.0, 0.0)
            .id();
        Self {
            model: m,
            body: [body0, body1, body2],
            heads: [head0, head1, head2],
        }
    }
}

impl EntityModel for WitherModel {
    fn model(&self) -> &Model {
        &self.model
    }

    fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// java: `ModelWither.setLivingAnimations`.
    fn set_living_animations(&mut self, p: &AnimParams) {
        for side in 0..2 {
            let head = self.heads[side + 1];
            self.model.get_mut(head).rotate_angle[1] =
                model_radians(p.entity.wither_side_head_yaw[side]);
            self.model.get_mut(head).rotate_angle[0] =
                model_radians(p.entity.wither_side_head_pitch[side]);
        }
    }

    /// java: `ModelWither.setRotationAngles`.
    fn set_rotation_angles(&mut self, p: &AnimParams) {
        let f = cos(p.age_in_ticks * 0.1);
        let body1_angle_x = (0.065 + 0.05 * f) * PI;
        let body2_angle_x = (0.265 + 0.1 * f) * PI;
        self.model.get_mut(self.body[1]).rotate_angle[0] = body1_angle_x;
        let body2 = self.model.get_mut(self.body[2]);
        body2.rotation_point = [
            -2.0,
            6.9 + cos(body1_angle_x) * 10.0,
            -0.5 + sin(body1_angle_x) * 10.0,
        ];
        body2.rotate_angle[0] = body2_angle_x;

        let centre = self.heads[0];
        self.model.get_mut(centre).rotate_angle[1] = model_radians(p.net_head_yaw);
        self.model.get_mut(centre).rotate_angle[0] = model_radians(p.head_pitch);
    }

    /// java: `ModelWither.render` — skulls first, followed by the three body parts.
    fn render(&self, p: &AnimParams, base: Mat4, out: &mut Vec<EntityVertex>) {
        self.model.emit_some(&self.heads, p.scale, base, out);
        self.model.emit_some(&self.body, p.scale, base, out);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::anim::{DragonRenderState, EntityAnimState};

    // ---- creeper -----------------------------------------------------------------------------

    #[test]
    fn creeper_geometry_matches_the_java_constructor() {
        let c = CreeperModel::new(0.0);
        let m = &c.model;
        // head: addBox(-4, -8, -4, 8, 8, 8), setRotationPoint(0, 6, 0)
        assert_eq!(m.get(c.head).cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(m.get(c.head).cubes[0].pos2, [4.0, 0.0, 4.0]);
        assert_eq!(m.get(c.head).rotation_point, [0.0, 6.0, 0.0]);
        // body: addBox(-4, 0, -2, 8, 12, 4) at (0, 6, 0), texture (16, 16)
        assert_eq!(m.get(c.body).cubes[0].pos2, [4.0, 12.0, 2.0]);
        assert_eq!(m.get(c.body).texture_offset, [16, 16]);
        // legs: 4x6x4 at (+-2, 18, +-4); 1 and 2 are the back pair (z = +4).
        assert_eq!(m.get(c.legs[0]).cubes[0].pos2, [2.0, 6.0, 2.0]);
        assert_eq!(m.get(c.legs[0]).rotation_point, [-2.0, 18.0, 4.0]);
        assert_eq!(m.get(c.legs[1]).rotation_point, [2.0, 18.0, 4.0]);
        assert_eq!(m.get(c.legs[2]).rotation_point, [-2.0, 18.0, -4.0]);
        assert_eq!(m.get(c.legs[3]).rotation_point, [2.0, 18.0, -4.0]);
        // the charge overlay is the head inflated by half a unit on every side
        assert_eq!(
            m.get(c.creeper_armor).cubes[0].quads[4].verts[1].pos,
            [-4.5, -8.5, -4.5]
        );
    }

    /// java: `leg1.rotateAngleX = cos(limbSwing * 0.6662F) * 1.4F * limbSwingAmount`, with legs
    /// 2 and 3 a half-period out of phase — the diagonal gait.
    #[test]
    fn creeper_legs_walk_diagonally() {
        let mut c = CreeperModel::new(0.0);
        let mut p = AnimParams::still();
        p.limb_swing = 7.5;
        p.limb_swing_amount = 0.8;
        c.set_rotation_angles(&p);

        let front = cos(7.5 * 0.6662) * 1.4 * 0.8;
        let back = cos(7.5 * 0.6662 + PI) * 1.4 * 0.8;
        assert_eq!(c.model.get(c.legs[0]).rotate_angle[0], front);
        assert_eq!(c.model.get(c.legs[1]).rotate_angle[0], back);
        assert_eq!(c.model.get(c.legs[2]).rotate_angle[0], back);
        assert_eq!(c.model.get(c.legs[3]).rotate_angle[0], front);
        // Diagonal pairs are (1, 4) and (2, 3), and the two groups oppose each other.
        assert!(front * back < 0.0, "{front} vs {back}");
    }

    /// java: `ModelCreeper.render` never draws `creeperArmor`.
    #[test]
    fn creeper_render_skips_the_charge_overlay() {
        let c = CreeperModel::new(0.0);
        let mut out = Vec::new();
        c.render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
        // 6 drawn parts x 1 box x 6 faces x 6 verts.
        assert_eq!(out.len(), 6 * 6 * 6);
        assert_eq!(
            c.model.parts().len(),
            7,
            "the armour part still exists, it is just unused"
        );
    }

    // ---- spider ------------------------------------------------------------------------------

    #[test]
    fn spider_geometry_matches_the_java_constructor() {
        let s = SpiderModel::new();
        let m = &s.model;
        assert_eq!(m.get(s.head).texture_offset, [32, 4]);
        assert_eq!(m.get(s.head).cubes[0].pos1, [-4.0, -4.0, -8.0]);
        assert_eq!(m.get(s.head).rotation_point, [0.0, 15.0, -3.0]);
        assert_eq!(m.get(s.neck).cubes[0].pos2, [3.0, 3.0, 3.0]);
        assert_eq!(m.get(s.body).cubes[0].pos1, [-5.0, -4.0, -6.0]);
        assert_eq!(m.get(s.body).rotation_point, [0.0, 15.0, 9.0]);
        // Odd Java legs point left (box runs to -15), even ones right.
        assert_eq!(m.get(s.legs[0]).cubes[0].pos1, [-15.0, -1.0, -1.0]);
        assert_eq!(m.get(s.legs[0]).cubes[0].pos2, [1.0, 1.0, 1.0]);
        assert_eq!(m.get(s.legs[1]).cubes[0].pos1, [-1.0, -1.0, -1.0]);
        assert_eq!(m.get(s.legs[1]).cubes[0].pos2, [15.0, 1.0, 1.0]);
        // The four Z rows: 2, 1, 0, -1.
        let zs: Vec<f32> = s.legs.iter().map(|&l| m.get(l).rotation_point[2]).collect();
        assert_eq!(zs, vec![2.0, 2.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0]);
    }

    /// Every one of the eight legs, against the literal Java formula.
    #[test]
    #[allow(clippy::approx_constant)] // vanilla's `0.3926991F`, not `FRAC_PI_8`
    fn spider_eight_legs_match_the_java_formula() {
        let mut s = SpiderModel::new();
        let mut p = AnimParams::still();
        p.limb_swing = 4.25;
        p.limb_swing_amount = 0.9;
        s.set_rotation_angles(&p);

        let (ls, la) = (4.25_f32, 0.9_f32);
        let f = PI / 4.0;
        let f1 = -0.0_f32;
        let f2 = 0.3926991_f32;
        let f3 = -(cos(ls * 0.6662 * 2.0) * 0.4) * la;
        let f4 = -(cos(ls * 0.6662 * 2.0 + PI) * 0.4) * la;
        let f5 = -(cos(ls * 0.6662 * 2.0 + PI / 2.0) * 0.4) * la;
        let f6 = -(cos(ls * 0.6662 * 2.0 + PI * 3.0 / 2.0) * 0.4) * la;
        let f7 = (sin(ls * 0.6662) * 0.4).abs() * la;
        let f8 = (sin(ls * 0.6662 + PI) * 0.4).abs() * la;
        let f9 = (sin(ls * 0.6662 + PI / 2.0) * 0.4).abs() * la;
        let f10 = (sin(ls * 0.6662 + PI * 3.0 / 2.0) * 0.4).abs() * la;

        let want_y = [
            f2 * 2.0 + f1 + f3,
            -f2 * 2.0 - f1 - f3,
            f2 + f1 + f4,
            -f2 - f1 - f4,
            -f2 + f1 + f5,
            f2 - f1 - f5,
            -f2 * 2.0 + f1 + f6,
            f2 * 2.0 - f1 - f6,
        ];
        let want_z = [
            -f + f7,
            f - f7,
            -f * 0.74 + f8,
            f * 0.74 - f8,
            -f * 0.74 + f9,
            f * 0.74 - f9,
            -f + f10,
            f - f10,
        ];
        for i in 0..8 {
            let got = s.model.get(s.legs[i]).rotate_angle;
            assert!(
                (got[1] - want_y[i]).abs() < 1e-6,
                "leg {i} yaw {} vs {}",
                got[1],
                want_y[i]
            );
            assert!(
                (got[2] - want_z[i]).abs() < 1e-6,
                "leg {i} roll {} vs {}",
                got[2],
                want_z[i]
            );
        }
        // Standing still the legs sit at the static splay only.
        let mut still = SpiderModel::new();
        still.set_rotation_angles(&AnimParams::still());
        assert_eq!(still.model.get(still.legs[0]).rotate_angle[2], -f);
        assert_eq!(still.model.get(still.legs[6]).rotate_angle[1], -f2 * 2.0);
    }

    // ---- slime -------------------------------------------------------------------------------

    #[test]
    fn slime_outer_gel_has_eyes_and_a_smaller_body_than_the_core() {
        let outer = SlimeModel::new(16);
        let core = SlimeModel::new(0);
        assert!(
            outer.model.parts().len() > core.model.parts().len(),
            "16 -> body + two eyes + mouth, 0 -> body only"
        );
        assert_eq!(outer.model.parts().len(), 4);
        assert_eq!(core.model.parts().len(), 1);

        // The core keeps the 8x8x8 box; the gel is rebuilt as 6x6x6 one unit in on every side.
        assert_eq!(
            core.model.get(core.bodies).cubes[0].pos1,
            [-4.0, 16.0, -4.0]
        );
        assert_eq!(core.model.get(core.bodies).cubes[0].pos2, [4.0, 24.0, 4.0]);
        assert_eq!(core.model.get(core.bodies).texture_offset, [0, 0]);
        assert_eq!(
            outer.model.get(outer.bodies).cubes.len(),
            1,
            "the 8x8x8 box is replaced"
        );
        assert_eq!(
            outer.model.get(outer.bodies).cubes[0].pos1,
            [-3.0, 17.0, -3.0]
        );
        assert_eq!(
            outer.model.get(outer.bodies).cubes[0].pos2,
            [3.0, 23.0, 3.0]
        );
        assert_eq!(outer.model.get(outer.bodies).texture_offset, [0, 16]);

        assert!(core.right_eye.is_none() && core.left_eye.is_none() && core.mouth.is_none());
        let re = outer.right_eye.expect("outer gel has a right eye");
        let le = outer.left_eye.expect("outer gel has a left eye");
        let mo = outer.mouth.expect("outer gel has a mouth");
        assert_eq!(outer.model.get(re).cubes[0].pos1, [-3.25, 18.0, -3.5]);
        assert_eq!(outer.model.get(le).cubes[0].pos1, [1.25, 18.0, -3.5]);
        assert_eq!(outer.model.get(mo).cubes[0].pos2, [1.0, 22.0, -2.5]);
        assert_eq!(outer.model.get(mo).texture_offset, [32, 8]);
    }

    // ---- magma cube --------------------------------------------------------------------------

    #[test]
    fn magma_cube_crust_stacks_and_separates_when_squished() {
        let mut mc = MagmaCubeModel::new();
        // Slab i sits one unit lower than slab i-1 and all eight share the 8x1x8 footprint.
        for (i, &id) in mc.segments.iter().enumerate() {
            let b = &mc.model.get(id).cubes[0];
            assert_eq!(b.pos1, [-4.0, 16.0 + i as f32, -4.0]);
            assert_eq!(b.pos2, [4.0, 17.0 + i as f32, 4.0]);
        }
        // java: only slabs 2 and 3 move off the (0, i) texture column.
        assert_eq!(mc.model.get(mc.segments[1]).texture_offset, [0, 1]);
        assert_eq!(mc.model.get(mc.segments[2]).texture_offset, [24, 10]);
        assert_eq!(mc.model.get(mc.segments[3]).texture_offset, [24, 19]);
        assert_eq!(mc.model.get(mc.segments[7]).texture_offset, [0, 7]);
        assert_eq!(mc.model.get(mc.core).cubes[0].pos1, [-2.0, 18.0, -2.0]);

        // java: rotationPointY = -(4 - i) * f * 1.7
        let mut p = AnimParams::still();
        p.entity.squish_factor = 0.5; // java: the interpolated squishFactor
        mc.set_living_animations(&p);
        for (i, &id) in mc.segments.iter().enumerate() {
            let want = -(4 - i as i32) as f32 * 0.5 * 1.7;
            assert_eq!(mc.model.get(id).rotation_point[1], want, "segment {i}");
        }
        assert_eq!(
            mc.model.get(mc.segments[4]).rotation_point[1],
            0.0,
            "slab 4 is the pivot"
        );

        // A negative squish is clamped to zero, so the crust collapses back into a stack.
        p.entity.squish_factor = -3.0;
        mc.set_living_animations(&p);
        assert!(mc
            .segments
            .iter()
            .all(|&id| mc.model.get(id).rotation_point[1] == 0.0));
    }

    // ---- blaze -------------------------------------------------------------------------------

    #[test]
    fn blaze_rods_orbit_in_three_rings() {
        let mut b = BlazeModel::new();
        assert_eq!(b.model.get(b.sticks[0]).cubes[0].pos2, [2.0, 8.0, 2.0]);
        assert_eq!(b.model.get(b.sticks[11]).texture_offset, [0, 16]);
        assert_eq!(b.model.get(b.head).cubes[0].pos1, [-4.0, -4.0, -4.0]);

        let age = 13.0_f32;
        let mut p = AnimParams::still();
        p.age_in_ticks = age;
        b.set_rotation_angles(&p);

        // Ring 1: radius 9, starting angle age * PI * -0.1, one radian per rod.
        let mut f = age * PI * -0.1;
        for i in 0..4 {
            let s = b.model.get(b.sticks[i]);
            assert_eq!(
                s.rotation_point[1],
                -2.0 + cos(((i * 2) as f32 + age) * 0.25)
            );
            assert_eq!(s.rotation_point[0], cos(f) * 9.0);
            assert_eq!(s.rotation_point[2], sin(f) * 9.0);
            f += 1.0;
        }
        // Ring 2: radius 7. Ring 3: radius 5, and eleven units up.
        //
        // The tolerance is 1e-3 rather than 1e-6 on purpose: `MathHelper.cos` is
        // `SIN_TABLE[(int)(x * 10430.378F + 16384.0F) & 65535]` and the `(int)` cast truncates
        // *toward zero*, so for a negative angle the cosine samples an index 16383 rather than
        // 16384 away from the sine's. `cos^2 + sin^2` is then off by ~2e-4 and the orbit is
        // very slightly elliptical — vanilla's, not ours.
        let radius = |id: PartId| {
            let s = b.model.get(id);
            (s.rotation_point[0] * s.rotation_point[0] + s.rotation_point[2] * s.rotation_point[2])
                .sqrt()
        };
        assert!(
            (radius(b.sticks[5]) - 7.0).abs() < 1e-3,
            "{}",
            radius(b.sticks[5])
        );
        assert!(
            (radius(b.sticks[9]) - 5.0).abs() < 1e-3,
            "{}",
            radius(b.sticks[9])
        );
        assert!(b.model.get(b.sticks[9]).rotation_point[1] > 9.0);
    }

    // ---- enderman ----------------------------------------------------------------------------

    #[test]
    fn enderman_geometry_matches_the_java_constructor() {
        let e = EndermanModel::new(0.0);
        let m = &e.model;
        // The head is ModelBiped's, only lifted by the -14 y-offset.
        assert_eq!(m.get(e.head).texture_offset, [0, 0]);
        assert_eq!(m.get(e.head).cubes[0].pos1, [-4.0, -8.0, -4.0]);
        assert_eq!(m.get(e.head).rotation_point, [0.0, -14.0, 0.0]);
        // The headwear is rebuilt at (0, 16) and *deflated* by half a unit.
        assert_eq!(m.get(e.headwear).texture_offset, [0, 16]);
        assert_eq!(m.get(e.headwear).cubes.len(), 1);
        assert_eq!(
            m.get(e.headwear).cubes[0].quads[4].verts[1].pos,
            [-3.5, -7.5, -3.5]
        );
        // The body moves to (32, 16); the limbs become 2x30x2 at (56, 0).
        assert_eq!(m.get(e.body).texture_offset, [32, 16]);
        assert_eq!(m.get(e.right_arm).texture_offset, [56, 0]);
        assert_eq!(m.get(e.right_arm).cubes[0].pos1, [-1.0, -2.0, -1.0]);
        assert_eq!(m.get(e.right_arm).cubes[0].pos2, [1.0, 28.0, 1.0]);
        assert_eq!(m.get(e.right_arm).rotation_point, [-3.0, -12.0, 0.0]);
        assert_eq!(m.get(e.left_arm).rotation_point, [5.0, -12.0, 0.0]);
        assert_eq!(m.get(e.right_leg).cubes[0].pos1, [-1.0, 0.0, -1.0]);
        assert_eq!(m.get(e.right_leg).rotation_point, [-2.0, -2.0, 0.0]);
        assert_eq!(m.get(e.left_leg).rotation_point, [2.0, -2.0, 0.0]);
        assert!(m.get(e.left_arm).mirror && m.get(e.left_leg).mirror);
        assert!(!m.get(e.right_arm).mirror && !m.get(e.right_leg).mirror);
    }

    #[test]
    fn enderman_head_tracks_yaw_and_pitch_and_sits_at_minus_thirteen() {
        let mut e = EndermanModel::new(0.0);
        let mut p = AnimParams::still();
        p.net_head_yaw = 90.0;
        p.head_pitch = -45.0;
        e.set_rotation_angles(&p);
        let h = e.model.get(e.head);
        assert!((h.rotate_angle[1] - PI / 2.0).abs() < 1e-6);
        assert!((h.rotate_angle[0] + PI / 4.0).abs() < 1e-6);
        // java: `this.bipedHead.rotationPointY = f + 1.0F` with f = -14.
        assert_eq!(h.rotation_point, [0.0, -13.0, -0.0]);
        // The headwear is pinned to the head.
        assert_eq!(e.model.get(e.headwear).rotate_angle, h.rotate_angle);
        assert_eq!(e.model.get(e.headwear).rotation_point, h.rotation_point);
        // java: the body is forced upright and back to the -14 pivot.
        assert_eq!(e.model.get(e.body).rotate_angle[0], 0.0);
        assert_eq!(e.model.get(e.body).rotation_point[1], -14.0);
        assert_eq!(e.model.get(e.right_leg).rotation_point, [-2.0, -5.0, 0.0]);
    }

    /// java: every limb X angle is halved and then clamped into `[-0.4, 0.4]`.
    #[test]
    fn enderman_limb_swing_is_halved_and_clamped() {
        let mut e = EndermanModel::new(0.0);
        let mut p = AnimParams::still();
        p.limb_swing = 0.0;
        p.limb_swing_amount = 1.0;
        e.set_rotation_angles(&p);
        // At limbSwing 0 the biped gives legs +-1.4 and arms +-1.0, all well past the clamp.
        assert_eq!(e.model.get(e.right_leg).rotate_angle[0], 0.4);
        assert_eq!(e.model.get(e.left_leg).rotate_angle[0], -0.4);
        assert_eq!(e.model.get(e.right_arm).rotate_angle[0], -0.4);
        assert_eq!(e.model.get(e.left_arm).rotate_angle[0], 0.4);

        // A small swing stays inside the clamp and is exactly half the biped value.
        let mut e2 = EndermanModel::new(0.0);
        let mut p2 = AnimParams::still();
        p2.limb_swing = 2.358_0; // near the cosine zero, so the raw angle is tiny
        p2.limb_swing_amount = 0.2;
        e2.set_rotation_angles(&p2);
        let want = (f64::from(cos(2.358_0 * 0.6662) * 1.4 * 0.2) * 0.5) as f32;
        assert!(
            want.abs() < 0.4,
            "the fixture must stay inside the clamp, got {want}"
        );
        assert_eq!(e2.model.get(e2.right_leg).rotate_angle[0], want);
    }

    #[test]
    fn enderman_carrying_and_attacking_poses() {
        let mut e = EndermanModel::new(0.0);
        let mut p = AnimParams::still();
        p.limb_swing = 6.0;
        p.limb_swing_amount = 1.0;
        p.entity = EntityAnimState {
            carrying_block: true, // java: ModelEnderman.isCarrying
            aggressive: true,     // java: ModelEnderman.isAttacking
            ..Default::default()
        };
        e.set_rotation_angles(&p);
        // java: both arms are forced forward and splayed, overriding the clamped swing.
        assert_eq!(e.model.get(e.right_arm).rotate_angle[0], -0.5);
        assert_eq!(e.model.get(e.left_arm).rotate_angle[0], -0.5);
        assert_eq!(e.model.get(e.right_arm).rotate_angle[2], 0.05);
        assert_eq!(e.model.get(e.left_arm).rotate_angle[2], -0.05);
        // java: attacking lifts the head five units — but only after the headwear was pinned.
        assert_eq!(e.model.get(e.head).rotation_point[1], -18.0);
        assert_eq!(e.model.get(e.headwear).rotation_point[1], -13.0);

        let mut calm = EndermanModel::new(0.0);
        calm.set_rotation_angles(&AnimParams::still());
        assert_eq!(calm.model.get(calm.head).rotation_point[1], -13.0);
    }

    // ---- ghast -------------------------------------------------------------------------------

    #[test]
    fn ghast_tentacles_come_off_the_vanilla_random_seed() {
        let g = GhastModel::new();
        // java: `new Random(1660L)` then `nextInt(7) + 8` nine times.
        assert_eq!(g.tentacle_lengths(), [8, 13, 9, 11, 11, 10, 12, 9, 12]);
        for (j, &id) in g.tentacles.iter().enumerate() {
            let b = &g.model.get(id).cubes[0];
            assert_eq!(b.pos1, [-1.0, 0.0, -1.0]);
            assert_eq!(b.pos2, [1.0, g.tentacle_lengths()[j] as f32, 1.0]);
            assert_eq!(g.model.get(id).rotation_point[1], 15.0, "31 + i, i = -16");
        }
        // The 3x3 grid: X cycles -3.75, 1.25, 6.25 shifted by half a unit on the middle row.
        let x: Vec<f32> = g
            .tentacles
            .iter()
            .map(|&t| g.model.get(t).rotation_point[0])
            .collect();
        assert_eq!(
            x,
            vec![-3.75, 1.25, 6.25, -6.25, -1.25, 3.75, -3.75, 1.25, 6.25]
        );
        let z: Vec<f32> = g
            .tentacles
            .iter()
            .map(|&t| g.model.get(t).rotation_point[2])
            .collect();
        assert_eq!(z, vec![-5.0, -5.0, -5.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0]);
        // body: 16^3 centred, lifted to 24 + i = 8.
        assert_eq!(g.model.get(g.body).cubes[0].pos1, [-8.0, -8.0, -8.0]);
        assert_eq!(g.model.get(g.body).rotation_point, [0.0, 8.0, 0.0]);
    }

    #[test]
    fn ghast_tentacles_wobble_one_radian_apart() {
        let mut g = GhastModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 42.0;
        g.set_rotation_angles(&p);
        for (i, &id) in g.tentacles.iter().enumerate() {
            let want = 0.2 * sin(42.0 * 0.3 + i as f32) + 0.4;
            assert_eq!(g.model.get(id).rotate_angle[0], want, "tentacle {i}");
        }
        // java: the render pass sits everything 0.6 blocks up.
        let mut a = Vec::new();
        let mut b = Vec::new();
        g.render(&p, Mat4::IDENTITY, &mut a);
        g.model.emit_all(p.scale, Mat4::IDENTITY, &mut b);
        assert_eq!(a.len(), b.len());
        assert!((a[0].pos[1] - b[0].pos[1] - 0.6).abs() < 1e-6);
    }

    // ---- silverfish and endermite ------------------------------------------------------------

    #[test]
    fn silverfish_segments_chain_along_z() {
        let s = SilverfishModel::new();
        let m = &s.model;
        // java: f starts at -3.5 and advances by half the sum of neighbouring depths.
        let want_z = [-3.5, -1.5, 1.0, 4.0, 7.0, 9.5, 11.5];
        for (i, &id) in s.body_parts.iter().enumerate() {
            let len = SILVERFISH_BOX_LENGTH[i];
            let p = m.get(id);
            assert_eq!(
                p.rotation_point,
                [0.0, (24 - len[1]) as f32, want_z[i]],
                "segment {i}"
            );
            assert_eq!(
                p.cubes[0].pos1,
                [len[0] as f32 * -0.5, 0.0, len[2] as f32 * -0.5]
            );
            assert_eq!(p.texture_offset, SILVERFISH_TEXTURE_POSITIONS[i]);
        }
        // The three plates sit on segments 2, 4 and 1.
        assert_eq!(m.get(s.wings[0]).rotation_point, [0.0, 16.0, 1.0]);
        assert_eq!(m.get(s.wings[1]).rotation_point, [0.0, 20.0, 7.0]);
        assert_eq!(m.get(s.wings[2]).rotation_point, [0.0, 19.0, -1.5]);
        // wing 2 takes its z offset from segment 4 (depth 3) but its depth from segment 1 (2).
        assert_eq!(m.get(s.wings[2]).cubes[0].pos1, [-3.0, 0.0, -1.5]);
        assert_eq!(m.get(s.wings[2]).cubes[0].pos2, [3.0, 5.0, 0.5]);
    }

    #[test]
    fn silverfish_wave_grows_away_from_the_middle_and_the_plates_follow() {
        let mut s = SilverfishModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 20.0;
        s.set_rotation_angles(&p);
        for (i, &id) in s.body_parts.iter().enumerate() {
            let i = i as i32;
            let phase = 20.0 * 0.9 + i as f32 * 0.15 * PI;
            let part = s.model.get(id);
            assert_eq!(
                part.rotate_angle[1],
                cos(phase) * PI * 0.05 * (1 + (i - 2).abs()) as f32
            );
            assert_eq!(
                part.rotation_point[0],
                sin(phase) * PI * 0.2 * (i - 2).abs() as f32
            );
        }
        // Segment 2 is the node: it yaws but never slides sideways.
        assert_eq!(s.model.get(s.body_parts[2]).rotation_point[0], 0.0);
        // java: the plates copy segments 2, 4 and 1.
        let m = &s.model;
        assert_eq!(
            m.get(s.wings[0]).rotate_angle[1],
            m.get(s.body_parts[2]).rotate_angle[1]
        );
        assert_eq!(
            m.get(s.wings[1]).rotation_point[0],
            m.get(s.body_parts[4]).rotation_point[0]
        );
        assert_eq!(
            m.get(s.wings[2]).rotate_angle[1],
            m.get(s.body_parts[1]).rotate_angle[1]
        );
        // wing 0 copies only the yaw, never the sideways slide.
        assert_eq!(m.get(s.wings[0]).rotation_point[0], 0.0);
    }

    #[test]
    fn endermite_is_a_four_segment_silverfish_with_a_fifth_of_the_yaw() {
        let mut e = EndermiteModel::new();
        let m = &e.model;
        let want_z = [-3.5, 0.0, 3.0, 4.0];
        for (i, &id) in e.body_parts.iter().enumerate() {
            let len = ENDERMITE_BOX_LENGTH[i];
            let p = m.get(id);
            assert_eq!(
                p.rotation_point,
                [0.0, (24 - len[1]) as f32, want_z[i]],
                "segment {i}"
            );
            assert_eq!(
                p.cubes[0].pos1,
                [len[0] as f32 * -0.5, 0.0, len[2] as f32 * -0.5]
            );
            assert_eq!(
                p.cubes[0].pos2,
                [len[0] as f32 * 0.5, len[1] as f32, len[2] as f32 * 0.5]
            );
        }
        assert_eq!(m.get(e.body_parts[1]).texture_offset, [0, 5]);

        let mut p = AnimParams::still();
        p.age_in_ticks = 20.0;
        e.set_rotation_angles(&p);
        for (i, &id) in e.body_parts.iter().enumerate() {
            let i = i as i32;
            let phase = 20.0 * 0.9 + i as f32 * 0.15 * PI;
            let part = e.model.get(id);
            assert_eq!(
                part.rotate_angle[1],
                cos(phase) * PI * 0.01 * (1 + (i - 2).abs()) as f32
            );
            assert_eq!(
                part.rotation_point[0],
                sin(phase) * PI * 0.1 * (i - 2).abs() as f32
            );
        }
    }

    // ---- guardian ----------------------------------------------------------------------------

    #[test]
    fn guardian_body_is_five_boxes_and_owns_every_other_part() {
        let g = GuardianModel::new();
        let m = &g.model;
        assert_eq!(m.texture_width, 64);
        assert_eq!(m.texture_height, 64);
        assert_eq!(
            m.roots(),
            &[g.body],
            "render() draws the body and nothing else"
        );

        let b = m.get(g.body);
        assert_eq!(b.cubes.len(), 5);
        assert_eq!(b.cubes[0].pos1, [-6.0, 10.0, -8.0]);
        assert_eq!(b.cubes[0].pos2, [6.0, 22.0, 8.0]);
        assert_eq!(b.cubes[1].pos1, [-8.0, 10.0, -6.0]);
        assert_eq!(b.cubes[2].pos1, [6.0, 10.0, -6.0]);
        assert_eq!(b.cubes[3].pos1, [-6.0, 8.0, -6.0]);
        assert_eq!(b.cubes[4].pos1, [-6.0, 22.0, -6.0]);
        // java: the right flank uses the per-box mirror overload, which must not latch.
        assert!(!b.mirror, "addBox(..., true) is per box, not a mode");
        assert_eq!(
            b.cubes[1].quads[4].verts[0].uv,
            b.cubes[2].quads[4].verts[3].uv
        );

        // 12 spines + eye + tail root hang off the body; the tail is a chain.
        assert_eq!(b.children.len(), 14);
        assert_eq!(m.get(g.tail[0]).children, vec![g.tail[1]]);
        assert_eq!(m.get(g.tail[1]).children, vec![g.tail[2]]);
        assert_eq!(m.get(g.tail[2]).cubes.len(), 2);
        assert_eq!(m.get(g.tail[2]).cubes[1].pos1, [1.0, 10.5, 3.0]);
        assert_eq!(m.get(g.spines[0]).cubes[0].pos1, [-1.0, -4.5, -1.0]);
        assert_eq!(m.get(g.eye).cubes[0].pos2, [1.0, 17.0, 1.0]);
    }

    #[test]
    fn guardian_spines_retract_as_the_spike_animation_falls() {
        let mut g = GuardianModel::new();
        let mut p = AnimParams::still();
        p.age_in_ticks = 8.0;
        p.entity.ticks_existed = 8; // so the java `f` (partial ticks) is 0

        // Fully extended: f1 = 0, so each spine sits at its raw afloat3..5 position.
        p.entity.spikes_animation = 1.0;
        g.set_rotation_angles(&p);
        for i in 0..12 {
            let wobble = 1.0 + cos(8.0 * 1.5 + i as f32) * 0.01;
            let s = g.model.get(g.spines[i]);
            assert_eq!(
                s.rotate_angle,
                [
                    PI * SPINE_ROT_X[i],
                    PI * SPINE_ROT_Y[i],
                    PI * SPINE_ROT_Z[i]
                ]
            );
            assert_eq!(s.rotation_point[0], SPINE_POS_X[i] * wobble);
            assert_eq!(s.rotation_point[1], 16.0 + SPINE_POS_Y[i] * wobble);
            assert_eq!(s.rotation_point[2], SPINE_POS_Z[i] * wobble);
        }

        // Fully retracted: f1 = 0.55, so the offsets shrink to 45% (plus the wobble).
        p.entity.spikes_animation = 0.0;
        g.set_rotation_angles(&p);
        let wobble = 1.0 + cos(8.0 * 1.5 + 2.0) * 0.01 - 0.55;
        assert_eq!(g.model.get(g.spines[2]).rotation_point[0], 8.0 * wobble);
        assert!(g.model.get(g.spines[2]).rotation_point[0] < 8.0 * 0.5);
    }

    /// java: `EntityGuardian.onLivingUpdate` (`:298-345`), the `isRemote` half, plus its two
    /// interpolators `func_175471_a` (`:411`) and `func_175469_o` (`:416`).
    #[test]
    fn the_guardian_tick_springs_the_tail_and_the_spikes() {
        use crate::entity::anim::{GuardianAnim, SpeciesState};

        // Swimming and still: the tail spring eases towards 0.125 and the spikes towards 1.
        let mut g = GuardianAnim::new(7);
        for _ in 0..200 {
            g.prev_tail_phase = g.tail_phase;
            g.tail_speed += (0.125 - g.tail_speed) * 0.2;
            g.tail_phase += g.tail_speed;
            g.prev_spikes = g.spikes;
            g.spikes += (1.0 - g.spikes) * 0.06;
        }
        assert!((g.tail_speed - 0.125).abs() < 1e-3, "{}", g.tail_speed);
        assert!(g.spikes > 0.99, "{}", g.spikes);
        // java: `func_175469_o(p)` = prev + (cur - prev) * p.
        g.prev_spikes = 0.0;
        g.spikes = 1.0;
        assert!((g.spikes(0.5) - 0.5).abs() < 1e-6);
        g.prev_tail_phase = 10.0;
        g.tail_phase = 12.0;
        assert!((g.tail_phase(0.25) - 10.5).abs() < 1e-6);

        // The two interpolations reach the model as ordinary `EntityAnimState` fields.
        let state = super::super::super::anim::LivingRenderState {
            species: SpeciesState::Guardian(g),
            ..Default::default()
        };
        let p = state.params(0.5);
        assert!((p.entity.spikes_animation - 0.5).abs() < 1e-6);
        assert!((p.entity.tail_rotation - 11.0).abs() < 1e-6);
        // java: `RenderGuardian` does **not** override `handleRotationFloat`, and
        // `ModelGuardian` relies on that to recover `f = ageInTicks - ticksExisted`.
        assert!((p.age_in_ticks - 0.5).abs() < 1e-6);

        // Beached: the tail speed is pinned at 2.0 and the spikes are redrawn from `Entity.rand`
        // every tick, so consecutive values differ and all of them are in `0..1`.
        let mut g = GuardianAnim::new(7);
        let mut seen = Vec::new();
        for _ in 0..8 {
            g.tail_speed = 2.0;
            g.tail_phase += g.tail_speed;
            g.spikes = g.next_float();
            assert!((0.0..1.0).contains(&g.spikes), "{}", g.spikes);
            seen.push(g.spikes);
        }
        assert!((g.tail_phase - 16.0).abs() < 1e-6);
        assert!(seen.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-6));
    }

    #[test]
    fn guardian_tail_and_eye_follow_the_tail_phase() {
        let mut g = GuardianModel::new();
        let mut p = AnimParams::still();
        // java: `f2 = entityguardian.func_175471_a(f)` -> EntityAnimState::tail_rotation
        p.entity.tail_rotation = 1.75;
        g.set_rotation_angles(&p);
        assert_eq!(
            g.model.get(g.tail[0]).rotate_angle[1],
            sin(1.75) * PI * 0.05
        );
        assert_eq!(g.model.get(g.tail[1]).rotate_angle[1], sin(1.75) * PI * 0.1);
        assert_eq!(
            g.model.get(g.tail[2]).rotate_angle[1],
            sin(1.75) * PI * 0.15
        );
        assert_eq!(g.model.get(g.tail[1]).rotation_point, [-1.5, 0.5, 14.0]);
        assert_eq!(g.model.get(g.tail[2]).rotation_point, [0.5, 0.5, 6.0]);
        assert_eq!(g.model.get(g.eye).rotation_point[2], -8.25);
        assert!(g.model.get(g.eye).show_model);

        // java: the look-at block. A target dead ahead gives a dot product of 0 and a centred
        // eye; a target below the guardian pushes the pupil down to y = 1.
        g.look_at([0.0, 10.0, 0.0], [0.0, 4.0, -5.0], [0.0, 0.0, -1.0]);
        assert_eq!(g.model.get(g.eye).rotation_point[1], 1.0);
        assert!(g.model.get(g.eye).rotation_point[0].abs() < 1e-3);
        // A target off to one side swings the pupil, and the two sides are mirror images.
        g.look_at([0.0, 4.0, 0.0], [5.0, 10.0, -5.0], [0.0, 0.0, -1.0]);
        assert_eq!(
            g.model.get(g.eye).rotation_point[1],
            0.0,
            "target above -> pupil up"
        );
        let right = g.model.get(g.eye).rotation_point[0];
        g.look_at([0.0, 4.0, 0.0], [-5.0, 10.0, -5.0], [0.0, 0.0, -1.0]);
        let left = g.model.get(g.eye).rotation_point[0];
        assert!(
            right.abs() > 1e-3,
            "an off-axis target must move the pupil, got {right}"
        );
        assert!((right + left).abs() < 1e-5, "{right} vs {left}");

        // The production trait-object path enters through set_rotation_angles, rather than
        // calling the concrete helper. Omitting that bridge must leave this off-axis target
        // centred and fail here.
        p.entity.guardian_look_at = Some(crate::entity::anim::GuardianLookAt {
            self_eyes: [0.0, 4.0, 0.0],
            target_eyes: [5.0, 10.0, -5.0],
            self_look: [0.0, 0.0, -1.0],
        });
        let mut bridged = GuardianModel::new();
        bridged.set_rotation_angles(&p);
        assert_eq!(bridged.model.get(bridged.eye).rotation_point[1], 0.0);
        assert!(bridged.model.get(bridged.eye).rotation_point[0].abs() > 1e-3);
    }

    // ---- dragon -----------------------------------------------------------------------------

    #[test]
    fn dragon_geometry_matches_the_java_constructor_and_part_tree() {
        let dragon = DragonModel::new();
        let m = &dragon.model;
        assert_eq!((m.texture_width, m.texture_height), (256, 256));
        assert_eq!(m.parts().len(), 12);
        assert_eq!(
            m.roots().len(),
            6,
            "head, neck, body, wing and two leg roots"
        );

        let head = m.get(dragon.parts.head);
        assert_eq!(head.cubes.len(), 6);
        assert_eq!(head.rotation_point, [0.0; 3]);
        assert_eq!(head.cubes[0].pos1, [-6.0, -1.0, -24.0]);
        assert_eq!(head.cubes[0].pos2, [6.0, 4.0, -8.0]);
        assert_eq!(head.cubes[1].pos1, [-8.0, -8.0, -10.0]);
        assert_eq!(head.cubes[1].pos2, [8.0, 8.0, 6.0]);
        assert_eq!(head.children, vec![dragon.parts.jaw]);
        assert_eq!(m.get(dragon.parts.jaw).cubes[0].pos1, [-6.0, 0.0, -16.0]);
        assert_eq!(m.get(dragon.parts.jaw).rotation_point, [0.0, 4.0, -8.0]);

        let spine = m.get(dragon.parts.spine);
        assert_eq!(spine.cubes.len(), 2);
        assert_eq!(spine.cubes[0].pos1, [-5.0, -5.0, -5.0]);
        assert_eq!(spine.cubes[1].pos2, [1.0, -5.0, 3.0]);
        assert_eq!(m.get(dragon.parts.body).cubes[0].pos2, [12.0, 24.0, 48.0]);

        // Java's wing membranes are zero-height boxes and intentionally begin at U = -56.
        assert_eq!(m.get(dragon.parts.wing).cubes[1].pos2, [0.0, 0.0, 58.0]);
        assert_eq!(
            m.get(dragon.parts.wing).cubes[1].quads[1].verts[1].uv[0],
            -56.0 / 256.0
        );
        assert_eq!(
            m.get(dragon.parts.wing).children,
            vec![dragon.parts.wing_tip]
        );
        assert_eq!(m.get(dragon.parts.wing_tip).cubes[1].pos2, [0.0, 0.0, 58.0]);
        assert_eq!(
            m.get(dragon.parts.front_leg).children,
            vec![dragon.parts.front_leg_tip]
        );
        assert_eq!(
            m.get(dragon.parts.front_leg_tip).children,
            vec![dragon.parts.front_foot]
        );
        assert_eq!(
            m.get(dragon.parts.rear_leg).children,
            vec![dragon.parts.rear_leg_tip]
        );
        assert_eq!(
            m.get(dragon.parts.rear_leg_tip).children,
            vec![dragon.parts.rear_foot]
        );
    }

    #[test]
    fn dragon_ring_state_matches_short_arc_interpolation_and_client_tick_order() {
        let mut state = DragonRenderState {
            ring_buffer_index: 10,
            ..Default::default()
        };
        state.ring_buffer[10] = [170.0, 4.0, 7.0];
        state.ring_buffer[9] = [-170.0, 8.0, 11.0];
        assert_eq!(state.movement_offset(0, 0.25), [185.0, 7.0, 10.0]);

        state.dead = true;
        assert_eq!(state.movement_offset(0, 0.25), [190.0, 8.0, 11.0]);

        let mut ticked = DragonRenderState::default();
        ticked.tick(45.0, 12.0, 0.0, 0.0, 0.0, false, false, false);
        assert_eq!(ticked.prev_anim_time, 0.0);
        assert_eq!(ticked.anim_time, 0.2);
        assert_eq!(ticked.ring_buffer_index, 0);
        assert!(ticked
            .ring_buffer
            .iter()
            .all(|sample| *sample == [45.0, 12.0, 0.0]));

        // NoAI pins the phase and does not advance the ring; death only snapshots the prior phase.
        ticked.tick(90.0, 20.0, 1.0, 1.0, 1.0, false, false, true);
        assert_eq!(ticked.prev_anim_time, 0.2);
        assert_eq!(ticked.anim_time, 0.5);
        assert_eq!(ticked.ring_buffer_index, 0);
        ticked.tick(180.0, 30.0, 0.0, 0.0, 0.0, false, true, false);
        assert_eq!(ticked.prev_anim_time, 0.5);
        assert_eq!(ticked.anim_time, 0.5);
        assert!(ticked.dead);
    }

    #[test]
    fn dragon_pose_and_render_use_animation_state_and_render_all_repeated_segments() {
        let mut dragon = DragonModel::new();
        let mut p = AnimParams::still();
        p.partial_ticks = 0.5;
        p.entity.dragon = Some(DragonRenderState {
            prev_anim_time: 0.0,
            anim_time: 0.25,
            ..DragonRenderState::neutral(0.0)
        });
        let mut out = Vec::new();
        dragon.pose_and_render(&p, Mat4::IDENTITY, &mut out);

        let phase = 0.25 * PI;
        let jaw = (dragon_sin(phase) + 1.0) * 0.2;
        assert_eq!(dragon.model.get(dragon.parts.jaw).rotate_angle[0], jaw);
        assert_eq!(dragon.model.get(dragon.parts.wing).rotate_angle[1], 0.25);
        assert_eq!(out.len(), 65 * 36);
        assert!(out.iter().any(|v| v.pos != [0.0; 3]));
        assert!(out.iter().all(|v| {
            v.pos.iter().all(|c| c.is_finite()) && v.uv.iter().all(|c| c.is_finite())
        }));
    }

    // ---- wither -----------------------------------------------------------------------------

    #[test]
    fn wither_geometry_and_draw_order_match_the_java_model() {
        let wither = WitherModel::new(0.0);
        assert_eq!(wither.model.parts().len(), 6);
        assert_eq!(
            wither.model.get(wither.body[0]).cubes[0].pos1,
            [-10.0, 3.9, -0.5]
        );
        assert_eq!(
            wither.model.get(wither.body[0]).cubes[0].pos2,
            [10.0, 6.9, 2.5]
        );
        assert_eq!(wither.model.get(wither.body[1]).cubes.len(), 4);
        assert_eq!(
            wither.model.get(wither.body[1]).rotation_point,
            [-2.0, 6.9, -0.5]
        );
        assert_eq!(
            wither.model.get(wither.heads[1]).rotation_point,
            [-8.0, 4.0, 0.0]
        );
        assert_eq!(
            wither.model.get(wither.heads[2]).rotation_point,
            [10.0, 4.0, 0.0]
        );

        let mut out = Vec::new();
        wither.render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
        // 3 skulls + 1 rib + 4 spine ribs + 1 tail = 9 boxes, six faces each, six vertices.
        assert_eq!(out.len(), 9 * 6 * 6);
    }

    #[test]
    fn wither_pose_uses_central_look_side_look_and_waving_tail() {
        let mut wither = WitherModel::new(0.0);
        let mut p = AnimParams::still();
        p.age_in_ticks = 7.25;
        p.net_head_yaw = 35.0;
        p.head_pitch = -12.0;
        p.entity.wither_side_head_yaw = [-45.0, 70.0];
        p.entity.wither_side_head_pitch = [15.0, -20.0];
        wither.pose_and_render(&p, Mat4::IDENTITY, &mut Vec::new());

        assert_eq!(
            wither.model.get(wither.heads[0]).rotate_angle,
            [model_radians(-12.0), model_radians(35.0), 0.0]
        );
        assert_eq!(
            wither.model.get(wither.heads[1]).rotate_angle,
            [model_radians(15.0), model_radians(-45.0), 0.0]
        );
        assert_eq!(
            wither.model.get(wither.heads[2]).rotate_angle,
            [model_radians(-20.0), model_radians(70.0), 0.0]
        );

        let body1_x = (0.065 + 0.05 * cos(7.25 * 0.1)) * PI;
        assert_eq!(wither.model.get(wither.body[1]).rotate_angle[0], body1_x);
        assert_eq!(
            wither.model.get(wither.body[2]).rotation_point,
            [-2.0, 6.9 + cos(body1_x) * 10.0, -0.5 + sin(body1_x) * 10.0]
        );
        assert_eq!(
            wither.model.get(wither.body[2]).rotate_angle[0],
            (0.265 + 0.1 * cos(7.25 * 0.1)) * PI
        );
    }
}
