//! The concrete vanilla model definitions.
//!
//! java: `client/model/Model*.java` — one Rust type per Java class, named `<Thing>Model`.
//!
//! Every model here is a straight port: the constructor reproduces the Java constructor's
//! `addBox`/`setRotationPoint` calls in order, and [`EntityModel::set_rotation_angles`] the
//! Java `setRotationAngles` body statement for statement. Two rules make that mechanical:
//!
//! * Trig goes through [`mc_math::sin`]/[`mc_math::cos`] — vanilla's `MathHelper`, not libm
//!   (hazard H1).
//! * Degrees become radians through [`super::anim::model_radians`], preserving vanilla's
//!   `angle / (180F / (float)Math.PI)` division and its `f32` rounding point.
//!
//! [`ModelKind`] is the closed set of models and [`ModelKind::build`] is the factory
//! [`super::manager`] dispatches through.

use super::anim::EntityModel;

pub mod biped;
pub mod hostile;
pub mod object;
pub mod passive;
pub mod quadruped;
pub mod tileent;

/// Which vanilla model to build.
///
/// java: the `new Model*()` argument at each `RenderManager.entityRenderMap.put` site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ModelKind {
    /// java: `ModelBiped`
    Biped,
    /// java: `ModelPlayer` (classic arms)
    Player,
    /// java: `ModelPlayer` with `smallArms = true` (the Alex model)
    PlayerSlim,
    /// java: `ModelZombie`
    Zombie,
    /// java: `ModelZombieVillager`
    ZombieVillager,
    /// java: `ModelSkeleton`
    Skeleton,
    /// java: `ModelVillager`
    Villager,
    /// java: `ModelWitch`
    Witch,
    /// java: `ModelPig`
    Pig,
    /// java: `ModelCow`
    Cow,
    /// java: `ModelSheep2` — the sheared body
    Sheep,
    /// java: `ModelSheep1` — the wool layer
    SheepWool,
    /// java: `ModelChicken`
    Chicken,
    /// java: `ModelWolf`
    Wolf,
    /// java: `ModelOcelot`
    Ocelot,
    /// java: `ModelRabbit`
    Rabbit,
    /// java: `ModelSquid`
    Squid,
    /// java: `ModelBat`
    Bat,
    /// java: `ModelSnowMan`
    SnowMan,
    /// java: `ModelIronGolem`
    IronGolem,
    /// java: `ModelHorse`
    ///
    /// One model covers horses, donkeys, mules, zombie horses and skeleton horses; the watched
    /// horse type selects ears/textures at render time.
    Horse,
    /// java: `ModelCreeper`
    Creeper,
    /// java: `ModelSpider`
    Spider,
    /// java: `new ModelSlime(16)` — `RenderSlime`'s **main** model: the six-unit inner
    /// body plus the two eyes and the mouth. The constructor argument is the texture V
    /// offset, and `> 0` is also what switches the eyes on (`ModelSlime:24`).
    Slime,
    /// java: `new ModelSlime(0)` — `LayerSlimeGel`'s **outer** shell: one plain
    /// eight-unit cube with no face, drawn translucent over the body.
    ///
    /// These two doc comments used to be the wrong way round, which is how
    /// `LayerKind::SlimeGel` came to point at `ModelKind::Slime` — the body's own
    /// geometry.
    SlimeGel,
    /// java: `ModelMagmaCube`
    MagmaCube,
    /// java: `ModelBlaze`
    Blaze,
    /// java: `ModelEnderman`
    Enderman,
    /// java: `ModelGhast`
    Ghast,
    /// java: `ModelSilverfish`
    Silverfish,
    /// java: `ModelEnderMite`
    Endermite,
    /// java: `ModelGuardian`
    Guardian,
    /// java: `ModelDragon`
    Dragon,
    /// java: `ModelWither`
    Wither,
    /// java: `ModelBoat`
    Boat,
    /// java: `ModelMinecart`
    Minecart,
    /// java: `ModelLeashKnot`
    LeashKnot,
    /// java: `ModelEnderCrystal` with the base
    EnderCrystal,
    /// java: `ModelArmorStand`
    ArmorStand,
    /// java: `ModelArmorStandArmor`
    ArmorStandArmor,
    /// java: `ModelChest`
    Chest,
    /// java: `ModelLargeChest`
    LargeChest,
    /// java: `ModelSign`
    Sign,
    /// java: `ModelBanner`
    Banner,
    /// java: `ModelBook`
    Book,
    /// java: `ModelSkeletonHead`
    SkeletonHead,
    /// java: `RenderWitherSkull`'s no-argument `new ModelSkeletonHead()` — `(0, 35, 64, 64)`.
    ///
    /// This must stay separate from [`ModelKind::SkeletonHead`], which is the tile-skull
    /// renderer's explicitly-sized `(0, 0, 64, 32)` model. Vanilla keeps those as distinct
    /// model instances with distinct texture sheets.
    WitherSkull,
    /// java: `ModelHumanoidHead`
    HumanoidHead,
}

impl ModelKind {
    /// Builds the model at zero inflation — the shape every entity's own body is drawn with.
    ///
    /// Equivalent to `self.build(0.0)`. Kept as the common case because most call sites (the
    /// body cache in [`super::manager::RenderManager`], every test in this module) never want a
    /// layer's inflated copy.
    #[must_use]
    pub fn build_default(self) -> Box<dyn EntityModel> {
        self.build(0.0)
    }

    /// Builds the model at the given inflation — vanilla's `scaleFactor`/`modelSize`
    /// constructor argument, baked into the box geometry at construction (see
    /// [`super::model::ModelBox::new`]'s `delta`).
    ///
    /// Only the handful of models whose Java constructor actually exposes that argument use it;
    /// the rest have no inflatable form in vanilla either (there is no `LayerZombieCharge`) and
    /// silently ignore it, which mirrors the Java exactly — those `Model*` classes simply have
    /// no such parameter to pass. java: the `new Model*(...)` call at each `RenderManager`/
    /// `Layer*` construction site — [`super::layers::LayerKind::model_inflation`] and
    /// [`super::layers::ArmorSlot::model_inflation`] are what a layer passes here.
    #[must_use]
    pub fn build(self, inflation: f32) -> Box<dyn EntityModel> {
        use ModelKind as K;
        match self {
            K::Biped => Box::new(biped::BipedModel::new(inflation)),
            K::Player => Box::new(biped::PlayerModel::new(inflation, false)),
            K::PlayerSlim => Box::new(biped::PlayerModel::new(inflation, true)),
            K::Zombie => Box::new(biped::ZombieModel::new()),
            K::ZombieVillager => Box::new(biped::ZombieVillagerModel::new()),
            K::Skeleton => Box::new(biped::SkeletonModel::sized(inflation)),
            K::Villager => Box::new(passive::VillagerModel::new(inflation)),
            K::Witch => Box::new(passive::WitchModel::new(inflation)),
            K::Pig => Box::new(quadruped::PigModel::sized(inflation)),
            K::Cow => Box::new(quadruped::CowModel::new()),
            K::Sheep => Box::new(quadruped::SheepModel::new()),
            K::SheepWool => Box::new(quadruped::SheepWoolModel::new()),
            K::Chicken => Box::new(passive::ChickenModel::new()),
            K::Wolf => Box::new(passive::WolfModel::new()),
            K::Ocelot => Box::new(passive::OcelotModel::new()),
            K::Rabbit => Box::new(passive::RabbitModel::new()),
            K::Squid => Box::new(passive::SquidModel::new()),
            K::Bat => Box::new(passive::BatModel::new()),
            K::SnowMan => Box::new(passive::SnowManModel::new()),
            K::IronGolem => Box::new(passive::IronGolemModel::new()),
            K::Horse => Box::new(passive::HorseModel::new()),
            K::Creeper => Box::new(hostile::CreeperModel::new(inflation)),
            K::Spider => Box::new(hostile::SpiderModel::new()),
            // java: `ModelSlime`'s constructor takes a box *count*, not a scale factor — there
            // is no inflatable form, so the inflation argument has nothing to reach.
            K::Slime => Box::new(hostile::SlimeModel::new(16)),
            K::SlimeGel => Box::new(hostile::SlimeModel::new(0)),
            K::MagmaCube => Box::new(hostile::MagmaCubeModel::new()),
            K::Blaze => Box::new(hostile::BlazeModel::new()),
            K::Enderman => Box::new(hostile::EndermanModel::new(inflation)),
            K::Ghast => Box::new(hostile::GhastModel::new()),
            K::Silverfish => Box::new(hostile::SilverfishModel::new()),
            K::Endermite => Box::new(hostile::EndermiteModel::new()),
            K::Guardian => Box::new(hostile::GuardianModel::new()),
            K::Dragon => Box::new(hostile::DragonModel::new()),
            K::Wither => Box::new(hostile::WitherModel::new(inflation)),
            K::Boat => Box::new(object::BoatModel::new()),
            K::Minecart => Box::new(object::MinecartModel::new()),
            K::LeashKnot => Box::new(object::LeashKnotModel::new()),
            K::EnderCrystal => Box::new(object::EnderCrystalModel::new(true)),
            K::ArmorStand => Box::new(object::ArmorStandModel::new()),
            K::ArmorStandArmor => Box::new(object::ArmorStandArmorModel::new(inflation)),
            K::Chest => Box::new(tileent::ChestModel::new()),
            K::LargeChest => Box::new(tileent::LargeChestModel::new()),
            K::Sign => Box::new(tileent::SignModel::new()),
            K::Banner => Box::new(tileent::BannerModel::new()),
            K::Book => Box::new(tileent::BookModel::new()),
            K::SkeletonHead => Box::new(tileent::SkeletonHeadModel::new(0, 0, 64, 32)),
            K::WitherSkull => Box::new(tileent::SkeletonHeadModel::new_wither_skull()),
            K::HumanoidHead => Box::new(tileent::HumanoidHeadModel::new()),
        }
    }

    /// Every kind, for exhaustive tests.
    #[must_use]
    pub fn all() -> &'static [ModelKind] {
        use ModelKind as K;
        &[
            K::Biped,
            K::Player,
            K::PlayerSlim,
            K::Zombie,
            K::ZombieVillager,
            K::Skeleton,
            K::Villager,
            K::Witch,
            K::Pig,
            K::Cow,
            K::Sheep,
            K::SheepWool,
            K::Chicken,
            K::Wolf,
            K::Ocelot,
            K::Rabbit,
            K::Squid,
            K::Bat,
            K::SnowMan,
            K::IronGolem,
            K::Horse,
            K::Creeper,
            K::Spider,
            K::Slime,
            K::SlimeGel,
            K::MagmaCube,
            K::Blaze,
            K::Enderman,
            K::Ghast,
            K::Silverfish,
            K::Endermite,
            K::Guardian,
            K::Dragon,
            K::Wither,
            K::Boat,
            K::Minecart,
            K::LeashKnot,
            K::EnderCrystal,
            K::ArmorStand,
            K::ArmorStandArmor,
            K::Chest,
            K::LargeChest,
            K::Sign,
            K::Banner,
            K::Book,
            K::SkeletonHead,
            K::WitherSkull,
            K::HumanoidHead,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::AnimParams;
    use glam::Mat4;

    /// Every model must build, pose and emit geometry without panicking, at a spread of
    /// animation inputs. This is the net that catches an out-of-range `PartId` or a part that
    /// was never given a box.
    #[test]
    fn every_model_builds_poses_and_emits() {
        for &kind in ModelKind::all() {
            let mut m = kind.build_default();
            assert!(
                !m.model().parts().is_empty(),
                "{kind:?} has no parts — is it still a stub?"
            );
            for (swing, amount, age) in [
                (0.0_f32, 0.0_f32, 0.0_f32),
                (12.5, 0.7, 40.0),
                (-3.0, 1.0, 1000.0),
            ] {
                let mut p = AnimParams::still();
                p.limb_swing = swing;
                p.limb_swing_amount = amount;
                p.age_in_ticks = age;
                p.net_head_yaw = 25.0;
                p.head_pitch = -15.0;
                let mut out = Vec::new();
                m.pose_and_render(&p, Mat4::IDENTITY, &mut out);
                assert!(
                    out.iter().all(|v| v.pos.iter().all(|c| c.is_finite())
                        && v.uv.iter().all(|c| c.is_finite())),
                    "{kind:?} emitted a non-finite vertex"
                );
            }
        }
    }

    /// A model that emits nothing at rest is almost certainly a stub or a mis-wired part tree.
    #[test]
    fn every_model_emits_geometry_at_rest() {
        for &kind in ModelKind::all() {
            let mut m = kind.build_default();
            let mut out = Vec::new();
            m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
            assert!(!out.is_empty(), "{kind:?} emitted no geometry");
            assert_eq!(out.len() % 6, 0, "{kind:?} emitted a partial quad");
        }
    }

    /// UVs must land inside the model's own texture. A UV outside `0..1` means the texture size
    /// on the part disagrees with the `ModelBase` texture size — the classic porting slip for
    /// the models that call `setTextureSize`.
    ///
    /// [`ModelKind::Minecart`] and [`ModelKind::Dragon`] are the genuine exceptions, and they are
    /// vanilla's, not ours. Dragon's membrane boxes intentionally use negative U offsets on its
    /// 256-pixel sheet, exactly as `ModelDragon` does; those faces are wrapped by the texture
    /// sampler in the original client.
    ///
    /// Minecart's exception is:
    /// `ModelMinecart`'s `sideModels[5]` is `new ModelRenderer(this, 44, 10)` with an 18-wide
    /// box, so its `+Y` and `+Z` faces reach `u = 81` and `u = 82` on a 64-pixel texture.
    /// `minecart.png` really is 64x32; GL_REPEAT wraps those two faces, and there is no
    /// integer shift that both fixes the overflow and keeps `u >= 0`. Reproduced rather than
    /// "corrected" — see `MinecartModel::new`.
    #[test]
    fn every_models_uvs_are_inside_its_texture() {
        for &kind in ModelKind::all() {
            if matches!(kind, ModelKind::Minecart | ModelKind::Dragon) {
                continue;
            }
            let mut m = kind.build_default();
            let mut out = Vec::new();
            m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
            for v in &out {
                assert!(
                    (-0.001..=1.001).contains(&v.uv[0]) && (-0.001..=1.001).contains(&v.uv[1]),
                    "{kind:?} has a uv outside the texture: {:?}",
                    v.uv
                );
            }
        }
    }

    /// ...and the minecart's overflow is exactly the two faces vanilla wraps, no more.
    #[test]
    fn the_minecarts_uv_overflow_is_the_documented_vanilla_one() {
        let mut m = ModelKind::Minecart.build_default();
        let mut out = Vec::new();
        m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
        let over: Vec<_> = out
            .iter()
            .filter(|v| v.uv[0] > 1.001 || v.uv[1] > 1.001 || v.uv[0] < -0.001 || v.uv[1] < -0.001)
            .collect();
        assert!(
            !over.is_empty(),
            "the vanilla overflow is expected to be present"
        );
        for v in &over {
            assert!(
                v.uv[0] > 1.0 && v.uv[0] < 1.3,
                "only the u axis overflows, and only just: {:?}",
                v.uv
            );
        }
    }

    /// `build(inflation)` is what a layer uses to draw a fatter (or thinner) copy of a body's
    /// own model — the charged-creeper aura ([`ModelKind::Creeper`] at `2.0`) and biped armour
    /// ([`ModelKind::Biped`] at `0.5`/`1.0`) are the two real vanilla users. The bounding box
    /// must grow with the inflation for the kinds whose Java constructor exposes it, and stay
    /// byte-for-byte identical for the many that do not (there is no `LayerZombieCharge`).
    #[test]
    fn build_with_inflation_grows_the_kinds_that_support_it() {
        let extent = |out: &[crate::entity::EntityVertex]| {
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in out {
                lo = lo.min(v.pos[0]);
                hi = hi.max(v.pos[0]);
            }
            hi - lo
        };
        let pose_extent = |kind: ModelKind, inflation: f32| {
            let mut m = kind.build(inflation);
            let mut out = Vec::new();
            m.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut out);
            extent(&out)
        };

        for kind in [
            ModelKind::Biped,
            ModelKind::Player,
            ModelKind::PlayerSlim,
            ModelKind::Skeleton,
            ModelKind::Villager,
            ModelKind::Witch,
            ModelKind::Pig,
            ModelKind::Creeper,
            ModelKind::Enderman,
            ModelKind::ArmorStandArmor,
        ] {
            let plain = pose_extent(kind, 0.0);
            let fat = pose_extent(kind, 1.0);
            assert!(fat > plain, "{kind:?}: {fat} should exceed {plain}");
        }

        // A kind with no inflatable constructor in vanilla ignores the argument rather than
        // panicking or silently producing different geometry.
        for kind in [
            ModelKind::Cow,
            ModelKind::Sheep,
            ModelKind::Slime,
            ModelKind::Spider,
        ] {
            assert_eq!(
                pose_extent(kind, 0.0),
                pose_extent(kind, 3.0),
                "{kind:?} has no inflation parameter in vanilla"
            );
        }
    }

    /// `build_default` is exactly `build(0.0)` — the shape every body cache in
    /// [`super::super::manager::RenderManager`] relies on.
    #[test]
    fn build_default_is_zero_inflation() {
        for &kind in ModelKind::all() {
            let mut a = kind.build_default();
            let mut b = kind.build(0.0);
            let mut oa = Vec::new();
            let mut ob = Vec::new();
            a.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut oa);
            b.pose_and_render(&AnimParams::still(), Mat4::IDENTITY, &mut ob);
            assert_eq!(oa, ob, "{kind:?}");
        }
    }

    /// `RenderWitherSkull` calls the no-argument `ModelSkeletonHead` constructor, while the
    /// tile renderer calls the explicitly-sized `(0, 0, 64, 32)` constructor. Keep the two
    /// cache keys distinct and assert the first UV in the divergent V region, rather than only
    /// checking that both models emit the same number of vertices.
    #[test]
    fn wither_skull_kind_keeps_the_no_arg_constructor_distinct_from_tile_skulls() {
        let tile = ModelKind::SkeletonHead.build_default();
        let wither = ModelKind::WitherSkull.build_default();
        let tile_part = tile
            .model()
            .find("skeleton_head")
            .map(|id| tile.model().get(id))
            .expect("tile skeleton head has its named part");
        let wither_part = wither
            .model()
            .find("skeleton_head")
            .map(|id| wither.model().get(id))
            .expect("wither skull has its named part");

        assert_eq!(
            (tile.model().texture_width, tile.model().texture_height),
            (64, 32)
        );
        assert_eq!(
            (wither.model().texture_width, wither.model().texture_height),
            (64, 64)
        );
        assert_eq!(tile_part.texture_offset, [0, 0]);
        assert_eq!(wither_part.texture_offset, [0, 35]);
        assert_eq!(
            tile_part.cubes[0].quads[4].verts[1].uv,
            [8.0 / 64.0, 8.0 / 32.0]
        );
        assert_eq!(
            wither_part.cubes[0].quads[4].verts[1].uv,
            [8.0 / 64.0, 43.0 / 64.0]
        );
        assert_ne!(
            tile_part.cubes[0].quads[4].verts[1].uv, wither_part.cubes[0].quads[4].verts[1].uv,
            "the tile and projectile heads must not share the tile UV layout"
        );
    }
}
