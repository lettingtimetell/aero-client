//! 1.7.10 entity registry data: which mob and object ids exist, and the box, eye height and
//! `yOffset` a 1.7.10 client gives each entity it spawns.
//!
//! 1.7.10 positions an entity by `posY = box bottom + yOffset - ySize` (java17:
//! `Entity.setPosition`, `entity/Entity.java:256`), where 1.8.9 has no `yOffset` and `posY` is the
//! box bottom. [`EntityGeometry5::y_offset`] is the value each spawn constructor leaves; the
//! packet handlers that turn 1.7.10 positions into the client's box-bottom model read it.
//!
//! Sizes, eye heights and offsets were dumped from the real 1.7.10 classes (every `EntityList`
//! class built against a client world) and are checked against that dump by
//! `tests/v5_entity_oracle.rs`; the two client-only spawn constructors (XP orb, falling block),
//! the egg (no `EntityList` id), the fishing hook and players are read from source.

use mc_data::{entity_type, object_type, EntityType, ObjectType};

use crate::entity::EntityKind;

/// Whether an entity of this kind exists in 1.7.10: every 1.8.9 kind but the armor stand,
/// endermite, guardian and rabbit (java17: `EntityList` static init, `entity/EntityList.java`).
#[inline]
#[must_use]
pub fn exists_in_v5(kind: EntityKind) -> bool {
    !matches!(
        kind,
        EntityKind::Type(
            EntityType::ArmorStand
                | EntityType::Endermite
                | EntityType::Guardian
                | EntityType::Rabbit
        )
    )
}

/// The type an `S0FPacketSpawnMob` id creates in 1.7.10, `None` where 1.7.10's `EntityList`
/// has no class and the spawn is ignored (java17: `EntityList.createEntityByID` returns null).
///
/// The 57 ids 1.7.10 knows have the same class as in 1.8.9; 1.8.9 adds 7 (thrown egg: 1.7.10
/// creates eggs only through object type 62), 30, 67, 68 and 101.
#[inline]
#[must_use]
pub fn mob_type_v5(net_id: u8) -> Option<EntityType> {
    entity_type(net_id).filter(|&t| t != EntityType::ThrownEgg && exists_in_v5(EntityKind::Type(t)))
}

/// The object an `S0EPacketSpawnObject` type creates in 1.7.10: 1.8.9's table without the
/// armor stand (78), which `handleSpawnObject` does not handle (java17:
/// `NetHandlerPlayClient.handleSpawnObject`). Every other type and its data int mean the same,
/// except the falling block's data packing (`mc_data::v5::state_from_v5_falling_block`).
#[inline]
#[must_use]
pub fn object_type_v5(id: u8) -> Option<ObjectType> {
    object_type(id).filter(|&t| t != ObjectType::ArmorStand)
}

/// An entity's box size, eye height and `yOffset` as a 1.7.10 client constructs it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityGeometry5 {
    /// `Entity.width` after the spawn constructor.
    pub width: f32,
    /// `Entity.height` after the spawn constructor.
    pub height: f32,
    /// `getEyeHeight()` at that size: 0 for every non-living entity (java17:
    /// `Entity.getEyeHeight`, `entity/Entity.java:997-1000`), `height * 0.85` for living ones
    /// (wolves `* 0.8`), 1.82 for a remote player.
    pub eye_height: f32,
    /// `Entity.yOffset` after the spawn constructor.
    pub y_offset: f32,
}

const fn geometry(width: f32, height: f32, eye_height: f32, y_offset: f32) -> EntityGeometry5 {
    EntityGeometry5 {
        width,
        height,
        eye_height,
        y_offset,
    }
}

/// A non-living entity: eye height 0 (java17: `Entity.getEyeHeight`).
const fn object(width: f32, height: f32, y_offset: f32) -> EntityGeometry5 {
    geometry(width, height, 0.0, y_offset)
}

/// A living entity: `yOffset` 0 and eye height `height * 0.85` (java17:
/// `EntityLivingBase.getEyeHeight`, `entity/EntityLivingBase.java:1948-1951`).
const fn living(width: f32, height: f32) -> EntityGeometry5 {
    geometry(width, height, height * 0.85, 0.0)
}

/// A remote player, `EntityOtherPlayerMP`: 0.6 x 1.8, `yOffset` 0 and eye height 1.82 (java17:
/// `EntityOtherPlayerMP.<init>` / `getEyeHeight`, `client/entity/EntityOtherPlayerMP.java:26,
/// 151-154`).
pub const REMOTE_PLAYER_GEOMETRY_V5: EntityGeometry5 = geometry(0.6, 1.8, 1.82, 0.0);

/// The local player while standing: 0.6 x 1.8 with `yOffset` 1.62, so `posY` is the eye
/// position, and `getEyeHeight` 0.12 (java17: `EntityPlayer.<init>` / `preparePlayerToSpawn` /
/// `resetHeight` / `getEyeHeight`, `entity/player/EntityPlayer.java:121, 468, 842-849`).
/// Sleeping sets 0.2 x 0.2 with `yOffset` 0.2 and dying sets `yOffset` 0.1 (`:1232-1238`,
/// `:610`).
pub const LOCAL_PLAYER_GEOMETRY_V5: EntityGeometry5 = geometry(0.6, 1.8, 0.12, 1.62);

/// The 1.7.10 spawn geometry of an entity, `None` for the kinds 1.7.10 does not have.
///
/// Three rules change it after spawning, as in 1.8.9 but with 1.7.10's numbers: a slime or magma
/// cube is resized every client tick ([`slime_dimensions_v5`]); a wither skeleton grows once its
/// type byte is 1 ([`WITHER_SKELETON_DIMENSIONS_V5`]); an ageable mob is scaled every client tick
/// by [`ageable_scale_v5`] of its growing age (applied to these base sizes; a horse foal grows
/// with its age in 1.7.10). Every living entity's `yOffset` is reset to 0 by its first
/// interpolated move (java17: `EntityLivingBase.setPositionAndRotation2`), which only matters
/// for players.
#[must_use]
pub fn geometry_v5(kind: EntityKind) -> Option<EntityGeometry5> {
    use EntityType as E;
    let t = match kind {
        EntityKind::Player => return Some(REMOTE_PLAYER_GEOMETRY_V5),
        // java17: EntityFishHook.<init>(World, double, double, double, EntityPlayer) ->
        // setSize(0.25F, 0.25F), yOffset 0 (entity/projectile/EntityFishHook.java:92).
        EntityKind::FishingBobber => return Some(object(0.25, 0.25, 0.0)),
        EntityKind::Type(t) => t,
    };
    Some(match t {
        // java17: EntityItem.<init>(World, double, double, double): setSize(0.25F, 0.25F),
        // yOffset = height / 2 (entity/item/EntityItem.java:37).
        E::Item => object(0.25, 0.25, 0.125),
        // java17: the client spawn constructor EntityXPOrb.<init>(World, x, y, z, xp):
        // setSize(0.5F, 0.5F), yOffset = height / 2 (entity/item/EntityXPOrb.java:22-27).
        E::XPOrb => object(0.5, 0.5, 0.25),
        // java17: EntityThrowable.<init>: setSize(0.25F, 0.25F), yOffset 0. The egg has no
        // EntityList id in 1.7.10 but is spawned by object type 62.
        E::ThrownEgg | E::Snowball | E::ThrownEnderpearl | E::ThrownPotion => {
            object(0.25, 0.25, 0.0)
        }
        E::ThrownExpBottle | E::EyeOfEnderSignal | E::FireworksRocketEntity => {
            object(0.25, 0.25, 0.0)
        }
        // java17: EntityHanging.<init>: setSize(0.5F, 0.5F), yOffset 0; the box itself comes
        // from setDirection (paintings, frames) or the leash knot's own bounds.
        E::LeashKnot | E::Painting | E::ItemFrame => object(0.5, 0.5, 0.0),
        E::Arrow => object(0.5, 0.5, 0.0),
        E::Fireball => object(1.0, 1.0, 0.0),
        E::SmallFireball | E::WitherSkull => object(0.3125, 0.3125, 0.0),
        // java17: EntityTNTPrimed.<init>: 0.98 x 0.98, yOffset = height / 2.
        E::PrimedTnt => object(0.98, 0.98, 0.49),
        // java17: the client spawn constructor EntityFallingBlock.<init>(World, x, y, z, Block,
        // int): setSize(0.98F, 0.98F), yOffset = height / 2
        // (entity/item/EntityFallingBlock.java:45-56).
        E::FallingSand => object(0.98, 0.98, 0.49),
        // java17: EntityMinecart.<init>: 0.98 x 0.7, yOffset = height / 2.
        E::MinecartCommandBlock
        | E::MinecartRideable
        | E::MinecartChest
        | E::MinecartFurnace
        | E::MinecartTNT
        | E::MinecartHopper
        | E::MinecartSpawner => object(0.98, 0.7, 0.35),
        // java17: EntityBoat.<init>: 1.5 x 0.6, yOffset = height / 2.
        E::Boat => object(1.5, 0.6, 0.3),
        // java17: EntityEnderCrystal.<init>: 2 x 2, yOffset = height / 2.
        E::EnderCrystal => object(2.0, 2.0, 1.0),
        // `Mob` and `Monster` are abstract in both versions; they never spawn.
        E::Mob | E::Monster => return None,
        E::Creeper | E::Skeleton | E::Zombie | E::PigZombie | E::Blaze => living(0.6, 1.8),
        E::Witch | E::Villager => living(0.6, 1.8),
        E::Spider => living(1.4, 0.9),
        // java17: EntityGiantZombie.<init>: setSize(width * 6, height * 6) from 0.6 x 1.8.
        E::Giant => living(0.6 * 6.0, 1.8 * 6.0),
        // The constructor's random size is replaced on the first client tick:
        // slime_dimensions_v5. Size 1 here.
        E::Slime | E::LavaSlime => living(0.6, 0.6),
        E::Ghast => living(4.0, 4.0),
        E::Enderman => living(0.6, 2.9),
        E::CaveSpider => living(0.7, 0.5),
        E::Silverfish => living(0.3, 0.7),
        E::EnderDragon => living(16.0, 8.0),
        E::WitherBoss => living(0.9, 4.0),
        E::Bat => living(0.5, 0.9),
        E::Pig => living(0.9, 0.9),
        E::Sheep | E::Cow | E::MushroomCow => living(0.9, 1.3),
        E::Chicken => living(0.3, 0.7),
        E::Squid => living(0.95, 0.95),
        // java17: EntityWolf.getEyeHeight returns height * 0.8 (entity/passive/EntityWolf.java:266-269).
        E::Wolf => geometry(0.6, 0.8, 0.8 * 0.8, 0.0),
        E::SnowMan => living(0.4, 1.8),
        E::Ozelot => living(0.6, 0.8),
        E::VillagerGolem => living(1.4, 2.9),
        E::EntityHorse => living(1.4, 1.6),
        E::ArmorStand | E::Endermite | E::Guardian | E::Rabbit => return None,
    })
}

/// The `EntityAgeable` classes 1.7.10 has (their watched index 12 is the growing age).
pub(super) fn is_ageable(t: EntityType) -> bool {
    use EntityType as E;
    matches!(
        t,
        E::Pig
            | E::Sheep
            | E::Cow
            | E::Chicken
            | E::Wolf
            | E::MushroomCow
            | E::Ozelot
            | E::EntityHorse
            | E::Villager
    )
}

/// The seven `EntityMinecart` classes.
pub(super) fn is_minecart(t: EntityType) -> bool {
    use EntityType as E;
    matches!(
        t,
        E::MinecartRideable
            | E::MinecartChest
            | E::MinecartFurnace
            | E::MinecartTNT
            | E::MinecartHopper
            | E::MinecartSpawner
            | E::MinecartCommandBlock
    )
}

/// The scale a 1.7.10 client gives an ageable mob with growing age `age`, every client tick
/// (java17: `EntityAgeable.onLivingUpdate` calls `setScaleForAge(isChild())`, which resizes the
/// box to the base size times this, `entity/EntityAgeable.java:114-116, 140-160`): 0.5 for a
/// child and 1 for an adult, except a horse foal, which grows from 0.5 at age -24000 towards 1
/// (java17: `EntityHorse.getHorseSize`, `entity/passive/EntityHorse.java:205-221`; java18's
/// `getHorseSize` is a constant 0.5). 1 for a kind that is not ageable.
///
/// `age` is the full int ([`super::growing_age_v5`]), not the sign the 1.8.9 model stores: a
/// newborn foal is 0.7 x 0.8, where the sign alone would give the 1.8.9 0.5 for every foal.
#[must_use]
pub fn ageable_scale_v5(t: EntityType, age: i32) -> f32 {
    if !is_ageable(t) || age >= 0 {
        return 1.0;
    }
    if t == EntityType::EntityHorse {
        // `0.5F + (float)(-24000 - age) / -24000.0F * 0.5F`, int arithmetic first.
        0.5 + (-24000_i32).wrapping_sub(age) as f32 / -24000.0 * 0.5
    } else {
        0.5
    }
}

/// A slime's or magma cube's box for its watched size: `setSize(0.6F * size, 0.6F * size)`,
/// every client tick (java17: `EntitySlime.onUpdate`, `entity/monster/EntitySlime.java:124-128`;
/// 1.8.9 uses 0.51 and only on a watcher update).
#[inline]
#[must_use]
pub fn slime_dimensions_v5(size: i32) -> (f32, f32) {
    let edge = 0.6_f32 * size as f32;
    (edge, edge)
}

/// The box a 1.7.10 client gives a skeleton once its type byte (13) is 1, every client tick
/// (java17: `EntitySkeleton.onLivingUpdate`, `entity/monster/EntitySkeleton.java:156-158`;
/// 1.8.9: 0.72 x 2.535).
pub const WITHER_SKELETON_DIMENSIONS_V5: (f32, f32) = (0.72, 2.34);

/// `getCollisionBorderSize()`, the margin a mouse-over pick adds to an entity's box: 1.0 for
/// fireballs and wither skulls (java17: `EntityFireball`, `entity/projectile/EntityFireball.java:256-259`),
/// 0.1 for everything else, item frames included (1.8.9 gives frames 0).
#[inline]
#[must_use]
pub fn collision_border_v5(kind: EntityKind) -> f32 {
    match kind {
        EntityKind::Type(
            EntityType::Fireball | EntityType::SmallFireball | EntityType::WitherSkull,
        ) => 1.0,
        _ => 0.1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::entity_types;

    #[test]
    fn mob_ids_are_the_1_7_10_entity_list() {
        let ids: Vec<u8> = (0..=u8::MAX)
            .filter(|&id| mob_type_v5(id).is_some())
            .collect();
        assert_eq!(ids.len(), 57, "java17 EntityList registers 57 ids");
        for id in [7, 30, 67, 68, 101] {
            assert!(entity_type(id).is_some(), "1.8.9 knows {id}");
            assert_eq!(mob_type_v5(id), None, "1.7.10 does not know {id}");
        }
        assert_eq!(mob_type_v5(54), Some(EntityType::Zombie));
        assert_eq!(mob_type_v5(200), Some(EntityType::EnderCrystal));
    }

    #[test]
    fn object_types_drop_only_the_armor_stand() {
        for id in 0..=u8::MAX {
            let expected = object_type(id).filter(|&t| t != ObjectType::ArmorStand);
            assert_eq!(object_type_v5(id), expected);
        }
        assert!(object_type(78).is_some() && object_type_v5(78).is_none());
        assert_eq!(object_type_v5(62), Some(ObjectType::ThrownEgg));
    }

    #[test]
    fn every_1_7_10_kind_has_geometry_and_the_1_8_9_only_kinds_none() {
        for t in entity_types() {
            let kind = EntityKind::Type(t);
            let abstract_class = matches!(t, EntityType::Mob | EntityType::Monster);
            assert_eq!(
                geometry_v5(kind).is_some(),
                exists_in_v5(kind) && !abstract_class,
                "{t:?}"
            );
        }
        assert!(geometry_v5(EntityKind::Player).is_some());
        assert!(geometry_v5(EntityKind::FishingBobber).is_some());
    }

    #[test]
    fn the_rows_that_differ_from_1_8_9() {
        let g = |t| geometry_v5(EntityKind::Type(t)).unwrap();
        // The classes whose 1.7.10 constructor size differs from 1.8.9's (java17 `setSize` in
        // EntityZombie, inherited by EntityPigZombie, EntityChicken, EntityOcelot,
        // EntitySilverfish, EntitySnowman and EntityWither; EntityWitch keeps `Entity`'s
        // 0.6 x 1.8): the 1.7.10 value, then the 1.8.9 table as a negative control.
        for (t, w, h) in [
            (EntityType::Zombie, 0.6, 1.8),
            (EntityType::PigZombie, 0.6, 1.8),
            (EntityType::Witch, 0.6, 1.8),
            (EntityType::Chicken, 0.3, 0.7),
            (EntityType::Ozelot, 0.6, 0.8),
            (EntityType::Silverfish, 0.3, 0.7),
            (EntityType::SnowMan, 0.4, 1.8),
            (EntityType::WitherBoss, 0.9, 4.0),
        ] {
            assert_eq!((g(t).width, g(t).height), (w, h), "{t:?}");
            let info = t.info();
            assert_ne!(
                (info.width, info.height),
                (w, h),
                "{t:?} must differ from 1.8.9"
            );
        }
        // Rows that are the same in both versions.
        for t in [
            EntityType::Creeper,
            EntityType::Spider,
            EntityType::Boat,
            EntityType::Arrow,
        ] {
            assert_eq!(
                (g(t).width, g(t).height),
                (t.info().width, t.info().height),
                "{t:?}"
            );
        }
        // The client spawn constructors, not the EntityList ones.
        assert_eq!(
            (g(EntityType::XPOrb).width, g(EntityType::XPOrb).y_offset),
            (0.5, 0.25)
        );
        assert_eq!(g(EntityType::FallingSand).y_offset, 0.49);
    }

    #[test]
    fn y_offsets_are_half_height_exactly_for_the_vanilla_set() {
        // java17: the seven classes whose constructor sets `yOffset = height / 2.0F`.
        let half = [
            EntityType::Item,
            EntityType::XPOrb,
            EntityType::Boat,
            EntityType::MinecartRideable,
            EntityType::PrimedTnt,
            EntityType::FallingSand,
            EntityType::EnderCrystal,
        ];
        for t in entity_types() {
            let Some(g) = geometry_v5(EntityKind::Type(t)) else {
                continue;
            };
            let is_minecart = t.name().starts_with("Minecart");
            if half.contains(&t) || is_minecart {
                assert_eq!(g.y_offset, g.height / 2.0, "{t:?}");
            } else {
                assert_eq!(g.y_offset, 0.0, "{t:?}");
            }
        }
        assert_eq!(LOCAL_PLAYER_GEOMETRY_V5.y_offset, 1.62);
        assert_eq!(REMOTE_PLAYER_GEOMETRY_V5.y_offset, 0.0);
    }

    #[test]
    fn dynamic_sizes() {
        assert_eq!(slime_dimensions_v5(1), (0.6, 0.6));
        assert_eq!(slime_dimensions_v5(4), (2.4, 2.4));
        // 1.8.9's factor is 0.51: a negative control on the constant.
        assert_ne!(slime_dimensions_v5(2).0, 0.510_000_05_f32 * 2.0);
        assert_eq!(WITHER_SKELETON_DIMENSIONS_V5, (0.72, 2.34));
    }

    #[test]
    fn horse_foals_grow_with_their_age() {
        let horse = EntityType::EntityHorse;
        let base = geometry_v5(EntityKind::Type(horse)).unwrap();
        // A newborn (age -24000) is half size: 0.7 x 0.8.
        let newborn = ageable_scale_v5(horse, -24000);
        assert_eq!(newborn, 0.5);
        assert_eq!((base.width * newborn, base.height * newborn), (0.7, 0.8));
        assert_eq!(ageable_scale_v5(horse, -12000), 0.75);
        // One tick from grown: java17 `getHorseSize` in float arithmetic.
        let almost = ageable_scale_v5(horse, -1);
        assert_eq!(almost, 0.5 + (-23999_f32) / -24000.0 * 0.5);
        assert!(almost > 0.9999);
        assert_eq!(ageable_scale_v5(horse, 0), 1.0);
        assert_eq!(ageable_scale_v5(horse, 6000), 1.0);
        assert!(ageable_scale_v5(horse, i32::MIN).is_finite());
        // Negative control: 1.8.9 gives every foal 0.5 (java18 `getHorseSize`), and the sign byte
        // the 1.8.9 model stores cannot tell a newborn from a grown foal.
        assert_ne!(almost, 0.5);
        // Other ageables are halved as children whatever the age, as in 1.8.9.
        for t in [EntityType::Cow, EntityType::Wolf, EntityType::Villager] {
            assert_eq!(ageable_scale_v5(t, -24000), 0.5, "{t:?}");
            assert_eq!(ageable_scale_v5(t, -1), 0.5, "{t:?}");
            assert_eq!(ageable_scale_v5(t, 0), 1.0, "{t:?}");
        }
        // Not ageable: no scaling.
        assert_eq!(ageable_scale_v5(EntityType::Zombie, -1), 1.0);
    }

    #[test]
    fn pick_borders() {
        assert_eq!(
            collision_border_v5(EntityKind::Type(EntityType::ItemFrame)),
            0.1
        );
        assert_eq!(
            EntityType::ItemFrame.info().collision_border_size,
            0.0,
            "1.8.9 differs"
        );
        for t in [
            EntityType::Fireball,
            EntityType::SmallFireball,
            EntityType::WitherSkull,
        ] {
            assert_eq!(collision_border_v5(EntityKind::Type(t)), 1.0);
            assert_eq!(t.info().collision_border_size, 1.0);
        }
        assert_eq!(collision_border_v5(EntityKind::Player), 0.1);
    }
}
