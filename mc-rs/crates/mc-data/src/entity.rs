//! Entity network-id tables.
//!
//! 1.8.9 has **two** disjoint id spaces on the wire and mixing them up is a classic
//! porting bug:
//!
//! * [`EntityType`] - `EntityList`'s mob ids, carried by `S0FPacketSpawnMob`.
//! * [`ObjectType`] - a hard-coded table for `S0EPacketSpawnObject`, which lives in
//!   `NetHandlerPlayClient.handleSpawnObject` and is **not** an `EntityList` id.
//!
//! Java inputs: `net/minecraft/entity/EntityList.java`,
//! `net/minecraft/client/network/NetHandlerPlayClient.java:handleSpawnObject`.

use crate::generated::entities::{ENTITY_BY_NET_ID, ENTITY_INFO};

pub use crate::generated::entities::EntityType;

/// Dimensions and registry data for an entity type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityInfo {
    /// `EntityList` network id.
    pub net_id: u8,
    /// `EntityList` save name, e.g. `Zombie` or `MinecartChest`.
    pub name: &'static str,
    /// Originating Java class.
    pub java_class: &'static str,
    /// True for `EntityLivingBase` subclasses.
    pub living: bool,
    /// Bounding-box width, from the entity's `setSize` call.
    pub width: f32,
    /// Bounding-box height.
    pub height: f32,
    /// Eye height above the entity's feet.
    pub eye_height: f32,
    /// Exact virtual `Entity.canBeCollidedWith()` result for this client-side entity type.
    ///
    /// This is intentionally separate from collision physics: vanilla uses it for interaction
    /// picking, while movement asks `getCollisionBoundingBox`/`getCollisionBox` instead.
    pub can_be_collided_with: bool,
    /// Exact virtual `Entity.getCollisionBorderSize()` used to expand a mouse-over pick box.
    pub collision_border_size: f32,
    /// Exact `Entity.renderDistanceWeight` left by the client-side spawn constructor.
    ///
    /// Vanilla's shared entity culler multiplies its box-derived range by this value. Arrows
    /// raise it to `10.0`; registered entities otherwise retain `Entity`'s `1.0` default.
    pub render_distance_weight: f64,
    /// Spawn-egg base colour, when the entity has an egg.
    pub egg_primary: Option<u32>,
    /// Spawn-egg spot colour.
    pub egg_secondary: Option<u32>,
}

impl EntityType {
    /// Dimensions and registry data. `O(1)`.
    #[inline]
    #[must_use]
    pub fn info(self) -> &'static EntityInfo {
        &ENTITY_INFO[self as usize]
    }

    /// `EntityList` network id.
    #[inline]
    #[must_use]
    pub fn net_id(self) -> u8 {
        self.info().net_id
    }

    /// `EntityList` save name.
    #[inline]
    #[must_use]
    pub fn name(self) -> &'static str {
        self.info().name
    }

    /// True for `EntityLivingBase` subclasses.
    #[inline]
    #[must_use]
    pub fn is_living(self) -> bool {
        self.info().living
    }
}

/// Maps a `S0FPacketSpawnMob` network id to its type. Frozen S01 API.
///
/// Returns `None` for ids vanilla never assigns; the caller should ignore the spawn rather
/// than disconnect, which is what vanilla does.
///
/// # Example
/// ```
/// use mc_data::entity_type;
/// assert_eq!(entity_type(54).map(|t| t.name()), Some("Zombie"));
/// assert!(entity_type(200).is_some());  // EnderCrystal
/// assert!(entity_type(3).is_none());
/// ```
#[inline]
#[must_use]
pub fn entity_type(net_id: u8) -> Option<EntityType> {
    ENTITY_BY_NET_ID[net_id as usize]
}

/// Looks up an entity type by its `EntityList` save name (`Zombie`, `MinecartChest`, ...).
#[must_use]
pub fn entity_by_name(name: &str) -> Option<EntityType> {
    ENTITY_BY_NET_ID
        .iter()
        .flatten()
        .copied()
        .find(|t| t.name() == name)
}

/// Every registered entity type, ascending by network id.
pub fn entity_types() -> impl Iterator<Item = EntityType> {
    ENTITY_BY_NET_ID.iter().flatten().copied()
}

/// The `S0EPacketSpawnObject` type ids.
///
/// Hard-coded in vanilla rather than registry-driven, so it is transcribed here.
// java: NetHandlerPlayClient.java handleSpawnObject
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ObjectType {
    /// `1` - boat.
    Boat = 1,
    /// `2` - dropped item stack.
    ItemStack = 2,
    /// `10` - minecart; the object data field selects the minecart variant.
    Minecart = 10,
    /// `50` - primed TNT.
    PrimedTnt = 50,
    /// `51` - ender crystal.
    EnderCrystal = 51,
    /// `60` - arrow; object data is the shooter's entity id plus one.
    Arrow = 60,
    /// `61` - snowball.
    Snowball = 61,
    /// `62` - thrown egg.
    ThrownEgg = 62,
    /// `63` - ghast fireball.
    LargeFireball = 63,
    /// `64` - blaze fireball.
    SmallFireball = 64,
    /// `65` - thrown ender pearl.
    ThrownEnderpearl = 65,
    /// `66` - wither skull.
    WitherSkull = 66,
    /// `70` - falling block; object data is a `Block.getStateId` value, see
    /// [`crate::BlockState::from_java_state_id`].
    FallingBlock = 70,
    /// `71` - item frame; object data is the facing.
    ItemFrame = 71,
    /// `72` - eye of ender.
    EyeOfEnder = 72,
    /// `73` - splash potion; object data is the potion damage value.
    ThrownPotion = 73,
    /// `75` - thrown bottle o' enchanting.
    ThrownExpBottle = 75,
    /// `76` - firework rocket.
    FireworkRocket = 76,
    /// `77` - leash knot.
    LeashKnot = 77,
    /// `78` - armour stand.
    ArmorStand = 78,
    /// `90` - fishing bobber; object data is the owning player's entity id.
    FishingBobber = 90,
}

/// Maps a `S0EPacketSpawnObject` type id to its object type.
///
/// # Example
/// ```
/// use mc_data::{object_type, ObjectType};
/// assert_eq!(object_type(70), Some(ObjectType::FallingBlock));
/// assert_eq!(object_type(3), None);
/// ```
#[must_use]
pub fn object_type(id: u8) -> Option<ObjectType> {
    Some(match id {
        1 => ObjectType::Boat,
        2 => ObjectType::ItemStack,
        10 => ObjectType::Minecart,
        50 => ObjectType::PrimedTnt,
        51 => ObjectType::EnderCrystal,
        60 => ObjectType::Arrow,
        61 => ObjectType::Snowball,
        62 => ObjectType::ThrownEgg,
        63 => ObjectType::LargeFireball,
        64 => ObjectType::SmallFireball,
        65 => ObjectType::ThrownEnderpearl,
        66 => ObjectType::WitherSkull,
        70 => ObjectType::FallingBlock,
        71 => ObjectType::ItemFrame,
        72 => ObjectType::EyeOfEnder,
        73 => ObjectType::ThrownPotion,
        75 => ObjectType::ThrownExpBottle,
        76 => ObjectType::FireworkRocket,
        77 => ObjectType::LeashKnot,
        78 => ObjectType::ArmorStand,
        90 => ObjectType::FishingBobber,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_net_id_resolves_without_panic() {
        for id in 0..=u8::MAX {
            if let Some(t) = entity_type(id) {
                assert_eq!(t.net_id(), id);
                assert!(t.info().height >= 0.0);
            }
        }
    }

    #[test]
    fn known_mobs() {
        let zombie = entity_type(54).expect("zombie");
        assert_eq!(zombie.name(), "Zombie");
        assert!(zombie.is_living());
        // 1.8 gave zombies a 1.95-block hitbox, not the player's 1.8.
        assert!(
            (zombie.info().height - 1.95).abs() < 1e-4,
            "{}",
            zombie.info().height
        );
        assert!(
            (zombie.info().width - 0.6).abs() < 1e-4,
            "{}",
            zombie.info().width
        );

        let item = entity_type(1).expect("item");
        assert_eq!(item.name(), "Item");
        assert!(!item.is_living());
    }

    /// The extractor must preserve both virtual pick APIs, rather than inferring either one
    /// from dimensions or `EntityLivingBase`.  These cases cover the ordinary inherited values,
    /// both exceptional border sizes, and the two classes whose collision answer differs from
    /// their nearby peers.
    ///
    /// java: `Entity.canBeCollidedWith` / `getCollisionBorderSize`, sampled from the client-side
    /// spawn constructors by `McDataExtractor.entitiesJson`.
    #[test]
    fn generated_pick_collision_columns_keep_vanillas_exceptions() {
        let all: Vec<_> = entity_types().collect();
        assert_eq!(
            all.len(),
            62,
            "every EntityList row has both generated columns"
        );
        assert!(all.iter().all(|kind| {
            let info = kind.info();
            info.collision_border_size.is_finite() && info.collision_border_size >= 0.0
        }));

        for (kind, collidable, border) in [
            // Ordinary `Entity` defaults must remain available for a non-living item and the
            // two vehicle classes a player can actually pick.
            (EntityType::Item, false, 0.1),
            (EntityType::Boat, true, 0.1),
            (EntityType::MinecartRideable, true, 0.1),
            (EntityType::Zombie, true, 0.1),
            // Projectiles normally cannot be picked, but fireballs enlarge their target by a
            // whole block even when their `canBeCollidedWith` override is false.
            (EntityType::Arrow, false, 0.1),
            (EntityType::Fireball, true, 1.0),
            (EntityType::SmallFireball, false, 1.0),
            (EntityType::WitherSkull, false, 1.0),
            // Hanging frames opt out of the inherited 0.1 expansion, while dragons opt out of
            // collision altogether despite being living.
            (EntityType::ItemFrame, true, 0.0),
            (EntityType::EnderDragon, false, 0.1),
        ] {
            let info = kind.info();
            assert_eq!(info.can_be_collided_with, collidable, "{kind:?}");
            assert_eq!(info.collision_border_size, border, "{kind:?}");
        }

        // This distribution is a compact all-row negative control: restoring either old
        // zero-filled generated column (or using one universal border) changes it immediately.
        assert_eq!(
            all.iter()
                .filter(|kind| kind.info().can_be_collided_with)
                .count(),
            47
        );
        assert_eq!(
            all.iter()
                .filter(|kind| kind.info().collision_border_size == 0.0)
                .count(),
            3
        );
        assert_eq!(
            all.iter()
                .filter(|kind| kind.info().collision_border_size == 1.0)
                .count(),
            3
        );
    }

    /// The value must remain an extracted table column. A class-name special case in the
    /// renderer would let stale extractor JSON silently erase the constructor-owned fact.
    #[test]
    fn generated_render_distance_weights_keep_arrows_special() {
        let all: Vec<_> = entity_types().collect();
        assert_eq!(
            all.len(),
            62,
            "every EntityList row has the generated weight"
        );
        assert!(all.iter().all(|kind| {
            let weight = kind.info().render_distance_weight;
            weight.is_finite() && weight > 0.0
        }));
        assert_eq!(EntityType::Arrow.info().render_distance_weight, 10.0);
        assert_eq!(EntityType::Zombie.info().render_distance_weight, 1.0);
        assert_eq!(EntityType::Item.info().render_distance_weight, 1.0);
        assert_eq!(
            all.iter()
                .filter(|kind| kind.info().render_distance_weight == 10.0)
                .count(),
            1,
            "Arrow is the sole 1.8.9 EntityList weight override",
        );
        assert_eq!(
            all.iter()
                .filter(|kind| kind.info().render_distance_weight == 1.0)
                .count(),
            61,
        );
    }

    /// `Entity.width`/`height` are whatever the constructor that ran last called `setSize` with,
    /// and for two entity types the client's spawn path picks a constructor that disagrees with
    /// the plain `(World)` one `EntityList.createEntityByID` uses. The extractor instantiates the
    /// client's, so these two rows must not read as the `EntityList` sizes.
    ///
    /// java:
    /// * `NetHandlerPlayClient.handleSpawnExperienceOrb:450` -> `new EntityXPOrb(w, x, y, z, xp)`
    ///   -> `EntityXPOrb:36` `setSize(0.5F, 0.5F)`. The `(World)` constructor (`EntityXPOrb:57`)
    ///   says `setSize(0.25F, 0.25F)` and is what a naive extraction picks up.
    /// * `NetHandlerPlayClient.handleSpawnObject:401` (object type 70) -> `new
    ///   EntityFallingBlock(w, x, y, z, state)` -> `EntityFallingBlock:45`
    ///   `setSize(0.98F, 0.98F)`. The `(World)` constructor (`:35`) calls `setSize` not at all, so
    ///   a naive extraction reports `Entity`'s `width = 0.6F; height = 1.8F` field initialisers
    ///   (`Entity.java:269-270`).
    ///
    /// `eye_height` is `Entity.getEyeHeight` = `height * 0.85F` (`Entity.java:2691`); neither
    /// class overrides it.
    #[test]
    fn the_two_entities_whose_client_spawn_constructor_is_not_the_entitylist_one() {
        let orb = entity_type(2).expect("xp orb");
        assert_eq!(orb.name(), "XPOrb");
        assert_eq!(orb.info().width, 0.5, "EntityXPOrb width");
        assert_eq!(orb.info().height, 0.5, "EntityXPOrb height");
        assert!((orb.info().eye_height - 0.5 * 0.85).abs() < 1e-6);

        let falling = entity_type(21).expect("falling block");
        assert_eq!(falling.name(), "FallingSand");
        assert_eq!(falling.info().width, 0.98, "EntityFallingBlock width");
        assert_eq!(falling.info().height, 0.98, "EntityFallingBlock height");
        assert!((falling.info().eye_height - 0.98 * 0.85).abs() < 1e-6);
    }

    #[test]
    fn name_lookup_round_trips() {
        for t in entity_types() {
            assert_eq!(entity_by_name(t.name()), Some(t));
        }
    }

    #[test]
    fn object_ids_are_disjoint_from_mob_ids_where_it_matters() {
        // Object id 70 is a falling block, but EntityList id 70 is nothing at all: the two
        // spaces are unrelated and must never be looked up through the same table.
        assert_eq!(object_type(70), Some(ObjectType::FallingBlock));
        assert!(entity_type(70).is_none());
    }
}
