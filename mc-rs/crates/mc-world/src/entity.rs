//! ECS components and the entity-arena bundle type.
//!
//! The arena-index rule: entities are `hecs` arena indices ([`crate::Entity`]). No component here, and
//! nothing built on top of [`crate::World`], may hold a `World` reference or wrap state behind
//! reference-counted interior mutability — behaviour is systems over `&mut World`, not methods
//! on the entity.
//!
//! java: `entity/Entity.java` (position/motion/rotation fields),
//! `entity/DataWatcher.java` (metadata)

use mc_data::EntityType;
use mc_math::{BlockPos, DVec3, Facing};

/// World-space position. java: `Entity.posX/posY/posZ`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Position(pub DVec3);

/// World-space velocity, in blocks/tick. java: `Entity.motionX/motionY/motionZ`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Velocity(pub DVec3);

/// Per-tick acceleration, in blocks/tick². java: `EntityFireball.accelerationX/Y/Z`
///
/// This is deliberately separate from [`Velocity`]: a spawn-object fireball starts with zero
/// motion even though its packet carries a non-zero *direction*. Its constructor normalizes that
/// direction into this field, and its tick later adds it to motion before drag.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Acceleration(pub DVec3);

/// The horizontal side an item frame is attached to. java: `EntityItemFrame.facingDirection`
/// from `NetHandlerPlayClient.handleSpawnObject` type 71.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ItemFrameFacing(pub Facing);

/// The raw damage/meta value of a spawned splash potion. java: `EntityPotion.potionDamage`.
///
/// It is an `i32` because `new ItemStack(Items.potionitem, 1, packet.data)` receives the wire
/// value directly, including zero and negative values, before any item consumer interprets it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplashPotionDamage(pub i32);

/// A resolved living entity that shot a spawned arrow, by network id.
///
/// `handleSpawnObject` only sets `EntityArrow.shootingEntity` when `data > 0` and the current
/// world lookup is an `EntityLivingBase`; a missing, nonliving, zero, or negative id has no
/// component. The raw data is not offset in 1.8.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArrowShooter(pub i32);

/// Look direction. java: `Entity.rotationYaw`/`rotationPitch`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Rotation {
    /// Yaw in degrees, vanilla convention (0 = south, increases clockwise looking down).
    pub yaw: f32,
    /// Pitch in degrees (-90 = straight up, 90 = straight down).
    pub pitch: f32,
}

/// Which mob/object this entity is, or that it is a remote player.
///
/// [`EntityKind::Type`] wraps `mc-data`'s registry type rather than redefining one — S01
/// already resolved the two disjoint network-id spaces (`EntityList` mobs vs.
/// `S0EPacketSpawnObject` objects), see `mc_data::entity`.
///
/// [`EntityKind::Player`] is the one entry this type adds on its own. `EntityList` never
/// registers the player class — vanilla constructs `EntityOtherPlayerMP` directly instead of
/// going through the registry (`NetHandlerPlayClient.handleSpawnPlayer`) — so there is no
/// `EntityType` to wrap. Before this variant existed, a remote player was spawned with an
/// entirely separate marker component (`mc_client::sync::RemotePlayer`) standing in for
/// `EntityKind`, which meant a caller after "every drawable entity" needed two disjoint ECS
/// queries: `query::<&EntityKind>()` for mobs/objects, `query::<&RemotePlayer>()` for players.
/// Folding `Player` into this enum instead means every tracked entity — mob, object, or player
/// — carries exactly one `EntityKind` component, and `query::<&EntityKind>()` alone finds all
/// of them. Closes the S14 -> S01 §13 cross-section request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityKind {
    /// An `EntityList`-registered mob or object.
    Type(EntityType),
    /// A remote player. See the type doc for why this is not an `EntityType`.
    Player,
    /// A client-side fishing hook. Java constructs `EntityFishHook` directly for S0E object
    /// type 90, so it has neither an `EntityList` id nor player identity.
    FishingBobber,
}

impl EntityKind {
    /// The `EntityList` type, if this names one rather than a player.
    #[inline]
    #[must_use]
    pub fn entity_type(self) -> Option<EntityType> {
        match self {
            EntityKind::Type(t) => Some(t),
            EntityKind::Player | EntityKind::FishingBobber => None,
        }
    }

    /// Whether vanilla's `EntityLivingBase` interpolation applies (as opposed to `Entity`'s
    /// immediate snap). Players are `EntityLivingBase`, so this is `true` for
    /// [`EntityKind::Player`] unconditionally, same as [`EntityType::is_living`].
    #[inline]
    #[must_use]
    pub fn is_living(self) -> bool {
        match self {
            EntityKind::Type(t) => t.is_living(),
            EntityKind::Player => true,
            EntityKind::FishingBobber => false,
        }
    }
}

/// One decoded `DataWatcher` (entity metadata) entry.
///
/// 1.8.9's `DataWatcher` wire format has eight value types (`DataWatcher.WatchableObject`,
/// type ids 0-7). The full *codec* — reading these off `PacketBuffer` — is `mc-protocol`'s
/// job (S06); this is only the *storage* shape so `mc-world` has somewhere to put decoded
/// values.
///
/// `ItemStack` (type 5) carries `mc_protocol::ItemStack` directly, matching
/// `mc_protocol::MetadataValue::ItemStack`'s own shape exactly. S08 originally left this arm as
/// still-encoded bytes (`ItemStackRaw(Vec<u8>)`) because `mc_protocol::ItemStack` did not exist
/// yet; S14 closed that placeholder (the S14 -> S08 §13 cross-section request) once it did, so
/// `mc-client`'s metadata conversion (`sync::entities::convert_metadata`) no longer has to
/// re-encode an already-decoded stack back to wire bytes just to fit this enum.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Type 0.
    Byte(i8),
    /// Type 1.
    Short(i16),
    /// Type 2.
    Int(i32),
    /// Type 3.
    Float(f32),
    /// Type 4.
    Str(String),
    /// Type 5. `None` is a legal watched value (e.g. an empty equipment slot) — matches
    /// `mc_protocol::MetadataValue::ItemStack`'s own `Option`, which in turn matches
    /// `ItemStack::read`'s "no item" case (the sentinel `-1` item id decodes to `None`, never
    /// to some empty-but-`Some` stack).
    ItemStack(Option<mc_protocol::ItemStack>),
    /// Type 6.
    Pos(BlockPos),
    /// Type 7 — (pitch, yaw, roll) in degrees.
    Rotations(f32, f32, f32),
}

/// An entity's full `DataWatcher` snapshot: index -> value.
///
/// Vanilla keys watched fields by a small `int` index (`DataWatcher.WatchableObject.getDataValueId`)
/// out of a couple dozen possible slots per entity type; a linear-scan `Vec` beats a hash map
/// at that size and preserves update order the way `DataWatcher`'s own change list does.
///
/// The private second field is the bag's change revision (see [`Self::revision`]). It is not
/// part of the value: equality compares the entries only.
#[derive(Debug, Clone, Default)]
pub struct Metadata(pub Vec<(u8, MetadataValue)>, u32);

impl PartialEq for Metadata {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl From<Vec<(u8, MetadataValue)>> for Metadata {
    fn from(entries: Vec<(u8, MetadataValue)>) -> Self {
        Self(entries, 0)
    }
}

/// Whether two watched values are the same, with floats compared by bits as Java's
/// `Float.equals` does (`0.0` and `-0.0` differ; a NaN equals the same NaN).
fn same_bits(a: &MetadataValue, b: &MetadataValue) -> bool {
    match (a, b) {
        (MetadataValue::Float(x), MetadataValue::Float(y)) => x.to_bits() == y.to_bits(),
        (MetadataValue::Rotations(ax, ay, az), MetadataValue::Rotations(bx, by, bz)) => {
            ax.to_bits() == bx.to_bits()
                && ay.to_bits() == by.to_bits()
                && az.to_bits() == bz.to_bits()
        }
        _ => a == b,
    }
}

impl Metadata {
    /// An empty metadata set.
    #[inline]
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new(), 0)
    }

    /// The value at `index`, if this entity has one watched there.
    #[must_use]
    pub fn get(&self, index: u8) -> Option<&MetadataValue> {
        self.0.iter().find(|(i, _)| *i == index).map(|(_, v)| v)
    }

    /// Sets (inserting or overwriting) the value at `index`.
    ///
    /// The value is always stored, as `DataWatcher.updateWatchedObjectsFromList` calls
    /// `setObject` unconditionally. [`Self::revision`] is bumped only when the stored entries
    /// actually change: a new index, or a value whose bits differ from the stored one. Floats are
    /// compared by bits, like Java's `Float.equals`, so `0.0` → `-0.0` is a change and is kept.
    /// Per-entry callbacks are the packet appliers' job, not this bag's.
    pub fn set(&mut self, index: u8, value: MetadataValue) {
        match self.0.iter_mut().find(|(i, _)| *i == index) {
            Some(slot) => {
                let changed = !same_bits(&slot.1, &value);
                slot.1 = value;
                if changed {
                    self.1 = self.1.wrapping_add(1);
                }
            }
            None => {
                self.0.push((index, value));
                self.1 = self.1.wrapping_add(1);
            }
        }
    }

    /// A change counter for this bag, so per-tick and per-frame consumers can skip re-deriving
    /// state from an unchanged bag.
    ///
    /// Every write goes through [`Self::set`], which bumps this exactly when the entries change,
    /// so for one bag an unchanged revision means unchanged entries. Revisions of *different*
    /// bags are unrelated (each starts at zero): key a cache on the owning ECS entity (a
    /// generation-unique handle) plus this value. Writing `self.0` directly bypasses the
    /// counter; nothing in the client does, and a caller that must should rebuild the bag
    /// through `set` instead.
    #[inline]
    #[must_use]
    pub fn revision(&self) -> u32 {
        self.1
    }
}

/// The component set for a newly spawned entity. java: `Entity`'s constructor plus its first
/// `readEntityFromNBT`/spawn-packet application.
///
/// No `Default` impl: [`EntityKind`] has no meaningful default variant — neither an arbitrary
/// `mc_data::EntityType` nor [`EntityKind::Player`] is any more "default" than another — so
/// every field must be supplied explicitly at the call site.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityBundle {
    /// Spawn position.
    pub position: Position,
    /// Spawn velocity.
    pub velocity: Velocity,
    /// Spawn look direction.
    pub rotation: Rotation,
    /// The entity's kind — an `EntityList` mob/object, or [`EntityKind::Player`]. Required
    /// rather than `Option`-wrapped so it stays a plain `EntityKind` component for `hecs`
    /// queries (`world.ecs.query::<&EntityKind>()`); a caller that gets an unrecognised network
    /// id from the wire should simply not call [`crate::World::spawn`] at all, which is what
    /// vanilla effectively does too (`entity_type` returns `None` and the spawn packet is
    /// ignored).
    pub kind: EntityKind,
    /// Initial metadata, if the spawn packet carried any.
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_get_set_overwrites_in_place() {
        let mut m = Metadata::new();
        m.set(0, MetadataValue::Byte(1));
        m.set(7, MetadataValue::Str("hi".into()));
        m.set(0, MetadataValue::Byte(2));
        assert_eq!(m.get(0), Some(&MetadataValue::Byte(2)));
        assert_eq!(m.get(7), Some(&MetadataValue::Str("hi".into())));
        assert_eq!(m.get(1), None);
        assert_eq!(m.0.len(), 2, "overwrite must not duplicate the slot");
    }

    #[test]
    fn metadata_revision_bumps_only_on_change_and_is_not_part_of_equality() {
        let mut m = Metadata::new();
        assert_eq!(m.revision(), 0);
        m.set(0, MetadataValue::Byte(1));
        assert_eq!(m.revision(), 1, "a new index is a change");
        m.set(0, MetadataValue::Byte(1));
        assert_eq!(m.revision(), 1, "rewriting an equal value is not a change");
        m.set(0, MetadataValue::Byte(2));
        m.set(7, MetadataValue::Str("hi".into()));
        assert_eq!(m.revision(), 3);
        let rebuilt = Metadata::from(vec![
            (0, MetadataValue::Byte(2)),
            (7, MetadataValue::Str("hi".into())),
        ]);
        assert_eq!(rebuilt.revision(), 0);
        assert_eq!(m, rebuilt, "equality compares entries, not revisions");
        assert_eq!(m.clone().revision(), m.revision());
    }

    /// java: `updateWatchedObjectsFromList` stores every value, and `Float.equals` tells `-0.0`
    /// from `0.0`, so a sign-only change is stored and counts as a change.
    #[test]
    fn metadata_set_stores_negative_zero_and_counts_it_as_a_change() {
        let mut m = Metadata::new();
        m.set(6, MetadataValue::Float(0.0));
        assert_eq!(m.revision(), 1);
        m.set(6, MetadataValue::Float(-0.0));
        assert_eq!(m.revision(), 2, "0.0 -> -0.0 is a change, as in Java");
        match m.get(6) {
            Some(MetadataValue::Float(v)) => assert_eq!(v.to_bits(), (-0.0_f32).to_bits()),
            other => panic!("expected the stored -0.0, got {other:?}"),
        }
        m.set(6, MetadataValue::Float(-0.0));
        assert_eq!(m.revision(), 2, "the same bits again are not a change");
        m.set(9, MetadataValue::Rotations(0.0, 1.0, 2.0));
        m.set(9, MetadataValue::Rotations(-0.0, 1.0, 2.0));
        assert_eq!(m.revision(), 4);
    }

    /// java: S14 -> S01 §13 cross-section request, closed. `Player` is not backed by any
    /// `EntityList` registration, so it must not resolve to a type and must still interpolate
    /// like every other `EntityLivingBase`.
    #[test]
    fn player_kind_has_no_entity_list_type_but_still_interpolates() {
        assert_eq!(EntityKind::Player.entity_type(), None);
        assert!(EntityKind::Player.is_living());
        assert_eq!(EntityKind::FishingBobber.entity_type(), None);
        assert!(!EntityKind::FishingBobber.is_living());
    }

    #[test]
    fn typed_kind_delegates_to_its_entity_type() {
        let zombie = mc_data::entity_type(54).expect("zombie");
        let kind = EntityKind::Type(zombie);
        assert_eq!(kind.entity_type(), Some(zombie));
        assert_eq!(kind.is_living(), zombie.is_living());

        let item = mc_data::entity_type(1).expect("item");
        assert!(
            !EntityKind::Type(item).is_living(),
            "dropped items do not interpolate"
        );
    }

    /// java: S14 -> S08 §13 cross-section request, closed. The stored value must equal the
    /// decoded `mc_protocol::ItemStack` directly — comparable by `==`, no byte blob to decode.
    #[test]
    fn metadata_carries_the_decoded_item_stack_directly_not_wire_bytes() {
        let stack = mc_protocol::ItemStack {
            item_id: 278,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let mut m = Metadata::new();
        m.set(6, MetadataValue::ItemStack(Some(stack.clone())));
        assert_eq!(m.get(6), Some(&MetadataValue::ItemStack(Some(stack))));
    }

    /// DataWatcher's type 5 permits a `null` value (an empty equipment slot); that must survive
    /// as `None`, not as some empty-but-present stack.
    #[test]
    fn an_empty_item_slot_is_a_data_watcher_none() {
        let mut m = Metadata::new();
        m.set(4, MetadataValue::ItemStack(None));
        assert_eq!(m.get(4), Some(&MetadataValue::ItemStack(None)));
    }
}
