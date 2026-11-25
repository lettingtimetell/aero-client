//! 1.7.10 `DataWatcher` registrations, and the remap of protocol-5 metadata onto the 1.8.9
//! indices the client stores and reads.
//!
//! The two versions register different indices for the same meanings, in their `entityInit`
//! chains:
//! - the custom name sits at 10/11 on `EntityLiving` in 1.7.10 (java17:
//!   `entity/EntityLiving.java:130-131`) and at 2/3 on `Entity` in 1.8.9 (java18:
//!   `entity/Entity.java:288-289`);
//! - the ageable age is an int in 1.7.10 and a byte in 1.8.9 (java17:
//!   `entity/EntityAgeable.java:71`; java18: `entity/EntityAgeable.java:78`). The byte keeps the
//!   sign, which is all the 1.8.9 code reads; 1.7.10 also sizes a horse foal from the full value,
//!   so [`growing_age_v5`] hands it to the caller;
//! - an enderman's carried block is two bytes (id, meta) in 1.7.10 and one packed short in 1.8.9
//!   (java17: `entity/monster/EntityEnderman.java:52-54`; java18: `:84-86`);
//! - an item frame's item and rotation move from 2/3 to 8/9 (java17: `EntityItemFrame.entityInit`,
//!   `entity/item/EntityItemFrame.java:32-33`; java18: `:35-36`), with 90-degree rotation steps
//!   becoming 45-degree ones;
//! - a player's cape switch moves from bit 1 of byte 16 (java17: `EntityPlayer.getHideCape`) to
//!   the skin-part byte 10 (java18: `EntityPlayer.isWearing`).
//!
//! Two indices keep their place but not their encoding:
//! - a wolf's collar (20) is a wool meta, `~dye & 15`, in 1.7.10 (java17:
//!   `EntityWolf.setCollarColor` with `BlockColored.func_150032_b`, read by `RenderWolf` as a
//!   fleece colour) and a dye damage in 1.8.9 (java18: `EntityWolf.getCollarColor` is
//!   `EnumDyeColor.byDyeDamage(b & 15)`);
//! - a minecart's display block (20) is `id & 0xFFFF | meta << 16` in 1.7.10 (java17:
//!   `EntityMinecart.getDisplayTile` / `getDisplayTileData`) and `Block.getStateId` (`id | meta <<
//!   12`) in 1.8.9 (java18: `EntityMinecart.getDisplayTile`).
//!
//! Everything else keeps its index and type.
//!
//! [`remap_metadata_v5`] takes a decoded protocol-5 list (with the protocol-5 values already in
//! the 1.8.9 value model, `MetadataEntry5::to_v47`) and returns the list with 1.8.9 indices, so
//! the existing metadata path stores and reads it unchanged. Entries are filtered first by the
//! 1.7.10 registrations: `DataWatcher.updateWatchedObjectsFromList` ignores an id the entity
//! does not register (java17: `entity/DataWatcher.java`), and an entry of the wrong type, which
//! would make 1.7.10 throw on its next typed read, is dropped as the 1.8.9 path drops it.

use mc_data::v5::{state_from_v5_enderman, state_from_v5_id_meta};
use mc_data::EntityType;
use mc_protocol::{MetadataEntry, MetadataValue};

use super::entity::{is_ageable, is_minecart};
use crate::entity::{EntityKind, Metadata, MetadataValue as Stored};

/// `MetadataValue::type_tag` values.
const BYTE: u8 = 0;
const SHORT: u8 = 1;
const INT: u8 = 2;
const FLOAT: u8 = 3;
const STRING: u8 = 4;
const ITEM_STACK: u8 = 5;

/// `filled_map`, the one item an item frame rotates in 90-degree steps in 1.8.9 too.
const FILLED_MAP: i16 = 358;

/// 1.8.9 `EnumPlayerModelParts` masks: the cape and the hat layer.
const SKIN_PART_CAPE: i8 = 0x01;
const SKIN_PART_HAT: i8 = 0x40;

/// The 1.8.9 skin-part byte (index 10) for a 1.7.10 player whose watched byte 16 is `byte16`.
///
/// 1.7.10 has no skin parts: bit 1 of byte 16 hides the cape (java17:
/// `EntityPlayer.getHideCape`, `entity/player/EntityPlayer.java:1421-1429`) and the hat layer is
/// always drawn (java17: `ModelBiped.render` renders `bipedHeadwear` unconditionally). The other
/// 1.8.9 layers do not exist in 1.7.10 and stay off.
#[inline]
#[must_use]
pub fn player_skin_parts_v5(byte16: i8) -> i8 {
    if byte16 & 0x02 != 0 {
        SKIN_PART_HAT
    } else {
        SKIN_PART_HAT | SKIN_PART_CAPE
    }
}

/// The skin-part byte a 1.7.10 player starts with (byte 16 registered as 0, java17:
/// `EntityPlayer.entityInit`). The local player's watcher is never sent whole, so a 1.7.10
/// session seeds the local player's index 10 with this at join and respawn.
pub const PLAYER_DEFAULT_SKIN_PARTS_V5: i8 = SKIN_PART_HAT | SKIN_PART_CAPE;

/// The raw 1.7.10 growing age a protocol-5 list carries for `kind`: the last index-12 int the
/// 1.7.10 registration accepts, `None` when there is none (or `kind` is not ageable).
///
/// [`remap_metadata_v5`] stores only its sign, which is all 1.8.9 code reads. A 1.7.10 client
/// also sizes a horse foal from the full value every tick (java17:
/// `EntityAgeable.onLivingUpdate` -> `EntityHorse.setScaleForAge` -> `getHorseSize`), so the
/// caller keeps this per entity and sizes ageables with [`super::ageable_scale_v5`]. The server
/// sends a foal's age as it grows, so the box grows with it.
#[must_use]
pub fn growing_age_v5(kind: EntityKind, incoming: &[MetadataEntry]) -> Option<i32> {
    incoming
        .iter()
        .rev()
        .filter(|entry| entry.index == 12 && registered_type_tag_v5(kind, 12) == Some(INT))
        .find_map(|entry| match entry.value {
            MetadataValue::Int(age) => Some(age),
            _ => None,
        })
}

/// The type tag (`MetadataValue::type_tag`) 1.7.10's `entityInit` chain registers at `index` for
/// `kind`, `None` when the index is not registered.
///
/// java17: `Entity.<init>` (0 byte flags, 1 short air), `EntityLivingBase.entityInit` (6 float
/// health, 7 int potion colour, 8 byte ambient, 9 byte arrows), `EntityLiving.entityInit` (10
/// string custom name, 11 byte always-show), `EntityAgeable.entityInit` (12 int age), and the
/// class-specific registrations; `EntityPlayer` extends `EntityLivingBase` directly and adds 16
/// byte, 17 float absorption, 18 int score (`entity/player/EntityPlayer.java:136-139`). Checked
/// against a dump of every class by `tests/v5_entity_oracle.rs`.
#[must_use]
pub fn registered_type_tag_v5(kind: EntityKind, index: u8) -> Option<u8> {
    match index {
        0 => return Some(BYTE),
        1 => return Some(SHORT),
        _ => {}
    }
    let t = match kind {
        EntityKind::Player => {
            return match index {
                6 | 17 => Some(FLOAT),
                7 | 18 => Some(INT),
                8 | 9 | 16 => Some(BYTE),
                _ => None,
            };
        }
        EntityKind::FishingBobber => return None,
        EntityKind::Type(t) => t,
    };
    if t.is_living() {
        match index {
            6 => return Some(FLOAT),
            7 => return Some(INT),
            8 | 9 | 11 => return Some(BYTE),
            10 => return Some(STRING),
            12 if is_ageable(t) => return Some(INT),
            _ => {}
        }
    }
    use EntityType as E;
    match t {
        E::Item => (index == 10).then_some(ITEM_STACK),
        E::Arrow => (index == 16).then_some(BYTE),
        E::WitherSkull => (index == 10).then_some(BYTE),
        E::ItemFrame => match index {
            2 => Some(ITEM_STACK),
            3 => Some(BYTE),
            _ => None,
        },
        E::FireworksRocketEntity => (index == 8).then_some(ITEM_STACK),
        E::EnderCrystal => (index == 8).then_some(INT),
        E::Creeper | E::Enderman => (16..=18).contains(&index).then_some(BYTE),
        E::Skeleton => (index == 13).then_some(BYTE),
        E::Zombie | E::PigZombie => (12..=14).contains(&index).then_some(BYTE),
        E::Spider | E::CaveSpider | E::Slime | E::LavaSlime | E::Ghast | E::Blaze => {
            (index == 16).then_some(BYTE)
        }
        E::Bat | E::VillagerGolem | E::Pig | E::Sheep => (index == 16).then_some(BYTE),
        E::WitherBoss => (17..=20).contains(&index).then_some(INT),
        E::Witch => (index == 21).then_some(BYTE),
        E::Wolf => match index {
            16 | 19 | 20 => Some(BYTE),
            17 => Some(STRING),
            18 => Some(FLOAT),
            _ => None,
        },
        E::Ozelot => match index {
            16 | 18 => Some(BYTE),
            17 => Some(STRING),
            _ => None,
        },
        E::Villager => (index == 16).then_some(INT),
        E::EntityHorse => match index {
            16 | 20 | 22 => Some(INT),
            19 => Some(BYTE),
            21 => Some(STRING),
            _ => None,
        },
        E::MinecartRideable
        | E::MinecartChest
        | E::MinecartFurnace
        | E::MinecartTNT
        | E::MinecartHopper
        | E::MinecartSpawner
        | E::MinecartCommandBlock => match index {
            16 if t == E::MinecartFurnace => Some(BYTE),
            17 | 18 | 20 | 21 => Some(INT),
            19 => Some(FLOAT),
            22 => Some(BYTE),
            23 | 24 if t == E::MinecartCommandBlock => Some(STRING),
            _ => None,
        },
        E::Boat => match index {
            17 | 18 => Some(INT),
            19 => Some(FLOAT),
            _ => None,
        },
        _ => None,
    }
}

/// Rewrites a protocol-5 metadata list for `kind` onto the 1.8.9 indices and types.
///
/// `current` is the entity's stored watcher (1.8.9 indices), needed where one 1.8.9 value is
/// built from two 1.7.10 ones that can arrive in separate packets (the enderman's carried block,
/// the item frame's rotation). Wire order is kept, and every accepted entry produces its own
/// output entry in place, so per-entry callbacks still run in Java's order. A derived value is
/// emitted after the entry it depends on, only when it changes.
///
/// The one 1.7.10 value the 1.8.9 model cannot hold is a growing age's magnitude: read it from
/// the same list with [`growing_age_v5`].
#[must_use]
pub fn remap_metadata_v5(
    kind: EntityKind,
    current: &Metadata,
    incoming: &[MetadataEntry],
) -> Vec<MetadataEntry> {
    let mut out = Vec::with_capacity(incoming.len() + 1);
    let t = kind.entity_type();
    let living_named = t.is_some_and(EntityType::is_living);
    let mut enderman = (t == Some(EntityType::Enderman)).then(|| Carried::from_stored(current));
    let mut frame = (t == Some(EntityType::ItemFrame)).then(|| Frame::from_stored(current));
    let mut skin_parts = match current.get(10) {
        Some(Stored::Byte(b)) => Some(*b),
        _ => None,
    };

    for entry in incoming {
        if registered_type_tag_v5(kind, entry.index) != Some(entry.value.type_tag()) {
            continue;
        }
        let value = &entry.value;
        if let Some(carried) = enderman.as_mut() {
            match (entry.index, value) {
                (16, MetadataValue::Byte(id)) => {
                    carried.id = *id;
                    out.push(carried.packed_entry());
                    continue;
                }
                (17, MetadataValue::Byte(meta)) => {
                    let before = carried.packed();
                    carried.meta = *meta;
                    out.push(entry.clone());
                    if carried.packed() != before {
                        out.push(carried.packed_entry());
                    }
                    continue;
                }
                _ => {}
            }
        }
        if let Some(frame) = frame.as_mut() {
            match (entry.index, value) {
                (2, MetadataValue::ItemStack(item)) => {
                    let before = frame.rotation_v18();
                    frame.is_map = item.as_ref().is_some_and(|s| s.item_id == FILLED_MAP);
                    out.push(MetadataEntry {
                        index: 8,
                        value: value.clone(),
                    });
                    if frame.rotation_v18() != before {
                        out.push(frame.rotation_entry());
                    }
                    continue;
                }
                (3, MetadataValue::Byte(rotation)) => {
                    frame.rotation = *rotation;
                    out.push(frame.rotation_entry());
                    continue;
                }
                _ => {}
            }
        }
        match (kind, entry.index, value) {
            (EntityKind::Player, 16, MetadataValue::Byte(byte16)) => {
                out.push(entry.clone());
                let parts = player_skin_parts_v5(*byte16);
                if skin_parts != Some(parts) {
                    skin_parts = Some(parts);
                    out.push(MetadataEntry {
                        index: 10,
                        value: MetadataValue::Byte(parts),
                    });
                }
            }
            (EntityKind::Type(_), 10 | 11, _) if living_named => out.push(MetadataEntry {
                index: entry.index - 8,
                value: value.clone(),
            }),
            (EntityKind::Type(_), 12, MetadataValue::Int(age)) => out.push(MetadataEntry {
                // 1.8.9's server sends `clamp(age, -1, 1)` and its client reads only the sign
                // (java18: `EntityAgeable.setGrowingAge` / `getGrowingAge`). The full value,
                // which sizes 1.7.10 horse foals, is `growing_age_v5`'s.
                index: 12,
                value: MetadataValue::Byte(age.signum() as i8),
            }),
            (EntityKind::Type(EntityType::Wolf), 20, MetadataValue::Byte(wool)) => {
                // Wool meta to dye damage: `EnumDyeColor` pairs them as `15 - x`.
                out.push(MetadataEntry {
                    index: 20,
                    value: MetadataValue::Byte(!wool & 15),
                });
            }
            (EntityKind::Type(t), 20, MetadataValue::Int(packed)) if is_minecart(t) => {
                // `id & 0xFFFF | meta << 16` to `Block.getStateId`, through the block decode
                // (an id 1.7.10 does not register is air, as `Block.getBlockById` makes it).
                let state = state_from_v5_id_meta(packed & 0xFFFF, packed >> 16);
                out.push(MetadataEntry {
                    index: 20,
                    value: MetadataValue::Int(state.to_java_state_id()),
                });
            }
            _ => out.push(entry.clone()),
        }
    }
    out
}

/// An enderman's carried block in 1.7.10 terms: the id byte (16) and the meta byte, which the
/// remap keeps at 1.8.9 index 17 (registered as a byte there and never read).
struct Carried {
    id: i8,
    meta: i8,
}

impl Carried {
    fn from_stored(current: &Metadata) -> Self {
        // The stored short is `Block.getStateId` of a decoded state: its low 12 bits are the
        // block id, which the decode keeps for every block 1.7.10 registers. A carried id is a
        // non-negative byte (a negative one decoded to air, id 0, which every meta keeps), so it
        // fits back into one.
        let id = match current.get(16) {
            Some(Stored::Short(packed)) => ((*packed as u16) & 0x0FFF) as u8 as i8,
            _ => 0,
        };
        let meta = match current.get(17) {
            Some(Stored::Byte(meta)) => *meta,
            _ => 0,
        };
        Self { id, meta }
    }

    fn packed(&self) -> i16 {
        state_from_v5_enderman(self.id, self.meta).to_java_state_id() as i16
    }

    fn packed_entry(&self) -> MetadataEntry {
        MetadataEntry {
            index: 16,
            value: MetadataValue::Short(self.packed()),
        }
    }
}

/// An item frame's rotation in 1.7.10 terms (90-degree steps), and whether it holds a map.
struct Frame {
    rotation: i8,
    is_map: bool,
}

impl Frame {
    fn from_stored(current: &Metadata) -> Self {
        let is_map = matches!(
            current.get(8),
            Some(Stored::ItemStack(Some(stack))) if stack.item_id == FILLED_MAP
        );
        let stored = match current.get(9) {
            Some(Stored::Byte(rotation)) => *rotation,
            _ => 0,
        };
        // Inverse of `rotation_v18` for every value it produces.
        let rotation = if is_map { stored } else { stored / 2 };
        Self { rotation, is_map }
    }

    /// The 1.8.9 rotation that draws like 1.7.10's.
    ///
    /// 1.7.10 turns the item by -90 degrees per step about an axis pointing at the viewer
    /// (java17: `RenderItemFrame.func_82402_b`, `glRotatef(-90 * getRotation(), 0, 0, 1)` after
    /// `glRotatef(180 + rotationYaw, 0, 1, 0)`); 1.8.9 by +45 degrees per step about an axis
    /// pointing away from the viewer, and by 90 degrees per `rotation % 4` for a map (java18:
    /// `RenderItemFrame.renderItem`, after `rotate(180 - rotationYaw, 0, 1, 0)`). Both turn
    /// clockwise, so step `r` is 1.8.9 rotation `2 * r`, or `r` for a map. 1.7.10 draws only
    /// `r & 3` distinct angles.
    fn rotation_v18(&self) -> i8 {
        let steps = self.rotation & 3;
        if self.is_map {
            steps
        } else {
            steps * 2
        }
    }

    fn rotation_entry(&self) -> MetadataEntry {
        MetadataEntry {
            index: 9,
            value: MetadataValue::Byte(self.rotation_v18()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::v5::state_from_v5;
    use mc_data::{entity_types, BlockId, BlockState};
    use mc_protocol::ItemStack;

    fn e(index: u8, value: MetadataValue) -> MetadataEntry {
        MetadataEntry { index, value }
    }

    fn kind(t: EntityType) -> EntityKind {
        EntityKind::Type(t)
    }

    /// Stores remapped entries the way the client's metadata path does (index -> value).
    fn store(metadata: &mut Metadata, entries: &[MetadataEntry]) {
        for entry in entries {
            let value = match &entry.value {
                MetadataValue::Byte(v) => Stored::Byte(*v),
                MetadataValue::Short(v) => Stored::Short(*v),
                MetadataValue::Int(v) => Stored::Int(*v),
                MetadataValue::Float(v) => Stored::Float(*v),
                MetadataValue::String(v) => Stored::Str(v.clone()),
                MetadataValue::ItemStack(v) => Stored::ItemStack(v.clone()),
                MetadataValue::BlockPos(p) => Stored::Pos(*p),
                MetadataValue::Rotations(x, y, z) => Stored::Rotations(*x, *y, *z),
            };
            metadata.set(entry.index, value);
        }
    }

    fn stack(item_id: i16) -> Option<ItemStack> {
        Some(ItemStack {
            item_id,
            count: 1,
            damage: 0,
            nbt: None,
        })
    }

    #[test]
    fn living_custom_name_moves_to_the_1_8_9_entity_slots() {
        let name = MetadataValue::String("Bob".into());
        let out = remap_metadata_v5(
            kind(EntityType::Zombie),
            &Metadata::new(),
            &[e(10, name.clone()), e(11, MetadataValue::Byte(1))],
        );
        assert_eq!(out, vec![e(2, name.clone()), e(3, MetadataValue::Byte(1))]);
        // Negative control: on an item, index 10 is the item stack and stays put; a string there
        // is the wrong type and is dropped.
        let item = MetadataValue::ItemStack(stack(1));
        let out = remap_metadata_v5(
            kind(EntityType::Item),
            &Metadata::new(),
            &[e(10, item.clone()), e(10, name.clone())],
        );
        assert_eq!(out, vec![e(10, item)]);
        // A player is not an EntityLiving: 10 and 11 are not registered for it.
        let out = remap_metadata_v5(EntityKind::Player, &Metadata::new(), &[e(10, name)]);
        assert!(out.is_empty());
    }

    #[test]
    fn ageable_int_age_becomes_the_1_8_9_sign_byte() {
        for (age, byte) in [(-24000, -1), (-1, -1), (0, 0), (6000, 1), (i32::MIN, -1)] {
            let out = remap_metadata_v5(
                kind(EntityType::Cow),
                &Metadata::new(),
                &[e(12, MetadataValue::Int(age))],
            );
            assert_eq!(out, vec![e(12, MetadataValue::Byte(byte))], "age {age}");
        }
        // A zombie's 12 is already a byte (child flag) in both versions; an int is dropped.
        let out = remap_metadata_v5(
            kind(EntityType::Zombie),
            &Metadata::new(),
            &[e(12, MetadataValue::Int(-1)), e(12, MetadataValue::Byte(1))],
        );
        assert_eq!(out, vec![e(12, MetadataValue::Byte(1))]);
    }

    #[test]
    fn enderman_carried_block_is_rebuilt_from_id_and_meta() {
        let k = kind(EntityType::Enderman);
        let packed = |b: u16, m: u8| {
            let state = state_from_v5((b << 4) | u16::from(m));
            MetadataValue::Short(state.to_java_state_id() as i16)
        };
        // Spawn list: both bytes and the scream flag.
        let spawn = [
            e(16, MetadataValue::Byte(98)),
            e(17, MetadataValue::Byte(3)),
            e(18, MetadataValue::Byte(0)),
        ];
        let out = remap_metadata_v5(k, &Metadata::new(), &spawn);
        assert_eq!(
            out,
            vec![
                e(16, packed(98, 0)),
                e(17, MetadataValue::Byte(3)),
                e(16, packed(98, 3)),
                e(18, MetadataValue::Byte(0)),
            ]
        );
        let mut stored = Metadata::new();
        store(&mut stored, &out);
        // A later packet with only the id keeps the stored meta.
        let out = remap_metadata_v5(k, &stored, &[e(16, MetadataValue::Byte(1))]);
        assert_eq!(out, vec![e(16, packed(1, 3))]);
        store(&mut stored, &out);
        // And one with only the meta keeps the stored id. Stone ignores its meta in 1.7.10, so
        // the packed state does not change and no derived entry follows.
        let out = remap_metadata_v5(k, &stored, &[e(17, MetadataValue::Byte(5))]);
        assert_eq!(out, vec![e(17, MetadataValue::Byte(5))]);
        // A non-vanilla meta goes through the block decode: a torch with meta 0 stands.
        let out = remap_metadata_v5(
            k,
            &Metadata::new(),
            &[
                e(16, MetadataValue::Byte(50)),
                e(17, MetadataValue::Byte(0)),
            ],
        );
        let torch = BlockState::new(BlockId(50), 5).to_java_state_id() as i16;
        assert_eq!(out[0], e(16, MetadataValue::Short(torch)));
        // The 1.8.9 short form must not be passed through.
        let out = remap_metadata_v5(k, &Metadata::new(), &[e(16, MetadataValue::Short(98))]);
        assert!(out.is_empty());
    }

    #[test]
    fn item_frame_moves_and_rotation_steps_double_except_for_maps() {
        let k = kind(EntityType::ItemFrame);
        let sword = MetadataValue::ItemStack(stack(276));
        let out = remap_metadata_v5(
            k,
            &Metadata::new(),
            &[e(2, sword.clone()), e(3, MetadataValue::Byte(1))],
        );
        assert_eq!(out, vec![e(8, sword), e(9, MetadataValue::Byte(2))]);
        let mut stored = Metadata::new();
        store(&mut stored, &out);
        // Swapping in a map keeps the 1.7.10 step: one 90-degree step is map rotation 1.
        let map = MetadataValue::ItemStack(stack(FILLED_MAP));
        let out = remap_metadata_v5(k, &stored, &[e(2, map.clone())]);
        assert_eq!(out, vec![e(8, map), e(9, MetadataValue::Byte(1))]);
        store(&mut stored, &out);
        // Rotating the map: step 3 is map rotation 3.
        let out = remap_metadata_v5(k, &stored, &[e(3, MetadataValue::Byte(3))]);
        assert_eq!(out, vec![e(9, MetadataValue::Byte(3))]);
        store(&mut stored, &out);
        // Emptying the frame keeps the step for the next item (2 * 3 = 6).
        let out = remap_metadata_v5(k, &stored, &[e(2, MetadataValue::ItemStack(None))]);
        assert_eq!(
            out,
            vec![
                e(8, MetadataValue::ItemStack(None)),
                e(9, MetadataValue::Byte(6))
            ]
        );
        // Index 2 is not the 1.8.9 custom-name slot for a frame: a string there is dropped.
        let out = remap_metadata_v5(k, &stored, &[e(2, MetadataValue::String("x".into()))]);
        assert!(out.is_empty());
    }

    #[test]
    fn player_cape_bit_becomes_skin_parts() {
        let out = remap_metadata_v5(
            EntityKind::Player,
            &Metadata::new(),
            &[
                e(16, MetadataValue::Byte(0)),
                e(17, MetadataValue::Float(2.0)),
            ],
        );
        assert_eq!(
            out,
            vec![
                e(16, MetadataValue::Byte(0)),
                e(10, MetadataValue::Byte(0x41)),
                e(17, MetadataValue::Float(2.0)),
            ]
        );
        let mut stored = Metadata::new();
        store(&mut stored, &out);
        // Hiding the cape (bit 1) clears only the cape part; an unchanged byte adds nothing.
        let out = remap_metadata_v5(
            EntityKind::Player,
            &stored,
            &[e(16, MetadataValue::Byte(2))],
        );
        assert_eq!(
            out,
            vec![
                e(16, MetadataValue::Byte(2)),
                e(10, MetadataValue::Byte(0x40))
            ]
        );
        store(&mut stored, &out);
        let out = remap_metadata_v5(
            EntityKind::Player,
            &stored,
            &[e(16, MetadataValue::Byte(2))],
        );
        assert_eq!(out, vec![e(16, MetadataValue::Byte(2))]);
        assert_eq!(player_skin_parts_v5(0), PLAYER_DEFAULT_SKIN_PARTS_V5);
        // Bit 0 is not the cape bit.
        assert_eq!(player_skin_parts_v5(1), PLAYER_DEFAULT_SKIN_PARTS_V5);
    }

    #[test]
    fn unregistered_and_wrong_typed_entries_are_dropped() {
        // 1.8.9-only indices: silent (4), no-AI (15), an armor-stand rotation.
        let out = remap_metadata_v5(
            kind(EntityType::Creeper),
            &Metadata::new(),
            &[
                e(4, MetadataValue::Byte(1)),
                e(15, MetadataValue::Byte(1)),
                e(16, MetadataValue::Rotations(0.0, 0.0, 0.0)),
                e(16, MetadataValue::Byte(1)),
            ],
        );
        assert_eq!(out, vec![e(16, MetadataValue::Byte(1))]);
        // Fishing hooks register only the Entity base.
        let out = remap_metadata_v5(
            EntityKind::FishingBobber,
            &Metadata::new(),
            &[
                e(0, MetadataValue::Byte(0)),
                e(6, MetadataValue::Float(1.0)),
            ],
        );
        assert_eq!(out, vec![e(0, MetadataValue::Byte(0))]);
    }

    #[test]
    fn unchanged_classes_pass_through() {
        let list = [
            e(0, MetadataValue::Byte(0x08)),
            e(1, MetadataValue::Short(300)),
            e(17, MetadataValue::Int(3)),
            e(18, MetadataValue::Int(1)),
            e(19, MetadataValue::Float(4.5)),
        ];
        let out = remap_metadata_v5(kind(EntityType::Boat), &Metadata::new(), &list);
        assert_eq!(out, list.to_vec());
    }

    #[test]
    fn registrations_cover_the_base_layout_for_every_kind() {
        for t in entity_types() {
            let k = kind(t);
            assert_eq!(registered_type_tag_v5(k, 0), Some(BYTE));
            assert_eq!(registered_type_tag_v5(k, 1), Some(SHORT));
            assert_eq!(
                registered_type_tag_v5(k, 6).is_some(),
                t.is_living(),
                "{t:?}"
            );
            // 1.8.9's Entity-level name/silent slots do not exist in 1.7.10, except where a
            // class registers 2/3 for itself (the item frame).
            assert_eq!(
                registered_type_tag_v5(k, 2).is_some(),
                t == EntityType::ItemFrame
            );
            assert_eq!(registered_type_tag_v5(k, 4), None, "{t:?}");
            assert_eq!(registered_type_tag_v5(k, 15), None, "{t:?}");
        }
    }

    #[test]
    fn the_raw_growing_age_survives_beside_the_sign_byte() {
        let horse = kind(EntityType::EntityHorse);
        let newborn = [e(12, MetadataValue::Int(-24000))];
        let grown = [e(12, MetadataValue::Int(-1))];
        // The 1.8.9 model stores the same byte for both...
        assert_eq!(
            remap_metadata_v5(horse, &Metadata::new(), &newborn),
            remap_metadata_v5(horse, &Metadata::new(), &grown)
        );
        // ...so the full value comes from the list itself.
        assert_eq!(growing_age_v5(horse, &newborn), Some(-24000));
        assert_eq!(growing_age_v5(horse, &grown), Some(-1));
        // The last accepted entry wins, as in the watcher.
        let two = [e(12, MetadataValue::Int(-5)), e(12, MetadataValue::Int(-4))];
        assert_eq!(growing_age_v5(horse, &two), Some(-4));
        // Wrong type, a list without index 12, or a kind whose 12 is not the age: none.
        assert_eq!(
            growing_age_v5(horse, &[e(12, MetadataValue::Byte(-1))]),
            None
        );
        assert_eq!(growing_age_v5(horse, &[e(16, MetadataValue::Int(4))]), None);
        let zombie = kind(EntityType::Zombie);
        assert_eq!(
            growing_age_v5(zombie, &[e(12, MetadataValue::Int(-1))]),
            None
        );
        assert_eq!(growing_age_v5(EntityKind::Player, &newborn), None);
    }

    #[test]
    fn wolf_collar_wool_meta_becomes_a_dye_damage() {
        // java17 stores `~dye & 15` (a wool meta, red = 14); java18 reads a dye damage (red = 1).
        for (wool, dye) in [(14, 1), (0, 15), (15, 0), (11, 4), (-2, 1)] {
            let out = remap_metadata_v5(
                kind(EntityType::Wolf),
                &Metadata::new(),
                &[e(20, MetadataValue::Byte(wool))],
            );
            assert_eq!(
                out,
                vec![e(20, MetadataValue::Byte(dye))],
                "wool meta {wool}"
            );
        }
        // Negative control: a horse's index 20 is an int variant and passes through.
        let variant = [e(20, MetadataValue::Int(14))];
        let out = remap_metadata_v5(kind(EntityType::EntityHorse), &Metadata::new(), &variant);
        assert_eq!(out, variant.to_vec());
    }

    #[test]
    fn minecart_display_block_is_repacked_through_the_block_decode() {
        let packed = |block: u16, meta: u16| {
            let state = mc_data::v5::state_from_v5((block << 4) | meta);
            MetadataValue::Int(state.to_java_state_id())
        };
        for t in [
            EntityType::MinecartRideable,
            EntityType::MinecartChest,
            EntityType::MinecartFurnace,
            EntityType::MinecartTNT,
            EntityType::MinecartHopper,
            EntityType::MinecartSpawner,
            EntityType::MinecartCommandBlock,
        ] {
            // java17 packs `id & 0xFFFF | meta << 16`; java18 reads `Block.getStateId`.
            for (wire, want) in [
                ((98 | (3 << 16)), packed(98, 3)),
                // Torch meta 0 has no 1.8.9 state: the decode makes it the standing torch.
                (50, packed(50, 5)),
                (0, MetadataValue::Int(0)),
                // An id 1.7.10 does not register is air.
                ((300 | (1 << 16)), MetadataValue::Int(0)),
            ] {
                let out = remap_metadata_v5(
                    kind(t),
                    &Metadata::new(),
                    &[e(20, MetadataValue::Int(wire))],
                );
                assert_eq!(out, vec![e(20, want.clone())], "{t:?} {wire:#x}");
            }
        }
        // The 1.8.9 packing of the same block is not what arrives: 98 | 3 << 12 means meta 0
        // (and bits 12-15 of a block id) in 1.7.10.
        let out = remap_metadata_v5(
            kind(EntityType::MinecartRideable),
            &Metadata::new(),
            &[e(20, MetadataValue::Int(98 | (3 << 12)))],
        );
        assert_ne!(out, vec![e(20, packed(98, 3))]);
    }
}
