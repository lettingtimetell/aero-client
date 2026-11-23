//! DataWatcher interpretation.
//!
//! Where `mc_world::entity::Metadata`'s raw indices become meaning — the shared `Entity` flag
//! byte, health, custom name, child, and the per-type indices — for the renderer and the HUD to
//! read. Before this existed, `Metadata::get` had zero non-test callers in the workspace: every
//! watched value was decoded off the wire, stored, and never interpreted.
//!
//! Its consumers include [`crate::app::appearance`] for visible variants and tints,
//! [`crate::sync::entities`] for ticked pose/state, and the live layer paths in
//! [`crate::app::driver`]. [`crate::app::entrender`] reads the charged Wither-skull accessor;
//! item-frame metadata is read directly there, while minecart display accessors drive the live
//! pre-shell cargo pass. The public surface uses vanilla concepts rather than index numbers because
//! one index means several unrelated things depending on the entity type — index 16 alone is the
//! sheep fleece byte, the creeper state, the villager profession, the horse flags and the slime
//! size, and every accessor here is gated on `EntityType` so they cannot bleed together.
//!
//! # How to read this module
//!
//! Every accessor is derived from the `entityInit()` of the Java class that *declares* the index,
//! and cites the accessor that reads it back. Nothing here guesses: an index whose vanilla owner
//! registers no meaning is simply absent.
//!
//! # Why every per-type accessor is gated on [`mc_data::EntityType`]
//!
//! Index 16 is `EntitySheep`'s fleece byte, `EntityCreeper`'s state byte, `EntityVillager`'s
//! profession `Integer`, `EntityHorse`'s flags `Integer`, `EntitySlime`'s size and
//! `EntityTameable`'s tame/sit bits — all at once, all different types. Vanilla resolves that by
//! the `instanceof` implicit in having an `EntitySheep` reference in hand; this module has only a
//! `Metadata` bag, so the entity type is carried alongside and every per-type accessor refuses to
//! answer for the wrong type. Without that gate, [`Watched::is_charged`] would be true for a
//! sheep whose fleece byte happens to be `1`.
//!
//! # What is deliberately *not* here
//!
//! - **TNT fuse, falling-block state, XP-orb value.** None of the three is a watched value —
//!   `EntityTNTPrimed.entityInit`, `EntityFallingBlock.entityInit` and `EntityXPOrb.entityInit`
//!   are all empty bodies. The fuse is pure client-side ticked state (`fuse = 80` at spawn,
//!   decremented in `onUpdate`), the falling block's state comes from `S0EPacketSpawnObject`'s
//!   `objectData`, and the orb's value comes from `S11PacketSpawnExperienceOrb.getXPValue()`.
//!   GAP A10-10/A10-22/A10-23 are spawn-packet gaps, not DataWatcher gaps.
//! - **Attributes** (`S20PacketEntityProperties`) — a separate packet.
//! - **Riding** (`S1BPacketEntityAttach`) — a separate packet. Note that index 0's
//!   bit 2 *is* vanilla's `isRiding` flag and [`EntityFlags::is_riding`] reads it, but the server
//!   does not keep it up to date for remote entities; the attach packet is authoritative.

use mc_data::EntityType;
use mc_world::entity::{Metadata, MetadataValue};

use super::entities::{EntityTracker, TrackedEntity, TrackedKind};

// ---------------------------------------------------------------------------------------------
// The shared flag byte
// ---------------------------------------------------------------------------------------------

/// Vanilla's index-0 flag byte, shared by every entity in the game.
///
/// java: `Entity.entityInit` (`:286`) registers it and `Entity.getFlag` (`:2207`) reads it as
/// `(byte0 & 1 << flag) != 0`. The bit meanings are spelled out in `getFlag`'s own javadoc —
/// "0) is burning; 1) is sneaking; 2) is riding something; 3) is sprinting; 4) is eating" — plus
/// bit 5, which `Entity.isInvisible` (`:2173`) reads and the javadoc forgets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EntityFlags(i8);

impl EntityFlags {
    /// java: `Entity.isBurning`'s client half (`:2129`) — `getFlag(0)`. The server half is a
    /// `fire > 0` counter a client never sees, so this bit is the *only* signal that an entity
    /// should be drawn with the fire overlay.
    #[must_use]
    pub fn is_burning(self) -> bool {
        self.get(0)
    }

    /// java: `Entity.isSneaking` (`:2146`) — `getFlag(1)`.
    #[must_use]
    pub fn is_sneaking(self) -> bool {
        self.get(1)
    }

    /// java: `getFlag(2)`, per `getFlag`'s javadoc. Unreliable for remote entities — vanilla's
    /// own client learns about riding from `S1BPacketEntityAttach`, not from this bit.
    #[must_use]
    pub fn is_riding(self) -> bool {
        self.get(2)
    }

    /// java: `Entity.isSprinting` (`:2158`) — `getFlag(3)`.
    #[must_use]
    pub fn is_sprinting(self) -> bool {
        self.get(3)
    }

    /// java: `Entity.isEating` (`:2193`) — `getFlag(4)`. Despite the name this is the generic
    /// "using an item" bit: eating, drinking, drawing a bow and blocking with a sword all set it.
    #[must_use]
    pub fn is_using_item(self) -> bool {
        self.get(4)
    }

    /// java: `Entity.isInvisible` (`:2173`) — `getFlag(5)`.
    #[must_use]
    pub fn is_invisible(self) -> bool {
        self.get(5)
    }

    /// The raw byte, for a consumer that wants to store or compare the whole set.
    #[must_use]
    pub fn bits(self) -> i8 {
        self.0
    }

    fn get(self, flag: u32) -> bool {
        self.0 & (1_i8 << flag) != 0
    }
}

// ---------------------------------------------------------------------------------------------
// Dye colours
// ---------------------------------------------------------------------------------------------

/// java: `item/EnumDyeColor.java`. Sixteen colours with **two** disjoint numbering schemes, which
/// is the whole reason this is a type rather than an `i32`: wool/sheep metadata counts white → 0,
/// black → 15, and dye *item damage* counts the other way (`dyeDamage = 15 - meta`).
/// `EntitySheep.getFleeceColor` reads its nibble with `byMetadata`; `EntityWolf.getCollarColor`
/// reads its nibble with `byDyeDamage`. Mixing them up recolours every collar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)] // Sixteen colour names; the enum's own doc says what they are.
pub enum DyeColour {
    White,
    Orange,
    Magenta,
    LightBlue,
    Yellow,
    Lime,
    Pink,
    Gray,
    Silver,
    Cyan,
    Purple,
    Blue,
    Brown,
    Green,
    Red,
    Black,
}

/// java: the declaration order of `EnumDyeColor`, which *is* the metadata order.
const DYE_BY_METADATA: [DyeColour; 16] = [
    DyeColour::White,
    DyeColour::Orange,
    DyeColour::Magenta,
    DyeColour::LightBlue,
    DyeColour::Yellow,
    DyeColour::Lime,
    DyeColour::Pink,
    DyeColour::Gray,
    DyeColour::Silver,
    DyeColour::Cyan,
    DyeColour::Purple,
    DyeColour::Blue,
    DyeColour::Brown,
    DyeColour::Green,
    DyeColour::Red,
    DyeColour::Black,
];

impl DyeColour {
    /// java: `EnumDyeColor.byMetadata` (`:75`) — out-of-range falls back to index 0, white.
    #[must_use]
    pub fn from_metadata(meta: i32) -> Self {
        match usize::try_from(meta) {
            Ok(i) if i < DYE_BY_METADATA.len() => DYE_BY_METADATA[i],
            _ => DyeColour::White,
        }
    }

    /// java: `EnumDyeColor.byDyeDamage` (`:65`). Every colour's `dyeDamage` is `15 - meta` (read
    /// straight off the enum constants: `WHITE(0, 15)` … `BLACK(15, 0)`), so the lookup table is
    /// the metadata table reversed. Out-of-range falls back to `DYE_DMG_LOOKUP[0]`, which is
    /// **black**, not white — the one place the two schemes' fallbacks differ.
    #[must_use]
    pub fn from_dye_damage(damage: i32) -> Self {
        match usize::try_from(damage) {
            Ok(i) if i < DYE_BY_METADATA.len() => DYE_BY_METADATA[DYE_BY_METADATA.len() - 1 - i],
            _ => DyeColour::Black,
        }
    }

    /// java: `EnumDyeColor.getMetadata`.
    #[must_use]
    pub fn metadata(self) -> i32 {
        self as i32
    }

    /// java: `EnumDyeColor.getDyeDamage` — `15 - metadata` for all sixteen constants.
    #[must_use]
    pub fn dye_damage(self) -> i32 {
        15 - self.metadata()
    }

    /// The tint `GlStateManager.color` is called with before drawing a fleece or a collar.
    ///
    /// java: `EntitySheep.DYE_TO_RGB` (`:372-387`), read by `EntitySheep.getDyeRgb` (`:58`) and
    /// used by both `LayerSheepWool` and `LayerWolfCollar` (`:27`). These are *not* the wool
    /// block's colours and not `MapColor`'s — they are a separate hand-authored table.
    #[must_use]
    pub fn fleece_rgb(self) -> [f32; 3] {
        match self {
            DyeColour::White => [1.0, 1.0, 1.0],
            DyeColour::Orange => [0.85, 0.5, 0.2],
            DyeColour::Magenta => [0.7, 0.3, 0.85],
            DyeColour::LightBlue => [0.4, 0.6, 0.85],
            DyeColour::Yellow => [0.9, 0.9, 0.2],
            DyeColour::Lime => [0.5, 0.8, 0.1],
            DyeColour::Pink => [0.95, 0.5, 0.65],
            DyeColour::Gray => [0.3, 0.3, 0.3],
            DyeColour::Silver => [0.6, 0.6, 0.6],
            DyeColour::Cyan => [0.3, 0.5, 0.6],
            DyeColour::Purple => [0.5, 0.25, 0.7],
            DyeColour::Blue => [0.2, 0.3, 0.7],
            DyeColour::Brown => [0.4, 0.3, 0.2],
            DyeColour::Green => [0.4, 0.5, 0.2],
            DyeColour::Red => [0.6, 0.2, 0.2],
            DyeColour::Black => [0.1, 0.1, 0.1],
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Small per-family value types
// ---------------------------------------------------------------------------------------------

/// java: `EnumPlayerModelParts` — the seven skin toggles a player sends in their client settings
/// and the server republishes at `EntityPlayer`'s index 10 (`EntityPlayer.entityInit`).
/// `RenderPlayer.setModelVisibilities` gates the cape, jacket, sleeve, trouser and hat parts on
/// these bits.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SkinParts(i8);

impl SkinParts {
    /// java: `EnumPlayerModelParts.CAPE(0)` — `1 << 0`.
    #[must_use]
    pub fn cape(self) -> bool {
        self.0 & 1 != 0
    }
    /// java: `EnumPlayerModelParts.JACKET(1)`.
    #[must_use]
    pub fn jacket(self) -> bool {
        self.0 & 2 != 0
    }
    /// java: `EnumPlayerModelParts.LEFT_SLEEVE(2)`.
    #[must_use]
    pub fn left_sleeve(self) -> bool {
        self.0 & 4 != 0
    }
    /// java: `EnumPlayerModelParts.RIGHT_SLEEVE(3)`.
    #[must_use]
    pub fn right_sleeve(self) -> bool {
        self.0 & 8 != 0
    }
    /// java: `EnumPlayerModelParts.LEFT_PANTS_LEG(4)`.
    #[must_use]
    pub fn left_trouser(self) -> bool {
        self.0 & 16 != 0
    }
    /// java: `EnumPlayerModelParts.RIGHT_PANTS_LEG(5)`.
    #[must_use]
    pub fn right_trouser(self) -> bool {
        self.0 & 32 != 0
    }
    /// java: `EnumPlayerModelParts.HAT(6)`.
    #[must_use]
    pub fn hat(self) -> bool {
        self.0 & 64 != 0
    }
    /// The raw bitmask.
    #[must_use]
    pub fn bits(self) -> i8 {
        self.0
    }
}

/// One `Rotations` triple, in **degrees** about X, Y and Z.
///
/// java: `util/Rotations.java` — `getX`/`getY`/`getZ`, which `ModelArmorStand` converts with
/// `* (PI / 180)` into `rotateAngleX/Y/Z`.
pub type Rotation3 = [f32; 3];

/// java: `EntityArmorStand`'s six `Rotations` slots, indices 11–16 (`:696-731`), with the
/// defaults its `entityInit` registers (`:28-33`). The arms and legs are *not* zeroed by default:
/// a freshly placed stand already has a slight asymmetric pose.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArmorStandPose {
    /// Index 11. java: `DEFAULT_HEAD_ROTATION = (0, 0, 0)`
    pub head: Rotation3,
    /// Index 12. java: `DEFAULT_BODY_ROTATION = (0, 0, 0)`
    pub body: Rotation3,
    /// Index 13. java: `DEFAULT_LEFTARM_ROTATION = (-10, 0, -10)`
    pub left_arm: Rotation3,
    /// Index 14. java: `DEFAULT_RIGHTARM_ROTATION = (-15, 0, 10)`
    pub right_arm: Rotation3,
    /// Index 15. java: `DEFAULT_LEFTLEG_ROTATION = (-1, 0, -1)`
    pub left_leg: Rotation3,
    /// Index 16. java: `DEFAULT_RIGHTLEG_ROTATION = (1, 0, 1)`
    pub right_leg: Rotation3,
}

impl Default for ArmorStandPose {
    fn default() -> Self {
        Self {
            head: [0.0, 0.0, 0.0],
            body: [0.0, 0.0, 0.0],
            left_arm: [-10.0, 0.0, -10.0],
            right_arm: [-15.0, 0.0, 10.0],
            left_leg: [-1.0, 0.0, -1.0],
            right_leg: [1.0, 0.0, 1.0],
        }
    }
}

/// java: `EntityArmorStand`'s index-10 flag byte (`:813-920`) plus its six pose slots.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ArmorStandState {
    /// java: `isSmall` (`:829`) — `& 1`. Halves the stand and its hitbox.
    pub small: bool,
    /// java: `hasNoGravity` (`:850`) — `& 2`. Server-side physics only; no render effect.
    pub no_gravity: bool,
    /// java: `getShowArms` (`:871`) — `& 4`. Arms are hidden entirely when clear.
    pub show_arms: bool,
    /// java: `hasNoBasePlate` (`:892`) — `& 8`.
    pub no_base_plate: bool,
    /// java: `hasMarker` (`:920`) — `& 16`. A marker stand has a zero-size hitbox and vanilla
    /// still draws it.
    pub marker: bool,
    /// The six limb rotations, indices 11–16.
    pub pose: ArmorStandPose,
}

/// java: `EntityMinecart` (`:999-1040`) and, identically, `EntityBoat` (`:571-611`) — the three
/// indices `RenderMinecart.doRender` turns into the damage rock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DamageWobble {
    /// Index 17, `Integer`. java: `getRollingAmplitude` — ticks since the last hit, counted down.
    pub time_since_hit: i32,
    /// Index 18, `Integer`, default **1**. java: `getRollingDirection` — the sign of the rock.
    pub rock_direction: i32,
    /// Index 19, `Float`. java: `getDamage` — scales the rock's amplitude.
    pub damage_taken: f32,
}

impl Default for DamageWobble {
    /// java: `entityInit` registers `(17, 0)`, `(18, 1)`, `(19, 0.0F)` — note 18 is **not** zero.
    fn default() -> Self {
        Self {
            time_since_hit: 0,
            rock_direction: 1,
            damage_taken: 0.0,
        }
    }
}

/// The two strings `EntityMinecartCommandBlock.entityInit` adds after the base minecart's
/// indices. `last_output_json` deliberately remains JSON text here: Java gives it to
/// `IChatComponent.Serializer.jsonToComponent` only when the command-block editor opens, and
/// that parser intentionally catches every failure and substitutes an empty component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandMinecartState<'a> {
    /// Index 23, `String`. java: `CommandBlockLogic.getCommand`.
    pub command: &'a str,
    /// Index 24, `String`, containing component JSON. java: `CommandBlockLogic.getLastOutput`.
    pub last_output_json: &'a str,
}

/// java: `EntityHorse`'s five registered indices (`entityInit`), which between them decide the
/// model, the texture, the saddle, the chest and the armour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HorseState {
    /// java: `isTame` (`:199`) — index 16 `& 2`.
    pub tame: bool,
    /// java: `isHorseSaddled` (`:589`) — index 16 `& 4`.
    pub saddled: bool,
    /// java: `isChested` (`:270`) — index 16 `& 8`.
    pub chested: bool,
    /// java: `isBreeding` (`:309`) — index 16 `& 16`.
    pub breeding: bool,
    /// java: `isEatingHaystack` (`:299`) — index 16 `& 32`.
    pub eating_haystack: bool,
    /// java: `isRearing` (`:304`) — index 16 `& 64`.
    pub rearing: bool,
    /// java: `onUpdate`'s `getHorseWatchableBoolean(128)` (`:1197`), which drives
    /// `mouthOpenness`.
    pub mouth_open: bool,
    /// Index 19. java: `getHorseType` (`:127`) — 0 horse, 1 donkey, 2 mule, 3 zombie horse,
    /// 4 skeleton horse.
    pub horse_type: i32,
    /// Index 20. java: `getHorseVariant` (`:138`) — colour in the low byte, markings in the high.
    pub variant: i32,
    /// Index 22. java: `getHorseArmorIndexSynced` (`:278`) — 0 none, 1 iron, 2 gold,
    /// 3 diamond. This is the renderer's armour *index*, not an item id.
    pub armour: i32,
}

// ---------------------------------------------------------------------------------------------
// The view
// ---------------------------------------------------------------------------------------------

/// A read-only, interpreting view over one entity's watched values.
///
/// Construct it from the entity's kind and its `Metadata` component and ask it vanilla questions.
/// Nothing is cached: every accessor is a linear scan of a list that is at most a couple of dozen
/// entries long, which is the same cost `DataWatcher.getWatchableObjectByte` pays.
#[derive(Debug, Clone, Copy)]
pub struct Watched<'a> {
    kind: TrackedKind,
    metadata: &'a Metadata,
}

impl<'a> Watched<'a> {
    /// A view over `metadata`, interpreted as belonging to an entity of `kind`.
    #[must_use]
    pub fn new(kind: TrackedKind, metadata: &'a Metadata) -> Self {
        Self { kind, metadata }
    }

    /// What this entity is.
    #[must_use]
    pub fn kind(self) -> TrackedKind {
        self.kind
    }

    /// The `EntityList` type, or `None` for a direct-constructed player/fishing bobber.
    #[must_use]
    pub fn entity_type(self) -> Option<EntityType> {
        self.kind.entity_type()
    }

    /// The raw bag, for a consumer that needs an index this module does not model.
    #[must_use]
    pub fn raw(self) -> &'a Metadata {
        self.metadata
    }

    /// Whether `index` was registered by this entity's Java `entityInit` chain.
    ///
    /// java: `DataWatcher.updateWatchedObjectsFromList` (`:365-378`) first looks the incoming
    /// id up in the owner's watcher map and silently skips ids that are not present.  The wire
    /// decoder quite deliberately accepts every legal metadata type, so the entity-specific
    /// registration table has to live beside the type-gated readers rather than in the codec.
    /// Keeping this predicate here also prevents an unrelated index 16 from becoming a slime,
    /// guardian, or player state when a server sends a hostile extra entry.
    #[must_use]
    pub(crate) fn is_registered(self, index: u8) -> bool {
        self.registered_type_tag(index).is_some()
    }

    /// The Java `DataWatcher` type registered for `index`, as a protocol metadata tag.
    ///
    /// `DataWatcher.dataTypes` uses the same eight tags as `mc_protocol::MetadataValue`. The
    /// packet codec can therefore prove only that an incoming value is one of those eight legal
    /// wire shapes; it cannot know which shape the receiving entity's `entityInit` registered.
    /// Keep that second, entity-specific check here so every S1C merge and every typed callback
    /// shares one table. `None` means the index is not present in this entity's watcher map.
    #[must_use]
    pub(crate) fn registered_type_tag(self, index: u8) -> Option<u8> {
        if matches!(index, 0..=4) {
            return match index {
                0 | 3 | 4 => Some(0), // byte
                1 => Some(1),         // short
                2 => Some(4),         // string
                _ => None,
            };
        }

        match self.kind {
            // EntityPlayer extends EntityLivingBase directly: it gets 6..=9, but not
            // EntityLiving's index 15. Its own four registrations are 10, 16, 17 and 18.
            TrackedKind::Player => match index {
                6 => Some(3),               // float
                7 => Some(2),               // int
                8 | 9 | 10 | 16 => Some(0), // byte
                17 => Some(3),              // float
                18 => Some(2),              // int
                _ => None,
            },
            // EntityFishHook extends Entity directly and adds no watcher values of its own.
            TrackedKind::FishingBobber => None,
            TrackedKind::Type(entity_type) => {
                let living = entity_type.is_living();
                // EntityLivingBase's four values are inherited by every living EntityList row.
                if living && matches!(index, 6..=9) {
                    return match index {
                        6 => Some(3),     // float
                        7 => Some(2),     // int
                        8 | 9 => Some(0), // byte
                        _ => None,
                    };
                }
                // EntityLiving adds NoAI at 15. ArmorStand is the only direct
                // EntityLivingBase subclass in the registry; EnderDragon extends
                // EntityLiving and therefore inherits this slot through EntityDragon's
                // `super.entityInit()` call.
                if living && entity_type != EntityType::ArmorStand && index == 15 {
                    return Some(0); // byte
                }
                // EntityAgeable's index 12 is inherited by the animal/villager family.
                if is_ageable(entity_type) && index == 12 {
                    return Some(0); // byte
                }

                use EntityType as E;
                match entity_type {
                    E::Item => (index == 10).then_some(5),        // item stack
                    E::Arrow => (index == 16).then_some(0),       // byte
                    E::WitherSkull => (index == 10).then_some(0), // byte
                    E::ItemFrame => match index {
                        8 => Some(5), // item stack
                        9 => Some(0), // byte
                        _ => None,
                    },
                    E::FireworksRocketEntity => (index == 8).then_some(5), // item stack
                    E::EnderCrystal => (index == 8).then_some(2),          // int
                    E::ArmorStand => match index {
                        10 => Some(0),      // byte
                        11..=16 => Some(7), // rotations
                        _ => None,
                    },
                    E::Creeper => (16..=18).contains(&index).then_some(0), // byte
                    E::Skeleton => (index == 13).then_some(0),             // byte
                    E::Spider | E::CaveSpider => (index == 16).then_some(0), // byte
                    E::Slime | E::LavaSlime => (index == 16).then_some(0), // byte
                    E::Ghast | E::Blaze | E::Bat | E::VillagerGolem => {
                        (index == 16).then_some(0) // byte
                    }
                    E::Zombie | E::PigZombie => (12..=14).contains(&index).then_some(0), // byte
                    E::Enderman => match index {
                        16 => Some(1),      // short
                        17 | 18 => Some(0), // byte
                        _ => None,
                    },
                    E::WitherBoss => (17..=20).contains(&index).then_some(2), // int
                    E::Guardian => (16..=17).contains(&index).then_some(2),   // int
                    E::Witch => (index == 21).then_some(0),                   // byte
                    E::Wolf => match index {
                        16 => Some(0),      // byte (tameable flags)
                        17 => Some(4),      // string (tameable owner)
                        18 => Some(3),      // float
                        19 | 20 => Some(0), // byte
                        _ => None,
                    },
                    E::Ozelot => match index {
                        16 => Some(0), // byte (tameable flags)
                        17 => Some(4), // string (tameable owner)
                        18 => Some(0), // byte
                        _ => None,
                    },
                    E::Rabbit => (index == 18).then_some(0), // byte
                    E::Pig | E::Sheep => (index == 16).then_some(0), // byte
                    E::Villager => (index == 16).then_some(2), // int
                    E::EntityHorse => match index {
                        16 => Some(2), // int
                        19 => Some(0), // byte
                        20 => Some(2), // int
                        21 => Some(4), // string
                        22 => Some(2), // int
                        _ => None,
                    },
                    E::MinecartRideable
                    | E::MinecartChest
                    | E::MinecartFurnace
                    | E::MinecartTNT
                    | E::MinecartHopper
                    | E::MinecartSpawner
                    | E::MinecartCommandBlock => match index {
                        16 if entity_type == E::MinecartFurnace => Some(0), // byte
                        17 | 18 | 20 | 21 => Some(2),                       // int
                        19 => Some(3),                                      // float
                        22 => Some(0),                                      // byte
                        23 | 24 if entity_type == E::MinecartCommandBlock => Some(4), // string
                        _ => None,
                    },
                    E::Boat => match index {
                        17 | 18 => Some(2), // int
                        19 => Some(3),      // float
                        _ => None,
                    },
                    // The remaining EntityList rows either have only the shared values above or
                    // are projectile subclasses whose `entityInit` is empty.
                    _ => None,
                }
            }
        }
    }

    /// Whether an incoming protocol value has both a registered index and that index's Java
    /// watcher type. The wire decoder accepts all eight legal types, but only one is valid for a
    /// particular watcher.
    #[must_use]
    pub(crate) fn accepts_registered_type(
        self,
        index: u8,
        value: &mc_protocol::MetadataValue,
    ) -> bool {
        self.is_registered(index) && self.registered_type_tag(index) == Some(value.type_tag())
    }

    // ---- shared `Entity` indices 0..=4 ---------------------------------------------------

    /// java: `Entity.getFlag` — index 0. Absent metadata reads as all-clear, which is what
    /// `entityInit`'s `addObject(0, (byte)0)` default means anyway.
    #[must_use]
    pub fn flags(self) -> EntityFlags {
        EntityFlags(self.byte(0).unwrap_or(0))
    }

    /// java: `Entity.isBurning`'s watched half — [`EntityFlags::is_burning`].
    #[must_use]
    pub fn is_burning(self) -> bool {
        self.flags().is_burning()
    }

    /// java: `Entity.isSneaking` — [`EntityFlags::is_sneaking`].
    #[must_use]
    pub fn is_sneaking(self) -> bool {
        self.flags().is_sneaking()
    }

    /// java: `Entity.isSprinting` — [`EntityFlags::is_sprinting`].
    #[must_use]
    pub fn is_sprinting(self) -> bool {
        self.flags().is_sprinting()
    }

    /// java: `Entity.isInvisible` — [`EntityFlags::is_invisible`].
    #[must_use]
    pub fn is_invisible(self) -> bool {
        self.flags().is_invisible()
    }

    /// java: `Entity.isEating` — [`EntityFlags::is_using_item`].
    #[must_use]
    pub fn is_using_item(self) -> bool {
        self.flags().is_using_item()
    }

    /// java: `Entity.getAir` (`:2231`) — index 1, a `Short`. `entityInit` (`:287`) defaults it to
    /// 300, which is what an entity that has never sent one has.
    #[must_use]
    pub fn air(self) -> i16 {
        self.short(1).unwrap_or(300)
    }

    /// java: `Entity.getCustomNameTag` (`:2616`) — index 2. `Entity.hasCustomName` (`:2624`) is
    /// `length() > 0`, so an empty string is *not* a name and this returns `None` for it.
    #[must_use]
    pub fn custom_name(self) -> Option<&'a str> {
        self.string(2).filter(|s| !s.is_empty())
    }

    /// java: `Entity.getAlwaysRenderNameTag` (`:2634`) — index 3, `== 1` (not `!= 0`).
    #[must_use]
    pub fn custom_name_visible(self) -> bool {
        self.byte(3) == Some(1)
    }

    /// java: `Entity.isSilent` (`:1014`) — index 4, `== 1`.
    #[must_use]
    pub fn is_silent(self) -> bool {
        self.byte(4) == Some(1)
    }

    /// The label `RendererLivingEntity.renderName` would draw over this entity, or `None`.
    ///
    /// java: ordinary living renderers reach `RendererLivingEntity.canRenderName` (`:547`), which
    /// refuses an entity invisible to the local player. `ArmorStandRenderer.canRenderName` (`:53`)
    /// overrides that path: it consults only `Entity.getAlwaysRenderNameTag` (index 3), so an
    /// invisible armour stand's forced custom-name hologram remains visible. Its small and marker
    /// flags do not take part in this decision.
    ///
    /// The caller additionally applies vanilla's pointed-entity fallback for ordinary living
    /// entities. Players are handled separately: `EntityPlayer.getAlwaysRenderNameTagForRender`
    /// (`:2163`) is hard-coded `true` and the text comes from the tab list, not from index 2.
    #[must_use]
    pub fn nameplate_text(self) -> Option<&'a str> {
        if !self.custom_name_visible()
            || (self.is_invisible() && self.entity_type() != Some(EntityType::ArmorStand))
        {
            return None;
        }
        self.custom_name()
    }

    // ---- `EntityLivingBase` indices 6..=9, `EntityLiving` 15 -----------------------------

    /// java: `EntityLivingBase.getHealth` (`:852`) — index 6, a `Float`. `None` when the server
    /// has not sent one, which is not the same as zero: a missing health value must never be
    /// read as "dead".
    #[must_use]
    pub fn health(self) -> Option<f32> {
        self.float(6)
    }

    /// java: `EntityLivingBase.updatePotionEffects` (`:642`) publishes the blended swirl colour
    /// to index 7 as an `Integer` `0xRRGGBB`; zero means no effects. This is the whole reason a
    /// client does not need remote `activePotionsMap` to draw swirls.
    #[must_use]
    pub fn potion_colour(self) -> i32 {
        self.int(7).unwrap_or(0)
    }

    /// java: `EntityLivingBase` (`:643`) — index 8, read as `> 0` rather than `== 1`. Ambient
    /// effects (beacon range) draw their swirls at reduced alpha.
    #[must_use]
    pub fn is_potion_ambient(self) -> bool {
        self.byte(8).is_some_and(|b| b > 0)
    }

    /// java: `EntityLivingBase.getArrowCountInEntity` (`:1319`) — index 9. `LayerArrow` draws
    /// this many shafts stuck in the model.
    #[must_use]
    pub fn arrows_in_body(self) -> i32 {
        i32::from(self.byte(9).unwrap_or(0))
    }

    /// java: `EntityLiving.isAIDisabled` (`:1301`) — index 15, `!= 0`.
    ///
    /// **Not purely a server concern**, which is what this comment used to claim. Index 15 has
    /// exactly one client reader in 1.8.9: `EntityDragon.onLivingUpdate` (`EntityDragon:172`)
    /// pins `animTime = 0.5F` when NoAI is set, freezing the wing beat mid-flap. That path is
    /// unreachable here only because `ModelDragon` is unported (`manager.rs` lists
    /// `E::EnderDragon` as having no renderer), so this accessor stays caller-less until the
    /// dragon lands — at which point it is required, not optional.
    #[must_use]
    pub fn has_no_ai(self) -> bool {
        self.byte(15).is_some_and(|b| b != 0)
    }

    // ---- ageing --------------------------------------------------------------------------

    /// java: `EntityAgeable.getGrowingAge` (`:88`) — index 12, read as a **byte** on the client
    /// (the server's own field is an `int`, so the client only ever sees a clamped sign).
    #[must_use]
    pub fn growing_age(self) -> Option<i8> {
        self.byte(12)
    }

    /// Whether this entity draws with the child model.
    ///
    /// java: **three** incompatible readings, on two different indices. `EntityAgeable.isChild`
    /// is `getGrowingAge() < 0`; `EntityZombie.isChild` (`:155`) is
    /// `getWatchableObjectByte(12) == 1` — `EntityZombie` is not an `EntityAgeable` at all and
    /// re-registers index 12 for its own boolean, so a zombie with a negative age byte is an
    /// adult; and `EntityArmorStand.isChild` (`:793`) is `isSmall()`, which is index **10**
    /// bit 0 (`:827`) and has nothing to do with an age at all.
    ///
    /// The armour-stand arm was missing, so every small stand drew full size.
    /// `RendererLivingEntity.doRender` (`:95`) assigns `mainModel.isChild = entity.isChild()`
    /// for every living entity, and both `ModelArmorStand.render` (`:100-107`) and the
    /// `ModelBiped.render` it calls through `super` have a real half-scale child branch.
    #[must_use]
    pub fn is_child(self) -> bool {
        match self.entity_type() {
            Some(EntityType::Zombie | EntityType::PigZombie) => self.byte(12) == Some(1),
            // java: `EntityArmorStand.isSmall` — index 10 bit 0, the same byte
            // [`Watched::armor_stand`] reads its other four flags out of.
            Some(EntityType::ArmorStand) => self.byte(10).is_some_and(|b| b & 1 != 0),
            // A remote player is an `EntityPlayer`, which registers no index 12 and is never a
            // child; `byte(12)` is `None` for it either way.
            _ => self.byte(12).is_some_and(|age| age < 0),
        }
    }

    // ---- sheep ---------------------------------------------------------------------------

    /// java: `EntitySheep.getFleeceColor` (`:262`) — `EnumDyeColor.byMetadata(byte16 & 15)`.
    #[must_use]
    pub fn fleece_colour(self) -> Option<DyeColour> {
        self.typed_byte(&[EntityType::Sheep], 16)
            .map(|b| DyeColour::from_metadata(i32::from(b & 15)))
    }

    /// java: `EntitySheep.getSheared` (`:279`) — `(byte16 & 16) != 0`. `LayerSheepWool` skips the
    /// fleece entirely when set.
    #[must_use]
    pub fn is_sheared(self) -> bool {
        self.typed_byte(&[EntityType::Sheep], 16)
            .is_some_and(|b| b & 16 != 0)
    }

    // ---- creeper -------------------------------------------------------------------------

    /// java: `EntityCreeper.getCreeperState` (`:236`) — index 16, defaulting to **-1** (idle);
    /// `1` means the fuse is lit, which `ModelCreeper`/`RenderCreeper` turn into the swell.
    #[must_use]
    pub fn creeper_state(self) -> i32 {
        i32::from(self.typed_byte(&[EntityType::Creeper], 16).unwrap_or(-1))
    }

    /// java: `EntityCreeper.getPowered` (`:215`) — index 17, `== 1`. Drives
    /// `LayerCreeperCharge`.
    #[must_use]
    pub fn is_charged(self) -> bool {
        self.typed_byte(&[EntityType::Creeper], 17) == Some(1)
    }

    /// java: `EntityCreeper.hasIgnited` (`:295`) — index 18, `!= 0` (flint-and-steel ignition).
    #[must_use]
    pub fn is_ignited(self) -> bool {
        self.typed_byte(&[EntityType::Creeper], 18)
            .is_some_and(|b| b != 0)
    }

    // ---- tameable (wolf, ocelot) ----------------------------------------------------------

    /// java: `EntityTameable.isSitting` (`:143`) — index 16 `& 1`.
    #[must_use]
    pub fn is_sitting(self) -> bool {
        self.tameable_flags().is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityTameable.isTamed` (`:118`) — index 16 `& 4`.
    #[must_use]
    pub fn is_tamed(self) -> bool {
        self.tameable_flags().is_some_and(|b| b & 4 != 0)
    }

    /// java: `EntityWolf.isAngry` (`:520`) — index 16 `& 2`. Only the wolf uses this bit; an
    /// ocelot's index 16 has no bit 1 meaning at all.
    #[must_use]
    pub fn is_angry(self) -> bool {
        self.typed_byte(&[EntityType::Wolf], 16)
            .is_some_and(|b| b & 2 != 0)
    }

    /// The owner's UUID, as the string the server publishes it as.
    ///
    /// java: two different slots for the same idea — `EntityTameable.getOwnerId` (`:162`) is
    /// index **17** for the wolf and the ocelot, while `EntityHorse.getOwnerId` (`:214`) is index
    /// **21**, because `EntityHorse` is an `EntityAnimal` rather than an `EntityTameable` and had
    /// to register its own. Empty means unowned in both.
    #[must_use]
    pub fn owner_id(self) -> Option<&'a str> {
        let index = match self.entity_type()? {
            EntityType::Wolf | EntityType::Ozelot => 17,
            EntityType::EntityHorse => 21,
            _ => return None,
        };
        self.string(index).filter(|s| !s.is_empty())
    }

    /// java: `EntityWolf.entityInit` registers index 18 as the wolf's *own* copy of its health,
    /// so a client can droop a hurt wolf's tail (`getTailRotation`, `:495`) without knowing the
    /// generic index-6 health.
    #[must_use]
    pub fn wolf_health(self) -> Option<f32> {
        self.typed_float(&[EntityType::Wolf], 18)
    }

    /// java: `EntityWolf.isBegging` (`:602`) — index 19, `== 1`.
    #[must_use]
    pub fn is_begging(self) -> bool {
        self.typed_byte(&[EntityType::Wolf], 19) == Some(1)
    }

    /// java: `EntityWolf.getCollarColor` (`:542`) — index 20, read with **`byDyeDamage`**, not
    /// `byMetadata`; `setCollarColor` (`:547`) stores `collarcolor.getDyeDamage() & 15` to match.
    ///
    /// Note the vanilla inconsistency this faithfully reproduces: `entityInit` seeds index 20
    /// with `EnumDyeColor.RED.getMetadata()` — `14`, a *metadata* value in a *dye-damage* slot —
    /// so a wolf that has never had its collar set reads back as `byDyeDamage(14)`, i.e. orange,
    /// not red. Reading this nibble with `byMetadata` would "fix" the default and mis-tint every
    /// collar a server actually sets.
    #[must_use]
    pub fn collar_colour(self) -> Option<DyeColour> {
        self.typed_byte(&[EntityType::Wolf], 20)
            .map(|b| DyeColour::from_dye_damage(i32::from(b & 15)))
    }

    /// java: `EntityWolf.getTailRotation` (`:495`) — angry, tamed and wild each pick a different
    /// tail angle, and the tamed case interpolates on the wolf's index-18 health.
    #[must_use]
    pub fn wolf_tail_rotation(self) -> f32 {
        if self.is_angry() {
            1.539_380_4
        } else if self.is_tamed() {
            // `Entity` calls the virtual `entityInit` from its constructor, before
            // `EntityLivingBase` later raises health to the maximum.  `EntityWolf.entityInit`
            // therefore copies index 6's registered 1.0F into index 18, not the wolf's eventual
            // 20.0F health.  Spawn packets normally include the slot, but a partial metadata bag
            // must still reproduce that construction-time default.
            (0.55 - (20.0 - self.wolf_health().unwrap_or(1.0)) * 0.02) * core::f32::consts::PI
        } else {
            core::f32::consts::PI / 5.0
        }
    }

    // ---- other passives -------------------------------------------------------------------

    /// java: `EntityVillager.getProfession` (`:358`) — index 16, an `Integer`, wrapped as
    /// `Math.max(value % 5, 0)`.
    ///
    /// The wrap is not decoration: it is why `RenderVillager.getEntityTexture`'s `default:`
    /// branch (the plain green villager) is **unreachable** in 1.8.9, and why profession 5 is
    /// the farmer rather than the default. Rust's `%` truncates toward zero exactly as Java's
    /// does, so a negative value lands on `max(_, 0)` the same way.
    #[must_use]
    pub fn profession(self) -> i32 {
        (self.typed_int(&[EntityType::Villager], 16).unwrap_or(0) % 5).max(0)
    }

    /// java: `EntityOcelot.getTameSkin` — index 18 (0 wild, 1 tuxedo, 2 tabby, 3 siamese).
    #[must_use]
    pub fn ocelot_skin(self) -> i32 {
        i32::from(self.typed_byte(&[EntityType::Ozelot], 18).unwrap_or(0))
    }

    /// java: `EntityRabbit.getRabbitType` — index 18 (0–5, plus 99 for the Killer Bunny).
    #[must_use]
    pub fn rabbit_type(self) -> i32 {
        i32::from(self.typed_byte(&[EntityType::Rabbit], 18).unwrap_or(0))
    }

    /// java: `EntitySkeleton.getSkeletonType` — index 13 (0 normal, 1 wither).
    #[must_use]
    pub fn skeleton_type(self) -> i32 {
        i32::from(self.typed_byte(&[EntityType::Skeleton], 13).unwrap_or(0))
    }

    /// java: `EntityPig.getSaddled` — index 16 `& 1`. `LayerSaddle` is gated on it.
    #[must_use]
    pub fn is_saddled(self) -> bool {
        self.typed_byte(&[EntityType::Pig], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityBat.getIsBatHanging` (`:96`) — index 16 `& 1`.
    #[must_use]
    pub fn is_bat_hanging(self) -> bool {
        self.typed_byte(&[EntityType::Bat], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityHorse`'s five indices, gathered. `None` for anything that is not a horse.
    #[must_use]
    pub fn horse(self) -> Option<HorseState> {
        if self.entity_type() != Some(EntityType::EntityHorse) {
            return None;
        }
        // java: `getHorseWatchableBoolean(mask)` is `(getWatchableObjectInt(16) & mask) != 0`.
        let f = self.int(16).unwrap_or(0);
        Some(HorseState {
            tame: f & 2 != 0,
            saddled: f & 4 != 0,
            chested: f & 8 != 0,
            breeding: f & 16 != 0,
            eating_haystack: f & 32 != 0,
            rearing: f & 64 != 0,
            mouth_open: f & 128 != 0,
            horse_type: i32::from(self.byte(19).unwrap_or(0)),
            variant: self.int(20).unwrap_or(0),
            armour: self.int(22).unwrap_or(0),
        })
    }

    // ---- monsters -------------------------------------------------------------------------

    /// java: `EntityEnderman.getHeldBlockState` reads index 16 as an unsigned packed state id.
    /// `Block.getStateId` lays it out as `block_id + (metadata << 12)`; index 17 is registered but
    /// vanilla never reads or writes it. `None` means the packed state resolves to air.
    #[must_use]
    pub fn carried_block(self) -> Option<(i16, i8)> {
        // The watcher transports a Java `short`, so metadata 8..=15 sets the sign bit. Cast to
        // `u16` first to reproduce `getWatchableObjectShort(16) & 65535` before unpacking it the
        // same way `Block.getStateById` does.
        let state_id = self.typed_short(&[EntityType::Enderman], 16)? as u16;
        let block_id = (state_id & 0x0fff) as i16;
        if block_id == 0 {
            return None;
        }
        Some((block_id, (state_id >> 12) as i8))
    }

    /// java: `EntityEnderman.isScreaming` — index 18, `!= 0`.
    #[must_use]
    pub fn is_screaming(self) -> bool {
        self.typed_byte(&[EntityType::Enderman], 18)
            .is_some_and(|b| b != 0)
    }

    /// java: `EntitySlime.getSlimeSize` — index 16, defaulting to **1** (`entityInit` registers
    /// `(byte)1`). `RenderSlime.preRenderCallback` scales the model by it.
    #[must_use]
    pub fn slime_size(self) -> i32 {
        i32::from(
            self.typed_byte(&[EntityType::Slime, EntityType::LavaSlime], 16)
                .unwrap_or(1),
        )
    }

    /// java: `EntitySpider.isBesideClimbableBlock` (`:175`) — index 16 `& 1`.
    #[must_use]
    pub fn is_climbing(self) -> bool {
        self.typed_byte(&[EntityType::Spider, EntityType::CaveSpider], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityBlaze.isBurning` (`:187`) — index 16 `& 1`. This is the blaze's *own* flag,
    /// separate from the shared index-0 fire bit.
    #[must_use]
    pub fn is_blaze_burning(self) -> bool {
        self.typed_byte(&[EntityType::Blaze], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityGhast.isAttacking` (`:44`) — index 16, `!= 0`. Swaps the ghast's face
    /// texture.
    #[must_use]
    pub fn is_ghast_attacking(self) -> bool {
        self.typed_byte(&[EntityType::Ghast], 16)
            .is_some_and(|b| b != 0)
    }

    /// java: `EntityIronGolem.isPlayerCreated` (`:275`) — index 16 `& 1`. Server-side only: it
    /// decides whether the golem defends a village or its builder, and no renderer reads it.
    /// (The poppy is *not* this flag — `LayerIronGolemFlower` gates on `getHoldRoseTick() != 0`,
    /// which is `S19PacketEntityStatus` opcode 11, not a watched value.)
    #[must_use]
    pub fn is_player_created(self) -> bool {
        self.typed_byte(&[EntityType::VillagerGolem], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityWitch.getAggressive` (`:98`) — index **21**, `== 1`; the witch is the only
    /// mob that watches that slot. **Server-side only**, exactly like `isPlayerCreated` above:
    /// its two readers are `onLivingUpdate` (`:118`, inside `if (!worldObj.isRemote)`) and
    /// `attackEntityWithRangedAttack` (`:255`), neither of which a client runs.
    ///
    /// The doc here used to say it "drives `LayerHeldItemWitch`'s potion". It does not:
    /// `LayerHeldItemWitch.doRenderLayer` (`:26`) and `RenderWitch.doRender` (`:25`) both gate on
    /// `getHeldItem() != null`, i.e. equipment slot 0 off `S04PacketEntityEquipment`. See
    /// `EntityAnimState::witch_holding_item`, which is the flag the nose pose actually reads.
    #[must_use]
    pub fn is_witch_aggressive(self) -> bool {
        self.typed_byte(&[EntityType::Witch], 21) == Some(1)
    }

    /// java: `EntityGuardian.isElder` (`:155`) — index 16 `& 4`.
    #[must_use]
    pub fn is_elder_guardian(self) -> bool {
        self.typed_int(&[EntityType::Guardian], 16)
            .is_some_and(|f| f & 4 != 0)
    }

    /// java: `EntityGuardian.func_175472_n` (`:139`) — index 16 `& 2`, the retracted-spikes pose.
    #[must_use]
    pub fn guardian_spikes_retracted(self) -> bool {
        self.typed_int(&[EntityType::Guardian], 16)
            .is_some_and(|f| f & 2 != 0)
    }

    /// java: `EntityGuardian.hasTargetedEntity` (`:189`) — index 17, the laser target's *network*
    /// entity id; `0` means none.
    #[must_use]
    pub fn guardian_target(self) -> Option<i32> {
        self.typed_int(&[EntityType::Guardian], 17)
            .filter(|id| *id != 0)
    }

    /// The id assigned to one of the wither's three heads, if it has a live target.
    ///
    /// java: `EntityWither.getWatchedTargetId` (`:636-639`) reads integer slot `17 + head`.
    /// Slot 17 is the server-only centre head target, while the client reads slots 18 and 19 for
    /// the two side heads in `onLivingUpdate` (`:177-187`). The registered default is zero and
    /// every client-side caller treats only a positive id as an entity id, so `None` is the
    /// useful client interpretation for zero and hostile negative values.
    #[must_use]
    pub fn wither_target(self, head: u8) -> Option<i32> {
        if head > 2 {
            return None;
        }
        self.typed_int(&[EntityType::WitherBoss], 17 + head)
            .filter(|id| *id > 0)
    }

    /// java: `EntityWither.getInvulTime` (`:623-626`) -- index 20, defaulting to zero.
    #[must_use]
    pub fn wither_invul_time(self) -> i32 {
        self.typed_int(&[EntityType::WitherBoss], 20).unwrap_or(0)
    }

    /// Whether `RenderWither.getEntityTexture` selects its invulnerable sheet this frame.
    ///
    /// java: `RenderWither.getEntityTexture` (`:33-37`). The last 80 invulnerability ticks
    /// blink in five-tick bands; testing merely `invul_time > 0` would leave that visible cadence
    /// wrong. Integer division and remainder here deliberately have Java/Rust truncation rules.
    #[must_use]
    pub fn is_wither_invulnerable_texture(self) -> bool {
        let time = self.wither_invul_time();
        time > 0 && (time > 80 || time / 5 % 2 != 1)
    }

    /// java: `EntityZombie.isVillager` (`:197`) — index 13, `== 1`.
    #[must_use]
    pub fn is_zombie_villager(self) -> bool {
        self.typed_byte(&[EntityType::Zombie], 13) == Some(1)
    }

    /// java: `EntityZombie.isConverting` (`:686`) — index 14, `== 1` (a curing villager zombie).
    #[must_use]
    pub fn is_converting(self) -> bool {
        self.typed_byte(&[EntityType::Zombie], 14) == Some(1)
    }

    // ---- objects ---------------------------------------------------------------------------

    /// java: `EntityArrow.getIsCritical` (`:599`) — index 16 `& 1`, the crit particle trail.
    #[must_use]
    pub fn is_arrow_critical(self) -> bool {
        self.typed_byte(&[EntityType::Arrow], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    /// java: `EntityItem.getEntityItem` (`:457`) — index 10, an `ItemStack`. The dropped stack
    /// `RenderEntityItem`/`DroppedItem` draws.
    #[must_use]
    pub fn dropped_item(self) -> Option<&'a mc_protocol::ItemStack> {
        self.typed_stack(&[EntityType::Item], 10)
    }

    /// java: `EntityItemFrame.getDisplayedItem` (`:148`) — index 8, an `ItemStack`.
    #[must_use]
    pub fn frame_item(self) -> Option<&'a mc_protocol::ItemStack> {
        self.typed_stack(&[EntityType::ItemFrame], 8)
    }

    /// java: `EntityItemFrame.getRotation` (`:179`) — index 8's companion at 9, and the
    /// eighth-turn the frame's contents are drawn at (`rotation * 360 / 8`).
    ///
    /// The reduction is vanilla's own: `func_174865_a` (`:189`) stores `(byte)(rotation % 8)`, so
    /// a server that follows the rules never sends anything outside `0..8` and `getRotation`
    /// returns the raw byte. `rem_euclid` here is the hostile-server guard — a raw `%` would let
    /// a negative byte through as a negative eighth-turn.
    #[must_use]
    pub fn frame_rotation(self) -> i32 {
        i32::from(self.typed_byte(&[EntityType::ItemFrame], 9).unwrap_or(0)).rem_euclid(8)
    }

    /// java: `EntityFireworkRocket.entityInit` — index 8, the firework `ItemStack` whose NBT
    /// holds the explosion recipe.
    #[must_use]
    pub fn firework_item(self) -> Option<&'a mc_protocol::ItemStack> {
        self.typed_stack(&[EntityType::FireworksRocketEntity], 8)
    }

    /// java: `EntityEnderCrystal.entityInit` — index 8, an `Integer` health.
    #[must_use]
    pub fn ender_crystal_health(self) -> Option<i32> {
        self.typed_int(&[EntityType::EnderCrystal], 8)
    }

    /// java: `EntityWitherSkull.isInvulnerable` — index 10, `== 1` (the blue charged skull).
    #[must_use]
    pub fn is_skull_charged(self) -> bool {
        self.typed_byte(&[EntityType::WitherSkull], 10) == Some(1)
    }

    /// java: `EntityArmorStand`'s index-10 flags and its six `Rotations` (`:696-731`, `:813-920`).
    #[must_use]
    pub fn armor_stand(self) -> Option<ArmorStandState> {
        if self.entity_type() != Some(EntityType::ArmorStand) {
            return None;
        }
        let flags = self.byte(10).unwrap_or(0);
        let default = ArmorStandPose::default();
        Some(ArmorStandState {
            small: flags & 1 != 0,
            no_gravity: flags & 2 != 0,
            show_arms: flags & 4 != 0,
            no_base_plate: flags & 8 != 0,
            marker: flags & 16 != 0,
            pose: ArmorStandPose {
                head: self.rotations(11).unwrap_or(default.head),
                body: self.rotations(12).unwrap_or(default.body),
                left_arm: self.rotations(13).unwrap_or(default.left_arm),
                right_arm: self.rotations(14).unwrap_or(default.right_arm),
                left_leg: self.rotations(15).unwrap_or(default.left_leg),
                right_leg: self.rotations(16).unwrap_or(default.right_leg),
            },
        })
    }

    /// java: `EntityMinecart` (`:1008-1040`) / `EntityBoat` (`:579-611`) — indices 17/18/19.
    /// `None` for anything that is neither.
    #[must_use]
    pub fn damage_wobble(self) -> Option<DamageWobble> {
        if !is_minecart_or_boat(self.entity_type()?) {
            return None;
        }
        let default = DamageWobble::default();
        Some(DamageWobble {
            time_since_hit: self.int(17).unwrap_or(default.time_since_hit),
            rock_direction: self.int(18).unwrap_or(default.rock_direction),
            damage_taken: self.float(19).unwrap_or(default.damage_taken),
        })
    }

    /// The block a minecart displays instead of its default contents, as a
    /// `Block.getStateId`-packed id.
    ///
    /// java: `EntityMinecart.getDisplayTile` (`:1047`) — index 20, but **only** when
    /// `hasDisplayTile()` (index 22, `== 1`, `:1079`); otherwise vanilla returns the cart's own
    /// `getDefaultDisplayTile()`, which is per-subclass and not a watched value at all. `None`
    /// here therefore means "use the cart's default", not "no block".
    #[must_use]
    pub fn minecart_display_block(self) -> Option<i32> {
        if !self.entity_type().is_some_and(is_minecart_or_boat) || !self.has_custom_display() {
            return None;
        }
        Some(self.int(20).unwrap_or(0))
    }

    /// java: `EntityMinecart.getDisplayTileOffset` (`:1057`) — index 21, default **6**, and like
    /// [`Self::minecart_display_block`] only meaningful when `hasDisplayTile()`.
    #[must_use]
    pub fn minecart_display_offset(self) -> Option<i32> {
        if !self.entity_type().is_some_and(is_minecart_or_boat) || !self.has_custom_display() {
            return None;
        }
        Some(self.int(21).unwrap_or(6))
    }

    /// java: `EntityMinecartCommandBlock.entityInit` (`:66-70`) and
    /// `onDataWatcherUpdate` (`:126-145`) -- its command and last-output component JSON.
    ///
    /// The command-block GUI is responsible for parsing `last_output_json` leniently, exactly
    /// like Java's `try { jsonToComponent(...) } catch (Throwable) { ... }`; metadata decoding
    /// must preserve the wire string even when that JSON is malformed.
    #[must_use]
    pub fn command_minecart(self) -> Option<CommandMinecartState<'a>> {
        if self.entity_type() != Some(EntityType::MinecartCommandBlock) {
            return None;
        }
        Some(CommandMinecartState {
            command: self.string(23).unwrap_or(""),
            last_output_json: self.string(24).unwrap_or(""),
        })
    }

    /// java: `EntityMinecartFurnace.isMinecartPowered` — index 16 `& 1`.
    #[must_use]
    pub fn is_furnace_minecart_powered(self) -> bool {
        self.typed_byte(&[EntityType::MinecartFurnace], 16)
            .is_some_and(|b| b & 1 != 0)
    }

    // ---- players ----------------------------------------------------------------------------

    /// java: `EntityPlayer.entityInit` — index 10, the `EnumPlayerModelParts` bitmask.
    #[must_use]
    pub fn skin_parts(self) -> SkinParts {
        match self.kind {
            TrackedKind::Player => SkinParts(self.byte(10).unwrap_or(0)),
            TrackedKind::Type(_) | TrackedKind::FishingBobber => SkinParts(0),
        }
    }

    /// java: `EntityPlayer.getAbsorptionAmount` (`:2355`) — index 17, a `Float` (the golden
    /// hearts).
    #[must_use]
    pub fn absorption(self) -> f32 {
        match self.kind {
            TrackedKind::Player => self.float(17).unwrap_or(0.0),
            TrackedKind::Type(_) | TrackedKind::FishingBobber => 0.0,
        }
    }

    /// java: `EntityPlayer.getScore` (`:690`) — index 18.
    #[must_use]
    pub fn score(self) -> i32 {
        match self.kind {
            TrackedKind::Player => self.int(18).unwrap_or(0),
            TrackedKind::Type(_) | TrackedKind::FishingBobber => 0,
        }
    }

    // ---- the derived variant ----------------------------------------------------------------

    /// The texture/model variant [`mc_render::entity::manager::texture_for`] switches on.
    ///
    /// java: each renderer's `getEntityTexture`. This is deliberately one number rather than a
    /// per-species enum because that is the shape `texture_for` was written against, and it is
    /// the field `mc_render::entity::EntityAnimState::variant` already carries. GAP A10-06 is
    /// that the caller used to pass the literal `0` here.
    #[must_use]
    pub fn variant(self) -> i32 {
        let Some(t) = self.entity_type() else {
            return 0;
        };
        use EntityType as E;
        match t {
            // The fleece nibble, which `LayerSheepWool` tints with rather than swapping texture.
            E::Sheep => self.fleece_colour().map_or(0, DyeColour::metadata),
            // java: `RenderWolf.getEntityTexture` is
            // `isTamed() ? tamed : (isAngry() ? angry : wolf)` — **tamed wins**. The comment
            // that used to sit here asserted the opposite, and the code matched the comment, so
            // a tamed wolf that had been angered swapped to `wolf_angry.png` (which vanilla
            // shows only for wild angry wolves) and, because `LayerWolfCollar` is gated on the
            // tame texture downstream, lost its collar.
            E::Wolf => {
                if self.is_tamed() {
                    1
                } else if self.is_angry() {
                    2
                } else {
                    0
                }
            }
            E::Ozelot => self.ocelot_skin(),
            E::Rabbit => self.rabbit_type(),
            E::Villager => self.profession(),
            E::Skeleton => self.skeleton_type(),
            E::Pig => i32::from(self.is_saddled()),
            E::Creeper => i32::from(self.is_charged()),
            E::Slime | E::LavaSlime => self.slime_size(),
            E::EntityHorse => self.horse().map_or(0, |h| h.horse_type),
            // Three `getEntityTexture` forks that `mc_render::entity::manager::texture_for`
            // already has arms for and that nothing could ever reach, because this match had no
            // case for them and fell through to `0`:
            //
            // java: `RenderZombie.getEntityTexture` (`:66`) — `isVillager()`, index 13.
            E::Zombie => i32::from(self.is_zombie_villager()),
            // java: `RenderGhast.getEntityTexture` (`:20`) — `isAttacking()`, the red-eyed sheet
            // that is the only warning a fireball is coming.
            E::Ghast => i32::from(self.is_ghast_attacking()),
            // java: `RenderGuardian.getEntityTexture` (`:129`) — `isElder()`.
            E::Guardian => i32::from(self.is_elder_guardian()),
            // java: `RenderWither.getEntityTexture` (`:33-37`) -- the invulnerable sheet is
            // solid through tick 81, then flickers in five-tick bands for the final 80 ticks.
            E::WitherBoss => i32::from(self.is_wither_invulnerable_texture()),
            _ => 0,
        }
    }

    // ---- typed readers ------------------------------------------------------------------

    fn byte(self, index: u8) -> Option<i8> {
        match self.metadata.get(index) {
            Some(MetadataValue::Byte(v)) => Some(*v),
            _ => None,
        }
    }

    fn short(self, index: u8) -> Option<i16> {
        match self.metadata.get(index) {
            Some(MetadataValue::Short(v)) => Some(*v),
            _ => None,
        }
    }

    fn int(self, index: u8) -> Option<i32> {
        match self.metadata.get(index) {
            Some(MetadataValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    fn float(self, index: u8) -> Option<f32> {
        match self.metadata.get(index) {
            Some(MetadataValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    fn string(self, index: u8) -> Option<&'a str> {
        match self.metadata.get(index) {
            Some(MetadataValue::Str(v)) => Some(v.as_str()),
            _ => None,
        }
    }

    fn stack(self, index: u8) -> Option<&'a mc_protocol::ItemStack> {
        match self.metadata.get(index) {
            Some(MetadataValue::ItemStack(v)) => v.as_ref(),
            _ => None,
        }
    }

    fn rotations(self, index: u8) -> Option<Rotation3> {
        match self.metadata.get(index) {
            Some(MetadataValue::Rotations(x, y, z)) => Some([*x, *y, *z]),
            _ => None,
        }
    }

    fn is_a(self, types: &[EntityType]) -> bool {
        self.entity_type().is_some_and(|t| types.contains(&t))
    }

    fn typed_byte(self, types: &[EntityType], index: u8) -> Option<i8> {
        self.is_a(types).then(|| self.byte(index)).flatten()
    }

    fn typed_short(self, types: &[EntityType], index: u8) -> Option<i16> {
        self.is_a(types).then(|| self.short(index)).flatten()
    }

    fn typed_int(self, types: &[EntityType], index: u8) -> Option<i32> {
        self.is_a(types).then(|| self.int(index)).flatten()
    }

    fn typed_float(self, types: &[EntityType], index: u8) -> Option<f32> {
        self.is_a(types).then(|| self.float(index)).flatten()
    }

    fn typed_stack(self, types: &[EntityType], index: u8) -> Option<&'a mc_protocol::ItemStack> {
        self.is_a(types).then(|| self.stack(index)).flatten()
    }

    /// java: `EntityMinecart.hasDisplayTile` (`:1079`) — index 22, `== 1`.
    fn has_custom_display(self) -> bool {
        self.byte(22) == Some(1)
    }

    /// java: `EntityTameable.entityInit` registers index 16 for the wolf and the ocelot, and only
    /// for them — every other index 16 in the game belongs to a different class.
    fn tameable_flags(self) -> Option<i8> {
        self.typed_byte(&[EntityType::Wolf, EntityType::Ozelot], 16)
    }
}

/// java: `EntityMinecart` and `EntityBoat` register indices 17/18/19 identically, and the seven
/// minecart variants all inherit `EntityMinecart`'s `entityInit`.
fn is_minecart_or_boat(t: EntityType) -> bool {
    use EntityType as E;
    matches!(
        t,
        E::Boat
            | E::MinecartRideable
            | E::MinecartChest
            | E::MinecartFurnace
            | E::MinecartTNT
            | E::MinecartHopper
            | E::MinecartSpawner
            | E::MinecartCommandBlock
    )
}

/// The EntityAgeable branch of the Java `entityInit` chain.
///
/// `EntityHorse` and `EntityVillager` are easy to miss here: neither has an `Animal` name, but
/// both extend `EntityAgeable` and therefore register the same byte index 12 as the ordinary
/// passive animals.
fn is_ageable(t: EntityType) -> bool {
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
            | E::Villager
            | E::EntityHorse
            | E::Rabbit
    )
}

// ---------------------------------------------------------------------------------------------
// Reaching the view from a tracked entity
// ---------------------------------------------------------------------------------------------

impl TrackedEntity {
    /// A [`Watched`] view over this entity's `Metadata` component.
    ///
    /// Takes a closure rather than returning the view because the component lives behind the
    /// ECS's borrow guard: handing a `Watched<'_>` back out would either extend that guard's
    /// lifetime past the call or need a second, owned copy of the metadata every frame.
    ///
    /// Returns `None` when the entity has no `Metadata` component — which for a live tracked
    /// entity means it has been despawned out from under the tracker.
    pub fn with_watched<R>(
        &self,
        world: &mc_world::World,
        f: impl FnOnce(Watched<'_>) -> R,
    ) -> Option<R> {
        let metadata = world.ecs.get::<&Metadata>(self.entity).ok()?;
        Some(f(Watched::new(self.kind, &metadata)))
    }
}

impl EntityTracker {
    /// [`TrackedEntity::with_watched`] by network id.
    pub fn with_watched<R>(
        &self,
        world: &mc_world::World,
        network_id: i32,
        f: impl FnOnce(Watched<'_>) -> R,
    ) -> Option<R> {
        self.get(network_id)?.with_watched(world, f)
    }
}

// ---------------------------------------------------------------------------------------------
// Potion swirls
// ---------------------------------------------------------------------------------------------

/// One tick of every tracked entity's potion-swirl particles.
///
/// java: the tail of `EntityLivingBase.updatePotionEffects` (`:642-670`), which is the *only*
/// part of that method a client runs — the rest is `!worldObj.isRemote`. It reads indices 7 and 8
/// and nothing else, which is precisely why a client does not need to model a remote entity's
/// `activePotionsMap` to draw its swirls (`driver.rs` restricts `EntityEffectAdded`
/// to the local player and notes that remote effects were never modelled — they do not need to
/// be).
///
/// Two deliberate divergences, both invisible:
/// - vanilla rolls from each entity's own `Entity.rand`; there is one shared
///   [`mc_render::particle::ParticleSystem`] RNG here, so the *sequence* differs while the
///   distribution does not.
/// - `spawnParticle` is called through `RenderGlobal.spawnParticle`'s range/density gate, which
///   is [`mc_render::particle::ParticleSystem::should_spawn`]; `SPELL_MOB` and
///   `SPELL_MOB_AMBIENT` both have `shouldIgnoreRange == false`, so the 16-block cull applies.
///
/// Attempt the one potion-swirl emission owned by one living entity for this tick.
///
/// The local player is not part of [`EntityTracker`], so the driver calls this with its predictor
/// position and active-effect colour. It mirrors [`tick_potion_swirls`]'s watcher-driven attempt,
/// giving the local route the same Java RNG order, range/density gate, colour decode and variant.
#[allow(clippy::too_many_arguments)] // Mirrors the Java call's independent per-entity inputs.
pub fn tick_potion_swirl(
    world: &mc_world::World,
    view_pos: mc_math::DVec3,
    base: mc_math::DVec3,
    width: f32,
    height: f32,
    colour: i32,
    ambient: bool,
    invisible: bool,
    particles: &mut mc_render::particle::ParticleSystem,
) {
    use mc_render::particle::spawn::{self, SpellVariant};

    if colour <= 0 {
        return;
    }
    let flag = {
        let rng = particles.rng_mut();
        let mut flag = if invisible {
            rng.next_i32_bound(15) == 0
        } else {
            rng.next_bool()
        };
        if ambient {
            let rarer = rng.next_i32_bound(5) == 0;
            flag &= rarer;
        }
        flag
    };
    if !flag {
        return;
    }

    let pos = {
        let rng = particles.rng_mut();
        mc_math::DVec3::new(
            base.x + (rng.next_f64() - 0.5) * f64::from(width),
            base.y + rng.next_f64() * f64::from(height),
            base.z + (rng.next_f64() - 0.5) * f64::from(width),
        )
    };
    if !particles.should_spawn(false, view_pos, pos) {
        return;
    }
    let rgb = mc_math::DVec3::new(
        f64::from((colour >> 16) & 255) / 255.0,
        f64::from((colour >> 8) & 255) / 255.0,
        f64::from(colour & 255) / 255.0,
    );
    let variant = if ambient {
        SpellVariant::AmbientMob
    } else {
        SpellVariant::Mob
    };
    // Particles mod: the potion-swirl group. Without its table this is `spell` on the
    // system RNG, then `spawn`.
    particles.spawn_attempt(world, swirl_particle_id(ambient), |rng| {
        Some(spawn::spell(rng, pos, rgb, variant))
    });
}

/// The `EnumParticleTypes` id of a potion swirl (the particles mod's group key).
fn swirl_particle_id(ambient: bool) -> i32 {
    use mc_render::particle::packet::{SPELL_MOB, SPELL_MOB_AMBIENT};
    if ambient {
        SPELL_MOB_AMBIENT
    } else {
        SPELL_MOB
    }
}

/// Attempt potion-swirl emission for every tracked remote living entity from watcher 7/8 state.
pub fn tick_potion_swirls(
    entities: &EntityTracker,
    world: &mc_world::World,
    view_pos: mc_math::DVec3,
    particles: &mut mc_render::particle::ParticleSystem,
) {
    use mc_render::particle::spawn::{self, SpellVariant};

    let lazy = crate::sync::entities::nw_f7_lazy_tick();
    for (_, tracked) in entities.iter() {
        // java: `updatePotionEffects` is `EntityLivingBase`'s; an arrow or a minecart never
        // reaches it.
        if !tracked.kind.is_living() {
            continue;
        }
        // The ambient and invisible flags are read only when the entity has
        // an effect colour; every other entity continues straight after this read.
        let Some((colour, ambient, invisible)) = tracked.with_watched(world, |w| {
            let colour = w.potion_colour();
            if lazy && colour <= 0 {
                return (colour, false, false);
            }
            (colour, w.is_potion_ambient(), w.is_invisible())
        }) else {
            continue;
        };
        // java: `if (i > 0)` — zero is "no effects", and a negative packed colour cannot happen.
        if colour <= 0 {
            continue;
        }

        let Ok(base) = world
            .ecs
            .get::<&mc_world::entity::Position>(tracked.entity)
            .map(|p| p.0)
        else {
            continue;
        };

        let flag = {
            let rng = particles.rng_mut();
            // java: an invisible entity trickles at 1-in-15 instead of the usual coin flip, so a
            // potion-invisible mob is still findable by its swirls.
            let mut flag = if invisible {
                rng.next_i32_bound(15) == 0
            } else {
                rng.next_bool()
            };
            if ambient {
                // java: `flag &= this.rand.nextInt(5) == 0` — a **non-short-circuiting** `&=` on
                // a `boolean`, so the roll happens even when `flag` is already false. Writing it
                // as `flag = flag && …` would consume a different number of values from the RNG
                // and desynchronise every later particle in the same tick.
                let rarer = rng.next_i32_bound(5) == 0;
                flag &= rarer;
            }
            flag
        };
        if !flag {
            continue;
        }

        let (width, height) = tracked.dimensions();
        // java: `posX + (rand.nextDouble() - 0.5) * width`, `posY + rand.nextDouble() * height`,
        // `posZ + (rand.nextDouble() - 0.5) * width` — in that argument order, which is the
        // order the three `nextDouble()` calls are made in.
        let pos = {
            let rng = particles.rng_mut();
            mc_math::DVec3::new(
                base.x + (rng.next_f64() - 0.5) * f64::from(width),
                base.y + rng.next_f64() * f64::from(height),
                base.z + (rng.next_f64() - 0.5) * f64::from(width),
            )
        };

        // java: `d0/d1/d2 = (i >> 16 & 255) / 255`, `(i >> 8 & 255) / 255`, `(i & 255) / 255` —
        // handed to `spawnParticle` in the *speed* arguments, which `MobFactory` then reads back
        // as the particle's RGB rather than as motion.
        let rgb = mc_math::DVec3::new(
            f64::from((colour >> 16) & 255) / 255.0,
            f64::from((colour >> 8) & 255) / 255.0,
            f64::from(colour & 255) / 255.0,
        );
        let variant = if ambient {
            SpellVariant::AmbientMob
        } else {
            SpellVariant::Mob
        };
        if !particles.should_spawn(false, view_pos, pos) {
            continue;
        }
        // Particles mod: see `tick_potion_swirl`.
        particles.spawn_attempt(world, swirl_particle_id(ambient), |rng| {
            Some(spawn::spell(rng, pos, rgb, variant))
        });
    }
}

/// Every per-tick, client-only particle emitter that reads tracked entity state.
///
/// java: three separate `onLivingUpdate` bodies, all inside `if (worldObj.isRemote)` and all
/// driven off watched flags the server sends without any accompanying particle packet — so a
/// client that does not run them draws nothing at all:
///
/// * `EntityLivingBase.updatePotionEffects` (`:642-670`) — [`tick_potion_swirls`].
/// * `EntityGuardian.onLivingUpdate:348-355` — the bubble wake.
/// * `EntityGuardian.onLivingUpdate:357-383` — the target beam's bubble trail.
/// * `EntityWolf.onLivingUpdate:265-277` — the shake-off spray.
///
/// This is the entry point the driver calls; the three bodies stay separate because they share
/// nothing but the loop.
pub fn tick_entity_particles(
    entities: &mut EntityTracker,
    world: &mc_world::World,
    view_pos: mc_math::DVec3,
    particles: &mut mc_render::particle::ParticleSystem,
) {
    mc_core::profile_scope!("tracker.entity_particles");
    tick_potion_swirls(&*entities, world, view_pos, particles);
    tick_species_particles(entities, world, view_pos, particles);
}

/// The guardian's bubble wake and laser trail, plus the wolf's shake-off spray.
///
/// Both are `worldObj.spawnParticle` calls from inside `onLivingUpdate`'s `isRemote` half, i.e.
/// `RenderGlobal.spawnParticle` — so both go through the 16-block range and density gate
/// ([`mc_render::particle::ParticleSystem::should_spawn`]); neither type sets
/// `shouldIgnoreRange`.
///
/// Queued to this lane by ENTITY-2: the state both read is advanced by
/// `sync::entities::advance_species_state`, which runs earlier in the same tick.
///
pub fn tick_species_particles(
    entities: &mut EntityTracker,
    world: &mc_world::World,
    view_pos: mc_math::DVec3,
    particles: &mut mc_render::particle::ParticleSystem,
) {
    use mc_render::entity::anim::SpeciesState;

    // Preserve World.loadedEntityList order while still permitting each Guardian's private
    // Entity.rand stream to advance. Borrowing the tracker mutably during a HashMap walk would
    // lose that order, so snapshot only the stable network ids before the effect pass.
    //
    // Only a Guardian (by kind) or a Wolf (by species) can do anything below; every other entity
    // used to cost two id lookups and a `Position` read before falling through `_ => {}`. Both
    // properties are fixed for a record's lifetime (`kind` and the species variant are set by
    // `spawn` and never reassigned), so filtering here keeps the same ids in the same order.
    let network_ids: Vec<i32> = entities
        .iter()
        .filter(|(_, tracked)| {
            tracked.kind.entity_type() == Some(EntityType::Guardian)
                || matches!(tracked.render.species, SpeciesState::Wolf(_))
        })
        .map(|(network_id, _)| network_id)
        .collect();
    for network_id in network_ids {
        if entities
            .get(network_id)
            .is_some_and(|tracked| tracked.kind.entity_type() == Some(EntityType::Guardian))
        {
            tick_guardian_particles(entities, network_id, world, view_pos, particles);
            continue;
        }

        let Some(tracked) = entities.get(network_id) else {
            continue;
        };
        let Some(_t) = tracked.kind.entity_type() else {
            continue;
        };
        let width = tracked.dimensions().0;
        let Ok(base) = world
            .ecs
            .get::<&mc_world::entity::Position>(tracked.entity)
            .map(|p| p.0)
        else {
            continue;
        };

        match tracked.render.species {
            // Guardians are handled above so their positional draws can use Entity.rand.
            SpeciesState::Guardian(_) => continue,
            // java: `EntityWolf.onLivingUpdate:265-277` — the spray a wet wolf throws off while
            // shaking, `sin((t - 0.4) * PI) * 7` drops a tick, peaking mid-shake.
            SpeciesState::Wolf(w) => {
                // java: the emitter sits inside `else if ((isWet || isShaking) && isShaking)`,
                // so a wolf that is merely wet throws nothing.
                if !w.is_shaking || w.shake_time <= 0.4 {
                    continue;
                }
                let motion = world
                    .ecs
                    .get::<&mc_world::entity::Velocity>(tracked.entity)
                    .map_or(mc_math::DVec3::ZERO, |v| v.0);
                // java: `(float)this.getEntityBoundingBox().minY` — the box's floor, which for a
                // living entity is `posY`.
                let floor = base.y as f32;
                let count =
                    (mc_math::sin((w.shake_time - 0.4) * core::f32::consts::PI) * 7.0) as i32;
                for _ in 0..count {
                    let (f1, f2) = {
                        // java: `EntityWolf.rand`, not EffectRenderer's particle-factory RNG.
                        // The two streams are distinct; using the latter changes later local
                        // entity effects whenever the view-distance gate drops a spray.
                        let rng = entities
                            .particle_random_mut(network_id)
                            .expect("Wolf must retain its Entity.rand stream");
                        let f1 = (rng.next_f32() * 2.0 - 1.0) * width * 0.5;
                        let f2 = (rng.next_f32() * 2.0 - 1.0) * width * 0.5;
                        (f1, f2)
                    };
                    let pos = mc_math::DVec3::new(
                        base.x + f64::from(f1),
                        f64::from(floor + 0.8),
                        base.z + f64::from(f2),
                    );
                    if !particles.should_spawn(false, view_pos, pos) {
                        continue;
                    }
                    let particle =
                        mc_render::particle::spawn::splash(particles.rng_mut(), pos, motion);
                    particles.spawn(world, particle);
                }
            }
            _ => {}
        }
    }
}

/// Emit one Guardian's client-only wake/trail using the entity's persistent `Entity.rand` stream.
///
/// java: `EntityGuardian.onLivingUpdate:348-383`. The call-site coordinates and beam spacing use
/// `this.rand`; only the `EntityBubbleFX` factory's own jitter uses the separate effect-renderer
/// random stream represented by [`mc_render::particle::ParticleSystem`]. Keeping those streams
/// distinct is observable when later client-local Guardian effects consume `Entity.rand`.
fn tick_guardian_particles(
    entities: &mut EntityTracker,
    network_id: i32,
    world: &mc_world::World,
    view_pos: mc_math::DVec3,
    particles: &mut mc_render::particle::ParticleSystem,
) {
    let Some((base, width, height, moving, in_water, head_yaw, pitch, laser, target)) =
        entities.get(network_id).and_then(|tracked| {
            // java: `EntityLivingBase.onUpdate` switches to `onDeathUpdate` at zero health, so
            // the Guardian subclass hook (and both of its particle loops) is not entered.
            if tracked.is_dying() {
                return None;
            }
            let base = world
                .ecs
                .get::<&mc_world::entity::Position>(tracked.entity)
                .ok()?
                .0;
            let rotation = world
                .ecs
                .get::<&mc_world::entity::Rotation>(tracked.entity)
                .ok();
            let (_, height) = tracked.dimensions();
            let (width, _) = tracked.dimensions();
            let target = tracked
                .guardian_laser_state()
                .and_then(|laser| laser.target)
                .and_then(|target_id| entities.get(target_id))
                .filter(|target| target.kind.is_living())
                .and_then(|target| {
                    let position = world
                        .ecs
                        .get::<&mc_world::entity::Position>(target.entity)
                        .ok()?
                        .0;
                    Some((position, target.dimensions().1))
                });
            Some((
                base,
                width,
                height,
                tracked
                    .with_watched(world, |watched| watched.guardian_spikes_retracted())
                    .unwrap_or(false),
                crate::sync::entities::in_water(world, tracked),
                tracked
                    .guardian_particle_head_yaw()
                    .unwrap_or(tracked.render.head_yaw),
                rotation.map_or(0.0, |rotation| rotation.pitch),
                tracked.guardian_laser_state(),
                target,
            ))
        })
    else {
        return;
    };

    // java: `getLook(0.0F)` is EntityLivingBase's head-facing vector, not the body yaw that
    // EntityBodyHelper may settle towards after this hook. Use the live head field at hook entry.
    if moving && in_water {
        let look = crate::app::interact::look_vector(head_yaw, pitch);
        for _ in 0..2 {
            // java: x/y/z argument expressions consume Entity.rand doubles left-to-right before
            // RenderGlobal applies its range/density gate or constructs EntityBubbleFX.
            let pos = {
                let rng = entities
                    .particle_random_mut(network_id)
                    .expect("Guardian must retain its Entity.rand stream");
                mc_math::DVec3::new(
                    base.x + (rng.next_f64() - 0.5) * f64::from(width) - look.x * 1.5,
                    base.y + rng.next_f64() * f64::from(height) - look.y * 1.5,
                    base.z + (rng.next_f64() - 0.5) * f64::from(width) - look.z * 1.5,
                )
            };
            if !particles.should_spawn(false, view_pos, pos) {
                continue;
            }
            let particle =
                mc_render::particle::spawn::bubble(particles.rng_mut(), pos, mc_math::DVec3::ZERO);
            particles.spawn(world, particle);
        }
    }

    // Java samples d4 even when the target is at zero distance (the while condition then simply
    // fails). Keep that draw before the finite/zero-distance guard so a malformed/overlapping
    // target cannot shift the entity stream relative to vanilla.
    let Some(laser) = laser else {
        return;
    };
    let Some((target_position, target_height)) = target else {
        return;
    };
    let direction = target_position + mc_math::DVec3::new(0.0, f64::from(target_height * 0.5), 0.0)
        - (base + mc_math::DVec3::new(0.0, f64::from(height * 0.5), 0.0));
    let distance = direction.length();
    let mut offset = entities
        .particle_random_mut(network_id)
        .expect("Guardian must retain its Entity.rand stream")
        .next_f64();
    if !distance.is_finite() || distance <= 0.0 {
        return;
    }
    let direction = direction / distance;
    let elder = entities
        .get(network_id)
        .and_then(|tracked| tracked.with_watched(world, |watched| watched.is_elder_guardian()))
        .unwrap_or(false);
    let max_ticks = if elder { 60.0_f32 } else { 80.0_f32 };
    // java: `func_175477_p(0.0F)` performs the division in float before the result widens to
    // double for the spacing expressions.
    let progress = f64::from(laser.ticks as f32 / max_ticks);
    while offset < distance {
        let step = {
            let rng = entities
                .particle_random_mut(network_id)
                .expect("Guardian must retain its Entity.rand stream");
            1.8 - progress + rng.next_f64() * (1.7 - progress)
        };
        offset += step;
        let pos =
            base + mc_math::DVec3::new(0.0, f64::from(height * 0.5), 0.0) + direction * offset;
        if !particles.should_spawn(false, view_pos, pos) {
            continue;
        }
        let particle =
            mc_render::particle::spawn::bubble(particles.rng_mut(), pos, mc_math::DVec3::ZERO);
        particles.spawn(world, particle);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::entity_type;
    use mc_protocol::{MetadataEntry, MetadataValue as Wire};
    use mc_world::entity::Metadata as Stored;

    #[test]
    fn watched_consumer_documentation_matches_live_call_sites() {
        let watched = include_str!("watched.rs");
        let renderer = include_str!("../app/entrender.rs");
        let summary = watched.lines().take(15).collect::<Vec<_>>().join("\n");

        assert!(summary.contains("reads the charged Wither-skull accessor"));
        assert!(summary.contains("item-frame metadata is read directly there"));
        assert!(summary.contains("minecart display accessors drive the live"));
        assert!(renderer.contains(".is_skull_charged()"));
        assert!(renderer.contains("watched_stack("));
        assert!(!renderer.contains(".armor_stand("));
        let driver = include_str!("../app/driver.rs");
        assert!(driver.contains("watched.minecart_display_block()"));
    }

    /// Build a `Metadata` the way a real `S1CPacketEntityMetadata` would: as wire entries run
    /// through the same `convert_metadata` `sync::entities` uses, never by hand-constructing the
    /// stored enum. A test that skipped the conversion could not catch a wire/stored type
    /// mismatch, which is exactly the class of bug an index-by-index decode invites.
    fn meta(entries: Vec<(u8, Wire)>) -> Stored {
        let wire: mc_protocol::Metadata = entries
            .into_iter()
            .map(|(index, value)| MetadataEntry { index, value })
            .collect();
        let mut out = Stored::new();
        for entry in &wire {
            out.set(
                entry.index,
                super::super::entities::convert_metadata(&entry.value),
            );
        }
        out
    }

    fn of(net_id: u8, entries: Vec<(u8, Wire)>) -> (TrackedKind, Stored) {
        (
            TrackedKind::Type(entity_type(net_id).expect("known entity type")),
            meta(entries),
        )
    }

    fn assert_registered_slots(kind: TrackedKind, accepted: &[u8], rejected: &[u8]) {
        let stored = Stored::new();
        let watched = Watched::new(kind, &stored);
        for &index in accepted {
            assert!(
                watched.is_registered(index),
                "vanilla-registered watcher index {index} was rejected"
            );
        }
        for &index in rejected {
            assert!(
                !watched.is_registered(index),
                "unregistered watcher index {index} was accepted"
            );
        }
    }

    fn assert_registered_types(kind: TrackedKind, expected: &[(u8, u8)]) {
        let stored = Stored::new();
        let watched = Watched::new(kind, &stored);
        for &(index, type_tag) in expected {
            assert_eq!(
                watched.registered_type_tag(index),
                Some(type_tag),
                "vanilla watcher index {index} has the wrong registered type"
            );
        }
    }

    /// java: `DataWatcher.updateWatchedObjectsFromList` only updates slots present in the
    /// receiving entity's watcher map. The high boundary cases are included because accepting
    /// every legal wire byte would let a valid entry shift into an unrelated consumer slot.
    #[test]
    fn watcher_registration_rejects_unowned_and_out_of_range_indices() {
        assert_registered_slots(
            TrackedKind::Type(EntityType::Zombie),
            &[0, 4, 6, 9, 12, 15],
            &[5, 10, 11, 16, 31, 32, u8::MAX],
        );
        assert_registered_slots(
            TrackedKind::Player,
            &[0, 4, 6, 9, 10, 16, 18],
            &[5, 11, 12, 15, 19, 32, u8::MAX],
        );
        assert_registered_slots(
            TrackedKind::Type(EntityType::Slime),
            &[0, 4, 6, 9, 15, 16],
            &[5, 12, 17, 31, 32, u8::MAX],
        );
        assert_registered_slots(
            TrackedKind::Type(EntityType::ItemFrame),
            &[0, 4, 8, 9],
            &[5, 6, 7, 10, 31, 32, u8::MAX],
        );
        assert_registered_slots(
            TrackedKind::Type(EntityType::MinecartFurnace),
            &[0, 4, 16, 17, 22],
            &[5, 15, 23, 31, 32, u8::MAX],
        );
        assert_registered_slots(
            TrackedKind::Type(EntityType::Boat),
            &[0, 4, 17, 19],
            &[5, 16, 20, 31, 32, u8::MAX],
        );
    }

    /// The wire decoder accepts every legal tag, while Java's `entityInit` fixes a tag for each
    /// registered slot. Keep the two layers distinct: these rows are the casts reached by the
    /// concrete entity callbacks and typed readers in the S1C path.
    #[test]
    fn watcher_registration_enforces_java_value_types() {
        assert_registered_types(
            TrackedKind::Player,
            &[
                (0, 0),
                (1, 1),
                (2, 4),
                (3, 0),
                (4, 0),
                (6, 3),
                (7, 2),
                (8, 0),
                (9, 0),
                (10, 0),
                (16, 0),
                (17, 3),
                (18, 2),
            ],
        );
        assert_registered_types(
            TrackedKind::Type(EntityType::Slime),
            &[(6, 3), (7, 2), (8, 0), (9, 0), (15, 0), (16, 0)],
        );
        assert_registered_types(TrackedKind::Type(EntityType::Guardian), &[(16, 2), (17, 2)]);
        assert_registered_types(TrackedKind::Type(EntityType::ItemFrame), &[(8, 5), (9, 0)]);
        assert_registered_types(
            TrackedKind::Type(EntityType::FireworksRocketEntity),
            &[(8, 5)],
        );
        assert_registered_types(TrackedKind::Type(EntityType::EnderCrystal), &[(8, 2)]);
        assert_registered_types(
            TrackedKind::Type(EntityType::ArmorStand),
            &[(6, 3), (7, 2), (8, 0), (9, 0), (10, 0), (11, 7), (16, 7)],
        );
        assert_registered_types(TrackedKind::Type(EntityType::EnderDragon), &[(15, 0)]);
        assert_registered_slots(
            TrackedKind::Type(EntityType::EnderDragon),
            &[6, 9, 15],
            &[5, 31, 32, u8::MAX],
        );
        assert_registered_types(
            TrackedKind::Type(EntityType::Enderman),
            &[(16, 1), (17, 0), (18, 0)],
        );
        assert_registered_types(TrackedKind::Type(EntityType::Villager), &[(12, 0), (16, 2)]);
        assert_registered_types(
            TrackedKind::Type(EntityType::MinecartCommandBlock),
            &[
                (17, 2),
                (18, 2),
                (19, 3),
                (20, 2),
                (21, 2),
                (22, 0),
                (23, 4),
                (24, 4),
            ],
        );

        let stored = Stored::new();
        let slime = Watched::new(TrackedKind::Type(EntityType::Slime), &stored);
        assert!(slime.accepts_registered_type(16, &Wire::Byte(4)));
        assert!(!slime.accepts_registered_type(16, &Wire::Int(4)));
        assert!(!slime.accepts_registered_type(17, &Wire::Byte(4)));
    }

    /// java: `Entity.getFlag`'s javadoc — 0 burning, 1 sneaking, 2 riding, 3 sprinting, 4 eating,
    /// plus 5 invisible from `isInvisible`. Every one of these read as `false` before A10-01.
    #[test]
    fn the_shared_flag_byte_decodes_all_six_bits() {
        let (kind, m) = of(54, vec![(0, Wire::Byte(0b0011_1111))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_burning());
        assert!(w.is_sneaking());
        assert!(w.flags().is_riding());
        assert!(w.is_sprinting());
        assert!(w.is_using_item());
        assert!(w.is_invisible());

        // Bit 0 alone: burning and nothing else. The classic off-by-one here is reading bit 1 as
        // burning because the javadoc numbers from zero and the byte is often written as `1`.
        let (kind, m) = of(54, vec![(0, Wire::Byte(1))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_burning());
        assert!(!w.is_sneaking());
        assert!(!w.is_invisible());
    }

    /// java: `Entity.entityInit` (`:286-290`) — air defaults to 300 and the name defaults to the
    /// empty string, which `hasCustomName` (`length() > 0`) treats as no name.
    #[test]
    fn the_shared_defaults_match_entity_init() {
        let (kind, m) = of(54, vec![]);
        let w = Watched::new(kind, &m);
        assert_eq!(w.air(), 300);
        assert_eq!(w.custom_name(), None);
        assert!(!w.custom_name_visible());
        assert!(!w.is_silent());
        assert_eq!(w.flags().bits(), 0);

        let (kind, m) = of(54, vec![(2, Wire::String(String::new()))]);
        assert_eq!(
            Watched::new(kind, &m).custom_name(),
            None,
            "an empty string is not a custom name"
        );
    }

    /// java: ordinary living entities use `RendererLivingEntity.canRenderName`, which rejects
    /// invisibility; `ArmorStandRenderer.canRenderName` instead tests only index 3.
    /// Nothing but tab-list players got a plate before.
    #[test]
    fn a_nameplate_needs_both_a_name_and_the_visible_flag() {
        let named = vec![(2, Wire::String("Hologram".into()))];
        let (kind, m) = of(54, named.clone());
        assert_eq!(
            Watched::new(kind, &m).nameplate_text(),
            None,
            "a custom name alone is not enough; index 3 must be 1"
        );

        let mut both = named.clone();
        both.push((3, Wire::Byte(1)));
        let (kind, m) = of(54, both.clone());
        assert_eq!(Watched::new(kind, &m).nameplate_text(), Some("Hologram"));

        // java: index 3 is compared `== 1`, not `!= 0`.
        let mut two = named.clone();
        two.push((3, Wire::Byte(2)));
        let (kind, m) = of(54, two);
        assert_eq!(Watched::new(kind, &m).nameplate_text(), None);

        // `RendererLivingEntity.canRenderName` still suppresses an invisible ordinary entity.
        let mut ordinary_hidden = both.clone();
        ordinary_hidden.push((0, Wire::Byte(0b0010_0000)));
        let (kind, m) = of(54, ordinary_hidden);
        assert_eq!(Watched::new(kind, &m).nameplate_text(), None);

        // `ArmorStandRenderer.canRenderName` bypasses that inherited gate. Its small (bit 0)
        // and marker (bit 4) flags do not alter the renderer's name decision either.
        let mut hologram = both;
        hologram.extend([(0, Wire::Byte(0b0010_0000)), (10, Wire::Byte(0b0001_0001))]);
        let (kind, m) = of(30, hologram);
        assert_eq!(Watched::new(kind, &m).nameplate_text(), Some("Hologram"));
    }

    /// java: `EntityLivingBase.entityInit` — 6 health `Float`, 7 swirl colour `Integer`,
    /// 8 ambient `Byte` read as `> 0`, 9 arrow count `Byte`.
    #[test]
    fn the_living_base_block_decodes() {
        let (kind, m) = of(
            54,
            vec![
                (6, Wire::Float(7.5)),
                (7, Wire::Int(0x38_5D_C6)),
                (8, Wire::Byte(1)),
                (9, Wire::Byte(3)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert_eq!(w.health(), Some(7.5));
        assert_eq!(w.potion_colour(), 0x38_5D_C6);
        assert!(w.is_potion_ambient());
        assert_eq!(w.arrows_in_body(), 3);

        // A server that has sent no health must not read as dead.
        let (kind, m) = of(54, vec![]);
        assert_eq!(Watched::new(kind, &m).health(), None);
        assert_eq!(Watched::new(kind, &m).potion_colour(), 0);
    }

    /// java: `EntityAgeable.isChild` is `growingAge < 0`; `EntityZombie.isChild` is
    /// `getWatchableObjectByte(12) == 1` at the *same* index. The two readings disagree on every
    /// value except zero, so a shared branch would be wrong for one of them.
    #[test]
    fn index_twelve_means_two_incompatible_things() {
        for (net_id, byte, want) in [
            (90_u8, -25_i8, true), // pig, EntityAgeable
            (90, 0, false),
            (90, 1, false),
            (54, 1, true), // zombie, EntityZombie
            (54, -1, false),
        ] {
            let (kind, m) = of(net_id, vec![(12, Wire::Byte(byte))]);
            assert_eq!(
                Watched::new(kind, &m).is_child(),
                want,
                "entity {net_id} with age byte {byte}"
            );
        }
    }

    /// java: `EntitySheep.getFleeceColor` (`& 15`, `byMetadata`) and `getSheared` (`& 16`) share
    /// index 16. `0x14` is "sheared, yellow".
    #[test]
    fn a_sheared_yellow_sheep_decodes_from_one_byte() {
        let (kind, m) = of(91, vec![(16, Wire::Byte(0x14))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_sheared());
        assert_eq!(w.fleece_colour(), Some(DyeColour::Yellow));
        assert_eq!(w.variant(), 4, "texture_for wants the metadata nibble");
        assert_eq!(w.fleece_colour().unwrap().fleece_rgb(), [0.9, 0.9, 0.2]);

        // The same byte on a creeper must not be read as fleece — index 16 is its state byte.
        let (kind, m) = of(50, vec![(16, Wire::Byte(0x14))]);
        assert_eq!(Watched::new(kind, &m).fleece_colour(), None);
        assert!(!Watched::new(kind, &m).is_sheared());
    }

    /// The whole point of the type gate: a *charged* creeper is index 17 `== 1`, and no other
    /// mob's index 17 may answer to it.
    #[test]
    fn a_charged_creeper_decodes_and_nothing_else_claims_to_be_one() {
        let (kind, m) = of(
            50,
            vec![
                (16, Wire::Byte(1)),
                (17, Wire::Byte(1)),
                (18, Wire::Byte(1)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert!(w.is_charged());
        assert!(w.is_ignited());
        assert_eq!(w.creeper_state(), 1, "fuse lit");
        assert_eq!(w.variant(), 1, "the charged texture variant");

        // java: `entityInit` registers index 16 as `(byte)-1`, so an idle creeper the server has
        // said nothing about is -1, not 0.
        let (kind, m) = of(50, vec![]);
        let w = Watched::new(kind, &m);
        assert_eq!(w.creeper_state(), -1);
        assert!(!w.is_charged());

        // A wolf's index 17 is its owner-uuid *string*; asking it for a creeper charge is false,
        // not a panic and not a type confusion.
        let (kind, m) = of(95, vec![(17, Wire::String("owner".into()))]);
        assert!(!Watched::new(kind, &m).is_charged());
    }

    /// java: `EntityTameable.isSitting`/`isTamed` (`& 1` / `& 4`) and `EntityWolf.isAngry`
    /// (`& 2`), all in index 16; `getTailRotation` picks from the same bits plus index 18.
    ///
    /// The two forks a tamed+angry wolf hits **disagree**, and getting them the same way round
    /// is the bug this test used to encode:
    ///
    /// * `EntityWolf.getTailRotation` (`:495`) is `isAngry() ? 1.5393804F : (isTamed() ? ...)`
    ///   — **angry first**;
    /// * `RenderWolf.getEntityTexture` (`:48`) is `isTamed() ? tamed : (isAngry() ? angry : ...)`
    ///   — **tamed first**.
    ///
    /// So an angered pet keeps `wolf_tame.png` (and therefore its collar) while its tail snaps to
    /// the angry angle.
    #[test]
    fn wolf_state_comes_off_one_byte_and_its_two_forks_disagree() {
        // Exhaust the texture fork's two independent bits.  In particular 0x06 is not
        // equivalent to the merely-angry 0x02 case: Java gives the tame arm precedence.
        for (flags, expected_variant) in [(0x00, 0), (0x02, 2), (0x04, 1), (0x06, 1)] {
            let (kind, m) = of(95, vec![(16, Wire::Byte(flags))]);
            assert_eq!(
                Watched::new(kind, &m).variant(),
                expected_variant,
                "wolf flags {flags:#04x}"
            );
        }

        let (kind, m) = of(95, vec![(16, Wire::Byte(0x05)), (18, Wire::Float(20.0))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_sitting());
        assert!(w.is_tamed());
        assert!(!w.is_angry());
        assert_eq!(w.variant(), 1, "the tame texture");
        assert!((w.wolf_tail_rotation() - 0.55 * core::f32::consts::PI).abs() < 1e-5);

        // `Entity` invokes the wolf's `entityInit` before `EntityLivingBase` calls
        // `setHealth(getMaxHealth())`.  At that moment generic health index 6 is still its
        // registered 1.0F, so that is the value copied into wolf-health index 18.  This missing
        // slot case distinguishes the vanilla construction default from the old, nicer-looking
        // but incorrect full-health fallback of 20.0F.
        let (kind, m) = of(95, vec![(16, Wire::Byte(0x04))]);
        let tail = Watched::new(kind, &m).wolf_tail_rotation();
        let vanilla_default = (0.55 - (20.0 - 1.0) * 0.02) * core::f32::consts::PI;
        assert!((tail - vanilla_default).abs() < 1e-5);
        assert!(
            (tail - 0.55 * core::f32::consts::PI).abs() > 1.0,
            "the absent slot must not masquerade as a full-health wolf"
        );

        let (kind, m) = of(95, vec![(16, Wire::Byte(0x06)), (18, Wire::Float(20.0))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_angry());
        assert!(w.is_tamed());
        assert_eq!(
            w.variant(),
            1,
            "java: getEntityTexture tests isTamed() first, so a tame wolf stays wolf_tame.png"
        );
        assert!(
            (w.wolf_tail_rotation() - 1.539_380_4).abs() < 1e-6,
            "but getTailRotation tests isAngry() first"
        );

        // Wild and angry: the only case that reaches wolf_angry.png.
        let (kind, m) = of(95, vec![(16, Wire::Byte(0x02)), (18, Wire::Float(20.0))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_angry());
        assert!(!w.is_tamed());
        assert_eq!(w.variant(), 2);

        // An ocelot shares `EntityTameable`'s index 16 but has no angry bit.
        let (kind, m) = of(98, vec![(16, Wire::Byte(0x06)), (18, Wire::Byte(3))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_tamed());
        assert!(!w.is_angry());
        assert_eq!(w.variant(), 3, "the siamese skin, from index 18");
    }

    /// java: `EntityWolf.getCollarColor` reads index 20 with **`byDyeDamage`** while
    /// `EntitySheep.getFleeceColor` reads index 16 with `byMetadata`. `EnumDyeColor`'s constants
    /// give `dyeDamage = 15 - meta`, so the two schemes are mirror images and swapping them
    /// silently recolours every collar.
    #[test]
    fn the_collar_nibble_is_dye_damage_not_metadata() {
        // Dye damage 1 is RED (metadata 14) — what `setCollarColor(RED)` writes.
        let (kind, m) = of(95, vec![(20, Wire::Byte(1))]);
        assert_eq!(Watched::new(kind, &m).collar_colour(), Some(DyeColour::Red));

        // And the vanilla inconsistency: `entityInit` seeds index 20 with RED's *metadata* (14),
        // which read back through `byDyeDamage` is ORANGE. Reproduced, not "fixed".
        let (kind, m) = of(95, vec![(20, Wire::Byte(14))]);
        assert_eq!(
            Watched::new(kind, &m).collar_colour(),
            Some(DyeColour::Orange)
        );

        assert_eq!(DyeColour::from_dye_damage(1), DyeColour::Red);
        assert_eq!(DyeColour::from_metadata(1), DyeColour::Orange);
        for meta in 0..16 {
            let c = DyeColour::from_metadata(meta);
            assert_eq!(c.metadata(), meta);
            assert_eq!(c.dye_damage(), 15 - meta);
            assert_eq!(DyeColour::from_dye_damage(c.dye_damage()), c);
        }
        // java: the two lookups' out-of-range fallbacks differ — `byMetadata` falls to
        // `META_LOOKUP[0]` (white) and `byDyeDamage` to `DYE_DMG_LOOKUP[0]` (black, whose dye
        // damage is 0).
        assert_eq!(DyeColour::from_metadata(99), DyeColour::White);
        assert_eq!(DyeColour::from_dye_damage(99), DyeColour::Black);
    }

    /// java: `EntityVillager.getProfession` is an `Integer` at index 16, not a byte — a decode
    /// that read it as a byte would find nothing and leave every villager a farmer.
    #[test]
    fn the_villager_profession_is_an_int_not_a_byte() {
        let (kind, m) = of(120, vec![(16, Wire::Int(2))]);
        let w = Watched::new(kind, &m);
        assert_eq!(w.profession(), 2);
        assert_eq!(w.variant(), 2, "the priest texture");
        assert_eq!(
            mc_render::entity::manager::texture_for(EntityType::Villager, w.variant()),
            "textures/entity/villager/priest.png"
        );

        // A `Byte(2)` at the same index is the wrong wire type and must not be accepted.
        let (kind, m) = of(120, vec![(16, Wire::Byte(2))]);
        assert_eq!(Watched::new(kind, &m).profession(), 0);
    }

    /// Java's `%` truncates toward zero. This must not become `rem_euclid(5)`: a negative
    /// profession maps to farmer after `Math.max`, rather than wrapping to a later profession.
    #[test]
    fn villager_profession_uses_java_remainder_then_clamps_negatives() {
        let cases = [
            (i32::MIN, 0),
            (-11, 0),
            (-6, 0),
            (-5, 0),
            (-4, 0),
            (-1, 0),
            (0, 0),
            (1, 1),
            (4, 4),
            (5, 0),
            (6, 1),
            (9, 4),
            (10, 0),
            (i32::MAX, 2),
        ];
        for (raw, expected) in cases {
            let (kind, m) = of(120, vec![(16, Wire::Int(raw))]);
            assert_eq!(
                Watched::new(kind, &m).profession(),
                expected,
                "raw profession {raw}"
            );
        }
    }

    /// java: `EntityItemFrame.entityInit` — index 8 an `ItemStack`, index 9 a rotation byte every
    /// reader takes `% 8` of.
    #[test]
    fn an_item_frame_carries_its_stack_and_rotation() {
        let stack = mc_protocol::ItemStack {
            item_id: 358,
            count: 1,
            damage: 0,
            nbt: None,
        };
        let (kind, m) = of(
            18,
            vec![
                (8, Wire::ItemStack(Some(stack.clone()))),
                (9, Wire::Byte(11)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert_eq!(w.frame_item(), Some(&stack));
        assert_eq!(w.frame_rotation(), 3, "11 % 8");

        // An empty frame: the `-1` sentinel decodes to `None` on the wire and stays `None`.
        let (kind, m) = of(18, vec![(8, Wire::ItemStack(None))]);
        assert_eq!(Watched::new(kind, &m).frame_item(), None);
        assert_eq!(Watched::new(kind, &m).frame_rotation(), 0);
    }

    /// java: `EntityArmorStand.entityInit` (`:28-33`) — the arms and legs have **non-zero**
    /// defaults, so a stand the server has said nothing about is not in a T-pose.
    #[test]
    fn an_armour_stand_decodes_its_flags_and_pose_with_vanilla_defaults() {
        let (kind, m) = of(30, vec![]);
        let stand = Watched::new(kind, &m).armor_stand().expect("armour stand");
        assert!(!stand.small);
        assert!(!stand.show_arms);
        assert_eq!(stand.pose.head, [0.0, 0.0, 0.0]);
        assert_eq!(stand.pose.left_arm, [-10.0, 0.0, -10.0]);
        assert_eq!(stand.pose.right_arm, [-15.0, 0.0, 10.0]);
        assert_eq!(stand.pose.left_leg, [-1.0, 0.0, -1.0]);
        assert_eq!(stand.pose.right_leg, [1.0, 0.0, 1.0]);

        // The hologram configuration servers actually send: small + marker + no base plate.
        let (kind, m) = of(
            30,
            vec![
                (10, Wire::Byte(1 | 8 | 16)),
                (11, Wire::Rotations(15.0, -30.0, 0.0)),
            ],
        );
        let stand = Watched::new(kind, &m).armor_stand().expect("armour stand");
        assert!(stand.small);
        assert!(stand.no_base_plate);
        assert!(stand.marker);
        assert!(!stand.show_arms);
        assert_eq!(stand.pose.head, [15.0, -30.0, 0.0]);

        // Nothing else answers to `armor_stand`.
        let (kind, m) = of(54, vec![(10, Wire::Byte(1))]);
        assert_eq!(Watched::new(kind, &m).armor_stand(), None);
    }

    /// java: `EntityArmorStand.isChild` (`:793`) is `isSmall()`, i.e. index **10** bit 0
    /// (`:827`) — not an age byte. `RendererLivingEntity.doRender` (`:95`) copies it onto
    /// `mainModel.isChild`, and `ModelArmorStand.render` (`:100`) halves the stand for it.
    /// `is_child` had no armour-stand arm, so every small stand drew full size.
    #[test]
    fn a_small_armour_stand_is_a_child_and_the_flag_is_not_an_age_byte() {
        let (kind, m) = of(30, vec![(10, Wire::Byte(1))]);
        assert!(Watched::new(kind, &m).is_child(), "small stand");

        // The hologram byte — small plus no base plate plus marker — is still small.
        let (kind, m) = of(30, vec![(10, Wire::Byte(1 | 8 | 16))]);
        assert!(Watched::new(kind, &m).is_child());

        // Negative controls. A stand with the *other* four flags set is full size...
        let (kind, m) = of(30, vec![(10, Wire::Byte(2 | 4 | 8 | 16))]);
        assert!(
            !Watched::new(kind, &m).is_child(),
            "bit 0 alone means small"
        );
        // ...a stand the server has said nothing about is full size...
        let (kind, m) = of(30, vec![]);
        assert!(!Watched::new(kind, &m).is_child());
        // ...and index 10 means something else entirely on a mob, so the arm must be typed.
        // java: index 10 on `EntityCreeper` is `getPowered`'s slot, not a size flag.
        let (kind, m) = of(50, vec![(10, Wire::Byte(1))]);
        assert!(
            !Watched::new(kind, &m).is_child(),
            "a creeper is not a child"
        );
    }

    /// java: `EntityMinecart.entityInit` registers index 18 as **1**, not 0, and index 21 as 6;
    /// the display block is only read when index 22 says the cart has one (`:1047`).
    #[test]
    fn minecart_wobble_and_display_block_respect_their_defaults() {
        let (kind, m) = of(42, vec![]);
        let w = Watched::new(kind, &m);
        let wobble = w.damage_wobble().expect("minecart");
        assert_eq!(wobble.time_since_hit, 0);
        assert_eq!(wobble.rock_direction, 1, "entityInit registers 1, not 0");
        assert_eq!(wobble.damage_taken, 0.0);
        assert_eq!(
            w.minecart_display_block(),
            None,
            "no display tile flag -> the cart's own default, which is not a watched value"
        );

        let (kind, m) = of(
            42,
            vec![
                (17, Wire::Int(10)),
                (18, Wire::Int(-1)),
                (19, Wire::Float(40.0)),
                (20, Wire::Int(1234)),
                (22, Wire::Byte(1)),
            ],
        );
        let w = Watched::new(kind, &m);
        let wobble = w.damage_wobble().expect("minecart");
        assert_eq!(wobble.time_since_hit, 10);
        assert_eq!(wobble.rock_direction, -1);
        assert_eq!(wobble.damage_taken, 40.0);
        assert_eq!(w.minecart_display_block(), Some(1234));
        assert_eq!(
            w.minecart_display_offset(),
            Some(6),
            "index 21 defaults to 6"
        );

        // The boat shares 17/18/19 and registers no display tile.
        let (kind, m) = of(41, vec![(19, Wire::Float(5.0))]);
        assert_eq!(
            Watched::new(kind, &m).damage_wobble().unwrap().damage_taken,
            5.0
        );
        // A mob has no wobble at all.
        let (kind, m) = of(54, vec![(19, Wire::Float(5.0))]);
        assert_eq!(Watched::new(kind, &m).damage_wobble(), None);
    }

    /// `EntityMinecartCommandBlock.entityInit` registers the command and the last-output JSON as
    /// empty strings. Its `onDataWatcherUpdate` then reads **only** indices 23 and 24, and only
    /// for the command-minecart subclass; index reuse must not leak a hostile cart's strings into
    /// another minecart's future UI.
    #[test]
    fn command_minecart_metadata_is_typed_and_preserves_raw_component_json() {
        let (kind, m) = of(
            40,
            vec![
                (23, Wire::String("/say hi".into())),
                (24, Wire::String(r#"{\"text\":\"hi\"}"#.into())),
            ],
        );
        let state = Watched::new(kind, &m)
            .command_minecart()
            .expect("command cart");
        assert_eq!(state.command, "/say hi");
        assert_eq!(state.last_output_json, r#"{\"text\":\"hi\"}"#);

        // Both defaults are registered by `entityInit`; a metadata delta may omit either one.
        let (kind, m) = of(40, vec![]);
        assert_eq!(
            Watched::new(kind, &m).command_minecart(),
            Some(CommandMinecartState {
                command: "",
                last_output_json: "",
            })
        );

        // Java stores the raw string before its GUI's deliberately catch-all JSON parser, so a
        // malformed component is still meaningful metadata here.
        let (kind, m) = of(40, vec![(24, Wire::String("{broken".into()))]);
        assert_eq!(
            Watched::new(kind, &m)
                .command_minecart()
                .expect("command cart")
                .last_output_json,
            "{broken"
        );

        // Index 23 has no minecart-wide meaning: neither another cart variant nor the wrong wire
        // type may answer through this typed accessor.
        let (kind, m) = of(42, vec![(23, Wire::String("leak".into()))]);
        assert_eq!(Watched::new(kind, &m).command_minecart(), None);
        let (kind, m) = of(40, vec![(23, Wire::Int(23))]);
        assert_eq!(
            Watched::new(kind, &m)
                .command_minecart()
                .expect("command cart")
                .command,
            ""
        );
    }

    /// java: `EntityHorse.getHorseWatchableBoolean` masks an `Integer` at index 16 with 2/4/8/16/
    /// 32/64/128, and the type/variant/armour live at 19/20/22.
    #[test]
    fn horse_flags_are_an_int_bitmask() {
        let (kind, m) = of(
            100,
            vec![
                (16, Wire::Int(2 | 4 | 8)),
                (19, Wire::Byte(1)),
                (20, Wire::Int(0x0102)),
                (22, Wire::Int(2)),
            ],
        );
        let h = Watched::new(kind, &m).horse().expect("horse");
        assert!(h.tame);
        assert!(h.saddled);
        assert!(h.chested);
        assert!(!h.rearing);
        assert!(!h.mouth_open);
        assert_eq!(h.horse_type, 1, "donkey");
        assert_eq!(h.variant, 0x0102);
        assert_eq!(h.armour, 2, "gold barding");

        let (kind, m) = of(54, vec![(16, Wire::Int(2))]);
        assert_eq!(Watched::new(kind, &m).horse(), None);
    }

    /// java: the owner uuid lives at index **17** for an `EntityTameable` (wolf, ocelot) and at
    /// index **21** for `EntityHorse`, which is an `EntityAnimal` and registers its own. Reading
    /// one slot for both would leave every horse unowned and every wolf's owner unreadable.
    #[test]
    fn the_owner_uuid_is_at_a_different_index_for_a_horse() {
        let (kind, m) = of(95, vec![(17, Wire::String("wolf-owner".into()))]);
        assert_eq!(Watched::new(kind, &m).owner_id(), Some("wolf-owner"));

        let (kind, m) = of(100, vec![(21, Wire::String("horse-owner".into()))]);
        assert_eq!(Watched::new(kind, &m).owner_id(), Some("horse-owner"));

        // A horse's index 17 is `EntityMinecart`-shaped nothing; a wolf's index 21 likewise.
        let (kind, m) = of(100, vec![(17, Wire::String("wrong".into()))]);
        assert_eq!(Watched::new(kind, &m).owner_id(), None);
        let (kind, m) = of(95, vec![(21, Wire::String("wrong".into()))]);
        assert_eq!(Watched::new(kind, &m).owner_id(), None);

        // java: `""` is how both spell "unowned".
        let (kind, m) = of(95, vec![(17, Wire::String(String::new()))]);
        assert_eq!(Watched::new(kind, &m).owner_id(), None);
    }

    /// The monster flags that each own a bit of somebody's index 16 — proof the gate keeps them
    /// apart rather than letting one byte answer for all of them.
    #[test]
    fn per_monster_index_sixteen_flags_do_not_bleed_into_each_other() {
        // Every one of these is `index 16 & 1` (or `!= 0`) in its own class.
        let one = vec![(16, Wire::Byte(1))];
        /// One `index 16 & 1` predicate, named so the table below is not a "very complex type".
        type Probe = fn(Watched<'_>) -> bool;
        let cases: [(u8, Probe); 6] = [
            (52, |w| w.is_climbing()),        // spider
            (61, |w| w.is_blaze_burning()),   // blaze
            (56, |w| w.is_ghast_attacking()), // ghast
            (99, |w| w.is_player_created()),  // iron golem
            (65, |w| w.is_bat_hanging()),     // bat
            (90, |w| w.is_saddled()),         // pig
        ];
        for (net_id, probe) in cases {
            let (kind, m) = of(net_id, one.clone());
            assert!(probe(Watched::new(kind, &m)), "entity {net_id} set");
            // The same predicate on a zombie, whose index 16 means nothing at all.
            let (kind, m) = of(54, one.clone());
            assert!(
                !probe(Watched::new(kind, &m)),
                "entity {net_id} on a zombie"
            );
        }

        // The cave spider inherits `EntitySpider`'s climbing bit.
        let (kind, m) = of(59, one.clone());
        assert!(Watched::new(kind, &m).is_climbing());

        // java: `EntitySlime.entityInit` registers `(byte)1`, so an unheard-of slime is size 1.
        let (kind, m) = of(55, vec![]);
        assert_eq!(Watched::new(kind, &m).slime_size(), 1);
        let (kind, m) = of(55, vec![(16, Wire::Byte(4))]);
        assert_eq!(Watched::new(kind, &m).slime_size(), 4);

        // java: the witch is alone in watching index 21.
        let (kind, m) = of(66, vec![(21, Wire::Byte(1))]);
        assert!(Watched::new(kind, &m).is_witch_aggressive());
    }

    /// java: `EntityEnderman` — index 16 is a signed `Short` carrying the unsigned packed block
    /// state id (`block_id + (metadata << 12)`); index 17 is vestigial, and 18 is the scream flag.
    #[test]
    fn the_enderman_unpacks_its_unsigned_short_state_id() {
        let (kind, m) = of(
            58,
            vec![
                (16, Wire::Short((3 + (2 << 12)) as i16)),
                (17, Wire::Byte(7)),
                (18, Wire::Byte(1)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert_eq!(
            w.carried_block(),
            Some((3, 2)),
            "podzol's metadata comes from slot 16's high nibble, not vestigial slot 17"
        );
        assert!(w.is_screaming());

        // Metadata 15 sets the Java short's sign bit. `& 65535` in
        // `EntityEnderman.getHeldBlockState` recovers black wool's unsigned state id.
        let black_wool_state = (35_u16 + (15 << 12)) as i16;
        let (kind, m) = of(58, vec![(16, Wire::Short(black_wool_state))]);
        assert_eq!(Watched::new(kind, &m).carried_block(), Some((35, 15)));

        // Even a hostile non-zero metadata nibble still resolves block id zero through
        // `Block.getStateById`, so it is air and leaves the hands empty.
        let (kind, m) = of(58, vec![(16, Wire::Short(1 << 12))]);
        assert_eq!(
            Watched::new(kind, &m).carried_block(),
            None,
            "packed block id 0 is air regardless of its metadata nibble"
        );
    }

    /// java: `EntityGuardian` masks an `Integer` at index 16 (2 spikes, 4 elder) and holds the
    /// laser target's entity id at 17, with `0` meaning none.
    #[test]
    fn guardian_flags_and_laser_target() {
        let (kind, m) = of(68, vec![(16, Wire::Int(4 | 2)), (17, Wire::Int(4321))]);
        let w = Watched::new(kind, &m);
        assert!(w.is_elder_guardian());
        assert!(w.guardian_spikes_retracted());
        assert_eq!(w.guardian_target(), Some(4321));

        let (kind, m) = of(68, vec![(17, Wire::Int(0))]);
        assert_eq!(Watched::new(kind, &m).guardian_target(), None);
    }

    /// `EntityWither` registers targets in 17..=19 and invulnerability in 20 as integers.
    /// The texture cadence is intentionally checked at both sides of its five-tick bands.
    #[test]
    fn wither_watcher_slots_preserve_target_and_invulnerability_semantics() {
        let (kind, m) = of(
            64,
            vec![
                (17, Wire::Int(100)),
                (18, Wire::Int(200)),
                (19, Wire::Int(-7)),
                (20, Wire::Int(5)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert_eq!(w.wither_target(0), Some(100));
        assert_eq!(w.wither_target(1), Some(200));
        assert_eq!(w.wither_target(2), None, "only positive ids are targets");
        assert_eq!(w.wither_target(3), None, "there are exactly three heads");
        assert_eq!(w.wither_invul_time(), 5);
        assert!(
            !w.is_wither_invulnerable_texture(),
            "tick 5 starts an off band"
        );
        assert_eq!(w.variant(), 0, "the off band uses the normal wither sheet");

        for (time, textured) in [
            (0, false),
            (1, true),
            (4, true),
            (5, false),
            (9, false),
            (10, true),
            (79, false),
            (80, true),
            (81, true),
            (220, true),
        ] {
            let (kind, m) = of(64, vec![(20, Wire::Int(time))]);
            let w = Watched::new(kind, &m);
            assert_eq!(
                w.is_wither_invulnerable_texture(),
                textured,
                "invulnerability tick {time}"
            );
            assert_eq!(w.variant(), i32::from(textured));
        }

        // A wolf can carry the same raw slots, but those slots belong to the wither alone.
        let (kind, m) = of(95, vec![(20, Wire::Int(81))]);
        let w = Watched::new(kind, &m);
        assert_eq!(w.wither_target(0), None);
        assert_eq!(w.wither_invul_time(), 0);
        assert!(!w.is_wither_invulnerable_texture());
    }

    /// java: `EntityZombie` re-registers 12/13/14 for child/villager/converting, all `== 1`.
    #[test]
    fn zombie_reuses_three_low_indices() {
        let (kind, m) = of(
            54,
            vec![
                (12, Wire::Byte(1)),
                (13, Wire::Byte(1)),
                (14, Wire::Byte(1)),
            ],
        );
        let w = Watched::new(kind, &m);
        assert!(w.is_child());
        assert!(w.is_zombie_villager());
        assert!(w.is_converting());
    }

    /// java: `EntityArrow.getIsCritical` (`& 1`) and `EntityItem.getEntityItem` (index 10).
    #[test]
    fn projectiles_and_dropped_items() {
        let (kind, m) = of(10, vec![(16, Wire::Byte(1))]);
        assert!(Watched::new(kind, &m).is_arrow_critical());

        let stack = mc_protocol::ItemStack {
            item_id: 264,
            count: 12,
            damage: 0,
            nbt: None,
        };
        let (kind, m) = of(1, vec![(10, Wire::ItemStack(Some(stack.clone())))]);
        assert_eq!(Watched::new(kind, &m).dropped_item(), Some(&stack));

        // The item frame's stack lives at 8, not 10, so a frame must not answer `dropped_item`.
        let (kind, m) = of(18, vec![(8, Wire::ItemStack(Some(stack)))]);
        assert_eq!(Watched::new(kind, &m).dropped_item(), None);
    }

    /// java: `EntityPlayer.entityInit` — index 10 model parts, 17 absorption, 18 score. A mob
    /// must not answer any of them: index 10 is the wither skull's charge byte, 17 the wolf's
    /// owner string.
    #[test]
    fn player_only_indices_are_gated_on_being_a_player() {
        let entries = vec![
            (10, Wire::Byte(0b0100_0001)),
            (17, Wire::Float(4.0)),
            (18, Wire::Int(77)),
        ];
        let m = meta(entries.clone());
        let w = Watched::new(TrackedKind::Player, &m);
        assert!(w.skin_parts().cape());
        assert!(w.skin_parts().hat());
        assert!(!w.skin_parts().jacket());
        assert_eq!(w.absorption(), 4.0);
        assert_eq!(w.score(), 77);

        let (kind, m) = of(54, entries);
        let w = Watched::new(kind, &m);
        assert_eq!(w.skin_parts().bits(), 0);
        assert_eq!(w.absorption(), 0.0);
        assert_eq!(w.score(), 0);
    }

    #[test]
    fn typed_accessors_bite_raw_slot_bypasses() {
        let stack = mc_protocol::ItemStack {
            item_id: 264,
            count: 1,
            damage: 0,
            nbt: None,
        };
        // A frame carrying a dropped-item-looking slot must not expose it as its own stack, and
        // a dropped item carrying a frame stack must not become a frame. The raw index is shared
        // by unrelated classes, so the kind gate is the part this test would catch removing.
        let (frame, frame_meta) = of(
            18,
            vec![
                (10, Wire::ItemStack(Some(stack.clone()))),
                (9, Wire::Byte(7)),
            ],
        );
        let frame_watched = Watched::new(frame, &frame_meta);
        assert_eq!(frame_watched.frame_item(), None);
        assert_eq!(frame_watched.frame_rotation(), 7);

        let (item, item_meta) = of(
            1,
            vec![(8, Wire::ItemStack(Some(stack))), (9, Wire::Byte(7))],
        );
        let item_watched = Watched::new(item, &item_meta);
        assert_eq!(item_watched.dropped_item(), None);
        assert_eq!(item_watched.frame_rotation(), 0);
    }

    /// GAP A10-06's exact shape: the variant a caller hands `texture_for`. Before A10-01 this was
    /// the literal `0` for every entity in the world.
    #[test]
    fn the_variant_matches_what_texture_for_expects() {
        let cases = [
            (
                120,
                vec![(16, Wire::Int(4))],
                "textures/entity/villager/butcher.png",
            ),
            (98, vec![(18, Wire::Byte(2))], "textures/entity/cat/red.png"),
            (
                95,
                vec![(16, Wire::Byte(0x04))],
                "textures/entity/wolf/wolf_tame.png",
            ),
            (
                101,
                vec![(18, Wire::Byte(1))],
                "textures/entity/rabbit/white.png",
            ),
            (
                51,
                vec![(13, Wire::Byte(1))],
                "textures/entity/skeleton/wither_skeleton.png",
            ),
        ];
        for (net_id, entries, want) in cases {
            let (kind, m) = of(net_id, entries);
            let t = kind.entity_type().unwrap();
            let got = mc_render::entity::manager::texture_for(t, Watched::new(kind, &m).variant());
            assert_eq!(got, want, "entity {net_id}");
        }
    }

    /// The view must reach a live tracked entity, not only a hand-built `Metadata`.
    #[test]
    fn with_watched_reads_the_ecs_component() {
        use super::super::entities::EntityTracker;
        use mc_math::DVec3;
        use mc_world::World;

        let mut world = World::new();
        let mut tracker = EntityTracker::new();
        tracker.spawn(
            &mut world,
            7,
            TrackedKind::Type(entity_type(50).expect("creeper")),
            None,
            [0, 0, 0],
            0.0,
            0.0,
            DVec3::ZERO,
            Stored::new(),
        );
        assert_eq!(
            tracker.with_watched(&world, 7, |w| w.is_charged()),
            Some(false)
        );

        assert!(tracker.apply_metadata(
            &mut world,
            7,
            &vec![MetadataEntry {
                index: 17,
                value: Wire::Byte(1),
            }],
        ));
        assert_eq!(
            tracker.with_watched(&world, 7, |w| w.is_charged()),
            Some(true),
            "a real S1CPacketEntityMetadata payload must reach the interpretation"
        );
        assert_eq!(tracker.with_watched(&world, 99, |w| w.is_charged()), None);
    }
}
