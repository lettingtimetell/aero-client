//! Client-side tile-entity storage.
//!
//! The `World` accessors for this store live here in their own `impl World { .. }` block rather
//! than in `world.rs`, so the whole tile-entity half of the world reads as one file. Only the
//! field itself and its initialiser sit in `world.rs`, where the struct is declared.
//!
//! # What a 1.8.9 client actually knows about a tile entity
//!
//! Two independent halves, and conflating them is the mistake this module exists to avoid:
//!
//! * **Which** tile entity is at a position comes from the **block**, never from the wire.
//!   `Chunk.getTileEntity(pos, CHECK)` (`world/chunk/Chunk.java:918`) looks in
//!   `chunkTileEntityMap` and, on a miss, calls `createNewTileEntity` (`:912`) — which is
//!   `((ITileEntityProvider) block).createNewTileEntity(...)`, i.e. a pure function of the block
//!   id. So a chest is a chest the instant its block arrives in a chunk packet, with no
//!   follow-up packet needed. [`TileKind::from_block`] is that function.
//! * **Extra state** (a sign's text, a banner's patterns, a skull's owner, a spawner's mob, a
//!   flower pot's plant, a beacon's levels) only ever arrives as a *separate* description packet
//!   pushed right behind the chunk (`server/management/PlayerManager.java:526-530` →
//!   `TileEntity.getDescriptionPacket()`); 1.8.9's `S21PacketChunkData` carries no tile-entity
//!   list at all. That is what [`World::apply_tile_entity_nbt`] and
//!   [`World::apply_sign_text`] land, and what this store holds.
//!
//! The consequence for the renderer is the important one: the store being **empty** must still
//! draw a chest, because vanilla's map is empty for chests too — the server never sends a chest
//! description packet. Anything that walks tile entities for rendering therefore keys off
//! [`World::tile_kind_at`] (the block) and treats the store as a source of *decoration*.
//!
//! # NBT stays out of this crate
//!
//! `mc-world` does not depend on `mc-nbt` (see `Cargo.toml`, which the Wave-1 rules freeze), so
//! the decode of an `S35PacketUpdateTileEntity` compound happens in `mc-client`, which already
//! has both crates, and reaches this store through the small typed setters below
//! ([`SkullState`], [`BannerState`], [`SpawnerState`], [`FlowerPotState`]). That is also the
//! cleaner split: `readFromNBT` in vanilla is per-class anyway.

use std::collections::HashMap;

use mc_data::{block_info, BlockId, BlockState, Material};
use mc_math::{Aabb, BlockPos, ChunkPos, DVec3, Facing, JavaRandom};

use crate::world::World;

/// Block ids that carry a tile entity this client can draw or animate.
///
/// Java's complete 27-block `ITileEntityProvider` census is represented here. The 21
/// `TileEntity.addMapping` registry entries expand to 24 block-facing kinds because some classes
/// serve multiple block ids with distinct block behavior; every provider is still mapped below.
/// The eight classes without a TESR or S35 action currently use `TileData::None`: inventory and
/// window-property state is owned by the container protocol, but their provider identity and NBT
/// registry id remain part of the world-side contract.
///
/// Note this is **not** the same set as `instanceof ITileEntityProvider`, which
/// `BlockPistonBase.canPush` tests: `BlockRedstoneComparator` implements that interface without
/// extending `BlockContainer`; it sets `isBlockContainer` itself, so `Block.hasTileEntity()` is
/// true for it as well. The piston-specific movement question is answered in
/// [`piston_can_push`], not here.
///
/// Deliberately **not** `#[non_exhaustive]`: the renderer's dispatch over this enum is the whole
/// of `TileEntityRendererDispatcher`, and a wildcard arm there is how a new tile entity becomes
/// an invisible block. Same reasoning as `driver.rs`'s `ClientEvent` match (GAP `T2`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TileKind {
    /// java: `TileEntityDispenser` on `BlockDispenser` (block 23).
    Dispenser,
    /// java: `TileEntityChest` on `BlockChest` with `chestType == 0` (block 54).
    Chest,
    /// java: `TileEntityChest` on `BlockChest` with `chestType == 1` (block 146).
    TrappedChest,
    /// java: `TileEntityEnderChest` (block 130).
    EnderChest,
    /// java: `TileEntityFurnace` on `BlockFurnace` (blocks 61 and 62).
    Furnace,
    /// java: `BlockJukebox.TileEntityJukebox` (block 84).
    Jukebox,
    /// java: `TileEntitySign` on `BlockStandingSign` (block 63).
    StandingSign,
    /// java: `TileEntitySign` on `BlockWallSign` (block 68).
    WallSign,
    /// java: `TileEntityBanner` on `BlockBannerStanding` (block 176).
    StandingBanner,
    /// java: `TileEntityBanner` on `BlockBannerHanging` (block 177).
    WallBanner,
    /// java: `TileEntitySkull` (block 144).
    Skull,
    /// java: `TileEntityBeacon` (block 138).
    Beacon,
    /// java: `TileEntityMobSpawner` (block 52).
    MobSpawner,
    /// java: `TileEntityEnchantmentTable` (block 116).
    EnchantmentTable,
    /// java: `TileEntityEndPortal` (block 119).
    EndPortal,
    /// java: `TileEntityFlowerPot` (block 140).
    FlowerPot,
    /// java: `TileEntityBrewingStand` on `BlockBrewingStand` (block 117).
    BrewingStand,
    /// java: `TileEntityComparator` on `BlockRedstoneComparator` (blocks 149 and 150).
    Comparator,
    /// java: `TileEntityDaylightDetector` on `BlockDaylightDetector` (blocks 151 and 178).
    DaylightDetector,
    /// java: `TileEntityHopper` on `BlockHopper` (block 154).
    Hopper,
    /// java: `TileEntityDropper` on `BlockDropper` (block 158).
    Dropper,
    /// java: `TileEntityPiston` on `BlockPistonMoving` (block 36).
    Piston,
    /// java: `TileEntityNote` (block 25). No TESR — it only answers a `BlockAction`.
    Note,
    /// java: `TileEntityCommandBlock` (block 137). No TESR either; kept because it is one of the
    /// six `S35PacketUpdateTileEntity` action bytes and dropping it silently would look like a
    /// decode bug the next time somebody reads the action switch.
    CommandBlock,
}

/// Block 54 — `minecraft:chest`.
pub const CHEST_ID: u16 = 54;
/// Block 146 — `minecraft:trapped_chest`.
pub const TRAPPED_CHEST_ID: u16 = 146;
/// Block 130 — `minecraft:ender_chest`.
pub const ENDER_CHEST_ID: u16 = 130;
/// Block 95 — `minecraft:stained_glass`.
pub const STAINED_GLASS_ID: u16 = 95;
/// Block 160 — `minecraft:stained_glass_pane`.
pub const STAINED_GLASS_PANE_ID: u16 = 160;
/// Block 7 — `minecraft:bedrock`, the one opaque block a beacon beam passes through.
pub const BEDROCK_ID: u16 = 7;
/// Block 29 — `minecraft:sticky_piston`.
pub const STICKY_PISTON_ID: u16 = 29;
/// Block 33 — `minecraft:piston`.
pub const PISTON_ID: u16 = 33;
/// Block 34 — `minecraft:piston_head` (`BlockPistonExtension`).
pub const PISTON_HEAD_ID: u16 = 34;
/// Block 36 — `minecraft:piston_extension` (`BlockPistonMoving`), the moving-block carrier.
pub const PISTON_EXTENSION_ID: u16 = 36;
/// Block 49 — `minecraft:obsidian`, the one block `BlockPistonBase.canPush` rejects by identity.
pub const OBSIDIAN_ID: u16 = 49;
/// Block 165 — `minecraft:slime`, the slime block.
pub const SLIME_BLOCK_ID: u16 = 165;
/// java: `BlockPistonStructureHelper`'s hard cap on how many blocks one push may carry.
const PISTON_PUSH_LIMIT: usize = 12;

impl TileKind {
    /// The tile entity a block id creates, or `None` for a block that has none.
    ///
    /// java: `Chunk.createNewTileEntity` (`world/chunk/Chunk.java:912`) —
    /// `!block.hasTileEntity() ? null : ((ITileEntityProvider) block).createNewTileEntity(...)`.
    /// A pure function of the block, which is why the client can draw a chest it was never sent
    /// a packet about.
    #[must_use]
    pub fn from_block(block: BlockId) -> Option<TileKind> {
        Some(match block.0 {
            23 => TileKind::Dispenser,
            25 => TileKind::Note,
            36 => TileKind::Piston,
            52 => TileKind::MobSpawner,
            CHEST_ID => TileKind::Chest,
            61 | 62 => TileKind::Furnace,
            63 => TileKind::StandingSign,
            68 => TileKind::WallSign,
            84 => TileKind::Jukebox,
            116 => TileKind::EnchantmentTable,
            117 => TileKind::BrewingStand,
            119 => TileKind::EndPortal,
            ENDER_CHEST_ID => TileKind::EnderChest,
            137 => TileKind::CommandBlock,
            138 => TileKind::Beacon,
            140 => TileKind::FlowerPot,
            144 => TileKind::Skull,
            TRAPPED_CHEST_ID => TileKind::TrappedChest,
            149 | 150 => TileKind::Comparator,
            151 | 178 => TileKind::DaylightDetector,
            154 => TileKind::Hopper,
            158 => TileKind::Dropper,
            176 => TileKind::StandingBanner,
            177 => TileKind::WallBanner,
            _ => return None,
        })
    }

    /// The registry `id` string vanilla writes into the NBT compound.
    ///
    /// java: the second argument of each `TileEntity.addMapping` call (`:295-315`). Exposed so a
    /// caller reading a description packet can check the tag against the block it landed on the
    /// way vanilla's `instanceof` chain does.
    #[must_use]
    pub fn id_string(self) -> &'static str {
        match self {
            TileKind::Dispenser => "Trap",
            TileKind::Chest | TileKind::TrappedChest => "Chest",
            TileKind::EnderChest => "EnderChest",
            TileKind::Furnace => "Furnace",
            TileKind::Jukebox => "RecordPlayer",
            TileKind::StandingSign | TileKind::WallSign => "Sign",
            TileKind::StandingBanner | TileKind::WallBanner => "Banner",
            TileKind::Skull => "Skull",
            TileKind::Beacon => "Beacon",
            TileKind::MobSpawner => "MobSpawner",
            TileKind::EnchantmentTable => "EnchantTable",
            TileKind::EndPortal => "Airportal",
            TileKind::FlowerPot => "FlowerPot",
            TileKind::BrewingStand => "Cauldron",
            TileKind::Comparator => "Comparator",
            TileKind::DaylightDetector => "DLDetector",
            TileKind::Hopper => "Hopper",
            TileKind::Dropper => "Dropper",
            TileKind::Piston => "Piston",
            TileKind::Note => "Music",
            TileKind::CommandBlock => "Control",
        }
    }

    /// The `S35PacketUpdateTileEntity` action byte whose payload this kind accepts, if any.
    ///
    /// java: `NetHandlerPlayClient.handleUpdateTileEntity` (`:1276`) is a six-way
    /// `action && instanceof` conjunction — `1` mob spawner, `2` command block, `3` beacon,
    /// `4` skull, `5` flower pot, `6` banner. **Nothing else** accepts an `S35`, which is why a
    /// sign's text has its own packet (`S33`) and a chest's lid has none at all.
    #[must_use]
    pub fn description_action(self) -> Option<u8> {
        Some(match self {
            TileKind::MobSpawner => 1,
            TileKind::CommandBlock => 2,
            TileKind::Beacon => 3,
            TileKind::Skull => 4,
            TileKind::FlowerPot => 5,
            TileKind::StandingBanner | TileKind::WallBanner => 6,
            _ => return None,
        })
    }

    /// True for the three chest-shaped kinds, which share [`ChestState`].
    #[must_use]
    pub fn is_chest(self) -> bool {
        matches!(
            self,
            TileKind::Chest | TileKind::TrappedChest | TileKind::EnderChest
        )
    }

    /// Whether `Chunk.getTileEntity`'s lazy-create arm produces one of these.
    ///
    /// java: `Chunk.createNewTileEntity` (`world/chunk/Chunk.java:912`) is
    /// `!block.hasTileEntity() ? null : ((ITileEntityProvider) block).createNewTileEntity(..)`.
    /// [`TileKind::from_block`] answers the first half — "is this block an
    /// `ITileEntityProvider`?", which gates `Chunk.addTileEntity` and therefore
    /// [`World::set_tile_entity`]. This answers the second: **`BlockPistonMoving`
    /// (`block/BlockPistonMoving.java:38-41`) returns `null`**, so a `piston_extension` block on
    /// its own has no tile entity, and the one it gets comes only from
    /// `BlockPistonBase.onBlockEventReceived`'s explicit `setTileEntity`. That null is precisely
    /// why `BlockPistonMoving.getCollisionBoundingBox` and `setBlockBoundsBasedOnState` have
    /// no-tile-entity branches at all.
    #[must_use]
    pub fn created_lazily(self) -> bool {
        !matches!(self, TileKind::Piston)
    }

    /// `TileEntity.getMaxRenderDistanceSquared()`, in blocks².
    ///
    /// java: `tileentity/TileEntity.java:171` returns `4096` for everything;
    /// `TileEntityBeacon.java:258` overrides it to `65536` so a 256-block beam stays visible far
    /// outside the normal 64-block TESR range.
    #[must_use]
    pub fn max_render_distance_sq(self) -> f64 {
        match self {
            TileKind::Beacon => 65536.0,
            _ => 4096.0,
        }
    }
}

/// The scan's registry-safe tile kind lookup. `TileKind::from_block` is a cheap block-id match;
/// only provider ids need the legality check that `ChunkSection::block` otherwise pays for every
/// cell, including air and ordinary terrain.
#[inline]
fn tile_kind_from_raw_state(raw: BlockState) -> Option<TileKind> {
    let block = raw.block();
    if block == BlockId::AIR {
        return None;
    }
    let kind = TileKind::from_block(block)?;
    raw.is_legal().then_some(kind)
}

/// java: `TileEntityChest`'s three animation fields plus its four adjacency links.
///
/// `num_players_using` is the **only** input the lid animation has, and the only way a client
/// ever learns it is `S24PacketBlockAction` → `receiveClientEvent(1, n)`
/// (`TileEntityChest.java:420-426`).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ChestState {
    /// java: `numPlayersUsing`
    pub num_players_using: i32,
    /// java: `lidAngle`, `0.0..=1.0`
    pub lid_angle: f32,
    /// java: `prevLidAngle`
    pub prev_lid_angle: f32,
}

/// java: `TileEntitySign.signText` plus `lineBeingEdited`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SignState {
    /// The four lines, already reduced to display text by the caller (which owns the
    /// chat-component parser). java: `IChatComponent[] signText`, rendered through
    /// `getFormattedText()`.
    pub lines: [String; 4],
    /// The original serialised chat components.  The renderer uses [`Self::lines`], but the
    /// editor must send untouched rows back verbatim: `GuiEditSign` only replaces a component
    /// after a key event targets that row.
    pub components: [String; 4],
    /// java: `TileEntitySign.isEditable`.  `GuiEditSign.initGui` clears this and
    /// `onGuiClosed` restores it, so an S33 arriving mid-edit is acknowledged but ignored.
    pub editable: bool,
    /// java: `lineBeingEdited` — `-1` unless the edit GUI is open on this sign, in which case
    /// that line renders as `"> " + s + " <"` (`TileEntitySignRenderer.java:96-99`).
    pub line_being_edited: i32,
}

impl Default for SignState {
    fn default() -> Self {
        Self {
            lines: Default::default(),
            components: core::array::from_fn(|_| "\"\"".to_owned()),
            editable: true,
            line_being_edited: -1,
        }
    }
}

/// java: `TileEntityCommandBlock.commandBlockLogic`'s client-visible fields.
///
/// `last_output` retains the exact `LastOutput` JSON string because the chat-component parser
/// belongs to the client/UI layer. This keeps the world crate independent of `mc-ui` while still
/// preserving the payload a future `GuiCommandBlock` reader needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommandBlockState {
    /// java: `CommandBlockLogic.commandStored` / NBT `Command`.
    pub command: String,
    /// java: `CommandBlockLogic.customName` / NBT `CustomName`.
    pub custom_name: String,
    /// java: `CommandBlockLogic.successCount` / NBT `SuccessCount`.
    pub success_count: i32,
    /// java: `CommandBlockLogic.trackOutput` / NBT `TrackOutput`.
    pub track_output: bool,
    /// java: `CommandBlockLogic.lastOutput` / NBT `LastOutput`, serialized as component JSON.
    pub last_output: Option<String>,
}

impl Default for CommandBlockState {
    /// java: `CommandBlockLogic` field initializers (`commandStored = ""`, `customName = "@"`,
    /// `successCount = 0`, `trackOutput = true`, `lastOutput = null`).
    fn default() -> Self {
        Self {
            command: String::new(),
            custom_name: "@".to_owned(),
            success_count: 0,
            track_output: true,
            last_output: None,
        }
    }
}

/// java: `TileEntityBanner`'s `Base`, `Patterns` and `field_175119_g`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BannerState {
    /// java: `baseColor` — an `EnumDyeColor` dye-damage value, `0..=15`.
    pub base_color: i32,
    /// java: the `Patterns` list, flattened to `(hashname, colour)` pairs in list order.
    ///
    /// The runtime `LayeredColorMaskTexture` compositor consumes this list in order. Unknown
    /// ids are discarded by the packet reader, matching `EnumBannerPattern.getPatternByID`.
    pub patterns: Vec<(String, i32)>,
    /// java: `field_175119_g`, set by `readFromNBT`/`setItemValues` and initially false.
    ///
    /// `TileEntityBannerRenderer` returns no texture while this is false, so a banner block
    /// created from a chunk before its S35 action-6 description is intentionally invisible for
    /// that short interval.
    pub initialized: bool,
}

/// The complete 1.8.9 `TileEntityBanner.EnumBannerPattern` table.
///
/// Each row is `(pattern id carried in NBT, mask texture basename)`.  The Java enum's crafting
/// recipes are server-side and do not affect a client renderer, so only these two fields cross
/// the state-to-render seam.  Keep this exhaustive: `getPatternByID` linearly searches all 39.
pub const BANNER_PATTERN_TABLE: &[(&str, &str)] = &[
    ("b", "base"),
    ("bl", "square_bottom_left"),
    ("br", "square_bottom_right"),
    ("tl", "square_top_left"),
    ("tr", "square_top_right"),
    ("bs", "stripe_bottom"),
    ("ts", "stripe_top"),
    ("ls", "stripe_left"),
    ("rs", "stripe_right"),
    ("cs", "stripe_center"),
    ("ms", "stripe_middle"),
    ("drs", "stripe_downright"),
    ("dls", "stripe_downleft"),
    ("ss", "small_stripes"),
    ("cr", "cross"),
    ("sc", "straight_cross"),
    ("bt", "triangle_bottom"),
    ("tt", "triangle_top"),
    ("bts", "triangles_bottom"),
    ("tts", "triangles_top"),
    ("ld", "diagonal_left"),
    ("rd", "diagonal_up_right"),
    ("lud", "diagonal_up_left"),
    ("rud", "diagonal_right"),
    ("mc", "circle"),
    ("mr", "rhombus"),
    ("vh", "half_vertical"),
    ("hh", "half_horizontal"),
    ("vhr", "half_vertical_right"),
    ("hhb", "half_horizontal_bottom"),
    ("bo", "border"),
    ("cbo", "curly_border"),
    ("cre", "creeper"),
    ("gra", "gradient"),
    ("gru", "gradient_up"),
    ("bri", "bricks"),
    ("sku", "skull"),
    ("flo", "flower"),
    ("moj", "mojang"),
];

/// java: `TileEntityBanner.EnumBannerPattern.getPatternByID`.
#[must_use]
pub fn banner_pattern_name(id: &str) -> Option<&'static str> {
    BANNER_PATTERN_TABLE
        .iter()
        .find_map(|(pattern_id, name)| (*pattern_id == id).then_some(*name))
}

/// java: `TileEntitySkull`'s `SkullType`, `Rot` and `Owner`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SkullState {
    /// java: `skullType`, `0..=4`.
    pub skull_type: u8,
    /// java: `skullRotation`, `0..=15`; the renderer turns it into `rot * 360 / 16` degrees.
    pub rotation: u8,
    /// java: `playerProfile.getName()`, from the `Owner` compound or the legacy `ExtraType`
    /// string.
    pub owner_name: Option<String>,
    /// The complete authlib profile decoded from the `Owner` compound, including its UUID and
    /// every signed or unsigned property. java: `TileEntitySkull.playerProfile`.
    pub owner_profile: Option<mc_auth::GameProfile>,
    /// The skin URL embedded in `Owner.Properties.textures[0].Value`'s base64 payload.
    ///
    /// This is the part worth having: it needs no session-server round trip, so a player head's
    /// real face is reachable offline. Resolving it to pixels still needs a download, which is
    /// not verifiable in this sandbox — see the report.
    pub skin_url: Option<String>,
}

/// One beacon beam segment. java: `TileEntityBeacon.BeamSegment`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BeamSegment {
    /// java: `getColors()` — RGB in `0..1`, already averaged across the stained glass above.
    pub colors: [f32; 3],
    /// java: `getHeight()` — how many blocks this colour runs for.
    pub height: i32,
}

/// java: `TileEntityBeacon`'s client-visible state.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BeaconState {
    /// java: `levels`, `0..=4`. Only ever arrives in an `S35` action-3 payload.
    pub levels: i32,
    /// java: `beamSegments`, rebuilt by [`World::scan_beacon_beam`].
    pub segments: Vec<BeamSegment>,
    /// java: `isComplete` — false aborts the beam entirely.
    pub complete: bool,
    /// java: `field_146014_j`, the per-render-call fade `shouldBeamRender()` returns.
    pub beam_fade: f32,
    /// java: `beamRenderCounter`, the world tick of the last successful `shouldBeamRender()`.
    /// Incomplete and out-of-range beacons deliberately leave it untouched so re-entry applies
    /// vanilla's elapsed-tick decay before the next `+0.025` frame step.
    pub beam_render_counter: i64,
}

/// java: `MobSpawnerBaseLogic`'s client half.
///
/// `SpawnData` is deliberately represented by a small, NBT-free value.  `mc-world` is kept
/// independent of the NBT decoder, while the client still needs the entity fields that vanilla
/// applies to the cached miniature before handing it to `RenderManager`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnerEquipment {
    /// `ItemStack.id`, resolved to the 1.8.9 numeric registry id.
    pub item_id: u16,
    /// `ItemStack.Damage` (negative values are clamped by Java's reader).
    pub damage: i16,
    /// `ItemArmor.getColor`'s decoded leather colour, when this stack carries one.
    pub leather_color: Option<u32>,
    /// Whether `ItemStack.hasEffect()` sees an enchantment list on this stack.
    pub enchanted: bool,
    /// `ItemStack.tag.SkullOwner`, when the stack is a player skull with a profile payload.
    ///
    /// java: `LayerCustomHead`/`TileEntityItemStackRenderer` pass this authlib profile through
    /// to `TileEntitySkullRenderer`; keeping it on the typed spawner snapshot avoids making the
    /// world crate depend on the NBT decoder while preserving UUID/name/properties exactly.
    pub skull_owner: Option<mc_auth::GameProfile>,
    /// `TileEntityBanner.setItemValues`' decoded `BlockEntityTag`, when this is a banner item.
    ///
    /// Keeping the state here lets the spawner's held/custom-head item paths use the same
    /// layered banner compositor as the ordinary item renderer without retaining raw NBT.
    pub banner_state: Option<BannerState>,
}

/// The client-visible subset of a spawner `SpawnData` compound.
///
/// java: `MobSpawnerBaseLogic.spawnNewEntity` overlays every SpawnData key onto the newly
/// constructed entity's NBT and calls `readFromNBT` before `TileEntityMobSpawnerRenderer` draws
/// it.  These fields are the appearance/state inputs consumed by the existing entity renderer;
/// gameplay-only fields (AI, health, drops, leash, attributes, and timers) are intentionally not
/// copied into the tile store.
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnerSpawnData {
    /// The renderer variant selected from the entity's NBT (villager zombie, skeleton type,
    /// rabbit skin, horse type, slime size, and similar stateful renderer branches).
    pub variant: i32,
    /// `EntityLivingBase.isChild()` after the entity NBT overlay.
    pub child: bool,
    /// `Entity.isInvisible()`.
    pub invisible: bool,
    /// `EntitySheep.getSheared()`.
    pub sheared: bool,
    /// `EntityCreeper.getPowered()`.
    pub powered: bool,
    /// `EntityPig.getSaddled()`.
    pub saddled: bool,
    /// `EntityTameable.isTamed()`.
    pub tamed: bool,
    /// `EntityWolf.isAngry()`.
    pub angry: bool,
    /// `EntityTameable.isSitting()`.
    pub sitting: bool,
    /// `EntitySheep`/`EntityWolf` dye metadata used by their texture layers.
    pub dye_color: Option<u8>,
    /// The horse's packed colour/markings `Variant` field, kept separate from its `Type`.
    pub horse_variant: i32,
    /// `EntityHorse`'s decoded `ArmorItem`: 0 none, 1 iron, 2 gold, 3 diamond.
    pub horse_armor: i32,
    /// `EntityHorse`'s `Saddle`/`ChestedHorse` flags.
    pub horse_chested: bool,
    /// `Entity.isRiding()` after `MobSpawnerBaseLogic.spawnNewEntity` overlays a nested
    /// `Riding` compound onto the cached outer entity.
    ///
    /// The vanilla spawner TESR renders only that returned outer entity; the nested mount is
    /// attached for pose/yaw state but is not recursively submitted as a second miniature.
    pub riding: bool,
    /// `EntityWither.isArmored()` resolved from the SpawnData health overlay. A missing health
    /// tag retains the constructor's full 300-health default and therefore leaves the aura off.
    pub wither_armored: bool,
    /// `EntityEnderman.getHeldBlockState()`, when the NBT supplied a valid block id.
    pub carried_block: Option<BlockState>,
    /// `EntityLiving.equipment`: slot 0 held item, slots 1..=4 boots through helmet.
    pub equipment: [Option<SpawnerEquipment>; 5],
}

impl Default for SpawnerSpawnData {
    fn default() -> Self {
        Self {
            variant: 0,
            child: false,
            invisible: false,
            sheared: false,
            powered: false,
            saddled: false,
            tamed: false,
            angry: false,
            sitting: false,
            dye_color: None,
            horse_variant: 0,
            horse_armor: 0,
            horse_chested: false,
            riding: false,
            wither_armored: false,
            carried_block: None,
            equipment: core::array::from_fn(|_| None),
        }
    }
}

/// java: `MobSpawnerBaseLogic`'s client-visible state.
#[derive(Clone, Debug, PartialEq)]
pub struct SpawnerState {
    /// java: `mobID` — the `EntityId` string from the NBT.
    pub entity_id: String,
    /// java: `spawnDelay` — the `Delay` short. Drives how fast the miniature spins.
    pub spawn_delay: i32,
    /// java: `minSpawnDelay` — the `MinSpawnDelay` short, and the value `setDelayToMin` snaps
    /// `spawnDelay` to when the server's `resetTimer` fires block event 1.
    pub min_spawn_delay: i32,
    /// java: `activatingRangeFromPlayer` — the `RequiredPlayerRange` short. `isActivated()`
    /// gates the whole of `updateSpawner`, animation included.
    pub activating_range: i32,
    /// java: `mobRotation`
    pub mob_rotation: f64,
    /// java: `prevMobRotation`
    pub prev_mob_rotation: f64,
    /// java: `MobSpawnerBaseLogic.randomEntity.nbtData`, reduced to the render-visible fields
    /// applied by `spawnNewEntity` to the cached miniature.
    pub spawn_data: Option<SpawnerSpawnData>,
}

impl Default for SpawnerState {
    /// java: `MobSpawnerBaseLogic`'s field initialisers (`tileentity/MobSpawnerBaseLogic.java:23-45`).
    ///
    /// `mobID` is **`"Pig"`**, not the empty string: a spawner the client has been sent no
    /// description packet for still renders a pig.
    fn default() -> SpawnerState {
        SpawnerState {
            entity_id: "Pig".to_owned(),
            spawn_delay: 20,
            min_spawn_delay: 200,
            activating_range: 16,
            mob_rotation: 0.0,
            prev_mob_rotation: 0.0,
            spawn_data: None,
        }
    }
}

/// java: `TileEntityEnchantmentTable`'s six paired animation fields plus its tick counter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnchantState {
    /// java: `tickCount`
    pub tick_count: i32,
    /// java: `pageFlip`
    pub page_flip: f32,
    /// java: `pageFlipPrev`
    pub page_flip_prev: f32,
    /// java: `field_145929_l` — the page-flip velocity.
    pub flip_velocity: f32,
    /// java: `field_145932_k` — the page-flip target, nudged randomly while a player is near.
    pub flip_target: f32,
    /// java: `bookSpread`
    pub book_spread: f32,
    /// java: `bookSpreadPrev`
    pub book_spread_prev: f32,
    /// java: `bookRotation`
    pub book_rotation: f32,
    /// java: `bookRotationPrev`
    pub book_rotation_prev: f32,
    /// java: `field_145924_q` — the angle toward the nearest player, chased by `bookRotation`.
    pub book_target: f32,
}

/// java: `TileEntityFlowerPot`'s `Item`/`Data`.
///
/// The *render* half of GAP `A05-16` — feeding this into `BlockFlowerPot.getActualState`'s
/// `CONTENTS` property so the mesher picks `flower_pot_rose` over `flower_pot` — is consumed by
/// the terrain actual-state hook. The data half is here so that hook has one canonical input;
/// [`FlowerPotState::contents`] is already the exact variant string
/// `assets/minecraft/blockstates/flower_pot.json` keys on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlowerPotState {
    /// java: `flowerPotItem`, as its numeric item id (the description packet rewrites the string
    /// form to an integer — `TileEntityFlowerPot.getDescriptionPacket`, `:55`).
    pub item_id: i32,
    /// java: `flowerPotData`
    pub data: i32,
}

impl FlowerPotState {
    /// The `contents=` variant key `flower_pot.json` selects on.
    ///
    /// java: `BlockFlowerPot.getActualState` (`block/BlockFlowerPot.java:317-400`) mapped onto
    /// `EnumFlowerType`'s names. Item ids are block ids for every potted plant (all of them are
    /// `ItemBlock`s), which is what makes this a pure function of the two stored numbers.
    #[must_use]
    pub fn contents(self) -> &'static str {
        match (self.item_id, self.data) {
            // java: `block == Blocks.sapling` → `BlockPlanks.EnumType.byMetadata(i)`, which
            // **clamps** (`block/BlockPlanks.java:118-126`: `meta < 0 || meta >= 6` becomes 0).
            // The `default: EMPTY` arm of the switch above it is therefore unreachable, so an
            // out-of-range sapling meta is an oak sapling and not an empty pot.
            (6, m) if !(0..=5).contains(&m) => "oak_sapling",
            // java: `block == Blocks.red_flower` → `EnumFlowerType.getType(RED, i)`, which
            // clamps the same way over its 9 entries (`block/BlockFlower.java:150-160`), so an
            // out-of-range red-flower meta is a poppy.
            (38, m) if !(0..=8).contains(&m) => "rose",
            (6, 0) => "oak_sapling",
            (6, 1) => "spruce_sapling",
            (6, 2) => "birch_sapling",
            (6, 3) => "jungle_sapling",
            (6, 4) => "acacia_sapling",
            (6, 5) => "dark_oak_sapling",
            // java: `block == Blocks.tallgrass` — only meta 0 and 2 are potted.
            (31, 0) => "dead_bush",
            (31, 2) => "fern",
            // java: `block == Blocks.deadbush`
            (32, _) => "dead_bush",
            // java: `block == Blocks.yellow_flower` — one variant only.
            (37, _) => "dandelion",
            // java: `block == Blocks.red_flower` → `BlockFlower.EnumFlowerType.getType(RED, i)`
            (38, 0) => "rose",
            (38, 1) => "blue_orchid",
            (38, 2) => "allium",
            (38, 3) => "houstonia",
            (38, 4) => "red_tulip",
            (38, 5) => "orange_tulip",
            (38, 6) => "white_tulip",
            (38, 7) => "pink_tulip",
            (38, 8) => "oxeye_daisy",
            // java: `block == Blocks.brown_mushroom` / `red_mushroom` / `cactus`
            (39, _) => "mushroom_brown",
            (40, _) => "mushroom_red",
            (81, _) => "cactus",
            _ => "empty",
        }
    }
}

/// `BlockFlowerPot.createNewTileEntity(World, meta)`'s legacy 1.7 metadata switch.
///
/// The returned ids are item ids. Every accepted plant is an `ItemBlock`, so in 1.8.9 those
/// numeric ids equal the corresponding block ids used below.
fn flower_pot_from_legacy_meta(meta: u8) -> FlowerPotState {
    let (item_id, data) = match meta {
        1 => (38, 0), // poppy
        2 => (37, 0), // dandelion
        3 => (6, 0),  // oak sapling
        4 => (6, 1),  // spruce sapling
        5 => (6, 2),  // birch sapling
        6 => (6, 3),  // jungle sapling
        7 => (40, 0), // red mushroom
        8 => (39, 0), // brown mushroom
        9 => (81, 0), // cactus
        10 => (32, 0),
        11 => (31, 2), // fern
        12 => (6, 4),  // acacia sapling
        13 => (6, 5),  // dark-oak sapling
        _ => (0, 0),
    };
    FlowerPotState { item_id, data }
}

/// java: `TileEntityPiston`'s six client-visible fields.
///
/// A client only ever gets one of these from its **own**
/// `BlockPistonBase.onBlockEventReceived`: `doMove` writes each moving block with flag `4`
/// (`block/BlockPistonBase.java:415`) and `World.setBlockState` (`world/World.java:376-389`)
/// only calls `markBlockForUpdate` when `flags & 2` is set, so the server never sends those
/// changes; `TileEntityPiston` has no `getDescriptionPacket` either. That is why the whole
/// state machine has to run locally.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PistonState {
    /// java: `pistonState` — the block state being carried, packed `id << 4 | meta`.
    pub piston_state: BlockState,
    /// java: `pistonFacing`
    pub facing: Facing,
    /// java: `extending` — true while the piston pushes, false while a sticky piston pulls.
    pub extending: bool,
    /// java: `shouldHeadBeRendered` — set only for the head the retract leaves behind.
    pub should_head_be_rendered: bool,
    /// java: `progress`, `0.0..=1.0` in `0.5` steps.
    pub progress: f32,
    /// java: `lastProgress`, the previous tick's `progress`, for render interpolation.
    pub last_progress: f32,
}

impl Default for PistonState {
    fn default() -> PistonState {
        PistonState {
            piston_state: BlockState::AIR,
            // java: the no-arg `TileEntityPiston()` leaves `pistonFacing` null; the only path
            // that reads a default one back is `readFromNBT`'s `EnumFacing.getFront(0)`, which
            // is DOWN. Unreachable in practice — see [`TileKind::created_lazily`].
            facing: Facing::Down,
            extending: false,
            should_head_be_rendered: false,
            progress: 0.0,
            last_progress: 0.0,
        }
    }
}

impl PistonState {
    /// java: `getProgress(float ticks)` (`tileentity/TileEntityPiston.java:72-80`), including
    /// the `ticks > 1.0F` clamp — note there is **no** lower clamp, so a negative partial tick
    /// extrapolates backwards exactly as vanilla does.
    #[must_use]
    pub fn progress_at(self, ticks: f32) -> f32 {
        let ticks = if ticks > 1.0 { 1.0 } else { ticks };
        self.last_progress + (self.progress - self.last_progress) * ticks
    }

    /// java: `getOffsetX/Y/Z(float ticks)` (`:82-95`) — the render translation of the carried
    /// block, `(progress - 1) * frontOffset` while extending and `(1 - progress) * frontOffset`
    /// while retracting.
    #[must_use]
    pub fn offset_at(self, ticks: f32) -> (f32, f32, f32) {
        let p = self.progress_at(ticks);
        let scale = if self.extending { p - 1.0 } else { 1.0 - p };
        let (x, y, z) = self.facing.offset();
        (scale * x as f32, scale * y as f32, scale * z as f32)
    }
}

/// Everything the client stores about one tile entity.
///
/// java: the `TileEntity` subclass instance in `Chunk.chunkTileEntityMap`. [`TileEntity::kind`]
/// is the class; [`TileEntity::data`] is the subset of its fields a client can observe.
#[derive(Clone, Debug, PartialEq)]
pub struct TileEntity {
    /// Which tile entity this is, as derived from the block.
    pub kind: TileKind,
    /// Its client-visible state.
    pub data: TileData,
}

/// The per-class payload of a [`TileEntity`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
// Keep this public sum type Java-shaped: callers match the typed payloads directly, and the
// world/client tile store already owns the surrounding `TileEntity`. Boxing only `Spawner`
// would change that construction/match API and add indirection to its typed state, so the
// deliberate inline representation is exempt from `large_enum_variant`.
#[allow(clippy::large_enum_variant)]
pub enum TileData {
    /// java: `TileEntityChest`/`TileEntityEnderChest`
    Chest(ChestState),
    /// java: `TileEntitySign`
    Sign(SignState),
    /// java: `TileEntityBanner`
    Banner(BannerState),
    /// java: `TileEntitySkull`
    Skull(SkullState),
    /// java: `TileEntityBeacon`
    Beacon(BeaconState),
    /// java: `TileEntityMobSpawner`
    Spawner(SpawnerState),
    /// java: `TileEntityEnchantmentTable`
    Enchant(EnchantState),
    /// java: `TileEntityFlowerPot`
    FlowerPot(FlowerPotState),
    /// java: `TileEntityCommandBlock`
    CommandBlock(CommandBlockState),
    /// java: `TileEntityPiston`
    Piston(PistonState),
    /// A tile entity with no client-visible state in this world-side seam — the end portal,
    /// note block, and provider-backed inventory/redstone classes whose state is carried by
    /// container/window packets. Command blocks use [`TileData::CommandBlock`] above.
    None,
}

impl TileEntity {
    /// A freshly created tile entity of `kind`, in the state vanilla's no-arg constructor leaves.
    ///
    /// java: `createNewTileEntity` — every `ITileEntityProvider` returns a default-constructed
    /// instance, which is why a chest that was never described still animates.
    #[must_use]
    pub fn new(kind: TileKind) -> TileEntity {
        Self::new_for_block_state(kind, BlockState::AIR)
    }

    /// A freshly created tile entity using the metadata passed to
    /// `ITileEntityProvider.createNewTileEntity(World, int)`.
    ///
    /// Most 1.8.9 providers ignore that integer. `BlockFlowerPot` does not: its legacy metadata
    /// pre-populates the tile entity for old-format worlds before (or without) an S35 update.
    #[must_use]
    pub fn new_for_block_state(kind: TileKind, state: BlockState) -> TileEntity {
        let data = match kind {
            TileKind::Chest | TileKind::TrappedChest | TileKind::EnderChest => {
                TileData::Chest(ChestState::default())
            }
            TileKind::StandingSign | TileKind::WallSign => TileData::Sign(SignState::default()),
            TileKind::StandingBanner | TileKind::WallBanner => {
                TileData::Banner(BannerState::default())
            }
            TileKind::Skull => TileData::Skull(SkullState::default()),
            TileKind::Beacon => TileData::Beacon(BeaconState::default()),
            // java: `MobSpawnerBaseLogic`'s field initialisers, including `mobID = "Pig"`.
            TileKind::MobSpawner => TileData::Spawner(SpawnerState::default()),
            TileKind::EnchantmentTable => TileData::Enchant(EnchantState::default()),
            TileKind::FlowerPot => TileData::FlowerPot(flower_pot_from_legacy_meta(state.meta())),
            TileKind::CommandBlock => TileData::CommandBlock(CommandBlockState::default()),
            // java: `BlockPistonMoving.createNewTileEntity` returns null, so this arm is only
            // reachable through an explicit `TileEntity::new(TileKind::Piston)` and never
            // through `tile_entity_or_create` — see [`TileKind::created_lazily`].
            TileKind::Piston => TileData::Piston(PistonState::default()),
            // These provider classes have no client-side TESR or S35 payload. Their identity
            // still matters for the provider gate and exact NBT registry id; inventory/window
            // state belongs to the container protocol.
            TileKind::Dispenser
            | TileKind::Furnace
            | TileKind::Jukebox
            | TileKind::BrewingStand
            | TileKind::Comparator
            | TileKind::DaylightDetector
            | TileKind::Hopper
            | TileKind::Dropper
            | TileKind::EndPortal
            | TileKind::Note => TileData::None,
        };
        TileEntity { kind, data }
    }

    /// java: `new TileEntityPiston(pistonStateIn, pistonFacingIn, extendingIn, renderHead)`
    /// (`tileentity/TileEntityPiston.java:31-37`), reached through
    /// `BlockPistonMoving.newTileEntity` (`block/BlockPistonMoving.java:43-46`).
    #[must_use]
    pub fn new_piston(
        piston_state: BlockState,
        facing: Facing,
        extending: bool,
        should_head_be_rendered: bool,
    ) -> TileEntity {
        TileEntity {
            kind: TileKind::Piston,
            data: TileData::Piston(PistonState {
                piston_state,
                facing,
                extending,
                should_head_be_rendered,
                progress: 0.0,
                last_progress: 0.0,
            }),
        }
    }

    /// The moving-block payload, if this is a `TileEntityPiston`.
    #[must_use]
    pub fn piston(&self) -> Option<&PistonState> {
        match &self.data {
            TileData::Piston(p) => Some(p),
            _ => None,
        }
    }

    /// The chest payload, if this is a chest.
    #[must_use]
    pub fn chest(&self) -> Option<&ChestState> {
        match &self.data {
            TileData::Chest(c) => Some(c),
            _ => None,
        }
    }

    /// The sign payload, if this is a sign.
    #[must_use]
    pub fn sign(&self) -> Option<&SignState> {
        match &self.data {
            TileData::Sign(s) => Some(s),
            _ => None,
        }
    }

    /// The skull payload, if this is a skull.
    #[must_use]
    pub fn skull(&self) -> Option<&SkullState> {
        match &self.data {
            TileData::Skull(s) => Some(s),
            _ => None,
        }
    }

    /// The banner payload, if this is a banner.
    #[must_use]
    pub fn banner(&self) -> Option<&BannerState> {
        match &self.data {
            TileData::Banner(b) => Some(b),
            _ => None,
        }
    }

    /// The beacon payload, if this is a beacon.
    #[must_use]
    pub fn beacon(&self) -> Option<&BeaconState> {
        match &self.data {
            TileData::Beacon(b) => Some(b),
            _ => None,
        }
    }

    /// The spawner payload, if this is a mob spawner.
    #[must_use]
    pub fn spawner(&self) -> Option<&SpawnerState> {
        match &self.data {
            TileData::Spawner(s) => Some(s),
            _ => None,
        }
    }

    /// The enchantment-table payload, if this is an enchantment table.
    #[must_use]
    pub fn enchant(&self) -> Option<&EnchantState> {
        match &self.data {
            TileData::Enchant(e) => Some(e),
            _ => None,
        }
    }

    /// The flower-pot payload, if this is a flower pot.
    #[must_use]
    pub fn flower_pot(&self) -> Option<&FlowerPotState> {
        match &self.data {
            TileData::FlowerPot(f) => Some(f),
            _ => None,
        }
    }

    /// The command-block payload, if this is a command block.
    #[must_use]
    pub fn command_block(&self) -> Option<&CommandBlockState> {
        match &self.data {
            TileData::CommandBlock(c) => Some(c),
            _ => None,
        }
    }
}

/// Which sides of a chest have a same-type chest against them.
///
/// java: `TileEntityChest.checkForAdjacentChests` (`:279-289`) and the four `adjacentChest*`
/// fields it fills. `isChestAt` (`:311-322`) requires the neighbour to be the *same* chest type,
/// so a chest and a trapped chest never pair.
///
/// **Audit correction.** `A05-07`'s fix sketch asks for a port of
/// `BlockChest.checkForSurroundingChests`. That function opens with
/// `if (worldIn.isRemote) { return state; }` (`block/BlockChest.java:178-181`) — it is a no-op on
/// a client and only ever rewrites *server*-side metadata. The renderer does call it
/// (`TileEntityChestRenderer.java:52-56`), but on a client it returns the state unchanged, and
/// the function that actually decides a double chest is `checkForAdjacentChests`. That is what
/// this is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ChestNeighbours {
    /// java: `adjacentChestXNeg` — a chest at `x - 1`.
    pub x_neg: bool,
    /// java: `adjacentChestXPos` — a chest at `x + 1`.
    pub x_pos: bool,
    /// java: `adjacentChestZNeg` — a chest at `z - 1`.
    pub z_neg: bool,
    /// java: `adjacentChestZPos` — a chest at `z + 1`.
    pub z_pos: bool,
}

impl ChestNeighbours {
    /// Whether this half is the one that draws.
    ///
    /// java: `TileEntityChestRenderer.renderTileEntityAt:58` — the whole draw is inside
    /// `if (te.adjacentChestZNeg == null && te.adjacentChestXNeg == null)`, so a double chest is
    /// drawn exactly once, from its `-X`/`-Z` half.
    #[must_use]
    pub fn draws(self) -> bool {
        !self.x_neg && !self.z_neg
    }

    /// Whether the large-chest model applies. java: `:61` — `adjacentChestXPos == null &&
    /// adjacentChestZPos == null` picks `simpleChest`, anything else picks `largeChest`.
    #[must_use]
    pub fn is_double(self) -> bool {
        self.x_pos || self.z_pos
    }

    /// True when this chest has any neighbour at all.
    #[must_use]
    pub fn any(self) -> bool {
        self.x_neg || self.x_pos || self.z_neg || self.z_pos
    }
}

/// A sound one tick of tile-entity animation asked for.
///
/// java: the two `worldObj.playSoundEffect` calls in `TileEntityChest.update` (`:355`, `:404`).
/// Returned rather than played because `mc-world` has no audio engine and must not grow one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TileEntitySound {
    /// Sound origin, already carrying the `+0.5` block centre and the double-chest shift.
    pub x: f64,
    /// Sound origin Y.
    pub y: f64,
    /// Sound origin Z.
    pub z: f64,
    /// java: `"random.chestopen"` / `"random.chestclosed"`.
    pub name: &'static str,
    /// java: `0.5F`
    pub volume: f32,
}

/// What a `S24PacketBlockAction` did, for the caller to turn into sound and particles.
///
/// java: `World.addBlockEvent` → `Block.onBlockEventReceived`. Only three implementations are
/// client-visible (`NetHandlerPlayClient.java:1317-1321`'s own comment says so).
///
/// Deliberately **not** `#[non_exhaustive]`: `driver.rs`'s event match learned the hard way
/// (GAP `T2`) that a wildcard arm is invisible to every kind of test, so a new client-visible
/// block event should be a compile error in `mc-client` rather than a silent drop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BlockActionEffect {
    /// A chest/ender chest changed its user count. java: `TileEntityChest.receiveClientEvent`
    /// (`:420-426`) — `id == 1` sets `numPlayersUsing`, the lid animation's only input.
    ChestUsers {
        /// The new `numPlayersUsing`.
        players: i32,
    },
    /// A note block was struck. java: `BlockNote.onBlockEventReceived` (`:108-113`).
    Note {
        /// java: `data1` — the instrument index, `0..=4`.
        instrument: u8,
        /// java: `data2` — the note, `0..=24`.
        note: u8,
        /// java: `(double)Math.pow(2.0D, (double)(param - 12) / 12.0D)`.
        pitch: f32,
    },
    /// A piston fired. The world-side mutation installs the moving carrier and its
    /// [`PistonState`]; the client tile pass consumes that state through the moving-block TESR.
    Piston {
        /// java: `data1` — `0` extend, `1` retract.
        extending: bool,
        /// java: `data2` — the `EnumFacing` index.
        facing: u8,
    },
}

/// java: `EntitySheep.DYE_TO_RGB` (`entity/passive/EntitySheep.java:372-387`), indexed by
/// `EnumDyeColor.getMetadata()`.
///
/// Lives here rather than in `mc-data` because the beacon's stained-glass scan is its only
/// in-scope consumer and this crate must not grow a table it would then have to keep in sync.
pub const DYE_RGB: [[f32; 3]; 16] = [
    [1.0, 1.0, 1.0],   // WHITE
    [0.85, 0.5, 0.2],  // ORANGE
    [0.7, 0.3, 0.85],  // MAGENTA
    [0.4, 0.6, 0.85],  // LIGHT_BLUE
    [0.9, 0.9, 0.2],   // YELLOW
    [0.5, 0.8, 0.1],   // LIME
    [0.95, 0.5, 0.65], // PINK
    [0.3, 0.3, 0.3],   // GRAY
    [0.6, 0.6, 0.6],   // SILVER
    [0.3, 0.5, 0.6],   // CYAN
    [0.5, 0.25, 0.7],  // PURPLE
    [0.2, 0.3, 0.7],   // BLUE
    [0.4, 0.3, 0.2],   // BROWN
    [0.4, 0.5, 0.2],   // GREEN
    [0.6, 0.2, 0.2],   // RED
    [0.1, 0.1, 0.1],   // BLACK
];

/// The dye colour a `0..=15` metadata value names.
#[must_use]
pub fn dye_rgb(meta: u8) -> [f32; 3] {
    DYE_RGB[(meta & 15) as usize]
}

impl World {
    // ---- the store ---------------------------------------------------------------------

    /// The tile entity stored at `pos`, if any.
    ///
    /// Absence does **not** mean "no tile entity here" — see this module's header. Ask
    /// [`World::tile_kind_at`] for that.
    #[must_use]
    pub fn tile_entity(&self, pos: BlockPos) -> Option<&TileEntity> {
        self.tile_entities.get(&pos)
    }

    /// The tile entity stored at `pos`, mutably.
    pub fn tile_entity_mut(&mut self, pos: BlockPos) -> Option<&mut TileEntity> {
        self.tile_entities.get_mut(&pos)
    }

    /// Which tile entity the **block** at `pos` creates.
    ///
    /// java: `Chunk.getTileEntity(pos, CHECK)`'s lazy-create arm, minus the caching. Reports
    /// `None` for an unloaded chunk, because `block_at` reads air there.
    #[must_use]
    pub fn tile_kind_at(&self, pos: BlockPos) -> Option<TileKind> {
        TileKind::from_block(self.block_at(pos).block())
    }

    /// Stores `entity` at `pos`, replacing whatever was there.
    ///
    /// java: `World.setTileEntity` → `Chunk.addTileEntity(pos, te)` (`Chunk.java:951-966`),
    /// including its "only if the block is an `ITileEntityProvider`" gate — a tile entity for a
    /// block that does not have one is dropped rather than stored.
    pub fn set_tile_entity(&mut self, pos: BlockPos, entity: TileEntity) {
        if self.tile_kind_at(pos).is_none() {
            return;
        }
        self.tile_entities.insert(pos, entity);
    }

    /// Removes the tile entity at `pos`. java: `Chunk.removeTileEntity`.
    pub fn remove_tile_entity(&mut self, pos: BlockPos) -> Option<TileEntity> {
        self.tile_entities.remove(&pos)
    }

    /// The tile entity at `pos`, creating a default one from the block first if the store has
    /// none and the block has one to give.
    ///
    /// java: `Chunk.getTileEntity(pos, IMMEDIATE)` (`:918-925`) — the arm that calls
    /// `createNewTileEntity` and then `worldObj.setTileEntity`.
    ///
    /// Returns `None` for a block whose `createNewTileEntity` returns null even though it is an
    /// `ITileEntityProvider` — in 1.8.9 that is `BlockPistonMoving` alone. See
    /// [`TileKind::created_lazily`].
    pub fn tile_entity_or_create(&mut self, pos: BlockPos) -> Option<&mut TileEntity> {
        let state = self.block_at(pos);
        let kind = TileKind::from_block(state.block())?;
        if !kind.created_lazily() {
            return self.tile_entities.get_mut(&pos);
        }
        let entry = self
            .tile_entities
            .entry(pos)
            .or_insert_with(|| TileEntity::new_for_block_state(kind, state));
        // The block changed under a stale entry (a chest replaced by a sign): rebuild rather
        // than hand back a `TileData` of the wrong shape. java: `Chunk.setBlockState` removes
        // the old tile entity when the block changes (`:585-590`).
        if entry.kind != kind {
            *entry = TileEntity::new_for_block_state(kind, state);
        }
        Some(entry)
    }

    /// Mirrors the client-visible tile-entity half of `Chunk.setBlockState` after a successful
    /// block write. A metadata-only change preserves the existing instance; a block identity
    /// change drops the old provider before creating the new one.
    pub(crate) fn reconcile_tile_entity_after_block_change(
        &mut self,
        pos: BlockPos,
        old: BlockState,
        new: BlockState,
    ) {
        if old.block() != new.block() && TileKind::from_block(old.block()).is_some() {
            self.tile_entities.remove(&pos);
        }

        let Some(kind) = TileKind::from_block(new.block()) else {
            return;
        };
        if !kind.created_lazily() {
            return;
        }
        self.tile_entities
            .entry(pos)
            .or_insert_with(|| TileEntity::new_for_block_state(kind, new));
    }

    /// Every stored tile entity, in unspecified order.
    pub fn tile_entities(&self) -> impl Iterator<Item = (BlockPos, &TileEntity)> + '_ {
        self.tile_entities.iter().map(|(p, t)| (*p, t))
    }

    /// How many tile entities are stored.
    #[must_use]
    pub fn tile_entity_count(&self) -> usize {
        self.tile_entities.len()
    }

    /// Drops every tile entity inside one chunk column.
    ///
    /// java: this is free in vanilla, where the map lives on the `Chunk` and goes with it
    /// (`ChunkProviderClient.unloadChunk`). The map is on `World` here — one flat `HashMap`
    /// beats a map-of-maps for the lookup that runs once per visible tile entity per frame — so
    /// the unload has to say so explicitly. Called from `World::unload_chunk` and from
    /// `apply_chunk_data`'s ground-up branch, which is `doPreChunk(x, z, true)` →
    /// `ChunkProviderClient.loadChunk`, i.e. a brand-new `Chunk` with an empty map.
    pub fn drop_tile_entities_in_chunk(&mut self, pos: ChunkPos) {
        self.tile_entities
            .retain(|p, _| p.chunk_pos().x != pos.x || p.chunk_pos().z != pos.z);
    }

    /// Drops entries whose block no longer creates that tile entity.
    ///
    /// java: `Chunk.setBlockState` (`:585-590`) removes the tile entity as part of the block
    /// write. [`World::set_block`] now does that inline too; this remains a cheap repair pass for
    /// bulk chunk storage writes, which bypass the per-block path by design.
    pub fn prune_tile_entities(&mut self) {
        let stale: Vec<BlockPos> = self
            .tile_entities
            .iter()
            .filter(|(pos, te)| TileKind::from_block(self.block_at(**pos).block()) != Some(te.kind))
            .map(|(pos, _)| *pos)
            .collect();
        for pos in stale {
            self.tile_entities.remove(&pos);
        }
    }

    // ---- packet application ------------------------------------------------------------

    /// Applies an `S33PacketUpdateSign`.
    ///
    /// java: `NetHandlerPlayClient.handleUpdateSign` (`:1234-1261`) — the lines are copied only
    /// when the block really is a sign, and the handler chat-warns "Unable to locate sign at
    /// x, y, z" when it is not. `false` here is that `flag == false` case, so the caller can
    /// emit the same message.
    ///
    /// The `getIsEditable()` gate vanilla applies (`:1247`) is reproduced by the
    /// [`SignState::editable`] check below. `isEditable` is cleared in two places, not one:
    /// `TileEntitySign.readFromNBT` (`:50`), which really is server-only on a client that never
    /// loads a sign from disk — **and `GuiEditSign.initGui` (`:46`), which is client code**, with
    /// `onGuiClosed` (`:62`) putting it back. So a vanilla client *does* have `isEditable ==
    /// false` for exactly as long as the sign editor is open, and an `S33` that lands in that
    /// window is dropped rather than stomping what the player is typing (note `flag` is still
    /// set, so no chat warning either).
    ///
    /// `apply_sign_components` is the live client path because it retains the raw components for
    /// the eventual C12 close packet; this display-only helper carries the same gate for callers
    /// that already decoded the packet.
    pub fn apply_sign_text(&mut self, pos: BlockPos, lines: [String; 4]) -> bool {
        let Some(entity) = self.tile_entity_or_create(pos) else {
            return false;
        };
        match &mut entity.data {
            TileData::Sign(sign) => {
                if sign.editable {
                    sign.lines = lines;
                }
                true
            }
            _ => false,
        }
    }

    /// Applies an S33 sign update while retaining its serialised components for a subsequent
    /// `C12PacketUpdateSign`.  `display_lines` are derived by the client-side chat parser;
    /// `mc-world` deliberately has no dependency on that parser.
    pub fn apply_sign_components(
        &mut self,
        pos: BlockPos,
        display_lines: [String; 4],
        components: [String; 4],
    ) -> bool {
        let Some(entity) = self.tile_entity_or_create(pos) else {
            return false;
        };
        match &mut entity.data {
            TileData::Sign(sign) => {
                if sign.editable {
                    sign.lines = display_lines;
                    sign.components = components;
                }
                true
            }
            _ => false,
        }
    }

    /// Updates the transient world-render view of an active sign editor.
    pub fn set_sign_editor_render_state(
        &mut self,
        pos: BlockPos,
        lines: [String; 4],
        line_being_edited: i32,
    ) -> bool {
        let Some(entity) = self.tile_entity_mut(pos) else {
            return false;
        };
        match &mut entity.data {
            TileData::Sign(sign) => {
                sign.lines = lines;
                sign.line_being_edited = line_being_edited;
                true
            }
            _ => false,
        }
    }

    /// Mirrors `TileEntitySign.setEditable` for a real in-world sign.
    pub fn set_sign_editable(&mut self, pos: BlockPos, editable: bool) -> bool {
        let Some(entity) = self.tile_entity_mut(pos) else {
            return false;
        };
        match &mut entity.data {
            TileData::Sign(sign) => {
                sign.editable = editable;
                if editable {
                    sign.line_being_edited = -1;
                }
                true
            }
            _ => false,
        }
    }

    /// Stores the exact C12 component payload after the local editor closes.  The display rows
    /// were updated while editing, so this only commits the component identity.
    pub fn set_sign_components(&mut self, pos: BlockPos, components: [String; 4]) -> bool {
        let Some(entity) = self.tile_entity_mut(pos) else {
            return false;
        };
        match &mut entity.data {
            TileData::Sign(sign) => {
                sign.components = components;
                true
            }
            _ => false,
        }
    }

    /// Applies an already-decoded `S35PacketUpdateTileEntity` payload.
    ///
    /// java: `handleUpdateTileEntity` (`:1267-1281`). The action byte is a **type check**, not a
    /// constructor: vanilla looks the tile entity up first and only calls `readFromNBT` when the
    /// action and the existing instance's class agree. So a payload whose action does not match
    /// the block at `pos` is silently dropped, exactly as the six-way `instanceof` conjunction
    /// drops it — which is what `false` reports here.
    ///
    /// (Audit note: `A05-01`'s fix sketch reads that line as "match `action` 1..=6 exactly as
    /// `NetHandlerPlayClient:1276` does". It is worth being precise, because the difference is
    /// observable: the action alone never decides what to build. The block does.)
    /// `pos` remains the packet/header identity in this flattened client store. The application
    /// reader separately retains NBT `x`/`y`/`z` as the render position, matching
    /// `TileEntity.readFromNBT` without moving the `chunkTileEntityMap` entry.
    pub fn apply_tile_entity_nbt(&mut self, pos: BlockPos, action: u8, data: TileData) -> bool {
        let Some(kind) = self.tile_kind_at(pos) else {
            return false;
        };
        if kind.description_action() != Some(action) {
            return false;
        }
        let matches = matches!(
            (&data, kind),
            (TileData::Spawner(_), TileKind::MobSpawner)
                | (TileData::Beacon(_), TileKind::Beacon)
                | (TileData::Skull(_), TileKind::Skull)
                | (TileData::FlowerPot(_), TileKind::FlowerPot)
                | (
                    TileData::Banner(_),
                    TileKind::StandingBanner | TileKind::WallBanner
                )
                | (TileData::CommandBlock(_), TileKind::CommandBlock)
        );
        if !matches {
            return false;
        }
        // A beacon's segment list and fade are *not* in the packet — they are recomputed by the
        // client's own `updateSegmentColors` — so keep whatever the last scan produced and take
        // only `levels` from the wire. java: `TileEntityBeacon.readFromNBT` (`:283`) reads
        // `Levels` and nothing else the renderer uses.
        let data = match (data, self.tile_entities.get(&pos).map(|t| &t.data)) {
            (TileData::Beacon(new), Some(TileData::Beacon(old))) => TileData::Beacon(BeaconState {
                levels: new.levels,
                segments: old.segments.clone(),
                complete: old.complete,
                beam_fade: old.beam_fade,
                beam_render_counter: old.beam_render_counter,
            }),
            // java: `MobSpawnerBaseLogic.readFromNBT` (`:249-296`) never touches `mobRotation`
            // or `prevMobRotation` — they are not in the compound at all. A server that resends
            // a spawner's description packet (any `markBlockForUpdate` on it does) must
            // therefore not reset the miniature's angle; replacing the whole payload did.
            (TileData::Spawner(new), Some(TileData::Spawner(old))) => {
                TileData::Spawner(SpawnerState {
                    mob_rotation: old.mob_rotation,
                    prev_mob_rotation: old.prev_mob_rotation,
                    ..new
                })
            }
            (data, _) => data,
        };
        self.tile_entities.insert(pos, TileEntity { kind, data });
        true
    }

    /// Applies an `S24PacketBlockAction`.
    ///
    /// java: `handleBlockAction` (`:1322-1326`) → `World.addBlockEvent` →
    /// `Block.onBlockEventReceived`. Returns what the caller still has to do
    /// (a sound, a particle) — the tile-entity half is already applied here.
    ///
    /// **The dispatch is on the packet's block, not the world's.** `handleBlockAction`
    /// (`:1322-1326`) passes `packetIn.getBlockType()` straight to `World.addBlockEvent`, and
    /// `World.addBlockEvent` (`world/World.java:3481-3484`) is one line:
    /// `blockIn.onBlockEventReceived(this, pos, this.getBlockState(pos), eventID, eventParam)`
    /// — the *packet's* block chooses the handler and the *world's* state is handed to it, with
    /// no equality test between them. The `getBlockState(pos).getBlock() == event.getBlock()`
    /// check does exist, but in `WorldServer.fireBlockEvent`, which is the queue drain a client
    /// never runs. mc-rs used to require the two to agree here and dropped the event otherwise,
    /// which silently lost every action that raced a chunk resend.
    ///
    /// Where vanilla would then throw on a genuinely mismatched state (`state.getValue(FACING)`
    /// on a non-piston), this returns `None` instead; that is a deliberate, documented
    /// departure and the only one.
    pub fn apply_block_action(
        &mut self,
        pos: BlockPos,
        block_id: i32,
        data1: u8,
        data2: u8,
    ) -> Option<BlockActionEffect> {
        self.apply_block_action_with_border(pos, block_id, data1, data2, None)
    }

    /// Applies an `S24PacketBlockAction` with the current client-world border projection.
    ///
    /// The projection is sampled by the caller at packet-application time, matching Java's
    /// `WorldBorder.getDiameter()` call inside `BlockPistonBase.canPush`. Non-piston block-action
    /// handlers ignore it. Passing `None` is the explicit world-border-disabled compatibility
    /// control used by callers that do not model S44 state.
    pub fn apply_block_action_with_border(
        &mut self,
        pos: BlockPos,
        block_id: i32,
        data1: u8,
        data2: u8,
        border: Option<PistonWorldBorder>,
    ) -> Option<BlockActionEffect> {
        let state_here = self.block_at(pos);
        let Ok(dispatch) = u16::try_from(block_id) else {
            return None;
        };
        match dispatch {
            CHEST_ID | TRAPPED_CHEST_ID | ENDER_CHEST_ID => {
                // java: `BlockContainer.onBlockEventReceived` forwards to the tile entity's
                // `receiveClientEvent`, and `TileEntityChest`'s only handles `id == 1`.
                if data1 != 1 {
                    return None;
                }
                let players = i32::from(data2);
                let entity = self.tile_entity_or_create(pos)?;
                if let TileData::Chest(chest) = &mut entity.data {
                    chest.num_players_using = players;
                }
                Some(BlockActionEffect::ChestUsers { players })
            }
            // java: `BlockNote.onBlockEventReceived` (`block/BlockNote.java:108-113`).
            25 => Some(BlockActionEffect::Note {
                instrument: data1,
                note: data2,
                // java: `(float)Math.pow(2.0D, (double)(data2 - 12) / 12.0D)`.
                // Keep the exponentiation in f64 and narrow only at the call boundary.  Doing
                // `f32::powf` performs the rounding at a different point.
                pitch: 2.0f64.powf((f64::from(data2) - 12.0) / 12.0) as f32,
            }),
            // java: `BlockMobSpawner` is a `BlockContainer`, so `onBlockEventReceived` forwards
            // to `TileEntityMobSpawner.receiveClientEvent`, which is
            // `spawnerLogic.setDelayToMin(id)` (`tileentity/MobSpawnerBaseLogic.java:358-369`):
            // `if (delay == 1 && isRemote) spawnDelay = minSpawnDelay`. The server sends it from
            // `resetTimer`'s `func_98267_a(1)`, i.e. every time the spawner fires — which is
            // what makes the miniature visibly slow down again after a spawn.
            52 => {
                if data1 != 1 {
                    return None;
                }
                let entity = self.tile_entity_or_create(pos)?;
                if let TileData::Spawner(spawner) = &mut entity.data {
                    spawner.spawn_delay = spawner.min_spawn_delay;
                }
                None
            }
            // java: `TileEntityBeacon.receiveClientEvent` (`:468-479`) — `id == 1` runs
            // `updateBeacon()`, whose client-visible half is `updateSegmentColors()`. Without
            // this the beam only catches up on the next 80-tick sweep, so a block placed in it
            // takes up to four seconds to show.
            138 => {
                if data1 != 1 {
                    return None;
                }
                self.refresh_beacon(pos);
                None
            }
            // java: `BlockPistonBase.onBlockEventReceived` (`:143-222`). Everything past the
            // `!worldIn.isRemote` pre-check runs on the client, and `doMove`'s writes use flag
            // `4`, so the server never re-sends them: the client is *required* to reproduce the
            // whole move locally. [`World::piston_block_event`] is that reproduction.
            //
            // **No sound is produced here, deliberately.** `worldIn.playSoundEffect` dispatches
            // to every registered `IWorldAccess`, and on a client the only one is `RenderGlobal`
            // (`client/renderer/RenderGlobal.java:474`), whose `playSound` body is **empty**
            // (`:2025-2027`). The bang the player hears is the server's `S29PacketSoundEffect`,
            // sent from `world/WorldManager.java:52-55`. Synthesising it here would double it.
            STICKY_PISTON_ID | PISTON_ID => {
                let extending = data1 == 0;
                // Java dispatches the virtual handler from the packet's block, while passing
                // the world's state as the handler's `state` argument. The two piston classes
                // share their state shape, so a stale chunk can leave those ids different;
                // `isSticky` and `getStateFromMeta` must still come from the packet handler.
                if !self.piston_block_event(
                    pos,
                    state_here,
                    BlockId(dispatch),
                    data1,
                    data2,
                    border,
                ) {
                    return None;
                }
                Some(BlockActionEffect::Piston {
                    extending,
                    facing: data2,
                })
            }
            _ => None,
        }
    }

    // ---- per-tick animation ------------------------------------------------------------

    /// One 20 Hz tick of every stored tile entity's animation.
    ///
    /// java: the client-side branches of `TileEntityChest.update` (`:327-410`),
    /// `MobSpawnerBaseLogic.updateSpawner` (`:88-102`) and
    /// `TileEntityEnchantmentTable.update` (`:54-129`). Returns the sounds the chest animation
    /// asked for; `mc-world` has no audio engine and must not grow one.
    ///
    /// Compatibility wrapper for callers that have only one player position. Live multiplayer
    /// code should use [`Self::tick_tile_entities_for_players`] so each enchanting table can run
    /// `getClosestPlayer(x + 0.5, y + 0.5, z + 0.5, 3.0)` against the full ordered player list.
    pub fn tick_tile_entities(&mut self, player: Option<(f64, f64, f64)>) -> Vec<TileEntitySound> {
        match player {
            Some((x, y, z)) => self.tick_tile_entities_for_players(&[DVec3::new(x, y, z)]),
            None => self.tick_tile_entities_for_players(&[]),
        }
    }

    /// Tick against every non-spectating player in `World.playerEntities` order.
    ///
    /// Spawners use an any-player range query; each enchanting table independently selects its
    /// nearest player, retaining the first candidate on an exact-distance tie.
    pub fn tick_tile_entities_for_players(&mut self, players: &[DVec3]) -> Vec<TileEntitySound> {
        let mut sounds = Vec::new();
        // The adjacency scan reads the world while the store is being mutated, so it is done up
        // front. Chests are rare enough that one pass over the store costs nothing.
        let chest_neighbours: Vec<(BlockPos, ChestNeighbours)> = self
            .tile_entities
            .iter()
            .filter(|(_, te)| te.kind.is_chest())
            .map(|(pos, _)| (*pos, self.chest_neighbours(*pos)))
            .collect();
        let neighbours: HashMap<BlockPos, ChestNeighbours> = chest_neighbours.into_iter().collect();

        for (pos, entity) in &mut self.tile_entities {
            match &mut entity.data {
                TileData::Chest(chest) => {
                    let adj = neighbours.get(pos).copied().unwrap_or_default();
                    tick_chest(*pos, chest, adj, &mut sounds);
                }
                TileData::Spawner(spawner) => tick_spawner(*pos, spawner, players),
                TileData::Enchant(enchant) => {
                    tick_enchant(*pos, enchant, players, &mut self.tile_entity_random);
                }
                _ => {}
            }
        }
        sounds
    }

    /// Every stored mob spawner that `isActivated()` this tick, in no particular order.
    ///
    /// java: the `if (this.isActivated())` guard around `MobSpawnerBaseLogic.updateSpawner`'s
    /// whole body (`:84`). [`World::tick_tile_entities`] already applies it to the animation;
    /// this reports it so `mc-client` can spawn the `SMOKE_NORMAL` + `FLAME` pair the same
    /// branch emits (`:93-94`) without re-deriving the range test — the two must agree, or a
    /// spawner smokes while frozen.
    ///
    /// Safe to call immediately after `tick_tile_entities`: nothing in the tick mutates
    /// `activating_range` or the block position, so the answer is the same one the tick used.
    #[must_use]
    pub fn activated_spawners(&self, player: Option<(f64, f64, f64)>) -> Vec<BlockPos> {
        match player {
            Some((x, y, z)) => self.activated_spawners_for_players(&[DVec3::new(x, y, z)]),
            None => self.activated_spawners_for_players(&[]),
        }
    }

    /// Multi-player form of [`Self::activated_spawners`], sharing the tick's ordered snapshot.
    #[must_use]
    pub fn activated_spawners_for_players(&self, players: &[DVec3]) -> Vec<BlockPos> {
        self.tile_entities
            .iter()
            .filter_map(|(pos, te)| match &te.data {
                TileData::Spawner(s) => {
                    spawner_is_activated(*pos, s.activating_range, players).then_some(*pos)
                }
                _ => None,
            })
            .collect()
    }

    /// Refreshes every stored beacon's segment list. java: `TileEntityBeacon.update`
    /// calls `updateSegmentColors` every 80 ticks; `shouldBeamRender` runs per frame.
    ///
    /// Split out from [`World::tick_tile_entities`] because the scan reads up to 256 blocks per
    /// beacon and vanilla does not run it every tick either.
    pub fn refresh_beacons(&mut self) {
        let positions: Vec<BlockPos> = self
            .tile_entities
            .iter()
            .filter(|(_, te)| te.kind == TileKind::Beacon)
            .map(|(pos, _)| *pos)
            .collect();
        for pos in positions {
            self.refresh_beacon(pos);
        }
    }

    /// One beacon's `updateSegmentColors`. java: `TileEntityBeacon.updateBeacon` (`:63-67`) —
    /// the other half, `addEffectsToPlayers`, is `!isRemote`-gated and server-only.
    ///
    /// Split out from [`World::refresh_beacons`] so `S24PacketBlockAction`'s beacon arm can run
    /// it for one position the moment the packet lands, which is what vanilla's
    /// `receiveClientEvent(1, _)` does.
    pub fn refresh_beacon(&mut self, pos: BlockPos) {
        let (segments, mut complete) = self.scan_beacon_beam(pos);
        let levels = self.beacon_levels(pos);
        // java: `updateSegmentColors:200` — `if (this.levels == 0) this.isComplete = false;`
        // A beacon with no pyramid under it has no beam however clear the sky is.
        complete = complete && levels > 0;
        if let Some(TileData::Beacon(beacon)) =
            self.tile_entities.get_mut(&pos).map(|t| &mut t.data)
        {
            beacon.levels = levels;
            beacon.segments = segments;
            beacon.complete = complete;
        }
    }

    /// Every position in one chunk column whose block creates a tile entity.
    ///
    /// java: vanilla never needs this — `Chunk.chunkTileEntityMap` is filled as blocks are
    /// written, and `RenderGlobal` collects the visible ones from the render chunks the mesher
    /// already walked. Neither hook exists here (the mesher is another author's file this wave),
    /// so the renderer's "which tile entities are near me" question is answered by a scan of the
    /// column's section arrays instead. Empty sections are skipped, which is most of a column.
    #[must_use]
    pub fn scan_chunk_tile_entities(&self, pos: ChunkPos) -> Vec<(BlockPos, TileKind)> {
        let mut found = Vec::new();
        let Some(chunk) = self.chunk(pos) else {
            return found;
        };
        for (index, section) in chunk.sections().iter().enumerate() {
            let Some(section) = section else { continue };
            if !section.may_contain_tile_provider() {
                continue;
            }
            for y in 0..16u8 {
                for z in 0..16u8 {
                    for x in 0..16u8 {
                        let raw = section.raw_block(x, y, z);
                        if let Some(kind) = tile_kind_from_raw_state(raw) {
                            found.push((
                                BlockPos::new(
                                    (pos.x << 4) + i32::from(x),
                                    (index as i32) * 16 + i32::from(y),
                                    (pos.z << 4) + i32::from(z),
                                ),
                                kind,
                            ));
                        }
                    }
                }
            }
        }
        found
    }

    /// Applies one visible render call of `TileEntityBeacon.shouldBeamRender`.
    ///
    /// The renderer calls this only after its strict distance gate. An incomplete beacon returns
    /// zero without touching either field; consequently an obstructed or out-of-range beacon
    /// retains the last render tick and pays the elapsed `ticks / 40` decay on re-entry. Multiple
    /// frames in one world tick each still add `0.025`, exactly like the mutating Java getter.
    pub fn advance_beacon_render_frame(&mut self, pos: BlockPos, total_world_time: i64) -> bool {
        let Some(TileData::Beacon(beacon)) = self
            .tile_entities
            .get_mut(&pos)
            .map(|entity| &mut entity.data)
        else {
            return false;
        };
        if !beacon.complete {
            return false;
        }

        // java: `int i = (int)(getTotalWorldTime() - beamRenderCounter)`,
        // a wrapping narrowing, then two separate clamps: the decay floors at zero *before* this
        // frame's `+= 0.025F`, so the first frame back after a long absence draws at 0.025
        // rather than being skipped.
        let elapsed = total_world_time.wrapping_sub(beacon.beam_render_counter) as i32;
        beacon.beam_render_counter = total_world_time;
        if elapsed > 1 {
            beacon.beam_fade -= elapsed as f32 / 40.0;
            if beacon.beam_fade < 0.0 {
                beacon.beam_fade = 0.0;
            }
        }
        beacon.beam_fade += 0.025;
        if beacon.beam_fade > 1.0 {
            beacon.beam_fade = 1.0;
        }
        true
    }

    // ---- pure scans --------------------------------------------------------------------

    /// Which sides of the chest at `pos` carry a same-type chest.
    ///
    /// java: `TileEntityChest.checkForAdjacentChests` (`:279-289`) via `isChestAt` (`:311-322`),
    /// whose test is `block instanceof BlockChest && block.chestType == this.getChestType()` —
    /// so `chest`(54) pairs only with `chest`, `trapped_chest`(146) only with `trapped_chest`,
    /// and an ender chest never pairs at all (`TileEntityEnderChest` is not a `TileEntityChest`).
    #[must_use]
    pub fn chest_neighbours(&self, pos: BlockPos) -> ChestNeighbours {
        let here = self.block_at(pos).block().0;
        if here != CHEST_ID && here != TRAPPED_CHEST_ID {
            return ChestNeighbours::default();
        }
        let same = |p: BlockPos| self.block_at(p).block().0 == here;
        ChestNeighbours {
            x_neg: same(pos.west()),
            x_pos: same(pos.east()),
            z_neg: same(pos.north()),
            z_pos: same(pos.south()),
        }
    }

    /// The beacon beam above `pos`: one segment per colour run, and whether the beam completes.
    ///
    /// java: `TileEntityBeacon.updateSegmentColors` (`:102-207`), whose upward half runs on both
    /// sides. Reproduced statement for statement, including the two non-obvious rules:
    ///
    /// * the *first* coloured block takes the dye straight, and every later one is **averaged**
    ///   with the running colour (`:145`, gated on `flag`), which is why a white→red→blue stack
    ///   is not the same as red→blue;
    /// * `getLightOpacity() >= 15` aborts the whole beam (`:131`) — except bedrock, which is the
    ///   only opaque block a beam passes through.
    #[must_use]
    pub fn scan_beacon_beam(&self, pos: BlockPos) -> (Vec<BeamSegment>, bool) {
        let mut segments = vec![BeamSegment {
            // java: `new BeamSegment(EntitySheep.getDyeRgb(EnumDyeColor.WHITE))`
            // `BeamSegment`'s constructor sets `height = 1` before the upward scan begins.
            // The first unit is the beacon block itself; starting at zero makes every segment
            // boundary one block too low and leaves a zero-height white segment when the first
            // block above the beacon is stained glass.
            colors: DYE_RGB[0],
            height: 1,
        }];
        // java: `boolean flag = true` — "no colour has been applied yet".
        let mut first = true;
        for y in (pos.y + 1)..256 {
            let state = self.block_at(BlockPos::new(pos.x, y, pos.z));
            let block = state.block().0;
            let dye = match block {
                STAINED_GLASS_ID | STAINED_GLASS_PANE_ID => dye_rgb(state.meta()),
                _ => {
                    if state.props().light_opacity >= 15 && block != BEDROCK_ID {
                        return (Vec::new(), false);
                    }
                    let last = segments.last_mut().expect("segments is never empty");
                    last.height += 1;
                    continue;
                }
            };
            let running = segments.last().expect("segments is never empty").colors;
            let colors = if first {
                dye
            } else {
                [
                    (running[0] + dye[0]) / 2.0,
                    (running[1] + dye[1]) / 2.0,
                    (running[2] + dye[2]) / 2.0,
                ]
            };
            if colors == running {
                segments.last_mut().expect("non-empty").height += 1;
            } else {
                segments.push(BeamSegment { colors, height: 1 });
            }
            first = false;
        }
        (segments, true)
    }

    /// Whether the beacon at `pos` stands on a complete pyramid, and how many levels of it.
    ///
    /// java: `updateSegmentColors`' second half (`:169-198`) — up to four rings of
    /// emerald/gold/diamond/iron below the beacon. This is what `levels` means; the server sends
    /// it in the `S35` action-3 payload too, so this exists for a client that has not been sent
    /// one yet (and as the check `isComplete` really depends on).
    #[must_use]
    pub fn beacon_levels(&self, pos: BlockPos) -> i32 {
        // java: emerald 133, gold 41, diamond 57, iron 42.
        const BASE: [u16; 4] = [133, 41, 57, 42];
        let mut levels = 0;
        for l in 1..=4i32 {
            let y = pos.y - l;
            if y < 0 {
                break;
            }
            let mut ok = true;
            'ring: for x in (pos.x - l)..=(pos.x + l) {
                for z in (pos.z - l)..=(pos.z + l) {
                    if !BASE.contains(&self.block_at(BlockPos::new(x, y, z)).block().0) {
                        ok = false;
                        break 'ring;
                    }
                }
            }
            if !ok {
                break;
            }
            levels = l;
        }
        levels
    }
}

impl World {
    /// One 20 Hz tick of every stored `TileEntityPiston`.
    ///
    /// java: `TileEntityPiston.update` (`tileentity/TileEntityPiston.java:166-199`).
    /// `World.updateEntities`' `tickableTileEntities` loop carries **no** `isRemote` guard, so
    /// this runs on a vanilla client — it is what makes the moving block slide instead of
    /// teleport, and (through `launchWithSlimeBlock`) what makes a piston shove the player.
    ///
    /// Separate from [`World::tick_tile_entities`] because completion mutates the *world*
    /// (`removeTileEntity` then `setBlockState(pos, pistonState, 3)`) while that function holds
    /// a `&mut` borrow of the store. Returns the shoves the caller still has to apply; this
    /// crate has no entity list the local player is in.
    pub fn tick_pistons(&mut self) -> Vec<PistonPush> {
        let mut pushes = Vec::new();
        let mut states: Vec<(BlockPos, PistonState)> = self
            .tile_entities
            .iter()
            .filter_map(|(pos, te)| te.piston().map(|p| (*pos, *p)))
            .collect();
        // The store is a `HashMap`, so fix an order. Vanilla's is insertion order over
        // `tickableTileEntities`; nothing here is order-sensitive except which shove lands
        // first when two pistons overlap, which vanilla does not define either.
        states.sort_unstable_by_key(|(pos, _)| (pos.y, pos.x, pos.z));

        let mut finished: Vec<(BlockPos, BlockState)> = Vec::new();
        for (pos, mut p) in states {
            // java: `this.lastProgress = this.progress;`
            p.last_progress = p.progress;
            if p.last_progress >= 1.0 {
                // java: `launchWithSlimeBlock(1.0F, 0.25F)`, then remove the tile entity and
                // write the carried state back with flag 3.
                pushes.extend(piston_launch(self, pos, p, 1.0, 0.25));
                finished.push((pos, p.piston_state));
            } else {
                p.progress = (p.progress + 0.5).min(1.0);
                if p.extending {
                    // java: `launchWithSlimeBlock(progress, progress - lastProgress + 0.0625F)`
                    let amount = p.progress - p.last_progress + 0.0625;
                    pushes.extend(piston_launch(self, pos, p, p.progress, amount));
                }
            }
            if let Some(TileData::Piston(cur)) =
                self.tile_entities.get_mut(&pos).map(|t| &mut t.data)
            {
                *cur = p;
            }
        }

        for (pos, carried) in finished {
            self.remove_tile_entity(pos);
            // java: the `getBlockState(pos).getBlock() == Blocks.piston_extension` guard — it
            // matters, because another piston may already have overwritten this position.
            if self.block_at(pos).block().0 == PISTON_EXTENSION_ID {
                self.set_block(pos, carried);
            }
        }
        pushes
    }

    // ---- the client half of a piston move ----------------------------------------------

    /// java: `BlockPistonBase.onBlockEventReceived` (`block/BlockPistonBase.java:143-222`),
    /// minus the `!worldIn.isRemote` pre-check, which is the only server-only part of it.
    ///
    /// `base` is the piston block state at `pos`; `data1`/`data2` are the `S24PacketBlockAction`
    /// fields, which `checkForMove` fills with `(0 | 1, facing.getIndex())`
    /// (`block/BlockPistonBase.java:99-107`). Returns vanilla's own boolean — `false` means the
    /// move was refused and nothing changed.
    fn piston_block_event(
        &mut self,
        pos: BlockPos,
        base: BlockState,
        handler_block: BlockId,
        data1: u8,
        data2: u8,
        border: Option<PistonWorldBorder>,
    ) -> bool {
        // The state comes from the *world* (`World.addBlockEvent` passes `getBlockState(pos)`)
        // while the handler was chosen by the *packet*. When the two disagree vanilla would
        // throw inside `state.getValue(FACING)`; refusing is the documented departure.
        if base.block().0 != PISTON_ID && base.block().0 != STICKY_PISTON_ID {
            return false;
        }
        // java: `(EnumFacing) state.getValue(FACING)` — the facing comes from the *block state*,
        // not from the packet. `BlockPistonBase.getFacing(meta)` is `meta & 7`, null above 5.
        let Some(facing) = piston_facing(base.meta()) else {
            return false;
        };
        // `this.isSticky` is a field on the packet-selected handler (`this`), not on the
        // world-state argument. This is observable when an S24 races a stale chunk update.
        let sticky = handler_block.0 == STICKY_PISTON_ID;

        if data1 == 0 {
            if !self.piston_do_move(pos, facing, true, sticky, border) {
                return false;
            }
            // java: `setBlockState(pos, state.withProperty(EXTENDED, true), 2)`, and
            // `BlockPistonBase.getMetaFromState` is `facing.getIndex() | (extended ? 8 : 0)`.
            self.set_block(pos, BlockState::new(base.block(), facing.index() | 8));
            return true;
        }
        if data1 != 1 {
            // java: neither branch matches and `onBlockEventReceived` falls through to
            // `return true` having done nothing.
            return true;
        }

        // Validate the untrusted event parameter before mutating the moving head. Java's
        // generated events always carry 0..=5, but a malformed S24 must not leave a half-applied
        // retract behind when this client deliberately chooses a safe no-op.
        let Some(param_facing) = piston_facing(data2) else {
            return false;
        };
        // java: the head's tile entity is stopped where it stands before the base is rewritten.
        self.clear_piston_tile_entity(pos.offset(facing, 1));
        // java: `this.getStateFromMeta(eventParam)` on the *base* block — FACING from the
        // packet, EXTENDED from bit 3, which `checkForMove` never sets.
        // Java calls `this.getStateFromMeta(eventParam)`, so the moving carrier's piston block
        // id follows the packet handler even when the stale world base has the other variant.
        let carried = BlockState::new(handler_block, param_facing.index() | (data2 & 8));
        let ext_meta = facing.index() | (u8::from(sticky) * 8);
        self.set_block(pos, BlockState::new(BlockId(PISTON_EXTENSION_ID), ext_meta));
        self.set_tile_entity(pos, TileEntity::new_piston(carried, facing, false, true));

        if sticky {
            // java: `:186-215` — a sticky piston drags whatever is two blocks out back with it.
            let two = pos.offset(facing, 2);
            let there = self.block_at(two);
            let mut already_moving = false;
            if there.block().0 == PISTON_EXTENSION_ID {
                if let Some(other) = self.tile_entity(two).and_then(TileEntity::piston) {
                    already_moving = other.facing == facing && other.extending;
                }
                if already_moving {
                    self.clear_piston_tile_entity(two);
                }
            }
            let mobility = block_info(there.block()).mobility_flag;
            if !already_moving
                && there.props().material != Material::Air
                && piston_can_push_with_border(self, two, facing.opposite(), false, border)
                && (mobility == 0
                    || there.block().0 == PISTON_ID
                    || there.block().0 == STICKY_PISTON_ID)
            {
                self.piston_do_move(pos, facing, false, sticky, border);
            }
        } else {
            // java: `worldIn.setBlockToAir(pos.offset(enumfacing))` — the head simply vanishes.
            self.set_block(pos.offset(facing, 1), BlockState::AIR);
            self.remove_tile_entity(pos.offset(facing, 1));
        }
        true
    }

    /// java: `BlockPistonBase.doMove` (`block/BlockPistonBase.java:376-450`).
    ///
    /// `sticky` is the *class* flag (`BlockPistonBase.isSticky`), not a property of the block at
    /// `pos`: by the time the retract path calls this, the base has already been overwritten
    /// with `piston_extension`.
    fn piston_do_move(
        &mut self,
        pos: BlockPos,
        direction: Facing,
        extending: bool,
        sticky: bool,
        border: Option<PistonWorldBorder>,
    ) -> bool {
        if !extending {
            self.set_block(pos.offset(direction, 1), BlockState::AIR);
            self.remove_tile_entity(pos.offset(direction, 1));
        }
        let (to_move, to_destroy) = {
            let mut helper = PistonStructure::new(self, pos, direction, extending, border);
            if !helper.can_move() {
                return false;
            }
            (helper.to_move, helper.to_destroy)
        };

        let move_dir = if extending {
            direction
        } else {
            direction.opposite()
        };
        // java: `block.dropBlockAsItem(...)` then `setBlockToAir`. `dropBlockAsItemWithChance`
        // opens with `if (!worldIn.isRemote)` (`block/Block.java:958`), so the client half of a
        // destroy is the air write alone.
        for &p in to_destroy.iter().rev() {
            self.set_block(p, BlockState::AIR);
            self.remove_tile_entity(p);
        }
        // Reverse order matters: the farthest block is written first, so a destination is
        // always vacated before the block behind it lands on it.
        for &p in to_move.iter().rev() {
            let carried = self.block_at(p);
            self.set_block(p, BlockState::AIR);
            self.remove_tile_entity(p);
            let dest = p.offset(move_dir, 1);
            // java: `Blocks.piston_extension.getDefaultState().withProperty(FACING, direction)`
            // — TYPE stays DEFAULT here, so bit 3 is clear whatever the piston is.
            self.set_block(
                dest,
                BlockState::new(BlockId(PISTON_EXTENSION_ID), direction.index()),
            );
            self.set_tile_entity(
                dest,
                TileEntity::new_piston(carried, direction, extending, false),
            );
        }

        if extending {
            // java: `:421-431` — the head itself becomes a moving block carrying `piston_head`.
            let head_pos = pos.offset(direction, 1);
            let head_meta = direction.index() | (u8::from(sticky) * 8);
            self.set_block(
                head_pos,
                BlockState::new(BlockId(PISTON_EXTENSION_ID), head_meta),
            );
            self.set_tile_entity(
                head_pos,
                TileEntity::new_piston(
                    BlockState::new(BlockId(PISTON_HEAD_ID), head_meta),
                    direction,
                    true,
                    false,
                ),
            );
        }
        // java: the two `notifyNeighborsOfStateChange` loops. mc-rs has no
        // `onNeighborBlockChange` dispatch at all — a separate and larger gap — and on a client
        // every listener those notifications would reach (`checkForMove`, the redstone diodes)
        // is server-side; the render invalidation they would also cause is already done by
        // `set_block`.
        true
    }

    /// java: `TileEntityPiston.clearPistonTileEntity`
    /// (`tileentity/TileEntityPiston.java:150-164`).
    fn clear_piston_tile_entity(&mut self, pos: BlockPos) {
        let Some(p) = self.tile_entity(pos).and_then(TileEntity::piston).copied() else {
            return;
        };
        // java: `if (this.lastProgress < 1.0F && this.worldObj != null)`
        if p.last_progress >= 1.0 {
            return;
        }
        self.remove_tile_entity(pos);
        if self.block_at(pos).block().0 == PISTON_EXTENSION_ID {
            self.set_block(pos, p.piston_state);
        }
    }

    /// The collision box `BlockPistonMoving` reports at `pos`, or `None` when there is none.
    ///
    /// java: `BlockPistonMoving.getCollisionBoundingBox`
    /// (`block/BlockPistonMoving.java:159-177`) — `null` without a tile entity, otherwise the
    /// carried block's own box shifted by `getProgress(0.0F)`, inverted while extending.
    /// mc-data's extracted `collision: Empty` for block 36 is only the *no*-tile-entity case,
    /// which is why physics has to ask here.
    #[must_use]
    pub fn piston_collision_box(&self, pos: BlockPos) -> Option<Aabb> {
        let p = self.tile_entity(pos)?.piston()?;
        let mut f = p.progress_at(0.0);
        if p.extending {
            f = 1.0 - f;
        }
        piston_bounding_box_at(Some(self), pos, p.piston_state, f, p.facing)
    }
}

/// java: `BlockPistonBase.getFacing(int meta)` / `BlockPistonExtension.getFacing(int meta)` —
/// both are `meta & 7`, null above 5.
#[must_use]
pub fn piston_facing(meta: u8) -> Option<Facing> {
    let i = meta & 7;
    if i > 5 {
        None
    } else {
        Some(Facing::from_index(i32::from(i)))
    }
}

/// The part of Java's `WorldBorder` that `BlockPistonBase.canPush` observes.
///
/// `S44PacketWorldBorder` and the transition clock belong to the client environment, not to
/// this block-storage crate. A caller therefore snapshots the border's current diameter and
/// threads this small, copyable projection through the piston move. `None` means the caller has
/// no live border (the compatibility/default path used by older callers), just as an unmodelled
/// border would leave the old piston behaviour unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PistonWorldBorder {
    /// Border centre X in the current dimension's coordinates.
    pub center_x: f64,
    /// Border centre Z in the current dimension's coordinates.
    pub center_z: f64,
    /// Current border diameter after any S44 transition interpolation.
    pub diameter: f64,
    /// Java `WorldBorder.worldSize`, normally `29_999_984`.
    pub world_size: i32,
}

impl PistonWorldBorder {
    /// Construct a current, dimension-local border projection.
    #[must_use]
    pub const fn new(center_x: f64, center_z: f64, diameter: f64, world_size: i32) -> Self {
        Self {
            center_x,
            center_z,
            diameter,
            world_size,
        }
    }

    /// Java `WorldBorder.contains(BlockPos)` (`world/border/WorldBorder.java:34`).
    ///
    /// The comparisons are strict, and the minimum side widens the integer block position by
    /// one before converting it to `double`. Keep the Java-shaped clamp rather than using
    /// `f64::min`/`max`: those methods have different NaN behaviour than Java's relational
    /// `if` statements.
    #[must_use]
    pub fn contains(self, pos: BlockPos) -> bool {
        let min_x = self.min_x();
        let max_x = self.max_x();
        let min_z = self.min_z();
        let max_z = self.max_z();
        f64::from(pos.x.wrapping_add(1)) > min_x
            && f64::from(pos.x) < max_x
            && f64::from(pos.z.wrapping_add(1)) > min_z
            && f64::from(pos.z) < max_z
    }

    /// Alias matching the client environment's `BorderState::contains_block` spelling.
    #[must_use]
    pub fn contains_block(self, pos: BlockPos) -> bool {
        self.contains(pos)
    }

    fn min_x(self) -> f64 {
        let value = self.center_x - self.diameter / 2.0;
        if value < -f64::from(self.world_size) {
            -f64::from(self.world_size)
        } else {
            value
        }
    }

    fn max_x(self) -> f64 {
        let value = self.center_x + self.diameter / 2.0;
        if value > f64::from(self.world_size) {
            f64::from(self.world_size)
        } else {
            value
        }
    }

    fn min_z(self) -> f64 {
        let value = self.center_z - self.diameter / 2.0;
        if value < -f64::from(self.world_size) {
            -f64::from(self.world_size)
        } else {
            value
        }
    }

    fn max_z(self) -> f64 {
        let value = self.center_z + self.diameter / 2.0;
        if value > f64::from(self.world_size) {
            f64::from(self.world_size)
        } else {
            value
        }
    }
}

/// java: `BlockPistonBase.canPush` (`block/BlockPistonBase.java:321-372`).
///
/// [`piston_can_push`] is the compatibility form for callers without a live S44 snapshot. The
/// border-aware movement path uses [`piston_can_push_with_border`] so every structure-helper
/// query observes the same current border projection.
#[must_use]
pub fn piston_can_push(
    world: &World,
    pos: BlockPos,
    direction: Facing,
    allow_destroy: bool,
) -> bool {
    piston_can_push_with_border(world, pos, direction, allow_destroy, None)
}

/// `BlockPistonBase.canPush` with the caller's current `WorldBorder` projection.
///
/// `border == None` deliberately disables the dynamic border gate for compatibility with
/// callers that model only blocks and height. When present, the Java `contains(pos)` check runs
/// before build-height and mobility checks, including the `allowDestroy` branch.
#[must_use]
pub fn piston_can_push_with_border(
    world: &World,
    pos: BlockPos,
    direction: Facing,
    allow_destroy: bool,
    border: Option<PistonWorldBorder>,
) -> bool {
    let state = world.block_at(pos);
    let block = state.block();
    if block.0 == OBSIDIAN_ID {
        return false;
    }
    if border.is_some_and(|border| !border.contains(pos)) {
        return false;
    }
    if pos.y < 0 || (direction == Facing::Down && pos.y == 0) {
        return false;
    }
    // java: `worldIn.getHeight() - 1`, and `World.getHeight()` is 256 in every 1.8.9 dimension.
    if pos.y > 255 || (direction == Facing::Up && pos.y == 255) {
        return false;
    }
    if block.0 != PISTON_ID && block.0 != STICKY_PISTON_ID {
        // java: `blockIn.getBlockHardness(worldIn, pos) == -1.0F`
        if state.props().hardness == -1.0 {
            return false;
        }
        match block_info(block).mobility_flag {
            2 => return false,
            1 => return allow_destroy,
            _ => {}
        }
    } else if state.meta() & 8 != 0 {
        // java: `getValue(EXTENDED)` — bit 3 of a piston base's metadata.
        return false;
    }
    // java: `return !(blockIn instanceof ITileEntityProvider);` (`BlockPistonBase.java:363`)
    //
    // `has_tile_entity` is `Block.hasTileEntity()`, i.e. `isBlockContainer`
    // (`block/Block.java:436-439`), which is a *different spelling* — but not a different set.
    // An earlier version of this line carried `&& block.0 != 149 && block.0 != 150` on the
    // theory that `BlockRedstoneComparator` implements `ITileEntityProvider` without extending
    // `BlockContainer` and so would slip through. It does not extend `BlockContainer`, but
    // `BlockRedstoneComparator.java:38` sets `this.isBlockContainer = true` by hand, so
    // `hasTileEntity()` is true for 149/150 and the patch was dead code. The two sets are equal
    // over the whole 1.8.9 registry and `mc-data/tests/golden_extractor.rs` asserts it against
    // the live one, so this needs no special case.
    !state.props().has_tile_entity
}

/// java: `BlockPistonMoving.getBoundingBox` (`block/BlockPistonMoving.java:218-271`).
///
/// `progress` is the already-inverted value both vanilla call sites compute; they invert it
/// *differently* (`1 - p` versus `p - 1` when retracting), so the inversion stays at the call
/// site.
///
/// `BlockPistonMoving` does **not** call the carried block's
/// `addCollisionBoxesToList`. It calls `getCollisionBoundingBox`, which returns one AABB from
/// that block instance's mutable `minX..maxZ` fields. `mc-data` snapshots those fields as
/// [`mc_data::BlockProps::bounds`]; the extracted [`CollisionShape`](mc_data::CollisionShape)
/// is a different oracle and may contain several independent boxes (stairs, panes, fences,
/// etc.). Using the union of those boxes here makes a moving carrier wider/taller than Java's
/// single-box lookup. Keep the two paths separate: ordinary world collision still iterates the
/// `CollisionShape` boxes, while this moving-block path selects one carried AABB.
#[must_use]
pub fn piston_bounding_box(
    pos: BlockPos,
    carried: BlockState,
    progress: f32,
    direction: Facing,
) -> Option<Aabb> {
    piston_bounding_box_at(None, pos, carried, progress, direction)
}

/// World-aware half of [`piston_bounding_box`].  Most carried blocks use the ordinary block
/// AABB rule (`int position + float-local bounds`), but a few 1.8.9 overrides deliberately form
/// some coordinates in Java `float`, and `BlockWall.getCollisionBoundingBox` reads live
/// neighbours before forcing its height to 1.5.  The moving carrier invokes those overrides at
/// its extension position, so retaining this small world seam avoids silently substituting the
/// carried block's ordinary-world collision query.
fn piston_bounding_box_at(
    world: Option<&World>,
    pos: BlockPos,
    carried: BlockState,
    progress: f32,
    direction: Facing,
) -> Option<Aabb> {
    let props = carried.props();
    // java: `extendingBlock.getBlock() != this && getMaterial() != Material.air`
    if carried.block().0 == PISTON_EXTENSION_ID || props.material == Material::Air {
        return None;
    }
    // Java returns null for carried blocks whose own getCollisionBoundingBox is null.  The
    // generated collision shape is the stable client-side representation of that predicate;
    // `bounds` alone is not sufficient because plants/liquids/buttons retain non-empty mutable
    // render bounds even though their collision method returns null.
    if props.collision.is_empty() {
        return None;
    }

    let class = block_info(props.block).java_class.rsplit('.').next();
    let world_bounds = if class == Some("BlockWall") {
        world.map(|world| crate::collision::wall_carried_collision_bounds(world, pos, props.block))
    } else {
        None
    };

    let (mut d0, mut d1, mut d2, mut d3, mut d4, mut d5) = if let Some(bounds) = world_bounds {
        (
            bounds.min.x,
            bounds.min.y,
            bounds.min.z,
            bounds.max.x,
            bounds.max.y,
            bounds.max.z,
        )
    } else if let Some(bounds) = piston_direct_world_bounds(pos, carried, class) {
        (
            bounds.min.x,
            bounds.min.y,
            bounds.min.z,
            bounds.max.x,
            bounds.max.y,
            bounds.max.z,
        )
    } else {
        // One mutable Java Block AABB, not the union of `props.collision.boxes()`. For the
        // neighbour-dependent classes the generated `bounds` snapshot is the state-aware
        // mutable singleton (`setBlockBoundsBasedOnState`); direct one-box collision
        // overrides (cactus, farmland, snow, slabs, etc.) retain their exact state-local
        // collision box. The wall's no-world compatibility path uses its singular extracted
        // collision box so its Java `maxY = 1.5D` override is not lost.
        let boxes = props.collision.boxes();
        let bounds = if class == Some("BlockWall") && boxes.len() == 1 {
            boxes[0]
        } else if props.collision_depends_on_neighbors {
            props.bounds
        } else if boxes.len() == 1 {
            boxes[0]
        } else {
            props.bounds
        };
        (
            f64::from(pos.x) + bounds.min_x,
            f64::from(pos.y) + bounds.min_y,
            f64::from(pos.z) + bounds.min_z,
            f64::from(pos.x) + bounds.max_x,
            f64::from(pos.y) + bounds.max_y,
            f64::from(pos.z) + bounds.max_z,
        )
    };

    let (ox, oy, oz) = direction.offset();
    // java: `(double)((float) direction.getFrontOffsetX() * progress)` — the product is formed
    // in `f32` and only then widened.
    let shift = |o: i32| f64::from(o as f32 * progress);
    if ox < 0 {
        d0 -= shift(ox);
    } else {
        d3 -= shift(ox);
    }
    if oy < 0 {
        d1 -= shift(oy);
    } else {
        d4 -= shift(oy);
    }
    if oz < 0 {
        d2 -= shift(oz);
    } else {
        d5 -= shift(oz);
    }
    Some(Aabb::new(DVec3::new(d0, d1, d2), DVec3::new(d3, d4, d5)))
}

/// The direct `getCollisionBoundingBox` overrides whose world coordinates are formed in Java
/// `float`.  Returning the already-anchored box preserves the same far-positive and far-negative
/// rounding while leaving ordinary `Block.getCollisionBoundingBox` callers on the int-plus-local
/// path above.
fn piston_direct_world_bounds(
    pos: BlockPos,
    carried: BlockState,
    class: Option<&str>,
) -> Option<Aabb> {
    let f32x = |value: i32| value as f32;
    let fx = f32x(pos.x);
    let fy = f32x(pos.y);
    let fz = f32x(pos.z);
    let fx1 = f32x(pos.x + 1);
    let fy1 = f32x(pos.y + 1);
    let fz1 = f32x(pos.z + 1);
    match class {
        Some("BlockCactus") => {
            let inset = 0.0625_f32;
            Some(Aabb::new(
                DVec3::new(
                    f64::from(fx + inset),
                    f64::from(pos.y),
                    f64::from(fz + inset),
                ),
                DVec3::new(
                    f64::from(fx1 - inset),
                    f64::from(fy1 - inset),
                    f64::from(fz1 - inset),
                ),
            ))
        }
        Some("BlockCake") => {
            let inset = 0.0625_f32;
            let bites = f32x(i32::from(carried.meta()));
            let bite_x = (1.0_f32 + bites * 2.0_f32) / 16.0_f32;
            Some(Aabb::new(
                DVec3::new(
                    f64::from(fx + bite_x),
                    f64::from(pos.y),
                    f64::from(fz + inset),
                ),
                DVec3::new(
                    f64::from(fx1 - inset),
                    f64::from(fy + 0.5_f32),
                    f64::from(fz1 - inset),
                ),
            ))
        }
        Some("BlockFenceGate") => {
            let meta = carried.meta();
            if meta & 4 != 0 {
                return None;
            }
            let inset = 0.375_f32;
            let other = 0.625_f32;
            if matches!(meta & 3, 0 | 2) {
                Some(Aabb::new(
                    DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(fz + inset)),
                    DVec3::new(
                        f64::from(pos.x + 1),
                        f64::from(fy + 1.5_f32),
                        f64::from(fz + other),
                    ),
                ))
            } else {
                Some(Aabb::new(
                    DVec3::new(f64::from(fx + inset), f64::from(pos.y), f64::from(pos.z)),
                    DVec3::new(
                        f64::from(fx + other),
                        f64::from(fy + 1.5_f32),
                        f64::from(pos.z + 1),
                    ),
                ))
            }
        }
        Some("BlockSoulSand") => Some(Aabb::new(
            DVec3::new(f64::from(pos.x), f64::from(pos.y), f64::from(pos.z)),
            DVec3::new(
                f64::from(pos.x + 1),
                f64::from(fy1 - 0.125_f32),
                f64::from(pos.z + 1),
            ),
        )),
        _ => None,
    }
}

/// One tick's worth of `TileEntityPiston.launchWithSlimeBlock`
/// (`tileentity/TileEntityPiston.java:97-145`), for a caller that owns an entity list.
///
/// `mc-world` has no list the local player is in — the predictor lives in `mc-client` — so this
/// is returned rather than applied. The method is **not** `isRemote`-gated: on a vanilla client
/// it moves the local player, which is the whole mechanism of being pushed by a piston.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PistonPush {
    /// The box `getEntitiesWithinAABBExcludingEntity(null, aabb)` is queried with.
    pub area: Aabb,
    /// java: the `pistonState.getBlock() == Blocks.slime_block && this.extending` branch, which
    /// sets `motionX/Y/Z` outright on the facing's axis instead of calling `moveEntity`.
    pub slime: bool,
    /// java: `pistonFacing` — the axis and sign of the shove.
    pub facing: Facing,
    /// java: `p_145863_2_` — the `moveEntity` distance along `facing`.
    pub amount: f32,
}

/// java: the head of `launchWithSlimeBlock` — the progress inversion and the `getBoundingBox`
/// lookup that decides whether anything is shoved at all.
fn piston_launch(
    world: &World,
    pos: BlockPos,
    p: PistonState,
    progress: f32,
    amount: f32,
) -> Option<PistonPush> {
    let progress = if p.extending {
        1.0 - progress
    } else {
        progress - 1.0
    };
    let area = piston_bounding_box_at(Some(world), pos, p.piston_state, progress, p.facing)?;
    Some(PistonPush {
        area,
        slime: p.piston_state.block().0 == SLIME_BLOCK_ID && p.extending,
        facing: p.facing,
        amount,
    })
}

/// java: `BlockPistonStructureHelper` (`block/state/BlockPistonStructureHelper.java`).
///
/// The unit brief assumed this class was server-only. It is not:
/// `BlockPistonBase.onBlockEventReceived` is commented "Called on both Client and Server" and
/// only its `shouldBeExtended` pre-check sits inside `if (!worldIn.isRemote)`, so `doMove` — and
/// with it this whole class, its 12-block cap, its slime-block recursion and its list
/// reordering — executes on the client too.
struct PistonStructure<'w> {
    world: &'w World,
    border: Option<PistonWorldBorder>,
    piston_pos: BlockPos,
    block_to_move: BlockPos,
    move_direction: Facing,
    to_move: Vec<BlockPos>,
    to_destroy: Vec<BlockPos>,
}

impl<'w> PistonStructure<'w> {
    fn new(
        world: &'w World,
        pos: BlockPos,
        piston_facing: Facing,
        extending: bool,
        border: Option<PistonWorldBorder>,
    ) -> Self {
        let (move_direction, block_to_move) = if extending {
            (piston_facing, pos.offset(piston_facing, 1))
        } else {
            (piston_facing.opposite(), pos.offset(piston_facing, 2))
        };
        PistonStructure {
            world,
            border,
            piston_pos: pos,
            block_to_move,
            move_direction,
            to_move: Vec::new(),
            to_destroy: Vec::new(),
        }
    }

    /// java: `canMove()` (`:38-71`).
    fn can_move(&mut self) -> bool {
        self.to_move.clear();
        self.to_destroy.clear();
        let block = self.world.block_at(self.block_to_move).block();
        if !piston_can_push_with_border(
            self.world,
            self.block_to_move,
            self.move_direction,
            false,
            self.border,
        ) {
            if block_info(block).mobility_flag != 1 {
                return false;
            }
            self.to_destroy.push(self.block_to_move);
            return true;
        }
        if !self.add_branch(self.block_to_move) {
            return false;
        }
        // java: `for (int i = 0; i < this.toMove.size(); ++i)` — the bound is re-read every
        // iteration and `func_177250_b` can append, so this must not be a snapshot range.
        let mut i = 0;
        while i < self.to_move.len() {
            let p = self.to_move[i];
            if self.world.block_at(p).block().0 == SLIME_BLOCK_ID && !self.check_slime_sides(p) {
                return false;
            }
            i += 1;
        }
        true
    }

    /// java: `func_177251_a(BlockPos origin)` (`:73-176`).
    fn add_branch(&mut self, origin: BlockPos) -> bool {
        let mut block = self.world.block_at(origin).block();
        if self.world.block_at(origin).props().material == Material::Air {
            return true;
        }
        if !piston_can_push_with_border(self.world, origin, self.move_direction, false, self.border)
        {
            return true;
        }
        if origin == self.piston_pos {
            return true;
        }
        if self.to_move.contains(&origin) {
            return true;
        }
        let back = self.move_direction.opposite();
        let mut i = 1usize;
        if i + self.to_move.len() > PISTON_PUSH_LIMIT {
            return false;
        }
        // java: the slime-block chain walk — a slime block drags whatever is stuck behind it.
        while block.0 == SLIME_BLOCK_ID {
            let p = origin.offset(back, i as i32);
            let state = self.world.block_at(p);
            block = state.block();
            if state.props().material == Material::Air
                || !piston_can_push_with_border(
                    self.world,
                    p,
                    self.move_direction,
                    false,
                    self.border,
                )
                || p == self.piston_pos
            {
                break;
            }
            i += 1;
            if i + self.to_move.len() > PISTON_PUSH_LIMIT {
                return false;
            }
        }

        let mut i1 = 0usize;
        for j in (0..i).rev() {
            self.to_move.push(origin.offset(back, j as i32));
            i1 += 1;
        }

        let mut j1 = 1i32;
        loop {
            let p = origin.offset(self.move_direction, j1);
            if let Some(k) = self.to_move.iter().position(|q| *q == p) {
                // java: `func_177255_a(i1, k)` — the block just found is already scheduled, so
                // the list is rotated to put this whole branch in front of it.
                self.reorder(i1, k);
                for l in 0..=(k + i1) {
                    let q = self.to_move[l];
                    if self.world.block_at(q).block().0 == SLIME_BLOCK_ID
                        && !self.check_slime_sides(q)
                    {
                        return false;
                    }
                }
                return true;
            }
            let state = self.world.block_at(p);
            block = state.block();
            if state.props().material == Material::Air {
                return true;
            }
            if !piston_can_push_with_border(self.world, p, self.move_direction, true, self.border)
                || p == self.piston_pos
            {
                return false;
            }
            if block_info(block).mobility_flag == 1 {
                self.to_destroy.push(p);
                return true;
            }
            if self.to_move.len() >= PISTON_PUSH_LIMIT {
                return false;
            }
            self.to_move.push(p);
            i1 += 1;
            j1 += 1;
        }
    }

    /// java: `func_177255_a(int, int)` (`:178-190`).
    fn reorder(&mut self, tail: usize, split: usize) {
        let len = self.to_move.len();
        let mut rebuilt = Vec::with_capacity(len);
        rebuilt.extend_from_slice(&self.to_move[0..split]);
        rebuilt.extend_from_slice(&self.to_move[len - tail..len]);
        rebuilt.extend_from_slice(&self.to_move[split..len - tail]);
        self.to_move = rebuilt;
    }

    /// java: `func_177250_b(BlockPos)` (`:192-203`) — a slime block also drags whatever sticks
    /// to its four *perpendicular* faces.
    fn check_slime_sides(&mut self, pos: BlockPos) -> bool {
        for facing in Facing::VALUES {
            if facing.axis() != self.move_direction.axis()
                && !self.add_branch(pos.offset(facing, 1))
            {
                return false;
            }
        }
        true
    }
}

/// One tick of `TileEntityChest.update`'s client half (`:349-409`).
fn tick_chest(
    pos: BlockPos,
    chest: &mut ChestState,
    adj: ChestNeighbours,
    sounds: &mut Vec<TileEntitySound>,
) {
    chest.prev_lid_angle = chest.lid_angle;
    let step = 0.1f32;

    // java: the open sound fires from the `-X`/`-Z` half only, shifted toward the other half.
    if chest.num_players_using > 0 && chest.lid_angle == 0.0 && !adj.z_neg && !adj.x_neg {
        sounds.push(chest_sound(pos, adj, "random.chestopen"));
    }

    if (chest.num_players_using == 0 && chest.lid_angle > 0.0)
        || (chest.num_players_using > 0 && chest.lid_angle < 1.0)
    {
        let was = chest.lid_angle;
        if chest.num_players_using > 0 {
            chest.lid_angle += step;
        } else {
            chest.lid_angle -= step;
        }
        if chest.lid_angle > 1.0 {
            chest.lid_angle = 1.0;
        }
        // java: `if (this.lidAngle < f3 && f2 >= f3 ...)` with `f3 = 0.5F` — the *downward*
        // crossing of the halfway point, which is why a closing chest clicks and an opening one
        // does not.
        if chest.lid_angle < 0.5 && was >= 0.5 && !adj.z_neg && !adj.x_neg {
            sounds.push(chest_sound(pos, adj, "random.chestclosed"));
        }
        if chest.lid_angle < 0.0 {
            chest.lid_angle = 0.0;
        }
    }
}

/// java: the `d1`/`d2` pair both chest sounds are played at (`:352-368`, `:396-410`).
fn chest_sound(pos: BlockPos, adj: ChestNeighbours, name: &'static str) -> TileEntitySound {
    let mut x = f64::from(pos.x) + 0.5;
    let mut z = f64::from(pos.z) + 0.5;
    if adj.z_pos {
        z += 0.5;
    }
    if adj.x_pos {
        x += 0.5;
    }
    TileEntitySound {
        x,
        y: f64::from(pos.y) + 0.5,
        z,
        name,
        volume: 0.5,
    }
}

/// java: `MobSpawnerBaseLogic.updateSpawner`'s `isRemote` branch (`:82-102`).
///
/// The miniature spins **faster as the spawn approaches**, because the increment is
/// `1000 / (spawnDelay + 200)` and `spawnDelay` counts down.
///
/// The whole body — animation included — sits inside `if (this.isActivated())` (`:84`), which is
/// `world.isAnyPlayerWithinRangeAt(x + 0.5, y + 0.5, z + 0.5, activatingRangeFromPlayer)`
/// (`:76-80`, `world/World.java:3319-3337`), a **squared** distance test against the player's
/// feet with a 16-block default. So a spawner outside that radius is frozen, not merely
/// unspawning: `spawnDelay` stops counting down and `prevMobRotation`/`mobRotation` stop
/// advancing, which also means it is still where it was when you left.
///
/// The two particles the same branch spawns (`SMOKE_NORMAL` and `FLAME`, one of each per tick at
/// a random point inside the block) *are* client-generated — `RenderGlobal.spawnParticle` is
/// implemented, unlike `playSound`. They are spawned by `mc-client`'s `tiles.rs`, which has the
/// particle system and an RNG; this crate reports *which* spawners were activated through
/// [`World::activated_spawners`], so the activation rule below stays the only copy of itself.
fn tick_spawner(pos: BlockPos, spawner: &mut SpawnerState, players: &[DVec3]) {
    if !spawner_is_activated(pos, spawner.activating_range, players) {
        return;
    }
    if spawner.spawn_delay > 0 {
        spawner.spawn_delay -= 1;
    }
    spawner.prev_mob_rotation = spawner.mob_rotation;
    spawner.mob_rotation = (spawner.mob_rotation
        + f64::from(1000.0f32 / (spawner.spawn_delay as f32 + 200.0)))
        % 360.0;
}

/// java: `MobSpawnerBaseLogic.isActivated` (`tileentity/MobSpawnerBaseLogic.java:76-80`) →
/// `World.isAnyPlayerWithinRangeAt` (`world/World.java:3319-3337`).
///
/// A **squared** distance test against the player's *feet*, from the block's centre, with a
/// `range < 0.0` escape hatch that means "always on". Strict `<`, so a player exactly
/// `activatingRangeFromPlayer` away does not activate it.
fn spawner_is_activated(pos: BlockPos, range: i32, players: &[DVec3]) -> bool {
    let range = f64::from(range);
    players.iter().any(|player| {
        let dx = player.x - (f64::from(pos.x) + 0.5);
        let dy = player.y - (f64::from(pos.y) + 0.5);
        let dz = player.z - (f64::from(pos.z) + 0.5);
        // java: `range < 0.0D || d0 < range * range`
        range < 0.0 || dx * dx + dy * dy + dz * dz < range * range
    })
}

/// java: `TileEntityEnchantmentTable.update` (`:54-129`).
///
/// The page target uses the world's shared Java RNG exactly.  In particular, the `||` below
/// short-circuits: while the book is still opening, Java does not consume the one-in-forty
/// gate draw, but it does consume the two (or more, after a zero difference) step draws.
fn tick_enchant(
    pos: BlockPos,
    enchant: &mut EnchantState,
    players: &[DVec3],
    random: &mut JavaRandom,
) {
    enchant.book_spread_prev = enchant.book_spread;
    enchant.book_rotation_prev = enchant.book_rotation;

    let centre = DVec3::new(
        f64::from(pos.x) + 0.5,
        f64::from(pos.y) + 0.5,
        f64::from(pos.z) + 0.5,
    );
    let mut best_distance_sq = 9.0;
    let mut near = None;
    for player in players {
        let distance_sq = player.distance_squared(centre);
        if distance_sq < best_distance_sq {
            best_distance_sq = distance_sq;
            near = Some(*player);
        }
    }

    if let Some(player) = near {
        // java: `field_145924_q = (float) MathHelper.atan2(d1, d0)`
        // (`tileentity/TileEntityEnchantmentTable.java:64`) with `d0 = px - (x + 0.5)` and
        // `d1 = pz - (z + 0.5)` — note the argument order is `(dz, dx)`.
        //
        // **`MathHelper.atan2`, not `Math.atan2`.** `MathHelper.java:411-470` is a table-driven
        // approximation (a 257-entry `asin` table, a fast inverse square root, and a cubic
        // correction), not a wrapper — it disagrees with the libm result in the last few ulps,
        // and the difference compounds through the `* 0.4` easing below. `mc_math::atan2` is
        // that function, tables and all; `f64::atan2` was the wrong one.
        enchant.book_target = mc_math::atan2(player.z - centre.z, player.x - centre.x) as f32;
        enchant.book_spread += 0.1;
        // java: `if (bookSpread < 0.5F || rand.nextInt(40) == 0)`, followed by the
        // retry-until-different `nextInt(4) - nextInt(4)` target nudge.
        if enchant.book_spread < 0.5 || random.next_i32_bound(40) == 0 {
            let previous_target = enchant.flip_target;
            while enchant.flip_target == previous_target {
                enchant.flip_target += (random.next_i32_bound(4) - random.next_i32_bound(4)) as f32;
            }
        }
    } else {
        enchant.book_target += 0.02;
        enchant.book_spread -= 0.1;
    }

    enchant.book_rotation = wrap_pi(enchant.book_rotation);
    enchant.book_target = wrap_pi(enchant.book_target);
    let delta = wrap_pi(enchant.book_target - enchant.book_rotation);
    enchant.book_rotation += delta * 0.4;
    enchant.book_spread = enchant.book_spread.clamp(0.0, 1.0);
    enchant.tick_count += 1;
    enchant.page_flip_prev = enchant.page_flip;
    let f = ((enchant.flip_target - enchant.page_flip) * 0.4).clamp(-0.2, 0.2);
    enchant.flip_velocity += (f - enchant.flip_velocity) * 0.9;
    enchant.page_flip += enchant.flip_velocity;
}

/// java: the `while (x >= PI) x -= 2PI; while (x < -PI) x += 2PI;` pairs the enchantment table
/// runs on three separate angles.
fn wrap_pi(mut angle: f32) -> f32 {
    use core::f32::consts::PI;
    while angle >= PI {
        angle -= PI * 2.0;
    }
    while angle < -PI {
        angle += PI * 2.0;
    }
    angle
}

/// The already-interpolated lid angles of the chest at `pos` and of its `-X`/`-Z` neighbours.
///
/// java: `TileEntityChestRenderer.java:148-160` — `f = prev + (lid - prev) * partialTicks`, then
/// the same for `adjacentChestZNeg` and `adjacentChestXNeg`. The renderer draws only the half
/// without those negative neighbours, so a valid double chest normally supplies an empty slice;
/// both halves swing together because the server sends the same `numPlayersUsing` event to both
/// tiles. The max rule itself remains in
/// `mc_render::entity::tile::chest_lid_angle` owns the max rule; this slice carries the exact
/// Java fallback/incomplete-adjacency input.
#[must_use]
pub fn chest_lid_inputs(world: &World, pos: BlockPos, partial_ticks: f32) -> (f32, f32, Vec<f32>) {
    let lerp = |c: &ChestState| c.prev_lid_angle + (c.lid_angle - c.prev_lid_angle) * partial_ticks;
    let (prev, cur) = world
        .tile_entity(pos)
        .and_then(TileEntity::chest)
        .map_or((0.0, 0.0), |c| (c.prev_lid_angle, c.lid_angle));
    let adj = world.chest_neighbours(pos);
    let mut neighbours = Vec::new();
    for (has, p) in [(adj.z_neg, pos.north()), (adj.x_neg, pos.west())] {
        if !has {
            continue;
        }
        if let Some(chest) = world.tile_entity(p).and_then(TileEntity::chest) {
            neighbours.push(lerp(chest));
        }
    }
    (prev, cur, neighbours)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_data::BlockState;

    fn state(block: u16, meta: u8) -> BlockState {
        BlockState::new(BlockId(block), meta)
    }

    #[test]
    fn raw_tile_prefilter_matches_validated_block_lookup_for_every_packed_state() {
        let mut section = crate::chunk::ChunkSection::empty();
        for raw_id in 0..=u16::MAX {
            let raw = BlockState::from_raw(raw_id);
            section.set_block(0, 0, 0, raw);
            let validated = section.block(0, 0, 0);
            let expected = if validated.id == 0 {
                None
            } else {
                TileKind::from_block(validated.block())
            };
            assert_eq!(
                tile_kind_from_raw_state(raw),
                expected,
                "raw packed state {raw_id:#06x} changed scan semantics"
            );
        }
    }

    #[test]
    fn provider_hint_never_skips_a_provider_for_any_packed_state() {
        for raw_id in 0..=u16::MAX {
            // Start fresh each time so a previous provider's sticky hint cannot hide a miss.
            let mut section = crate::chunk::ChunkSection::empty();
            section.set_block(0, 0, 0, BlockState::from_raw(raw_id));
            let expected = TileKind::from_block(section.block(0, 0, 0).block());
            assert!(
                expected.is_none() || section.may_contain_tile_provider(),
                "raw packed state {raw_id:#06x} would skip a provider"
            );
        }
    }

    #[test]
    fn raw_tile_prefilter_preserves_section_y_z_x_scan_order() {
        let first = BlockPos::new(1, 0, 1);
        let second = BlockPos::new(2, 0, 1);
        let third = BlockPos::new(0, 0, 2);
        let fourth = BlockPos::new(0, 16, 0);
        let world = world_with(&[
            (fourth, state(138, 0)),
            (third, state(63, 0)),
            (second, state(61, 2)),
            (first, state(54, 2)),
        ]);
        let found = world.scan_chunk_tile_entities(ChunkPos::new(0, 0));
        assert_eq!(
            found,
            vec![
                (first, TileKind::Chest),
                (second, TileKind::Furnace),
                (third, TileKind::StandingSign),
                (fourth, TileKind::Beacon),
            ]
        );
    }

    fn world_with(blocks: &[(BlockPos, BlockState)]) -> World {
        let mut world = World::new();
        let mut seen = Vec::new();
        for (pos, _) in blocks {
            let cp = pos.chunk_pos();
            if !seen.contains(&(cp.x, cp.z)) {
                seen.push((cp.x, cp.z));
                world.load_empty_chunk(cp);
            }
        }
        for &(pos, s) in blocks {
            world.set_block(pos, s);
        }
        world
    }

    #[test]
    fn every_vanilla_tile_entity_provider_block_maps_to_the_right_kind_and_id() {
        // java: `BlockContainer` plus `BlockRedstoneComparator`'s 27 provider ids, with
        // `TileEntity.addMapping`'s exact registry strings. Metadata is intentionally included:
        // `createNewTileEntity(World, int)` receives it, but every provider except flower pots
        // has the same class for all metadata values.
        for (id, want) in [
            (23u16, TileKind::Dispenser),
            (25u16, TileKind::Note),
            (36, TileKind::Piston),
            (52, TileKind::MobSpawner),
            (54, TileKind::Chest),
            (61, TileKind::Furnace),
            (62, TileKind::Furnace),
            (63, TileKind::StandingSign),
            (68, TileKind::WallSign),
            (84, TileKind::Jukebox),
            (116, TileKind::EnchantmentTable),
            (117, TileKind::BrewingStand),
            (119, TileKind::EndPortal),
            (130, TileKind::EnderChest),
            (137, TileKind::CommandBlock),
            (138, TileKind::Beacon),
            (140, TileKind::FlowerPot),
            (144, TileKind::Skull),
            (146, TileKind::TrappedChest),
            (149, TileKind::Comparator),
            (150, TileKind::Comparator),
            (151, TileKind::DaylightDetector),
            (154, TileKind::Hopper),
            (158, TileKind::Dropper),
            (176, TileKind::StandingBanner),
            (177, TileKind::WallBanner),
            (178, TileKind::DaylightDetector),
        ] {
            for meta in 0..=15 {
                assert_eq!(
                    TileKind::from_block(BlockId(id)),
                    Some(want),
                    "block {id} meta {meta} should create {want:?}"
                );
            }
            assert_eq!(
                want.id_string(),
                match want {
                    TileKind::Dispenser => "Trap",
                    TileKind::Chest | TileKind::TrappedChest => "Chest",
                    TileKind::EnderChest => "EnderChest",
                    TileKind::Furnace => "Furnace",
                    TileKind::Jukebox => "RecordPlayer",
                    TileKind::StandingSign | TileKind::WallSign => "Sign",
                    TileKind::StandingBanner | TileKind::WallBanner => "Banner",
                    TileKind::Skull => "Skull",
                    TileKind::Beacon => "Beacon",
                    TileKind::MobSpawner => "MobSpawner",
                    TileKind::EnchantmentTable => "EnchantTable",
                    TileKind::EndPortal => "Airportal",
                    TileKind::FlowerPot => "FlowerPot",
                    TileKind::BrewingStand => "Cauldron",
                    TileKind::Comparator => "Comparator",
                    TileKind::DaylightDetector => "DLDetector",
                    TileKind::Hopper => "Hopper",
                    TileKind::Dropper => "Dropper",
                    TileKind::Piston => "Piston",
                    TileKind::Note => "Music",
                    TileKind::CommandBlock => "Control",
                }
            );
        }
        for id in [
            0, 1, 22, 24, 60, 83, 115, 118, 139, 143, 148, 153, 157, 175, 179, 197,
        ] {
            assert_eq!(
                TileKind::from_block(BlockId(id)),
                None,
                "block {id} is not a provider"
            );
        }
    }

    #[test]
    fn every_provider_creates_a_matching_world_tile_entity_except_piston_null_factory() {
        // Java's `Chunk.getTileEntity(..., IMMEDIATE)` creates the provider object. Piston
        // extension is the one deliberate exception: BlockPistonMoving.createNewTileEntity
        // returns null, despite implementing ITileEntityProvider.
        let providers = [
            (23u16, 2u8),
            (25, 0),
            (52, 0),
            (54, 2),
            (61, 2),
            (62, 2),
            (63, 0),
            (68, 2),
            (84, 0),
            (116, 0),
            (117, 0),
            (119, 0),
            (130, 2),
            (137, 0),
            (138, 0),
            (140, 0),
            (144, 1),
            (146, 2),
            (149, 2),
            (150, 2),
            (151, 0),
            (154, 0),
            (158, 2),
            (176, 0),
            (177, 2),
            (178, 0),
        ];
        for (id, meta) in providers {
            let pos = BlockPos::new(i32::from(id), 64, 0);
            let mut world = world_with(&[(pos, state(id, meta))]);
            let kind = TileKind::from_block(BlockId(id)).expect("provider kind");
            let entity = world
                .tile_entity_or_create(pos)
                .unwrap_or_else(|| panic!("non-piston provider factory for block {id}"));
            assert_eq!(entity.kind, kind, "block {id} kind");
            assert_eq!(entity.kind.id_string(), kind.id_string(), "block {id} id");
        }
        let pos = BlockPos::new(36, 64, 0);
        let mut world = world_with(&[(pos, state(36, 0))]);
        assert!(world.tile_entity_or_create(pos).is_none());
    }

    #[test]
    fn the_block_creates_the_tile_entity_without_any_packet() {
        // The whole point of A05-01: a chest is a chest the moment its block lands, because
        // `Chunk.createNewTileEntity` is a function of the block. No `S35` is ever sent for one.
        let pos = BlockPos::new(4, 70, 9);
        let mut world = world_with(&[(pos, state(CHEST_ID, 2))]);
        assert_eq!(world.tile_kind_at(pos), Some(TileKind::Chest));
        assert_eq!(world.tile_entity_count(), 1, "created inside set_block");
        let te = world.tile_entity_or_create(pos).expect("chest");
        assert_eq!(te.kind, TileKind::Chest);
        assert!(matches!(te.data, TileData::Chest(_)));
        assert_eq!(world.tile_entity_count(), 1);
    }

    #[test]
    fn a_tile_entity_cannot_be_stored_on_a_block_that_has_none() {
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_with(&[(pos, state(1, 0))]);
        world.set_tile_entity(pos, TileEntity::new(TileKind::Chest));
        assert_eq!(world.tile_entity_count(), 0);
        assert!(world.tile_entity_or_create(pos).is_none());
    }

    #[test]
    fn sign_text_lands_only_on_a_sign() {
        let sign = BlockPos::new(1, 64, 1);
        let stone = BlockPos::new(2, 64, 1);
        let mut world = world_with(&[(sign, state(63, 4)), (stone, state(1, 0))]);
        let lines = [
            "hello".to_owned(),
            "world".to_owned(),
            String::new(),
            String::new(),
        ];
        assert!(world.apply_sign_text(sign, lines.clone()));
        assert_eq!(
            world
                .tile_entity(sign)
                .and_then(TileEntity::sign)
                .unwrap()
                .lines[0],
            "hello"
        );
        // java: `handleUpdateSign`'s `flag` stays false and the client chat-warns.
        assert!(!world.apply_sign_text(stone, lines));
    }

    #[test]
    fn sign_update_is_acknowledged_but_ignored_while_gui_edit_sign_locks_it() {
        let sign = BlockPos::new(4, 64, 1);
        let stone = BlockPos::new(5, 64, 1);
        let missing = BlockPos::new(40, 64, 1);
        let mut world = world_with(&[(sign, state(63, 4)), (stone, state(1, 0))]);
        let initial_display = [
            "old".to_owned(),
            "keep".to_owned(),
            String::new(),
            String::new(),
        ];
        let initial_components = [
            r#"{"text":"old","color":"red"}"#.to_owned(),
            r#"{"text":"keep","bold":true}"#.to_owned(),
            "\"\"".to_owned(),
            "\"\"".to_owned(),
        ];
        assert!(world.apply_sign_components(
            sign,
            initial_display.clone(),
            initial_components.clone()
        ));

        // The editor's local text is a render-state update, not an S33 packet.  Once
        // `GuiEditSign.initGui` has called setEditable(false), both S33 helpers must retain it.
        let editing_display = [
            "typing".to_owned(),
            "keep".to_owned(),
            String::new(),
            String::new(),
        ];
        assert!(world.set_sign_editor_render_state(sign, editing_display.clone(), 0));
        assert!(world.set_sign_editable(sign, false));
        let incoming_display = [
            "server".to_owned(),
            "replacement".to_owned(),
            "new".to_owned(),
            String::new(),
        ];
        let incoming_components = [
            r#"{"text":"server"}"#.to_owned(),
            r#"{"text":"replacement"}"#.to_owned(),
            r#"{"text":"new"}"#.to_owned(),
            "\"\"".to_owned(),
        ];
        assert!(world.apply_sign_text(sign, incoming_display.clone()));
        assert!(world.apply_sign_components(
            sign,
            incoming_display.clone(),
            incoming_components.clone()
        ));
        let locked = world.tile_entity(sign).and_then(TileEntity::sign).unwrap();
        assert!(!locked.editable);
        assert_eq!(locked.lines, editing_display);
        assert_eq!(locked.components, initial_components);
        assert_eq!(locked.line_being_edited, 0);

        // Java sets flag=true for a real sign even when getIsEditable() rejects the copy; only a
        // non-sign or an unloaded position takes the warning path.
        assert!(!world.apply_sign_components(
            stone,
            incoming_display.clone(),
            incoming_components.clone()
        ));
        assert!(!world.apply_sign_components(
            missing,
            incoming_display.clone(),
            incoming_components.clone()
        ));
        assert!(world.tile_entity(missing).is_none());

        // GuiEditSign.onGuiClosed restores editability before the next server update.  The host
        // also clears the transient caret when reopening the ordinary render path.
        assert!(world.set_sign_editable(sign, true));
        let reopened = world.tile_entity(sign).and_then(TileEntity::sign).unwrap();
        assert!(reopened.editable);
        assert_eq!(reopened.line_being_edited, -1);
        assert!(world.apply_sign_components(sign, incoming_display, incoming_components.clone()));
        let updated = world.tile_entity(sign).and_then(TileEntity::sign).unwrap();
        assert_eq!(updated.lines[0], "server");
        assert_eq!(updated.components, incoming_components);
    }

    #[test]
    fn the_action_byte_must_agree_with_the_block() {
        // java: `handleUpdateTileEntity:1276` — `i == 4 && tileentity instanceof TileEntitySkull`.
        let skull = BlockPos::new(3, 64, 3);
        let mut world = world_with(&[(skull, state(144, 1))]);
        let data = TileData::Skull(SkullState {
            skull_type: 3,
            rotation: 5,
            owner_name: Some("Notch".to_owned()),
            owner_profile: None,
            skin_url: Some("https://textures.example/notch.png".to_owned()),
        });
        assert!(
            !world.apply_tile_entity_nbt(skull, 3, data.clone()),
            "wrong action"
        );
        assert!(world.apply_tile_entity_nbt(skull, 4, data), "right action");
        let stored = world
            .tile_entity(skull)
            .and_then(TileEntity::skull)
            .unwrap();
        assert_eq!(stored.skull_type, 3);
        assert_eq!(stored.rotation, 5);
        assert_eq!(stored.owner_name.as_deref(), Some("Notch"));
        assert_eq!(
            stored.skin_url.as_deref(),
            Some("https://textures.example/notch.png")
        );
    }

    #[test]
    fn command_block_state_is_typed_and_rejected_updates_are_atomic() {
        let pos = BlockPos::new(3, 64, 3);
        let stone = BlockPos::new(4, 64, 3);
        let mut world = world_with(&[(pos, state(137, 0)), (stone, state(1, 0))]);

        let initial = world
            .tile_entity(pos)
            .and_then(TileEntity::command_block)
            .expect("command blocks are lazily created with typed state");
        assert_eq!(initial, &CommandBlockState::default());

        let accepted = CommandBlockState {
            command: "/say hello".to_owned(),
            custom_name: "Builder".to_owned(),
            success_count: 7,
            track_output: true,
            last_output: Some(r#"{"text":"ok"}"#.to_owned()),
        };
        assert!(world.apply_tile_entity_nbt(pos, 2, TileData::CommandBlock(accepted.clone())));
        assert_eq!(
            world.tile_entity(pos).and_then(TileEntity::command_block),
            Some(&accepted)
        );

        // Java's action/type conjunction rejects before readFromNBT, so no field may be
        // partially replaced by a wrong action.
        let before_rejected = accepted.clone();
        let rejected = CommandBlockState {
            command: "/say rejected".to_owned(),
            success_count: 99,
            ..accepted.clone()
        };
        assert!(!world.apply_tile_entity_nbt(pos, 3, TileData::CommandBlock(rejected.clone())));
        assert_eq!(
            world.tile_entity(pos).and_then(TileEntity::command_block),
            Some(&before_rejected)
        );

        // A matching action cannot manufacture a command-block tile on a non-provider block.
        assert!(!world.apply_tile_entity_nbt(stone, 2, TileData::CommandBlock(rejected)));
        assert!(world.tile_entity(stone).is_none());
    }

    #[test]
    fn a_block_action_opens_the_chest_and_the_lid_integrates() {
        // java: `TileEntityChest.receiveClientEvent(1, n)` then `update()`'s ±0.1 per tick.
        let pos = BlockPos::new(8, 64, 8);
        let mut world = world_with(&[(pos, state(CHEST_ID, 2))]);
        let effect = world.apply_block_action(pos, i32::from(CHEST_ID), 1, 1);
        assert_eq!(effect, Some(BlockActionEffect::ChestUsers { players: 1 }));

        let sounds = world.tick_tile_entities(None);
        assert_eq!(sounds.len(), 1, "the 0 -> open transition plays a sound");
        assert_eq!(sounds[0].name, "random.chestopen");
        let lid = world.tile_entity(pos).and_then(TileEntity::chest).unwrap();
        assert!((lid.lid_angle - 0.1).abs() < 1e-6, "{}", lid.lid_angle);
        assert_eq!(lid.prev_lid_angle, 0.0);

        for _ in 0..20 {
            world.tick_tile_entities(None);
        }
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::chest)
                .unwrap()
                .lid_angle,
            1.0,
            "clamped fully open"
        );

        // Closing crosses 0.5 downward exactly once, which is the click.
        world.apply_block_action(pos, i32::from(CHEST_ID), 1, 0);
        let mut clicks = 0;
        for _ in 0..20 {
            clicks += world
                .tick_tile_entities(None)
                .iter()
                .filter(|s| s.name == "random.chestclosed")
                .count();
        }
        assert_eq!(clicks, 1);
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::chest)
                .unwrap()
                .lid_angle,
            0.0
        );
    }

    #[test]
    fn a_chest_action_on_a_non_tile_block_has_no_receiver() {
        // java: `World.addBlockEvent` still dispatches this packet's chest block, but
        // `BlockContainer.onBlockEventReceived` only forwards it when `getTileEntity(pos)` is
        // non-null. Stone has none, so this is ignored for that per-block reason -- not because
        // `World.addBlockEvent` compared the packet block with the world's state.
        let pos = BlockPos::new(8, 64, 8);
        let mut world = world_with(&[(pos, state(1, 0))]);
        assert_eq!(
            world.apply_block_action(pos, i32::from(CHEST_ID), 1, 1),
            None
        );
    }

    #[test]
    fn a_note_block_action_carries_the_vanilla_pitch() {
        // java: `(double) Math.pow(2.0D, (double)(param - 12) / 12.0D)`.
        let pos = BlockPos::new(0, 64, 5);
        let mut world = world_with(&[(pos, state(25, 0))]);
        let effect = world.apply_block_action(pos, 25, 2, 12);
        match effect {
            Some(BlockActionEffect::Note { pitch, .. }) => {
                assert!(
                    (pitch - 1.0).abs() < 1e-6,
                    "note 12 is pitch 1.0, got {pitch}"
                );
            }
            other => panic!("expected a note, got {other:?}"),
        }
        match world.apply_block_action(pos, 25, 2, 24) {
            Some(BlockActionEffect::Note { pitch, .. }) => {
                assert!(
                    (pitch - 2.0).abs() < 1e-5,
                    "an octave up is 2.0, got {pitch}"
                );
            }
            other => panic!("expected a note, got {other:?}"),
        }
    }

    #[test]
    fn note_block_pitch_is_double_pow_then_single_narrow_like_java() {
        // java: BlockNote.onBlockEventReceived casts the result of
        // `Math.pow(2.0D, (double)(data2 - 12) / 12.0D)` to float.  Check all wire-byte values:
        // the packet does not validate `data2`, and this includes values for which f32::powf
        // rounds differently.
        let pos = BlockPos::new(0, 64, 5);
        let mut differs_from_f32_powf = false;
        for note in 0..=u8::MAX {
            let mut world = world_with(&[(pos, state(25, 0))]);
            let expected = 2.0f64.powf((f64::from(note) - 12.0) / 12.0) as f32;
            differs_from_f32_powf |=
                (2.0f32.powf((f32::from(note) - 12.0) / 12.0)).to_bits() != expected.to_bits();
            let Some(BlockActionEffect::Note { pitch, .. }) =
                world.apply_block_action(pos, 25, 0, note)
            else {
                panic!("note block action must yield an effect");
            };
            assert_eq!(
                pitch.to_bits(),
                expected.to_bits(),
                "data2={note} must follow Java's f64 Math.pow then float cast"
            );
        }
        assert!(
            differs_from_f32_powf,
            "the test range must contain a negative control for the old f32::powf port"
        );
    }

    #[test]
    fn a_note_block_action_dispatches_when_the_world_block_differs() {
        // java: `World.addBlockEvent` calls the packet's `BlockNote` directly. Its
        // `onBlockEventReceived` plays the note and spawns the particle without inspecting
        // `state`, so an S24 for a note block still takes effect while this position is stale.
        // Restoring the old `here.block() == block_id` guard makes this fail.
        let pos = BlockPos::new(0, 64, 5);
        let mut world = world_with(&[(pos, state(1, 0))]);
        assert_eq!(
            world.apply_block_action(pos, 25, 2, 12),
            Some(BlockActionEffect::Note {
                instrument: 2,
                note: 12,
                pitch: 1.0,
            })
        );
    }

    #[test]
    fn double_chests_pair_only_with_their_own_type() {
        // java: `isChestAt` — `block.chestType == this.getChestType()`.
        let a = BlockPos::new(0, 64, 0);
        let b = BlockPos::new(1, 64, 0);
        let world = world_with(&[(a, state(CHEST_ID, 2)), (b, state(CHEST_ID, 2))]);
        let na = world.chest_neighbours(a);
        assert!(na.x_pos && !na.x_neg, "{na:?}");
        assert!(
            na.draws() && na.is_double(),
            "the -X half draws the big model"
        );
        let nb = world.chest_neighbours(b);
        assert!(!nb.draws(), "the +X half draws nothing");

        let mixed = world_with(&[(a, state(CHEST_ID, 2)), (b, state(TRAPPED_CHEST_ID, 2))]);
        assert!(
            !mixed.chest_neighbours(a).any(),
            "a trapped chest is a different type"
        );
    }

    #[test]
    fn chest_lid_inputs_interpolate_each_half_and_only_probe_negative_neighbours() {
        // Java's renderer draws the half with no -X/-Z neighbour.  Keep both halves asymmetric
        // so a producer that swaps prev/current values, probes +X, or ignores the neighbour
        // direction cannot pass this oracle.
        let west = BlockPos::new(0, 64, 0);
        let east = west.east();
        let mut world = world_with(&[(west, state(CHEST_ID, 2)), (east, state(CHEST_ID, 2))]);
        let set_lid = |world: &mut World, pos: BlockPos, prev: f32, current: f32| {
            let chest = world
                .tile_entity_mut(pos)
                .and_then(|te| match &mut te.data {
                    TileData::Chest(chest) => Some(chest),
                    _ => None,
                })
                .expect("set_block creates the chest tile");
            chest.prev_lid_angle = prev;
            chest.lid_angle = current;
        };
        set_lid(&mut world, west, 0.2, 0.8);
        set_lid(&mut world, east, 0.7, 0.3);

        let (west_prev, west_current, west_neighbours) = chest_lid_inputs(&world, west, 0.25);
        assert_eq!((west_prev, west_current), (0.2, 0.8));
        assert!(west_neighbours.is_empty());
        assert!(world.chest_neighbours(west).draws());
        assert!(world.chest_neighbours(west).is_double());

        let (east_prev, east_current, east_neighbours) = chest_lid_inputs(&world, east, 0.25);
        assert_eq!((east_prev, east_current), (0.7, 0.3));
        assert_eq!(east_neighbours.len(), 1);
        assert!(
            (east_neighbours[0] - 0.35).abs() < 1e-6,
            "the west half interpolates 0.2 -> 0.8 at partial 0.25"
        );
        assert!(!world.chest_neighbours(east).draws());

        // A single chest and a blocked/mismatched neighbour must not manufacture a lid input.
        let single = BlockPos::new(5, 64, 0);
        let single_world = world_with(&[(single, state(CHEST_ID, 2))]);
        let (_, _, none) = chest_lid_inputs(&single_world, single, 0.5);
        assert!(none.is_empty());

        let mixed = world_with(&[
            (single, state(CHEST_ID, 2)),
            (single.east(), state(TRAPPED_CHEST_ID, 2)),
        ]);
        let (_, _, mismatched) = chest_lid_inputs(&mixed, single.east(), 0.5);
        assert!(mismatched.is_empty(), "different chest types cannot pair");
    }

    #[test]
    fn an_ender_chest_never_pairs() {
        // java: `TileEntityEnderChest` is not a `TileEntityChest`, so `checkForAdjacentChests`
        // is not even on it — `TileEntityEnderChestRenderer` always draws `simpleChest`.
        let a = BlockPos::new(0, 64, 0);
        let b = BlockPos::new(1, 64, 0);
        let world = world_with(&[(a, state(ENDER_CHEST_ID, 2)), (b, state(ENDER_CHEST_ID, 2))]);
        assert!(!world.chest_neighbours(a).any());
    }

    #[test]
    fn a_clear_beacon_beam_reaches_the_sky_as_one_white_segment() {
        let pos = BlockPos::new(2, 60, 2);
        let world = world_with(&[(pos, state(138, 0))]);
        let (segments, complete) = world.scan_beacon_beam(pos);
        assert!(complete);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].colors, [1.0, 1.0, 1.0]);
        // Java's BeamSegment constructor contributes the beacon block itself (height 1),
        // then the scan adds y+1 .. 255. The old zero-initialised port reported 195 here.
        assert_eq!(segments[0].height, 256 - 60, "beacon block plus y+1 .. 255");
    }

    #[test]
    fn a_glass_block_immediately_above_beacon_keeps_the_initial_white_unit() {
        // Java starts with `new BeamSegment(WHITE)` whose height is one. A colour change on the
        // first scanned block must therefore produce [WHITE x1, RED x1], never a zero-height
        // white segment followed by red.
        let pos = BlockPos::new(2, 60, 2);
        let world = world_with(&[
            (pos, state(138, 0)),
            (BlockPos::new(2, 61, 2), state(STAINED_GLASS_ID, 14)),
        ]);
        let (segments, complete) = world.scan_beacon_beam(pos);
        assert!(complete);
        assert_eq!(segments.len(), 2);
        assert_eq!(
            segments[0],
            BeamSegment {
                colors: DYE_RGB[0],
                height: 1
            }
        );
        assert_eq!(segments[1].colors, DYE_RGB[14]);
        assert_eq!(segments[1].height, 256 - 61);
    }

    #[test]
    fn an_opaque_block_aborts_the_beam_but_bedrock_does_not() {
        // java: `getLightOpacity() >= 15 && block != Blocks.bedrock` -> isComplete = false.
        let pos = BlockPos::new(2, 60, 2);
        let stone = world_with(&[(pos, state(138, 0)), (BlockPos::new(2, 70, 2), state(1, 0))]);
        assert_eq!(stone.scan_beacon_beam(pos), (Vec::new(), false));

        let bedrock = world_with(&[
            (pos, state(138, 0)),
            (BlockPos::new(2, 70, 2), state(BEDROCK_ID, 0)),
        ]);
        let (segments, complete) = bedrock.scan_beacon_beam(pos);
        assert!(complete, "bedrock is the one exception");
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn beacon_fade_advances_per_visible_frame_and_decays_after_a_render_gap() {
        let pos = BlockPos::new(2, 60, 2);
        let mut world = world_with(&[(pos, state(138, 0))]);
        let entity = world.tile_entity_or_create(pos).expect("beacon tile");
        let TileData::Beacon(beacon) = &mut entity.data else {
            panic!("block 138 must create BeaconState");
        };
        beacon.complete = true;
        beacon.beam_fade = 0.5;
        beacon.beam_render_counter = 100;

        assert!(world.advance_beacon_render_frame(pos, 101));
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert!((beacon.beam_fade - 0.525).abs() < 1e-6);
        assert_eq!(beacon.beam_render_counter, 101);

        // The world clock has not ticked, but Java's mutating renderer getter is called again.
        assert!(world.advance_beacon_render_frame(pos, 101));
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert!((beacon.beam_fade - 0.55).abs() < 1e-6);

        // Incomplete calls return zero without touching either persistent field.
        let entity = world.tile_entity_mut(pos).expect("beacon tile");
        let TileData::Beacon(beacon) = &mut entity.data else {
            unreachable!();
        };
        beacon.complete = false;
        assert!(!world.advance_beacon_render_frame(pos, 120));
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert!((beacon.beam_fade - 0.55).abs() < 1e-6);
        assert_eq!(beacon.beam_render_counter, 101);

        // On re-entry, twenty missed ticks cost 0.5 before this frame contributes 0.025.
        let entity = world.tile_entity_mut(pos).expect("beacon tile");
        let TileData::Beacon(beacon) = &mut entity.data else {
            unreachable!();
        };
        beacon.complete = true;
        assert!(world.advance_beacon_render_frame(pos, 121));
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert!((beacon.beam_fade - 0.075).abs() < 1e-6);
        assert_eq!(beacon.beam_render_counter, 121);
    }

    /// java: `shouldBeamRender` floors the decay at zero before adding
    /// this frame's 0.025, and narrows the elapsed ticks with a wrapping `(int)` cast.
    #[test]
    fn beacon_fade_comes_back_at_one_step_after_any_absence_like_java() {
        let pos = BlockPos::new(2, 60, 2);
        let fresh = |total: i64| {
            let mut world = world_with(&[(pos, state(138, 0))]);
            let entity = world.tile_entity_or_create(pos).expect("beacon tile");
            let TileData::Beacon(beacon) = &mut entity.data else {
                panic!("block 138 must create BeaconState");
            };
            beacon.complete = true;
            assert!(world.advance_beacon_render_frame(pos, total));
            let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
            beacon.beam_fade
        };
        // Never rendered (counter 0) in a 50,000-tick world: 0 - 1250 floors at 0, then +0.025.
        assert!(
            (fresh(50_000) - 0.025).abs() < 1e-6,
            "drawn on its first frame"
        );
        // Past 2^31 ticks `(int)` wraps negative, so Java does not decay at all.
        assert!((fresh(3_000_000_000) - 0.025).abs() < 1e-6);
        // A long absence after a full beam: 1.0 - 400/40 floors at 0, then 0.025.
        let mut world = world_with(&[(pos, state(138, 0))]);
        let entity = world.tile_entity_or_create(pos).expect("beacon tile");
        let TileData::Beacon(beacon) = &mut entity.data else {
            unreachable!();
        };
        beacon.complete = true;
        beacon.beam_fade = 1.0;
        beacon.beam_render_counter = 1_000;
        assert!(world.advance_beacon_render_frame(pos, 1_400));
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert!(
            (beacon.beam_fade - 0.025).abs() < 1e-6,
            "400 ticks away: floor, then a step"
        );
    }

    #[test]
    fn stained_glass_splits_the_beam_and_the_second_colour_is_averaged() {
        // java: `:145` — the first colour is taken straight (`flag` is still true), every later
        // one is `(running + dye) / 2`.
        let pos = BlockPos::new(2, 60, 2);
        let world = world_with(&[
            (pos, state(138, 0)),
            // meta 14 is RED = (0.6, 0.2, 0.2)
            (BlockPos::new(2, 62, 2), state(STAINED_GLASS_ID, 14)),
            // meta 11 is BLUE = (0.2, 0.3, 0.7)
            (BlockPos::new(2, 64, 2), state(STAINED_GLASS_ID, 11)),
        ]);
        let (segments, complete) = world.scan_beacon_beam(pos);
        assert!(complete);
        assert_eq!(
            segments.len(),
            3,
            "white, red, red-blue average: {segments:?}"
        );
        assert_eq!(segments[0].colors, [1.0, 1.0, 1.0]);
        assert_eq!(segments[1].colors, [0.6, 0.2, 0.2]);
        let want = [(0.6 + 0.2) / 2.0, (0.2 + 0.3) / 2.0, (0.2 + 0.7) / 2.0];
        assert_eq!(segments[2].colors, want);
    }

    #[test]
    fn a_full_pyramid_reports_four_levels() {
        // java: `updateSegmentColors`' four rings of iron/gold/diamond/emerald.
        let pos = BlockPos::new(0, 60, 0);
        let mut blocks = vec![(pos, state(138, 0))];
        for l in 1..=4i32 {
            for x in (pos.x - l)..=(pos.x + l) {
                for z in (pos.z - l)..=(pos.z + l) {
                    blocks.push((BlockPos::new(x, pos.y - l, z), state(42, 0)));
                }
            }
        }
        let world = world_with(&blocks);
        assert_eq!(world.beacon_levels(pos), 4);
    }

    #[test]
    fn the_spawner_spins_faster_as_the_delay_falls() {
        // java: `mobRotation += 1000 / (spawnDelay + 200)`.
        let mut fast = SpawnerState {
            spawn_delay: 1,
            ..SpawnerState::default()
        };
        let mut slow = SpawnerState {
            spawn_delay: 800,
            ..SpawnerState::default()
        };
        let pos = BlockPos::new(0, 64, 0);
        let near = [DVec3::new(0.5, 64.5, 0.5)];
        tick_spawner(pos, &mut fast, &near);
        tick_spawner(pos, &mut slow, &near);
        assert!(
            fast.mob_rotation > slow.mob_rotation,
            "{} vs {}",
            fast.mob_rotation,
            slow.mob_rotation
        );
        assert_eq!(fast.prev_mob_rotation, 0.0);
    }

    #[test]
    fn the_enchantment_book_opens_near_a_player_and_closes_away_from_one() {
        let pos = BlockPos::new(0, 64, 0);
        let mut near = EnchantState::default();
        let mut random = JavaRandom::new(0);
        for _ in 0..5 {
            tick_enchant(pos, &mut near, &[DVec3::new(0.5, 64.5, 0.5)], &mut random);
        }
        assert!(near.book_spread > 0.4, "{}", near.book_spread);
        let mut far = near;
        for _ in 0..20 {
            tick_enchant(pos, &mut far, &[], &mut random);
        }
        assert_eq!(far.book_spread, 0.0, "clamped shut");
        assert!(far.tick_count > near.tick_count);
    }

    #[test]
    fn enchantment_nearest_player_is_strict_and_first_on_ties() {
        let pos = BlockPos::new(0, 64, 0);
        let centre = DVec3::new(0.5, 64.5, 0.5);
        let east = centre + DVec3::new(1.0, 0.0, 0.0);
        let west = centre + DVec3::new(-1.0, 0.0, 0.0);

        let mut tied = EnchantState::default();
        tick_enchant(pos, &mut tied, &[east, west], &mut JavaRandom::new(7));
        assert_eq!(
            tied.book_target.to_bits(),
            0.0_f32.to_bits(),
            "equal-distance candidates retain world insertion order"
        );
        assert!(tied.book_spread > 0.0);

        let mut nearer_second = EnchantState::default();
        tick_enchant(
            pos,
            &mut nearer_second,
            &[centre + DVec3::new(2.0, 0.0, 0.0), west],
            &mut JavaRandom::new(7),
        );
        assert!(
            (nearer_second.book_target.abs() - core::f32::consts::PI).abs() < 1e-5,
            "the nearer later candidate must replace a farther first candidate"
        );

        let mut exact_three = EnchantState::default();
        tick_enchant(
            pos,
            &mut exact_three,
            &[centre + DVec3::new(3.0, 0.0, 0.0)],
            &mut JavaRandom::new(7),
        );
        assert_eq!(
            exact_three.book_spread, 0.0,
            "distance squared 9 is excluded"
        );
        assert_eq!(
            exact_three.book_target, 0.02,
            "the book follows its idle arm"
        );

        let mut none = EnchantState::default();
        tick_enchant(pos, &mut none, &[], &mut JavaRandom::new(7));
        assert_eq!(none.book_spread, 0.0);
        assert_eq!(none.book_target, 0.02);
    }

    #[test]
    fn the_enchantment_book_uses_the_idle_one_in_forty_page_turn() {
        let pos = BlockPos::new(0, 64, 0);
        let mut book = EnchantState {
            // At exactly 0.5 Java must consult the random gate; Random(0)'s first
            // nextInt(40) is zero, then its next two nextInt(4)s are 3 and 0.
            book_spread: 0.5,
            flip_target: 1.0,
            ..EnchantState::default()
        };
        let mut random = JavaRandom::new(0);
        tick_enchant(pos, &mut book, &[DVec3::new(0.5, 64.5, 0.5)], &mut random);

        assert_eq!(book.flip_target, 4.0, "the signed Java step is +3, not +1");
        let mut oracle = JavaRandom::new(0);
        assert_eq!(
            oracle.next_i32_bound(40),
            0,
            "test seed must open the idle gate"
        );
        assert_eq!(oracle.next_i32_bound(4) - oracle.next_i32_bound(4), 3);
        assert_eq!(
            random.raw_seed(),
            oracle.raw_seed(),
            "gate and step consume Java draws"
        );
    }

    #[test]
    fn the_enchantment_book_short_circuits_its_idle_gate_while_opening() {
        let pos = BlockPos::new(0, 64, 0);
        let mut book = EnchantState {
            // The `< 0.5` arm must not call nextInt(40), but it still takes a page step.
            book_spread: 0.3,
            ..EnchantState::default()
        };
        let mut random = JavaRandom::new(0);
        tick_enchant(pos, &mut book, &[DVec3::new(0.5, 64.5, 0.5)], &mut random);

        let mut oracle = JavaRandom::new(0);
        let expected_step = oracle.next_i32_bound(4) - oracle.next_i32_bound(4);
        assert_ne!(
            expected_step, 0,
            "the chosen seed must avoid the retry path"
        );
        assert_eq!(book.flip_target, expected_step as f32);
        assert_eq!(
            random.raw_seed(),
            oracle.raw_seed(),
            "a random gate draw while opening would desynchronise World.rand"
        );
    }

    #[test]
    fn unloading_a_chunk_drops_its_tile_entities() {
        let pos = BlockPos::new(4, 70, 9);
        let mut world = world_with(&[(pos, state(CHEST_ID, 2))]);
        world.tile_entity_or_create(pos);
        assert_eq!(world.tile_entity_count(), 1);
        world.unload_chunk(pos.chunk_pos());
        assert_eq!(world.tile_entity_count(), 0);
    }

    #[test]
    fn replacing_the_block_removes_the_stale_tile_entity_inline() {
        let pos = BlockPos::new(4, 70, 9);
        let mut world = world_with(&[(pos, state(CHEST_ID, 2))]);
        assert_eq!(world.tile_entity_count(), 1);
        world.set_block(pos, state(1, 0));
        assert_eq!(world.tile_entity_count(), 0);
    }

    #[test]
    fn changing_only_metadata_preserves_the_tile_entity_instance() {
        // java: `Chunk.setBlockState` removes the old tile entity only when `block1 != block`.
        let pos = BlockPos::new(4, 70, 9);
        let mut world = world_with(&[(pos, state(CHEST_ID, 2))]);
        world
            .tile_entity_mut(pos)
            .and_then(|te| match &mut te.data {
                TileData::Chest(chest) => Some(chest),
                _ => None,
            })
            .expect("chest")
            .lid_angle = 0.75;

        world.set_block(pos, state(CHEST_ID, 5));

        assert_eq!(world.tile_entity_count(), 1);
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::chest)
                .unwrap()
                .lid_angle,
            0.75,
            "metadata updates do not reset client animation state"
        );
    }

    #[test]
    fn flower_pot_creation_uses_all_sixteen_legacy_metadata_values() {
        // java: `BlockFlowerPot.createNewTileEntity` is a tableswitch over 1..=13; the default
        // arm covers 0, 14 and 15. This table pins the item *and* data fed to the tile entity.
        let expected = [
            (0, 0),
            (38, 0),
            (37, 0),
            (6, 0),
            (6, 1),
            (6, 2),
            (6, 3),
            (40, 0),
            (39, 0),
            (81, 0),
            (32, 0),
            (31, 2),
            (6, 4),
            (6, 5),
            (0, 0),
            (0, 0),
        ];
        let mut world = World::new();
        world.load_empty_chunk(ChunkPos::new(0, 0));
        for (meta, &(item_id, data)) in expected.iter().enumerate() {
            let pos = BlockPos::new(meta as i32, 64, 0);
            world.set_block(pos, state(140, meta as u8));
            let pot = world
                .tile_entity(pos)
                .and_then(TileEntity::flower_pot)
                .unwrap_or_else(|| panic!("meta {meta} did not create a flower pot inline"));
            assert_eq!((pot.item_id, pot.data), (item_id, data), "meta {meta}");
        }
    }

    #[test]
    fn flower_pot_contents_name_the_blockstate_variant() {
        // java: `BlockFlowerPot.getActualState` -> `EnumFlowerType`, keyed by
        // `assets/minecraft/blockstates/flower_pot.json`.
        assert_eq!(
            FlowerPotState {
                item_id: 38,
                data: 0
            }
            .contents(),
            "rose"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 38,
                data: 1
            }
            .contents(),
            "blue_orchid"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 6,
                data: 2
            }
            .contents(),
            "birch_sapling"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 31,
                data: 2
            }
            .contents(),
            "fern"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 81,
                data: 0
            }
            .contents(),
            "cactus"
        );
        assert_eq!(FlowerPotState::default().contents(), "empty");
    }

    #[test]
    fn flower_pot_contents_cover_every_model_variant_and_reject_unknown_payloads() {
        // Java's `BlockFlowerPot.getActualState` exposes precisely these 22 values in
        // `assets/minecraft/blockstates/flower_pot.json`. Keep the table here as the content
        // oracle so a new Item/Data arm cannot silently fall back to the empty model.
        let variants = [
            ((0, 0), "empty"),
            ((38, 0), "rose"),
            ((38, 1), "blue_orchid"),
            ((38, 2), "allium"),
            ((38, 3), "houstonia"),
            ((38, 4), "red_tulip"),
            ((38, 5), "orange_tulip"),
            ((38, 6), "white_tulip"),
            ((38, 7), "pink_tulip"),
            ((38, 8), "oxeye_daisy"),
            ((37, 0), "dandelion"),
            ((6, 0), "oak_sapling"),
            ((6, 1), "spruce_sapling"),
            ((6, 2), "birch_sapling"),
            ((6, 3), "jungle_sapling"),
            ((6, 4), "acacia_sapling"),
            ((6, 5), "dark_oak_sapling"),
            ((40, 0), "mushroom_red"),
            ((39, 0), "mushroom_brown"),
            ((32, 0), "dead_bush"),
            ((31, 2), "fern"),
            ((81, 0), "cactus"),
        ];
        for &((item_id, data), expected) in &variants {
            assert_eq!(FlowerPotState { item_id, data }.contents(), expected);
        }

        // Negative controls: tallgrass meta 1 is not one of the two accepted potted variants,
        // and an arbitrary ItemBlock must never invent a model. The two clamping controls make
        // sure malformed sapling/red-flower payloads retain Java's index-zero behaviour.
        assert_eq!(
            FlowerPotState {
                item_id: 31,
                data: 1,
            }
            .contents(),
            "empty"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 99,
                data: 0,
            }
            .contents(),
            "empty"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 6,
                data: 15,
            }
            .contents(),
            "oak_sapling"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 38,
                data: 15,
            }
            .contents(),
            "rose"
        );
    }

    #[test]
    fn a_block_action_dispatches_on_the_packets_block_not_the_worlds() {
        // java: `handleBlockAction` hands `packetIn.getBlockType()` to `World.addBlockEvent`,
        // which is `blockIn.onBlockEventReceived(this, pos, this.getBlockState(pos), ..)` — no
        // equality test. The `getBlockState(pos).getBlock() == event.getBlock()` check lives in
        // `WorldServer.fireBlockEvent`, the queue drain a client never runs.
        //
        // This is the negative control for the old behaviour: mc-rs required the two to agree
        // and returned `None`, so this assertion is exactly what used to fail.
        let pos = BlockPos::new(8, 64, 8);
        let mut world = world_with(&[(pos, state(TRAPPED_CHEST_ID, 2))]);
        let effect = world.apply_block_action(pos, i32::from(CHEST_ID), 1, 2);
        assert_eq!(
            effect,
            Some(BlockActionEffect::ChestUsers { players: 2 }),
            "the packet said chest; the world holds a trapped chest; vanilla still runs it"
        );
        // ... and the tile entity it created came from the *world's* block, because
        // `BlockContainer.onBlockEventReceived` calls `worldIn.getTileEntity(pos)`.
        assert_eq!(
            world.tile_entity(pos).map(|t| t.kind),
            Some(TileKind::TrappedChest)
        );
    }

    #[test]
    fn the_particle_list_and_the_animation_gate_never_disagree() {
        // java: `MobSpawnerBaseLogic.updateSpawner:88-94` — the two `spawnParticle` calls and
        // the `spawnDelay--`/`mobRotation` update are statements in **one** `isActivated()`
        // branch. mc-client draws the particles and mc-world runs the animation, so the two
        // halves are separated here in a way vanilla never separates them; this pins that they
        // still answer the same question.
        //
        // The failure this rules out is a spawner that smokes while frozen (or the reverse),
        // which is exactly what a second copy of the range test would eventually produce.
        let a = BlockPos::new(0, 64, 0);
        let b = BlockPos::new(40, 64, 0);
        let mut world = world_with(&[(a, state(52, 0)), (b, state(52, 0))]);
        world.tile_entity_or_create(a);
        world.tile_entity_or_create(b);

        let player = Some((4.5, 64.5, 0.5));
        let before: Vec<(BlockPos, i32, f64)> = [a, b]
            .iter()
            .map(|p| {
                let s = world.tile_entity(*p).and_then(TileEntity::spawner).unwrap();
                (*p, s.spawn_delay, s.mob_rotation)
            })
            .collect();
        world.tick_tile_entities(player);
        let listed = world.activated_spawners(player);

        for (pos, delay, rot) in before {
            let now = world
                .tile_entity(pos)
                .and_then(TileEntity::spawner)
                .unwrap();
            let animated = now.spawn_delay != delay || now.mob_rotation != rot;
            assert_eq!(
                animated,
                listed.contains(&pos),
                "{pos:?}: animated={animated} but activated_spawners says {}",
                listed.contains(&pos)
            );
        }
        assert_eq!(listed, vec![a], "only the near spawner is activated");

        // Negative control on the list itself: with no player nothing is activated, so a test
        // that merely counted entries could not pass by accident.
        assert!(world.activated_spawners(None).is_empty());
    }

    #[test]
    fn a_spawner_freezes_outside_its_activation_radius() {
        // java: the whole of `updateSpawner` is inside `if (this.isActivated())`
        // (`MobSpawnerBaseLogic.java:84`), which is a squared 16-block test against the player.
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_with(&[(pos, state(52, 0))]);
        world.tile_entity_or_create(pos);

        // 20 blocks away: nothing moves, not even the countdown.
        world.tick_tile_entities(Some((20.5, 64.5, 0.5)));
        let far = world
            .tile_entity(pos)
            .and_then(TileEntity::spawner)
            .unwrap();
        assert_eq!(far.spawn_delay, 20, "the delay does not tick down");
        assert_eq!(far.mob_rotation, 0.0);

        // 4 blocks away: it runs.
        world.tick_tile_entities(Some((4.5, 64.5, 0.5)));
        let near = world
            .tile_entity(pos)
            .and_then(TileEntity::spawner)
            .unwrap();
        assert_eq!(near.spawn_delay, 19);
        assert!(near.mob_rotation > 0.0);

        // and the boundary is `d0 < range * range`, strictly — 16 blocks out is *not* activated.
        let mut edge = world_with(&[(pos, state(52, 0))]);
        edge.tile_entity_or_create(pos);
        edge.tick_tile_entities(Some((16.5, 64.5, 0.5)));
        assert_eq!(
            edge.tile_entity(pos)
                .and_then(TileEntity::spawner)
                .unwrap()
                .spawn_delay,
            20
        );
    }

    #[test]
    fn an_undescribed_spawner_is_a_pig() {
        // java: `MobSpawnerBaseLogic.mobID = "Pig"` (`:24`). `Chunk.getTileEntity` hands the
        // TESR a default-constructed instance, so a spawner nobody sent an `S35` for still
        // renders something.
        assert_eq!(SpawnerState::default().entity_id, "Pig");
        assert_eq!(SpawnerState::default().spawn_delay, 20);
        assert_eq!(SpawnerState::default().min_spawn_delay, 200);
        assert_eq!(SpawnerState::default().activating_range, 16);
        let te = TileEntity::new(TileKind::MobSpawner);
        assert_eq!(te.spawner().unwrap().entity_id, "Pig");
    }

    #[test]
    fn the_spawners_block_event_snaps_the_delay_to_min() {
        // java: `TileEntityMobSpawner.receiveClientEvent` -> `setDelayToMin(id)` (`:358-369`),
        // sent by the server's `resetTimer` as `func_98267_a(1)` every time it fires.
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_with(&[(pos, state(52, 0))]);
        {
            let te = world.tile_entity_or_create(pos).unwrap();
            if let TileData::Spawner(s) = &mut te.data {
                s.spawn_delay = 3;
                s.min_spawn_delay = 137;
            }
        }
        // id 0 does nothing; only 1 does.
        world.apply_block_action(pos, 52, 0, 0);
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::spawner)
                .unwrap()
                .spawn_delay,
            3
        );
        world.apply_block_action(pos, 52, 1, 0);
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::spawner)
                .unwrap()
                .spawn_delay,
            137
        );
    }

    #[test]
    fn a_repeat_description_packet_does_not_snap_the_miniature_back() {
        // java: `MobSpawnerBaseLogic.readFromNBT` never touches `mobRotation`, so any
        // `markBlockForUpdate` on a spawner must leave the animation where it is.
        let pos = BlockPos::new(0, 64, 0);
        let mut world = world_with(&[(pos, state(52, 0))]);
        world.tile_entity_or_create(pos);
        for _ in 0..5 {
            world.tick_tile_entities(Some((0.5, 64.5, 0.5)));
        }
        let spun = world
            .tile_entity(pos)
            .and_then(TileEntity::spawner)
            .unwrap()
            .mob_rotation;
        let prev_spun = world
            .tile_entity(pos)
            .and_then(TileEntity::spawner)
            .unwrap()
            .prev_mob_rotation;
        assert!(spun > 0.0);
        assert!(prev_spun > 0.0);

        assert!(world.apply_tile_entity_nbt(
            pos,
            1,
            TileData::Spawner(SpawnerState {
                entity_id: "Zombie".to_owned(),
                spawn_delay: 40,
                ..SpawnerState::default()
            })
        ));
        let after = world
            .tile_entity(pos)
            .and_then(TileEntity::spawner)
            .unwrap();
        assert_eq!(after.entity_id, "Zombie", "the packet's fields do land");
        assert_eq!(after.spawn_delay, 40);
        assert_eq!(after.mob_rotation, spun, "but the angle is preserved");
        assert_eq!(
            after.prev_mob_rotation, prev_spun,
            "the interpolation baseline is preserved too"
        );
    }

    #[test]
    fn the_beacons_block_event_rescans_the_beam_at_once() {
        // java: `TileEntityBeacon.receiveClientEvent(1, _)` -> `updateBeacon()` ->
        // `updateSegmentColors()`. Without it the beam only catches up on the 80-tick sweep.
        let pos = BlockPos::new(2, 60, 2);
        let untouched = BlockPos::new(8, 60, 2);
        let mut blocks = vec![(pos, state(138, 0))];
        blocks.push((untouched, state(138, 0)));
        for x in (pos.x - 1)..=(pos.x + 1) {
            for z in (pos.z - 1)..=(pos.z + 1) {
                blocks.push((BlockPos::new(x, pos.y - 1, z), state(42, 0)));
            }
        }
        let mut world = world_with(&blocks);
        world.tile_entity_or_create(pos);
        {
            // This second beacon has deliberately stale render state.  The packet must refresh
            // *only* its position, not behave like the ordinary all-beacon 80-tick sweep.
            let other = world.tile_entity_or_create(untouched).unwrap();
            if let TileData::Beacon(beacon) = &mut other.data {
                beacon.levels = 4;
                beacon.complete = true;
                beacon.beam_fade = 0.75;
                beacon.segments = vec![BeamSegment {
                    colors: [0.25, 0.5, 0.75],
                    height: 99,
                }];
            }
        }
        assert_eq!(
            world
                .tile_entity(pos)
                .and_then(TileEntity::beacon)
                .unwrap()
                .segments
                .len(),
            0,
            "nothing scanned yet"
        );
        // The event selector is not a generic redraw: `TileEntityBeacon` handles precisely 1.
        world.apply_block_action(pos, 138, 0, 255);
        assert!(world
            .tile_entity(pos)
            .and_then(TileEntity::beacon)
            .unwrap()
            .segments
            .is_empty());
        world.apply_block_action(pos, 138, 1, 0);
        let beacon = world.tile_entity(pos).and_then(TileEntity::beacon).unwrap();
        assert_eq!(beacon.levels, 1);
        assert!(beacon.complete);
        assert_eq!(beacon.segments.len(), 1);
        assert_eq!(
            world
                .tile_entity(untouched)
                .and_then(TileEntity::beacon)
                .unwrap(),
            &BeaconState {
                levels: 4,
                complete: true,
                beam_fade: 0.75,
                beam_render_counter: 0,
                segments: vec![BeamSegment {
                    colors: [0.25, 0.5, 0.75],
                    height: 99,
                }],
            },
            "a beacon block action is not the global beacon refresh"
        );
    }

    #[test]
    fn an_out_of_range_flower_pot_meta_clamps_to_a_plant() {
        // java: `BlockPlanks.EnumType.byMetadata` and `BlockFlower.EnumFlowerType.getType` both
        // clamp an out-of-range meta to index 0, which makes the `default: EMPTY` arms of
        // `BlockFlowerPot.getActualState`'s two switches unreachable.
        assert_eq!(
            FlowerPotState {
                item_id: 6,
                data: 9
            }
            .contents(),
            "oak_sapling"
        );
        assert_eq!(
            FlowerPotState {
                item_id: 38,
                data: 12
            }
            .contents(),
            "rose"
        );
        // tallgrass really does fall through to empty: its switch has no clamping helper.
        assert_eq!(
            FlowerPotState {
                item_id: 31,
                data: 1
            }
            .contents(),
            "empty"
        );
    }

    // ---- pistons -----------------------------------------------------------------------
    //
    // Every one of these is a differential against `BlockPistonBase.onBlockEventReceived` +
    // `doMove` + `TileEntityPiston.update` read out of the 1.8.9 source, not against mc-rs's
    // own previous behaviour — before this change `apply_block_action` mutated nothing at all, so
    // every assertion below fails on the old code by construction.

    /// A piston base pointing `facing`, not yet extended. java: meta is
    /// `facing.getIndex() | (extended ? 8 : 0)`.
    fn piston(facing: Facing) -> BlockState {
        BlockState::new(BlockId(PISTON_ID), facing.index())
    }

    #[test]
    fn a_piston_extension_block_has_no_lazily_created_tile_entity() {
        // java: `BlockPistonMoving.createNewTileEntity` returns **null**, which is why
        // `getCollisionBoundingBox` has a no-tile-entity branch at all.
        let pos = BlockPos::new(3, 64, 3);
        let mut world = world_with(&[(pos, state(PISTON_EXTENSION_ID, 0))]);
        assert_eq!(world.tile_kind_at(pos), Some(TileKind::Piston));
        assert!(
            world.tile_entity_or_create(pos).is_none(),
            "36 is an ITileEntityProvider whose createNewTileEntity is null"
        );
        assert_eq!(world.tile_entity_count(), 0);
        // ... but an explicit `setTileEntity` still lands, exactly as `doMove` needs it to.
        world.set_tile_entity(
            pos,
            TileEntity::new_piston(state(1, 0), Facing::East, true, false),
        );
        assert_eq!(world.tile_entity_count(), 1);
    }

    #[test]
    fn piston_action_keeps_packet_handler_variant_over_stale_world_base() {
        // `World.addBlockEvent` dispatches the packet's BlockPistonBase instance while passing
        // the world's state separately.  If a sticky-piston S24 arrives while the stale chunk
        // still says ordinary piston, Java's `this.isSticky` and `this.getStateFromMeta` still
        // make the retract sticky and carry a sticky piston state.
        let base = BlockPos::new(2, 64, 2);
        let head = base.east();
        let pulled = head.east();
        let mut world = world_with(&[
            (base, state(PISTON_ID, 5 | 8)),
            (head, state(PISTON_EXTENSION_ID, 5)),
            (pulled, state(1, 0)),
        ]);
        world.set_tile_entity(
            head,
            TileEntity::new_piston(state(PISTON_HEAD_ID, 5), Facing::East, true, false),
        );

        assert_eq!(
            world.apply_block_action(base, i32::from(STICKY_PISTON_ID), 1, 5),
            Some(BlockActionEffect::Piston {
                extending: false,
                facing: 5,
            })
        );
        assert_eq!(
            world
                .tile_entity(base)
                .and_then(TileEntity::piston)
                .map(|p| p.piston_state.block().0),
            Some(STICKY_PISTON_ID),
            "the moving base carrier follows packet `this`, not stale world state"
        );
        assert_eq!(world.block_at(head).block().0, PISTON_EXTENSION_ID);
        assert_eq!(world.block_at(pulled), BlockState::AIR);
    }

    #[test]
    fn malformed_piston_retract_facing_is_a_safe_noop_without_partial_clear() {
        let base = BlockPos::new(2, 64, 2);
        let head = base.east();
        let mut world = world_with(&[
            (base, state(PISTON_ID, 5 | 8)),
            (head, state(PISTON_EXTENSION_ID, 5)),
        ]);
        world.set_tile_entity(
            head,
            TileEntity::new_piston(state(PISTON_HEAD_ID, 5), Facing::East, true, false),
        );

        assert_eq!(
            world.apply_block_action(base, i32::from(PISTON_ID), 1, 6),
            None,
            "metadata 6 maps to Java's null facing and must not panic"
        );
        assert_eq!(world.block_at(base), state(PISTON_ID, 5 | 8));
        assert_eq!(world.block_at(head).block().0, PISTON_EXTENSION_ID);
        assert!(
            world
                .tile_entity(head)
                .and_then(TileEntity::piston)
                .is_some(),
            "invalid facing must not clear the moving head before returning"
        );
    }

    #[test]
    fn extending_moves_the_pushed_block_and_plants_the_head_locally() {
        // java: `onBlockEventReceived` event 0 -> `doMove(.., true)`. The destination writes use
        // flag 4, so the server never sends them and the client must produce them itself.
        let base = BlockPos::new(2, 64, 2);
        let push = base.east();
        let dest = push.east();
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            (push, state(1, 0)),
            (dest, state(0, 0)),
        ]);

        let effect = world.apply_block_action(base, i32::from(PISTON_ID), 0, 5);
        assert_eq!(
            effect,
            Some(BlockActionEffect::Piston {
                extending: true,
                facing: 5
            })
        );

        // java: `setBlockState(pos, state.withProperty(EXTENDED, true), 2)`
        assert_eq!(world.block_at(base), state(PISTON_ID, 5 | 8));

        // the stone is now a moving block one further out
        assert_eq!(world.block_at(dest).block().0, PISTON_EXTENSION_ID);
        let carried = world
            .tile_entity(dest)
            .and_then(TileEntity::piston)
            .unwrap();
        assert_eq!(carried.piston_state, state(1, 0), "the stone is carried");
        assert!(carried.extending);
        assert_eq!(carried.facing, Facing::East);
        assert_eq!(carried.progress, 0.0);

        // and the head is a moving block at the old stone position
        assert_eq!(world.block_at(push).block().0, PISTON_EXTENSION_ID);
        let head = world
            .tile_entity(push)
            .and_then(TileEntity::piston)
            .unwrap();
        assert_eq!(head.piston_state, state(PISTON_HEAD_ID, 5));
        assert!(head.extending);
    }

    #[test]
    fn the_extended_flag_is_written_only_after_a_successful_move() {
        // java: `if (!this.doMove(...)) return false;` precedes the EXTENDED write, so a piston
        // that cannot push must be left completely untouched.
        let base = BlockPos::new(2, 64, 2);
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            // obsidian is `canPush`'s one identity rejection.
            (base.east(), state(OBSIDIAN_ID, 0)),
        ]);
        assert_eq!(
            world.apply_block_action(base, i32::from(PISTON_ID), 0, 5),
            None
        );
        assert_eq!(
            world.block_at(base),
            piston(Facing::East),
            "no EXTENDED bit"
        );
        assert_eq!(world.block_at(base.east()).block().0, OBSIDIAN_ID);
        assert_eq!(world.tile_entity_count(), 0);
    }

    #[test]
    fn the_extend_sets_the_extended_bit() {
        let base = BlockPos::new(2, 64, 2);
        let mut world = world_with(&[(base, piston(Facing::East)), (base.east(), state(0, 0))]);
        assert!(world
            .apply_block_action(base, i32::from(PISTON_ID), 0, 5)
            .is_some());
        // java: `BlockPistonBase.getMetaFromState` -> `5 | 8`
        assert_eq!(world.block_at(base).meta(), 5 | 8);
    }

    #[test]
    fn the_moving_block_slides_in_two_ticks_and_then_becomes_real() {
        // java: `TileEntityPiston.update` — `progress += 0.5F` twice, and on the tick after
        // `lastProgress >= 1.0F` the tile entity is removed and `pistonState` written back.
        let base = BlockPos::new(2, 64, 2);
        let push = base.east();
        let dest = push.east();
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            (push, state(1, 0)),
            (dest, state(0, 0)),
        ]);
        world.apply_block_action(base, i32::from(PISTON_ID), 0, 5);

        world.tick_pistons();
        assert_eq!(
            world
                .tile_entity(dest)
                .and_then(TileEntity::piston)
                .unwrap()
                .progress,
            0.5
        );
        world.tick_pistons();
        let p = *world
            .tile_entity(dest)
            .and_then(TileEntity::piston)
            .unwrap();
        assert_eq!(p.progress, 1.0);
        assert_eq!(p.last_progress, 0.5);
        // still a moving block: vanilla needs one more tick to see `lastProgress >= 1.0F`.
        assert_eq!(world.block_at(dest).block().0, PISTON_EXTENSION_ID);

        world.tick_pistons();
        assert_eq!(world.block_at(dest), state(1, 0), "the stone landed");
        assert_eq!(
            world.block_at(push),
            state(PISTON_HEAD_ID, 5),
            "the head landed"
        );
        assert_eq!(world.tile_entity_count(), 0, "both tile entities cleared");
    }

    #[test]
    fn a_sticky_piston_retract_drags_the_block_back() {
        // java: event 1 -> the base becomes `piston_extension` carrying the piston itself, and
        // `isSticky` re-runs `doMove(.., false)` from two blocks out.
        let base = BlockPos::new(2, 64, 2);
        let head = base.east();
        let dragged = head.east();
        let mut world = world_with(&[
            // an *extended* sticky piston: meta 5 | 8
            (base, state(STICKY_PISTON_ID, 5 | 8)),
            (head, state(PISTON_HEAD_ID, 5 | 8)),
            (dragged, state(1, 0)),
        ]);
        let effect = world.apply_block_action(base, i32::from(STICKY_PISTON_ID), 1, 5);
        assert_eq!(
            effect,
            Some(BlockActionEffect::Piston {
                extending: false,
                facing: 5
            })
        );

        // the base carries the piston back to itself
        assert_eq!(world.block_at(base).block().0, PISTON_EXTENSION_ID);
        let at_base = *world
            .tile_entity(base)
            .and_then(TileEntity::piston)
            .unwrap();
        assert!(!at_base.extending);
        assert!(at_base.should_head_be_rendered, "java: renderHead = true");
        assert_eq!(
            at_base.piston_state,
            state(STICKY_PISTON_ID, 5),
            "getStateFromMeta(eventParam): EXTENDED cleared"
        );

        // and the dragged block is now a moving block where the head used to be
        assert_eq!(world.block_at(head).block().0, PISTON_EXTENSION_ID);
        let pulled = *world
            .tile_entity(head)
            .and_then(TileEntity::piston)
            .unwrap();
        assert_eq!(pulled.piston_state, state(1, 0));
        assert!(!pulled.extending);
        assert_eq!(world.block_at(dragged), BlockState::AIR);

        for _ in 0..3 {
            world.tick_pistons();
        }
        assert_eq!(world.block_at(base), state(STICKY_PISTON_ID, 5));
        assert_eq!(world.block_at(head), state(1, 0), "dragged one block back");
        assert_eq!(world.block_at(dragged), BlockState::AIR);
    }

    #[test]
    fn a_plain_piston_retract_leaves_nothing_behind() {
        // java: the `else` arm — `worldIn.setBlockToAir(pos.offset(enumfacing))`.
        let base = BlockPos::new(2, 64, 2);
        let head = base.east();
        let neighbour = head.east();
        let mut world = world_with(&[
            (base, state(PISTON_ID, 5 | 8)),
            (head, state(PISTON_HEAD_ID, 5)),
            (neighbour, state(1, 0)),
        ]);
        world.apply_block_action(base, i32::from(PISTON_ID), 1, 5);
        assert_eq!(world.block_at(head), BlockState::AIR);
        assert_eq!(world.block_at(neighbour), state(1, 0), "not dragged");
        for _ in 0..3 {
            world.tick_pistons();
        }
        assert_eq!(world.block_at(base), state(PISTON_ID, 5));
    }

    #[test]
    fn can_push_reproduces_all_four_vanilla_rejections() {
        // java: `BlockPistonBase.canPush` — obsidian by identity, hardness -1, mobility 2, and
        // `instanceof ITileEntityProvider`.
        let p = BlockPos::new(5, 64, 5);
        for (id, meta, why) in [
            (OBSIDIAN_ID, 0u8, "obsidian"),
            (7, 0, "bedrock: hardness -1"),
            (145, 0, "anvil: mobility 2"),
            (CHEST_ID, 2, "chest: ITileEntityProvider"),
            (
                149,
                0,
                "comparator: ITileEntityProvider without isBlockContainer",
            ),
        ] {
            let world = world_with(&[(p, state(id, meta))]);
            assert!(
                !piston_can_push(&world, p, Facing::East, false),
                "block {id} should be unpushable ({why})"
            );
        }
        // and one that is pushable
        let world = world_with(&[(p, state(1, 0))]);
        assert!(piston_can_push(&world, p, Facing::East, false));
    }

    #[test]
    fn piston_world_border_uses_java_inside_and_strict_edge_controls() {
        // Java's 4-wide border has maxX/maxZ == 2. A block at 1 occupies [1, 2) and is
        // inside; a block at 2 starts exactly on the strict maximum edge and is outside.
        let border = PistonWorldBorder::new(0.0, 0.0, 4.0, 29_999_984);
        assert!(border.contains(BlockPos::new(1, 64, 1)));
        assert!(!border.contains(BlockPos::new(2, 64, 1)));
        assert!(!border.contains(BlockPos::new(1, 64, 2)));

        let inside = BlockPos::new(1, 64, 1);
        let outside = BlockPos::new(2, 64, 1);
        let world = world_with(&[(inside, state(1, 0)), (outside, state(1, 0))]);
        assert!(piston_can_push_with_border(
            &world,
            inside,
            Facing::East,
            false,
            Some(border)
        ));
        assert!(!piston_can_push_with_border(
            &world,
            outside,
            Facing::East,
            false,
            Some(border)
        ));
    }

    #[test]
    fn piston_move_rejects_an_outside_chain_but_disabled_border_keeps_vanilla_controls() {
        let base = BlockPos::new(0, 64, 0);
        let first = base.east();
        let second = first.east();
        let third = second.east();
        let border = PistonWorldBorder::new(0.0, 0.0, 4.0, 29_999_984);

        // The first carried block is inside, but the next occupied destination is exactly on the
        // border's outside edge. `BlockPistonStructureHelper` reaches that state through the
        // same canPush call Java uses, so the entire event is refused without partial writes.
        let blocks = [
            (base, piston(Facing::East)),
            (first, state(1, 0)),
            (second, state(1, 0)),
        ];
        let mut bounded = world_with(&blocks);
        assert!(bounded
            .apply_block_action_with_border(base, i32::from(PISTON_ID), 0, 5, Some(border))
            .is_none());
        assert_eq!(bounded.block_at(base), piston(Facing::East));
        assert_eq!(bounded.block_at(first), state(1, 0));
        assert_eq!(bounded.block_at(second), state(1, 0));

        // `None` is the explicit border-disabled compatibility path. The same ordinary piston
        // move succeeds and carries both blocks into the empty third cell.
        let mut unbounded = world_with(&blocks);
        assert!(unbounded
            .apply_block_action_with_border(base, i32::from(PISTON_ID), 0, 5, None)
            .is_some());
        assert_eq!(unbounded.block_at(third).block().0, PISTON_EXTENSION_ID);
        assert_eq!(unbounded.block_at(second).block().0, PISTON_EXTENSION_ID);
    }

    #[test]
    fn piston_border_gate_does_not_replace_height_and_build_limit_controls() {
        let border = PistonWorldBorder::new(0.0, 0.0, 4.0, 29_999_984);
        let below = BlockPos::new(0, -1, 0);
        let bottom = BlockPos::new(0, 0, 0);
        let top = BlockPos::new(0, 255, 0);
        let above = BlockPos::new(0, 256, 0);
        let world = world_with(&[
            (below, state(1, 0)),
            (bottom, state(1, 0)),
            (top, state(1, 0)),
            (above, state(1, 0)),
        ]);
        assert!(!piston_can_push_with_border(
            &world,
            below,
            Facing::East,
            false,
            Some(border)
        ));
        assert!(!piston_can_push_with_border(
            &world,
            bottom,
            Facing::Down,
            false,
            Some(border)
        ));
        assert!(!piston_can_push_with_border(
            &world,
            top,
            Facing::Up,
            false,
            Some(border)
        ));
        assert!(!piston_can_push_with_border(
            &world,
            above,
            Facing::East,
            false,
            Some(border)
        ));
    }

    #[test]
    fn a_mobility_one_block_is_destroyed_rather_than_pushed() {
        // java: `canMove()`'s first branch — `getMobilityFlag() == 1` goes on `toDestroy`.
        let base = BlockPos::new(2, 64, 2);
        let torch = base.east();
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            // block 50 is `minecraft:torch`, `Material.circuits`, mobility 1.
            (torch, state(50, 5)),
        ]);
        assert!(world
            .apply_block_action(base, i32::from(PISTON_ID), 0, 5)
            .is_some());
        // the torch is gone and the head took its place, with nothing carried past it.
        assert_eq!(world.block_at(torch).block().0, PISTON_EXTENSION_ID);
        let head = world
            .tile_entity(torch)
            .and_then(TileEntity::piston)
            .unwrap();
        assert_eq!(head.piston_state, state(PISTON_HEAD_ID, 5));
        assert_eq!(world.block_at(torch.east()), BlockState::AIR);
    }

    #[test]
    fn thirteen_blocks_is_one_too_many() {
        // java: `BlockPistonStructureHelper`'s `> 12` cap, checked in two places.
        let base = BlockPos::new(0, 64, 0);
        let mut blocks = vec![(base, piston(Facing::East))];
        for i in 1..=13 {
            blocks.push((base.add(i, 0, 0), state(1, 0)));
        }
        let mut world = world_with(&blocks);
        assert_eq!(
            world.apply_block_action(base, i32::from(PISTON_ID), 0, 5),
            None,
            "13 blocks cannot be pushed"
        );

        let mut blocks = vec![(base, piston(Facing::East))];
        for i in 1..=12 {
            blocks.push((base.add(i, 0, 0), state(1, 0)));
        }
        blocks.push((base.add(13, 0, 0), state(0, 0)));
        let mut world12 = world_with(&blocks);
        assert!(
            world12
                .apply_block_action(base, i32::from(PISTON_ID), 0, 5)
                .is_some(),
            "12 is the limit, not 11"
        );
        assert_eq!(
            world12.block_at(base.add(13, 0, 0)).block().0,
            PISTON_EXTENSION_ID
        );
    }

    #[test]
    fn the_moving_block_has_a_collision_box_the_whole_way_across() {
        // java: `BlockPistonMoving.getCollisionBoundingBox` — mc-data's extracted
        // `collision: Empty` for block 36 is only the *no*-tile-entity case, so before this the
        // player fell through a moving block for the two ticks it existed.
        let base = BlockPos::new(2, 64, 2);
        let push = base.east();
        let dest = push.east();
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            (push, state(1, 0)),
            (dest, state(0, 0)),
        ]);
        world.apply_block_action(base, i32::from(PISTON_ID), 0, 5);

        // progress 0 while extending -> the leading edge has not entered the cell yet, so the
        // box is degenerate; vanilla is identical (`1.0F - 0.0F` shrinks maxX by a full block).
        let at_zero = world.piston_collision_box(dest).expect("a box exists");
        assert!((at_zero.max.x - at_zero.min.x).abs() < 1e-9, "{at_zero:?}");

        world.tick_pistons();
        let half = world.piston_collision_box(dest).expect("a box exists");
        // getProgress(0.0) is *lastProgress*, still 0 after the first tick.
        assert!((half.max.x - half.min.x).abs() < 1e-9, "{half:?}");

        world.tick_pistons();
        let most = world.piston_collision_box(dest).expect("a box exists");
        assert!(
            (most.max.x - most.min.x - 0.5).abs() < 1e-6,
            "half a block of floor: {most:?}"
        );
        assert_eq!(most.min.y, 64.0);
        assert_eq!(most.max.y, 65.0);
    }

    #[test]
    fn moving_piston_uses_one_carried_bounds_box_for_multi_box_states() {
        // BlockPistonMoving.getBoundingBox calls the carried block's singular
        // getCollisionBoundingBox. It must not return one union synthesized from the two
        // stair collision boxes. The oak-stairs state is a useful multi-box control even though
        // its isolated union happens to cover the unit cell.
        let carried = state(53, 0); // oak_stairs, outer-corner collision has two boxes
        assert!(carried.props().collision.boxes().len() > 1);

        let half = piston_bounding_box(BlockPos::ORIGIN, carried, 0.5, Facing::East)
            .expect("stairs have a carried collision box");
        assert_eq!(half.min, DVec3::new(0.0, 0.0, 0.0));
        assert_eq!(half.max, DVec3::new(0.5, 1.0, 1.0));
    }

    #[test]
    fn moving_piston_uses_fence_mutable_bounds_not_tall_collision_arm() {
        // BlockFence.addCollisionBoxesToList emits a 1.5-high movement arm, but the carried
        // Block.getCollisionBoundingBox is one mutable AABB. The extracted bounds snapshot is
        // the one-block-high post used by that singleton path.
        let carried = state(85, 0); // fence
        let half = piston_bounding_box(BlockPos::ORIGIN, carried, 0.5, Facing::East)
            .expect("fences have a carried collision box");
        assert_eq!(half.min, DVec3::new(0.125, 0.0, 0.375));
        assert_eq!(half.max, DVec3::new(0.375, 1.0, 0.625));
    }

    #[test]
    fn moving_wall_uses_its_collision_height_and_live_carrier_neighbours() {
        // BlockWall.getCollisionBoundingBox recomputes connections at the moving extension
        // position and then forces maxY to 1.5.  It is not the one-block selection bounds row.
        let pos = BlockPos::new(-7, 64, -11);
        let mut world = world_with(&[
            (pos, state(PISTON_EXTENSION_ID, 5)),
            (pos.east(), state(1, 0)),
        ]);
        world.set_tile_entity(
            pos,
            TileEntity::new_piston(state(139, 0), Facing::East, false, false),
        );

        let connected = world.piston_collision_box(pos).expect("wall carrier box");
        assert_eq!(connected.min.x, -6.75);
        assert_eq!(connected.max.x, -6.0);
        assert_eq!(connected.max.y, 65.5);

        world.set_block(pos.east(), BlockState::AIR);
        let isolated = world.piston_collision_box(pos).expect("wall carrier box");
        assert_eq!(isolated.min.x, -6.75);
        assert_eq!(isolated.max.x, -6.25);
        assert_eq!(isolated.max.y, 65.5);
    }

    #[test]
    fn moving_cactus_keeps_java_float_world_rounding_at_negative_coordinates() {
        // BlockCactus constructs x/z and its upper y face through float world coordinates before
        // widening to double.  Int-plus-local f64 arithmetic diverges around the 2^21 binade.
        let pos = BlockPos::new(-2_097_153, 2_097_152, 2_097_152);
        let got = piston_bounding_box(pos, state(81, 0), 0.0, Facing::East)
            .expect("cactus has a moving collision box");
        assert_eq!(got.min.x, f64::from((pos.x as f32) + 0.0625_f32));
        assert_eq!(got.max.x, f64::from(((pos.x + 1) as f32) - 0.0625_f32));
        assert_eq!(got.min.z, f64::from((pos.z as f32) + 0.0625_f32));
        assert_eq!(got.max.y, f64::from(((pos.y + 1) as f32) - 0.0625_f32));
        assert_ne!(
            got.min.x,
            f64::from(pos.x) + 0.0625,
            "the old int-plus-local path loses Java float rounding"
        );
    }

    #[test]
    fn moving_direct_float_collision_overrides_keep_their_java_coordinate_rules() {
        let pos = BlockPos::new(-2_097_153, 64, 2_097_152);

        // Cake: BITES controls the local x face, while x/z and the upper y face are widened
        // from Java float world coordinates.
        let cake = piston_bounding_box(pos, state(92, 2), 0.0, Facing::East)
            .expect("cake has a moving collision box");
        assert_eq!(cake.min.x, f64::from((pos.x as f32) + (5.0_f32 / 16.0_f32)));
        assert_eq!(cake.max.y, f64::from((pos.y as f32) + 0.5_f32));

        // A closed north/south gate is the Z-thin direct override; its y face is 1.5 blocks.
        let gate = piston_bounding_box(pos, state(107, 0), 0.0, Facing::East)
            .expect("closed gate has a moving collision box");
        assert_eq!(gate.min.z, f64::from((pos.z as f32) + 0.375_f32));
        assert_eq!(gate.max.y, f64::from((pos.y as f32) + 1.5_f32));

        // Soul sand uses integer x/z anchors but forms maxY as `(float)(y + 1) - .125F`.
        let soul = piston_bounding_box(pos, state(88, 0), 0.0, Facing::East)
            .expect("soul sand has a moving collision box");
        assert_eq!(soul.max.y, f64::from(((pos.y + 1) as f32) - 0.125_f32));
    }

    #[test]
    fn moving_piston_shifts_only_the_directional_face_for_all_six_facings() {
        let carried = state(1, 0); // stone: a simple one-box control
        let expected = [
            (
                Facing::Down,
                DVec3::new(0.0, 0.5, 0.0),
                DVec3::new(1.0, 1.0, 1.0),
            ),
            (
                Facing::Up,
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 0.5, 1.0),
            ),
            (
                Facing::North,
                DVec3::new(0.0, 0.0, 0.5),
                DVec3::new(1.0, 1.0, 1.0),
            ),
            (
                Facing::South,
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 0.5),
            ),
            (
                Facing::West,
                DVec3::new(0.5, 0.0, 0.0),
                DVec3::new(1.0, 1.0, 1.0),
            ),
            (
                Facing::East,
                DVec3::new(0.0, 0.0, 0.0),
                DVec3::new(0.5, 1.0, 1.0),
            ),
        ];
        for (facing, min, max) in expected {
            let actual = piston_bounding_box(BlockPos::ORIGIN, carried, 0.5, facing)
                .expect("stone has a carried collision box");
            assert_eq!(actual.min, min, "facing {facing:?}");
            assert_eq!(actual.max, max, "facing {facing:?}");
        }
    }

    #[test]
    fn moving_piston_collision_inverts_extension_but_not_retraction() {
        let pos = BlockPos::new(4, 64, 4);
        let carried = state(1, 0);
        let mut extending = world_with(&[(pos, state(PISTON_EXTENSION_ID, 5))]);
        extending.set_tile_entity(
            pos,
            TileEntity::new_piston(carried, Facing::East, true, false),
        );
        if let Some(TileData::Piston(p)) = extending.tile_entity_mut(pos).map(|te| &mut te.data) {
            p.progress = 0.25;
            p.last_progress = 0.25;
        }
        let ext = extending
            .piston_collision_box(pos)
            .expect("extending carrier collision");
        assert_eq!(ext.min.x, 4.0);
        assert_eq!(ext.max.x, 4.25);

        let mut retracting = world_with(&[(pos, state(PISTON_EXTENSION_ID, 5))]);
        retracting.set_tile_entity(
            pos,
            TileEntity::new_piston(carried, Facing::East, false, false),
        );
        if let Some(TileData::Piston(p)) = retracting.tile_entity_mut(pos).map(|te| &mut te.data) {
            p.progress = 0.25;
            p.last_progress = 0.25;
        }
        let retract = retracting
            .piston_collision_box(pos)
            .expect("retracting carrier collision");
        assert_eq!(retract.min.x, 4.0);
        assert_eq!(retract.max.x, 4.75);
    }

    #[test]
    fn the_shove_box_and_distance_match_launch_with_slime_block() {
        // java: `launchWithSlimeBlock(progress, progress - lastProgress + 0.0625F)` while
        // extending — 0.5 + 0.0625 on each of the two moving ticks, 0.25 on the last.
        let base = BlockPos::new(2, 64, 2);
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            (base.east(), state(1, 0)),
            (base.east().east(), state(0, 0)),
        ]);
        world.apply_block_action(base, i32::from(PISTON_ID), 0, 5);

        let first = world.tick_pistons();
        assert_eq!(first.len(), 2, "the pushed stone and the head both shove");
        for push in &first {
            assert!((push.amount - 0.5625).abs() < 1e-6, "{}", push.amount);
            assert_eq!(push.facing, Facing::East);
            assert!(!push.slime);
        }
        world.tick_pistons();
        let last = world.tick_pistons();
        assert_eq!(last.len(), 2);
        for push in &last {
            assert!((push.amount - 0.25).abs() < 1e-6, "{}", push.amount);
        }
    }

    #[test]
    fn a_slime_block_sets_motion_instead_of_nudging() {
        // java: `pistonState.getBlock() == Blocks.slime_block && this.extending`.
        let base = BlockPos::new(2, 64, 2);
        let mut world = world_with(&[
            (base, piston(Facing::East)),
            (base.east(), state(SLIME_BLOCK_ID, 0)),
            (base.east().east(), state(0, 0)),
        ]);
        world.apply_block_action(base, i32::from(PISTON_ID), 0, 5);
        let pushes = world.tick_pistons();
        assert!(
            pushes.iter().any(|p| p.slime),
            "the slime block's own shove sets motion: {pushes:?}"
        );
        assert!(
            pushes.iter().any(|p| !p.slime),
            "the head's does not: {pushes:?}"
        );
    }

    #[test]
    fn get_progress_and_the_render_offset_follow_vanillas_formulae() {
        // java: `getProgress(ticks)` clamps `ticks > 1.0F`; `getOffsetX` is
        // `(progress - 1) * frontOffset` extending and `(1 - progress) * frontOffset` retracting.
        let p = PistonState {
            piston_state: state(1, 0),
            facing: Facing::East,
            extending: true,
            should_head_be_rendered: false,
            progress: 1.0,
            last_progress: 0.5,
        };
        assert!((p.progress_at(0.0) - 0.5).abs() < 1e-6);
        assert!((p.progress_at(0.5) - 0.75).abs() < 1e-6);
        assert!((p.progress_at(4.0) - 1.0).abs() < 1e-6, "clamped at 1");
        let (x, y, z) = p.offset_at(0.0);
        assert!((x - -0.5).abs() < 1e-6, "{x}");
        assert_eq!((y, z), (0.0, 0.0));

        let q = PistonState {
            extending: false,
            ..p
        };
        let (x, _, _) = q.offset_at(0.0);
        assert!((x - 0.5).abs() < 1e-6, "{x}");
    }

    #[test]
    fn the_banner_pattern_registry_matches_java_and_rejects_unknown_ids() {
        // java: `TileEntityBanner.EnumBannerPattern` — every one of the 39 enum rows is a
        // render-mask basename plus the short id carried in `Patterns[i].Pattern`.
        assert_eq!(BANNER_PATTERN_TABLE.len(), 39);
        for (id, name) in BANNER_PATTERN_TABLE {
            assert_eq!(banner_pattern_name(id), Some(*name), "pattern {id}");
        }
        // Negative control: Java's getPatternByID returns null, so a plugin's arbitrary id must
        // not turn into a guessed texture or survive into a composite design.
        assert_eq!(banner_pattern_name("not_a_vanilla_pattern"), None);
        assert_eq!(banner_pattern_name("base.png"), None);
    }

    #[test]
    fn the_beacon_is_the_only_tile_entity_visible_past_sixty_four_blocks() {
        // java: `TileEntity.getMaxRenderDistanceSquared` is 4096; the beacon overrides to 65536.
        assert_eq!(TileKind::Beacon.max_render_distance_sq(), 65536.0);
        for kind in [TileKind::Chest, TileKind::StandingSign, TileKind::Skull] {
            assert_eq!(kind.max_render_distance_sq(), 4096.0);
        }
    }
    #[test]
    fn skipping_an_unhinted_section_does_not_skip_existing_tile_ticks() {
        let pos = BlockPos::new(1, 64, 1);
        let chunk = pos.chunk_pos();
        let mut world = world_with(&[(pos, state(1, 0))]);
        world
            .tile_entities
            .insert(pos, TileEntity::new(TileKind::Chest));
        if let Some(TileData::Chest(chest)) = world.tile_entity_mut(pos).map(|e| &mut e.data) {
            chest.num_players_using = 1;
        }
        assert!(!world
            .chunk(chunk)
            .unwrap()
            .section(4)
            .unwrap()
            .may_contain_tile_provider());
        assert!(world.scan_chunk_tile_entities(chunk).is_empty());
        world.tick_tile_entities(None);
        assert!(
            (world
                .tile_entity(pos)
                .and_then(TileEntity::chest)
                .unwrap()
                .lid_angle
                - 0.1)
                .abs()
                < 1e-6
        );
    }
}
