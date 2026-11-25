//! Protocol-5 clientbound Play packets `0x00`-`0x40`, one struct per packet (the two chunk
//! packets are in [`super::chunk`]).
//!
//! Each `read` is the 1.7.10 class's `readPacketData` and each `write` its `writePacketData`.
//! Fields keep their wire width. Where `readPacketData` itself maps a value through a table this
//! crate has (a world type name, a game mode or difficulty looked up in an enum, S24's
//! `& 4095`), the struct holds the result, the value the 1.7.10 handler sees, and `write`
//! produces what Java's `writePacketData` would for it. Block, item and stat ids stay numbers:
//! Java turns them into registry objects (`Block.getBlockById` gives air for an unknown id, which
//! S23 and S24 would then write back as 0), and the 1.7.10 registries belong to the code that
//! applies the packet. Vanilla servers only send values these mappings leave unchanged, so their
//! packets re-encode byte for byte.
//!
//! Positions are separate `x`, `y`, `z` fields; packets with a block position offer
//! `block_pos()`. Chat JSON stays unparsed, as in [`crate::play::s2c`]: S02 and S40 are JSON,
//! while S2D, S33, S38, S3B and S3E carry plain text in protocol 5.
//!
//! java17: `network/play/server/S00PacketKeepAlive.java` .. `S40PacketDisconnect.java`

use mc_math::BlockPos;
use mc_nbt::NbtCompound;
use uuid::Uuid;

use super::buffer::PacketBufferV5;
use super::metadata::{self, Metadata5};
use super::nbt::NbtBudget;
use super::tables::canonical_world_type;
use crate::item::ItemStack;
use crate::packet::Packet;
use crate::{Angle, PacketBuffer, ProtoError};

/// java17: `WorldSettings.GameType.getByID`: SURVIVAL (0), CREATIVE (1), ADVENTURE (2); any other
/// id is SURVIVAL. 1.7.10 has no spectator mode.
fn game_type_by_id(id: u8) -> u8 {
    if id <= 2 {
        id
    } else {
        0
    }
}

/// java17: `EnumDifficulty.getDifficultyEnum`, `difficultyEnums[id % 4]`.
fn difficulty_by_id(id: u8) -> u8 {
    id % 4
}

/// `0x00 Keep Alive`: an `int` id the client echoes in C00. java17: `S00PacketKeepAlive`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive {
    /// The id (an `int`, not a VarInt).
    pub id: i32,
}

impl Packet for KeepAlive {
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(KeepAlive {
            id: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.id);
    }
}

/// `0x01 Join Game`. java17: `S01PacketJoinGame`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinGame {
    /// The local player's entity id.
    pub entity_id: i32,
    /// Bit 3 of the game-mode byte.
    pub hardcore_mode: bool,
    /// `GameType.getByID(byte & ~8)`: 0 survival, 1 creative, 2 adventure (anything else is
    /// survival).
    pub game_mode: u8,
    /// A signed byte: -1 nether, 0 overworld, 1 end.
    pub dimension: i8,
    /// `EnumDifficulty.getDifficultyEnum`, 0 peaceful .. 3 hard.
    pub difficulty: u8,
    /// The server's player cap.
    pub max_players: u8,
    /// The canonical world type name ([`super::tables::canonical_world_type`]).
    pub level_type: String,
}

impl Packet for JoinGame {
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        let raw = buf.read_u8()?;
        let dimension = buf.read_i8()?;
        let difficulty = difficulty_by_id(buf.read_u8()?);
        let max_players = buf.read_u8()?;
        let level_type = canonical_world_type(&buf.read_string(16)?).to_owned();
        Ok(JoinGame {
            entity_id,
            hardcore_mode: raw & 8 == 8,
            game_mode: game_type_by_id(raw & !8),
            dimension,
            difficulty,
            max_players,
            level_type,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        let mut raw = self.game_mode;
        if self.hardcore_mode {
            raw |= 8;
        }
        buf.write_u8(raw);
        buf.write_i8(self.dimension);
        buf.write_u8(self.difficulty);
        buf.write_u8(self.max_players);
        buf.write_string(canonical_world_type(&self.level_type));
    }
}

/// `0x02 Chat`: chat JSON only, no position byte. java17: `S02PacketChat`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chat {
    /// The component as JSON (`readStringFromBuffer(32767)`).
    pub component: String,
}

impl Packet for Chat {
    const ID: i32 = 0x02;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Chat {
            component: buf.read_chat()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_chat(&self.component);
    }
}

/// `0x03 Time Update`. java17: `S03PacketTimeUpdate`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeUpdate {
    /// Total world time.
    pub total_world_time: i64,
    /// Time of day; negative means the daylight cycle is frozen.
    pub world_time: i64,
}

impl Packet for TimeUpdate {
    const ID: i32 = 0x03;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(TimeUpdate {
            total_world_time: buf.read_i64()?,
            world_time: buf.read_i64()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i64(self.total_world_time);
        buf.write_i64(self.world_time);
    }
}

/// `0x04 Entity Equipment`. java17: `S04PacketEntityEquipment`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityEquipment {
    /// An `int` entity id.
    pub entity_id: i32,
    /// 0 held item, 1-4 armour (boots .. helmet).
    pub equipment_slot: i16,
    /// The stack, gzip NBT on the wire.
    pub item: Option<ItemStack>,
}

impl Packet for EntityEquipment {
    const ID: i32 = 0x04;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(EntityEquipment {
            entity_id: buf.read_i32()?,
            equipment_slot: buf.read_i16()?,
            item: buf.read_item_v5(&mut budget)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i16(self.equipment_slot);
        buf.write_item_v5(self.item.as_ref());
    }
}

/// `0x05 Spawn Position`: three ints. java17: `S05PacketSpawnPosition`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnPosition {
    /// X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Z.
    pub z: i32,
}

impl SpawnPosition {
    /// The position as a block position.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, self.y, self.z)
    }
}

impl Packet for SpawnPosition {
    const ID: i32 = 0x05;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnPosition {
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
    }
}

/// `0x06 Update Health`. java17: `S06PacketUpdateHealth`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateHealth {
    /// Health.
    pub health: f32,
    /// Food level, a `short` (1.8.9: VarInt).
    pub food_level: i16,
    /// Saturation.
    pub saturation_level: f32,
}

impl Packet for UpdateHealth {
    const ID: i32 = 0x06;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(UpdateHealth {
            health: buf.read_f32()?,
            food_level: buf.read_i16()?,
            saturation_level: buf.read_f32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.health);
        buf.write_i16(self.food_level);
        buf.write_f32(self.saturation_level);
    }
}

/// `0x07 Respawn`. java17: `S07PacketRespawn`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Respawn {
    /// The dimension, an `int`.
    pub dimension: i32,
    /// `EnumDifficulty.getDifficultyEnum`.
    pub difficulty: u8,
    /// `GameType.getByID` (no hardcore bit here).
    pub game_mode: u8,
    /// The canonical world type name.
    pub level_type: String,
}

impl Packet for Respawn {
    const ID: i32 = 0x07;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Respawn {
            dimension: buf.read_i32()?,
            difficulty: difficulty_by_id(buf.read_u8()?),
            game_mode: game_type_by_id(buf.read_u8()?),
            level_type: canonical_world_type(&buf.read_string(16)?).to_owned(),
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.dimension);
        buf.write_u8(self.difficulty);
        buf.write_u8(self.game_mode);
        buf.write_string(canonical_world_type(&self.level_type));
    }
}

/// `0x08 Player Position And Look`: absolute, `y` at eye level, and an `onGround` byte the
/// client echoes in its C06 reply. No relative flags. java17: `S08PacketPlayerPosLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPosLook {
    /// X.
    pub x: f64,
    /// Y of the eyes (the server sends feet + 1.62).
    pub y: f64,
    /// Z.
    pub z: f64,
    /// Yaw.
    pub yaw: f32,
    /// Pitch.
    pub pitch: f32,
    /// Echoed in the C06 reply.
    pub on_ground: bool,
}

impl Packet for PlayerPosLook {
    const ID: i32 = 0x08;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerPosLook {
            x: buf.read_f64()?,
            y: buf.read_f64()?,
            z: buf.read_f64()?,
            yaw: buf.read_f32()?,
            pitch: buf.read_f32()?,
            on_ground: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f64(self.x);
        buf.write_f64(self.y);
        buf.write_f64(self.z);
        buf.write_f32(self.yaw);
        buf.write_f32(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x09 Held Item Change`. java17: `S09PacketHeldItemChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeldItemChange {
    /// Hotbar slot.
    pub slot: i8,
}

impl Packet for HeldItemChange {
    const ID: i32 = 0x09;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(HeldItemChange {
            slot: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.slot);
    }
}

/// `0x0A Use Bed`. java17: `S0APacketUseBed`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UseBed {
    /// An `int` entity id.
    pub player_id: i32,
    /// X.
    pub x: i32,
    /// Y as a **signed** byte (`readByte`): a bed above y 127 arrives negative.
    pub y: i8,
    /// Z.
    pub z: i32,
}

impl UseBed {
    /// The position the handler uses (`y` sign-extended).
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for UseBed {
    const ID: i32 = 0x0A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(UseBed {
            player_id: buf.read_i32()?,
            x: buf.read_i32()?,
            y: buf.read_i8()?,
            z: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.player_id);
        buf.write_i32(self.x);
        buf.write_i8(self.y);
        buf.write_i32(self.z);
    }
}

/// `0x0B Animation`. java17: `S0BPacketAnimation`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Animation {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// The animation (`readUnsignedByte`).
    pub animation_type: u8,
}

impl Packet for Animation {
    const ID: i32 = 0x0B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Animation {
            entity_id: buf.read_varint()?,
            animation_type: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_u8(self.animation_type);
    }
}

/// One `GameProfile` property in [`SpawnPlayer`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileProperty {
    /// Property name, e.g. `textures`.
    pub name: String,
    /// Value (base64 for textures).
    pub value: String,
    /// Signature; always present in protocol 5 (possibly empty).
    pub signature: String,
}

/// `0x0C Spawn Player`: carries the player's identity (dashed UUID string, name, skin
/// properties) itself; protocol 5's S38 has none of it. java17: `S0CPacketSpawnPlayer`
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnPlayer {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// `UUID.fromString(readStringFromBuffer(36))`; written back as `UUID.toString()`.
    pub uuid: Uuid,
    /// `readStringFromBuffer(16)`.
    pub name: String,
    /// The profile's properties in wire order.
    pub properties: Vec<ProfileProperty>,
    /// X * 32.
    pub x: i32,
    /// Y * 32 (feet).
    pub y: i32,
    /// Z * 32.
    pub z: i32,
    /// Yaw byte.
    pub yaw: Angle,
    /// Pitch byte.
    pub pitch: Angle,
    /// The held item's id, a `short`.
    pub current_item: i16,
    /// The watcher list.
    pub metadata: Metadata5,
}

impl Packet for SpawnPlayer {
    const ID: i32 = 0x0C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let uuid_text = buf.read_string(36)?;
        let uuid =
            mc_auth::java_uuid_from_string(&uuid_text).ok_or(ProtoError::InvalidUuid(uuid_text))?;
        let name = buf.read_string(16)?;
        let count = buf.read_varint()?;
        let mut properties = Vec::new();
        for _ in 0..count.max(0) {
            properties.push(ProfileProperty {
                name: buf.read_string(32767)?,
                value: buf.read_string(32767)?,
                signature: buf.read_string(32767)?,
            });
        }
        let x = buf.read_i32()?;
        let y = buf.read_i32()?;
        let z = buf.read_i32()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let current_item = buf.read_i16()?;
        let metadata = metadata::read(buf, &mut NbtBudget::per_packet())?;
        Ok(SpawnPlayer {
            entity_id,
            uuid,
            name,
            properties,
            x,
            y,
            z,
            yaw,
            pitch,
            current_item,
            metadata,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_string(&self.uuid.hyphenated().to_string());
        buf.write_string(&self.name);
        buf.write_varint(self.properties.len() as i32);
        for property in &self.properties {
            buf.write_string(&property.name);
            buf.write_string(&property.value);
            buf.write_string(&property.signature);
        }
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_i16(self.current_item);
        metadata::write(&self.metadata, buf);
    }
}

/// `0x0D Collect Item`. java17: `S0DPacketCollectItem`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectItem {
    /// The item (or arrow, or orb) entity, an `int`.
    pub collected_entity_id: i32,
    /// The collector, an `int`.
    pub collector_entity_id: i32,
}

impl Packet for CollectItem {
    const ID: i32 = 0x0D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(CollectItem {
            collected_entity_id: buf.read_i32()?,
            collector_entity_id: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.collected_entity_id);
        buf.write_i32(self.collector_entity_id);
    }
}

/// `0x0E Spawn Object`. Same layout as 1.8.9; a falling block's `data` is
/// `blockId | meta << 16` and an item frame's position is the wall block (handler matters).
/// java17: `S0EPacketSpawnObject`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnObject {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// The object type.
    pub object_type: i8,
    /// X * 32.
    pub x: i32,
    /// Y * 32.
    pub y: i32,
    /// Z * 32.
    pub z: i32,
    /// Pitch byte (`floor(pitch * 256 / 360)`).
    pub pitch: Angle,
    /// Yaw byte.
    pub yaw: Angle,
    /// Type-specific data.
    pub data: i32,
    /// Velocity * 8000, on the wire exactly when `data > 0`.
    pub velocity: Option<(i16, i16, i16)>,
}

impl Packet for SpawnObject {
    const ID: i32 = 0x0E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let object_type = buf.read_i8()?;
        let x = buf.read_i32()?;
        let y = buf.read_i32()?;
        let z = buf.read_i32()?;
        let pitch = buf.read_angle()?;
        let yaw = buf.read_angle()?;
        let data = buf.read_i32()?;
        let velocity = if data > 0 {
            Some((buf.read_i16()?, buf.read_i16()?, buf.read_i16()?))
        } else {
            None
        };
        Ok(SpawnObject {
            entity_id,
            object_type,
            x,
            y,
            z,
            pitch,
            yaw,
            data,
            velocity,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.object_type);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.pitch);
        buf.write_angle(self.yaw);
        buf.write_i32(self.data);
        if self.data > 0 {
            let (vx, vy, vz) = self.velocity.unwrap_or_default();
            buf.write_i16(vx);
            buf.write_i16(vy);
            buf.write_i16(vz);
        }
    }
}

/// `0x0F Spawn Mob`. java17: `S0FPacketSpawnMob`
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnMob {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// `EntityList` id (`readByte() & 255`).
    pub entity_type: u8,
    /// X * 32.
    pub x: i32,
    /// Y * 32.
    pub y: i32,
    /// Z * 32.
    pub z: i32,
    /// Yaw byte.
    pub yaw: Angle,
    /// Pitch byte.
    pub pitch: Angle,
    /// Head yaw byte (`rotationYawHead`; 1.8.9's source calls the same field `headPitch`).
    pub head_yaw: Angle,
    /// Velocity X * 8000.
    pub velocity_x: i16,
    /// Velocity Y * 8000.
    pub velocity_y: i16,
    /// Velocity Z * 8000.
    pub velocity_z: i16,
    /// The watcher list.
    pub metadata: Metadata5,
}

impl Packet for SpawnMob {
    const ID: i32 = 0x0F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnMob {
            entity_id: buf.read_varint()?,
            entity_type: buf.read_u8()?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            yaw: buf.read_angle()?,
            pitch: buf.read_angle()?,
            head_yaw: buf.read_angle()?,
            velocity_x: buf.read_i16()?,
            velocity_y: buf.read_i16()?,
            velocity_z: buf.read_i16()?,
            metadata: metadata::read(buf, &mut NbtBudget::per_packet())?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_u8(self.entity_type);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_angle(self.head_yaw);
        buf.write_i16(self.velocity_x);
        buf.write_i16(self.velocity_y);
        buf.write_i16(self.velocity_z);
        metadata::write(&self.metadata, buf);
    }
}

/// `0x10 Spawn Painting`: the position is the wall block and the direction an `int`.
/// java17: `S10PacketSpawnPainting`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnPainting {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// The art title, at most 13 chars (`EnumArt.maxArtTitleLength`).
    pub title: String,
    /// X of the block the painting hangs on.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Z.
    pub z: i32,
    /// `hangingDirection`: 0 south, 1 west, 2 north, 3 east.
    pub direction: i32,
}

impl SpawnPainting {
    /// The wall block.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, self.y, self.z)
    }
}

impl Packet for SpawnPainting {
    const ID: i32 = 0x10;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnPainting {
            entity_id: buf.read_varint()?,
            title: buf.read_string(13)?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            direction: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_string(&self.title);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_i32(self.direction);
    }
}

/// `0x11 Spawn Experience Orb`. java17: `S11PacketSpawnExperienceOrb`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnExperienceOrb {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// X * 32 (java17 `NetHandlerPlayClient.handleSpawnExperienceOrb` passes it to the
    /// `EntityXPOrb` constructor without dividing by 32).
    pub x: i32,
    /// Y * 32.
    pub y: i32,
    /// Z * 32.
    pub z: i32,
    /// Experience value.
    pub xp_value: i16,
}

impl Packet for SpawnExperienceOrb {
    const ID: i32 = 0x11;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnExperienceOrb {
            entity_id: buf.read_varint()?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            xp_value: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_i16(self.xp_value);
    }
}

/// `0x12 Entity Velocity`. java17: `S12PacketEntityVelocity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityVelocity {
    /// An `int` entity id.
    pub entity_id: i32,
    /// Motion X * 8000.
    pub motion_x: i16,
    /// Motion Y * 8000.
    pub motion_y: i16,
    /// Motion Z * 8000.
    pub motion_z: i16,
}

impl Packet for EntityVelocity {
    const ID: i32 = 0x12;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityVelocity {
            entity_id: buf.read_i32()?,
            motion_x: buf.read_i16()?,
            motion_y: buf.read_i16()?,
            motion_z: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i16(self.motion_x);
        buf.write_i16(self.motion_y);
        buf.write_i16(self.motion_z);
    }
}

/// `0x13 Destroy Entities`: a signed byte count, then `int` ids. java17:
/// `S13PacketDestroyEntities`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestroyEntities {
    /// The ids. A negative count is `new int[n]`'s exception; `write` narrows the count with
    /// `writeByte` as Java does, so more than 127 ids cannot be represented.
    pub entity_ids: Vec<i32>,
}

impl Packet for DestroyEntities {
    const ID: i32 = 0x13;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_i8()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(i32::from(count)));
        }
        let mut entity_ids = Vec::with_capacity(count as usize);
        for _ in 0..count {
            entity_ids.push(buf.read_i32()?);
        }
        Ok(DestroyEntities { entity_ids })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.entity_ids.len() as u8);
        for id in &self.entity_ids {
            buf.write_i32(*id);
        }
    }
}

/// `0x14 Entity`: no movement. java17: `S14PacketEntity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entity {
    /// An `int` entity id.
    pub entity_id: i32,
}

impl Packet for Entity {
    const ID: i32 = 0x14;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Entity {
            entity_id: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
    }
}

/// `0x15 Entity Relative Move`: no `onGround`. java17: `S14PacketEntity.S15PacketEntityRelMove`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityRelMove {
    /// An `int` entity id.
    pub entity_id: i32,
    /// dX * 32.
    pub delta_x: i8,
    /// dY * 32.
    pub delta_y: i8,
    /// dZ * 32.
    pub delta_z: i8,
}

impl Packet for EntityRelMove {
    const ID: i32 = 0x15;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityRelMove {
            entity_id: buf.read_i32()?,
            delta_x: buf.read_i8()?,
            delta_y: buf.read_i8()?,
            delta_z: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.delta_x);
        buf.write_i8(self.delta_y);
        buf.write_i8(self.delta_z);
    }
}

/// `0x16 Entity Look`: no `onGround`. java17: `S14PacketEntity.S16PacketEntityLook`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLook {
    /// An `int` entity id.
    pub entity_id: i32,
    /// Yaw byte.
    pub yaw: Angle,
    /// Pitch byte.
    pub pitch: Angle,
}

impl Packet for EntityLook {
    const ID: i32 = 0x16;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityLook {
            entity_id: buf.read_i32()?,
            yaw: buf.read_angle()?,
            pitch: buf.read_angle()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
    }
}

/// `0x17 Entity Look And Relative Move`: no `onGround`.
/// java17: `S14PacketEntity.S17PacketEntityLookMove`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLookAndRelMove {
    /// An `int` entity id.
    pub entity_id: i32,
    /// dX * 32.
    pub delta_x: i8,
    /// dY * 32.
    pub delta_y: i8,
    /// dZ * 32.
    pub delta_z: i8,
    /// Yaw byte.
    pub yaw: Angle,
    /// Pitch byte.
    pub pitch: Angle,
}

impl Packet for EntityLookAndRelMove {
    const ID: i32 = 0x17;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityLookAndRelMove {
            entity_id: buf.read_i32()?,
            delta_x: buf.read_i8()?,
            delta_y: buf.read_i8()?,
            delta_z: buf.read_i8()?,
            yaw: buf.read_angle()?,
            pitch: buf.read_angle()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.delta_x);
        buf.write_i8(self.delta_y);
        buf.write_i8(self.delta_z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
    }
}

/// `0x18 Entity Teleport`: no `onGround`. java17: `S18PacketEntityTeleport`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityTeleport {
    /// An `int` entity id.
    pub entity_id: i32,
    /// X * 32.
    pub x: i32,
    /// Y * 32.
    pub y: i32,
    /// Z * 32.
    pub z: i32,
    /// Yaw byte.
    pub yaw: Angle,
    /// Pitch byte.
    pub pitch: Angle,
}

impl Packet for EntityTeleport {
    const ID: i32 = 0x18;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityTeleport {
            entity_id: buf.read_i32()?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            yaw: buf.read_angle()?,
            pitch: buf.read_angle()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
    }
}

/// `0x19 Entity Head Look`. java17: `S19PacketEntityHeadLook`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityHeadLook {
    /// An `int` entity id.
    pub entity_id: i32,
    /// Head yaw byte.
    pub yaw: Angle,
}

impl Packet for EntityHeadLook {
    const ID: i32 = 0x19;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityHeadLook {
            entity_id: buf.read_i32()?,
            yaw: buf.read_angle()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_angle(self.yaw);
    }
}

/// `0x1A Entity Status`. java17: `S19PacketEntityStatus`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityStatus {
    /// An `int` entity id.
    pub entity_id: i32,
    /// The status opcode.
    pub opcode: i8,
}

impl Packet for EntityStatus {
    const ID: i32 = 0x1A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityStatus {
            entity_id: buf.read_i32()?,
            opcode: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.opcode);
    }
}

/// `0x1B Attach Entity`. java17: `S1BPacketEntityAttach`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityAttach {
    /// The rider (or leashed entity).
    pub entity_id: i32,
    /// The vehicle (or leash holder); -1 detaches.
    pub vehicle_entity_id: i32,
    /// 0 riding, 1 leash (`readUnsignedByte`).
    pub leash: u8,
}

impl Packet for EntityAttach {
    const ID: i32 = 0x1B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityAttach {
            entity_id: buf.read_i32()?,
            vehicle_entity_id: buf.read_i32()?,
            leash: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i32(self.vehicle_entity_id);
        buf.write_u8(self.leash);
    }
}

/// `0x1C Entity Metadata`. java17: `S1CPacketEntityMetadata`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMetadata {
    /// An `int` entity id.
    pub entity_id: i32,
    /// The changed entries (1.7.10 indices).
    pub metadata: Metadata5,
}

impl Packet for EntityMetadata {
    const ID: i32 = 0x1C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityMetadata {
            entity_id: buf.read_i32()?,
            metadata: metadata::read(buf, &mut NbtBudget::per_packet())?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        metadata::write(&self.metadata, buf);
    }
}

/// `0x1D Entity Effect`: no hide-particles flag. java17: `S1DPacketEntityEffect`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityEffect {
    /// An `int` entity id.
    pub entity_id: i32,
    /// Potion id.
    pub effect_id: i8,
    /// Amplifier.
    pub amplifier: i8,
    /// Duration in ticks, a `short`; 32767 is "infinite" (`func_149429_c`).
    pub duration: i16,
}

impl Packet for EntityEffect {
    const ID: i32 = 0x1D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityEffect {
            entity_id: buf.read_i32()?,
            effect_id: buf.read_i8()?,
            amplifier: buf.read_i8()?,
            duration: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.effect_id);
        buf.write_i8(self.amplifier);
        buf.write_i16(self.duration);
    }
}

/// `0x1E Remove Entity Effect`. java17: `S1EPacketRemoveEntityEffect`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoveEntityEffect {
    /// An `int` entity id.
    pub entity_id: i32,
    /// Potion id (`readUnsignedByte`).
    pub effect_id: u8,
}

impl Packet for RemoveEntityEffect {
    const ID: i32 = 0x1E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(RemoveEntityEffect {
            entity_id: buf.read_i32()?,
            effect_id: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_u8(self.effect_id);
    }
}

/// `0x1F Set Experience`: bar, level, total, the last two `short`s. java17:
/// `S1FPacketSetExperience` (`field_149400_c` is the level, written first)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetExperience {
    /// Progress bar, 0-1.
    pub experience_bar: f32,
    /// Level.
    pub level: i16,
    /// Total experience.
    pub total_experience: i16,
}

impl Packet for SetExperience {
    const ID: i32 = 0x1F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SetExperience {
            experience_bar: buf.read_f32()?,
            level: buf.read_i16()?,
            total_experience: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.experience_bar);
        buf.write_i16(self.level);
        buf.write_i16(self.total_experience);
    }
}

/// One modifier of an [`AttributeSnapshot`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributeModifier {
    /// Modifier UUID (two longs).
    pub uuid: Uuid,
    /// Amount.
    pub amount: f64,
    /// Operation 0-2 (`AttributeModifier`'s constructor rejects anything else).
    pub operation: u8,
}

/// One attribute of [`EntityProperties`].
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeSnapshot {
    /// Attribute name, at most 64 chars.
    pub key: String,
    /// Base value.
    pub base_value: f64,
    /// Modifiers in wire order.
    pub modifiers: Vec<AttributeModifier>,
}

/// `0x20 Entity Properties`. java17: `S20PacketEntityProperties`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityProperties {
    /// An `int` entity id.
    pub entity_id: i32,
    /// The attributes; an `int` count (a negative one reads none).
    pub properties: Vec<AttributeSnapshot>,
}

impl Packet for EntityProperties {
    const ID: i32 = 0x20;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        let count = buf.read_i32()?;
        let mut properties = Vec::new();
        for _ in 0..count.max(0) {
            let key = buf.read_string(64)?;
            let base_value = buf.read_f64()?;
            let modifier_count = buf.read_i16()?;
            let mut modifiers = Vec::new();
            for _ in 0..modifier_count.max(0) {
                let most = buf.read_i64()? as u64;
                let least = buf.read_i64()? as u64;
                let amount = buf.read_f64()?;
                let operation = buf.read_i8()?;
                // java17: Validate.inclusiveBetween(0, 2, operation, "Invalid operation")
                if !(0..=2).contains(&operation) {
                    return Err(ProtoError::InvalidEnum {
                        what: "attribute modifier operation",
                        value: i32::from(operation),
                    });
                }
                modifiers.push(AttributeModifier {
                    uuid: Uuid::from_u64_pair(most, least),
                    amount,
                    operation: operation as u8,
                });
            }
            properties.push(AttributeSnapshot {
                key,
                base_value,
                modifiers,
            });
        }
        Ok(EntityProperties {
            entity_id,
            properties,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i32(self.properties.len() as i32);
        for property in &self.properties {
            buf.write_string(&property.key);
            buf.write_f64(property.base_value);
            buf.write_i16(property.modifiers.len() as i16);
            for modifier in &property.modifiers {
                let (most, least) = modifier.uuid.as_u64_pair();
                buf.write_i64(most as i64);
                buf.write_i64(least as i64);
                buf.write_f64(modifier.amount);
                buf.write_u8(modifier.operation);
            }
        }
    }
}

/// One record of a [`MultiBlockChange`], as the 1.7.10 handler unpacks it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiBlockRecord {
    /// X within the chunk, 0-15.
    pub rel_x: u8,
    /// Y, 0-255.
    pub y: u8,
    /// Z within the chunk, 0-15.
    pub rel_z: u8,
    /// Block id, 12 bits.
    pub block_id: u16,
    /// Metadata, 4 bits.
    pub metadata: u8,
}

/// `0x22 Multi Block Change`: a record count, then an `int`-length blob of
/// `(short position, short id << 4 | meta)` records. java17: `S22PacketMultiBlockChange`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiBlockChange {
    /// Chunk X.
    pub chunk_x: i32,
    /// Chunk Z.
    pub chunk_z: i32,
    /// `readShort() & 65535`: how many records the handler reads.
    pub record_count: u16,
    /// The record bytes; `None` when the length was zero or negative (Java leaves the array
    /// `null`, and `writePacketData` then writes a length of 0).
    pub data: Option<Vec<u8>>,
}

impl MultiBlockChange {
    /// The records as `NetHandlerPlayClient.handleMultiBlockChange` reads them: up to
    /// `record_count` from `data`, stopping silently where the data runs out (its
    /// `IOException` is caught).
    pub fn records(&self) -> impl Iterator<Item = MultiBlockRecord> + '_ {
        let data = self.data.as_deref().unwrap_or(&[]);
        data.chunks_exact(4)
            .take(usize::from(self.record_count))
            .map(|r| {
                let pos = u16::from_be_bytes([r[0], r[1]]);
                let state = u16::from_be_bytes([r[2], r[3]]);
                MultiBlockRecord {
                    rel_x: (pos >> 12 & 15) as u8,
                    rel_z: (pos >> 8 & 15) as u8,
                    y: (pos & 255) as u8,
                    block_id: state >> 4 & 4095,
                    metadata: (state & 15) as u8,
                }
            })
    }
}

impl Packet for MultiBlockChange {
    const ID: i32 = 0x22;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let chunk_x = buf.read_i32()?;
        let chunk_z = buf.read_i32()?;
        let record_count = buf.read_u16()?;
        let len = buf.read_i32()?;
        let data = if len > 0 {
            Some(buf.read_bytes(len as usize)?)
        } else {
            None
        };
        Ok(MultiBlockChange {
            chunk_x,
            chunk_z,
            record_count,
            data,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.chunk_x);
        buf.write_i32(self.chunk_z);
        buf.write_u16(self.record_count);
        match &self.data {
            Some(data) => {
                buf.write_i32(data.len() as i32);
                buf.write_bytes(data);
            }
            None => buf.write_i32(0),
        }
    }
}

/// `0x23 Block Change`: a VarInt block id and a separate metadata byte.
/// java17: `S23PacketBlockChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockChange {
    /// X.
    pub x: i32,
    /// Y (`readUnsignedByte`).
    pub y: u8,
    /// Z.
    pub z: i32,
    /// The VarInt id `Block.getBlockById` resolves (an unknown id resolves to air there).
    pub block_id: i32,
    /// Metadata (`readUnsignedByte`).
    pub metadata: u8,
}

impl BlockChange {
    /// The changed block.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for BlockChange {
    const ID: i32 = 0x23;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(BlockChange {
            x: buf.read_i32()?,
            y: buf.read_u8()?,
            z: buf.read_i32()?,
            block_id: buf.read_varint()?,
            metadata: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_u8(self.y);
        buf.write_i32(self.z);
        buf.write_varint(self.block_id);
        buf.write_u8(self.metadata);
    }
}

/// `0x24 Block Action`. java17: `S24PacketBlockAction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAction {
    /// X.
    pub x: i32,
    /// Y, a `short`.
    pub y: i16,
    /// Z.
    pub z: i32,
    /// First data byte.
    pub data1: u8,
    /// Second data byte.
    pub data2: u8,
    /// `readVarIntFromBuffer() & 4095`.
    pub block_id: i32,
}

impl BlockAction {
    /// The block.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for BlockAction {
    const ID: i32 = 0x24;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(BlockAction {
            x: buf.read_i32()?,
            y: buf.read_i16()?,
            z: buf.read_i32()?,
            data1: buf.read_u8()?,
            data2: buf.read_u8()?,
            block_id: buf.read_varint()? & 4095,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_i16(self.y);
        buf.write_i32(self.z);
        buf.write_u8(self.data1);
        buf.write_u8(self.data2);
        buf.write_varint(self.block_id & 4095);
    }
}

/// `0x25 Block Break Animation`. java17: `S25PacketBlockBreakAnim`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockBreakAnimation {
    /// A VarInt entity id of the breaker.
    pub breaker_id: i32,
    /// X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Z.
    pub z: i32,
    /// Progress 0-9 (`readUnsignedByte`).
    pub progress: u8,
}

impl BlockBreakAnimation {
    /// The block.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, self.y, self.z)
    }
}

impl Packet for BlockBreakAnimation {
    const ID: i32 = 0x25;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(BlockBreakAnimation {
            breaker_id: buf.read_varint()?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            progress: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.breaker_id);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_u8(self.progress);
    }
}

/// `0x27 Explosion`. java17: `S27PacketExplosion`
#[derive(Debug, Clone, PartialEq)]
pub struct Explosion {
    /// Centre X (a float on the wire, widened to double by Java).
    pub x: f32,
    /// Centre Y.
    pub y: f32,
    /// Centre Z.
    pub z: f32,
    /// Strength.
    pub strength: f32,
    /// Destroyed blocks as byte offsets from `(int) x`, `(int) y`, `(int) z`.
    pub offsets: Vec<(i8, i8, i8)>,
    /// Player knockback X.
    pub motion_x: f32,
    /// Player knockback Y.
    pub motion_y: f32,
    /// Player knockback Z.
    pub motion_z: f32,
}

impl Explosion {
    /// The destroyed blocks as the handler sees them (`ChunkPosition`s).
    pub fn affected_blocks(&self) -> impl Iterator<Item = BlockPos> + '_ {
        // java: (int) of the double-widened float truncates toward zero.
        let (bx, by, bz) = (self.x as i32, self.y as i32, self.z as i32);
        self.offsets.iter().map(move |&(dx, dy, dz)| {
            BlockPos::new(
                i32::from(dx).wrapping_add(bx),
                i32::from(dy).wrapping_add(by),
                i32::from(dz).wrapping_add(bz),
            )
        })
    }
}

impl Packet for Explosion {
    const ID: i32 = 0x27;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let x = buf.read_f32()?;
        let y = buf.read_f32()?;
        let z = buf.read_f32()?;
        let strength = buf.read_f32()?;
        let count = buf.read_i32()?;
        if count < 0 {
            // java: new ArrayList(count) throws IllegalArgumentException.
            return Err(ProtoError::NegativeLength(count));
        }
        let mut offsets = Vec::new();
        for _ in 0..count {
            offsets.push((buf.read_i8()?, buf.read_i8()?, buf.read_i8()?));
        }
        Ok(Explosion {
            x,
            y,
            z,
            strength,
            offsets,
            motion_x: buf.read_f32()?,
            motion_y: buf.read_f32()?,
            motion_z: buf.read_f32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.x);
        buf.write_f32(self.y);
        buf.write_f32(self.z);
        buf.write_f32(self.strength);
        buf.write_i32(self.offsets.len() as i32);
        for &(dx, dy, dz) in &self.offsets {
            buf.write_i8(dx);
            buf.write_i8(dy);
            buf.write_i8(dz);
        }
        buf.write_f32(self.motion_x);
        buf.write_f32(self.motion_y);
        buf.write_f32(self.motion_z);
    }
}

/// `0x28 Effect` (sound or particle event). java17: `S28PacketEffect`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Effect {
    /// Event id.
    pub effect_id: i32,
    /// X.
    pub x: i32,
    /// Y (`readByte() & 255`).
    pub y: u8,
    /// Z.
    pub z: i32,
    /// Event data.
    pub data: i32,
    /// Play at full volume regardless of distance (`func_149244_c`).
    pub disable_relative_volume: bool,
}

impl Effect {
    /// The position.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for Effect {
    const ID: i32 = 0x28;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Effect {
            effect_id: buf.read_i32()?,
            x: buf.read_i32()?,
            y: buf.read_u8()?,
            z: buf.read_i32()?,
            data: buf.read_i32()?,
            disable_relative_volume: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.effect_id);
        buf.write_i32(self.x);
        buf.write_u8(self.y);
        buf.write_i32(self.z);
        buf.write_i32(self.data);
        buf.write_bool(self.disable_relative_volume);
    }
}

/// `0x29 Sound Effect` (named). java17: `S29PacketSoundEffect`
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEffect {
    /// Sound event name, at most 256 chars.
    pub sound_name: String,
    /// X * 8.
    pub x: i32,
    /// Y * 8.
    pub y: i32,
    /// Z * 8.
    pub z: i32,
    /// Volume.
    pub volume: f32,
    /// Pitch * 63 (`readUnsignedByte`).
    pub pitch: u8,
}

impl Packet for SoundEffect {
    const ID: i32 = 0x29;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SoundEffect {
            sound_name: buf.read_string(256)?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
            volume: buf.read_f32()?,
            pitch: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.sound_name);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_f32(self.volume);
        buf.write_u8(self.pitch);
    }
}

/// `0x2A Particles`: by name (`readStringFromBuffer(64)`), no long-distance flag and no extra
/// arguments; block and item particles encode theirs in the name. java17: `S2APacketParticles`
#[derive(Debug, Clone, PartialEq)]
pub struct Particles {
    /// The particle name `RenderGlobal.doSpawnParticle` dispatches on, e.g. `iconcrack_1_0`.
    pub particle_name: String,
    /// X.
    pub x: f32,
    /// Y.
    pub y: f32,
    /// Z.
    pub z: f32,
    /// Offset X.
    pub offset_x: f32,
    /// Offset Y.
    pub offset_y: f32,
    /// Offset Z.
    pub offset_z: f32,
    /// Speed.
    pub speed: f32,
    /// Count.
    pub count: i32,
}

impl Packet for Particles {
    const ID: i32 = 0x2A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Particles {
            particle_name: buf.read_string(64)?,
            x: buf.read_f32()?,
            y: buf.read_f32()?,
            z: buf.read_f32()?,
            offset_x: buf.read_f32()?,
            offset_y: buf.read_f32()?,
            offset_z: buf.read_f32()?,
            speed: buf.read_f32()?,
            count: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.particle_name);
        buf.write_f32(self.x);
        buf.write_f32(self.y);
        buf.write_f32(self.z);
        buf.write_f32(self.offset_x);
        buf.write_f32(self.offset_y);
        buf.write_f32(self.offset_z);
        buf.write_f32(self.speed);
        buf.write_i32(self.count);
    }
}

/// `0x2B Change Game State`. java17: `S2BPacketChangeGameState`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangeGameState {
    /// Reason (`readUnsignedByte`).
    pub reason: u8,
    /// Value.
    pub value: f32,
}

impl Packet for ChangeGameState {
    const ID: i32 = 0x2B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ChangeGameState {
            reason: buf.read_u8()?,
            value: buf.read_f32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.reason);
        buf.write_f32(self.value);
    }
}

/// `0x2C Spawn Global Entity` (lightning). java17: `S2CPacketSpawnGlobalEntity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnGlobalEntity {
    /// A VarInt entity id.
    pub entity_id: i32,
    /// Type (1 = lightning bolt).
    pub kind: i8,
    /// X * 32.
    pub x: i32,
    /// Y * 32.
    pub y: i32,
    /// Z * 32.
    pub z: i32,
}

impl Packet for SpawnGlobalEntity {
    const ID: i32 = 0x2C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnGlobalEntity {
            entity_id: buf.read_varint()?,
            kind: buf.read_i8()?,
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.kind);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
    }
}

/// `0x2D Open Window`: a numeric inventory type and a plain-text title.
/// java17: `S2DPacketOpenWindow`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenWindow {
    /// Window id.
    pub window_id: u8,
    /// 0 chest, 1 workbench, 2 furnace, 3 dispenser, 4 enchanting table, 5 brewing stand,
    /// 6 villager, 7 beacon, 8 anvil, 9 hopper, 10 dropper, 11 horse
    /// (`NetHandlerPlayClient.handleOpenWindow`).
    pub inventory_type: u8,
    /// Plain-text title, at most 32 chars.
    pub title: String,
    /// Slot count.
    pub slot_count: u8,
    /// Whether `title` replaces the container's default name.
    pub use_provided_title: bool,
    /// The horse's entity id, on the wire only for type 11 (0 otherwise).
    pub entity_id: i32,
}

impl Packet for OpenWindow {
    const ID: i32 = 0x2D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let inventory_type = buf.read_u8()?;
        let title = buf.read_string(32)?;
        let slot_count = buf.read_u8()?;
        let use_provided_title = buf.read_bool()?;
        let entity_id = if inventory_type == 11 {
            buf.read_i32()?
        } else {
            0
        };
        Ok(OpenWindow {
            window_id,
            inventory_type,
            title,
            slot_count,
            use_provided_title,
            entity_id,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_u8(self.inventory_type);
        buf.write_string(&self.title);
        buf.write_u8(self.slot_count);
        buf.write_bool(self.use_provided_title);
        if self.inventory_type == 11 {
            buf.write_i32(self.entity_id);
        }
    }
}

/// `0x2E Close Window`. java17: `S2EPacketCloseWindow`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseWindow {
    /// Window id (`readUnsignedByte`).
    pub window_id: u8,
}

impl Packet for CloseWindow {
    const ID: i32 = 0x2E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(CloseWindow {
            window_id: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
    }
}

/// `0x2F Set Slot`. java17: `S2FPacketSetSlot`
#[derive(Debug, Clone, PartialEq)]
pub struct SetSlot {
    /// Window id, signed (-1 with slot -1 is the cursor stack).
    pub window_id: i8,
    /// Slot.
    pub slot: i16,
    /// The stack.
    pub item: Option<ItemStack>,
}

impl Packet for SetSlot {
    const ID: i32 = 0x2F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(SetSlot {
            window_id: buf.read_i8()?,
            slot: buf.read_i16()?,
            item: buf.read_item_v5(&mut budget)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
        buf.write_i16(self.slot);
        buf.write_item_v5(self.item.as_ref());
    }
}

/// `0x30 Window Items`. java17: `S30PacketWindowItems`
#[derive(Debug, Clone, PartialEq)]
pub struct WindowItems {
    /// Window id (`readUnsignedByte`).
    pub window_id: u8,
    /// Every slot; a `short` count (negative is `new ItemStack[n]`'s exception).
    pub items: Vec<Option<ItemStack>>,
}

impl Packet for WindowItems {
    const ID: i32 = 0x30;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let count = buf.read_i16()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(i32::from(count)));
        }
        let mut budget = NbtBudget::per_packet();
        let mut items = Vec::new();
        for _ in 0..count {
            items.push(buf.read_item_v5(&mut budget)?);
        }
        Ok(WindowItems { window_id, items })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.items.len() as i16);
        for item in &self.items {
            buf.write_item_v5(item.as_ref());
        }
    }
}

/// `0x31 Window Property`. java17: `S31PacketWindowProperty`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowProperty {
    /// Window id (`readUnsignedByte`).
    pub window_id: u8,
    /// Property.
    pub property: i16,
    /// Value.
    pub value: i16,
}

impl Packet for WindowProperty {
    const ID: i32 = 0x31;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(WindowProperty {
            window_id: buf.read_u8()?,
            property: buf.read_i16()?,
            value: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.property);
        buf.write_i16(self.value);
    }
}

/// `0x32 Confirm Transaction`. java17: `S32PacketConfirmTransaction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfirmTransaction {
    /// Window id (`readUnsignedByte`).
    pub window_id: u8,
    /// Action number.
    pub action_number: i16,
    /// Accepted.
    pub accepted: bool,
}

impl Packet for ConfirmTransaction {
    const ID: i32 = 0x32;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ConfirmTransaction {
            window_id: buf.read_u8()?,
            action_number: buf.read_i16()?,
            accepted: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.action_number);
        buf.write_bool(self.accepted);
    }
}

/// `0x33 Update Sign`: four plain-text lines of at most 15 chars. java17: `S33PacketUpdateSign`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateSign {
    /// X.
    pub x: i32,
    /// Y, a `short`.
    pub y: i16,
    /// Z.
    pub z: i32,
    /// The lines.
    pub lines: [String; 4],
}

impl UpdateSign {
    /// The sign.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for UpdateSign {
    const ID: i32 = 0x33;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(UpdateSign {
            x: buf.read_i32()?,
            y: buf.read_i16()?,
            z: buf.read_i32()?,
            lines: [
                buf.read_string(15)?,
                buf.read_string(15)?,
                buf.read_string(15)?,
                buf.read_string(15)?,
            ],
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_i16(self.y);
        buf.write_i32(self.z);
        for line in &self.lines {
            buf.write_string(line);
        }
    }
}

/// `0x34 Maps`: a map id and an unsigned-short-length blob whose first byte selects colours
/// (0), icons (1) or scale (2) (`MapData.updateMPMapData`). java17: `S34PacketMaps`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Maps {
    /// A VarInt map id.
    pub map_id: i32,
    /// The blob (`readUnsignedShort` length). Writing more than 65535 bytes is refused.
    pub data: Vec<u8>,
}

/// One icon of a [`MapUpdate::Icons`] update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapIcon {
    /// `(byte)(b >> 4)` of the **signed** first byte: 0-7 for vanilla icons, negative above.
    pub kind: i8,
    /// `b & 15`.
    pub rotation: u8,
    /// X on the map, a signed byte.
    pub x: i8,
    /// Z on the map, a signed byte.
    pub z: i8,
}

/// What a [`Maps`] blob does, read as `MapData.updateMPMapData` reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MapUpdate<'a> {
    /// First byte 0: `colors[(i + y_start) * 128 + x] = colors[i]` for every byte after the
    /// header. Java writes them in order and throws at the first index past 16383.
    Colors {
        /// Column.
        x: u8,
        /// First row.
        y_start: u8,
        /// The colour bytes.
        colors: &'a [u8],
    },
    /// First byte 1: every icon, replacing the previous ones; `(len - 1) / 3` triples, a partial
    /// one ignored.
    Icons(Vec<MapIcon>),
    /// First byte 2: the map scale (a signed byte).
    Scale(i8),
    /// Any other first byte: nothing happens.
    Ignored,
}

impl Maps {
    /// java17: `MapData.updateMPMapData`.
    ///
    /// # Errors
    ///
    /// [`ProtoError::Eof`] where Java's handler indexes past the array: an empty blob, a colour
    /// update shorter than its three header bytes, a scale update without its byte.
    pub fn update(&self) -> Result<MapUpdate<'_>, ProtoError> {
        let short = || ProtoError::Eof("a map update");
        let kind = *self.data.first().ok_or_else(short)?;
        Ok(match kind {
            0 => {
                let [_, x, y_start, ..] = self.data[..] else {
                    return Err(short());
                };
                MapUpdate::Colors {
                    x,
                    y_start,
                    colors: &self.data[3..],
                }
            }
            1 => MapUpdate::Icons(
                self.data[1..]
                    .chunks_exact(3)
                    .map(|icon| MapIcon {
                        kind: (icon[0] as i8) >> 4,
                        rotation: icon[0] & 15,
                        x: icon[1] as i8,
                        z: icon[2] as i8,
                    })
                    .collect(),
            ),
            2 => MapUpdate::Scale(*self.data.get(1).ok_or_else(short)? as i8),
            _ => MapUpdate::Ignored,
        })
    }
}

impl Packet for Maps {
    const ID: i32 = 0x34;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let map_id = buf.read_varint()?;
        let len = buf.read_u16()?;
        Ok(Maps {
            map_id,
            data: buf.read_bytes(usize::from(len))?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.map_id);
        if self.data.len() > usize::from(u16::MAX) {
            buf.write_bounded_payload(&self.data, usize::from(u16::MAX));
            return;
        }
        buf.write_u16(self.data.len() as u16);
        buf.write_bytes(&self.data);
    }
}

/// `0x35 Update Tile Entity`. java17: `S35PacketUpdateTileEntity`
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateTileEntity {
    /// X.
    pub x: i32,
    /// Y, a `short`.
    pub y: i16,
    /// Z.
    pub z: i32,
    /// Which tile entity kind the tag is for (`readUnsignedByte`).
    pub action: u8,
    /// The tag, gzip on the wire.
    pub nbt: Option<NbtCompound>,
}

impl UpdateTileEntity {
    /// The tile entity's block.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for UpdateTileEntity {
    const ID: i32 = 0x35;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(UpdateTileEntity {
            x: buf.read_i32()?,
            y: buf.read_i16()?,
            z: buf.read_i32()?,
            action: buf.read_u8()?,
            nbt: buf.read_nbt_v5(&mut budget)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_i16(self.y);
        buf.write_i32(self.z);
        buf.write_u8(self.action);
        buf.write_nbt_v5(self.nbt.as_ref());
    }
}

/// `0x36 Sign Editor Open`: three ints. java17: `S36PacketSignEditorOpen`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignEditorOpen {
    /// X.
    pub x: i32,
    /// Y.
    pub y: i32,
    /// Z.
    pub z: i32,
}

impl SignEditorOpen {
    /// The sign.
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, self.y, self.z)
    }
}

impl Packet for SignEditorOpen {
    const ID: i32 = 0x36;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SignEditorOpen {
            x: buf.read_i32()?,
            y: buf.read_i32()?,
            z: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
    }
}

/// `0x37 Statistics`. java17: `S37PacketStatistics`
///
/// `readPacketData` resolves each name with `StatList.getOneShotStat` and drops names it does
/// not know (numeric ids such as `stat.mineBlock.1` in 1.7.10); that lookup needs the 1.7.10
/// registries, so the entries are kept in wire order here and the filter is the handler's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Statistics {
    /// `(stat id, value)` in wire order.
    pub entries: Vec<(String, i32)>,
}

impl Packet for Statistics {
    const ID: i32 = 0x37;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_varint()?;
        let mut entries = Vec::new();
        for _ in 0..count.max(0) {
            entries.push((buf.read_string(32767)?, buf.read_varint()?));
        }
        Ok(Statistics { entries })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entries.len() as i32);
        for (name, value) in &self.entries {
            buf.write_string(name);
            buf.write_varint(*value);
        }
    }
}

/// `0x38 Player List Item`: one row by display name. java17: `S38PacketPlayerListItem`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerListItem {
    /// The name shown (at most 16 chars); the row's key.
    pub name: String,
    /// `true` adds or updates the row, `false` removes it.
    pub online: bool,
    /// Ping in milliseconds, a `short`.
    pub ping: i16,
}

impl Packet for PlayerListItem {
    const ID: i32 = 0x38;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerListItem {
            name: buf.read_string(16)?,
            online: buf.read_bool()?,
            ping: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.name);
        buf.write_bool(self.online);
        buf.write_i16(self.ping);
    }
}

/// `0x39 Player Abilities`. java17: `S39PacketPlayerAbilities`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerAbilities {
    /// Flag 1.
    pub invulnerable: bool,
    /// Flag 2.
    pub flying: bool,
    /// Flag 4.
    pub allow_flying: bool,
    /// Flag 8.
    pub creative_mode: bool,
    /// Fly speed.
    pub fly_speed: f32,
    /// Walk speed.
    pub walk_speed: f32,
}

impl Packet for PlayerAbilities {
    const ID: i32 = 0x39;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let flags = buf.read_u8()?;
        Ok(PlayerAbilities {
            invulnerable: flags & 1 != 0,
            flying: flags & 2 != 0,
            allow_flying: flags & 4 != 0,
            creative_mode: flags & 8 != 0,
            fly_speed: buf.read_f32()?,
            walk_speed: buf.read_f32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        let mut flags = 0u8;
        if self.invulnerable {
            flags |= 1;
        }
        if self.flying {
            flags |= 2;
        }
        if self.allow_flying {
            flags |= 4;
        }
        if self.creative_mode {
            flags |= 8;
        }
        buf.write_u8(flags);
        buf.write_f32(self.fly_speed);
        buf.write_f32(self.walk_speed);
    }
}

/// `0x3A Tab Complete`. java17: `S3APacketTabComplete`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabComplete {
    /// Completions; a VarInt count (negative is `new String[n]`'s exception).
    pub matches: Vec<String>,
}

impl Packet for TabComplete {
    const ID: i32 = 0x3A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_varint()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        let mut matches = Vec::new();
        for _ in 0..count {
            matches.push(buf.read_string(32767)?);
        }
        Ok(TabComplete { matches })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.matches.len() as i32);
        for m in &self.matches {
            buf.write_string(m);
        }
    }
}

/// `0x3B Scoreboard Objective`: the display name is always sent, there is no render type.
/// java17: `S3BPacketScoreboardObjective`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreboardObjective {
    /// Objective name, at most 16 chars.
    pub objective_name: String,
    /// Display name, at most 32 chars (plain text).
    pub display_name: String,
    /// 0 create, 1 remove, 2 update.
    pub mode: i8,
}

impl Packet for ScoreboardObjective {
    const ID: i32 = 0x3B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ScoreboardObjective {
            objective_name: buf.read_string(16)?,
            display_name: buf.read_string(32)?,
            mode: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.objective_name);
        buf.write_string(&self.display_name);
        buf.write_i8(self.mode);
    }
}

/// `0x3C Update Score`. java17: `S3CPacketUpdateScore`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateScore {
    /// The score holder, at most 16 chars.
    pub score_name: String,
    /// 0 change, 1 remove (a signed byte).
    pub action: i8,
    /// Objective, at most 16 chars; on the wire unless `action == 1`.
    pub objective_name: String,
    /// Value, an `int`; on the wire unless `action == 1`.
    pub value: i32,
}

impl Packet for UpdateScore {
    const ID: i32 = 0x3C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let score_name = buf.read_string(16)?;
        let action = buf.read_i8()?;
        let (objective_name, value) = if action != 1 {
            (buf.read_string(16)?, buf.read_i32()?)
        } else {
            (String::new(), 0)
        };
        Ok(UpdateScore {
            score_name,
            action,
            objective_name,
            value,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.score_name);
        buf.write_i8(self.action);
        if self.action != 1 {
            buf.write_string(&self.objective_name);
            buf.write_i32(self.value);
        }
    }
}

/// `0x3D Display Scoreboard`. java17: `S3DPacketDisplayScoreboard`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayScoreboard {
    /// 0 list, 1 sidebar, 2 below name.
    pub position: i8,
    /// Objective name, at most 16 chars.
    pub score_name: String,
}

impl Packet for DisplayScoreboard {
    const ID: i32 = 0x3D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(DisplayScoreboard {
            position: buf.read_i8()?,
            score_name: buf.read_string(16)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.position);
        buf.write_string(&self.score_name);
    }
}

/// `0x3E Teams`: no name-tag visibility and no colour. java17: `S3EPacketTeams`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Teams {
    /// Team name, at most 16 chars.
    pub team_name: String,
    /// 0 create, 1 remove, 2 update info, 3 add players, 4 remove players (a signed byte).
    pub mode: i8,
    /// Display name, at most 32 chars; on the wire for modes 0 and 2.
    pub display_name: String,
    /// Prefix, at most 16 chars; modes 0 and 2.
    pub prefix: String,
    /// Suffix, at most 16 chars; modes 0 and 2.
    pub suffix: String,
    /// Friendly-fire flags; modes 0 and 2.
    pub friendly_flags: i8,
    /// Player names (at most 40 chars each, a `short` count); modes 0, 3 and 4.
    pub players: Vec<String>,
}

impl Teams {
    fn has_info(&self) -> bool {
        self.mode == 0 || self.mode == 2
    }

    fn has_players(&self) -> bool {
        self.mode == 0 || self.mode == 3 || self.mode == 4
    }
}

impl Packet for Teams {
    const ID: i32 = 0x3E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut teams = Teams {
            team_name: buf.read_string(16)?,
            mode: buf.read_i8()?,
            display_name: String::new(),
            prefix: String::new(),
            suffix: String::new(),
            friendly_flags: 0,
            players: Vec::new(),
        };
        if teams.has_info() {
            teams.display_name = buf.read_string(32)?;
            teams.prefix = buf.read_string(16)?;
            teams.suffix = buf.read_string(16)?;
            teams.friendly_flags = buf.read_i8()?;
        }
        if teams.has_players() {
            let count = buf.read_i16()?;
            for _ in 0..count.max(0) {
                teams.players.push(buf.read_string(40)?);
            }
        }
        Ok(teams)
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.team_name);
        buf.write_i8(self.mode);
        if self.has_info() {
            buf.write_string(&self.display_name);
            buf.write_string(&self.prefix);
            buf.write_string(&self.suffix);
            buf.write_i8(self.friendly_flags);
        }
        if self.has_players() {
            buf.write_i16(self.players.len() as i16);
            for player in &self.players {
                buf.write_string(player);
            }
        }
    }
}

/// `0x3F Plugin Message`: an unsigned-short length before the payload.
/// java17: `S3FPacketCustomPayload`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMessage {
    /// Channel, at most 20 chars.
    pub channel: String,
    /// Payload (`readUnsignedShort` length). Writing more than 65535 bytes is refused.
    pub data: Vec<u8>,
}

impl Packet for PluginMessage {
    const ID: i32 = 0x3F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let channel = buf.read_string(20)?;
        let len = buf.read_u16()?;
        Ok(PluginMessage {
            channel,
            data: buf.read_bytes(usize::from(len))?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.channel);
        if self.data.len() > usize::from(u16::MAX) {
            buf.write_bounded_payload(&self.data, usize::from(u16::MAX));
            return;
        }
        buf.write_u16(self.data.len() as u16);
        buf.write_bytes(&self.data);
    }
}

/// `0x40 Disconnect`. java17: `S40PacketDisconnect`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    /// The reason as chat JSON.
    pub reason: String,
}

impl Packet for Disconnect {
    const ID: i32 = 0x40;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Disconnect {
            reason: buf.read_chat()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_chat(&self.reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode<P: Packet>(bytes: &[u8]) -> P {
        let mut buf = PacketBuffer::from_slice(bytes);
        let packet = P::read(&mut buf).unwrap();
        assert!(buf.is_empty(), "trailing bytes");
        packet
    }

    fn encode<P: Packet>(packet: &P) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        buf.finish_write().unwrap();
        buf.as_slice().to_vec()
    }

    #[test]
    fn join_game_maps_like_readpacketdata() {
        // entity 7, hardcore creative, nether, difficulty 6 (-> 2), 20 players, "FLAT"
        let wire = [0, 0, 0, 7, 9, 0xFF, 6, 20, 4, b'F', b'L', b'A', b'T'];
        let packet: JoinGame = decode(&wire);
        assert!(packet.hardcore_mode);
        assert_eq!(packet.game_mode, 1);
        assert_eq!(packet.dimension, -1);
        assert_eq!(packet.difficulty, 2);
        assert_eq!(packet.level_type, "flat");
        // Spectator (3) does not exist in 1.7.10: GameType.getByID falls back to SURVIVAL.
        let spectator: JoinGame = decode(&[
            0, 0, 0, 7, 3, 0, 0, 20, 7, b'd', b'e', b'f', b'a', b'u', b'l', b't',
        ]);
        assert_eq!(spectator.game_mode, 0);
        // Canonical input re-encodes byte for byte.
        let canonical = [0, 0, 0, 7, 9, 0xFF, 2, 20, 4, b'f', b'l', b'a', b't'];
        assert_eq!(encode(&decode::<JoinGame>(&canonical)), canonical);
    }

    #[test]
    fn spawn_object_velocity_follows_data() {
        let packet = SpawnObject {
            entity_id: 300,
            object_type: 70,
            x: -32,
            y: 2048,
            z: 64,
            pitch: Angle(0),
            yaw: Angle(0x80),
            data: 12 | 2 << 16,
            velocity: Some((1, -2, 3)),
        };
        let bytes = encode(&packet);
        assert_eq!(decode::<SpawnObject>(&bytes), packet);
        let no_velocity = SpawnObject {
            data: 0,
            velocity: None,
            ..packet
        };
        let bytes = encode(&no_velocity);
        assert_eq!(bytes.len(), 2 + 1 + 12 + 2 + 4);
        assert_eq!(decode::<SpawnObject>(&bytes), no_velocity);
    }

    #[test]
    fn multi_block_records_unpack_like_the_handler() {
        let packet = MultiBlockChange {
            chunk_x: 1,
            chunk_z: -1,
            record_count: 3,
            // x 3, z 4, y 70 -> stone:2 ; x 15, z 15, y 255 -> id 4095:15 ; half a record
            data: Some(vec![0x34, 70, 0x00, 0x12, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]),
        };
        let records: Vec<_> = packet.records().collect();
        assert_eq!(records.len(), 2);
        assert_eq!(
            records[0],
            MultiBlockRecord {
                rel_x: 3,
                y: 70,
                rel_z: 4,
                block_id: 1,
                metadata: 2
            }
        );
        assert_eq!(records[1].block_id, 4095);
        assert_eq!(records[1].metadata, 15);
        assert_eq!(decode::<MultiBlockChange>(&encode(&packet)), packet);

        // A non-positive length leaves no data and nothing more is read.
        let none: MultiBlockChange =
            decode(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert_eq!(none.data, None);
        assert_eq!(none.records().count(), 0);
    }

    #[test]
    fn spawn_player_carries_the_profile() {
        let packet = SpawnPlayer {
            entity_id: 5,
            uuid: Uuid::from_u128(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef),
            name: "Steve".into(),
            properties: vec![ProfileProperty {
                name: "textures".into(),
                value: "e30=".into(),
                signature: String::new(),
            }],
            x: 1,
            y: 2,
            z: 3,
            yaw: Angle(4),
            pitch: Angle(5),
            current_item: 0,
            metadata: Vec::new(),
        };
        let bytes = encode(&packet);
        assert_eq!(&bytes[1..2], &[36]);
        assert_eq!(&bytes[2..38], b"01234567-89ab-cdef-0123-456789abcdef");
        assert_eq!(decode::<SpawnPlayer>(&bytes), packet);
        let mut bad = bytes.clone();
        bad[10] = b'x';
        assert!(SpawnPlayer::read(&mut PacketBuffer::from_slice(&bad)).is_err());
    }

    #[test]
    fn entity_properties_rejects_an_invalid_operation() {
        let packet = EntityProperties {
            entity_id: 1,
            properties: vec![AttributeSnapshot {
                key: "generic.movementSpeed".into(),
                base_value: 0.1,
                modifiers: vec![AttributeModifier {
                    uuid: Uuid::from_u128(1),
                    amount: 0.3,
                    operation: 2,
                }],
            }],
        };
        let mut bytes = encode(&packet);
        assert_eq!(decode::<EntityProperties>(&bytes), packet);
        let last = bytes.len() - 1;
        bytes[last] = 3;
        assert!(EntityProperties::read(&mut PacketBuffer::from_slice(&bytes)).is_err());
    }

    #[test]
    fn negative_counts_follow_java() {
        // S13 and S30: `new T[n]` throws; S20, S37 and S3E: the loop never runs.
        assert!(DestroyEntities::read(&mut PacketBuffer::from_slice(&[0xFF])).is_err());
        assert!(WindowItems::read(&mut PacketBuffer::from_slice(&[0, 0xFF, 0xFF])).is_err());
        let props: EntityProperties = decode(&[0, 0, 0, 1, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(props.properties.is_empty());
        let teams: Teams = decode(&[1, b't', 3, 0xFF, 0xFF]);
        assert!(teams.players.is_empty());
        assert!(Explosion::read(&mut PacketBuffer::from_slice(
            &[0; 16]
                .iter()
                .chain(&[0xFF; 4])
                .copied()
                .collect::<Vec<_>>()
        ))
        .is_err());
    }

    #[test]
    fn explosion_offsets_are_relative_to_the_truncated_centre() {
        let packet = Explosion {
            x: -1.5,
            y: 64.9,
            z: 0.5,
            strength: 4.0,
            offsets: vec![(1, -1, 0)],
            motion_x: 0.0,
            motion_y: 0.0,
            motion_z: 0.0,
        };
        let blocks: Vec<_> = packet.affected_blocks().collect();
        assert_eq!(blocks, vec![BlockPos::new(0, 63, 0)]);
        assert_eq!(decode::<Explosion>(&encode(&packet)), packet);
    }

    #[test]
    fn open_window_reads_the_entity_id_only_for_horses() {
        let chest = OpenWindow {
            window_id: 1,
            inventory_type: 0,
            title: "Chest".into(),
            slot_count: 27,
            use_provided_title: false,
            entity_id: 0,
        };
        assert_eq!(encode(&chest).len(), 1 + 1 + 6 + 1 + 1);
        let horse = OpenWindow {
            inventory_type: 11,
            entity_id: 77,
            ..chest
        };
        assert_eq!(decode::<OpenWindow>(&encode(&horse)), horse);
    }

    #[test]
    fn plugin_message_and_maps_are_short_prefixed() {
        let packet = PluginMessage {
            channel: "MC|Brand".into(),
            data: b"vanilla".to_vec(),
        };
        let bytes = encode(&packet);
        assert_eq!(&bytes[9..11], &[0, 7]);
        assert_eq!(decode::<PluginMessage>(&bytes), packet);
        let mut too_big = PacketBuffer::new();
        PluginMessage {
            channel: "x".into(),
            data: vec![0; 65_536],
        }
        .write(&mut too_big);
        assert!(too_big.finish_write().is_err());
    }

    #[test]
    fn map_updates_read_like_update_mp_map_data() {
        let maps = |data: Vec<u8>| Maps { map_id: 1, data };
        assert_eq!(
            maps(vec![0, 5, 6, 7, 8]).update().unwrap(),
            MapUpdate::Colors {
                x: 5,
                y_start: 6,
                colors: &[7, 8]
            }
        );
        // 0xF5 >> 4 of the signed byte is -1; a trailing partial triple is ignored.
        assert_eq!(
            maps(vec![1, 0xF5, 0x80, 0x7F, 0x21, 0x01])
                .update()
                .unwrap(),
            MapUpdate::Icons(vec![MapIcon {
                kind: -1,
                rotation: 5,
                x: -128,
                z: 127
            }])
        );
        assert_eq!(maps(vec![2, 3]).update().unwrap(), MapUpdate::Scale(3));
        assert_eq!(maps(vec![9]).update().unwrap(), MapUpdate::Ignored);
        // Where Java's handler indexes past the array.
        for short in [vec![], vec![0, 1], vec![2]] {
            assert!(maps(short).update().is_err());
        }
    }
}
