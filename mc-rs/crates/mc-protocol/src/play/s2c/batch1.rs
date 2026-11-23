//! Clientbound Play, ids `0x00`-`0x13`: `KeepAlive` .. `DestroyEntities`.
//!
//! Fixtures in this file's tests are synthetic (hand-built plausible values), not captured
//! vanilla traffic — see `tests/golden.rs` for why, and for a handful of independently
//! hand-derived byte vectors covering the state-transition packets.
//!
//! java: `network/play/server/S00PacketKeepAlive.java` .. `S13PacketDestroyEntities.java`

use mc_math::Facing;
use uuid::Uuid;

use crate::metadata::{self, Metadata};
use crate::packet::Packet;
use crate::{Angle, PacketBuffer, ProtoError};

/// Return the canonical wire spelling for the 1.8.9 `WorldType` registry.
///
/// `S01PacketJoinGame` and `S07PacketRespawn` decode the string through
/// `WorldType.parseWorldType`, which compares names case-insensitively, and then replace a
/// missing entry with `WorldType.DEFAULT`.  Keeping that normalization at the packet boundary
/// means every downstream consumer observes the same enum value Java does, and packet writes
/// use `WorldType.getWorldTypeName()`'s canonical spelling.
fn canonical_level_type_name(level_type: &str) -> &'static str {
    const WORLD_TYPES: [&str; 7] = [
        "default",
        "flat",
        "largeBiomes",
        "amplified",
        "customized",
        "debug_all_block_states",
        "default_1_1",
    ];

    WORLD_TYPES
        .into_iter()
        .find(|name| level_type.eq_ignore_ascii_case(name))
        .unwrap_or("default")
}

/// `0x00 Keep Alive`: an opaque id the client must echo back (`C2SPlay::KeepAlive`) to avoid
/// being timed out. java: `S00PacketKeepAlive`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive {
    /// Server-chosen id (VarInt on the wire — not the plain `i32` some later protocol versions
    /// use).
    pub id: i32,
}

impl Packet for KeepAlive {
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(KeepAlive {
            id: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.id);
    }
}

/// `0x01 Join Game`: the first packet after login success, describing the world the player is
/// entering. java: `S01PacketJoinGame`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinGame {
    /// The local player's own entity id.
    pub entity_id: i32,
    /// Whether the world is hardcore (packed as bit 3 of the gamemode byte on the wire — see
    /// `game_mode`'s doc).
    pub hardcore_mode: bool,
    /// `0`=survival, `1`=creative, `2`=adventure, `3`=spectator. Already unpacked from the
    /// wire's combined `gamemode | (hardcore ? 8 : 0)` byte.
    pub game_mode: u8,
    /// A signed byte on the wire (unlike [`super::batch1::Respawn::dimension`], which is a full
    /// `i32`) — this asymmetry is vanilla's, not a simplification here. `-1`=Nether, `0`=
    /// Overworld, `1`=End are the vanilla values, but nothing stops an odd server sending more.
    pub dimension: i8,
    /// `0`=peaceful .. `3`=hard.
    pub difficulty: u8,
    /// The server's configured player cap (informational — vanilla doesn't clamp on join).
    pub max_players: u8,
    /// The canonical level-type name (`"default"`, `"flat"`, `"largeBiomes"`, `"amplified"`,
    /// ...). Java resolves this through `WorldType.parseWorldType` case-insensitively and falls
    /// back to `WorldType.DEFAULT` for an unknown name before the packet reaches its handler.
    pub level_type: String,
    /// Whether the F3 debug screen should hide coordinates/seed (a "reduced info" server
    /// policy).
    pub reduced_debug_info: bool,
}

impl Packet for JoinGame {
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        let raw_game_mode = buf.read_u8()?;
        let hardcore_mode = raw_game_mode & 8 == 8;
        let game_mode = raw_game_mode & !8;
        let dimension = buf.read_i8()?;
        let difficulty = buf.read_u8()?;
        let max_players = buf.read_u8()?;
        let level_type = canonical_level_type_name(&buf.read_string(16)?).to_owned();
        let reduced_debug_info = buf.read_bool()?;
        Ok(JoinGame {
            entity_id,
            hardcore_mode,
            game_mode,
            dimension,
            difficulty,
            max_players,
            level_type,
            reduced_debug_info,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        let mut raw_game_mode = self.game_mode;
        if self.hardcore_mode {
            raw_game_mode |= 8;
        }
        buf.write_u8(raw_game_mode);
        buf.write_i8(self.dimension);
        buf.write_u8(self.difficulty);
        buf.write_u8(self.max_players);
        buf.write_string(canonical_level_type_name(&self.level_type));
        buf.write_bool(self.reduced_debug_info);
    }
}

/// `0x02 Chat Message`: a chat-component JSON payload plus a display-position hint.
/// java: `S02PacketChat`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chat {
    /// The message, as chat-component JSON (unparsed — see `PacketBuffer::read_chat`).
    pub component: String,
    /// `0`=chat box, `1`=system message (still chat box), `2`=above the hotbar (action bar).
    /// Vanilla stores this as a raw byte with no validation, so this crate does too.
    pub position: i8,
}

impl Packet for Chat {
    const ID: i32 = 0x02;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let component = buf.read_chat()?;
        let position = buf.read_i8()?;
        Ok(Chat {
            component,
            position,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_chat(&self.component);
        buf.write_i8(self.position);
    }
}

/// `0x03 Time Update`. java: `S03PacketTimeUpdate`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeUpdate {
    /// Total ticks since the world was created (used for lightning/moon phase, never resets).
    pub total_world_time: i64,
    /// Absolute world time in ticks (not reduced modulo a day). When `doDaylightCycle=false`,
    /// vanilla sends the negated value; a negative nonzero value signals a frozen day/night
    /// cycle. Wire value `0` cannot distinguish frozen-zero from active-zero and is treated as
    /// active.
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

/// `0x04 Entity Equipment`. java: `S04PacketEntityEquipment`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityEquipment {
    /// The entity to update.
    pub entity_id: i32,
    /// `0`=held item, `1..=4`=boots/leggings/chestplate/helmet.
    pub equipment_slot: i16,
    /// The item now in that slot, or `None` for an empty slot.
    pub item: Option<crate::ItemStack>,
}

impl Packet for EntityEquipment {
    const ID: i32 = 0x04;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let equipment_slot = buf.read_i16()?;
        let item = buf.read_item_stack()?;
        Ok(EntityEquipment {
            entity_id,
            equipment_slot,
            item,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i16(self.equipment_slot);
        buf.write_item_stack(self.item.as_ref());
    }
}

/// `0x05 Spawn Position`: the world's compass/respawn anchor point. java: `S05PacketSpawnPosition`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnPosition {
    /// The world spawn block.
    pub position: mc_math::BlockPos,
}

impl Packet for SpawnPosition {
    const ID: i32 = 0x05;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SpawnPosition {
            position: buf.read_block_pos()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
    }
}

/// `0x06 Update Health`. java: `S06PacketUpdateHealth`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdateHealth {
    /// `0.0` means dead; vanilla's max is `20.0` but nothing on the wire enforces that.
    pub health: f32,
    /// `0..=20`.
    pub food_level: i32,
    /// `0.0..=5.0`, drives when natural regen consumes food.
    pub saturation_level: f32,
}

impl Packet for UpdateHealth {
    const ID: i32 = 0x06;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(UpdateHealth {
            health: buf.read_f32()?,
            food_level: buf.read_varint()?,
            saturation_level: buf.read_f32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.health);
        buf.write_varint(self.food_level);
        buf.write_f32(self.saturation_level);
    }
}

/// `0x07 Respawn`: dimension change (including the "respawn after death" case).
/// java: `S07PacketRespawn`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Respawn {
    /// A full `i32` on the wire here — unlike [`JoinGame::dimension`]'s single byte.
    pub dimension: i32,
    /// `0..=3`.
    pub difficulty: u8,
    /// `0`=survival .. `3`=spectator (no separate hardcore bit here, unlike `JoinGame`).
    pub game_mode: u8,
    /// The canonical level-type name, same normalization as [`JoinGame::level_type`].
    pub level_type: String,
}

impl Packet for Respawn {
    const ID: i32 = 0x07;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Respawn {
            dimension: buf.read_i32()?,
            difficulty: buf.read_u8()?,
            game_mode: buf.read_u8()?,
            level_type: canonical_level_type_name(&buf.read_string(16)?).to_owned(),
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.dimension);
        buf.write_u8(self.difficulty);
        buf.write_u8(self.game_mode);
        buf.write_string(canonical_level_type_name(&self.level_type));
    }
}

/// `0x08 Player Position And Look` (clientbound): an authoritative correction the client must
/// apply, absolutely or relative-to-current-position per bit in `flags`.
/// java: `S08PacketPlayerPosLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPositionLook {
    /// Absolute position or a relative delta — see `flags` bit 0.
    pub x: f64,
    /// Absolute position or a relative delta — see `flags` bit 1.
    pub y: f64,
    /// Absolute position or a relative delta — see `flags` bit 2.
    pub z: f64,
    /// Absolute angle or a relative delta — see `flags` bit 3.
    pub yaw: f32,
    /// Absolute angle or a relative delta — see `flags` bit 4.
    pub pitch: f32,
    /// Bit `0`=X relative, `1`=Y relative, `2`=Z relative, `3`=yaw relative, `4`=pitch relative;
    /// an unset bit means that field is absolute. java: `S08PacketPlayerPosLook.EnumFlags`
    pub flags: u8,
}

impl Packet for PlayerPositionLook {
    const ID: i32 = 0x08;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerPositionLook {
            x: buf.read_f64()?,
            y: buf.read_f64()?,
            z: buf.read_f64()?,
            yaw: buf.read_f32()?,
            pitch: buf.read_f32()?,
            flags: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f64(self.x);
        buf.write_f64(self.y);
        buf.write_f64(self.z);
        buf.write_f32(self.yaw);
        buf.write_f32(self.pitch);
        buf.write_u8(self.flags);
    }
}

/// `0x09 Held Item Change` (clientbound): the server forcing the client's selected hotbar slot.
/// java: `S09PacketHeldItemChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeldItemChange {
    /// `0..=8`.
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

/// `0x0A Use Bed`: a player entered a bed (drives the "sleeping" render pose for other
/// clients). java: `S0APacketUseBed`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UseBed {
    /// The sleeping player's entity id.
    pub player_id: i32,
    /// The bed's head-block position.
    pub bed_position: mc_math::BlockPos,
}

impl Packet for UseBed {
    const ID: i32 = 0x0A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(UseBed {
            player_id: buf.read_varint()?,
            bed_position: buf.read_block_pos()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.player_id);
        buf.write_block_pos(self.bed_position);
    }
}

/// `0x0B Animation` (clientbound): a one-shot visual animation on an entity (arm swing, hurt
/// flash, wake up, eat, critical-hit particles). java: `S0BPacketAnimation`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Animation {
    /// The animating entity.
    pub entity_id: i32,
    /// `0`=swing arm, `1`=hurt, `2`=leave bed, `3`=eat, `4`=critical hit, `5`=magic critical
    /// hit. Vanilla stores this as a raw byte with no validation.
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

/// `0x0C Spawn Player`: a remote player entered view range.
/// java: `S0CPacketSpawnPlayer`
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnPlayer {
    /// The new entity's id (arena index on the wire, not the `mc-world` ECS index).
    pub entity_id: i32,
    /// The player's account UUID (packed two-`i64` form, the *normal* play-protocol
    /// encoding — unlike `LoginSuccess`'s dashed-string UUID).
    pub player_uuid: Uuid,
    /// Fixed-point: the true world X is `x as f64 / 32.0`.
    pub x: i32,
    /// Fixed-point: the true world Y is `y as f64 / 32.0`.
    pub y: i32,
    /// Fixed-point: the true world Z is `z as f64 / 32.0`.
    pub z: i32,
    /// Body yaw.
    pub yaw: Angle,
    /// Body pitch.
    pub pitch: Angle,
    /// The held item's registry id, or `0` for empty hand.
    pub current_item: i16,
    /// Initial entity-metadata snapshot (skin layers, name-tag visibility, etc).
    pub metadata: Metadata,
}

impl Packet for SpawnPlayer {
    const ID: i32 = 0x0C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let player_uuid = buf.read_uuid()?;
        let x = buf.read_i32()?;
        let y = buf.read_i32()?;
        let z = buf.read_i32()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let current_item = buf.read_i16()?;
        let metadata = metadata::read(buf)?;
        Ok(SpawnPlayer {
            entity_id,
            player_uuid,
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
        buf.write_uuid(self.player_uuid);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_i16(self.current_item);
        metadata::write(&self.metadata, buf);
    }
}

/// `0x0D Collect Item`: an item/arrow/XP-orb entity was picked up (drives the "fly toward
/// collector" animation). java: `S0DPacketCollectItem`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectItem {
    /// The entity being picked up (removed shortly after on a `DestroyEntities`).
    pub collected_entity_id: i32,
    /// The collecting entity.
    pub collector_entity_id: i32,
}

impl Packet for CollectItem {
    const ID: i32 = 0x0D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(CollectItem {
            collected_entity_id: buf.read_varint()?,
            collector_entity_id: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.collected_entity_id);
        buf.write_varint(self.collector_entity_id);
    }
}

/// `0x0E Spawn Object`: a non-living entity (item frame, arrow, boat, minecart, TNT, falling
/// block, ...). java: `S0EPacketSpawnObject`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnObject {
    /// The new entity's id.
    pub entity_id: i32,
    /// The object type id (`Entity.getEntityId`-equivalent for objects, a separate table from
    /// `SpawnMob`'s entity-type ids — resolving it is `mc-data`'s job).
    pub object_type: i8,
    /// Fixed-point: true X is `x as f64 / 32.0`.
    pub x: i32,
    /// Fixed-point: true Y is `y as f64 / 32.0`.
    pub y: i32,
    /// Fixed-point: true Z is `z as f64 / 32.0`.
    pub z: i32,
    /// Pitch.
    pub pitch: Angle,
    /// Yaw.
    pub yaw: Angle,
    /// Type-specific auxiliary data (e.g. block-state id for a falling block); `> 0` is the
    /// signal that `velocity` follows on the wire.
    pub data: i32,
    /// Present only when `data > 0`; each axis is a fixed-point velocity (true units/tick is
    /// `component as f64 / 8000.0`).
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
            let (vx, vy, vz) = self.velocity.unwrap_or((0, 0, 0));
            buf.write_i16(vx);
            buf.write_i16(vy);
            buf.write_i16(vz);
        }
    }
}

/// `0x0F Spawn Mob`: a living entity (any mob, or another player's `EntityLivingBase` seen as
/// a mob — actual players use `SpawnPlayer` instead). java: `S0FPacketSpawnMob`
#[derive(Debug, Clone, PartialEq)]
pub struct SpawnMob {
    /// The new entity's id.
    pub entity_id: i32,
    /// `EntityList` network id — resolving it to a species is `mc-data`'s job.
    pub entity_type: u8,
    /// Fixed-point: true X is `x as f64 / 32.0`.
    pub x: i32,
    /// Fixed-point: true Y is `y as f64 / 32.0`.
    pub y: i32,
    /// Fixed-point: true Z is `z as f64 / 32.0`.
    pub z: i32,
    /// Body yaw.
    pub yaw: Angle,
    /// Body pitch.
    pub pitch: Angle,
    /// Head yaw (independent of body yaw — e.g. a zombie looking sideways while walking
    /// straight).
    pub head_pitch: Angle,
    /// Fixed-point velocity, true units/tick is `component as f64 / 8000.0`.
    pub velocity_x: i16,
    /// Fixed-point velocity Y.
    pub velocity_y: i16,
    /// Fixed-point velocity Z.
    pub velocity_z: i16,
    /// Initial entity-metadata snapshot.
    pub metadata: Metadata,
}

impl Packet for SpawnMob {
    const ID: i32 = 0x0F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let entity_type = buf.read_u8()?;
        let x = buf.read_i32()?;
        let y = buf.read_i32()?;
        let z = buf.read_i32()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let head_pitch = buf.read_angle()?;
        let velocity_x = buf.read_i16()?;
        let velocity_y = buf.read_i16()?;
        let velocity_z = buf.read_i16()?;
        let metadata = metadata::read(buf)?;
        Ok(SpawnMob {
            entity_id,
            entity_type,
            x,
            y,
            z,
            yaw,
            pitch,
            head_pitch,
            velocity_x,
            velocity_y,
            velocity_z,
            metadata,
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
        buf.write_angle(self.head_pitch);
        buf.write_i16(self.velocity_x);
        buf.write_i16(self.velocity_y);
        buf.write_i16(self.velocity_z);
        metadata::write(&self.metadata, buf);
    }
}

/// `0x10 Spawn Painting`. java: `S10PacketSpawnPainting`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpawnPainting {
    /// The new entity's id.
    pub entity_id: i32,
    /// The painting's art name (max 13 ASCII chars — `"SkullAndRoses"` is vanilla's longest).
    pub title: String,
    /// The block position the painting is mounted against.
    pub position: mc_math::BlockPos,
    /// Which wall the painting hangs on (only the four horizontal directions are legal; a
    /// decoded `Down`/`Up` would mean a malformed packet, not a real vanilla one).
    pub facing: Facing,
}

impl Packet for SpawnPainting {
    const ID: i32 = 0x10;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let title = buf.read_string(13)?;
        let position = buf.read_block_pos()?;
        let facing = Facing::from_horizontal(i32::from(buf.read_u8()?));
        Ok(SpawnPainting {
            entity_id,
            title,
            position,
            facing,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_string(&self.title);
        buf.write_block_pos(self.position);
        buf.write_u8(self.facing.horizontal_index().unwrap_or(0));
    }
}

/// `0x11 Spawn Experience Orb`. java: `S11PacketSpawnExperienceOrb`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnExperienceOrb {
    /// The new entity's id.
    pub entity_id: i32,
    /// Fixed-point: true X is `x as f64 / 32.0`.
    pub x: i32,
    /// Fixed-point: true Y is `y as f64 / 32.0`.
    pub y: i32,
    /// Fixed-point: true Z is `z as f64 / 32.0`.
    pub z: i32,
    /// The XP value this orb carries.
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

/// `0x12 Entity Velocity`. java: `S12PacketEntityVelocity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityVelocity {
    /// The entity to update.
    pub entity_id: i32,
    /// Fixed-point, true units/tick is `component as f64 / 8000.0`, clamped by the server to
    /// ±3.9 before scaling.
    pub motion_x: i16,
    /// Fixed-point velocity Y.
    pub motion_y: i16,
    /// Fixed-point velocity Z.
    pub motion_z: i16,
}

impl Packet for EntityVelocity {
    const ID: i32 = 0x12;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityVelocity {
            entity_id: buf.read_varint()?,
            motion_x: buf.read_i16()?,
            motion_y: buf.read_i16()?,
            motion_z: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i16(self.motion_x);
        buf.write_i16(self.motion_y);
        buf.write_i16(self.motion_z);
    }
}

/// `0x13 Destroy Entities`: entities to remove from the client's world. java:
/// `S13PacketDestroyEntities`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestroyEntities {
    /// The entity ids to despawn.
    pub entity_ids: Vec<i32>,
}

impl Packet for DestroyEntities {
    const ID: i32 = 0x13;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_varint()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        // Don't pre-allocate the full (attacker-controlled) count up front — grow as bytes
        // actually arrive, same defensive shape as `chunk::MapChunkBulk`'s column list.
        let mut entity_ids = Vec::with_capacity((count as usize).min(1024));
        for _ in 0..count {
            entity_ids.push(buf.read_varint()?);
        }
        Ok(DestroyEntities { entity_ids })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_ids.len() as i32);
        for &id in &self.entity_ids {
            buf.write_varint(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<P: Packet + PartialEq + std::fmt::Debug>(pkt: P) {
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(P::read(&mut read_buf).unwrap(), pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn keep_alive_round_trips() {
        round_trip(KeepAlive { id: 123_456 });
    }

    #[test]
    fn join_game_round_trips_and_packs_hardcore_bit() {
        let pkt = JoinGame {
            entity_id: 42,
            hardcore_mode: true,
            game_mode: 1,
            dimension: -1,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        // The hardcore bit and game mode share one byte at a known offset (after the i32 id).
        assert_eq!(buf.as_slice()[4], 0b0000_1001); // game_mode=1 | hardcore(8)
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(JoinGame::read(&mut read_buf).unwrap(), pkt);
    }

    fn raw_join_game(level_type: &str) -> PacketBuffer {
        let mut buf = PacketBuffer::new();
        buf.write_i32(42);
        buf.write_u8(0b0000_1001);
        buf.write_i8(-1);
        buf.write_u8(2);
        buf.write_u8(20);
        buf.write_string(level_type);
        buf.write_bool(false);
        buf
    }

    #[test]
    fn join_game_world_type_decode_normalizes_java_registry_names() {
        // WorldType.parseWorldType uses equalsIgnoreCase, while S01 replaces a null result with
        // WorldType.DEFAULT. Its writer then emits getWorldTypeName(), i.e. these exact spellings.
        for (wire_name, canonical_name) in [
            ("DEFAULT", "default"),
            ("fLaT", "flat"),
            ("LARGEBIOMES", "largeBiomes"),
            ("AmPlIfIeD", "amplified"),
            ("CUSTOMIZED", "customized"),
            ("DEFAULT_1_1", "default_1_1"),
            ("not_real_type", "default"),
        ] {
            let mut input = PacketBuffer::from_slice(raw_join_game(wire_name).as_slice());
            let packet = JoinGame::read(&mut input).unwrap();
            assert_eq!(packet.level_type, canonical_name, "wire name: {wire_name}");

            let mut output = PacketBuffer::new();
            packet.write(&mut output);
            assert_eq!(output.as_slice(), raw_join_game(canonical_name).as_slice());
        }
    }

    #[test]
    fn chat_round_trips() {
        round_trip(Chat {
            component: r#"{"text":"hi"}"#.into(),
            position: 2,
        });
    }

    #[test]
    fn time_update_round_trips_negative_frozen_time() {
        round_trip(TimeUpdate {
            total_world_time: 123_456,
            world_time: -6000,
        });
    }

    #[test]
    fn entity_equipment_round_trips_empty_and_full() {
        round_trip(EntityEquipment {
            entity_id: 5,
            equipment_slot: 0,
            item: None,
        });
        round_trip(EntityEquipment {
            entity_id: 5,
            equipment_slot: 4,
            item: Some(crate::ItemStack {
                item_id: 298,
                count: 1,
                damage: 0,
                nbt: None,
            }),
        });
    }

    #[test]
    fn spawn_position_round_trips_negative_coords() {
        round_trip(SpawnPosition {
            position: mc_math::BlockPos::new(-100, 64, -200),
        });
    }

    #[test]
    fn update_health_round_trips() {
        round_trip(UpdateHealth {
            health: 10.0,
            food_level: 18,
            saturation_level: 3.5,
        });
    }

    #[test]
    fn respawn_round_trips() {
        round_trip(Respawn {
            dimension: -1,
            difficulty: 1,
            game_mode: 0,
            level_type: "default".into(),
        });
    }

    fn raw_respawn(level_type: &str) -> PacketBuffer {
        let mut buf = PacketBuffer::new();
        buf.write_i32(-1);
        buf.write_u8(2);
        buf.write_u8(1);
        buf.write_string(level_type);
        buf
    }

    #[test]
    fn respawn_world_type_decode_normalizes_unknown_and_mixed_case() {
        for (wire_name, canonical_name) in [
            ("LaRgEbIoMeS", "largeBiomes"),
            ("AMPLIFIED", "amplified"),
            ("unknown_type", "default"),
        ] {
            let mut input = PacketBuffer::from_slice(raw_respawn(wire_name).as_slice());
            let packet = Respawn::read(&mut input).unwrap();
            assert_eq!(packet.level_type, canonical_name, "wire name: {wire_name}");

            let mut output = PacketBuffer::new();
            packet.write(&mut output);
            assert_eq!(output.as_slice(), raw_respawn(canonical_name).as_slice());
        }
    }

    #[test]
    fn player_position_look_round_trips() {
        round_trip(PlayerPositionLook {
            x: 8.5,
            y: 65.0,
            z: -12.25,
            yaw: 90.0,
            pitch: 0.0,
            flags: 0b0001_1111,
        });
    }

    #[test]
    fn held_item_change_round_trips() {
        round_trip(HeldItemChange { slot: 8 });
    }

    #[test]
    fn use_bed_round_trips() {
        round_trip(UseBed {
            player_id: 7,
            bed_position: mc_math::BlockPos::new(1, 4, 1),
        });
    }

    #[test]
    fn animation_round_trips() {
        round_trip(Animation {
            entity_id: 9,
            animation_type: 0,
        });
    }

    #[test]
    fn spawn_player_round_trips_with_metadata() {
        round_trip(SpawnPlayer {
            entity_id: 55,
            player_uuid: Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0),
            x: 32 * 10,
            y: 32 * 70,
            z: 32 * -5,
            yaw: Angle(64),
            pitch: Angle(0),
            current_item: 0,
            metadata: vec![crate::metadata::MetadataEntry {
                index: 0,
                value: crate::metadata::MetadataValue::Byte(0),
            }],
        });
    }

    #[test]
    fn collect_item_round_trips() {
        round_trip(CollectItem {
            collected_entity_id: 3,
            collector_entity_id: 4,
        });
    }

    #[test]
    fn spawn_object_round_trips_with_and_without_velocity() {
        round_trip(SpawnObject {
            entity_id: 10,
            object_type: 1,
            x: 100,
            y: 200,
            z: -300,
            pitch: Angle(0),
            yaw: Angle(128),
            data: 0,
            velocity: None,
        });
        round_trip(SpawnObject {
            entity_id: 11,
            object_type: 50,
            x: 0,
            y: 0,
            z: 0,
            pitch: Angle(0),
            yaw: Angle(0),
            data: 1,
            velocity: Some((100, -100, 200)),
        });
    }

    #[test]
    fn spawn_mob_round_trips() {
        round_trip(SpawnMob {
            entity_id: 20,
            entity_type: 90, // zombie
            x: 320,
            y: 2048,
            z: -320,
            yaw: Angle(10),
            pitch: Angle(20),
            head_pitch: Angle(30),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Metadata::new(),
        });
    }

    #[test]
    fn spawn_painting_round_trips() {
        round_trip(SpawnPainting {
            entity_id: 30,
            title: "Aztec".into(),
            position: mc_math::BlockPos::new(5, 65, 5),
            facing: Facing::North,
        });
    }

    #[test]
    fn spawn_painting_preserves_unknown_title_for_the_client_constructor() {
        round_trip(SpawnPainting {
            entity_id: 31,
            title: "NotAPainting".into(),
            position: mc_math::BlockPos::new(-5, 65, 5),
            facing: Facing::South,
        });
    }

    #[test]
    fn spawn_experience_orb_round_trips() {
        round_trip(SpawnExperienceOrb {
            entity_id: 40,
            x: 10,
            y: 20,
            z: 30,
            xp_value: 5,
        });
    }

    #[test]
    fn entity_velocity_round_trips() {
        round_trip(EntityVelocity {
            entity_id: 12,
            motion_x: 100,
            motion_y: -50,
            motion_z: 0,
        });
    }

    #[test]
    fn destroy_entities_round_trips_including_empty() {
        round_trip(DestroyEntities { entity_ids: vec![] });
        round_trip(DestroyEntities {
            entity_ids: vec![1, 2, 3, 999_999],
        });
    }

    #[test]
    fn destroy_entities_rejects_negative_count_without_panicking() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(DestroyEntities::read(&mut read_buf).is_err());
    }

    #[test]
    fn destroy_entities_huge_claimed_count_errors_instead_of_allocating() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(i32::MAX); // no payload follows
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(DestroyEntities::read(&mut read_buf).is_err());
    }
}
