//! Clientbound Play, ids `0x14`-`0x27` (minus `0x21`/`0x26`, in [`crate::chunk`]):
//! `Entity` .. `Explosion`.
//!
//! java: `network/play/server/S14PacketEntity.java` through `S27PacketExplosion.java`. One
//! file, `S14PacketEntity.java`, backs four ids: the `S14PacketEntity` base class (`0x14`) plus
//! three nested static subclasses, `S15PacketEntityRelMove` (`0x15`), `S16PacketEntityLook`
//! (`0x16`), and `S17PacketEntityLookMove` (`0x17`) — modelled here as four independent structs
//! rather than a Rust inheritance hierarchy Rust has no use for. The crate-level doc's `S19`
//! naming quirk also lives in this batch: the *file* `S19PacketEntityHeadLook.java` registers at
//! `0x19`, while the *file* `S19PacketEntityStatus.java` registers at `0x1A`.

use crate::metadata::{self, Metadata};
use crate::packet::Packet;
use crate::{Angle, PacketBuffer, ProtoError};
use mc_math::BlockPos;
use uuid::Uuid;

// ---- 0x14-0x17: S14PacketEntity and its nested subclasses ---------------------------------

/// `0x14 Entity`: a no-op position/look keep-alive — sent for an entity that is still being
/// tracked but hasn't moved since the last update. `S15`/`S16`/`S17` below all read this same
/// `entity_id` field first (`super.readPacketData(buf)`), then append their own fields.
/// java: `S14PacketEntity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entity {
    /// The entity id.
    pub entity_id: i32,
}

impl Packet for Entity {
    const ID: i32 = 0x14;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Entity {
            entity_id: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
    }
}

/// `0x15 Entity Relative Move`: a relative position delta, each axis a 1/32-block fixed-point
/// signed byte (so at most ~4 blocks per packet per axis — vanilla falls back to
/// [`crate::play::EntityTeleport`] for larger jumps).
/// java: `S14PacketEntity.S15PacketEntityRelMove`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityRelMove {
    /// The entity id.
    pub entity_id: i32,
    /// X delta, in 1/32nds of a block.
    pub delta_x: i8,
    /// Y delta, in 1/32nds of a block.
    pub delta_y: i8,
    /// Z delta, in 1/32nds of a block.
    pub delta_z: i8,
    /// Whether the entity is on the ground.
    pub on_ground: bool,
}

impl Packet for EntityRelMove {
    const ID: i32 = 0x15;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let delta_x = buf.read_i8()?;
        let delta_y = buf.read_i8()?;
        let delta_z = buf.read_i8()?;
        let on_ground = buf.read_bool()?;
        Ok(EntityRelMove {
            entity_id,
            delta_x,
            delta_y,
            delta_z,
            on_ground,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.delta_x);
        buf.write_i8(self.delta_y);
        buf.write_i8(self.delta_z);
        buf.write_bool(self.on_ground);
    }
}

/// `0x16 Entity Look`: a new head/body yaw and pitch, no position change.
/// java: `S14PacketEntity.S16PacketEntityLook`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLook {
    /// The entity id.
    pub entity_id: i32,
    /// New yaw.
    pub yaw: Angle,
    /// New pitch.
    pub pitch: Angle,
    /// Whether the entity is on the ground.
    pub on_ground: bool,
}

impl Packet for EntityLook {
    const ID: i32 = 0x16;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let on_ground = buf.read_bool()?;
        Ok(EntityLook {
            entity_id,
            yaw,
            pitch,
            on_ground,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x17 Entity Look and Relative Move`: [`EntityRelMove`] and [`EntityLook`] combined.
/// java: `S14PacketEntity.S17PacketEntityLookMove`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLookAndRelMove {
    /// The entity id.
    pub entity_id: i32,
    /// X delta, in 1/32nds of a block.
    pub delta_x: i8,
    /// Y delta, in 1/32nds of a block.
    pub delta_y: i8,
    /// Z delta, in 1/32nds of a block.
    pub delta_z: i8,
    /// New yaw.
    pub yaw: Angle,
    /// New pitch.
    pub pitch: Angle,
    /// Whether the entity is on the ground.
    pub on_ground: bool,
}

impl Packet for EntityLookAndRelMove {
    const ID: i32 = 0x17;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let delta_x = buf.read_i8()?;
        let delta_y = buf.read_i8()?;
        let delta_z = buf.read_i8()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let on_ground = buf.read_bool()?;
        Ok(EntityLookAndRelMove {
            entity_id,
            delta_x,
            delta_y,
            delta_z,
            yaw,
            pitch,
            on_ground,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.delta_x);
        buf.write_i8(self.delta_y);
        buf.write_i8(self.delta_z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

// ---- 0x18-0x1A -------------------------------------------------------------------------------

/// `0x18 Entity Teleport`: an absolute position snap (a big jump, a dimension change follow-up,
/// or periodic resync). Position is ×32 fixed-point, exactly like [`crate::play::SpawnPlayer`]
/// — **not** the packed [`BlockPos`]/`Position` wire type, since this is an entity's continuous
/// position, not a block coordinate.
/// java: `S18PacketEntityTeleport`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityTeleport {
    /// The entity id.
    pub entity_id: i32,
    /// X position, ×32 fixed-point (`floor(x * 32.0)`).
    pub x: i32,
    /// Y position, ×32 fixed-point.
    pub y: i32,
    /// Z position, ×32 fixed-point.
    pub z: i32,
    /// New yaw.
    pub yaw: Angle,
    /// New pitch.
    pub pitch: Angle,
    /// Whether the entity is on the ground.
    pub on_ground: bool,
}

impl Packet for EntityTeleport {
    const ID: i32 = 0x18;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let x = buf.read_i32()?;
        let y = buf.read_i32()?;
        let z = buf.read_i32()?;
        let yaw = buf.read_angle()?;
        let pitch = buf.read_angle()?;
        let on_ground = buf.read_bool()?;
        Ok(EntityTeleport {
            entity_id,
            x,
            y,
            z,
            yaw,
            pitch,
            on_ground,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i32(self.x);
        buf.write_i32(self.y);
        buf.write_i32(self.z);
        buf.write_angle(self.yaw);
        buf.write_angle(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x19 Entity Head Look`: the head yaw alone (separate from body yaw, e.g. a player looking
/// around while standing still). java: `S19PacketEntityHeadLook` — this *file*, despite its
/// `S19` name, registers at `0x19` (see the crate-level doc for the naming quirk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityHeadLook {
    /// The entity id.
    pub entity_id: i32,
    /// New head yaw.
    pub yaw: Angle,
}

impl Packet for EntityHeadLook {
    const ID: i32 = 0x19;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let yaw = buf.read_angle()?;
        Ok(EntityHeadLook { entity_id, yaw })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_angle(self.yaw);
    }
}

/// `0x1A Entity Status`: a one-byte, entity-type-dependent event code (hurt animation, death,
/// wolf shaking off water, iron golem's rose offer, ...) — `opcode`'s meaning is decided by the
/// receiving entity's class, so it stays a plain integer rather than one closed Rust enum.
///
/// Unlike almost every other entity packet in this batch, `entity_id` here is a **plain
/// big-endian `i32`** (`readInt`), not a VarInt — an inconsistency inherited directly from
/// vanilla's `readPacketData`, not a transcription error.
///
/// java: `S19PacketEntityStatus` — this *file*, despite its `S19` name, registers at `0x1A`
/// (see the crate-level doc for the naming quirk).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityStatus {
    /// The entity id, sent as a plain `i32` (not a VarInt).
    pub entity_id: i32,
    /// The entity-type-dependent status/event opcode.
    pub opcode: i8,
}

impl Packet for EntityStatus {
    const ID: i32 = 0x1A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        let opcode = buf.read_i8()?;
        Ok(EntityStatus { entity_id, opcode })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.opcode);
    }
}

// ---- 0x1B EntityAttach -------------------------------------------------------------------

/// What kind of attachment [`EntityAttach`] describes — vanilla's `leash` field, read as
/// `readUnsignedByte()`. `NetHandlerPlayClient.handleEntityAttach` acts only on values `0` and
/// `1`; every other byte falls through without disconnecting.
/// java: `S1BPacketEntityAttach` (the `leash` field/constructor parameter)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachType {
    /// `vehicle_entity_id` is a vehicle to mount (or `-1` to dismount whatever it's currently
    /// riding).
    Vehicle,
    /// `vehicle_entity_id` is a leash holder to attach to (or `-1` to detach an existing leash).
    Leash,
    /// A wire value vanilla's handler ignores without disconnecting.
    Unknown(u8),
}

impl AttachType {
    fn from_id(id: u8) -> AttachType {
        match id {
            0 => AttachType::Vehicle,
            1 => AttachType::Leash,
            other => AttachType::Unknown(other),
        }
    }

    fn to_id(self) -> u8 {
        match self {
            AttachType::Vehicle => 0,
            AttachType::Leash => 1,
            AttachType::Unknown(id) => id,
        }
    }
}

/// `0x1B Entity Attach`: mounts/dismounts an entity on a vehicle, or attaches/detaches a leash.
///
/// Both id fields are plain big-endian `i32`s (`readInt`), not VarInts — like [`EntityStatus`],
/// this is what vanilla's `readPacketData` actually does, not a transcription slip.
/// java: `S1BPacketEntityAttach`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityAttach {
    /// The entity being attached/detached.
    pub entity_id: i32,
    /// The vehicle or leash-holder entity id, or `-1` to dismount/detach.
    pub vehicle_entity_id: i32,
    /// Whether this is a vehicle mount, leash attach, or a byte vanilla ignores.
    pub attach_type: AttachType,
}

impl Packet for EntityAttach {
    const ID: i32 = 0x1B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        let vehicle_entity_id = buf.read_i32()?;
        let attach_type = AttachType::from_id(buf.read_u8()?);
        Ok(EntityAttach {
            entity_id,
            vehicle_entity_id,
            attach_type,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i32(self.vehicle_entity_id);
        buf.write_u8(self.attach_type.to_id());
    }
}

// ---- 0x1C EntityMetadata -----------------------------------------------------------------

/// `0x1C Entity Metadata`: a `DataWatcher` snapshot — either every watched slot, or just the
/// ones that changed, depending on why vanilla sent it; the wire form doesn't distinguish.
/// java: `S1CPacketEntityMetadata`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityMetadata {
    /// The entity id.
    pub entity_id: i32,
    /// The watched-object list. See [`crate::metadata`].
    pub metadata: Metadata,
}

impl Packet for EntityMetadata {
    const ID: i32 = 0x1C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let metadata = metadata::read(buf)?;
        Ok(EntityMetadata {
            entity_id,
            metadata,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        metadata::write(&self.metadata, buf);
    }
}

// ---- 0x1D-0x1F ----------------------------------------------------------------------------

/// `0x1D Entity Effect`: applies (or refreshes) a potion effect on an entity.
/// java: `S1DPacketEntityEffect`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityEffect {
    /// The entity id.
    pub entity_id: i32,
    /// The potion effect's registry id (`Potion` ids index `mc-data`, which this crate does not
    /// depend on).
    pub effect_id: i8,
    /// Effect amplifier (`0` = level I).
    pub amplifier: i8,
    /// Remaining duration, in ticks. Vanilla's sender caps this at `32767` before encoding, but
    /// that cap isn't enforced on decode — any VarInt value round-trips.
    pub duration: i32,
    /// Whether the effect's particles should be shown. Wire: a single byte where vanilla treats
    /// any nonzero value as `showParticles = true`
    /// (`S1DPacketEntityEffect.func_179707_f`); the sender only ever writes `0`/`1`.
    pub show_particles: bool,
}

impl Packet for EntityEffect {
    const ID: i32 = 0x1D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let effect_id = buf.read_i8()?;
        let amplifier = buf.read_i8()?;
        let duration = buf.read_varint()?;
        let show_particles = buf.read_bool()?;
        Ok(EntityEffect {
            entity_id,
            effect_id,
            amplifier,
            duration,
            show_particles,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.effect_id);
        buf.write_i8(self.amplifier);
        buf.write_varint(self.duration);
        buf.write_bool(self.show_particles);
    }
}

/// `0x1E Remove Entity Effect`: clears one potion effect from an entity.
/// java: `S1EPacketRemoveEntityEffect`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RemoveEntityEffect {
    /// The entity id.
    pub entity_id: i32,
    /// The potion effect's registry id to remove.
    pub effect_id: u8,
}

impl Packet for RemoveEntityEffect {
    const ID: i32 = 0x1E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let effect_id = buf.read_u8()?;
        Ok(RemoveEntityEffect {
            entity_id,
            effect_id,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_u8(self.effect_id);
    }
}

/// `0x1F Set Experience`: updates the client's experience bar and level (always about the local
/// player; there is no `entity_id` field).
/// java: `S1FPacketSetExperience`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SetExperience {
    /// Progress toward the next level, in `[0.0, 1.0)`.
    pub experience_bar: f32,
    /// Current level.
    pub level: i32,
    /// Total accumulated experience points.
    pub total_experience: i32,
}

impl Packet for SetExperience {
    const ID: i32 = 0x1F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let experience_bar = buf.read_f32()?;
        let level = buf.read_varint()?;
        let total_experience = buf.read_varint()?;
        Ok(SetExperience {
            experience_bar,
            level,
            total_experience,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.experience_bar);
        buf.write_varint(self.level);
        buf.write_varint(self.total_experience);
    }
}

// ---- 0x20 EntityProperties ----------------------------------------------------------------

/// An attribute modifier's combine operation (`AttributeModifier.Operation`'s three legal
/// values). java: `AttributeModifier` (the `operation` field, written/read as a single byte)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeModifierOperation {
    /// Add the amount directly to the base value.
    Add,
    /// Add `amount * base value` (modifiers with this operation are summed before applying).
    MultiplyBase,
    /// Multiply the running total by `1 + amount`.
    Multiply,
}

impl AttributeModifierOperation {
    fn from_id(id: i32) -> Result<AttributeModifierOperation, ProtoError> {
        match id {
            0 => Ok(AttributeModifierOperation::Add),
            1 => Ok(AttributeModifierOperation::MultiplyBase),
            2 => Ok(AttributeModifierOperation::Multiply),
            other => Err(ProtoError::InvalidEnum {
                what: "attribute modifier operation",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            AttributeModifierOperation::Add => 0,
            AttributeModifierOperation::MultiplyBase => 1,
            AttributeModifierOperation::Multiply => 2,
        }
    }
}

/// One attribute modifier within an [`AttributePropertySnapshot`].
/// java: `S20PacketEntityProperties.readPacketData` (the inner modifier loop)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttributeModifierSnapshot {
    /// The modifier's unique id.
    pub uuid: Uuid,
    /// The modifier's amount (interpretation depends on `operation`).
    pub amount: f64,
    /// How this modifier combines with the base value and other modifiers.
    pub operation: AttributeModifierOperation,
}

/// One attribute's current value plus any active modifiers.
/// java: `S20PacketEntityProperties.Snapshot`
#[derive(Debug, Clone, PartialEq)]
pub struct AttributePropertySnapshot {
    /// The attribute's unlocalized name (e.g. `"generic.maxHealth"`) — attribute *registration*
    /// is `mc-data`'s concern, so this stays a raw string rather than a resolved enum.
    pub key: String,
    /// The attribute's base value, before modifiers.
    pub base_value: f64,
    /// Active modifiers, in wire order.
    pub modifiers: Vec<AttributeModifierSnapshot>,
}

/// `0x20 Entity Properties`: a batch of attribute (max health, movement speed, ...) snapshots
/// for one entity.
/// java: `S20PacketEntityProperties`
#[derive(Debug, Clone, PartialEq)]
pub struct EntityProperties {
    /// The entity id.
    pub entity_id: i32,
    /// The attribute snapshots, in wire order.
    pub properties: Vec<AttributePropertySnapshot>,
}

impl Packet for EntityProperties {
    const ID: i32 = 0x20;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        // Unlike almost every other list length in this batch, this one is a plain `i32`
        // (`readInt`), not a VarInt.
        let count = buf.read_i32()?;
        // Vanilla uses a plain `for (int j = 0; j < count; ++j)`: negative counts simply make
        // this loop empty.  Do not turn that tolerated malformed input into a disconnect.
        let mut properties = Vec::new();
        for _ in 0..count {
            let key = buf.read_string(64)?;
            let base_value = buf.read_f64()?;
            let modifier_count = buf.read_varint()?;
            // Same signed-loop semantics as the outer list in S20PacketEntityProperties.
            let mut modifiers = Vec::new();
            for _ in 0..modifier_count {
                let uuid = buf.read_uuid()?;
                let amount = buf.read_f64()?;
                let operation = AttributeModifierOperation::from_id(i32::from(buf.read_i8()?))?;
                modifiers.push(AttributeModifierSnapshot {
                    uuid,
                    amount,
                    operation,
                });
            }
            properties.push(AttributePropertySnapshot {
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
        buf.write_varint(self.entity_id);
        buf.write_i32(self.properties.len() as i32);
        for prop in &self.properties {
            buf.write_string(&prop.key);
            buf.write_f64(prop.base_value);
            buf.write_varint(prop.modifiers.len() as i32);
            for modifier in &prop.modifiers {
                buf.write_uuid(modifier.uuid);
                buf.write_f64(modifier.amount);
                buf.write_i8(modifier.operation.to_id() as i8);
            }
        }
    }
}

// ---- 0x22 MultiBlockChange ----------------------------------------------------------------

/// One changed block within a [`MultiBlockChange`]. The chunk-relative coordinates and the
/// (still chunk-relative-packed) block state id are unpacked from a single wire `i16` — see
/// [`MultiBlockChange`]'s doc for the exact bit layout.
/// java: `S22PacketMultiBlockChange.BlockUpdateData`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MultiBlockChangeEntry {
    /// Chunk-relative X (`0..=15`).
    pub rel_x: u8,
    /// Absolute Y (`0..=255`).
    pub y: u8,
    /// Chunk-relative Z (`0..=15`).
    pub rel_z: u8,
    /// The new block state id (`Block.BLOCK_STATE_IDS`'s registry — `mc-data`'s concern, so
    /// this stays a plain integer).
    pub block_state_id: i32,
}

/// `0x22 Multi Block Change`: several block updates within one chunk column, batched into one
/// packet.
///
/// Each entry's position is packed into one `i16`, **not** the packed 26/12/26 `Position`
/// [`PacketBuffer::read_block_pos`] handles elsewhere — a different, chunk-relative packing:
/// bits 12-15 are relative X (`0..=15`), bits 8-11 are relative Z (`0..=15`), and bits 0-7 are
/// the *absolute* Y (`0..=255`, a full byte — 1.8.9's world height fits exactly).
/// java: `S22PacketMultiBlockChange` — `BlockUpdateData.getPos()`:
/// `chunkPosCrammed >> 12 & 15` (X), `chunkPosCrammed & 255` (Y), `chunkPosCrammed >> 8 & 15` (Z)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MultiBlockChange {
    /// The chunk column's X coordinate (in chunks).
    pub chunk_x: i32,
    /// The chunk column's Z coordinate (in chunks).
    pub chunk_z: i32,
    /// The changed blocks, in wire order.
    pub changes: Vec<MultiBlockChangeEntry>,
}

impl Packet for MultiBlockChange {
    const ID: i32 = 0x22;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let chunk_x = buf.read_i32()?;
        let chunk_z = buf.read_i32()?;
        let count = buf.read_varint()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        let mut changes = Vec::new();
        for _ in 0..count {
            let crammed = buf.read_i16()? as u16;
            let rel_x = ((crammed >> 12) & 0xF) as u8;
            let rel_z = ((crammed >> 8) & 0xF) as u8;
            let y = (crammed & 0xFF) as u8;
            let block_state_id = buf.read_varint()?;
            changes.push(MultiBlockChangeEntry {
                rel_x,
                y,
                rel_z,
                block_state_id,
            });
        }
        Ok(MultiBlockChange {
            chunk_x,
            chunk_z,
            changes,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.chunk_x);
        buf.write_i32(self.chunk_z);
        buf.write_varint(self.changes.len() as i32);
        for change in &self.changes {
            let crammed: u16 = (u16::from(change.rel_x & 0xF) << 12)
                | (u16::from(change.rel_z & 0xF) << 8)
                | u16::from(change.y);
            buf.write_i16(crammed as i16);
            buf.write_varint(change.block_state_id);
        }
    }
}

// ---- 0x23-0x25: single block-position packets ----------------------------------------------

/// `0x23 Block Change`: one block's state changed.
/// java: `S23PacketBlockChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockChange {
    /// The changed block's position.
    pub position: BlockPos,
    /// The new block state id (`mc-data`'s registry — a plain integer here).
    pub block_state_id: i32,
}

impl Packet for BlockChange {
    const ID: i32 = 0x23;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let block_state_id = buf.read_varint()?;
        Ok(BlockChange {
            position,
            block_state_id,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
        buf.write_varint(self.block_state_id);
    }
}

/// `0x24 Block Action`: a block-type-dependent "action" event — note block instrument/pitch,
/// a piston extending/retracting, a chest's viewer count, and more, each interpreting `data1`/
/// `data2` differently.
/// java: `S24PacketBlockAction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockAction {
    /// The block's position.
    pub position: BlockPos,
    /// The first action byte (e.g. a note block's instrument).
    pub data1: u8,
    /// The second action byte (e.g. a note block's pitch).
    pub data2: u8,
    /// The block type this action applies to, masked to 12 bits (`& 4095`) exactly as vanilla
    /// does on both read and write — block ids never exceed 4095 in 1.8.9, so the mask is
    /// purely defensive. Indexes `mc-data`'s block registry, hence a plain integer here.
    pub block_id: i32,
}

impl Packet for BlockAction {
    const ID: i32 = 0x24;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let data1 = buf.read_u8()?;
        let data2 = buf.read_u8()?;
        let block_id = buf.read_varint()? & 0xFFF;
        Ok(BlockAction {
            position,
            data1,
            data2,
            block_id,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
        buf.write_u8(self.data1);
        buf.write_u8(self.data2);
        buf.write_varint(self.block_id & 0xFFF);
    }
}

/// `0x25 Block Break Animation`: the cracking overlay shown while a player mines a block.
/// java: `S25PacketBlockBreakAnim`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockBreakAnimation {
    /// The id of the entity (player) doing the breaking — used as a key to know which
    /// animation to replace/clear, not necessarily a currently-tracked entity.
    pub breaker_id: i32,
    /// The block being broken.
    pub position: BlockPos,
    /// Break stage, `0..=9` show progressively more cracked overlays; any other value (vanilla
    /// uses `10` and up) clears the overlay.
    pub progress: u8,
}

impl Packet for BlockBreakAnimation {
    const ID: i32 = 0x25;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let breaker_id = buf.read_varint()?;
        let position = buf.read_block_pos()?;
        let progress = buf.read_u8()?;
        Ok(BlockBreakAnimation {
            breaker_id,
            position,
            progress,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.breaker_id);
        buf.write_block_pos(self.position);
        buf.write_u8(self.progress);
    }
}

// ---- 0x27 Explosion -----------------------------------------------------------------------

/// One block destroyed by an [`Explosion`], as an offset from the explosion's own (floored)
/// position — **not** an absolute [`BlockPos`]; recovering the absolute position requires
/// flooring [`Explosion::x`]/`y`/`z` and adding these deltas, which this crate leaves to the
/// caller (no floor-and-add math is performed here).
/// java: `S27PacketExplosion.readPacketData` (the affected-block loop)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExplosionAffectedBlock {
    /// X offset from the explosion's floored position.
    pub dx: i8,
    /// Y offset from the explosion's floored position.
    pub dy: i8,
    /// Z offset from the explosion's floored position.
    pub dz: i8,
}

/// `0x27 Explosion`: an explosion's origin, blast strength, the blocks it destroyed, and the
/// velocity kick applied to the local player if caught in the blast.
///
/// `x`/`y`/`z` are `f64` fields on the Rust side (matching `Entity` position conventions
/// elsewhere), but are narrowed to `f32` on the wire — vanilla does the same narrowing
/// (`buf.writeFloat((float)this.posX)`) and widens back to `double` on read, so this is a
/// lossless round trip of whatever value vanilla itself would carry, not an added loss.
/// java: `S27PacketExplosion`
#[derive(Debug, Clone, PartialEq)]
pub struct Explosion {
    /// Explosion origin X.
    pub x: f64,
    /// Explosion origin Y.
    pub y: f64,
    /// Explosion origin Z.
    pub z: f64,
    /// Blast radius/strength.
    pub strength: f32,
    /// Blocks destroyed by the blast, as offsets from the (floored) origin.
    pub affected_blocks: Vec<ExplosionAffectedBlock>,
    /// Extra X velocity applied to the local player, if caught in the blast.
    pub motion_x: f32,
    /// Extra Y velocity applied to the local player, if caught in the blast.
    pub motion_y: f32,
    /// Extra Z velocity applied to the local player, if caught in the blast.
    pub motion_z: f32,
}

impl Packet for Explosion {
    const ID: i32 = 0x27;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let x = f64::from(buf.read_f32()?);
        let y = f64::from(buf.read_f32()?);
        let z = f64::from(buf.read_f32()?);
        let strength = buf.read_f32()?;
        // A plain `i32` (`readInt`), not a VarInt.
        let count = buf.read_i32()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        let mut affected_blocks = Vec::new();
        for _ in 0..count {
            let dx = buf.read_i8()?;
            let dy = buf.read_i8()?;
            let dz = buf.read_i8()?;
            affected_blocks.push(ExplosionAffectedBlock { dx, dy, dz });
        }
        let motion_x = buf.read_f32()?;
        let motion_y = buf.read_f32()?;
        let motion_z = buf.read_f32()?;
        Ok(Explosion {
            x,
            y,
            z,
            strength,
            affected_blocks,
            motion_x,
            motion_y,
            motion_z,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.x as f32);
        buf.write_f32(self.y as f32);
        buf.write_f32(self.z as f32);
        buf.write_f32(self.strength);
        buf.write_i32(self.affected_blocks.len() as i32);
        for block in &self.affected_blocks {
            buf.write_i8(block.dx);
            buf.write_i8(block.dy);
            buf.write_i8(block.dz);
        }
        buf.write_f32(self.motion_x);
        buf.write_f32(self.motion_y);
        buf.write_f32(self.motion_z);
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
    fn entity_round_trips() {
        round_trip(Entity { entity_id: 42 });
    }

    #[test]
    fn entity_rel_move_round_trips_negative_deltas() {
        round_trip(EntityRelMove {
            entity_id: 7,
            delta_x: -128,
            delta_y: 127,
            delta_z: 0,
            on_ground: true,
        });
    }

    #[test]
    fn entity_look_round_trips() {
        round_trip(EntityLook {
            entity_id: 7,
            yaw: Angle(200),
            pitch: Angle(10),
            on_ground: false,
        });
    }

    #[test]
    fn entity_look_and_rel_move_round_trips() {
        round_trip(EntityLookAndRelMove {
            entity_id: 99,
            delta_x: 1,
            delta_y: -1,
            delta_z: 2,
            yaw: Angle(0),
            pitch: Angle(255),
            on_ground: true,
        });
    }

    #[test]
    fn entity_teleport_round_trips() {
        round_trip(EntityTeleport {
            entity_id: 5,
            x: -12345,
            y: 2048,
            z: 999_999,
            yaw: Angle(64),
            pitch: Angle(192),
            on_ground: false,
        });
    }

    #[test]
    fn entity_head_look_round_trips() {
        round_trip(EntityHeadLook {
            entity_id: 3,
            yaw: Angle(128),
        });
    }

    #[test]
    fn entity_status_round_trips_and_uses_plain_i32_id() {
        let pkt = EntityStatus {
            entity_id: 300,
            opcode: 3,
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        // A VarInt for 300 would take two bytes; a plain i32 takes four. Confirm the latter.
        let mut peek = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(peek.read_i32().unwrap(), 300);
        assert_eq!(peek.read_i8().unwrap(), 3);
        assert!(peek.is_empty());
        round_trip(pkt);
    }

    #[test]
    fn entity_attach_round_trips_vehicle_and_leash() {
        round_trip(EntityAttach {
            entity_id: 1,
            vehicle_entity_id: -1,
            attach_type: AttachType::Vehicle,
        });
        round_trip(EntityAttach {
            entity_id: 1,
            vehicle_entity_id: 55,
            attach_type: AttachType::Leash,
        });
    }

    #[test]
    fn entity_attach_preserves_unknown_attach_type_for_handler_fallthrough() {
        for id in 2..=u8::MAX {
            let mut buf = PacketBuffer::new();
            buf.write_i32(1);
            buf.write_i32(-1);
            buf.write_u8(id);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet =
                EntityAttach::read(&mut read_buf).expect("unknown leash byte must still decode");
            assert_eq!(packet.attach_type, AttachType::Unknown(id));
            assert!(read_buf.is_empty());
            round_trip(packet);
        }
    }

    #[test]
    fn entity_metadata_round_trips() {
        round_trip(EntityMetadata {
            entity_id: 8,
            metadata: vec![
                metadata::MetadataEntry {
                    index: 0,
                    value: metadata::MetadataValue::Byte(-1),
                },
                metadata::MetadataEntry {
                    index: 12,
                    value: metadata::MetadataValue::String("hi".into()),
                },
            ],
        });
    }

    #[test]
    fn entity_effect_round_trips() {
        round_trip(EntityEffect {
            entity_id: 10,
            effect_id: 1,
            amplifier: 2,
            duration: 32767,
            show_particles: true,
        });
        round_trip(EntityEffect {
            entity_id: 10,
            effect_id: 1,
            amplifier: 0,
            duration: 20,
            show_particles: false,
        });
    }

    #[test]
    fn entity_effect_particle_byte_uses_vanilla_show_polarity() {
        fn decode_particle_byte(raw: u8) -> EntityEffect {
            let mut buf = PacketBuffer::new();
            buf.write_varint(10);
            buf.write_i8(1);
            buf.write_i8(2);
            buf.write_varint(40);
            buf.write_u8(raw);

            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = EntityEffect::read(&mut read_buf).unwrap();
            assert!(read_buf.is_empty());
            packet
        }

        // S1DPacketEntityEffect stores the raw byte in a misleadingly named `hideParticles`
        // member, but func_179707_f passes `byte != 0` to PotionEffect's `showParticles` argument.
        assert!(!decode_particle_byte(0).show_particles);
        assert!(decode_particle_byte(1).show_particles);
        assert!(decode_particle_byte(u8::MAX).show_particles);
    }

    #[test]
    fn remove_entity_effect_round_trips() {
        round_trip(RemoveEntityEffect {
            entity_id: 10,
            effect_id: 200,
        });
    }

    #[test]
    fn set_experience_round_trips() {
        round_trip(SetExperience {
            experience_bar: 0.75,
            level: 30,
            total_experience: 1500,
        });
    }

    #[test]
    fn entity_properties_round_trips_with_modifiers() {
        let uuid = Uuid::from_u128(0x1234_5678_9abc_def0_1234_5678_9abc_def0);
        round_trip(EntityProperties {
            entity_id: 11,
            properties: vec![
                AttributePropertySnapshot {
                    key: "generic.maxHealth".into(),
                    base_value: 20.0,
                    modifiers: vec![AttributeModifierSnapshot {
                        uuid,
                        amount: 4.0,
                        operation: AttributeModifierOperation::Add,
                    }],
                },
                AttributePropertySnapshot {
                    key: "generic.movementSpeed".into(),
                    base_value: 0.1,
                    modifiers: vec![],
                },
            ],
        });
    }

    #[test]
    fn entity_properties_tolerates_negative_counts_like_vanilla_signed_loops() {
        // The outer Java loop is `for (int j = 0; j < i; ++j)`, so a negative count decodes an
        // otherwise valid empty snapshot list. An old `NegativeLength` guard mutant rejects it.
        let mut buf = PacketBuffer::new();
        buf.write_varint(1);
        buf.write_i32(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let packet = EntityProperties::read(&mut read_buf).unwrap();
        assert_eq!(packet.entity_id, 1);
        assert!(packet.properties.is_empty());
        assert!(read_buf.is_empty());

        // The nested modifier loop has exactly the same form. It must retain the enclosing
        // property while treating the negative modifier count as an empty modifier list.
        let mut buf = PacketBuffer::new();
        buf.write_varint(2);
        buf.write_i32(1);
        buf.write_string("generic.maxHealth");
        buf.write_f64(20.0);
        buf.write_varint(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let packet = EntityProperties::read(&mut read_buf).unwrap();
        assert_eq!(packet.entity_id, 2);
        assert_eq!(packet.properties.len(), 1);
        assert_eq!(packet.properties[0].key, "generic.maxHealth");
        assert_eq!(packet.properties[0].base_value, 20.0);
        assert!(packet.properties[0].modifiers.is_empty());
        assert!(read_buf.is_empty());
    }

    #[test]
    fn entity_properties_rejects_out_of_range_modifier_operations() {
        // AttributeModifier's constructor validates `0 <= operation <= 2`; the Rust enum
        // rejection must retain that semantic even though it reports a ProtoError rather than
        // Java's unchecked IllegalArgumentException.
        for operation in [-1_i8, 3_i8, i8::MAX] {
            let mut buf = PacketBuffer::new();
            buf.write_varint(2);
            buf.write_i32(1);
            buf.write_string("generic.maxHealth");
            buf.write_f64(20.0);
            buf.write_varint(1);
            buf.write_uuid(Uuid::nil());
            buf.write_f64(1.0);
            buf.write_i8(operation);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            assert!(matches!(
                EntityProperties::read(&mut read_buf),
                Err(ProtoError::InvalidEnum {
                    what: "attribute modifier operation",
                    value
                }) if value == i32::from(operation)
            ));
        }
    }

    #[test]
    fn multi_block_change_round_trips_and_packs_bits_like_vanilla() {
        let pkt = MultiBlockChange {
            chunk_x: -4,
            chunk_z: 9,
            changes: vec![
                MultiBlockChangeEntry {
                    rel_x: 1,
                    y: 200,
                    rel_z: 14,
                    block_state_id: 123,
                },
                MultiBlockChangeEntry {
                    rel_x: 15,
                    y: 0,
                    rel_z: 0,
                    block_state_id: 0,
                },
            ],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);

        // Confirm the exact bit layout: chunkPosCrammed >> 12 & 15 (x), & 255 (y),
        // >> 8 & 15 (z), matching S22PacketMultiBlockChange.BlockUpdateData.getPos().
        let mut peek = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(peek.read_i32().unwrap(), -4);
        assert_eq!(peek.read_i32().unwrap(), 9);
        assert_eq!(peek.read_varint().unwrap(), 2);
        let crammed = peek.read_i16().unwrap() as u16;
        assert_eq!((crammed >> 12) & 0xF, 1);
        assert_eq!(crammed & 0xFF, 200);
        assert_eq!((crammed >> 8) & 0xF, 14);

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(MultiBlockChange::read(&mut read_buf).unwrap(), pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn block_change_round_trips() {
        round_trip(BlockChange {
            position: BlockPos::new(-10, 64, 100),
            block_state_id: 4080,
        });
    }

    #[test]
    fn block_action_round_trips_and_masks_block_id() {
        let mut buf = PacketBuffer::new();
        buf.write_block_pos(BlockPos::new(1, 2, 3));
        buf.write_u8(5);
        buf.write_u8(6);
        // Deliberately over 12 bits: the reader must mask it down.
        buf.write_varint(0x1FFF);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let pkt = BlockAction::read(&mut read_buf).unwrap();
        assert_eq!(pkt.block_id, 0x1FFF & 0xFFF);

        round_trip(BlockAction {
            position: BlockPos::new(1, 2, 3),
            data1: 5,
            data2: 6,
            block_id: 0xABC,
        });
    }

    #[test]
    fn note_block_action_wire_preserves_the_full_pitch_byte() {
        // S24's data2 is a raw unsigned byte.  BlockNote receives it unchanged and performs
        // its double-precision pitch calculation even for values outside the server's usual
        // 0..=24 note range.
        let position = BlockPos::new(-11, 72, 19);
        for note in 0..=u8::MAX {
            let mut buf = PacketBuffer::new();
            buf.write_block_pos(position);
            buf.write_u8(0); // harp
            buf.write_u8(note);
            buf.write_varint(25); // BlockNote
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let action = BlockAction::read(&mut read_buf).expect("valid raw S24 payload");
            assert_eq!(action.position, position);
            assert_eq!(action.block_id, 25);
            assert_eq!(action.data2, note, "wire pitch byte {note} was changed");
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn block_break_animation_round_trips() {
        round_trip(BlockBreakAnimation {
            breaker_id: 77,
            position: BlockPos::new(0, 0, 0),
            progress: 9,
        });
    }

    #[test]
    fn explosion_round_trips() {
        round_trip(Explosion {
            x: 100.5,
            y: 64.0,
            z: -200.25,
            strength: 4.0,
            affected_blocks: vec![
                ExplosionAffectedBlock {
                    dx: -1,
                    dy: 0,
                    dz: 1,
                },
                ExplosionAffectedBlock {
                    dx: 2,
                    dy: -2,
                    dz: 0,
                },
            ],
            motion_x: 0.1,
            motion_y: 0.5,
            motion_z: -0.1,
        });
    }

    #[test]
    fn explosion_narrows_position_through_f32_like_vanilla() {
        // Vanilla itself narrows posX/Y/Z to float on the wire; a value that isn't exactly
        // representable in f32 must come back rounded the same way vanilla would send it,
        // not preserved at full f64 precision.
        let pkt = Explosion {
            x: 1.0 / 3.0,
            y: 0.0,
            z: 0.0,
            strength: 1.0,
            affected_blocks: vec![],
            motion_x: 0.0,
            motion_y: 0.0,
            motion_z: 0.0,
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let round_tripped = Explosion::read(&mut read_buf).unwrap();
        assert_eq!(round_tripped.x, f64::from(1.0f32 / 3.0f32));
    }

    #[test]
    fn explosion_rejects_negative_count() {
        let mut buf = PacketBuffer::new();
        buf.write_f32(0.0);
        buf.write_f32(0.0);
        buf.write_f32(0.0);
        buf.write_f32(1.0);
        buf.write_i32(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            Explosion::read(&mut read_buf),
            Err(ProtoError::NegativeLength(-1))
        ));
    }
}
