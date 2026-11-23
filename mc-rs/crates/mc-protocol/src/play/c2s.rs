//! Serverbound Play, ids `0x00`-`0x19`: `KeepAlive` .. `ResourcePackStatus` (all 26).
//!
//! java: `network/play/client/**`. Four of these ids (`0x03`-`0x06`) come from a single Java
//! file, `C03PacketPlayer.java`: a base class (`on_ground` only) plus three nested static
//! subclasses (`C04PacketPlayerPosition`, `C05PacketPlayerLook`, `C06PacketPlayerPosLook`) that
//! each read their own fields first, then delegate to the base class's `readPacketData` for the
//! trailing `on_ground` byte. Since Rust has no inheritance, they're four independent structs
//! here, each re-implementing that trailing-byte read/write directly.

use uuid::Uuid;

use crate::item::ItemStack;
use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError};
use mc_math::{BlockPos, Facing};

/// `0x00 Keep Alive` (serverbound): echoes the id from the clientbound `KeepAlive` back, so the
/// server can measure round-trip time and detect a stalled connection.
/// java: `C00PacketKeepAlive`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive {
    /// The id last sent by the server's clientbound `KeepAlive`.
    pub key: i32,
}

impl Packet for KeepAlive {
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(KeepAlive {
            key: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.key);
    }
}

/// `0x01 Chat Message` (serverbound): the raw text a player typed, before any `/`-command
/// parsing. java: `C01PacketChatMessage`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// The typed text, bounded to 100 UTF-16 code units (vanilla's client-side truncation is
    /// mirrored server-side by this same cap, via [`PacketBuffer::read_string`]). `write` does
    /// not cut: the client's 256 Character Chat mod sends up to 256 (1.11's limit), which a real
    /// 1.8 server's `read` rejects.
    pub message: String,
}

impl Packet for ChatMessage {
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ChatMessage {
            message: buf.read_string(100)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.message);
    }
}

/// The action a [`UseEntity`] packet reports. java: `C02PacketUseEntity.Action`
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UseEntityAction {
    /// Ordinal 0: right-click with no precise hit position (open a GUI, mount, etc).
    Interact,
    /// Ordinal 1: left-click / attack.
    Attack,
    /// Ordinal 2: right-click at a precise point on the entity's hitbox, carrying that point.
    InteractAt {
        /// X of the clicked point, in the entity's local (hitbox-relative) space.
        target_x: f32,
        /// Y of the clicked point.
        target_y: f32,
        /// Z of the clicked point.
        target_z: f32,
    },
}

impl UseEntityAction {
    fn ordinal(self) -> i32 {
        match self {
            UseEntityAction::Interact => 0,
            UseEntityAction::Attack => 1,
            UseEntityAction::InteractAt { .. } => 2,
        }
    }
}

/// `0x02 Use Entity` (serverbound): a right-click or left-click aimed at another entity.
/// java: `C02PacketUseEntity`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UseEntity {
    /// The target entity's id.
    pub entity_id: i32,
    /// Which interaction this was, and (for [`UseEntityAction::InteractAt`]) exactly where.
    pub action: UseEntityAction,
}

impl Packet for UseEntity {
    const ID: i32 = 0x02;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        // java: `readEnumValue` — a VarInt ordinal indexing the `Action` enum's constant array.
        let ordinal = buf.read_varint()?;
        let action = match ordinal {
            0 => UseEntityAction::Interact,
            1 => UseEntityAction::Attack,
            2 => UseEntityAction::InteractAt {
                target_x: buf.read_f32()?,
                target_y: buf.read_f32()?,
                target_z: buf.read_f32()?,
            },
            other => {
                return Err(ProtoError::InvalidEnum {
                    what: "use-entity action",
                    value: other,
                })
            }
        };
        Ok(UseEntity { entity_id, action })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_varint(self.action.ordinal());
        if let UseEntityAction::InteractAt {
            target_x,
            target_y,
            target_z,
        } = self.action
        {
            buf.write_f32(target_x);
            buf.write_f32(target_y);
            buf.write_f32(target_z);
        }
    }
}

/// `0x03 Player` (serverbound): the base case of the four `on_ground`-only-carrying movement
/// packets — sent when neither position nor look changed since the last tick.
/// java: `C03PacketPlayer`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    /// Whether the client believes it is standing on solid ground.
    pub on_ground: bool,
}

impl Packet for Player {
    const ID: i32 = 0x03;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Player {
            on_ground: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_bool(self.on_ground);
    }
}

/// `0x04 Player Position` (serverbound): a position-only movement update.
/// java: `C03PacketPlayer.C04PacketPlayerPosition`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPosition {
    /// The player's feet X, as a plain double (**not** [`PacketBuffer::read_block_pos`] — this
    /// is the live float position, not a block coordinate).
    pub x: f64,
    /// The player's feet Y.
    pub y: f64,
    /// The player's feet Z.
    pub z: f64,
    /// Whether the client believes it is standing on solid ground.
    pub on_ground: bool,
}

impl Packet for PlayerPosition {
    const ID: i32 = 0x04;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let x = buf.read_f64()?;
        let y = buf.read_f64()?;
        let z = buf.read_f64()?;
        let on_ground = buf.read_bool()?;
        Ok(PlayerPosition { x, y, z, on_ground })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f64(self.x);
        buf.write_f64(self.y);
        buf.write_f64(self.z);
        buf.write_bool(self.on_ground);
    }
}

/// `0x05 Player Look` (serverbound): a look-only movement update.
/// java: `C03PacketPlayer.C05PacketPlayerLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerLook {
    /// Full-precision yaw in degrees, as a plain float (**not** the single-byte [`crate::Angle`]
    /// used by entity-movement packets elsewhere in the protocol).
    pub yaw: f32,
    /// Full-precision pitch in degrees.
    pub pitch: f32,
    /// Whether the client believes it is standing on solid ground.
    pub on_ground: bool,
}

impl Packet for PlayerLook {
    const ID: i32 = 0x05;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let yaw = buf.read_f32()?;
        let pitch = buf.read_f32()?;
        let on_ground = buf.read_bool()?;
        Ok(PlayerLook {
            yaw,
            pitch,
            on_ground,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.yaw);
        buf.write_f32(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x06 Player Position And Look` (serverbound): a combined position+look movement update.
/// java: `C03PacketPlayer.C06PacketPlayerPosLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPositionAndLook {
    /// The player's feet X.
    pub x: f64,
    /// The player's feet Y.
    pub y: f64,
    /// The player's feet Z.
    pub z: f64,
    /// Full-precision yaw in degrees.
    pub yaw: f32,
    /// Full-precision pitch in degrees.
    pub pitch: f32,
    /// Whether the client believes it is standing on solid ground.
    pub on_ground: bool,
}

impl Packet for PlayerPositionAndLook {
    const ID: i32 = 0x06;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let x = buf.read_f64()?;
        let y = buf.read_f64()?;
        let z = buf.read_f64()?;
        let yaw = buf.read_f32()?;
        let pitch = buf.read_f32()?;
        let on_ground = buf.read_bool()?;
        Ok(PlayerPositionAndLook {
            x,
            y,
            z,
            yaw,
            pitch,
            on_ground,
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

/// The status a [`PlayerDigging`] packet reports. java: `C07PacketPlayerDigging.Action`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayerDiggingStatus {
    /// Ordinal 0: the player started breaking a block.
    StartDestroyBlock,
    /// Ordinal 1: digging was cancelled (block unchanged).
    AbortDestroyBlock,
    /// Ordinal 2: the block finished breaking.
    StopDestroyBlock,
    /// Ordinal 3: drop the entire held stack (`Q` with a modifier).
    DropAllItems,
    /// Ordinal 4: drop a single item from the held stack (`Q`).
    DropItem,
    /// Ordinal 5: finished using the held item (releasing a bow, finishing food/a potion).
    ReleaseUseItem,
}

impl PlayerDiggingStatus {
    fn from_id(id: i32) -> Result<Self, ProtoError> {
        match id {
            0 => Ok(PlayerDiggingStatus::StartDestroyBlock),
            1 => Ok(PlayerDiggingStatus::AbortDestroyBlock),
            2 => Ok(PlayerDiggingStatus::StopDestroyBlock),
            3 => Ok(PlayerDiggingStatus::DropAllItems),
            4 => Ok(PlayerDiggingStatus::DropItem),
            5 => Ok(PlayerDiggingStatus::ReleaseUseItem),
            other => Err(ProtoError::InvalidEnum {
                what: "player-digging status",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            PlayerDiggingStatus::StartDestroyBlock => 0,
            PlayerDiggingStatus::AbortDestroyBlock => 1,
            PlayerDiggingStatus::StopDestroyBlock => 2,
            PlayerDiggingStatus::DropAllItems => 3,
            PlayerDiggingStatus::DropItem => 4,
            PlayerDiggingStatus::ReleaseUseItem => 5,
        }
    }
}

/// `0x07 Player Digging` (serverbound): block-breaking progress, or an item drop/use-release
/// (which reuse this packet with a sentinel status rather than carrying meaningful position/face
/// fields). java: `C07PacketPlayerDigging`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerDigging {
    /// What stage of digging (or which non-digging action) this is.
    pub status: PlayerDiggingStatus,
    /// The block being dug (meaningless, but still present on the wire, for the drop/release
    /// statuses).
    pub position: BlockPos,
    /// Which face of the block is being dug.
    pub face: Facing,
}

impl Packet for PlayerDigging {
    const ID: i32 = 0x07;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let status = PlayerDiggingStatus::from_id(buf.read_varint()?)?;
        let position = buf.read_block_pos()?;
        // java: `EnumFacing.getFront(buf.readUnsignedByte())` — wraps rather than erroring on an
        // out-of-range byte, exactly what `Facing::from_index` reproduces.
        let face = Facing::from_index(i32::from(buf.read_u8()?));
        Ok(PlayerDigging {
            status,
            position,
            face,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.status.to_id());
        buf.write_block_pos(self.position);
        buf.write_u8(self.face.index());
    }
}

/// `0x08 Player Block Placement` (serverbound): placing a block, or right-clicking with an item
/// while not aimed at a block. java: `C08PacketPlayerBlockPlacement`
///
/// Vanilla signals "not placing against a block" (using an item in the air) with the sentinel
/// position `(-1, -1, -1)` and `placed_block_direction == 255`; this type decodes and
/// re-encodes whatever bytes are on the wire without special-casing that sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerBlockPlacement {
    /// The block being placed against (or the `(-1, -1, -1)` sentinel — see the type doc).
    pub position: BlockPos,
    /// Raw wire value for the face being placed against. Kept as a plain byte rather than a
    /// [`Facing`] because `255` is a legal sentinel here (see the type doc), which
    /// `Facing::from_index` would silently fold into a real face.
    pub placed_block_direction: u8,
    /// The item stack being placed/used.
    pub stack: Option<ItemStack>,
    /// Click-point X within the target block, raw byte (vanilla divides this by 16.0 to get an
    /// offset in `[0, 1)`; kept raw here for exact wire round-tripping).
    pub cursor_x: u8,
    /// Click-point Y within the target block, raw byte.
    pub cursor_y: u8,
    /// Click-point Z within the target block, raw byte.
    pub cursor_z: u8,
}

impl Packet for PlayerBlockPlacement {
    const ID: i32 = 0x08;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let placed_block_direction = buf.read_u8()?;
        let stack = buf.read_item_stack()?;
        let cursor_x = buf.read_u8()?;
        let cursor_y = buf.read_u8()?;
        let cursor_z = buf.read_u8()?;
        Ok(PlayerBlockPlacement {
            position,
            placed_block_direction,
            stack,
            cursor_x,
            cursor_y,
            cursor_z,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
        buf.write_u8(self.placed_block_direction);
        buf.write_item_stack(self.stack.as_ref());
        buf.write_u8(self.cursor_x);
        buf.write_u8(self.cursor_y);
        buf.write_u8(self.cursor_z);
    }
}

/// `0x09 Held Item Change` (serverbound): the player switched hotbar slots.
/// java: `C09PacketHeldItemChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeldItemChange {
    /// The newly selected hotbar slot, `0..=8`.
    pub slot_id: i16,
}

impl Packet for HeldItemChange {
    const ID: i32 = 0x09;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(HeldItemChange {
            slot_id: buf.read_i16()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i16(self.slot_id);
    }
}

/// `0x0A Animation` (serverbound): the player swung its arm. No fields — this packet's whole
/// meaning is its id and arrival. java: `C0APacketAnimation`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Animation;

impl Packet for Animation {
    const ID: i32 = 0x0A;

    fn read(_buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Animation)
    }

    fn write(&self, _buf: &mut PacketBuffer) {}
}

/// The action an [`EntityAction`] packet reports. java: `C0BPacketEntityAction.Action`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntityActionType {
    /// Ordinal 0: started sneaking.
    StartSneaking,
    /// Ordinal 1: stopped sneaking.
    StopSneaking,
    /// Ordinal 2: got out of bed.
    StopSleeping,
    /// Ordinal 3: started sprinting.
    StartSprinting,
    /// Ordinal 4: stopped sprinting.
    StopSprinting,
    /// Ordinal 5: a ridden horse was given a jump (charged via [`EntityAction::aux_data`]).
    RidingJump,
    /// Ordinal 6: opened a ridden horse's inventory.
    OpenInventory,
}

impl EntityActionType {
    fn from_id(id: i32) -> Result<Self, ProtoError> {
        match id {
            0 => Ok(EntityActionType::StartSneaking),
            1 => Ok(EntityActionType::StopSneaking),
            2 => Ok(EntityActionType::StopSleeping),
            3 => Ok(EntityActionType::StartSprinting),
            4 => Ok(EntityActionType::StopSprinting),
            5 => Ok(EntityActionType::RidingJump),
            6 => Ok(EntityActionType::OpenInventory),
            other => Err(ProtoError::InvalidEnum {
                what: "entity action",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            EntityActionType::StartSneaking => 0,
            EntityActionType::StopSneaking => 1,
            EntityActionType::StopSleeping => 2,
            EntityActionType::StartSprinting => 3,
            EntityActionType::StopSprinting => 4,
            EntityActionType::RidingJump => 5,
            EntityActionType::OpenInventory => 6,
        }
    }
}

/// `0x0B Entity Action` (serverbound): a player-driven state change (sneak/sprint/etc), or an
/// action on a ridden entity. java: `C0BPacketEntityAction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityAction {
    /// The acting entity's id (always the sending player's own id in vanilla).
    pub entity_id: i32,
    /// Which action this is.
    pub action: EntityActionType,
    /// Extra payload: the horse jump-boost strength (`0..=100`) for
    /// [`EntityActionType::RidingJump`], `0` for every other action.
    pub aux_data: i32,
}

impl EntityAction {
    /// Construct an action whose auxiliary payload is the vanilla default (`0`).
    ///
    /// `C0BPacketEntityAction(Entity, Action)` uses this form for every action except
    /// `RIDING_JUMP`.  Keeping the constructor beside the wire type makes the three
    /// non-edge actions (`STOP_SLEEPING`, `OPEN_INVENTORY`, and `RIDING_JUMP`) available to
    /// callers instead of leaving construction coupled to the four walking-state edges.
    #[must_use]
    pub const fn new(entity_id: i32, action: EntityActionType) -> Self {
        Self {
            entity_id,
            action,
            aux_data: 0,
        }
    }

    /// Construct an action with an explicit auxiliary payload.
    ///
    /// Vanilla only supplies a non-zero value for `RIDING_JUMP`, where it is the truncated
    /// horse-jump power multiplied by 100.  The generic form mirrors the packet's Java fields
    /// and deliberately leaves validation to the same server-side rules as vanilla.
    #[must_use]
    pub const fn with_aux_data(entity_id: i32, action: EntityActionType, aux_data: i32) -> Self {
        Self {
            entity_id,
            action,
            aux_data,
        }
    }

    /// Construct `STOP_SLEEPING`, the packet sent by the sleep screen's wake action.
    #[must_use]
    pub const fn stop_sleeping(entity_id: i32) -> Self {
        Self::new(entity_id, EntityActionType::StopSleeping)
    }

    /// Construct `RIDING_JUMP` with its wire payload.
    #[must_use]
    pub const fn riding_jump(entity_id: i32, aux_data: i32) -> Self {
        Self::with_aux_data(entity_id, EntityActionType::RidingJump, aux_data)
    }

    /// Construct `OPEN_INVENTORY`, the packet sent while mounted on a horse.
    #[must_use]
    pub const fn open_inventory(entity_id: i32) -> Self {
        Self::new(entity_id, EntityActionType::OpenInventory)
    }
}

impl Packet for EntityAction {
    const ID: i32 = 0x0B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let action = EntityActionType::from_id(buf.read_varint()?)?;
        let aux_data = buf.read_varint()?;
        Ok(EntityAction {
            entity_id,
            action,
            aux_data,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_varint(self.action.to_id());
        buf.write_varint(self.aux_data);
    }
}

/// `0x0C Steer Vehicle` (serverbound): the WASD/jump/dismount state while riding something.
/// java: `C0CPacketInput`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteerVehicle {
    /// Positive for left strafe, negative for right, `0` for none.
    pub sideways: f32,
    /// Positive for forward, negative for backward, `0` for none.
    pub forward: f32,
    /// Bit `0`: jumping. Bit `1`: sneaking (dismounts a ridden entity). Vanilla discards every
    /// other bit when decoding and reconstructs this byte from those two booleans when encoding.
    pub flags: u8,
}

impl Packet for SteerVehicle {
    const ID: i32 = 0x0C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let sideways = buf.read_f32()?;
        let forward = buf.read_f32()?;
        let flags = buf.read_u8()? & 0b11;
        Ok(SteerVehicle {
            sideways,
            forward,
            flags,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.sideways);
        buf.write_f32(self.forward);
        buf.write_u8(self.flags & 0b11);
    }
}

/// `0x0D Close Window` (serverbound): the player closed an inventory GUI (including their own,
/// via `Esc`). java: `C0DPacketCloseWindow`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseWindow {
    /// The window that was closed. `0` is the player's own inventory.
    pub window_id: u8,
}

impl Packet for CloseWindow {
    const ID: i32 = 0x0D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(CloseWindow {
            window_id: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
    }
}

/// `0x0E Click Window` (serverbound): a click inside an open inventory GUI. The `mode`/
/// `used_button` pair's interpretation (shift-click, number-key swap, drag, etc) is a
/// higher layer's concern (`Container`'s slot-click logic); this type only carries the raw
/// wire fields. java: `C0EPacketClickWindow`
#[derive(Debug, Clone, PartialEq)]
pub struct ClickWindow {
    /// The window clicked in. `0` is the player's own inventory.
    pub window_id: u8,
    /// The clicked slot's index within that window.
    pub slot_id: i16,
    /// The mouse button (or number key) used; meaning depends on `mode`.
    pub used_button: u8,
    /// A unique id for this click, echoed back by the server's `ConfirmTransaction` so the
    /// client can match acceptance/rejection to the click that caused it.
    pub action_number: i16,
    /// Click mode: `0` normal, `1` shift, `2` number-key swap, `3` middle-click, `4` drop,
    /// `5` drag, `6` double-click.
    pub mode: u8,
    /// The item stack the client believes is now in the clicked slot (a client-side prediction,
    /// authoritative only after the server's confirmation).
    pub clicked_item: Option<ItemStack>,
}

impl Packet for ClickWindow {
    const ID: i32 = 0x0E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let slot_id = buf.read_i16()?;
        let used_button = buf.read_u8()?;
        let action_number = buf.read_i16()?;
        let mode = buf.read_u8()?;
        let clicked_item = buf.read_item_stack()?;
        Ok(ClickWindow {
            window_id,
            slot_id,
            used_button,
            action_number,
            mode,
            clicked_item,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.slot_id);
        buf.write_u8(self.used_button);
        buf.write_i16(self.action_number);
        buf.write_u8(self.mode);
        buf.write_item_stack(self.clicked_item.as_ref());
    }
}

/// `0x0F Confirm Transaction` (serverbound): the client's ack/nack of a server-initiated
/// inventory transaction (see [`ClickWindow::action_number`]). java: `C0FPacketConfirmTransaction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfirmTransaction {
    /// The window the transaction happened in.
    pub window_id: u8,
    /// The transaction id being confirmed (java field name: `uid`).
    pub action_id: i16,
    /// Whether the client accepted the server's version of the transaction.
    pub accepted: bool,
}

impl Packet for ConfirmTransaction {
    const ID: i32 = 0x0F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let action_id = buf.read_i16()?;
        let accepted = buf.read_bool()?;
        Ok(ConfirmTransaction {
            window_id,
            action_id,
            accepted,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.action_id);
        buf.write_bool(self.accepted);
    }
}

/// `0x10 Creative Inventory Action` (serverbound): a direct slot-set, only legal in creative
/// mode (no server-side "do you actually have this item" check applies the way it does for
/// [`ClickWindow`]). java: `C10PacketCreativeInventoryAction`
#[derive(Debug, Clone, PartialEq)]
pub struct CreativeInventoryAction {
    /// The slot being set.
    pub slot_id: i16,
    /// The stack to place there (`None` clears the slot).
    pub stack: Option<ItemStack>,
}

impl Packet for CreativeInventoryAction {
    const ID: i32 = 0x10;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let slot_id = buf.read_i16()?;
        let stack = buf.read_item_stack()?;
        Ok(CreativeInventoryAction { slot_id, stack })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i16(self.slot_id);
        buf.write_item_stack(self.stack.as_ref());
    }
}

/// `0x11 Enchant Item` (serverbound): a click on one of the three enchanting-table options.
/// java: `C11PacketEnchantItem`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnchantItem {
    /// The enchanting-table window being used.
    pub window_id: u8,
    /// Which of the (up to three) enchantment options was clicked, `0..=2`.
    pub button: u8,
}

impl Packet for EnchantItem {
    const ID: i32 = 0x11;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let button = buf.read_u8()?;
        Ok(EnchantItem { window_id, button })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_u8(self.button);
    }
}

/// `0x12 Update Sign` (serverbound): the four lines typed into a sign-edit GUI.
/// java: `C12PacketUpdateSign`
///
/// Vanilla reads each line with `readStringFromBuffer(384)`, then parses that string as an
/// `IChatComponent`. On the write side Java serializes the component again with
/// `IChatComponent.Serializer.componentToJson`; the live sign editor performs that parse/serialize
/// step before constructing this packet. The protocol crate intentionally keeps the four fields as
/// strings because the component tree belongs to `mc-ui`, which already depends on this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateSign {
    /// The sign block's position.
    pub position: BlockPos,
    /// The four lines of text, as canonical serialized chat-component JSON, in top-to-bottom
    /// order. Production sign-editor callers must parse and serialize each line with the shared
    /// `mc-ui::font::component::ChatComponent` implementation before constructing the packet.
    pub lines: [String; 4],
}

impl Packet for UpdateSign {
    const ID: i32 = 0x12;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let lines = [
            buf.read_string(384)?,
            buf.read_string(384)?,
            buf.read_string(384)?,
            buf.read_string(384)?,
        ];
        Ok(UpdateSign { position, lines })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
        for line in &self.lines {
            buf.write_chat(line);
        }
    }
}

/// `0x13 Player Abilities` (serverbound): the client's report of its own fly state (vanilla
/// sends this whenever flight is toggled, in addition to the server pushing its own copy on
/// join). java: `C13PacketPlayerAbilities`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerAbilities {
    /// Whether the player takes no damage (creative mode).
    pub invulnerable: bool,
    /// Whether the player is currently flying.
    pub flying: bool,
    /// Whether the player is allowed to toggle flight.
    pub allow_flying: bool,
    /// Whether the player is in creative mode.
    pub creative_mode: bool,
    /// Flight speed.
    pub fly_speed: f32,
    /// Walk speed.
    pub walk_speed: f32,
}

impl Packet for PlayerAbilities {
    const ID: i32 = 0x13;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let flags = buf.read_u8()?;
        let invulnerable = flags & 1 != 0;
        let flying = flags & 2 != 0;
        let allow_flying = flags & 4 != 0;
        let creative_mode = flags & 8 != 0;
        let fly_speed = buf.read_f32()?;
        let walk_speed = buf.read_f32()?;
        Ok(PlayerAbilities {
            invulnerable,
            flying,
            allow_flying,
            creative_mode,
            fly_speed,
            walk_speed,
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

/// `0x14 Tab Complete` (serverbound): a tab-completion request, either for chat/a command or
/// (when aimed at a block, e.g. completing a `/setblock` argument) scoped to that block.
/// java: `C14PacketTabComplete`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabComplete {
    /// The partial text being completed.
    pub message: String,
    /// The block the client was looking at when it requested completion, if any.
    pub target_block: Option<BlockPos>,
}

impl Packet for TabComplete {
    const ID: i32 = 0x14;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let message = buf.read_string(32767)?;
        let has_target = buf.read_bool()?;
        let target_block = if has_target {
            Some(buf.read_block_pos()?)
        } else {
            None
        };
        Ok(TabComplete {
            message,
            target_block,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        // java: StringUtils.substring(this.message, 0, 32767), then PacketBuffer.writeString.
        // The substring limit is UTF-16 units, while writeString's limit is encoded UTF-8
        // bytes.  Keep those two limits separate (and preserve Java's replacement for a cut
        // surrogate) instead of applying the generic writer to the untruncated Rust string.
        if !write_java_utf16_prefix(buf, &self.message, 32767) {
            // Java throws from writeString before it can emit the optional-target fields.  The
            // buffer records the same deferred error; stop here so no partial packet body is
            // mistaken for a valid C14 payload.
            return;
        }
        buf.write_bool(self.target_block.is_some());
        if let Some(pos) = self.target_block {
            buf.write_block_pos(pos);
        }
    }
}

/// How the client wants incoming chat filtered. java: `EntityPlayer.EnumChatVisibility`
///
/// Wire representation note: unlike the other closed-set fields in this module, this one is
/// **not** sent through `readEnumValue`/`writeEnumValue` (a VarInt ordinal) — vanilla reads/
/// writes it as a single raw byte (`readByte`/`writeByte`), which happens to equal the ordinal
/// for all three variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatVisibility {
    /// Show all chat.
    Full,
    /// Show only system messages (deaths, advancements, etc), not player chat.
    System,
    /// Show no chat at all.
    Hidden,
}

impl ChatVisibility {
    fn from_id(id: i32) -> Result<Self, ProtoError> {
        // java: EnumChatVisibility.getEnumChatVisibility(int) returns ID_LOOKUP[id % 3].
        // The wire value is a signed byte, so a negative id would throw Java's
        // ArrayIndexOutOfBoundsException; keep malformed network input fallible here rather than
        // panicking, while retaining the modulo wrap for every nonnegative ordinal.
        if id < 0 {
            return Err(ProtoError::InvalidEnum {
                what: "chat visibility",
                value: id,
            });
        }
        match id {
            0 => Ok(ChatVisibility::Full),
            1 => Ok(ChatVisibility::System),
            2 => Ok(ChatVisibility::Hidden),
            other => match other % 3 {
                0 => Ok(ChatVisibility::Full),
                1 => Ok(ChatVisibility::System),
                _ => Ok(ChatVisibility::Hidden),
            },
        }
    }

    fn to_id(self) -> i32 {
        match self {
            ChatVisibility::Full => 0,
            ChatVisibility::System => 1,
            ChatVisibility::Hidden => 2,
        }
    }
}

/// `0x15 Client Settings` (serverbound): the client's locale/render-distance/chat/skin options,
/// sent on join and whenever changed. java: `C15PacketClientSettings`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSettings {
    /// The client's locale, e.g. `"en_US"`.
    pub locale: String,
    /// Requested render distance, in chunks.
    pub view_distance: i8,
    /// How much chat the client wants to receive.
    pub chat_visibility: ChatVisibility,
    /// Whether the client wants chat color codes rendered (vs. stripped to plain text).
    pub chat_colors: bool,
    /// Bitmask of which skin layers (cape, jacket, hat, sleeves, etc) are enabled; the bit
    /// layout is a rendering concern, not this crate's.
    pub displayed_skin_parts: u8,
}

impl Packet for ClientSettings {
    const ID: i32 = 0x15;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let locale = buf.read_string(7)?;
        let view_distance = buf.read_i8()?;
        let chat_visibility = ChatVisibility::from_id(i32::from(buf.read_i8()?))?;
        let chat_colors = buf.read_bool()?;
        let displayed_skin_parts = buf.read_u8()?;
        Ok(ClientSettings {
            locale,
            view_distance,
            chat_visibility,
            chat_colors,
            displayed_skin_parts,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.locale);
        buf.write_i8(self.view_distance);
        buf.write_i8(self.chat_visibility.to_id() as i8);
        buf.write_bool(self.chat_colors);
        buf.write_u8(self.displayed_skin_parts);
    }
}

/// The action a [`ClientStatus`] packet reports. java: `C16PacketClientStatus.EnumState`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientStatusAction {
    /// Ordinal 0: respawn after death, or exit the end-poem/credits screen.
    PerformRespawn,
    /// Ordinal 1: request the "stat.*"/achievement values for the stats screen.
    RequestStats,
    /// Ordinal 2: opened the achievements/statistics GUI (vanilla uses this to lazily send
    /// achievement data).
    OpenInventoryAchievement,
}

impl ClientStatusAction {
    fn from_id(id: i32) -> Result<Self, ProtoError> {
        match id {
            0 => Ok(ClientStatusAction::PerformRespawn),
            1 => Ok(ClientStatusAction::RequestStats),
            2 => Ok(ClientStatusAction::OpenInventoryAchievement),
            other => Err(ProtoError::InvalidEnum {
                what: "client status action",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            ClientStatusAction::PerformRespawn => 0,
            ClientStatusAction::RequestStats => 1,
            ClientStatusAction::OpenInventoryAchievement => 2,
        }
    }
}

/// `0x16 Client Status` (serverbound). java: `C16PacketClientStatus`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientStatus {
    /// Which client-driven state change this is.
    pub action: ClientStatusAction,
}

impl Packet for ClientStatus {
    const ID: i32 = 0x16;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ClientStatus {
            action: ClientStatusAction::from_id(buf.read_varint()?)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.action.to_id());
    }
}

/// `0x17 Plugin Message` (serverbound): an opaque, mod/plugin-defined payload on a named
/// channel. java: `C17PacketCustomPayload`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMessage {
    /// The channel name, e.g. `"MC|BEdit"`.
    pub channel: String,
    /// The raw payload. Its length is never sent explicitly — it is everything left in this
    /// packet's frame after `channel`, hence [`PacketBuffer::read_remaining`] rather than a
    /// length-prefixed read.
    pub data: Vec<u8>,
}

impl Packet for PluginMessage {
    const ID: i32 = 0x17;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        const MAX_PAYLOAD: usize = 32_767;
        let channel = buf.read_string(20)?;
        if buf.remaining() > MAX_PAYLOAD {
            return Err(ProtoError::PayloadTooLong {
                len: buf.remaining(),
                max: MAX_PAYLOAD,
            });
        }
        let data = buf.read_remaining();
        Ok(PluginMessage { channel, data })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        const MAX_PAYLOAD: usize = 32_767;
        buf.write_string(&self.channel);
        buf.write_bounded_payload(&self.data, MAX_PAYLOAD);
    }
}

/// `0x18 Spectate` (serverbound): a spectator-mode player teleporting to another entity.
/// java: `C18PacketSpectate`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spectate {
    /// The UUID of the entity to spectate.
    pub target: Uuid,
}

impl Packet for Spectate {
    const ID: i32 = 0x18;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Spectate {
            target: buf.read_uuid()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_uuid(self.target);
    }
}

/// The result a [`ResourcePackStatus`] packet reports.
/// java: `C19PacketResourcePackStatus.Action`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourcePackStatusResult {
    /// Ordinal 0: the pack downloaded and applied successfully.
    SuccessfullyLoaded,
    /// Ordinal 1: the player declined the prompt.
    Declined,
    /// Ordinal 2: the download failed.
    FailedDownload,
    /// Ordinal 3: the client accepted and started downloading (sent before the eventual
    /// loaded/failed follow-up).
    Accepted,
}

impl ResourcePackStatusResult {
    fn from_id(id: i32) -> Result<Self, ProtoError> {
        match id {
            0 => Ok(ResourcePackStatusResult::SuccessfullyLoaded),
            1 => Ok(ResourcePackStatusResult::Declined),
            2 => Ok(ResourcePackStatusResult::FailedDownload),
            3 => Ok(ResourcePackStatusResult::Accepted),
            other => Err(ProtoError::InvalidEnum {
                what: "resource pack status result",
                value: other,
            }),
        }
    }

    fn to_id(self) -> i32 {
        match self {
            ResourcePackStatusResult::SuccessfullyLoaded => 0,
            ResourcePackStatusResult::Declined => 1,
            ResourcePackStatusResult::FailedDownload => 2,
            ResourcePackStatusResult::Accepted => 3,
        }
    }
}

/// `0x19 Resource Pack Status` (serverbound): the client's response to a server-pushed resource
/// pack. java: `C19PacketResourcePackStatus`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePackStatus {
    /// The pack's hash, as sent in the clientbound resource-pack-send packet, bounded to 40
    /// characters (a SHA-1 hex digest's length).
    pub hash: String,
    /// The outcome being reported.
    pub result: ResourcePackStatusResult,
}

impl Packet for ResourcePackStatus {
    const ID: i32 = 0x19;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let hash = buf.read_string(40)?;
        let result = ResourcePackStatusResult::from_id(buf.read_varint()?)?;
        Ok(ResourcePackStatus { hash, result })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        if write_java_utf16_prefix(buf, &self.hash, 40) {
            buf.write_varint(self.result.to_id());
        }
    }
}

/// Write the prefix Java's `String.substring(0, max_utf16)` would produce, including Java's
/// UTF-8 encoder replacement (`?`) when the cut splits a surrogate pair.  Returns `false` when
/// the resulting UTF-8 bytes exceed `PacketBuffer::writeString`'s 32767-byte limit; the error is
/// deferred through that same writer so packet encoding remains infallible by trait contract.
fn write_java_utf16_prefix(buf: &mut PacketBuffer, value: &str, max_utf16: usize) -> bool {
    let mut encoded = Vec::with_capacity(value.len().min(max_utf16));
    let mut used_utf16 = 0;
    for ch in value.chars() {
        let units = ch.len_utf16();
        if used_utf16 + units > max_utf16 {
            if used_utf16 < max_utf16 {
                encoded.push(b'?');
            }
            break;
        }
        let mut utf8 = [0_u8; 4];
        encoded.extend_from_slice(ch.encode_utf8(&mut utf8).as_bytes());
        used_utf16 += units;
    }
    if encoded.len() > 32767 {
        // PacketBuffer intentionally keeps write-side errors private and exposes them through
        // finish_write().  Feed it an ASCII string of the exact encoded length to record the same
        // StringTooLong { len, max } error without putting an invalid prefix on the wire.
        buf.write_string(&"x".repeat(encoded.len()));
        return false;
    }
    buf.write_varint(encoded.len() as i32);
    buf.write_bytes(&encoded);
    true
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
        round_trip(KeepAlive { key: 123_456 });
    }

    #[test]
    fn chat_message_round_trips() {
        round_trip(ChatMessage {
            message: "hello, world".into(),
        });
    }

    /// The caveat of 256 Character Chat: the encoder writes a
    /// 256-character message in full, and a real 1.8 server's decoder (this `read`, vanilla's
    /// `readStringFromBuffer(100)`) rejects it, which disconnects the player. 100 still passes.
    #[test]
    fn a_long_chat_message_encodes_whole_and_a_1_8_decoder_rejects_it() {
        let long = ChatMessage {
            message: "a".repeat(256),
        };
        let mut buf = PacketBuffer::new();
        long.write(&mut buf);
        assert_eq!(buf.as_slice().len(), 2 + 256, "VarInt 256 then every byte");
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            ChatMessage::read(&mut read_buf),
            Err(ProtoError::StringTooLong { len: 256, max: 100 })
        ));
        round_trip(ChatMessage {
            message: "a".repeat(100),
        });
    }

    #[test]
    fn use_entity_round_trips_every_action() {
        round_trip(UseEntity {
            entity_id: 42,
            action: UseEntityAction::Interact,
        });
        round_trip(UseEntity {
            entity_id: 42,
            action: UseEntityAction::Attack,
        });
        round_trip(UseEntity {
            entity_id: 42,
            action: UseEntityAction::InteractAt {
                target_x: 0.25,
                target_y: 0.5,
                target_z: 0.75,
            },
        });
    }

    #[test]
    fn bad_use_entity_action_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(1);
        buf.write_varint(99);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            UseEntity::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { .. })
        ));
    }

    #[test]
    fn player_round_trips() {
        round_trip(Player { on_ground: true });
        round_trip(Player { on_ground: false });
    }

    #[test]
    fn player_position_round_trips() {
        round_trip(PlayerPosition {
            x: 1.5,
            y: 64.0,
            z: -30.25,
            on_ground: true,
        });
    }

    #[test]
    fn player_look_round_trips() {
        round_trip(PlayerLook {
            yaw: 90.0,
            pitch: -45.5,
            on_ground: false,
        });
    }

    #[test]
    fn player_position_and_look_round_trips() {
        round_trip(PlayerPositionAndLook {
            x: 1.5,
            y: 64.0,
            z: -30.25,
            yaw: 180.0,
            pitch: 0.0,
            on_ground: true,
        });
    }

    #[test]
    fn player_digging_round_trips_every_status() {
        let statuses = [
            PlayerDiggingStatus::StartDestroyBlock,
            PlayerDiggingStatus::AbortDestroyBlock,
            PlayerDiggingStatus::StopDestroyBlock,
            PlayerDiggingStatus::DropAllItems,
            PlayerDiggingStatus::DropItem,
            PlayerDiggingStatus::ReleaseUseItem,
        ];
        for status in statuses {
            round_trip(PlayerDigging {
                status,
                position: BlockPos::new(-5, 70, 12),
                face: Facing::North,
            });
        }
    }

    #[test]
    fn bad_player_digging_status_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(99);
        buf.write_block_pos(BlockPos::new(0, 0, 0));
        buf.write_u8(0);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            PlayerDigging::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { .. })
        ));
    }

    #[test]
    fn player_block_placement_round_trips() {
        round_trip(PlayerBlockPlacement {
            position: BlockPos::new(10, 65, -3),
            placed_block_direction: 1,
            stack: Some(ItemStack {
                item_id: 1,
                count: 1,
                damage: 0,
                nbt: None,
            }),
            cursor_x: 8,
            cursor_y: 0,
            cursor_z: 15,
        });
    }

    #[test]
    fn player_block_placement_round_trips_air_use_sentinel() {
        // Using an item in the air: vanilla's "not aimed at a block" sentinel.
        round_trip(PlayerBlockPlacement {
            position: BlockPos::new(-1, -1, -1),
            placed_block_direction: 255,
            stack: None,
            cursor_x: 0,
            cursor_y: 0,
            cursor_z: 0,
        });
    }

    #[test]
    fn held_item_change_round_trips() {
        round_trip(HeldItemChange { slot_id: 8 });
    }

    #[test]
    fn animation_has_no_body() {
        let mut buf = PacketBuffer::new();
        Animation.write(&mut buf);
        assert!(buf.is_empty());
        let mut read_buf = PacketBuffer::from_slice(&[]);
        assert_eq!(Animation::read(&mut read_buf).unwrap(), Animation);
    }

    #[test]
    fn entity_action_round_trips_every_action() {
        let actions = [
            EntityActionType::StartSneaking,
            EntityActionType::StopSneaking,
            EntityActionType::StopSleeping,
            EntityActionType::StartSprinting,
            EntityActionType::StopSprinting,
            EntityActionType::RidingJump,
            EntityActionType::OpenInventory,
        ];
        for action in actions {
            round_trip(EntityAction {
                entity_id: 7,
                action,
                aux_data: if action == EntityActionType::RidingJump {
                    80
                } else {
                    0
                },
            });
        }
    }

    /// The packet's ordinal is an on-wire enum index, not a discriminant that can be inferred
    /// from a Rust layout.  Keep an explicit byte control for every 1.8.9 action, including the
    /// three actions that were historically left without a producer.
    #[test]
    fn entity_action_constructors_encode_all_ordinals_and_aux_data() {
        let actions = [
            (EntityActionType::StartSneaking, 0_u8),
            (EntityActionType::StopSneaking, 1),
            (EntityActionType::StopSleeping, 2),
            (EntityActionType::StartSprinting, 3),
            (EntityActionType::StopSprinting, 4),
            (EntityActionType::RidingJump, 5),
            (EntityActionType::OpenInventory, 6),
        ];

        for (action, ordinal) in actions {
            let packet = EntityAction::new(7, action);
            let mut buf = PacketBuffer::new();
            packet.write(&mut buf);
            assert_eq!(buf.as_slice(), &[7, ordinal, 0], "{action:?}");
        }

        // Java's horse path sends `(int)(power * 100.0F)`, so a non-zero value must survive as
        // the third VarInt rather than being silently forced to the default zero.
        let packet = EntityAction::riding_jump(7, 80);
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert_eq!(buf.as_slice(), &[7, 5, 80]);

        // The explicit constructor is also the escape hatch used by a caller that already did
        // the Java truncation.  This control prevents a future "all aux is zero" helper from
        // looking correct under round-trip-only coverage.
        let packet = EntityAction::with_aux_data(7, EntityActionType::RidingJump, 37);
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert_eq!(buf.as_slice(), &[7, 5, 37]);

        assert_eq!(
            EntityAction::stop_sleeping(7),
            EntityAction::new(7, EntityActionType::StopSleeping)
        );
        assert_eq!(
            EntityAction::open_inventory(7),
            EntityAction::new(7, EntityActionType::OpenInventory)
        );
    }

    /// A malformed ordinal is rejected instead of being silently mapped onto one of the live
    /// actions.  This is the negative control for the seven-entry constructor table.
    #[test]
    fn entity_action_unknown_ordinal_is_rejected() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(7);
        buf.write_varint(7);
        buf.write_varint(0);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            EntityAction::read(&mut read_buf),
            Err(ProtoError::InvalidEnum {
                what: "entity action",
                value: 7
            })
        ));
    }

    #[test]
    fn steer_vehicle_round_trips() {
        round_trip(SteerVehicle {
            sideways: -1.0,
            forward: 1.0,
            flags: 0b11,
        });
    }

    #[test]
    fn steer_vehicle_discards_non_vanilla_flag_bits() {
        let packet = SteerVehicle {
            sideways: -0.25,
            forward: 0.75,
            flags: 0b1111_0110,
        };
        let mut encoded = PacketBuffer::new();
        packet.write(&mut encoded);
        assert_eq!(encoded.as_slice().last(), Some(&0b10));

        let mut wire = PacketBuffer::new();
        wire.write_f32(packet.sideways);
        wire.write_f32(packet.forward);
        wire.write_u8(0b1111_0110);
        let mut input = PacketBuffer::from_slice(wire.as_slice());
        assert_eq!(SteerVehicle::read(&mut input).unwrap().flags, 0b10);
    }

    #[test]
    fn close_window_round_trips() {
        round_trip(CloseWindow { window_id: 1 });
    }

    #[test]
    fn click_window_round_trips() {
        round_trip(ClickWindow {
            window_id: 1,
            slot_id: 12,
            used_button: 0,
            action_number: 5,
            mode: 1,
            clicked_item: Some(ItemStack {
                item_id: 264,
                count: 3,
                damage: 0,
                nbt: None,
            }),
        });
        round_trip(ClickWindow {
            window_id: 0,
            slot_id: -999,
            used_button: 0,
            action_number: 1,
            mode: 4,
            clicked_item: None,
        });
    }

    #[test]
    fn confirm_transaction_round_trips() {
        round_trip(ConfirmTransaction {
            window_id: 1,
            action_id: -12,
            accepted: false,
        });
    }

    #[test]
    fn creative_inventory_action_round_trips() {
        round_trip(CreativeInventoryAction {
            slot_id: 36,
            stack: Some(ItemStack {
                item_id: 1,
                count: 64,
                damage: 0,
                nbt: None,
            }),
        });
        round_trip(CreativeInventoryAction {
            slot_id: 36,
            stack: None,
        });
    }

    #[test]
    fn enchant_item_round_trips() {
        round_trip(EnchantItem {
            window_id: 1,
            button: 2,
        });
    }

    #[test]
    fn update_sign_round_trips() {
        round_trip(UpdateSign {
            position: BlockPos::new(4, 70, 4),
            lines: [
                r#"{"text":"line one"}"#.into(),
                r#"{"text":""}"#.into(),
                r#"{"text":""}"#.into(),
                r#"{"text":"line four"}"#.into(),
            ],
        });
    }

    #[test]
    fn update_sign_writes_the_four_canonical_component_strings_in_order() {
        // `IChatComponent.Serializer.componentToJson` runs before this protocol packet is
        // constructed by the live sign editor.  Keep this codec oracle deliberately byte-facing:
        // it proves that the canonical primitive/object forms and their top-to-bottom order reach
        // the four bounded string fields unchanged.
        let packet = UpdateSign {
            position: BlockPos::new(-4, 70, 9),
            lines: [
                r#""top""#.into(),
                r#"{"bold":true,"text":"second"}"#.into(),
                r#""third""#.into(),
                r#"{"color":"red","text":"bottom"}"#.into(),
            ],
        };
        let mut encoded = PacketBuffer::new();
        packet.write(&mut encoded);
        assert!(encoded.finish_write().is_ok());

        let mut decoded = PacketBuffer::from_slice(encoded.as_slice());
        assert_eq!(decoded.read_block_pos().unwrap(), packet.position);
        let lines = core::array::from_fn(|_| decoded.read_string(384).unwrap());
        assert_eq!(lines, packet.lines);
        assert!(decoded.is_empty());
    }

    #[test]
    fn update_sign_preserves_gson_escapes_and_supplementary_utf8() {
        // C12's protocol layer deliberately has no dependency on the UI component tree.  The
        // sign editor hands it Gson-shaped strings, and this codec must carry those bytes
        // verbatim: in particular, it must not turn a Gson escape into a Rust string escape or
        // replace a supplementary scalar with a UTF-16 surrogate spelling.
        let packet = UpdateSign {
            position: BlockPos::new(0, -64, 0),
            lines: [
                r#""quote\\slash""#.into(),
                r#""html \u003c \u003e \u0026 \u003d \u0027""#.into(),
                "\"astral 😀\"".into(),
                r#"{"bold":true,"extra":["tail"],"text":"ordered"}"#.into(),
            ],
        };

        let mut encoded = PacketBuffer::new();
        packet.write(&mut encoded);
        assert!(encoded.finish_write().is_ok());

        let mut decoded = PacketBuffer::from_slice(encoded.as_slice());
        assert_eq!(decoded.read_block_pos().unwrap(), packet.position);
        let lines = core::array::from_fn(|_| decoded.read_string(384).unwrap());
        assert_eq!(lines, packet.lines);
        assert!(decoded.is_empty());
    }

    #[test]
    fn update_sign_read_uses_java_utf16_units_for_non_bmp_limit() {
        // Java's readStringFromBuffer(384) counts String.length(), i.e. UTF-16 code units.
        // The JSON quotes consume two units, leaving room for exactly 191 supplementary
        // scalars.  The next scalar is rejected before the following rows are read.
        let fits = format!("\"{}\"", "😀".repeat(191));
        assert_eq!(fits.encode_utf16().count(), 384);
        let too_long = format!("\"{}\"", "😀".repeat(192));
        assert_eq!(too_long.encode_utf16().count(), 386);

        let mut wire = PacketBuffer::new();
        wire.write_block_pos(BlockPos::ORIGIN);
        wire.write_string(&fits);
        for _ in 0..3 {
            wire.write_string("\"\"");
        }
        let mut read_buf = PacketBuffer::from_slice(wire.as_slice());
        assert_eq!(UpdateSign::read(&mut read_buf).unwrap().lines[0], fits);

        let mut too_long_wire = PacketBuffer::new();
        too_long_wire.write_block_pos(BlockPos::ORIGIN);
        too_long_wire.write_string(&too_long);
        for _ in 0..3 {
            too_long_wire.write_string("\"\"");
        }
        let mut read_buf = PacketBuffer::from_slice(too_long_wire.as_slice());
        assert!(matches!(
            UpdateSign::read(&mut read_buf),
            Err(ProtoError::StringTooLong { max: 384, .. })
        ));
    }

    #[test]
    fn update_sign_rejects_a_line_over_384_utf16_units() {
        let mut buf = PacketBuffer::new();
        buf.write_block_pos(BlockPos::new(4, 70, 4));
        buf.write_string(&"x".repeat(385));
        for _ in 0..3 {
            buf.write_string(r#"{"text":""}"#);
        }
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            UpdateSign::read(&mut read_buf),
            Err(ProtoError::StringTooLong { max: 384, .. })
        ));
    }

    #[test]
    fn player_abilities_round_trips() {
        round_trip(PlayerAbilities {
            invulnerable: true,
            flying: true,
            allow_flying: true,
            creative_mode: true,
            fly_speed: 0.05,
            walk_speed: 0.1,
        });
        round_trip(PlayerAbilities {
            invulnerable: false,
            flying: false,
            allow_flying: false,
            creative_mode: false,
            fly_speed: 0.05,
            walk_speed: 0.1,
        });
    }

    #[test]
    fn tab_complete_round_trips_with_and_without_target() {
        round_trip(TabComplete {
            message: "/setblock ~ ~ ~ ".into(),
            target_block: None,
        });
        round_trip(TabComplete {
            message: "/setblock ~ ~ ~ ".into(),
            target_block: Some(BlockPos::new(1, 2, 3)),
        });
    }

    #[test]
    fn tab_complete_write_truncates_at_java_utf16_limit() {
        let packet = TabComplete {
            message: "x".repeat(32_768),
            target_block: None,
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert!(buf.finish_write().is_ok());

        let mut encoded = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(encoded.read_string(32_767).unwrap(), "x".repeat(32_767));
        assert!(!encoded.read_bool().unwrap());
        assert!(encoded.is_empty());
    }

    #[test]
    fn tab_complete_write_replaces_a_surrogate_cut_and_keeps_target() {
        // Java substring(0, 32767) keeps the high surrogate when this supplementary scalar is
        // the next character after 32766 UTF-16 units; Java's UTF-8 encoder emits '?' for it.
        let packet = TabComplete {
            message: format!("{}💣tail", "x".repeat(32_766)),
            target_block: Some(BlockPos::new(1, 2, 3)),
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert!(buf.finish_write().is_ok());

        let mut encoded = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            encoded.read_string(32_767).unwrap(),
            format!("{}?", "x".repeat(32_766))
        );
        assert!(encoded.read_bool().unwrap());
        assert_eq!(encoded.read_block_pos().unwrap(), BlockPos::new(1, 2, 3));
        assert!(encoded.is_empty());
    }

    #[test]
    fn tab_complete_write_preserves_java_string_byte_limit_after_truncation() {
        // The UTF-16 prefix is legal (well below 32767 units), but its UTF-8 encoding is 32768
        // bytes. Java's subsequent PacketBuffer.writeString throws; Rust defers that error and
        // does not append the optional-target fields.
        let packet = TabComplete {
            message: "😀".repeat(8_192),
            target_block: Some(BlockPos::new(1, 2, 3)),
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert!(matches!(
            buf.finish_write(),
            Err(ProtoError::StringTooLong {
                len: 32_768,
                max: 32_767
            })
        ));
        assert!(buf.is_empty());
    }

    #[test]
    fn client_settings_round_trips() {
        round_trip(ClientSettings {
            locale: "en_US".into(),
            view_distance: 12,
            chat_visibility: ChatVisibility::System,
            chat_colors: true,
            displayed_skin_parts: 0x7F,
        });
    }

    #[test]
    fn client_settings_view_distance_round_trips_signed_byte_boundaries_without_wire_changes() {
        for (view_distance, wire_byte) in [(-128_i8, 0x80_u8), (-1, 0xFF), (127, 0x7F)] {
            let settings = ClientSettings {
                locale: "en_US".into(),
                view_distance,
                chat_visibility: ChatVisibility::System,
                chat_colors: true,
                displayed_skin_parts: 0x7F,
            };
            let expected_wire = [5, b'e', b'n', b'_', b'U', b'S', wire_byte, 1, 1, 0x7F];

            let mut write_buf = PacketBuffer::new();
            settings.write(&mut write_buf);
            assert_eq!(write_buf.as_slice(), expected_wire);

            let mut read_buf = PacketBuffer::from_slice(&expected_wire);
            assert_eq!(ClientSettings::read(&mut read_buf).unwrap(), settings);
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn client_settings_chat_visibility_wraps_nonnegative_java_ordinals() {
        for (ordinal, expected) in [
            (3_i8, ChatVisibility::Full),
            (4, ChatVisibility::System),
            (5, ChatVisibility::Hidden),
            (127, ChatVisibility::System),
        ] {
            let mut buf = PacketBuffer::new();
            buf.write_string("en_US");
            buf.write_i8(12);
            buf.write_i8(ordinal);
            buf.write_bool(true);
            buf.write_u8(0x7F);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            assert_eq!(
                ClientSettings::read(&mut read_buf).unwrap().chat_visibility,
                expected,
                "ordinal {ordinal}"
            );
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn bad_chat_visibility_negative_java_ordinal_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_string("en_US");
        buf.write_u8(12);
        buf.write_i8(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            ClientSettings::read(&mut read_buf),
            Err(ProtoError::InvalidEnum {
                what: "chat visibility",
                value: -1
            })
        ));
    }

    #[test]
    fn client_status_round_trips_every_action() {
        round_trip(ClientStatus {
            action: ClientStatusAction::PerformRespawn,
        });
        round_trip(ClientStatus {
            action: ClientStatusAction::RequestStats,
        });
        round_trip(ClientStatus {
            action: ClientStatusAction::OpenInventoryAchievement,
        });
    }

    #[test]
    fn plugin_message_round_trips() {
        round_trip(PluginMessage {
            channel: "MC|Brand".into(),
            data: b"vanilla".to_vec(),
        });
        round_trip(PluginMessage {
            channel: "MC|Brand".into(),
            data: Vec::new(),
        });
    }

    #[test]
    fn plugin_message_rejects_payload_over_32767_bytes_on_read_and_write() {
        let data = vec![0_u8; 32_768];
        let mut wire = PacketBuffer::new();
        wire.write_string("MC|Brand");
        wire.write_bytes(&data);
        let mut read_buf = PacketBuffer::from_slice(wire.as_slice());
        assert!(matches!(
            PluginMessage::read(&mut read_buf),
            Err(ProtoError::PayloadTooLong {
                len: 32_768,
                max: 32_767
            })
        ));

        let mut write_buf = PacketBuffer::new();
        PluginMessage {
            channel: "MC|Brand".into(),
            data,
        }
        .write(&mut write_buf);
        assert!(matches!(
            write_buf.finish_write(),
            Err(ProtoError::PayloadTooLong {
                len: 32_768,
                max: 32_767
            })
        ));
    }

    #[test]
    fn spectate_round_trips() {
        round_trip(Spectate {
            target: Uuid::from_u128(0x1111_2222_3333_4444_5555_6666_7777_8888),
        });
    }

    #[test]
    fn resource_pack_status_round_trips_every_result() {
        let results = [
            ResourcePackStatusResult::SuccessfullyLoaded,
            ResourcePackStatusResult::Declined,
            ResourcePackStatusResult::FailedDownload,
            ResourcePackStatusResult::Accepted,
        ];
        for result in results {
            round_trip(ResourcePackStatus {
                hash: "0123456789abcdef0123456789abcdef01234567"
                    .chars()
                    .take(40)
                    .collect(),
                result,
            });
        }
    }

    #[test]
    fn resource_pack_status_write_truncates_hash_to_40_utf16_units() {
        let packet = ResourcePackStatus {
            hash: format!("{}💣tail", "x".repeat(39)),
            result: ResourcePackStatusResult::Accepted,
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            read_buf.read_string(40).unwrap(),
            format!("{}?", "x".repeat(39))
        );
        assert_eq!(read_buf.read_varint().unwrap(), 3);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn bad_resource_pack_status_result_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_string("hash");
        buf.write_varint(99);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            ResourcePackStatus::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { .. })
        ));
    }
}
