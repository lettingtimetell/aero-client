//! Protocol-5 serverbound Play packets `0x00`-`0x17`, one struct per packet, and the exact
//! conversion from the 1.8.9 model ([`super::play::C2SPlay5::from_v47`]).
//!
//! Each `write` is the 1.7.10 class's `writePacketData` and each `read` its `readPacketData`
//! (the server's view, used by the tests). Fields keep their wire width; values the 1.7.10
//! client writes as sentinels (C07's face `-1`/`255`, C08's `(-1, -1, -1, 255)`) are therefore
//! representable as they are.
//!
//! java17: `network/play/client/C00PacketKeepAlive.java` .. `C17PacketCustomPayload.java`

use mc_math::BlockPos;
use thiserror::Error;

use super::buffer::PacketBufferV5;
use super::nbt::NbtBudget;
use crate::item::ItemStack;
use crate::packet::Packet;
use crate::play::c2s as v47;
use crate::play::C2SPlay;
use crate::{PacketBuffer, ProtoError};

/// `java.lang.String` length limit that `C17PacketCustomPayload`'s constructor enforces:
/// `length >= 32767` throws `IllegalArgumentException`.
pub const MAX_PLUGIN_PAYLOAD: usize = 32766;

/// `0x00 Keep Alive`: the server's id echoed as an `int`. java17: `C00PacketKeepAlive`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeepAlive {
    /// The id from S00.
    pub key: i32,
}

impl Packet for KeepAlive {
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(KeepAlive {
            key: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.key);
    }
}

/// `0x01 Chat Message`. java17: `C01PacketChatMessage` (its constructor keeps the first 100
/// chars; the server reads at most 100).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    /// The message or command.
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

/// [`UseEntity`]'s action. java17: `C02PacketUseEntity.Action`; there is no `INTERACT_AT`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseEntityAction {
    /// Id 0: right click.
    Interact,
    /// Id 1: left click.
    Attack,
}

/// `0x02 Use Entity`: an `int` entity id and a byte action. java17: `C02PacketUseEntity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UseEntity {
    /// The target.
    pub entity_id: i32,
    /// The action.
    pub action: UseEntityAction,
}

impl Packet for UseEntity {
    const ID: i32 = 0x02;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_i32()?;
        // java17: ACTIONS[readByte() % ACTIONS.length]; a negative remainder throws.
        let raw = buf.read_i8()?;
        let action = match raw % 2 {
            0 => UseEntityAction::Interact,
            1 => UseEntityAction::Attack,
            _ => {
                return Err(ProtoError::InvalidEnum {
                    what: "use-entity action",
                    value: i32::from(raw),
                })
            }
        };
        Ok(UseEntity { entity_id, action })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_u8(match self.action {
            UseEntityAction::Interact => 0,
            UseEntityAction::Attack => 1,
        });
    }
}

/// `0x03 Player`: on-ground only. java17: `C03PacketPlayer`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Player {
    /// On ground (`readUnsignedByte() != 0`).
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

/// `0x04 Player Position`: `x`, feet `y`, `stance` (eye height, the player's `posY`), `z`.
/// java17: `C03PacketPlayer.C04PacketPlayerPosition`, built from
/// `posX, boundingBox.minY, posY, posZ` (`EntityClientPlayerMP.sendMotionUpdates`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPosition {
    /// X.
    pub x: f64,
    /// Feet Y (`boundingBox.minY`).
    pub y: f64,
    /// Eye Y (`posY`); the server kicks when `stance - y` is outside `[0.1, 1.65]`.
    pub stance: f64,
    /// Z.
    pub z: f64,
    /// On ground.
    pub on_ground: bool,
}

impl Packet for PlayerPosition {
    const ID: i32 = 0x04;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerPosition {
            x: buf.read_f64()?,
            y: buf.read_f64()?,
            stance: buf.read_f64()?,
            z: buf.read_f64()?,
            on_ground: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f64(self.x);
        buf.write_f64(self.y);
        buf.write_f64(self.stance);
        buf.write_f64(self.z);
        buf.write_bool(self.on_ground);
    }
}

/// `0x05 Player Look`. java17: `C03PacketPlayer.C05PacketPlayerLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerLook {
    /// Yaw.
    pub yaw: f32,
    /// Pitch.
    pub pitch: f32,
    /// On ground.
    pub on_ground: bool,
}

impl Packet for PlayerLook {
    const ID: i32 = 0x05;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerLook {
            yaw: buf.read_f32()?,
            pitch: buf.read_f32()?,
            on_ground: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.yaw);
        buf.write_f32(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x06 Player Position And Look`: as [`PlayerPosition`] plus yaw and pitch. Also the reply to
/// S08, which echoes the packet's position, look and `onGround`.
/// java17: `C03PacketPlayer.C06PacketPlayerPosLook`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerPositionAndLook {
    /// X.
    pub x: f64,
    /// Feet Y.
    pub y: f64,
    /// Eye Y.
    pub stance: f64,
    /// Z.
    pub z: f64,
    /// Yaw.
    pub yaw: f32,
    /// Pitch.
    pub pitch: f32,
    /// On ground.
    pub on_ground: bool,
}

impl Packet for PlayerPositionAndLook {
    const ID: i32 = 0x06;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerPositionAndLook {
            x: buf.read_f64()?,
            y: buf.read_f64()?,
            stance: buf.read_f64()?,
            z: buf.read_f64()?,
            yaw: buf.read_f32()?,
            pitch: buf.read_f32()?,
            on_ground: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f64(self.x);
        buf.write_f64(self.y);
        buf.write_f64(self.stance);
        buf.write_f64(self.z);
        buf.write_f32(self.yaw);
        buf.write_f32(self.pitch);
        buf.write_bool(self.on_ground);
    }
}

/// `0x07 Player Digging`: every field a raw byte except the `int` x and z.
/// java17: `C07PacketPlayerDigging`
///
/// The 1.7.10 client writes `face = -1` (`0xFF`) when `PlayerControllerMP.resetBlockRemoving`
/// aborts, and `(5, 0, 0, 0, 255)` from `onStoppedUsingItem`; both are `face: 0xFF` here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerDigging {
    /// 0 start, 1 abort, 2 finish, 3 drop stack, 4 drop item, 5 release use item.
    pub status: u8,
    /// X.
    pub x: i32,
    /// Y (low eight bits).
    pub y: u8,
    /// Z.
    pub z: i32,
    /// Face 0-5, or the `0xFF` sentinel.
    pub face: u8,
}

impl PlayerDigging {
    /// The block, as the server reads it (`y` unsigned).
    #[must_use]
    pub fn block_pos(&self) -> BlockPos {
        BlockPos::new(self.x, i32::from(self.y), self.z)
    }
}

impl Packet for PlayerDigging {
    const ID: i32 = 0x07;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(PlayerDigging {
            status: buf.read_u8()?,
            x: buf.read_i32()?,
            y: buf.read_u8()?,
            z: buf.read_i32()?,
            face: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.status);
        buf.write_i32(self.x);
        buf.write_u8(self.y);
        buf.write_i32(self.z);
        buf.write_u8(self.face);
    }
}

/// `0x08 Player Block Placement`. java17: `C08PacketPlayerBlockPlacement`
///
/// Using an item in the air is `(-1, -1, -1)` with direction 255 (the `y` byte is then `0xFF`).
#[derive(Debug, Clone, PartialEq)]
pub struct PlayerBlockPlacement {
    /// X.
    pub x: i32,
    /// Y (low eight bits).
    pub y: u8,
    /// Z.
    pub z: i32,
    /// Face 0-5, or 255 for "not at a block".
    pub direction: u8,
    /// The held stack.
    pub stack: Option<ItemStack>,
    /// `(int)(hitX * 16)`.
    pub cursor_x: u8,
    /// `(int)(hitY * 16)`.
    pub cursor_y: u8,
    /// `(int)(hitZ * 16)`.
    pub cursor_z: u8,
}

impl Packet for PlayerBlockPlacement {
    const ID: i32 = 0x08;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(PlayerBlockPlacement {
            x: buf.read_i32()?,
            y: buf.read_u8()?,
            z: buf.read_i32()?,
            direction: buf.read_u8()?,
            stack: buf.read_item_v5(&mut budget)?,
            cursor_x: buf.read_u8()?,
            cursor_y: buf.read_u8()?,
            cursor_z: buf.read_u8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.x);
        buf.write_u8(self.y);
        buf.write_i32(self.z);
        buf.write_u8(self.direction);
        buf.write_item_v5(self.stack.as_ref());
        buf.write_u8(self.cursor_x);
        buf.write_u8(self.cursor_y);
        buf.write_u8(self.cursor_z);
    }
}

/// `0x09 Held Item Change`. java17: `C09PacketHeldItemChange`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeldItemChange {
    /// Hotbar slot.
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

/// `0x0A Animation`: the player's own entity id and type 1 (swing).
/// java17: `C0APacketAnimation`, sent by `EntityClientPlayerMP.swingItem`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Animation {
    /// The local player's entity id.
    pub entity_id: i32,
    /// 1 = swing arm, the only value the client sends.
    pub animation_type: i8,
}

impl Packet for Animation {
    const ID: i32 = 0x0A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Animation {
            entity_id: buf.read_i32()?,
            animation_type: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.animation_type);
    }
}

/// `0x0B Entity Action`: `int` entity id, byte action **1-based**, `int` aux.
/// java17: `C0BPacketEntityAction`; ids 1 sneak, 2 unsneak, 3 leave bed, 4 sprint, 5 stop
/// sprinting, 6 horse jump (aux = jump power * 100), 7 open horse inventory
/// (`EntityClientPlayerMP`, `GuiSleepMP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityAction {
    /// The local player's entity id.
    pub entity_id: i32,
    /// The action id, 1-7.
    pub action_id: i8,
    /// Horse jump power, else 0.
    pub aux_data: i32,
}

impl Packet for EntityAction {
    const ID: i32 = 0x0B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EntityAction {
            entity_id: buf.read_i32()?,
            action_id: buf.read_i8()?,
            aux_data: buf.read_i32()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.entity_id);
        buf.write_i8(self.action_id);
        buf.write_i32(self.aux_data);
    }
}

/// `0x0C Steer Vehicle`: two booleans instead of a flags byte. java17: `C0CPacketInput`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SteerVehicle {
    /// Strafe speed.
    pub strafe: f32,
    /// Forward speed.
    pub forward: f32,
    /// Jumping.
    pub jump: bool,
    /// Sneaking (dismount).
    pub sneak: bool,
}

impl Packet for SteerVehicle {
    const ID: i32 = 0x0C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SteerVehicle {
            strafe: buf.read_f32()?,
            forward: buf.read_f32()?,
            jump: buf.read_bool()?,
            sneak: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_f32(self.strafe);
        buf.write_f32(self.forward);
        buf.write_bool(self.jump);
        buf.write_bool(self.sneak);
    }
}

/// `0x0D Close Window`. java17: `C0DPacketCloseWindow`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseWindow {
    /// Window id.
    pub window_id: i8,
}

impl Packet for CloseWindow {
    const ID: i32 = 0x0D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(CloseWindow {
            window_id: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
    }
}

/// `0x0E Click Window`: the 1.8.9 layout (`actionNumber` before `mode`) with a protocol-5 slot.
/// java17: `C0EPacketClickWindow`
#[derive(Debug, Clone, PartialEq)]
pub struct ClickWindow {
    /// Window id.
    pub window_id: i8,
    /// Slot (-999 outside the window).
    pub slot_id: i16,
    /// Mouse button, or hotbar key for mode 2.
    pub used_button: i8,
    /// Transaction id.
    pub action_number: i16,
    /// Click mode 0-6.
    pub mode: i8,
    /// The clicked slot's stack as the client predicted it.
    pub clicked_item: Option<ItemStack>,
}

impl Packet for ClickWindow {
    const ID: i32 = 0x0E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(ClickWindow {
            window_id: buf.read_i8()?,
            slot_id: buf.read_i16()?,
            used_button: buf.read_i8()?,
            action_number: buf.read_i16()?,
            mode: buf.read_i8()?,
            clicked_item: buf.read_item_v5(&mut budget)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
        buf.write_i16(self.slot_id);
        buf.write_i8(self.used_button);
        buf.write_i16(self.action_number);
        buf.write_i8(self.mode);
        buf.write_item_v5(self.clicked_item.as_ref());
    }
}

/// `0x0F Confirm Transaction`. java17: `C0FPacketConfirmTransaction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfirmTransaction {
    /// Window id.
    pub window_id: i8,
    /// Transaction id.
    pub action_number: i16,
    /// Accepted (`writeByte(accepted ? 1 : 0)`).
    pub accepted: bool,
}

impl Packet for ConfirmTransaction {
    const ID: i32 = 0x0F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ConfirmTransaction {
            window_id: buf.read_i8()?,
            action_number: buf.read_i16()?,
            accepted: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
        buf.write_i16(self.action_number);
        buf.write_bool(self.accepted);
    }
}

/// `0x10 Creative Inventory Action`. java17: `C10PacketCreativeInventoryAction`
#[derive(Debug, Clone, PartialEq)]
pub struct CreativeInventoryAction {
    /// Slot (-1 drops the stack).
    pub slot_id: i16,
    /// The stack.
    pub stack: Option<ItemStack>,
}

impl Packet for CreativeInventoryAction {
    const ID: i32 = 0x10;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let mut budget = NbtBudget::per_packet();
        Ok(CreativeInventoryAction {
            slot_id: buf.read_i16()?,
            stack: buf.read_item_v5(&mut budget)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i16(self.slot_id);
        buf.write_item_v5(self.stack.as_ref());
    }
}

/// `0x11 Enchant Item`. java17: `C11PacketEnchantItem`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnchantItem {
    /// Window id.
    pub window_id: i8,
    /// Enchantment button 0-2.
    pub button: i8,
}

impl Packet for EnchantItem {
    const ID: i32 = 0x11;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EnchantItem {
            window_id: buf.read_i8()?,
            button: buf.read_i8()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
        buf.write_i8(self.button);
    }
}

/// `0x12 Update Sign`: four plain-text lines (the server reads at most 15 chars each).
/// java17: `C12PacketUpdateSign`
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

impl Packet for UpdateSign {
    const ID: i32 = 0x12;

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

/// `0x13 Player Abilities`. java17: `C13PacketPlayerAbilities`
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
    const ID: i32 = 0x13;

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

/// `0x14 Tab Complete`: the text only, no target block. java17: `C14PacketTabComplete`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabComplete {
    /// The text to complete; written as `StringUtils.substring(message, 0, 32767)`.
    pub message: String,
}

impl Packet for TabComplete {
    const ID: i32 = 0x14;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(TabComplete {
            message: buf.read_string(32767)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        write_java_utf16_prefix(buf, &self.message, 32767);
    }
}

/// `0x15 Client Settings`: difficulty and a show-cape flag where 1.8.9 has skin parts.
/// java17: `C15PacketClientSettings`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClientSettings {
    /// Language code, e.g. `en_US` (the server reads at most 7 chars).
    pub locale: String,
    /// Render distance in chunks.
    pub view_distance: i8,
    /// 0 full, 1 system, 2 hidden (`EnumChatVisibility`).
    pub chat_visibility: u8,
    /// Chat colours.
    pub chat_colors: bool,
    /// The client's own difficulty option, 0-3.
    pub difficulty: u8,
    /// Show cape.
    pub show_cape: bool,
}

impl Packet for ClientSettings {
    const ID: i32 = 0x15;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let locale = buf.read_string(7)?;
        let view_distance = buf.read_i8()?;
        // java17: getEnumChatVisibility(b) and getDifficultyEnum(b) index `values[b % n]`.
        let chat_visibility = java_enum_index(buf.read_i8()?, 3, "chat visibility")?;
        let chat_colors = buf.read_bool()?;
        let difficulty = java_enum_index(buf.read_i8()?, 4, "difficulty")?;
        Ok(ClientSettings {
            locale,
            view_distance,
            chat_visibility,
            chat_colors,
            difficulty,
            show_cape: buf.read_bool()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.locale);
        buf.write_i8(self.view_distance);
        buf.write_u8(self.chat_visibility);
        buf.write_bool(self.chat_colors);
        buf.write_u8(self.difficulty);
        buf.write_bool(self.show_cape);
    }
}

/// `0x16 Client Status`: a byte. java17: `C16PacketClientStatus`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClientStatus {
    /// 0 respawn, 1 request statistics, 2 open inventory achievement.
    pub status: u8,
}

impl Packet for ClientStatus {
    const ID: i32 = 0x16;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ClientStatus {
            status: java_enum_index(buf.read_i8()?, 3, "client status")?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.status);
    }
}

/// `0x17 Plugin Message`: an `i16` length before the payload. java17: `C17PacketCustomPayload`
///
/// The payload formats are the 1.7.10 ones: `MC|Brand` and `MC|ItemName` are raw UTF-8 with no
/// length prefix, and payloads vanilla builds in a Netty buffer go out as its whole backing
/// array. Building them is the caller's job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMessage {
    /// Channel (the server reads at most 20 chars).
    pub channel: String,
    /// Payload, at most [`MAX_PLUGIN_PAYLOAD`] bytes; a longer one is a deferred error, like
    /// the constructor's `IllegalArgumentException`.
    pub data: Vec<u8>,
}

impl Packet for PluginMessage {
    const ID: i32 = 0x17;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let channel = buf.read_string(20)?;
        let len = buf.read_i16()?;
        // java17: the payload is read only for 0 < length < 32767.
        let data = if len > 0 && len < i16::MAX {
            buf.read_bytes(len as usize)?
        } else {
            Vec::new()
        };
        Ok(PluginMessage { channel, data })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        if self.data.len() > MAX_PLUGIN_PAYLOAD {
            buf.write_bounded_payload(&self.data, MAX_PLUGIN_PAYLOAD);
            return;
        }
        buf.write_string(&self.channel);
        buf.write_i16(self.data.len() as i16);
        buf.write_bytes(&self.data);
    }
}

/// `values[b % n]` with Java's remainder: a negative remainder is an
/// `ArrayIndexOutOfBoundsException`.
fn java_enum_index(raw: i8, n: i8, what: &'static str) -> Result<u8, ProtoError> {
    let index = raw % n;
    if index < 0 {
        return Err(ProtoError::InvalidEnum {
            what,
            value: i32::from(raw),
        });
    }
    Ok(index as u8)
}

/// `writeStringToBuffer(StringUtils.substring(value, 0, max_utf16))`: cut at `max_utf16` UTF-16
/// units (a cut surrogate pair becomes `?`, as `String.getBytes(UTF_8)` encodes a lone
/// surrogate), then the 32767-byte limit as a deferred error.
fn write_java_utf16_prefix(buf: &mut PacketBuffer, value: &str, max_utf16: usize) {
    let mut cut = String::with_capacity(value.len().min(max_utf16));
    let mut used = 0;
    for ch in value.chars() {
        let units = ch.len_utf16();
        if used + units > max_utf16 {
            if used < max_utf16 {
                cut.push('?');
            }
            break;
        }
        cut.push(ch);
        used += units;
    }
    buf.write_string(&cut);
}

/// What [`super::play::C2SPlay5::from_v47`] needs that the 1.8.9 packets do not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V5Ctx {
    /// The local player's entity id, which protocol-5 C0A carries.
    pub entity_id: i32,
}

/// A 1.8.9 packet with no exact protocol-5 equivalent; sending it would be wrong, so it is
/// refused. 1.7.10 code builds these packets in their protocol-5 form directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("{packet} has no exact protocol-5 form: {reason}")]
pub struct NotInProtocol5 {
    /// The 1.8.9 packet.
    pub packet: &'static str,
    /// Why.
    pub reason: &'static str,
}

/// [`super::play::C2SPlay5::from_v47`]'s body; see there.
pub(super) fn from_v47(
    packet: &C2SPlay,
    ctx: &V5Ctx,
) -> Result<super::play::C2SPlay5, NotInProtocol5> {
    use super::play::C2SPlay5 as P5;
    let refuse = |packet, reason| Err(NotInProtocol5 { packet, reason });
    Ok(match packet {
        C2SPlay::KeepAlive(p) => P5::KeepAlive(KeepAlive { key: p.key }),
        C2SPlay::ChatMessage(p) => P5::ChatMessage(ChatMessage {
            message: p.message.clone(),
        }),
        C2SPlay::UseEntity(p) => P5::UseEntity(UseEntity {
            entity_id: p.entity_id,
            action: match p.action {
                v47::UseEntityAction::Interact => UseEntityAction::Interact,
                v47::UseEntityAction::Attack => UseEntityAction::Attack,
                v47::UseEntityAction::InteractAt { .. } => {
                    return refuse("C02 UseEntity", "INTERACT_AT does not exist in protocol 5")
                }
            },
        }),
        C2SPlay::Player(p) => P5::Player(Player {
            on_ground: p.on_ground,
        }),
        C2SPlay::PlayerPosition(_) => {
            return refuse("C04 PlayerPosition", "protocol 5 needs feet y and stance")
        }
        C2SPlay::PlayerLook(p) => P5::PlayerLook(PlayerLook {
            yaw: p.yaw,
            pitch: p.pitch,
            on_ground: p.on_ground,
        }),
        C2SPlay::PlayerPositionAndLook(_) => {
            return refuse(
                "C06 PlayerPositionAndLook",
                "protocol 5 needs feet y and stance",
            )
        }
        C2SPlay::PlayerDigging(_) => {
            return refuse(
                "C07 PlayerDigging",
                "1.7.10 sends face -1/255 where 1.8.9 sends DOWN; build it in protocol-5 form",
            )
        }
        C2SPlay::PlayerBlockPlacement(p) => {
            // The server reads y unsigned; -1 (the use-item sentinel) and 0..=255 survive the
            // byte, nothing else does.
            if !(-1..=255).contains(&p.position.y) {
                return refuse("C08 PlayerBlockPlacement", "y does not fit the byte");
            }
            P5::PlayerBlockPlacement(PlayerBlockPlacement {
                x: p.position.x,
                y: p.position.y as u8,
                z: p.position.z,
                direction: p.placed_block_direction,
                stack: p.stack.clone(),
                cursor_x: p.cursor_x,
                cursor_y: p.cursor_y,
                cursor_z: p.cursor_z,
            })
        }
        C2SPlay::HeldItemChange(p) => P5::HeldItemChange(HeldItemChange { slot_id: p.slot_id }),
        C2SPlay::Animation(_) => P5::Animation(Animation {
            entity_id: ctx.entity_id,
            animation_type: 1,
        }),
        C2SPlay::EntityAction(p) => P5::EntityAction(EntityAction {
            entity_id: p.entity_id,
            action_id: match p.action {
                v47::EntityActionType::StartSneaking => 1,
                v47::EntityActionType::StopSneaking => 2,
                v47::EntityActionType::StopSleeping => 3,
                v47::EntityActionType::StartSprinting => 4,
                v47::EntityActionType::StopSprinting => 5,
                v47::EntityActionType::RidingJump => 6,
                v47::EntityActionType::OpenInventory => 7,
            },
            aux_data: p.aux_data,
        }),
        C2SPlay::SteerVehicle(p) => {
            if p.flags & !3 != 0 {
                return refuse("C0C SteerVehicle", "flags beyond jump and unmount");
            }
            P5::SteerVehicle(SteerVehicle {
                strafe: p.sideways,
                forward: p.forward,
                jump: p.flags & 1 != 0,
                sneak: p.flags & 2 != 0,
            })
        }
        C2SPlay::CloseWindow(p) => P5::CloseWindow(CloseWindow {
            window_id: p.window_id as i8,
        }),
        C2SPlay::ClickWindow(p) => P5::ClickWindow(ClickWindow {
            window_id: p.window_id as i8,
            slot_id: p.slot_id,
            used_button: p.used_button as i8,
            action_number: p.action_number,
            mode: p.mode as i8,
            clicked_item: p.clicked_item.clone(),
        }),
        C2SPlay::ConfirmTransaction(p) => P5::ConfirmTransaction(ConfirmTransaction {
            window_id: p.window_id as i8,
            action_number: p.action_id,
            accepted: p.accepted,
        }),
        C2SPlay::CreativeInventoryAction(p) => {
            P5::CreativeInventoryAction(CreativeInventoryAction {
                slot_id: p.slot_id,
                stack: p.stack.clone(),
            })
        }
        C2SPlay::EnchantItem(p) => P5::EnchantItem(EnchantItem {
            window_id: p.window_id as i8,
            button: p.button as i8,
        }),
        C2SPlay::UpdateSign(_) => {
            return refuse(
                "C12 UpdateSign",
                "protocol 5 sends plain lines, not chat JSON",
            )
        }
        C2SPlay::PlayerAbilities(p) => P5::PlayerAbilities(PlayerAbilities {
            invulnerable: p.invulnerable,
            flying: p.flying,
            allow_flying: p.allow_flying,
            creative_mode: p.creative_mode,
            fly_speed: p.fly_speed,
            walk_speed: p.walk_speed,
        }),
        C2SPlay::TabComplete(p) => {
            if p.target_block.is_some() {
                return refuse("C14 TabComplete", "protocol 5 has no target block");
            }
            P5::TabComplete(TabComplete {
                message: p.message.clone(),
            })
        }
        C2SPlay::ClientSettings(_) => {
            return refuse(
                "C15 ClientSettings",
                "protocol 5 sends difficulty and show-cape, not skin parts",
            )
        }
        C2SPlay::ClientStatus(p) => P5::ClientStatus(ClientStatus {
            status: match p.action {
                v47::ClientStatusAction::PerformRespawn => 0,
                v47::ClientStatusAction::RequestStats => 1,
                v47::ClientStatusAction::OpenInventoryAchievement => 2,
            },
        }),
        C2SPlay::PluginMessage(_) => {
            return refuse(
                "C17 PluginMessage",
                "payload formats differ by channel (MC|Brand is raw UTF-8 in 1.7.10)",
            )
        }
        C2SPlay::Spectate(_) => return refuse("C18 Spectate", "1.8.9 only"),
        C2SPlay::ResourcePackStatus(_) => return refuse("C19 ResourcePackStatus", "1.8.9 only"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode<P: Packet>(packet: &P) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        buf.finish_write().unwrap();
        buf.as_slice().to_vec()
    }

    fn decode<P: Packet>(bytes: &[u8]) -> P {
        let mut buf = PacketBuffer::from_slice(bytes);
        let packet = P::read(&mut buf).unwrap();
        assert!(buf.is_empty());
        packet
    }

    #[test]
    fn digging_sentinels_are_plain_bytes() {
        let abort = PlayerDigging {
            status: 1,
            x: -5,
            y: 64,
            z: 9,
            face: 0xFF,
        };
        assert_eq!(
            encode(&abort),
            [1, 0xFF, 0xFF, 0xFF, 0xFB, 64, 0, 0, 0, 9, 0xFF]
        );
        let release = PlayerDigging {
            status: 5,
            x: 0,
            y: 0,
            z: 0,
            face: 255,
        };
        assert_eq!(decode::<PlayerDigging>(&encode(&release)), release);
    }

    #[test]
    fn movement_carries_stance() {
        let packet = PlayerPositionAndLook {
            x: 1.5,
            y: 64.0,
            stance: 65.62,
            z: -2.5,
            yaw: 90.0,
            pitch: 0.0,
            on_ground: true,
        };
        let bytes = encode(&packet);
        assert_eq!(bytes.len(), 8 * 4 + 4 * 2 + 1);
        assert_eq!(&bytes[16..24], &65.62f64.to_be_bytes());
        assert_eq!(decode::<PlayerPositionAndLook>(&bytes), packet);
    }

    #[test]
    fn plugin_payload_is_short_prefixed_and_capped_below_32767() {
        let brand = PluginMessage {
            channel: "MC|Brand".into(),
            data: b"Aero".to_vec(),
        };
        let bytes = encode(&brand);
        assert_eq!(&bytes[9..], &[0, 4, b'A', b'e', b'r', b'o']);
        assert_eq!(decode::<PluginMessage>(&bytes), brand);

        let mut buf = PacketBuffer::new();
        PluginMessage {
            channel: "x".into(),
            data: vec![0; MAX_PLUGIN_PAYLOAD + 1],
        }
        .write(&mut buf);
        assert!(buf.finish_write().is_err());
        assert!(buf.as_slice().is_empty());
        let ok = PluginMessage {
            channel: "x".into(),
            data: vec![1; MAX_PLUGIN_PAYLOAD],
        };
        assert_eq!(decode::<PluginMessage>(&encode(&ok)), ok);
    }

    #[test]
    fn java_remainder_indexing() {
        // -2 % 2 == 0 is INTERACT in java; -1 % 2 == -1 throws.
        assert_eq!(
            decode::<UseEntity>(&[0, 0, 0, 1, 0xFE]).action,
            UseEntityAction::Interact
        );
        assert!(UseEntity::read(&mut PacketBuffer::from_slice(&[0, 0, 0, 1, 0xFF])).is_err());
        assert_eq!(decode::<ClientStatus>(&[5]).status, 2);
        assert!(ClientStatus::read(&mut PacketBuffer::from_slice(&[0xFF])).is_err());
    }

    #[test]
    fn tab_complete_cuts_at_32767_utf16_units() {
        let long = format!("{}\u{1F600}", "a".repeat(32766));
        let bytes = encode(&TabComplete { message: long });
        let cut: TabComplete = decode(&bytes);
        assert_eq!(cut.message.len(), 32767);
        assert!(cut.message.ends_with('?'));
    }

    #[test]
    fn from_v47_converts_identical_packets_and_refuses_the_rest() {
        let ctx = V5Ctx { entity_id: 42 };
        let swing = super::from_v47(&C2SPlay::Animation(v47::Animation), &ctx).unwrap();
        assert_eq!(
            swing,
            super::super::play::C2SPlay5::Animation(Animation {
                entity_id: 42,
                animation_type: 1
            })
        );
        let sprint = C2SPlay::EntityAction(v47::EntityAction {
            entity_id: 42,
            action: v47::EntityActionType::StartSprinting,
            aux_data: 0,
        });
        assert!(matches!(
            super::from_v47(&sprint, &ctx),
            Ok(super::super::play::C2SPlay5::EntityAction(EntityAction {
                action_id: 4,
                ..
            }))
        ));
        let air_use = C2SPlay::PlayerBlockPlacement(v47::PlayerBlockPlacement {
            position: BlockPos::new(-1, -1, -1),
            placed_block_direction: 255,
            stack: None,
            cursor_x: 0,
            cursor_y: 0,
            cursor_z: 0,
        });
        let converted = super::from_v47(&air_use, &ctx).unwrap();
        let mut buf = PacketBuffer::new();
        converted.write(&mut buf);
        assert_eq!(
            buf.as_slice(),
            &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0, 0]
        );
        for refused in [
            C2SPlay::PlayerPosition(v47::PlayerPosition {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                on_ground: true,
            }),
            C2SPlay::TabComplete(v47::TabComplete {
                message: "/tp".into(),
                target_block: Some(BlockPos::new(0, 0, 0)),
            }),
            C2SPlay::SteerVehicle(v47::SteerVehicle {
                sideways: 0.0,
                forward: 0.0,
                flags: 4,
            }),
        ] {
            assert!(super::from_v47(&refused, &ctx).is_err(), "{refused:?}");
        }
    }
}
