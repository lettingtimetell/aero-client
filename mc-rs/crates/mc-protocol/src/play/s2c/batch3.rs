//! Clientbound Play, ids `0x28`-`0x38`: `Effect` .. `PlayerListItem`.
//!
//! java: `network/play/server/S28PacketEffect.java` .. `S38PacketPlayerListItem.java`

use mc_math::BlockPos;
use mc_nbt::NbtCompound;
use uuid::Uuid;

use crate::item::ItemStack;
use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError};

// ---- 0x28 Effect ---------------------------------------------------------------------------

/// `0x28 Effect`: a one-shot sound/particle effect (e.g. door open, block break, wither spawn,
/// record play) keyed by a numeric effect id. java: `S28PacketEffect`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Effect {
    /// The effect id (vanilla has no enum for this — a raw, effect-registry-shaped int fed
    /// straight into a client-side dispatch table; resolving its meaning is out of scope here,
    /// same rationale as `ItemStack::item_id`).
    pub effect_id: i32,
    /// Where the effect happens.
    pub position: BlockPos,
    /// Extra data for effects that need it (e.g. the block/item id for a break/place effect,
    /// or the record id for record play); unused (`0`) for effects that don't need it.
    pub data: i32,
    /// If `true`, the client uses `RenderGlobal.broadcastSound` instead of the ordinary
    /// `playAuxSFX` table. That broadcast table handles only ids 1013 and 1018, moving their
    /// sounds two blocks from the listener toward `position`; every other id is ignored.
    /// java: `S28PacketEffect.serverWide`, `NetHandlerPlayClient.handleEffect`
    pub disable_relative_volume: bool,
}

impl Packet for Effect {
    const ID: i32 = 0x28;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let effect_id = buf.read_i32()?;
        let position = buf.read_block_pos()?;
        let data = buf.read_i32()?;
        let disable_relative_volume = buf.read_bool()?;
        Ok(Effect {
            effect_id,
            position,
            data,
            disable_relative_volume,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.effect_id);
        buf.write_block_pos(self.position);
        buf.write_i32(self.data);
        buf.write_bool(self.disable_relative_volume);
    }
}

// ---- 0x29 SoundEffect -----------------------------------------------------------------------

/// `0x29 Sound Effect`: play a named sound at a position. Unlike `Effect`'s `BlockPos`, the
/// position here is **not** the packed hazard-H5 encoding — it's three raw big-endian `i32`s,
/// each the source coordinate fixed-point scaled by `8` (`(int)(coord * 8.0D)`).
/// java: `S29PacketSoundEffect`
#[derive(Debug, Clone, PartialEq)]
pub struct SoundEffect {
    /// The sound's registry name (e.g. `"random.explode"`), not resolved further here.
    pub sound_name: String,
    /// Source X, fixed-point: wire value is `(int)(x * 8.0)`.
    pub x_fixed: i32,
    /// Source Y, fixed-point: wire value is `(int)(y * 8.0)`.
    pub y_fixed: i32,
    /// Source Z, fixed-point: wire value is `(int)(z * 8.0)`.
    pub z_fixed: i32,
    /// Playback volume; `1.0` is normal, values above `1.0` extend the falloff radius.
    pub volume: f32,
    /// Playback pitch, quantized: wire value is `(int)(pitch * 63.0)`, `0..=255`.
    pub pitch_quantized: u8,
}

impl Packet for SoundEffect {
    const ID: i32 = 0x29;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let sound_name = buf.read_string(256)?;
        let x_fixed = buf.read_i32()?;
        let y_fixed = buf.read_i32()?;
        let z_fixed = buf.read_i32()?;
        let volume = buf.read_f32()?;
        let pitch_quantized = buf.read_u8()?;
        Ok(SoundEffect {
            sound_name,
            x_fixed,
            y_fixed,
            z_fixed,
            volume,
            pitch_quantized,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.sound_name);
        buf.write_i32(self.x_fixed);
        buf.write_i32(self.y_fixed);
        buf.write_i32(self.z_fixed);
        buf.write_f32(self.volume);
        buf.write_u8(self.pitch_quantized);
    }
}

// ---- 0x2A Particles -------------------------------------------------------------------------

/// The count of trailing `extra_data` VarInts a given particle id carries, matching
/// `EnumParticleTypes.getArgumentCount()`: `2` for `ITEM_CRACK` (id `36`), `1` for
/// `BLOCK_CRACK`/`BLOCK_DUST` (ids `37`/`38`), and `0` for every other id. The read path
/// normalizes unknown wire ids to `BARRIER` before calling this helper, just as vanilla does.
fn particle_extra_data_count(particle_id: i32) -> usize {
    match particle_id {
        36 => 2,
        37 | 38 => 1,
        _ => 0,
    }
}

/// Resolves the wire id the same way `EnumParticleTypes.getParticleFromId` followed by
/// `S2APacketParticles.readPacketData` does. Vanilla's 42 particle ids are the contiguous
/// range `0..=41`; an id outside that range becomes `BARRIER` (id `35`) before the packet's
/// argument count is selected.
fn normalize_particle_id(particle_id: i32) -> i32 {
    if (0..=41).contains(&particle_id) {
        particle_id
    } else {
        35
    }
}

/// `0x2A Particles`: spawn a burst of one particle type at a point.
///
/// Unknown wire ids are resolved to `BARRIER` during decoding, before choosing the fixed number
/// of trailing argument VarInts. This is the same enum substitution performed by vanilla's
/// `S2APacketParticles.readPacketData`; consequently a decoded unknown packet re-encodes with
/// id `35`, as a Java packet whose `particleType` was set to `BARRIER` does.
/// java: `S2APacketParticles`
#[derive(Debug, Clone, PartialEq)]
pub struct Particles {
    /// Particle type id (`EnumParticleTypes.getParticleID()`). Wire ids unknown to vanilla are
    /// normalized to `BARRIER` (35) during [`Packet::read`].
    pub particle_id: i32,
    /// If `true`, the particle renders even far outside the client's normal particle-view
    /// distance (`shouldIgnoreRange` on a handful of particle types, e.g. explosions).
    pub long_distance: bool,
    /// Spawn center X.
    pub x: f32,
    /// Spawn center Y.
    pub y: f32,
    /// Spawn center Z.
    pub z: f32,
    /// Random per-particle X offset/spread.
    pub offset_x: f32,
    /// Random per-particle Y offset/spread.
    pub offset_y: f32,
    /// Random per-particle Z offset/spread.
    pub offset_z: f32,
    /// Particle motion speed (meaning varies by particle type).
    pub particle_speed: f32,
    /// Number of particles to spawn.
    pub particle_count: i32,
    /// Trailing per-particle-type data (e.g. block/item id, and metadata for `ITEM_CRACK`); its
    /// length is implied by `particle_id` — see [`particle_extra_data_count`] — and is not
    /// itself sent on the wire.
    pub extra_data: Vec<i32>,
}

impl Packet for Particles {
    const ID: i32 = 0x2A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let particle_id = normalize_particle_id(buf.read_i32()?);
        let long_distance = buf.read_bool()?;
        let x = buf.read_f32()?;
        let y = buf.read_f32()?;
        let z = buf.read_f32()?;
        let offset_x = buf.read_f32()?;
        let offset_y = buf.read_f32()?;
        let offset_z = buf.read_f32()?;
        let particle_speed = buf.read_f32()?;
        let particle_count = buf.read_i32()?;
        let n = particle_extra_data_count(particle_id);
        let mut extra_data = Vec::with_capacity(n);
        for _ in 0..n {
            extra_data.push(buf.read_varint()?);
        }
        Ok(Particles {
            particle_id,
            long_distance,
            x,
            y,
            z,
            offset_x,
            offset_y,
            offset_z,
            particle_speed,
            particle_count,
            extra_data,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i32(self.particle_id);
        buf.write_bool(self.long_distance);
        buf.write_f32(self.x);
        buf.write_f32(self.y);
        buf.write_f32(self.z);
        buf.write_f32(self.offset_x);
        buf.write_f32(self.offset_y);
        buf.write_f32(self.offset_z);
        buf.write_f32(self.particle_speed);
        buf.write_i32(self.particle_count);
        // S2APacketParticles does not serialize the supplied vararg array's length.  It asks
        // the particle enum for its fixed arity and indexes that many entries: surplus caller
        // values are ignored, while a missing required value is an immediate Java array-bounds
        // failure.  Preserve both boundaries rather than allowing an arbitrary Vec to change
        // this packet's wire body.
        let n = particle_extra_data_count(self.particle_id);
        assert!(
            self.extra_data.len() >= n,
            "particle id {particle_id} requires {n} extra data values, got {actual}",
            particle_id = self.particle_id,
            actual = self.extra_data.len(),
        );
        for v in self.extra_data.iter().take(n) {
            buf.write_varint(*v);
        }
    }
}

// ---- 0x2B ChangeGameState -------------------------------------------------------------------

/// The reason code for a [`ChangeGameState`]. Vanilla's own handler (`NetHandlerPlayClient`)
/// dispatches on this with a plain `if`/`else if` chain, silently ignoring any value it doesn't
/// recognize (no exception, no default case) — so unlike e.g. [`PlayerListAction`], an unknown
/// byte here is not a protocol error. [`GameStateReason::Other`] carries it through unchanged
/// rather than rejecting the packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameStateReason {
    /// `0`: the bed a player tried to sleep in isn't valid right now; `value` is unused.
    InvalidBed,
    /// `1`: end raining; `value` is unused. A vanilla naming quirk, not a transcription error —
    /// see `mc-client`'s `sync::env::EnvState::apply_game_state` for the full explanation: this
    /// reason also sets `WorldInfo.raining = true` and resets the strength to `0.0` so it visibly
    /// fades *in*, which is the opposite of what the name suggests.
    EndRaining,
    /// `2`: begin raining; `value` is unused. Same quirk as [`GameStateReason::EndRaining`],
    /// mirrored: sets `WorldInfo.raining = false` and the strength to `1.0` so it fades *out*.
    BeginRaining,
    /// `3`: change game mode; `value` is the new [`GameMode`] id, floored+rounded.
    ChangeGameMode,
    /// `4`: show the end-credits screen; `value` is unused.
    EnterCredits,
    /// `5`: a demo-mode prompt; `value` selects which (`0`/`101`/`102`/`103`).
    DemoMessage,
    /// `6`: play the "arrow hit another player" sound; `value` is unused.
    ArrowHittingPlayer,
    /// `7`: set rain strength directly; `value` is the new strength.
    RainLevelChange,
    /// `8`: set thunder strength directly; `value` is the new strength.
    ThunderLevelChange,
    /// `10`: play the elder-guardian "mob appearance" effect; `value` is unused.
    PlayMobAppearance,
    /// Any other byte value vanilla's client silently ignores; carried through as-is.
    Other(u8),
}

impl GameStateReason {
    fn from_id(id: u8) -> GameStateReason {
        match id {
            0 => GameStateReason::InvalidBed,
            1 => GameStateReason::EndRaining,
            2 => GameStateReason::BeginRaining,
            3 => GameStateReason::ChangeGameMode,
            4 => GameStateReason::EnterCredits,
            5 => GameStateReason::DemoMessage,
            6 => GameStateReason::ArrowHittingPlayer,
            7 => GameStateReason::RainLevelChange,
            8 => GameStateReason::ThunderLevelChange,
            10 => GameStateReason::PlayMobAppearance,
            other => GameStateReason::Other(other),
        }
    }

    fn to_id(self) -> u8 {
        match self {
            GameStateReason::InvalidBed => 0,
            GameStateReason::EndRaining => 1,
            GameStateReason::BeginRaining => 2,
            GameStateReason::ChangeGameMode => 3,
            GameStateReason::EnterCredits => 4,
            GameStateReason::DemoMessage => 5,
            GameStateReason::ArrowHittingPlayer => 6,
            GameStateReason::RainLevelChange => 7,
            GameStateReason::ThunderLevelChange => 8,
            GameStateReason::PlayMobAppearance => 10,
            GameStateReason::Other(v) => v,
        }
    }
}

/// `0x2B Change Game State`: a grab-bag of world/UI state changes that don't warrant their own
/// packet (rain, thunder, game-mode change, demo prompts, ...). java: `S2BPacketChangeGameState`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChangeGameState {
    /// What's changing.
    pub reason: GameStateReason,
    /// The reason-specific payload float; see each [`GameStateReason`] variant for its meaning.
    pub value: f32,
}

impl Packet for ChangeGameState {
    const ID: i32 = 0x2B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let reason = GameStateReason::from_id(buf.read_u8()?);
        let value = buf.read_f32()?;
        Ok(ChangeGameState { reason, value })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.reason.to_id());
        buf.write_f32(self.value);
    }
}

// ---- 0x2C SpawnGlobalEntity -----------------------------------------------------------------

/// `0x2C Spawn Global Entity`: spawns a world-scoped entity with no owning chunk-render entry —
/// in 1.8.9 this is used for exactly one thing, a lightning bolt. Position fields are raw
/// big-endian `i32`s, fixed-point scaled by `32` (`floor(coord * 32.0)`), the same convention
/// `S0CPacketSpawnPlayer`/`S0EPacketSpawnObject` use — **not** the packed hazard-H5 `BlockPos`.
/// java: `S2CPacketSpawnGlobalEntity`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpawnGlobalEntity {
    /// The new entity's id.
    pub entity_id: i32,
    /// Entity type; vanilla only ever sends `1` (`EntityLightningBolt`), `0` for anything else.
    pub entity_type: i8,
    /// X, fixed-point: wire value is `floor(x * 32.0)`.
    pub x_fixed: i32,
    /// Y, fixed-point: wire value is `floor(y * 32.0)`.
    pub y_fixed: i32,
    /// Z, fixed-point: wire value is `floor(z * 32.0)`.
    pub z_fixed: i32,
}

impl Packet for SpawnGlobalEntity {
    const ID: i32 = 0x2C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let entity_type = buf.read_i8()?;
        let x_fixed = buf.read_i32()?;
        let y_fixed = buf.read_i32()?;
        let z_fixed = buf.read_i32()?;
        Ok(SpawnGlobalEntity {
            entity_id,
            entity_type,
            x_fixed,
            y_fixed,
            z_fixed,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_i8(self.entity_type);
        buf.write_i32(self.x_fixed);
        buf.write_i32(self.y_fixed);
        buf.write_i32(self.z_fixed);
    }
}

// ---- 0x2D OpenWindow ------------------------------------------------------------------------

/// `0x2D Open Window`: tells the client to display a server-side inventory GUI.
///
/// `inventory_type` is a free-form string id (e.g. `"minecraft:chest"`, `"EntityHorse"`) rather
/// than a VarInt enum in 1.8.9 — vanilla's own read/write side only ever branches on it once,
/// an exact `"EntityHorse"` string match gating `entity_id`, so it's kept as a plain `String`
/// here rather than a closed Rust enum. java: `S2DPacketOpenWindow`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenWindow {
    /// The window id the client should use to refer to this inventory in later packets
    /// (`ClickWindow`, `CloseWindow`, ...).
    pub window_id: u8,
    /// The GUI type id, e.g. `"minecraft:chest"`; see the struct-level note on `"EntityHorse"`.
    pub inventory_type: String,
    /// The window's title, as chat-component JSON.
    pub window_title: String,
    /// Number of slots in the inventory (excluding the player's own inventory, always appended
    /// client-side).
    pub slot_count: u8,
    /// The ridden/owning entity's id — present only when `inventory_type == "EntityHorse"`.
    pub entity_id: Option<i32>,
}

impl Packet for OpenWindow {
    const ID: i32 = 0x2D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let inventory_type = buf.read_string(32)?;
        let window_title = buf.read_chat()?;
        let slot_count = buf.read_u8()?;
        let entity_id = if inventory_type == "EntityHorse" {
            Some(buf.read_i32()?)
        } else {
            None
        };
        Ok(OpenWindow {
            window_id,
            inventory_type,
            window_title,
            slot_count,
            entity_id,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_string(&self.inventory_type);
        buf.write_chat(&self.window_title);
        buf.write_u8(self.slot_count);
        if self.inventory_type == "EntityHorse" {
            // Matches vanilla's own field default (0) if `entity_id` wasn't actually set on an
            // "EntityHorse" window — `write` is infallible by contract, so there's no error path.
            buf.write_i32(self.entity_id.unwrap_or(0));
        }
    }
}

// ---- 0x2E CloseWindow -----------------------------------------------------------------------

/// `0x2E Close Window`: the server is force-closing a window the client had open (e.g. because
/// the block/entity backing it went away). java: `S2EPacketCloseWindow`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CloseWindow {
    /// The window id being closed.
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

// ---- 0x2F SetSlot ---------------------------------------------------------------------------

/// `0x2F Set Slot`: sets a single inventory slot's contents.
///
/// `window_id` is a **signed** byte here (`readByte`, unlike the unsigned `readUnsignedByte`
/// every other window-id field in this batch uses) because `-1` is a real value: it addresses
/// the cursor stack (the item currently picked up by the mouse), not a numbered window.
/// java: `S2FPacketSetSlot`
#[derive(Debug, Clone, PartialEq)]
pub struct SetSlot {
    /// The target window id; `-1` means the cursor stack.
    pub window_id: i8,
    /// The slot index within that window.
    pub slot: i16,
    /// The slot's new contents (`None` = empty).
    pub item: Option<ItemStack>,
}

impl Packet for SetSlot {
    const ID: i32 = 0x2F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_i8()?;
        let slot = buf.read_i16()?;
        let item = buf.read_item_stack()?;
        Ok(SetSlot {
            window_id,
            slot,
            item,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.window_id);
        buf.write_i16(self.slot);
        buf.write_item_stack(self.item.as_ref());
    }
}

// ---- 0x30 WindowItems -----------------------------------------------------------------------

/// `0x30 Window Items`: replaces an entire window's slot contents in one shot (e.g. on window
/// open). The item count is a raw big-endian `i16` (`readShort`/`writeShort`), **not** a
/// VarInt — every other count in this batch is a VarInt, so this is worth calling out.
/// java: `S30PacketWindowItems`
#[derive(Debug, Clone, PartialEq)]
pub struct WindowItems {
    /// The window being updated (`0` is always the player's own inventory).
    pub window_id: u8,
    /// Every slot's contents, in slot-index order.
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
        let mut items = Vec::with_capacity(count as usize);
        for _ in 0..count {
            items.push(buf.read_item_stack()?);
        }
        Ok(WindowItems { window_id, items })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.items.len() as i16);
        for item in &self.items {
            buf.write_item_stack(item.as_ref());
        }
    }
}

// ---- 0x31 WindowProperty --------------------------------------------------------------------

/// `0x31 Window Property`: updates one numeric "property" of a window (e.g. furnace burn time,
/// enchanting-table levels) — the property's meaning is defined by the window type and
/// `var_index` together, which this crate does not resolve. java: `S31PacketWindowProperty`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowProperty {
    /// The window being updated.
    pub window_id: u8,
    /// Which property (window-type-specific meaning).
    pub var_index: i16,
    /// The property's new value.
    pub var_value: i16,
}

impl Packet for WindowProperty {
    const ID: i32 = 0x31;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let var_index = buf.read_i16()?;
        let var_value = buf.read_i16()?;
        Ok(WindowProperty {
            window_id,
            var_index,
            var_value,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.var_index);
        buf.write_i16(self.var_value);
    }
}

// ---- 0x32 ConfirmTransaction -----------------------------------------------------------------

/// `0x32 Confirm Transaction`: the server's reply to a client `ConfirmTransaction`, telling the
/// client whether a queued window action was accepted (and, if not, that it must resync the
/// window). java: `S32PacketConfirmTransaction`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfirmTransaction {
    /// The window the action applied to.
    pub window_id: u8,
    /// The action number being confirmed, echoed back from the client's own transaction.
    pub action_number: i16,
    /// Whether the server accepted the action.
    pub accepted: bool,
}

impl Packet for ConfirmTransaction {
    const ID: i32 = 0x32;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let window_id = buf.read_u8()?;
        let action_number = buf.read_i16()?;
        let accepted = buf.read_bool()?;
        Ok(ConfirmTransaction {
            window_id,
            action_number,
            accepted,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.window_id);
        buf.write_i16(self.action_number);
        buf.write_bool(self.accepted);
    }
}

// ---- 0x33 UpdateSign ------------------------------------------------------------------------

/// `0x33 Update Sign`: sets a sign's four text lines (e.g. right after a player finishes editing
/// one in the sign GUI). Each line is chat-component JSON, like elsewhere in this crate.
/// java: `S33PacketUpdateSign`
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateSign {
    /// The sign block's position.
    pub position: BlockPos,
    /// The four lines of text, top to bottom, as chat-component JSON.
    pub lines: [String; 4],
}

impl Packet for UpdateSign {
    const ID: i32 = 0x33;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let lines = [
            buf.read_chat()?,
            buf.read_chat()?,
            buf.read_chat()?,
            buf.read_chat()?,
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

// ---- 0x34 Maps -------------------------------------------------------------------------------

/// One map-decoration icon (a player/frame/etc. marker drawn on the map). java: `Vec4b`, as used
/// by `S34PacketMaps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapIcon {
    /// The icon's type/texture id, `0..=15` (packed into the high nibble of the first wire
    /// byte): e.g. white arrow, item frame, player, mansion, ocean monument.
    pub icon_type: u8,
    /// X position on the 128x128 map image.
    pub x: i8,
    /// Z ("Y" on the wire, but it's a world-Z-derived map coordinate) position on the map image.
    pub z: i8,
    /// Rotation, `0..=15` (packed into the low nibble of the first wire byte): 16 steps around a
    /// full turn.
    pub direction: u8,
}

impl MapIcon {
    fn read(buf: &mut PacketBuffer) -> Result<MapIcon, ProtoError> {
        let packed = buf.read_u8()?;
        let x = buf.read_i8()?;
        let z = buf.read_i8()?;
        Ok(MapIcon {
            icon_type: (packed >> 4) & 0x0F,
            x,
            z,
            direction: packed & 0x0F,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(((self.icon_type & 0x0F) << 4) | (self.direction & 0x0F));
        buf.write_i8(self.x);
        buf.write_i8(self.z);
    }
}

/// A rectangular pixel update to a map's 128x128 color-index image, present only when the
/// updated region is non-empty. java: the `mapMaxX > 0` branch of `S34PacketMaps`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MapPixelUpdate {
    /// Width of the updated rectangle, in pixels. Always `> 0` for a well-formed update — the
    /// packet-level presence of this whole struct already encodes that.
    pub width: u8,
    /// Height of the updated rectangle, in pixels.
    pub height: u8,
    /// X offset of the rectangle's left edge within the 128x128 image.
    pub min_x: u8,
    /// Y offset of the rectangle's top edge within the 128x128 image.
    pub min_y: u8,
    /// `width * height` color-index bytes, row-major.
    pub data: Vec<u8>,
}

/// `0x34 Maps`: updates an item-map's icons and/or a rectangular region of its pixel data.
/// java: `S34PacketMaps`
#[derive(Debug, Clone, PartialEq)]
pub struct Maps {
    /// Which map item this updates (matches the map item's damage/metadata value).
    pub map_id: i32,
    /// The map's zoom scale, `0` (1 block/pixel) to `4` (16 blocks/pixel).
    pub scale: i8,
    /// Decoration icons currently visible on the map (players, frames, ...).
    pub icons: Vec<MapIcon>,
    /// A pixel-data update, if any pixels changed this tick.
    pub update: Option<MapPixelUpdate>,
}

impl Packet for Maps {
    const ID: i32 = 0x34;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let map_id = buf.read_varint()?;
        let scale = buf.read_i8()?;

        let icon_count = buf.read_varint()?;
        if icon_count < 0 {
            return Err(ProtoError::NegativeLength(icon_count));
        }
        let mut icons = Vec::with_capacity((icon_count as usize).min(4096));
        for _ in 0..icon_count {
            icons.push(MapIcon::read(buf)?);
        }

        let width = buf.read_u8()?;
        let update = if width > 0 {
            let height = buf.read_u8()?;
            let min_x = buf.read_u8()?;
            let min_y = buf.read_u8()?;
            let data = buf.read_byte_array()?;
            Some(MapPixelUpdate {
                width,
                height,
                min_x,
                min_y,
                data,
            })
        } else {
            None
        };

        Ok(Maps {
            map_id,
            scale,
            icons,
            update,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.map_id);
        buf.write_i8(self.scale);

        buf.write_varint(self.icons.len() as i32);
        for icon in &self.icons {
            icon.write(buf);
        }

        match &self.update {
            Some(u) => {
                buf.write_u8(u.width);
                // Mirrors vanilla's own write-side condition, which re-checks the field's
                // value rather than a separate "present" flag: a `width == 0` update (not
                // constructible via a normal read, but not prevented by the type either) writes
                // only the leading zero, same as `None`.
                if u.width > 0 {
                    buf.write_u8(u.height);
                    buf.write_u8(u.min_x);
                    buf.write_u8(u.min_y);
                    buf.write_byte_array(&u.data);
                }
            }
            None => buf.write_u8(0),
        }
    }
}

// ---- 0x35 UpdateTileEntity -------------------------------------------------------------------

/// `0x35 Update Tile Entity`: pushes a full NBT-serialized tile entity to the client (mob
/// spawners, command blocks, skulls, flower pots, banners, ...). java: `S35PacketUpdateTileEntity`
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateTileEntity {
    /// The tile entity's block position.
    pub position: BlockPos,
    /// A vanilla-tile-entity-type discriminant (e.g. `1` mob spawner, `2` command block, `3`
    /// beacon, `4` skull, `5` flower pot, `6` banner, `7` structure block, `8` end gateway);
    /// modded tile entities generally send `0`/unused here and rely on `nbt`'s `id` tag instead.
    pub action: u8,
    /// The tile entity's full NBT data.
    pub nbt: Option<NbtCompound>,
}

impl Packet for UpdateTileEntity {
    const ID: i32 = 0x35;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_block_pos()?;
        let action = buf.read_u8()?;
        let nbt = buf.read_nbt_compound_opt()?;
        Ok(UpdateTileEntity {
            position,
            action,
            nbt,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
        buf.write_u8(self.action);
        buf.write_nbt_compound_opt(self.nbt.as_ref());
    }
}

// ---- 0x36 SignEditorOpen ---------------------------------------------------------------------

/// `0x36 Sign Editor Open`: tells the client to open the sign-editing GUI for a
/// just-placed sign. java: `S36PacketSignEditorOpen`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SignEditorOpen {
    /// The sign block's position.
    pub position: BlockPos,
}

impl Packet for SignEditorOpen {
    const ID: i32 = 0x36;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SignEditorOpen {
            position: buf.read_block_pos()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_block_pos(self.position);
    }
}

// ---- 0x37 Statistics -------------------------------------------------------------------------

// S37PacketStatistics resolves each wire id through StatList.getOneShotStat and
// retains only registered ids. Keep the 1.8.9 registry here rather than accepting
// arbitrary strings: mc-protocol intentionally has no dependency on mc-data, while
// the packet's filtering is part of the wire behavior. This list is the exact 883-id
// 1.8.9 registry (including achievements), kept as a protocol-local immutable table.
const KNOWN_STAT_IDS: &[&str] = &[
    "achievement.acquireIron",
    "achievement.bakeCake",
    "achievement.blazeRod",
    "achievement.bookcase",
    "achievement.breedCow",
    "achievement.buildBetterPickaxe",
    "achievement.buildFurnace",
    "achievement.buildHoe",
    "achievement.buildPickaxe",
    "achievement.buildSword",
    "achievement.buildWorkBench",
    "achievement.cookFish",
    "achievement.diamonds",
    "achievement.diamondsToYou",
    "achievement.enchantments",
    "achievement.exploreAllBiomes",
    "achievement.flyPig",
    "achievement.fullBeacon",
    "achievement.ghast",
    "achievement.killCow",
    "achievement.killEnemy",
    "achievement.killWither",
    "achievement.makeBread",
    "achievement.mineWood",
    "achievement.onARail",
    "achievement.openInventory",
    "achievement.overkill",
    "achievement.overpowered",
    "achievement.portal",
    "achievement.potion",
    "achievement.snipeSkeleton",
    "achievement.spawnWither",
    "achievement.theEnd",
    "achievement.theEnd2",
    "stat.animalsBred",
    "stat.armorCleaned",
    "stat.bannerCleaned",
    "stat.beaconInteraction",
    "stat.boatOneCm",
    "stat.breakItem.minecraft.bow",
    "stat.breakItem.minecraft.carrot_on_a_stick",
    "stat.breakItem.minecraft.chainmail_boots",
    "stat.breakItem.minecraft.chainmail_chestplate",
    "stat.breakItem.minecraft.chainmail_helmet",
    "stat.breakItem.minecraft.chainmail_leggings",
    "stat.breakItem.minecraft.diamond_axe",
    "stat.breakItem.minecraft.diamond_boots",
    "stat.breakItem.minecraft.diamond_chestplate",
    "stat.breakItem.minecraft.diamond_helmet",
    "stat.breakItem.minecraft.diamond_hoe",
    "stat.breakItem.minecraft.diamond_leggings",
    "stat.breakItem.minecraft.diamond_pickaxe",
    "stat.breakItem.minecraft.diamond_shovel",
    "stat.breakItem.minecraft.diamond_sword",
    "stat.breakItem.minecraft.fishing_rod",
    "stat.breakItem.minecraft.flint_and_steel",
    "stat.breakItem.minecraft.golden_axe",
    "stat.breakItem.minecraft.golden_boots",
    "stat.breakItem.minecraft.golden_chestplate",
    "stat.breakItem.minecraft.golden_helmet",
    "stat.breakItem.minecraft.golden_hoe",
    "stat.breakItem.minecraft.golden_leggings",
    "stat.breakItem.minecraft.golden_pickaxe",
    "stat.breakItem.minecraft.golden_shovel",
    "stat.breakItem.minecraft.golden_sword",
    "stat.breakItem.minecraft.iron_axe",
    "stat.breakItem.minecraft.iron_boots",
    "stat.breakItem.minecraft.iron_chestplate",
    "stat.breakItem.minecraft.iron_helmet",
    "stat.breakItem.minecraft.iron_hoe",
    "stat.breakItem.minecraft.iron_leggings",
    "stat.breakItem.minecraft.iron_pickaxe",
    "stat.breakItem.minecraft.iron_shovel",
    "stat.breakItem.minecraft.iron_sword",
    "stat.breakItem.minecraft.leather_boots",
    "stat.breakItem.minecraft.leather_chestplate",
    "stat.breakItem.minecraft.leather_helmet",
    "stat.breakItem.minecraft.leather_leggings",
    "stat.breakItem.minecraft.shears",
    "stat.breakItem.minecraft.stone_axe",
    "stat.breakItem.minecraft.stone_hoe",
    "stat.breakItem.minecraft.stone_pickaxe",
    "stat.breakItem.minecraft.stone_shovel",
    "stat.breakItem.minecraft.stone_sword",
    "stat.breakItem.minecraft.wooden_axe",
    "stat.breakItem.minecraft.wooden_hoe",
    "stat.breakItem.minecraft.wooden_pickaxe",
    "stat.breakItem.minecraft.wooden_shovel",
    "stat.breakItem.minecraft.wooden_sword",
    "stat.brewingstandInteraction",
    "stat.cakeSlicesEaten",
    "stat.cauldronFilled",
    "stat.cauldronUsed",
    "stat.chestOpened",
    "stat.climbOneCm",
    "stat.craftingTableInteraction",
    "stat.craftItem.minecraft.acacia_door",
    "stat.craftItem.minecraft.acacia_fence",
    "stat.craftItem.minecraft.acacia_fence_gate",
    "stat.craftItem.minecraft.acacia_stairs",
    "stat.craftItem.minecraft.activator_rail",
    "stat.craftItem.minecraft.anvil",
    "stat.craftItem.minecraft.armor_stand",
    "stat.craftItem.minecraft.arrow",
    "stat.craftItem.minecraft.baked_potato",
    "stat.craftItem.minecraft.banner",
    "stat.craftItem.minecraft.beacon",
    "stat.craftItem.minecraft.bed",
    "stat.craftItem.minecraft.birch_door",
    "stat.craftItem.minecraft.birch_fence",
    "stat.craftItem.minecraft.birch_fence_gate",
    "stat.craftItem.minecraft.birch_stairs",
    "stat.craftItem.minecraft.blaze_powder",
    "stat.craftItem.minecraft.boat",
    "stat.craftItem.minecraft.book",
    "stat.craftItem.minecraft.bookshelf",
    "stat.craftItem.minecraft.bow",
    "stat.craftItem.minecraft.bowl",
    "stat.craftItem.minecraft.bread",
    "stat.craftItem.minecraft.brewing_stand",
    "stat.craftItem.minecraft.brick",
    "stat.craftItem.minecraft.brick_block",
    "stat.craftItem.minecraft.brick_stairs",
    "stat.craftItem.minecraft.bucket",
    "stat.craftItem.minecraft.cake",
    "stat.craftItem.minecraft.carpet",
    "stat.craftItem.minecraft.carrot_on_a_stick",
    "stat.craftItem.minecraft.cauldron",
    "stat.craftItem.minecraft.chest",
    "stat.craftItem.minecraft.chest_minecart",
    "stat.craftItem.minecraft.clay",
    "stat.craftItem.minecraft.clock",
    "stat.craftItem.minecraft.coal",
    "stat.craftItem.minecraft.coal_block",
    "stat.craftItem.minecraft.cobblestone_wall",
    "stat.craftItem.minecraft.comparator",
    "stat.craftItem.minecraft.compass",
    "stat.craftItem.minecraft.cooked_beef",
    "stat.craftItem.minecraft.cooked_chicken",
    "stat.craftItem.minecraft.cooked_fish",
    "stat.craftItem.minecraft.cooked_mutton",
    "stat.craftItem.minecraft.cooked_porkchop",
    "stat.craftItem.minecraft.cooked_rabbit",
    "stat.craftItem.minecraft.cookie",
    "stat.craftItem.minecraft.crafting_table",
    "stat.craftItem.minecraft.dark_oak_door",
    "stat.craftItem.minecraft.dark_oak_fence",
    "stat.craftItem.minecraft.dark_oak_fence_gate",
    "stat.craftItem.minecraft.dark_oak_stairs",
    "stat.craftItem.minecraft.daylight_detector",
    "stat.craftItem.minecraft.detector_rail",
    "stat.craftItem.minecraft.diamond",
    "stat.craftItem.minecraft.diamond_axe",
    "stat.craftItem.minecraft.diamond_block",
    "stat.craftItem.minecraft.diamond_boots",
    "stat.craftItem.minecraft.diamond_chestplate",
    "stat.craftItem.minecraft.diamond_helmet",
    "stat.craftItem.minecraft.diamond_hoe",
    "stat.craftItem.minecraft.diamond_leggings",
    "stat.craftItem.minecraft.diamond_pickaxe",
    "stat.craftItem.minecraft.diamond_shovel",
    "stat.craftItem.minecraft.diamond_sword",
    "stat.craftItem.minecraft.dirt",
    "stat.craftItem.minecraft.dispenser",
    "stat.craftItem.minecraft.dropper",
    "stat.craftItem.minecraft.dye",
    "stat.craftItem.minecraft.emerald",
    "stat.craftItem.minecraft.emerald_block",
    "stat.craftItem.minecraft.enchanting_table",
    "stat.craftItem.minecraft.ender_chest",
    "stat.craftItem.minecraft.ender_eye",
    "stat.craftItem.minecraft.fence",
    "stat.craftItem.minecraft.fence_gate",
    "stat.craftItem.minecraft.fermented_spider_eye",
    "stat.craftItem.minecraft.fire_charge",
    "stat.craftItem.minecraft.fishing_rod",
    "stat.craftItem.minecraft.flint_and_steel",
    "stat.craftItem.minecraft.flower_pot",
    "stat.craftItem.minecraft.furnace",
    "stat.craftItem.minecraft.furnace_minecart",
    "stat.craftItem.minecraft.glass",
    "stat.craftItem.minecraft.glass_bottle",
    "stat.craftItem.minecraft.glass_pane",
    "stat.craftItem.minecraft.glowstone",
    "stat.craftItem.minecraft.gold_block",
    "stat.craftItem.minecraft.gold_ingot",
    "stat.craftItem.minecraft.gold_nugget",
    "stat.craftItem.minecraft.golden_apple",
    "stat.craftItem.minecraft.golden_axe",
    "stat.craftItem.minecraft.golden_boots",
    "stat.craftItem.minecraft.golden_carrot",
    "stat.craftItem.minecraft.golden_chestplate",
    "stat.craftItem.minecraft.golden_helmet",
    "stat.craftItem.minecraft.golden_hoe",
    "stat.craftItem.minecraft.golden_leggings",
    "stat.craftItem.minecraft.golden_pickaxe",
    "stat.craftItem.minecraft.golden_rail",
    "stat.craftItem.minecraft.golden_shovel",
    "stat.craftItem.minecraft.golden_sword",
    "stat.craftItem.minecraft.hardened_clay",
    "stat.craftItem.minecraft.hay_block",
    "stat.craftItem.minecraft.heavy_weighted_pressure_plate",
    "stat.craftItem.minecraft.hopper",
    "stat.craftItem.minecraft.hopper_minecart",
    "stat.craftItem.minecraft.iron_axe",
    "stat.craftItem.minecraft.iron_bars",
    "stat.craftItem.minecraft.iron_block",
    "stat.craftItem.minecraft.iron_boots",
    "stat.craftItem.minecraft.iron_chestplate",
    "stat.craftItem.minecraft.iron_door",
    "stat.craftItem.minecraft.iron_helmet",
    "stat.craftItem.minecraft.iron_hoe",
    "stat.craftItem.minecraft.iron_ingot",
    "stat.craftItem.minecraft.iron_leggings",
    "stat.craftItem.minecraft.iron_pickaxe",
    "stat.craftItem.minecraft.iron_shovel",
    "stat.craftItem.minecraft.iron_sword",
    "stat.craftItem.minecraft.iron_trapdoor",
    "stat.craftItem.minecraft.item_frame",
    "stat.craftItem.minecraft.jukebox",
    "stat.craftItem.minecraft.jungle_door",
    "stat.craftItem.minecraft.jungle_fence",
    "stat.craftItem.minecraft.jungle_fence_gate",
    "stat.craftItem.minecraft.jungle_stairs",
    "stat.craftItem.minecraft.ladder",
    "stat.craftItem.minecraft.lapis_block",
    "stat.craftItem.minecraft.lead",
    "stat.craftItem.minecraft.leather",
    "stat.craftItem.minecraft.leather_boots",
    "stat.craftItem.minecraft.leather_chestplate",
    "stat.craftItem.minecraft.leather_helmet",
    "stat.craftItem.minecraft.leather_leggings",
    "stat.craftItem.minecraft.lever",
    "stat.craftItem.minecraft.light_weighted_pressure_plate",
    "stat.craftItem.minecraft.lit_pumpkin",
    "stat.craftItem.minecraft.magma_cream",
    "stat.craftItem.minecraft.map",
    "stat.craftItem.minecraft.melon_block",
    "stat.craftItem.minecraft.melon_seeds",
    "stat.craftItem.minecraft.minecart",
    "stat.craftItem.minecraft.mossy_cobblestone",
    "stat.craftItem.minecraft.mushroom_stew",
    "stat.craftItem.minecraft.nether_brick",
    "stat.craftItem.minecraft.nether_brick_fence",
    "stat.craftItem.minecraft.nether_brick_stairs",
    "stat.craftItem.minecraft.netherbrick",
    "stat.craftItem.minecraft.noteblock",
    "stat.craftItem.minecraft.oak_stairs",
    "stat.craftItem.minecraft.painting",
    "stat.craftItem.minecraft.paper",
    "stat.craftItem.minecraft.piston",
    "stat.craftItem.minecraft.planks",
    "stat.craftItem.minecraft.prismarine",
    "stat.craftItem.minecraft.pumpkin_pie",
    "stat.craftItem.minecraft.pumpkin_seeds",
    "stat.craftItem.minecraft.quartz",
    "stat.craftItem.minecraft.quartz_block",
    "stat.craftItem.minecraft.quartz_stairs",
    "stat.craftItem.minecraft.rabbit_stew",
    "stat.craftItem.minecraft.rail",
    "stat.craftItem.minecraft.red_sandstone",
    "stat.craftItem.minecraft.red_sandstone_stairs",
    "stat.craftItem.minecraft.redstone",
    "stat.craftItem.minecraft.redstone_block",
    "stat.craftItem.minecraft.redstone_lamp",
    "stat.craftItem.minecraft.redstone_torch",
    "stat.craftItem.minecraft.repeater",
    "stat.craftItem.minecraft.sandstone",
    "stat.craftItem.minecraft.sandstone_stairs",
    "stat.craftItem.minecraft.sea_lantern",
    "stat.craftItem.minecraft.shears",
    "stat.craftItem.minecraft.sign",
    "stat.craftItem.minecraft.slime",
    "stat.craftItem.minecraft.slime_ball",
    "stat.craftItem.minecraft.snow",
    "stat.craftItem.minecraft.snow_layer",
    "stat.craftItem.minecraft.speckled_melon",
    "stat.craftItem.minecraft.sponge",
    "stat.craftItem.minecraft.spruce_door",
    "stat.craftItem.minecraft.spruce_fence",
    "stat.craftItem.minecraft.spruce_fence_gate",
    "stat.craftItem.minecraft.spruce_stairs",
    "stat.craftItem.minecraft.stained_glass",
    "stat.craftItem.minecraft.stained_glass_pane",
    "stat.craftItem.minecraft.stained_hardened_clay",
    "stat.craftItem.minecraft.stick",
    "stat.craftItem.minecraft.sticky_piston",
    "stat.craftItem.minecraft.stone",
    "stat.craftItem.minecraft.stone_axe",
    "stat.craftItem.minecraft.stone_brick_stairs",
    "stat.craftItem.minecraft.stone_button",
    "stat.craftItem.minecraft.stone_hoe",
    "stat.craftItem.minecraft.stone_pickaxe",
    "stat.craftItem.minecraft.stone_pressure_plate",
    "stat.craftItem.minecraft.stone_shovel",
    "stat.craftItem.minecraft.stone_slab",
    "stat.craftItem.minecraft.stone_slab2",
    "stat.craftItem.minecraft.stone_stairs",
    "stat.craftItem.minecraft.stone_sword",
    "stat.craftItem.minecraft.stonebrick",
    "stat.craftItem.minecraft.sugar",
    "stat.craftItem.minecraft.tnt",
    "stat.craftItem.minecraft.tnt_minecart",
    "stat.craftItem.minecraft.torch",
    "stat.craftItem.minecraft.trapdoor",
    "stat.craftItem.minecraft.trapped_chest",
    "stat.craftItem.minecraft.tripwire_hook",
    "stat.craftItem.minecraft.wheat",
    "stat.craftItem.minecraft.wooden_axe",
    "stat.craftItem.minecraft.wooden_button",
    "stat.craftItem.minecraft.wooden_door",
    "stat.craftItem.minecraft.wooden_hoe",
    "stat.craftItem.minecraft.wooden_pickaxe",
    "stat.craftItem.minecraft.wooden_pressure_plate",
    "stat.craftItem.minecraft.wooden_shovel",
    "stat.craftItem.minecraft.wooden_slab",
    "stat.craftItem.minecraft.wooden_sword",
    "stat.craftItem.minecraft.wool",
    "stat.craftItem.minecraft.writable_book",
    "stat.crouchOneCm",
    "stat.damageDealt",
    "stat.damageTaken",
    "stat.deaths",
    "stat.dispenserInspected",
    "stat.diveOneCm",
    "stat.drop",
    "stat.dropperInspected",
    "stat.enderchestOpened",
    "stat.entityKilledBy.Bat",
    "stat.entityKilledBy.Blaze",
    "stat.entityKilledBy.CaveSpider",
    "stat.entityKilledBy.Chicken",
    "stat.entityKilledBy.Cow",
    "stat.entityKilledBy.Creeper",
    "stat.entityKilledBy.Enderman",
    "stat.entityKilledBy.Endermite",
    "stat.entityKilledBy.EntityHorse",
    "stat.entityKilledBy.Ghast",
    "stat.entityKilledBy.Guardian",
    "stat.entityKilledBy.LavaSlime",
    "stat.entityKilledBy.MushroomCow",
    "stat.entityKilledBy.Ozelot",
    "stat.entityKilledBy.Pig",
    "stat.entityKilledBy.PigZombie",
    "stat.entityKilledBy.Rabbit",
    "stat.entityKilledBy.Sheep",
    "stat.entityKilledBy.Silverfish",
    "stat.entityKilledBy.Skeleton",
    "stat.entityKilledBy.Slime",
    "stat.entityKilledBy.Spider",
    "stat.entityKilledBy.Squid",
    "stat.entityKilledBy.Villager",
    "stat.entityKilledBy.Witch",
    "stat.entityKilledBy.Wolf",
    "stat.entityKilledBy.Zombie",
    "stat.fallOneCm",
    "stat.fishCaught",
    "stat.flowerPotted",
    "stat.flyOneCm",
    "stat.furnaceInteraction",
    "stat.hopperInspected",
    "stat.horseOneCm",
    "stat.itemEnchanted",
    "stat.jump",
    "stat.junkFished",
    "stat.killEntity.Bat",
    "stat.killEntity.Blaze",
    "stat.killEntity.CaveSpider",
    "stat.killEntity.Chicken",
    "stat.killEntity.Cow",
    "stat.killEntity.Creeper",
    "stat.killEntity.Enderman",
    "stat.killEntity.Endermite",
    "stat.killEntity.EntityHorse",
    "stat.killEntity.Ghast",
    "stat.killEntity.Guardian",
    "stat.killEntity.LavaSlime",
    "stat.killEntity.MushroomCow",
    "stat.killEntity.Ozelot",
    "stat.killEntity.Pig",
    "stat.killEntity.PigZombie",
    "stat.killEntity.Rabbit",
    "stat.killEntity.Sheep",
    "stat.killEntity.Silverfish",
    "stat.killEntity.Skeleton",
    "stat.killEntity.Slime",
    "stat.killEntity.Spider",
    "stat.killEntity.Squid",
    "stat.killEntity.Villager",
    "stat.killEntity.Witch",
    "stat.killEntity.Wolf",
    "stat.killEntity.Zombie",
    "stat.leaveGame",
    "stat.mineBlock.minecraft.acacia_fence",
    "stat.mineBlock.minecraft.acacia_fence_gate",
    "stat.mineBlock.minecraft.acacia_stairs",
    "stat.mineBlock.minecraft.activator_rail",
    "stat.mineBlock.minecraft.anvil",
    "stat.mineBlock.minecraft.beacon",
    "stat.mineBlock.minecraft.birch_fence",
    "stat.mineBlock.minecraft.birch_fence_gate",
    "stat.mineBlock.minecraft.birch_stairs",
    "stat.mineBlock.minecraft.bookshelf",
    "stat.mineBlock.minecraft.brick_block",
    "stat.mineBlock.minecraft.brick_stairs",
    "stat.mineBlock.minecraft.brown_mushroom",
    "stat.mineBlock.minecraft.brown_mushroom_block",
    "stat.mineBlock.minecraft.cactus",
    "stat.mineBlock.minecraft.carpet",
    "stat.mineBlock.minecraft.chest",
    "stat.mineBlock.minecraft.clay",
    "stat.mineBlock.minecraft.coal_block",
    "stat.mineBlock.minecraft.coal_ore",
    "stat.mineBlock.minecraft.cobblestone",
    "stat.mineBlock.minecraft.cobblestone_wall",
    "stat.mineBlock.minecraft.command_block",
    "stat.mineBlock.minecraft.crafting_table",
    "stat.mineBlock.minecraft.dark_oak_fence",
    "stat.mineBlock.minecraft.dark_oak_fence_gate",
    "stat.mineBlock.minecraft.dark_oak_stairs",
    "stat.mineBlock.minecraft.daylight_detector",
    "stat.mineBlock.minecraft.deadbush",
    "stat.mineBlock.minecraft.detector_rail",
    "stat.mineBlock.minecraft.diamond_block",
    "stat.mineBlock.minecraft.diamond_ore",
    "stat.mineBlock.minecraft.dirt",
    "stat.mineBlock.minecraft.dispenser",
    "stat.mineBlock.minecraft.double_plant",
    "stat.mineBlock.minecraft.dragon_egg",
    "stat.mineBlock.minecraft.dropper",
    "stat.mineBlock.minecraft.emerald_block",
    "stat.mineBlock.minecraft.emerald_ore",
    "stat.mineBlock.minecraft.enchanting_table",
    "stat.mineBlock.minecraft.end_portal_frame",
    "stat.mineBlock.minecraft.end_stone",
    "stat.mineBlock.minecraft.ender_chest",
    "stat.mineBlock.minecraft.fence",
    "stat.mineBlock.minecraft.fence_gate",
    "stat.mineBlock.minecraft.furnace",
    "stat.mineBlock.minecraft.glass",
    "stat.mineBlock.minecraft.glass_pane",
    "stat.mineBlock.minecraft.glowstone",
    "stat.mineBlock.minecraft.gold_block",
    "stat.mineBlock.minecraft.gold_ore",
    "stat.mineBlock.minecraft.golden_rail",
    "stat.mineBlock.minecraft.gravel",
    "stat.mineBlock.minecraft.hardened_clay",
    "stat.mineBlock.minecraft.hay_block",
    "stat.mineBlock.minecraft.heavy_weighted_pressure_plate",
    "stat.mineBlock.minecraft.hopper",
    "stat.mineBlock.minecraft.ice",
    "stat.mineBlock.minecraft.iron_bars",
    "stat.mineBlock.minecraft.iron_block",
    "stat.mineBlock.minecraft.iron_ore",
    "stat.mineBlock.minecraft.jukebox",
    "stat.mineBlock.minecraft.jungle_fence",
    "stat.mineBlock.minecraft.jungle_fence_gate",
    "stat.mineBlock.minecraft.jungle_stairs",
    "stat.mineBlock.minecraft.ladder",
    "stat.mineBlock.minecraft.lapis_block",
    "stat.mineBlock.minecraft.lapis_ore",
    "stat.mineBlock.minecraft.leaves",
    "stat.mineBlock.minecraft.leaves2",
    "stat.mineBlock.minecraft.lever",
    "stat.mineBlock.minecraft.light_weighted_pressure_plate",
    "stat.mineBlock.minecraft.log",
    "stat.mineBlock.minecraft.log2",
    "stat.mineBlock.minecraft.melon_block",
    "stat.mineBlock.minecraft.monster_egg",
    "stat.mineBlock.minecraft.mossy_cobblestone",
    "stat.mineBlock.minecraft.mycelium",
    "stat.mineBlock.minecraft.nether_brick",
    "stat.mineBlock.minecraft.nether_brick_fence",
    "stat.mineBlock.minecraft.nether_brick_stairs",
    "stat.mineBlock.minecraft.netherrack",
    "stat.mineBlock.minecraft.noteblock",
    "stat.mineBlock.minecraft.oak_stairs",
    "stat.mineBlock.minecraft.obsidian",
    "stat.mineBlock.minecraft.packed_ice",
    "stat.mineBlock.minecraft.piston",
    "stat.mineBlock.minecraft.planks",
    "stat.mineBlock.minecraft.prismarine",
    "stat.mineBlock.minecraft.pumpkin",
    "stat.mineBlock.minecraft.quartz_block",
    "stat.mineBlock.minecraft.quartz_ore",
    "stat.mineBlock.minecraft.quartz_stairs",
    "stat.mineBlock.minecraft.rail",
    "stat.mineBlock.minecraft.red_flower",
    "stat.mineBlock.minecraft.red_mushroom",
    "stat.mineBlock.minecraft.red_mushroom_block",
    "stat.mineBlock.minecraft.red_sandstone",
    "stat.mineBlock.minecraft.red_sandstone_stairs",
    "stat.mineBlock.minecraft.redstone_block",
    "stat.mineBlock.minecraft.redstone_lamp",
    "stat.mineBlock.minecraft.redstone_ore",
    "stat.mineBlock.minecraft.redstone_torch",
    "stat.mineBlock.minecraft.sand",
    "stat.mineBlock.minecraft.sandstone",
    "stat.mineBlock.minecraft.sandstone_stairs",
    "stat.mineBlock.minecraft.sapling",
    "stat.mineBlock.minecraft.sea_lantern",
    "stat.mineBlock.minecraft.slime",
    "stat.mineBlock.minecraft.snow",
    "stat.mineBlock.minecraft.snow_layer",
    "stat.mineBlock.minecraft.soul_sand",
    "stat.mineBlock.minecraft.sponge",
    "stat.mineBlock.minecraft.spruce_fence",
    "stat.mineBlock.minecraft.spruce_fence_gate",
    "stat.mineBlock.minecraft.spruce_stairs",
    "stat.mineBlock.minecraft.stained_glass",
    "stat.mineBlock.minecraft.stained_glass_pane",
    "stat.mineBlock.minecraft.stained_hardened_clay",
    "stat.mineBlock.minecraft.sticky_piston",
    "stat.mineBlock.minecraft.stone",
    "stat.mineBlock.minecraft.stone_brick_stairs",
    "stat.mineBlock.minecraft.stone_button",
    "stat.mineBlock.minecraft.stone_pressure_plate",
    "stat.mineBlock.minecraft.stone_slab",
    "stat.mineBlock.minecraft.stone_slab2",
    "stat.mineBlock.minecraft.stone_stairs",
    "stat.mineBlock.minecraft.stonebrick",
    "stat.mineBlock.minecraft.tallgrass",
    "stat.mineBlock.minecraft.tnt",
    "stat.mineBlock.minecraft.torch",
    "stat.mineBlock.minecraft.trapped_chest",
    "stat.mineBlock.minecraft.tripwire_hook",
    "stat.mineBlock.minecraft.vine",
    "stat.mineBlock.minecraft.waterlily",
    "stat.mineBlock.minecraft.web",
    "stat.mineBlock.minecraft.wooden_button",
    "stat.mineBlock.minecraft.wooden_pressure_plate",
    "stat.mineBlock.minecraft.wooden_slab",
    "stat.mineBlock.minecraft.wool",
    "stat.mineBlock.minecraft.yellow_flower",
    "stat.minecartOneCm",
    "stat.mobKills",
    "stat.noteblockPlayed",
    "stat.noteblockTuned",
    "stat.pigOneCm",
    "stat.playerKills",
    "stat.playOneMinute",
    "stat.recordPlayed",
    "stat.sprintOneCm",
    "stat.swimOneCm",
    "stat.talkedToVillager",
    "stat.timeSinceDeath",
    "stat.tradedWithVillager",
    "stat.trappedChestTriggered",
    "stat.treasureFished",
    "stat.useItem.minecraft.acacia_door",
    "stat.useItem.minecraft.acacia_fence",
    "stat.useItem.minecraft.acacia_fence_gate",
    "stat.useItem.minecraft.acacia_stairs",
    "stat.useItem.minecraft.activator_rail",
    "stat.useItem.minecraft.anvil",
    "stat.useItem.minecraft.apple",
    "stat.useItem.minecraft.armor_stand",
    "stat.useItem.minecraft.arrow",
    "stat.useItem.minecraft.baked_potato",
    "stat.useItem.minecraft.banner",
    "stat.useItem.minecraft.barrier",
    "stat.useItem.minecraft.beacon",
    "stat.useItem.minecraft.bed",
    "stat.useItem.minecraft.bedrock",
    "stat.useItem.minecraft.beef",
    "stat.useItem.minecraft.birch_door",
    "stat.useItem.minecraft.birch_fence",
    "stat.useItem.minecraft.birch_fence_gate",
    "stat.useItem.minecraft.birch_stairs",
    "stat.useItem.minecraft.blaze_powder",
    "stat.useItem.minecraft.blaze_rod",
    "stat.useItem.minecraft.boat",
    "stat.useItem.minecraft.bone",
    "stat.useItem.minecraft.book",
    "stat.useItem.minecraft.bookshelf",
    "stat.useItem.minecraft.bow",
    "stat.useItem.minecraft.bowl",
    "stat.useItem.minecraft.bread",
    "stat.useItem.minecraft.brewing_stand",
    "stat.useItem.minecraft.brick",
    "stat.useItem.minecraft.brick_block",
    "stat.useItem.minecraft.brick_stairs",
    "stat.useItem.minecraft.brown_mushroom",
    "stat.useItem.minecraft.brown_mushroom_block",
    "stat.useItem.minecraft.bucket",
    "stat.useItem.minecraft.cactus",
    "stat.useItem.minecraft.cake",
    "stat.useItem.minecraft.carpet",
    "stat.useItem.minecraft.carrot",
    "stat.useItem.minecraft.carrot_on_a_stick",
    "stat.useItem.minecraft.cauldron",
    "stat.useItem.minecraft.chainmail_boots",
    "stat.useItem.minecraft.chainmail_chestplate",
    "stat.useItem.minecraft.chainmail_helmet",
    "stat.useItem.minecraft.chainmail_leggings",
    "stat.useItem.minecraft.chest",
    "stat.useItem.minecraft.chest_minecart",
    "stat.useItem.minecraft.chicken",
    "stat.useItem.minecraft.clay",
    "stat.useItem.minecraft.clay_ball",
    "stat.useItem.minecraft.clock",
    "stat.useItem.minecraft.coal",
    "stat.useItem.minecraft.coal_block",
    "stat.useItem.minecraft.coal_ore",
    "stat.useItem.minecraft.cobblestone",
    "stat.useItem.minecraft.cobblestone_wall",
    "stat.useItem.minecraft.command_block",
    "stat.useItem.minecraft.command_block_minecart",
    "stat.useItem.minecraft.comparator",
    "stat.useItem.minecraft.compass",
    "stat.useItem.minecraft.cooked_beef",
    "stat.useItem.minecraft.cooked_chicken",
    "stat.useItem.minecraft.cooked_fish",
    "stat.useItem.minecraft.cooked_mutton",
    "stat.useItem.minecraft.cooked_porkchop",
    "stat.useItem.minecraft.cooked_rabbit",
    "stat.useItem.minecraft.cookie",
    "stat.useItem.minecraft.crafting_table",
    "stat.useItem.minecraft.dark_oak_door",
    "stat.useItem.minecraft.dark_oak_fence",
    "stat.useItem.minecraft.dark_oak_fence_gate",
    "stat.useItem.minecraft.dark_oak_stairs",
    "stat.useItem.minecraft.daylight_detector",
    "stat.useItem.minecraft.deadbush",
    "stat.useItem.minecraft.detector_rail",
    "stat.useItem.minecraft.diamond",
    "stat.useItem.minecraft.diamond_axe",
    "stat.useItem.minecraft.diamond_block",
    "stat.useItem.minecraft.diamond_boots",
    "stat.useItem.minecraft.diamond_chestplate",
    "stat.useItem.minecraft.diamond_helmet",
    "stat.useItem.minecraft.diamond_hoe",
    "stat.useItem.minecraft.diamond_horse_armor",
    "stat.useItem.minecraft.diamond_leggings",
    "stat.useItem.minecraft.diamond_ore",
    "stat.useItem.minecraft.diamond_pickaxe",
    "stat.useItem.minecraft.diamond_shovel",
    "stat.useItem.minecraft.diamond_sword",
    "stat.useItem.minecraft.dirt",
    "stat.useItem.minecraft.dispenser",
    "stat.useItem.minecraft.double_plant",
    "stat.useItem.minecraft.dragon_egg",
    "stat.useItem.minecraft.dropper",
    "stat.useItem.minecraft.dye",
    "stat.useItem.minecraft.egg",
    "stat.useItem.minecraft.emerald",
    "stat.useItem.minecraft.emerald_block",
    "stat.useItem.minecraft.emerald_ore",
    "stat.useItem.minecraft.enchanted_book",
    "stat.useItem.minecraft.enchanting_table",
    "stat.useItem.minecraft.end_portal_frame",
    "stat.useItem.minecraft.end_stone",
    "stat.useItem.minecraft.ender_chest",
    "stat.useItem.minecraft.ender_eye",
    "stat.useItem.minecraft.ender_pearl",
    "stat.useItem.minecraft.experience_bottle",
    "stat.useItem.minecraft.feather",
    "stat.useItem.minecraft.fence",
    "stat.useItem.minecraft.fence_gate",
    "stat.useItem.minecraft.fermented_spider_eye",
    "stat.useItem.minecraft.filled_map",
    "stat.useItem.minecraft.fire_charge",
    "stat.useItem.minecraft.firework_charge",
    "stat.useItem.minecraft.fireworks",
    "stat.useItem.minecraft.fish",
    "stat.useItem.minecraft.fishing_rod",
    "stat.useItem.minecraft.flint",
    "stat.useItem.minecraft.flint_and_steel",
    "stat.useItem.minecraft.flower_pot",
    "stat.useItem.minecraft.furnace",
    "stat.useItem.minecraft.furnace_minecart",
    "stat.useItem.minecraft.ghast_tear",
    "stat.useItem.minecraft.glass",
    "stat.useItem.minecraft.glass_bottle",
    "stat.useItem.minecraft.glass_pane",
    "stat.useItem.minecraft.glowstone",
    "stat.useItem.minecraft.glowstone_dust",
    "stat.useItem.minecraft.gold_block",
    "stat.useItem.minecraft.gold_ingot",
    "stat.useItem.minecraft.gold_nugget",
    "stat.useItem.minecraft.gold_ore",
    "stat.useItem.minecraft.golden_apple",
    "stat.useItem.minecraft.golden_axe",
    "stat.useItem.minecraft.golden_boots",
    "stat.useItem.minecraft.golden_carrot",
    "stat.useItem.minecraft.golden_chestplate",
    "stat.useItem.minecraft.golden_helmet",
    "stat.useItem.minecraft.golden_hoe",
    "stat.useItem.minecraft.golden_horse_armor",
    "stat.useItem.minecraft.golden_leggings",
    "stat.useItem.minecraft.golden_pickaxe",
    "stat.useItem.minecraft.golden_rail",
    "stat.useItem.minecraft.golden_shovel",
    "stat.useItem.minecraft.golden_sword",
    "stat.useItem.minecraft.gravel",
    "stat.useItem.minecraft.gunpowder",
    "stat.useItem.minecraft.hardened_clay",
    "stat.useItem.minecraft.hay_block",
    "stat.useItem.minecraft.heavy_weighted_pressure_plate",
    "stat.useItem.minecraft.hopper",
    "stat.useItem.minecraft.hopper_minecart",
    "stat.useItem.minecraft.ice",
    "stat.useItem.minecraft.iron_axe",
    "stat.useItem.minecraft.iron_bars",
    "stat.useItem.minecraft.iron_block",
    "stat.useItem.minecraft.iron_boots",
    "stat.useItem.minecraft.iron_chestplate",
    "stat.useItem.minecraft.iron_door",
    "stat.useItem.minecraft.iron_helmet",
    "stat.useItem.minecraft.iron_hoe",
    "stat.useItem.minecraft.iron_horse_armor",
    "stat.useItem.minecraft.iron_ingot",
    "stat.useItem.minecraft.iron_leggings",
    "stat.useItem.minecraft.iron_ore",
    "stat.useItem.minecraft.iron_pickaxe",
    "stat.useItem.minecraft.iron_shovel",
    "stat.useItem.minecraft.iron_sword",
    "stat.useItem.minecraft.iron_trapdoor",
    "stat.useItem.minecraft.item_frame",
    "stat.useItem.minecraft.jukebox",
    "stat.useItem.minecraft.jungle_door",
    "stat.useItem.minecraft.jungle_fence",
    "stat.useItem.minecraft.jungle_fence_gate",
    "stat.useItem.minecraft.jungle_stairs",
    "stat.useItem.minecraft.ladder",
    "stat.useItem.minecraft.lapis_block",
    "stat.useItem.minecraft.lapis_ore",
    "stat.useItem.minecraft.lava_bucket",
    "stat.useItem.minecraft.lead",
    "stat.useItem.minecraft.leather",
    "stat.useItem.minecraft.leather_boots",
    "stat.useItem.minecraft.leather_chestplate",
    "stat.useItem.minecraft.leather_helmet",
    "stat.useItem.minecraft.leather_leggings",
    "stat.useItem.minecraft.leaves",
    "stat.useItem.minecraft.leaves2",
    "stat.useItem.minecraft.lever",
    "stat.useItem.minecraft.light_weighted_pressure_plate",
    "stat.useItem.minecraft.log",
    "stat.useItem.minecraft.log2",
    "stat.useItem.minecraft.magma_cream",
    "stat.useItem.minecraft.map",
    "stat.useItem.minecraft.melon",
    "stat.useItem.minecraft.melon_block",
    "stat.useItem.minecraft.melon_seeds",
    "stat.useItem.minecraft.milk_bucket",
    "stat.useItem.minecraft.minecart",
    "stat.useItem.minecraft.mob_spawner",
    "stat.useItem.minecraft.monster_egg",
    "stat.useItem.minecraft.mossy_cobblestone",
    "stat.useItem.minecraft.mushroom_stew",
    "stat.useItem.minecraft.mutton",
    "stat.useItem.minecraft.mycelium",
    "stat.useItem.minecraft.name_tag",
    "stat.useItem.minecraft.nether_brick",
    "stat.useItem.minecraft.nether_brick_fence",
    "stat.useItem.minecraft.nether_brick_stairs",
    "stat.useItem.minecraft.nether_star",
    "stat.useItem.minecraft.nether_wart",
    "stat.useItem.minecraft.netherbrick",
    "stat.useItem.minecraft.netherrack",
    "stat.useItem.minecraft.noteblock",
    "stat.useItem.minecraft.oak_stairs",
    "stat.useItem.minecraft.obsidian",
    "stat.useItem.minecraft.packed_ice",
    "stat.useItem.minecraft.painting",
    "stat.useItem.minecraft.paper",
    "stat.useItem.minecraft.piston",
    "stat.useItem.minecraft.planks",
    "stat.useItem.minecraft.poisonous_potato",
    "stat.useItem.minecraft.porkchop",
    "stat.useItem.minecraft.potato",
    "stat.useItem.minecraft.potion",
    "stat.useItem.minecraft.prismarine",
    "stat.useItem.minecraft.prismarine_crystals",
    "stat.useItem.minecraft.prismarine_shard",
    "stat.useItem.minecraft.pumpkin",
    "stat.useItem.minecraft.pumpkin_pie",
    "stat.useItem.minecraft.pumpkin_seeds",
    "stat.useItem.minecraft.quartz",
    "stat.useItem.minecraft.quartz_block",
    "stat.useItem.minecraft.quartz_ore",
    "stat.useItem.minecraft.quartz_stairs",
    "stat.useItem.minecraft.rabbit",
    "stat.useItem.minecraft.rabbit_foot",
    "stat.useItem.minecraft.rabbit_hide",
    "stat.useItem.minecraft.rabbit_stew",
    "stat.useItem.minecraft.rail",
    "stat.useItem.minecraft.record_11",
    "stat.useItem.minecraft.record_13",
    "stat.useItem.minecraft.record_blocks",
    "stat.useItem.minecraft.record_cat",
    "stat.useItem.minecraft.record_chirp",
    "stat.useItem.minecraft.record_far",
    "stat.useItem.minecraft.record_mall",
    "stat.useItem.minecraft.record_mellohi",
    "stat.useItem.minecraft.record_stal",
    "stat.useItem.minecraft.record_strad",
    "stat.useItem.minecraft.record_wait",
    "stat.useItem.minecraft.record_ward",
    "stat.useItem.minecraft.red_flower",
    "stat.useItem.minecraft.red_mushroom",
    "stat.useItem.minecraft.red_mushroom_block",
    "stat.useItem.minecraft.red_sandstone",
    "stat.useItem.minecraft.red_sandstone_stairs",
    "stat.useItem.minecraft.redstone",
    "stat.useItem.minecraft.redstone_block",
    "stat.useItem.minecraft.redstone_lamp",
    "stat.useItem.minecraft.redstone_ore",
    "stat.useItem.minecraft.redstone_torch",
    "stat.useItem.minecraft.reeds",
    "stat.useItem.minecraft.repeater",
    "stat.useItem.minecraft.rotten_flesh",
    "stat.useItem.minecraft.saddle",
    "stat.useItem.minecraft.sand",
    "stat.useItem.minecraft.sandstone",
    "stat.useItem.minecraft.sandstone_stairs",
    "stat.useItem.minecraft.sapling",
    "stat.useItem.minecraft.sea_lantern",
    "stat.useItem.minecraft.shears",
    "stat.useItem.minecraft.sign",
    "stat.useItem.minecraft.skull",
    "stat.useItem.minecraft.slime",
    "stat.useItem.minecraft.slime_ball",
    "stat.useItem.minecraft.snow",
    "stat.useItem.minecraft.snow_layer",
    "stat.useItem.minecraft.snowball",
    "stat.useItem.minecraft.soul_sand",
    "stat.useItem.minecraft.spawn_egg",
    "stat.useItem.minecraft.speckled_melon",
    "stat.useItem.minecraft.spider_eye",
    "stat.useItem.minecraft.sponge",
    "stat.useItem.minecraft.spruce_door",
    "stat.useItem.minecraft.spruce_fence",
    "stat.useItem.minecraft.spruce_fence_gate",
    "stat.useItem.minecraft.spruce_stairs",
    "stat.useItem.minecraft.stained_glass",
    "stat.useItem.minecraft.stained_glass_pane",
    "stat.useItem.minecraft.stained_hardened_clay",
    "stat.useItem.minecraft.stick",
    "stat.useItem.minecraft.sticky_piston",
    "stat.useItem.minecraft.stone",
    "stat.useItem.minecraft.stone_axe",
    "stat.useItem.minecraft.stone_brick_stairs",
    "stat.useItem.minecraft.stone_button",
    "stat.useItem.minecraft.stone_hoe",
    "stat.useItem.minecraft.stone_pickaxe",
    "stat.useItem.minecraft.stone_pressure_plate",
    "stat.useItem.minecraft.stone_shovel",
    "stat.useItem.minecraft.stone_slab",
    "stat.useItem.minecraft.stone_slab2",
    "stat.useItem.minecraft.stone_stairs",
    "stat.useItem.minecraft.stone_sword",
    "stat.useItem.minecraft.stonebrick",
    "stat.useItem.minecraft.string",
    "stat.useItem.minecraft.sugar",
    "stat.useItem.minecraft.tallgrass",
    "stat.useItem.minecraft.tnt",
    "stat.useItem.minecraft.tnt_minecart",
    "stat.useItem.minecraft.torch",
    "stat.useItem.minecraft.trapdoor",
    "stat.useItem.minecraft.trapped_chest",
    "stat.useItem.minecraft.tripwire_hook",
    "stat.useItem.minecraft.vine",
    "stat.useItem.minecraft.water_bucket",
    "stat.useItem.minecraft.waterlily",
    "stat.useItem.minecraft.web",
    "stat.useItem.minecraft.wheat",
    "stat.useItem.minecraft.wheat_seeds",
    "stat.useItem.minecraft.wooden_axe",
    "stat.useItem.minecraft.wooden_button",
    "stat.useItem.minecraft.wooden_door",
    "stat.useItem.minecraft.wooden_hoe",
    "stat.useItem.minecraft.wooden_pickaxe",
    "stat.useItem.minecraft.wooden_pressure_plate",
    "stat.useItem.minecraft.wooden_shovel",
    "stat.useItem.minecraft.wooden_slab",
    "stat.useItem.minecraft.wooden_sword",
    "stat.useItem.minecraft.wool",
    "stat.useItem.minecraft.writable_book",
    "stat.useItem.minecraft.written_book",
    "stat.useItem.minecraft.yellow_flower",
    "stat.walkOneCm",
];

#[inline]
fn is_known_stat_id(stat_id: &str) -> bool {
    KNOWN_STAT_IDS.contains(&stat_id)
}

/// One recognized statistic update: a raw stat-id string (e.g. `"stat.jump"`) and its value.
/// java: the loop body of `S37PacketStatistics.readPacketData`/`writePacketData`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatisticEntry {
    /// The stat's registry key. Unknown wire entries are consumed and dropped by the decoder,
    /// matching vanilla's `StatList.getOneShotStat` lookup.
    pub stat_id: String,
    /// The stat's new value.
    pub value: i32,
}

/// `0x37 Statistics`: a batch of statistic values, sent in reply to a client's stats request.
/// java: `S37PacketStatistics`
#[derive(Debug, Clone, PartialEq)]
pub struct Statistics {
    /// The updated statistics, in wire order.
    pub entries: Vec<StatisticEntry>,
}

impl Packet for Statistics {
    const ID: i32 = 0x37;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        // java: `for (int j = 0; j < i; ++j)` over a signed VarInt into a HashMap: a negative
        // count is an empty loop, not an error (unlike the `new T[i]` readers).
        let count = buf.read_varint()?;
        let mut entries = Vec::with_capacity(usize::try_from(count).unwrap_or(0).min(4096));
        for _ in 0..count {
            let stat_id = buf.read_string(32767)?;
            let value = buf.read_varint()?;
            if is_known_stat_id(&stat_id) {
                entries.push(StatisticEntry { stat_id, value });
            }
        }
        Ok(Statistics { entries })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entries.len() as i32);
        for entry in &self.entries {
            buf.write_string(&entry.stat_id);
            buf.write_varint(entry.value);
        }
    }
}

// ---- 0x38 PlayerListItem ---------------------------------------------------------------------

/// A player's game mode, matching `WorldSettings.GameType`. Decoding an id outside `-1..=3`
/// falls back to [`GameMode::Survival`], exactly matching `WorldSettings.GameType.getByID`'s own
/// linear-search-with-default (never an error, unlike e.g. [`PlayerListAction`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GameMode {
    /// `-1`: no game mode assigned (rare; mostly a placeholder default).
    NotSet,
    /// `0`.
    Survival,
    /// `1`.
    Creative,
    /// `2`.
    Adventure,
    /// `3`.
    Spectator,
}

impl GameMode {
    fn from_id(id: i32) -> GameMode {
        match id {
            -1 => GameMode::NotSet,
            0 => GameMode::Survival,
            1 => GameMode::Creative,
            2 => GameMode::Adventure,
            3 => GameMode::Spectator,
            _ => GameMode::Survival,
        }
    }

    fn to_id(self) -> i32 {
        match self {
            GameMode::NotSet => -1,
            GameMode::Survival => 0,
            GameMode::Creative => 1,
            GameMode::Adventure => 2,
            GameMode::Spectator => 3,
        }
    }
}

/// The action a [`PlayerListItem`] packet performs, and (via `readEnumValue`/`writeEnumValue`)
/// the packet's own wire ordinal. Unlike [`GameStateReason`]/particle ids, an out-of-range
/// ordinal here **is** a protocol error: vanilla's `readEnumValue` indexes straight into the
/// enum's constant array (`enumClass.getEnumConstants()[id]`), which throws on an out-of-bounds
/// index rather than degrading gracefully.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayerListAction {
    AddPlayer,
    UpdateGameMode,
    UpdateLatency,
    UpdateDisplayName,
    RemovePlayer,
}

impl PlayerListAction {
    fn from_id(id: i32) -> Result<PlayerListAction, ProtoError> {
        match id {
            0 => Ok(PlayerListAction::AddPlayer),
            1 => Ok(PlayerListAction::UpdateGameMode),
            2 => Ok(PlayerListAction::UpdateLatency),
            3 => Ok(PlayerListAction::UpdateDisplayName),
            4 => Ok(PlayerListAction::RemovePlayer),
            other => Err(ProtoError::InvalidEnum {
                what: "player-list-item action",
                value: other,
            }),
        }
    }
}

/// One game-profile property (`com.mojang.authlib.properties.Property`): a name/value pair,
/// optionally signed by Mojang's session server (used for the `"textures"` property, which
/// carries the player's skin/cape URLs). Only sent for [`PlayerListItem::AddPlayer`] entries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileProperty {
    /// The property's name (e.g. `"textures"`).
    pub name: String,
    /// The property's value (often base64-encoded).
    pub value: String,
    /// The session-server signature over `value`, if this property is signed.
    pub signature: Option<String>,
}

impl ProfileProperty {
    fn read(buf: &mut PacketBuffer) -> Result<ProfileProperty, ProtoError> {
        let name = buf.read_string(32767)?;
        let value = buf.read_string(32767)?;
        let signature = if buf.read_bool()? {
            Some(buf.read_string(32767)?)
        } else {
            None
        };
        Ok(ProfileProperty {
            name,
            value,
            signature,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.name);
        buf.write_string(&self.value);
        match &self.signature {
            Some(sig) => {
                buf.write_bool(true);
                buf.write_string(sig);
            }
            None => buf.write_bool(false),
        }
    }
}

/// One entry under [`PlayerListItem::AddPlayer`]: a full game profile plus initial tab-list
/// state. java: `S38PacketPlayerListItem.AddPlayerData`, `ADD_PLAYER` case
#[derive(Debug, Clone, PartialEq)]
pub struct AddPlayerEntry {
    /// The player's UUID.
    pub uuid: Uuid,
    /// The player's username.
    pub username: String,
    /// The profile's signed properties (in practice, usually just `"textures"`).
    pub properties: Vec<ProfileProperty>,
    /// The player's starting game mode.
    pub game_mode: GameMode,
    /// The player's starting ping, in milliseconds.
    pub ping: i32,
    /// A tab-list display-name override, as chat-component JSON, if set.
    pub display_name: Option<String>,
}

/// One entry under [`PlayerListItem::UpdateGameMode`]. java: `UPDATE_GAME_MODE` case
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateGameModeEntry {
    /// The player being updated.
    pub uuid: Uuid,
    /// Their new game mode.
    pub game_mode: GameMode,
}

/// One entry under [`PlayerListItem::UpdateLatency`]. java: `UPDATE_LATENCY` case
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UpdateLatencyEntry {
    /// The player being updated.
    pub uuid: Uuid,
    /// Their new ping, in milliseconds.
    pub ping: i32,
}

/// One entry under [`PlayerListItem::UpdateDisplayName`]. java: `UPDATE_DISPLAY_NAME` case
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateDisplayNameEntry {
    /// The player being updated.
    pub uuid: Uuid,
    /// Their new tab-list display-name override, as chat-component JSON, or `None` to clear it
    /// back to their plain username.
    pub display_name: Option<String>,
}

/// `0x38 Player List Item`: adds, removes, or updates entries in the tab-list player roster.
///
/// Vanilla represents this as one `action` field plus a `List<AddPlayerData>` whose unused
/// fields are simply left at their zero value depending on `action` (see the Java `switch` in
/// `readPacketData`). This port instead makes the action-to-payload relationship a Rust
/// invariant: each action carries a `Vec` of *only* the fields that action's entries actually
/// have on the wire, so a decoded [`PlayerListItem::UpdateLatency`] literally cannot carry a
/// stray `username` the way vanilla's shared `AddPlayerData` class could (harmlessly) hold one.
/// java: `S38PacketPlayerListItem`
#[derive(Debug, Clone, PartialEq)]
pub enum PlayerListItem {
    /// A player joined visibility range / the tab list (usually right after they join the
    /// server); `action == 0`.
    AddPlayer(Vec<AddPlayerEntry>),
    /// A player's game mode changed; `action == 1`.
    UpdateGameMode(Vec<UpdateGameModeEntry>),
    /// A player's measured ping changed; `action == 2`.
    UpdateLatency(Vec<UpdateLatencyEntry>),
    /// A player's tab-list display name changed; `action == 3`.
    UpdateDisplayName(Vec<UpdateDisplayNameEntry>),
    /// Players left visibility range / the tab list; `action == 4`. Only a UUID is sent per
    /// entry, so this carries a plain `Vec<Uuid>` rather than a single-field entry struct.
    RemovePlayer(Vec<Uuid>),
}

impl Packet for PlayerListItem {
    const ID: i32 = 0x38;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let action = PlayerListAction::from_id(buf.read_varint()?)?;
        // java: both counts drive `for (int j = 0; j < i; ++j)` loops into a list/multimap, so
        // a negative signed VarInt is an empty loop, never a decoder exception. The capacity
        // hint only bounds an up-front allocation; entries are still read one at a time.
        let count = usize::try_from(buf.read_varint()?).unwrap_or(0);
        let capacity = count.min(4096);

        match action {
            PlayerListAction::AddPlayer => {
                let mut entries = Vec::with_capacity(capacity);
                for _ in 0..count {
                    let uuid = buf.read_uuid()?;
                    let username = buf.read_string(16)?;
                    let prop_count = buf.read_varint()?;
                    let mut properties =
                        Vec::with_capacity(usize::try_from(prop_count).unwrap_or(0).min(4096));
                    for _ in 0..prop_count {
                        properties.push(ProfileProperty::read(buf)?);
                    }
                    let game_mode = GameMode::from_id(buf.read_varint()?);
                    let ping = buf.read_varint()?;
                    let display_name = if buf.read_bool()? {
                        Some(buf.read_chat()?)
                    } else {
                        None
                    };
                    entries.push(AddPlayerEntry {
                        uuid,
                        username,
                        properties,
                        game_mode,
                        ping,
                        display_name,
                    });
                }
                Ok(PlayerListItem::AddPlayer(entries))
            }
            PlayerListAction::UpdateGameMode => {
                let mut entries = Vec::with_capacity(capacity);
                for _ in 0..count {
                    let uuid = buf.read_uuid()?;
                    let game_mode = GameMode::from_id(buf.read_varint()?);
                    entries.push(UpdateGameModeEntry { uuid, game_mode });
                }
                Ok(PlayerListItem::UpdateGameMode(entries))
            }
            PlayerListAction::UpdateLatency => {
                let mut entries = Vec::with_capacity(capacity);
                for _ in 0..count {
                    let uuid = buf.read_uuid()?;
                    let ping = buf.read_varint()?;
                    entries.push(UpdateLatencyEntry { uuid, ping });
                }
                Ok(PlayerListItem::UpdateLatency(entries))
            }
            PlayerListAction::UpdateDisplayName => {
                let mut entries = Vec::with_capacity(capacity);
                for _ in 0..count {
                    let uuid = buf.read_uuid()?;
                    let display_name = if buf.read_bool()? {
                        Some(buf.read_chat()?)
                    } else {
                        None
                    };
                    entries.push(UpdateDisplayNameEntry { uuid, display_name });
                }
                Ok(PlayerListItem::UpdateDisplayName(entries))
            }
            PlayerListAction::RemovePlayer => {
                let mut entries = Vec::with_capacity(capacity);
                for _ in 0..count {
                    entries.push(buf.read_uuid()?);
                }
                Ok(PlayerListItem::RemovePlayer(entries))
            }
        }
    }

    fn write(&self, buf: &mut PacketBuffer) {
        match self {
            PlayerListItem::AddPlayer(entries) => {
                buf.write_varint(0);
                buf.write_varint(entries.len() as i32);
                for e in entries {
                    buf.write_uuid(e.uuid);
                    buf.write_string(&e.username);
                    buf.write_varint(e.properties.len() as i32);
                    for p in &e.properties {
                        p.write(buf);
                    }
                    buf.write_varint(e.game_mode.to_id());
                    buf.write_varint(e.ping);
                    match &e.display_name {
                        Some(name) => {
                            buf.write_bool(true);
                            buf.write_chat(name);
                        }
                        None => buf.write_bool(false),
                    }
                }
            }
            PlayerListItem::UpdateGameMode(entries) => {
                buf.write_varint(1);
                buf.write_varint(entries.len() as i32);
                for e in entries {
                    buf.write_uuid(e.uuid);
                    buf.write_varint(e.game_mode.to_id());
                }
            }
            PlayerListItem::UpdateLatency(entries) => {
                buf.write_varint(2);
                buf.write_varint(entries.len() as i32);
                for e in entries {
                    buf.write_uuid(e.uuid);
                    buf.write_varint(e.ping);
                }
            }
            PlayerListItem::UpdateDisplayName(entries) => {
                buf.write_varint(3);
                buf.write_varint(entries.len() as i32);
                for e in entries {
                    buf.write_uuid(e.uuid);
                    match &e.display_name {
                        Some(name) => {
                            buf.write_bool(true);
                            buf.write_chat(name);
                        }
                        None => buf.write_bool(false),
                    }
                }
            }
            PlayerListItem::RemovePlayer(uuids) => {
                buf.write_varint(4);
                buf.write_varint(uuids.len() as i32);
                for uuid in uuids {
                    buf.write_uuid(*uuid);
                }
            }
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
    fn effect_round_trips() {
        round_trip(Effect {
            effect_id: 1003,
            position: BlockPos::new(10, 64, -20),
            data: 0,
            disable_relative_volume: true,
        });
    }

    #[test]
    fn effect_server_wide_byte_selects_the_broadcast_table() {
        fn decode_server_wide_byte(raw: u8) -> Effect {
            let mut buf = PacketBuffer::new();
            buf.write_i32(1013);
            buf.write_block_pos(BlockPos::new(10, 64, -20));
            buf.write_i32(0);
            buf.write_u8(raw);

            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = Effect::read(&mut read_buf).unwrap();
            assert!(read_buf.is_empty());
            packet
        }

        // PacketBuffer.readBoolean and Rust's read_bool both use raw-byte != 0. The field's
        // meaning is dispatch-table selection, not merely disabled distance attenuation.
        assert!(!decode_server_wide_byte(0).disable_relative_volume);
        assert!(decode_server_wide_byte(1).disable_relative_volume);
        assert!(decode_server_wide_byte(u8::MAX).disable_relative_volume);
    }

    #[test]
    fn sound_effect_round_trips() {
        round_trip(SoundEffect {
            sound_name: "random.explode".into(),
            x_fixed: 80,
            y_fixed: 512,
            z_fixed: -160,
            volume: 1.0,
            pitch_quantized: 63,
        });
    }

    #[test]
    fn particles_with_no_extra_data_round_trips() {
        round_trip(Particles {
            particle_id: 30, // REDSTONE
            long_distance: false,
            x: 1.0,
            y: 2.0,
            z: 3.0,
            offset_x: 0.1,
            offset_y: 0.2,
            offset_z: 0.3,
            particle_speed: 0.0,
            particle_count: 10,
            extra_data: vec![],
        });
    }

    fn raw_particle_packet(particle_id: i32) -> PacketBuffer {
        let mut buf = PacketBuffer::new();
        buf.write_i32(particle_id);
        buf.write_bool(false);
        for _ in 0..7 {
            buf.write_f32(0.0);
        }
        buf.write_i32(0);
        buf
    }

    /// S37's entry count and S38's entry and property counts
    /// drive Java `for (int j = 0; j < i; ++j)` loops, so a negative signed VarInt reads nothing
    /// and decoding carries on. The former `NegativeLength` guards disconnected instead. (The
    /// `new T[i]` readers, such as S13, S22, S30 and S3A, do throw and keep their guards.)
    #[test]
    fn statistics_and_player_list_counts_are_signed_loops_like_vanilla() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(
            Statistics::read(&mut read_buf).unwrap(),
            Statistics { entries: vec![] }
        );
        assert!(read_buf.is_empty());

        for action in 0..=4 {
            let mut buf = PacketBuffer::new();
            buf.write_varint(action);
            buf.write_varint(i32::MIN);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = PlayerListItem::read(&mut read_buf)
                .unwrap_or_else(|error| panic!("action {action}: {error}"));
            assert!(read_buf.is_empty());
            let len = match packet {
                PlayerListItem::AddPlayer(entries) => entries.len(),
                PlayerListItem::UpdateGameMode(entries) => entries.len(),
                PlayerListItem::UpdateLatency(entries) => entries.len(),
                PlayerListItem::UpdateDisplayName(entries) => entries.len(),
                PlayerListItem::RemovePlayer(entries) => entries.len(),
            };
            assert_eq!(len, 0, "action {action}");
        }

        // A negative property count skips the property loop and the entry keeps decoding.
        let mut buf = PacketBuffer::new();
        buf.write_varint(0);
        buf.write_varint(1);
        buf.write_uuid(Uuid::from_u128(7));
        buf.write_string("Notch");
        buf.write_varint(-1);
        buf.write_varint(1);
        buf.write_varint(42);
        buf.write_bool(false);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        match PlayerListItem::read(&mut read_buf).unwrap() {
            PlayerListItem::AddPlayer(entries) => {
                assert_eq!(entries.len(), 1);
                assert!(entries[0].properties.is_empty());
                assert_eq!(entries[0].username, "Notch");
                assert_eq!(entries[0].game_mode, GameMode::Creative);
                assert_eq!(entries[0].ping, 42);
                assert_eq!(entries[0].display_name, None);
            }
            other => panic!("expected ADD_PLAYER, got {other:?}"),
        }
        assert!(read_buf.is_empty());
    }

    #[test]
    fn particles_unknown_positive_and_negative_ids_decode_as_barrier() {
        for particle_id in [42, i32::MAX, -1, i32::MIN] {
            let buf = raw_particle_packet(particle_id);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let decoded = Particles::read(&mut read_buf).unwrap();
            assert_eq!(decoded.particle_id, 35, "wire id {particle_id}");
            assert!(decoded.extra_data.is_empty());
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn particles_unknown_ids_select_barrier_argument_count() {
        for particle_id in [42, -1] {
            let mut buf = raw_particle_packet(particle_id);
            // This is the next framed value, not a particle argument. Unknown ids resolve to
            // BARRIER (zero arguments), so the decoder must leave it unread.
            buf.write_varint(1234);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let decoded = Particles::read(&mut read_buf).unwrap();
            assert_eq!(decoded.particle_id, 35);
            assert!(decoded.extra_data.is_empty());
            assert_eq!(read_buf.read_varint().unwrap(), 1234);
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn particles_unknown_id_decode_then_encode_uses_barrier_wire_id() {
        let input = raw_particle_packet(9999);
        let mut read_buf = PacketBuffer::from_slice(input.as_slice());
        let decoded = Particles::read(&mut read_buf).unwrap();
        let mut output = PacketBuffer::new();
        decoded.write(&mut output);

        let mut encoded = PacketBuffer::from_slice(output.as_slice());
        assert_eq!(encoded.read_i32().unwrap(), 35);
        let encoded_body = encoded.as_slice().to_vec();
        assert_eq!(
            encoded_body.as_slice(),
            &input.as_slice()[4..],
            "normal fields remain unchanged after unknown-id normalization"
        );
    }

    #[test]
    fn particles_all_known_ids_round_trip_without_normalization() {
        for particle_id in 0..=41 {
            let extra_data = match particle_id {
                36 => vec![280, 2],
                37 | 38 => vec![1],
                _ => vec![],
            };
            round_trip(Particles {
                particle_id,
                long_distance: false,
                x: 1.0,
                y: 2.0,
                z: 3.0,
                offset_x: 0.1,
                offset_y: 0.2,
                offset_z: 0.3,
                particle_speed: 0.4,
                particle_count: 5,
                extra_data,
            });
        }
    }

    #[test]
    fn particles_item_crack_carries_two_extra_ints() {
        let pkt = Particles {
            particle_id: 36, // ITEM_CRACK
            long_distance: true,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 1.0,
            particle_count: 1,
            extra_data: vec![280, 0],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let decoded = Particles::read(&mut read_buf).unwrap();
        assert_eq!(decoded, pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn particles_block_crack_carries_one_extra_int() {
        let pkt = Particles {
            particle_id: 37, // BLOCK_CRACK
            long_distance: false,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 0.0,
            particle_count: 4,
            extra_data: vec![1],
        };
        round_trip(pkt);
    }

    #[test]
    fn particles_writer_ignores_surplus_extra_data_at_the_particle_enum_boundary() {
        // S2APacketParticles.writePacketData iterates getArgumentCount(), not the supplied
        // varargs array.  The second value must not become a trailing byte in a framed 0x2A
        // body: BLOCK_CRACK has exactly one argument.
        let pkt = Particles {
            particle_id: 37, // BLOCK_CRACK
            long_distance: false,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 0.0,
            particle_count: 0,
            extra_data: vec![0x1234, 0x7f],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let decoded = Particles::read(&mut read_buf).unwrap();
        assert_eq!(decoded.extra_data, vec![0x1234]);
        assert!(read_buf.is_empty());
    }

    #[test]
    #[should_panic(expected = "requires 2 extra data values, got 1")]
    fn particles_writer_rejects_missing_required_extra_data_like_java_array_indexing() {
        // ITEM_CRACK has two enum-declared entries.  Java indexes both and throws rather than
        // emitting a truncated packet; the Rust wire writer must not silently do the latter.
        Particles {
            particle_id: 36, // ITEM_CRACK
            long_distance: false,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 0.0,
            particle_count: 0,
            extra_data: vec![280],
        }
        .write(&mut PacketBuffer::new());
    }

    #[test]
    fn change_game_state_round_trips_known_reason() {
        round_trip(ChangeGameState {
            reason: GameStateReason::RainLevelChange,
            value: 0.5,
        });
    }

    #[test]
    fn change_game_state_unknown_reason_is_not_an_error() {
        // Vanilla's own handler silently ignores unrecognized reason bytes rather than
        // rejecting the packet, so decoding must not error either.
        let mut buf = PacketBuffer::new();
        buf.write_u8(200);
        buf.write_f32(1.0);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let decoded = ChangeGameState::read(&mut read_buf).unwrap();
        assert_eq!(decoded.reason, GameStateReason::Other(200));
        assert_eq!(decoded.reason.to_id(), 200);
    }

    #[test]
    fn spawn_global_entity_round_trips() {
        round_trip(SpawnGlobalEntity {
            entity_id: 42,
            entity_type: 1,
            x_fixed: 320,
            y_fixed: 2048,
            z_fixed: -640,
        });
    }

    #[test]
    fn open_window_without_horse_round_trips() {
        round_trip(OpenWindow {
            window_id: 1,
            inventory_type: "minecraft:chest".into(),
            window_title: r#"{"text":"Chest"}"#.into(),
            slot_count: 27,
            entity_id: None,
        });
    }

    #[test]
    fn open_window_horse_carries_entity_id() {
        let pkt = OpenWindow {
            window_id: 2,
            inventory_type: "EntityHorse".into(),
            window_title: r#"{"text":"Horse"}"#.into(),
            slot_count: 2,
            entity_id: Some(1234),
        };
        // `round_trip` (via its `read_buf.is_empty()` check) already proves the entity-id field
        // is both written and consumed exactly when `inventory_type == "EntityHorse"`: if either
        // side got the condition wrong, the two would desync and either error out or leave
        // trailing bytes.
        round_trip(pkt);
    }

    #[test]
    fn close_window_round_trips() {
        round_trip(CloseWindow { window_id: 3 });
    }

    #[test]
    fn set_slot_round_trips_including_negative_cursor_window() {
        round_trip(SetSlot {
            window_id: -1,
            slot: -1,
            item: None,
        });
        round_trip(SetSlot {
            window_id: 0,
            slot: 5,
            item: Some(ItemStack {
                item_id: 1,
                count: 64,
                damage: 0,
                nbt: None,
            }),
        });
    }

    #[test]
    fn window_items_round_trips() {
        round_trip(WindowItems {
            window_id: 0,
            items: vec![
                None,
                Some(ItemStack {
                    item_id: 5,
                    count: 1,
                    damage: 0,
                    nbt: None,
                }),
            ],
        });
    }

    #[test]
    fn window_property_round_trips() {
        round_trip(WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 200,
        });
    }

    #[test]
    fn confirm_transaction_round_trips() {
        round_trip(ConfirmTransaction {
            window_id: 1,
            action_number: 7,
            accepted: false,
        });
    }

    #[test]
    fn update_sign_round_trips() {
        round_trip(UpdateSign {
            position: BlockPos::new(1, 65, 1),
            lines: [
                r#"{"text":"line1"}"#.into(),
                r#"{"text":"line2"}"#.into(),
                r#"{"text":""}"#.into(),
                r#"{"text":""}"#.into(),
            ],
        });
    }

    #[test]
    fn maps_round_trips_with_icons_and_no_update() {
        round_trip(Maps {
            map_id: 0,
            scale: 3,
            icons: vec![MapIcon {
                icon_type: 1,
                x: -10,
                z: 20,
                direction: 15,
            }],
            update: None,
        });
    }

    #[test]
    fn maps_round_trips_with_pixel_update() {
        round_trip(Maps {
            map_id: 7,
            scale: 0,
            icons: vec![],
            update: Some(MapPixelUpdate {
                width: 4,
                height: 2,
                min_x: 1,
                min_y: 1,
                data: (0..8).collect(),
            }),
        });
    }

    #[test]
    fn map_icon_nibble_packing_round_trips_at_the_edges() {
        let mut buf = PacketBuffer::new();
        let icon = MapIcon {
            icon_type: 15,
            x: -128,
            z: 127,
            direction: 0,
        };
        icon.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(MapIcon::read(&mut read_buf).unwrap(), icon);
    }

    #[test]
    fn update_tile_entity_round_trips_with_and_without_nbt() {
        round_trip(UpdateTileEntity {
            position: BlockPos::new(-5, 70, 5),
            action: 1,
            nbt: None,
        });
        let mut nbt = NbtCompound::new();
        nbt.insert("id", mc_nbt::Nbt::String("MobSpawner".into()));
        round_trip(UpdateTileEntity {
            position: BlockPos::new(-5, 70, 5),
            action: 1,
            nbt: Some(nbt),
        });
    }

    #[test]
    fn sign_editor_open_round_trips() {
        round_trip(SignEditorOpen {
            position: BlockPos::new(3, 4, 5),
        });
    }

    #[test]
    fn statistics_round_trips() {
        round_trip(Statistics {
            entries: vec![
                StatisticEntry {
                    stat_id: "stat.mineBlock.minecraft.stone".into(),
                    value: 12,
                },
                StatisticEntry {
                    stat_id: "stat.jump".into(),
                    value: 999,
                },
            ],
        });
    }

    #[test]
    fn statistics_filters_unknown_ids_and_preserves_wire_alignment() {
        assert_eq!(KNOWN_STAT_IDS.len(), 883);
        let wire_entries = [
            StatisticEntry {
                stat_id: "stat.jump".into(),
                value: 7,
            },
            StatisticEntry {
                stat_id: "stat.jump.not_registered".into(),
                value: 1234,
            },
            StatisticEntry {
                stat_id: "stat.mineBlock.minecraft.stone".into(),
                value: -3,
            },
            StatisticEntry {
                stat_id: "stat.mineBlock.minecraft.not_a_block".into(),
                value: 99,
            },
        ];
        let mut buf = PacketBuffer::new();
        buf.write_varint(wire_entries.len() as i32);
        for entry in &wire_entries {
            buf.write_string(&entry.stat_id);
            buf.write_varint(entry.value);
        }
        buf.write_varint(0x5a);

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let decoded = Statistics::read(&mut read_buf).unwrap();
        assert_eq!(
            decoded.entries,
            vec![wire_entries[0].clone(), wire_entries[2].clone(),]
        );
        assert_eq!(read_buf.read_varint().unwrap(), 0x5a);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn statistics_empty_round_trips() {
        round_trip(Statistics { entries: vec![] });
    }

    #[test]
    fn player_list_item_add_player_round_trips() {
        let uuid = Uuid::from_u128(0x1111_2222_3333_4444_5555_6666_7777_8888);
        round_trip(PlayerListItem::AddPlayer(vec![AddPlayerEntry {
            uuid,
            username: "Notch".into(),
            properties: vec![ProfileProperty {
                name: "textures".into(),
                value: "base64stuff".into(),
                signature: Some("sig".into()),
            }],
            game_mode: GameMode::Creative,
            ping: 42,
            display_name: Some(r#"{"text":"Notch"}"#.into()),
        }]));
    }

    #[test]
    fn player_list_item_add_player_with_unsigned_property_and_no_display_name() {
        let uuid = Uuid::from_u128(1);
        round_trip(PlayerListItem::AddPlayer(vec![AddPlayerEntry {
            uuid,
            username: "Steve".into(),
            properties: vec![ProfileProperty {
                name: "textures".into(),
                value: "base64stuff".into(),
                signature: None,
            }],
            game_mode: GameMode::Survival,
            ping: 10,
            display_name: None,
        }]));
    }

    #[test]
    fn player_list_item_update_game_mode_round_trips() {
        round_trip(PlayerListItem::UpdateGameMode(vec![UpdateGameModeEntry {
            uuid: Uuid::from_u128(2),
            game_mode: GameMode::Spectator,
        }]));
    }

    #[test]
    fn player_list_item_update_latency_round_trips() {
        round_trip(PlayerListItem::UpdateLatency(vec![UpdateLatencyEntry {
            uuid: Uuid::from_u128(3),
            ping: 123,
        }]));
    }

    #[test]
    fn player_list_item_update_display_name_round_trips() {
        round_trip(PlayerListItem::UpdateDisplayName(vec![
            UpdateDisplayNameEntry {
                uuid: Uuid::from_u128(4),
                display_name: Some(r#"{"text":"Fancy"}"#.into()),
            },
            UpdateDisplayNameEntry {
                uuid: Uuid::from_u128(5),
                display_name: None,
            },
        ]));
    }

    #[test]
    fn player_list_item_remove_player_round_trips() {
        round_trip(PlayerListItem::RemovePlayer(vec![
            Uuid::from_u128(6),
            Uuid::from_u128(7),
        ]));
    }

    #[test]
    fn player_list_item_unknown_action_errors() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(99);
        buf.write_varint(0);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            PlayerListItem::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { .. })
        ));
    }

    #[test]
    fn player_list_item_decodes_past_the_preallocation_cap() {
        const COUNT: usize = (1 << 20) + 1;

        let mut buf = PacketBuffer::new();
        buf.write_varint(4); // RemovePlayer: the smallest entry is one 16-byte UUID.
        buf.write_varint(COUNT as i32);
        for id in 0..COUNT {
            buf.write_uuid(Uuid::from_u128(id as u128));
        }

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let packet = PlayerListItem::read(&mut read_buf)
            .expect("the allocation cap must not truncate the declared entry count");
        let PlayerListItem::RemovePlayer(entries) = packet else {
            panic!("remove-player action decoded as another variant");
        };
        assert_eq!(entries.len(), COUNT);
        assert_eq!(entries.last(), Some(&Uuid::from_u128((COUNT - 1) as u128)));
        assert!(read_buf.is_empty());
    }

    #[test]
    fn game_mode_unknown_id_falls_back_to_survival() {
        assert_eq!(GameMode::from_id(42), GameMode::Survival);
        assert_eq!(GameMode::from_id(-1), GameMode::NotSet);
    }
}
