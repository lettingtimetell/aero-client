//! [`ClientEvent`]: what the dispatch reports that is *not* a mutation of
//! [`ClientState`](crate::sync::ClientState).
//!
//! Vanilla's handlers reach straight into the renderer, the sound engine and the GUI. Those
//! subsystems live in other crates here, and S14 does not own them, so anything that would have
//! been such a call becomes an event instead. The rule the dispatch follows:
//!
//! * a change to world/ECS/tab-list/environment state is **applied**, and the event (if any)
//!   only announces that it happened;
//! * a call into a subsystem this section does not own — sound, particles, containers, screens,
//!   the scoreboard overlay — is **reported verbatim**, so the owning section can consume it
//!   without S14 having guessed at its API.

use mc_math::BlockPos;
use mc_nbt::NbtCompound;
use mc_protocol::play::s2c;
use mc_protocol::ItemStack;
use mc_ui::font::component::ChatComponent;
use mc_world::Entity;

use crate::net::DisconnectReason;
use crate::sync::state::Abilities;

/// Where a chat message goes. java: `S02PacketChat.getType`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatPosition {
    /// `0` — an ordinary chat message.
    Chat,
    /// `1` — a system message. Vanilla still routes it to the chat box.
    System,
    /// `2` — the action bar above the hotbar. java: `GuiIngame.setRecordPlaying`
    ActionBar,
    /// Any other byte. Vanilla does no validation, so neither does this; it is treated as
    /// [`ChatPosition::Chat`] by `is_chat_box`.
    Other(i8),
}

impl ChatPosition {
    /// Decode the wire byte.
    #[must_use]
    pub fn from_raw(position: i8) -> ChatPosition {
        match position {
            0 => ChatPosition::Chat,
            1 => ChatPosition::System,
            2 => ChatPosition::ActionBar,
            other => ChatPosition::Other(other),
        }
    }

    /// Whether this message belongs in the chat box.
    ///
    /// java: `handleChat` (`:849`) branches on `getType() == 2` and sends **everything else**
    /// to the chat GUI — including unknown positions.
    #[must_use]
    pub fn is_chat_box(self) -> bool {
        !matches!(self, ChatPosition::ActionBar)
    }
}

/// Container/inventory traffic. Owned by S17; S14 only carries it across.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum WindowEvent {
    /// java: `handleOpenWindow` (`:1092`)
    Open(s2c::OpenWindow),
    /// java: `handleCloseWindow` (`:1311`)
    Close {
        /// The window id the server closed.
        window_id: u8,
    },
    /// java: `handleSetSlot` (`:1133`)
    SetSlot {
        /// `-1` is the cursor stack; `0` is the player inventory.
        window_id: i8,
        /// The slot index within that window.
        slot: i16,
        /// The new contents.
        item: Option<ItemStack>,
    },
    /// java: `handleWindowItems` (`:1198`)
    Items {
        /// The window being filled.
        window_id: u8,
        /// One entry per slot, in slot order.
        items: Vec<Option<ItemStack>>,
    },
    /// java: `handleWindowProperty` (`:1286`) — furnace burn time, enchanting levels, …
    Property {
        /// The window.
        window_id: u8,
        /// Which property.
        index: i16,
        /// Its new value.
        value: i16,
    },
    /// java: `handleConfirmTransaction` (`:1174`)
    ConfirmTransaction {
        /// The window.
        window_id: u8,
        /// The action number being acknowledged or rejected.
        action_number: i16,
        /// Whether the server accepted it. A rejection must be echoed back.
        accepted: bool,
    },
}

/// Scoreboard and team traffic. Owned by S16's HUD; S14 only carries it across.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ScoreboardEvent {
    /// java: `handleScoreboardObjective` (`:1869`)
    Objective(s2c::ScoreboardObjective),
    /// java: `handleUpdateScore` (`:1899`)
    Score(s2c::UpdateScore),
    /// java: `handleDisplayScoreboard` (`:1927`)
    Display(s2c::DisplayScoreboard),
    /// java: `handleTeams` (`:1947`)
    Teams(Box<s2c::Teams>),
}

/// Something the dispatch wants the rest of the client to know about.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClientEvent {
    // ---- lifecycle ----
    /// The TCP socket is open and the login handshake is starting.
    SocketConnected,
    /// `S01PacketJoinGame` landed and [`crate::sync::ClientState`] has been reset for it.
    /// java: `handleJoinGame` (`:277`)
    JoinedGame(Box<s2c::JoinGame>),
    /// `S07PacketRespawn`. `changed_world` is true when the dimension differed, which is the
    /// case in which vanilla throws the whole `WorldClient` away. java: `handleRespawn` (`:1056`)
    Respawned {
        /// The dimension respawned into.
        dimension: i32,
        /// Whether the world was rebuilt.
        changed_world: bool,
    },
    /// `S08PacketPlayerPosLook`. The caller must feed this to `mc_physics::Predictor` and send
    /// the resulting `C06PacketPlayerPosLook` back — the server discards everything the client
    /// says until it does. java: `handlePlayerPosLook` (`:669`)
    PositionCorrection(mc_physics::prediction::PositionCorrection),
    /// The first position correction after a join or a dimension change, which is what dismisses
    /// vanilla's `GuiDownloadTerrain`. java: the `doneLoadingTerrain` flip at `:719`
    TerrainDownloaded,
    /// The connection ended.
    Disconnected(DisconnectReason),

    // ---- chat and screens ----
    /// java: `handleChat` (`:849`)
    Chat {
        /// The parsed component tree.
        component: Box<ChatComponent>,
        /// Where it goes.
        position: ChatPosition,
    },
    /// java: `handleTitle` (`:1565`)
    Title(Box<s2c::Title>),
    /// java: `handleTabComplete` (`:1683`)
    TabComplete(Vec<String>),
    /// The tab list header/footer changed; read them off
    /// [`PlayerList`](crate::sync::players::PlayerList). java: `:1600`
    PlayerListHeaderFooter,
    /// java: `handleResourcePack` (`:1701`). The caller decides whether to accept, and must
    /// answer with `C19PacketResourcePackStatus`.
    ResourcePack {
        /// Where to fetch it.
        url: String,
        /// Its SHA-1, echoed back in the status packet.
        hash: String,
    },
    /// java: `handleCustomPayload` (`:1822`)
    PluginMessage {
        /// The channel name (`MC|Brand`, `MC|TrList`, …).
        channel: String,
        /// The raw payload.
        data: Vec<u8>,
    },
    /// java: `handleStatistics` (`:1458`)
    Statistics(Vec<s2c::StatisticEntry>),
    /// java: `handleMaps` (`:1433`)
    Maps(Box<s2c::Maps>),
    /// java: `handleSignEditorOpen` (`:1216`)
    OpenSignEditor(BlockPos),

    // ---- local player ----
    /// java: `handleUpdateHealth` (`:1042`)
    HealthChanged {
        /// `0.0` means dead — the caller should send `C16PacketClientStatus::PerformRespawn`.
        health: f32,
        /// `0..=20`.
        food_level: i32,
        /// `0.0..=5.0`.
        saturation: f32,
    },
    /// java: `handleSetExperience` (`:1050`)
    ExperienceChanged {
        /// Progress across the current level, `0.0..=1.0`.
        bar: f32,
        /// The current level.
        level: i32,
        /// Lifetime experience.
        total: i32,
    },
    /// java: `handleChangeGameState` reason 3 (`:1381`)
    GameModeChanged(s2c::GameMode),
    /// java: `handlePlayerAbilities` (`:1668`)
    AbilitiesChanged(Abilities),
    /// java: `handleHeldItemChange` (`:598`)
    HeldSlotChanged(u8),
    /// java: `handleSpawnPosition` (`:959`)
    SpawnPositionChanged(BlockPos),
    /// java: `handleServerDifficulty` (`:1540`)
    DifficultyChanged(s2c::ServerDifficultyLevel),
    /// java: `handleCamera` (`:1547`)
    CameraChanged {
        /// The entity to view the world through.
        entity_id: i32,
    },
    /// java: `handleCombatEvent` (`:1515`)
    CombatEvent(Box<s2c::CombatEvent>),
    /// Rain or thunder strength changed; read them off
    /// [`Environment`](crate::sync::env::Environment). java: reasons 1, 2, 7, 8
    WeatherChanged,
    /// java: `handleTimeUpdate` (`:952`)
    TimeUpdated,
    /// java: `handleWorldBorder` (`:1558`)
    WorldBorderChanged,
    /// A `S2BPacketChangeGameState` reason with no world/ECS effect: the credits screen, a demo
    /// prompt, the arrow-hit sound, the elder-guardian effect, or an id vanilla itself ignores.
    ///
    /// Reason `0` is intentionally absent: vanilla's one `MESSAGE_NAMES` entry is handled in
    /// dispatch as an ordinary [`ClientEvent::Chat`] translation (`tile.bed.notValid`), before
    /// this handler's reason switch. java: `handleChangeGameState` (`:1366-1369`).
    GameState {
        /// What the server asked for.
        reason: s2c::GameStateReason,
        /// The reason-specific float.
        value: f32,
    },

    // ---- entities ----
    /// An entity was added to the ECS.
    EntitySpawned {
        /// The server's id for it.
        network_id: i32,
        /// The ECS handle (an arena index — the arena-index rule).
        entity: Entity,
        /// What it is. `TrackedKind::Player` for another player, which has no `EntityList`
        /// type — see [`RemotePlayer`](crate::sync::entities::RemotePlayer).
        kind: crate::sync::entities::TrackedKind,
    },
    /// A spawn packet arrived for a type this client cannot represent as an ECS entity, so
    /// nothing was spawned. See [`crate::sync::dispatch`] for the two cases where this happens.
    EntitySpawnIgnored {
        /// The server's id for it.
        network_id: i32,
        /// Why it was skipped.
        reason: &'static str,
    },
    /// java: `handleDestroyEntities` (`:654`)
    EntityRemoved {
        /// The server's id for it.
        network_id: i32,
    },
    /// java: `handleEntityStatus` (`:1024`) — damage flashes, death, particle effects.
    EntityStatus {
        /// The entity.
        network_id: i32,
        /// The status opcode, uninterpreted.
        opcode: i8,
    },
    /// java: `handleAnimation` (`:867`)
    EntityAnimation {
        /// The entity.
        network_id: i32,
        /// `0` swing, `1` hurt, `2` leave bed, `3` eat, `4` crit, `5` magic crit.
        animation: u8,
    },
    /// java: `handleEntityEquipment` (`:1297`)
    EntityEquipment {
        /// The entity.
        network_id: i32,
        /// `0` held, `1..=4` boots → helmet.
        slot: i16,
        /// The item now in that slot.
        item: Option<ItemStack>,
    },
    /// java: `handleEntityEffect` (`:1502`)
    EntityEffectAdded(s2c::EntityEffect),
    /// java: `handleRemoveEntityEffect` (`:1606`)
    EntityEffectRemoved {
        /// The entity.
        network_id: i32,
        /// The potion id.
        effect_id: u8,
    },
    /// java: `handleEntityAttach` (`:966`) — mounting a vehicle, or a lead.
    EntityAttached(s2c::EntityAttach),
    /// java: `handleEntityProperties` (`:2051`) — attribute modifiers, notably movement speed.
    EntityProperties(Box<s2c::EntityProperties>),
    /// java: `handleEntityNBT` (`:1805`)
    EntityNbt {
        /// The entity.
        network_id: i32,
        /// The tag, or `None` to clear.
        tag: Option<Box<NbtCompound>>,
    },
    /// java: `handleUseBed` (`:903`)
    EnteredBed {
        /// The sleeping player.
        network_id: i32,
        /// The bed's head block.
        bed_position: BlockPos,
    },
    /// java: `handleSpawnGlobalEntity` (`:463`), which in 1.8.9 only ever means a lightning
    /// bolt. It is not an `EntityList` type and vanilla adds it through `addWeatherEffect`
    /// rather than `addEntityToWorld`, so nothing is spawned into the ECS — S20 owns the
    /// rendering.
    LightningBolt {
        /// The server's id for it.
        network_id: i32,
        /// Where it struck.
        position: mc_math::DVec3,
    },
    /// java: `handleCollectItem` (`:819`) - the pickup animation and sound, **and the removal**.
    ///
    /// An earlier version of this comment said the collected entity is **not** removed here and
    /// that a `DestroyEntities` follows. That is false. `handleCollectItem` calls
    /// `removeEntityFromWorld(packetIn.getCollectedItemEntityID())` on the line immediately after
    /// `addEffect` (`NetHandlerPlayClient.java:841`); no `S13PacketDestroyEntities` is required or
    /// sent. A consumer that waits for one leaves the item entity in the world forever.
    /// Corrected 2026-08-23 - F-P2-client.particle-03-3, found by the wave-1 ENTITY lane.
    ItemCollected {
        /// The item or orb being picked up.
        collected: i32,
        /// Who picked it up.
        collector: i32,
    },

    // ---- world ----
    /// A chunk column was replaced or patched. java: `handleChunkData` (`:747`)
    ChunkLoaded(mc_math::ChunkPos),
    /// A chunk column was dropped. java: `doPreChunk(x, z, false)`
    ChunkUnloaded(mc_math::ChunkPos),
    /// java: `handleBlockAction` (`:1322`) — note piston, chest lid, mob-spawner reset.
    BlockAction(s2c::BlockAction),
    /// java: `handleBlockBreakAnim` (`:1331`)
    BlockBreakProgress {
        /// Who is breaking it.
        breaker_id: i32,
        /// The block.
        position: BlockPos,
        /// `0..=9`, or anything outside that to clear.
        progress: u8,
    },
    /// java: `handleExplosion` (`:1078`). The affected blocks have already been set to air.
    Explosion(Box<s2c::Explosion>),
    /// java: `handleEffect` (`:1441`) — door sounds, records, block-break particles.
    Effect(s2c::Effect),
    /// java: `handleSoundEffect` (`:1695`)
    SoundEffect(Box<s2c::SoundEffect>),
    /// java: `handleParticles` (`:2003`)
    Particles(Box<s2c::Particles>),
    /// java: `handleUpdateTileEntity` (`:1267`)
    TileEntityUpdated {
        /// Where.
        position: BlockPos,
        /// Which tile-entity kind the action byte selects.
        action: u8,
        /// The tag.
        nbt: Option<Box<NbtCompound>>,
    },
    /// java: `handleUpdateSign` (`:1234`)
    SignUpdated {
        /// Where.
        position: BlockPos,
        /// Four lines of chat-component JSON.
        lines: Box<[String; 4]>,
    },

    // ---- passed through to other sections ----
    /// Container traffic, for S17.
    Window(Box<WindowEvent>),
    /// Scoreboard traffic, for S16.
    Scoreboard(Box<ScoreboardEvent>),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// java: `handleChat` branches only on `== 2`; everything else goes to the chat box.
    #[test]
    fn chat_positions_decode_and_route_like_vanilla() {
        assert_eq!(ChatPosition::from_raw(0), ChatPosition::Chat);
        assert_eq!(ChatPosition::from_raw(1), ChatPosition::System);
        assert_eq!(ChatPosition::from_raw(2), ChatPosition::ActionBar);
        assert_eq!(ChatPosition::from_raw(9), ChatPosition::Other(9));

        assert!(ChatPosition::Chat.is_chat_box());
        assert!(ChatPosition::System.is_chat_box());
        assert!(!ChatPosition::ActionBar.is_chat_box());
        assert!(ChatPosition::Other(9).is_chat_box());
    }
}
