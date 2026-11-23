//! Clientbound Play, ids `0x39`-`0x49`: `PlayerAbilities` .. `UpdateEntityNBT`.
//!
//! java: `network/play/server/S39PacketPlayerAbilities.java` through `S49PacketUpdateEntityNBT.java`.
//!
//! Four packets in this slice (`ScoreboardObjective`, `UpdateScore`, `Teams`, `WorldBorder`,
//! `CombatEvent`, `Title`) share one wire id across several structurally different payloads,
//! selected by a leading mode/action/type field. Where Java reads that selector with
//! `PacketBuffer.readEnumValue` (a VarInt ordinal index into the enum's `values()`) the two are
//! equivalent bit-for-bit; where Java instead reads a raw `byte` fed into an `if`/`else` chain
//! (`ScoreboardObjective`'s mode, `Teams`'s action) an out-of-range byte reads no extra fields
//! and decodes as that enum's `Unknown` variant, exactly as vanilla's chain falls through.

use mc_nbt::NbtCompound;

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError};

// =============================================================================================
// 0x39 PlayerAbilities
// =============================================================================================

/// `0x39 Player Abilities`: the server's authoritative view of the player's flight/creative
/// state, sent on join and whenever it changes server-side.
/// java: `S39PacketPlayerAbilities`
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlayerAbilities {
    /// Bit flags: bit 0 `invulnerable`, bit 1 `flying`, bit 2 `allowFlying`, bit 3
    /// `creativeMode`. Bits above 3 are unused by vanilla but round-trip unchanged.
    pub flags: u8,
    /// Flying speed (vanilla default `0.05`).
    pub fly_speed: f32,
    /// Walking speed (vanilla default `0.1`).
    pub walk_speed: f32,
}

impl Packet for PlayerAbilities {
    const ID: i32 = 0x39;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let flags = buf.read_u8()?;
        let fly_speed = buf.read_f32()?;
        let walk_speed = buf.read_f32()?;
        Ok(PlayerAbilities {
            flags,
            fly_speed,
            walk_speed,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.flags);
        buf.write_f32(self.fly_speed);
        buf.write_f32(self.walk_speed);
    }
}

// =============================================================================================
// 0x3A TabComplete
// =============================================================================================

/// `0x3A Tab-Complete` (clientbound): the list of completions for a client's tab-complete
/// request. java: `S3APacketTabComplete`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TabComplete {
    /// The suggested completions, in server order.
    pub matches: Vec<String>,
}

impl Packet for TabComplete {
    const ID: i32 = 0x3A;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let count = buf.read_varint()?;
        if count < 0 {
            return Err(ProtoError::NegativeLength(count));
        }
        let mut matches = Vec::with_capacity((count as usize).min(4096));
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

// =============================================================================================
// 0x3B ScoreboardObjective
// =============================================================================================

/// How a scoreboard objective's score values are rendered client-side.
/// java: `IScoreObjectiveCriteria.EnumRenderType`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScoreboardRenderType {
    /// Plain integer score values.
    Integer,
    /// Score values rendered as hearts (used by `health`-criteria objectives).
    Hearts,
}

impl ScoreboardRenderType {
    /// Decode from the wire string. Unlike this crate's usual closed-enum convention, this
    /// mirrors `EnumRenderType.func_178795_a` exactly: an unrecognized string is **not** an
    /// error, it silently falls back to `Integer`, so this is infallible rather than returning
    /// a [`ProtoError`].
    fn from_wire(s: &str) -> ScoreboardRenderType {
        match s {
            "hearts" => ScoreboardRenderType::Hearts,
            _ => ScoreboardRenderType::Integer,
        }
    }

    fn as_wire(self) -> &'static str {
        match self {
            ScoreboardRenderType::Integer => "integer",
            ScoreboardRenderType::Hearts => "hearts",
        }
    }
}

/// The create/remove/update payload of a [`ScoreboardObjective`] packet, selected by the wire
/// mode byte (`0`/`1`/`2`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScoreboardObjectiveMode {
    /// Mode `0`: register a new objective.
    Create {
        /// The objective's display text.
        display_name: String,
        /// How score values under this objective are rendered.
        render_type: ScoreboardRenderType,
    },
    /// Mode `1`: unregister the objective named by [`ScoreboardObjective::objective_name`].
    Remove,
    /// Mode `2`: update an existing objective's display text/render type.
    Update {
        /// The objective's new display text.
        display_name: String,
        /// The objective's new render type.
        render_type: ScoreboardRenderType,
    },
    /// Any other signed wire byte. Vanilla reads no additional fields and its handler no-ops.
    Unknown(i8),
}

/// `0x3B Scoreboard Objective`: create, remove, or update one scoreboard objective.
/// java: `S3BPacketScoreboardObjective`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreboardObjective {
    /// The objective's unique internal name.
    pub objective_name: String,
    /// Which action to take, and that action's payload.
    pub mode: ScoreboardObjectiveMode,
}

impl Packet for ScoreboardObjective {
    const ID: i32 = 0x3B;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let objective_name = buf.read_string(16)?;
        let mode_id = buf.read_i8()?;
        let mode = match mode_id {
            0 => {
                let display_name = buf.read_string(32)?;
                let render_type = ScoreboardRenderType::from_wire(&buf.read_string(16)?);
                ScoreboardObjectiveMode::Create {
                    display_name,
                    render_type,
                }
            }
            1 => ScoreboardObjectiveMode::Remove,
            2 => {
                let display_name = buf.read_string(32)?;
                let render_type = ScoreboardRenderType::from_wire(&buf.read_string(16)?);
                ScoreboardObjectiveMode::Update {
                    display_name,
                    render_type,
                }
            }
            other => ScoreboardObjectiveMode::Unknown(other),
        };
        Ok(ScoreboardObjective {
            objective_name,
            mode,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.objective_name);
        match &self.mode {
            ScoreboardObjectiveMode::Create {
                display_name,
                render_type,
            } => {
                buf.write_i8(0);
                buf.write_string(display_name);
                buf.write_string(render_type.as_wire());
            }
            ScoreboardObjectiveMode::Remove => buf.write_i8(1),
            ScoreboardObjectiveMode::Update {
                display_name,
                render_type,
            } => {
                buf.write_i8(2);
                buf.write_string(display_name);
                buf.write_string(render_type.as_wire());
            }
            ScoreboardObjectiveMode::Unknown(mode) => buf.write_i8(*mode),
        }
    }
}

// =============================================================================================
// 0x3C UpdateScore
// =============================================================================================

/// The change/remove payload of an [`UpdateScore`] packet, selected by
/// `PacketBuffer.readEnumValue` (a VarInt ordinal: `0` = change, `1` = remove).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateScoreAction {
    /// Set the named player's score under the objective to `value`.
    Change {
        /// The new score value.
        value: i32,
    },
    /// Remove the named player's score entry (from the objective named in
    /// [`UpdateScore::objective_name`], or from every objective if that name is empty).
    Remove,
}

/// `0x3C Update Score`: set or remove one player's score entry under a scoreboard objective.
/// java: `S3CPacketUpdateScore`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpdateScore {
    /// The score entry's name (usually a player name, but may be an arbitrary string).
    pub player_name: String,
    /// The objective this score belongs to.
    pub objective_name: String,
    /// Whether this sets a value or removes the entry.
    pub action: UpdateScoreAction,
}

impl Packet for UpdateScore {
    const ID: i32 = 0x3C;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let player_name = buf.read_string(40)?;
        let action_id = buf.read_varint()?;
        let objective_name = buf.read_string(16)?;
        let action = match action_id {
            0 => UpdateScoreAction::Change {
                value: buf.read_varint()?,
            },
            1 => UpdateScoreAction::Remove,
            other => {
                return Err(ProtoError::InvalidEnum {
                    what: "update-score action",
                    value: other,
                })
            }
        };
        Ok(UpdateScore {
            player_name,
            objective_name,
            action,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.player_name);
        buf.write_varint(match self.action {
            UpdateScoreAction::Change { .. } => 0,
            UpdateScoreAction::Remove => 1,
        });
        buf.write_string(&self.objective_name);
        if let UpdateScoreAction::Change { value } = self.action {
            buf.write_varint(value);
        }
    }
}

// =============================================================================================
// 0x3D DisplayScoreboard
// =============================================================================================

/// `0x3D Display Scoreboard`: bind an objective to a scoreboard display slot.
/// java: `S3DPacketDisplayScoreboard`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayScoreboard {
    /// The display slot: `0` = list (tab list sidebar), `1` = sidebar, `2` = below name,
    /// `3..=18` = sidebar filtered to one of the 16 chat-format team colors
    /// (`3 + EnumChatFormatting.getColorIndex()`).
    pub position: i8,
    /// The objective's internal name to bind, or `""` to clear the slot.
    pub objective_name: String,
}

impl Packet for DisplayScoreboard {
    const ID: i32 = 0x3D;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let position = buf.read_i8()?;
        let objective_name = buf.read_string(16)?;
        Ok(DisplayScoreboard {
            position,
            objective_name,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_i8(self.position);
        buf.write_string(&self.objective_name);
    }
}

// =============================================================================================
// 0x3E Teams
// =============================================================================================

/// Fields shared by the `Create` and `UpdateInfo` [`TeamsAction`] variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeamInfo {
    /// The team's display text.
    pub display_name: String,
    /// Chat prefix prepended to team members' names.
    pub prefix: String,
    /// Chat suffix appended to team members' names.
    pub suffix: String,
    /// Bit flags: bit 0 = friendly fire allowed, bit 1 = can see friendly invisible teammates.
    /// java: `ScorePlayerTeam.func_98299_i`
    pub friendly_flags: i8,
    /// The `Team.EnumVisible` internal name governing name-tag visibility (e.g. `"always"`,
    /// `"never"`, `"hideForOtherTeams"`, `"hideForOwnTeam"`) — left as a raw string since
    /// resolving it to an enum is not this crate's concern.
    pub name_tag_visibility: String,
    /// `EnumChatFormatting.getColorIndex()` for the team's chat color: `0..=15` for a color,
    /// `-1` for reset/no color.
    pub color: i8,
}

/// The create/remove/update/add-players/remove-players payload of a [`Teams`] packet, selected
/// by a raw wire byte (`0..=4`; Java reads it with a plain `readByte()` fed into an `if`/`else`
/// chain rather than `readEnumValue`, but the set of legal values is exactly as closed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TeamsAction {
    /// Mode `0`: register a new team.
    Create {
        /// The new team's display/format info.
        info: TeamInfo,
        /// Initial member names.
        players: Vec<String>,
    },
    /// Mode `1`: unregister the team.
    Remove,
    /// Mode `2`: update an existing team's display/format info (membership unchanged).
    UpdateInfo {
        /// The team's new display/format info.
        info: TeamInfo,
    },
    /// Mode `3`: add players to the team's roster.
    AddPlayers {
        /// Names to add.
        players: Vec<String>,
    },
    /// Mode `4`: remove players from the team's roster.
    RemovePlayers {
        /// Names to remove.
        players: Vec<String>,
    },
    /// Any other signed wire byte. Vanilla reads no additional fields and its handler no-ops.
    Unknown(i8),
}

/// `0x3E Teams`: create, remove, or reconfigure a scoreboard team, or change its membership.
/// java: `S3EPacketTeams`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Teams {
    /// The team's unique internal (registered) name.
    pub team_name: String,
    /// Which action to take, and that action's payload.
    pub action: TeamsAction,
}

fn read_teams_players(buf: &mut PacketBuffer) -> Result<Vec<String>, ProtoError> {
    // java: `for (int j = 0; j < i; ++j)` adds to a list: a negative count is an empty roster,
    // not a decoder exception.
    let count = buf.read_varint()?;
    let mut players = Vec::with_capacity(usize::try_from(count).unwrap_or(0).min(4096));
    for _ in 0..count {
        players.push(buf.read_string(40)?);
    }
    Ok(players)
}

fn write_teams_players(buf: &mut PacketBuffer, players: &[String]) {
    buf.write_varint(players.len() as i32);
    for p in players {
        buf.write_string(p);
    }
}

impl Packet for Teams {
    const ID: i32 = 0x3E;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let team_name = buf.read_string(16)?;
        let action_id = buf.read_i8()?;

        // The Java source reads this as two independent `if`s in sequence (not a `switch`):
        // the info block first (modes 0, 2), then the player-list block (modes 0, 3, 4).
        let info = if action_id == 0 || action_id == 2 {
            let display_name = buf.read_string(32)?;
            let prefix = buf.read_string(16)?;
            let suffix = buf.read_string(16)?;
            let friendly_flags = buf.read_i8()?;
            let name_tag_visibility = buf.read_string(32)?;
            let color = buf.read_i8()?;
            Some(TeamInfo {
                display_name,
                prefix,
                suffix,
                friendly_flags,
                name_tag_visibility,
                color,
            })
        } else {
            None
        };

        let players = if action_id == 0 || action_id == 3 || action_id == 4 {
            Some(read_teams_players(buf)?)
        } else {
            None
        };

        // Matched on `(action_id, info, players)` together rather than pulling `info`/`players`
        // back out with `.expect()`: every legal `action_id` lines up with the `Some`/`None`
        // shape the two blocks above actually produced, and the fallback arm turns any
        // impossible combination into an ordinary [`ProtoError`] instead of a reachable panic.
        let action = match (action_id, info, players) {
            (0, Some(info), Some(players)) => TeamsAction::Create { info, players },
            (1, _, _) => TeamsAction::Remove,
            (2, Some(info), _) => TeamsAction::UpdateInfo { info },
            (3, _, Some(players)) => TeamsAction::AddPlayers { players },
            (4, _, Some(players)) => TeamsAction::RemovePlayers { players },
            (other, _, _) => TeamsAction::Unknown(other),
        };

        Ok(Teams { team_name, action })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.team_name);
        let action_id: i8 = match &self.action {
            TeamsAction::Create { .. } => 0,
            TeamsAction::Remove => 1,
            TeamsAction::UpdateInfo { .. } => 2,
            TeamsAction::AddPlayers { .. } => 3,
            TeamsAction::RemovePlayers { .. } => 4,
            TeamsAction::Unknown(action) => *action,
        };
        buf.write_i8(action_id);

        let info = match &self.action {
            TeamsAction::Create { info, .. } | TeamsAction::UpdateInfo { info } => Some(info),
            _ => None,
        };
        if let Some(info) = info {
            buf.write_string(&info.display_name);
            buf.write_string(&info.prefix);
            buf.write_string(&info.suffix);
            buf.write_i8(info.friendly_flags);
            buf.write_string(&info.name_tag_visibility);
            buf.write_i8(info.color);
        }

        let players = match &self.action {
            TeamsAction::Create { players, .. }
            | TeamsAction::AddPlayers { players }
            | TeamsAction::RemovePlayers { players } => Some(players.as_slice()),
            _ => None,
        };
        if let Some(players) = players {
            write_teams_players(buf, players);
        }
    }
}

// =============================================================================================
// 0x3F PluginMessage
// =============================================================================================

/// `0x3F Plugin Message` (clientbound): a mod/plugin-channel payload. The payload has no length
/// prefix of its own — the outer packet frame already delimits it — so decoding just consumes
/// whatever bytes remain. java: `S3FPacketCustomPayload`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PluginMessage {
    /// The plugin channel name (e.g. `"MC|Brand"`).
    pub channel: String,
    /// The raw, channel-specific payload.
    pub data: Vec<u8>,
}

impl Packet for PluginMessage {
    const ID: i32 = 0x3F;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        const MAX_PAYLOAD: usize = 1_048_576;
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
        const MAX_PAYLOAD: usize = 1_048_576;
        buf.write_string(&self.channel);
        buf.write_bounded_payload(&self.data, MAX_PAYLOAD);
    }
}

// =============================================================================================
// 0x40 Disconnect
// =============================================================================================

/// `0x40 Disconnect` (play state): the server is closing the connection with `reason` shown to
/// the player. java: `S40PacketDisconnect`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    /// The disconnect reason, as chat-component JSON.
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

// =============================================================================================
// 0x41 ServerDifficulty
// =============================================================================================

/// The world's difficulty setting, as carried by this packet specifically. java: `EnumDifficulty`
///
/// Named `ServerDifficultyLevel` rather than the more obvious `Difficulty` because `JoinGame`
/// and `Respawn` (batch1, authored in parallel with this file) carry the same concept and
/// `play::s2c`'s `pub use batch*::*` flattens every batch's public items into one namespace —
/// two same-named public types there would be an ambiguous glob re-export. Whoever integrates
/// the batches may still want to consolidate this with batch1's equivalent into one shared type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerDifficultyLevel {
    /// No hostile mobs, no hunger.
    Peaceful,
    /// Easy.
    Easy,
    /// Normal.
    Normal,
    /// Hard.
    Hard,
}

impl ServerDifficultyLevel {
    /// Mirrors `EnumDifficulty.getDifficultyEnum`: the unsigned wire byte indexes the four
    /// difficulties modulo the array length, so every possible byte is accepted.
    fn from_id(id: u8) -> ServerDifficultyLevel {
        match id % 4 {
            0 => ServerDifficultyLevel::Peaceful,
            1 => ServerDifficultyLevel::Easy,
            2 => ServerDifficultyLevel::Normal,
            3 => ServerDifficultyLevel::Hard,
            _ => unreachable!("u8 modulo four is always in 0..=3"),
        }
    }

    fn to_id(self) -> u8 {
        match self {
            ServerDifficultyLevel::Peaceful => 0,
            ServerDifficultyLevel::Easy => 1,
            ServerDifficultyLevel::Normal => 2,
            ServerDifficultyLevel::Hard => 3,
        }
    }
}

/// `0x41 Server Difficulty`. java: `S41PacketServerDifficulty`
///
/// Deviation from the Java class: it also declares a `difficultyLocked` field, but neither
/// `readPacketData` nor `writePacketData` touches it (the difficulty-lock UI is a later-version
/// feature; this decompile's field is dead weight for protocol 47), so it has no wire
/// representation and is omitted here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerDifficulty {
    /// The new difficulty.
    pub difficulty: ServerDifficultyLevel,
}

impl Packet for ServerDifficulty {
    const ID: i32 = 0x41;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(ServerDifficulty {
            difficulty: ServerDifficultyLevel::from_id(buf.read_u8()?),
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_u8(self.difficulty.to_id());
    }
}

// =============================================================================================
// 0x42 CombatEvent
// =============================================================================================

/// `0x42 Combat Event`: enter/end/death notifications from a `CombatTracker`. The whole packet
/// is one `PacketBuffer.readEnumValue`-selected event, with no field shared across variants, so
/// this enum *is* the packet (no wrapper struct). java: `S42PacketCombatEvent`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CombatEvent {
    /// Ordinal `0`: a combat encounter started. No fields.
    EnterCombat,
    /// Ordinal `1`: a combat encounter ended.
    EndCombat {
        /// Duration of the encounter, in ticks.
        duration_ticks: i32,
        /// The entity id involved in ending combat, or `-1` if none.
        entity_id: i32,
    },
    /// Ordinal `2`: the tracked entity died in combat.
    EntityDead {
        /// The id of the entity that died.
        player_id: i32,
        /// The id of the entity responsible, or `-1` if none.
        entity_id: i32,
        /// The death message, as plain (already-formatted) text — **not** chat-component JSON;
        /// vanilla reads/writes it with the plain string codec, not `readChatComponent`.
        death_message: String,
    },
}

impl Packet for CombatEvent {
    const ID: i32 = 0x42;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let event_id = buf.read_varint()?;
        match event_id {
            0 => Ok(CombatEvent::EnterCombat),
            1 => {
                let duration_ticks = buf.read_varint()?;
                let entity_id = buf.read_i32()?;
                Ok(CombatEvent::EndCombat {
                    duration_ticks,
                    entity_id,
                })
            }
            2 => {
                let player_id = buf.read_varint()?;
                let entity_id = buf.read_i32()?;
                let death_message = buf.read_string(32767)?;
                Ok(CombatEvent::EntityDead {
                    player_id,
                    entity_id,
                    death_message,
                })
            }
            other => Err(ProtoError::InvalidEnum {
                what: "combat event type",
                value: other,
            }),
        }
    }

    fn write(&self, buf: &mut PacketBuffer) {
        match self {
            CombatEvent::EnterCombat => buf.write_varint(0),
            CombatEvent::EndCombat {
                duration_ticks,
                entity_id,
            } => {
                buf.write_varint(1);
                buf.write_varint(*duration_ticks);
                buf.write_i32(*entity_id);
            }
            CombatEvent::EntityDead {
                player_id,
                entity_id,
                death_message,
            } => {
                buf.write_varint(2);
                buf.write_varint(*player_id);
                buf.write_i32(*entity_id);
                buf.write_string(death_message);
            }
        }
    }
}

// =============================================================================================
// 0x43 Camera
// =============================================================================================

/// `0x43 Camera`: switch the client's point of view to spectate through another entity.
/// java: `S43PacketCamera`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Camera {
    /// The entity id to view through.
    pub entity_id: i32,
}

impl Packet for Camera {
    const ID: i32 = 0x43;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Camera {
            entity_id: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
    }
}

// =============================================================================================
// 0x44 WorldBorder
// =============================================================================================

/// `0x44 World Border`: one of six very different updates to the world border, selected by a
/// `PacketBuffer.readEnumValue` ordinal. No field is shared across variants, so this enum *is*
/// the packet (no wrapper struct). java: `S44PacketWorldBorder`
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorldBorder {
    /// Ordinal `0`: jump straight to a new diameter (no lerp).
    SetSize {
        /// The new diameter, in blocks.
        diameter: f64,
    },
    /// Ordinal `1`: interpolate the diameter over time.
    LerpSize {
        /// The diameter to start from.
        old_diameter: f64,
        /// The diameter to end at.
        new_diameter: f64,
        /// How long the transition takes, in milliseconds.
        speed_millis: i64,
    },
    /// Ordinal `2`: move the border's center, diameter unchanged.
    SetCenter {
        /// New center X.
        x: f64,
        /// New center Z.
        z: f64,
    },
    /// Ordinal `3`: the full state, sent once on join.
    Initialize {
        /// New center X.
        x: f64,
        /// New center Z.
        z: f64,
        /// The diameter to start from (equal to `new_diameter` if no lerp is in progress).
        old_diameter: f64,
        /// The diameter to end at.
        new_diameter: f64,
        /// How long the (possible) transition takes, in milliseconds.
        speed_millis: i64,
        /// The maximum world radius in blocks (`WorldBorder.worldSize`; vanilla default
        /// `29999984`), independent of the current diameter.
        world_size: i32,
        /// Warning distance from the border, in blocks, at which the screen tint appears.
        warning_blocks: i32,
        /// Warning lead time, in seconds, before a shrinking border reaches the player.
        warning_time: i32,
    },
    /// Ordinal `4`: change the warning lead time only.
    SetWarningTime {
        /// New warning lead time, in seconds.
        warning_time: i32,
    },
    /// Ordinal `5`: change the warning distance only.
    SetWarningBlocks {
        /// New warning distance, in blocks.
        warning_blocks: i32,
    },
}

impl Packet for WorldBorder {
    const ID: i32 = 0x44;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let action_id = buf.read_varint()?;
        match action_id {
            0 => Ok(WorldBorder::SetSize {
                diameter: buf.read_f64()?,
            }),
            1 => {
                let old_diameter = buf.read_f64()?;
                let new_diameter = buf.read_f64()?;
                let speed_millis = buf.read_varlong()?;
                Ok(WorldBorder::LerpSize {
                    old_diameter,
                    new_diameter,
                    speed_millis,
                })
            }
            2 => {
                let x = buf.read_f64()?;
                let z = buf.read_f64()?;
                Ok(WorldBorder::SetCenter { x, z })
            }
            3 => {
                let x = buf.read_f64()?;
                let z = buf.read_f64()?;
                let old_diameter = buf.read_f64()?;
                let new_diameter = buf.read_f64()?;
                let speed_millis = buf.read_varlong()?;
                let world_size = buf.read_varint()?;
                let warning_blocks = buf.read_varint()?;
                let warning_time = buf.read_varint()?;
                Ok(WorldBorder::Initialize {
                    x,
                    z,
                    old_diameter,
                    new_diameter,
                    speed_millis,
                    world_size,
                    warning_blocks,
                    warning_time,
                })
            }
            4 => Ok(WorldBorder::SetWarningTime {
                warning_time: buf.read_varint()?,
            }),
            5 => Ok(WorldBorder::SetWarningBlocks {
                warning_blocks: buf.read_varint()?,
            }),
            other => Err(ProtoError::InvalidEnum {
                what: "world border action",
                value: other,
            }),
        }
    }

    fn write(&self, buf: &mut PacketBuffer) {
        match *self {
            WorldBorder::SetSize { diameter } => {
                buf.write_varint(0);
                buf.write_f64(diameter);
            }
            WorldBorder::LerpSize {
                old_diameter,
                new_diameter,
                speed_millis,
            } => {
                buf.write_varint(1);
                buf.write_f64(old_diameter);
                buf.write_f64(new_diameter);
                buf.write_varlong(speed_millis);
            }
            WorldBorder::SetCenter { x, z } => {
                buf.write_varint(2);
                buf.write_f64(x);
                buf.write_f64(z);
            }
            WorldBorder::Initialize {
                x,
                z,
                old_diameter,
                new_diameter,
                speed_millis,
                world_size,
                warning_blocks,
                warning_time,
            } => {
                buf.write_varint(3);
                buf.write_f64(x);
                buf.write_f64(z);
                buf.write_f64(old_diameter);
                buf.write_f64(new_diameter);
                buf.write_varlong(speed_millis);
                buf.write_varint(world_size);
                buf.write_varint(warning_blocks);
                buf.write_varint(warning_time);
            }
            WorldBorder::SetWarningTime { warning_time } => {
                buf.write_varint(4);
                buf.write_varint(warning_time);
            }
            WorldBorder::SetWarningBlocks { warning_blocks } => {
                buf.write_varint(5);
                buf.write_varint(warning_blocks);
            }
        }
    }
}

// =============================================================================================
// 0x45 Title
// =============================================================================================

/// `0x45 Title`: on-screen title/subtitle control, selected by a `PacketBuffer.readEnumValue`
/// ordinal. No field is shared across variants, so this enum *is* the packet (no wrapper
/// struct). java: `S45PacketTitle`
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Title {
    /// Ordinal `0`: set (and start showing) the main title text.
    SetTitle {
        /// The title, as chat-component JSON.
        text: String,
    },
    /// Ordinal `1`: set (and start showing) the subtitle text.
    SetSubtitle {
        /// The subtitle, as chat-component JSON.
        text: String,
    },
    /// Ordinal `2`: set the fade-in/stay/fade-out timing, in ticks, for the next title shown.
    SetTimes {
        /// Ticks to fade in over.
        fade_in_ticks: i32,
        /// Ticks to stay fully visible.
        stay_ticks: i32,
        /// Ticks to fade out over.
        fade_out_ticks: i32,
    },
    /// Ordinal `3`: hide the currently displayed title/subtitle.
    Clear,
    /// Ordinal `4`: hide the title and reset its timing to the client's defaults.
    Reset,
}

impl Packet for Title {
    const ID: i32 = 0x45;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let type_id = buf.read_varint()?;
        match type_id {
            0 => Ok(Title::SetTitle {
                text: buf.read_chat()?,
            }),
            1 => Ok(Title::SetSubtitle {
                text: buf.read_chat()?,
            }),
            2 => {
                let fade_in_ticks = buf.read_i32()?;
                let stay_ticks = buf.read_i32()?;
                let fade_out_ticks = buf.read_i32()?;
                Ok(Title::SetTimes {
                    fade_in_ticks,
                    stay_ticks,
                    fade_out_ticks,
                })
            }
            3 => Ok(Title::Clear),
            4 => Ok(Title::Reset),
            other => Err(ProtoError::InvalidEnum {
                what: "title action",
                value: other,
            }),
        }
    }

    fn write(&self, buf: &mut PacketBuffer) {
        match self {
            Title::SetTitle { text } => {
                buf.write_varint(0);
                buf.write_chat(text);
            }
            Title::SetSubtitle { text } => {
                buf.write_varint(1);
                buf.write_chat(text);
            }
            Title::SetTimes {
                fade_in_ticks,
                stay_ticks,
                fade_out_ticks,
            } => {
                buf.write_varint(2);
                buf.write_i32(*fade_in_ticks);
                buf.write_i32(*stay_ticks);
                buf.write_i32(*fade_out_ticks);
            }
            Title::Clear => buf.write_varint(3),
            Title::Reset => buf.write_varint(4),
        }
    }
}

// =============================================================================================
// 0x46 SetCompression
// =============================================================================================

/// `0x46 Set Compression` (play state): changes the compression threshold mid-session. A
/// distinct type from [`crate::login::SetCompression`] (same purpose, but the login-state
/// packet of the same name) — they live in different modules, so there is no name clash.
/// java: `S46PacketSetCompressionLevel`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCompression {
    /// The new compression threshold, in bytes.
    pub threshold: i32,
}

impl Packet for SetCompression {
    const ID: i32 = 0x46;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SetCompression {
            threshold: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.threshold);
    }
}

// =============================================================================================
// 0x47 PlayerListHeaderFooter
// =============================================================================================

/// `0x47 Player List Header And Footer`: extra chat components shown above/below the tab-list
/// player names. java: `S47PacketPlayerListHeaderFooter`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlayerListHeaderFooter {
    /// Header text, as chat-component JSON.
    pub header: String,
    /// Footer text, as chat-component JSON.
    pub footer: String,
}

impl Packet for PlayerListHeaderFooter {
    const ID: i32 = 0x47;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let header = buf.read_chat()?;
        let footer = buf.read_chat()?;
        Ok(PlayerListHeaderFooter { header, footer })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_chat(&self.header);
        buf.write_chat(&self.footer);
    }
}

// =============================================================================================
// 0x48 ResourcePackSend
// =============================================================================================

/// `0x48 Resource Pack Send`: asks the client to download and apply a resource pack.
/// java: `S48PacketResourcePackSend`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourcePackSend {
    /// The pack's download URL.
    pub url: String,
    /// The pack's SHA-1 hash, as a (up to 40-character) hex string.
    pub hash: String,
}

impl Packet for ResourcePackSend {
    const ID: i32 = 0x48;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let url = buf.read_string(32767)?;
        let hash = buf.read_string(40)?;
        Ok(ResourcePackSend { url, hash })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.url);
        buf.write_string(&self.hash);
    }
}

// =============================================================================================
// 0x49 UpdateEntityNBT
// =============================================================================================

/// `0x49 Update Entity NBT`: overwrite an entity's NBT tag (used by the `/blockdata`-style
/// debug/creative tools). java: `S49PacketUpdateEntityNBT`
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateEntityNBT {
    /// The target entity's id.
    pub entity_id: i32,
    /// The entity's new NBT tag. `readNBTTagCompoundFromBuffer`/`writeNBTTagCompoundToBuffer`
    /// permit the null-tag sentinel here just like every other NBT-bearing field in this
    /// protocol, so this is genuinely optional, not always-present.
    pub tag: Option<NbtCompound>,
}

impl Packet for UpdateEntityNBT {
    const ID: i32 = 0x49;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let entity_id = buf.read_varint()?;
        let tag = buf.read_nbt_compound_opt()?;
        Ok(UpdateEntityNBT { entity_id, tag })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.entity_id);
        buf.write_nbt_compound_opt(self.tag.as_ref());
    }
}

// =============================================================================================
// tests
// =============================================================================================

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
    fn player_abilities_round_trips() {
        round_trip(PlayerAbilities {
            flags: 0b1011,
            fly_speed: 0.05,
            walk_speed: 0.1,
        });
    }

    #[test]
    fn tab_complete_round_trips_empty_and_populated() {
        round_trip(TabComplete { matches: vec![] });
        round_trip(TabComplete {
            matches: vec!["foo".into(), "foobar".into(), "food".into()],
        });
    }

    #[test]
    fn tab_complete_rejects_negative_count() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(-1);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            TabComplete::read(&mut read_buf),
            Err(ProtoError::NegativeLength(-1))
        ));
    }

    /// java: S3E's roster count drives `for (int j = 0; j < i;
    /// ++j)`, so a negative count is an empty roster and decoding carries on (unlike S3A's
    /// `new String[i]`, above). The former `NegativeLength` guard disconnected.
    #[test]
    fn teams_roster_count_is_a_signed_loop_like_vanilla() {
        for (action, expected) in [
            (
                0,
                TeamsAction::Create {
                    info: TeamInfo {
                        display_name: "Red".into(),
                        prefix: "§c".into(),
                        suffix: String::new(),
                        friendly_flags: 3,
                        name_tag_visibility: "always".into(),
                        color: 12,
                    },
                    players: vec![],
                },
            ),
            (3, TeamsAction::AddPlayers { players: vec![] }),
            (4, TeamsAction::RemovePlayers { players: vec![] }),
        ] {
            let mut buf = PacketBuffer::new();
            buf.write_string("red");
            buf.write_i8(action);
            if action == 0 {
                buf.write_string("Red");
                buf.write_string("§c");
                buf.write_string("");
                buf.write_i8(3);
                buf.write_string("always");
                buf.write_i8(12);
            }
            buf.write_varint(-3);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let teams = Teams::read(&mut read_buf)
                .unwrap_or_else(|error| panic!("action {action}: {error}"));
            assert!(read_buf.is_empty(), "action {action}");
            assert_eq!(teams.team_name, "red");
            assert_eq!(teams.action, expected, "action {action}");
        }
    }

    #[test]
    fn scoreboard_objective_round_trips_all_modes() {
        round_trip(ScoreboardObjective {
            objective_name: "obj".into(),
            mode: ScoreboardObjectiveMode::Create {
                display_name: "Objective".into(),
                render_type: ScoreboardRenderType::Hearts,
            },
        });
        round_trip(ScoreboardObjective {
            objective_name: "obj".into(),
            mode: ScoreboardObjectiveMode::Remove,
        });
        round_trip(ScoreboardObjective {
            objective_name: "obj".into(),
            mode: ScoreboardObjectiveMode::Update {
                display_name: "Objective 2".into(),
                render_type: ScoreboardRenderType::Integer,
            },
        });
    }

    #[test]
    fn scoreboard_objective_preserves_every_unknown_signed_mode() {
        for mode in i8::MIN..=i8::MAX {
            if (0..=2).contains(&mode) {
                continue;
            }
            let mut buf = PacketBuffer::new();
            buf.write_string("obj");
            buf.write_i8(mode);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = ScoreboardObjective::read(&mut read_buf)
                .expect("unknown objective modes are not decoder errors in vanilla");
            assert_eq!(packet.mode, ScoreboardObjectiveMode::Unknown(mode));
            assert!(read_buf.is_empty());
            round_trip(packet);
        }
    }

    #[test]
    fn scoreboard_render_type_falls_back_on_unknown_string_like_vanilla() {
        let mut buf = PacketBuffer::new();
        buf.write_string("obj");
        buf.write_u8(0);
        buf.write_string("Objective");
        buf.write_string("not-a-real-type");
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        let pkt = ScoreboardObjective::read(&mut read_buf).unwrap();
        assert_eq!(
            pkt.mode,
            ScoreboardObjectiveMode::Create {
                display_name: "Objective".into(),
                render_type: ScoreboardRenderType::Integer,
            }
        );
    }

    #[test]
    fn update_score_round_trips_both_actions() {
        round_trip(UpdateScore {
            player_name: "Notch".into(),
            objective_name: "obj".into(),
            action: UpdateScoreAction::Change { value: 42 },
        });
        round_trip(UpdateScore {
            player_name: "Notch".into(),
            objective_name: "obj".into(),
            action: UpdateScoreAction::Remove,
        });
    }

    #[test]
    fn display_scoreboard_round_trips() {
        round_trip(DisplayScoreboard {
            position: 1,
            objective_name: "obj".into(),
        });
        // Position 3..=18 are team-color-filtered sidebar slots.
        round_trip(DisplayScoreboard {
            position: 18,
            objective_name: "".into(),
        });
    }

    #[test]
    fn teams_round_trips_every_action() {
        let info = TeamInfo {
            display_name: "Red Team".into(),
            prefix: "[R] ".into(),
            suffix: "".into(),
            friendly_flags: 0b11,
            name_tag_visibility: "always".into(),
            color: -1,
        };
        round_trip(Teams {
            team_name: "red".into(),
            action: TeamsAction::Create {
                info: info.clone(),
                players: vec!["Notch".into(), "Jeb_".into()],
            },
        });
        round_trip(Teams {
            team_name: "red".into(),
            action: TeamsAction::Remove,
        });
        round_trip(Teams {
            team_name: "red".into(),
            action: TeamsAction::UpdateInfo { info },
        });
        round_trip(Teams {
            team_name: "red".into(),
            action: TeamsAction::AddPlayers {
                players: vec!["Dinnerbone".into()],
            },
        });
        round_trip(Teams {
            team_name: "red".into(),
            action: TeamsAction::RemovePlayers {
                players: vec!["Dinnerbone".into()],
            },
        });
    }

    #[test]
    fn teams_preserves_every_unknown_signed_action() {
        for action in i8::MIN..=i8::MAX {
            if (0..=4).contains(&action) {
                continue;
            }
            let mut buf = PacketBuffer::new();
            buf.write_string("red");
            buf.write_i8(action);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = Teams::read(&mut read_buf)
                .expect("unknown team actions are not decoder errors in vanilla");
            assert_eq!(packet.action, TeamsAction::Unknown(action));
            assert!(read_buf.is_empty());
            round_trip(packet);
        }
    }

    #[test]
    fn plugin_message_round_trips_and_uses_outer_frame_length() {
        round_trip(PluginMessage {
            channel: "MC|Brand".into(),
            data: vec![1, 2, 3, 4, 5],
        });
        // Confirm the payload really has no self-describing length prefix: the raw bytes
        // written are exactly `channel-prefix + data`, nothing more.
        let pkt = PluginMessage {
            channel: "c".into(),
            data: vec![0xAA; 10],
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut expected = PacketBuffer::new();
        expected.write_string("c");
        expected.write_bytes(&[0xAA; 10]);
        assert_eq!(buf.as_slice(), expected.as_slice());
    }

    #[test]
    fn plugin_message_rejects_payload_over_one_mib_on_read_and_write() {
        // Java's `S3FPacketCustomPayload` accepts its inclusive 1 MiB limit.  Pin the
        // boundary before the max+1 negative controls below so a `<`/`>=` mutation cannot
        // accidentally turn this into an overzealous disconnect.
        let boundary = vec![0x5A_u8; 1_048_576];
        let mut boundary_wire = PacketBuffer::new();
        boundary_wire.write_string("MC|Brand");
        boundary_wire.write_bytes(&boundary);
        let mut boundary_read = PacketBuffer::from_slice(boundary_wire.as_slice());
        assert_eq!(
            PluginMessage::read(&mut boundary_read).unwrap(),
            PluginMessage {
                channel: "MC|Brand".into(),
                data: boundary.clone(),
            }
        );
        assert!(boundary_read.is_empty());

        let mut boundary_write = PacketBuffer::new();
        PluginMessage {
            channel: "MC|Brand".into(),
            data: boundary,
        }
        .write(&mut boundary_write);
        assert!(boundary_write.finish_write().is_ok());

        let data = vec![0_u8; 1_048_577];
        let mut wire = PacketBuffer::new();
        wire.write_string("MC|Brand");
        wire.write_bytes(&data);
        let mut read_buf = PacketBuffer::from_slice(wire.as_slice());
        assert!(matches!(
            PluginMessage::read(&mut read_buf),
            Err(ProtoError::PayloadTooLong {
                len: 1_048_577,
                max: 1_048_576
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
                len: 1_048_577,
                max: 1_048_576
            })
        ));
    }

    #[test]
    fn disconnect_round_trips() {
        round_trip(Disconnect {
            reason: r#"{"text":"kicked"}"#.into(),
        });
    }

    #[test]
    fn server_difficulty_round_trips_all_levels() {
        for difficulty in [
            ServerDifficultyLevel::Peaceful,
            ServerDifficultyLevel::Easy,
            ServerDifficultyLevel::Normal,
            ServerDifficultyLevel::Hard,
        ] {
            round_trip(ServerDifficulty { difficulty });
        }
    }

    #[test]
    fn server_difficulty_wraps_every_unsigned_byte_modulo_four() {
        let expected = [
            ServerDifficultyLevel::Peaceful,
            ServerDifficultyLevel::Easy,
            ServerDifficultyLevel::Normal,
            ServerDifficultyLevel::Hard,
        ];

        for id in 0..=u8::MAX {
            let mut buf = PacketBuffer::new();
            buf.write_u8(id);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            let packet = ServerDifficulty::read(&mut read_buf)
                .expect("every unsigned difficulty byte is valid in vanilla");
            assert_eq!(packet.difficulty, expected[usize::from(id % 4)]);
            assert!(read_buf.is_empty());
        }
    }

    #[test]
    fn combat_event_round_trips_every_variant() {
        round_trip(CombatEvent::EnterCombat);
        round_trip(CombatEvent::EndCombat {
            duration_ticks: 100,
            entity_id: -1,
        });
        round_trip(CombatEvent::EntityDead {
            player_id: 5,
            entity_id: 12,
            death_message: "Notch was slain by Herobrine".into(),
        });
    }

    #[test]
    fn camera_round_trips() {
        round_trip(Camera { entity_id: 1234 });
    }

    #[test]
    fn world_border_round_trips_every_action() {
        round_trip(WorldBorder::SetSize { diameter: 100.0 });
        round_trip(WorldBorder::LerpSize {
            old_diameter: 100.0,
            new_diameter: 50.0,
            speed_millis: 60_000,
        });
        round_trip(WorldBorder::SetCenter { x: 12.5, z: -8.25 });
        round_trip(WorldBorder::Initialize {
            x: 0.0,
            z: 0.0,
            old_diameter: 60_000_000.0,
            new_diameter: 60_000_000.0,
            speed_millis: 0,
            world_size: 29_999_984,
            warning_blocks: 5,
            warning_time: 15,
        });
        round_trip(WorldBorder::SetWarningTime { warning_time: 15 });
        round_trip(WorldBorder::SetWarningBlocks { warning_blocks: 5 });
    }

    #[test]
    fn world_border_wire_ordinals_match_s44_action_declaration() {
        // Java declares SET_SIZE, LERP_SIZE, SET_CENTER, INITIALIZE, SET_WARNING_TIME,
        // SET_WARNING_BLOCKS in that order (`S44PacketWorldBorder.java:176-183`). A round trip
        // alone cannot catch a consistently permuted encoder/decoder, so assert the raw first
        // VarInt byte independently for every action. The final pair is the negative control:
        // their ids must remain distinct because the Java packet switches on 4 vs 5.
        let packets = [
            (WorldBorder::SetSize { diameter: 1.0 }, 0),
            (
                WorldBorder::LerpSize {
                    old_diameter: 2.0,
                    new_diameter: 1.0,
                    speed_millis: 10,
                },
                1,
            ),
            (WorldBorder::SetCenter { x: 3.0, z: 4.0 }, 2),
            (
                WorldBorder::Initialize {
                    x: 5.0,
                    z: 6.0,
                    old_diameter: 7.0,
                    new_diameter: 8.0,
                    speed_millis: 9,
                    world_size: 10,
                    warning_blocks: 11,
                    warning_time: 12,
                },
                3,
            ),
            (WorldBorder::SetWarningTime { warning_time: 13 }, 4),
            (WorldBorder::SetWarningBlocks { warning_blocks: 14 }, 5),
        ];

        let mut raw_ids = Vec::new();
        for (packet, expected_id) in packets {
            let mut buf = PacketBuffer::new();
            packet.write(&mut buf);
            assert_eq!(buf.as_slice()[0], expected_id);
            raw_ids.push(buf.as_slice()[0]);
        }
        assert_ne!(raw_ids[4], raw_ids[5]);
    }

    #[test]
    fn world_border_initialize_wire_payload_keeps_java_field_order() {
        let packet = WorldBorder::Initialize {
            x: 1.25,
            z: -2.5,
            old_diameter: 200.0,
            new_diameter: 100.0,
            speed_millis: 300,
            world_size: 29_999_984,
            warning_blocks: 5,
            warning_time: 15,
        };
        let mut actual = PacketBuffer::new();
        packet.write(&mut actual);

        // Java `writePacketData` (`S44PacketWorldBorder.java:114-122`) writes center, diameter,
        // target, time, world size, warning distance, then warning time after ordinal 3.
        let mut expected = PacketBuffer::new();
        expected.write_varint(3);
        expected.write_f64(1.25);
        expected.write_f64(-2.5);
        expected.write_f64(200.0);
        expected.write_f64(100.0);
        expected.write_varlong(300);
        expected.write_varint(29_999_984);
        expected.write_varint(5);
        expected.write_varint(15);
        assert_eq!(actual.as_slice(), expected.as_slice());

        // Negative control: swapping the two warning values must change the wire bytes.
        let mut swapped = PacketBuffer::new();
        swapped.write_varint(3);
        swapped.write_f64(1.25);
        swapped.write_f64(-2.5);
        swapped.write_f64(200.0);
        swapped.write_f64(100.0);
        swapped.write_varlong(300);
        swapped.write_varint(29_999_984);
        swapped.write_varint(15);
        swapped.write_varint(5);
        assert_ne!(actual.as_slice(), swapped.as_slice());
    }

    #[test]
    fn world_border_rejects_unknown_action() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(6);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            WorldBorder::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { value: 6, .. })
        ));
    }

    #[test]
    fn title_round_trips_every_variant() {
        round_trip(Title::SetTitle {
            text: r#"{"text":"Welcome"}"#.into(),
        });
        round_trip(Title::SetSubtitle {
            text: r#"{"text":"to the server"}"#.into(),
        });
        round_trip(Title::SetTimes {
            fade_in_ticks: 10,
            stay_ticks: 70,
            fade_out_ticks: 20,
        });
        round_trip(Title::Clear);
        round_trip(Title::Reset);
    }

    #[test]
    fn title_rejects_unknown_action() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(5);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            Title::read(&mut read_buf),
            Err(ProtoError::InvalidEnum { value: 5, .. })
        ));
    }

    #[test]
    fn set_compression_round_trips() {
        round_trip(SetCompression { threshold: 256 });
        round_trip(SetCompression { threshold: -1 });
    }

    #[test]
    fn player_list_header_footer_round_trips() {
        round_trip(PlayerListHeaderFooter {
            header: r#"{"text":"header"}"#.into(),
            footer: r#"{"text":"footer"}"#.into(),
        });
    }

    #[test]
    fn resource_pack_send_round_trips() {
        round_trip(ResourcePackSend {
            url: "https://example.com/pack.zip".into(),
            hash: "a".repeat(40),
        });
    }

    #[test]
    fn update_entity_nbt_round_trips_none_and_some() {
        round_trip(UpdateEntityNBT {
            entity_id: 99,
            tag: None,
        });
        let mut tag = NbtCompound::new();
        tag.insert("Health", mc_nbt::Nbt::Float(20.0));
        round_trip(UpdateEntityNBT {
            entity_id: 99,
            tag: Some(tag),
        });
    }
}
