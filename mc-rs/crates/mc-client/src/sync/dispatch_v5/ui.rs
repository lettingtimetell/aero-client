//! The 1.7.10 packets the screens, the tab list and the scoreboard consume: windows, statistics,
//! the player list, tab completion, scoreboard and teams, and plugin channels (java17
//! `client/network/NetHandlerPlayClient.java`, `NHPC` below).

use mc_data::EntityType;
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::v5::s2c as s2c5;
use mc_protocol::v5::{NbtBudget, PacketBufferV5};
use mc_protocol::{PacketBuffer, ProtoError};

use crate::error::ClientError;
use crate::sync::entities::TrackedKind;
use crate::sync::event::{ClientEvent, ScoreboardEvent, WindowEvent};
use crate::sync::state::ClientState;

/// The display slots 1.7.10's scoreboard has: list, sidebar, below name (java17
/// `scoreboard/Scoreboard.java:16`, `new ScoreObjective[3]`; java18 has 19).
const DISPLAY_SLOTS_V5: i8 = 3;

fn window(event: WindowEvent) -> ClientEvent {
    ClientEvent::Window(Box::new(event))
}

fn scoreboard(event: ScoreboardEvent) -> ClientEvent {
    ClientEvent::Scoreboard(Box::new(event))
}

// ---- windows -----------------------------------------------------------------------------

/// java17 `NHPC.handleOpenWindow` (`:877-989`): a window by number, 0-11, each opening its 1.7.10
/// container; any other number opens nothing (and keeps no window id). Type 11 opens only for an
/// entity that exists and is a horse. The window travels on as the 1.8.9 packet the screens take
/// ([`mc_ui::screens::container::v5::open_window_v47`]); a 1.7.10 session's screen builder makes
/// the 1.7.10 container from it.
pub(super) fn open_window(
    state: &ClientState,
    packet: &s2c5::OpenWindow,
    events: &mut Vec<ClientEvent>,
) {
    let Some(lowered) = mc_ui::screens::container::v5::open_window_v47(packet) else {
        return;
    };
    if packet.inventory_type == 11 {
        // `getEntityByID` answers the local player for its own id, which is not a horse.
        let horse = state.local.entity_id != packet.entity_id
            && state
                .entities
                .get(packet.entity_id)
                .is_some_and(|tracked| tracked.kind == TrackedKind::Type(EntityType::EntityHorse));
        if !horse {
            return;
        }
    }
    events.push(window(WindowEvent::Open(lowered)));
}

/// java17 `NHPC.handleCloseWindow` (`:1161-1164`) = java18 (`:1311-1315`): `closeScreenNoPacket`,
/// whatever the window id.
pub(super) fn close_window(packet: &s2c5::CloseWindow, events: &mut Vec<ClientEvent>) {
    events.push(window(WindowEvent::Close {
        window_id: packet.window_id,
    }));
}

/// java17 `NHPC.handleSetSlot` (`:991-1025`) = java18 (`:1133-1168`).
pub(super) fn set_slot(packet: s2c5::SetSlot, events: &mut Vec<ClientEvent>) {
    events.push(window(WindowEvent::SetSlot {
        window_id: packet.window_id,
        slot: packet.slot,
        item: packet.item,
    }));
}

/// java17 `NHPC.handleWindowItems` (`:1047-1059`) = java18 (`:1198-1211`).
pub(super) fn window_items(packet: s2c5::WindowItems, events: &mut Vec<ClientEvent>) {
    events.push(window(WindowEvent::Items {
        window_id: packet.window_id,
        items: packet.items,
    }));
}

/// java17 `NHPC.handleWindowProperty` (`:1141-1149`) = java18 (`:1286-1295`); the property ids
/// mean what the 1.7.10 container says, which the 1.7.10 screen applies.
pub(super) fn window_property(packet: &s2c5::WindowProperty, events: &mut Vec<ClientEvent>) {
    events.push(window(WindowEvent::Property {
        window_id: packet.window_id,
        index: packet.property,
        value: packet.value,
    }));
}

/// java17 `NHPC.handleConfirmTransaction` (`:1027-1045`) = java18 (`:1174-1193`): the client's
/// `C0F` reply is the screens' (`Client::react`), sent while this tick's drain runs.
pub(super) fn confirm_transaction(
    packet: &s2c5::ConfirmTransaction,
    events: &mut Vec<ClientEvent>,
) {
    events.push(window(WindowEvent::ConfirmTransaction {
        window_id: packet.window_id,
        action_number: packet.action_number,
        accepted: packet.accepted,
    }));
}

// ---- statistics, player list, completion ----------------------------------------------------

/// java17 `NHPC.handleStatistics` (`:1279-1321`) = java18 (`:1458-1500`) once each 1.7.10
/// statistic id is resolved ([`mc_data::v5::stat_from_v5`]); `S37PacketStatistics.readPacketData`
/// keeps only the ids `StatList.func_151177_a` knows, which drops the rest.
pub(super) fn lower_statistics(statistics: s2c5::Statistics) -> S2CPlay {
    S2CPlay::Statistics(s2c::Statistics {
        entries: statistics
            .entries
            .into_iter()
            .filter_map(|(id, value)| {
                mc_data::v5::stat_from_v5(&id).map(|stat| s2c::StatisticEntry {
                    stat_id: stat.id.to_owned(),
                    value,
                })
            })
            .collect(),
    })
}

/// java17 `NHPC.handlePlayerListItem` (`:1345-1366`): the name-keyed 1.7.10 list.
pub(super) fn player_list_item(state: &mut ClientState, item: &s2c5::PlayerListItem) {
    state
        .players_v5
        .handle_player_list_item(&item.name, item.online, item.ping);
}

// ---- scoreboard --------------------------------------------------------------------------

/// java17 `NHPC.handleScoreboardObjective` (`:1471-1494`): modes 0 (create), 1 (remove) and 2
/// (rename) on java18's scoreboard (`:1869-1894`), with no render type: every 1.7.10 objective
/// shows integers. 1.7.10's removal clears its three display slots, which are the only ones a
/// 1.7.10 session ever fills.
pub(super) fn scoreboard_objective(
    objective: s2c5::ScoreboardObjective,
    events: &mut Vec<ClientEvent>,
) {
    let mode = match objective.mode {
        0 => s2c::ScoreboardObjectiveMode::Create {
            display_name: objective.display_name,
            render_type: s2c::ScoreboardRenderType::Integer,
        },
        1 => s2c::ScoreboardObjectiveMode::Remove,
        2 => s2c::ScoreboardObjectiveMode::Update {
            display_name: objective.display_name,
            render_type: s2c::ScoreboardRenderType::Integer,
        },
        other => s2c::ScoreboardObjectiveMode::Unknown(other),
    };
    events.push(scoreboard(ScoreboardEvent::Objective(
        s2c::ScoreboardObjective {
            objective_name: objective.objective_name,
            mode,
        },
    )));
}

/// java17 `NHPC.handleUpdateScore` (`:1496-1510`): action 0 sets the score as java18 does;
/// action 1 is `func_96515_c(name)`, which drops the name's scores in **every** objective (the
/// 1.7.10 packet names no objective for it), java18's removal with an empty objective name;
/// any other action does nothing.
pub(super) fn update_score(score: s2c5::UpdateScore, events: &mut Vec<ClientEvent>) {
    let (objective_name, action) = match score.action {
        0 => (
            score.objective_name,
            s2c::UpdateScoreAction::Change { value: score.value },
        ),
        1 => (String::new(), s2c::UpdateScoreAction::Remove),
        _ => return,
    };
    events.push(scoreboard(ScoreboardEvent::Score(s2c::UpdateScore {
        player_name: score.score_name,
        objective_name,
        action,
    })));
}

/// java17 `NHPC.handleDisplayScoreboard` (`:1512-1525`) = java18 (`:1927-1941`) for the three
/// 1.7.10 slots; any other slot indexes past `objectiveDisplaySlots` and throws in 1.7.10.
pub(super) fn display_scoreboard(
    display: s2c5::DisplayScoreboard,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    if !(0..DISPLAY_SLOTS_V5).contains(&display.position) {
        return Err(ClientError::Protocol(ProtoError::InvalidEnum {
            what: "scoreboard display slot past 1.7.10's three",
            value: i32::from(display.position),
        }));
    }
    events.push(scoreboard(ScoreboardEvent::Display(
        s2c::DisplayScoreboard {
            position: display.position,
            objective_name: display.score_name,
        },
    )));
    Ok(())
}

/// java17 `NHPC.handleTeams` (`:1527-1578`): modes 0-4 with java18's membership rules
/// (`:1947-1997`) and no colour or name-tag visibility: the team keeps java18's defaults for
/// both (no colour, always visible), so a name is coloured by its prefix alone.
pub(super) fn teams(teams: s2c5::Teams, events: &mut Vec<ClientEvent>) {
    let info = || s2c::TeamInfo {
        display_name: teams.display_name.clone(),
        prefix: teams.prefix.clone(),
        suffix: teams.suffix.clone(),
        friendly_flags: teams.friendly_flags,
        name_tag_visibility: "always".to_owned(),
        color: -1,
    };
    let action = match teams.mode {
        0 => s2c::TeamsAction::Create {
            info: info(),
            players: teams.players.clone(),
        },
        1 => s2c::TeamsAction::Remove,
        2 => s2c::TeamsAction::UpdateInfo { info: info() },
        3 => s2c::TeamsAction::AddPlayers {
            players: teams.players.clone(),
        },
        4 => s2c::TeamsAction::RemovePlayers {
            players: teams.players.clone(),
        },
        other => s2c::TeamsAction::Unknown(other),
    };
    events.push(scoreboard(ScoreboardEvent::Teams(Box::new(s2c::Teams {
        team_name: teams.team_name,
        action,
    }))));
}

// ---- plugin channels ---------------------------------------------------------------------

/// java17 `NHPC.handleCustomPayload` (`:1405-1469`):
///
/// - `MC|Brand`: the whole payload is the brand, `new String(bytes, UTF_8)`. The client keeps a
///   server brand from java18's payload, a VarInt-prefixed string, so the decoded brand goes on
///   in that form.
/// - `MC|RPack`: the payload is the pack's URL (`new String(bytes, UTF_8)`); 1.7.10 has no hash
///   and sends no status, which the resource-pack prompt applies for a 1.7.10 session.
/// - `MC|TrList`: a window id and 1.7.10's recipe list, which goes on as java18's
///   ([`trade_list_v47`]).
/// - `MC|BOpen` does not exist in 1.7.10 and is not passed on (java18 would open the held book).
/// - Any other channel goes on unchanged, as java18 passes it.
pub(super) fn plugin_message(message: s2c5::PluginMessage, events: &mut Vec<ClientEvent>) {
    let data = match message.channel.as_str() {
        "MC|Brand" => {
            let mut buf = PacketBuffer::new();
            buf.write_string(&String::from_utf8_lossy(&message.data));
            buf.as_slice().to_vec()
        }
        "MC|RPack" => {
            events.push(ClientEvent::ResourcePack {
                url: String::from_utf8_lossy(&message.data).into_owned(),
                hash: String::new(),
            });
            return;
        }
        "MC|TrList" => trade_list_v47(&message.data),
        "MC|BOpen" => return,
        _ => message.data,
    };
    events.push(ClientEvent::PluginMessage {
        channel: message.channel,
        data,
    });
}

/// A 1.7.10 `MC|TrList` payload in java18's layout: the window id, then
/// `MerchantRecipeList.func_151390_b`'s list (java17 `village/MerchantRecipeList.java:85-113`)
/// rewritten as java18's `readFromBuf` reads it (`village/MerchantRecipeList.java:76-106`): the
/// same count, stacks and flags, item stacks in the 1.8.9 encoding, and the two use counters
/// 1.7.10 does not send at the values its recipe takes (`MerchantRecipe` constructor: 0 uses of
/// 7, java17 `village/MerchantRecipe.java:21-27`; a disabled recipe has used all 7).
///
/// A list that does not decode is rewritten as far as it decodes, so java18's reader stops where
/// 1.7.10's did and the client reports it as 1.7.10 does ("Couldn't load trade info").
fn trade_list_v47(data: &[u8]) -> Vec<u8> {
    let mut input = PacketBuffer::from_slice(data);
    let mut out = PacketBuffer::new();
    let Ok(window_id) = input.read_i32() else {
        return data.to_vec();
    };
    out.write_i32(window_id);
    let Ok(count) = input.read_u8() else {
        return out.as_slice().to_vec();
    };
    out.write_u8(count);
    let mut budget = NbtBudget::per_packet();
    for _ in 0..count {
        let recipe = (|| -> Result<_, ProtoError> {
            let buy = input.read_item_v5(&mut budget)?;
            let sell = input.read_item_v5(&mut budget)?;
            let second = if input.read_bool()? {
                Some(input.read_item_v5(&mut budget)?)
            } else {
                None
            };
            let disabled = input.read_bool()?;
            Ok((buy, sell, second, disabled))
        })();
        let Ok((buy, sell, second, disabled)) = recipe else {
            break;
        };
        out.write_item_stack(buy.as_ref());
        out.write_item_stack(sell.as_ref());
        out.write_bool(second.is_some());
        if let Some(second) = &second {
            out.write_item_stack(second.as_ref());
        }
        out.write_bool(disabled);
        out.write_i32(0);
        out.write_i32(7);
    }
    out.as_slice().to_vec()
}
