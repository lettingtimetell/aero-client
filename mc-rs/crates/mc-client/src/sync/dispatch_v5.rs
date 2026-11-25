//! The 1.7.10 dispatch: every protocol-5 clientbound Play packet, applied to [`ClientState`].
//!
//! java17: `client/network/NetHandlerPlayClient.java` (`NHPC` below), against java18's file of
//! the same name where an arm says the two handlers agree.
//!
//! # How a packet gets here
//!
//! `Session::poll_next` queues a 1.7.10 connection's packets; the tick drains them (at most 1001
//! per tick, java17 `NetworkManager.processReceivedPackets`) through `session/v5.rs`, which calls
//! [`apply_v5`] for one packet, then `Session::react_to` for its events, exactly as the 1.8.9
//! frame poll calls `dispatch::apply`. So everything `react_to` and the client do with a
//! [`ClientEvent`] happens for 1.7.10 too.
//!
//! Not here, by design:
//!
//! * `S00 KeepAlive` and `S40 Disconnect` are handled on the network thread (`net/task_v5.rs`),
//!   as 1.7.10 handles them on the Netty thread (java17 `NetworkManager.channelRead0`,
//!   `network/NetworkManager.java:103-116`). Their arms only serve a caller that
//!   feeds packets in directly.
//! * Full-column `S21`/`S26` arrive already converted to the 1.8.9 form and applied through
//!   `dispatch::apply_prebuilt_chunks` (`net/chunk_v5.rs`). Only the rest (an unload, a partial
//!   `S21`, a chunk packet with no background thread) reach the arms below, which convert with
//!   the world's own sky flag.
//! * `S08 PlayerPosLook` belongs to the session, which owns the player and the C06 reply
//!   (`session/v5.rs`, java17 `NHPC.handlePlayerPosLook`); it never passes S08 here.
//!
//! # The pattern for each arm
//!
//! [`apply_v5`] matches every [`S2CPlay5`] variant, with no `_ =>` arm: a variant added to
//! `mc-protocol` breaks this build instead of being dropped. Each arm is one of:
//!
//! 1. **Lowered**: the packet is rewritten into its 1.8.9 [`S2CPlay`] and handed to
//!    [`dispatch::apply`]. Only where the 1.7.10 handler is the 1.8.9 one once decoded; the arm
//!    or its `lower_*` function cites both handlers.
//! 2. **A 1.7.10 handler**: a function here or in the `entity`, `world` and `ui` modules beside
//!    this file, using the existing state APIs (the tracker's 1.7.10 seams, the world, the tab
//!    list...), named after the java17 handler and citing it, for packets whose data or handling
//!    differs.
//!
//! Errors follow 1.8.9's rule: a handler error is returned, the session logs it and the
//! connection goes on, where vanilla 1.7.10 would crash (java17 `processReceivedPackets` has no
//! `catch`, `network/NetworkManager.java:188-215`; java18 `Util.runTask` logs and goes on).

mod entity;
mod ui;
mod world;

#[cfg(test)]
mod captures;
#[cfg(test)]
mod tests;

use mc_math::BlockPos;
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::v5::{s2c as s2c5, S2CPlay5};
use mc_world::entity::Metadata;

use crate::error::ClientError;
use crate::net::chunk_v5;
use crate::sync::dispatch;
use crate::sync::event::ClientEvent;
use crate::sync::state::ClientState;

/// Apply one protocol-5 Play packet; see the module docs for what each arm may do.
///
/// # Errors
///
/// What the handler reports: a [`dispatch::apply`] error for a lowered packet, a chunk whose
/// masks its data does not fill, the disconnect a directly fed `S40` stands for, or a packet
/// vanilla 1.7.10 throws on (a null metadata entry, a malformed particle name, a display slot
/// past its three, a map update past its image).
#[allow(clippy::too_many_lines)] // 65 arms; splitting it would only hide the exhaustiveness.
pub fn apply_v5(
    state: &mut ClientState,
    packet: S2CPlay5,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    match packet {
        // ---- network-thread packets (`hasPriority`), here only when fed in directly ----
        S2CPlay5::KeepAlive(_) => Ok(()),
        S2CPlay5::Disconnect(s2c5::Disconnect { reason }) => {
            Err(ClientError::Disconnected { reason })
        }
        // ---- session-owned: java17 NHPC.handlePlayerPosLook (:545-566) ----
        S2CPlay5::PlayerPosLook(_) => {
            tracing::warn!("S08 is applied by the session; ignored here");
            Ok(())
        }

        // ---- the connection and the local player ----
        // java17 NHPC.handleJoinGame (:229-242) = java18 (:277-292) less the reduced-debug flag
        // and the difficulty option. The C15 + MC|Brand replies are `Session::react_to`'s.
        S2CPlay5::JoinGame(join) => {
            dispatch::apply(state, lower_join_game(join), events)?;
            entity::seed_player_skin_parts(&mut state.local.metadata);
            Ok(())
        }
        // java17 NHPC.handleRespawn (:849-865) = java18 (:1056-1073), except that the new player
        // starts from a fresh watcher (java17 `Minecraft.setDimensionAndSpawnPlayer`,
        // `client/Minecraft.java:2202-2232`, copies nothing; java18 copies the old player's,
        // `Minecraft.java:2445-2447`): not sprinting, not burning, full air.
        S2CPlay5::Respawn(respawn) => {
            dispatch::apply(state, lower_respawn(respawn), events)?;
            state.local.metadata = Metadata::new();
            entity::seed_player_skin_parts(&mut state.local.metadata);
            Ok(())
        }
        // java17 NHPC.handleChat (:684-687): no position byte, every message to the chat.
        S2CPlay5::Chat(chat) => dispatch::apply(state, lower_chat(chat), events),
        // java17 NHPC.handleTimeUpdate (:764-768) = java18 (:952-957)
        S2CPlay5::TimeUpdate(time) => dispatch::apply(state, lower_time(time), events),
        // java17 NHPC.handleSpawnPosition (:770-774) = java18 (:959-964)
        S2CPlay5::SpawnPosition(spawn) => dispatch::apply(state, lower_spawn(&spawn), events),
        // java17 NHPC.handleUpdateHealth (:837-842) = java18 (:1042-1048)
        S2CPlay5::UpdateHealth(health) => dispatch::apply(state, lower_health(health), events),
        // java17 NHPC.handleHeldItemChange (:500-506) = java18 (:598-606); the C09 echo's tick
        // follows from where the drain runs.
        S2CPlay5::HeldItemChange(held) => dispatch::apply(state, lower_held_item(held), events),
        // java17 NHPC.handlePlayerAbilities (:1378-1387) = java18 (:1668-1678)
        S2CPlay5::PlayerAbilities(abilities) => {
            dispatch::apply(state, lower_abilities(&abilities), events)
        }
        // java17 NHPC.handleSetExperience (:844-847) = java18 (:1050-1054)
        S2CPlay5::SetExperience(experience) => {
            dispatch::apply(state, lower_experience(&experience), events)
        }
        // java17 NHPC.handleChangeGameState (:1195-1258): reason 3's chat line, 1.7.10's game
        // types, no reason 10.
        S2CPlay5::ChangeGameState(game_state) => {
            world::change_game_state(state, &game_state, events)
        }

        // ---- entity spawns ----
        S2CPlay5::SpawnPlayer(player) => entity::spawn_player(state, player, events),
        S2CPlay5::SpawnObject(object) => entity::spawn_object(state, object, events),
        S2CPlay5::SpawnMob(mob) => entity::spawn_mob(state, mob, events),
        S2CPlay5::SpawnPainting(painting) => entity::spawn_painting(state, painting, events),
        S2CPlay5::SpawnExperienceOrb(orb) => entity::spawn_experience_orb(state, orb, events),
        S2CPlay5::SpawnGlobalEntity(global) => {
            dispatch::apply(state, entity::lower_spawn_global(&global), events)
        }

        // ---- entity movement ----
        S2CPlay5::Entity(moved) => {
            entity::relative_move(state, moved.entity_id, [0; 3], None);
            Ok(())
        }
        S2CPlay5::EntityRelMove(moved) => {
            let delta = [moved.delta_x, moved.delta_y, moved.delta_z];
            entity::relative_move(state, moved.entity_id, delta, None);
            Ok(())
        }
        S2CPlay5::EntityLook(look) => {
            let rotation = (look.yaw.to_degrees(), look.pitch.to_degrees());
            entity::relative_move(state, look.entity_id, [0; 3], Some(rotation));
            Ok(())
        }
        S2CPlay5::EntityLookAndRelMove(moved) => {
            let delta = [moved.delta_x, moved.delta_y, moved.delta_z];
            let rotation = (moved.yaw.to_degrees(), moved.pitch.to_degrees());
            entity::relative_move(state, moved.entity_id, delta, Some(rotation));
            Ok(())
        }
        S2CPlay5::EntityTeleport(teleport) => {
            entity::teleport(state, &teleport);
            Ok(())
        }
        // java17 NHPC.handleEntityHeadLook (:526-535) = java18 (:637-647)
        S2CPlay5::EntityHeadLook(head) => dispatch::apply(
            state,
            S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
                entity_id: head.entity_id,
                yaw: head.yaw,
            }),
            events,
        ),
        // java17 NHPC.handleEntityVelocity (:429-437) = java18 (:501-510)
        S2CPlay5::EntityVelocity(velocity) => dispatch::apply(
            state,
            S2CPlay::EntityVelocity(s2c::EntityVelocity {
                entity_id: velocity.entity_id,
                motion_x: velocity.motion_x,
                motion_y: velocity.motion_y,
                motion_z: velocity.motion_z,
            }),
            events,
        ),
        // java17 NHPC.handleDestroyEntities (:537-543) = java18 (:654-662)
        S2CPlay5::DestroyEntities(destroy) => dispatch::apply(
            state,
            S2CPlay::DestroyEntities(s2c::DestroyEntities {
                entity_ids: destroy.entity_ids,
            }),
            events,
        ),
        // java17 NHPC.handleEntityAttach (:776-825) = java18 (:966-1016)
        S2CPlay5::EntityAttach(attach) => {
            dispatch::apply(state, entity::lower_entity_attach(&attach), events)
        }

        // ---- per-entity state ----
        S2CPlay5::EntityMetadata(metadata) => entity::entity_metadata(state, &metadata),
        S2CPlay5::EntityStatus(status) => entity::entity_status(state, status, events),
        // java17 NHPC.handleAnimation (:689-719) = java18 (:867-897)
        S2CPlay5::Animation(animation) => dispatch::apply(
            state,
            S2CPlay::Animation(s2c::Animation {
                entity_id: animation.entity_id,
                animation_type: animation.animation_type,
            }),
            events,
        ),
        // java17 NHPC.handleEntityEquipment (:1151-1159) = java18 (:1297-1306)
        S2CPlay5::EntityEquipment(equipment) => dispatch::apply(
            state,
            S2CPlay::EntityEquipment(s2c::EntityEquipment {
                entity_id: equipment.entity_id,
                equipment_slot: equipment.equipment_slot,
                item: equipment.item,
            }),
            events,
        ),
        S2CPlay5::EntityEffect(effect) => {
            dispatch::apply(state, entity::lower_entity_effect(effect), events)
        }
        // java17 NHPC.handleRemoveEntityEffect (:1335-1343) = java18 (:1606-1615)
        S2CPlay5::RemoveEntityEffect(effect) => dispatch::apply(
            state,
            S2CPlay::RemoveEntityEffect(s2c::RemoveEntityEffect {
                entity_id: effect.entity_id,
                effect_id: effect.effect_id,
            }),
            events,
        ),
        S2CPlay5::EntityProperties(properties) => {
            dispatch::apply(state, entity::lower_entity_properties(properties), events)
        }
        // java17 NHPC.handleUseBed (:721-724) -> `sleepInBedAt`, java18 (:903-907) -> `trySleep`:
        // the same bed. The local player's 1.7.10 bed pose (`yOffset` 0.2 with `posY = y +
        // 0.9375`, java17 `EntityPlayer.sleepInBedAt:1198-1284`) is the session's
        // (`session/movement_v5.rs`); a remote sleeper keeps 1.8.9's box, 0.05 lower. The y is a
        // signed byte in 1.7.10's packet.
        S2CPlay5::UseBed(bed) => dispatch::apply(
            state,
            S2CPlay::UseBed(s2c::UseBed {
                player_id: bed.player_id,
                bed_position: BlockPos::new(bed.x, i32::from(bed.y), bed.z),
            }),
            events,
        ),
        // java17 NHPC.handleCollectItem (:658-682) = java18 (:819-844): the same sound and
        // removal; the pickup particle flies to `posY - 0.5` where java18's flies to
        // `posY + 0.5` (java17 `EntityPickupFX`), a drawing difference this client keeps.
        S2CPlay5::CollectItem(collect) => dispatch::apply(
            state,
            S2CPlay::CollectItem(s2c::CollectItem {
                collected_entity_id: collect.collected_entity_id,
                collector_entity_id: collect.collector_entity_id,
            }),
            events,
        ),

        // ---- blocks and chunks ----
        // java17 NHPC.handleBlockChange (:622-625) = java18 (:776-780), the cell through the
        // 1.7.10 block decode.
        S2CPlay5::BlockChange(change) => dispatch::apply(state, lower_block(&change), events),
        // java17 NHPC.handleMultiBlockChange (:568-596) = java18 (:734-742), each record through
        // the 1.7.10 block decode.
        S2CPlay5::MultiBlockChange(change) => {
            dispatch::apply(state, lower_multi_block(&change), events)
        }
        // java17 NHPC.handleChunkData (:598-620) / handleMapChunkBulk (:1176-1193) = java18
        // (:747-771 / :1337-1356), converted with the world's sky flag (`net/chunk_v5.rs`).
        S2CPlay5::ChunkData(chunk) => {
            let chunk = chunk_v5::chunk_data_v47(&chunk, state.world.has_sky_light())?;
            dispatch::apply(state, S2CPlay::ChunkData(chunk), events)
        }
        S2CPlay5::MapChunkBulk(bulk) => {
            let bulk = chunk_v5::map_chunk_bulk_v47(&bulk, state.world.has_sky_light())?;
            dispatch::apply(state, S2CPlay::MapChunkBulk(bulk), events)
        }
        S2CPlay5::BlockAction(action) => {
            dispatch::apply(state, world::lower_block_action(&action), events)
        }
        S2CPlay5::BlockBreakAnimation(animation) => {
            dispatch::apply(state, world::lower_break_animation(&animation), events)
        }
        S2CPlay5::Explosion(explosion) => {
            dispatch::apply(state, world::lower_explosion(explosion), events)
        }

        // ---- world events, sounds and particles ----
        S2CPlay5::Effect(effect) => match world::lower_effect(&effect) {
            Some(lowered) => dispatch::apply(state, lowered, events),
            None => Ok(()),
        },
        S2CPlay5::SoundEffect(sound) => match world::lower_sound(sound) {
            Some(lowered) => dispatch::apply(state, lowered, events),
            None => Ok(()),
        },
        S2CPlay5::Particles(particles) => world::particles(&particles, events),

        // ---- signs, tile entities, maps ----
        S2CPlay5::UpdateSign(sign) => {
            dispatch::apply(state, world::lower_update_sign(sign), events)
        }
        S2CPlay5::UpdateTileEntity(tile) => match world::lower_tile_entity(tile) {
            Some(lowered) => dispatch::apply(state, lowered, events),
            None => Ok(()),
        },
        S2CPlay5::SignEditorOpen(open) => {
            world::sign_editor_open(state, &open, events);
            Ok(())
        }
        S2CPlay5::Maps(maps) => world::maps(state, &maps, events),

        // ---- windows ----
        S2CPlay5::OpenWindow(window) => {
            ui::open_window(state, &window, events);
            Ok(())
        }
        S2CPlay5::CloseWindow(window) => {
            ui::close_window(&window, events);
            Ok(())
        }
        S2CPlay5::SetSlot(slot) => {
            ui::set_slot(slot, events);
            Ok(())
        }
        S2CPlay5::WindowItems(items) => {
            ui::window_items(items, events);
            Ok(())
        }
        S2CPlay5::WindowProperty(property) => {
            ui::window_property(&property, events);
            Ok(())
        }
        S2CPlay5::ConfirmTransaction(transaction) => {
            ui::confirm_transaction(&transaction, events);
            Ok(())
        }

        // ---- statistics, tab list, completion ----
        S2CPlay5::Statistics(statistics) => {
            dispatch::apply(state, ui::lower_statistics(statistics), events)
        }
        S2CPlay5::PlayerListItem(item) => {
            ui::player_list_item(state, &item);
            Ok(())
        }
        // java17 NHPC.handleTabComplete (:1389-1398) = java18 (:1683-1693)
        S2CPlay5::TabComplete(complete) => dispatch::apply(
            state,
            S2CPlay::TabComplete(s2c::TabComplete {
                matches: complete.matches,
            }),
            events,
        ),

        // ---- scoreboard and teams ----
        S2CPlay5::ScoreboardObjective(objective) => {
            ui::scoreboard_objective(objective, events);
            Ok(())
        }
        S2CPlay5::UpdateScore(score) => {
            ui::update_score(score, events);
            Ok(())
        }
        S2CPlay5::DisplayScoreboard(display) => ui::display_scoreboard(display, events),
        S2CPlay5::Teams(teams) => {
            ui::teams(teams, events);
            Ok(())
        }

        // ---- plugin channels ----
        S2CPlay5::PluginMessage(message) => {
            ui::plugin_message(message, events);
            Ok(())
        }
    }
}

/// Protocol 5 has no reduced-debug flag (java17 `S01PacketJoinGame.readPacketData`); `level_type`
/// is already one of 1.7.10's names.
fn lower_join_game(join: s2c5::JoinGame) -> S2CPlay {
    S2CPlay::JoinGame(s2c::JoinGame {
        entity_id: join.entity_id,
        hardcore_mode: join.hardcore_mode,
        game_mode: join.game_mode,
        dimension: join.dimension,
        difficulty: join.difficulty,
        max_players: join.max_players,
        level_type: join.level_type,
        reduced_debug_info: false,
    })
}

fn lower_respawn(respawn: s2c5::Respawn) -> S2CPlay {
    S2CPlay::Respawn(s2c::Respawn {
        dimension: respawn.dimension,
        difficulty: respawn.difficulty,
        game_mode: respawn.game_mode,
        level_type: respawn.level_type,
    })
}

/// java17 `handleChat` prints every message to the chat (`NetHandlerPlayClient.java:684-687`),
/// which is 1.8.9's position 0.
fn lower_chat(chat: s2c5::Chat) -> S2CPlay {
    S2CPlay::Chat(s2c::Chat {
        component: chat.component,
        position: 0,
    })
}

fn lower_time(time: s2c5::TimeUpdate) -> S2CPlay {
    S2CPlay::TimeUpdate(s2c::TimeUpdate {
        total_world_time: time.total_world_time,
        world_time: time.world_time,
    })
}

fn lower_spawn(spawn: &s2c5::SpawnPosition) -> S2CPlay {
    S2CPlay::SpawnPosition(s2c::SpawnPosition {
        position: BlockPos::new(spawn.x, spawn.y, spawn.z),
    })
}

fn lower_health(health: s2c5::UpdateHealth) -> S2CPlay {
    S2CPlay::UpdateHealth(s2c::UpdateHealth {
        health: health.health,
        food_level: i32::from(health.food_level),
        saturation_level: health.saturation_level,
    })
}

fn lower_held_item(held: s2c5::HeldItemChange) -> S2CPlay {
    S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: held.slot })
}

/// The flags byte both versions read the four booleans from (java17
/// `S39PacketPlayerAbilities.readPacketData`: 1 invulnerable, 2 flying, 4 may fly, 8 creative).
fn lower_abilities(abilities: &s2c5::PlayerAbilities) -> S2CPlay {
    let flags = u8::from(abilities.invulnerable)
        | u8::from(abilities.flying) << 1
        | u8::from(abilities.allow_flying) << 2
        | u8::from(abilities.creative_mode) << 3;
    S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
        flags,
        fly_speed: abilities.fly_speed,
        walk_speed: abilities.walk_speed,
    })
}

/// 1.7.10 sends the level and the total as shorts (java17 `S1FPacketSetExperience`).
fn lower_experience(experience: &s2c5::SetExperience) -> S2CPlay {
    S2CPlay::SetExperience(s2c::SetExperience {
        experience_bar: experience.experience_bar,
        level: i32::from(experience.level),
        total_experience: i32::from(experience.total_experience),
    })
}

/// java17 `handleBlockChange`: `func_147492_c(x, y, z, block, meta)`, the 1.7.10 cell stored as
/// `mc_data::v5` decodes it (always a registered 1.8.9 state).
fn lower_block(change: &s2c5::BlockChange) -> S2CPlay {
    let state = mc_data::v5::state_from_v5_id_meta(change.block_id, i32::from(change.metadata));
    S2CPlay::BlockChange(s2c::BlockChange {
        position: BlockPos::new(change.x, i32::from(change.y), change.z),
        block_state_id: i32::from(state.id),
    })
}

/// java17 `handleMultiBlockChange`: each record's `id << 4 | meta` through the 1.7.10 block
/// decode; a record cut short ends the list, as the handler's `IOException` does.
fn lower_multi_block(change: &s2c5::MultiBlockChange) -> S2CPlay {
    let changes = change
        .records()
        .map(|record| {
            let raw = record.block_id << 4 | u16::from(record.metadata);
            s2c::MultiBlockChangeEntry {
                rel_x: record.rel_x,
                y: record.y,
                rel_z: record.rel_z,
                block_state_id: i32::from(mc_data::v5::state_from_v5(raw).id),
            }
        })
        .collect();
    S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
        chunk_x: change.chunk_x,
        chunk_z: change.chunk_z,
        changes,
    })
}
