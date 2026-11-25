//! The 1.7.10 entity packets: spawns, movement, metadata, status and the rest of the per-entity
//! traffic (java17 `client/network/NetHandlerPlayClient.java`, `NHPC` below).
//!
//! Where the 1.7.10 handler is the 1.8.9 one once decoded, the packet is lowered into its 1.8.9
//! form and applied by [`dispatch::apply`]; the function says why that is exact. Everything else
//! is the 1.7.10 handler itself over the tracker's 1.7.10 seams (`sync/entities/tracker_v5.rs`),
//! which convert a 1.7.10 `posY` (box bottom plus `yOffset`) into the client's box bottom.

use mc_data::{EntityType, ObjectType};
use mc_math::{BlockPos, DVec3, Facing};
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::v5::{s2c as s2c5, MetadataEntry5};
use mc_protocol::{MetadataEntry, ProtoError};
use mc_world::entity::{EntityKind, Metadata};

use crate::error::ClientError;
use crate::sync::dispatch;
use crate::sync::entities::tracker_v5::{entity_kind_v5, hanging_position_v5, spawn_box_bottom_v5};
use crate::sync::entities::{
    merge_registered_metadata, metadata_for_kind, PlayerProfileSnapshot, TrackedKind,
    FIXED_POINT_SCALE,
};
use crate::sync::event::ClientEvent;
use crate::sync::state::ClientState;

/// Whether `network_id` names the local player, as java17 `WorldClient.getEntityByID` answers
/// it (`thePlayer` for its own id, `client/multiplayer/WorldClient.java:236-239`). Zero is a
/// real id once JoinGame has set it.
fn is_local(state: &ClientState, network_id: i32) -> bool {
    (state.local.entity_id_known || state.local.entity_id != 0)
        && network_id == state.local.entity_id
}

/// A protocol-5 metadata list in the 1.8.9 value model, cut at its first type-7 entry.
///
/// 1.7.10 decodes a type-7 header as a `null` entry (java17
/// `DataWatcher.readWatchedListFromPacketBuffer`), and `updateWatchedObjectsFromList` applies the
/// entries before it and then throws on it (`entity/DataWatcher.java:322-341`). The second value
/// is the error that throw stands for; the caller applies the prefix first and then reports it.
fn metadata_prefix(list: &[MetadataEntry5]) -> (Vec<MetadataEntry>, Option<ClientError>) {
    let mut prefix = Vec::with_capacity(list.len());
    for entry in list {
        match entry.to_v47() {
            Some(converted) => prefix.push(converted),
            None => {
                return (
                    prefix,
                    Some(ClientError::Protocol(ProtoError::InvalidEnum {
                        what: "metadata entry of type 7 (a null WatchableObject)",
                        value: i32::from(entry.index),
                    })),
                );
            }
        }
    }
    (prefix, None)
}

/// `Err` for a list that stopped at a null entry, else `Ok`.
fn finish(error: Option<ClientError>) -> Result<(), ClientError> {
    error.map_or(Ok(()), Err)
}

/// The 1.8.9 skin-part byte (index 10) a 1.7.10 player watcher starts with: 1.7.10 has no skin
/// parts, and its byte 16 (registered 0, java17 `EntityPlayer.entityInit`) shows the cape and
/// the always-drawn hat ([`mc_world::v5::player_skin_parts_v5`]).
pub(crate) fn seed_player_skin_parts(metadata: &mut Metadata) {
    if metadata.get(10).is_none() {
        metadata.set(
            10,
            mc_world::entity::MetadataValue::Byte(mc_world::v5::PLAYER_DEFAULT_SKIN_PARTS_V5),
        );
    }
}

// ---- spawns ------------------------------------------------------------------------------

/// java17 `NHPC.handleSpawnPlayer` (`:449-480`): the remote player is built from the packet's own
/// `GameProfile` (UUID, name and properties, textures included); the tab list is not consulted.
/// The rest is 1.8.9's (java18 `:530-561`): the held item from the item id, the position
/// `serverPos / 32` (a remote player's `yOffset` is 0), the packet's metadata on the new watcher.
pub(super) fn spawn_player(
    state: &mut ClientState,
    player: s2c5::SpawnPlayer,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let (entries, null_entry) = metadata_prefix(&player.metadata);
    let mut metadata = Metadata::new();
    seed_player_skin_parts(&mut metadata);
    let remapped = mc_world::v5::remap_metadata_v5(EntityKind::Player, &metadata, &entries);
    merge_registered_metadata(TrackedKind::Player, &mut metadata, &remapped);
    let entity = state.entities.spawn_packet(
        &mut state.world,
        player.entity_id,
        TrackedKind::Player,
        Some(player.uuid),
        [player.x, player.y, player.z],
        player.yaw.to_degrees(),
        player.pitch.to_degrees(),
        DVec3::ZERO,
        metadata,
    );
    // java17 `S0CPacketSpawnPlayer.readPacketData` puts every property with its signature into
    // the profile; `AbstractClientPlayer`'s skin lookup then verifies the textures one.
    let profile = PlayerProfileSnapshot {
        username: player.name,
        properties: player
            .properties
            .into_iter()
            .map(|property| s2c::ProfileProperty {
                name: property.name,
                value: property.value,
                signature: Some(property.signature),
            })
            .collect(),
        trusted_textures: crate::sync::players::TrustedTextures::default(),
    };
    let _ = state
        .entities
        .set_player_profile(player.entity_id, entity, Some(profile));
    // `inventory.mainInventory[currentItem] = new ItemStack(Item.getItemById(id), 1, 0)`, or
    // empty for id 0 (`:461-470`).
    let current_item = (player.current_item != 0).then_some(mc_protocol::ItemStack {
        item_id: player.current_item,
        count: 1,
        damage: 0,
        nbt: None,
    });
    let _ = state
        .entities
        .apply_equipment(player.entity_id, 0, current_item);
    state.entities.init_v5(&state.world, player.entity_id, None);
    events.push(ClientEvent::EntitySpawned {
        network_id: player.entity_id,
        entity,
        kind: TrackedKind::Player,
    });
    finish(null_entry)
}

/// java17 `NHPC.handleSpawnMob` (`:726-762`): the 1.7.10 `EntityList` id and constructor size
/// ([`mc_world::v5::geometry_v5`]), the head yaw only (the body keeps `renderYawOffset`'s
/// default; java18 also turns it, `:925`), the position `serverPos / 32` (every living entity has
/// `yOffset` 0), the motion `(double)((float)v / 8000.0F)`, then the metadata, whose growing age
/// sizes an ageable from its first tick. An id 1.7.10 does not register makes `createEntityByID`
/// return null and the handler throw before anything exists; it is ignored here.
pub(super) fn spawn_mob(
    state: &mut ClientState,
    mob: s2c5::SpawnMob,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let Some(entity_type) = mc_world::v5::mob_type_v5(mob.entity_type) else {
        events.push(ClientEvent::EntitySpawnIgnored {
            network_id: mob.entity_id,
            reason: "no 1.7.10 EntityList class for this id",
        });
        return Ok(());
    };
    let kind = TrackedKind::Type(entity_type);
    let (entries, null_entry) = metadata_prefix(&mob.metadata);
    let remapped =
        mc_world::v5::remap_metadata_v5(EntityKind::Type(entity_type), &Metadata::new(), &entries);
    let velocity = |v: i16| f64::from(f32::from(v) / 8000.0_f32);
    let entity = state.entities.spawn_packet(
        &mut state.world,
        mob.entity_id,
        kind,
        None,
        [mob.x, mob.y, mob.z],
        mob.yaw.to_degrees(),
        mob.pitch.to_degrees(),
        DVec3::new(
            velocity(mob.velocity_x),
            velocity(mob.velocity_y),
            velocity(mob.velocity_z),
        ),
        metadata_for_kind(kind, &remapped),
    );
    state
        .entities
        .apply_spawn_head_look_v5(mob.entity_id, mob.head_yaw.to_degrees());
    let age = mc_world::v5::growing_age_v5(EntityKind::Type(entity_type), &entries);
    state.entities.init_v5(&state.world, mob.entity_id, age);
    events.push(ClientEvent::EntitySpawned {
        network_id: mob.entity_id,
        entity,
        kind,
    });
    finish(null_entry)
}

/// java17 `NHPC.handleSpawnObject` (`:244-385`) is java18's (`:297-442`) for every type 1.7.10
/// has, with the same constructor arguments, the same nine types whose `data` is cleared before
/// the velocity test and the same arrow-shooter rule, so the packet is lowered, except that:
///
/// - type 78 (the armor stand) is not in 1.7.10's chain: nothing is created;
/// - a falling block's data is `id | meta << 16` (`Block.getBlockById(data & 65535)`, meta
///   `data >> 16`), decoded with the 1.7.10 block table into 1.8.9's `Block.getStateId`;
/// - the spawn constructors of the half-height types place the box by their own `yOffset`
///   rule ([`spawn_box_bottom_v5`]), applied once the 1.8.9 arm has built the entity;
/// - an item frame hangs from the wall block the packet names (`(int)` of each coordinate) and
///   sits where `EntityHanging.setDirection` puts a 9-pixel frame (java17
///   `EntityItemFrame.java:58-66`, [`hanging_position_v5`]).
pub(super) fn spawn_object(
    state: &mut ClientState,
    object: s2c5::SpawnObject,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let raw_type = object.object_type as u8;
    if mc_world::v5::object_type_v5(raw_type).is_none() && mc_data::object_type(raw_type).is_some()
    {
        events.push(ClientEvent::EntitySpawnIgnored {
            network_id: object.entity_id,
            reason: "no 1.7.10 object for this type",
        });
        return Ok(());
    }
    let data = if mc_data::object_type(raw_type) == Some(ObjectType::FallingBlock) {
        mc_data::v5::state_from_v5_falling_block(object.data).to_java_state_id()
    } else {
        object.data
    };
    let before = events.len();
    dispatch::apply(
        state,
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id: object.entity_id,
            object_type: object.object_type,
            x: object.x,
            y: object.y,
            z: object.z,
            pitch: object.pitch,
            yaw: object.yaw,
            data,
            velocity: object.velocity,
        }),
        events,
    )?;
    let spawned = events[before..].iter().find_map(|event| match event {
        ClientEvent::EntitySpawned {
            network_id, kind, ..
        } if *network_id == object.entity_id => Some(*kind),
        _ => None,
    });
    let Some(kind) = spawned else {
        return Ok(());
    };
    state.entities.init_v5(&state.world, object.entity_id, None);
    // A fishing hook (`yOffset` 0) stays where the 1.8.9 arm put it.
    let TrackedKind::Type(entity_type) = kind else {
        return Ok(());
    };
    let wire = |v: i32| f64::from(v) / FIXED_POINT_SCALE;
    if entity_type == EntityType::ItemFrame {
        // `new EntityItemFrame(world, (int)x, (int)y, (int)z, data)`.
        let wall = BlockPos::new(
            wire(object.x) as i32,
            wire(object.y) as i32,
            wire(object.z) as i32,
        );
        let position = hanging_position_v5(wall, object.data, 9, 9);
        state
            .entities
            .place_hanging_v5(&mut state.world, object.entity_id, position);
        return Ok(());
    }
    let pos_y = wire(object.y);
    let box_y = spawn_box_bottom_v5(entity_type, pos_y);
    if box_y.to_bits() != pos_y.to_bits() {
        let position = DVec3::new(wire(object.x), box_y, wire(object.z));
        state
            .entities
            .place_spawn_v5(&mut state.world, object.entity_id, position);
    }
    Ok(())
}

/// java17 `NHPC.handleSpawnExperienceOrb` (`:387-397`) passes the packet's fixed-point integers
/// to the constructor unscaled (`S11PacketSpawnExperienceOrb`'s getters return them raw), so the
/// orb is built at 32 times its coordinates with its box `yOffset` (0.25) below that, while its
/// `serverPos` is right; its first move puts it in place. Everything else is java18's handler
/// (`:447-458`): the same constructor (random motion, the XP value), rotation zero.
pub(super) fn spawn_experience_orb(
    state: &mut ClientState,
    orb: s2c5::SpawnExperienceOrb,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    dispatch::apply(
        state,
        S2CPlay::SpawnExperienceOrb(s2c::SpawnExperienceOrb {
            entity_id: orb.entity_id,
            x: orb.x,
            y: orb.y,
            z: orb.z,
            xp_value: orb.xp_value,
        }),
        events,
    )?;
    state.entities.init_v5(&state.world, orb.entity_id, None);
    let position = DVec3::new(
        f64::from(orb.x),
        spawn_box_bottom_v5(EntityType::XPOrb, f64::from(orb.y)),
        f64::from(orb.z),
    );
    state
        .entities
        .place_spawn_v5(&mut state.world, orb.entity_id, position);
    Ok(())
}

/// java17 `NHPC.handleSpawnPainting` (`:423-427`): `new EntityPainting(world, x, y, z,
/// direction, title)`, the art java18's handler (`:491-496`) also picks: the title's, after the
/// same random draw over the arts that fit. The direction is 1.7.10's `Direction` index, whose
/// 0-3 are java18's `EnumFacing.getHorizontal` order (south, west, north, east).
///
/// 1.7.10 hangs a painting from the wall block the packet names, 1.8.9 from the block in front
/// of it: the 1.8.9 arm gets that block (the same wall, the same fit test), then the painting
/// sits where `EntityHanging.setDirection` puts it ([`hanging_position_v5`]).
pub(super) fn spawn_painting(
    state: &mut ClientState,
    painting: s2c5::SpawnPainting,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let wall = BlockPos::new(painting.x, painting.y, painting.z);
    let facing = Facing::from_horizontal(painting.direction);
    let (dx, _, dz) = facing.offset();
    let entity_id = painting.entity_id;
    let before = events.len();
    dispatch::apply(
        state,
        S2CPlay::SpawnPainting(s2c::SpawnPainting {
            entity_id,
            title: painting.title,
            position: BlockPos::new(wall.x + dx, wall.y, wall.z + dz),
            facing,
        }),
        events,
    )?;
    let spawned = events[before..].iter().find_map(|event| match event {
        ClientEvent::EntitySpawned {
            network_id, entity, ..
        } if *network_id == entity_id => Some(*entity),
        _ => None,
    });
    let Some(entity) = spawned else {
        return Ok(());
    };
    let size = state
        .world
        .ecs
        .get::<&crate::app::entrender::Hanging>(entity)
        .map(|hanging| (hanging.art.size_x, hanging.art.size_y))
        .ok();
    state.entities.init_v5(&state.world, entity_id, None);
    if let Some((width, height)) = size {
        let position = hanging_position_v5(wall, painting.direction, width, height);
        state
            .entities
            .place_hanging_v5(&mut state.world, entity_id, position);
    }
    Ok(())
}

/// java17 `NHPC.handleSpawnGlobalEntity` (`:399-421`) = java18 (`:463-486`): a lightning bolt for
/// type 1, nothing else.
pub(super) fn lower_spawn_global(global: &s2c5::SpawnGlobalEntity) -> S2CPlay {
    S2CPlay::SpawnGlobalEntity(s2c::SpawnGlobalEntity {
        entity_id: global.entity_id,
        entity_type: global.kind,
        x_fixed: global.x,
        y_fixed: global.y,
        z_fixed: global.z,
    })
}

// ---- movement ----------------------------------------------------------------------------

/// Whether a move or teleport names the local player, which a 1.7.10 session ignores.
///
/// java17 `NHPC.handleEntityMovement` / `handleEntityTeleport` apply such a packet to the local
/// player too (`WorldClient.getEntityByID` answers `thePlayer`), and its
/// `EntityLivingBase.setPositionAndRotation2` zeroes `yOffset` (`entity/EntityLivingBase.java:
/// 1815-1824`): the eye drops to the box bottom and a vanilla server then kicks the player for an
/// illegal stance. Vanilla servers never send these; a server that does would make vanilla
/// 1.7.10 fail, and the client does not reproduce that failure.
fn ignores_local_move(state: &ClientState, entity_id: i32) -> bool {
    is_local(state, entity_id)
}

/// java17 `NHPC.handleEntityMovement` (`:508-524`): `serverPos += delta`, the target is
/// `serverPos / 32` with the packet's rotation if it has one, three steps; the packets carry no
/// on-ground flag and the handler sets none.
pub(super) fn relative_move(
    state: &mut ClientState,
    entity_id: i32,
    delta: [i8; 3],
    look: Option<(f32, f32)>,
) {
    if ignores_local_move(state, entity_id) {
        return;
    }
    state
        .entities
        .apply_relative_move_v5(&mut state.world, entity_id, delta, look);
}

/// java17 `NHPC.handleEntityTeleport` (`:482-498`): `serverPos` = the packet; the target is
/// `(x / 32, y / 32 + 0.015625, z / 32)` whatever the current position (java18 holds still
/// within a dead zone and drops the 1/64), three steps, no on-ground flag.
pub(super) fn teleport(state: &mut ClientState, teleport: &s2c5::EntityTeleport) {
    if ignores_local_move(state, teleport.entity_id) {
        return;
    }
    state.entities.apply_teleport_v5(
        &mut state.world,
        teleport.entity_id,
        [teleport.x, teleport.y, teleport.z],
        teleport.yaw.to_degrees(),
        teleport.pitch.to_degrees(),
    );
}

// ---- per-entity state ----------------------------------------------------------------------

/// java17 `NHPC.handleEntityMetadata` (`:439-447`) = java18 (`:516-525`) on the 1.7.10 watcher:
/// the list is remapped onto the 1.8.9 indices the client stores
/// ([`mc_world::v5::remap_metadata_v5`], against the entity's current watcher) and merged as
/// 1.8.9 merges it, without the watcher callbacks only 1.8.9 has
/// ([`crate::sync::EntityTracker::apply_metadata_v5`]); an ageable's growing age is also kept
/// whole, which the client tick sizes it from. An id with no entity is ignored, null entries
/// included.
pub(super) fn entity_metadata(
    state: &mut ClientState,
    packet: &s2c5::EntityMetadata,
) -> Result<(), ClientError> {
    if packet.entity_id == state.local.entity_id {
        let (entries, null_entry) = metadata_prefix(&packet.metadata);
        let remapped =
            mc_world::v5::remap_metadata_v5(EntityKind::Player, &state.local.metadata, &entries);
        merge_registered_metadata(TrackedKind::Player, &mut state.local.metadata, &remapped);
        return finish(null_entry);
    }
    let Some((kind, entity)) = state
        .entities
        .get(packet.entity_id)
        .map(|tracked| (tracked.kind, tracked.entity))
    else {
        return Ok(());
    };
    let (entries, null_entry) = metadata_prefix(&packet.metadata);
    let remapped = {
        let empty = Metadata::new();
        let current = state.world.ecs.get::<&Metadata>(entity).ok();
        mc_world::v5::remap_metadata_v5(
            entity_kind_v5(kind),
            current.as_deref().unwrap_or(&empty),
            &entries,
        )
    };
    state
        .entities
        .apply_metadata_v5(&mut state.world, packet.entity_id, &remapped);
    if let Some(age) = mc_world::v5::growing_age_v5(entity_kind_v5(kind), &entries) {
        state.entities.set_age_v5(packet.entity_id, age);
    }
    finish(null_entry)
}

/// java17 `NHPC.handleEntityStatus` (`:827-835`): `handleHealthUpdate(opcode)` on the entity,
/// whose overrides are java18's `handleStatusUpdate` ones for every class 1.7.10 has, except
/// that 1.7.10 has no opcode 20 (`EntityLiving`'s spawn particles), 21 (the guardian sound,
/// java18 `:1031-1035`), 22 or 23 (the reduced-debug switch), and its squid has no 19. Those do
/// nothing; everything else is lowered.
pub(super) fn entity_status(
    state: &mut ClientState,
    status: s2c5::EntityStatus,
    events: &mut Vec<ClientEvent>,
) -> Result<(), ClientError> {
    let squid = || {
        state
            .entities
            .get(status.entity_id)
            .is_some_and(|tracked| tracked.kind == TrackedKind::Type(EntityType::Squid))
    };
    if matches!(status.opcode, 20..=23) || (status.opcode == 19 && squid()) {
        return Ok(());
    }
    dispatch::apply(
        state,
        S2CPlay::EntityStatus(s2c::EntityStatus {
            entity_id: status.entity_id,
            opcode: status.opcode,
        }),
        events,
    )
}

/// java17 `NHPC.handleEntityEffect` (`:1323-1333`): `new PotionEffect(id, duration,
/// amplifier)` with the maximum-duration flag; java18 (`:1502-1513`) is the same call with a
/// particle flag that 1.7.10 does not have (its particles always show).
pub(super) fn lower_entity_effect(effect: s2c5::EntityEffect) -> S2CPlay {
    S2CPlay::EntityEffect(s2c::EntityEffect {
        entity_id: effect.entity_id,
        effect_id: effect.effect_id,
        amplifier: effect.amplifier,
        duration: i32::from(effect.duration),
        show_particles: true,
    })
}

/// java17 `NHPC.handleEntityProperties` (`:1604-1641`) = java18 (`:2051-2085`): the decoder
/// already refused an operation outside 0-2 as `AttributeModifier`'s constructor does.
pub(super) fn lower_entity_properties(properties: s2c5::EntityProperties) -> S2CPlay {
    S2CPlay::EntityProperties(s2c::EntityProperties {
        entity_id: properties.entity_id,
        properties: properties
            .properties
            .into_iter()
            .map(|property| s2c::AttributePropertySnapshot {
                key: property.key,
                base_value: property.base_value,
                modifiers: property
                    .modifiers
                    .into_iter()
                    .map(|modifier| s2c::AttributeModifierSnapshot {
                        uuid: modifier.uuid,
                        amount: modifier.amount,
                        operation: match modifier.operation {
                            0 => s2c::AttributeModifierOperation::Add,
                            1 => s2c::AttributeModifierOperation::MultiplyBase,
                            _ => s2c::AttributeModifierOperation::Multiply,
                        },
                    })
                    .collect(),
            })
            .collect(),
    })
}

/// java17 `NHPC.handleEntityAttach` (`:776-825`) = java18 (`:966-1016`): the same leash byte,
/// mount and boat-occupancy rules (`Entity.mountEntity` is the same in both, java17
/// `entity/Entity.java:1551-1587`).
pub(super) fn lower_entity_attach(attach: &s2c5::EntityAttach) -> S2CPlay {
    S2CPlay::EntityAttach(s2c::EntityAttach {
        entity_id: attach.entity_id,
        vehicle_entity_id: attach.vehicle_entity_id,
        attach_type: match attach.leash {
            0 => s2c::AttachType::Vehicle,
            1 => s2c::AttachType::Leash,
            other => s2c::AttachType::Unknown(other),
        },
    })
}
