//! Unit tests for the 1.7.10 dispatch: one or more per handler, with the expectation taken from
//! the java17 handler each test cites and a 1.8.9 control where the two differ.

use mc_math::DVec3;
use mc_protocol::play::s2c::GameMode;
use mc_protocol::v5::{MetadataEntry5, MetadataValue5, PacketBufferV5};
use mc_protocol::{Angle, PacketBuffer};
use mc_world::entity::{Metadata as Stored, MetadataValue as StoredValue, Position};

use super::*;
use crate::sync::entities::TrackedKind;
use crate::sync::event::{ChatPosition, ScoreboardEvent, WindowEvent};

const LOCAL_ID: i32 = 7;

/// Apply one packet and return its events; the handler must succeed.
fn apply(state: &mut ClientState, packet: S2CPlay5) -> Vec<ClientEvent> {
    let mut events = Vec::new();
    apply_v5(state, packet, &mut events).expect("the handler succeeds");
    events
}

/// Apply one packet that must fail (vanilla 1.7.10 throws on it), returning its events.
fn apply_failing(state: &mut ClientState, packet: S2CPlay5) -> Vec<ClientEvent> {
    let mut events = Vec::new();
    assert!(apply_v5(state, packet, &mut events).is_err());
    events
}

/// A 1.7.10 client after `S01JoinGame` (entity id 7, survival, overworld).
fn joined() -> ClientState {
    let mut state = ClientState::new();
    apply(
        &mut state,
        S2CPlay5::JoinGame(s2c5::JoinGame {
            entity_id: LOCAL_ID,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 1,
            max_players: 20,
            level_type: "flat".to_owned(),
        }),
    );
    state
}

/// Load an all-air column through the shared chunk path, so blocks can be set and entities in
/// it are resident.
fn load_column(state: &mut ClientState, x: i32, z: i32) {
    let data = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
    dispatch::apply(
        state,
        S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
            chunk_x: x,
            chunk_z: z,
            ground_up_continuous: true,
            primary_bit_mask: 0b1,
            data,
        }),
        &mut Vec::new(),
    )
    .expect("an all-air column loads");
}

fn byte(index: u8, value: i8) -> MetadataEntry5 {
    MetadataEntry5 {
        index,
        value: MetadataValue5::Byte(value),
    }
}

fn position(state: &ClientState, id: i32) -> DVec3 {
    let tracked = state.entities.get(id).expect("tracked");
    state
        .world
        .ecs
        .get::<&Position>(tracked.entity)
        .expect("position")
        .0
}

fn stored(state: &ClientState, id: i32, index: u8) -> Option<StoredValue> {
    let tracked = state.entities.get(id)?;
    let metadata = state.world.ecs.get::<&Stored>(tracked.entity).ok()?;
    metadata.get(index).cloned()
}

fn spawn_object(id: i32, object_type: i8, y: i32, data: i32) -> S2CPlay5 {
    S2CPlay5::SpawnObject(s2c5::SpawnObject {
        entity_id: id,
        object_type,
        x: 100 * 32 + 16,
        y,
        z: 100 * 32 + 16,
        pitch: Angle(0),
        yaw: Angle(0),
        data,
        velocity: (data > 0).then_some((0, 0, 0)),
    })
}

fn spawn_mob(id: i32, entity_type: u8, metadata: Vec<MetadataEntry5>) -> S2CPlay5 {
    S2CPlay5::SpawnMob(s2c5::SpawnMob {
        entity_id: id,
        entity_type,
        x: 32,
        y: 64 * 32,
        z: 32,
        yaw: Angle(64),
        pitch: Angle(0),
        head_yaw: Angle(32),
        velocity_x: 0,
        velocity_y: 0,
        velocity_z: 0,
        metadata,
    })
}

fn spawned(events: &[ClientEvent], id: i32) -> bool {
    events
        .iter()
        .any(|e| matches!(e, ClientEvent::EntitySpawned { network_id, .. } if *network_id == id))
}

// ---- the local player ------------------------------------------------------------------------

/// java17 `EntityPlayer.entityInit` registers byte 16 as 0 (cape shown) and 1.7.10 always draws
/// the hat, so a joined player's 1.8.9 skin-part byte starts as hat and cape.
#[test]
fn join_seeds_the_local_skin_parts() {
    let state = joined();
    assert_eq!(
        state.local.metadata.get(10),
        Some(&StoredValue::Byte(
            mc_world::v5::PLAYER_DEFAULT_SKIN_PARTS_V5
        ))
    );
    assert_eq!(state.local.entity_id, LOCAL_ID);
}

/// java17 `Minecraft.setDimensionAndSpawnPlayer` (`client/Minecraft.java:2202-2232`) builds the
/// new player with a fresh watcher: the old sprint flag is gone. java18 copies the old watcher
/// (`Minecraft.java:2445-2447`), the control.
#[test]
fn respawn_starts_from_a_fresh_watcher() {
    let mut state = joined();
    state.local.metadata.set(0, StoredValue::Byte(0x08));
    let respawn = s2c5::Respawn {
        dimension: 0,
        difficulty: 1,
        game_mode: 0,
        level_type: "flat".to_owned(),
    };
    let mut control = joined();
    control.local.metadata.set(0, StoredValue::Byte(0x08));
    dispatch::apply(
        &mut control,
        lower_respawn(respawn.clone()),
        &mut Vec::new(),
    )
    .expect("1.8.9 respawn");
    assert_eq!(
        control.local.metadata.get(0),
        Some(&StoredValue::Byte(0x08))
    );

    let events = apply(&mut state, S2CPlay5::Respawn(respawn));
    assert!(events
        .iter()
        .any(|e| matches!(e, ClientEvent::Respawned { .. })));
    assert_eq!(
        state.local.metadata.get(0),
        None,
        "flags back to the default 0"
    );
    assert_eq!(
        state.local.metadata.get(10),
        Some(&StoredValue::Byte(
            mc_world::v5::PLAYER_DEFAULT_SKIN_PARTS_V5
        ))
    );
}

/// java17 `NHPC.handleEntityMetadata` on the local player: byte 16 bit 1 hides the cape
/// (`EntityPlayer.getHideCape`), which the client stores as the 1.8.9 skin-part byte.
#[test]
fn local_metadata_maps_the_cape_switch() {
    let mut state = joined();
    apply(
        &mut state,
        S2CPlay5::EntityMetadata(s2c5::EntityMetadata {
            entity_id: LOCAL_ID,
            metadata: vec![byte(0, 0x08), byte(16, 0x02)],
        }),
    );
    assert_eq!(state.local.metadata.get(0), Some(&StoredValue::Byte(0x08)));
    assert_eq!(state.local.metadata.get(10), Some(&StoredValue::Byte(0x40)));
}

/// java17 `NHPC.handleChangeGameState` (`:1195-1258`): reason 3 prints `gameMode.changed` and
/// uses 1.7.10's `GameType.getByID`, where 3 is survival (java18: spectator, the control).
#[test]
fn game_state_reason_3_prints_and_maps_1_7_10_game_types() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::ChangeGameState(s2c5::ChangeGameState {
            reason: 3,
            value: 1.0,
        }),
    );
    assert!(matches!(
        &events[0],
        ClientEvent::Chat { position: ChatPosition::Chat, component }
            if component.to_json().contains("gameMode.changed")
    ));
    assert!(matches!(
        events[1],
        ClientEvent::GameModeChanged(GameMode::Creative)
    ));

    apply(
        &mut state,
        S2CPlay5::ChangeGameState(s2c5::ChangeGameState {
            reason: 3,
            value: 3.0,
        }),
    );
    assert_eq!(state.local.game_mode, GameMode::Survival);
    let mut control = joined();
    let mut events = Vec::new();
    dispatch::apply(
        &mut control,
        S2CPlay::ChangeGameState(s2c::ChangeGameState {
            reason: s2c::GameStateReason::ChangeGameMode,
            value: 3.0,
        }),
        &mut events,
    )
    .expect("1.8.9 game state");
    assert_eq!(control.local.game_mode, GameMode::Spectator);
    assert!(!events.iter().any(|e| matches!(e, ClientEvent::Chat { .. })));
}

/// Reason 10 is java18's elder-guardian effect; 1.7.10 has no such reason. Reason 0 keeps its
/// chat line and rain (1) is java18's.
#[test]
fn game_state_reason_10_does_nothing_and_the_rest_lowers() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::ChangeGameState(s2c5::ChangeGameState {
            reason: 10,
            value: 0.0,
        }),
    );
    assert!(events.is_empty());
    let events = apply(
        &mut state,
        S2CPlay5::ChangeGameState(s2c5::ChangeGameState {
            reason: 0,
            value: 0.0,
        }),
    );
    assert!(matches!(&events[..], [ClientEvent::Chat { .. }]));
    apply(
        &mut state,
        S2CPlay5::ChangeGameState(s2c5::ChangeGameState {
            reason: 1,
            value: 0.0,
        }),
    );
    assert!(state.env.raining);
}

// ---- spawns ------------------------------------------------------------------------------

fn spawn_player(metadata: Vec<MetadataEntry5>) -> S2CPlay5 {
    S2CPlay5::SpawnPlayer(s2c5::SpawnPlayer {
        entity_id: 40,
        uuid: uuid::Uuid::from_u128(0x1234),
        name: "Remote".to_owned(),
        properties: vec![s2c5::ProfileProperty {
            name: "textures".to_owned(),
            value: "dGV4dHVyZXM=".to_owned(),
            signature: "c2ln".to_owned(),
        }],
        x: 32,
        y: 64 * 32,
        z: 32,
        yaw: Angle(0),
        pitch: Angle(0),
        current_item: 276,
        metadata,
    })
}

/// java17 `NHPC.handleSpawnPlayer` (`:449-480`) builds the player from the packet's own profile;
/// java18 (the control) looks the profile up in the tab list and has none without an entry.
#[test]
fn spawn_player_takes_its_profile_from_the_packet() {
    let mut state = joined();
    let events = apply(&mut state, spawn_player(vec![byte(16, 0x02)]));
    assert!(spawned(&events, 40));
    let tracked = state.entities.get(40).expect("spawned");
    assert_eq!(tracked.kind, TrackedKind::Player);
    assert_eq!(tracked.uuid, Some(uuid::Uuid::from_u128(0x1234)));
    let profile = tracked.player_profile().expect("a profile from the packet");
    assert_eq!(profile.username, "Remote");
    assert_eq!(profile.properties[0].signature.as_deref(), Some("c2ln"));
    assert_eq!(tracked.equipment[0].as_ref().map(|s| s.item_id), Some(276));
    // Byte 16 bit 1 hides the cape: the hat alone.
    assert_eq!(stored(&state, 40, 10), Some(StoredValue::Byte(0x40)));
    assert_eq!(position(&state, 40), DVec3::new(1.0, 64.0, 1.0));

    let mut control = joined();
    dispatch::apply(
        &mut control,
        S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
            entity_id: 40,
            player_uuid: uuid::Uuid::from_u128(0x1234),
            x: 32,
            y: 64 * 32,
            z: 32,
            yaw: Angle(0),
            pitch: Angle(0),
            current_item: 276,
            metadata: Vec::new(),
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 spawn");
    assert!(control
        .entities
        .get(40)
        .expect("spawned")
        .player_profile()
        .is_none());
}

/// Without byte 16 in the spawn list the constructor's 0 applies: hat and cape.
#[test]
fn spawn_player_without_byte_16_shows_the_cape() {
    let mut state = joined();
    apply(&mut state, spawn_player(Vec::new()));
    assert_eq!(
        stored(&state, 40, 10),
        Some(StoredValue::Byte(
            mc_world::v5::PLAYER_DEFAULT_SKIN_PARTS_V5
        ))
    );
}

/// java17 `NHPC.handleSpawnMob` (`:726-762`): the 1.7.10 registry (101, the rabbit, is unknown
/// and nothing spawns), the head yaw alone (the body keeps 0; java18 turns it to the head, the
/// control), and 1.7.10 metadata indices (`EntityLiving` custom name 10 -> 1.8.9's 2).
#[test]
fn spawn_mob_uses_the_1_7_10_registry_head_and_metadata() {
    let mut state = joined();
    let name = MetadataEntry5 {
        index: 10,
        value: MetadataValue5::String("Bob".to_owned()),
    };
    let events = apply(&mut state, spawn_mob(50, 54, vec![name]));
    assert!(spawned(&events, 50));
    let tracked = state.entities.get(50).expect("zombie");
    assert_eq!(tracked.kind, TrackedKind::Type(mc_data::EntityType::Zombie));
    assert_eq!(tracked.head_yaw, 45.0);
    assert_eq!(tracked.render.body_yaw, 0.0);
    assert_eq!(
        stored(&state, 50, 2),
        Some(StoredValue::Str("Bob".to_owned()))
    );

    let mut control = joined();
    dispatch::apply(
        &mut control,
        S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: 50,
            entity_type: 54,
            x: 32,
            y: 64 * 32,
            z: 32,
            yaw: Angle(64),
            pitch: Angle(0),
            head_pitch: Angle(32),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Vec::new(),
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 spawn");
    assert_eq!(
        control.entities.get(50).expect("zombie").render.body_yaw,
        45.0
    );

    let events = apply(&mut state, spawn_mob(51, 101, Vec::new()));
    assert!(state.entities.get(51).is_none());
    assert!(matches!(
        &events[..],
        [ClientEvent::EntitySpawnIgnored { network_id: 51, .. }]
    ));
}

/// The half-height objects start where their 1.7.10 spawn constructors put the box: a boat at
/// `y` (`setPosition(x, y + yOffset, z)`), an item `yOffset` lower (`setPosition(x, y, z)`); an
/// arrow (yOffset 0) at `y`, as in 1.8.9; type 78 (the armor stand) is not a 1.7.10 object.
#[test]
fn spawn_object_places_half_height_boxes_like_the_1_7_10_constructors() {
    let mut state = joined();
    let wire_y = 64 * 32 + 10;
    let y = f64::from(wire_y) / 32.0;
    apply(&mut state, spawn_object(60, 1, wire_y, 0));
    let boat_offset = f64::from(0.3_f32);
    assert_eq!(position(&state, 60).y, (y + boat_offset) - boat_offset);
    apply(&mut state, spawn_object(61, 2, wire_y, 1));
    assert_eq!(position(&state, 61).y, y - f64::from(0.125_f32));
    assert_eq!(state.entities.get(61).expect("item").server_pos[1], wire_y);
    apply(&mut state, spawn_object(62, 50, wire_y, 0));
    assert_eq!(position(&state, 62).y, y - f64::from(0.49_f32));
    apply(&mut state, spawn_object(63, 60, wire_y, 0));
    assert_eq!(position(&state, 63).y, y);

    let events = apply(&mut state, spawn_object(64, 78, wire_y, 0));
    assert!(state.entities.get(64).is_none());
    assert!(matches!(
        &events[..],
        [ClientEvent::EntitySpawnIgnored { network_id: 64, .. }]
    ));
}

/// A falling block's data is `id | meta << 16` in 1.7.10 (`Block.getBlockById(data & 65535)`,
/// meta `data >> 16`); the client keeps 1.8.9's `Block.getStateId` packing.
#[test]
fn spawn_object_decodes_1_7_10_falling_block_data() {
    let mut state = joined();
    // Sand (12) with meta 1: red sand in both versions.
    apply(&mut state, spawn_object(70, 70, 64 * 32, 12 | 1 << 16));
    let tracked = state.entities.get(70).expect("falling block");
    let data = state
        .world
        .ecs
        .get::<&crate::app::entrender::ObjectData>(tracked.entity)
        .expect("data")
        .0;
    assert_eq!(data, 12 | 1 << 12);
    assert_eq!(
        position(&state, 70).y,
        64.0 - f64::from(0.49_f32),
        "yOffset below the wire y"
    );
}

/// java17 `NHPC.handleSpawnExperienceOrb` (`:387-397`) builds the orb at the raw fixed-point
/// values, its `serverPos` right; its first move puts it back. Here it waits in the spawn queue
/// of the unloaded column out there until that move, then joins the world where it lands.
#[test]
fn xp_orb_spawns_at_raw_coordinates_until_it_moves() {
    let mut state = joined();
    load_column(&mut state, 6, 6);
    let (x, y, z) = (100 * 32, 64 * 32, 100 * 32);
    apply(
        &mut state,
        S2CPlay5::SpawnExperienceOrb(s2c5::SpawnExperienceOrb {
            entity_id: 80,
            x,
            y,
            z,
            xp_value: 3,
        }),
    );
    assert_eq!(
        position(&state, 80),
        DVec3::new(3200.0, 2048.0 - f64::from(0.25_f32), 3200.0)
    );
    assert_eq!(state.entities.get(80).expect("orb").server_pos, [x, y, z]);
    assert!(
        !state.entities.is_resident(80),
        "queued in its unloaded column"
    );

    apply(
        &mut state,
        S2CPlay5::EntityRelMove(s2c5::EntityRelMove {
            entity_id: 80,
            delta_x: 0,
            delta_y: 0,
            delta_z: 0,
        }),
    );
    assert_eq!(
        position(&state, 80),
        DVec3::new(100.0, 64.0 - f64::from(0.25_f32), 100.0)
    );
    assert!(
        state.entities.is_resident(80),
        "re-admitted in the loaded column"
    );
}

fn dimensions(state: &ClientState, id: i32) -> (f32, f32) {
    state.entities.get(id).expect("tracked").dimensions()
}

fn int(index: u8, value: i32) -> MetadataEntry5 {
    MetadataEntry5 {
        index,
        value: MetadataValue5::Int(value),
    }
}

/// 1.7.10 constructor sizes through client ticks: a zombie keeps java17's 0.6 x 1.8 (java18's
/// 0.6 x 1.95 is the control), a slime is `0.6F * size` (`EntitySlime.onUpdate`), a wither
/// skeleton 0.72 x 2.34 (`EntitySkeleton.onLivingUpdate`), a foal sized by its growing age
/// (`EntityHorse.getHorseSize`).
#[test]
fn entities_take_1_7_10_sizes() {
    let mut state = joined();
    load_column(&mut state, 0, 0);
    apply(&mut state, spawn_mob(50, 54, Vec::new()));
    apply(&mut state, spawn_mob(51, 55, vec![byte(16, 4)]));
    apply(&mut state, spawn_mob(52, 51, vec![byte(13, 1)]));
    apply(&mut state, spawn_mob(53, 100, vec![int(12, -12_000)]));
    let foal = |age: i32| {
        let scale = mc_world::v5::ageable_scale_v5(mc_data::EntityType::EntityHorse, age);
        (1.4_f32 * scale, 1.6_f32 * scale)
    };
    for _ in 0..2 {
        state.entities.tick_interpolation(&mut state.world);
        assert_eq!(dimensions(&state, 50), (0.6, 1.8));
        assert_eq!(dimensions(&state, 51), (0.6_f32 * 4.0, 0.6_f32 * 4.0));
        assert_eq!(dimensions(&state, 52), (0.72, 2.34));
        assert_eq!(dimensions(&state, 53), foal(-12_000));
    }
    assert_ne!(
        foal(-12_000),
        foal(-24_000),
        "the age curve is not 1.8.9's constant half"
    );
    apply(
        &mut state,
        S2CPlay5::EntityMetadata(s2c5::EntityMetadata {
            entity_id: 53,
            metadata: vec![int(12, -24_000)],
        }),
    );
    state.entities.tick_interpolation(&mut state.world);
    assert_eq!(dimensions(&state, 53), foal(-24_000));

    let mut control = joined();
    dispatch::apply(
        &mut control,
        S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: 50,
            entity_type: 54,
            x: 32,
            y: 64 * 32,
            z: 32,
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Vec::new(),
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 spawn");
    assert_eq!(dimensions(&control, 50), (0.6, 1.95));
}

/// 1.7.10 hangs an item frame or a painting from the wall block the packet names and
/// `EntityHanging.setDirection` puts it 0.5625 out of the wall (and half a block along and up
/// for a 32-pixel side); java18's block-anchored pose, the control, would put the same packet
/// inside the wall.
#[test]
fn hanging_entities_sit_where_set_direction_puts_them() {
    let mut state = joined();
    load_column(&mut state, 0, 1);
    apply(
        &mut state,
        S2CPlay5::SpawnObject(s2c5::SpawnObject {
            entity_id: 90,
            object_type: 71,
            x: 10 * 32,
            y: 64 * 32,
            z: 20 * 32,
            pitch: Angle(0),
            yaw: Angle(0),
            data: 0,
            velocity: None,
        }),
    );
    apply(
        &mut state,
        S2CPlay5::SpawnPainting(s2c5::SpawnPainting {
            entity_id: 91,
            title: "Match".to_owned(),
            x: 10,
            y: 66,
            z: 20,
            direction: 0,
        }),
    );
    let frame = DVec3::new(10.5, 64.5, 21.0625);
    let painting = DVec3::new(11.0, 67.0, 21.0625);
    for _ in 0..2 {
        assert_eq!(position(&state, 90), frame);
        assert_eq!(
            state
                .entities
                .get(90)
                .expect("frame")
                .position(&state.world),
            frame
        );
        assert_eq!(position(&state, 91), painting);
        state.entities.tick_interpolation(&mut state.world);
    }

    let mut control = joined();
    dispatch::apply(
        &mut control,
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id: 90,
            object_type: 71,
            x: 10 * 32,
            y: 64 * 32,
            z: 20 * 32,
            pitch: Angle(0),
            yaw: Angle(0),
            data: 0,
            velocity: None,
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 spawn");
    assert_eq!(
        control
            .entities
            .get(90)
            .expect("frame")
            .position(&control.world)
            .z,
        20.03125
    );
}

fn item_frame(id: i32, wall: BlockPos, direction: i32) -> S2CPlay5 {
    S2CPlay5::SpawnObject(s2c5::SpawnObject {
        entity_id: id,
        object_type: 71,
        x: wall.x * 32,
        y: wall.y * 32,
        z: wall.z * 32,
        pitch: Angle(0),
        yaw: Angle(0),
        data: direction,
        velocity: None,
    })
}

/// A frame on a wall at the east edge of its column, facing east, sits 0.5625 out of the wall,
/// in the next column, and is filed there: shown as soon as that column is loaded, whether or
/// not the wall's column is (java17 places it by its `setDirection` position).
#[test]
fn an_item_frame_facing_into_the_next_column_lives_in_that_column() {
    let wall = BlockPos::new(15, 64, 20);
    let mut state = joined();
    load_column(&mut state, 1, 1);
    apply(&mut state, item_frame(90, wall, 3));
    assert_eq!(position(&state, 90), DVec3::new(16.0625, 64.5, 20.5));
    assert!(state.entities.is_resident(90), "its own column is loaded");
    load_column(&mut state, 0, 1);
    assert!(
        state.entities.is_resident(90),
        "the wall's column changes nothing"
    );

    let mut state = joined();
    load_column(&mut state, 0, 1);
    apply(&mut state, item_frame(91, wall, 3));
    assert!(
        !state.entities.is_resident(91),
        "its own column is not loaded yet"
    );
    load_column(&mut state, 1, 1);
    assert!(
        state.entities.is_resident(91),
        "shown once its column arrives"
    );
}

/// A teleport of a 1.7.10 hanging entity (sent after 400 watcher changes) takes the base
/// `Entity` setter: the frame goes to the packet position, `y / 32 + 1/64`, and stays there
/// (java18's item-frame setter would anchor it to that block: (11.5, 70.5, 21.03125)).
#[test]
fn a_hanging_entity_moves_to_its_teleport() {
    let mut state = joined();
    load_column(&mut state, 0, 1);
    apply(&mut state, item_frame(90, BlockPos::new(10, 64, 20), 0));
    apply(
        &mut state,
        S2CPlay5::EntityTeleport(s2c5::EntityTeleport {
            entity_id: 90,
            x: 11 * 32,
            y: 70 * 32,
            z: 21 * 32,
            yaw: Angle(0),
            pitch: Angle(0),
        }),
    );
    let moved = DVec3::new(11.0, 70.015_625, 21.0);
    for _ in 0..2 {
        assert_eq!(position(&state, 90), moved);
        let frame = state.entities.get(90).expect("frame");
        assert_eq!(frame.position(&state.world), moved);
        state.entities.tick_interpolation(&mut state.world);
    }
}

fn spawn_player_at(id: i32, x: i32, z: i32) -> S2CPlay5 {
    S2CPlay5::SpawnPlayer(s2c5::SpawnPlayer {
        entity_id: id,
        uuid: uuid::Uuid::from_u128(id as u128),
        name: format!("P{id}"),
        properties: Vec::new(),
        x: x * 32,
        y: 64 * 32,
        z: z * 32,
        yaw: Angle(0),
        pitch: Angle(0),
        current_item: 0,
        metadata: Vec::new(),
    })
}

fn attach(rider: i32, vehicle: i32) -> S2CPlay5 {
    S2CPlay5::EntityAttach(s2c5::EntityAttach {
        entity_id: rider,
        vehicle_entity_id: vehicle,
        leash: 0,
    })
}

/// Remote riders take 1.7.10's seats (java17 `Entity.updateRiderPosition`, `EntityBoat`'s): the
/// vehicle's `posY` (its box bottom plus `yOffset`, 0.3 for a boat and 0.35 for a minecart),
/// plus its `getMountedYOffset` (-0.30000001192092896 for boats and minecarts, else
/// `height * 0.75`), plus the player's `getYOffset`, -0.5; a boat also seats its rider 0.4
/// forward along its yaw. java18's seats (the control: the same scene through the 1.8.9
/// handlers, `getMountedYOffset` -0.3 for a boat and 0 for a minecart, `getYOffset` -0.35) put
/// the player 0.15 lower in a boat, 0.10 higher in a minecart and 0.15 higher on a pig.
#[test]
fn remote_riders_take_1_7_10_seats() {
    let pig_mount = f64::from(0.9_f32) * 0.75;
    // (rider, vehicle, the seat over the vehicle's box bottom in 1.7.10, and in 1.8.9)
    let seats = [
        (
            40,
            60,
            f64::from(0.3_f32) - 0.300_000_011_920_928_96 - 0.5,
            -0.3 - 0.35,
        ),
        (
            41,
            61,
            f64::from(0.35_f32) - 0.300_000_011_920_928_96 - 0.5,
            -0.35,
        ),
        (42, 62, pig_mount - 0.5, pig_mount - 0.35),
    ];
    let rise =
        |state: &ClientState, rider, vehicle| position(state, rider).y - position(state, vehicle).y;

    let mut state = joined();
    load_column(&mut state, 0, 0);
    load_column(&mut state, 6, 6);
    apply(&mut state, spawn_object(60, 1, 70 * 32, 0));
    apply(&mut state, spawn_object(61, 10, 70 * 32, 0));
    apply(&mut state, spawn_mob(62, 90, Vec::new()));
    for (rider, vehicle, ..) in seats {
        apply(&mut state, spawn_player_at(rider, 2, 2));
        apply(&mut state, attach(rider, vehicle));
    }
    state.entities.tick_interpolation(&mut state.world);
    for (rider, vehicle, v5, _) in seats {
        assert!(
            (rise(&state, rider, vehicle) - v5).abs() < 1e-9,
            "1.7.10 seat of {rider}"
        );
    }
    let (boat, seat) = (position(&state, 60), position(&state, 40));
    let boat_yaw = state
        .world
        .ecs
        .get::<&mc_world::entity::Rotation>(state.entities.get(60).expect("boat").entity)
        .expect("rotation")
        .yaw;
    let radians = f64::from(boat_yaw) * core::f64::consts::PI / 180.0;
    assert!((seat.x - boat.x - radians.cos() * 0.4).abs() < 1e-9);
    assert!((seat.z - boat.z - radians.sin() * 0.4).abs() < 1e-9);

    let mut control = joined();
    load_column(&mut control, 0, 0);
    load_column(&mut control, 6, 6);
    let object = |entity_id, object_type| {
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id,
            object_type,
            x: 100 * 32 + 16,
            y: 70 * 32,
            z: 100 * 32 + 16,
            pitch: Angle(0),
            yaw: Angle(0),
            data: 0,
            velocity: None,
        })
    };
    let mut packets = vec![
        object(60, 1),
        object(61, 10),
        S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: 62,
            entity_type: 90,
            x: 32,
            y: 64 * 32,
            z: 32,
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Vec::new(),
        }),
    ];
    for (rider, vehicle, ..) in seats {
        packets.push(S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
            entity_id: rider,
            player_uuid: uuid::Uuid::from_u128(rider as u128),
            x: 2 * 32,
            y: 64 * 32,
            z: 2 * 32,
            yaw: Angle(0),
            pitch: Angle(0),
            current_item: 0,
            metadata: Vec::new(),
        }));
        packets.push(S2CPlay::EntityAttach(s2c::EntityAttach {
            entity_id: rider,
            vehicle_entity_id: vehicle,
            attach_type: s2c::AttachType::Vehicle,
        }));
    }
    for packet in packets {
        dispatch::apply(&mut control, packet, &mut Vec::new()).expect("a 1.8.9 handler");
    }
    control.entities.tick_interpolation(&mut control.world);
    for (rider, vehicle, _, v18) in seats {
        assert!(
            (rise(&control, rider, vehicle) - v18).abs() < 1e-9,
            "1.8.9 seat of {rider}"
        );
    }
}

/// java17 `EntitySlime` has no watcher callback: a size byte after the spawn leaves the body yaw
/// and the rotation where the spawn put them; java18's `onDataWatcherUpdate(16)` (the control,
/// the 1.8.9 handler on the same slime) turns both to the head.
#[test]
fn a_slime_size_update_does_not_turn_the_body_in_1_7_10() {
    let yaws = |state: &ClientState| {
        let slime = state.entities.get(55).expect("slime");
        let rotation = state
            .world
            .ecs
            .get::<&mc_world::entity::Rotation>(slime.entity)
            .expect("rotation")
            .yaw;
        (slime.render.head_yaw, slime.render.body_yaw, rotation)
    };
    let mut state = joined();
    load_column(&mut state, 0, 0);
    apply(&mut state, spawn_mob(55, 55, Vec::new()));
    let spawned = yaws(&state);
    assert_eq!((spawned.0, spawned.1), (45.0, 0.0));
    apply(
        &mut state,
        S2CPlay5::EntityMetadata(s2c5::EntityMetadata {
            entity_id: 55,
            metadata: vec![byte(16, 2)],
        }),
    );
    assert_eq!(yaws(&state), spawned);
    assert_eq!(stored(&state, 55, 16), Some(StoredValue::Byte(2)));

    let mut control = joined();
    load_column(&mut control, 0, 0);
    apply(&mut control, spawn_mob(55, 55, Vec::new()));
    dispatch::apply(
        &mut control,
        S2CPlay::EntityMetadata(s2c::EntityMetadata {
            entity_id: 55,
            metadata: vec![mc_protocol::MetadataEntry {
                index: 16,
                value: mc_protocol::MetadataValue::Byte(2),
            }],
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 metadata");
    assert_eq!(yaws(&control), (45.0, 45.0, 45.0));
}

// ---- movement ----------------------------------------------------------------------------

/// java17 `NHPC.handleEntityTeleport` (`:482-498`): the target is `y / 32 + 1/64` even when the
/// entity is already there (java18 holds still inside its dead zone, the control), and the
/// packet carries no on-ground flag.
#[test]
fn teleport_adds_a_sixty_fourth_and_never_holds_still() {
    let mut state = joined();
    apply(&mut state, spawn_mob(50, 54, Vec::new()));
    let teleport = s2c5::EntityTeleport {
        entity_id: 50,
        x: 32,
        y: 64 * 32,
        z: 32,
        yaw: Angle(0),
        pitch: Angle(0),
    };
    apply(&mut state, S2CPlay5::EntityTeleport(teleport));
    let tracked = state.entities.get(50).expect("zombie");
    assert_eq!(tracked.target_pos, DVec3::new(1.0, 64.015_625, 1.0));
    assert_eq!(tracked.interpolation_steps, 3);
    assert!(!tracked.on_ground);

    let mut control = joined();
    apply(&mut control, spawn_mob(50, 54, Vec::new()));
    dispatch::apply(
        &mut control,
        S2CPlay::EntityTeleport(s2c::EntityTeleport {
            entity_id: 50,
            x: 32,
            y: 64 * 32,
            z: 32,
            yaw: Angle(0),
            pitch: Angle(0),
            on_ground: true,
        }),
        &mut Vec::new(),
    )
    .expect("1.8.9 teleport");
    let tracked = control.entities.get(50).expect("zombie");
    assert_eq!(tracked.target_pos, DVec3::new(1.0, 64.0, 1.0));
    assert!(tracked.on_ground);
}

/// java17 `NHPC.handleEntityMovement` (`:508-524`): `serverPos += delta`, the target
/// `serverPos / 32`, and for a half-height object its box `yOffset` lower (an item snaps there,
/// `Entity.setPositionAndRotation2`).
#[test]
fn relative_moves_target_the_box_bottom_of_the_1_7_10_pos_y() {
    let mut state = joined();
    load_column(&mut state, 6, 6);
    apply(&mut state, spawn_object(61, 2, 80 * 32, 1));
    apply(
        &mut state,
        S2CPlay5::EntityRelMove(s2c5::EntityRelMove {
            entity_id: 61,
            delta_x: 0,
            delta_y: -32,
            delta_z: 0,
        }),
    );
    let tracked = state.entities.get(61).expect("item");
    assert_eq!(tracked.server_pos[1], 79 * 32);
    assert_eq!(position(&state, 61).y, 79.0 - f64::from(0.125_f32));
}

/// A move or teleport naming the local player is ignored in a 1.7.10 session (vanilla 1.7.10
/// would zero the player's `yOffset` and be kicked by a vanilla server; only other servers send
/// these). The same packets move a remote entity, the control.
#[test]
fn moves_naming_the_local_player_are_ignored() {
    let mut state = joined();
    let before = (
        state.local.server_pos,
        state.local.target_pos,
        state.local.interpolation_steps,
    );
    let packets = |entity_id| {
        [
            S2CPlay5::EntityRelMove(s2c5::EntityRelMove {
                entity_id,
                delta_x: 32,
                delta_y: 0,
                delta_z: 0,
            }),
            S2CPlay5::EntityLookAndRelMove(s2c5::EntityLookAndRelMove {
                entity_id,
                delta_x: 0,
                delta_y: 32,
                delta_z: 0,
                yaw: Angle(64),
                pitch: Angle(0),
            }),
            S2CPlay5::EntityTeleport(s2c5::EntityTeleport {
                entity_id,
                x: 320,
                y: 2048,
                z: 320,
                yaw: Angle(0),
                pitch: Angle(0),
            }),
        ]
    };
    for packet in packets(LOCAL_ID) {
        assert!(apply(&mut state, packet).is_empty());
        assert_eq!(
            (
                state.local.server_pos,
                state.local.target_pos,
                state.local.interpolation_steps,
            ),
            before
        );
    }
    apply(&mut state, spawn_mob(50, 54, Vec::new()));
    for packet in packets(50) {
        apply(&mut state, packet);
    }
    assert_eq!(
        state.entities.get(50).expect("zombie").server_pos,
        [320, 2048, 320]
    );
}

// ---- per-entity state ----------------------------------------------------------------------

/// java17 `NHPC.handleEntityMetadata`: entries before a type-7 (null) entry are applied, then
/// 1.7.10 throws on it.
#[test]
fn metadata_stops_at_a_null_entry() {
    let mut state = joined();
    apply(&mut state, spawn_mob(50, 54, Vec::new()));
    let null = MetadataEntry5 {
        index: 4,
        value: MetadataValue5::Null,
    };
    apply_failing(
        &mut state,
        S2CPlay5::EntityMetadata(s2c5::EntityMetadata {
            entity_id: 50,
            metadata: vec![byte(0, 0x20), null, byte(11, 1)],
        }),
    );
    assert_eq!(stored(&state, 50, 0), Some(StoredValue::Byte(0x20)));
    assert_eq!(stored(&state, 50, 3), None, "the entry after the null one");
}

/// 1.7.10 has no status 20-23 (java18: `EntityLiving` spawn particles, the guardian sound, the
/// reduced-debug switch); 2 is java18's.
#[test]
fn entity_status_drops_the_1_8_9_only_opcodes() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::EntityStatus(s2c5::EntityStatus {
            entity_id: LOCAL_ID,
            opcode: 22,
        }),
    );
    assert!(events.is_empty());
    let events = apply(
        &mut state,
        S2CPlay5::EntityStatus(s2c5::EntityStatus {
            entity_id: LOCAL_ID,
            opcode: 2,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::EntityStatus { opcode: 2, .. }]
    ));
    assert_eq!(state.local.hurt_time, 10);
}

/// java17 `NHPC.handleEntityEffect` (`:1323-1333`): no particle flag; particles always show.
#[test]
fn entity_effects_always_show_particles() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::EntityEffect(s2c5::EntityEffect {
            entity_id: LOCAL_ID,
            effect_id: 8,
            amplifier: 1,
            duration: 32_767,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::EntityEffectAdded(effect)]
            if effect.show_particles && effect.duration == 32_767 && effect.effect_id == 8
    ));
}

/// java17 `NHPC.handleEntityProperties` (`:1604-1641`): an absent entity is ignored; a non-living
/// one throws (logged here).
#[test]
fn entity_properties_for_a_non_living_entity_fail() {
    let mut state = joined();
    apply(&mut state, spawn_object(61, 2, 64 * 32, 1));
    let properties = |entity_id| {
        S2CPlay5::EntityProperties(s2c5::EntityProperties {
            entity_id,
            properties: vec![s2c5::AttributeSnapshot {
                key: "generic.movementSpeed".to_owned(),
                base_value: 0.1,
                modifiers: Vec::new(),
            }],
        })
    };
    apply_failing(&mut state, properties(61));
    assert!(apply(&mut state, properties(999)).is_empty());
    let events = apply(&mut state, properties(LOCAL_ID));
    assert!(matches!(&events[..], [ClientEvent::EntityProperties(_)]));
}

// ---- world -------------------------------------------------------------------------------

/// java17 `RenderGlobal.doSpawnParticle`: names, with only the three big ones skipping the
/// 16-block check; unknown names spawn nothing; a malformed crack name throws.
#[test]
fn particles_resolve_1_7_10_names() {
    let particles = |name: &str| s2c5::Particles {
        particle_name: name.to_owned(),
        x: 0.0,
        y: 64.0,
        z: 0.0,
        offset_x: 0.0,
        offset_y: 0.0,
        offset_z: 0.0,
        speed: 0.0,
        count: 1,
    };
    let mut state = joined();
    let events = apply(&mut state, S2CPlay5::Particles(particles("fireworksSpark")));
    assert!(matches!(
        &events[..],
        [ClientEvent::Particles(p)] if p.particle_id == 3 && p.long_distance
    ));
    let events = apply(&mut state, S2CPlay5::Particles(particles("explode")));
    assert!(matches!(
        &events[..],
        [ClientEvent::Particles(p)] if p.particle_id == 0 && !p.long_distance
    ));
    let events = apply(
        &mut state,
        S2CPlay5::Particles(particles("iconcrack_276_3")),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::Particles(p)] if p.particle_id == 36 && p.extra_data == [276, 3]
    ));
    assert!(apply(&mut state, S2CPlay5::Particles(particles("barrier"))).is_empty());
    assert!(apply_failing(&mut state, S2CPlay5::Particles(particles("blockcrack_1"))).is_empty());
}

/// A 1.8.9-only sound event is unknown to 1.7.10 and does not play; others (a resource pack's
/// own included) go on.
#[test]
fn sounds_only_1_8_9_has_are_dropped() {
    let sound = |name: &str| {
        S2CPlay5::SoundEffect(s2c5::SoundEffect {
            sound_name: name.to_owned(),
            x: 8,
            y: 512,
            z: 8,
            volume: 1.0,
            pitch: 63,
        })
    };
    let mut state = joined();
    assert!(apply(&mut state, sound("mob.rabbit.hop")).is_empty());
    let events = apply(&mut state, sound("random.click"));
    assert!(matches!(
        &events[..],
        [ClientEvent::SoundEffect(s)] if s.sound_name == "random.click" && s.y_fixed == 512
    ));
    assert_eq!(apply(&mut state, sound("custom.pack.sound")).len(), 1);
}

/// 1006 is java18's door-close event, unknown to 1.7.10; 2001's block goes through the 1.7.10
/// block table (stone meta 1 is plain stone in 1.7.10).
#[test]
fn effects_drop_1006_and_decode_2001_blocks() {
    let effect = |effect_id, data| {
        S2CPlay5::Effect(s2c5::Effect {
            effect_id,
            x: 1,
            y: 64,
            z: 1,
            data,
            disable_relative_volume: false,
        })
    };
    let mut state = joined();
    assert!(apply(&mut state, effect(1006, 0)).is_empty());
    let events = apply(&mut state, effect(2001, 1 | 1 << 12));
    assert!(matches!(&events[..], [ClientEvent::Effect(e)] if e.data == 1));
    let events = apply(&mut state, effect(2002, 5));
    assert!(matches!(&events[..], [ClientEvent::Effect(e)] if e.data == 5));
}

/// 1.7.10 sign lines are plain strings; the client stores chat components.
#[test]
fn sign_lines_become_text_components() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::UpdateSign(s2c5::UpdateSign {
            x: 1,
            y: 64,
            z: 1,
            lines: ["a".into(), "\u{a7}cb".into(), String::new(), "{x".into()],
        }),
    );
    let ClientEvent::SignUpdated { lines, .. } = &events[0] else {
        panic!("a sign update");
    };
    for (line, text) in lines.iter().zip(["a", "\u{a7}cb", "", "{x"]) {
        let component = mc_ui::font::component::ChatComponent::from_json(line).expect("json");
        assert_eq!(component, mc_ui::font::component::ChatComponent::text(text));
    }
}

/// java17 `NHPC.handleUpdateTileEntity`: actions 1-5; 1.7.10 has no banner (6).
#[test]
fn tile_entity_action_6_is_ignored() {
    let tile = |action| {
        S2CPlay5::UpdateTileEntity(s2c5::UpdateTileEntity {
            x: 1,
            y: 64,
            z: 1,
            action,
            nbt: None,
        })
    };
    let mut state = joined();
    assert!(apply(&mut state, tile(6)).is_empty());
    assert_eq!(apply(&mut state, tile(4)).len(), 1);
}

/// java17 `NHPC.handleSignEditorOpen` + `EntityPlayerSP.displayGUIEditSign`: the editor opens on a
/// sign or where no tile entity is; another tile entity (a chest) opens nothing (java18 opens the
/// editor anyway).
#[test]
fn sign_editor_opens_only_on_a_sign_or_nothing() {
    let mut state = joined();
    load_column(&mut state, 0, 0);
    let sign = BlockPos::new(1, 4, 1);
    let chest = BlockPos::new(2, 4, 1);
    state
        .world
        .set_block(sign, mc_data::BlockState::from_raw(63 << 4));
    state
        .world
        .set_block(chest, mc_data::BlockState::from_raw(54 << 4 | 2));
    let open = |p: BlockPos| {
        S2CPlay5::SignEditorOpen(s2c5::SignEditorOpen {
            x: p.x,
            y: p.y,
            z: p.z,
        })
    };
    assert!(matches!(
        &apply(&mut state, open(sign))[..],
        [ClientEvent::OpenSignEditor(p)] if *p == sign
    ));
    assert!(apply(&mut state, open(chest)).is_empty());
    assert_eq!(apply(&mut state, open(BlockPos::new(3, 4, 1))).len(), 1);
}

/// java17 `MapData.updateMPMapData`: a column of colours, the icons, or the scale, each alone;
/// a column at `x >= 128` is a row lower; rows past the image throw after what fit.
#[test]
fn map_updates_apply_one_part_at_a_time() {
    let mut state = joined();
    let maps = |data: Vec<u8>| S2CPlay5::Maps(s2c5::Maps { map_id: 3, data });
    apply(&mut state, maps(vec![2, 4]));
    apply(&mut state, maps(vec![1, 0x21, 5, 6]));
    apply(&mut state, maps(vec![0, 5, 10, 11, 12, 13]));
    let map = state.world.map_data(3).expect("map");
    assert_eq!(map.scale, 4, "the colour and icon updates keep the scale");
    assert_eq!(map.decorations.len(), 1);
    assert_eq!(
        (map.decorations[0].icon_type, map.decorations[0].direction),
        (2, 1)
    );
    assert_eq!(
        [
            map.colors[5 + 10 * 128],
            map.colors[5 + 11 * 128],
            map.colors[5 + 12 * 128]
        ],
        [11, 12, 13]
    );
    apply(&mut state, maps(vec![0, 130, 0, 7]));
    assert_eq!(state.world.map_data(3).expect("map").colors[2 + 128], 7);
    apply_failing(&mut state, maps(vec![0, 0, 126, 1, 2, 3]));
    let map = state.world.map_data(3).expect("map");
    assert_eq!([map.colors[126 * 128], map.colors[127 * 128]], [1, 2]);
    apply_failing(&mut state, maps(Vec::new()));
    assert!(state.world.map_data(3).is_some());
}

// ---- windows, lists, scoreboard, channels ---------------------------------------------------

/// java17 `NHPC.handleOpenWindow` (`:877-989`): numbers 0-11; 12 opens nothing; 11 needs a horse.
#[test]
fn open_window_by_number_with_the_horse_check() {
    let open = |inventory_type, entity_id| {
        S2CPlay5::OpenWindow(s2c5::OpenWindow {
            window_id: 3,
            inventory_type,
            title: "Box".to_owned(),
            slot_count: 27,
            use_provided_title: true,
            entity_id,
        })
    };
    let mut state = joined();
    let events = apply(&mut state, open(0, 0));
    assert!(matches!(
        &events[..],
        [ClientEvent::Window(w)]
            if matches!(&**w, WindowEvent::Open(o) if o.inventory_type == "minecraft:chest")
    ));
    assert!(apply(&mut state, open(12, 0)).is_empty());
    apply(&mut state, spawn_mob(90, 100, Vec::new()));
    apply(&mut state, spawn_mob(91, 90, Vec::new()));
    assert_eq!(apply(&mut state, open(11, 90)).len(), 1, "a horse");
    assert!(apply(&mut state, open(11, 91)).is_empty(), "a pig");
    assert!(apply(&mut state, open(11, 999)).is_empty(), "no entity");
}

/// java17 `NHPC.handleStatistics`: numeric 1.7.10 ids resolve to 1.8.9's statistics, unknown
/// ones are dropped.
#[test]
fn statistics_resolve_1_7_10_ids() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::Statistics(s2c5::Statistics {
            entries: vec![
                ("stat.mineBlock.1".to_owned(), 5),
                ("stat.jump".to_owned(), 2),
                ("stat.mineBlock.minecraft.stone".to_owned(), 9),
            ],
        }),
    );
    let ClientEvent::Statistics(entries) = &events[0] else {
        panic!("statistics");
    };
    let ids: Vec<&str> = entries.iter().map(|e| e.stat_id.as_str()).collect();
    assert_eq!(ids, ["stat.mineBlock.minecraft.stone", "stat.jump"]);
}

/// java17 `NHPC.handlePlayerListItem` into the name-keyed list.
#[test]
fn player_list_items_fill_the_name_keyed_list() {
    let mut state = joined();
    let item = |name: &str, online| {
        S2CPlay5::PlayerListItem(s2c5::PlayerListItem {
            name: name.to_owned(),
            online,
            ping: 30,
        })
    };
    apply(&mut state, item("B", true));
    apply(&mut state, item("A", true));
    apply(&mut state, item("B", false));
    let names: Vec<&str> = state.players_v5.iter().map(|p| p.name.as_str()).collect();
    assert_eq!(names, ["A"]);
    assert!(state.players.is_empty(), "the 1.8.9 list is untouched");
}

/// java17 `NHPC.handleUpdateScore` action 1 removes the name from every objective; the slot past
/// 1.7.10's three throws; a team has no colour.
#[test]
fn scoreboard_packets_have_1_7_10_semantics() {
    let mut state = joined();
    let mut scoreboard_events = Vec::new();
    for (name, mode) in [("a", 0), ("b", 0)] {
        scoreboard_events.extend(apply(
            &mut state,
            S2CPlay5::ScoreboardObjective(s2c5::ScoreboardObjective {
                objective_name: name.to_owned(),
                display_name: name.to_owned(),
                mode,
            }),
        ));
        scoreboard_events.extend(apply(
            &mut state,
            S2CPlay5::UpdateScore(s2c5::UpdateScore {
                score_name: "Steve".to_owned(),
                action: 0,
                objective_name: name.to_owned(),
                value: 3,
            }),
        ));
    }
    scoreboard_events.extend(apply(
        &mut state,
        S2CPlay5::UpdateScore(s2c5::UpdateScore {
            score_name: "Steve".to_owned(),
            action: 1,
            objective_name: String::new(),
            value: 0,
        }),
    ));
    for event in &scoreboard_events {
        let ClientEvent::Scoreboard(event) = event else {
            panic!("a scoreboard event");
        };
        state.scoreboard.apply(event);
    }
    assert_eq!(state.scoreboard.score("Steve", "a"), None);
    assert_eq!(state.scoreboard.score("Steve", "b"), None);

    let display = |position| {
        S2CPlay5::DisplayScoreboard(s2c5::DisplayScoreboard {
            position,
            score_name: "a".to_owned(),
        })
    };
    assert!(apply_failing(&mut state, display(3)).is_empty());
    assert_eq!(apply(&mut state, display(1)).len(), 1);

    let events = apply(
        &mut state,
        S2CPlay5::Teams(s2c5::Teams {
            team_name: "t".to_owned(),
            mode: 0,
            display_name: "T".to_owned(),
            prefix: "\u{a7}c".to_owned(),
            suffix: String::new(),
            friendly_flags: 1,
            players: vec!["Steve".to_owned()],
        }),
    );
    let ClientEvent::Scoreboard(event) = &events[0] else {
        panic!("teams");
    };
    let ScoreboardEvent::Teams(teams) = &**event else {
        panic!("teams");
    };
    let s2c::TeamsAction::Create { info, players } = &teams.action else {
        panic!("create");
    };
    assert_eq!(
        (info.color, info.name_tag_visibility.as_str()),
        (-1, "always")
    );
    assert_eq!(players, &["Steve".to_owned()]);
}

/// java17 `NHPC.handleCustomPayload`: the raw brand, `MC|RPack` as a pack with no hash, no
/// `MC|BOpen`, other channels unchanged.
#[test]
fn plugin_channels_have_1_7_10_formats() {
    let message = |channel: &str, data: &[u8]| {
        S2CPlay5::PluginMessage(s2c5::PluginMessage {
            channel: channel.to_owned(),
            data: data.to_vec(),
        })
    };
    let mut state = joined();
    let events = apply(&mut state, message("MC|Brand", b"abc"));
    let ClientEvent::PluginMessage { channel, data } = &events[0] else {
        panic!("brand");
    };
    assert_eq!(channel, "MC|Brand");
    assert_eq!(
        PacketBuffer::from_slice(data)
            .read_string(32_767)
            .expect("a string"),
        "abc"
    );
    let events = apply(&mut state, message("MC|RPack", b"http://x/p.zip"));
    assert!(matches!(
        &events[..],
        [ClientEvent::ResourcePack { url, hash }] if url == "http://x/p.zip" && hash.is_empty()
    ));
    assert!(apply(&mut state, message("MC|BOpen", &[])).is_empty());
    let events = apply(&mut state, message("custom", &[1, 2]));
    assert!(matches!(
        &events[..],
        [ClientEvent::PluginMessage { data, .. }] if data == &[1, 2]
    ));
}

/// `MC|TrList` in 1.7.10's recipe layout reaches the client in java18's: the stacks, the second
/// stack and the disabled flag kept, with the two counters 1.7.10's recipe has (0 of 7).
#[test]
fn trade_lists_are_rewritten_in_the_1_8_9_layout() {
    let mut payload = PacketBuffer::new();
    payload.write_i32(5);
    payload.write_u8(1);
    let emerald = mc_protocol::ItemStack {
        item_id: 388,
        count: 2,
        damage: 0,
        nbt: None,
    };
    payload.write_item_v5(Some(&emerald));
    payload.write_item_v5(Some(&mc_protocol::ItemStack {
        item_id: 1,
        count: 8,
        damage: 0,
        nbt: None,
    }));
    payload.write_bool(false);
    payload.write_bool(true);
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::PluginMessage(s2c5::PluginMessage {
            channel: "MC|TrList".to_owned(),
            data: payload.as_slice().to_vec(),
        }),
    );
    let ClientEvent::PluginMessage { data, .. } = &events[0] else {
        panic!("trade list");
    };
    let mut read = PacketBuffer::from_slice(data);
    assert_eq!(read.read_i32().expect("window"), 5);
    assert_eq!(read.read_u8().expect("count"), 1);
    assert_eq!(read.read_item_stack().expect("buy"), Some(emerald));
    assert_eq!(
        read.read_item_stack().expect("sell").map(|s| s.item_id),
        Some(1)
    );
    assert!(!read.read_bool().expect("second"));
    assert!(read.read_bool().expect("disabled"));
    assert_eq!(read.read_i32().expect("uses"), 0);
    assert_eq!(read.read_i32().expect("max uses"), 7);
    assert_eq!(read.remaining(), 0);
}

/// Every lowered packet without a test of its own reaches the 1.8.9 handler as the same event.
#[test]
fn lowered_packets_reach_the_1_8_9_handlers() {
    let mut state = joined();
    let events = apply(
        &mut state,
        S2CPlay5::EntityVelocity(s2c5::EntityVelocity {
            entity_id: LOCAL_ID,
            motion_x: 8000,
            motion_y: 0,
            motion_z: -4000,
        }),
    );
    assert!(events.is_empty());
    assert_eq!(
        state.local.pending_velocity,
        Some(DVec3::new(1.0, 0.0, -0.5))
    );
    let events = apply(
        &mut state,
        S2CPlay5::SetSlot(s2c5::SetSlot {
            window_id: 0,
            slot: 36,
            item: None,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::Window(w)] if matches!(**w, WindowEvent::SetSlot { slot: 36, .. })
    ));
    let events = apply(
        &mut state,
        S2CPlay5::ConfirmTransaction(s2c5::ConfirmTransaction {
            window_id: 0,
            action_number: 4,
            accepted: false,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::Window(w)]
            if matches!(**w, WindowEvent::ConfirmTransaction { action_number: 4, accepted: false, .. })
    ));
    let events = apply(
        &mut state,
        S2CPlay5::BlockBreakAnimation(s2c5::BlockBreakAnimation {
            breaker_id: 9,
            x: 1,
            y: 2,
            z: 3,
            progress: 255,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::BlockBreakProgress { progress: 255, .. }]
    ));
    let events = apply(
        &mut state,
        S2CPlay5::SetExperience(s2c5::SetExperience {
            experience_bar: 0.5,
            level: 30,
            total_experience: 1395,
        }),
    );
    assert!(matches!(
        &events[..],
        [ClientEvent::ExperienceChanged {
            level: 30,
            total: 1395,
            ..
        }]
    ));
    assert!(matches!(
        apply(
            &mut state,
            S2CPlay5::TabComplete(s2c5::TabComplete {
                matches: vec!["/help".to_owned()],
            }),
        )
        .as_slice(),
        [ClientEvent::TabComplete(m)] if m == &["/help".to_owned()]
    ));
}
