//! Every one of the 74 clientbound Play packets, fed through the real dispatch.
//!
//! # What this proves, and what it does not
//!
//! [`mc_client::sync::apply`] matches `S2CPlay` exhaustively, so the *compiler* already
//! guarantees that no variant is missing. What the compiler cannot guarantee is that each arm
//! is reachable with a well-formed packet and does something other than panic. This file
//! constructs one concrete instance of all 74 variants — by hand, from `mc-protocol`'s field
//! definitions, not by reflection — and drives each through `apply`, asserting:
//!
//! * every wire id `0x00..=0x49` is represented exactly once, so the list really is complete;
//! * no packet panics, including on an empty world with no chunks and no entities;
//! * the only one that returns an error is `0x40 Disconnect`, which is a control-flow signal
//!   rather than a failure.
//!
//! It does **not** prove each arm is semantically right — that is what the per-packet unit
//! tests in `sync::dispatch` and `sync::entities` are for.

use mc_client::sync::{apply, ClientEvent, ClientState};
use mc_client::ClientError;
use mc_math::{BlockPos, Facing};
use mc_nbt::NbtCompound;
use mc_protocol::chunk::{ChunkData, MapChunkBulk};
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::{Angle, MetadataEntry, MetadataValue};
use uuid::Uuid;

fn pos() -> BlockPos {
    BlockPos::new(1, 2, 3)
}

/// One valid instance of every clientbound Play packet, in `S2CPlay` declaration order.
#[allow(clippy::too_many_lines)]
fn one_of_each() -> Vec<S2CPlay> {
    vec![
        S2CPlay::KeepAlive(s2c::KeepAlive { id: 1 }),
        S2CPlay::JoinGame(s2c::JoinGame {
            entity_id: 1,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 20,
            level_type: "default".into(),
            reduced_debug_info: false,
        }),
        S2CPlay::Chat(s2c::Chat {
            component: r#"{"text":"hi"}"#.into(),
            position: 0,
        }),
        S2CPlay::TimeUpdate(s2c::TimeUpdate {
            total_world_time: 100,
            world_time: 6000,
        }),
        S2CPlay::EntityEquipment(s2c::EntityEquipment {
            entity_id: 1,
            equipment_slot: 0,
            item: None,
        }),
        S2CPlay::SpawnPosition(s2c::SpawnPosition { position: pos() }),
        S2CPlay::UpdateHealth(s2c::UpdateHealth {
            health: 20.0,
            food_level: 20,
            saturation_level: 5.0,
        }),
        S2CPlay::Respawn(s2c::Respawn {
            dimension: 0,
            difficulty: 1,
            game_mode: 0,
            level_type: "default".into(),
        }),
        S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
            x: 0.0,
            y: 64.0,
            z: 0.0,
            yaw: 0.0,
            pitch: 0.0,
            flags: 0,
        }),
        S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: 0 }),
        S2CPlay::UseBed(s2c::UseBed {
            player_id: 1,
            bed_position: pos(),
        }),
        S2CPlay::Animation(s2c::Animation {
            entity_id: 1,
            animation_type: 0,
        }),
        S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
            entity_id: 10,
            player_uuid: Uuid::from_u128(1),
            x: 0,
            y: 0,
            z: 0,
            yaw: Angle(0),
            pitch: Angle(0),
            current_item: 0,
            metadata: vec![MetadataEntry {
                index: 0,
                value: MetadataValue::Byte(0),
            }],
        }),
        S2CPlay::CollectItem(s2c::CollectItem {
            collected_entity_id: 11,
            collector_entity_id: 10,
        }),
        S2CPlay::SpawnObject(s2c::SpawnObject {
            entity_id: 12,
            object_type: 2,
            x: 0,
            y: 0,
            z: 0,
            pitch: Angle(0),
            yaw: Angle(0),
            data: 0,
            velocity: None,
        }),
        S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: 13,
            entity_type: 54,
            x: 0,
            y: 0,
            z: 0,
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: Vec::new(),
        }),
        S2CPlay::SpawnPainting(s2c::SpawnPainting {
            entity_id: 14,
            title: "Kebab".into(),
            position: pos(),
            facing: Facing::North,
        }),
        S2CPlay::SpawnExperienceOrb(s2c::SpawnExperienceOrb {
            entity_id: 15,
            x: 0,
            y: 0,
            z: 0,
            xp_value: 3,
        }),
        S2CPlay::EntityVelocity(s2c::EntityVelocity {
            entity_id: 13,
            motion_x: 0,
            motion_y: 0,
            motion_z: 0,
        }),
        S2CPlay::DestroyEntities(s2c::DestroyEntities {
            entity_ids: vec![15],
        }),
        S2CPlay::Entity(s2c::Entity { entity_id: 13 }),
        S2CPlay::EntityRelMove(s2c::EntityRelMove {
            entity_id: 13,
            delta_x: 1,
            delta_y: 0,
            delta_z: -1,
            on_ground: true,
        }),
        S2CPlay::EntityLook(s2c::EntityLook {
            entity_id: 13,
            yaw: Angle(64),
            pitch: Angle(0),
            on_ground: true,
        }),
        S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
            entity_id: 13,
            delta_x: 1,
            delta_y: 1,
            delta_z: 1,
            yaw: Angle(32),
            pitch: Angle(8),
            on_ground: false,
        }),
        S2CPlay::EntityTeleport(s2c::EntityTeleport {
            entity_id: 13,
            x: 320,
            y: 2048,
            z: -320,
            yaw: Angle(0),
            pitch: Angle(0),
            on_ground: true,
        }),
        S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
            entity_id: 13,
            yaw: Angle(128),
        }),
        S2CPlay::EntityStatus(s2c::EntityStatus {
            entity_id: 13,
            opcode: 2,
        }),
        S2CPlay::EntityAttach(s2c::EntityAttach {
            entity_id: 13,
            vehicle_entity_id: 12,
            attach_type: s2c::AttachType::Vehicle,
        }),
        S2CPlay::EntityMetadata(s2c::EntityMetadata {
            entity_id: 13,
            metadata: vec![MetadataEntry {
                index: 6,
                value: MetadataValue::Float(20.0),
            }],
        }),
        S2CPlay::EntityEffect(s2c::EntityEffect {
            entity_id: 13,
            effect_id: 1,
            amplifier: 0,
            duration: 200,
            show_particles: false,
        }),
        S2CPlay::RemoveEntityEffect(s2c::RemoveEntityEffect {
            entity_id: 13,
            effect_id: 1,
        }),
        S2CPlay::SetExperience(s2c::SetExperience {
            experience_bar: 0.5,
            level: 3,
            total_experience: 40,
        }),
        S2CPlay::EntityProperties(s2c::EntityProperties {
            entity_id: 13,
            properties: vec![s2c::AttributePropertySnapshot {
                key: "generic.movementSpeed".into(),
                base_value: 0.1,
                modifiers: vec![s2c::AttributeModifierSnapshot {
                    uuid: Uuid::from_u128(7),
                    amount: 0.3,
                    operation: s2c::AttributeModifierOperation::Add,
                }],
            }],
        }),
        S2CPlay::ChunkData(ChunkData {
            chunk_x: 0,
            chunk_z: 0,
            ground_up_continuous: true,
            primary_bit_mask: 0b1,
            data: vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)],
        }),
        S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
            chunk_x: 0,
            chunk_z: 0,
            changes: vec![s2c::MultiBlockChangeEntry {
                rel_x: 1,
                y: 2,
                rel_z: 3,
                block_state_id: 16,
            }],
        }),
        S2CPlay::BlockChange(s2c::BlockChange {
            position: pos(),
            block_state_id: 16,
        }),
        S2CPlay::BlockAction(s2c::BlockAction {
            position: pos(),
            data1: 1,
            data2: 0,
            block_id: 33,
        }),
        S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
            breaker_id: 10,
            position: pos(),
            progress: 4,
        }),
        S2CPlay::MapChunkBulk(MapChunkBulk {
            has_sky_light: true,
            columns: vec![],
        }),
        S2CPlay::Explosion(s2c::Explosion {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            strength: 4.0,
            affected_blocks: vec![s2c::ExplosionAffectedBlock {
                dx: 1,
                dy: 2,
                dz: 3,
            }],
            motion_x: 0.0,
            motion_y: 0.0,
            motion_z: 0.0,
        }),
        S2CPlay::Effect(s2c::Effect {
            effect_id: 1003,
            position: pos(),
            data: 0,
            disable_relative_volume: false,
        }),
        S2CPlay::SoundEffect(s2c::SoundEffect {
            sound_name: "random.pop".into(),
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
            volume: 1.0,
            pitch_quantized: 63,
        }),
        S2CPlay::Particles(s2c::Particles {
            particle_id: 0,
            long_distance: false,
            x: 0.0,
            y: 0.0,
            z: 0.0,
            offset_x: 0.0,
            offset_y: 0.0,
            offset_z: 0.0,
            particle_speed: 0.0,
            particle_count: 1,
            extra_data: vec![],
        }),
        S2CPlay::ChangeGameState(s2c::ChangeGameState {
            reason: s2c::GameStateReason::BeginRaining,
            value: 0.0,
        }),
        S2CPlay::SpawnGlobalEntity(s2c::SpawnGlobalEntity {
            entity_id: 20,
            entity_type: 1,
            x_fixed: 0,
            y_fixed: 0,
            z_fixed: 0,
        }),
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id: 1,
            inventory_type: "minecraft:chest".into(),
            window_title: r#"{"text":"Chest"}"#.into(),
            slot_count: 27,
            entity_id: None,
        }),
        S2CPlay::CloseWindow(s2c::CloseWindow { window_id: 1 }),
        S2CPlay::SetSlot(s2c::SetSlot {
            window_id: 0,
            slot: 36,
            item: None,
        }),
        S2CPlay::WindowItems(s2c::WindowItems {
            window_id: 0,
            items: vec![None, None],
        }),
        S2CPlay::WindowProperty(s2c::WindowProperty {
            window_id: 1,
            var_index: 0,
            var_value: 100,
        }),
        S2CPlay::ConfirmTransaction(s2c::ConfirmTransaction {
            window_id: 1,
            action_number: 1,
            accepted: true,
        }),
        S2CPlay::UpdateSign(s2c::UpdateSign {
            position: pos(),
            lines: [
                r#"{"text":"a"}"#.into(),
                r#"{"text":"b"}"#.into(),
                r#"{"text":"c"}"#.into(),
                r#"{"text":"d"}"#.into(),
            ],
        }),
        S2CPlay::Maps(s2c::Maps {
            map_id: 1,
            scale: 0,
            icons: vec![s2c::MapIcon {
                icon_type: 0,
                x: 0,
                z: 0,
                direction: 0,
            }],
            update: None,
        }),
        S2CPlay::UpdateTileEntity(s2c::UpdateTileEntity {
            position: pos(),
            action: 1,
            nbt: Some(NbtCompound::new()),
        }),
        S2CPlay::SignEditorOpen(s2c::SignEditorOpen { position: pos() }),
        S2CPlay::Statistics(s2c::Statistics {
            entries: vec![s2c::StatisticEntry {
                stat_id: "stat.jump".into(),
                value: 7,
            }],
        }),
        S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
            uuid: Uuid::from_u128(1),
            username: "Notch".into(),
            properties: vec![],
            game_mode: s2c::GameMode::Survival,
            ping: 20,
            display_name: None,
        }])),
        S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
            flags: 0x04,
            fly_speed: 0.05,
            walk_speed: 0.1,
        }),
        S2CPlay::TabComplete(s2c::TabComplete {
            matches: vec!["/help".into()],
        }),
        S2CPlay::ScoreboardObjective(s2c::ScoreboardObjective {
            objective_name: "obj".into(),
            mode: s2c::ScoreboardObjectiveMode::Create {
                display_name: "Objective".into(),
                render_type: s2c::ScoreboardRenderType::Integer,
            },
        }),
        S2CPlay::UpdateScore(s2c::UpdateScore {
            player_name: "Notch".into(),
            objective_name: "obj".into(),
            action: s2c::UpdateScoreAction::Change { value: 5 },
        }),
        S2CPlay::DisplayScoreboard(s2c::DisplayScoreboard {
            position: 1,
            objective_name: "obj".into(),
        }),
        S2CPlay::Teams(s2c::Teams {
            team_name: "red".into(),
            action: s2c::TeamsAction::Remove,
        }),
        S2CPlay::PluginMessage(s2c::PluginMessage {
            channel: "MC|Brand".into(),
            data: vec![7, b'v', b'a', b'n', b'i', b'l', b'l', b'a'],
        }),
        S2CPlay::Disconnect(s2c::Disconnect {
            reason: r#"{"text":"bye"}"#.into(),
        }),
        S2CPlay::ServerDifficulty(s2c::ServerDifficulty {
            difficulty: s2c::ServerDifficultyLevel::Hard,
        }),
        S2CPlay::CombatEvent(s2c::CombatEvent::EnterCombat),
        S2CPlay::Camera(s2c::Camera { entity_id: 13 }),
        S2CPlay::WorldBorder(s2c::WorldBorder::SetSize { diameter: 128.0 }),
        S2CPlay::Title(s2c::Title::Clear),
        S2CPlay::SetCompression(s2c::SetCompression { threshold: 256 }),
        S2CPlay::PlayerListHeaderFooter(s2c::PlayerListHeaderFooter {
            header: r#"{"text":"head"}"#.into(),
            footer: r#"{"text":""}"#.into(),
        }),
        S2CPlay::ResourcePackSend(s2c::ResourcePackSend {
            url: "http://example.invalid/pack.zip".into(),
            hash: "0".repeat(40),
        }),
        S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
            entity_id: 13,
            tag: Some(NbtCompound::new()),
        }),
    ]
}

#[test]
fn the_sample_set_covers_every_wire_id_exactly_once() {
    let mut ids: Vec<i32> = one_of_each().iter().map(S2CPlay::id).collect();
    assert_eq!(ids.len(), 74, "S06 registers 74 clientbound Play packets");
    ids.sort_unstable();
    assert_eq!(
        ids,
        (0x00..=0x49).collect::<Vec<i32>>(),
        "the sample set must span 0x00..=0x49 with no gaps and no duplicates"
    );
}

#[test]
fn every_packet_applies_without_panicking() {
    for packet in one_of_each() {
        let mut state = ClientState::new();
        let mut events = Vec::new();
        let id = packet.id();
        let is_disconnect = matches!(packet, S2CPlay::Disconnect(_));
        match apply(&mut state, packet, &mut events) {
            Ok(()) => assert!(
                !is_disconnect,
                "Disconnect must surface as an error, not silently succeed"
            ),
            Err(ClientError::Disconnected { .. }) => assert!(
                is_disconnect,
                "only 0x40 may report Disconnected, id {id:#04x} did"
            ),
            Err(err) => panic!("packet id {id:#04x} failed to apply: {err}"),
        }
    }
}

/// The same set again, but applied to one accumulating state in order, so that later packets
/// see the entities and chunks earlier ones created. This is closer to a real session and
/// catches arms that only work against a blank world.
#[test]
fn the_whole_set_applies_in_sequence_against_one_accumulating_state() {
    let mut state = ClientState::new();
    let mut events = Vec::new();
    for packet in one_of_each() {
        if matches!(packet, S2CPlay::Disconnect(_)) {
            continue;
        }
        let id = packet.id();
        apply(&mut state, packet, &mut events)
            .unwrap_or_else(|err| panic!("packet id {id:#04x} failed in sequence: {err}"));
    }

    // A few things the sequence must have left behind.
    assert!(state.world.is_chunk_loaded(mc_math::ChunkPos::new(0, 0)));
    assert!(state.entities.get(13).is_some(), "the zombie is tracked");
    assert!(state.entities.get(10).is_some(), "the player is tracked");
    assert!(
        state.entities.get(15).is_none(),
        "the XP orb was destroyed by DestroyEntities"
    );
    assert_eq!(state.players.len(), 1);
    assert_eq!(state.env.total_world_time, 100);
    assert_eq!(
        state.env.difficulty,
        mc_protocol::play::s2c::ServerDifficultyLevel::Hard
    );
    assert!(state.local.abilities.allow_flying);
    assert!(events.iter().any(|e| matches!(e, ClientEvent::Chat { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e, ClientEvent::LightningBolt { .. })));
}
