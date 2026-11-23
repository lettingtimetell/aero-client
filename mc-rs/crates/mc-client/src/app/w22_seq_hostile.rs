//! Hostile packet *sequences* through the live client.
//!
//! The decoders meet damaged bytes elsewhere (`tests/z1_hostile_server.rs`) and random values in
//! the live fuzzer (`random_wire_reachable_packets_never_panic_the_live_client`). This module
//! sends *valid* clientbound packets in the orders and with the sizes a broken or malicious
//! server can choose, through the path the game loop takes: `Session::poll_next` ->
//! `sync::apply` -> `Session::react_to` -> `Client::react`, with client ticks in between.
//!
//! Each scenario is bounded (fixed sizes, no randomness, seconds in the debug build) and checks
//! one of:
//! - nothing panics and nothing hangs (a watchdog aborts the binary with the scenario's name);
//! - a packet costs time linear in its size, as its vanilla handler does;
//! - client-side state returns to its size once the server has removed what it added;
//! - the client answers exactly what vanilla 1.8.9's handlers answer.
//!
//! A child of `driver` (like the movement and interaction oracles) so it can read `Client`'s
//! private per-entity maps for the state census.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::*;
use crate::net::channel::NetHandle;
use crate::net::{ToGame, ToNet};
use mc_protocol::play::{c2s, s2c, C2SPlay, S2CPlay};
use mc_protocol::{Angle, MetadataEntry, MetadataValue};
use uuid::Uuid;

/// The local player's entity id in every scenario.
const LOCAL: i32 = 4_000;

/// Stone, as a 1.8 block state id (`id << 4 | meta`).
const STONE_STATE: u16 = 1 << 4;

// ---- harness -----------------------------------------------------------------------------

/// Aborts the test binary when a scenario outlives its limit, naming it: a hang would
/// otherwise stall the whole suite with no output at all.
struct Watchdog {
    done: Arc<AtomicBool>,
}

impl Watchdog {
    fn arm(name: &'static str, limit: Duration) -> Self {
        let done = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&done);
        let start = Instant::now();
        std::thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                if start.elapsed() > limit {
                    eprintln!("SEQ watchdog: '{name}' still running after {limit:?}: a hang");
                    std::process::abort();
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        });
        Self { done }
    }
}

impl Drop for Watchdog {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Relaxed);
    }
}

/// A client in game on an offline session: the test plays the server's side of the channel.
struct Rig {
    client: Client,
    to_game: tokio::sync::mpsc::Sender<ToGame>,
    to_net: tokio::sync::mpsc::UnboundedReceiver<ToNet>,
    _dir: tempfile::TempDir,
}

impl Rig {
    /// JoinGame, a stone floor column under the spawn and the first S08 (which closes the
    /// terrain screen and is answered with a C06), with every reply drained.
    fn joined() -> Option<Self> {
        let dir = tempfile::tempdir().expect("tempdir");
        let config = ClientConfig {
            game_dir: dir.path().to_path_buf(),
            login: LoginMode::Offline("Dev".to_owned()),
            ..ClientConfig::default()
        }
        .with_discovered_assets();
        if !config
            .version_jar
            .as_ref()
            .is_some_and(|path| path.is_file())
        {
            eprintln!("skip: no vanilla jar in this checkout");
            return None;
        }
        let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
        let (handle, to_game, to_net) = NetHandle::new_pair();
        let mut rig = Self {
            client,
            to_game,
            to_net,
            _dir: dir,
        };
        rig.client.session = Some(Session::new(handle, JoinSettings::default()));
        rig.client.ui = Ui::InGame;
        rig.apply([join(0), floor_column(0, 0), teleport(8.5, 16.0, 8.5)]);
        rig.ticks(2);
        rig.sent();
        Some(rig)
    }

    /// Queue one packet behind the others, letting the game loop drain the bounded channel
    /// when it is full, as the network task's spill queue does.
    fn queue(&mut self, packet: S2CPlay) {
        let mut message = ToGame::Packet(packet);
        loop {
            match self.to_game.try_send(message) {
                Ok(()) => return,
                Err(tokio::sync::mpsc::error::TrySendError::Full(back)) => {
                    message = back;
                    self.client.poll_in_place();
                }
                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                    panic!("the session dropped its inbound channel")
                }
            }
        }
    }

    /// Queue `packets` and run one game-loop poll over them.
    fn apply(&mut self, packets: impl IntoIterator<Item = S2CPlay>) {
        for packet in packets {
            self.queue(packet);
        }
        self.client.poll_in_place();
    }

    /// `count` game-loop iterations of one tick each.
    fn ticks(&mut self, count: usize) {
        for _ in 0..count {
            self.client.poll_in_place();
            self.client.tick(&MoveInput::default());
        }
    }

    /// Everything the client put on the wire since the last call, in order.
    fn sent(&mut self) -> Vec<C2SPlay> {
        let mut out = Vec::new();
        while let Ok(message) = self.to_net.try_recv() {
            if let ToNet::Send(packet) = message {
                out.push(packet);
            }
        }
        out
    }

    fn state(&self) -> &crate::sync::ClientState {
        self.client
            .session
            .as_ref()
            .expect("a live session")
            .state()
    }
}

/// Every client-side collection that holds per-entity, per-player, per-position or per-name
/// state, by name. A server-driven cycle that adds and then removes something must leave each
/// of these where it started.
fn census(rig: &Rig) -> Vec<(&'static str, usize)> {
    let client = &rig.client;
    let state = rig.state();
    vec![
        ("tracked entities", state.entities.len()),
        ("parked spawns", state.entities.pending_spawn_count()),
        ("waiting leashes", state.entities.waiting_leash_count()),
        ("tab list", state.players.len()),
        ("loaded chunks", state.world.loaded_chunk_count()),
        ("objectives", state.scoreboard.objective_count()),
        ("teams", state.scoreboard.teams().count()),
        ("wither heads", client.wither_heads.len()),
        ("wither max health", client.wither_max_health.len()),
        ("max health attributes", client.max_health_attributes.len()),
        ("damaged blocks", client.damaged_blocks.len()),
        (
            "furnace minecart predictions",
            client.furnace_minecart_predictions.len(),
        ),
        (
            "command minecart outputs",
            client.command_minecart_track_output.len(),
        ),
        ("record sounds", client.record_sounds.len()),
        ("guardian sounds", client.guardian_sounds.len()),
        ("minecart sounds", client.minecart_sounds.len()),
        ("pickup effects", client.pickup_effects.len()),
        ("pending local drops", client.pending_local_item_drops.len()),
        (
            "inventory preview uploads",
            client.inventory_preview_uploads.len(),
        ),
        ("tab skin uploads", client.tab_skin_uploads.len()),
    ]
}

/// The census entries that differ between two snapshots.
fn census_growth(before: &[(&'static str, usize)], after: &[(&'static str, usize)]) -> Vec<String> {
    before
        .iter()
        .zip(after)
        .filter(|((_, a), (_, b))| a != b)
        .map(|((name, a), (_, b))| format!("{name}: {a} -> {b}"))
        .collect()
}

/// Whether a server can deliver `packet`: it survives the real encoder and decoder whole.
fn deliverable(packet: &S2CPlay) -> bool {
    crate::sync::fuzz::wire_reachable(packet.clone()).is_some()
}

// ---- packets -------------------------------------------------------------------------------

fn join(dimension: i8) -> S2CPlay {
    S2CPlay::JoinGame(s2c::JoinGame {
        entity_id: LOCAL,
        hardcore_mode: false,
        game_mode: 0,
        dimension,
        difficulty: 2,
        max_players: 20,
        level_type: "default".into(),
        reduced_debug_info: false,
    })
}

fn respawn(dimension: i32) -> S2CPlay {
    S2CPlay::Respawn(s2c::Respawn {
        dimension,
        difficulty: 2,
        game_mode: 0,
        level_type: "default".into(),
    })
}

fn teleport(x: f64, y: f64, z: f64) -> S2CPlay {
    S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
        x,
        y,
        z,
        yaw: 0.0,
        pitch: 0.0,
        flags: 0,
    })
}

/// A full column whose lowest section is solid stone: the player stands on y = 16.
fn floor_column(chunk_x: i32, chunk_z: i32) -> S2CPlay {
    let mut data = vec![0u8; mc_world::expected_chunk_data_len(0b1, true, true)];
    for block in data.chunks_exact_mut(2).take(16 * 16 * 16) {
        block.copy_from_slice(&STONE_STATE.to_le_bytes());
    }
    S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
        chunk_x,
        chunk_z,
        ground_up_continuous: true,
        primary_bit_mask: 0b1,
        data,
    })
}

/// java: `S21PacketChunkData` with `groundUp` and an empty bitmask is `doPreChunk(x, z, false)`.
fn unload_column(chunk_x: i32, chunk_z: i32) -> S2CPlay {
    S2CPlay::ChunkData(mc_protocol::chunk::ChunkData {
        chunk_x,
        chunk_z,
        ground_up_continuous: true,
        primary_bit_mask: 0,
        data: vec![0u8; mc_world::expected_chunk_data_len(0, true, true)],
    })
}

fn zombie(entity_id: i32, x: i32, z: i32) -> S2CPlay {
    S2CPlay::SpawnMob(s2c::SpawnMob {
        entity_id,
        entity_type: 54,
        x,
        y: 16 * 32,
        z,
        yaw: Angle(0),
        pitch: Angle(0),
        head_pitch: Angle(0),
        velocity_x: 0,
        velocity_y: 0,
        velocity_z: 0,
        metadata: Vec::new(),
    })
}

fn minecart(entity_id: i32, data: i32) -> S2CPlay {
    S2CPlay::SpawnObject(s2c::SpawnObject {
        entity_id,
        object_type: 10,
        x: 8 * 32,
        y: 16 * 32,
        z: 8 * 32,
        pitch: Angle(0),
        yaw: Angle(0),
        data,
        velocity: Some((0, 0, 0)),
    })
}

fn destroy(entity_ids: Vec<i32>) -> S2CPlay {
    S2CPlay::DestroyEntities(s2c::DestroyEntities { entity_ids })
}

fn attach(entity_id: i32, vehicle_entity_id: i32, attach_type: s2c::AttachType) -> S2CPlay {
    S2CPlay::EntityAttach(s2c::EntityAttach {
        entity_id,
        vehicle_entity_id,
        attach_type,
    })
}

fn team_info() -> s2c::TeamInfo {
    s2c::TeamInfo {
        display_name: "t".into(),
        prefix: String::new(),
        suffix: String::new(),
        friendly_flags: 0,
        name_tag_visibility: "always".into(),
        color: -1,
    }
}

fn teams(team_name: &str, action: s2c::TeamsAction) -> S2CPlay {
    S2CPlay::Teams(s2c::Teams {
        team_name: team_name.into(),
        action,
    })
}

fn objective(name: &str, mode: s2c::ScoreboardObjectiveMode) -> S2CPlay {
    S2CPlay::ScoreboardObjective(s2c::ScoreboardObjective {
        objective_name: name.into(),
        mode,
    })
}

fn create_objective(name: &str) -> S2CPlay {
    objective(
        name,
        s2c::ScoreboardObjectiveMode::Create {
            display_name: name.into(),
            render_type: s2c::ScoreboardRenderType::Integer,
        },
    )
}

fn score(player: &str, objective_name: &str, action: s2c::UpdateScoreAction) -> S2CPlay {
    S2CPlay::UpdateScore(s2c::UpdateScore {
        player_name: player.into(),
        objective_name: objective_name.into(),
        action,
    })
}

/// Every packet vanilla resolves through `getEntityByID`, aimed at `id` (and `other` where a
/// packet names two entities).
fn entity_targeted(id: i32, other: i32) -> Vec<S2CPlay> {
    let mut packets = vec![
        S2CPlay::EntityEquipment(s2c::EntityEquipment {
            entity_id: id,
            equipment_slot: 0,
            item: None,
        }),
        S2CPlay::UseBed(s2c::UseBed {
            player_id: id,
            bed_position: BlockPos::new(8, 16, 8),
        }),
        S2CPlay::CollectItem(s2c::CollectItem {
            collected_entity_id: id,
            collector_entity_id: other,
        }),
        S2CPlay::EntityVelocity(s2c::EntityVelocity {
            entity_id: id,
            motion_x: 8000,
            motion_y: 8000,
            motion_z: -8000,
        }),
        S2CPlay::Entity(s2c::Entity { entity_id: id }),
        S2CPlay::EntityRelMove(s2c::EntityRelMove {
            entity_id: id,
            delta_x: 1,
            delta_y: 0,
            delta_z: -1,
            on_ground: true,
        }),
        S2CPlay::EntityLook(s2c::EntityLook {
            entity_id: id,
            yaw: Angle(64),
            pitch: Angle(0),
            on_ground: true,
        }),
        S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
            entity_id: id,
            delta_x: 1,
            delta_y: 1,
            delta_z: 1,
            yaw: Angle(32),
            pitch: Angle(8),
            on_ground: false,
        }),
        S2CPlay::EntityTeleport(s2c::EntityTeleport {
            entity_id: id,
            x: 320,
            y: 2048,
            z: -320,
            yaw: Angle(0),
            pitch: Angle(0),
            on_ground: true,
        }),
        S2CPlay::EntityHeadLook(s2c::EntityHeadLook {
            entity_id: id,
            yaw: Angle(128),
        }),
        attach(id, other, s2c::AttachType::Vehicle),
        attach(id, other, s2c::AttachType::Leash),
        S2CPlay::EntityMetadata(s2c::EntityMetadata {
            entity_id: id,
            metadata: vec![
                MetadataEntry {
                    index: 0,
                    value: MetadataValue::Byte(0x20),
                },
                MetadataEntry {
                    index: 2,
                    value: MetadataValue::String("name".into()),
                },
                MetadataEntry {
                    index: 6,
                    value: MetadataValue::Float(1.0),
                },
                MetadataEntry {
                    index: 17,
                    value: MetadataValue::Int(other),
                },
            ],
        }),
        S2CPlay::EntityEffect(s2c::EntityEffect {
            entity_id: id,
            effect_id: 1,
            amplifier: 0,
            duration: 200,
            show_particles: true,
        }),
        S2CPlay::RemoveEntityEffect(s2c::RemoveEntityEffect {
            entity_id: id,
            effect_id: 1,
        }),
        S2CPlay::EntityProperties(s2c::EntityProperties {
            entity_id: id,
            properties: vec![s2c::AttributePropertySnapshot {
                key: "generic.maxHealth".into(),
                base_value: 300.0,
                modifiers: vec![],
            }],
        }),
        S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
            breaker_id: id,
            position: BlockPos::new(8, 15, 8),
            progress: 5,
        }),
        // Progress 10 is vanilla's "remove" for the same breaker.
        S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
            breaker_id: id,
            position: BlockPos::new(8, 15, 8),
            progress: 10,
        }),
        S2CPlay::Camera(s2c::Camera { entity_id: id }),
        S2CPlay::Camera(s2c::Camera { entity_id: LOCAL }),
        S2CPlay::UpdateEntityNBT(s2c::UpdateEntityNBT {
            entity_id: id,
            tag: Some(mc_nbt::NbtCompound::new()),
        }),
        S2CPlay::CombatEvent(s2c::CombatEvent::EndCombat {
            duration_ticks: 5,
            entity_id: id,
        }),
    ];
    for opcode in [
        2, 3, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 21, 22, 23,
    ] {
        packets.push(S2CPlay::EntityStatus(s2c::EntityStatus {
            entity_id: id,
            opcode,
        }));
    }
    for animation_type in 0..=5 {
        packets.push(S2CPlay::Animation(s2c::Animation {
            entity_id: id,
            animation_type,
        }));
    }
    packets
}

// ---- scenarios: work linear in the input --------------------------------------------------

/// Names a server can put in one S3E or S3C: up to 40 characters each (vanilla's
/// `readStringFromBuffer(40)`), here short and distinct.
fn names(count: usize) -> Vec<String> {
    (0..count).map(|index| format!("p{index}")).collect()
}

/// java: `ScorePlayerTeam.membershipSet` is a `HashSet`, so adding,
/// moving and removing each listed player costs constant time and one S3E costs time linear in
/// its list. A list scan per player made one packet quadratic: a server could freeze the game
/// thread with a few hundred kilobytes.
#[test]
fn a_team_roster_packet_costs_time_linear_in_its_length() {
    const PLAYERS: usize = 30_000;
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("team roster", Duration::from_secs(300));
    let players = names(PLAYERS);
    let create = teams(
        "a",
        s2c::TeamsAction::Create {
            info: team_info(),
            players: players.clone(),
        },
    );
    assert!(deliverable(&create), "one S3E carries the whole roster");
    let start = Instant::now();
    rig.apply([
        create,
        teams(
            "b",
            s2c::TeamsAction::Create {
                info: team_info(),
                players: Vec::new(),
            },
        ),
        // Every player moves from "a" to "b": `addPlayerToTeam` removes them from "a" first.
        teams(
            "b",
            s2c::TeamsAction::AddPlayers {
                players: players.clone(),
            },
        ),
    ]);
    let moved = start.elapsed();
    {
        let board = &rig.state().scoreboard;
        assert!(board.team("a").expect("team a").members.is_empty());
        assert_eq!(board.team("b").expect("team b").members.len(), PLAYERS);
        assert_eq!(
            board.player_team("p0").map(|team| team.name.as_str()),
            Some("b")
        );
    }
    let start = Instant::now();
    rig.apply([teams("b", s2c::TeamsAction::RemovePlayers { players })]);
    let removed = start.elapsed();
    let board = &rig.state().scoreboard;
    assert!(board.team("b").expect("team b").members.is_empty());
    assert!(board.player_team("p0").is_none());
    eprintln!("SEQ team roster of {PLAYERS}: create+move {moved:?}, remove {removed:?}");
    assert!(
        moved + removed < Duration::from_secs(5),
        "a {PLAYERS}-player roster took {moved:?} to create and move and {removed:?} to remove"
    );
}

/// java: `Scoreboard.entitiesScoreObjectives` is a map of maps, so an
/// S3C REMOVE touches one player's row. A scan over every score per REMOVE made a server's
/// churn of N rows cost N^2.
#[test]
fn score_removals_cost_time_linear_in_their_count() {
    const ROWS: usize = 30_000;
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("score rows", Duration::from_secs(300));
    let players = names(ROWS);
    rig.apply([
        create_objective("obj"),
        create_objective("other"),
        S2CPlay::DisplayScoreboard(s2c::DisplayScoreboard {
            position: 1,
            objective_name: "obj".into(),
        }),
    ]);
    let start = Instant::now();
    for (index, player) in players.iter().enumerate() {
        rig.queue(score(
            player,
            "obj",
            s2c::UpdateScoreAction::Change {
                value: index as i32,
            },
        ));
    }
    rig.client.poll_in_place();
    let added = start.elapsed();
    assert_eq!(rig.state().scoreboard.sorted_scores("obj").len(), ROWS);
    let start = Instant::now();
    // Half go one objective at a time, half all at once (an empty objective name).
    for (index, player) in players.iter().enumerate() {
        let objective_name = if index.is_multiple_of(2) { "obj" } else { "" };
        rig.queue(score(
            player,
            objective_name,
            s2c::UpdateScoreAction::Remove,
        ));
    }
    rig.client.poll_in_place();
    let removed = start.elapsed();
    assert!(rig.state().scoreboard.sorted_scores("obj").is_empty());
    eprintln!("SEQ {ROWS} score rows: add {added:?}, remove {removed:?}");
    assert!(
        added + removed < Duration::from_secs(5),
        "{ROWS} score rows took {added:?} to add and {removed:?} to remove"
    );
}

/// A deterministic xorshift64* stream: every failure reproduces from its seed.
struct Stream(u64);

impl Stream {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 33) as usize % n
    }

    fn pick<'a>(&mut self, items: &[&'a str]) -> &'a str {
        items[self.below(items.len())]
    }

    fn names(&mut self, items: &[&str]) -> Vec<String> {
        (0..self.below(6))
            .map(|_| self.pick(items).to_owned())
            .collect()
    }
}

/// The roster and score rules exactly as the scoreboard applied them before the linear-time
/// rewrite: one `Vec` scan per listed player and one flattened `(player, objective)` map.
#[derive(Default)]
struct ScoreboardModel {
    teams: HashMap<String, Vec<String>>,
    memberships: HashMap<String, String>,
    objectives: HashSet<String>,
    scores: HashMap<(String, String), i32>,
}

impl ScoreboardModel {
    fn add_player_to_team(&mut self, player: &str, team: &str) {
        if !self.teams.contains_key(team) {
            return;
        }
        if let Some(previous) = self.memberships.insert(player.to_owned(), team.to_owned()) {
            if previous != team {
                if let Some(old) = self.teams.get_mut(&previous) {
                    old.retain(|m| m != player);
                }
            }
        }
        let members = self.teams.get_mut(team).expect("checked above");
        if !members.iter().any(|m| m == player) {
            members.push(player.to_owned());
        }
    }

    fn apply(&mut self, packet: &S2CPlay) {
        match packet {
            S2CPlay::Teams(s2c::Teams { team_name, action }) => match action {
                s2c::TeamsAction::Create { players, .. } => {
                    if self.teams.contains_key(team_name) {
                        return;
                    }
                    self.teams.insert(team_name.clone(), Vec::new());
                    for player in players {
                        self.add_player_to_team(player, team_name);
                    }
                }
                s2c::TeamsAction::Remove => {
                    if let Some(members) = self.teams.remove(team_name) {
                        for member in &members {
                            self.memberships.remove(member);
                        }
                    }
                }
                s2c::TeamsAction::AddPlayers { players } => {
                    for player in players {
                        self.add_player_to_team(player, team_name);
                    }
                }
                s2c::TeamsAction::RemovePlayers { players } => {
                    let Some(members) = self.teams.get_mut(team_name) else {
                        return;
                    };
                    for player in players {
                        if self.memberships.get(player) != Some(team_name) {
                            break;
                        }
                        self.memberships.remove(player);
                        members.retain(|m| m != player);
                    }
                }
                s2c::TeamsAction::UpdateInfo { .. } | s2c::TeamsAction::Unknown(_) => {}
            },
            S2CPlay::ScoreboardObjective(s2c::ScoreboardObjective {
                objective_name,
                mode,
            }) => match mode {
                s2c::ScoreboardObjectiveMode::Create { .. } => {
                    self.objectives.insert(objective_name.clone());
                }
                s2c::ScoreboardObjectiveMode::Remove => {
                    self.objectives.remove(objective_name);
                    self.scores.retain(|(_, obj), _| obj != objective_name);
                }
                _ => {}
            },
            S2CPlay::UpdateScore(s2c::UpdateScore {
                player_name,
                objective_name,
                action,
            }) => match action {
                s2c::UpdateScoreAction::Change { value } => {
                    if self.objectives.contains(objective_name) {
                        self.scores
                            .insert((player_name.clone(), objective_name.clone()), *value);
                    }
                }
                s2c::UpdateScoreAction::Remove => {
                    if objective_name.is_empty() {
                        self.scores.retain(|(player, _), _| player != player_name);
                    } else if self.objectives.contains(objective_name) {
                        self.scores
                            .remove(&(player_name.clone(), objective_name.clone()));
                    }
                }
            },
            _ => {}
        }
    }
}

/// The linear-time roster and score rows give
/// exactly the state the former per-player scans gave, after every packet of a long random
/// sequence over a small universe of names (so creates, moves, duplicates, case-equivalent
/// names, unknown teams and objectives, and removals that stop at a non-member all recur):
/// every roster in order, every membership, every score, and every objective's rows.
#[test]
fn the_linear_scoreboard_matches_the_former_per_player_scans_packet_for_packet() {
    const PLAYERS: [&str; 7] = ["a", "A", "b", "c", "dd", "Dd", "e"];
    const TEAMS: [&str; 3] = ["t1", "t2", "t3"];
    const OBJECTIVES: [&str; 4] = ["o1", "o2", "o3", ""];
    let mut board = crate::sync::scoreboard::Scoreboard::new();
    let mut model = ScoreboardModel::default();
    let mut stream = Stream(0x5EC0_2026_1122);
    for step in 0..40_000 {
        let team_name = stream.pick(&TEAMS).to_owned();
        let objective_name = stream.pick(&OBJECTIVES).to_owned();
        let player = stream.pick(&PLAYERS);
        let packet = match stream.below(10) {
            0 => teams(
                &team_name,
                s2c::TeamsAction::Create {
                    info: team_info(),
                    players: stream.names(&PLAYERS),
                },
            ),
            1 => teams(&team_name, s2c::TeamsAction::Remove),
            2 | 3 => teams(
                &team_name,
                s2c::TeamsAction::AddPlayers {
                    players: stream.names(&PLAYERS),
                },
            ),
            4 => teams(
                &team_name,
                s2c::TeamsAction::RemovePlayers {
                    players: stream.names(&PLAYERS),
                },
            ),
            5 => create_objective(&objective_name),
            6 => objective(&objective_name, s2c::ScoreboardObjectiveMode::Remove),
            7 | 8 => score(
                player,
                &objective_name,
                s2c::UpdateScoreAction::Change {
                    value: stream.below(5) as i32,
                },
            ),
            _ => score(player, &objective_name, s2c::UpdateScoreAction::Remove),
        };
        model.apply(&packet);
        // `Client::react` applies `ClientEvent::Scoreboard` exactly like this.
        let event = match packet.clone() {
            S2CPlay::Teams(teams) => crate::sync::ScoreboardEvent::Teams(Box::new(teams)),
            S2CPlay::ScoreboardObjective(objective) => {
                crate::sync::ScoreboardEvent::Objective(objective)
            }
            S2CPlay::UpdateScore(score) => crate::sync::ScoreboardEvent::Score(score),
            other => unreachable!("only scoreboard packets are generated: {other:?}"),
        };
        board.apply(&event);
        for team in TEAMS {
            assert_eq!(
                board.team(team).map(|team| team.members.clone()),
                model.teams.get(team).cloned(),
                "step {step}: roster of {team} after {packet:?}"
            );
        }
        for name in PLAYERS {
            assert_eq!(
                board.player_team(name).map(|team| team.name.clone()),
                model.memberships.get(name).cloned(),
                "step {step}: team of {name} after {packet:?}"
            );
            for objective_name in OBJECTIVES {
                assert_eq!(
                    board.score(name, objective_name),
                    model
                        .scores
                        .get(&(name.to_owned(), objective_name.to_owned()))
                        .copied(),
                    "step {step}: score of {name} in {objective_name:?} after {packet:?}"
                );
            }
        }
        for objective_name in OBJECTIVES {
            let mut rows: Vec<(String, i32)> = board
                .sorted_scores(objective_name)
                .into_iter()
                .map(|(name, points)| (name.to_owned(), points))
                .collect();
            rows.sort();
            let mut expected: Vec<(String, i32)> = model
                .scores
                .iter()
                .filter(|((_, obj), _)| obj == objective_name)
                .map(|((name, _), points)| (name.clone(), *points))
                .collect();
            expected.sort();
            assert_eq!(rows, expected, "step {step}: rows of {objective_name:?}");
        }
    }
}

/// java: `WorldClient.removeEntityFromWorld` is a map removal and the world list drops the
/// dead entity on its next tick. Spawning N entities and destroying them in one S13 must stay
/// near-linear, whether their column is loaded or not.
#[test]
fn spawning_and_destroying_many_entities_costs_near_linear_time() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("entity mass destroy", Duration::from_secs(300));
    let mut timings = Vec::new();
    // Column 40 was never sent: those entities wait for their chunk. Column -40 is also
    // absent, and a leash names a holder that never spawns (a leash to an untracked player).
    for (count, column_x) in [
        (2_000_usize, 0),
        (8_000, 0),
        (2_000, 40),
        (8_000, 40),
        (2_000, -40),
        (8_000, -40),
    ] {
        let ids: Vec<i32> = (0..count as i32).map(|index| 100_000 + index).collect();
        if column_x < 0 {
            rig.apply([
                zombie(99_999, 8 * 32, 8 * 32),
                attach(99_999, 77_777, s2c::AttachType::Leash),
            ]);
        }
        let start = Instant::now();
        for &id in &ids {
            rig.queue(zombie(id, (column_x * 16 + 8) * 32, 8 * 32));
        }
        rig.client.poll_in_place();
        let spawned = start.elapsed();
        let leashed = usize::from(column_x < 0);
        assert_eq!(
            rig.state().entities.len(),
            count + leashed,
            "all {count} tracked"
        );
        let start = Instant::now();
        rig.apply([destroy(ids)]);
        rig.ticks(1);
        let destroyed = start.elapsed();
        if column_x < 0 {
            rig.apply([destroy(vec![99_999])]);
        }
        assert_eq!(rig.state().entities.len(), 0);
        eprintln!(
            "SEQ {count} entities in column {column_x}: spawn {spawned:?}, \
             destroy {destroyed:?}"
        );
        timings.push((count, column_x, spawned, destroyed));
    }
    for pair in timings.chunks(2) {
        let [(small, column, small_spawn, small_destroy), (large, _, large_spawn, large_destroy)] =
            pair
        else {
            unreachable!("pairs");
        };
        let ratio =
            |big: Duration, little: Duration| big.as_secs_f64() / little.as_secs_f64().max(0.001);
        let factor = (*large as f64) / (*small as f64);
        // Linear would be `factor` (4); quadratic `factor^2` (16). Below 100 ms the ratio is
        // timer noise, so a fast large run passes on its own.
        assert!(
            ratio(*large_spawn, *small_spawn) < factor * 2.5
                || *large_spawn < Duration::from_millis(100),
            "spawning in column {column}: {small} took {small_spawn:?}, \
             {large} took {large_spawn:?}"
        );
        assert!(
            ratio(*large_destroy, *small_destroy) < factor * 2.5
                || *large_destroy < Duration::from_millis(100),
            "destroying in column {column}: {small} took {small_destroy:?}, \
             {large} took {large_destroy:?}"
        );
    }
}

/// java: `UUID.hashCode` is the XOR of the UUID's four 32-bit
/// words, so a server can make every modifier of one S20 share a `HashMap` bin; Java 8 turns
/// such a bin into a tree, and `applyModifier` stays O(log m). The port's model of that set's
/// iteration order scanned the bin for duplicates, so m colliding modifiers cost m^2 at every
/// resolution (each S20, and each attack for `generic.attackDamage`).
#[test]
fn attribute_modifiers_with_colliding_uuids_cost_near_linear_time() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("colliding modifiers", Duration::from_secs(300));
    rig.apply([zombie(40_000, 8 * 32, 8 * 32)]);
    // Every one of these hashes to 0: most = (i << 32 | i), least = 0.
    let colliding = |count: u32| -> Vec<s2c::AttributeModifierSnapshot> {
        (1..=count)
            .map(|index| {
                let most = (u128::from(index) << 32) | u128::from(index);
                s2c::AttributeModifierSnapshot {
                    uuid: Uuid::from_u128(most << 64),
                    amount: 0.0,
                    operation: s2c::AttributeModifierOperation::Add,
                }
            })
            .collect()
    };
    let mut timings = Vec::new();
    // 4x the modifiers: linear is 4x; the former bucket scan was about 11x at 2k -> 8k and
    // grows toward 16x with size.
    for count in [4_000_u32, 16_000] {
        let properties = |key: &str| {
            vec![s2c::AttributePropertySnapshot {
                key: key.into(),
                base_value: 20.0,
                modifiers: colliding(count),
            }]
        };
        let packets = [
            S2CPlay::EntityProperties(s2c::EntityProperties {
                entity_id: LOCAL,
                properties: properties("generic.maxHealth"),
            }),
            S2CPlay::EntityProperties(s2c::EntityProperties {
                entity_id: LOCAL,
                properties: properties("generic.attackDamage"),
            }),
            S2CPlay::EntityProperties(s2c::EntityProperties {
                entity_id: 40_000,
                properties: properties("generic.maxHealth"),
            }),
        ];
        assert!(packets.iter().all(deliverable), "a server can send these");
        let start = Instant::now();
        rig.apply(packets);
        rig.ticks(1);
        let took = start.elapsed();
        eprintln!("SEQ {count} colliding modifiers x3 attributes: {took:?}");
        timings.push(took);
    }
    let ratio = timings[1].as_secs_f64() / timings[0].as_secs_f64().max(0.001);
    assert!(
        ratio < 10.0 || timings[1] < Duration::from_millis(100),
        "4x the modifiers took {ratio:.1}x as long: {timings:?}"
    );
}

/// java: `GuiNewChat.setChatLine` trims its lists to their cap, so every message costs the same
/// however many came before. A flood of chat, action-bar and title text (a mix of plain,
/// formatted and hover components) must cost time linear in its length, with history at its cap.
#[test]
fn a_chat_flood_costs_time_linear_in_its_length() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("chat flood", Duration::from_secs(300));
    let message = |index: usize| {
        let component = match index % 4 {
            0 => format!(r#"{{"text":"message {index}"}}"#),
            1 => format!(
                r#"{{"text":"","extra":[{{"text":"[{index}] ","color":"gold","bold":true}},{{"text":"{}"}}]}}"#,
                "x".repeat(200)
            ),
            2 => format!(
                r#"{{"text":"hover {index}","hoverEvent":{{"action":"show_text","value":"{}"}}}}"#,
                "h".repeat(100)
            ),
            _ => format!(r#"{{"translate":"chat.type.text","with":["p{index}","hello"]}}"#),
        };
        S2CPlay::Chat(s2c::Chat {
            component,
            position: [0, 1, 2][index % 3],
        })
    };
    let mut timings = Vec::new();
    for count in [2_000_usize, 8_000] {
        let start = Instant::now();
        for index in 0..count {
            rig.queue(message(index));
            if index.is_multiple_of(512) {
                rig.ticks(1);
            }
        }
        rig.client.poll_in_place();
        let took = start.elapsed();
        eprintln!("SEQ {count} chat packets: {took:?}");
        timings.push(took);
    }
    let ratio = timings[1].as_secs_f64() / timings[0].as_secs_f64().max(0.001);
    assert!(
        ratio < 10.0 || timings[1] < Duration::from_millis(200),
        "4x the messages took {ratio:.1}x as long: {timings:?}"
    );
}

/// java: `World.updateEntity` ticks a vehicle and then its rider, so
/// every mounted entity is placed once per tick, in any spawn order. A chain whose riders
/// spawned before their vehicles must cost a tick time linear in its length, as one whose
/// vehicles came first does.
#[test]
fn a_riding_chain_costs_tick_time_linear_in_its_length_in_any_spawn_order() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("riding chain", Duration::from_secs(300));
    let mut timings = Vec::new();
    for (round, length) in [250_i32, 1_000].into_iter().enumerate() {
        let base = 70_000 + round as i32 * 10_000;
        let ids: Vec<i32> = (0..length).map(|index| base + index).collect();
        for &id in &ids {
            rig.queue(zombie(id, 8 * 32 + 16, 8 * 32 + 16));
        }
        // Each entity rides the one spawned after it: every rider precedes its vehicle.
        for pair in ids.windows(2) {
            rig.queue(attach(pair[0], pair[1], s2c::AttachType::Vehicle));
        }
        rig.client.poll_in_place();
        rig.ticks(1);
        // The bottom vehicle walks, so every rider above it moves on every tick.
        let root = ids[ids.len() - 1];
        let start = Instant::now();
        for tick in 0..10 {
            rig.queue(S2CPlay::EntityTeleport(s2c::EntityTeleport {
                entity_id: root,
                x: 8 * 32 + 16 + (tick % 2) * 32,
                y: 16 * 32,
                z: 8 * 32 + 16,
                yaw: Angle(0),
                pitch: Angle(0),
                on_ground: true,
            }));
            rig.ticks(1);
        }
        let took = start.elapsed();
        // The top rider sits one mount offset above the next, all the way down the chain.
        let top = rig.state().entities.get(ids[0]).expect("top rider");
        let bottom = rig
            .state()
            .entities
            .get(ids[ids.len() - 1])
            .expect("bottom vehicle");
        assert_eq!(top.riding_entity(), Some(ids[1]));
        assert!(bottom.riding_entity().is_none());
        eprintln!("SEQ riding chain of {length}, riders first: 10 ticks {took:?}");
        timings.push(took);
        rig.apply([destroy(ids)]);
    }
    let ratio = timings[1].as_secs_f64() / timings[0].as_secs_f64().max(0.001);
    assert!(
        ratio < 10.0 || timings[1] < Duration::from_millis(200),
        "a 4x longer chain took {ratio:.1}x as long to tick: {timings:?}"
    );
}

/// java: `Explosion.doExplosionB` and `handleMultiBlockChange` set each listed block in turn, so
/// both packets cost time linear in their record count (relighting included). Records here are
/// distinct positions inside loaded stone columns, the most work a record can cause.
#[test]
fn explosion_and_multi_block_records_cost_time_linear_in_their_count() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("block records", Duration::from_secs(600));
    for x in -1..=1 {
        for z in -1..=1 {
            rig.queue(floor_column(x, z));
        }
    }
    rig.client.poll_in_place();
    let mut timings = Vec::new();
    let sizes: Vec<usize> = std::env::var("SEQ_BLOCK_RECORDS")
        .ok()
        .and_then(|sizes| sizes.split(',').map(|n| n.parse().ok()).collect())
        .unwrap_or_else(|| vec![4_000, 16_000]);
    for count in sizes {
        // Distinct offsets inside the stone floor around (8, 8, 8): a cube of side `count^(1/3)`.
        let side = (count as f64).cbrt().ceil() as i32;
        let affected_blocks: Vec<s2c::ExplosionAffectedBlock> = (0..count as i32)
            .map(|index| s2c::ExplosionAffectedBlock {
                dx: (index % side - side / 2) as i8,
                dy: ((index / side) % side - side / 2) as i8,
                dz: (index / (side * side) - side / 2) as i8,
            })
            .collect();
        let explosion = S2CPlay::Explosion(s2c::Explosion {
            x: 8.0,
            y: 8.0,
            z: 8.0,
            strength: 4.0,
            affected_blocks,
            motion_x: 0.0,
            motion_y: 0.0,
            motion_z: 0.0,
        });
        let changes: Vec<s2c::MultiBlockChangeEntry> = (0..count)
            .map(|index| s2c::MultiBlockChangeEntry {
                rel_x: (index % 16) as u8,
                y: (index / 256 % 16) as u8,
                rel_z: (index / 16 % 16) as u8,
                block_state_id: i32::from(STONE_STATE),
            })
            .collect();
        let restore = S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
            chunk_x: 0,
            chunk_z: 0,
            changes,
        });
        assert!(deliverable(&explosion) && deliverable(&restore));
        let start = Instant::now();
        rig.apply([explosion, restore]);
        rig.ticks(2);
        let took = start.elapsed();
        eprintln!("SEQ {count} explosion records + {count} multi-block records: {took:?}");
        timings.push(took);
    }
    let ratio = timings[1].as_secs_f64() / timings[0].as_secs_f64().max(0.001);
    assert!(
        ratio < 10.0 || timings[1] < Duration::from_millis(200),
        "4x the records took {ratio:.1}x as long: {timings:?}"
    );
}

// ---- scenarios: state that must not outlive what the server removed ------------------------

/// java: every entity-targeted handler starts with `getEntityByID` and does nothing for an id
/// that is not in the world, whether it never arrived or was already destroyed. None of them may
/// leave client-side state behind for such an id, or a server could grow it without bound by
/// naming fresh ids.
#[test]
fn packets_for_unknown_or_destroyed_entities_leave_no_state_behind() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("unknown entity ids", Duration::from_secs(300));
    // Warm-up: one full cycle, then let every timed owner expire (damaged blocks: 400 ticks).
    let cycle = |rig: &mut Rig, base: i32| {
        for id in base..base + 50 {
            // Never spawned.
            rig.apply(entity_targeted(id, id + 1));
            // Spawned, destroyed, then addressed again.
            let respawned = id + 10_000;
            rig.apply([zombie(respawned, 8 * 32, 8 * 32), destroy(vec![respawned])]);
            rig.apply(entity_targeted(respawned, id));
        }
        rig.apply([S2CPlay::Camera(s2c::Camera { entity_id: LOCAL })]);
        rig.ticks(420);
    };
    cycle(&mut rig, 1_000);
    let before = census(&rig);
    for round in 0..3 {
        cycle(&mut rig, 20_000 + round * 1_000);
    }
    let after = census(&rig);
    let growth = census_growth(&before, &after);
    assert!(growth.is_empty(), "state left behind: {growth:?}");
}

/// java: the death of a vehicle, rider or leash holder, the riders of a destroyed vehicle and
/// the vehicle of a destroyed rider: all of it is dropped with the entity. Long churn of
/// mounted, leashed, named, effect-carrying entities with fresh ids must come back to the
/// baseline.
#[test]
fn long_entity_churn_returns_client_state_to_its_baseline() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("entity churn", Duration::from_secs(300));
    let cycle = |rig: &mut Rig, base: i32| {
        for id in (base..base + 40).step_by(4) {
            rig.apply([
                zombie(id, 8 * 32, 8 * 32),
                minecart(id + 1, 0),
                minecart(id + 2, 2),
                minecart(id + 3, 6),
                attach(id, id + 1, s2c::AttachType::Vehicle),
                attach(id + 2, id, s2c::AttachType::Leash),
                // A holder that never spawns: the leash waits for it until its record goes.
                attach(id + 1, base + 99_999, s2c::AttachType::Leash),
            ]);
            rig.apply(entity_targeted(id, id + 3));
            rig.ticks(2);
            rig.apply([destroy(vec![id + 1, id, id + 2, id + 3])]);
        }
        // A wither (64): head targets and the boss bar's max health.
        rig.apply([S2CPlay::SpawnMob(s2c::SpawnMob {
            entity_id: base + 900,
            entity_type: 64,
            x: 8 * 32,
            y: 20 * 32,
            z: 8 * 32,
            yaw: Angle(0),
            pitch: Angle(0),
            head_pitch: Angle(0),
            velocity_x: 0,
            velocity_y: 0,
            velocity_z: 0,
            metadata: vec![MetadataEntry {
                index: 17,
                value: MetadataValue::Int(base + 901),
            }],
        })]);
        rig.apply(entity_targeted(base + 900, base + 901));
        rig.ticks(2);
        rig.apply([destroy(vec![base + 900])]);
        rig.ticks(420);
    };
    cycle(&mut rig, 1_000);
    let before = census(&rig);
    for round in 0..3 {
        cycle(&mut rig, 50_000 + round * 1_000);
    }
    let after = census(&rig);
    let growth = census_growth(&before, &after);
    assert!(growth.is_empty(), "state left behind: {growth:?}");
}

/// java: `ChunkProviderClient.unloadChunk` of a column that was never loaded only touches the
/// blank chunk, and `World.setBlockState` refuses positions outside 0..256 (`isValid`) and
/// writes nothing into an absent column (`EmptyChunk.setBlockState`). Thousands of such
/// packets must not create columns or queue work that never drains.
#[test]
fn unloading_absent_columns_and_editing_outside_the_world_keep_nothing() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("absent columns", Duration::from_secs(300));
    rig.ticks(2);
    let before = census(&rig);
    let start = Instant::now();
    let extremes = [
        i32::MIN,
        i32::MIN + 1,
        -1_875_000,
        1_875_000,
        i32::MAX - 1,
        i32::MAX,
    ];
    for (index, chunk_x) in (1_000..4_000).enumerate() {
        rig.queue(unload_column(chunk_x, -chunk_x));
        let edge = extremes[index % extremes.len()];
        rig.queue(unload_column(edge, edge));
    }
    for y in [-1, 256, 1_000, 2047, -2048] {
        for x in [0, 30_000_000, -30_000_000, 33_554_431, -33_554_432] {
            rig.queue(S2CPlay::BlockChange(s2c::BlockChange {
                position: BlockPos::new(x, y, 5),
                block_state_id: i32::from(STONE_STATE),
            }));
            rig.queue(S2CPlay::BlockAction(s2c::BlockAction {
                position: BlockPos::new(x, y, 5),
                data1: 1,
                data2: 1,
                block_id: 54,
            }));
            rig.queue(S2CPlay::BlockBreakAnimation(s2c::BlockBreakAnimation {
                breaker_id: 7,
                position: BlockPos::new(x, y, 5),
                progress: 3,
            }));
        }
    }
    for chunk in [i32::MIN, -1, 1 << 20, i32::MAX] {
        rig.queue(S2CPlay::MultiBlockChange(s2c::MultiBlockChange {
            chunk_x: chunk,
            chunk_z: chunk,
            changes: (0..=255u8)
                .map(|y| s2c::MultiBlockChangeEntry {
                    rel_x: y % 16,
                    y,
                    rel_z: y / 16,
                    block_state_id: i32::from(STONE_STATE),
                })
                .collect(),
        }));
    }
    rig.client.poll_in_place();
    rig.ticks(420);
    let took = start.elapsed();
    let after = census(&rig);
    let growth = census_growth(&before, &after);
    assert!(growth.is_empty(), "state left behind: {growth:?}");
    assert!(
        rig.client.mesh_queue_len() < 64,
        "mesh work queued for absent columns: {}",
        rig.client.mesh_queue_len()
    );
    eprintln!("SEQ absent columns and out-of-world edits: {took:?}");
}

// ---- scenarios: orders vanilla survives ----------------------------------------------------

/// java: Play packets can arrive after login and before `S01PacketJoinGame`; vanilla's handlers
/// then meet a null `thePlayer` or world, and `Util.runTask` only logs the exception. Every kind
/// of packet before the join (and a frame drawn meanwhile) must neither panic nor stall, and the
/// join that follows must still work.
#[test]
fn play_packets_before_the_join_neither_panic_nor_stall() {
    let Some(rig) = Rig::joined() else {
        return;
    };
    // A fresh session that has logged in but not joined.
    let mut rig = rig;
    let (handle, to_game, to_net) = NetHandle::new_pair();
    rig.client.session = Some(Session::new(handle, JoinSettings::default()));
    rig.to_game = to_game;
    rig.to_net = to_net;
    let _watchdog = Watchdog::arm("before the join", Duration::from_secs(300));
    let mut packets = vec![
        teleport(8.5, 16.0, 8.5),
        respawn(-1),
        floor_column(0, 0),
        unload_column(0, 0),
        zombie(50, 8 * 32, 8 * 32),
        minecart(51, 0),
        destroy(vec![50, 51, LOCAL]),
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id: 3,
            inventory_type: "minecraft:chest".into(),
            window_title: r#"{"text":"Chest"}"#.into(),
            slot_count: 27,
            entity_id: None,
        }),
        S2CPlay::SetSlot(s2c::SetSlot {
            window_id: 0,
            slot: 36,
            item: Some(extreme_stack(1)),
        }),
        S2CPlay::WindowItems(s2c::WindowItems {
            window_id: 0,
            items: vec![None; 45],
        }),
        S2CPlay::ConfirmTransaction(s2c::ConfirmTransaction {
            window_id: 0,
            action_number: 1,
            accepted: false,
        }),
        S2CPlay::UpdateHealth(s2c::UpdateHealth {
            health: 0.0,
            food_level: 0,
            saturation_level: 0.0,
        }),
        S2CPlay::SetExperience(s2c::SetExperience {
            experience_bar: 0.5,
            level: 1,
            total_experience: 1,
        }),
        S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
            flags: 0x0f,
            fly_speed: 0.05,
            walk_speed: 0.1,
        }),
        S2CPlay::HeldItemChange(s2c::HeldItemChange { slot: 3 }),
        S2CPlay::TimeUpdate(s2c::TimeUpdate {
            total_world_time: 1,
            world_time: 6_000,
        }),
        S2CPlay::SpawnPosition(s2c::SpawnPosition {
            position: BlockPos::new(1, 2, 3),
        }),
        S2CPlay::ChangeGameState(s2c::ChangeGameState {
            reason: s2c::GameStateReason::BeginRaining,
            value: 0.0,
        }),
        S2CPlay::ChangeGameState(s2c::ChangeGameState {
            reason: s2c::GameStateReason::ChangeGameMode,
            value: 1.0,
        }),
        S2CPlay::WorldBorder(s2c::WorldBorder::SetSize { diameter: 10.0 }),
        S2CPlay::Title(s2c::Title::SetTitle {
            text: r#"{"text":"t"}"#.into(),
        }),
        S2CPlay::Chat(s2c::Chat {
            component: r#"{"text":"before the join"}"#.into(),
            position: 0,
        }),
        S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
            uuid: Uuid::from_u128(77),
            username: "early".into(),
            properties: Vec::new(),
            game_mode: s2c::GameMode::Survival,
            ping: 1,
            display_name: None,
        }])),
        create_objective("early"),
        score(
            "early",
            "early",
            s2c::UpdateScoreAction::Change { value: 1 },
        ),
        S2CPlay::Camera(s2c::Camera { entity_id: 50 }),
        S2CPlay::SignEditorOpen(s2c::SignEditorOpen {
            position: BlockPos::new(8, 16, 8),
        }),
        S2CPlay::Explosion(s2c::Explosion {
            x: 8.0,
            y: 16.0,
            z: 8.0,
            strength: 4.0,
            affected_blocks: vec![s2c::ExplosionAffectedBlock {
                dx: 0,
                dy: -1,
                dz: 0,
            }],
            motion_x: 1.0,
            motion_y: 1.0,
            motion_z: 1.0,
        }),
        S2CPlay::BlockChange(s2c::BlockChange {
            position: BlockPos::new(8, 15, 8),
            block_state_id: 0,
        }),
        S2CPlay::EntityEffect(s2c::EntityEffect {
            entity_id: LOCAL,
            effect_id: 1,
            amplifier: 0,
            duration: 200,
            show_particles: true,
        }),
        S2CPlay::EntityProperties(s2c::EntityProperties {
            entity_id: LOCAL,
            properties: vec![s2c::AttributePropertySnapshot {
                key: "generic.movementSpeed".into(),
                base_value: 0.2,
                modifiers: Vec::new(),
            }],
        }),
        attach(LOCAL, 51, s2c::AttachType::Vehicle),
        S2CPlay::EntityVelocity(s2c::EntityVelocity {
            entity_id: LOCAL,
            motion_x: 100,
            motion_y: 100,
            motion_z: 100,
        }),
    ];
    packets.extend(entity_targeted(LOCAL, 50));
    packets.extend(entity_targeted(50, LOCAL));
    for packet in packets {
        let shown = format!("{packet:?}");
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            rig.apply([packet]);
            rig.ticks(1);
        }));
        assert!(
            outcome.is_ok(),
            "before the join, this panicked: {shown:.300}"
        );
    }
    // They really were applied, as vanilla applies them (not dropped for want of a join).
    assert_eq!(
        rig.state().players.len(),
        1,
        "the early Tab entry was applied"
    );
    assert_eq!(rig.state().scoreboard.score("early", "early"), Some(1));
    // Then a normal join, which must still bring the player into the world.
    rig.apply([join(0), floor_column(0, 0), teleport(8.5, 16.0, 8.5)]);
    rig.ticks(3);
    assert!(
        rig.sent()
            .iter()
            .any(|packet| matches!(packet, C2SPlay::PlayerPositionAndLook(_))),
        "the S08 after the join is answered"
    );
}

/// java: `Entity.mountEntity` refuses a vehicle whose own riding chain already contains the
/// rider, so a server's attach cycle (A on B on A, or A on itself) never makes a loop. Every
/// walk up or down the chain (ticks, the camera, the render snapshot) must stay bounded.
#[test]
fn riding_cycles_and_self_attachment_cannot_hang_the_client() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("riding cycles", Duration::from_secs(120));
    rig.apply([
        zombie(10, 8 * 32, 8 * 32),
        zombie(11, 8 * 32, 8 * 32),
        zombie(12, 8 * 32, 8 * 32),
        minecart(13, 0),
    ]);
    for (rider, vehicle) in [
        (10, 10),
        (10, 11),
        (11, 10),
        (11, 12),
        (12, 10),
        (LOCAL, 13),
        (13, LOCAL),
        (LOCAL, LOCAL),
        (13, 10),
        (10, LOCAL),
    ] {
        rig.apply([attach(rider, vehicle, s2c::AttachType::Vehicle)]);
        rig.apply([attach(vehicle, rider, s2c::AttachType::Leash)]);
        rig.apply([S2CPlay::Camera(s2c::Camera { entity_id: rider })]);
        rig.ticks(3);
    }
    rig.apply([S2CPlay::Camera(s2c::Camera { entity_id: LOCAL })]);
    rig.apply([destroy(vec![10, 11, 12, 13])]);
    rig.ticks(5);
}

/// java: `WorldClient.addEntityToWorld` removes whatever already had the id first, including
/// `thePlayer` when a spawn reuses the local id; S13 naming the local id removes it the same
/// way. None of it may panic, and the session must keep running.
#[test]
fn reused_and_local_entity_ids_do_not_break_the_session() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("reused ids", Duration::from_secs(120));
    let player = |entity_id: i32| {
        S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
            entity_id,
            player_uuid: Uuid::from_u128(0x5e9),
            x: 8 * 32,
            y: 16 * 32,
            z: 8 * 32,
            yaw: Angle(0),
            pitch: Angle(0),
            current_item: 0,
            metadata: Vec::new(),
        })
    };
    rig.apply([
        zombie(20, 8 * 32, 8 * 32),
        player(20),
        minecart(20, 0),
        S2CPlay::SpawnExperienceOrb(s2c::SpawnExperienceOrb {
            entity_id: 20,
            x: 0,
            y: 0,
            z: 0,
            xp_value: 3,
        }),
        zombie(LOCAL, 8 * 32, 8 * 32),
        player(LOCAL),
        minecart(LOCAL, 0),
    ]);
    rig.ticks(3);
    rig.apply([destroy(vec![LOCAL, LOCAL, 20, 20, -1, i32::MIN, i32::MAX])]);
    rig.ticks(3);
    rig.apply([teleport(8.5, 16.0, 8.5)]);
    rig.ticks(3);
    let replies = rig.sent();
    assert!(
        replies
            .iter()
            .any(|packet| matches!(packet, C2SPlay::PlayerPositionAndLook(_))),
        "the session still answers an S08: {replies:?}"
    );
}

/// java: `handleCloseWindow` is `closeScreenAndDropStack()`, which sends nothing, and
/// `handleOpenWindow` over an open container replaces it without a C0D (`GuiContainer`'s
/// `onGuiClosed` only calls `Container.onContainerClosed`). The client must not answer either.
#[test]
fn server_window_opens_and_closes_are_not_answered() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("server windows", Duration::from_secs(120));
    let chest = |window_id: u8, slot_count: u8| {
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id,
            inventory_type: "minecraft:chest".into(),
            window_title: r#"{"text":"Chest"}"#.into(),
            slot_count,
            entity_id: None,
        })
    };
    rig.apply([chest(1, 27)]);
    assert!(
        matches!(rig.client.ui, Ui::Container(_)),
        "the chest opened"
    );
    rig.apply([chest(2, 54), chest(3, 0), chest(4, 255)]);
    rig.apply([S2CPlay::CloseWindow(s2c::CloseWindow { window_id: 4 })]);
    assert!(matches!(rig.client.ui, Ui::InGame), "S2E closed the screen");
    rig.apply([S2CPlay::CloseWindow(s2c::CloseWindow { window_id: 9 })]);
    rig.ticks(2);
    let replies = rig.sent();
    assert!(
        !replies
            .iter()
            .any(|packet| matches!(packet, C2SPlay::CloseWindow(_))),
        "vanilla sends no C0D for a server's open or close: {replies:?}"
    );
}

/// java: `handleConfirmTransaction` answers `C0F(window, action, true)` only for a refused
/// transaction (`!accepted`) on window 0 or the open container's id, once per packet, in order.
#[test]
fn transactions_are_answered_exactly_as_vanilla_answers_them() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("transactions", Duration::from_secs(120));
    let transaction = |window_id: u8, action_number: i16, accepted: bool| {
        S2CPlay::ConfirmTransaction(s2c::ConfirmTransaction {
            window_id,
            action_number,
            accepted,
        })
    };
    let mut expected = Vec::new();
    for action in [i16::MIN, -1, 0, 1, i16::MAX] {
        rig.apply([
            transaction(0, action, false),
            transaction(0, action, true),
            transaction(5, action, false),
            transaction(255, action, false),
        ]);
        expected.push((0_u8, action));
    }
    let answered: Vec<(u8, i16)> = rig
        .sent()
        .into_iter()
        .filter_map(|packet| match packet {
            C2SPlay::ConfirmTransaction(c2s::ConfirmTransaction {
                window_id,
                action_id,
                accepted,
            }) => {
                assert!(accepted, "the answer always says accepted");
                Some((window_id, action_id))
            }
            _ => None,
        })
        .collect();
    assert_eq!(answered, expected);
    // A flood: one answer per refused transaction, no more, no fewer.
    for action in 0..5_000 {
        rig.queue(transaction(0, action, false));
    }
    rig.client.poll_in_place();
    let flood = rig
        .sent()
        .into_iter()
        .filter(|packet| matches!(packet, C2SPlay::ConfirmTransaction(_)))
        .count();
    assert_eq!(flood, 5_000);
}

/// java: `handlePlayerPosLook` sends one C06 per S08, from the packet's place in the stream.
/// A flood of teleports, including ones with non-finite or out-of-world values, gets exactly
/// one C06 each and never a panic or a stall while ticking afterwards.
#[test]
fn every_teleport_is_answered_once_even_in_a_flood_of_extreme_values() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("teleport flood", Duration::from_secs(300));
    let values = [
        8.5,
        -8.5,
        0.0,
        -0.0,
        29_999_999.0,
        -30_000_001.0,
        1e30,
        -1e30,
        f64::MAX,
        f64::MIN_POSITIVE,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let mut count = 0;
    for (index, &x) in values.iter().enumerate() {
        for flags in [0_u8, 0x1f] {
            let y = values[(index + 3) % values.len()];
            let z = values[(index + 7) % values.len()];
            rig.queue(S2CPlay::PlayerPositionLook(s2c::PlayerPositionLook {
                x,
                y,
                z,
                yaw: f32::NAN,
                pitch: f32::INFINITY,
                flags,
            }));
            count += 1;
        }
    }
    rig.client.poll_in_place();
    let answers = rig
        .sent()
        .into_iter()
        .filter(|packet| matches!(packet, C2SPlay::PlayerPositionAndLook(_)))
        .count();
    assert_eq!(answers, count, "one C06 per S08");
    // Ticks at each edge position (movement, chunk residency, light, the camera). Coordinates
    // beyond Java's `int` range are left out here: there the block arithmetic wraps in the
    // user's build exactly as Java's `int` does, and only this debug build's overflow checks
    // stop (the light engine's `origin.x - 18`, as the live fuzzer documents).
    for &x in &[
        8.5,
        -8.5,
        29_999_999.0,
        -30_000_001.0,
        1_000_000_000.0,
        f64::NAN,
    ] {
        for &y in &[16.0, 300.0, -100.0, f64::NAN] {
            let start = Instant::now();
            rig.apply([teleport(x, y, -x)]);
            rig.ticks(4);
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "ticking at ({x}, {y}, {}) took {:?}",
                -x,
                start.elapsed()
            );
        }
    }
    rig.apply([teleport(8.5, 16.0, 8.5)]);
    rig.ticks(4);
}

/// java: `handleRespawn` rebuilds the world only for another dimension and keeps whatever
/// screen is open otherwise; `handleJoinGame` mid-session rebuilds it too. Neither may panic
/// with a container, sign editor or book open, or for a dimension vanilla has no provider for.
#[test]
fn respawns_and_joins_arrive_safely_over_open_screens() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("respawn over screens", Duration::from_secs(300));
    let screens: Vec<S2CPlay> = vec![
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id: 7,
            inventory_type: "minecraft:chest".into(),
            window_title: r#"{"text":"Chest"}"#.into(),
            slot_count: 27,
            entity_id: None,
        }),
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id: 8,
            inventory_type: "minecraft:villager".into(),
            window_title: r#"{"text":"Villager"}"#.into(),
            slot_count: 3,
            entity_id: None,
        }),
        S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id: 9,
            inventory_type: "EntityHorse".into(),
            window_title: r#"{"text":"Horse"}"#.into(),
            slot_count: 2,
            entity_id: Some(12_345),
        }),
        S2CPlay::SignEditorOpen(s2c::SignEditorOpen {
            position: BlockPos::new(8, 16, 8),
        }),
        S2CPlay::ChangeGameState(s2c::ChangeGameState {
            reason: s2c::GameStateReason::EnterCredits,
            value: 0.0,
        }),
        S2CPlay::UpdateHealth(s2c::UpdateHealth {
            health: 0.0,
            food_level: 20,
            saturation_level: 5.0,
        }),
    ];
    let dimensions = [0, -1, 1, 0, 2, -2, i32::MAX, i32::MIN, 0];
    for screen in &screens {
        for &dimension in &dimensions {
            rig.apply([screen.clone()]);
            rig.apply([respawn(dimension), floor_column(0, 0)]);
            rig.apply([
                S2CPlay::SetSlot(s2c::SetSlot {
                    window_id: 7,
                    slot: 3,
                    item: None,
                }),
                S2CPlay::WindowItems(s2c::WindowItems {
                    window_id: 7,
                    items: vec![None; 90],
                }),
                S2CPlay::WindowProperty(s2c::WindowProperty {
                    window_id: 7,
                    var_index: 0,
                    var_value: 100,
                }),
                S2CPlay::UpdateHealth(s2c::UpdateHealth {
                    health: 20.0,
                    food_level: 20,
                    saturation_level: 5.0,
                }),
                teleport(8.5, 16.0, 8.5),
            ]);
            rig.ticks(2);
        }
        rig.apply([screen.clone(), join(-1), floor_column(0, 0), join(0)]);
        rig.apply([floor_column(0, 0), teleport(8.5, 16.0, 8.5)]);
        rig.ticks(2);
    }
    rig.sent();
}

/// A headless GPU and everything one shell frame needs.
struct Screen {
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    input: InputState,
    settings: GameSettings,
    stats: mc_render::FrameStats,
    /// Time spent in `ShellApp::frame` (building the frame on the CPU), summed.
    build_time: Duration,
    /// Time spent rendering and presenting (the phases' encoding and the GPU), summed.
    render_time: Duration,
}

impl Screen {
    const WIDTH: u32 = 854;
    const HEIGHT: u32 = 480;

    /// `None` without a GPU adapter (the scenario is then skipped, as the other GPU tests are).
    fn install(rig: &mut Rig) -> Option<Self> {
        let gpu = match Gpu::headless(Self::WIDTH, Self::HEIGHT, mc_render::GpuOptions::default()) {
            Ok(gpu) => gpu,
            Err(error) => {
                eprintln!("skip: no headless GPU ({error})");
                return None;
            }
        };
        let mut phases = PhaseList::new();
        rig.client
            .install(&gpu, &mut phases, Self::WIDTH, Self::HEIGHT);
        let mut camera = Camera::default();
        camera.set_viewport(Self::WIDTH, Self::HEIGHT);
        Some(Self {
            gpu,
            phases,
            camera,
            input: InputState::default(),
            settings: GameSettings::default(),
            stats: mc_render::FrameStats::new(),
            build_time: Duration::ZERO,
            render_time: Duration::ZERO,
        })
    }

    /// [`Self::frame`], with a panic turned into its message.
    fn try_frame(&mut self, rig: &mut Rig) -> Result<Duration, String> {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.frame(rig))).map_err(
            |panic| {
                panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default()
            },
        )
    }

    /// One shell iteration: drain the packets, one tick, then build and present a frame.
    fn frame(&mut self, rig: &mut Rig) -> Duration {
        let start = Instant::now();
        <Client as mc_render::ShellApp>::before_ticks(&mut rig.client, &mut self.camera);
        rig.client.tick(&MoveInput::default());
        let built = Instant::now();
        <Client as mc_render::ShellApp>::frame(
            &mut rig.client,
            &mut mc_render::FrameCtx {
                gpu: &self.gpu,
                phases: &mut self.phases,
                camera: &mut self.camera,
                input: &mut self.input,
                settings: &self.settings,
                partial_ticks: 0.5,
                stats: &self.stats,
            },
        );
        self.camera.update();
        self.build_time += built.elapsed();
        let rendered = Instant::now();
        let frame = self.gpu.acquire().expect("a headless frame");
        self.phases
            .render(&self.gpu, frame.view(), &self.camera, 0.5);
        frame.present(&self.gpu.queue);
        self.render_time += rendered.elapsed();
        start.elapsed()
    }
}

/// What a run of hostile frames found.
#[derive(Default)]
struct FrameLog {
    slowest: Duration,
    /// Debug-build arithmetic overflow checks, by step. The user's build wraps there, as Java's
    /// `int` does in the same vanilla expression; they are reported, not failures.
    overflows: Vec<String>,
    /// Every other panic: a crash in the user's build too.
    panics: Vec<String>,
}

impl FrameLog {
    fn frame(&mut self, screen: &mut Screen, rig: &mut Rig, step: &str) {
        match screen.try_frame(rig) {
            Ok(took) => self.slowest = self.slowest.max(took),
            Err(message) if message.contains("with overflow") => {
                self.overflows.push(format!("{step}: {message}"));
            }
            Err(message) => self.panics.push(format!("{step}: {message}")),
        }
    }
}

fn living(entity_id: i32, entity_type: u8, metadata: Vec<MetadataEntry>) -> S2CPlay {
    S2CPlay::SpawnMob(s2c::SpawnMob {
        entity_id,
        entity_type,
        x: (8 * 32) + 64,
        y: 17 * 32,
        z: (8 * 32) + 96,
        yaw: Angle(0),
        pitch: Angle(0),
        head_pitch: Angle(0),
        velocity_x: 0,
        velocity_y: 0,
        velocity_z: 0,
        metadata,
    })
}

fn max_health(entity_id: i32, base_value: f64) -> S2CPlay {
    S2CPlay::EntityProperties(s2c::EntityProperties {
        entity_id,
        properties: vec![s2c::AttributePropertySnapshot {
            key: "generic.maxHealth".into(),
            base_value,
            modifiers: vec![s2c::AttributeModifierSnapshot {
                uuid: Uuid::from_u128(9),
                amount: base_value,
                operation: s2c::AttributeModifierOperation::Add,
            }],
        }],
    })
}

/// Frames drawn over the state a hostile server can build: a 20,000-row sidebar and a
/// 5,000-player Tab list held open (both sorted for drawing), titles with extreme times, a
/// world border at its limits, bosses with non-finite or extreme health and long names, armor
/// stands with non-finite poses, a size-127 slime, a 200-long riding chain and a map with
/// out-of-range icons in an item frame. No frame may panic or stall.
#[test]
fn hostile_hud_and_world_state_draws_in_bounded_frames() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let Some(mut screen) = Screen::install(&mut rig) else {
        return;
    };
    let _watchdog = Watchdog::arm("hostile frames", Duration::from_secs(600));
    for _ in 0..3 {
        screen.frame(&mut rig);
    }
    // The scoreboard in every slot, a full sidebar, and team affixes on every row.
    let long_name = "W".repeat(40);
    rig.apply([
        create_objective("side"),
        create_objective("below"),
        create_objective("list"),
        S2CPlay::DisplayScoreboard(s2c::DisplayScoreboard {
            position: 1,
            objective_name: "side".into(),
        }),
        S2CPlay::DisplayScoreboard(s2c::DisplayScoreboard {
            position: 2,
            objective_name: "below".into(),
        }),
        S2CPlay::DisplayScoreboard(s2c::DisplayScoreboard {
            position: 0,
            objective_name: "list".into(),
        }),
        teams(
            "affix",
            s2c::TeamsAction::Create {
                info: s2c::TeamInfo {
                    display_name: "affix".into(),
                    prefix: "\u{a7}k".repeat(8),
                    suffix: "\u{a7}l".repeat(8),
                    friendly_flags: 3,
                    name_tag_visibility: "always".into(),
                    color: 15,
                },
                players: names(2_000),
            },
        ),
    ]);
    for (index, player) in names(20_000).iter().enumerate() {
        let value = [i32::MIN, i32::MAX, 0, -1][index % 4];
        rig.queue(score(
            player,
            "side",
            s2c::UpdateScoreAction::Change { value },
        ));
    }
    rig.queue(score(
        &long_name,
        "side",
        s2c::UpdateScoreAction::Change { value: 7 },
    ));
    // A held-open Tab list of 5,000 players, with header and footer.
    rig.queue(S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(
        (0..5_000_u128)
            .map(|index| s2c::AddPlayerEntry {
                uuid: Uuid::from_u128(0x7AB_0000 + index),
                username: format!("p{index}"),
                properties: Vec::new(),
                game_mode: s2c::GameMode::Survival,
                ping: [i32::MIN, -1, 0, 150, 1_000, i32::MAX][index as usize % 6],
                display_name: index
                    .is_multiple_of(3)
                    .then(|| format!(r#"{{"text":"{}"}}"#, "N".repeat(40))),
            })
            .collect(),
    )));
    rig.queue(S2CPlay::PlayerListHeaderFooter(
        s2c::PlayerListHeaderFooter {
            header: format!(r#"{{"text":"{}"}}"#, "H\\n".repeat(500)),
            footer: format!(r#"{{"text":"{}"}}"#, "F".repeat(4_000)),
        },
    ));
    rig.client.tab_list_held = true;
    // Bosses: a wither and a dragon with non-finite and extreme health, long names.
    let named = |health: f32| {
        vec![
            MetadataEntry {
                index: 2,
                value: MetadataValue::String("B".repeat(400)),
            },
            MetadataEntry {
                index: 3,
                value: MetadataValue::Byte(1),
            },
            MetadataEntry {
                index: 6,
                value: MetadataValue::Float(health),
            },
        ]
    };
    rig.apply([
        living(30_001, 64, named(f32::NAN)),
        max_health(30_001, f64::NAN),
        living(30_002, 63, named(f32::INFINITY)),
        max_health(30_002, 0.0),
        living(30_003, 64, named(-1e30)),
        max_health(30_003, f64::MAX),
    ]);
    // Armor stands with non-finite poses, a giant slime, a long riding chain.
    for (index, angle) in [f32::NAN, f32::INFINITY, -1e30, 1e30]
        .into_iter()
        .enumerate()
    {
        let mut pose = vec![MetadataEntry {
            index: 10,
            value: MetadataValue::Byte(0x0f),
        }];
        for part in 11..=16 {
            pose.push(MetadataEntry {
                index: part,
                value: MetadataValue::Rotations(angle, -angle, angle),
            });
        }
        rig.queue(living(30_100 + index as i32, 30, pose));
    }
    rig.queue(living(
        30_200,
        55,
        vec![MetadataEntry {
            index: 16,
            value: MetadataValue::Byte(127),
        }],
    ));
    for link in 0..200 {
        rig.queue(zombie(31_000 + link, 8 * 32 + 64, 8 * 32 + 64));
        if link > 0 {
            rig.queue(attach(
                31_000 + link,
                31_000 + link - 1,
                s2c::AttachType::Vehicle,
            ));
        }
    }
    // A map with out-of-range icons, in an item frame in front of the player.
    rig.queue(S2CPlay::Maps(s2c::Maps {
        map_id: 32_767,
        scale: 127,
        icons: (0..=255_u8)
            .map(|index| s2c::MapIcon {
                icon_type: index,
                x: index as i8,
                z: (index as i8).wrapping_neg(),
                direction: index,
            })
            .collect(),
        update: Some(s2c::MapPixelUpdate {
            width: 128,
            height: 128,
            min_x: 0,
            min_y: 0,
            data: (0..128 * 128).map(|index| index as u8).collect(),
        }),
    }));
    rig.queue(S2CPlay::SpawnObject(s2c::SpawnObject {
        entity_id: 30_300,
        object_type: 71,
        x: 8 * 32,
        y: 17 * 32,
        z: 10 * 32,
        pitch: Angle(0),
        yaw: Angle(0),
        data: 2,
        velocity: None,
    }));
    rig.queue(S2CPlay::EntityMetadata(s2c::EntityMetadata {
        entity_id: 30_300,
        metadata: vec![
            MetadataEntry {
                index: 8,
                value: MetadataValue::ItemStack(Some(mc_protocol::ItemStack {
                    item_id: 358,
                    count: 1,
                    damage: 32_767,
                    nbt: None,
                })),
            },
            MetadataEntry {
                index: 9,
                value: MetadataValue::Byte(-128),
            },
        ],
    }));
    let mut log = FrameLog::default();
    log.frame(
        &mut screen,
        &mut rig,
        "scoreboard, tab list, bosses, stands, chain, map",
    );
    // Finite boss health again, or the boss bar would stop every later frame at the same
    // debug-build overflow before the rest of the HUD is drawn.
    for boss in [30_001, 30_002, 30_003] {
        rig.queue(max_health(boss, 20.0));
        rig.queue(S2CPlay::EntityMetadata(s2c::EntityMetadata {
            entity_id: boss,
            metadata: named(10.0),
        }));
    }
    log.frame(&mut screen, &mut rig, "bosses at finite health");
    for (fade_in, stay, fade_out) in [(0, i32::MAX, i32::MIN), (i32::MIN, 0, 0), (-5, -5, -5)] {
        rig.queue(S2CPlay::Title(s2c::Title::SetTimes {
            fade_in_ticks: fade_in,
            stay_ticks: stay,
            fade_out_ticks: fade_out,
        }));
        rig.queue(S2CPlay::Title(s2c::Title::SetTitle {
            text: format!(r#"{{"text":"{}"}}"#, "T".repeat(2_000)),
        }));
        rig.queue(S2CPlay::Title(s2c::Title::SetSubtitle {
            text: r#"{"text":"sub"}"#.into(),
        }));
        for border in [
            s2c::WorldBorder::SetCenter { x: 8.5, z: 8.5 },
            s2c::WorldBorder::SetSize { diameter: 3.0 },
            s2c::WorldBorder::SetWarningBlocks {
                warning_blocks: i32::MAX,
            },
            s2c::WorldBorder::LerpSize {
                old_diameter: 1e300,
                new_diameter: -1e300,
                speed_millis: i64::MAX,
            },
        ] {
            let step = format!("title times {fade_in}/{stay}/{fade_out}, border {border:?}");
            rig.queue(S2CPlay::WorldBorder(border));
            for _ in 0..2 {
                log.frame(&mut screen, &mut rig, &step);
            }
        }
        for health in [f32::NAN, f32::INFINITY, 1e30] {
            rig.queue(S2CPlay::UpdateHealth(s2c::UpdateHealth {
                health,
                food_level: i32::MAX,
                saturation_level: health,
            }));
            rig.queue(S2CPlay::SetExperience(s2c::SetExperience {
                experience_bar: health,
                level: i32::MIN,
                total_experience: i32::MAX,
            }));
            rig.queue(S2CPlay::TimeUpdate(s2c::TimeUpdate {
                total_world_time: i64::MAX,
                world_time: i64::MIN,
            }));
            log.frame(
                &mut screen,
                &mut rig,
                &format!("health, food, XP, time at {health}"),
            );
            // A finite bar again, or its draw would stop every later frame at the same place.
            rig.queue(S2CPlay::SetExperience(s2c::SetExperience {
                experience_bar: 0.5,
                level: 3,
                total_experience: 40,
            }));
        }
        for diameter in [f64::NAN, f64::INFINITY, 0.0, -1.0, 6e7] {
            let x = [1e300, f64::NAN, 8.5][fade_out.rem_euclid(3) as usize];
            rig.queue(S2CPlay::WorldBorder(s2c::WorldBorder::SetSize { diameter }));
            rig.queue(S2CPlay::WorldBorder(s2c::WorldBorder::SetCenter {
                x,
                z: 8.5,
            }));
            log.frame(&mut screen, &mut rig, &format!("border {diameter} at {x}"));
        }
    }
    eprintln!("SEQ slowest hostile frame (debug build): {:?}", log.slowest);
    for overflow in &log.overflows {
        eprintln!("SEQ debug-only overflow (Java's int wraps the same way): {overflow}");
    }
    assert!(log.panics.is_empty(), "frames panicked: {:#?}", log.panics);
    assert!(
        log.slowest < Duration::from_secs(20),
        "a frame over hostile state took {:?}",
        log.slowest
    );
}

/// What the renderer holds for the world: uploaded sections, terrain sections and their GPU
/// bytes, and the mesh work still queued.
fn render_census(rig: &Rig) -> Vec<(&'static str, u64)> {
    let client = &rig.client;
    let render = client.render.as_ref().expect("installed render");
    let terrain = render.terrain.borrow();
    vec![
        ("uploaded sections", client.uploaded_sections.len() as u64),
        ("terrain sections", terrain.loaded_sections() as u64),
        ("terrain GPU bytes", terrain.gpu_memory().bytes),
        ("mesh queue", client.mesh_queue_len() as u64),
        ("mesh in flight", client.mesh_queue_in_flight_len() as u64),
    ]
}

/// java: `doPreChunk(x, z, false)` drops a column and `RenderGlobal` lets its render chunks go;
/// a destroyed entity leaves nothing to draw. Columns and entities a server sends and then takes
/// back, cycle after cycle at fresh positions and ids, must leave the renderer and the client
/// where they started (drawn in real frames, so meshes are built, uploaded and released).
#[test]
fn column_and_entity_churn_returns_the_renderer_to_its_baseline() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let Some(mut screen) = Screen::install(&mut rig) else {
        return;
    };
    let _watchdog = Watchdog::arm("render churn", Duration::from_secs(600));
    let settle = |screen: &mut Screen, rig: &mut Rig| {
        for _ in 0..200 {
            screen.frame(rig);
            if rig.client.mesh_queue_len() == 0 && rig.client.mesh_queue_in_flight_len() == 0 {
                break;
            }
        }
        for _ in 0..3 {
            screen.frame(rig);
        }
    };
    let cycle = |screen: &mut Screen, rig: &mut Rig, round: i32| {
        let columns: Vec<(i32, i32)> = (-3..=3)
            .flat_map(|x| (-3..=3).map(move |z| (x, z)))
            .filter(|&column| column != (0, 0))
            .map(|(x, z)| (x + round * 7, z))
            .collect();
        for &(x, z) in &columns {
            rig.queue(floor_column(x, z));
        }
        let ids: Vec<i32> = (0..120)
            .map(|index| 60_000 + round * 1_000 + index)
            .collect();
        for (index, &id) in ids.iter().enumerate() {
            let (x, z) = columns[index % columns.len()];
            let packet = match index % 3 {
                0 => zombie(id, (x * 16 + 8) * 32, (z * 16 + 8) * 32),
                1 => minecart(id, 0),
                _ => living(id, 30, Vec::new()),
            };
            rig.queue(packet);
        }
        settle(screen, rig);
        // The cycle must really have drawn: its columns uploaded, its entities tracked.
        let loaded = render_census(rig)[0].1;
        assert!(
            loaded >= 16 * columns.len() as u64,
            "only {loaded} sections uploaded"
        );
        assert_eq!(rig.state().entities.len(), ids.len());
        rig.queue(destroy(ids));
        for &(x, z) in &columns {
            rig.queue(unload_column(x, z));
        }
        settle(screen, rig);
    };
    cycle(&mut screen, &mut rig, 1);
    let before = (census(&rig), render_census(&rig));
    for round in 2..5 {
        cycle(&mut screen, &mut rig, round);
    }
    let after = (census(&rig), render_census(&rig));
    let growth = census_growth(&before.0, &after.0);
    assert!(growth.is_empty(), "client state left behind: {growth:?}");
    let render_growth: Vec<String> = before
        .1
        .iter()
        .zip(&after.1)
        .filter(|((_, a), (_, b))| a != b)
        .map(|((name, a), (_, b))| format!("{name}: {a} -> {b}"))
        .collect();
    eprintln!(
        "SEQ render census before {:?} after {:?}",
        before.1, after.1
    );
    assert!(
        render_growth.is_empty(),
        "render state left behind: {render_growth:?}"
    );
}

/// An item stack at the edges of what S30 can carry: odd ids, counts and damage, and NBT with a
/// long name, many lore lines, many enchantments, potion effects, a skull owner and pages.
fn extreme_stack(index: usize) -> mc_protocol::ItemStack {
    use mc_nbt::{Nbt, NbtCompound};
    let ids = [
        1_i16, 276, 358, 373, 397, 387, 403, 401, 32_000, 0, 261, 298,
    ];
    let mut tag = NbtCompound::new();
    let mut display = NbtCompound::new();
    display.insert("Name", Nbt::String("\u{a7}k".repeat(200)));
    display.insert(
        "Lore",
        Nbt::list(
            (0..300)
                .map(|line| Nbt::String(format!("lore {line}")))
                .collect(),
        ),
    );
    display.insert("color", Nbt::Int(i32::MIN));
    tag.insert("display", Nbt::Compound(display));
    tag.insert(
        "ench",
        Nbt::list(
            (0..64)
                .map(|n| {
                    let mut enchantment = NbtCompound::new();
                    let id = [i16::MIN, -1, 0, 16, 255, i16::MAX][n % 6];
                    enchantment.insert("id", Nbt::Short(id));
                    enchantment.insert("lvl", Nbt::Short([i16::MIN, -1, 0, 5, i16::MAX][n % 5]));
                    Nbt::Compound(enchantment)
                })
                .collect(),
        ),
    );
    tag.insert("HideFlags", Nbt::Int([0, -1, 63, i32::MIN][index % 4]));
    tag.insert("SkullOwner", Nbt::String("S".repeat(16)));
    let mut potion = NbtCompound::new();
    potion.insert("Id", Nbt::Byte(-128));
    potion.insert("Amplifier", Nbt::Byte(127));
    potion.insert("Duration", Nbt::Int(i32::MIN));
    tag.insert(
        "CustomPotionEffects",
        Nbt::list(vec![Nbt::Compound(potion)]),
    );
    tag.insert(
        "pages",
        Nbt::list((0..50).map(|_| Nbt::String("p".repeat(256))).collect()),
    );
    tag.insert("title", Nbt::String("T".repeat(32)));
    mc_protocol::ItemStack {
        item_id: ids[index % ids.len()],
        count: [127, -128, 0, 1, 64, -1][index % 6],
        damage: [i16::MIN, -1, 0, 1, 32_767, 16_384][index % 6],
        nbt: (!index.is_multiple_of(3)).then_some(tag),
    }
}

/// Every server-opened container type drawn in real frames with every slot holding an extreme
/// stack, every window property at the limits of a short, and the cursor swept across the GUI
/// so that tooltips, enchantment and beacon buttons and slot highlights are drawn too. Integer
/// division by a server value would panic in the user's build as well; none may.
#[test]
fn container_screens_with_extreme_properties_and_stacks_draw_without_panics() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let Some(mut screen) = Screen::install(&mut rig) else {
        return;
    };
    let _watchdog = Watchdog::arm("container screens", Duration::from_secs(900));
    let mut log = FrameLog::default();
    let windows = [
        ("minecraft:furnace", 3),
        ("minecraft:brewing_stand", 4),
        ("minecraft:enchanting_table", 0),
        ("minecraft:anvil", 0),
        ("minecraft:beacon", 1),
        ("minecraft:hopper", 5),
        ("minecraft:dispenser", 9),
        ("minecraft:crafting_table", 0),
        ("minecraft:villager", 3),
        ("minecraft:chest", 54),
        ("minecraft:container", 255),
    ];
    let mut opened = 0;
    for (index, (inventory_type, slot_count)) in windows.into_iter().enumerate() {
        let window_id = 1 + index as u8;
        rig.apply([S2CPlay::OpenWindow(s2c::OpenWindow {
            window_id,
            inventory_type: inventory_type.into(),
            window_title: format!(r#"{{"text":"{}"}}"#, "W".repeat(300)),
            slot_count,
            entity_id: None,
        })]);
        if matches!(rig.client.ui, Ui::Container(_)) {
            opened += 1;
        }
        let slots = usize::from(slot_count) + 36;
        rig.queue(S2CPlay::WindowItems(s2c::WindowItems {
            window_id,
            items: (0..slots).map(|slot| Some(extreme_stack(slot))).collect(),
        }));
        rig.queue(S2CPlay::SetSlot(s2c::SetSlot {
            window_id: -1,
            slot: -1,
            item: Some(extreme_stack(index)),
        }));
        for value in [i16::MIN, -1, 0, 1, 400, i16::MAX] {
            for var_index in 0..12 {
                rig.queue(S2CPlay::WindowProperty(s2c::WindowProperty {
                    window_id,
                    var_index,
                    var_value: value,
                }));
            }
            // Sweep the pointer over the window and its surroundings.
            for step in 0..6 {
                let x = 200.0 + f64::from(step) * 90.0 + f64::from(value).abs() % 37.0;
                let y = 40.0 + f64::from(step % 3) * 140.0;
                screen.input.on_cursor_moved(x, y);
                log.frame(
                    &mut screen,
                    &mut rig,
                    &format!("{inventory_type} property {value} at ({x}, {y})"),
                );
            }
        }
        rig.apply([S2CPlay::CloseWindow(s2c::CloseWindow { window_id })]);
    }
    assert_eq!(opened, windows.len(), "every window type opened its screen");
    // The player's own inventory with the same stacks.
    rig.queue(S2CPlay::WindowItems(s2c::WindowItems {
        window_id: 0,
        items: (0..45).map(|slot| Some(extreme_stack(slot))).collect(),
    }));
    rig.client.open_inventory();
    for step in 0..8 {
        screen.input.on_cursor_moved(
            250.0 + f64::from(step) * 40.0,
            120.0 + f64::from(step) * 20.0,
        );
        log.frame(
            &mut screen,
            &mut rig,
            &format!("inventory, pointer step {step}"),
        );
    }
    eprintln!(
        "SEQ slowest container frame (debug build): {:?}",
        log.slowest
    );
    for overflow in &log.overflows {
        eprintln!("SEQ debug-only overflow (Java's int wraps the same way): {overflow}");
    }
    assert!(log.panics.is_empty(), "frames panicked: {:#?}", log.panics);
    assert!(
        log.slowest < Duration::from_secs(20),
        "slowest frame {:?}",
        log.slowest
    );
}

/// java: `World.updateEntities` ticks each entity once and `RenderGlobal.renderEntities` draws
/// each once, so a tick and a frame cost time linear in the entity count. Named mobs, armor
/// stands, players (nameplates, skins, the Tab list) and items, all in view, at two sizes.
#[test]
fn ticks_and_frames_cost_time_linear_in_the_entity_count() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let Some(mut screen) = Screen::install(&mut rig) else {
        return;
    };
    let _watchdog = Watchdog::arm("entity count", Duration::from_secs(900));
    for x in -1..=1 {
        for z in -1..=1 {
            rig.queue(floor_column(x, z));
        }
    }
    rig.client.poll_in_place();
    let mut log = FrameLog::default();
    let mut costs = Vec::new();
    let sizes: Vec<i32> = std::env::var("SEQ_ENTITY_COUNTS")
        .ok()
        .and_then(|sizes| sizes.split(',').map(|n| n.parse().ok()).collect())
        .unwrap_or_else(|| vec![500, 4_000]);
    for (round, count) in sizes.into_iter().enumerate() {
        let base = 200_000 + round as i32 * 100_000;
        let ids: Vec<i32> = (0..count).map(|index| base + index).collect();
        let mut players = Vec::new();
        for (index, &id) in ids.iter().enumerate() {
            // A block of entities in front of the camera (which looks along +z from (8.5, 8.5)):
            // 32 across, 16 deep and stacked in layers, inside the loaded columns.
            let index = index as i32;
            let (x, z) = (16 + (index % 32) * 16, (14 + (index / 32) % 16) * 32);
            let y = (16 + (index / 512) * 2) * 32;
            let named = vec![
                MetadataEntry {
                    index: 2,
                    value: MetadataValue::String(format!("n{index}")),
                },
                MetadataEntry {
                    index: 3,
                    value: MetadataValue::Byte(1),
                },
            ];
            // Diagnosis: `SEQ_ENTITY_KIND=<0..3>` spawns that one kind only.
            let kind = std::env::var("SEQ_ENTITY_KIND")
                .ok()
                .and_then(|kind| kind.parse::<usize>().ok())
                .unwrap_or(index as usize % 4);
            let packet = match kind {
                0 => S2CPlay::SpawnMob(s2c::SpawnMob {
                    entity_id: id,
                    entity_type: 54,
                    x,
                    y,
                    z,
                    yaw: Angle(0),
                    pitch: Angle(0),
                    head_pitch: Angle(0),
                    velocity_x: 0,
                    velocity_y: 0,
                    velocity_z: 0,
                    metadata: named,
                }),
                1 => S2CPlay::SpawnMob(s2c::SpawnMob {
                    entity_id: id,
                    entity_type: 30,
                    x,
                    y,
                    z,
                    yaw: Angle(0),
                    pitch: Angle(0),
                    head_pitch: Angle(0),
                    velocity_x: 0,
                    velocity_y: 0,
                    velocity_z: 0,
                    metadata: named,
                }),
                2 => {
                    let uuid = Uuid::from_u128(0xE17_0000_0000 + id as u128);
                    players.push(s2c::AddPlayerEntry {
                        uuid,
                        username: format!("q{index}"),
                        properties: Vec::new(),
                        game_mode: s2c::GameMode::Survival,
                        ping: 50,
                        display_name: None,
                    });
                    S2CPlay::SpawnPlayer(s2c::SpawnPlayer {
                        entity_id: id,
                        player_uuid: uuid,
                        x,
                        y,
                        z,
                        yaw: Angle(0),
                        pitch: Angle(0),
                        current_item: 276,
                        metadata: Vec::new(),
                    })
                }
                _ => S2CPlay::SpawnObject(s2c::SpawnObject {
                    entity_id: id,
                    object_type: 2,
                    x,
                    y,
                    z,
                    pitch: Angle(0),
                    yaw: Angle(0),
                    data: 1,
                    velocity: Some((0, 0, 0)),
                }),
            };
            rig.queue(packet);
        }
        rig.queue(S2CPlay::PlayerListItem(s2c::PlayerListItem::AddPlayer(
            players,
        )));
        rig.client.poll_in_place();
        assert!(
            ids.iter().all(|&id| rig.state().entities.is_resident(id)),
            "every entity is in a loaded column, so it is ticked and drawn"
        );
        // Settle (meshes, first-frame uploads), then measure ticks and frames separately.
        for _ in 0..5 {
            log.frame(
                &mut screen,
                &mut rig,
                &format!("{count} entities, settling"),
            );
        }
        let start = Instant::now();
        rig.ticks(20);
        let ticks = start.elapsed();
        let start = Instant::now();
        let (build_before, render_before) = (screen.build_time, screen.render_time);
        // Diagnosis only: `SEQ_PROFILE=1 MC_PROFILE=1` (with this test filtered alone) prints
        // the client's own profile scopes for the measured frames.
        let profile = std::env::var_os("SEQ_PROFILE").is_some();
        if profile {
            if round == 0 {
                mc_core::init_tracing();
            }
            mc_core::profile::reset();
        }
        for _ in 0..10 {
            log.frame(&mut screen, &mut rig, &format!("{count} entities"));
        }
        let frames = start.elapsed();
        if profile {
            let rows = mc_core::profile::snapshot();
            eprint!(
                "{}",
                mc_core::profile::format_table(&format!("{count} entities"), &rows, frames)
            );
        }
        // The frustum decides how many of them are drawn (the F3 "E:" count), so the frame
        // cost is compared per drawn entity; the tick walks all of them.
        let drawn = rig.client.debug_entities.rendered.max(1);
        let build = screen.build_time - build_before;
        eprintln!(
            "SEQ {count} entities ({drawn} drawn): 20 ticks {ticks:?}, 10 frames {frames:?} \
             (build {build:?}, render {:?})",
            screen.render_time - render_before
        );
        costs.push((ticks, build / u32::try_from(drawn).unwrap_or(u32::MAX)));
        rig.apply([destroy(ids)]);
    }
    assert!(log.panics.is_empty(), "frames panicked: {:#?}", log.panics);
    let ratio =
        |big: Duration, small: Duration| big.as_secs_f64() / small.as_secs_f64().max(0.000_001);
    // 8x the entities: a linear tick is 8x, a quadratic one 64x.
    assert!(
        ratio(costs[1].0, costs[0].0) < 24.0 || costs[1].0 < Duration::from_millis(200),
        "ticks: {costs:?}"
    );
    // Per drawn entity, a frame's cost must not grow with the count (quadratic would be 8x).
    assert!(
        ratio(costs[1].1, costs[0].1) < 4.0,
        "frame cost per drawn entity: {costs:?}"
    );
}

/// The HUD's server-driven values at their extremes, with ticks in between: titles with
/// negative and huge times, a world border with non-finite or inverted sizes and lerps,
/// health, food, experience, abilities, time and weather at their limits.
#[test]
fn extreme_hud_and_world_values_survive_ticking() {
    let Some(mut rig) = Rig::joined() else {
        return;
    };
    let _watchdog = Watchdog::arm("extreme values", Duration::from_secs(300));
    let floats = [
        0.0_f32,
        -0.0,
        1.0,
        -1.0,
        1e30,
        -1e30,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::MIN_POSITIVE,
    ];
    let doubles = floats.map(f64::from);
    let ints = [0, 1, -1, i32::MAX, i32::MIN, 72_000, -72_000];
    for (index, &value) in floats.iter().enumerate() {
        let int = ints[index % ints.len()];
        let double = doubles[index];
        rig.apply([
            S2CPlay::Title(s2c::Title::SetTimes {
                fade_in_ticks: int,
                stay_ticks: ints[(index + 1) % ints.len()],
                fade_out_ticks: ints[(index + 2) % ints.len()],
            }),
            S2CPlay::Title(s2c::Title::SetTitle {
                text: r#"{"text":"t"}"#.into(),
            }),
            S2CPlay::Title(s2c::Title::SetSubtitle {
                text: r#"{"text":"s"}"#.into(),
            }),
            S2CPlay::WorldBorder(s2c::WorldBorder::SetSize { diameter: double }),
            S2CPlay::WorldBorder(s2c::WorldBorder::LerpSize {
                old_diameter: double,
                new_diameter: -double,
                speed_millis: i64::from(int) * 1_000,
            }),
            S2CPlay::WorldBorder(s2c::WorldBorder::SetCenter {
                x: double,
                z: -double,
            }),
            S2CPlay::WorldBorder(s2c::WorldBorder::SetWarningTime { warning_time: int }),
            S2CPlay::WorldBorder(s2c::WorldBorder::SetWarningBlocks {
                warning_blocks: int,
            }),
            S2CPlay::UpdateHealth(s2c::UpdateHealth {
                health: value,
                food_level: int,
                saturation_level: value,
            }),
            S2CPlay::SetExperience(s2c::SetExperience {
                experience_bar: value,
                level: int,
                total_experience: int,
            }),
            S2CPlay::PlayerAbilities(s2c::PlayerAbilities {
                flags: 0x0f,
                fly_speed: value,
                walk_speed: value,
            }),
            S2CPlay::TimeUpdate(s2c::TimeUpdate {
                total_world_time: i64::from(int) * 1_000_003,
                world_time: if index.is_multiple_of(2) {
                    i64::MIN
                } else {
                    i64::MAX
                },
            }),
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::RainLevelChange,
                value: value.clamp(-20.0, 20.0),
            }),
            S2CPlay::ChangeGameState(s2c::ChangeGameState {
                reason: s2c::GameStateReason::ThunderLevelChange,
                value: value.clamp(-20.0, 20.0),
            }),
            S2CPlay::SpawnPosition(s2c::SpawnPosition {
                position: BlockPos::new(int.clamp(-33_554_432, 33_554_431), 2047, -2048),
            }),
            S2CPlay::HeldItemChange(s2c::HeldItemChange {
                slot: (index as i8) - 4,
            }),
        ]);
        rig.ticks(3);
    }
    rig.apply([
        S2CPlay::UpdateHealth(s2c::UpdateHealth {
            health: 20.0,
            food_level: 20,
            saturation_level: 5.0,
        }),
        S2CPlay::WorldBorder(s2c::WorldBorder::SetSize {
            diameter: 60_000_000.0,
        }),
    ]);
    rig.ticks(3);
}
