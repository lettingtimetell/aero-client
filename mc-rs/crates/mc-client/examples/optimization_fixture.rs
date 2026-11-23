//! Deterministic protocol fixtures for the O0 optimization harness.
//!
//! This example writes a synthetic, diagnostic-only `.mccap` plus a sidecar recipe.
//! It uses the real protocol packet encoders and the repository's extracted 1.8.9
//! block/biome registry.  It does not claim image qualification or performance.
//!
//! ```text
//! cargo run -p mc-client --example optimization_fixture -- --scene v1 --output out/v1.mccap
//! cargo run -p mc-client --example optimization_fixture -- --scene e1-50 --output out/e1-50.mccap
//! cargo run -p mc-client --example optimization_fixture -- --scene e1-100 --output out/e1-100.mccap
//! cargo run -p mc-client --example optimization_fixture -- --scene lifecycle-gap --output out/lifecycle-gap.mccap
//! ```

#[cfg(test)]
use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use mc_data::{BlockState, biomes, block_by_name};
use mc_math::BlockPos;
#[cfg(test)]
use mc_math::ChunkPos;
use mc_parity::{Capture, CaptureMeta, Direction, Recorder};
use mc_protocol::login::LoginSuccess;
use mc_protocol::packet::{Packet, State};
use mc_protocol::play::s2c;
use mc_protocol::{Angle, MetadataEntry, MetadataValue, PacketBuffer, chunk};
use serde_json::{Value, json};
use uuid::Uuid;

const PROTOCOL: i32 = 47;
const LOCAL_ENTITY_ID: i32 = 1;
const FIRST_REMOTE_ENTITY_ID: i32 = 100;
const LOCAL_UUID: u128 = 0x0000_0000_0000_4000_8000_0000_0000_0001;
const ISLAND_SURFACE_Y: i32 = 63;
const ISLAND_RADIUS: i32 = 20;
const TERRAIN_MIN_Y: i32 = 60;
const TERRAIN_MAX_Y: i32 = 63;
/// With `--underground`: a full stone layer across every loaded column, and the players below it.
const SEALED_FLOOR_Y: i32 = 48;
const UNDERGROUND_PLAYER_Y: i32 = 30;
/// Mojang's real signed `textures` payload, identical to the vector `mc-auth/src/hash.rs` tests
/// verify. Offline servers send no signatures, so a fixture without this cannot exercise the
/// signature and profile path that every online-mode server hits.
const SIGNED_TEXTURES_VALUE: &str = "ewogICJ0aW1lc3RhbXAiIDogMTc4NzY5MTA3NTE3MiwKICAicHJvZmlsZUlkIiA6ICIwNjlhNzlmNDQ0ZTk0NzI2YTViZWZjYTkwZTM4YWFmNSIsCiAgInByb2ZpbGVOYW1lIiA6ICJOb3RjaCIsCiAgInNpZ25hdHVyZVJlcXVpcmVkIiA6IHRydWUsCiAgInRleHR1cmVzIiA6IHsKICAgICJTS0lOIiA6IHsKICAgICAgInVybCIgOiAiaHR0cDovL3RleHR1cmVzLm1pbmVjcmFmdC5uZXQvdGV4dHVyZS8yOTIwMDlhNDkyNWI1OGYwMmM3N2RhZGMzZWNlZjA3ZWE0Yzc0NzJmNjRlMGZkYzMyY2U1NTIyNDg5MzYyNjgwIgogICAgfQogIH0KfQ==";
const SIGNED_TEXTURES_SIGNATURE: &str = "oqgTiH1ZWCjCcU9a5bOJpskVzloygYgbdNZ16Xtn1RgKrRsPLutWL9Uok/rkg4re1O+TbsKL0lKYzQtZH0+OfWdfWIoWE4bBanHuhejGWu69EWVvcxlpEiNhZptzER9KfPGP5Q7SPsT+4MaSj/AbJP+yKWNjriYZZFV00bWSW/aexY6PJSoXMD/R57gn0TucdaZp1nrbfgL9DXf+6R6T8ou2Lz4qsPAoB2CJI2I8F9HPsLJ/dUinrp8gmjfKnvvJUf+a6XeG+BESWqqChuiD+XJjvnZSyM3Fs1zdb+RkqvFNqT4W14ohyh3mxyX3Q2S/8S+Q2axT47nnjvy9WMVpaDay1o8pquzy+zRqPps1tRFTScrp4w4jVdSzm8xL4FhNqm2H/QUoAYczVUyAfBx4Q8w9daelERlNOcQAIEeauBzan/728P/F4Rxc24Wl9iV5Qb+XeE/858QHNplwLIcSv4Wfp6G29++0XVljN7K2Vtbilp9LipVrDG028XMFY6eY/lb1n9C4hiMWF/N4YTXz8Tm+IBIBqKpgeUCzb3ZmB7bvVubPUgWdL3K7piiClRQ8ssIAN67DEOm/sTgyv0eFitWiZopPhBvgizOBPDv2/GSa3WdFRiNAtaxrqOZVO3W4HUn62XNz+leI640BMjAYMvIcpNU2x0R+S8ILf7v0zcM=";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scene {
    V1,
    E1 {
        players: usize,
    },
    /// A fixed-tick terrain lifecycle diagnostic; it is never a score fixture.
    Lifecycle,
    /// The lifecycle diagnostic with a corridor unload/reload extension.
    LifecycleGap,
}

impl Scene {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw {
            "v1" => Ok(Self::V1),
            "e1-1" => Ok(Self::E1 { players: 1 }),
            "e1-16" => Ok(Self::E1 { players: 16 }),
            "e1-50" => Ok(Self::E1 { players: 50 }),
            "e1-100" => Ok(Self::E1 { players: 100 }),
            "lifecycle" | "diagnostic-terrain" => Ok(Self::Lifecycle),
            "lifecycle-gap" => Ok(Self::LifecycleGap),
            other => Err(format!(
                "--scene must be v1, e1-1, e1-16, e1-50, e1-100, lifecycle, or lifecycle-gap; got {other:?}"
            )),
        }
    }

    const fn id(self) -> &'static str {
        match self {
            Self::V1 => "V1",
            Self::E1 { players: 50 } => "E1-50",
            Self::E1 { players: 100 } => "E1-100",
            Self::E1 { .. } => "E1",
            Self::Lifecycle => "DIAGNOSTIC_TERRAIN",
            Self::LifecycleGap => "DIAGNOSTIC_TERRAIN_GAP",
        }
    }

    const fn player_count(self) -> usize {
        match self {
            Self::V1 => 8,
            Self::E1 { players } => players,
            Self::Lifecycle | Self::LifecycleGap => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct PlayerPlan {
    entity_id: i32,
    uuid: Uuid,
    username: String,
    x: i32,
    y: i32,
    z: i32,
    yaw: u8,
    /// Explicit S19 head pose emitted immediately after the S0C spawn.
    head_yaw: u8,
}

#[derive(Debug, Clone, PartialEq)]
struct WorldPlan {
    columns: Vec<(i32, i32, u16)>,
    stone_state: u16,
    dirt_state: u16,
    grass_state: u16,
    plains_biome: u8,
    solid_blocks: usize,
    loaded_air_sections: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct FixturePlan {
    /// Attach Mojang's signed `textures` property to every roster row, including the local player.
    signed_textures: bool,
    /// Add a fight's worth of sound effects and one streamed record to the timeline.
    sound_burst: bool,
    /// Seal the players under a stone layer, where the culling walk cannot reach them.
    underground: bool,
    scene: Scene,
    players: Vec<PlayerPlan>,
    world: WorldPlan,
    local_position: (i32, i32, i32),
}

const LIFECYCLE_STEP_MICROS: u64 = 50_000;
const LIFECYCLE_INITIAL_CHECKPOINT: u64 = 200;
const LIFECYCLE_TRANSITION_WINDOW: u64 = 120;
const LIFECYCLE_FINAL_CHECKPOINT: u64 = 1_280;
const LIFECYCLE_GAP_UNLOAD_TICK: u64 = 1_282;
const LIFECYCLE_GAP_RELOAD_TICK: u64 = 1_402;
const LIFECYCLE_GAP_CHECKPOINT: u64 = 1_401;
const LIFECYCLE_GAP_FINAL_CHECKPOINT: u64 = 1_521;
const LIFECYCLE_SECTION_MASK: u16 = u16::MAX;
const LIFECYCLE_PLATFORM_SECTION: i32 = 4;
const LIFECYCLE_PLATFORM_Y: i32 = 64;

/// The 24-column lifecycle topology.  A/B/C are two-by-two platform islands;
/// Hx/Hz are all-air corridors that keep the loaded-section graph connected.
const LIFECYCLE_COLUMNS: [(i32, i32); 24] = [
    // A
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    // B
    (8, 0),
    (9, 0),
    (8, 1),
    (9, 1),
    // C
    (0, 8),
    (1, 8),
    (0, 9),
    (1, 9),
    // Hx
    (2, 0),
    (3, 0),
    (4, 0),
    (5, 0),
    (6, 0),
    (7, 0),
    // Hz
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (0, 6),
    (0, 7),
];

#[derive(Debug, Clone, PartialEq, Eq)]
struct LifecyclePlan {
    stone_state: u16,
    plains_biome: u8,
    columns: Vec<(i32, i32)>,
    platform_columns: Vec<(i32, i32)>,
    loaded_sections: usize,
    air_sections: usize,
}

fn lifecycle_plan() -> LifecyclePlan {
    let stone_state = registry_state("stone", 0).id;
    let plains_biome = registry_biome("Plains");
    let columns = LIFECYCLE_COLUMNS.to_vec();
    let platform_columns = columns
        .iter()
        .copied()
        .filter(|&(x, z)| (x < 2 && z < 2) || (x >= 8 && z < 2) || (x < 2 && z >= 8))
        .collect::<Vec<_>>();
    let loaded_sections = columns.len() * 16;
    let air_sections = loaded_sections - platform_columns.len();
    LifecyclePlan {
        stone_state,
        plains_biome,
        columns,
        platform_columns,
        loaded_sections,
        air_sections,
    }
}

fn lifecycle_has_platform(plan: &LifecyclePlan, chunk_x: i32, chunk_z: i32) -> bool {
    plan.platform_columns
        .iter()
        .any(|&(x, z)| x == chunk_x && z == chunk_z)
}

fn lifecycle_block_at(
    plan: &LifecyclePlan,
    chunk_x: i32,
    chunk_z: i32,
    section: i32,
    index: usize,
) -> u16 {
    if section != LIFECYCLE_PLATFORM_SECTION || !lifecycle_has_platform(plan, chunk_x, chunk_z) {
        return 0;
    }
    let local_y = (index >> 8) as i32;
    // Keep a one-block platform at y=64 while every Hx/Hz section stays air.
    if local_y == 0 { plan.stone_state } else { 0 }
}

fn lifecycle_chunk_payload(plan: &LifecyclePlan, chunk_x: i32, chunk_z: i32, mask: u16) -> Vec<u8> {
    let sections = mask.count_ones() as usize;
    let expected = mc_world::expected_chunk_data_len(mask, true, true);
    let mut blocks = Vec::with_capacity(sections * 8192);
    let block_light = vec![0u8; sections * 2048];
    let mut sky_light = vec![0u8; sections * 2048];
    let mut selected_section = 0usize;
    for section in 0..16i32 {
        if mask & (1 << section) == 0 {
            continue;
        }
        let section_block_start = blocks.len();
        blocks.resize(section_block_start + 8192, 0);
        for index in 0..4096usize {
            let state = lifecycle_block_at(plan, chunk_x, chunk_z, section, index);
            let offset = section_block_start + index * 2;
            blocks[offset] = (state & 0xFF) as u8;
            blocks[offset + 1] = (state >> 8) as u8;
            // Sky light is encoded as 4096 nibbles (2048 bytes) per selected
            // section.  The section stride is deliberately 4096, since
            // set_nibble converts the nibble index to a byte index.
            let global_y = section * 16 + (index >> 8) as i32;
            let sky = if lifecycle_has_platform(plan, chunk_x, chunk_z)
                && global_y <= LIFECYCLE_PLATFORM_Y
            {
                0
            } else {
                15
            };
            set_nibble(&mut sky_light, selected_section * 4096 + index, sky);
        }
        selected_section += 1;
    }
    let mut payload = blocks;
    payload.extend_from_slice(&block_light);
    payload.extend_from_slice(&sky_light);
    payload.extend(std::iter::repeat_n(plan.plains_biome, 256));
    assert_eq!(payload.len(), expected);
    payload
}

fn lifecycle_tick_micros(tick: u64, ordinal: usize, packet_count: usize) -> u64 {
    assert!(packet_count > 0 && ordinal < packet_count);
    tick * LIFECYCLE_STEP_MICROS - packet_count as u64 + ordinal as u64
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LifecycleCamera {
    tick: u64,
    x: f64,
    y: f64,
    z: f64,
    yaw: f32,
    pitch: f32,
    label: &'static str,
}

const LIFECYCLE_CAMERAS: [LifecycleCamera; 4] = [
    LifecycleCamera {
        tick: 1,
        x: 16.5,
        y: 65.0,
        z: 16.5,
        yaw: -90.0,
        pitch: 0.0,
        label: "A_plus_x",
    },
    LifecycleCamera {
        tick: 201,
        x: 16.5,
        y: 65.0,
        z: 16.5,
        yaw: 0.0,
        pitch: 0.0,
        label: "A_plus_z",
    },
    LifecycleCamera {
        tick: 321,
        x: 144.5,
        y: 65.0,
        z: 16.5,
        yaw: 90.0,
        pitch: 0.0,
        label: "B_minus_x",
    },
    LifecycleCamera {
        tick: 801,
        x: 16.5,
        y: 65.0,
        z: 144.5,
        yaw: 180.0,
        pitch: 0.0,
        label: "C_minus_z",
    },
];

fn lifecycle_columns_json(columns: &[(i32, i32)]) -> Vec<Value> {
    columns
        .iter()
        .map(|&(x, z)| json!({"chunk_x": x, "chunk_z": z, "sections": 16}))
        .collect()
}

fn lifecycle_checkpoint(
    tick: u64,
    label: &'static str,
    resident_columns: usize,
    required_visible: &[&'static str],
    required_resident: &[&'static str],
    forbidden_visible: &[&'static str],
    forbidden_resident: &[&'static str],
) -> Value {
    json!({
        "tick": tick,
        "label": label,
        "resident_columns": resident_columns,
        "resident_sections": resident_columns * 16,
        "required_visible_sentinels": required_visible,
        "required_resident_groups": required_resident,
        "forbidden_visible_sentinels": forbidden_visible,
        "forbidden_resident_groups": forbidden_resident,
        "identity_policy": "verify exact resident set and visible uniqueness/subset/sentinels; record visible order without reference-order qualification; track all-air corridors as residency",
    })
}

fn registry_state(name: &str, meta: u8) -> BlockState {
    BlockState::new(
        block_by_name(name).unwrap_or_else(|| panic!("mc-data registry missing block {name}")),
        meta,
    )
}

fn registry_biome(name: &str) -> u8 {
    biomes()
        .find(|biome| biome.name == name)
        .unwrap_or_else(|| panic!("mc-data registry missing biome {name}"))
        .id
}

fn build_plan(scene: Scene) -> FixturePlan {
    let stone = registry_state("stone", 0).id;
    let dirt = registry_state("dirt", 0).id;
    // 1.8.9 calls the block "grass"; this is the actual registry name, not a
    // hand-written packed state id.
    let grass = registry_state("grass", 0).id;
    let plains = registry_biome("Plains");
    let players = (0..scene.player_count())
        .map(|index| {
            let grid_width = 10i32;
            let x = (index as i32 % grid_width) * 2 - 9;
            let z = (index as i32 / grid_width) * 2 - 9;
            let yaw = (index as u8).wrapping_mul(32);
            PlayerPlan {
                entity_id: FIRST_REMOTE_ENTITY_ID + index as i32,
                uuid: Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0001_0000 + index as u128),
                username: match scene {
                    Scene::V1 => format!("V1P{index:02}"),
                    Scene::E1 { .. } => format!("E1P{index:03}"),
                    Scene::Lifecycle | Scene::LifecycleGap => format!("LIFE{index:03}"),
                },
                x: x * 32 + 16,
                y: (ISLAND_SURFACE_Y + 1) * 32,
                z: z * 32 + 16,
                yaw,
                head_yaw: yaw,
            }
        })
        .collect::<Vec<_>>();

    let mut columns = Vec::new();
    let mut solid_blocks = 0usize;
    let mut loaded_air_sections = 0usize;
    for chunk_x in -2..=2 {
        for chunk_z in -2..=2 {
            // Load sections 0..4 in every column so the radius-20 island is not
            // clipped at the central 3x3 boundary and the air above its surface
            // carries an explicit sky-light array.  Most sections are deliberately
            // all-air, which makes the loaded-air contract explicit.
            let mask = 0x001F;
            for section in 0..5i32 {
                let mut section_has_solid = false;
                for local_y in 0..16i32 {
                    let y = section * 16 + local_y;
                    for local_z in 0..16i32 {
                        for local_x in 0..16i32 {
                            let x = chunk_x * 16 + local_x;
                            let z = chunk_z * 16 + local_z;
                            if (TERRAIN_MIN_Y..=TERRAIN_MAX_Y).contains(&y)
                                && x * x + z * z <= ISLAND_RADIUS * ISLAND_RADIUS
                            {
                                solid_blocks += 1;
                                section_has_solid = true;
                            }
                        }
                    }
                }
                if !section_has_solid {
                    loaded_air_sections += 1;
                }
            }
            columns.push((chunk_x, chunk_z, mask));
        }
    }

    FixturePlan {
        signed_textures: false,
        sound_burst: false,
        underground: false,
        scene,
        players,
        world: WorldPlan {
            columns,
            stone_state: stone,
            dirt_state: dirt,
            grass_state: grass,
            plains_biome: plains,
            solid_blocks,
            loaded_air_sections,
        },
        // Intentional diagnostic camera/player feet: inside the island at a
        // supported block, looking +Z, so both the V1 row and E1 10x10 grid fit
        // the documented 854x480/FOV70 framing target. Pixel visibility remains
        // unqualified.
        local_position: (16, 64 * 32, -18 * 32),
    }
}

fn block_at(plan: &FixturePlan, x: i32, y: i32, z: i32) -> u16 {
    if plan.underground && y == SEALED_FLOOR_Y {
        return plan.world.stone_state;
    }
    if x * x + z * z > ISLAND_RADIUS * ISLAND_RADIUS {
        return 0;
    }
    match y {
        60 | 61 => plan.world.stone_state,
        62 => plan.world.dirt_state,
        63 => plan.world.grass_state,
        _ => 0,
    }
}

fn set_nibble(bytes: &mut [u8], index: usize, value: u8) {
    let byte = &mut bytes[index >> 1];
    if index & 1 == 0 {
        *byte = (*byte & 0xF0) | (value & 0x0F);
    } else {
        *byte = (*byte & 0x0F) | ((value & 0x0F) << 4);
    }
}

fn chunk_payload(plan: &FixturePlan, chunk_x: i32, chunk_z: i32, mask: u16) -> Vec<u8> {
    let sections = mask.count_ones() as usize;
    let expected = mc_world::expected_chunk_data_len(mask, true, true);
    let mut blocks = Vec::with_capacity(sections * 8192);
    let block_light = vec![0u8; sections * 2048];
    let mut sky_light = vec![0u8; sections * 2048];
    let mut selected_section = 0usize;
    for section in 0..16i32 {
        if mask & (1 << section) == 0 {
            continue;
        }
        let section_block_start = blocks.len();
        blocks.resize(section_block_start + 8192, 0);
        for local_y in 0..16i32 {
            for local_z in 0..16i32 {
                for local_x in 0..16i32 {
                    let x = chunk_x * 16 + local_x;
                    let z = chunk_z * 16 + local_z;
                    let y = section * 16 + local_y;
                    let state = block_at(plan, x, y, z);
                    let index =
                        ((local_y as usize) << 8) | ((local_z as usize) << 4) | local_x as usize;
                    let offset = section_block_start + index * 2;
                    blocks[offset] = (state & 0xFF) as u8;
                    blocks[offset + 1] = (state >> 8) as u8;
                    let top = if x * x + z * z <= ISLAND_RADIUS * ISLAND_RADIUS {
                        ISLAND_SURFACE_Y
                    } else {
                        -1
                    };
                    let light = if y > top { 15 } else { 0 };
                    // `index` is a nibble index; set_nibble divides it by two to
                    // address the byte array.  Keep each section's nibble span at
                    // 4096 entries (2048 bytes), rather than overlapping adjacent
                    // sections by using a byte stride here.
                    let light_index = selected_section * 4096 + index;
                    set_nibble(&mut sky_light, light_index, light);
                }
            }
        }
        selected_section += 1;
    }
    let mut payload = blocks;
    payload.extend_from_slice(&block_light);
    payload.extend_from_slice(&sky_light);
    payload.extend(std::iter::repeat_n(plan.world.plains_biome, 256));
    assert_eq!(payload.len(), expected);
    payload
}

fn encode_packet<P: Packet>(packet: &P) -> Vec<u8> {
    let mut bytes = PacketBuffer::new();
    bytes.write_varint(P::ID);
    packet.write(&mut bytes);
    bytes.as_slice().to_vec()
}

fn record_packet<P: Packet>(recorder: &mut Recorder, state: State, micros: u64, packet: &P) {
    recorder.record_at(
        Direction::Clientbound,
        state,
        micros,
        &encode_packet(packet),
    );
}

fn record_lifecycle_camera(recorder: &mut Recorder, camera: LifecycleCamera) {
    record_packet(
        recorder,
        State::Play,
        lifecycle_tick_micros(camera.tick, 0, 1),
        &s2c::PlayerPositionLook {
            x: camera.x,
            y: camera.y,
            z: camera.z,
            yaw: camera.yaw,
            pitch: camera.pitch,
            flags: 0,
        },
    );
}

fn record_lifecycle_chunk_batch(
    recorder: &mut Recorder,
    plan: &LifecyclePlan,
    tick: u64,
    columns: &[(i32, i32)],
    mask: u16,
) {
    let packet_count = columns.len();
    for (ordinal, &(chunk_x, chunk_z)) in columns.iter().enumerate() {
        record_packet(
            recorder,
            State::Play,
            lifecycle_tick_micros(tick, ordinal, packet_count),
            &chunk::ChunkData {
                chunk_x,
                chunk_z,
                ground_up_continuous: true,
                primary_bit_mask: mask,
                data: lifecycle_chunk_payload(plan, chunk_x, chunk_z, mask),
            },
        );
    }
}

fn lifecycle_group(plan: &LifecyclePlan, name: &str) -> Vec<(i32, i32)> {
    plan.columns
        .iter()
        .copied()
        .filter(|&(x, z)| match name {
            "A" => x < 2 && z < 2,
            "B" => x >= 8 && z < 2,
            "C" => x < 2 && z >= 8,
            "Hx" => (2..=7).contains(&x) && z == 0,
            "Hz" => x == 0 && (2..=7).contains(&z),
            _ => false,
        })
        .collect()
}

#[cfg(test)]
fn build_lifecycle_capture(plan: &LifecyclePlan) -> Capture {
    build_lifecycle_capture_variant(plan, false)
}

fn build_lifecycle_capture_variant(plan: &LifecyclePlan, gap_variant: bool) -> Capture {
    let mut recorder = Recorder::new(CaptureMeta {
        protocol: PROTOCOL,
        address: "synthetic-diagnostic-terrain".into(),
        started_at_unix_millis: 0,
    });
    record_packet(
        &mut recorder,
        State::Login,
        0,
        &LoginSuccess {
            uuid: Uuid::from_u128(LOCAL_UUID),
            username: "LifecycleLocal".into(),
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_000,
        &s2c::JoinGame {
            entity_id: LOCAL_ENTITY_ID,
            hardcore_mode: false,
            game_mode: 1,
            dimension: 0,
            difficulty: 2,
            max_players: 1,
            level_type: "default".into(),
            reduced_debug_info: false,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_100,
        &s2c::TimeUpdate {
            total_world_time: 6_000,
            world_time: 6_000,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_200,
        &s2c::UpdateHealth {
            health: 20.0,
            food_level: 20,
            saturation_level: 5.0,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_300,
        &s2c::SpawnPosition {
            position: BlockPos::new(16, LIFECYCLE_PLATFORM_Y, 16),
        },
    );
    // Keep the diagnostic avatar at the scripted feet without asking the live
    // input path to synthesize movement.  This is a typed protocol capability
    // packet, and ordinary captures never enter this scene.
    record_packet(
        &mut recorder,
        State::Play,
        1_400,
        &s2c::PlayerAbilities {
            flags: 0x0e, // flying + allowFlying + creativeMode
            fly_speed: 0.05,
            walk_speed: 0.1,
        },
    );

    let initial = plan.columns.as_slice();
    record_lifecycle_chunk_batch(&mut recorder, plan, 1, initial, LIFECYCLE_SECTION_MASK);
    record_lifecycle_camera(&mut recorder, LIFECYCLE_CAMERAS[0]);
    record_lifecycle_camera(&mut recorder, LIFECYCLE_CAMERAS[1]);
    record_lifecycle_camera(&mut recorder, LIFECYCLE_CAMERAS[2]);

    let group_b = lifecycle_group(plan, "B");
    record_lifecycle_chunk_batch(&mut recorder, plan, 441, &group_b, 0);
    record_lifecycle_chunk_batch(&mut recorder, plan, 561, &group_b, LIFECYCLE_SECTION_MASK);
    // A paired edit exercises block-change invalidation without changing the
    // final geometry at the 800-frame checkpoint.
    let edit_position = BlockPos::new(8 * 16, LIFECYCLE_PLATFORM_Y, 0);
    record_packet(
        &mut recorder,
        State::Play,
        lifecycle_tick_micros(681, 0, 1),
        &s2c::BlockChange {
            position: edit_position,
            block_state_id: 0,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        lifecycle_tick_micros(682, 0, 1),
        &s2c::BlockChange {
            position: edit_position,
            block_state_id: i32::from(plan.stone_state),
        },
    );
    record_lifecycle_camera(&mut recorder, LIFECYCLE_CAMERAS[3]);

    let group_a = lifecycle_group(plan, "A");
    record_lifecycle_chunk_batch(&mut recorder, plan, 921, &group_a, 0);
    record_lifecycle_chunk_batch(&mut recorder, plan, 1_041, &group_a, LIFECYCLE_SECTION_MASK);
    if gap_variant {
        // Return to A, then remove only Hx=(4,0) after the base checkpoints.
        // The typed camera packet makes the culler face +X from the A platform;
        // the unload/reload pair isolates the corridor residency regression.
        record_lifecycle_camera(
            &mut recorder,
            LifecycleCamera {
                tick: 1_281,
                x: 16.5,
                y: 65.0,
                z: 16.5,
                yaw: -90.0,
                pitch: 0.0,
                label: "gap_A_plus_x",
            },
        );
        record_lifecycle_chunk_batch(&mut recorder, plan, LIFECYCLE_GAP_UNLOAD_TICK, &[(4, 0)], 0);
        record_lifecycle_chunk_batch(
            &mut recorder,
            plan,
            LIFECYCLE_GAP_RELOAD_TICK,
            &[(4, 0)],
            LIFECYCLE_SECTION_MASK,
        );
    }
    recorder.finish()
}

#[cfg(test)]
fn lifecycle_recipe(plan: &LifecyclePlan) -> Value {
    lifecycle_recipe_variant(plan, false)
}

fn lifecycle_recipe_variant(plan: &LifecyclePlan, gap_variant: bool) -> Value {
    let group_a = lifecycle_group(plan, "A");
    let group_b = lifecycle_group(plan, "B");
    let group_c = lifecycle_group(plan, "C");
    let corridor_x = lifecycle_group(plan, "Hx");
    let corridor_z = lifecycle_group(plan, "Hz");
    let mut recipe = json!({
        "contract": "mc-rs.optimization.synthetic-lifecycle-fixture",
        "version": 1,
        "fixture_id": "DIAGNOSTIC_TERRAIN",
        "status": "DIAGNOSTIC",
        "scoring_eligible": false,
        "disclaimer": "Typed protocol lifecycle diagnostic; it certifies packet/world identity contracts only and is not a Java renderer or performance reference.",
        "protocol": PROTOCOL,
        "clock": {
            "simulation": "fixed_20hz",
            "step_micros": LIFECYCLE_STEP_MICROS,
            "capture": "synthetic explicit microseconds with strict same-tick ordering",
            "input": "scripted typed PositionLook camera positions; no OS input; local abilities hold flight",
            "rng": "none (no entities, particles, or emitters)",
            "readiness": "fixed checkpoint; never extended by observed readiness"
        },
        "bounds": {
            "initial_checkpoint_tick": LIFECYCLE_INITIAL_CHECKPOINT,
            "transition_window_ticks": LIFECYCLE_TRANSITION_WINDOW,
            "final_checkpoint_tick": LIFECYCLE_FINAL_CHECKPOINT,
            "max_tick": LIFECYCLE_FINAL_CHECKPOINT
        },
        "world": {
            "loaded_columns": lifecycle_columns_json(&plan.columns),
            "initial_loaded_column_count": plan.columns.len(),
            "initial_loaded_section_count": plan.loaded_sections,
            "initial_air_section_count": plan.air_sections,
            "platform_section_y": LIFECYCLE_PLATFORM_SECTION,
            "platform_block_y": LIFECYCLE_PLATFORM_Y,
            "sky_light_above_air": 15,
            "block_light": 0,
            "groups": {"A": lifecycle_columns_json(&group_a), "B": lifecycle_columns_json(&group_b), "C": lifecycle_columns_json(&group_c), "Hx": lifecycle_columns_json(&corridor_x), "Hz": lifecycle_columns_json(&corridor_z)},
            "sentinels": {
                "A": [0, 4, 0],
                "B": [8, 4, 0],
                "C": [0, 4, 8],
                "Hx": [4, 4, 0],
                "Hz": [0, 4, 4]
            }
        },
        "camera_script": [
            {"tick": 1, "label": "A_plus_x", "position": [16.5, 65.0, 16.5], "yaw": -90.0, "pitch": 0.0},
            {"tick": 201, "label": "A_plus_z", "position": [16.5, 65.0, 16.5], "yaw": 0.0, "pitch": 0.0},
            {"tick": 321, "label": "B_minus_x", "position": [144.5, 65.0, 16.5], "yaw": 90.0, "pitch": 0.0},
            {"tick": 801, "label": "C_minus_z", "position": [16.5, 65.0, 144.5], "yaw": 180.0, "pitch": 0.0}
        ],
        "checkpoints": [
            lifecycle_checkpoint(200, "initial_ready", 24, &["B"], &["Hx"], &[], &[]),
            lifecycle_checkpoint(320, "camera_turn_plus_z", 24, &["C"], &["Hz"], &[], &[]),
            lifecycle_checkpoint(440, "teleport_B", 24, &["A"], &["Hx"], &[], &[]),
            lifecycle_checkpoint(560, "unload_B", 20, &["A"], &["Hx"], &["B"], &["B"]),
            lifecycle_checkpoint(680, "reload_B", 24, &["A"], &["Hx"], &[], &[]),
            lifecycle_checkpoint(800, "edit_round_trip", 24, &["A"], &["Hx"], &[], &[]),
            lifecycle_checkpoint(920, "teleport_C", 24, &["A"], &["Hz"], &[], &[]),
            lifecycle_checkpoint(1_040, "unload_A", 20, &[], &["Hz"], &[], &["A"]),
            lifecycle_checkpoint(1_160, "reload_A", 24, &["A"], &["Hz"], &[], &[]),
            lifecycle_checkpoint(1_280, "final", 24, &["A"], &["Hz"], &[], &[])
        ],
        "generation": "mc-protocol Packet::write and mc-data registry lookups; corrected per-section sky nibble stride; no raw invented packet bytes"
    });
    if gap_variant {
        recipe["fixture_variant"] = json!("unloaded_corridor");
        recipe["bounds"]["gap_checkpoint_tick"] = json!(LIFECYCLE_GAP_CHECKPOINT);
        recipe["bounds"]["gap_final_checkpoint_tick"] = json!(LIFECYCLE_GAP_FINAL_CHECKPOINT);
        recipe["bounds"]["max_tick"] = json!(LIFECYCLE_GAP_FINAL_CHECKPOINT);
        recipe["gap_extension"] = json!({
            "camera": {"tick": 1281, "position": [16.5, 65.0, 16.5], "yaw": -90.0, "pitch": 0.0},
            "unload": {"tick": LIFECYCLE_GAP_UNLOAD_TICK, "chunk": [4, 0], "sections_removed": 16},
            "reload": {"tick": LIFECYCLE_GAP_RELOAD_TICK, "chunk": [4, 0], "sections_restored": 16},
            "checkpoints": [
                {
                    "tick": LIFECYCLE_GAP_CHECKPOINT,
                    "label": "unload_Hx",
                    "resident_columns": 23,
                    "resident_sections": 368,
                    "required_visible_sentinels": ["B"],
                    "forbidden_resident_columns": [{"chunk_x": 4, "chunk_z": 0, "sections": 16}]
                },
                {
                    "tick": LIFECYCLE_GAP_FINAL_CHECKPOINT,
                    "label": "reload_Hx",
                    "resident_columns": 24,
                    "resident_sections": 384,
                    "required_visible_sentinels": ["B"]
                }
            ]
        });
        let generation = recipe["generation"].as_str().unwrap_or_default().to_owned();
        recipe["generation"] = json!(format!(
            "{}; gap extension uses typed PositionLook and ChunkData unload/reload packets",
            generation
        ));
    }
    recipe
}

fn write_lifecycle_scene(output: &Path) -> Result<(PathBuf, LifecyclePlan), String> {
    write_lifecycle_scene_variant(output, false)
}

fn write_lifecycle_scene_variant(
    output: &Path,
    gap_variant: bool,
) -> Result<(PathBuf, LifecyclePlan), String> {
    let plan = lifecycle_plan();
    let capture = build_lifecycle_capture_variant(&plan, gap_variant)
        .encode()
        .map_err(|error| format!("encoding lifecycle capture failed: {error}"))?;
    let sidecar = recipe_path(output)?;
    let mut recipe_bytes = serde_json::to_vec_pretty(&lifecycle_recipe_variant(&plan, gap_variant))
        .map_err(|error| format!("encoding lifecycle recipe failed: {error}"))?;
    recipe_bytes.push(b'\n');
    reserve_and_write(output, &capture, &sidecar, &recipe_bytes)?;
    Ok((sidecar, plan))
}

fn build_capture(plan: &FixturePlan) -> Capture {
    let meta = CaptureMeta {
        protocol: PROTOCOL,
        address: format!("synthetic-{}", plan.scene.id().to_ascii_lowercase()),
        started_at_unix_millis: 0,
    };
    let mut recorder = Recorder::new(meta);
    record_packet(
        &mut recorder,
        State::Login,
        0,
        &LoginSuccess {
            uuid: Uuid::from_u128(LOCAL_UUID),
            username: "BenchLocal".into(),
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_000,
        &s2c::JoinGame {
            entity_id: LOCAL_ENTITY_ID,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: plan.scene.player_count().min(255) as u8,
            level_type: "default".into(),
            reduced_debug_info: false,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_100,
        &s2c::TimeUpdate {
            total_world_time: 6_000,
            world_time: 6_000,
        },
    );
    record_packet(
        &mut recorder,
        State::Play,
        1_200,
        &s2c::UpdateHealth {
            health: 20.0,
            food_level: 20,
            saturation_level: 5.0,
        },
    );
    let mut micros = 2_000;
    for &(chunk_x, chunk_z, mask) in &plan.world.columns {
        record_packet(
            &mut recorder,
            State::Play,
            micros,
            &chunk::ChunkData {
                chunk_x,
                chunk_z,
                ground_up_continuous: true,
                primary_bit_mask: mask,
                data: chunk_payload(plan, chunk_x, chunk_z, mask),
            },
        );
        micros += 100;
    }
    let properties = if plan.signed_textures {
        vec![s2c::ProfileProperty {
            name: "textures".into(),
            value: SIGNED_TEXTURES_VALUE.into(),
            signature: Some(SIGNED_TEXTURES_SIGNATURE.into()),
        }]
    } else {
        vec![]
    };
    let mut roster: Vec<s2c::AddPlayerEntry> = plan
        .players
        .iter()
        .map(|player| s2c::AddPlayerEntry {
            uuid: player.uuid,
            username: player.username.clone(),
            properties: properties.clone(),
            game_mode: s2c::GameMode::Survival,
            ping: 0,
            display_name: None,
        })
        .collect();
    if plan.signed_textures {
        // An online-mode server sends the local player's own row too, and the first-person arm
        // resolves its skin through that row on every frame.
        roster.push(s2c::AddPlayerEntry {
            uuid: Uuid::from_u128(LOCAL_UUID),
            username: "BenchLocal".into(),
            properties,
            game_mode: s2c::GameMode::Survival,
            ping: 0,
            display_name: None,
        });
    }
    record_packet(
        &mut recorder,
        State::Play,
        micros,
        &s2c::PlayerListItem::AddPlayer(roster),
    );
    micros += 100;
    for player in &plan.players {
        record_packet(
            &mut recorder,
            State::Play,
            micros,
            &s2c::SpawnPlayer {
                entity_id: player.entity_id,
                player_uuid: player.uuid,
                x: player.x,
                y: player.y,
                z: player.z,
                yaw: Angle(player.yaw),
                pitch: Angle(0),
                current_item: 0,
                // Vanilla Entity's constructor always registers watcher 0 as
                // the zero-valued flags byte.  A completely empty metadata
                // list is encoded as a null Java list, and 1.8.9's
                // S0CPacketSpawnPlayer getter dereferences that list before
                // the client handler can treat it as absent.
                metadata: vec![MetadataEntry {
                    index: 0,
                    value: MetadataValue::Byte(0),
                }],
            },
        );
        record_packet(
            &mut recorder,
            State::Play,
            micros,
            &s2c::EntityHeadLook {
                entity_id: player.entity_id,
                yaw: Angle(player.head_yaw),
            },
        );
        micros += 100;
    }
    record_packet(
        &mut recorder,
        State::Play,
        micros,
        &s2c::SpawnPosition {
            position: BlockPos::new(0, ISLAND_SURFACE_Y, 0),
        },
    );
    micros += 100;
    record_packet(
        &mut recorder,
        State::Play,
        micros,
        &s2c::PlayerPositionLook {
            x: plan.local_position.0 as f64 / 32.0,
            y: plan.local_position.1 as f64 / 32.0,
            z: plan.local_position.2 as f64 / 32.0,
            yaw: 0.0,
            pitch: 6.0,
            flags: 0,
        },
    );
    if plan.sound_burst {
        record_sound_burst(&mut recorder, plan);
    }
    recorder.finish()
}

/// Twenty sounds per second for twenty seconds of captured time at the local player, then a record
/// partway through. Starts after a quiet stretch so the world has settled first.
fn record_sound_burst(recorder: &mut Recorder, plan: &FixturePlan) {
    const FIRST_MICROS: u64 = 3_000_000;
    const SPACING_MICROS: u64 = 50_000;
    const SOUNDS: [&str; 5] = [
        "game.player.hurt",
        "random.successful_hit",
        "step.stone",
        "dig.stone",
        "random.bow",
    ];
    const RECORD_AT_INDEX: u64 = 100;
    const RECORD_CAT_ITEM: i32 = 2257;
    let (x, y, z) = plan.local_position;
    for index in 0..400u64 {
        let at = FIRST_MICROS + index * SPACING_MICROS;
        record_packet(
            recorder,
            State::Play,
            at,
            &s2c::SoundEffect {
                sound_name: SOUNDS[index as usize % SOUNDS.len()].into(),
                // Positions are 1/32 block in the plan and 1/8 block on this packet.
                x_fixed: x / 4,
                y_fixed: y / 4,
                z_fixed: z / 4,
                volume: 1.0,
                pitch_quantized: 63,
            },
        );
        if index == RECORD_AT_INDEX {
            record_packet(
                recorder,
                State::Play,
                at,
                &s2c::Effect {
                    effect_id: 1005,
                    position: BlockPos::new(x / 32, y / 32, z / 32),
                    data: RECORD_CAT_ITEM,
                    disable_relative_volume: false,
                },
            );
        }
    }
}

fn recipe(plan: &FixturePlan) -> Value {
    let columns = plan
        .world
        .columns
        .iter()
        .map(|&(x, z, mask)| json!({"chunk_x": x, "chunk_z": z, "section_mask": mask}))
        .collect::<Vec<_>>();
    let players = plan
        .players
        .iter()
        .map(|player| {
            json!({
                "entity_id": player.entity_id,
                "uuid": player.uuid.to_string(),
                "username": player.username,
                "fixed_position_32": [player.x, player.y, player.z],
                "yaw_byte": player.yaw,
                "head_yaw_byte": player.head_yaw,
                "profile_properties": if plan.signed_textures { "signed_textures" } else { "empty_default_skin" }
            })
        })
        .collect::<Vec<_>>();
    let mut recipe = json!({
        "contract": "mc-rs.optimization.synthetic-fixture-recipe",
        "version": 1,
        "fixture_id": plan.scene.id(),
        "status": "UNQUALIFIED_DIAGNOSTIC",
        "scoring_eligible": false,
        "disclaimer": "Synthetic typed-packet diagnostic input only; no image, visibility, asset, or performance qualification is claimed.",
        "protocol": PROTOCOL,
        "player_count": plan.players.len(),
        "players": players,
        "spawn_player_metadata": [{"index": 0, "type": "Byte", "value": 0, "meaning": "vanilla Entity flags default"}],
        "entity_head_pose": {"packet": "EntityHeadLook", "ordering": "immediately_after_each_spawn_player", "meaning": "explicit deterministic head pose for cross-runtime fixture comparison; same timestamp as its spawn"},
        "local_player": {"entity_id": LOCAL_ENTITY_ID, "uuid": Uuid::from_u128(LOCAL_UUID).to_string(), "fixed_position_32": [plan.local_position.0, plan.local_position.1, plan.local_position.2]},
        "camera": {
            "position": [plan.local_position.0 as f64 / 32.0, plan.local_position.1 as f64 / 32.0, plan.local_position.2 as f64 / 32.0],
            "yaw_degrees": 0.0,
            "pitch_degrees": 6.0,
            "viewport": [854, 480],
            "vertical_fov_degrees": 70.0,
            "intent": "Diagnostic framing target for the complete V1 row or E1 10x10 grid; image visibility remains unverified."
        },
        "world": {
            "chunk_columns": columns,
            "loaded_column_count": plan.world.columns.len(),
            "loaded_air_section_count": plan.world.loaded_air_sections,
            "island_radius_blocks": ISLAND_RADIUS,
            "surface_y": ISLAND_SURFACE_Y,
            "solid_y_range": [TERRAIN_MIN_Y, TERRAIN_MAX_Y],
            "solid_block_count": plan.world.solid_blocks,
            "block_states": {"stone": plan.world.stone_state, "dirt": plan.world.dirt_state, "grass": plan.world.grass_state},
            "biome": {"name": "Plains", "id": plan.world.plains_biome},
            "block_light_nibble": 0,
            "sky_light_nibble_above_surface": 15,
            "ground_up_continuous": true
        },
        "clocks": {"capture": "synthetic explicit microseconds in Recorder::record_at", "replay": "runtime ReplaySource virtual cadence is not generated or verified here; pin replay_step_micros", "wall_time": "not used by generation; runtime wall-clock/frame trace audit is required"},
        "assets": "player profile properties are empty and request the default skin path; asset discovery and runtime skin visibility are not verified by this generator",
        "generation": "mc-protocol Packet::write and mc-data registry lookups; no raw invented packet or block-state bytes"
    });
    if plan.underground {
        recipe["world"]["sealed_floor_y"] = json!(SEALED_FLOOR_Y);
    }
    recipe
}

fn recipe_path(output: &Path) -> Result<PathBuf, String> {
    let stem = output
        .file_stem()
        .and_then(|stem| stem.to_str())
        .filter(|stem| !stem.is_empty())
        .ok_or_else(|| "--output must have a filename stem".to_owned())?;
    Ok(output
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.recipe.json")))
}

fn reserve_and_write(
    output: &Path,
    capture: &[u8],
    recipe_path: &Path,
    recipe: &[u8],
) -> Result<(), String> {
    if output == recipe_path {
        return Err("capture output and recipe sidecar paths must differ".into());
    }
    let mut reserved: Vec<(PathBuf, File)> = Vec::new();
    for path in [output, recipe_path] {
        let result = OpenOptions::new().write(true).create_new(true).open(path);
        match result {
            Ok(file) => reserved.push((path.to_path_buf(), file)),
            Err(error) => {
                for (reserved_path, _) in reserved {
                    let _ = fs::remove_file(reserved_path);
                }
                if error.kind() == io::ErrorKind::AlreadyExists {
                    return Err(format!(
                        "refusing to overwrite existing output {}",
                        path.display()
                    ));
                }
                return Err(format!("cannot reserve output {}: {error}", path.display()));
            }
        }
    }
    let write_result = (|| {
        reserved[0].1.write_all(capture)?;
        reserved[0].1.flush()?;
        reserved[1].1.write_all(recipe)?;
        reserved[1].1.flush()?;
        Ok::<(), io::Error>(())
    })();
    if let Err(error) = write_result {
        let paths = reserved
            .iter()
            .map(|(path, _)| path.clone())
            .collect::<Vec<_>>();
        drop(reserved);
        for path in paths {
            let _ = fs::remove_file(path);
        }
        return Err(format!("writing fixture outputs failed: {error}"));
    }
    Ok(())
}

fn write_scene(
    scene: Scene,
    signed_textures: bool,
    sound_burst: bool,
    underground: bool,
    output: &Path,
) -> Result<(PathBuf, FixturePlan), String> {
    let mut plan = build_plan(scene);
    plan.signed_textures = signed_textures;
    plan.sound_burst = sound_burst;
    if underground {
        plan.underground = true;
        for player in &mut plan.players {
            player.y = UNDERGROUND_PLAYER_Y * 32;
        }
        plan.world.solid_blocks += 256 * plan.world.columns.len();
    }
    let capture = build_capture(&plan)
        .encode()
        .map_err(|error| format!("encoding capture failed: {error}"))?;
    let sidecar = recipe_path(output)?;
    let mut recipe_bytes = serde_json::to_vec_pretty(&recipe(&plan))
        .map_err(|error| format!("encoding recipe failed: {error}"))?;
    recipe_bytes.push(b'\n');
    reserve_and_write(output, &capture, &sidecar, &recipe_bytes)?;
    Ok((sidecar, plan))
}

fn parse_args() -> Result<(Scene, bool, bool, bool, PathBuf), String> {
    let mut scene = None;
    let mut output = None;
    let mut signed_textures = false;
    let mut sound_burst = false;
    let mut underground = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scene" => scene = Some(Scene::parse(&args.next().ok_or("--scene needs a value")?)?),
            "--output" => {
                output = Some(PathBuf::from(args.next().ok_or("--output needs a value")?))
            }
            "--signed-textures" => signed_textures = true,
            "--sound-burst" => sound_burst = true,
            "--underground" => underground = true,
            "-h" | "--help" => {
                println!(
                    "usage: optimization_fixture --scene v1|e1-1|e1-16|e1-50|e1-100|lifecycle|lifecycle-gap [--signed-textures] [--sound-burst] --output PATH"
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let scene: Scene = scene.ok_or("--scene is required")?;
    if (signed_textures || sound_burst || underground)
        && matches!(scene, Scene::Lifecycle | Scene::LifecycleGap)
    {
        return Err(
            "--signed-textures, --sound-burst and --underground apply to the player scenes only"
                .to_owned(),
        );
    }
    Ok((
        scene,
        signed_textures,
        sound_burst,
        underground,
        output.ok_or("--output is required")?,
    ))
}

fn main() {
    let (scene, signed_textures, sound_burst, underground, output) = match parse_args() {
        Ok(args) => args,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    let result = match scene {
        Scene::Lifecycle => write_lifecycle_scene(&output).map(|(sidecar, plan)| {
            println!("wrote {} ({})", output.display(), scene.id());
            println!(
                "wrote {} ({} columns, {} resident sections)",
                sidecar.display(),
                plan.columns.len(),
                plan.loaded_sections
            );
        }),
        Scene::LifecycleGap => {
            write_lifecycle_scene_variant(&output, true).map(|(sidecar, plan)| {
                println!("wrote {} ({})", output.display(), scene.id());
                println!(
                "wrote {} ({} base columns, {} base resident sections, gap tail through tick {})",
                sidecar.display(),
                plan.columns.len(),
                plan.loaded_sections,
                LIFECYCLE_GAP_FINAL_CHECKPOINT
            );
            })
        }
        _ => write_scene(scene, signed_textures, sound_burst, underground, &output).map(
            |(sidecar, plan)| {
                println!("wrote {} ({})", output.display(), scene.id());
                println!(
                    "wrote {} ({} players, {} columns)",
                    sidecar.display(),
                    plan.players.len(),
                    plan.world.columns.len()
                );
            },
        ),
    };
    match result {
        Ok(()) => {}
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_protocol::play::s2c::PlayerListItem;

    fn capture_bytes(scene: Scene) -> Vec<u8> {
        build_capture(&build_plan(scene))
            .encode()
            .expect("capture encoding")
    }

    #[test]
    fn lifecycle_capture_has_fixed_schedule_and_unload_packets() {
        let plan = lifecycle_plan();
        assert_eq!(plan.columns.len(), 24);
        assert_eq!(plan.loaded_sections, 384);
        assert_eq!(plan.air_sections, 372);
        let capture = Capture::parse(
            &build_lifecycle_capture(&plan)
                .encode()
                .expect("lifecycle encoding"),
        )
        .expect("lifecycle parse");
        let chunks = capture
            .frames()
            .iter()
            .filter(|frame| frame.packet_id() == Some(chunk::ChunkData::ID))
            .map(|frame| {
                let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
                (
                    frame.micros,
                    chunk::ChunkData::read(&mut body).expect("chunk"),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 40);
        assert_eq!(
            chunks[..24]
                .iter()
                .filter(|(_, packet)| packet.primary_bit_mask == LIFECYCLE_SECTION_MASK)
                .count(),
            24
        );
        assert_eq!(
            chunks[24..28]
                .iter()
                .filter(|(_, packet)| packet.primary_bit_mask == 0)
                .count(),
            4
        );
        assert_eq!(
            chunks[28..32]
                .iter()
                .filter(|(_, packet)| packet.primary_bit_mask == LIFECYCLE_SECTION_MASK)
                .count(),
            4
        );
        assert_eq!(
            chunks[32..36]
                .iter()
                .filter(|(_, packet)| packet.primary_bit_mask == 0)
                .count(),
            4
        );
        assert_eq!(
            chunks[36..40]
                .iter()
                .filter(|(_, packet)| packet.primary_bit_mask == LIFECYCLE_SECTION_MASK)
                .count(),
            4
        );
        assert!(chunks.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        assert!(chunks.iter().all(|(_, packet)| packet.ground_up_continuous));
        let edits = capture
            .frames()
            .iter()
            .filter(|frame| frame.packet_id() == Some(s2c::BlockChange::ID))
            .map(|frame| frame.micros)
            .collect::<Vec<_>>();
        assert_eq!(edits.len(), 2);
        assert!(edits[0] < 681 * LIFECYCLE_STEP_MICROS);
        assert!(edits[1] >= 682 * LIFECYCLE_STEP_MICROS - 1);
    }

    #[test]
    fn lifecycle_base_variant_wrapper_is_byte_identical() {
        let plan = lifecycle_plan();
        let base = build_lifecycle_capture(&plan)
            .encode()
            .expect("base lifecycle encoding");
        let explicit_base = build_lifecycle_capture_variant(&plan, false)
            .encode()
            .expect("explicit base lifecycle encoding");
        assert_eq!(
            base, explicit_base,
            "gap extension must not alter base capture bytes"
        );
        let recipe = lifecycle_recipe(&plan);
        assert!(
            recipe.get("fixture_variant").is_none(),
            "base recipe stays untagged"
        );
        assert_eq!(
            recipe["checkpoints"]
                .as_array()
                .expect("base checkpoints")
                .len(),
            10
        );
    }

    #[test]
    fn lifecycle_gap_has_exact_corridor_tail_and_resident_counts() {
        let plan = lifecycle_plan();
        let capture = Capture::parse(
            &build_lifecycle_capture_variant(&plan, true)
                .encode()
                .expect("gap lifecycle encoding"),
        )
        .expect("gap lifecycle parse");
        let chunks = capture
            .frames()
            .iter()
            .filter(|frame| frame.packet_id() == Some(chunk::ChunkData::ID))
            .map(|frame| {
                let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
                (
                    frame.micros,
                    chunk::ChunkData::read(&mut body).expect("chunk"),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(chunks.len(), 42);
        let unload = &chunks[40];
        assert_eq!(
            unload.0,
            lifecycle_tick_micros(LIFECYCLE_GAP_UNLOAD_TICK, 0, 1)
        );
        assert_eq!((unload.1.chunk_x, unload.1.chunk_z), (4, 0));
        assert_eq!(unload.1.primary_bit_mask, 0);
        let reload = &chunks[41];
        assert_eq!(
            reload.0,
            lifecycle_tick_micros(LIFECYCLE_GAP_RELOAD_TICK, 0, 1)
        );
        assert_eq!((reload.1.chunk_x, reload.1.chunk_z), (4, 0));
        assert_eq!(reload.1.primary_bit_mask, LIFECYCLE_SECTION_MASK);

        let camera_micros = lifecycle_tick_micros(1_281, 0, 1);
        let camera_frame = capture
            .frames()
            .iter()
            .find(|frame| {
                frame.packet_id() == Some(s2c::PlayerPositionLook::ID)
                    && frame.micros == camera_micros
            })
            .expect("gap return-to-A camera packet");
        let mut camera_body = PacketBuffer::from_slice(camera_frame.body().expect("camera body"));
        let camera = s2c::PlayerPositionLook::read(&mut camera_body).expect("camera packet");
        assert_eq!((camera.x, camera.y, camera.z), (16.5, 65.0, 16.5));
        assert_eq!((camera.yaw, camera.pitch, camera.flags), (-90.0, 0.0, 0));

        let mut world = mc_world::World::new();
        let mut saw_unload = false;
        let mut saw_reload = false;
        for frame in capture.frames() {
            if frame.packet_id() != Some(chunk::ChunkData::ID) {
                continue;
            }
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            let packet = chunk::ChunkData::read(&mut body).expect("chunk");
            world
                .apply_chunk_data(
                    ChunkPos::new(packet.chunk_x, packet.chunk_z),
                    true,
                    packet.primary_bit_mask,
                    &packet.data,
                )
                .expect("valid gap chunk");
            if frame.micros == unload.0 {
                saw_unload = true;
                assert_eq!(world.loaded_chunk_count(), 23);
                assert_eq!(world.loaded_chunk_count() * 16, 368);
            }
            if frame.micros == reload.0 {
                saw_reload = true;
                assert_eq!(world.loaded_chunk_count(), 24);
                assert_eq!(world.loaded_chunk_count() * 16, 384);
            }
        }
        assert!(saw_unload && saw_reload);
    }

    #[test]
    fn lifecycle_gap_recipe_pins_variant_and_tail_contract() {
        let recipe = lifecycle_recipe_variant(&lifecycle_plan(), true);
        assert_eq!(recipe["fixture_variant"], "unloaded_corridor");
        assert_eq!(recipe["bounds"]["max_tick"], LIFECYCLE_GAP_FINAL_CHECKPOINT);
        let extension = recipe["gap_extension"].as_object().expect("gap extension");
        assert_eq!(extension["unload"]["chunk"], json!([4, 0]));
        let checkpoints = extension["checkpoints"]
            .as_array()
            .expect("gap checkpoints");
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0]["resident_sections"], 368);
        assert_eq!(checkpoints[1]["resident_sections"], 384);
        assert_eq!(checkpoints[0]["required_visible_sentinels"], json!(["B"]));
    }

    #[test]
    fn lifecycle_world_identity_round_trips_and_restores_after_reload() {
        let plan = lifecycle_plan();
        let capture = Capture::parse(
            &build_lifecycle_capture(&plan)
                .encode()
                .expect("lifecycle encoding"),
        )
        .expect("lifecycle parse");
        let mut world = mc_world::World::new();
        let mut saw_unload_b = false;
        let mut saw_reload_b = false;
        for frame in capture.frames() {
            if frame.packet_id() != Some(chunk::ChunkData::ID) {
                continue;
            }
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            let packet = chunk::ChunkData::read(&mut body).expect("chunk");
            world
                .apply_chunk_data(
                    ChunkPos::new(packet.chunk_x, packet.chunk_z),
                    true,
                    packet.primary_bit_mask,
                    &packet.data,
                )
                .expect("valid lifecycle chunk");
            if packet.primary_bit_mask == 0 && packet.chunk_x == 9 && packet.chunk_z == 1 {
                saw_unload_b = true;
                assert_eq!(world.loaded_chunk_count(), 20);
            }
            if packet.primary_bit_mask == LIFECYCLE_SECTION_MASK && packet.chunk_x == 8 {
                saw_reload_b = true;
            }
        }
        assert!(saw_unload_b && saw_reload_b);
        assert_eq!(world.loaded_chunk_count(), 24);
        assert_eq!(
            world
                .block_at(BlockPos::new(8 * 16, LIFECYCLE_PLATFORM_Y, 0))
                .id,
            plan.stone_state
        );
        assert_eq!(world.sky_light_at(BlockPos::new(4 * 16, 64, 0)), 15);
        assert_eq!(world.sky_light_at(BlockPos::new(0, 64, 4 * 16)), 15);
    }

    #[test]
    fn lifecycle_decoded_light_shades_platform_below_and_at_floor() {
        let plan = lifecycle_plan();
        let capture = Capture::parse(
            &build_lifecycle_capture(&plan)
                .encode()
                .expect("lifecycle encoding"),
        )
        .expect("lifecycle parse");
        let mut world = mc_world::World::new();
        for frame in capture
            .frames()
            .iter()
            .filter(|frame| frame.packet_id() == Some(chunk::ChunkData::ID))
            .take(24)
        {
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            let packet = chunk::ChunkData::read(&mut body).expect("typed chunk");
            if packet.primary_bit_mask == LIFECYCLE_SECTION_MASK {
                world
                    .apply_chunk_data(
                        ChunkPos::new(packet.chunk_x, packet.chunk_z),
                        true,
                        packet.primary_bit_mask,
                        &packet.data,
                    )
                    .expect("valid lifecycle chunk");
            }
        }
        for y in 0..256i32 {
            let platform_sky = world.sky_light_at(BlockPos::new(0, y, 0));
            let corridor_sky = world.sky_light_at(BlockPos::new(4 * 16, y, 0));
            assert_eq!(
                platform_sky,
                if y <= LIFECYCLE_PLATFORM_Y { 0 } else { 15 },
                "platform sky at y={y}"
            );
            assert_eq!(corridor_sky, 15, "corridor sky at y={y}");
        }
        assert_eq!(
            world.block_at(BlockPos::new(0, LIFECYCLE_PLATFORM_Y, 0)).id,
            plan.stone_state
        );
        assert_eq!(
            world.block_at(BlockPos::new(0, LIFECYCLE_PLATFORM_Y - 1, 0)),
            BlockState::AIR
        );
    }

    #[test]
    fn lifecycle_recipe_contains_bounded_checkpoint_contract() {
        let recipe = lifecycle_recipe(&lifecycle_plan());
        assert_eq!(recipe["fixture_id"], "DIAGNOSTIC_TERRAIN");
        assert_eq!(recipe["scoring_eligible"], false);
        assert_eq!(recipe["bounds"]["initial_checkpoint_tick"], 200);
        assert_eq!(recipe["bounds"]["transition_window_ticks"], 120);
        assert_eq!(recipe["bounds"]["max_tick"], 1280);
        assert_eq!(recipe["world"]["sentinels"]["C"], json!([0, 4, 8]));
        assert_eq!(recipe["world"]["sentinels"]["A"], json!([0, 4, 0]));
        let checkpoints = recipe["checkpoints"].as_array().expect("checkpoints");
        assert_eq!(checkpoints.len(), 10);
        let unload_a = checkpoints
            .iter()
            .find(|checkpoint| checkpoint["label"] == "unload_A")
            .expect("unload A checkpoint");
        assert_eq!(unload_a["required_visible_sentinels"], json!([]));
        assert_eq!(unload_a["forbidden_resident_groups"], json!(["A"]));
        let reload_a = checkpoints
            .iter()
            .find(|checkpoint| checkpoint["label"] == "reload_A")
            .expect("reload A checkpoint");
        assert_eq!(reload_a["required_visible_sentinels"], json!(["A"]));
    }

    #[test]
    fn typed_capture_round_trips_and_contains_play_packets() {
        let bytes = capture_bytes(Scene::V1);
        let capture = Capture::parse(&bytes).expect("capture parse");
        assert_eq!(capture.meta.protocol, PROTOCOL);
        assert!(
            capture
                .frames()
                .iter()
                .any(|frame| frame.state == State::Play)
        );
        assert!(
            capture
                .frames()
                .iter()
                .all(|frame| frame.packet_id().is_some())
        );
    }

    #[test]
    fn generated_bytes_and_recipe_are_deterministic() {
        let first = build_plan(Scene::E1 { players: 50 });
        let second = build_plan(Scene::E1 { players: 50 });
        assert_eq!(first, second);
        assert_eq!(
            capture_bytes(Scene::E1 { players: 50 }),
            capture_bytes(Scene::E1 { players: 50 })
        );
        assert_eq!(recipe(&first), recipe(&second));
    }

    #[test]
    fn player_list_and_spawn_packets_match() {
        let plan = build_plan(Scene::V1);
        let capture = Capture::parse(&capture_bytes(Scene::V1)).expect("parse");
        let recipe = recipe(&plan);
        assert_eq!(recipe["entity_head_pose"]["packet"], "EntityHeadLook");
        for player in recipe["players"].as_array().expect("recipe players") {
            assert_eq!(player["head_yaw_byte"], player["yaw_byte"]);
        }
        let mut roster = BTreeSet::new();
        let mut spawns = BTreeSet::new();
        for frame in capture.frames() {
            if frame.state != State::Play {
                continue;
            }
            let Some(id) = frame.packet_id() else {
                continue;
            };
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            if id == PlayerListItem::ID {
                let PlayerListItem::AddPlayer(entries) =
                    PlayerListItem::read(&mut body).expect("list")
                else {
                    panic!("expected add player")
                };
                roster.extend(
                    entries
                        .into_iter()
                        .map(|entry| (entry.uuid, entry.username)),
                );
            } else if id == s2c::SpawnPlayer::ID {
                let spawn = s2c::SpawnPlayer::read(&mut body).expect("spawn");
                assert_eq!(
                    spawn.metadata,
                    vec![MetadataEntry {
                        index: 0,
                        value: MetadataValue::Byte(0),
                    }],
                    "SpawnPlayer must carry vanilla watcher 0 flags metadata"
                );
                spawns.insert(spawn.player_uuid);
            }
        }
        assert_eq!(roster.len(), plan.players.len());
        assert_eq!(spawns.len(), plan.players.len());
        for player in &plan.players {
            assert!(roster.contains(&(player.uuid, player.username.clone())));
            assert!(spawns.contains(&player.uuid));
        }
    }

    #[test]
    fn every_spawn_has_an_immediate_matching_head_pose() {
        let plan = build_plan(Scene::V1);
        let capture = Capture::parse(&capture_bytes(Scene::V1)).expect("parse");
        let mut pending_spawn = None;
        let mut head_poses = 0usize;
        for frame in capture.frames() {
            if frame.state != State::Play {
                continue;
            }
            let Some(id) = frame.packet_id() else {
                continue;
            };
            if let Some((entity_id, yaw, spawn_micros)) = pending_spawn.take() {
                assert_eq!(
                    id,
                    s2c::EntityHeadLook::ID,
                    "S19 must immediately follow its S0C spawn"
                );
                assert_eq!(frame.micros, spawn_micros);
                assert_eq!(frame.direction, Direction::Clientbound);
                assert_eq!(frame.state, State::Play);
                let mut body = PacketBuffer::from_slice(frame.body().expect("head body"));
                let head = s2c::EntityHeadLook::read(&mut body).expect("head pose");
                assert_eq!(head.entity_id, entity_id);
                assert_eq!(head.yaw, Angle(yaw));
                head_poses += 1;
                continue;
            }
            if id == s2c::SpawnPlayer::ID {
                let mut body = PacketBuffer::from_slice(frame.body().expect("spawn body"));
                let spawn = s2c::SpawnPlayer::read(&mut body).expect("spawn");
                pending_spawn = Some((spawn.entity_id, spawn.yaw.0, frame.micros));
            }
        }
        assert!(pending_spawn.is_none(), "last spawn must have a head pose");
        assert_eq!(head_poses, plan.players.len());
    }

    #[test]
    fn chunks_have_real_geometry_and_loaded_air_sections() {
        let plan = build_plan(Scene::V1);
        let capture = Capture::parse(&capture_bytes(Scene::V1)).expect("parse");
        let mut world = mc_world::World::new();
        for frame in capture.frames() {
            if frame.packet_id() != Some(chunk::ChunkData::ID) {
                continue;
            }
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            let packet = chunk::ChunkData::read(&mut body).expect("chunk");
            world
                .apply_chunk_data(
                    ChunkPos::new(packet.chunk_x, packet.chunk_z),
                    true,
                    packet.primary_bit_mask,
                    &packet.data,
                )
                .expect("valid chunk");
        }
        assert_eq!(world.loaded_chunk_count(), plan.world.columns.len());
        assert_eq!(
            world.block_at(BlockPos::new(0, 63, 0)).id,
            plan.world.grass_state
        );
        assert_eq!(
            world.block_at(BlockPos::new(0, 60, 0)).id,
            plan.world.stone_state
        );
        assert_eq!(world.block_at(BlockPos::new(0, 20, 0)), BlockState::AIR);
        assert!(
            world
                .chunk(ChunkPos::new(2, 2))
                .expect("outer column")
                .section(0)
                .is_some()
        );
        assert!(plan.world.solid_blocks > 0);
        assert!(plan.world.loaded_air_sections > 0);
    }

    #[test]
    fn decoded_light_payload_and_full_overworld_state_match_recipe() {
        for scene in [
            Scene::V1,
            Scene::E1 { players: 50 },
            Scene::E1 { players: 100 },
        ] {
            let plan = build_plan(scene);
            let capture = Capture::parse(&capture_bytes(scene)).expect("parse");
            let mut world = mc_world::World::new();
            for frame in capture.frames() {
                if frame.packet_id() != Some(chunk::ChunkData::ID) {
                    continue;
                }
                let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
                let packet = chunk::ChunkData::read(&mut body).expect("chunk");
                world
                    .apply_chunk_data(
                        ChunkPos::new(packet.chunk_x, packet.chunk_z),
                        true,
                        packet.primary_bit_mask,
                        &packet.data,
                    )
                    .expect("valid chunk");
            }

            assert_eq!(world.loaded_chunk_count(), plan.world.columns.len());
            for (position, column) in world.loaded_chunks() {
                assert_eq!(
                    column.queued_light_checks(),
                    4096,
                    "full overworld chunk {position:?} must start with relight sweep complete"
                );
                assert!(
                    !column.is_gap_lighting_updated(),
                    "full overworld chunk {position:?} must not have a sky-gap repair pending"
                );
            }

            // Walk every block in every loaded section.  This catches both the
            // surface-adjacent y=64 contract and section-stride overlap in the
            // packed sky nibble array.
            for chunk_x in -2..=2 {
                for chunk_z in -2..=2 {
                    for local_y in 0..80i32 {
                        let y = local_y;
                        for local_z in 0..16i32 {
                            for local_x in 0..16i32 {
                                let x = chunk_x * 16 + local_x;
                                let z = chunk_z * 16 + local_z;
                                let position = BlockPos::new(x, y, z);
                                let inside_island = x * x + z * z <= ISLAND_RADIUS * ISLAND_RADIUS;
                                let expected_sky: u8 = if inside_island && y <= ISLAND_SURFACE_Y {
                                    0
                                } else {
                                    15
                                };
                                assert_eq!(
                                    world.sky_light_at(position),
                                    expected_sky,
                                    "unexpected decoded sky nibble for {scene:?} at {position:?}"
                                );
                                assert_eq!(
                                    world.block_light_at(position),
                                    0,
                                    "synthetic fixture must not inject block light for {scene:?} at {position:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_packet_bytes_are_scene_independent() {
        fn chunk_packets(scene: Scene) -> Vec<chunk::ChunkData> {
            let capture = Capture::parse(&capture_bytes(scene)).expect("parse");
            capture
                .frames()
                .iter()
                .filter_map(|frame| {
                    (frame.packet_id() == Some(chunk::ChunkData::ID)).then(|| {
                        let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
                        chunk::ChunkData::read(&mut body).expect("chunk")
                    })
                })
                .collect()
        }

        let v1 = chunk_packets(Scene::V1);
        let e1_50 = chunk_packets(Scene::E1 { players: 50 });
        let e1_100 = chunk_packets(Scene::E1 { players: 100 });
        assert_eq!(v1.len(), 25);
        assert_eq!(v1, e1_50);
        assert_eq!(v1, e1_100);
    }

    #[test]
    fn local_anchor_is_supported_and_roster_fits_documented_frustum() {
        let plan = build_plan(Scene::E1 { players: 100 });
        let capture = Capture::parse(&capture_bytes(Scene::E1 { players: 100 })).expect("parse");
        let mut world = mc_world::World::new();
        for frame in capture.frames() {
            if frame.packet_id() != Some(chunk::ChunkData::ID) {
                continue;
            }
            let mut body = PacketBuffer::from_slice(frame.body().expect("body"));
            let packet = chunk::ChunkData::read(&mut body).expect("chunk");
            world
                .apply_chunk_data(
                    ChunkPos::new(packet.chunk_x, packet.chunk_z),
                    true,
                    packet.primary_bit_mask,
                    &packet.data,
                )
                .expect("valid chunk");
        }
        let feet = BlockPos::new(
            plan.local_position.0 / 32,
            plan.local_position.1 / 32,
            plan.local_position.2 / 32,
        );
        assert_ne!(world.block_at(feet.down()), BlockState::AIR);

        let half_horizontal_fov = (70.0_f64.to_radians() / 2.0)
            .tan()
            .mul_add(854.0 / 480.0, 0.0)
            .atan()
            .to_degrees();
        let camera_x = plan.local_position.0 as f64 / 32.0;
        let camera_z = plan.local_position.2 as f64 / 32.0;
        for player in &plan.players {
            let x = player.x as f64 / 32.0;
            let z = player.z as f64 / 32.0;
            let forward = z - camera_z;
            assert!(forward > 0.0, "player must be in front of +Z camera");
            let horizontal_angle = ((x - camera_x).abs() / forward).atan().to_degrees();
            assert!(
                horizontal_angle < half_horizontal_fov,
                "player at ({x},{z}) outside documented horizontal frustum"
            );
        }
    }

    #[test]
    fn exclusive_outputs_refuse_overwrite() {
        let temp = tempfile::tempdir().expect("tempdir");
        let output = temp.path().join("v1.mccap");
        write_scene(Scene::V1, false, false, false, &output).expect("first write");
        let capture_before = fs::read(&output).expect("capture");
        assert!(write_scene(Scene::V1, false, false, false, &output).is_err());
        assert_eq!(fs::read(&output).expect("capture remains"), capture_before);
    }

    #[test]
    fn cli_scene_counts_are_exact() {
        assert_eq!(build_plan(Scene::V1).players.len(), 8);
        assert_eq!(build_plan(Scene::E1 { players: 50 }).players.len(), 50);
        assert_eq!(build_plan(Scene::E1 { players: 100 }).players.len(), 100);
    }
}
