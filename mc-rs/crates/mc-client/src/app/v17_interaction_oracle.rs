//! 1.7.10 support: the client-level interaction oracle against the real 1.7.10 client.
//!
//! `javaref/V17OracleRef.java` drives the **real recompiled 1.7.10 client** headless
//! (`Minecraft.runTick`, the real `NetworkManager` queue, `PlayerControllerMP`,
//! `EntityClientPlayerMP`, `EntityRenderer.getMouseOver`) through the scenarios in
//! `tests/fixtures/v17_oracle/scenarios/` and writes, per frame, the inbound packets it was given
//! (protocol 5), every packet it sent (exact bytes) and the local player's state
//! (`tests/fixtures/v17_oracle/java17/`, format in `javaref/V17Scenario.java`).
//!
//! This replays those frames through the real Rust client in 1.7.10 mode the way the shell does
//! (`h2_interaction_oracle` is the 1.8.9 twin): the render-time look on the predictor, the key
//! edges through `KeyBindings::set_state_when_input_allowed`, the inbound bytes decoded with the
//! protocol-5 decoder into the session's 1.7.10 queue (applied inside the tick, after the pick),
//! `ShellApp::before_ticks`, then `ShellApp::tick` per tick.
//!
//! It compares what a click does: the **interaction packets** (C02, C07, C08, C09, C0A, and C0B,
//! whose sprint reset a hit decides) of each frame, in order and byte for byte in their
//! protocol-5 encoding, and the interaction state the Java side records: the crosshair target
//! (Java's null `objectMouseOver` included), dig in progress, dig damage, the five-tick hit delay,
//! item in use and its count, held slot, held stack and the scenario's watched blocks. A 1.8.9
//! packet the protocol-5 conversion refuses fails its frame: it would never reach a 1.7.10
//! server. Movement, screens and the rest are compared elsewhere.
//!
//! `V17_ONLY=<substring>` replays only matching scenarios; `V17_VERBOSE=1` prints every frame.
//! Skips (prints `skip:`) without the vanilla jar (`MC_JARS_DIR`) or the fixture.

use std::fmt::Write as _;

use super::*;
use crate::net::channel::NetHandle;
use crate::net::{ToGame, ToNet};
use mc_protocol::v5::{C2SPlay5, S2CPlay5, V5Ctx};
use mc_protocol::PacketBuffer;
use mc_render::input::keys;
use mc_render::{ShellApp, TickCtx};
use mc_version::GameVersion;
use serde_json::Value;

const FIXTURE_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v17_oracle/java17"
);
/// The scenario definitions, for the cells each one watches.
const SCENARIO_DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v17_oracle/scenarios"
);
/// java: the harness's local profile, as in the 1.8.9 oracles ("H1").
const LOCAL_UUID: u128 = 0x4831_f00d_0000_3000_8000_0000_0000_0001;

/// The protocol-5 ids of the packets a click produces, and C0B, whose `STOP_SPRINTING` follows
/// a knockback hit.
const COMPARED_IDS: [i32; 6] = [0x02, 0x07, 0x08, 0x09, 0x0A, 0x0B];

/// java: `V17Scenario.BINDINGS`, by the vanilla default binding they name.
fn binding(name: &str) -> Option<&'static str> {
    Some(match name {
        "attack" => keys::ATTACK,
        "use" => keys::USE_ITEM,
        "forward" => keys::FORWARD,
        "left" => keys::LEFT,
        "back" => keys::BACK,
        "right" => keys::RIGHT,
        "jump" => keys::JUMP,
        "sneak" => keys::SNEAK,
        "sprint" => keys::SPRINT,
        "drop" => keys::DROP,
        "inventory" => keys::INVENTORY,
        "pick" => keys::PICK_ITEM,
        "chat" => keys::CHAT,
        hotbar if hotbar.starts_with("hotbar") => {
            keys::hotbar_static(hotbar.trim_start_matches("hotbar").parse().ok()?)
        }
        _ => return None,
    })
}

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(out, "{byte:02x}");
    }
    out
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|at| u8::from_str_radix(&text[at..at + 2], 16).expect("fixture hex"))
        .collect()
}

fn encode_v5(packet: &C2SPlay5) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(packet.id());
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

fn packet_id(wire: &[u8]) -> i32 {
    let mut buf = PacketBuffer::from_slice(wire);
    buf.read_varint().unwrap_or(-1)
}

fn decode_inbound(wire: &[u8]) -> S2CPlay5 {
    let mut buf = PacketBuffer::from_slice(wire);
    let id = buf.read_varint().expect("inbound id");
    S2CPlay5::read(id, &mut buf)
        .unwrap_or_else(|error| panic!("cannot decode protocol-5 inbound 0x{id:02X}: {error:?}"))
}

/// A real client on a temp game dir with the mods at their defaults except `modern_keybinds`,
/// as the 1.8.9 oracles run it, holding a 1.7.10 session already past login.
fn client_v17() -> Option<(
    Client,
    tempfile::TempDir,
    tokio::sync::mpsc::Sender<ToGame>,
    tokio::sync::mpsc::UnboundedReceiver<ToNet>,
)> {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join(mc_mods::SETTINGS_FILE_NAME),
        r#"{"format":1,"mods":{"modern_keybinds":{"enabled":false}}}"#,
    )
    .expect("write the temp mod settings");
    let config = ClientConfig {
        game_dir: dir.path().to_path_buf(),
        login: LoginMode::Offline("H1".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
        return None;
    }
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let (handle, to_game, to_net) = NetHandle::new_pair();
    let join = client.join_settings();
    client.session = Some(Session::with_version(handle, join, GameVersion::V1_7_10));
    to_game
        .try_send(ToGame::Connected(mc_protocol::connection::LoginOutcome {
            uuid: uuid::Uuid::from_u128(LOCAL_UUID),
            username: "H1".into(),
            compression_threshold: None,
            encrypted: false,
        }))
        .expect("inbound channel");
    Some((client, dir, to_game, to_net))
}

/// The compared packets Rust queued, in their protocol-5 bytes. A 1.8.9 packet the protocol-5
/// conversion refuses is listed as `REFUSED ...`, so its frame fails whatever it was: the
/// network task would drop it, and vanilla 1.7.10 sends nothing it cannot encode.
fn drain_interactions(
    client: &Client,
    to_net: &mut tokio::sync::mpsc::UnboundedReceiver<ToNet>,
    out: &mut Vec<String>,
) {
    let entity_id = client
        .session
        .as_ref()
        .map_or(0, |session| session.state().local.entity_id);
    while let Ok(message) = to_net.try_recv() {
        let packet = match message {
            ToNet::SendV5(packet) => packet,
            ToNet::Send(packet) => match C2SPlay5::from_v47(&packet, &V5Ctx { entity_id }) {
                Ok(packet) => packet,
                Err(error) => {
                    out.push(format!("REFUSED {packet:?}: {error}"));
                    continue;
                }
            },
            ToNet::Disconnect => continue,
        };
        let wire = encode_v5(&packet);
        if COMPARED_IDS.contains(&packet_id(&wire)) {
            out.push(hex(&wire));
        }
    }
}

/// The interaction state both sides record, as the fixture spells it. `watch` is the scenario's
/// watched cells.
fn rust_state(client: &Client, watch: &[mc_math::BlockPos]) -> Vec<(&'static str, Value)> {
    let hit = match client.hit {
        // java17: `objectMouseOver == null`, written as JSON null.
        _ if client.pick_v5.null => Value::Null,
        HitResult::Block(hit) => Value::from(format!(
            "block {} {} {} {}",
            hit.pos.x,
            hit.pos.y,
            hit.pos.z,
            hit.face.index()
        )),
        HitResult::Entity(hit) => Value::from(format!("entity {}", hit.network_id)),
        HitResult::Miss => Value::from("miss"),
    };
    // java: `item(stack)`, `id:stackSize:metadata` or null.
    let held = client.held_item().map_or(Value::Null, |stack| {
        Value::from(format!(
            "{}:{}:{}",
            stack.item_id, stack.count, stack.damage
        ))
    });
    let blocks = client.session.as_ref().map_or(Value::Null, |session| {
        Value::from(
            watch
                .iter()
                .map(|pos| {
                    let state = session.state().world.block_at(*pos);
                    Value::from(format!("{}:{}", state.block().0, state.meta()))
                })
                .collect::<Vec<_>>(),
        )
    });
    vec![
        ("hit", hit),
        ("held", held),
        ("blocks", blocks),
        (
            "itemInUseCount",
            Value::from(client.controller.item_in_use_count()),
        ),
        (
            "digging",
            Value::from(client.controller.hitting_block().is_some()),
        ),
        (
            "digDamage",
            Value::from(f64::from(client.controller.damage())),
        ),
        ("digDelay", Value::from(client.controller.block_hit_delay())),
        ("usingItem", Value::from(client.controller.is_using_item())),
        ("slot", Value::from(client.player_hud.selected_slot)),
    ]
}

fn same_value(java: &Value, rust: &Value) -> bool {
    match (java.as_f64(), rust.as_f64()) {
        // Floats are written as the exact decimal of their double widening.
        (Some(j), Some(r)) => j.to_bits() == r.to_bits(),
        _ => java == rust,
    }
}

/// What one scenario's replay found: the first frame whose interaction packets differ and the
/// first frame whose interaction state differs.
#[derive(Default)]
struct Outcome {
    packets: Option<String>,
    state: Option<String>,
    frames: usize,
}

fn replay(
    name: &str,
    scenario: &Value,
    blobs: &[Vec<u8>],
    watch: &[mc_math::BlockPos],
) -> Option<Outcome> {
    let verbose = std::env::var_os("V17_VERBOSE").is_some();
    let (mut client, _dir, to_game, mut to_net) = client_v17()?;
    let mut camera = menu_camera();
    let mut input = InputState::new();
    let mut settings = GameSettings::default();
    let mut ticks_run = 0u64;
    let frames = scenario["frames"].as_array().expect("frames");
    let mut outcome = Outcome {
        frames: frames.len(),
        ..Outcome::default()
    };
    // The Java side records only what changed; carry the fields forward.
    let mut java_state = serde_json::Map::new();

    for frame in frames {
        let index = frame["frame"].as_u64().unwrap_or(0);
        if let Some(look) = frame.get("look").and_then(Value::as_array) {
            if let Some(predictor) = client.session.as_mut().and_then(Session::predictor_mut) {
                predictor.state.yaw = look[0].as_f64().unwrap_or(0.0) as f32;
                predictor.state.pitch = look[1].as_f64().unwrap_or(0.0) as f32;
            }
        }
        let mut wheel = 0i32;
        for event in frame
            .get("events")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
        {
            if let Some(notches) = event.strip_prefix("wheel ") {
                wheel += notches.trim().parse::<i32>().unwrap_or(0);
                continue;
            }
            let (down, name) = match event.split_at(1) {
                ("+", name) => (true, name),
                ("-", name) => (false, name),
                _ => continue,
            };
            let Some(name) = binding(name) else {
                continue;
            };
            let input_key = settings.keys.get(name).expect("vanilla binding").binding();
            let allowed = ShellApp::game_input_allowed(&client);
            settings
                .keys
                .set_state_when_input_allowed(input_key, down, allowed);
        }
        for inbound in frame
            .get("inbound")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let blob = inbound["blob"].as_u64().expect("blob index") as usize;
            let packet = decode_inbound(&blobs[blob]);
            // S00 and S40 are the network task's in 1.7.10; the task answers S00 itself.
            if matches!(packet, S2CPlay5::KeepAlive(_) | S2CPlay5::Disconnect(_)) {
                continue;
            }
            to_game
                .try_send(ToGame::PacketV5(Box::new(packet)))
                .expect("inbound channel");
        }
        let mut sent = Vec::new();
        ShellApp::before_ticks(&mut client, &mut camera);
        drain_interactions(&client, &mut to_net, &mut sent);
        for _ in 0..frame["ticks"].as_u64().unwrap_or(1) {
            ticks_run += 1;
            ShellApp::tick(
                &mut client,
                &mut TickCtx {
                    input: &mut input,
                    settings: &mut settings,
                    camera: &mut camera,
                    tick: ticks_run,
                },
            );
            drain_interactions(&client, &mut to_net, &mut sent);
        }
        for _ in 0..wheel.unsigned_abs() {
            client.mouse_wheel(if wheel > 0 { 1.0 } else { -1.0 });
        }

        let java_sent: Vec<String> = frame["sent"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|sent| sent["hex"].as_str())
            .filter(|wire| COMPARED_IDS.contains(&packet_id(&unhex(wire))))
            .map(str::to_owned)
            .collect();
        if verbose {
            eprintln!("[{name} {index}] java {java_sent:?}\n      rust {sent:?}");
        }
        if outcome.packets.is_none() && java_sent != sent {
            outcome.packets = Some(format!(
                "frame {index}: interaction packets differ\n      java {java_sent:?}\n      \
                 rust {sent:?}"
            ));
        }
        if let Some(player) = frame.get("player").and_then(Value::as_object) {
            for (key, value) in player {
                java_state.insert(key.clone(), value.clone());
            }
        }
        if outcome.state.is_none()
            && client
                .session
                .as_ref()
                .is_some_and(|s| s.predictor().is_some())
        {
            // A key the Java side never wrote (`blocks` without a watch list) is not compared;
            // a JSON null is (a null pick, an empty hand).
            let differences: Vec<String> = rust_state(&client, watch)
                .into_iter()
                .filter_map(|(key, rust)| {
                    let java = java_state.get(key)?;
                    (!same_value(java, &rust)).then(|| format!("{key}: java {java} rust {rust}"))
                })
                .collect();
            if !differences.is_empty() {
                outcome.state = Some(format!(
                    "frame {index}: state differs: {}",
                    differences.join(", ")
                ));
            }
        }
    }
    Some(outcome)
}

/// The recorded scenario, the inbound blobs and the cells the scenario watches.
type Loaded = (Value, Vec<Vec<u8>>, Vec<mc_math::BlockPos>);

fn load(name: &str) -> Option<Loaded> {
    let read = |dir: &str, file: &str| -> Option<Value> {
        let path = format!("{dir}/{file}");
        match std::fs::read_to_string(&path) {
            Ok(text) => Some(serde_json::from_str(&text).expect("fixture json")),
            Err(error) => {
                eprintln!("skip: no 1.7.10 oracle fixture at {path}: {error}");
                None
            }
        }
    };
    let inbound = read(FIXTURE_DIR, "inbound.json")?;
    let blobs = inbound["blobs"]
        .as_array()
        .expect("blobs")
        .iter()
        .map(|blob| unhex(blob.as_str().expect("blob hex")))
        .collect();
    // java: `V17Scenario.watch`, `[[x, y, z], ...]`, reported after every frame.
    let definition = read(SCENARIO_DIR, &format!("{name}.json"))?;
    let watch = definition
        .get("watch")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|cell| {
            let at = |index: usize| {
                cell[index]
                    .as_i64()
                    .and_then(|value| i32::try_from(value).ok())
                    .expect("watched cell")
            };
            mc_math::BlockPos::new(at(0), at(1), at(2))
        })
        .collect();
    Some((read(FIXTURE_DIR, &format!("{name}.json"))?, blobs, watch))
}

/// Replay the named scenarios; panic with each one's first divergences.
fn check(names: &[&str]) {
    let only = std::env::var("V17_ONLY").ok();
    let mut report = String::new();
    let mut failures = 0;
    for name in names {
        if only
            .as_ref()
            .is_some_and(|only| !name.contains(only.as_str()))
        {
            continue;
        }
        let Some((scenario, blobs, watch)) = load(name) else {
            return;
        };
        let Some(outcome) = replay(name, &scenario, &blobs, &watch) else {
            return;
        };
        if outcome.packets.is_none() && outcome.state.is_none() {
            let _ = writeln!(report, "OK   {name} ({} frames)", outcome.frames);
            continue;
        }
        failures += 1;
        let _ = writeln!(report, "FAIL {name} ({} frames)", outcome.frames);
        for line in [&outcome.packets, &outcome.state].into_iter().flatten() {
            let _ = writeln!(report, "  {line}");
        }
    }
    eprintln!("{report}");
    assert_eq!(
        failures, 0,
        "{failures} scenario(s) diverge from vanilla 1.7.10:\n{report}"
    );
}

macro_rules! scenarios {
    ($($test:ident => $name:literal,)+) => {
        $(
            #[test]
            fn $test() {
                check(&[$name]);
            }
        )+
    };
}

scenarios! {
    focus_miss_cooldown => "focus_miss_cooldown",
    focus_adventure_glass => "focus_adventure_glass",
    focus_water_dig_eye => "focus_water_dig_eye",
    focus_dig_fatigue0 => "focus_dig_fatigue0",
    focus_dig_fatigue1 => "focus_dig_fatigue1",
    focus_dig_while_blocking => "focus_dig_while_blocking",
    focus_far_cursor => "focus_far_cursor",
    focus_place_double_plant => "focus_place_double_plant",
    focus_rightclick_while_digging => "focus_rightclick_while_digging",
    focus_tool_ladder => "focus_tool_ladder",
    focus_tool_melon => "focus_tool_melon",
    focus_tool_packed_ice => "focus_tool_packed_ice",
    focus_reach_entity_3blocks => "focus_reach_entity_3blocks",
    focus_yaw_head_lag => "focus_yaw_head_lag",
    focus_pick_before_packets => "focus_pick_before_packets",
    focus_remote_teleport_offset => "focus_remote_teleport_offset",
    dig_adventure_mode => "dig_adventure_mode",
    dig_fatigue_and_tools => "dig_fatigue_and_tools",
    dig_start_abort_finish => "dig_start_abort_finish",
    dig_while_using_place_while_digging => "dig_while_using_place_while_digging",
    place_blocks => "place_blocks",
    place_far_coordinates => "place_far_coordinates",
    place_sneak_bridge => "place_sneak_bridge",
    use_eat_and_bow => "use_eat_and_bow",
    combat_block_hit => "combat_block_hit",
    combat_entity_right_click => "combat_entity_right_click",
    combat_miss_penalty => "combat_miss_penalty",
    combat_pick_before_packets => "combat_pick_before_packets",
    combat_reach_block_behind => "combat_reach_block_behind",
    combat_sprint_hits_wtap => "combat_sprint_hits_wtap",
}
