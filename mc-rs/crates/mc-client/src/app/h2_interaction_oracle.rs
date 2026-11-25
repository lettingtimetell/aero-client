//! The end-to-end interaction differential oracle.
//!
//! `javaref/InteractionDiffRef.java` drives the **real recompiled 1.8.9 client** headless with
//! the movement oracle's harness (`javaref/MovementDiffRef.java`):
//! the real `Minecraft.runTick()`, `NetHandlerPlayClient`, `PlayerControllerMP`, `EntityPlayerSP`
//! and `EntityRenderer.getMouseOver`. Its scenarios are about what the client sends when the
//! player interacts: digging (C07), block placement and right-clicks on blocks (C08 face and hit
//! vector), item use and release, entity attack and interaction (C02 with the swing), held item
//! changes (C09), the drop key and pick block (C10 in creative).
//! `generate-interaction-diff.ps1` writes the fixture, in the movement oracle's format.
//!
//! This replays the same frames through the real Rust client exactly as the shell does
//! (`mc-render/src/shell.rs`), the way `h1_movement_oracle` does: render-time look on the
//! predictor, device edges through `KeyBindings::set_state_when_input_allowed` gated by
//! `ShellApp::game_input_allowed`, the inbound packets decoded from the same bytes into the
//! network channel, `ShellApp::before_ticks`, then `ShellApp::tick`, and a wheel notch handed to
//! `Client::mouse_wheel` after the tick.
//!
//! Every outbound packet must match Java's bytes, in order, in the same phase of the same frame
//! (the text of C17 `MC|Brand`, the client's name, is the one exception: the movement oracle's
//! `comparable`), and the local player's state must match after every frame.
//!
//! Skips (prints `skip:`) without the vanilla jar (`MC_JARS_DIR`) or the fixture.
//! `H2_ONLY=<substring>` replays only the matching scenarios; `H2_VERBOSE=1` prints every frame;
//! `H2_ALL=1` reports every frame whose packets differ, not only the first;
//! `H2_MODS_FILE=<path>` replays with that mod settings file.

use std::fmt::Write as _;

use super::*;

// What a sent packet is compared by: all of its bytes, except the text of C17 `MC|Brand` (the
// movement oracle's rule, tested there).
use super::h1_movement_oracle::comparable;
use crate::net::channel::NetHandle;
use crate::net::{ToGame, ToNet};
use mc_protocol::play::{C2SPlay, S2CPlay};
use mc_protocol::PacketBuffer;
use mc_render::input::keys;
use mc_render::{ShellApp, TickCtx};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h2_interaction/interaction_diff.bin"
);
/// The movement oracle's format (`MovementDiffRef.Writer`).
const MAGIC: &[u8; 8] = b"MCH1DF01";
/// java: `MovementDiffRef.SELF_UUID`, the local profile (named "H1" on both sides).
const LOCAL_UUID: u128 = 0x4831_f00d_0000_3000_8000_0000_0000_0001;

/// java: `MovementDiffRef.B_*`, vanilla default keys.
fn binding_name(binding: i32) -> &'static str {
    match binding {
        0 => keys::ATTACK,
        1 => keys::USE_ITEM,
        2 => keys::FORWARD,
        3 => keys::LEFT,
        4 => keys::BACK,
        5 => keys::RIGHT,
        6 => keys::JUMP,
        7 => keys::SNEAK,
        8 => keys::SPRINT,
        9 => keys::DROP,
        10 => keys::INVENTORY,
        11 => keys::PICK_ITEM,
        12..=20 => keys::hotbar_static((binding - 11) as u8),
        21 => keys::CHAT,
        other => panic!("fixture names unknown binding {other}"),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct State {
    pos: [f64; 3],
    vel: [f64; 3],
    yaw: f32,
    pitch: f32,
    flags: u16,
    held_slot: i32,
    health: f32,
    food: i32,
    fall_distance: f32,
    item_in_use_count: i32,
    sprinting_ticks_left: i32,
    move_forward: f32,
    move_strafing: f32,
    entity_id: i32,
    dimension: i32,
    ticks_existed: i32,
}

const F_ON_GROUND: u16 = 1;
const F_SPRINTING: u16 = 2;
const F_SNEAKING: u16 = 4;
const F_FLYING: u16 = 8;
const F_COLLIDED_H: u16 = 16;
const F_USING_ITEM: u16 = 32;
const F_RIDING: u16 = 64;
const F_SCREEN: u16 = 128;
const F_DEAD: u16 = 256;

struct Frame {
    inbound: Vec<u32>,
    tick: bool,
    yaw: f32,
    pitch: f32,
    events: Vec<(u8, i32, i32)>,
    sent: Vec<(u8, Vec<u8>)>,
    state: Option<State>,
}

struct Scenario {
    name: String,
    frames: Vec<Frame>,
}

struct Fixture {
    blobs: Vec<Vec<u8>>,
    scenarios: Vec<Scenario>,
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> &'a [u8] {
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        out
    }
    fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    fn u16(&mut self) -> u16 {
        u16::from_be_bytes(self.take(2).try_into().unwrap())
    }
    fn u32(&mut self) -> u32 {
        u32::from_be_bytes(self.take(4).try_into().unwrap())
    }
    fn i32(&mut self) -> i32 {
        i32::from_be_bytes(self.take(4).try_into().unwrap())
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(u64::from_be_bytes(self.take(8).try_into().unwrap()))
    }
    fn len_prefixed(&mut self) -> Vec<u8> {
        let n = self.u32() as usize;
        self.take(n).to_vec()
    }
}

fn read_state(r: &mut Reader<'_>) -> State {
    State {
        pos: [r.f64(), r.f64(), r.f64()],
        vel: [r.f64(), r.f64(), r.f64()],
        yaw: r.f32(),
        pitch: r.f32(),
        flags: r.u16(),
        held_slot: r.i32(),
        health: r.f32(),
        food: r.i32(),
        fall_distance: r.f32(),
        item_in_use_count: r.i32(),
        sprinting_ticks_left: r.i32(),
        move_forward: r.f32(),
        move_strafing: r.f32(),
        entity_id: r.i32(),
        dimension: r.i32(),
        ticks_existed: r.i32(),
    }
}

fn load_fixture() -> Option<Fixture> {
    let bytes = match std::fs::read(FIXTURE) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("skip: no interaction differential fixture at {FIXTURE}: {error}");
            return None;
        }
    };
    let mut r = Reader {
        bytes: &bytes,
        at: 0,
    };
    assert_eq!(r.take(8), MAGIC, "fixture magic");
    let blobs = (0..r.u32()).map(|_| r.len_prefixed()).collect();
    let mut scenarios = Vec::new();
    for _ in 0..r.u32() {
        let name_len = usize::from(r.u16());
        let name = String::from_utf8(r.take(name_len).to_vec()).expect("utf-8 scenario name");
        let mut frames = Vec::new();
        for _ in 0..r.u32() {
            let inbound = (0..r.u32()).map(|_| r.u32()).collect();
            let tick = r.u8() != 0;
            let yaw = r.f32();
            let pitch = r.f32();
            let events = (0..r.u32()).map(|_| (r.u8(), r.i32(), r.i32())).collect();
            let sent = (0..r.u32())
                .map(|_| {
                    let phase = r.u8();
                    (phase, r.len_prefixed())
                })
                .collect();
            let state = (r.u8() != 0).then(|| read_state(&mut r));
            frames.push(Frame {
                inbound,
                tick,
                yaw,
                pitch,
                events,
                sent,
                state,
            });
        }
        scenarios.push(Scenario { name, frames });
    }
    assert_eq!(r.at, bytes.len(), "trailing fixture bytes");
    Some(Fixture { blobs, scenarios })
}

/// A real client on a temp game dir with the mods at their defaults, except `modern_keybinds`
/// (on by default; its re-press of keys held through a screen is a deliberate non-vanilla
/// feature), as in the movement oracle.
///
/// `H2_MODS_FILE=<path>` replays with that mod settings file instead (for example a copy of a
/// player's own), to see which scenarios a configuration takes away from vanilla.
fn vanilla_client() -> Option<(Client, tempfile::TempDir)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let settings = match std::env::var_os("H2_MODS_FILE") {
        Some(path) => std::fs::read_to_string(path).expect("read H2_MODS_FILE"),
        None => r#"{"format":1,"mods":{"modern_keybinds":{"enabled":false}}}"#.to_owned(),
    };
    std::fs::write(dir.path().join(mc_mods::SETTINGS_FILE_NAME), settings)
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
    let client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    Some((client, dir))
}

fn decode_inbound(wire: &[u8]) -> S2CPlay {
    let mut buf = PacketBuffer::from_slice(wire);
    let id = buf.read_varint().expect("inbound packet id");
    let packet = S2CPlay::read(id, &mut buf)
        .unwrap_or_else(|error| panic!("Rust cannot decode inbound 0x{id:02X}: {error:?}"));
    assert_eq!(
        buf.remaining(),
        0,
        "inbound 0x{id:02X} left {} bytes undecoded",
        buf.remaining()
    );
    packet
}

fn encode_outbound(packet: &C2SPlay) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_varint(packet.id());
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

/// One serverbound packet, readable.
fn describe(wire: &[u8]) -> String {
    let mut buf = PacketBuffer::from_slice(wire);
    let Ok(id) = buf.read_varint() else {
        return format!("<undecodable {wire:02X?}>");
    };
    match C2SPlay::read(id, &mut buf) {
        Ok(packet) => format!("{packet:?}"),
        Err(error) => format!("<0x{id:02X} {error:?} {wire:02X?}>"),
    }
}

fn drain(
    to_net: &mut tokio::sync::mpsc::UnboundedReceiver<ToNet>,
    phase: u8,
    out: &mut Vec<(u8, Vec<u8>)>,
) {
    while let Ok(message) = to_net.try_recv() {
        match message {
            ToNet::Send(packet) => out.push((phase, encode_outbound(&packet))),
            ToNet::Disconnect => out.push((phase, b"<disconnect>".to_vec())),
            // Never sent on a 1.8.9 connection: kept visible as a mismatch if it ever is.
            ToNet::SendV5(packet) => out.push((phase, format!("<1.7.10 {packet:?}>").into_bytes())),
        }
    }
}

fn rust_state(client: &Client) -> Option<State> {
    let session = client.session.as_ref()?;
    let predictor = session.predictor()?;
    let s = &predictor.state;
    let local = &session.state().local;
    let mut flags = 0;
    if s.on_ground {
        flags |= F_ON_GROUND;
    }
    if s.sprinting {
        flags |= F_SPRINTING;
    }
    if s.sneaking {
        flags |= F_SNEAKING;
    }
    if s.flying {
        flags |= F_FLYING;
    }
    if s.collided_horizontally {
        flags |= F_COLLIDED_H;
    }
    if client.controller.is_using_item() {
        flags |= F_USING_ITEM;
    }
    if s.riding {
        flags |= F_RIDING;
    }
    if !matches!(client.ui, Ui::InGame) {
        flags |= F_SCREEN;
    }
    if local.health <= 0.0 {
        flags |= F_DEAD;
    }
    Some(State {
        pos: [s.pos.x, s.pos.y, s.pos.z],
        vel: [s.vel.x, s.vel.y, s.vel.z],
        yaw: s.yaw,
        pitch: s.pitch,
        flags,
        held_slot: client.player_hud.selected_slot,
        health: local.health,
        // java: `FoodStats.foodLevel`, written as soon as S06 is handled. The session's value is
        // its owner; the physics copy is refreshed only at the top of a tick (the finding),
        // so after an S06 in a frame with no tick it would report a difference the client lacks.
        food: local.food_level,
        fall_distance: s.fall_distance,
        item_in_use_count: 0,
        sprinting_ticks_left: s.sprinting_ticks_left,
        move_forward: s.move_forward,
        move_strafing: s.move_strafing,
        entity_id: local.entity_id,
        dimension: local.dimension,
        ticks_existed: 0,
    })
}

/// The fields both clients expose with the same meaning.
fn state_differences(java: &State, rust: &State) -> Vec<String> {
    let mut out = Vec::new();
    let mut check = |name: &str, same: bool, j: String, r: String| {
        if !same {
            out.push(format!("{name}: java {j} rust {r}"));
        }
    };
    for (axis, (j, r)) in ["x", "y", "z"].iter().zip(java.pos.iter().zip(rust.pos)) {
        check(
            &format!("pos.{axis}"),
            j.to_bits() == r.to_bits(),
            format!("{j:?}"),
            format!("{r:?}"),
        );
    }
    for (axis, (j, r)) in ["x", "y", "z"].iter().zip(java.vel.iter().zip(rust.vel)) {
        check(
            &format!("vel.{axis}"),
            j.to_bits() == r.to_bits(),
            format!("{j:?}"),
            format!("{r:?}"),
        );
    }
    check(
        "yaw",
        java.yaw.to_bits() == rust.yaw.to_bits(),
        format!("{:?}", java.yaw),
        format!("{:?}", rust.yaw),
    );
    check(
        "pitch",
        java.pitch.to_bits() == rust.pitch.to_bits(),
        format!("{:?}", java.pitch),
        format!("{:?}", rust.pitch),
    );
    for (bit, name) in [
        (F_ON_GROUND, "onGround"),
        (F_SPRINTING, "sprinting"),
        (F_SNEAKING, "sneaking"),
        (F_FLYING, "flying"),
        (F_COLLIDED_H, "collidedHorizontally"),
        (F_USING_ITEM, "usingItem"),
        (F_RIDING, "riding"),
        (F_SCREEN, "screenOpen"),
        (F_DEAD, "dead"),
    ] {
        let (j, r) = (java.flags & bit != 0, rust.flags & bit != 0);
        check(name, j == r, j.to_string(), r.to_string());
    }
    check(
        "heldSlot",
        java.held_slot == rust.held_slot,
        java.held_slot.to_string(),
        rust.held_slot.to_string(),
    );
    check(
        "fallDistance",
        java.fall_distance.to_bits() == rust.fall_distance.to_bits(),
        format!("{:?}", java.fall_distance),
        format!("{:?}", rust.fall_distance),
    );
    check(
        "sprintingTicksLeft",
        java.sprinting_ticks_left == rust.sprinting_ticks_left,
        java.sprinting_ticks_left.to_string(),
        rust.sprinting_ticks_left.to_string(),
    );
    check(
        "moveForward",
        java.move_forward.to_bits() == rust.move_forward.to_bits(),
        format!("{:?}", java.move_forward),
        format!("{:?}", rust.move_forward),
    );
    check(
        "moveStrafing",
        java.move_strafing.to_bits() == rust.move_strafing.to_bits(),
        format!("{:?}", java.move_strafing),
        format!("{:?}", rust.move_strafing),
    );
    check(
        "health",
        java.health.to_bits() == rust.health.to_bits(),
        format!("{:?}", java.health),
        format!("{:?}", rust.health),
    );
    check(
        "food",
        java.food == rust.food,
        java.food.to_string(),
        rust.food.to_string(),
    );
    out
}

/// What one scenario's replay found.
#[derive(Default)]
struct Outcome {
    /// The first frame whose compared packets differ, with both lists.
    packets: Option<String>,
    /// The first frame whose player state differs, with every differing field.
    state: Option<String>,
    frames: usize,
}

fn packet_lines(list: &[(u8, Vec<u8>)]) -> String {
    let mut out = String::new();
    for (phase, wire) in list {
        let _ = writeln!(out, "      [{phase}] {}", describe(wire));
    }
    if out.is_empty() {
        out.push_str("      (none)\n");
    }
    out
}

fn replay(fixture: &Fixture, scenario: &Scenario) -> Option<Outcome> {
    let verbose = std::env::var_os("H2_VERBOSE").is_some();
    let all_frames = std::env::var_os("H2_ALL").is_some();
    let (mut client, _dir) = vanilla_client()?;
    let (handle, to_game, mut to_net) = NetHandle::new_pair();
    // What `Client::connect` joins with: the client's own options and its brand choice.
    let join = client.join_settings();
    client.session = Some(Session::new(handle, join));
    // The login a live connection completes first: Java's local profile is the same UUID/name.
    to_game
        .try_send(ToGame::Connected(mc_protocol::connection::LoginOutcome {
            uuid: uuid::Uuid::from_u128(LOCAL_UUID),
            username: "H1".into(),
            compression_threshold: None,
            encrypted: false,
        }))
        .expect("inbound channel");
    let mut camera = menu_camera();
    let mut input = InputState::new();
    let mut settings = GameSettings::default();
    let mut ticks = 0u64;
    let mut outcome = Outcome {
        frames: scenario.frames.len(),
        ..Outcome::default()
    };

    for (index, frame) in scenario.frames.iter().enumerate() {
        // (1) The previous frame's render-time look (`turn_local_player` writes the predictor).
        if !frame.yaw.is_nan() {
            if let Some(predictor) = client.session.as_mut().and_then(Session::predictor_mut) {
                predictor.state.yaw = frame.yaw;
                predictor.state.pitch = frame.pitch;
            }
        }
        // (2) Device edges, as shell.rs routes them between draws.
        let mut wheel = 0i32;
        for &(kind, a, b) in &frame.events {
            match kind {
                0 => {
                    let input = settings
                        .keys
                        .get(binding_name(a))
                        .expect("vanilla binding")
                        .binding();
                    let allowed = ShellApp::game_input_allowed(&client);
                    settings
                        .keys
                        .set_state_when_input_allowed(input, b != 0, allowed);
                }
                1 => wheel += a,
                other => panic!("unknown event kind {other}"),
            }
        }
        // (3) The frame's scheduled packets, then (4) its tick.
        for &blob in &frame.inbound {
            let packet = decode_inbound(&fixture.blobs[blob as usize]);
            to_game
                .try_send(ToGame::Packet(packet))
                .expect("inbound channel");
        }
        let mut sent = Vec::new();
        ShellApp::before_ticks(&mut client, &mut camera);
        drain(&mut to_net, 0, &mut sent);
        if frame.tick {
            ticks += 1;
            ShellApp::tick(
                &mut client,
                &mut TickCtx {
                    input: &mut input,
                    settings: &mut settings,
                    camera: &mut camera,
                    tick: ticks,
                },
            );
        }
        drain(&mut to_net, 1, &mut sent);
        // `ShellApp::frame` -> `handle_screen_input` delivers the wheel after the frame's ticks,
        // one `mouse_wheel` per winit event (one +-1.0 event per notch, as Java's harness pushes
        // one +-120 LWJGL event per notch).
        for _ in 0..wheel.unsigned_abs() {
            client.mouse_wheel(if wheel > 0 { 1.0 } else { -1.0 });
        }

        let java: Vec<_> = frame
            .sent
            .iter()
            .map(|(phase, wire)| (*phase, comparable(wire).to_vec()))
            .collect();
        let rust: Vec<_> = sent
            .iter()
            .map(|(phase, wire)| (*phase, comparable(wire).to_vec()))
            .collect();
        let rust_now = rust_state(&client);
        if verbose {
            eprintln!(
                "[{} {index}] java {:?}\n        rust {:?}",
                scenario.name,
                java.iter().map(|(p, w)| (*p, w[0])).collect::<Vec<_>>(),
                rust.iter().map(|(p, w)| (*p, w[0])).collect::<Vec<_>>()
            );
        }
        if java != rust {
            let difference = format!(
                "frame {index}: packets differ\n    java:\n{}    rust:\n{}",
                packet_lines(&java),
                packet_lines(&rust)
            );
            match &mut outcome.packets {
                None => outcome.packets = Some(difference),
                // `H2_ALL=1` lists every differing frame (up to 40), not only the first.
                Some(all) if all_frames && all.matches("packets differ").count() < 40 => {
                    all.push_str("\n  ");
                    all.push_str(&difference);
                }
                Some(_) => {}
            }
        }
        if outcome.state.is_none() {
            match (&frame.state, &rust_now) {
                (Some(j), Some(r)) => {
                    let differences = state_differences(j, r);
                    if !differences.is_empty() {
                        outcome.state = Some(format!(
                            "frame {index}: state differs\n      {}",
                            differences.join("\n      ")
                        ));
                    }
                }
                (None, None) => {}
                (j, r) => {
                    outcome.state = Some(format!(
                        "frame {index}: player presence differs (java {}, rust {})",
                        j.is_some(),
                        r.is_some()
                    ));
                }
            }
        }
    }
    Some(outcome)
}

/// Replay the named scenarios; panic with each one's first packet and first state divergence.
fn check(names: &[&str]) {
    let Some(fixture) = load_fixture() else {
        return;
    };
    let only = std::env::var("H2_ONLY").ok();
    let mut report = String::new();
    let mut failures = 0;
    for name in names {
        if only
            .as_ref()
            .is_some_and(|only| !name.contains(only.as_str()))
        {
            continue;
        }
        let scenario = fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.name == *name)
            .unwrap_or_else(|| panic!("the fixture has no scenario {name:?}; regenerate it"));
        let Some(outcome) = replay(&fixture, scenario) else {
            return;
        };
        if outcome.packets.is_none() && outcome.state.is_none() {
            let _ = writeln!(report, "OK   {} ({} frames)", scenario.name, outcome.frames);
            continue;
        }
        failures += 1;
        let _ = writeln!(report, "FAIL {} ({} frames)", scenario.name, outcome.frames);
        if let Some(packets) = &outcome.packets {
            let _ = writeln!(report, "  {packets}");
        }
        if let Some(state) = &outcome.state {
            let _ = writeln!(report, "  {state}");
        }
    }
    eprintln!("{report}");
    assert_eq!(
        failures, 0,
        "{failures} scenario(s) diverge from vanilla:\n{report}"
    );
}

/// One test per fixture scenario, so the runner parallelises them and a failure names its case.
macro_rules! scenarios {
    ($($test:ident => $name:literal,)+) => {
        $(
            #[test]
            fn $test() {
                check(&[$name]);
            }
        )+

        /// Every scenario the Java oracle wrote has a test above (a new scenario cannot be
        /// silently skipped), and every test names a scenario the fixture has.
        #[test]
        fn every_fixture_scenario_is_replayed() {
            let Some(fixture) = load_fixture() else {
                return;
            };
            let mut declared = vec![$($name),+];
            let mut written: Vec<&str> =
                fixture.scenarios.iter().map(|scenario| scenario.name.as_str()).collect();
            declared.sort_unstable();
            written.sort_unstable();
            assert_eq!(declared, written);
        }
    };
}

scenarios! {
    dig_break_and_continue => "dig_break_and_continue",
    dig_release_look_away_retarget => "dig_release_look_away_retarget",
    dig_hotbar_switch_mid_dig => "dig_hotbar_switch_mid_dig",
    dig_instant_inside_held_dig => "dig_instant_inside_held_dig",
    dig_creative_every_five_ticks => "dig_creative_every_five_ticks",
    dig_haste_efficiency_airborne => "dig_haste_efficiency_airborne",
    place_hit_vector_sweep => "place_hit_vector_sweep",
    place_stone_cases => "place_stone_cases",
    place_against_collision_boxes => "place_against_collision_boxes",
    right_click_gallery_empty_hand => "right_click_gallery_empty_hand",
    right_click_gallery_with_block => "right_click_gallery_with_block",
    right_click_gallery_sneaking_with_block => "right_click_gallery_sneaking_with_block",
    right_click_item_sweep => "right_click_item_sweep",
    item_use_at_blocks_and_release => "item_use_at_blocks_and_release",
    entity_attack_and_interact => "entity_attack_and_interact",
    drop_key_and_survival_pick_block => "drop_key_and_survival_pick_block",
    creative_pick_block_and_place => "creative_pick_block_and_place",
    adventure_mode_lobby_clicks => "adventure_mode_lobby_clicks",
    spectator_mode_clicks => "spectator_mode_clicks",
    combined_inputs => "combined_inputs",
    entity_item_sweep => "entity_item_sweep",
    right_click_items_on_a_wall => "right_click_items_on_a_wall",
    items_on_their_blocks => "items_on_their_blocks",
    sprint_attacks_on_objects => "sprint_attacks_on_objects",
}
