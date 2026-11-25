//! The end-to-end movement differential oracle.
//!
//! `javaref/MovementDiffRef.java` drives the **real recompiled 1.8.9 client** headless: the real
//! `Minecraft.runTick()` (LWJGL stubbed to an always-focused window whose event queues the
//! harness fills), the real `NetHandlerPlayClient` handlers, `PlayerControllerMP`, `WorldClient`,
//! `EntityPlayerSP` and `MovementInputFromOptions`. For every scenario it records, frame by
//! frame, the inbound packets (as wire bytes), the device events, every outbound packet's bytes
//! and the local player's state. `generate-movement-diff.ps1` writes the fixture.
//!
//! This replays the same frames through the real Rust client exactly as the shell does
//! (`mc-render/src/shell.rs`): render-time look on the predictor, device edges through
//! `KeyBindings::set_state_when_input_allowed` gated by `ShellApp::game_input_allowed`, the
//! inbound packets decoded from the same bytes into the network channel, `ShellApp::before_ticks`
//! (the scheduled-task drain), then `ShellApp::tick`. A wheel notch is handed to
//! `Client::mouse_wheel` after the tick, where `ShellApp::frame`'s `handle_screen_input`
//! delivers it.
//!
//! Every outbound packet must match Java's bytes, in order, in the same phase of the same frame.
//! The one exception is the text of C17 `MC|Brand`, the client's name, which is the player's
//! choice: that packet is compared by its id and channel only (`comparable`). C15 (client
//! settings) and every other custom payload are compared in full. The local player's position,
//! motion, rotation and flags must match after every frame.
//!
//! Skips (prints `skip:`) without the vanilla jar (`MC_JARS_DIR`) or the fixture.
//! `H1_ONLY=<substring>` replays only the matching scenarios; `H1_VERBOSE=1` prints every frame.
//! `H1_FIXTURE=<path>` makes `replay_every_scenario_of_an_override_fixture` replay every
//! scenario of another fixture, such as a hunting run with many more random seeds.

use std::fmt::Write as _;

use super::*;

use crate::net::channel::NetHandle;
use crate::net::{ToGame, ToNet};
use mc_protocol::play::{C2SPlay, S2CPlay};
use mc_protocol::PacketBuffer;
use mc_render::input::keys;
use mc_render::{ShellApp, TickCtx};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/h1_movement/movement_diff.bin"
);
const MAGIC: &[u8; 8] = b"MCH1DF01";
/// java: `MovementDiffRef.SELF_UUID`, the local profile.
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
    /// `runTick`s after the packets: 0 (a render-only frame), 1, or more for a slow frame
    /// (`Timer.elapsedTicks`, at most 10).
    ticks: u8,
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
    load_fixture_from(FIXTURE)
}

fn load_fixture_from(path: &str) -> Option<Fixture> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            eprintln!("skip: no movement differential fixture at {path}: {error}");
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
            let ticks = r.u8();
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
                ticks,
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

/// A real client on a temp game dir, every packet-relevant mod off.
///
/// Only `modern_keybinds` is on by default among the mods whose ON state is not vanilla (its
/// re-press after a screen closes is a deliberate Aero feature); the rest are visual or are the
/// vanilla features themselves (fog, vignette, tall grass, dynamic FOV).
fn vanilla_client() -> Option<(Client, tempfile::TempDir)> {
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

/// The start of the C17 that names the client: packet id 0x17, then the channel `MC|Brand` as a
/// protocol string (its length, 8, then the name).
const BRAND_HEAD: [u8; 10] = [0x17, 8, b'M', b'C', b'|', b'B', b'r', b'a', b'n', b'd'];

/// What a sent packet is compared by: all of its bytes, except for C17 `MC|Brand`.
///
/// The brand is the client's name (`Aero`, `mods/client_brand.rs`; vanilla says `vanilla`), so
/// its text is cut off and the packet is compared by its id and channel: it
/// must still go out in Java's frame, phase and order. Every other packet, C15 (client settings)
/// and every other custom payload (book edits and signing, item names, beacon and trade choices)
/// included, must match Java's bytes. The interaction oracle compares by the same rule.
pub(super) fn comparable(wire: &[u8]) -> &[u8] {
    if wire.starts_with(&BRAND_HEAD) {
        &wire[..BRAND_HEAD.len()]
    } else {
        wire
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
        // java: `FoodStats.foodLevel`, which `handleUpdateHealth` writes. Its Rust owner is the
        // session's `local.food_level` (set by the S06 dispatch); the physics state holds a copy
        // refreshed at the top of every local tick for the sprint gate, so after a render-only
        // frame that copy is one tick behind by design.
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

/// The fields both clients expose with the same meaning. Each entry: name, java, rust.
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
    replay_with(fixture, scenario, &|_| true, &|_, _| {})
}

/// [`replay`] with two test-only interventions for the negative controls: `keep` withholds every
/// inbound packet it returns `false` for (it sees the wire bytes, `[VarInt id][body]`), and
/// `hook` edits the client after a frame's packets were handled, before its tick.
fn replay_with(
    fixture: &Fixture,
    scenario: &Scenario,
    keep: &dyn Fn(&[u8]) -> bool,
    hook: &dyn Fn(usize, &mut Client),
) -> Option<Outcome> {
    let verbose = std::env::var_os("H1_VERBOSE").is_some();
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
            let wire = &fixture.blobs[blob as usize];
            if !keep(wire) {
                continue;
            }
            let packet = decode_inbound(wire);
            to_game
                .try_send(ToGame::Packet(packet))
                .expect("inbound channel");
        }
        let mut sent = Vec::new();
        ShellApp::before_ticks(&mut client, &mut camera);
        drain(&mut to_net, 0, &mut sent);
        hook(index, &mut client);
        // shell.rs: `before_ticks` once, then `tick` for each of the frame's elapsed ticks.
        for _ in 0..frame.ticks {
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
        // one `mouse_wheel` per winit event: a standard wheel sends one +-1.0 event per notch
        // (Java's harness pushes one +-120 LWJGL event per notch).
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
        if outcome.packets.is_none() && java != rust {
            outcome.packets = Some(format!(
                "frame {index}: packets differ\n    java:\n{}    rust:\n{}",
                packet_lines(&java),
                packet_lines(&rust)
            ));
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
    check_in(&fixture, names);
}

/// [`check`] against an already loaded fixture.
fn check_in(fixture: &Fixture, names: &[&str]) {
    let only = std::env::var("H1_ONLY").ok();
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
        let Some(outcome) = replay(fixture, scenario) else {
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

/// Swap every block whose id is in `from` for `to` across the EDGE world (x and z in [-16, 31])
/// in the client's own world, as if the server had sent plain blocks there.
fn substitute_edge_blocks(client: &mut Client, from: &[u16], to: mc_data::BlockState) {
    let Some(session) = client.session.as_mut() else {
        return;
    };
    let world = &mut session.state_mut().world;
    for x in -16..=31 {
        for z in -16..=31 {
            for y in 56..=76 {
                let pos = mc_math::BlockPos::new(x, y, z);
                if from.contains(&(world.block_at(pos).id >> 4)) {
                    world.set_block(pos, to);
                }
            }
        }
    }
}

/// Negative controls: every scenario added for these cases must depend on what it was
/// written to exercise. Withholding those server packets, or swapping its feature blocks in the
/// client's world for plain ones once the chunks are in, has to make it diverge from the Java
/// trace; a scenario that still matched would pass whether or not the client handled them.
#[test]
fn the_wave_2026_11_19_scenarios_diverge_without_their_packets_or_blocks() {
    let Some(fixture) = load_fixture() else {
        return;
    };
    let find = |name: &str| {
        fixture
            .scenarios
            .iter()
            .find(|scenario| scenario.name == name)
            .unwrap_or_else(|| panic!("the fixture has no scenario {name:?}; regenerate it"))
    };
    // The first wire byte is the packet id: every 1.8 play id is below 0x80.
    const S06: u8 = 0x06;
    const S08: u8 = 0x08;
    const S12: u8 = 0x12;
    const S1D: u8 = 0x1D;
    const S1E: u8 = 0x1E;
    const S20: u8 = 0x20;
    const S23: u8 = 0x23;
    const S27: u8 = 0x27;
    const S44: u8 = 0x44;
    const S15: u8 = 0x15;
    let packet_controls: &[(&str, &[u8])] = &[
        ("edge_ice_slime_soul_sand_sprint_jumps", &[S12]),
        ("edge_slime_fall_bounce_and_sneak_landing", &[S12]),
        ("edge_soul_sand_and_cobwebs", &[S12]),
        ("edge_soul_sand_and_cobwebs", &[S1D, S1E]),
        ("edge_water_flow_and_lava_swim", &[S27]),
        ("edge_water_flow_and_lava_swim", &[S20]),
        ("edge_stairs_slabs_step_up", &[S20]),
        ("edge_fence_wall_pane_rows_jump_boost", &[S1D, S1E]),
        ("edge_ladder_vine_climb_and_knockback", &[S12]),
        ("edge_sneak_platform_edges_and_knockback", &[S12]),
        ("edge_potion_jump_boost_amplifiers_and_expiry", &[S1D]),
        ("edge_potion_jump_boost_amplifiers_and_expiry", &[S1E]),
        ("edge_potion_jump_boost_amplifiers_and_expiry", &[S20]),
        ("world_border_walls_and_latch", &[S44]),
        ("edge_random_play_0", &[S12]),
        ("edge_random_play_1", &[S06, S08, S1D, S1E, S20]),
        ("edge_random_play_2", &[S23, S27]),
        ("edge_random_play_3", &[S12]),
        ("low_fps_multi_tick_frames", &[S12, S08]),
        ("edge_random_paced_0", &[S12]),
        ("edge_random_paced_1", &[S12]),
        // The target's moves decide which clicks hit; the target's knockback moves the player.
        ("combat_random_0", &[S15]),
        ("combat_random_0", &[S12]),
        ("combat_random_paced_0", &[S15]),
        ("combat_random_paced_0", &[S12]),
    ];
    // Block ids (`BlockState::id >> 4`).
    let stone = mc_data::BlockState::new(mc_data::BlockId(1), 0);
    let air = mc_data::BlockState::AIR;
    let block_controls: &[(&str, &[u16], mc_data::BlockState)] = &[
        ("edge_ice_slime_soul_sand_sprint_jumps", &[79, 174], stone),
        ("edge_ice_slime_soul_sand_sprint_jumps", &[165], stone),
        ("edge_ice_slime_soul_sand_sprint_jumps", &[88], stone),
        ("edge_slime_fall_bounce_and_sneak_landing", &[165], stone),
        ("edge_soul_sand_and_cobwebs", &[30], air),
        ("edge_soul_sand_and_cobwebs", &[88], stone),
        ("edge_water_flow_and_lava_swim", &[8, 9], air),
        ("edge_water_flow_and_lava_swim", &[10, 11], air),
        ("edge_stairs_slabs_step_up", &[53, 67], stone),
        ("edge_stairs_slabs_step_up", &[44], stone),
        ("edge_stairs_slabs_step_up", &[78, 171], air),
        ("edge_fence_wall_pane_rows_jump_boost", &[85, 107], air),
        ("edge_fence_wall_pane_rows_jump_boost", &[139], air),
        ("edge_fence_wall_pane_rows_jump_boost", &[101, 102], air),
        ("edge_ladder_vine_climb_and_knockback", &[65], air),
        ("edge_ladder_vine_climb_and_knockback", &[106], air),
        ("edge_sneak_platform_edges_and_knockback", &[44], air),
        ("edge_sneak_platform_edges_and_knockback", &[67], stone),
    ];
    let mut still_matching = Vec::new();
    for &(name, dropped) in packet_controls {
        let outcome = replay_with(
            &fixture,
            find(name),
            &|wire| !dropped.contains(&wire[0]),
            &|_, _| {},
        );
        let Some(outcome) = outcome else {
            return;
        };
        if outcome.packets.is_none() && outcome.state.is_none() {
            still_matching.push(format!("{name} without packets {dropped:02X?}"));
        }
    }
    for &(name, from, to) in block_controls {
        // The EDGE chunks arrive in frame 2; keep re-applying through frame 4 in case a column
        // lands a frame later (none of these scenarios sends a block change of its own).
        let outcome = replay_with(&fixture, find(name), &|_| true, &|index, client| {
            if (2..=4).contains(&index) {
                substitute_edge_blocks(client, from, to);
            }
        });
        let Some(outcome) = outcome else {
            return;
        };
        if outcome.packets.is_none() && outcome.state.is_none() {
            still_matching.push(format!("{name} with blocks {from:?} replaced by {}", to.id));
        }
    }
    assert!(
        still_matching.is_empty(),
        "these negative controls still match the Java trace, so the scenario does not \
         exercise what it claims:\n  {}",
        still_matching.join("\n  ")
    );
}

/// Only the brand's text is left out of the comparison: a brand with any text compares equal to
/// Java's `vanilla`, while every other custom payload (also a channel whose name merely starts
/// with the brand's, or one whose data holds it) and C15 are compared byte for byte.
#[test]
fn only_the_brand_text_is_left_out_of_the_comparison() {
    use mc_protocol::play::c2s;
    let payload = |channel: &str, data: &[u8]| {
        encode_outbound(&C2SPlay::PluginMessage(c2s::PluginMessage {
            channel: channel.to_owned(),
            data: data.to_vec(),
        }))
    };
    let string = |text: &str| {
        let mut buf = PacketBuffer::new();
        buf.write_string(text);
        buf.as_slice().to_vec()
    };
    let vanilla = payload("MC|Brand", &string("vanilla"));
    assert_eq!(&vanilla[..BRAND_HEAD.len()], &BRAND_HEAD[..]);
    assert_eq!(comparable(&vanilla), &BRAND_HEAD[..]);
    for brand in ["Aero", "", "vanilla but longer"] {
        let wire = payload("MC|Brand", &string(brand));
        assert_eq!(comparable(&wire), comparable(&vanilla), "brand {brand:?}");
    }
    for (channel, data) in [
        ("MC|BEdit", vec![1, 2, 3]),
        ("MC|BSign", vec![4, 5]),
        ("MC|ItemName", string("Sword")),
        ("MC|Beacon", vec![0, 0, 0, 1, 0, 0, 0, 0]),
        ("MC|TrSel", vec![0, 0, 0, 2]),
        ("MC|BrandX", string("vanilla")),
        ("REGISTER", b"MC|Brand".to_vec()),
    ] {
        let wire = payload(channel, &data);
        assert_eq!(
            comparable(&wire),
            &wire[..],
            "{channel} is compared in full"
        );
    }
    let settings = encode_outbound(&C2SPlay::ClientSettings(c2s::ClientSettings {
        locale: "en_US".to_owned(),
        view_distance: 12,
        chat_visibility: c2s::ChatVisibility::Full,
        chat_colors: true,
        displayed_skin_parts: 0x7F,
    }));
    assert_eq!(
        comparable(&settings),
        &settings[..],
        "C15 is compared in full"
    );
}

/// The comparison catches what the client sends and Java did not: an extra custom payload (an
/// item name, as an anvil sends it) or an extra C15 in a scenario that otherwise matches the
/// Java trace. Before only the brand's text was exempt, every C15 and C17 was skipped, so both
/// of these passed.
#[test]
fn a_custom_payload_or_c15_java_did_not_send_fails_the_comparison() {
    use mc_protocol::play::c2s;
    let Some(fixture) = load_fixture() else {
        return;
    };
    let scenario = fixture
        .scenarios
        .iter()
        .find(|scenario| scenario.name == "join_walk_sprint_jump_sneak")
        .expect("the fixture has the join scenario");
    // Well after the join: the session is playing, so `Session::send` queues the packet.
    const FRAME: usize = 12;
    assert!(scenario.frames.len() > FRAME);
    let Some(clean) = replay(&fixture, scenario) else {
        return;
    };
    assert!(
        clean.packets.is_none() && clean.state.is_none(),
        "the scenario itself must match: {:?} {:?}",
        clean.packets,
        clean.state
    );
    let extras = [
        (
            "MC|ItemName",
            C2SPlay::PluginMessage(c2s::PluginMessage {
                channel: "MC|ItemName".to_owned(),
                data: vec![5, b'S', b'w', b'o', b'r', b'd'],
            }),
        ),
        (
            "ClientSettings",
            C2SPlay::ClientSettings(c2s::ClientSettings {
                locale: "en_US".to_owned(),
                view_distance: 8,
                chat_visibility: c2s::ChatVisibility::Full,
                chat_colors: true,
                displayed_skin_parts: 0x7F,
            }),
        ),
    ];
    for (name, packet) in extras {
        let outcome = replay_with(&fixture, scenario, &|_| true, &|index, client| {
            if index == FRAME {
                if let Some(session) = client.session.as_mut() {
                    session.send(packet.clone());
                }
            }
        })
        .expect("the jar was found above");
        let packets = outcome.packets.unwrap_or_default();
        assert!(
            packets.contains(&format!("frame {FRAME}:")) && packets.contains(name),
            "an extra {name} at frame {FRAME} must fail the comparison; got {packets:?}"
        );
    }
}

/// Hunting aid, off by default: with `H1_FIXTURE=<path>`, replay every scenario of that fixture
/// (one written with `generate-movement-diff.ps1 -Out <path>`, for example with
/// `H1_EXTRA_RANDOM=<n>` set so the generator appends `n` more random-play seeds). Without the
/// variable this test does nothing; the committed fixture is checked by the tests below.
#[test]
fn replay_every_scenario_of_an_override_fixture() {
    let Ok(path) = std::env::var("H1_FIXTURE") else {
        return;
    };
    let Some(fixture) = load_fixture_from(&path) else {
        return;
    };
    let names: Vec<&str> = fixture
        .scenarios
        .iter()
        .map(|scenario| scenario.name.as_str())
        .collect();
    check_in(&fixture, &names);
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
    join_walk_sprint_jump_sneak => "join_walk_sprint_jump_sneak",
    spawn_before_chunks => "spawn_before_chunks",
    long_terrain_download_keepalives => "long_terrain_download_keepalives",
    keys_held_through_terrain_screen => "keys_held_through_terrain_screen",
    sprint_jump_strafe_turning => "sprint_jump_strafe_turning",
    sprint_into_wall_and_slide => "sprint_into_wall_and_slide",
    keepsprint_wtap_hits => "keepsprint_wtap_hits",
    block_hit_cycle => "block_hit_cycle",
    eating_while_moving => "eating_while_moving",
    bow_draw_while_moving => "bow_draw_while_moving",
    setbacks_absolute_relative_midjump => "setbacks_absolute_relative_midjump",
    knockback_s12_s27_s08_orders => "knockback_s12_s27_s08_orders",
    knockback_with_hits => "knockback_with_hits",
    s1c_sprint_flag_echoes => "s1c_sprint_flag_echoes",
    s20_attribute_echoes_speed_potion => "s20_attribute_echoes_speed_potion",
    respawn_same_dimension_while_sprinting => "respawn_same_dimension_while_sprinting",
    dimension_change_while_sprinting => "dimension_change_while_sprinting",
    proxy_switch_mid_connection_s01 => "proxy_switch_mid_connection_s01",
    held_slots_keys_wheel_clicks => "held_slots_keys_wheel_clicks",
    riding_minecart_enter_steer_exit => "riding_minecart_enter_steer_exit",
    flying_toggle_and_land => "flying_toggle_and_land",
    spectator_noclip_flight => "spectator_noclip_flight",
    death_and_respawn => "death_and_respawn",
    transactions_around_teleports => "transactions_around_teleports",
    mob_hits_have_no_keepsprint_slowdown => "mob_hits_have_no_keepsprint_slowdown",
    slot_switch_while_blocking => "slot_switch_while_blocking",
    food_and_blindness_sprint_gates => "food_and_blindness_sprint_gates",
    camera_spectates_another_entity => "camera_spectates_another_entity",
    water_swim_and_exit => "water_swim_and_exit",
    ladder_climb_sneak_hold_jump_off => "ladder_climb_sneak_hold_jump_off",
    creative_flight => "creative_flight",
    teleport_into_wall_push_out => "teleport_into_wall_push_out",
    inventory_and_chat_opened_while_sprinting => "inventory_and_chat_opened_while_sprinting",
    bungee_switch_s07_pair_while_sprinting => "bungee_switch_s07_pair_while_sprinting",
    sprint_jump_under_low_ceiling => "sprint_jump_under_low_ceiling",
    moving_target_interpolated_hits => "moving_target_interpolated_hits",
    walk_into_player_push => "walk_into_player_push",
    backwards_sneak_bridging => "backwards_sneak_bridging",
    // The EDGE world (ice, slime, soul sand, webs, water flow, lava,
    // slab and stair steps, fence/wall/pane rows, ladder and vines, a sneak platform), potion
    // amplifier edge cases and sprint-jump timing around landings.
    edge_ice_slime_soul_sand_sprint_jumps => "edge_ice_slime_soul_sand_sprint_jumps",
    edge_slime_fall_bounce_and_sneak_landing => "edge_slime_fall_bounce_and_sneak_landing",
    edge_soul_sand_and_cobwebs => "edge_soul_sand_and_cobwebs",
    edge_water_flow_and_lava_swim => "edge_water_flow_and_lava_swim",
    edge_stairs_slabs_step_up => "edge_stairs_slabs_step_up",
    edge_fence_wall_pane_rows_jump_boost => "edge_fence_wall_pane_rows_jump_boost",
    edge_ladder_vine_climb_and_knockback => "edge_ladder_vine_climb_and_knockback",
    edge_sneak_platform_edges_and_knockback => "edge_sneak_platform_edges_and_knockback",
    edge_potion_jump_boost_amplifiers_and_expiry => "edge_potion_jump_boost_amplifiers_and_expiry",
    sprint_jump_timing_around_landings => "sprint_jump_timing_around_landings",
    world_border_walls_and_latch => "world_border_walls_and_latch",
    // Seeded random play over the EDGE world with random knockback, explosions that clear the
    // floor, Jump Boost, speed snapshots, food, teleports and block changes.
    edge_random_play_0 => "edge_random_play_0",
    edge_random_play_1 => "edge_random_play_1",
    edge_random_play_2 => "edge_random_play_2",
    edge_random_play_3 => "edge_random_play_3",
    // Slow frames: several ticks after one packet drain and one device batch.
    low_fps_multi_tick_frames => "low_fps_multi_tick_frames",
    // Random play with random frame pacing: render-only frames and slow frames mixed in.
    edge_random_paced_0 => "edge_random_paced_0",
    edge_random_paced_1 => "edge_random_paced_1",
    // Random fighting beside a strafing remote player: hits, misses, block-hits, knockback.
    combat_random_0 => "combat_random_0",
    combat_random_paced_0 => "combat_random_paced_0",
}
