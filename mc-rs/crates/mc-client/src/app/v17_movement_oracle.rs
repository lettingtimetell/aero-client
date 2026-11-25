//! The 1.7.10 movement differential oracle, client level.
//!
//! `tests/fixtures/v17_oracle/java17/` holds recordings of the real 1.7.10 client
//! (`Minecraft.runTick` run headless over the real 1.7.10 network stack, `javaref/V17OracleRef`):
//! per frame the inbound packets (protocol-5 wire bytes in `inbound.json`), the device events,
//! every serverbound packet's bytes and the local player's state after the frame.
//!
//! This replays them through the real Rust client in 1.7.10 mode, the way the shell drives it:
//! the render-time look on the predictor, device edges through
//! `KeyBindings::set_state_when_input_allowed` gated by `ShellApp::game_input_allowed`, the
//! inbound packets decoded with the protocol-5 decoder into the network channel (the keep-alive
//! and disconnect are the network task's), `ShellApp::before_ticks`, then `ShellApp::tick` for
//! each of the frame's ticks, whose 1.7.10 drain applies the packets inside the tick. A wheel
//! notch is a scroll event of the client's input before the ticks, which the frame's first tick
//! reads, as java17's tick reads its LWJGL wheel event.
//!
//! Compared after every frame, exactly: the local player's movement state (`x`, the box bottom
//! `y`, `z`, the eye `posY`, motion, rotation, the ground, sprint, sneak, flight and collision
//! flags, `fallDistance`, `sprintingTicksLeft`, `moveForward`/`moveStrafing`, `jumpTicks`), and
//! the movement packets in order and byte for byte: `sendMotionUpdates`' C0B sneak and sprint
//! edges and C03..C06, and the C06 that answers an S08. The other packets are compared by the
//! interaction oracle and by the outbound check below.
//!
//! # The outbound check
//!
//! [`replay_outbound`] compares the **whole** serverbound stream instead: every packet the game
//! thread sends, in order and byte for byte as the 1.7.10 network task writes it (a packet built
//! in its protocol-5 form as it is, a 1.8.9 one through `C2SPlay5::from_v47`), tick by tick.
//! Every one of them must leave inside a tick, as all of vanilla 1.7.10's game-thread sends do
//! (`runTick`); only the network thread's keep-alive echo (the recording's `net` phase) is left
//! out, since the network task, not the game, answers it. Two recorded differences are not
//! compared byte for byte: the `MC|Brand` text (Aero against vanilla; the channel is compared),
//! and the order inside a drag's run of per-slot C0E packets, which vanilla takes from a
//! `HashSet` (compared as a set, with the run's transaction numbers in rising order). Screen
//! events (`click slot`, `click outside`, `drag`, Escape and the keys a screen reads) go through
//! the shell frame's screen pass before the frame's ticks, as `ShellApp::frame` runs it (which in
//! 1.7.10 mode queues them), and `ShellApp::tick`'s screen phase handles them. A packet
//! the 1.7.10 conversion refuses fails the replay: every 1.7.10 send site builds its own packet.
//!
//! Skips (prints `skip:`) without the vanilla jar (`MC_JARS_DIR`) or the recordings.
//! `V17_VERBOSE=1` prints every frame's packets.

use std::fmt::Write as _;

use super::*;

use crate::net::channel::NetHandle;
use crate::net::{ToGame, ToNet};
use mc_protocol::v5::{C2SPlay5, S2CPlay5, V5Ctx};
use mc_protocol::PacketBuffer;
use mc_render::input::keys;
use mc_render::{ShellApp, TickCtx};
use mc_version::GameVersion;
use serde_json::{Map, Value};
use tokio::sync::mpsc::UnboundedReceiver;
use winit::event::MouseButton;

const RECORDINGS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/v17_oracle/java17"
);
/// The recording harness's local profile (`V17OracleRef`, the same as the 1.8.9 oracle's).
const LOCAL_UUID: u128 = 0x4831_f00d_0000_3000_8000_0000_0000_0001;

/// The recordings' binding names (`V17Scenario.BINDINGS`) as the client's key bindings.
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
        other => {
            let slot = other.strip_prefix("hotbar")?.parse::<u8>().ok()?;
            keys::hotbar_static(slot)
        }
    })
}

fn read_json(path: &std::path::Path) -> Option<Value> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut out, byte| {
        let _ = write!(out, "{byte:02x}");
        out
    })
}

/// A vanilla-configured client: the real assets, and the mod that remaps keys off.
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

fn decode_inbound(wire: &[u8]) -> S2CPlay5 {
    let mut buf = PacketBuffer::from_slice(wire);
    let id = buf.read_varint().expect("inbound packet id");
    let packet = S2CPlay5::read(id, &mut buf)
        .unwrap_or_else(|error| panic!("cannot decode inbound 0x{id:02X}: {error:?}"));
    assert_eq!(
        buf.remaining(),
        0,
        "inbound 0x{id:02X} left bytes undecoded"
    );
    packet
}

/// One serverbound packet as the 1.7.10 network task writes it.
#[derive(Debug, Clone)]
struct Sent {
    /// The packet in its protocol-5 form.
    packet: C2SPlay5,
    /// Built in that form by the client (`ToNet::SendV5`), not converted from 1.8.9's.
    built: bool,
}

impl Sent {
    /// The frame the recordings hold: VarInt id, then the body.
    fn wire(&self) -> Vec<u8> {
        let mut buf = PacketBuffer::new();
        buf.write_varint(self.packet.id());
        self.packet.write(&mut buf);
        buf.finish_write().expect("a sent packet encodes");
        buf.as_slice().to_vec()
    }

    /// `sendMotionUpdates`' packets: C03..C06 and the C0B sneak and sprint edges.
    fn is_movement(&self) -> bool {
        match &self.packet {
            C2SPlay5::Player(_)
            | C2SPlay5::PlayerPosition(_)
            | C2SPlay5::PlayerLook(_)
            | C2SPlay5::PlayerPositionAndLook(_) => true,
            C2SPlay5::EntityAction(action) => matches!(action.action_id, 1 | 2 | 4 | 5),
            _ => false,
        }
    }
}

/// Everything queued for the network task since the last drain, in order, as the 1.7.10 task
/// sends it: a `ToNet::SendV5` as built, a `ToNet::Send` through `C2SPlay5::from_v47` with the
/// player's entity id (`net/task_v5.rs`). The task never sends a packet the conversion refuses;
/// here one is an error, since a 1.7.10 send site that reaches the conversion with a packet it
/// refuses has not built its own.
fn drain_sent(to_net: &mut UnboundedReceiver<ToNet>, entity_id: i32) -> Result<Vec<Sent>, String> {
    let mut sent = Vec::new();
    while let Ok(message) = to_net.try_recv() {
        match message {
            ToNet::SendV5(packet) => sent.push(Sent {
                packet,
                built: true,
            }),
            ToNet::Send(packet) => match C2SPlay5::from_v47(&packet, &V5Ctx { entity_id }) {
                Ok(packet) => sent.push(Sent {
                    packet,
                    built: false,
                }),
                Err(error) => {
                    return Err(format!(
                        "the 1.7.10 conversion refused {packet:?} ({error:?})"
                    ))
                }
            },
            ToNet::Disconnect => {}
        }
    }
    Ok(sent)
}

/// The movement packets among `sent`, as wire hex. Only built ones: a converted 1.8.9 packet
/// never carries the 1.7.10 movement.
fn movement_hex<'a>(sent: impl IntoIterator<Item = &'a Sent>) -> Vec<String> {
    sent.into_iter()
        .filter(|s| s.built && s.is_movement())
        .map(|s| hex(&s.wire()))
        .collect()
}

/// The recording's movement packets of one frame, as wire hex.
fn recorded_movement(frame: &Value) -> Vec<String> {
    frame["sent"]
        .as_array()
        .map(|sent| {
            sent.iter()
                .filter(|s| match s["p"]["kind"].as_str() {
                    Some(
                        "C03PacketPlayer"
                        | "C04PacketPlayerPosition"
                        | "C05PacketPlayerLook"
                        | "C06PacketPlayerPosLook",
                    ) => true,
                    Some("C0BPacketEntityAction") => matches!(
                        s["p"]["action"].as_str(),
                        Some(
                            "START_SNEAKING"
                                | "STOP_SNEAKING"
                                | "START_SPRINTING"
                                | "STOP_SPRINTING"
                        )
                    ),
                    _ => false,
                })
                .map(|s| s["hex"].as_str().expect("hex").to_owned())
                .collect()
        })
        .unwrap_or_default()
}

/// The window the screen events are aimed in.
const WINDOW: (u32, u32) = (854, 480);

/// One recorded screen event (`V17Scenario`'s `click` and `drag` steps).
#[derive(Debug, Clone, PartialEq)]
enum ScreenEvent {
    /// A press and release over a slot, by its container slot number, or outside the window.
    Click {
        slot: Option<u16>,
        button: MouseButton,
    },
    /// A press over the first slot, the pointer over the others, the release over the last.
    Drag {
        slots: Vec<u16>,
        button: MouseButton,
    },
}

impl ScreenEvent {
    fn parse(event: &str) -> Option<Self> {
        let (what, button) = event.rsplit_once(" b")?;
        let button = match button {
            "0" => MouseButton::Left,
            "1" => MouseButton::Right,
            "2" => MouseButton::Middle,
            _ => return None,
        };
        if what == "click outside" {
            return Some(Self::Click { slot: None, button });
        }
        if let Some(slot) = what.strip_prefix("click slot ") {
            let slot = slot.parse().ok()?;
            return Some(Self::Click {
                slot: Some(slot),
                button,
            });
        }
        let slots = what.strip_prefix("drag [")?.strip_suffix(']')?;
        let slots = slots
            .split(',')
            .map(|slot| slot.trim().parse().ok())
            .collect::<Option<Vec<u16>>>()?;
        Some(Self::Drag { slots, button })
    }
}

/// The shell frame's screen pass, as `ShellApp::frame` runs it with no diagnostic run (a test in
/// `app/send_sites_v5.rs` holds the frame to these lines): in 1.7.10 mode it queues the events
/// for the tick's screen phase, unless the replay's `Checks::in_tick` is off (a negative control).
fn screen_pass(client: &mut Client, input: &mut InputState) {
    if !client.queue_screen_input_v5(input, WINDOW.0, WINDOW.1) {
        client.handle_screen_input(input, WINDOW.0, WINDOW.1);
    }
}

/// The window pixel whose GUI cursor is `(x, y)`, for `Client::handle_screen_input`.
fn window_point(client: &Client, (x, y): (i32, i32)) -> Option<(f64, f64)> {
    let resolution = client.screen_resolution(WINDOW.0, WINDOW.1);
    let px = (0..WINDOW.0).find(|&px| resolution.cursor_to_gui(f64::from(px), 0.0).0 == x)?;
    let py =
        (0..WINDOW.1).find(|&py| resolution.cursor_to_gui(f64::from(px), f64::from(py)).1 == y)?;
    Some((f64::from(px), f64::from(py)))
}

/// Where a recorded click lands: a slot's centre, `nudge` GUI pixels to the right (still inside
/// the slot), or a point off the window panel.
fn screen_target(client: &Client, slot: Option<u16>, nudge: i32) -> Result<(f64, f64), String> {
    let Ui::Container(screen) = &client.ui else {
        return Err("a click with no container screen open".to_owned());
    };
    let gui = match slot {
        None => (2 + nudge, 2),
        Some(id) => {
            let slot = screen
                .container
                .slots()
                .iter()
                .find(|s| s.id == id)
                .ok_or_else(|| format!("no slot {id} in this window"))?;
            (
                screen.gui_left() + slot.x + 8 + nudge,
                screen.gui_top() + slot.y + 8,
            )
        }
    };
    window_point(client, gui).ok_or_else(|| format!("no window pixel for {gui:?}"))
}

/// One screen event through the screen pass, as the shell delivers it: the pointer at the
/// target, the press, the pointer over each further slot, the release.
fn play_screen_event(
    client: &mut Client,
    input: &mut InputState,
    event: &ScreenEvent,
) -> Result<(), String> {
    // The newly opened window's layout (`initGui`) first.
    screen_pass(client, input);
    let (slots, button) = match event {
        ScreenEvent::Click { slot, button } => (vec![*slot], *button),
        ScreenEvent::Drag { slots, button } => (slots.iter().copied().map(Some).collect(), *button),
    };
    let (first, rest) = slots.split_first().ok_or("an empty drag")?;
    let mut at = screen_target(client, *first, 0)?;
    input.on_cursor_moved(at.0, at.1);
    input.on_mouse_button(button, true);
    screen_pass(client, input);
    for slot in rest {
        // The recording's move events reach the screen even within one slot; the shell's only
        // when the pointer moves, so a repeated slot is a move one pixel over inside it.
        let mut point = screen_target(client, *slot, 0)?;
        if point == at {
            point = screen_target(client, *slot, 1)?;
        }
        at = point;
        input.on_cursor_moved(at.0, at.1);
        screen_pass(client, input);
    }
    input.on_mouse_button(button, false);
    screen_pass(client, input);
    Ok(())
}

/// One recorded packet in the outbound check's form: wire hex, with the `MC|Brand` text cut off
/// after its channel.
fn comparable(wire: &[u8]) -> String {
    let mut buf = PacketBuffer::from_slice(wire);
    if buf.read_varint().ok() == Some(0x17)
        && buf.read_string(20).ok().as_deref() == Some("MC|Brand")
    {
        let channel_end = wire.len() - buf.remaining();
        return hex(&wire[..channel_end]);
    }
    hex(wire)
}

/// A drag's per-slot packet: C0E mode 5 with button 1, 5 or 9.
fn is_drag_slot(wire: &[u8]) -> bool {
    wire.len() > 7 && wire[0] == 0x0E && wire[7] == 5 && matches!(wire[4], 1 | 5 | 9)
}

/// A frame's packets with each run of drag slots made order-free: sorted by slot, the run's
/// transaction numbers given back in rising order (vanilla iterates a `HashSet` there).
fn canonical(mut packets: Vec<Vec<u8>>) -> Vec<String> {
    let mut i = 0;
    while i < packets.len() {
        let mut j = i;
        while j < packets.len() && is_drag_slot(&packets[j]) {
            j += 1;
        }
        if j - i > 1 {
            let run = &mut packets[i..j];
            let mut actions: Vec<[u8; 2]> = run.iter().map(|p| [p[5], p[6]]).collect();
            actions.sort_by_key(|a| i16::from_be_bytes(*a));
            run.sort_by_key(|p| i16::from_be_bytes([p[2], p[3]]));
            for (packet, action) in run.iter_mut().zip(actions) {
                packet[5..7].copy_from_slice(&action);
            }
        }
        i = j.max(i + 1);
    }
    packets.iter().map(|p| comparable(p)).collect()
}

/// The recording's game-thread packets of one frame, one list per tick: every packet but the
/// network thread's (`net`), split after each tick's movement packet: C03..C06, or a riding
/// tick's C05 and C0C (java17 `EntityClientPlayerMP.onUpdate`).
fn recorded_ticks(frame: &Value, sites: &[Value]) -> Vec<Vec<Vec<u8>>> {
    let game: Vec<Vec<u8>> = frame["sent"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|sent| {
            sites[sent["at"].as_u64().expect("site") as usize][0]
                .as_str()
                .expect("phase")
                != "net"
        })
        .map(|sent| unhex(sent["hex"].as_str().expect("hex")))
        .collect();
    let mut ticks = vec![Vec::new()];
    for (index, wire) in game.iter().enumerate() {
        let steer_next = game.get(index + 1).is_some_and(|next| next[0] == 0x0C);
        let ends_tick = match wire[0] {
            0x03..=0x06 => !steer_next,
            0x0C => true,
            _ => false,
        };
        ticks.last_mut().expect("a tick").push(wire.clone());
        if ends_tick {
            ticks.push(Vec::new());
        }
    }
    if ticks.last().is_some_and(Vec::is_empty) {
        ticks.pop();
    }
    ticks
}

/// The outbound check of one frame: nothing outside a tick, then the frame's packets in order
/// and, when the frame ran several ticks, tick by tick.
fn outbound_difference(
    frame: &Value,
    sites: &[Value],
    outside: &[Sent],
    ticks: &[Vec<Sent>],
) -> Option<String> {
    if !outside.is_empty() {
        let kinds: Vec<String> = outside.iter().map(|s| hex(&s.wire())).collect();
        return Some(format!("sent outside a tick: {kinds:?}"));
    }
    let rust: Vec<Vec<Vec<u8>>> = ticks
        .iter()
        .map(|tick| tick.iter().map(Sent::wire).collect())
        .collect();
    let java = recorded_ticks(frame, sites);
    let flat = |ticks: &[Vec<Vec<u8>>]| canonical(ticks.concat());
    if flat(&java) != flat(&rust) {
        return Some(format!(
            "packets\n    java {:?}\n    rust {:?}",
            flat(&java),
            flat(&rust)
        ));
    }
    // The same packets in the same order: the tick each one left in must match too. A recorded
    // tick that ends without a movement packet cannot be told apart, so only a full split is
    // compared.
    let rust_split: Vec<Vec<Vec<u8>>> = rust.into_iter().filter(|t| !t.is_empty()).collect();
    if java.len() == rust_split.len() {
        for (index, (java, rust)) in java.iter().zip(&rust_split).enumerate() {
            if canonical(java.clone()) != canonical(rust.clone()) {
                return Some(format!("tick {index} of the frame holds other packets"));
            }
        }
    }
    None
}

/// Compares the recorded movement fields with the client's player, bit for bit.
fn state_difference(expected: &Map<String, Value>, client: &Client) -> Option<String> {
    let Some(predictor) = client.session.as_ref().and_then(Session::predictor) else {
        return Some("rust has no player".to_owned());
    };
    let Some(v5) = predictor.v5() else {
        return Some("the player is not a 1.7.10 one".to_owned());
    };
    let s = &predictor.state;
    let mut out = Vec::new();
    for (key, actual) in [
        ("x", s.pos.x),
        ("y", s.aabb().min.y),
        ("z", s.pos.z),
        ("posY", v5.pos_y()),
        ("vx", s.vel.x),
        ("vy", s.vel.y),
        ("vz", s.vel.z),
    ] {
        let want = expected[key].as_f64().expect(key);
        if want.to_bits() != actual.to_bits() {
            out.push(format!("{key}: java {want:?}, rust {actual:?}"));
        }
    }
    for (key, actual) in [
        ("yaw", s.yaw),
        ("pitch", s.pitch),
        ("fallDistance", s.fall_distance),
        ("moveForward", s.move_forward),
        ("moveStrafing", s.move_strafing),
    ] {
        let want = expected[key].as_f64().expect(key) as f32;
        if want.to_bits() != actual.to_bits() {
            out.push(format!("{key}: java {want:?}, rust {actual:?}"));
        }
    }
    for (key, actual) in [
        ("onGround", s.on_ground),
        ("sprinting", s.sprinting),
        ("sneaking", s.sneaking),
        ("flying", s.flying),
        ("collidedH", s.collided_horizontally),
    ] {
        let want = expected[key].as_bool().expect(key);
        if want != actual {
            out.push(format!("{key}: java {want}, rust {actual}"));
        }
    }
    for (key, actual) in [
        ("sprintingTicksLeft", s.sprinting_ticks_left),
        ("jumpTicks", s.jump_ticks),
    ] {
        let want = expected[key].as_i64().expect(key);
        if want != i64::from(actual) {
            out.push(format!("{key}: java {want}, rust {actual}"));
        }
    }
    (!out.is_empty()).then(|| out.join("; "))
}

/// What one recording's replay found.
#[derive(Debug)]
enum Outcome {
    Matched { frames: usize },
    Unsupported(String),
    Diverged(String),
}

/// What a replay compares.
#[derive(Debug, Clone, Copy)]
struct Checks {
    /// The movement packets and the player's movement state.
    movement: bool,
    /// The whole outbound stream ([`outbound_difference`]).
    outbound: bool,
    /// The 1.7.10 screen events handled in the tick's screen phase (off only in a negative
    /// control, which handles them in the frame as a 1.8.9 session does).
    in_tick: bool,
}

const MOVEMENT: Checks = Checks {
    movement: true,
    outbound: false,
    in_tick: true,
};

const OUTBOUND: Checks = Checks {
    movement: false,
    outbound: true,
    in_tick: true,
};

fn replay(name: &str) -> Option<Outcome> {
    replay_with(name, &|_| {})
}

/// [`replay`] with `hook` editing the client before each frame's ticks (negative controls).
fn replay_with(name: &str, hook: &dyn Fn(&mut Client)) -> Option<Outcome> {
    replay_checked(name, hook, MOVEMENT)
}

/// The outbound check over one recording.
fn replay_outbound(name: &str) -> Option<Outcome> {
    replay_checked(name, &|_| {}, OUTBOUND)
}

/// The replay itself, making the comparisons `checks` names.
fn replay_checked(name: &str, hook: &dyn Fn(&mut Client), checks: Checks) -> Option<Outcome> {
    let dir = std::path::Path::new(RECORDINGS);
    let (Some(blobs), Some(recording)) = (
        read_json(&dir.join("inbound.json")),
        read_json(&dir.join(format!("{name}.json"))),
    ) else {
        eprintln!("skip: no 1.7.10 recording {name}");
        return None;
    };
    let blobs: Vec<Vec<u8>> = blobs["blobs"]
        .as_array()
        .expect("blobs")
        .iter()
        .map(|b| unhex(b.as_str().expect("blob")))
        .collect();
    let frames = recording["frames"].as_array().expect("frames");
    let verbose = std::env::var_os("V17_VERBOSE").is_some();

    let (mut client, _dir) = vanilla_client()?;
    client.screen_input_v5.at_frame_time = !checks.in_tick;
    let (handle, to_game, mut to_net) = NetHandle::new_pair();
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
    let mut camera = menu_camera();
    let mut input = InputState::new();
    let mut settings = GameSettings::default();
    let mut ticks = 0u64;
    let mut expected: Option<Map<String, Value>> = None;

    for (index, frame) in frames.iter().enumerate() {
        let at = |what: String| {
            let note = frame["note"].as_str().unwrap_or("");
            format!("{name} frame {index} ({note}): {what}")
        };
        match frame.get("player") {
            Some(Value::Object(delta)) => {
                let state = expected.get_or_insert_with(Map::new);
                for (k, v) in delta {
                    state.insert(k.clone(), v.clone());
                }
            }
            Some(Value::Null) => expected = None,
            _ => {}
        }

        // (1) the previous frame's render-time look.
        if let Some(look) = frame["look"].as_array() {
            if let Some(predictor) = client.session.as_mut().and_then(Session::predictor_mut) {
                predictor.state.yaw = look[0].as_f64().expect("yaw") as f32;
                predictor.state.pitch = look[1].as_f64().expect("pitch") as f32;
            }
        }
        // (2) device edges, as shell.rs routes them between draws: a key edge reaches the
        // bindings while no screen is up, and the open screen's key queue otherwise.
        let mut wheel = 0i32;
        let mut screen_keys = false;
        let mut screen_events = Vec::new();
        for event in frame["events"].as_array().into_iter().flatten() {
            let event = event.as_str().expect("event");
            if let Some(notches) = event.strip_prefix("wheel ") {
                wheel += notches.parse::<i32>().expect("wheel notches");
                continue;
            }
            if let Some(screen_event) = ScreenEvent::parse(event) {
                screen_events.push(screen_event);
                continue;
            }
            let (down, name) = match event.split_at(1) {
                ("+", name) => (true, name),
                ("-", name) => (false, name),
                _ => return Some(Outcome::Unsupported(at(format!("event `{event}`")))),
            };
            let allowed = ShellApp::game_input_allowed(&client);
            let code = if name == "esc" {
                Some(winit::keyboard::KeyCode::Escape)
            } else {
                let Some(name) = binding(name) else {
                    return Some(Outcome::Unsupported(at(format!("key `{event}`"))));
                };
                let key = settings.keys.get(name).expect("vanilla binding").binding();
                settings
                    .keys
                    .set_state_when_input_allowed(key, down, allowed);
                // A keyboard key also reaches the client's raw key queue (an open screen's
                // input, and the held modifiers `isCtrlKeyDown` reads); a mouse binding has no
                // key code and needs only its binding edge here.
                mc_render::input::keycode_of_lwjgl(key.to_lwjgl())
            };
            if let Some(code) = code {
                client.handle_raw_key(code, down, false, None);
                screen_keys = true;
            }
        }
        // The screens' events, through the shell frame's screen pass. The pass runs after a
        // frame's ticks, so events recorded before this frame's ticks met the previous one, which
        // in 1.7.10 mode queues them for this frame's first tick (java17 handles them there).
        if screen_keys || !screen_events.is_empty() {
            // `GuiContainer`'s double-click window reads the clock: 50 ms per tick, as recorded.
            if let Some(start) =
                std::time::Instant::now().checked_sub(std::time::Duration::from_millis(50 * ticks))
            {
                client.started = start;
            }
            if screen_keys {
                screen_pass(&mut client, &mut input);
            }
            for event in &screen_events {
                if let Err(why) = play_screen_event(&mut client, &mut input, event) {
                    return Some(Outcome::Unsupported(at(why)));
                }
            }
        }
        // A wheel notch is one LWJGL event (+-120) that the tick's mouse loop reads (java17
        // `Minecraft.java:1646-1700`): the live client's scroll input, which the frame's first
        // tick takes (`pump_tick_wheel`, or the screen phase in 1.7.10 mode), or an open
        // screen's pass.
        for _ in 0..wheel.unsigned_abs() {
            let notch = if wheel > 0 { 1.0 } else { -1.0 };
            input.on_scroll(winit::event::MouseScrollDelta::LineDelta(0.0, notch));
        }
        if wheel != 0 && !ShellApp::game_input_allowed(&client) {
            screen_pass(&mut client, &mut input);
        }
        // (3) the inbound packets reach the channel; (4) the frame's ticks.
        for packet in frame["inbound"].as_array().into_iter().flatten() {
            let wire = &blobs[packet["blob"].as_u64().expect("blob") as usize];
            let packet = decode_inbound(wire);
            // The network task answers these on arrival (java17 `NetworkManager.channelRead0`).
            if packet.has_priority() {
                continue;
            }
            to_game
                .try_send(ToGame::PacketV5(Box::new(packet)))
                .expect("inbound channel");
        }
        let entity_id = |client: &Client| {
            client
                .session
                .as_ref()
                .map_or(0, |session| session.state().local.entity_id)
        };
        ShellApp::before_ticks(&mut client, &mut camera);
        let outside = match drain_sent(&mut to_net, entity_id(&client)) {
            Ok(sent) => sent,
            Err(why) => return Some(Outcome::Diverged(at(why))),
        };
        hook(&mut client);
        let mut per_tick = Vec::new();
        for _ in 0..frame["ticks"].as_u64().expect("ticks") {
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
            match drain_sent(&mut to_net, entity_id(&client)) {
                Ok(sent) => per_tick.push(sent),
                Err(why) => return Some(Outcome::Diverged(at(why))),
            }
        }

        if checks.movement {
            let want = recorded_movement(frame);
            let sent = movement_hex(outside.iter().chain(per_tick.iter().flatten()));
            if verbose {
                eprintln!("[{name} {index}] java {want:?}\n    rust {sent:?}");
            }
            if want != sent {
                return Some(Outcome::Diverged(at(format!(
                    "movement packets\n    java {want:?}\n    rust {sent:?}"
                ))));
            }
            if let Some(expected) = &expected {
                if let Some(what) = state_difference(expected, &client) {
                    return Some(Outcome::Diverged(at(what)));
                }
            }
        }
        if checks.outbound {
            let sites = recording["sites"].as_array().expect("sites");
            if verbose {
                let rust: Vec<Vec<String>> = per_tick
                    .iter()
                    .map(|tick| tick.iter().map(|s| hex(&s.wire())).collect())
                    .collect();
                eprintln!("[{name} {index}] rust per tick {rust:?}");
            }
            if let Some(what) = outbound_difference(frame, sites, &outside, &per_tick) {
                return Some(Outcome::Diverged(at(what)));
            }
        }
    }
    Some(Outcome::Matched {
        frames: frames.len(),
    })
}

fn check(name: &str) {
    match replay(name) {
        None => {}
        Some(Outcome::Matched { frames }) => println!("{name}: {frames} frames match"),
        Some(Outcome::Unsupported(why)) => panic!("{name} cannot be replayed: {why}"),
        Some(Outcome::Diverged(why)) => panic!("{why}"),
    }
}

/// One test per recording the client replays; each must match every frame.
macro_rules! matched_recordings {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check(stringify!($name));
            }
        )*
    };
}

matched_recordings!(
    combat_block_hit,
    combat_entity_right_click,
    combat_miss_penalty,
    combat_pick_before_packets,
    combat_reach_block_behind,
    combat_sprint_hits_wtap,
    dig_adventure_mode,
    dig_fatigue_and_tools,
    dig_start_abort_finish,
    dig_while_using_place_while_digging,
    drop_item_and_stack,
    focus_adventure_glass,
    focus_dig_fatigue0,
    focus_dig_fatigue1,
    focus_dig_while_blocking,
    focus_far_cursor,
    focus_fly_speeds,
    focus_miss_cooldown,
    focus_neg_pool_climb,
    focus_pick_before_packets,
    focus_place_double_plant,
    focus_reach_entity_3blocks,
    focus_remote_teleport_offset,
    focus_rightclick_while_digging,
    focus_sneak_ysize,
    focus_sprint_sneak_release,
    focus_step_bed,
    focus_step_ceiling_snow,
    focus_step_slab,
    focus_tool_ladder,
    focus_tool_melon,
    focus_tool_packed_ice,
    focus_water_dig_eye,
    focus_yaw_head_lag,
    hotbar_keys_wheel_server,
    inventory_clicks_drags,
    inventory_middle_drag,
    inventory_middle_drag_after_right_drag,
    move_creative_flight,
    move_falls,
    move_fences_walls_panes,
    move_ice_soul_sand,
    move_knockback_setbacks,
    move_ladder_vines,
    move_sneak_release_double_tap,
    move_sneak_stance_edge,
    move_stairs_slabs_carpet_snow,
    move_step_heights,
    move_swim_bank_boost,
    move_walk_sprint_jump_strafe,
    move_water_lava_flow,
    move_web_soul_sand,
    net_join_terrain_screen,
    net_keepalive_transactions_zero_tick,
    net_respawn_and_death,
    place_blocks,
    place_far_coordinates,
    place_sneak_bridge,
    use_eat_and_bow,
);

/// The comparison is not vacuous: the same replay with 1.8.9's step height diverges where the
/// recorded 1.7.10 player stops at the bed.
#[test]
fn a_1_8_9_step_height_diverges_from_the_bed_recording() {
    let tall_step = |client: &mut Client| {
        if let Some(predictor) = client.session.as_mut().and_then(Session::predictor_mut) {
            predictor.state.step_height = 0.6;
        }
    };
    match replay_with("focus_step_bed", &tall_step) {
        None => {}
        Some(Outcome::Diverged(why)) => println!("step 0.6: {why}"),
        Some(other) => panic!("a 0.6 step must diverge: {other:?}"),
    }
}

/// Replays every recording and prints what each one does; asserts nothing beyond the list
/// above (the inbound handling still grows). `V17_ALL=1` runs it.
#[test]
fn report_every_recording() {
    if std::env::var_os("V17_ALL").is_none() {
        return;
    }
    for name in recording_names() {
        println!("{name}: {:?}", replay(&name));
    }
}

fn recording_names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(RECORDINGS)
        .into_iter()
        .flatten()
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            (stem != "inbound").then(|| stem.to_owned())
        })
        .collect();
    names.sort();
    names
}

fn check_outbound(name: &str) {
    match replay_outbound(name) {
        None => {}
        Some(Outcome::Matched { frames }) => println!("{name}: {frames} frames, outbound match"),
        Some(Outcome::Unsupported(why)) => panic!("{name} cannot be replayed: {why}"),
        Some(Outcome::Diverged(why)) => panic!("{why}"),
    }
}

/// One test per recording whose whole outbound stream the client reproduces, packet for packet
/// and tick for tick.
macro_rules! outbound_recordings {
    ($($test:ident => $name:literal),* $(,)?) => {
        $(
            #[test]
            fn $test() {
                check_outbound($name);
            }
        )*
    };
}

outbound_recordings!(
    outbound_combat_block_hit => "combat_block_hit",
    outbound_combat_entity_right_click => "combat_entity_right_click",
    outbound_combat_miss_penalty => "combat_miss_penalty",
    outbound_combat_pick_before_packets => "combat_pick_before_packets",
    outbound_combat_reach_block_behind => "combat_reach_block_behind",
    outbound_combat_sprint_hits_wtap => "combat_sprint_hits_wtap",
    outbound_dig_adventure_mode => "dig_adventure_mode",
    outbound_dig_fatigue_and_tools => "dig_fatigue_and_tools",
    outbound_dig_start_abort_finish => "dig_start_abort_finish",
    outbound_dig_while_using_place_while_digging => "dig_while_using_place_while_digging",
    outbound_drop_item_and_stack => "drop_item_and_stack",
    outbound_focus_adventure_glass => "focus_adventure_glass",
    outbound_focus_dig_fatigue0 => "focus_dig_fatigue0",
    outbound_focus_dig_fatigue1 => "focus_dig_fatigue1",
    outbound_focus_dig_while_blocking => "focus_dig_while_blocking",
    outbound_focus_far_cursor => "focus_far_cursor",
    outbound_focus_fly_speeds => "focus_fly_speeds",
    outbound_focus_miss_cooldown => "focus_miss_cooldown",
    outbound_focus_neg_pool_climb => "focus_neg_pool_climb",
    outbound_focus_pick_before_packets => "focus_pick_before_packets",
    outbound_focus_place_double_plant => "focus_place_double_plant",
    outbound_focus_reach_entity_3blocks => "focus_reach_entity_3blocks",
    outbound_focus_remote_teleport_offset => "focus_remote_teleport_offset",
    outbound_focus_rightclick_while_digging => "focus_rightclick_while_digging",
    outbound_focus_sneak_ysize => "focus_sneak_ysize",
    outbound_focus_sprint_sneak_release => "focus_sprint_sneak_release",
    outbound_focus_step_bed => "focus_step_bed",
    outbound_focus_step_ceiling_snow => "focus_step_ceiling_snow",
    outbound_focus_step_slab => "focus_step_slab",
    outbound_focus_tool_ladder => "focus_tool_ladder",
    outbound_focus_tool_melon => "focus_tool_melon",
    outbound_focus_tool_packed_ice => "focus_tool_packed_ice",
    outbound_focus_water_dig_eye => "focus_water_dig_eye",
    outbound_focus_yaw_head_lag => "focus_yaw_head_lag",
    outbound_hotbar_keys_wheel_server => "hotbar_keys_wheel_server",
    outbound_inventory_clicks_drags => "inventory_clicks_drags",
    outbound_inventory_middle_drag => "inventory_middle_drag",
    outbound_inventory_middle_drag_after_right_drag => "inventory_middle_drag_after_right_drag",
    outbound_move_creative_flight => "move_creative_flight",
    outbound_move_falls => "move_falls",
    outbound_move_fences_walls_panes => "move_fences_walls_panes",
    outbound_move_ice_soul_sand => "move_ice_soul_sand",
    outbound_move_knockback_setbacks => "move_knockback_setbacks",
    outbound_move_ladder_vines => "move_ladder_vines",
    outbound_move_sneak_release_double_tap => "move_sneak_release_double_tap",
    outbound_move_sneak_stance_edge => "move_sneak_stance_edge",
    outbound_move_stairs_slabs_carpet_snow => "move_stairs_slabs_carpet_snow",
    outbound_move_step_heights => "move_step_heights",
    outbound_move_swim_bank_boost => "move_swim_bank_boost",
    outbound_move_walk_sprint_jump_strafe => "move_walk_sprint_jump_strafe",
    outbound_move_water_lava_flow => "move_water_lava_flow",
    outbound_move_web_soul_sand => "move_web_soul_sand",
    outbound_net_join_terrain_screen => "net_join_terrain_screen",
    outbound_net_keepalive_transactions_zero_tick => "net_keepalive_transactions_zero_tick",
    outbound_net_respawn_and_death => "net_respawn_and_death",
    outbound_place_blocks => "place_blocks",
    outbound_place_far_coordinates => "place_far_coordinates",
    outbound_place_sneak_bridge => "place_sneak_bridge",
    outbound_use_eat_and_bow => "use_eat_and_bow",
);

/// The tick's screen phase is what puts a click inside its tick: handled in the frame instead,
/// as a 1.8.9 session handles it, the inventory recording's first click leaves before the tick,
/// outside any tick (in the tick, the whole recording matches:
/// `outbound_inventory_clicks_drags`).
#[test]
fn screen_events_handled_in_the_frame_leave_outside_a_tick() {
    let in_frame = Checks {
        in_tick: false,
        ..OUTBOUND
    };
    match replay_checked("inventory_clicks_drags", &|_| {}, in_frame) {
        None => {}
        Some(Outcome::Diverged(why)) => {
            assert!(why.contains("sent outside a tick"), "{why}");
            println!("handled in the frame: {why}");
        }
        Some(other) => panic!("a click handled in the frame must leave outside a tick: {other:?}"),
    }
}

/// The outbound check is not vacuous: one packet more than the recording, sent from inside the
/// tick, is found in the frame it was sent in.
#[test]
fn an_extra_packet_diverges_from_the_recording() {
    let extra_swing = |client: &mut Client| {
        if let Some(session) = client.session.as_mut() {
            session.send(mc_protocol::play::C2SPlay::Animation(
                mc_protocol::play::c2s::Animation,
            ));
        }
    };
    match replay_checked("net_join_terrain_screen", &extra_swing, OUTBOUND) {
        None => {}
        Some(Outcome::Diverged(why)) => println!("extra C0A: {why}"),
        Some(other) => panic!("an extra packet must diverge: {other:?}"),
    }
}

/// Replays every recording with the outbound check and prints the first divergence of each;
/// asserts nothing beyond the list above. `V17_ALL=1` runs it.
#[test]
fn report_every_outbound_stream() {
    if std::env::var_os("V17_ALL").is_none() {
        return;
    }
    for name in recording_names() {
        println!("outbound {name}: {:?}", replay_outbound(&name));
    }
}
