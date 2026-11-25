//! The 1.7.10 movement oracle, physics half.
//!
//! `mc-client/tests/fixtures/v17_oracle/` holds recordings of the real 1.7.10 client
//! (`Minecraft.runTick` run headless over the real 1.7.10 network stack): the input scripts in
//! `scenarios/`, and in `java17/` the player state and outbound packets after every frame plus
//! `inbound.json`, the protocol-5 packets the frames delivered. This test replays the movement
//! scenarios through a 1.7.10 [`Predictor`] and compares, after every frame and bit for bit:
//!
//! * the player: `x`, the box bottom `y`, `z`, the eye `posY`, motion, rotation, `onGround`,
//!   the sprint, sneak and flight flags, `isCollidedHorizontally`, `fallDistance`,
//!   `sprintingTicksLeft`, `moveForward`/`moveStrafing` and `jumpTicks`;
//! * the movement packets: `sendMotionUpdates`' C0B edges and C03..C06, and the C06 that answers
//!   an S08, by their decoded fields.
//!
//! Around the player tick the replay does what java17's `Minecraft.runTick` does with a frame
//! (`client/Minecraft.java:1536-2029`): the frame's inbound packets are applied in the first
//! tick inside `PlayerControllerMP.updateController` (a frame without a tick applies nothing),
//! then the key events reach the bindings, then `WorldClient.updateEntities` ticks the player
//! and `EntityClientPlayerMP.sendMotionUpdates` reports it. What the client decides outside the
//! player's own update comes from the recording instead: whether an item is in use, and which
//! entity each attack hit (whose client-side effect on the attacker, the sprint reset, is
//! applied here). Screens, a respawn and the blocks the client places or breaks itself are not
//! modelled; a recording that needs them is reported as unsupported with the reason.

use std::path::{Path, PathBuf};

use mc_math::{BlockPos, ChunkPos, DVec3};
use mc_physics::scene;
use mc_physics::v5::{CorrectionReplyV5, OutboundV5};
use mc_physics::{MoveInput, PhysicsState, Predictor};
use mc_world::World;
use serde_json::{Map, Value};

/// java17: `EntityLivingBase.sprintingSpeedBoostModifierUUID`.
const SPRINT_UUID: u128 = 0x662A_6B8D_DA3E_4C1C_8813_96EA_6097_278D;

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../mc-client/tests/fixtures/v17_oracle")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// ------------------------------------------------------------------------------- the wire

/// A big-endian reader over one protocol-5 packet, `[VarInt id][body]`.
struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, at: 0 }
    }

    fn take<const N: usize>(&mut self) -> [u8; N] {
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.at..self.at + N]);
        self.at += N;
        out
    }

    fn u8(&mut self) -> u8 {
        self.take::<1>()[0]
    }

    fn i8(&mut self) -> i8 {
        i8::from_be_bytes(self.take())
    }

    fn i16(&mut self) -> i16 {
        i16::from_be_bytes(self.take())
    }

    fn i32(&mut self) -> i32 {
        i32::from_be_bytes(self.take())
    }

    fn i64(&mut self) -> i64 {
        i64::from_be_bytes(self.take())
    }

    fn f32(&mut self) -> f32 {
        f32::from_be_bytes(self.take())
    }

    fn f64(&mut self) -> f64 {
        f64::from_be_bytes(self.take())
    }

    fn bool(&mut self) -> bool {
        self.u8() != 0
    }

    fn varint(&mut self) -> i32 {
        let mut value = 0i32;
        for shift in (0..35).step_by(7) {
            let byte = self.u8();
            value |= i32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
        }
        value
    }

    fn string(&mut self) -> String {
        let len = usize::try_from(self.varint()).expect("string length");
        let text = String::from_utf8(self.bytes[self.at..self.at + len].to_vec()).expect("utf-8");
        self.at += len;
        text
    }
}

fn hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

// ------------------------------------------------------------------------- the recording

/// One movement packet, by the fields the recording decodes.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Sent {
    Ground(bool),
    Position(f64, f64, f64, f64, bool),
    Look(f32, f32, bool),
    PositionLook(f64, f64, f64, f64, f32, f32, bool),
    Action(&'static str),
}

impl Sent {
    /// Exact comparison: `Debug` prints every distinct `f64`/`f32` (and the sign of zero)
    /// differently.
    fn key(self) -> String {
        format!("{self:?}")
    }

    fn from_outbound(out: OutboundV5) -> Self {
        match out {
            OutboundV5::Ground { on_ground } => Sent::Ground(on_ground),
            OutboundV5::Position {
                x,
                y,
                stance,
                z,
                on_ground,
            } => Sent::Position(x, y, stance, z, on_ground),
            OutboundV5::Look {
                yaw,
                pitch,
                on_ground,
            } => Sent::Look(yaw, pitch, on_ground),
            OutboundV5::PositionLook {
                x,
                y,
                stance,
                z,
                yaw,
                pitch,
                on_ground,
            } => Sent::PositionLook(x, y, stance, z, yaw, pitch, on_ground),
        }
    }

    fn from_reply(r: CorrectionReplyV5) -> Self {
        Sent::PositionLook(r.x, r.y, r.stance, r.z, r.yaw, r.pitch, r.on_ground)
    }

    /// The recording's decode of a movement packet (`V17Scenario.Wire`), or `None` for other
    /// packets.
    fn from_recording(p: &Value) -> Option<Self> {
        let f = |k: &str| p[k].as_f64().unwrap_or_else(|| panic!("{k} in {p}"));
        let g = |k: &str| f(k) as f32;
        let b = |k: &str| p[k].as_bool().unwrap_or_else(|| panic!("{k} in {p}"));
        Some(match p["kind"].as_str()? {
            "C03PacketPlayer" => Sent::Ground(b("onGround")),
            "C04PacketPlayerPosition" => {
                Sent::Position(f("x"), f("y"), f("stance"), f("z"), b("onGround"))
            }
            "C05PacketPlayerLook" => Sent::Look(g("yaw"), g("pitch"), b("onGround")),
            "C06PacketPlayerPosLook" => Sent::PositionLook(
                f("x"),
                f("y"),
                f("stance"),
                f("z"),
                g("yaw"),
                g("pitch"),
                b("onGround"),
            ),
            "C0BPacketEntityAction" => Sent::Action(match p["action"].as_str()? {
                "START_SNEAKING" => "START_SNEAKING",
                "STOP_SNEAKING" => "STOP_SNEAKING",
                "START_SPRINTING" => "START_SPRINTING",
                "STOP_SPRINTING" => "STOP_SPRINTING",
                _ => return None,
            }),
            _ => return None,
        })
    }
}

/// java17 send sites whose packets this replay reproduces.
const MOVEMENT_SITE: &str = "EntityClientPlayerMP.sendMotionUpdates";
const S08_REPLY_SITE: &str = "NetHandlerPlayClient.handlePlayerPosLook";

/// The world file's boxes: `[x0, y0, z0, x1, y1, z1, id, meta]`, painted in order.
struct WorldSpec {
    boxes: Vec<[i32; 8]>,
}

impl WorldSpec {
    fn load(name: &str) -> Self {
        let json = read_json(&fixtures().join(format!("scenarios/worlds/{name}.json")));
        let boxes = json["boxes"]
            .as_array()
            .expect("boxes")
            .iter()
            .map(|b| {
                let b = b.as_array().expect("box");
                let mut out = [0i32; 8];
                for (i, v) in b.iter().enumerate() {
                    out[i] = i32::try_from(v.as_i64().expect("box int")).expect("box int");
                }
                out
            })
            .collect();
        Self { boxes }
    }
}

// --------------------------------------------------------------------------------- replay

#[derive(Debug, Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    jump: bool,
    sneak: bool,
    sprint: bool,
}

/// Why a scenario could not be replayed at the physics level.
#[derive(Debug)]
enum Outcome {
    Matched { frames: usize },
    Unsupported(String),
    Mismatch(String),
}

struct Replay {
    spec: WorldSpec,
    world: World,
    p: Predictor,
    local_id: i32,
    keys: Keys,
    was_sprinting: bool,
    was_sneaking: bool,
    /// java17: `GuiDownloadTerrain`, open from the join until the first S08; key events go to
    /// it rather than to the bindings.
    terrain_screen: bool,
    /// Entity ids of the remote players (S0C).
    players: Vec<i32>,
}

impl Replay {
    /// Applies one inbound packet the way its java17 handler changes the local player.
    fn apply(&mut self, wire: &[u8], sent: &mut Vec<Sent>) -> Result<(), String> {
        let mut r = Reader::new(wire);
        let id = r.varint();
        match id {
            // Keep-alive, chat, time, equipment, spawn position, held item, animation, collect,
            // spawn mob, painting, orb, destroy, head look, experience, block action, break
            // animation, world effect, sound, particles, global entity, window contents and
            // slots, close window, transaction, tile entity, statistics, player list, plugin
            // channel: none moves the local player.
            0x00
            | 0x02..=0x05
            | 0x09
            | 0x0B
            | 0x0D
            | 0x0F..=0x11
            | 0x13
            | 0x19
            | 0x1F
            | 0x24
            | 0x25
            | 0x28..=0x2A
            | 0x2C
            | 0x2E..=0x32
            | 0x35
            | 0x37
            | 0x38
            | 0x3F => Ok(()),
            0x0C => {
                // A remote player: the target whose client-side hit resets the sprint.
                self.players.push(r.varint());
                Ok(())
            }
            0x06 => {
                // java17: `handleUpdateHealth`.
                let health = r.f32();
                let food = r.i16();
                if health <= 0.0 {
                    return Err("the player dies (death screen and removal)".into());
                }
                self.p.state.food_level = i32::from(food);
                Ok(())
            }
            0x08 => {
                let (x, y, z) = (r.f64(), r.f64(), r.f64());
                let (yaw, pitch, on_ground) = (r.f32(), r.f32(), r.bool());
                let reply = self
                    .p
                    .apply_correction_v5(x, y, z, yaw, pitch, on_ground)
                    .expect("a 1.7.10 predictor");
                sent.push(Sent::from_reply(reply));
                // java17: the first S08 closes the terrain screen (`doneLoadingTerrain`).
                self.terrain_screen = false;
                Ok(())
            }
            0x0E => {
                let _entity = r.varint();
                match r.i8() {
                    1 | 10 => Err("a boat or minecart (an entity collision box)".into()),
                    _ => Ok(()),
                }
            }
            0x12 => {
                // java17: `handleEntityVelocity` -> `setVelocity(v / 8000.0D)`.
                if r.i32() == self.local_id {
                    self.p.state.vel = DVec3::new(
                        f64::from(r.i16()) / 8000.0,
                        f64::from(r.i16()) / 8000.0,
                        f64::from(r.i16()) / 8000.0,
                    );
                }
                Ok(())
            }
            0x1A => {
                // java17: `handleEntityStatus` -> `EntityPlayer.handleHealthUpdate`: 9 ends the
                // item use and 2 is the hurt animation (its `attackEntityFrom` returns false on
                // the client); 3 is death.
                if r.i32() == self.local_id && r.i8() == 3 {
                    Err("the player dies (entity status 3)".into())
                } else {
                    Ok(())
                }
            }
            0x14..=0x18 | 0x1B | 0x1C => {
                if r.i32() == self.local_id {
                    Err(format!(
                        "packet 0x{id:02x} moves or rewrites the local player"
                    ))
                } else {
                    Ok(())
                }
            }
            0x1D => {
                // java17: `handleEntityEffect` -> `addPotionEffect` (`combine`: a stronger
                // amplifier replaces). Speed and Slowness reach the client through S20 only.
                if r.i32() == self.local_id {
                    let s = &mut self.p.state;
                    let (effect, amplifier) = (r.i8(), r.i8());
                    match effect {
                        8 => {
                            if s.jump_boost.is_none_or(|old| amplifier > old) {
                                s.jump_boost = Some(amplifier);
                            }
                        }
                        15 => s.blindness = true,
                        _ => {}
                    }
                }
                Ok(())
            }
            0x1E => {
                if r.i32() == self.local_id {
                    let s = &mut self.p.state;
                    match r.u8() {
                        8 => s.jump_boost = None,
                        15 => s.blindness = false,
                        _ => {}
                    }
                }
                Ok(())
            }
            0x20 => {
                if r.i32() == self.local_id {
                    apply_attributes(&mut r, &mut self.p.state);
                }
                Ok(())
            }
            0x21 => {
                let (cx, cz) = (r.i32(), r.i32());
                let full = r.bool();
                let bitmask = r.i16() as u16;
                self.load_column(cx, cz, full, bitmask);
                Ok(())
            }
            0x23 => {
                // java17: `handleBlockChange`; a write into a column that is not loaded lands
                // in `EmptyChunk`, which keeps nothing.
                let (x, y, z) = (r.i32(), i32::from(r.u8()), r.i32());
                let (block, meta) = (r.varint(), i32::from(r.u8()));
                let pos = BlockPos::new(x, y, z);
                if self.world.is_chunk_loaded(pos.chunk_pos()) {
                    let state = mc_data::v5::state_from_v5_id_meta(block, meta);
                    self.world.set_block(pos, state);
                }
                Ok(())
            }
            0x27 => {
                // java17: `handleExplosion` adds the packet's player motion.
                let _centre = (r.f32(), r.f32(), r.f32(), r.f32());
                let records = usize::try_from(r.i32()).expect("records");
                r.at += records * 3;
                let s = &mut self.p.state;
                s.vel.x += f64::from(r.f32());
                s.vel.y += f64::from(r.f32());
                s.vel.z += f64::from(r.f32());
                Ok(())
            }
            0x2B => {
                let reason = r.u8();
                let value = r.f32();
                if reason == 3 {
                    // java17: `setGameType(getByID(floor_float(value + 0.5F)))`.
                    let creative = (value + 0.5).floor() as i32 == 1;
                    configure_capabilities(&mut self.p.state, creative);
                }
                Ok(())
            }
            0x39 => {
                // java17: `handlePlayerAbilities`.
                let flags = r.u8();
                let s = &mut self.p.state;
                s.flying = flags & 2 != 0;
                s.allow_flying = flags & 4 != 0;
                s.fly_speed = r.f32();
                Ok(())
            }
            0x01 => Err("a second S01".into()),
            0x07 => Err("a respawn (a new player)".into()),
            0x2D => Err("an open window (a screen)".into()),
            0x0A => Err("a bed (sleeping)".into()),
            _ => Err(format!("inbound packet 0x{id:02x} is not modelled")),
        }
    }

    /// java17: `handleChunkData`: a full chunk with an empty bitmask unloads the column;
    /// otherwise its listed sections are filled (`Chunk.fillChunk`). The sections are rebuilt
    /// from the world file, which is what the recorded S21 was built from.
    fn load_column(&mut self, cx: i32, cz: i32, full: bool, bitmask: u16) {
        let column = ChunkPos::new(cx, cz);
        if full && bitmask == 0 {
            self.world.unload_chunk(column);
            return;
        }
        scene::ensure_chunk(&mut self.world, column);
        let mut fills = Vec::new();
        for b in &self.spec.boxes {
            let x0 = b[0].max(cx * 16);
            let x1 = b[3].min(cx * 16 + 15);
            let z0 = b[2].max(cz * 16);
            let z1 = b[5].min(cz * 16 + 15);
            if x0 > x1 || z0 > z1 {
                continue;
            }
            let state = mc_data::v5::state_from_v5_id_meta(b[6], b[7]);
            for section in 0..16 {
                if bitmask & (1 << section) == 0 {
                    continue;
                }
                let y0 = b[1].max(section * 16);
                let y1 = b[4].min(section * 16 + 15);
                if y0 <= y1 {
                    fills.push((BlockPos::new(x0, y0, z0), BlockPos::new(x1, y1, z1), state));
                }
            }
        }
        scene::fill_static(&mut self.world, fills);
    }

    fn event(&mut self, event: &str) -> Result<(), String> {
        if event.starts_with("click") || event.starts_with("drag") {
            return Err(format!("event `{event}` (a container)"));
        }
        // java17: `runTick` reads the devices only while `currentScreen == null ||
        // allowUserInput` (`client/Minecraft.java:1646`); otherwise the screen's `handleInput`
        // consumes the events and the bindings keep their state. The wheel changes the held
        // item, never the movement.
        if self.terrain_screen || event.starts_with("wheel") {
            return Ok(());
        }
        let (down, binding) = match event.split_at(1) {
            ("+", b) => (true, b),
            ("-", b) => (false, b),
            _ => return Err(format!("event `{event}` is not modelled")),
        };
        let k = &mut self.keys;
        match binding {
            "forward" => k.forward = down,
            "back" => k.back = down,
            "left" => k.left = down,
            "right" => k.right = down,
            "jump" => k.jump = down,
            "sneak" => k.sneak = down,
            "sprint" => k.sprint = down,
            // Clicks, item use, hotbar and drop never move the player by themselves.
            "attack" | "use" | "drop" | "pick" => {}
            b if b.starts_with("hotbar") => {}
            other => return Err(format!("binding `{other}` opens a screen")),
        }
        Ok(())
    }

    /// java17: `EntityPlayer.attackTargetEntityWithCurrentItem`
    /// (`entity/player/EntityPlayer.java:1052-1110`), the part that moves the attacker: a hit
    /// that `attackEntityFrom` accepts on the client (only a remote player's,
    /// `client/entity/EntityOtherPlayerMP.java:38-41`) with knockback (sprinting; the recordings
    /// carry no Knockback enchantment) does `motionX/Z *= 0.6D` and `setSprinting(false)`.
    fn attack(&mut self, target: i32) {
        let s = &mut self.p.state;
        if self.players.contains(&target) && s.sprinting {
            s.vel.x *= 0.6;
            s.vel.z *= 0.6;
            s.set_sprinting(false);
        }
    }

    /// One `Entity.onUpdate` of the local player, then `sendMotionUpdates`' report.
    fn tick(&mut self, using_item: bool, sent: &mut Vec<Sent>) {
        let input = MoveInput {
            forward: self.keys.forward,
            back: self.keys.back,
            left: self.keys.left,
            right: self.keys.right,
            jump: self.keys.jump,
            sneak: self.keys.sneak,
            sprint: self.keys.sprint,
            using_item,
            yaw: self.p.state.yaw,
            pitch: self.p.state.pitch,
        };
        let out = self
            .p
            .tick_v5(&self.world, &input, &[])
            .expect("a 1.7.10 predictor");
        // java17: `sendMotionUpdates`' C0B edges (`client/entity/EntityClientPlayerMP.java:88-118`).
        let sprinting = self.p.state.sprinting;
        if sprinting != self.was_sprinting {
            sent.push(Sent::Action(if sprinting {
                "START_SPRINTING"
            } else {
                "STOP_SPRINTING"
            }));
            self.was_sprinting = sprinting;
        }
        let sneaking = self.p.state.sneaking;
        if sneaking != self.was_sneaking {
            sent.push(Sent::Action(if sneaking {
                "START_SNEAKING"
            } else {
                "STOP_SNEAKING"
            }));
            self.was_sneaking = sneaking;
        }
        sent.push(Sent::from_outbound(out));
    }
}

/// java17: `WorldSettings.GameType.configurePlayerCapabilities`.
fn configure_capabilities(s: &mut PhysicsState, creative: bool) {
    s.allow_flying = creative;
    if !creative {
        s.flying = false;
    }
}

/// java17: `handleEntityProperties` for `generic.movementSpeed`: base value, then
/// `removeAllModifiers` and the packet's modifiers, evaluated by `computeValue` in the
/// operation sets' UUID order. The canonical sprint UUID is kept apart, as the physics state
/// keeps it (`PhysicsState::sprint_modifier_amount`).
fn apply_attributes(r: &mut Reader<'_>, s: &mut PhysicsState) {
    let count = r.i32();
    for _ in 0..count {
        let key = r.string();
        let base = r.f64();
        let n = r.i16();
        let mut modifiers = Vec::new();
        for _ in 0..n {
            let uuid = (u128::from(r.i64() as u64) << 64) | u128::from(r.i64() as u64);
            let amount = r.f64();
            let op = r.u8();
            modifiers.push((uuid, amount, op));
        }
        if !key.eq_ignore_ascii_case("generic.movementSpeed")
            && !key.eq_ignore_ascii_case("Movement Speed")
        {
            continue;
        }
        s.sprint_modifier_amount = modifiers.iter().find(|m| m.0 == SPRINT_UUID).map(|m| m.1);
        let others: Vec<_> = modifiers
            .into_iter()
            .filter(|m| m.0 != SPRINT_UUID)
            .collect();
        let ordered = |op: u8| -> Vec<f64> {
            let of_op: Vec<_> = others.iter().filter(|m| m.2 == op).collect();
            let ids: Vec<u128> = of_op.iter().map(|m| m.0).collect();
            mc_physics::attribute::java_uuid_hash_set_order(&ids)
                .into_iter()
                .map(|i| of_op[i].1)
                .collect()
        };
        let mut value = base;
        for amount in ordered(0) {
            value += amount;
        }
        let summed = value;
        for amount in ordered(1) {
            value += summed * amount;
        }
        for amount in ordered(2) {
            value *= 1.0 + amount;
        }
        s.movement_speed = value;
    }
}

/// Compares the recorded player fields with the predictor, bit for bit.
fn compare_state(expected: &Map<String, Value>, p: &Predictor) -> Result<(), String> {
    let s = &p.state;
    let v5 = p.v5().expect("a 1.7.10 predictor");
    let doubles = [
        ("x", s.pos.x),
        ("y", s.aabb().min.y),
        ("z", s.pos.z),
        ("posY", v5.pos_y()),
        ("vx", s.vel.x),
        ("vy", s.vel.y),
        ("vz", s.vel.z),
    ];
    for (key, actual) in doubles {
        let want = expected[key].as_f64().unwrap_or_else(|| panic!("{key}"));
        if want.to_bits() != actual.to_bits() {
            return Err(format!("{key}: java {want:?}, rust {actual:?}"));
        }
    }
    let floats = [
        ("yaw", s.yaw),
        ("pitch", s.pitch),
        ("fallDistance", s.fall_distance),
        ("moveForward", s.move_forward),
        ("moveStrafing", s.move_strafing),
    ];
    for (key, actual) in floats {
        let want = expected[key].as_f64().unwrap_or_else(|| panic!("{key}")) as f32;
        if want.to_bits() != actual.to_bits() {
            return Err(format!("{key}: java {want:?}, rust {actual:?}"));
        }
    }
    let flags = [
        ("onGround", s.on_ground),
        ("sprinting", s.sprinting),
        ("sneaking", s.sneaking),
        ("flying", s.flying),
        ("collidedH", s.collided_horizontally),
    ];
    for (key, actual) in flags {
        let want = expected[key].as_bool().unwrap_or_else(|| panic!("{key}"));
        if want != actual {
            return Err(format!("{key}: java {want}, rust {actual}"));
        }
    }
    let ints = [
        ("sprintingTicksLeft", s.sprinting_ticks_left),
        ("jumpTicks", s.jump_ticks),
    ];
    for (key, actual) in ints {
        let want = expected[key].as_i64().unwrap_or_else(|| panic!("{key}"));
        if want != i64::from(actual) {
            return Err(format!("{key}: java {want}, rust {actual}"));
        }
    }
    Ok(())
}

fn replay(name: &str) -> Outcome {
    replay_with(name, |_| {})
}

/// [`replay`] with `mutate` applied to the predictor before the first tick (negative controls).
fn replay_with(name: &str, mutate: impl Fn(&mut Predictor)) -> Outcome {
    let dir = fixtures().join("java17");
    let blobs: Vec<Vec<u8>> = read_json(&dir.join("inbound.json"))["blobs"]
        .as_array()
        .expect("blobs")
        .iter()
        .map(|b| hex(b.as_str().expect("hex blob")))
        .collect();
    let recording = read_json(&dir.join(format!("{name}.json")));
    let sites: Vec<String> = recording["sites"]
        .as_array()
        .expect("sites")
        .iter()
        .map(|s| s[1].as_str().expect("site").to_owned())
        .collect();
    let frames = recording["frames"].as_array().expect("frames");
    let world = recording["world"].as_str().expect("world");

    // Frame 0: S01 creates the player (`handleJoinGame` -> `loadWorld`, `setGameType`). The
    // recorded state is the new `EntityClientPlayerMP` before its first tick.
    let mut expected = frames[0]["player"]
        .as_object()
        .expect("frame 0 player")
        .clone();
    let join = &blobs[frames[0]["inbound"][0]["blob"].as_u64().expect("S01") as usize];
    let mut r = Reader::new(join);
    assert_eq!(r.varint(), 0x01, "frame 0 is the join");
    let local_id = r.i32();
    let game_type = i32::from(r.u8() & !8);

    let feet = DVec3::new(
        expected["x"].as_f64().expect("x"),
        expected["y"].as_f64().expect("y"),
        expected["z"].as_f64().expect("z"),
    );
    let mut state = PhysicsState::new_player(feet);
    state.yaw = expected["yaw"].as_f64().expect("yaw") as f32;
    state.pitch = expected["pitch"].as_f64().expect("pitch") as f32;
    configure_capabilities(&mut state, game_type == 1);
    let mut p = Predictor::new_v5(state);
    let eye = expected["posY"].as_f64().expect("posY");
    if p.v5().expect("v5").pos_y().to_bits() != eye.to_bits() {
        return Outcome::Mismatch(format!("frame 0: posY {eye:?} is not minY + 1.62F"));
    }
    mutate(&mut p);

    let mut replay = Replay {
        spec: WorldSpec::load(world),
        world: World::new(),
        p,
        local_id,
        keys: Keys::default(),
        was_sprinting: false,
        was_sneaking: false,
        terrain_screen: true,
        players: Vec::new(),
    };
    let mut queued_packets: Vec<usize> = Vec::new();
    let mut queued_events: Vec<String> = Vec::new();
    let mut world_edit: Option<usize> = None;

    for (index, frame) in frames.iter().enumerate().skip(1) {
        let fail = |what: String| {
            let note = frame["note"].as_str().unwrap_or("");
            format!("{name} frame {index} ({note}): {what}")
        };
        match &frame["player"] {
            Value::Object(delta) => {
                for (k, v) in delta {
                    expected.insert(k.clone(), v.clone());
                }
            }
            Value::Null if frame.get("player").is_some() => {
                return Outcome::Unsupported(fail("the player is removed".into()));
            }
            _ => {}
        }
        if expected["screen"]
            .as_str()
            .is_some_and(|s| s != "GuiDownloadTerrain")
        {
            return Outcome::Unsupported(fail(format!("screen {}", expected["screen"])));
        }
        if expected["riding"].as_bool() == Some(true) {
            return Outcome::Unsupported(fail("riding".into()));
        }
        let recorded: Vec<(&str, &Value)> = frame["sent"]
            .as_array()
            .map(|sent| {
                sent.iter()
                    .map(|s| {
                        (
                            sites[s["at"].as_u64().expect("at") as usize].as_str(),
                            &s["p"],
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        // The client predicts the blocks it places and breaks (the interaction half). This
        // replay's world does not, so a divergence after such an edit is reported as that.
        if world_edit.is_none() {
            world_edit = recorded.iter().find_map(|(_, p)| {
                let placed =
                    p["kind"] == "C08PacketPlayerBlockPlacement" && p["face"].as_i64() != Some(-1);
                let broken =
                    p["kind"] == "C07PacketPlayerDigging" && p["status"] == "STOP_DESTROY_BLOCK";
                (placed || broken).then_some(index)
            });
        }
        // The attacks of each tick: a tick's input phase precedes its movement report, so an
        // attack belongs to the tick of the next C03..C06 sent from `sendMotionUpdates`.
        let ticks = frame["ticks"].as_u64().expect("ticks") as usize;
        let mut attacks: Vec<Vec<i32>> = vec![Vec::new(); ticks];
        let mut tick_of = 0usize;
        for (site, p) in &recorded {
            if p["kind"] == "C02PacketUseEntity" && p["action"] == "ATTACK" {
                let target = i32::try_from(p["entity"].as_i64().expect("entity")).expect("id");
                if let Some(list) = attacks.get_mut(tick_of) {
                    list.push(target);
                }
            }
            if site.ends_with(MOVEMENT_SITE)
                && matches!(Sent::from_recording(p), Some(s) if !matches!(s, Sent::Action(_)))
            {
                tick_of += 1;
            }
        }

        // (1) the render-time look; (2) the device events and (3) the packets queue up.
        if let Some(look) = frame["look"].as_array() {
            replay.p.state.yaw = look[0].as_f64().expect("yaw") as f32;
            replay.p.state.pitch = look[1].as_f64().expect("pitch") as f32;
        }
        if let Some(events) = frame["events"].as_array() {
            queued_events.extend(events.iter().map(|e| e.as_str().expect("event").to_owned()));
        }
        if let Some(inbound) = frame["inbound"].as_array() {
            for packet in inbound {
                let blob = packet["blob"].as_u64().expect("blob") as usize;
                // java17: S00 keep-alive and S40 disconnect are handled on arrival; neither
                // moves the player. Everything else waits for the tick.
                if !matches!(blobs[blob].first(), Some(0x00 | 0x40)) {
                    queued_packets.push(blob);
                }
            }
        }

        // (4) the ticks.
        let using_item = expected["usingItem"].as_bool().unwrap_or(false);
        let mut sent = Vec::new();
        for (tick, tick_attacks) in attacks.iter().enumerate() {
            if tick == 0 {
                for blob in std::mem::take(&mut queued_packets) {
                    if let Err(why) = replay.apply(&blobs[blob], &mut sent) {
                        return Outcome::Unsupported(fail(why));
                    }
                }
                for event in std::mem::take(&mut queued_events) {
                    if let Err(why) = replay.event(&event) {
                        return Outcome::Unsupported(fail(why));
                    }
                }
            }
            for &target in tick_attacks {
                replay.attack(target);
            }
            replay.tick(using_item, &mut sent);
        }

        // (5) compare: the movement packets, then the player.
        let want: Vec<Sent> = recorded
            .iter()
            .filter(|(site, _)| site.ends_with(MOVEMENT_SITE) || site.ends_with(S08_REPLY_SITE))
            .filter_map(|(_, p)| Sent::from_recording(p))
            .collect();
        let want_keys: Vec<String> = want.iter().map(|s| s.key()).collect();
        let got_keys: Vec<String> = sent.iter().map(|s| s.key()).collect();
        let difference = if want_keys != got_keys {
            Some(format!(
                "movement packets\n  java {want_keys:?}\n  rust {got_keys:?}"
            ))
        } else {
            compare_state(&expected, &replay.p).err()
        };
        if let Some(what) = difference {
            return match world_edit {
                Some(edit) => Outcome::Unsupported(fail(format!(
                    "diverges after the client-side block edit of frame {edit}: {what}"
                ))),
                None => Outcome::Mismatch(fail(what)),
            };
        }
    }
    Outcome::Matched {
        frames: frames.len() - 1,
    }
}

/// The recordings this physics-level replay cannot carry: the frame where each must stop and the
/// reason it reports there (screens, a respawn, and the blocks the client places itself, which are
/// the client-level replay's). The frame is pinned so that a movement regression earlier in one
/// of these recordings fails instead of hiding behind the reason: in `place_blocks` the first
/// edit is frame 8, but the frames up to 59 still match; java lands on the block the player
/// placed under itself at frame 60.
const UNSUPPORTED: &[(&str, usize, &str)] = &[
    ("inventory_clicks_drags", 8, "screen \"GuiInventory\""),
    ("inventory_middle_drag", 8, "screen \"GuiInventory\""),
    (
        "inventory_middle_drag_after_right_drag",
        8,
        "screen \"GuiInventory\"",
    ),
    ("net_respawn_and_death", 10, "a respawn"),
    ("place_blocks", 60, "after the client-side block edit"),
    ("place_sneak_bridge", 30, "after the client-side block edit"),
];

fn fixtures_present() -> bool {
    let present = fixtures().join("java17/inbound.json").is_file();
    if !present {
        println!(
            "skip: no 1.7.10 oracle fixtures at {}",
            fixtures().display()
        );
    }
    present
}

fn check(name: &str) {
    if !fixtures_present() {
        return;
    }
    match replay(name) {
        Outcome::Matched { frames } => println!("{name}: {frames} frames match"),
        Outcome::Unsupported(why) => panic!("{name} is not replayable here: {why}"),
        Outcome::Mismatch(why) => panic!("{why}"),
    }
}

/// One test per recording the replay carries; each must match every frame.
macro_rules! matched_recordings {
    ($($name:ident),* $(,)?) => {
        /// The recordings this replay carries.
        const MATCHED: &[&str] = &[$(stringify!($name)),*];
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
    place_far_coordinates,
    use_eat_and_bow,
);

/// Every recording is either carried (and matches) or listed as unsupported with the reason the
/// replay gives; a new recording must be added to one of the two lists. Prints the table.
#[test]
fn every_recording_is_accounted_for() {
    if !fixtures_present() {
        return;
    }
    let mut names: Vec<String> = std::fs::read_dir(fixtures().join("java17"))
        .expect("java17")
        .filter_map(|e| {
            let name = e.ok()?.file_name().into_string().ok()?;
            let stem = name.strip_suffix(".json")?;
            (stem != "inbound").then(|| stem.to_owned())
        })
        .collect();
    names.sort();
    for name in &names {
        let outcome = replay(name);
        println!("{name}: {outcome:?}");
        if MATCHED.contains(&name.as_str()) {
            assert!(
                matches!(outcome, Outcome::Matched { .. }),
                "{name}: {outcome:?}"
            );
        } else if let Some((_, frame, reason)) = UNSUPPORTED.iter().find(|(n, ..)| n == name) {
            let at = format!("{name} frame {frame} (");
            assert!(
                matches!(&outcome, Outcome::Unsupported(why)
                    if why.starts_with(&at) && why.contains(reason)),
                "{name}: expected unsupported at frame {frame} ({reason}), got {outcome:?}"
            );
        } else {
            panic!("{name} is in neither list: {outcome:?}");
        }
    }
    for name in MATCHED.iter().chain(UNSUPPORTED.iter().map(|(n, ..)| n)) {
        assert!(names.iter().any(|n| n == name), "{name} has no recording");
    }
}

/// The comparison is not vacuous: the same replay with 1.8.9's step height fails the bed
/// recording, and a player too hungry to sprint fails the sneak-release sprint recording.
#[test]
fn negative_controls_fail_the_recordings() {
    if !fixtures_present() {
        return;
    }
    assert!(matches!(replay("focus_step_bed"), Outcome::Matched { .. }));
    let tall_step = replay_with("focus_step_bed", |p| p.state.step_height = 0.6);
    assert!(matches!(tall_step, Outcome::Mismatch(_)), "{tall_step:?}");
    println!("step 0.6: {tall_step:?}");

    assert!(matches!(
        replay("focus_sprint_sneak_release"),
        Outcome::Matched { .. }
    ));
    let hungry = replay_with("focus_sprint_sneak_release", |p| p.state.food_level = 6);
    assert!(matches!(hungry, Outcome::Mismatch(_)), "{hungry:?}");
    println!("food 6: {hungry:?}");
}
