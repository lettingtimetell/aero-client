//! S13's Java-dumped physics fixtures, expressed as generic [`Trace`]s.
//!
//! # What is new here, and what is not
//!
//! **Not new:** the evidence. `crates/mc-physics/javaref/PhysicsRefGen.java` runs the real
//! recompiled vanilla 1.8.9 classes under a JDK, drives a genuine `EntityPlayer` through
//! `Entity.moveEntity`/`EntityLivingBase.moveEntityWithHeading`, and dumps raw IEEE-754 bit
//! patterns; `crates/mc-physics/tests/parity.rs` already replays them and already passes at
//! tolerance `0.0`. S13's original twelve scenarios totalled 4,570 ticks; the committed
//! fixture set now contains 22 scenarios and 6,783 ticks, including the ten GAP A11 additions.
//!
//! **New here:** the *shape*. Those fixtures are read into [`Trace`], the replay produces
//! another [`Trace`], and the two are compared by [`Trace::compare`] — the same comparator any
//! other subsystem can point at its own Java-dumped reference. Three things fall out of the
//! generalisation that the bespoke test did not have:
//!
//! 1. **The flags byte is decomposed into seven named boolean channels.** S13 asserts one `u8`
//!    equality, so a failure says "flags 0b0100011 != 0b0100111". Here a failure says
//!    `in_water`, which is the difference between a five-minute diagnosis and a five-second one.
//! 2. **Every channel reports its worst error, not just the one that failed first.** A run that
//!    passes with worst error `0.0` on all fifteen channels is a stronger statement than "the
//!    assertions did not fire", and [`TraceDiff::is_bit_exact`] makes it a checkable one.
//! 3. **The fixtures are consumed, not copied.** This module reads
//!    `crates/mc-physics/tests/fixtures/*.bin` in place. There is deliberately no second copy
//!    of them in this crate: two copies of a golden fixture is one copy too many.
//!
//! # The fixture format (`MCPHYS02`), as S13 emits it
//!
//! ```text
//!   8  magic "MCPHYS02"
//!   2  scenario name length (u16) + name bytes
//!  16  loaded chunk rect (4 × i32: cx0, cz0, cx1, cz1)
//!   4  fill count (u32); each fill is 6 × i32 (from/to block coords) + u16 block state
//!  24  start position (3 × f64)
//!   8  start yaw, start pitch (2 × f32)
//!   *  setup event list (u32 count; each event is u8 kind + 3 × f64)
//!   4  tick count (u32); each tick is:
//!        1  key bitmask
//!        8  yaw, pitch (2 × f32)
//!        *  event list, applied before the tick runs
//!       48  expected position and motion (6 × f64)
//!        4  expected fallDistance (f32)
//!        1  expected flag bitmask
//!        4  expected predicted fall damage (i32; -1 for none)
//!       20  limbSwing, limbSwingAmount, prevLimbSwingAmount,
//!           distanceWalkedModified, distanceWalkedOnStepModified (5 × f32)
//!        4  nextStepDistance (i32)
//!        1  second flag bitmask (isFlying, noClip, inPortal, cactus damage)
//!        *  the step sound `playStepSound` played (u16 length + UTF-8 name) + volume (f32)
//!        4  the swim sound volume this tick (f32, NaN for none)
//! ```
//!
//! **The trailing per-tick fields are read and discarded here.** They exist so `mc-physics`'
//! own `tests/parity.rs` can assert the walk-animation and footstep state bit-exactly; adding them as [`Channel`]s here would change this crate's encoded trace
//! shape, which is a separate decision from consuming the new fixtures. The **events** are not
//! discarded: they are the server-driven state (`S12PacketEntityVelocity`, capabilities,
//! potions, the movement-speed attribute, item use, Depth Strider) without which the replay
//! would not reproduce the reference at all.
//!
//! This module re-reads it with checked, panic-free reads. S13's own test uses direct slice
//! indexing, which is fine for a file it generated itself; a *harness* is a place other people
//! point at files it did not generate, so the same bytes get the untrusted-input treatment.

use std::path::PathBuf;

use mc_data::BlockState;
use mc_math::{clamp_f64, BlockPos, DVec3};
use mc_physics::{tick_player, MoveInput, PhysicsState};
use mc_world::World;

use crate::trace::{Channel, Kind, Tolerances, Trace, TraceDiff, Value};
use crate::ParityError;

/// The fixture container's magic bytes.
pub const PHYSICS_MAGIC: &[u8; 8] = b"MCPHYS02";

/// Every scenario S13 generated, in the order its own test lists them.
pub const SCENARIOS: [&str; 22] = [
    "flat_walk",
    "wall_collide",
    "step_up",
    "sneak_edge",
    "fall_damage",
    "friction",
    "water_swim",
    "lava_swim",
    "ladder_climb",
    "cobweb",
    "connected_blocks",
    "long_random",
    // GAP A11 - the branches the original twelve deliberately avoided: creative flight, a
    // server-sent movement speed, knockback, Jump Boost, item use, Depth Strider, spectator
    // `noClip`, a dead player, an unloaded column and the cactus/portal contact callbacks.
    "creative_fly",
    "speed_attribute",
    "knockback",
    "jump_boost",
    "item_use",
    "depth_strider",
    "spectator_noclip",
    "dead_player",
    "unloaded_chunk",
    "contact_blocks",
];

/// The channel list both the reference and the replay produce.
///
/// Order is fixed so the encoded form is stable; comparison is by name, so a future addition
/// in the middle would not silently shift anything.
#[must_use]
pub fn channels() -> Vec<Channel> {
    let f64c = |n: &str| Channel {
        name: n.to_owned(),
        kind: Kind::F64,
    };
    let boolc = |n: &str| Channel {
        name: n.to_owned(),
        kind: Kind::Bool,
    };
    vec![
        f64c("pos.x"),
        f64c("pos.y"),
        f64c("pos.z"),
        f64c("vel.x"),
        f64c("vel.y"),
        f64c("vel.z"),
        Channel {
            name: "fall_distance".to_owned(),
            kind: Kind::F32,
        },
        boolc("on_ground"),
        boolc("collided_horizontally"),
        boolc("collided_vertically"),
        boolc("in_water"),
        boolc("on_ladder"),
        boolc("sprinting"),
        boolc("in_lava"),
        Channel {
            name: "fall_damage".to_owned(),
            kind: Kind::I32,
        },
    ]
}

// The flag bits, exactly as `PhysicsRefGen.java` packs them.
const F_ON_GROUND: u8 = 1;
const F_COLLIDED_H: u8 = 2;
const F_COLLIDED_V: u8 = 4;
const F_IN_WATER: u8 = 8;
const F_ON_LADDER: u8 = 16;
const F_SPRINTING: u8 = 32;
const F_IN_LAVA: u8 = 64;

// The key bits, likewise.
const K_FORWARD: u8 = 1;
const K_BACK: u8 = 2;
const K_LEFT: u8 = 4;
const K_RIGHT: u8 = 8;
const K_JUMP: u8 = 16;
const K_SNEAK: u8 = 32;
const K_SPRINT: u8 = 64;

/// A loaded scenario: the world, the scripted inputs, and the Java-produced reference trace.
pub struct PhysicsFixture {
    /// The scenario name, as recorded inside the file.
    pub name: String,
    /// The synthetic world the scenario runs in, rebuilt from the fixture's fill list.
    pub world: World,
    /// The player's starting position.
    pub start: DVec3,
    /// The player's starting yaw.
    pub start_yaw: f32,
    /// The player's starting pitch.
    pub start_pitch: f32,
    /// One input per tick.
    pub inputs: Vec<MoveInput>,
    /// Server-driven changes applied before tick 0.
    pub setup: Vec<FixtureEvent>,
    /// Server-driven changes applied at the start of each tick, parallel to `inputs`.
    pub events: Vec<Vec<FixtureEvent>>,
    /// What the real Java client produced for those inputs.
    pub reference: Trace,
}

/// Where S13's committed fixtures live, relative to this crate.
///
/// A workspace path rather than a copy: see this module's header for why.
#[must_use]
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("mc-physics")
        .join("tests")
        .join("fixtures")
}

/// Load one scenario by name.
///
/// # Errors
/// [`ParityError::Io`] if the fixture is missing, or any parse variant if it is malformed.
pub fn load(name: &str) -> Result<PhysicsFixture, ParityError> {
    let path = fixture_dir().join(format!("{name}.bin"));
    let bytes = std::fs::read(&path)?;
    parse(&bytes)
}

/// Parse a fixture's bytes.
///
/// # Errors
/// [`ParityError::BadMagic`], [`ParityError::Truncated`], [`ParityError::InvalidUtf8`],
/// [`ParityError::TrailingBytes`], or [`ParityError::ShapeMismatch`] from building the trace.
pub fn parse(bytes: &[u8]) -> Result<PhysicsFixture, ParityError> {
    let mut r = Reader::new(bytes);

    let magic = r.take(8, "physics fixture magic")?;
    if magic != PHYSICS_MAGIC {
        return Err(ParityError::BadMagic {
            expected: "MCPHYS01",
            found: String::from_utf8_lossy(magic).into_owned(),
        });
    }

    let name_len = r.u16("scenario name length")?;
    let name_bytes = r.take(name_len as usize, "scenario name")?;
    let name = std::str::from_utf8(name_bytes)
        .map_err(|_| ParityError::InvalidUtf8("the scenario name"))?
        .to_owned();

    // The rect of chunk columns the fixture calls loaded. Outside it a client has the shared
    // `EmptyChunk`, which is what `moveEntityWithHeading`'s gravity pin tests.
    let cx0 = r.i32("loaded cx0")?;
    let cz0 = r.i32("loaded cz0")?;
    let cx1 = r.i32("loaded cx1")?;
    let cz1 = r.i32("loaded cz1")?;
    let mut world = World::new();
    for cx in cx0..=cx1 {
        for cz in cz0..=cz1 {
            mc_physics::scene::ensure_chunk(&mut world, mc_math::ChunkPos::new(cx, cz));
        }
    }

    // Static fixture terrain is installed in one lighting-free batch. This preserves the
    // encoded fill order and residency while avoiding a flood fill for every block write;
    // fixture light is intentionally not reconstructed because physics does not read it.
    let fills = r.u32("fill count")?;
    let mut static_fills = Vec::with_capacity(fills as usize);
    for _ in 0..fills {
        let x0 = r.i32("fill x0")?;
        let y0 = r.i32("fill y0")?;
        let z0 = r.i32("fill z0")?;
        let x1 = r.i32("fill x1")?;
        let y1 = r.i32("fill y1")?;
        let z1 = r.i32("fill z1")?;
        let state = BlockState::from_raw(r.u16("fill block state")?);
        static_fills.push((BlockPos::new(x0, y0, z0), BlockPos::new(x1, y1, z1), state));
    }
    mc_physics::scene::fill_static(&mut world, static_fills);

    let start = DVec3::new(r.f64("start x")?, r.f64("start y")?, r.f64("start z")?);
    let start_yaw = r.f32("start yaw")?;
    let start_pitch = r.f32("start pitch")?;
    let setup = read_events(&mut r)?;

    let ticks = r.u32("tick count")?;
    let mut inputs = Vec::with_capacity(ticks as usize);
    let mut events = Vec::with_capacity(ticks as usize);
    let mut reference = Trace::new(name.clone(), channels());
    for _ in 0..ticks {
        let keys = r.u8("input keys")?;
        let yaw = r.f32("input yaw")?;
        let pitch = r.f32("input pitch")?;
        inputs.push(MoveInput {
            forward: keys & K_FORWARD != 0,
            back: keys & K_BACK != 0,
            left: keys & K_LEFT != 0,
            right: keys & K_RIGHT != 0,
            jump: keys & K_JUMP != 0,
            sneak: keys & K_SNEAK != 0,
            sprint: keys & K_SPRINT != 0,
            // The recorded trace format has no item-use channel — `keys` is the seven movement
            // bits and nothing else — so a replayed trace never has an item in use.
            using_item: false,
            yaw,
            pitch,
        });
        events.push(read_events(&mut r)?);

        let pos = [r.f64("pos x")?, r.f64("pos y")?, r.f64("pos z")?];
        let vel = [r.f64("vel x")?, r.f64("vel y")?, r.f64("vel z")?];
        let fall_distance = r.f32("fall distance")?;
        let flags = r.u8("flags")?;
        let fall_damage = r.i32("fall damage")?;
        reference.push(vec![
            Value::F64(pos[0]),
            Value::F64(pos[1]),
            Value::F64(pos[2]),
            Value::F64(vel[0]),
            Value::F64(vel[1]),
            Value::F64(vel[2]),
            Value::F32(fall_distance),
            Value::Bool(flags & F_ON_GROUND != 0),
            Value::Bool(flags & F_COLLIDED_H != 0),
            Value::Bool(flags & F_COLLIDED_V != 0),
            Value::Bool(flags & F_IN_WATER != 0),
            Value::Bool(flags & F_ON_LADDER != 0),
            Value::Bool(flags & F_SPRINTING != 0),
            Value::Bool(flags & F_IN_LAVA != 0),
            Value::I32(fall_damage),
        ])?;

        // Read past the animation/footstep block; see this module's header for why it is not a
        // channel here.
        for what in [
            "limbSwing",
            "limbSwingAmount",
            "prevLimbSwingAmount",
            "distanceWalkedModified",
            "distanceWalkedOnStepModified",
        ] {
            r.f32(what)?;
        }
        r.i32("nextStepDistance")?;
        r.u8("second flag bitmask")?;
        let sound_len = r.u16("step sound name length")?;
        r.take(sound_len as usize, "step sound name")?;
        r.f32("step sound volume")?;
        r.f32("swim sound volume")?;
    }

    if r.remaining() != 0 {
        return Err(ParityError::TrailingBytes {
            what: "the physics fixture",
            count: r.remaining(),
        });
    }

    Ok(PhysicsFixture {
        name,
        world,
        start,
        start_yaw,
        start_pitch,
        inputs,
        setup,
        events,
        reference,
    })
}

/// One server-driven change, applied at the start of a tick.
///
/// java: the `NetHandlerPlayClient` handlers that reach physics. `PhysicsRefGen.java`'s format
/// comment owns the kind numbering; this mirrors it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixtureEvent {
    /// Which handler. See `PhysicsRefGen.java`'s format comment.
    pub kind: u8,
    /// First argument.
    pub a: f64,
    /// Second argument.
    pub b: f64,
    /// Third argument.
    pub c: f64,
}

fn read_events(r: &mut Reader<'_>) -> Result<Vec<FixtureEvent>, ParityError> {
    let count = r.u32("event count")?;
    let mut out = Vec::with_capacity(count as usize);
    for _ in 0..count {
        out.push(FixtureEvent {
            kind: r.u8("event kind")?,
            a: r.f64("event a")?,
            b: r.f64("event b")?,
            c: r.f64("event c")?,
        });
    }
    Ok(out)
}

/// A small, oracle-local representation of one modifier.
///
/// The fixture wire format predates multi-modifier attributes and carries one amount/operation
/// pair in an event. The full representation is kept here for the exact-bit controls below; it
/// deliberately does not call the client session's resolver.
#[derive(Debug, Clone, Copy, PartialEq)]
struct FixtureModifier {
    uuid: u128,
    amount: f64,
    operation: i32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttributeResolutionError {
    DuplicateUuid(u128),
}

/// Return modifier indices in Java's `HashSet<UUID>` traversal order for one operation.
fn modifier_indices_in_java_order(modifiers: &[FixtureModifier], operation: i32) -> Vec<usize> {
    let indices = modifiers
        .iter()
        .enumerate()
        .filter(|(_, modifier)| modifier.operation == operation)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let ids = indices
        .iter()
        .map(|&index| modifiers[index].uuid)
        .collect::<Vec<_>>();
    mc_physics::attribute::java_uuid_hash_set_order(&ids)
        .into_iter()
        .map(|index| indices[index])
        .collect()
}

/// `ModifiableAttributeInstance.computeValue` for a movement-speed snapshot.
///
/// Java (`ModifiableAttributeInstance.java:166-188`) left-folds each operation's UUID-keyed
/// `HashSet` in its own traversal order, then clamps through `RangedAttribute.clampValue`.
/// Keeping this copy independent from `mc-client` is intentional: this crate is the replay
/// oracle, not another caller of the implementation under test.
fn compute_attribute_value(
    base_value: f64,
    modifiers: &[FixtureModifier],
) -> Result<f64, AttributeResolutionError> {
    let mut seen = Vec::with_capacity(modifiers.len());
    for modifier in modifiers {
        if seen.contains(&modifier.uuid) {
            return Err(AttributeResolutionError::DuplicateUuid(modifier.uuid));
        }
        seen.push(modifier.uuid);
    }

    let mut d0 = base_value;
    for index in modifier_indices_in_java_order(modifiers, 0) {
        d0 += modifiers[index].amount;
    }

    let mut d1 = d0;
    for index in modifier_indices_in_java_order(modifiers, 1) {
        d1 += d0 * modifiers[index].amount;
    }
    for index in modifier_indices_in_java_order(modifiers, 2) {
        d1 *= 1.0 + modifiers[index].amount;
    }

    // generic.movementSpeed is RangedAttribute(0.0, 1024.0). Spell out Java's comparison-based
    // clamp via mc-math so NaN remains NaN, just as MathHelper.clamp_double does.
    Ok(clamp_f64(d1, 0.0, 1024.0))
}

/// The one-modifier event projection of `compute_attribute_value`.
fn attribute_value(base: f64, amount: f64, operation: i32) -> f64 {
    if !(0..=2).contains(&operation) {
        return clamp_f64(base, 0.0, 1024.0);
    }
    let modifier = FixtureModifier {
        uuid: 0,
        amount,
        operation,
    };
    compute_attribute_value(base, &[modifier])
        .expect("the one-modifier fixture projection cannot contain a duplicate UUID")
}

/// Apply one fixture event to the simulated player, as `PhysicsRefGen.RefPlayer.apply` applies
/// it to the real one.
fn apply_event(state: &mut PhysicsState, using_item: &mut bool, e: FixtureEvent) {
    match e.kind {
        0 => state.vel = DVec3::new(e.a, e.b, e.c),
        1 => state.flying = e.a != 0.0,
        2 => state.allow_flying = e.a != 0.0,
        3 => state.fly_speed = e.a as f32,
        // `capabilities.walkSpeed` has no client-side physics effect at all: `EntityPlayer
        // .onLivingUpdate:622` pushes it into the attribute inside `if (!isRemote)`.
        4 => {}
        // The oracle's `setGameType(SPECTATOR)` sets both the tab-row flag (`noClip`) and the
        // controller mode (forced flight, no landing clear); L2 split them in `PhysicsState`.
        5 => {
            state.spectator = e.a != 0.0;
            state.spectator_mode = e.a != 0.0;
        }
        6 => *using_item = e.a != 0.0,
        7 => state.movement_blocked = e.a <= 0.0,
        8 => state.food_level = e.a as i32,
        9 => match e.a as i32 {
            // S1D reads the amplifier with `readByte`; preserve Java's signed byte rather than
            // clamping wire values 128..=255 to zero before the physics consumer sees them.
            8 => {
                state.jump_boost = (e.c > 0.0).then(|| {
                    i8::try_from(e.b as i32).expect("Jump Boost amplifier must be a signed byte")
                })
            }
            15 => state.blindness = e.c > 0.0,
            _ => {}
        },
        10 => state.movement_speed = attribute_value(e.a, e.b, e.c as i32),
        11 => state.depth_strider = e.a as f32,
        _ => {}
    }
}

/// Replay a fixture's inputs through `mc-physics` and collect the result as a [`Trace`].
///
/// # Errors
/// [`ParityError::ShapeMismatch`] can only fire if [`channels`] and this function disagree,
/// which is a programming error in this file rather than a data problem — it is a `Result`
/// rather than a panic so that a harness never aborts a run it could have reported.
pub fn replay(fixture: &PhysicsFixture) -> Result<Trace, ParityError> {
    let mut state = PhysicsState::new_player(fixture.start);
    state.yaw = fixture.start_yaw;
    state.pitch = fixture.start_pitch;

    let mut using_item = false;
    for &event in &fixture.setup {
        apply_event(&mut state, &mut using_item, event);
    }

    let mut out = Trace::new(fixture.name.clone(), channels());
    for (i, input) in fixture.inputs.iter().enumerate() {
        for &event in &fixture.events[i] {
            apply_event(&mut state, &mut using_item, event);
        }
        let input = &MoveInput {
            using_item,
            ..*input
        };
        tick_player(&fixture.world, &mut state, input);
        out.push(vec![
            Value::F64(state.pos.x),
            Value::F64(state.pos.y),
            Value::F64(state.pos.z),
            Value::F64(state.vel.x),
            Value::F64(state.vel.y),
            Value::F64(state.vel.z),
            Value::F32(state.fall_distance),
            Value::Bool(state.on_ground),
            Value::Bool(state.collided_horizontally),
            Value::Bool(state.collided_vertically),
            Value::Bool(state.in_water),
            Value::Bool(mc_physics::player::is_on_ladder(&fixture.world, &state)),
            Value::Bool(state.sprinting),
            Value::Bool(mc_physics::fluid::is_in_lava(&fixture.world, &state)),
            // The fixture stores -1 for "no fall damage this tick"; `Option::None` maps back
            // onto that rather than onto 0, which is a real damage value.
            Value::I32(
                state
                    .predicted_fall_damage
                    .map_or(-1, |d| i32::try_from(d).unwrap_or(i32::MAX)),
            ),
        ])?;
    }
    Ok(out)
}

/// Load, replay and compare one scenario at a given tolerance.
///
/// # Errors
/// Anything [`load`], [`replay`] or [`Trace::compare`] returns.
pub fn check(name: &str, tolerances: &Tolerances) -> Result<TraceDiff, ParityError> {
    let fixture = load(name)?;
    let actual = replay(&fixture)?;
    fixture.reference.compare(&actual, tolerances)
}

/// A checked reader, as in [`crate::capture`] and [`crate::trace`].
struct Reader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Reader<'a> {
        Reader { data, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.offset
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], ParityError> {
        if self.remaining() < n {
            return Err(ParityError::Truncated {
                what,
                need: n,
                have: self.remaining(),
            });
        }
        let out = &self.data[self.offset..self.offset + n];
        self.offset += n;
        Ok(out)
    }

    fn u8(&mut self, what: &'static str) -> Result<u8, ParityError> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &'static str) -> Result<u16, ParityError> {
        let b = self.take(2, what)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, ParityError> {
        let b = self.take(4, what)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self, what: &'static str) -> Result<i32, ParityError> {
        Ok(self.u32(what)? as i32)
    }

    fn f32(&mut self, what: &'static str) -> Result<f32, ParityError> {
        Ok(f32::from_bits(self.u32(what)?))
    }

    fn f64(&mut self, what: &'static str) -> Result<f64, ParityError> {
        let b = self.take(8, what)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(f64::from_bits(u64::from_be_bytes(a)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_channel_list_matches_what_replay_emits() {
        // Fifteen channels, and the reference and the replay build their rows in this file, so
        // a divergence between the two would otherwise only show up as a shape error at
        // runtime on whichever fixture happened to load first.
        let channels = channels();
        assert_eq!(channels.len(), 15);
        let names: Vec<&str> = channels.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"in_lava"));
        assert!(names.contains(&"fall_damage"));
    }

    #[test]
    fn parse_rejects_a_file_that_is_not_a_physics_fixture() {
        // `PhysicsFixture` holds a `World` and is not `Debug`, so the `Ok` arm is matched
        // explicitly rather than through `expect_err`.
        match parse(b"NOTPHYS!and some more bytes") {
            Err(ParityError::BadMagic { .. }) => {}
            Err(other) => panic!("expected BadMagic, got {other:?}"),
            Ok(_) => panic!("a file that is not a fixture must not parse"),
        }
    }

    #[test]
    fn parse_rejects_a_truncated_header_without_panicking() {
        for cut in 0..PHYSICS_MAGIC.len() + 8 {
            let mut bytes = PHYSICS_MAGIC.to_vec();
            bytes.extend_from_slice(&[0u8; 16]);
            assert!(parse(&bytes[..cut]).is_err(), "{cut}-byte prefix");
        }
    }

    #[test]
    fn attribute_oracle_is_uuid_ordered_sequential_and_clamped_bit_exactly() {
        // UUID 18 is first on the wire, but the Java 8 HashSet visits UUID 1, then UUID 18 and
        // UUID 2 (`java_uuid_hash_set_order` -> [1, 0, 2]). These values make a wire-order
        // mutant differ by one f64 bit rather than disappearing under a tolerance.
        let add_wire = [
            FixtureModifier {
                uuid: 18,
                amount: -100_000_000.0,
                operation: 0,
            },
            FixtureModifier {
                uuid: 1,
                amount: 5.139_593_396_247_941,
                operation: 0,
            },
            FixtureModifier {
                uuid: 2,
                amount: -0.000_001,
                operation: 0,
            },
        ];
        let add_java = ((100_000_000.1_f64 + 5.139_593_396_247_941) - 100_000_000.0) - 0.000_001;
        let add_wire_order =
            ((100_000_000.1_f64 - 100_000_000.0) + 5.139_593_396_247_941) - 0.000_001;
        assert_eq!(
            compute_attribute_value(100_000_000.1, &add_wire)
                .unwrap()
                .to_bits(),
            add_java.to_bits()
        );
        assert_ne!(
            add_java.to_bits(),
            add_wire_order.to_bits(),
            "operation-0 wire-order mutant must bite"
        );

        let multiply_base_wire = [
            FixtureModifier {
                uuid: 18,
                amount: 1_000.0,
                operation: 1,
            },
            FixtureModifier {
                uuid: 1,
                amount: 0.000_01,
                operation: 1,
            },
            FixtureModifier {
                uuid: 2,
                amount: -9.659_309_795_323_193,
                operation: 1,
            },
        ];
        let multiply_base_java =
            ((0.1_f64 + 0.1 * 0.000_01) + 0.1 * 1_000.0) - 0.1 * 9.659_309_795_323_193;
        let multiply_base_wire_order =
            ((0.1_f64 + 0.1 * 1_000.0) + 0.1 * 0.000_01) - 0.1 * 9.659_309_795_323_193;
        assert_eq!(
            compute_attribute_value(0.1, &multiply_base_wire)
                .unwrap()
                .to_bits(),
            multiply_base_java.to_bits()
        );
        assert_ne!(
            multiply_base_java.to_bits(),
            multiply_base_wire_order.to_bits(),
            "operation-1 wire-order mutant must bite"
        );

        let multiply_wire = [
            FixtureModifier {
                uuid: 18,
                amount: 4.004_807_362_210_215,
                operation: 2,
            },
            FixtureModifier {
                uuid: 1,
                amount: 1.625_183_354_820_274_8,
                operation: 2,
            },
            FixtureModifier {
                uuid: 2,
                amount: 3.549_961_541_408_507,
                operation: 2,
            },
        ];
        let multiply_java =
            ((0.1_f64 * 2.625_183_354_820_274_8) * 5.004_807_362_210_215) * 4.549_961_541_408_507;
        let multiply_wire_order =
            ((0.1_f64 * 5.004_807_362_210_215) * 2.625_183_354_820_274_8) * 4.549_961_541_408_507;
        assert_eq!(
            compute_attribute_value(0.1, &multiply_wire)
                .unwrap()
                .to_bits(),
            multiply_java.to_bits()
        );
        assert_ne!(
            multiply_java.to_bits(),
            multiply_wire_order.to_bits(),
            "operation-2 wire-order mutant must bite"
        );

        // The test fixture's old one-modifier oracle returned the raw -0.1. Vanilla clamps the
        // final value to zero, so this negative control demonstrates that omission bit-for-bit.
        let below_minimum = [FixtureModifier {
            uuid: 7,
            amount: -2.0,
            operation: 2,
        }];
        let raw_below_minimum = 0.1_f64 * (1.0 - 2.0);
        assert_eq!(
            compute_attribute_value(0.1, &below_minimum)
                .unwrap()
                .to_bits(),
            0.0_f64.to_bits()
        );
        assert_ne!(
            raw_below_minimum.to_bits(),
            0.0_f64.to_bits(),
            "unclamped lower-bound mutant must bite"
        );

        let above_maximum = [FixtureModifier {
            uuid: 7,
            amount: 0.0,
            operation: 0,
        }];
        assert_eq!(
            compute_attribute_value(2_000.0, &above_maximum)
                .unwrap()
                .to_bits(),
            1024.0_f64.to_bits()
        );
        assert_ne!(
            2_000.0_f64.to_bits(),
            1024.0_f64.to_bits(),
            "unclamped upper-bound mutant must bite"
        );

        assert_eq!(
            compute_attribute_value(
                0.1,
                &[
                    below_minimum[0],
                    FixtureModifier {
                        uuid: 7,
                        amount: 0.3,
                        operation: 2,
                    },
                ],
            ),
            Err(AttributeResolutionError::DuplicateUuid(7))
        );
    }
}
