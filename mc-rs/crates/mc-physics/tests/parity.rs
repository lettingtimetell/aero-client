//! Differential test against the **real** 1.8.9 game.
//!
//! `javaref/PhysicsRefGen.java` runs the recompiled vanilla classes under a JDK, drives a genuine
//! `EntityPlayer` through `Entity.moveEntity`/`EntityLivingBase.moveEntityWithHeading` over a
//! scripted input sequence in a synthetic world, and dumps the per-tick position, motion,
//! `fallDistance` and collision flags as raw IEEE-754 bit patterns. This test replays the same
//! inputs through `mc-physics` and compares.
//!
//! The comparison is on **position and motion to 1e-9**, and on the flags and predicted fall
//! damage exactly. That is the plan's S13 acceptance criterion; in practice the two agree far
//! more tightly than that (see `parity_is_bit_exact_not_merely_within_tolerance`).
//!
//! # What each tick asserts (fixture format `MCPHYS02`)
//!
//! Position, motion, `fallDistance`, seven flag bits and the predicted fall damage were the
//! original set. Wave 1 added, all bit-exact:
//!
//! * `limbSwing`, `limbSwingAmount`, `prevLimbSwingAmount` — GAP `A11-22`, the walk animation.
//! * `distanceWalkedModified`, `distanceWalkedOnStepModified`, `nextStepDistance` and the *name
//!   and volume of the sound `playStepSound` actually played*. The sound is
//!   observed rather than re-derived (`RefPlayer.playSound` records what came out of vanilla's
//!   own `playStepSound` body), so this also checks `mc_data`'s block sound table.
//! * `capabilities.isFlying`, `noClip`, `inPortal` and the trace-only cactus contact callback.
//!
//! The fixture also carries a **loaded-column rect** and a per-tick **event list**. The rect is
//! what `RefWorld` answers `Chunk.isLoaded()` from, so `moveEntityWithHeading`'s unloaded-chunk
//! gravity pin is testable at all. The events are the `NetHandlerPlayClient` handlers
//! that reach physics — `S12PacketEntityVelocity`, `S39PacketPlayerAbilities`,
//! `S1DPacketEntityEffect`, `S20PacketEntityProperties`, `S06PacketUpdateHealth`, the game mode,
//! item use and an enchanted pair of boots — applied at the top of a tick on both sides.
//!
//! # A correction the reference generator itself needed
//!
//! `EntityPlayer.onLivingUpdate` (`entity/player/EntityPlayer.java:619-634`) calls
//! `super.onLivingUpdate()` — which runs `moveEntityWithHeading` — **before** it assigns
//! `jumpMovementFactor`. Earlier revisions of `PhysicsRefGen` hoisted those four lines above the
//! move, so the fixtures and `mc-physics` agreed with each other and not with the game: the
//! sprint air-control boost landed one tick early on the tick you leave the ground. The
//! generator now transcribes the order literally and `PhysicsState::jump_movement_factor` is a
//! stored field that lags by one tick, as vanilla's is.
//!
//! Regenerate with:
//!
//! ```text
//! powershell -ExecutionPolicy Bypass -File crates/mc-physics/javaref/generate.ps1
//! ```
//!
//! The fixtures are committed, so this test needs no JDK and no jars.

use std::path::PathBuf;

use mc_data::BlockState;
use mc_math::{clamp_f64, BlockPos, DVec3};
use mc_physics::player::{mounted_living_travel, mounted_on_living_update_preflight};
use mc_physics::{tick_player, MoveInput, PhysicsState};
use mc_world::World;

/// The plan's acceptance tolerance for S13.
const TOLERANCE: f64 = 1e-9;

const K_FORWARD: u8 = 1;
const K_BACK: u8 = 2;
const K_LEFT: u8 = 4;
const K_RIGHT: u8 = 8;
const K_JUMP: u8 = 16;
const K_SNEAK: u8 = 32;
const K_SPRINT: u8 = 64;

const F_ON_GROUND: u8 = 1;
const F_COLLIDED_H: u8 = 2;
const F_COLLIDED_V: u8 = 4;
const F_IN_WATER: u8 = 8;
const F_ON_LADDER: u8 = 16;
const F_SPRINTING: u8 = 32;
const F_IN_LAVA: u8 = 64;

struct Reader<'a> {
    d: &'a [u8],
    o: usize,
}

impl<'a> Reader<'a> {
    fn u8(&mut self) -> u8 {
        let v = self.d[self.o];
        self.o += 1;
        v
    }
    fn u16(&mut self) -> u16 {
        let v = u16::from_be_bytes(self.d[self.o..self.o + 2].try_into().unwrap());
        self.o += 2;
        v
    }
    fn i32(&mut self) -> i32 {
        let v = i32::from_be_bytes(self.d[self.o..self.o + 4].try_into().unwrap());
        self.o += 4;
        v
    }
    fn u32(&mut self) -> u32 {
        self.i32() as u32
    }
    fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    fn f64(&mut self) -> f64 {
        let v = u64::from_be_bytes(self.d[self.o..self.o + 8].try_into().unwrap());
        self.o += 8;
        f64::from_bits(v)
    }
    fn bytes(&mut self, n: usize) -> &'a [u8] {
        let v = &self.d[self.o..self.o + n];
        self.o += n;
        v
    }
}

const F2_FLYING: u8 = 1;
const F2_NO_CLIP: u8 = 2;
const F2_IN_PORTAL: u8 = 4;
const F2_CACTUS: u8 = 8;

struct Expected {
    pos: DVec3,
    vel: DVec3,
    fall_distance: f32,
    flags: u8,
    fall_damage: i32,
    limb_swing: f32,
    limb_swing_amount: f32,
    prev_limb_swing_amount: f32,
    distance_walked: f32,
    distance_walked_on_step: f32,
    next_step_distance: i32,
    flags2: u8,
    step_sound: Option<(String, f32)>,
    swim_volume: Option<f32>,
}

/// One server-driven change, applied at the start of a tick — the Rust half of
/// `PhysicsRefGen`'s `Event`. See that file's header for the kind numbering.
#[derive(Debug, Clone, Copy)]
struct Event {
    kind: u8,
    a: f64,
    b: f64,
    c: f64,
}

struct Trace {
    name: String,
    world: World,
    start: DVec3,
    start_yaw: f32,
    start_pitch: f32,
    setup: Vec<Event>,
    inputs: Vec<MoveInput>,
    events: Vec<Vec<Event>>,
    expected: Vec<Expected>,
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load(name: &str) -> Trace {
    let path = fixture_dir().join(format!("{name}.bin"));
    let data = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut r = Reader { d: &data, o: 0 };

    assert_eq!(r.bytes(8), b"MCPHYS02", "bad magic in {name}");
    let n = r.u16() as usize;
    let scenario = String::from_utf8(r.bytes(n).to_vec()).expect("utf-8 name");
    assert_eq!(scenario, name);

    let mut world = World::new();
    // The loaded-column rect. `RefWorld` answers `Chunk.isLoaded()` from exactly this rect, so
    // building the Rust world from it is what makes `moveEntityWithHeading`'s gravity pin (and
    // `Session`'s outer gate) testable at all: everything outside is a column the server never
    // sent, which on a client is the shared `EmptyChunk`.
    let (cx0, cz0, cx1, cz1) = (r.i32(), r.i32(), r.i32(), r.i32());
    for cx in cx0..=cx1 {
        for cz in cz0..=cz1 {
            mc_physics::scene::ensure_chunk(&mut world, mc_math::ChunkPos::new(cx, cz));
        }
    }

    // Static fixture terrain is installed in one lighting-free batch. This preserves the
    // encoded fill order and residency while avoiding a flood fill for every block write;
    // fixture light is intentionally not reconstructed because physics does not read it.
    let fills = r.u32();
    let mut static_fills = Vec::with_capacity(fills as usize);
    for _ in 0..fills {
        let x0 = r.i32();
        let y0 = r.i32();
        let z0 = r.i32();
        let x1 = r.i32();
        let y1 = r.i32();
        let z1 = r.i32();
        let state = BlockState::from_raw(r.u16());
        static_fills.push((BlockPos::new(x0, y0, z0), BlockPos::new(x1, y1, z1), state));
    }
    mc_physics::scene::fill_static(&mut world, static_fills);

    let start = DVec3::new(r.f64(), r.f64(), r.f64());
    let start_yaw = r.f32();
    let start_pitch = r.f32();
    let setup = read_events(&mut r);

    let ticks = r.u32() as usize;
    let mut inputs = Vec::with_capacity(ticks);
    let mut events = Vec::with_capacity(ticks);
    let mut expected = Vec::with_capacity(ticks);
    for _ in 0..ticks {
        let keys = r.u8();
        let yaw = r.f32();
        let pitch = r.f32();
        inputs.push(MoveInput {
            forward: keys & K_FORWARD != 0,
            back: keys & K_BACK != 0,
            left: keys & K_LEFT != 0,
            right: keys & K_RIGHT != 0,
            jump: keys & K_JUMP != 0,
            sneak: keys & K_SNEAK != 0,
            sprint: keys & K_SPRINT != 0,
            // Driven by the `usingItem` event, not by the key byte.
            using_item: false,
            yaw,
            pitch,
        });
        events.push(read_events(&mut r));
        let mut e = Expected {
            pos: DVec3::new(r.f64(), r.f64(), r.f64()),
            vel: DVec3::new(r.f64(), r.f64(), r.f64()),
            fall_distance: r.f32(),
            flags: r.u8(),
            fall_damage: r.i32(),
            limb_swing: r.f32(),
            limb_swing_amount: r.f32(),
            prev_limb_swing_amount: r.f32(),
            distance_walked: r.f32(),
            distance_walked_on_step: r.f32(),
            next_step_distance: r.i32(),
            flags2: r.u8(),
            step_sound: None,
            swim_volume: None,
        };
        let sound_len = r.u16() as usize;
        let sound_name = String::from_utf8(r.bytes(sound_len).to_vec()).expect("utf-8 sound");
        let sound_volume = r.f32();
        if !sound_name.is_empty() {
            e.step_sound = Some((sound_name, sound_volume));
        }
        let swim = r.f32();
        if !swim.is_nan() {
            e.swim_volume = Some(swim);
        }
        expected.push(e);
    }
    assert_eq!(r.o, data.len(), "{name}: trailing bytes in fixture");

    Trace {
        name: scenario,
        world,
        start,
        start_yaw,
        start_pitch,
        setup,
        inputs,
        events,
        expected,
    }
}

fn read_events(r: &mut Reader<'_>) -> Vec<Event> {
    let count = r.u32() as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(Event {
            kind: r.u8(),
            a: r.f64(),
            b: r.f64(),
            c: r.f64(),
        });
    }
    out
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
/// Keeping this copy independent from `mc-client` is intentional: this test is the replay
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

/// The Rust half of `PhysicsRefGen.RefPlayer.apply` — the `NetHandlerPlayClient` handlers, as
/// far as they reach physics.
fn apply_event(s: &mut PhysicsState, using_item: &mut bool, e: &Event) {
    match e.kind {
        // handleEntityVelocity -> Entity.setVelocity. It assigns; it does not add.
        0 => s.vel = DVec3::new(e.a, e.b, e.c),
        1 => s.flying = e.a != 0.0,
        2 => s.allow_flying = e.a != 0.0,
        3 => s.fly_speed = e.a as f32,
        // `capabilities.walkSpeed` deliberately has **no** physics effect on a client:
        // `EntityPlayer.onLivingUpdate:622` pushes it into the movementSpeed attribute inside
        // `if (!this.worldObj.isRemote)`. `mc-physics` therefore has no field for it, and the
        // fixture proves the position trace does not move when the server changes it.
        4 => {}
        // `RefPlayer` answers both the tab-list `isSpectator()` and the controller's
        // `isSpectatorMode()` from this one event, as a consistent live client does.
        5 => {
            s.spectator = e.a != 0.0;
            s.spectator_mode = e.a != 0.0;
        }
        6 => *using_item = e.a != 0.0,
        7 => s.movement_blocked = e.a <= 0.0,
        8 => s.food_level = e.a as i32,
        9 => {
            let id = e.a as i32;
            let amplifier = e.b as i32;
            let active = e.c > 0.0;
            match id {
                // Potion.jump. Read through `isPotionActive` in `jump()` and `fall()`, neither of
                // which is `isRemote`-gated, so it really does apply client-side.
                8 => {
                    s.jump_boost = if active {
                        Some(i8::try_from(amplifier).expect("a signed-byte amplifier"))
                    } else {
                        None
                    };
                }
                // Potion.blindness — the sprint-start gate.
                15 => s.blindness = active,
                // Everything else, Speed and Slowness included, reaches movement only through
                // the attribute map, which on a client only `S20PacketEntityProperties` writes.
                _ => {}
            }
        }
        // handleEntityProperties: setBaseValue, removeAllModifiers, re-apply. The removal drops
        // the sprint UUID's entry too, whatever the sprint flag says.
        10 => {
            s.movement_speed = attribute_value(e.a, e.b, e.c as i32);
            s.sprint_modifier_amount = None;
        }
        11 => s.depth_strider = e.a as f32,
        // The S1C echo of watched index 0 bit 3 — the flag alone.
        12 => s.sprinting = e.a != 0.0,
        // An S20 that still carries the canonical sprint UUID entry.
        13 => {
            s.movement_speed = attribute_value(e.a, e.b, e.c as i32);
            s.sprint_modifier_amount = Some(mc_physics::state::SPRINT_SPEED_BOOST);
        }
        other => panic!("unknown fixture event kind {other}"),
    }
}

/// Replays a trace and returns the largest absolute position/motion error seen.
fn replay(t: &Trace, tolerance: f64) -> f64 {
    replay_with(t, tolerance, apply_event)
}

/// [`replay`] with a substitute event handler, for negative controls that model an older
/// client seam against the same game trace.
fn replay_with(
    t: &Trace,
    tolerance: f64,
    apply_event: fn(&mut PhysicsState, &mut bool, &Event),
) -> f64 {
    replay_full(t, tolerance, apply_event, mounted_tick)
}

/// One mounted tick, as the session's riding branch runs it: `Entity.updateRidden`
/// is the preflight plus the rider's own travel, then `updateRiderPosition` seats the rider
/// (`setPosition`) and `EntityLivingBase.updateRidden` clears `fallDistance`.
fn mounted_tick(w: &World, s: &mut PhysicsState, input: &MoveInput, seat: DVec3) {
    mounted_on_living_update_preflight(w, s, input, true);
    let mut outside_border = false;
    mounted_living_travel(w, s, &[], None, &mut outside_border);
    s.set_position(seat);
    s.fall_distance = 0.0;
}

/// [`replay_with`] with a substitute mounted tick as well. Fixture event kinds 14 (mount at the
/// seat `a, b, c`) and 15 (dismount) select which tick body runs, as `PhysicsRefGen.run` does.
fn replay_full(
    t: &Trace,
    tolerance: f64,
    apply_event: fn(&mut PhysicsState, &mut bool, &Event),
    mounted_tick: fn(&World, &mut PhysicsState, &MoveInput, DVec3),
) -> f64 {
    let mut s = PhysicsState::new_player(t.start);
    s.yaw = t.start_yaw;
    s.pitch = t.start_pitch;

    let mut worst = 0.0f64;
    let mut using_item = false;
    let mut riding: Option<DVec3> = None;
    fn apply(
        s: &mut PhysicsState,
        using_item: &mut bool,
        riding: &mut Option<DVec3>,
        e: &Event,
        apply_event: fn(&mut PhysicsState, &mut bool, &Event),
    ) {
        match e.kind {
            14 => *riding = Some(DVec3::new(e.a, e.b, e.c)),
            15 => *riding = None,
            _ => apply_event(s, using_item, e),
        }
    }
    for e in &t.setup {
        apply(&mut s, &mut using_item, &mut riding, e, apply_event);
    }

    for (i, (input, want)) in t.inputs.iter().zip(&t.expected).enumerate() {
        for e in &t.events[i] {
            apply(&mut s, &mut using_item, &mut riding, e, apply_event);
        }
        let input = &MoveInput {
            using_item,
            ..*input
        };
        match riding {
            Some(seat) => mounted_tick(&t.world, &mut s, input, seat),
            None => tick_player(&t.world, &mut s, input),
        }

        let deltas = [
            ("pos.x", s.pos.x, want.pos.x),
            ("pos.y", s.pos.y, want.pos.y),
            ("pos.z", s.pos.z, want.pos.z),
            ("vel.x", s.vel.x, want.vel.x),
            ("vel.y", s.vel.y, want.vel.y),
            ("vel.z", s.vel.z, want.vel.z),
        ];
        for (label, got, expect) in deltas {
            let err = (got - expect).abs();
            if err > worst {
                worst = err;
            }
            assert!(
                err <= tolerance,
                "{}: tick {i} {label}: rust {got:.17} vs java {expect:.17} (err {err:e})\n\
                 input {input:?}\n\
                 rust pos {:?} vel {:?}\n\
                 java pos {:?} vel {:?}",
                t.name,
                s.pos,
                s.vel,
                want.pos,
                want.vel,
            );
        }

        assert_eq!(
            s.fall_distance, want.fall_distance,
            "{}: tick {i}: fallDistance",
            t.name
        );

        fn bit(set: bool, flag: u8) -> u8 {
            if set {
                flag
            } else {
                0
            }
        }
        let got_flags = bit(s.on_ground, F_ON_GROUND)
            | bit(s.collided_horizontally, F_COLLIDED_H)
            | bit(s.collided_vertically, F_COLLIDED_V)
            | bit(s.in_water, F_IN_WATER)
            | bit(mc_physics::player::is_on_ladder(&t.world, &s), F_ON_LADDER)
            | bit(s.sprinting, F_SPRINTING)
            | bit(mc_physics::fluid::is_in_lava(&t.world, &s), F_IN_LAVA);
        assert_eq!(
            got_flags, want.flags,
            "{}: tick {i}: flags (rust {got_flags:#08b} java {:#08b})",
            t.name, want.flags
        );

        let want_damage = if want.fall_damage < 0 {
            None
        } else {
            Some(want.fall_damage as u32)
        };
        assert_eq!(
            s.predicted_fall_damage, want_damage,
            "{}: tick {i}: predicted fall damage",
            t.name
        );

        // The walk-animation driver, bit-exact. `mc-render`'s `AnimParams` reads these.
        for (label, got, expect) in [
            ("limbSwing", s.limb_swing, want.limb_swing),
            (
                "limbSwingAmount",
                s.limb_swing_amount,
                want.limb_swing_amount,
            ),
            (
                "prevLimbSwingAmount",
                s.prev_limb_swing_amount,
                want.prev_limb_swing_amount,
            ),
            // The walk accumulators and the step-sound counter.
            (
                "distanceWalkedModified",
                s.distance_walked_modified,
                want.distance_walked,
            ),
            (
                "distanceWalkedOnStepModified",
                s.distance_walked_on_step_modified,
                want.distance_walked_on_step,
            ),
        ] {
            assert_eq!(got, expect, "{}: tick {i}: {label}", t.name);
        }
        assert_eq!(
            s.next_step_distance, want.next_step_distance,
            "{}: tick {i}: nextStepDistance",
            t.name
        );

        let got_flags2 = bit(s.flying, F2_FLYING)
            | bit(s.no_clip, F2_NO_CLIP)
            | bit(s.in_portal, F2_IN_PORTAL)
            | bit(s.touched_cactus, F2_CACTUS);
        assert_eq!(
            got_flags2, want.flags2,
            "{}: tick {i}: flags2 (rust {got_flags2:#08b} java {:#08b})",
            t.name, want.flags2
        );

        // A11-13's sound half. `mc_data`'s block table supplies the family name and the volume;
        // `Block.SoundType.getStepSound()` is `"step." + soundName` and `playStepSound` scales
        // the volume by `0.15F`. Comparing against what the *game* played also checks that
        // table, and the snow-layer substitution with it.
        let got_step = s.step_sound.map(|step| (step.sound_name(), step.volume()));
        match (&got_step, &want.step_sound) {
            (Some((got_name, got_vol)), Some((want_name, want_vol))) => {
                assert_eq!(got_name, want_name, "{}: tick {i}: step sound name", t.name);
                assert_eq!(got_vol, want_vol, "{}: tick {i}: step sound volume", t.name);
            }
            (None, None) => {}
            _ => panic!(
                "{}: tick {i}: step sound rust {got_step:?} java {:?}",
                t.name, want.step_sound
            ),
        }
        assert_eq!(
            s.swim_sound_volume, want.swim_volume,
            "{}: tick {i}: swim sound volume",
            t.name
        );
    }

    worst
}

macro_rules! parity {
    ($($fn_name:ident => $scenario:literal),+ $(,)?) => {
        $(
            #[test]
            fn $fn_name() {
                let t = load($scenario);
                let worst = replay(&t, TOLERANCE);
                println!("{}: {} ticks, worst error {worst:e}", t.name, t.inputs.len());
            }
        )+
    };
}

/// Every fixture in the directory, in generation order. The `parity!` macro below turns each
/// into its own `#[test]`, and `parity_is_bit_exact_not_merely_within_tolerance` re-runs the lot
/// at tolerance `0.0`.
const SCENARIOS: &[&str] = &[
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
    // GAP A11 — the branches the original twelve deliberately avoided.
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
    // The sprint flag (S1C) and the sprint attribute entry (S20) are separate state.
    "sprint_flag_echo",
    "sprint_attribute_echo",
    "sprint_expiry",
    // moveEntity edge cases from `javaref/MoveEdgeRefGen.java`.
    "l3_decor_hop",
    "l3_decor_walk",
    "l3_stair_pyramid",
    "l3_sneak_fence",
    "l3_sneak_cobblestone_wall",
    "l3_sneak_glass_pane",
    "l3_sneak_iron_bars",
    "l3_sneak_stone_slab",
    "l3_sneak_carpet",
    "l3_sneak_bed",
    "l3_sneak_trapdoor",
    "l3_sneak_end_portal_frame",
    "l3_sneak_hopper",
    "l3_sneak_skull",
    "l3_sneak_daylight_detector",
    "l3_slime_bounce",
    "l3_slime_sneak_land",
    "l3_ice_quirks",
    "l3_web_fall",
    "l3_ladder_climb_out",
    "l3_ladder_gap",
    "l3_vine_climb",
    "l3_water_flow",
    "l3_waterfall",
    "l3_water_ledge",
    "l3_lava_flow",
    "l3_head_hitters",
    "l3_decor_arena_0",
    "l3_decor_arena_1",
    "l3_decor_arena_2",
    // Riding (event kinds 14/15), flight timing, potions and combined block effects
    // from `javaref/H2EdgeRefGen.java`.
    "h2_ride_wall",
    "h2_ride_water_flight",
    "h2_ride_midair",
    "h2_soul_sand_ice",
    "h2_web_water",
    "h2_ladder_water",
    "h2_sneak_stairs",
    "h2_sneak_stairs_top",
    "h2_depth_strider_flow",
    "h2_potion_mix",
    "h2_fall_ladder",
    "h2_fall_water",
    "h2_fall_lava",
    "h2_fall_hay",
    "h2_fly_toggle",
    // Random arenas mixing liquids, webs, ladders, slippery and shaped blocks
    // under random server events, plus targeted combinations, from `javaref/ComboRefGen.java`.
    "w19_combo_arena_0",
    "w19_combo_arena_1",
    "w19_combo_arena_2",
    "w19_combo_arena_3",
    "w19_ice_head_hitter",
    "w19_sneak_edge_knockback_slab",
    "w19_sneak_edge_knockback_stair",
    "w19_sneak_edge_knockback_fence",
];

parity! {
    flat_walk => "flat_walk",
    wall_collide => "wall_collide",
    step_up => "step_up",
    sneak_edge => "sneak_edge",
    fall_damage => "fall_damage",
    friction => "friction",
    water_swim => "water_swim",
    lava_swim => "lava_swim",
    ladder_climb => "ladder_climb",
    cobweb => "cobweb",
    connected_blocks => "connected_blocks",
    long_random => "long_random",
    creative_fly => "creative_fly",
    speed_attribute => "speed_attribute",
    knockback => "knockback",
    jump_boost => "jump_boost",
    item_use => "item_use",
    depth_strider => "depth_strider",
    spectator_noclip => "spectator_noclip",
    dead_player => "dead_player",
    unloaded_chunk => "unloaded_chunk",
    contact_blocks => "contact_blocks",
    sprint_flag_echo => "sprint_flag_echo",
    sprint_attribute_echo => "sprint_attribute_echo",
    sprint_expiry => "sprint_expiry",
    l3_decor_hop => "l3_decor_hop",
    l3_decor_walk => "l3_decor_walk",
    l3_stair_pyramid => "l3_stair_pyramid",
    l3_sneak_fence => "l3_sneak_fence",
    l3_sneak_cobblestone_wall => "l3_sneak_cobblestone_wall",
    l3_sneak_glass_pane => "l3_sneak_glass_pane",
    l3_sneak_iron_bars => "l3_sneak_iron_bars",
    l3_sneak_stone_slab => "l3_sneak_stone_slab",
    l3_sneak_carpet => "l3_sneak_carpet",
    l3_sneak_bed => "l3_sneak_bed",
    l3_sneak_trapdoor => "l3_sneak_trapdoor",
    l3_sneak_end_portal_frame => "l3_sneak_end_portal_frame",
    l3_sneak_hopper => "l3_sneak_hopper",
    l3_sneak_skull => "l3_sneak_skull",
    l3_sneak_daylight_detector => "l3_sneak_daylight_detector",
    l3_slime_bounce => "l3_slime_bounce",
    l3_slime_sneak_land => "l3_slime_sneak_land",
    l3_ice_quirks => "l3_ice_quirks",
    l3_web_fall => "l3_web_fall",
    l3_ladder_climb_out => "l3_ladder_climb_out",
    l3_ladder_gap => "l3_ladder_gap",
    l3_vine_climb => "l3_vine_climb",
    l3_water_flow => "l3_water_flow",
    l3_waterfall => "l3_waterfall",
    l3_water_ledge => "l3_water_ledge",
    l3_lava_flow => "l3_lava_flow",
    l3_head_hitters => "l3_head_hitters",
    l3_decor_arena_0 => "l3_decor_arena_0",
    l3_decor_arena_1 => "l3_decor_arena_1",
    l3_decor_arena_2 => "l3_decor_arena_2",
    h2_ride_wall => "h2_ride_wall",
    h2_ride_water_flight => "h2_ride_water_flight",
    h2_ride_midair => "h2_ride_midair",
    h2_soul_sand_ice => "h2_soul_sand_ice",
    h2_web_water => "h2_web_water",
    h2_ladder_water => "h2_ladder_water",
    h2_sneak_stairs => "h2_sneak_stairs",
    h2_sneak_stairs_top => "h2_sneak_stairs_top",
    h2_depth_strider_flow => "h2_depth_strider_flow",
    h2_potion_mix => "h2_potion_mix",
    h2_fall_ladder => "h2_fall_ladder",
    h2_fall_water => "h2_fall_water",
    h2_fall_lava => "h2_fall_lava",
    h2_fall_hay => "h2_fall_hay",
    h2_fly_toggle => "h2_fly_toggle",
    w19_combo_arena_0 => "w19_combo_arena_0",
    w19_combo_arena_1 => "w19_combo_arena_1",
    w19_combo_arena_2 => "w19_combo_arena_2",
    w19_combo_arena_3 => "w19_combo_arena_3",
    w19_ice_head_hitter => "w19_ice_head_hitter",
    w19_sneak_edge_knockback_slab => "w19_sneak_edge_knockback_slab",
    w19_sneak_edge_knockback_stair => "w19_sneak_edge_knockback_stair",
    w19_sneak_edge_knockback_fence => "w19_sneak_edge_knockback_fence",
}

/// The plan's "done when": 1,000 ticks, within 1e-9.
#[test]
fn thousand_tick_replay_matches_java() {
    let t = load("long_random");
    assert_eq!(t.inputs.len(), 1000);
    replay(&t, TOLERANCE);
}

/// 1e-9 is the contract, but the two implementations should actually be *identical* — every
/// arithmetic step is the same operation on the same IEEE-754 values, in the same order. If
/// this test ever starts failing while the 1e-9 ones still pass, something has been "cleaned
/// up" into an algebraically-equal-but-differently-rounded form; find it before it drifts.
#[test]
fn parity_is_bit_exact_not_merely_within_tolerance() {
    for name in SCENARIOS {
        let t = load(name);
        let worst = replay(&t, 0.0);
        assert_eq!(worst, 0.0, "{name}");
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
    let add_wire_order = ((100_000_000.1_f64 - 100_000_000.0) + 5.139_593_396_247_941) - 0.000_001;
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

    // The old one-modifier oracle returned the raw -0.1. Vanilla clamps the final value to zero,
    // so this negative control demonstrates that omission bit-for-bit.
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

/// The fixtures are only meaningful if the scenarios actually exercise what they claim. This
/// guards against a scene edit quietly turning a test into "the player stands still for 300
/// ticks and matches perfectly".
#[test]
fn the_scenarios_exercise_what_they_claim() {
    /// (scenario, predicate over its expected trace, what the predicate is checking).
    type Guard = (&'static str, fn(&[Expected]) -> bool, &'static str);

    let assertions: &[Guard] = &[
        (
            "flat_walk",
            |e| e.iter().any(|x| x.flags & F_SPRINTING != 0) && e.iter().any(|x| x.pos.y > 64.5),
            "sprints and jumps",
        ),
        (
            "wall_collide",
            |e| e.iter().filter(|x| x.flags & F_COLLIDED_H != 0).count() > 50,
            "spends time against a wall",
        ),
        (
            "step_up",
            |e| e.iter().any(|x| x.pos.y == 64.5) && e.iter().any(|x| x.flags & F_COLLIDED_H != 0),
            "steps onto a slab and is blocked by a full block",
        ),
        (
            "sneak_edge",
            |e| e.iter().any(|x| x.pos.z > 13.0) && e.iter().any(|x| x.pos.y < 60.0),
            "hangs over the ledge, then falls once sneak is released",
        ),
        (
            "fall_damage",
            |e| e.iter().any(|x| x.fall_damage > 15) && e.iter().any(|x| x.fall_distance > 20.0),
            "takes a big fall",
        ),
        (
            "friction",
            // Sprint velocity settles at ~0.153 on stone and ~0.241 on ice, and soul sand
            // holds it near 0.012. All three bands have to appear.
            |e| {
                e.iter().any(|x| x.vel.z > 0.23)
                    && e.iter().any(|x| (0.15..0.16).contains(&x.vel.z))
                    && e.iter()
                        .any(|x| (0.0..0.02).contains(&x.vel.z) && x.pos.z > 23.0)
            },
            "covers ice, stone and soul sand speeds",
        ),
        (
            "water_swim",
            |e| e.iter().filter(|x| x.flags & F_IN_WATER != 0).count() > 100,
            "spends most of the run in water",
        ),
        (
            "lava_swim",
            |e| e.iter().filter(|x| x.flags & F_IN_LAVA != 0).count() > 100,
            "spends most of the run in lava",
        ),
        (
            "ladder_climb",
            |e| {
                e.iter().filter(|x| x.flags & F_ON_LADDER != 0).count() > 100
                    && e.iter().any(|x| x.pos.y > 72.0)
            },
            "climbs a long way up a ladder, then a vine",
        ),
        (
            "cobweb",
            // Inside the web the delta is a quarter of a already-zeroed motion, so progress
            // is a crawl: dozens of consecutive ticks moving less than 0.05 blocks.
            |e| {
                e.windows(2)
                    .filter(|w| {
                        (8.0..14.0).contains(&w[1].pos.z) && (w[1].pos.z - w[0].pos.z).abs() < 0.05
                    })
                    .count()
                    > 40
            },
            "crawls through the web instead of walking",
        ),
        (
            "connected_blocks",
            // Where the player stops is a direct readout of each neighbour-aware shape. With a
            // half-width of 0.30000001192092896, a stop at pos.z = X means the box front face
            // is at X + 0.3, i.e. the block-local inset of that barrier:
            //   8.375  fence      (0.375  post/bar inset)
            //   14.3125 wall      (0.3125 straight-run narrowing)
            //   20.4375 glass pane(0.4375 thin axis)
            //   26.4375 iron bars (0.4375, same shape, different block)
            //   32.8125 door      (1 - 0.1875, the leaf on the far side of its block)
            // If any of these regress, the wrong shape is being computed for that block.
            |e| {
                [8.075, 14.0125, 20.1375, 26.1375, 32.5125]
                    .iter()
                    .all(|want| {
                        e.iter()
                            .any(|x| x.flags & F_COLLIDED_H != 0 && (x.pos.z - want).abs() < 1e-6)
                    })
            },
            "is stopped at the exact face of a fence, wall, pane, bars and door",
        ),
        (
            "long_random",
            |e| {
                let count = |f: u8| e.iter().filter(|x| x.flags & f != 0).count();
                count(F_IN_WATER) > 100
                    && count(F_SPRINTING) > 100
                    && count(F_COLLIDED_H) > 20
                    && e.iter().filter(|x| x.flags & F_ON_GROUND == 0).count() > 100
                    && e.iter().any(|x| x.fall_damage > 0)
                    && e.iter().any(|x| x.pos.y > 64.4)
            },
            "spends real time in water, sprinting, airborne and against walls",
        ),
        // ---- GAP A11 ----------------------------------------------------------------------
        // Thresholds are read off the generated traces, not guessed: the recorded `vel` is the
        // motion *after* the tick's drag multiply, so it sits below the per-tick displacement.
        (
            "creative_fly",
            |e| {
                let flying = e.iter().filter(|x| x.flags2 & F2_FLYING != 0).count();
                flying > 150
                    && e.iter().any(|x| x.pos.y > 80.0)
                    && e.iter()
                        .any(|x| x.flags2 & F2_FLYING != 0 && x.flags & F_SPRINTING != 0)
                    // The landing clears `isFlying` again, and walking resumes.
                    && e.last().is_some_and(|x| x.flags2 & F2_FLYING == 0)
                    // `EntityPlayer.fall` is a no-op while `allowFlying`, however far you drop.
                    && !e.iter().any(|x| x.fall_damage > 0)
                    // While flying, `EntityPlayer.canTriggerWalking()` is false, so not one
                    // block of walked distance accumulates however far you travel.
                    && e.iter()
                        .filter(|x| x.flags2 & F2_FLYING != 0)
                        .all(|x| x.distance_walked == 0.0)
            },
            "toggles flight, climbs, sprint-flies, lands and clears the flag",
        ),
        (
            "speed_attribute",
            // Four speed bands: 0.1179 base, 0.1650 under Speed II, 0.2145 sprinting under it,
            // and 0.0825 under Slowness II. A `walkSpeed` of 0.3 lands between the first two and
            // must change nothing. The last hostile modifier computes negative before
            // RangedAttribute clamps it, so existing momentum decays without reversing.
            |e| {
                (e[35].vel.z - 0.117_859_048_5).abs() < 1e-6
                    && (e[75].vel.z - e[35].vel.z).abs() < 1e-9
                    && e[135].vel.z > 0.16
                    && e[195].vel.z > 0.21
                    && e[265].vel.z < 0.09
                    && e[309].vel.z >= 0.0
                    && e[309].vel.z < 1e-8
            },
            "is unmoved by walkSpeed and moved by the movementSpeed attribute",
        ),
        (
            "knockback",
            // Four assignments, each visible as a velocity the input could not have produced.
            |e| {
                e.iter().any(|x| x.vel.y > 0.7)
                    && e.iter().any(|x| x.vel.x < -0.25)
                    && e.iter().any(|x| x.vel.x > 0.3)
            },
            "is shoved around by velocity packets",
        ),
        (
            "jump_boost",
            // `(0.42 + (amplifier + 1) * 0.1 - 0.08) * 0.98 = 0.5292` on the jump tick, and the
            // 18.77-block fall costs `ceil(18.77 - 3 - 2) = 14` rather than 16.
            |e| {
                e.iter().any(|x| x.vel.y > 0.52)
                    && e.iter().any(|x| x.fall_distance > 18.0)
                    && e.iter().any(|x| x.fall_damage == 14)
            },
            "jumps higher and takes reduced fall damage",
        ),
        (
            "item_use",
            // Sprinting, then 20% input while using an item, then sprinting again, then blind
            // and unable to start one.
            |e| {
                e.iter().filter(|x| x.flags & F_SPRINTING != 0).count() > 60
                    && e[45..100]
                        .iter()
                        .all(|x| x.flags & F_SPRINTING == 0 && x.vel.z < 0.03)
                    && e[160..200].iter().all(|x| x.flags & F_SPRINTING == 0)
            },
            "loses sprint while using an item and cannot re-sprint while blind",
        ),
        (
            "depth_strider",
            // The swim drag rises from 0.8 toward 0.546 and the acceleration toward the walk
            // speed, so the steady-state swim speed goes 0.0784 -> 0.1179.
            |e| {
                e.iter().filter(|x| x.flags & F_IN_WATER != 0).count() > 150
                    && e[100..120].iter().all(|x| x.vel.z < 0.09)
                    && e[180..200].iter().any(|x| x.vel.z > 0.11)
            },
            "swims faster once Depth Strider III is on the boots",
        ),
        (
            "spectator_noclip",
            |e| {
                let noclip: Vec<_> = e.iter().filter(|x| x.flags2 & F2_NO_CLIP != 0).collect();
                noclip.len() > 200
                    // `EntityPlayer.onUpdate` forces `onGround = false` every tick, and the
                    // `noClip` branch of `moveEntity` never sets it again.
                    && noclip.iter().all(|x| x.flags & F_ON_GROUND == 0)
                    // Straight through 17 blocks of solid rock and out the far side.
                    && e.iter().any(|x| x.pos.z > 30.0)
                    // The ladder in the wall stops counting for a spectator.
                    && noclip.iter().all(|x| x.flags & F_ON_LADDER == 0)
                    // And the walked distance stops accumulating: `distanceWalkedModified` is
                    // cumulative, so it keeps the 30 ticks of walking that came first, but
                    // `moveEntity`'s `noClip` branch returns before the block that adds to it.
                    && noclip
                        .first()
                        .zip(noclip.last())
                        .is_some_and(|(a, b)| a.distance_walked == b.distance_walked)
            },
            "passes through solid rock without ground, ladder or footsteps",
        ),
        (
            "dead_player",
            |e| {
                // Sprinting before the death, then stopped dead with the input ignored, then
                // moving again after the respawn.
                e[..60].iter().any(|x| x.vel.z > 0.18)
                    && e[130..140].iter().all(|x| x.vel.z == 0.0)
                    && e[130..140].iter().all(|x| x.pos.y == 64.0)
                    && e[170..].iter().any(|x| x.vel.z > 0.1)
            },
            "stops moving while dead and resumes after the respawn",
        ),
        (
            "unloaded_chunk",
            // `-0.1 * 0.98` every tick instead of accelerating: the sink rate never passes 0.1.
            |e| {
                e.iter().any(|x| x.flags & F_ON_GROUND != 0)
                    && e[120..]
                        .iter()
                        .all(|x| x.vel.y == -0.1 * 0.980_000_019_073_486_3)
                    && e.last().is_some_and(|x| x.pos.y < 55.0)
            },
            "sinks at the pinned rate once it leaves the loaded column",
        ),
        (
            "contact_blocks",
            |e| {
                e.iter().any(|x| x.flags2 & F2_CACTUS != 0)
                    && e.iter().any(|x| x.flags2 & F2_IN_PORTAL != 0)
                    && e.iter()
                        .any(|x| x.step_sound.as_ref().is_some_and(|s| s.0 == "step.snow"))
                    && e.iter()
                        .any(|x| x.step_sound.as_ref().is_some_and(|s| s.0 == "step.stone"))
                    && e.iter().any(|x| x.next_step_distance > 4)
            },
            "walks over snow, through a portal and into a cactus",
        ),
        // ---- The sprint flag and the sprint attribute entry --------------------
        // Bands read off the generated traces: steady stone sprint 0.153217, walk 0.117859.
        (
            "sprint_flag_echo",
            |e| {
                let sprint = |x: &Expected| x.flags & F_SPRINTING != 0;
                // Echoed false: the flag drops, the boost does not.
                !sprint(&e[38]) && (e[38].vel.z - 0.153_217).abs() < 1e-5
                    // ...and the jump taken with the flag cleared gets no +0.2 sprint kick.
                    && e[40].vel.z < 0.16
                    // Echoed true on a walker: the flag rises, the boost does not...
                    && sprint(&e[99]) && (e[99].vel.z - 0.117_857).abs() < 1e-5
                    // ...but the jump does get the sprint kick.
                    && e[100].vel.z > 0.22
            },
            "keeps speed on the attribute entry and the jump kick on the flag",
        ),
        (
            "sprint_attribute_echo",
            |e| {
                let sprint = |x: &Expected| x.flags & F_SPRINTING != 0;
                // S20 without the entry while sprinting: still flagged, walking speed.
                sprint(&e[39]) && (e[39].vel.z - 0.117_859).abs() < 1e-5
                    // S20 with the entry on a non-sprinter (+ Speed II): sprint-under-Speed band.
                    && !sprint(&e[129]) && (e[129].vel.z - 0.214_503).abs() < 1e-5
                    && (e[159].vel.z - 0.165_003).abs() < 1e-5
            },
            "drops and restores the boost from S20 without touching the flag",
        ),
        (
            "sprint_expiry",
            |e| {
                let sprint = |x: &Expected| x.flags & F_SPRINTING != 0;
                // The echoes at 200/210 neither stop the sprint speed nor restart the countdown.
                (e[205].vel.z - 0.153_217).abs() < 1e-5
                    && sprint(&e[611])
                    && !sprint(&e[612])
                    && (e[649].vel.z - 0.117_859).abs() < 1e-5
            },
            "ends a double-tap sprint 600 ticks after setSprinting(true)",
        ),
    ];

    for (name, pred, what) in assertions {
        let t = load(name);
        assert!(pred(&t.expected), "{name} no longer {what}");
    }
}

/// Negative control: the client model before this change fed every S1C sprint-flag
/// change into physics as a full `setSprinting` (flag, attribute entry *and* the 600-tick
/// countdown). Replaying the new oracle traces with that model must diverge from the game,
/// which is what makes `sprint_flag_echo`/`sprint_expiry` a real guard rather than a tautology.
#[test]
fn the_old_flag_echo_model_is_rejected_by_the_oracle() {
    fn old_model(s: &mut PhysicsState, using_item: &mut bool, e: &Event) {
        if e.kind == 12 {
            let sprinting = e.a != 0.0;
            s.sprinting = sprinting;
            s.sprinting_ticks_left = if sprinting { 600 } else { 0 };
            s.sprint_modifier_amount = sprinting.then_some(mc_physics::state::SPRINT_SPEED_BOOST);
        } else {
            apply_event(s, using_item, e);
        }
    }
    for name in ["sprint_flag_echo", "sprint_attribute_echo", "sprint_expiry"] {
        let t = load(name);
        // The trace is only read; nothing observes it after a panic.
        let replay_old = std::panic::AssertUnwindSafe(|| replay_with(&t, TOLERANCE, old_model));
        let diverged = std::panic::catch_unwind(replay_old).is_err();
        assert!(
            diverged,
            "{name}: the pre-wave flag-echo model must not match the game"
        );
    }
}

/// The arena-index rule: no `Rc`/`RefCell` anywhere in this crate. `Send + Sync` is the type-level
/// half of that; the source grep in the same test catches wrappers `Send + Sync` would miss.
#[test]
fn no_interior_mutability_in_the_physics_state() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PhysicsState>();
    assert_send_sync::<MoveInput>();
    assert_send_sync::<mc_physics::Predictor>();

    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src");
    for entry in std::fs::read_dir(&src).expect("src/ is readable") {
        let path = entry.expect("a readable dir entry").path();
        if path.extension().is_some_and(|e| e == "rs") {
            let text = std::fs::read_to_string(&path).expect("a readable source file");
            for forbidden in ["Rc<", "RefCell", "Cell<", "Mutex", "RwLock"] {
                assert!(
                    !text.contains(forbidden),
                    "{} contains `{forbidden}` — the arena rule forbids interior mutability in \
                     mc-physics",
                    path.display()
                );
            }
        }
    }
}

/// The frozen three-argument `move_entity` still resolves a plain sweep, independently of
/// `tick_player`.
#[test]
fn frozen_move_entity_signature_works_standalone() {
    let t = load("flat_walk");
    let mut bb = mc_math::Aabb::from_bounds(8.2, 70.0, 8.2, 8.8, 71.8, 8.8);
    let (applied, on_ground) =
        mc_physics::move_entity(&t.world, &mut bb, DVec3::new(0.0, -8.0, 0.0));
    assert!(on_ground);
    assert_eq!(bb.min.y, 64.0);
    assert_eq!(applied, DVec3::new(0.0, -6.0, 0.0));
}

/// The `MoveEdgeRefGen` traces really exercise the edge they are named after, so a
/// regenerated fixture with broken geometry cannot pass vacuously.
#[test]
fn the_l3_edge_scenarios_exercise_what_they_claim() {
    type Guard = (&'static str, fn(&[Expected]) -> bool, &'static str);
    fn ground(x: &Expected) -> bool {
        x.flags & F_ON_GROUND != 0
    }
    fn flag(x: &Expected, f: u8) -> bool {
        x.flags & f != 0
    }
    // Every sneak scenario lands at tick 24, sneaks to every edge and corner until tick 404,
    // then steps off unsneaked: held on top throughout, off it by the end.
    fn held_then_dropped(e: &[Expected]) -> bool {
        let top = e[24].pos.y;
        e[25..=404].iter().all(|x| x.pos.y > top - 0.6)
            && e[25..=404].iter().filter(|x| ground(x)).count() > 300
            && e[456].pos.y < top - 1.0
    }
    let assertions: &[Guard] = &[
        (
            "l3_decor_hop",
            |e| e.iter().any(|x| x.pos.z > 100.0),
            "hops the whole course",
        ),
        (
            "l3_decor_walk",
            |e| e.iter().any(|x| flag(x, F_COLLIDED_H)) && e.iter().any(|x| x.pos.y >= 64.5),
            "steps up decor and stops at the tall blocks",
        ),
        (
            "l3_stair_pyramid",
            |e| e.iter().any(|x| x.pos.y >= 65.0),
            "climbs the stairs",
        ),
        (
            "l3_sneak_fence",
            held_then_dropped,
            "is held on the fence top",
        ),
        (
            "l3_sneak_cobblestone_wall",
            held_then_dropped,
            "is held on the wall top",
        ),
        (
            "l3_sneak_glass_pane",
            held_then_dropped,
            "is held on the pane top",
        ),
        (
            "l3_sneak_iron_bars",
            held_then_dropped,
            "is held on the bars",
        ),
        (
            "l3_sneak_stone_slab",
            held_then_dropped,
            "is held on the slabs",
        ),
        (
            "l3_sneak_carpet",
            held_then_dropped,
            "is held on the carpet",
        ),
        ("l3_sneak_bed", held_then_dropped, "is held on the beds"),
        (
            "l3_sneak_trapdoor",
            held_then_dropped,
            "is held on the trapdoors",
        ),
        (
            "l3_sneak_end_portal_frame",
            held_then_dropped,
            "is held on the frames",
        ),
        (
            "l3_sneak_hopper",
            held_then_dropped,
            "is held on the hoppers",
        ),
        (
            "l3_sneak_skull",
            // Vanilla quirk, reproduced: the skulls sit 0.5 above the pillar the guard also
            // probes, so the player steps down onto the pillar, is airborne for a tick
            // (`flag = onGround && isSneaking()` is false), and walks off the side unguarded.
            |e| {
                e[25..=110].iter().all(|x| x.pos.y == e[24].pos.y)
                    && e.iter().any(|x| x.pos.y < 62.0)
            },
            "is held on the skulls until a 0.5 step-down drops the guard",
        ),
        (
            "l3_sneak_daylight_detector",
            held_then_dropped,
            "is held on the sensors",
        ),
        (
            "l3_slime_bounce",
            |e| e[..60].iter().any(|x| x.vel.y > 0.5),
            "bounces off slime",
        ),
        (
            "l3_slime_sneak_land",
            |e| e[..90].iter().all(|x| x.vel.y <= 0.0) && e[..90].iter().any(ground),
            "lands on slime without a bounce while sneaking",
        ),
        (
            "l3_ice_quirks",
            |e| {
                e[..150].iter().any(|x| x.vel.z > 0.4)
                    && e.iter().any(|x| x.pos.y == 63.0625 && ground(x))
                    && e.iter().any(|x| x.pos.y == 63.5 && ground(x))
            },
            "sprint-jumps on ice and stands on carpet and slab over ice",
        ),
        (
            "l3_web_fall",
            |e| {
                e.iter().any(|x| x.fall_distance > 10.0)
                    && e.iter().any(|x| {
                        x.pos.y > 64.5 && x.pos.y < 70.0 && x.fall_distance == 0.0 && !ground(x)
                    })
            },
            "is caught mid-fall by the web",
        ),
        (
            "l3_ladder_climb_out",
            |e| e.iter().any(|x| x.pos.y == 71.0) && e.iter().any(|x| flag(x, F_ON_LADDER)),
            "climbs out onto the ledge",
        ),
        (
            "l3_ladder_gap",
            |e| e[40..130].iter().all(|x| x.pos.y < 67.7) && e.iter().any(|x| flag(x, F_ON_LADDER)),
            "cannot climb past the gap",
        ),
        (
            "l3_vine_climb",
            |e| e.iter().any(|x| x.pos.y == 71.0) && e.iter().any(|x| flag(x, F_ON_LADDER)),
            "climbs vines to the ledge",
        ),
        (
            "l3_water_flow",
            |e| e[59].pos.z - e[0].pos.z > 2.0,
            "is carried by the current",
        ),
        (
            "l3_waterfall",
            |e| {
                e[40..230]
                    .iter()
                    .any(|x| x.pos.y > 65.0 && flag(x, F_IN_WATER))
            },
            "swims up the waterfall",
        ),
        (
            "l3_water_ledge",
            |e| {
                e.iter().any(|x| x.vel.y == 0.300_000_011_920_928_96)
                    && e.iter().any(|x| x.pos.y == 63.0625 && ground(x))
            },
            "gets the 0.3 exit kick and swims out onto the carpet rim",
        ),
        (
            "l3_lava_flow",
            |e| e.iter().any(|x| flag(x, F_IN_LAVA)),
            "swims in lava",
        ),
        (
            "l3_head_hitters",
            |e| e.iter().any(|x| flag(x, F_COLLIDED_V) && !ground(x)),
            "hits its head",
        ),
        (
            "l3_decor_arena_0",
            |e| e.iter().filter(|x| flag(x, F_COLLIDED_H)).count() > 20,
            "bumps into the decor",
        ),
        (
            "l3_decor_arena_1",
            |e| e.iter().filter(|x| flag(x, F_COLLIDED_H)).count() > 20,
            "bumps into the decor",
        ),
        (
            "l3_decor_arena_2",
            |e| e.iter().filter(|x| flag(x, F_COLLIDED_H)).count() > 20,
            "bumps into the decor",
        ),
    ];
    for (name, pred, what) in assertions {
        let t = load(name);
        assert!(pred(&t.expected), "{name} no longer {what}");
    }
}

/// Each `h2_*` trace really reaches the state it is named for, so a regenerated
/// fixture cannot quietly stop testing the riding, flight and fall edges.
#[test]
fn the_h2_edge_scenarios_exercise_what_they_claim() {
    type Guard = (&'static str, fn(&[Expected]) -> bool, &'static str);
    fn flag(x: &Expected, f: u8) -> bool {
        x.flags & f != 0
    }
    fn flying(x: &Expected) -> bool {
        x.flags2 & F2_FLYING != 0
    }
    let assertions: &[Guard] = &[
        (
            "h2_ride_wall",
            |e| {
                let seat = DVec3::new(4.5, 64.0, 3.5);
                flag(&e[40], F_COLLIDED_H)
                    // Ride tick 1 travels into the wall from where the player was, and its sprint
                    // stop still read the walking tick's collision.
                    && flag(&e[41], F_COLLIDED_H)
                    && !flag(&e[41], F_SPRINTING)
                    && !flag(&e[42], F_COLLIDED_H)
                    && flag(&e[43], F_SPRINTING)
                    && e[41..47].iter().all(|x| x.pos == seat && x.vel.y < 0.0)
                    // The dismounted player lands on tick 47 and jumps on tick 48: jumpTicks was
                    // zeroed while jump was up during the ride.
                    && flag(&e[47], F_ON_GROUND)
                    && e[48].vel.y > 0.3
            },
            "refreshes the rider's collision, sprints mounted and jumps right after the dismount",
        ),
        (
            "h2_ride_water_flight",
            |e| {
                e[10..20]
                    .iter()
                    .all(|x| flag(x, F_IN_WATER) && x.vel.y > 0.0)
                    && flag(&e[25], F_ON_GROUND)
                    && !flying(&e[25])
                    && e[30..35].iter().all(|x| flying(x) && x.vel.y > 0.0)
            },
            "jumps in water while mounted, lands a flying rider and thrusts one up",
        ),
        (
            "h2_ride_midair",
            |e| {
                e[12..18].iter().all(|x| x.fall_distance == 0.0)
                    && e.iter().any(|x| x.fall_damage > 0 && x.fall_damage < 17)
            },
            "clears fallDistance while mounted, so only the second drop hurts",
        ),
        (
            "h2_soul_sand_ice",
            |e| {
                // Grounded on the sunk soul sand of each lane (blocks z 4-6 over stone, 7-9 over
                // slime, 10-12 over packed ice, 13-15 over ice), then on the plain ice beyond.
                [(4.3, 6.7), (7.3, 9.7), (10.3, 12.7), (13.3, 15.7)]
                    .iter()
                    .all(|&(lo, hi)| {
                        e.iter().any(|x| {
                            x.pos.y == 63.875
                                && x.pos.z > lo
                                && x.pos.z < hi
                                && flag(x, F_ON_GROUND)
                        })
                    })
                    && e.iter().any(|x| x.pos.y == 64.0 && x.pos.z > 17.0)
            },
            "walks on soul sand over every slippery block and reaches the ice",
        ),
        (
            "h2_web_water",
            |e| e.iter().filter(|x| flag(x, F_IN_WATER)).count() > 50,
            "swims through the webs",
        ),
        (
            "h2_ladder_water",
            |e| {
                e.iter()
                    .any(|x| flag(x, F_ON_LADDER) && flag(x, F_IN_WATER))
                    && e.iter()
                        .any(|x| flag(x, F_ON_LADDER) && !flag(x, F_IN_WATER))
            },
            "is on the ladder both in and above the water",
        ),
        (
            "h2_fall_ladder",
            |e| e.iter().any(|x| flag(x, F_ON_LADDER)) && e.iter().all(|x| x.fall_damage < 0),
            "is caught by the ladder",
        ),
        (
            "h2_fall_water",
            |e| e.iter().any(|x| flag(x, F_IN_WATER)) && e.iter().all(|x| x.fall_damage < 0),
            "is caught by the water",
        ),
        (
            "h2_fall_lava",
            |e| e.iter().any(|x| flag(x, F_IN_LAVA) && x.fall_damage > 0),
            "lands in lava and takes the fall",
        ),
        (
            "h2_fall_hay",
            |e| e.iter().any(|x| x.fall_damage == 16),
            "lands on hay with no 1.8.9 reduction",
        ),
        (
            "h2_fly_toggle",
            |e| {
                // A second tap 7 ticks after the first re-arms; 6 ticks after toggles.
                !flying(&e[12])
                    && flying(&e[18])
                    && flying(&e[60])
                    && !flying(&e[61])
                    && flying(&e[104])
                    && e[113].vel.y == 0.0
            },
            "toggles flight at the 7-tick window's boundary and off in mid-air",
        ),
    ];
    for (name, pred, what) in assertions {
        let t = load(name);
        assert!(pred(&t.expected), "{name} no longer {what}");
    }
}

/// Negative control: the mounted tick before this change ran only the preflight and
/// the C0C damping, never the rider's own travel, so motion stayed zero, the collision flag
/// went stale and `jumpTicks` only counted down. Replaying the riding traces with that model
/// must diverge from the game.
#[test]
fn the_pre_h2_mounted_tick_is_rejected_by_the_oracle() {
    fn pre_h2(w: &World, s: &mut PhysicsState, input: &MoveInput, seat: DVec3) {
        mounted_on_living_update_preflight(w, s, input, true);
        s.move_strafing *= 0.98;
        s.move_forward *= 0.98;
        // L4's onGround: false on every ride tick without a flight descent.
        s.on_ground = false;
        s.set_position(seat);
        s.fall_distance = 0.0;
    }
    for name in ["h2_ride_wall", "h2_ride_water_flight", "h2_ride_midair"] {
        let t = load(name);
        // The trace is only read; nothing observes it after a panic.
        let replay_old =
            std::panic::AssertUnwindSafe(|| replay_full(&t, TOLERANCE, apply_event, pre_h2));
        let diverged = std::panic::catch_unwind(replay_old).is_err();
        assert!(
            diverged,
            "{name}: the pre-H2 mounted tick must not match the game"
        );
    }
}

/// Negative controls: the combined-effect traces from `ComboRefGen` must depend
/// on their server events and on their feature blocks. Ignoring one event kind, or swapping the
/// feature blocks for plain ones in the Rust world, has to diverge from the game; a trace that
/// still matched would not be testing that handling at all.
#[test]
fn the_w19_traces_depend_on_their_events_and_blocks() {
    type Handler = fn(&mut PhysicsState, &mut bool, &Event);
    fn without_velocity(s: &mut PhysicsState, using_item: &mut bool, e: &Event) {
        if e.kind != 0 {
            apply_event(s, using_item, e);
        }
    }
    fn without_jump_boost(s: &mut PhysicsState, using_item: &mut bool, e: &Event) {
        if !(e.kind == 9 && e.a == 8.0) {
            apply_event(s, using_item, e);
        }
    }
    fn without_speed_snapshots(s: &mut PhysicsState, using_item: &mut bool, e: &Event) {
        if e.kind != 10 {
            apply_event(s, using_item, e);
        }
    }
    fn diverges(t: &Trace, handler: Handler) -> bool {
        // The trace is only read; nothing observes it after a panic.
        let run = std::panic::AssertUnwindSafe(|| replay_with(t, TOLERANCE, handler));
        std::panic::catch_unwind(run).is_err()
    }
    /// Swap every block whose id (`state.id >> 4`) is in `from` for `to` inside the box.
    fn swap_blocks(w: &mut World, lo: BlockPos, hi: BlockPos, from: &[u16], to: BlockState) {
        for x in lo.x..=hi.x {
            for y in lo.y..=hi.y {
                for z in lo.z..=hi.z {
                    let pos = BlockPos::new(x, y, z);
                    if from.contains(&(w.block_at(pos).id >> 4)) {
                        w.set_block(pos, to);
                    }
                }
            }
        }
    }

    let mut still_matching = Vec::new();
    let events: [(&str, Handler); 3] = [
        ("knockback", without_velocity),
        ("jump boost", without_jump_boost),
        ("speed snapshots", without_speed_snapshots),
    ];
    for name in [
        "w19_combo_arena_0",
        "w19_combo_arena_1",
        "w19_combo_arena_2",
        "w19_combo_arena_3",
    ] {
        let t = load(name);
        for (label, handler) in events {
            if !diverges(&t, handler) {
                still_matching.push(format!("{name} without {label}"));
            }
        }
    }
    for name in [
        "w19_ice_head_hitter",
        "w19_sneak_edge_knockback_slab",
        "w19_sneak_edge_knockback_stair",
        "w19_sneak_edge_knockback_fence",
    ] {
        if !diverges(&load(name), without_velocity) {
            still_matching.push(format!("{name} without knockback"));
        }
    }

    /// A trace, the box to edit, the block ids to replace and their replacement.
    type Swap<'a> = (&'a str, (BlockPos, BlockPos), &'a [u16], BlockState);
    let stone = BlockState::new(mc_data::BlockId(1), 0);
    let air = BlockState::AIR;
    let arena = (BlockPos::new(1, 56, 1), BlockPos::new(23, 70, 23));
    let swaps: &[Swap<'_>] = &[
        ("w19_combo_arena_0", arena, &[8, 9, 10, 11], air),
        ("w19_combo_arena_1", arena, &[8, 9, 10, 11], air),
        ("w19_combo_arena_2", arena, &[8, 9, 10, 11], air),
        ("w19_combo_arena_3", arena, &[8, 9, 10, 11], air),
        (
            "w19_ice_head_hitter",
            (BlockPos::new(6, 63, 2), BlockPos::new(10, 63, 100)),
            &[79, 174],
            stone,
        ),
        (
            "w19_sneak_edge_knockback_slab",
            (BlockPos::new(7, 65, 3), BlockPos::new(9, 65, 14)),
            &[44],
            stone,
        ),
        (
            "w19_sneak_edge_knockback_stair",
            (BlockPos::new(7, 65, 3), BlockPos::new(9, 65, 14)),
            &[53],
            stone,
        ),
        (
            "w19_sneak_edge_knockback_fence",
            (BlockPos::new(7, 65, 3), BlockPos::new(9, 65, 14)),
            &[85],
            stone,
        ),
    ];
    for &(name, (lo, hi), from, to) in swaps {
        let mut t = load(name);
        swap_blocks(&mut t.world, lo, hi, from, to);
        if !diverges(&t, apply_event) {
            still_matching.push(format!("{name} with blocks {from:?} replaced by {}", to.id));
        }
    }
    assert!(
        still_matching.is_empty(),
        "these negative controls still match the game, so the trace does not exercise what \
         it claims:\n  {}",
        still_matching.join("\n  ")
    );
}
