//! Differential test: vanilla 1.8.9's client-side projectile simulation against mc-rs's.
//!
//! `javaref/ProjectileRefGen.java` runs the real recompiled vanilla classes (a client `World` with
//! `isRemote == true`) and records, tick by tick, what a vanilla client's `EntityEnderPearl`,
//! `EntitySnowball`, `EntityEgg`, `EntityPotion`, `EntityExpBottle`, `EntityArrow` and
//! `EntityLargeFireball` do after `NetHandlerPlayClient` created them from the same spawn,
//! velocity, teleport and relative-move packets. Regenerate the fixtures with
//! `javaref/generate-projectiles.ps1`; nothing in them is hand-derived.
//!
//! This test feeds those packets through mc-rs's real dispatch (`mc_client::sync::apply`) and
//! entity tick (`EntityTracker::tick_interpolation`) and compares position, motion and rotation
//! bit for bit after every tick.

use mc_client::sync::{apply, ClientState};
use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, ChunkPos};
use mc_protocol::play::{s2c, S2CPlay};
use mc_protocol::Angle;
use mc_world::entity::{Position, Rotation, Velocity};

/// The generator's fixed entity id (`ProjectileRefGen.ENTITY_ID`).
const ENTITY_ID: i32 = 100;

/// Sections 3..=5 (y 48..95) cover every block and trajectory in the fixtures; the rest of each
/// column is air, which is also what the Java `RefWorld` answers outside its block map.
const SECTION_MASK: u16 = 0b0011_1000;

#[derive(Debug)]
struct Expected {
    tick: u32,
    pos: [u64; 3],
    motion: [u64; 3],
    yaw: u32,
    pitch: u32,
}

#[derive(Debug)]
enum Step {
    Fill([i32; 8]),
    Spawn(s2c::SpawnObject),
    Packet {
        before_tick: u32,
        packet: Box<S2CPlay>,
    },
    Tick(Expected),
}

fn angle(value: &str) -> Angle {
    Angle(value.parse::<i32>().expect("angle byte") as i8 as u8)
}

fn int(value: &str) -> i32 {
    value.parse().expect("integer")
}

fn parse(text: &str) -> (String, Vec<Step>) {
    let mut name = String::new();
    let mut steps = Vec::new();
    for line in text.lines() {
        let p: Vec<&str> = line.split_whitespace().collect();
        match p.first().copied() {
            Some("scenario") => name = p[1].to_owned(),
            Some("fill") => {
                let mut f = [0; 8];
                for (slot, value) in f.iter_mut().zip(&p[1..9]) {
                    *slot = int(value);
                }
                steps.push(Step::Fill(f));
            }
            Some("spawn") => {
                let data = int(p[7]);
                steps.push(Step::Spawn(s2c::SpawnObject {
                    entity_id: ENTITY_ID,
                    object_type: int(p[1]) as i8,
                    x: int(p[2]),
                    y: int(p[3]),
                    z: int(p[4]),
                    pitch: angle(p[6]),
                    yaw: angle(p[5]),
                    data,
                    // S0EPacketSpawnObject only serialises the speed shorts when data > 0.
                    velocity: (data > 0)
                        .then(|| (int(p[8]) as i16, int(p[9]) as i16, int(p[10]) as i16)),
                }));
            }
            Some("velocity") => steps.push(Step::Packet {
                before_tick: int(p[1]) as u32,
                packet: Box::new(S2CPlay::EntityVelocity(s2c::EntityVelocity {
                    entity_id: ENTITY_ID,
                    motion_x: int(p[2]) as i16,
                    motion_y: int(p[3]) as i16,
                    motion_z: int(p[4]) as i16,
                })),
            }),
            Some("teleport") => steps.push(Step::Packet {
                before_tick: int(p[1]) as u32,
                packet: Box::new(S2CPlay::EntityTeleport(s2c::EntityTeleport {
                    entity_id: ENTITY_ID,
                    x: int(p[2]),
                    y: int(p[3]),
                    z: int(p[4]),
                    yaw: angle(p[5]),
                    pitch: angle(p[6]),
                    on_ground: p[7] != "0",
                })),
            }),
            Some("move") => steps.push(Step::Packet {
                before_tick: int(p[1]) as u32,
                packet: Box::new(S2CPlay::EntityLookAndRelMove(s2c::EntityLookAndRelMove {
                    entity_id: ENTITY_ID,
                    delta_x: int(p[2]) as i8,
                    delta_y: int(p[3]) as i8,
                    delta_z: int(p[4]) as i8,
                    yaw: angle(p[5]),
                    pitch: angle(p[6]),
                    on_ground: p[7] != "0",
                })),
            }),
            Some("tick") => {
                let d = |i: usize| u64::from_str_radix(p[i], 16).expect("double bits");
                let f = |i: usize| u32::from_str_radix(p[i], 16).expect("float bits");
                steps.push(Step::Tick(Expected {
                    tick: int(p[1]) as u32,
                    pos: [d(2), d(3), d(4)],
                    motion: [d(5), d(6), d(7)],
                    yaw: f(8),
                    pitch: f(9),
                }));
            }
            Some("end") | None => {}
            Some(other) => panic!("unknown fixture record {other:?}"),
        }
    }
    (name, steps)
}

fn load_columns(state: &mut ClientState) {
    let payload = vec![0; mc_world::expected_chunk_data_len(SECTION_MASK, true, true)];
    for x in -5..=5 {
        for z in -3..=7 {
            state
                .world
                .apply_chunk_data(ChunkPos::new(x, z), true, SECTION_MASK, &payload)
                .expect("resident fixture column");
        }
    }
}

fn send(state: &mut ClientState, packet: S2CPlay) {
    let mut events = Vec::new();
    apply(state, packet, &mut events).expect("dispatch");
}

/// Replay one fixture and return every per-tick disagreement, formatted for a failure message.
fn replay(text: &str) -> (String, Vec<String>) {
    let (name, steps) = parse(text);
    let mut state = ClientState::new();
    load_columns(&mut state);
    let mut pending: Vec<(u32, S2CPlay)> = Vec::new();
    let mut mismatches = Vec::new();
    for step in steps {
        match step {
            Step::Fill([x0, y0, z0, x1, y1, z1, id, meta]) => {
                let block = BlockState::new(BlockId(id as u16), meta as u8);
                for x in x0..=x1 {
                    for y in y0..=y1 {
                        for z in z0..=z1 {
                            state.world.set_block(BlockPos::new(x, y, z), block);
                        }
                    }
                }
            }
            Step::Spawn(spawn) => send(&mut state, S2CPlay::SpawnObject(spawn)),
            Step::Packet {
                before_tick,
                packet,
            } => pending.push((before_tick, *packet)),
            Step::Tick(expected) => {
                // `Minecraft.runGameLoop` drains the packet queue before the tick loop.
                let due: Vec<S2CPlay> = pending
                    .iter()
                    .filter(|(tick, _)| *tick == expected.tick)
                    .map(|(_, packet)| packet.clone())
                    .collect();
                pending.retain(|(tick, _)| *tick != expected.tick);
                for packet in due {
                    send(&mut state, packet);
                }
                state.entities.tick_interpolation(&mut state.world);

                let tracked = state.entities.get(ENTITY_ID).expect("projectile tracked");
                let ecs = &state.world.ecs;
                let pos = ecs.get::<&Position>(tracked.entity).expect("position").0;
                let motion = ecs.get::<&Velocity>(tracked.entity).expect("velocity").0;
                let rotation = *ecs.get::<&Rotation>(tracked.entity).expect("rotation");
                let got_pos = [pos.x.to_bits(), pos.y.to_bits(), pos.z.to_bits()];
                let got_motion = [motion.x.to_bits(), motion.y.to_bits(), motion.z.to_bits()];
                let fields = [
                    ("pos", got_pos.to_vec(), expected.pos.to_vec()),
                    ("motion", got_motion.to_vec(), expected.motion.to_vec()),
                    (
                        "rotation",
                        vec![
                            u64::from(rotation.yaw.to_bits()),
                            u64::from(rotation.pitch.to_bits()),
                        ],
                        vec![u64::from(expected.yaw), u64::from(expected.pitch)],
                    ),
                ];
                for (field, got, want) in fields {
                    if got != want {
                        let show = |bits: &[u64]| -> Vec<f64> {
                            if field == "rotation" {
                                bits.iter()
                                    .map(|b| f64::from(f32::from_bits(*b as u32)))
                                    .collect()
                            } else {
                                bits.iter().map(|b| f64::from_bits(*b)).collect()
                            }
                        };
                        mismatches.push(format!(
                            "{name} tick {} {field}: mc-rs {:?} vanilla {:?}",
                            expected.tick,
                            show(&got),
                            show(&want)
                        ));
                    }
                }
            }
        }
    }
    (name, mismatches)
}

fn check(text: &str) {
    let (name, mismatches) = replay(text);
    assert!(
        mismatches.is_empty(),
        "{name}: {} per-tick disagreement(s) with vanilla; first ones:\n{}",
        mismatches.len(),
        mismatches
            .iter()
            .take(12)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

macro_rules! fixture {
    ($test:ident, $file:literal) => {
        #[test]
        fn $test() {
            check(include_str!(concat!("fixtures/projectiles/", $file)));
        }
    };
}

fixture!(pearl_flat_throw_matches_vanilla, "pearl_flat_throw.txt");
fixture!(
    pearl_into_still_water_matches_vanilla,
    "pearl_into_still_water.txt"
);
fixture!(
    pearl_into_flowing_water_matches_vanilla,
    "pearl_into_flowing_water.txt"
);
fixture!(
    pearl_teleport_lift_matches_vanilla,
    "pearl_teleport_lift.txt"
);
fixture!(
    snowball_into_tall_grass_matches_vanilla,
    "snowball_into_tall_grass.txt"
);
fixture!(egg_arc_matches_vanilla, "egg_arc.txt");
fixture!(potion_arc_matches_vanilla, "potion_arc.txt");
fixture!(xp_bottle_arc_matches_vanilla, "xp_bottle_arc.txt");
fixture!(arrow_grass_wall_matches_vanilla, "arrow_grass_wall.txt");
fixture!(arrow_long_shot_matches_vanilla, "arrow_long_shot.txt");
fixture!(arrow_into_water_matches_vanilla, "arrow_into_water.txt");
fixture!(
    fireball_into_water_matches_vanilla,
    "fireball_into_water.txt"
);
