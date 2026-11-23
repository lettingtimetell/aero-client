//! Remote-player interpolation and the click pick, bit for bit
//! against the real 1.8.9 classes.
//!
//! `javaref/PickRef.java` (`generate-pick.ps1`) drives a real `EntityOtherPlayerMP` through
//! `handleSpawnPlayer`/`handleEntityMovement`/`handleEntityTeleport` and its own
//! `onLivingUpdate`, then runs a verbatim `EntityRenderer.getMouseOver(1.0F)` over real players,
//! real blocks and the real AABB/Vec3 maths, with the viewer's `rotationYaw` set 37 degrees away
//! from its `rotationYawHead`. This replays both through the client's own code: [`EntityTracker`]
//! for the packets and ticks (the live S14-S18 handlers and the per-tick ease), and the driver's
//! candidate construction plus [`interact::pick_with_riding`] for the pick. Positions, eyes, look
//! vectors and hit results must match exactly.
//!
//! What this proves for hit registration: where a remote player is after any packet/tick
//! interleaving (the 1/32 `serverPos` accumulation, S18's dead zone, the three-step ease), and
//! what a click lands on from there (the 3.0 entity limit against 4.5/5.0 block reach and
//! creative's 6.0, the 0.1 border, the eye-inside case, blocks without collision boxes such as
//! tall grass in the way, spectators excluded, ties between overlapping boxes).

use mc_client::app::interact::{self, CollisionState, HitResult, Pickable, PlayerController};
use mc_client::sync::{EntityTracker, TrackedKind};
use mc_data::{BlockId, BlockState};
use mc_math::{BlockPos, DVec3};
use mc_physics::state::PhysicsState;
use mc_protocol::Angle;
use mc_world::entity::Metadata;

const REFERENCE: &str = include_str!("../javaref/pick_ref.txt");

fn f64_hex(field: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(field, 16).expect("f64 bits"))
}

fn f32_hex(field: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(field, 16).expect("f32 bits"))
}

fn int(field: &str) -> i32 {
    field.parse().expect("integer field")
}

fn angle(field: &str) -> f32 {
    // java: `(float)(byte * 360) / 256.0F` on the packet's signed byte.
    Angle(field.parse::<i8>().expect("signed byte") as u8).to_degrees()
}

/// Bit equality, with Java's `Vec3` folding of `-0.0` into `+0.0` treated as equal.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0)
}

fn same_vec(a: DVec3, b: DVec3) -> bool {
    same(a.x, b.x) && same(a.y, b.y) && same(a.z, b.z)
}

const PLAYER: i32 = 1;

#[test]
fn remote_player_positions_match_entity_other_player_mp_tick_for_tick() {
    let mut world = mc_world::World::new();
    let mut tracker = EntityTracker::new();
    let mut scenario = String::new();
    let (mut ticks, mut packets, mut dead_zone_candidates) = (0_usize, 0_usize, 0_usize);
    let mut rotation_mismatches = 0_usize;
    let mut failures = Vec::new();

    for line in REFERENCE.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("i") => {
                world = mc_world::World::new();
                tracker = EntityTracker::new();
                scenario = f[1].to_owned();
                tracker.spawn(
                    &mut world,
                    PLAYER,
                    TrackedKind::Player,
                    Some(uuid::Uuid::from_u128(0x4c37)),
                    [int(f[2]), int(f[3]), int(f[4])],
                    angle(f[5]),
                    angle(f[6]),
                    DVec3::ZERO,
                    Metadata::new(),
                );
            }
            Some("p") => {
                packets += 1;
                let kind = int(f[1]);
                let ground = Some(f[7] == "1");
                let delta = [
                    f[2].parse::<i8>().unwrap_or(0),
                    f[3].parse::<i8>().unwrap_or(0),
                    f[4].parse::<i8>().unwrap_or(0),
                ];
                let look = Some((angle(f[5]), angle(f[6])));
                let applied = match kind {
                    14 => tracker.apply_relative_move_grounded(
                        &mut world,
                        PLAYER,
                        [0, 0, 0],
                        None,
                        Some(false),
                    ),
                    15 => tracker
                        .apply_relative_move_grounded(&mut world, PLAYER, delta, None, ground),
                    16 => tracker.apply_relative_move_grounded(
                        &mut world,
                        PLAYER,
                        [0, 0, 0],
                        look,
                        ground,
                    ),
                    17 => tracker
                        .apply_relative_move_grounded(&mut world, PLAYER, delta, look, ground),
                    18 => {
                        let server = [int(f[2]), int(f[3]), int(f[4])];
                        let current = tracker.get(PLAYER).expect("tracked").position(&world);
                        let near = |live: f64, raw: i32, zone: f64| {
                            (live - f64::from(raw) / 32.0).abs() < zone
                        };
                        if near(current.x, server[0], 0.03125)
                            && near(current.y, server[1], 0.015_625)
                            && near(current.z, server[2], 0.03125)
                        {
                            dead_zone_candidates += 1;
                        }
                        tracker.apply_teleport_grounded(
                            &mut world,
                            PLAYER,
                            server,
                            angle(f[5]),
                            angle(f[6]),
                            ground,
                        )
                    }
                    other => panic!("unknown packet kind {other}"),
                };
                assert!(applied, "scenario {scenario}: the player is tracked");
            }
            Some("t") => {
                ticks += 1;
                tracker.tick_interpolation(&mut world);
                let tracked = tracker.get(PLAYER).expect("tracked");
                let rust = tracked.position(&world);
                let java = DVec3::new(f64_hex(f[1]), f64_hex(f[2]), f64_hex(f[3]));
                if !same_vec(rust, java) && failures.len() < 20 {
                    failures.push(format!(
                        "scenario {scenario} tick {ticks}: rust {rust:?} java {java:?}"
                    ));
                }
                let rotation = world
                    .ecs
                    .get::<&mc_world::Rotation>(tracked.entity)
                    .map(|rotation| (rotation.yaw, rotation.pitch))
                    .expect("a player has a rotation");
                if rotation.0.to_bits() != f32_hex(f[4]).to_bits()
                    || rotation.1.to_bits() != f32_hex(f[5]).to_bits()
                {
                    rotation_mismatches += 1;
                }
            }
            _ => {}
        }
    }

    assert!(
        ticks > 4000 && packets > 4000,
        "the reference covers {ticks} ticks"
    );
    assert!(
        dead_zone_candidates > 30,
        "S18's dead zone was exercised only {dead_zone_candidates} times"
    );
    assert!(
        failures.is_empty(),
        "position mismatches:\n{}",
        failures.join("\n")
    );
    // The rotation ease (body yaw/pitch) only steers the remote model and its look line; hit
    // boxes are axis-aligned. It is checked, and must be exact as well.
    assert_eq!(
        rotation_mismatches, 0,
        "remote rotation mismatches against Java"
    );
}

struct Query {
    creative: bool,
    feet: DVec3,
    head_yaw: f32,
    pitch: f32,
    sneak: bool,
    java_eye: DVec3,
    java_look: DVec3,
    targets: Vec<(DVec3, bool)>,
    result: Vec<String>,
}

fn parse_query(f: &[&str]) -> Query {
    let k = int(f[15]) as usize;
    let mut targets = Vec::with_capacity(k);
    for j in 0..k {
        let at = 16 + j * 4;
        targets.push((
            DVec3::new(f64_hex(f[at]), f64_hex(f[at + 1]), f64_hex(f[at + 2])),
            f[at + 3] == "1",
        ));
    }
    Query {
        creative: f[2] == "1",
        feet: DVec3::new(f64_hex(f[3]), f64_hex(f[4]), f64_hex(f[5])),
        head_yaw: f32_hex(f[6]),
        pitch: f32_hex(f[7]),
        sneak: f[8] == "1",
        java_eye: DVec3::new(f64_hex(f[9]), f64_hex(f[10]), f64_hex(f[11])),
        java_look: DVec3::new(f64_hex(f[12]), f64_hex(f[13]), f64_hex(f[14])),
        targets,
        result: f[16 + k * 4..].iter().map(|s| (*s).to_owned()).collect(),
    }
}

/// The driver's candidate construction (`Client::pick_result_at`) for remote players, then the
/// pick itself: `EntityPlayer.setSize(0.6F, 1.8F)`'s box, the `0.1F` border, the cheap reach
/// reject, `NOT_SPECTATING && canBeCollidedWith`, list order.
fn rust_pick(world: &mc_world::World, q: &Query, eye: DVec3, yaw: f32) -> HitResult {
    let look = interact::look_vector(yaw, q.pitch);
    let entity_reach = PlayerController::entity_reach(q.creative);
    let candidates = q
        .targets
        .iter()
        .enumerate()
        .filter_map(|(index, &(position, spectator))| {
            let bounds = interact::size_bounding_box(position, 0.6, 1.8);
            let border_size = interact::collision_border_size(None);
            interact::box_within_reach(bounds, border_size, eye, entity_reach).then_some(Pickable {
                network_id: index as i32,
                bounds,
                border_size,
                collidable: !spectator
                    && interact::can_be_collided_with(None, CollisionState::default()),
            })
        })
        .collect::<Vec<_>>();
    interact::pick_with_riding(
        world,
        eye,
        look,
        PlayerController::block_reach(q.creative),
        q.creative,
        None,
        candidates,
    )
}

fn matches_java(hit: HitResult, result: &[String]) -> bool {
    match (result[0].as_str(), hit) {
        ("M", HitResult::Miss) => true,
        ("B", HitResult::Block(block)) => {
            block.pos == BlockPos::new(int(&result[1]), int(&result[2]), int(&result[3]))
                && i32::from(block.face.index()) == int(&result[4])
                && same_vec(
                    block.hit_vec,
                    DVec3::new(
                        f64_hex(&result[5]),
                        f64_hex(&result[6]),
                        f64_hex(&result[7]),
                    ),
                )
        }
        ("E", HitResult::Entity(entity)) => {
            entity.network_id == int(&result[1])
                && same_vec(
                    entity.hit_vec,
                    DVec3::new(
                        f64_hex(&result[2]),
                        f64_hex(&result[3]),
                        f64_hex(&result[4]),
                    ),
                )
        }
        _ => false,
    }
}

#[test]
fn the_click_pick_matches_get_mouse_over_on_the_real_classes() {
    let mut world = mc_render::terrain::fixture::blank_world(1);
    let mut queries = Vec::new();
    for line in REFERENCE.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("b") => {
                let packed = int(f[4]);
                world.set_block(
                    BlockPos::new(int(f[1]), int(f[2]), int(f[3])),
                    BlockState::new(BlockId((packed >> 4) as u16), (packed & 15) as u8),
                );
            }
            Some("q") => queries.push(parse_query(&f)),
            _ => {}
        }
    }
    assert_eq!(queries.len(), 2500);

    let mut failures = Vec::new();
    let mut kinds = [0_usize; 3];
    let mut frame_yaw_differs = 0_usize;
    // Coverage: eye inside a bordered box; nearest bordered box on the ray between the 3.0
    // entity limit and the block reach; a spectator's box on the ray.
    let (mut inside, mut past_three, mut spectator_on_ray) = (0_usize, 0_usize, 0_usize);
    for (n, q) in queries.iter().enumerate() {
        // java: `getPositionEyes(1.0F)`, `posY + (double)getEyeHeight()` with `EntityPlayer`'s
        // float `1.62F - 0.08F` while sneaking: the same formula as `Session::tick_pick_ray`.
        let mut state = PhysicsState::new_player(q.feet);
        state.sneaking = q.sneak;
        let eye = state.eye_pos();
        // java: `EntityLivingBase.getLook(1.0F)` -> `getVectorForRotation(rotationPitch,
        // rotationYawHead)`.
        let look = interact::look_vector(q.head_yaw, q.pitch);
        let border = f64::from(interact::collision_border_size(None));
        let reach = PlayerController::block_reach(q.creative);
        let mut nearest = f64::INFINITY;
        for &(position, spectator) in &q.targets {
            let bordered =
                interact::size_bounding_box(position, 0.6, 1.8).expand(border, border, border);
            if bordered.is_vec_inside(eye) {
                inside += usize::from(!spectator);
                continue;
            }
            if let Some((point, _)) = bordered.calculate_intercept(eye, eye + look * reach) {
                if spectator {
                    spectator_on_ray += 1;
                } else {
                    nearest = nearest.min(mc_math::vec::distance_to(eye, point));
                }
            }
        }
        past_three += usize::from(nearest > 3.0 && nearest < reach);
        let hit = rust_pick(&world, q, eye, q.head_yaw);
        kinds[match q.result[0].as_str() {
            "M" => 0,
            "B" => 1,
            _ => 2,
        }] += 1;
        let eye_ok = same_vec(eye, q.java_eye);
        let look_ok = same_vec(look, q.java_look);
        if !(eye_ok && look_ok && matches_java(hit, &q.result)) && failures.len() < 20 {
            failures.push(format!(
                "query {n}: eye {eye_ok} look {look_ok}; rust {hit:?}; java {:?}",
                q.result
            ));
        }
        // Negative control: aiming with the viewer's `rotationYaw` (37 degrees off the head) must
        // usually land somewhere else, or this reference would not discriminate the yaw source.
        if !matches_java(rust_pick(&world, q, eye, q.head_yaw + 37.0), &q.result) {
            frame_yaw_differs += 1;
        }
    }

    assert!(
        failures.is_empty(),
        "pick mismatches:\n{}",
        failures.join("\n")
    );
    eprintln!(
        "picks: miss/block/entity {kinds:?}; eye inside a box {inside}; nearest box past 3.0 \
         {past_three}; spectator box on the ray {spectator_on_ray}"
    );
    assert!(
        kinds.iter().all(|&count| count > 400),
        "miss/block/entity coverage {kinds:?}"
    );
    assert!(
        inside > 50 && past_three > 100 && spectator_on_ray > 20,
        "edge coverage: inside {inside}, past 3.0 {past_three}, spectator {spectator_on_ray}"
    );
    assert!(
        frame_yaw_differs > queries.len() / 2,
        "only {frame_yaw_differs} picks change when aimed with rotationYaw"
    );
}
