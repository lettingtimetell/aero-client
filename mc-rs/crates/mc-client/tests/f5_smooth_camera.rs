//! Bug report: "F5 perspective looks glitchy and teleporty rather than
//! smooth".
//!
//! java: the F5 camera and the player's own body are drawn from the *same* interpolated point.
//! `EntityRenderer.orientCamera` puts the camera at the render-view entity's interpolated eye
//! (`prevPos + (pos - prevPos) * partialTicks`), and `RenderGlobal.renderEntities` draws the
//! client's own `EntityPlayerSP` at `lastTickPos + (pos - lastTickPos) * partialTicks` relative to
//! that same render origin, so in vanilla the body never moves against the camera.
//!
//! mc-rs interpolated the camera but drew the local body at the current tick's `pos`: between two
//! ticks the body ran up to one tick of motion ahead of the camera and snapped back at every tick
//! boundary, twenty times a second.
//!
//! This drives the real shell sequence of a joined client (`ShellApp::before_ticks`, the tick,
//! `ShellApp::frame`, `PhaseList::render`) at partial ticks 0, 0.25, 0.5 and 0.75 while the player
//! walks, jumps, and rides a server-driven pig, in both F5 views, and reads back where the
//! `entities` phase actually put the body. It checks:
//!
//! * the camera moves smoothly and monotonically (uniform steps, no jumps at tick boundaries);
//! * the body's head, in camera-relative space, holds still: body and camera share one point;
//! * the body's world position therefore moves smoothly and monotonically too.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_math::DVec3;
use mc_physics::state::MoveInput;
use mc_protocol::play::s2c;
use mc_protocol::Angle;
use mc_render::{FrameCtx, FrameStats, GameSettings, InputState, PhaseList, ShellApp};
use tokio::net::TcpListener;
use tokio::sync::mpsc;

/// The local player's entity id in the scripted join.
const PLAYER: i32 = 55;
/// The pig the riding case mounts.
const PIG: i32 = 77;

/// One scripted server packet for the riding case's peer to forward.
enum Out {
    Spawn(s2c::SpawnMob),
    Attach(s2c::EntityAttach),
    Move(s2c::EntityRelMove),
}

/// Join, then forward scripted packets until the test drops its sender.
async fn serve_riding(listener: TcpListener, mut rx: mpsc::UnboundedReceiver<Out>) {
    let mut peer = accept_and_join(listener, PLAYER).await;
    while let Some(out) = rx.recv().await {
        match out {
            Out::Spawn(packet) => peer.send(&packet).await,
            Out::Attach(packet) => peer.send(&packet).await,
            Out::Move(packet) => peer.send(&packet).await,
        }
    }
    peer.drain_until_closed().await;
}

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// The four partial ticks each tick is sampled at.
const PARTIALS: [f32; 4] = [0.0, 0.25, 0.5, 0.75];

/// One rendered frame: where the camera was and where the body's head was drawn.
#[derive(Debug, Clone, Copy)]
struct Sample {
    tick: u32,
    partial: f32,
    camera: DVec3,
    /// Centroid of the head/hat vertices, camera-relative (what the screen shows).
    head_rel: DVec3,
}

impl Sample {
    /// The head's world position: camera-relative position plus the camera.
    fn head_world(&self) -> DVec3 {
        self.camera + self.head_rel
    }
}

/// Centroid of the body's head region: every queued vertex within 0.3 blocks of the highest one.
///
/// The local body is the tallest thing in the `entities` phase: the walking case spawns nothing
/// else, and the riding case's pig sits well below its rider's head. The head and hat are rigid
/// while the look is fixed, unlike the limbs, which swing while walking.
fn head_centroid(client: &Client) -> DVec3 {
    let positions: Vec<[f32; 3]> = client
        .render()
        .expect("a live render stack")
        .entities
        .with(|phase| phase.vertex_positions().collect());
    assert!(!positions.is_empty(), "F5 must draw the local body");
    let top = positions
        .iter()
        .map(|p| p[1])
        .fold(f32::NEG_INFINITY, f32::max);
    let head: Vec<[f32; 3]> = positions.into_iter().filter(|p| p[1] > top - 0.3).collect();
    let n = head.len() as f64;
    let sum = head.iter().fold(DVec3::ZERO, |acc, p| {
        acc + DVec3::new(f64::from(p[0]), f64::from(p[1]), f64::from(p[2]))
    });
    sum / n
}

/// The shell's per-iteration sequence: `before_ticks`, the tick (if any), `frame`, render.
#[allow(clippy::too_many_arguments)]
fn shell_iteration(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut mc_render::Camera,
    input: &mut InputState,
    settings: &GameSettings,
    stats: &FrameStats,
    tick: Option<&MoveInput>,
    partial_ticks: f32,
) {
    ShellApp::before_ticks(client, camera);
    if let Some(movement) = tick {
        // `ShellApp::tick` builds exactly this `MoveInput` from the held keys and the
        // predictor's look, then calls `Client::tick` with it.
        client.tick(movement);
    }
    ShellApp::frame(
        client,
        &mut FrameCtx {
            gpu,
            phases,
            camera,
            input,
            settings,
            partial_ticks,
            stats,
        },
    );
    let frame = gpu.acquire().expect("acquire offscreen frame");
    phases.render(gpu, frame.view(), camera, partial_ticks);
}

/// Walk (or jump) for `ticks` ticks, sampling every tick at [`PARTIALS`].
#[allow(clippy::too_many_arguments)]
fn run(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut mc_render::Camera,
    input: &mut InputState,
    settings: &GameSettings,
    stats: &FrameStats,
    ticks: u32,
    movement: impl Fn(u32) -> MoveInput,
) -> Vec<Sample> {
    let mut samples = Vec::new();
    for tick in 0..ticks {
        let step = movement(tick);
        for (i, &partial) in PARTIALS.iter().enumerate() {
            let tick_input = (i == 0).then_some(&step);
            shell_iteration(
                client, gpu, phases, camera, input, settings, stats, tick_input, partial,
            );
            samples.push(Sample {
                tick,
                partial,
                camera: camera.pos,
                head_rel: head_centroid(client),
            });
        }
    }
    samples
}

/// The body must not move against the camera: its camera-relative head holds still.
fn assert_body_holds_still_on_screen(view: &str, samples: &[Sample]) {
    let first = samples[0].head_rel;
    for sample in samples {
        let drift = (sample.head_rel - first).length();
        assert!(
            drift < 1e-4,
            "{view}: at tick {} + {:.2} the body's head moved {drift:.4} blocks against the \
             camera (camera-relative {:?}, first frame {:?}); vanilla draws body and camera \
             from one interpolated point, so it must hold still on screen",
            sample.tick,
            sample.partial,
            sample.head_rel,
            first,
        );
    }
}

/// Positions along one axis must advance every frame by roughly the same step.
fn assert_smooth_monotone(view: &str, what: &str, values: &[(u32, f32, f64)]) {
    let steps: Vec<f64> = values.windows(2).map(|w| w[1].2 - w[0].2).collect();
    let mean = steps.iter().sum::<f64>() / steps.len() as f64;
    assert!(
        mean > 1e-3,
        "{view}: {what} must actually move (mean step {mean})"
    );
    for (window, step) in values.windows(2).zip(&steps) {
        assert!(
            *step > 0.0,
            "{view}: {what} must advance every frame, but from tick {} + {:.2} to tick {} + \
             {:.2} it moved {step:.5} (steps {steps:.4?})",
            window[0].0,
            window[0].1,
            window[1].0,
            window[1].1,
        );
        assert!(
            (step - mean).abs() < mean * 0.25,
            "{view}: {what} must move in even steps at a steady walk, but from tick {} + {:.2} \
             to tick {} + {:.2} it moved {step:.5} against a mean of {mean:.5} (steps \
             {steps:.4?})",
            window[0].0,
            window[0].1,
            window[1].0,
            window[1].1,
        );
    }
}

#[test]
fn f5_camera_and_local_body_move_smoothly_between_ticks() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();
    camera.set_viewport(WIDTH, HEIGHT);

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    // Nothing but the local body is ever drawn into the `entities` phase.
    let server = client.runtime().handle().spawn(async move {
        let mut peer = accept_and_join(listener, 55).await;
        peer.drain_until_closed().await;
    });

    client.connect("127.0.0.1", port);
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        |client| {
            client.machine.state().is_in_world()
                && client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
        },
    );

    let mut input = InputState::default();
    let settings = GameSettings::default();
    let stats = FrameStats::new();
    // No view bobbing: it rotates the whole view per frame and is not what is under test.
    client.settings.view_bobbing = false;

    for view in [1_u8, 2] {
        let name = if view == 1 {
            "F5 back view"
        } else {
            "F5 front view"
        };
        // Back at the spawn point, standing still, facing +Z (south) and level, so the eight
        // corner probes behind (or in front of) the eye clear the stone floor.
        let (yaw, pitch) = (0.0_f32, 0.0_f32);
        {
            let predictor = client
                .session_mut()
                .and_then(|session| session.predictor_mut())
                .expect("joined predictor");
            predictor
                .state
                .set_position(DVec3::new(common::SPAWN.0, common::SPAWN.1, 4.5));
            predictor.state.vel = DVec3::ZERO;
            predictor.state.yaw = yaw;
            predictor.state.pitch = pitch;
        }
        camera.yaw = yaw;
        camera.pitch = pitch;
        client.third_person_view = view;
        let still = MoveInput {
            yaw,
            pitch,
            ..MoveInput::default()
        };
        // Settle on the floor, with the body yaw caught up with the look.
        let _ = run(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &mut input,
            &settings,
            &stats,
            6,
            |_| still,
        );

        // ---- a steady walk along +Z -------------------------------------------------------
        let walk = MoveInput {
            forward: true,
            ..still
        };
        // Ramp up to the steady walking speed first; vanilla friction converges within ~8 ticks.
        let _ = run(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &mut input,
            &settings,
            &stats,
            12,
            |_| walk,
        );
        let walking = run(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &mut input,
            &settings,
            &stats,
            8,
            |_| walk,
        );
        let camera_z: Vec<(u32, f32, f64)> = walking
            .iter()
            .map(|s| (s.tick, s.partial, s.camera.z))
            .collect();
        assert_smooth_monotone(name, "the camera", &camera_z);
        for sample in &walking {
            assert!(
                (sample.camera.x - walking[0].camera.x).abs() < 1e-6
                    && (sample.camera.y - walking[0].camera.y).abs() < 1e-6,
                "{name}: walking straight along +Z must not move the camera sideways or up, \
                 {sample:?}"
            );
        }
        assert_body_holds_still_on_screen(name, &walking);
        let body_z: Vec<(u32, f32, f64)> = walking
            .iter()
            .map(|s| (s.tick, s.partial, s.head_world().z))
            .collect();
        assert_smooth_monotone(name, "the body", &body_z);

        // ---- a jump from standing ----------------------------------------------------------
        // Stop, then jump once: twelve ticks of rising and falling, where every frame between
        // two ticks must again keep the body on the camera's interpolated point.
        let _ = run(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &mut input,
            &settings,
            &stats,
            12,
            |_| still,
        );
        let jump = MoveInput {
            jump: true,
            ..still
        };
        let jumping = run(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &mut input,
            &settings,
            &stats,
            12,
            |tick| if tick == 0 { jump } else { still },
        );
        let peak = jumping
            .iter()
            .map(|s| s.camera.y)
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(
            peak > jumping[0].camera.y + 0.5,
            "{name}: the jump must lift the camera (start {}, peak {peak})",
            jumping[0].camera.y
        );
        // Rising: every frame higher than the last until the apex, no tick-boundary jumps.
        let rising: Vec<(u32, f32, f64)> = jumping
            .iter()
            .take_while(|s| s.tick < 4)
            .map(|s| (s.tick, s.partial, s.camera.y))
            .collect();
        for pair in rising.windows(2) {
            assert!(
                pair[1].2 > pair[0].2,
                "{name}: the camera must rise every frame early in a jump, {rising:?}"
            );
        }
        assert_body_holds_still_on_screen(&format!("{name}, jumping"), &jumping);
    }

    client.third_person_view = 0;
    client.disconnect();
    for _ in 0..10 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    client.runtime().handle().block_on(async {
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the peer must finish")
            .expect("the peer task must not panic");
    });
}

/// Riding: the server walks a pig south and the local player sits on it.
///
/// java: `Entity.updateRidden` -> `updateRiderPosition` moves the rider with its vehicle once a
/// tick, after `onEntityUpdate` took `prevPos`; the camera and the rider's body then interpolate
/// that pair exactly like a walking player's. mc-rs drew the riding body at the current tick's
/// seat as well, so it jittered against both the camera and the (interpolated) pig.
#[test]
fn f5_riding_camera_and_body_follow_the_vehicle_smoothly() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();
    camera.set_viewport(WIDTH, HEIGHT);

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let (tx, rx) = mpsc::unbounded_channel();
    let server = client.runtime().handle().spawn(serve_riding(listener, rx));

    client.connect("127.0.0.1", port);
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        |client| {
            client.machine.state().is_in_world()
                && client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
        },
    );

    // A pig on the floor at (8.5, 5, 3), then S1B puts the player on it.
    tx.send(Out::Spawn(s2c::SpawnMob {
        entity_id: PIG,
        entity_type: 90,
        x: 272,
        y: 160,
        z: 96,
        yaw: Angle(0),
        pitch: Angle(0),
        head_pitch: Angle(0),
        velocity_x: 0,
        velocity_y: 0,
        velocity_z: 0,
        metadata: Vec::new(),
    }))
    .expect("peer alive");
    tx.send(Out::Attach(s2c::EntityAttach {
        entity_id: PLAYER,
        vehicle_entity_id: PIG,
        attach_type: s2c::AttachType::Vehicle,
    }))
    .expect("peer alive");
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(10),
        "the player to mount the pig",
        |client| {
            client
                .session()
                .is_some_and(|session| session.riding() == Some(PIG))
        },
    );

    let (yaw, pitch) = (0.0_f32, 0.0_f32);
    {
        let predictor = client
            .session_mut()
            .and_then(|session| session.predictor_mut())
            .expect("joined predictor");
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;
    }
    camera.yaw = yaw;
    camera.pitch = pitch;
    client.settings.view_bobbing = false;
    let still = MoveInput {
        yaw,
        pitch,
        ..MoveInput::default()
    };
    let mut input = InputState::default();
    let settings = GameSettings::default();
    let stats = FrameStats::new();

    for view in [1_u8, 2] {
        let name = if view == 1 {
            "F5 back view, riding"
        } else {
            "F5 front view, riding"
        };
        client.third_person_view = view;
        let mut samples = Vec::new();
        for tick in 0..16_u32 {
            // The server walks the pig 6/32 of a block south every tick.
            tx.send(Out::Move(s2c::EntityRelMove {
                entity_id: PIG,
                delta_x: 0,
                delta_y: 0,
                delta_z: 6,
                on_ground: true,
            }))
            .expect("peer alive");
            // Let the loopback deliver it before this tick's `before_ticks` polls.
            std::thread::sleep(Duration::from_millis(20));
            for (i, &partial) in PARTIALS.iter().enumerate() {
                let tick_input = (i == 0).then_some(&still);
                shell_iteration(
                    &mut client,
                    &mut gpu,
                    &mut phases,
                    &mut camera,
                    &mut input,
                    &settings,
                    &stats,
                    tick_input,
                    partial,
                );
                samples.push(Sample {
                    tick,
                    partial,
                    camera: camera.pos,
                    head_rel: head_centroid(&client),
                });
            }
        }
        assert!(
            client
                .session()
                .is_some_and(|session| session.riding() == Some(PIG)),
            "{name}: still riding"
        );
        // The pig's three-step `setPositionAndRotation2` interpolation ramps up first.
        let riding = &samples[4 * 4..];
        let start = riding[0].camera.z;
        let end = riding[riding.len() - 1].camera.z;
        assert!(
            end > start + 0.5,
            "{name}: the pig must carry the camera south ({start} -> {end})"
        );
        for pair in riding.windows(2) {
            assert!(
                pair[1].camera.z >= pair[0].camera.z - 1e-9,
                "{name}: riding forward, the camera must never step back: {:?} -> {:?}",
                pair[0],
                pair[1]
            );
        }
        assert_body_holds_still_on_screen(name, riding);
    }

    drop(tx);
    client.third_person_view = 0;
    client.disconnect();
    for _ in 0..10 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    client.runtime().handle().block_on(async {
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the peer must finish")
            .expect("the peer task must not panic");
    });
}
