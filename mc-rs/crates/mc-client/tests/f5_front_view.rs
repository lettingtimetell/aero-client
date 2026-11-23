//! Bug report: F5's "second perspective (looking at players front) keeps switching between
//! looking forwards and backwards over and over again".
//!
//! `Client::place_camera` leaves the front view's render orientation, `(yaw + 180, -pitch)`, in
//! the shell `Camera`. The next frame's mouse look and the next tick's movement input used to
//! read the camera back as the player's own look, so the player (and the C05/C06 it sends) and
//! the orbit flipped every frame. java: `EntityRenderer.orientCamera` (`:663-716`) only reads the
//! entity's rotation, and `setAngles` turns `thePlayer` from its own rotation.
//!
//! This drives the real `ShellApp::before_ticks` / `tick` / `frame` sequence of a joined client,
//! the exact path the shell runs, with zero mouse motion, and then with a real mouse turn.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::{FrameCtx, FrameStats, GameSettings, InputState, PhaseList, ShellApp, TickCtx};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// `Entity.getLook` for a yaw/pitch pair, in doubles.
fn look(yaw: f32, pitch: f32) -> mc_math::DVec3 {
    let (yaw, pitch) = (f64::from(yaw).to_radians(), f64::from(pitch).to_radians());
    mc_math::DVec3::new(
        -yaw.sin() * pitch.cos(),
        -pitch.sin(),
        yaw.cos() * pitch.cos(),
    )
}

#[test]
fn front_view_holds_still_across_real_shell_frames_and_follows_the_mouse() {
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

    // A look with both angles non-zero, so a flip of either one shows.
    let (yaw, pitch) = (347.80002_f32, -19.817451_f32);
    {
        let predictor = client
            .session_mut()
            .and_then(|session| session.predictor_mut())
            .expect("joined predictor");
        predictor.state.yaw = yaw;
        predictor.state.pitch = pitch;
    }
    // The shell camera as a first-person frame leaves it, so only the front view can differ.
    camera.yaw = yaw;
    camera.pitch = pitch;
    client.third_person_view = 2;

    let mut input = InputState::default();
    let mut settings = GameSettings::default();
    let stats = FrameStats::new();
    let mut shell_frame =
        |client: &mut Client, camera: &mut mc_render::Camera, input: &mut InputState, n: u64| {
            ShellApp::before_ticks(client, camera);
            if n.is_multiple_of(2) {
                ShellApp::tick(
                    client,
                    &mut TickCtx {
                        input: &mut *input,
                        settings: &mut settings,
                        camera: &mut *camera,
                        tick: n,
                    },
                );
            }
            ShellApp::frame(
                client,
                &mut FrameCtx {
                    gpu: &gpu,
                    phases: &mut phases,
                    camera: &mut *camera,
                    input: &mut *input,
                    settings: &settings,
                    partial_ticks: 0.5,
                    stats: &stats,
                },
            );
            client
                .session()
                .and_then(|session| session.predictor())
                .expect("joined predictor")
                .state
        };

    let forward = look(yaw, pitch);
    for n in 0..12 {
        let state = shell_frame(&mut client, &mut camera, &mut input, n);
        assert_eq!(
            (state.yaw, state.pitch),
            (yaw, pitch),
            "frame {n}: the front view must not rewrite the player's look"
        );
        assert_eq!(
            (camera.yaw, camera.pitch),
            (yaw + 180.0, -pitch),
            "frame {n}: the front view looks back along the player's look"
        );
        let offset = camera.pos - state.eye_pos();
        assert!(
            offset.dot(forward) > 0.5,
            "frame {n}: the orbit must stay in front of the player, offset {offset:?}"
        );
    }

    // A real mouse turn still turns the player as in first person, and the orbit follows.
    // `InputState::look_delta_raw`'s cubic curve, associated exactly as it computes it.
    let f = client.settings.mouse_sensitivity * 0.6 + 0.2;
    let scale = f * f * f * 8.0;
    let raw_yaw = 40.0 * scale;
    let mut expected = mc_render::Camera {
        yaw,
        pitch,
        ..mc_render::Camera::default()
    };
    expected.turn_mouse(raw_yaw, 0.0);
    input.on_raw_motion(40.0, 0.0);
    let state = shell_frame(&mut client, &mut camera, &mut input, 13);
    assert_eq!((state.yaw, state.pitch), (expected.yaw, expected.pitch));
    assert!(
        state.yaw > yaw,
        "moving the mouse right turns the player right"
    );
    assert_eq!((camera.yaw, camera.pitch), (expected.yaw + 180.0, -pitch));

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
