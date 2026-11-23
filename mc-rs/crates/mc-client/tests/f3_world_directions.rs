//! Bug report: "Can't see crosshair in F3 screen".
//!
//! Vanilla hides the 2D crosshair while the debug screen is open (`GuiIngame.showCrosshair`,
//! `GuiIngame.java:513-518`) and draws `EntityRenderer.renderWorldDirections`
//! (`EntityRenderer.java:1265-1288`, called at `:1486`) in its place: three tiny world-aligned
//! boxes at the eye, red +X, green +Y, blue +Z. mc-rs hid the crosshair but drew nothing instead.
//!
//! This renders a real joined client frame through the real `PhaseList` into a headless target,
//! with F3 off and on, and reads back the pixels around the screen centre. The frame is 639x479
//! so the automatic GUI scale is 1 and the F3 text stays clear of the centre. With F3 off, the
//! scene (sky, terrain, crosshair) contains no pure red, green or blue pixel near the centre, so
//! the F3-on assertions can only be satisfied by the gizmo.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::PhaseList;
use tokio::net::TcpListener;

const WIDTH: u32 = 639;
const HEIGHT: u32 = 479;
/// Half-size of the square around the centre that is searched. The first-person arms of the
/// gizmo are about 17 by 11 pixels at this size.
const REACH: i64 = 30;

/// Pure red, green and blue pixels within [`REACH`] of the centre, with their offsets.
fn centre_primaries(pixels: &[u8]) -> [Vec<(i64, i64)>; 3] {
    let (cx, cy) = (i64::from(WIDTH / 2), i64::from(HEIGHT / 2));
    let mut out: [Vec<(i64, i64)>; 3] = Default::default();
    for dy in -REACH..=REACH {
        for dx in -REACH..=REACH {
            let i = (((cy + dy) * i64::from(WIDTH) + cx + dx) * 4) as usize;
            match pixels[i..i + 3] {
                [255, 0, 0] => out[0].push((dx, dy)),
                [0, 255, 0] => out[1].push((dx, dy)),
                [0, 0, 255] => out[2].push((dx, dy)),
                _ => {}
            }
        }
    }
    out
}

#[test]
fn f3_replaces_the_crosshair_with_the_world_axes_gizmo() {
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

    // Facing south-west and 30 degrees down, so all three axes have screen length: +X runs
    // left and down, +Z left and up, +Y straight up.
    let look = MoveInput {
        yaw: 45.0,
        pitch: 30.0,
        ..MoveInput::default()
    };
    let mut frame = |client: &mut Client| {
        pump(client, &mut gpu, &mut phases, &mut camera, &look);
        pump(client, &mut gpu, &mut phases, &mut camera, &look);
        centre_primaries(&gpu.read_pixels().expect("readback"))
    };

    assert!(!client.debug.visible, "F3 starts closed");
    let [red, green, blue] = frame(&mut client);
    assert!(
        red.is_empty() && green.is_empty() && blue.is_empty(),
        "without F3 nothing near the centre is a pure primary: {red:?} {green:?} {blue:?}"
    );

    // The same frames also drove the first-person hand. Its arm sway is EntityPlayerSP's
    // tick-damped `renderArmYaw/Pitch` (it lags the turn to 45 degrees), and its lightmap
    // texel is `World.getCombinedLight` at the eye (`setLightMapFromPlayer`).
    let state = client
        .session()
        .and_then(|session| session.predictor())
        .expect("joined predictor")
        .state;
    let (arm, light) = client
        .render()
        .expect("render stack")
        .first_person
        .with(|phase| {
            (
                [
                    phase.prev_render_arm_yaw,
                    phase.render_arm_yaw,
                    phase.prev_render_arm_pitch,
                    phase.render_arm_pitch,
                ],
                phase.combined_light,
            )
        });
    assert_eq!(
        arm,
        [
            state.prev_render_arm_yaw,
            state.render_arm_yaw,
            state.prev_render_arm_pitch,
            state.render_arm_pitch,
        ]
    );
    assert!(
        state.render_arm_yaw > 0.0 && state.render_arm_yaw < 45.0,
        "the arm lags the turn: {arm:?}"
    );
    let eye = state.eye_pos();
    let sample = mc_math::BlockPos::new(
        eye.x.floor() as i32,
        (eye.y.floor() as i32).max(0),
        eye.z.floor() as i32,
    );
    let world = client.session().expect("session").world();
    assert_eq!(
        light,
        mc_render::terrain::light::combined_light(world, sample, 0)
    );

    client.debug.visible = true;
    let [red, green, blue] = frame(&mut client);
    assert!(
        client
            .render()
            .expect("render stack")
            .world_directions
            .with(|phase| phase.visible),
        "F3 must enable the gizmo phase"
    );
    assert!(!red.is_empty(), "F3 must draw the red +X arm");
    assert!(!green.is_empty(), "F3 must draw the green +Y arm");
    assert!(!blue.is_empty(), "F3 must draw the blue +Z arm");
    // Orientation, from the camera above: +Y up the screen, +X and +Z both to the left.
    assert!(
        green.iter().all(|&(_, dy)| dy <= 1),
        "green points up: {green:?}"
    );
    assert!(
        green.iter().any(|&(_, dy)| dy < -5),
        "green has length: {green:?}"
    );
    assert!(red.iter().any(|&(dx, _)| dx < -5), "red runs left: {red:?}");
    assert!(
        blue.iter().any(|&(dx, dy)| dx < -5 && dy < 0),
        "blue runs up-left: {blue:?}"
    );
    // The whole gizmo stays near the centre: the hand projection's 70 degrees, 0.1 ahead.
    for (dx, dy) in red.iter().chain(&green).chain(&blue) {
        assert!(
            dx.abs() <= 20 && dy.abs() <= 14,
            "({dx}, {dy}) is too far out"
        );
    }

    client.debug.visible = false;
    let [red, green, blue] = frame(&mut client);
    assert!(
        red.is_empty() && green.is_empty() && blue.is_empty(),
        "closing F3 removes the gizmo again"
    );

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
