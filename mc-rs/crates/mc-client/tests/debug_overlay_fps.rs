//! F3, end to end: the key press, and the frame rate the overlay reports afterwards.
//!
//! # Why this file exists
//!
//! `DebugInfo::fps` was the literal `format!("{} fps", 0)`. Pressing F3 opened the overlay and
//! every line on it was live except the first, which said `0 fps` on a client rendering
//! thousands. The counter it should have been reading already existed — [`mc_render::FrameStats`]
//! keeps vanilla's one-second `debugFPS` window — but it lives in the shell, and nothing carried
//! it across into the client's debug snapshot.
//!
//! The unit tests in `driver.rs` cover the formatting and the window roll from literals. What
//! they cannot cover is the wiring, and the wiring is the whole bug: `Client::debug_info` is
//! reached through the inherent `Client::frame`, while the frame counter is only in scope in the
//! [`ShellApp::frame`] that wraps it. `tests/common`'s `pump` calls the inherent one — so a test
//! built on `pump`, like every other client test here, would have passed against the broken
//! version too. This file drives the `ShellApp` path with a real `FrameStats` instead, which is
//! what the shipping binary does.
//!
//! java: `Minecraft.runGameLoop`'s `debugUpdateTime` block (`:1194`) feeding `GuiOverlayDebug`.

mod common;

use std::time::{Duration, Instant};

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::{FrameCtx, FrameStats, GameSettings, InputState, PhaseList, ShellApp};
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::screens::Action;
use tokio::net::TcpListener;
use winit::keyboard::KeyCode;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;

/// Long enough that vanilla's one-second window is guaranteed to have closed at least once, with
/// margin for a loaded CI machine. Until it closes, `debugFPS` is legitimately zero — vanilla
/// shows `0 fps` for the first second too.
const RUN_FOR: Duration = Duration::from_millis(1_600);

#[test]
fn f3_opens_the_overlay_and_its_first_line_reports_a_real_frame_rate() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skipping: the real assets are not present");
        return;
    };

    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let peer = client
        .runtime()
        .handle()
        .spawn(async move { accept_and_join(listener, 42).await });

    client.handle_action(Action::Connect(Box::new(ServerEntry::new(
        "Loopback",
        format!("127.0.0.1:{port}"),
    ))));
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world and mesh it",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
        },
    );
    // The dirty set is drained into the mesh queue by `Client::tick`, and `pump` is the only
    // helper here that ticks — the `ShellApp::frame` loop below deliberately does not, because
    // the shell drives ticks separately. So the upload the chunk-update assertion needs has to
    // happen now, while ticks are still running.
    // Drained explicitly rather than by a guessed frame count. GAP `A12-05` grew the work a
    // single chunk arrival creates: `markBlockRangeForRenderUpdate`'s ±1 box means loading one
    // column now also dirties its eight neighbours, so ~144 sections queue where 16 used to, and
    // a fixed 16 frames at `MESH_BUDGET` = 8 per frame no longer empties the queue. Looping until
    // it is empty keeps this test about the F3 line rather than about the mesher's throughput.
    for _ in 0..256 {
        if client.mesh_queue_len() == 0 {
            break;
        }
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    assert_eq!(
        client.mesh_queue_len(),
        0,
        "the peer's one chunk must be meshed and uploaded before the window opens"
    );

    // The overlay starts closed, exactly as `gameSettings.showDebugInfo` starts false.
    assert!(!client.debug.visible, "F3 has not been pressed yet");

    // java: `dispatchKeypresses`'s `if (k == 61)`. Delivered the way the shell delivers it.
    client.handle_raw_key(KeyCode::F3, true, false, None);
    client.handle_screen_input(&mut InputState::new(), WIDTH, HEIGHT);
    assert!(client.debug.visible, "F3 must open the overlay");

    // Now let a window close with the overlay open, on the `ShellApp` path. Reaching the world
    // meshed and uploaded the peer's chunk, so this also covers the half of the line the
    // main-menu test cannot: `renderChunksUpdated` counts sections that reached the GPU.
    let mut input = InputState::new();
    let settings = GameSettings::default();
    let mut stats = FrameStats::new();
    let started = Instant::now();
    while started.elapsed() < RUN_FOR {
        <Client as ShellApp>::frame(
            &mut client,
            &mut FrameCtx {
                gpu: &gpu,
                phases: &mut phases,
                camera: &mut camera,
                input: &mut input,
                settings: &settings,
                partial_ticks: 0.0,
                stats: &stats,
            },
        );
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &camera, 0.0);
        stats.frame();
    }

    let line = client.debug_info(&gpu).fps;
    // Printed because this is the line a human reads off the screen after pressing F3, and a run
    // of this test is the evidence that it says something. `-- --nocapture` to see it.
    eprintln!("F3 line: {line}");
    assert_eq!(
        line.split_whitespace().next().and_then(|n| n.parse().ok()),
        Some(stats.fps()),
        "the open overlay must show the shell's frame rate: {line:?}"
    );
    let updates: u32 = line
        .split(" chunk update")
        .next()
        .and_then(|head| head.rsplit('(').next())
        .and_then(|n| n.parse().ok())
        .unwrap_or_else(|| panic!("no chunk-update count in {line:?}"));
    assert!(
        updates > 0,
        "joining a world uploads sections, so the window that just closed cannot be empty: \
         {line:?}"
    );

    // Shut the peer down rather than leaving a detached task holding the socket while the
    // runtime drops. Mirrors `input_dispatch.rs`'s `Rig::finish`.
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
    let handle = client.runtime().handle().clone();
    handle.block_on(async {
        tokio::time::timeout(Duration::from_secs(10), peer)
            .await
            .expect("the peer must finish")
            .expect("the peer task must not panic")
    });
    drop(dir);
}

/// The regression this file is named for, isolated from the join: drive the **`ShellApp`** frame
/// path with a real [`FrameStats`] and read the line the overlay would draw.
///
/// No session is needed — the FPS line is machine state, not world state — so this runs against
/// a client sitting at the main menu, which keeps it fast and independent of the loopback peer.
#[test]
fn the_fps_line_reports_the_shells_counter_rather_than_a_hardcoded_zero() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        eprintln!("skipping: no GPU adapter available");
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // vanilla's title screen: this measures the frame counter on its frames
    // (the Aero one builds its greenhouse on its first frame).
    common::vanilla_title_screen(dir.path());
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skipping: the real assets are not present");
        return;
    };

    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = mc_client::app::driver::menu_camera();
    let mut input = InputState::new();
    let settings = GameSettings::default();
    // Pinned on **`client.settings`**, not on the `GameSettings` handed to `FrameCtx` below.
    // Those are two different values: the shell is given a clone at startup and the client keeps
    // the original, which is the one the options screen writes and `options.txt` round-trips —
    // so it is the one the debug line reads, exactly as vanilla reads its single
    // `this.gameSettings`. Setting the copy instead is what the first run of this test did, and
    // it reported ` vsync` from the client's untouched default while the copy said otherwise.
    client.settings.limit_framerate = 120;
    client.settings.enable_vsync = false;
    client.settings.fancy_graphics = true;
    client.settings.clouds = 2;
    client.settings.use_vbo = false;

    // Nothing has been counted yet, so this is vanilla's genuine first-second reading.
    assert!(
        client.debug_info(&gpu).fps.starts_with("0 fps "),
        "an unstarted counter reads zero: {:?}",
        client.debug_info(&gpu).fps
    );

    // The shell's loop, reduced to the two calls that matter: hand the app a `FrameCtx` carrying
    // the counter, then record the frame once it has been presented — the same order as
    // `Shell::render_frame`, so the app sees the counter as of the previous frame.
    let mut stats = FrameStats::new();
    let started = Instant::now();
    let mut frames = 0_u64;
    while started.elapsed() < RUN_FOR {
        // Fully qualified: the inherent `Client::frame` shadows the trait method, and it is the
        // trait one — the only place `FrameStats` is in scope — that this test is about.
        <Client as ShellApp>::frame(
            &mut client,
            &mut FrameCtx {
                gpu: &gpu,
                phases: &mut phases,
                camera: &mut camera,
                input: &mut input,
                settings: &settings,
                partial_ticks: 0.0,
                stats: &stats,
            },
        );
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(&gpu, frame.view(), &camera, 0.0);
        stats.frame();
        frames += 1;
    }

    assert!(
        stats.fps_updates() >= 1,
        "the run must be long enough to close a window"
    );
    let line = client.debug_info(&gpu).fps;

    // The substantive claim: a number, and the *shell's* number, not a placeholder.
    let reported: u32 = line
        .split_whitespace()
        .next()
        .expect("the line starts with the frame count")
        .parse()
        .unwrap_or_else(|_| panic!("the line must start with an integer: {line:?}"));
    assert_eq!(
        reported,
        stats.fps(),
        "the overlay must report the shell's completed window, not its own arithmetic: {line:?}"
    );
    assert!(
        reported > 0,
        "{frames} frames in {RUN_FOR:?} cannot be 0 fps: {line:?}"
    );

    // And the rest of the line is vanilla's, with the settings pinned above.
    assert!(
        line.ends_with(" chunk updates) T: 120 fancy-clouds"),
        "unexpected tail: {line:?}"
    );

    drop(dir);
}
