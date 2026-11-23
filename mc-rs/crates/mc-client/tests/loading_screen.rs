//! The loading screen, rendered through the real pipeline.
//!
//! java: `client/LoadingScreenRenderer.setLoadingProgress` (`:119`) — a tiled
//! `options_background`, a grey 100×2 track with a green fill over it, and two shadowed strings.
//!
//! The unit tests in `app::loading` pin the geometry and the 100 ms throttle against the Java.
//! This file pins that the geometry actually reaches the framebuffer: `gui_scale = 1` makes one
//! GUI pixel one framebuffer pixel, so vanilla's coordinates can be read back directly.
//!
//! `the_progress_screen_is_presented_while_the_assets_are_still_loading` is the one that closes
//! S21's asset-load task: it drives the **real deferred startup** — `Client::new_deferred` plus
//! `step_loading`, the same two calls `ShellApp::frame` makes — and reads the framebuffer back
//! *while the model bake has not run yet*, proving the bar is on screen during the load rather
//! than after it. Before this run the assets had to be fully loaded before `Shell::run` created a
//! window, so there was nothing to present them on.

mod common;

use common::{config, gpu};
use mc_client::app::assets::{STAGE_COLORMAPS, STAGE_MODELS, STAGE_SOUNDS};
use mc_client::app::driver::Ui;
use mc_client::app::loading::{BAR_FILL_ARGB, BAR_TRACK_ARGB, BAR_WIDTH};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::{Camera, PhaseList};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

fn rgb(pixels: &[u8], x: u32, y: u32) -> [u8; 3] {
    let index = ((y * WIDTH + x) * 4) as usize;
    [pixels[index], pixels[index + 1], pixels[index + 2]]
}

fn argb_rgb(argb: u32) -> [u8; 3] {
    [
        ((argb >> 16) & 0xFF) as u8,
        ((argb >> 8) & 0xFF) as u8,
        (argb & 0xFF) as u8,
    ]
}

#[test]
fn the_progress_bar_lands_on_vanillas_exact_pixels() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    // One GUI pixel = one framebuffer pixel, so vanilla's coordinates are readable directly.
    client.settings.gui_scale = 1;

    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();

    client.ui = Ui::Loading;
    client.loading.reset_progress_and_message("Loading world");
    client.loading.display_loading_string("Baking block models");
    const PROGRESS: i32 = 40;
    client.loading.set_loading_progress(PROGRESS, 1_000);

    // Two frames: the first pulls `options_background` out of the pack stack.
    for _ in 0..2 {
        client.poll();
        client.tick(&MoveInput::default());
        client.frame(&gpu, &mut camera, 0.0);
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &camera, 0.0);
    }
    let pixels = gpu.read_pixels().expect("readback");

    // java: `k1 = k / 2 - 50`, `l1 = l / 2 + 16`, bar 100 wide and 2 tall.
    let bar_x = WIDTH as i32 / 2 - BAR_WIDTH / 2;
    let bar_y = HEIGHT as i32 / 2 + 16;
    assert_eq!((bar_x, bar_y), (110, 136));

    for offset in [0, 10, PROGRESS - 1] {
        assert_eq!(
            rgb(&pixels, (bar_x + offset) as u32, bar_y as u32),
            argb_rgb(BAR_FILL_ARGB),
            "the filled part of the bar must be vanilla's (128, 255, 128) at +{offset}"
        );
    }
    for offset in [PROGRESS, BAR_WIDTH / 2 + 20, BAR_WIDTH - 1] {
        assert_eq!(
            rgb(&pixels, (bar_x + offset) as u32, bar_y as u32),
            argb_rgb(BAR_TRACK_ARGB),
            "the unfilled track must be vanilla's (128, 128, 128) at +{offset}"
        );
    }
    // Both rows of the two-pixel-tall bar, and neither row outside it.
    assert_eq!(
        rgb(&pixels, bar_x as u32, (bar_y + 1) as u32),
        argb_rgb(BAR_FILL_ARGB)
    );
    for y in [bar_y - 1, bar_y + 2] {
        assert_ne!(
            rgb(&pixels, bar_x as u32, y as u32),
            argb_rgb(BAR_FILL_ARGB),
            "the bar is exactly two pixels tall"
        );
    }
    // And one pixel past each end is background, not bar.
    for x in [bar_x - 1, bar_x + BAR_WIDTH] {
        let pixel = rgb(&pixels, x as u32, bar_y as u32);
        assert_ne!(pixel, argb_rgb(BAR_FILL_ARGB));
        assert_ne!(pixel, argb_rgb(BAR_TRACK_ARGB));
    }
}

/// java: `displayLoadingString` calls `setLoadingProgress(-1)`, and `if (progress >= 0)` gates
/// the whole bar — so a message with no numeric progress must draw text and background only.
#[test]
fn a_message_with_no_progress_draws_no_bar() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    client.settings.gui_scale = 1;

    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();

    client.ui = Ui::Loading;
    client
        .loading
        .display_loading_string("Reloading resource packs");
    assert_eq!(client.loading.progress, -1);

    for _ in 0..2 {
        client.frame(&gpu, &mut camera, 0.0);
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &camera, 0.0);
    }
    let pixels = gpu.read_pixels().expect("readback");

    let bar_x = WIDTH as i32 / 2 - BAR_WIDTH / 2;
    let bar_y = HEIGHT as i32 / 2 + 16;
    for offset in 0..BAR_WIDTH {
        let pixel = rgb(&pixels, (bar_x + offset) as u32, bar_y as u32);
        assert_ne!(pixel, argb_rgb(BAR_FILL_ARGB), "no bar at +{offset}");
        assert_ne!(pixel, argb_rgb(BAR_TRACK_ARGB), "no track at +{offset}");
    }
    // The background is still drawn: `options_background` tinted to (64, 64, 64) is dark but not
    // the clear colour.
    assert!(
        common::drawn_pixels(&pixels, common::CLEAR_RGB) > (WIDTH * HEIGHT) as usize / 2,
        "the tiled dirt background must cover the screen"
    );
}

/// **The asset-load task.** Drive the real deferred startup and prove the progress screen is on
/// the framebuffer *before* the expensive stages have run.
///
/// This is the same sequence `ShellApp::frame` runs — `step_loading` then `build_ui`, with a
/// present in between — so what it exercises is the shipping path, not a test-only one. The
/// window itself is offscreen because that is the only kind this environment can read pixels
/// from; the real windowed run is recorded in the handoff.
#[test]
fn the_progress_screen_is_presented_while_the_assets_are_still_loading() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // The Aero Title Screen mod (on by default) draws its own loading screen (the logo and a
    // progress bar) over this one. This test is about vanilla's progress screen, so it turns the
    // mod off, which keeps vanilla's exactly (the Aero screen has its own tests in
    // `app/mods/tw_loading.rs`). The earlier "Loading screen" option (and its "Classic" choice,
    // used here before) is retired.
    std::fs::write(
        dir.path().join("mc-rs-mods.json"),
        r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#,
    )
    .expect("write the mod settings");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };

    // Only the pack stack, the language and the font — the window opens after this, not after
    // the model bake.
    let mut client = Client::new_deferred(config).expect("boot");
    client.settings.gui_scale = 1;
    assert!(client.is_loading());
    assert!(
        !client.assets.ready && !client.assets.models_installed,
        "the expensive stages must not have run yet"
    );
    assert!(
        matches!(client.ui, Ui::Loading),
        "the client sits on the loading screen, not the main menu"
    );

    let mut phases = PhaseList::new();
    client.install_loading_ui(&gpu, &mut phases, WIDTH, HEIGHT);
    assert_eq!(
        phases.names().collect::<Vec<_>>(),
        ["ui"],
        "nothing but the UI phase can exist yet: there is no atlas to build terrain from"
    );
    let mut camera = Camera::default();

    let bar_x = WIDTH as i32 / 2 - BAR_WIDTH / 2;
    let bar_y = HEIGHT as i32 / 2 + 16;

    let mut stages_seen: Vec<&'static str> = Vec::new();
    let mut frames_with_a_visible_bar = 0usize;
    let mut bar_widths: Vec<i32> = Vec::new();
    let mut frames = 0usize;

    // The frame loop, exactly as the shell runs it.
    while client.is_loading() && frames < 32 {
        frames += 1;
        if let Some(stage) = client.loading_stage() {
            if stages_seen.last() != Some(&stage) {
                stages_seen.push(stage);
            }
        }
        let assets_ready_at_frame_start = client.assets.ready;

        // `now_ms` climbs past the 100 ms redraw throttle every frame, so every stage change is
        // presented rather than swallowed. java: `LoadingScreenRenderer`'s `systemTime` guard.
        client.step_loading(&gpu, &mut phases, WIDTH, HEIGHT, frames as u64 * 200);
        client.frame(&gpu, &mut camera, 0.0);
        let frame = gpu.acquire().expect("acquire");
        phases.render(&gpu, frame.view(), &camera, 0.0);

        // Read back only while the load is genuinely unfinished, which is the whole claim.
        if !assets_ready_at_frame_start && matches!(client.ui, Ui::Loading) {
            let pixels = gpu.read_pixels().expect("readback");
            let mut width = 0;
            while width < BAR_WIDTH
                && rgb(&pixels, (bar_x + width) as u32, bar_y as u32) == argb_rgb(BAR_FILL_ARGB)
            {
                width += 1;
            }
            if width > 0 {
                frames_with_a_visible_bar += 1;
                bar_widths.push(width);
                assert_eq!(
                    rgb(&pixels, (bar_x + BAR_WIDTH - 1) as u32, bar_y as u32),
                    argb_rgb(BAR_TRACK_ARGB),
                    "the unfilled track must still be behind the fill while loading"
                );
            }
        }
    }

    assert!(!client.is_loading(), "the load must finish, not stall");
    assert!(client.assets.ready && client.assets.models_installed);
    assert!(
        client.fatal_error().is_none(),
        "{:?}",
        client.fatal_error().map(ToString::to_string)
    );

    // Every expensive stage was announced on screen before it ran.
    assert_eq!(
        stages_seen,
        [STAGE_MODELS, STAGE_COLORMAPS, STAGE_SOUNDS],
        "the three deferred stages must each get their own announced frame"
    );
    assert!(
        frames_with_a_visible_bar >= 2,
        "the progress bar must be on the framebuffer for more than one frame while loading, \
         got {frames_with_a_visible_bar}"
    );
    assert!(
        bar_widths.windows(2).all(|w| w[0] <= w[1]),
        "the bar must only ever grow: {bar_widths:?}"
    );
    assert!(
        bar_widths.last() > bar_widths.first(),
        "and it must actually advance, not sit still: {bar_widths:?}"
    );

    // Finishing the load is what installs the rest of the phases, through `FrameCtx::phases`.
    let names: Vec<&str> = phases.names().collect();
    assert!(
        names.contains(&"terrain/solid") && names.contains(&"ui"),
        "the full stack must be registered once the assets exist: {names:?}"
    );
    assert!(
        names.len() > 8,
        "every phase, not just terrain and the UI: {names:?}"
    );
    assert!(
        matches!(client.ui, Ui::MainMenu(_)),
        "and the main menu takes over, as `startGame:577` does"
    );

    // The other half of the shell contract this run added: `Minecraft.running`. Before
    // `ShellApp::keep_running` existed the main menu's Quit button had nowhere to land — the loop
    // only ended when the *window* closed or a `ShellConfig` bound was hit.
    use mc_render::ShellApp as _;
    assert!(
        client.keep_running(),
        "a running client keeps the loop alive"
    );
    client.quit();
    assert!(
        !client.keep_running(),
        "asking the client to quit must stop the shell"
    );
    let _ = MoveInput::default();
}
