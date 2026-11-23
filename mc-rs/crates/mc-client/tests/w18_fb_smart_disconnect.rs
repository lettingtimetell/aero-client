//! Confirm Disconnect's question in a real client frame.
//!
//! Replays the synthetic V1 capture through `Client`'s full frame path on a headless GPU, opens
//! the pause menu over it, and draws three frames: the pause menu, the question just after its
//! Disconnect was clicked (buttons still waiting out their arm time) and the question armed (the
//! client's own ticks count it down). A fourth frame is the same question asked before joining
//! another server from the pause menu's server list. The clicks themselves, the tick hook and
//! what reaches the network are tested in `app/mods/w18_fb.rs`; this file is the pictures and
//! the draw wiring.
//!
//! `MC_W18_FB_PNG=<dir>` saves the frames. Skips (loudly) without an adapter, the vanilla jar or
//! the capture.

mod common;

use std::path::{Path, PathBuf};

use common::{config, gpu, pump};
use mc_client::app::campath::CameraScript;
use mc_client::app::driver::Ui;
use mc_client::app::Client;
use mc_mods::mods::smart_disconnect::ARM_TICKS;
use mc_physics::state::MoveInput;
use mc_render::{Camera, PhaseList};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::pause::PauseScreen;
use mc_ui::widget::ScaledResolution;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
const WARMUP: usize = 100;

fn capture() -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .join("optimization")
        .join("fixtures")
        .join("generated-light-v2")
        .join("v1.mccap");
    path.is_file().then_some(path)
}

fn save(name: &str, pixels: &[u8]) {
    let Some(dir) = std::env::var_os("MC_W18_FB_PNG") else {
        return;
    };
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("output dir");
    image::save_buffer(
        dir.join(name),
        pixels,
        WIDTH,
        HEIGHT,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write the PNG");
}

#[test]
fn the_disconnect_question_is_drawn_over_the_game_and_arms_with_the_clients_ticks() {
    let Some(capture) = capture() else {
        eprintln!("SKIP: the V1 capture is not in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(
        dir.path().join(mc_mods::SETTINGS_FILE_NAME),
        r#"{"format":1,"mods":{"aero_title":{"enabled":false},
            "smart_disconnect":{"enabled":true}}}"#,
    )
    .expect("write the scratch mod settings");
    let Some(mut config) = config(dir.path(), "FbViewer") else {
        eprintln!("SKIP: no vanilla jar in this checkout");
        return;
    };
    config.replay = Some(capture);
    config.replay_step_micros = 50_000;
    config.replay_camera = CameraScript::Fixed;
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();
    client.connect("replay", 0);
    let input = MoveInput::default();
    for _ in 0..WARMUP {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &input);
    }

    // The pause menu, as Escape opens it.
    client.show_screen(Some(Ui::Pause(Box::default())));
    pump(&mut client, &mut gpu, &mut phases, &mut camera, &input);
    let pause = gpu.read_pixels().expect("read back");

    // Its Disconnect clicked with Confirm Disconnect on (the client sets this before each click).
    let ctx = ScreenCtx::new(ScaledResolution::new(WIDTH, HEIGHT, 0, false), 0.0);
    let mut lang = mc_assets::Language::new("en_US");
    lang.insert("menu.disconnect", "Disconnect");
    lang.insert("gui.cancel", "Cancel");
    let mut screen = PauseScreen::new();
    screen.init(&ctx, &lang);
    screen.set_confirm_disconnect(true);
    let (x, y) = (ctx.width / 2 - 100 + 1, ctx.height / 4 + 104 + 1);
    assert_eq!(
        screen.mouse_clicked(x, y, 0),
        mc_ui::screens::Action::None,
        "asking, not leaving"
    );
    assert!(screen.is_confirming_disconnect());
    client.show_screen(Some(Ui::Pause(Box::new(screen))));
    pump(&mut client, &mut gpu, &mut phases, &mut camera, &input);
    let waiting = gpu.read_pixels().expect("read back");

    // The client's ticks arm it (one per pump).
    for _ in 0..ARM_TICKS {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &input);
    }
    let Ui::Pause(screen) = &client.ui else {
        panic!("the pause menu is still open");
    };
    assert!(screen.confirmation_armed(), "armed by the client's ticks");
    let armed = gpu.read_pixels().expect("read back");
    assert!(client.session().is_some(), "still in the game");

    save("smart-disconnect-1-pause.png", &pause);
    save("smart-disconnect-2-question-waiting.png", &waiting);
    save("smart-disconnect-3-question-armed.png", &armed);
    assert_ne!(pause, waiting, "the question replaces the pause menu");
    assert_ne!(waiting, armed, "the buttons light up when armed");

    // The other way out: the pause menu's server list, then a join (what a double click on a row
    // sends). The join waits behind the same question, over the list.
    client.show_screen(Some(Ui::Pause(Box::default())));
    client.handle_action(mc_ui::screens::Action::OpenMultiplayer);
    assert!(matches!(client.ui, Ui::Multiplayer(_)));
    client.handle_action(mc_ui::screens::Action::Connect(Box::new(
        mc_ui::screens::multiplayer::ServerEntry::new("Test server", "192.0.2.1:25565"),
    )));
    assert!(
        matches!(client.ui, Ui::ConfirmLeaveServer(..)),
        "the join waits for an answer"
    );
    for _ in 0..=ARM_TICKS {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &input);
    }
    assert!(
        matches!(client.ui, Ui::ConfirmLeaveServer(..)),
        "still asking: nothing answers it but a click or Escape"
    );
    assert!(client.session().is_some(), "still in the game");
    save(
        "smart-disconnect-4-join-question.png",
        &gpu.read_pixels().expect("read back"),
    );
}
