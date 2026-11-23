//! The entity mods through the real client.
//!
//! Replays the synthetic `E1-50` capture (50 remote players laid out in front of the viewer,
//! `optimization/fixtures/generated-light-v2/e1-50.mccap`) through `Client`'s full frame path on a
//! headless GPU, once with the mods at their defaults and once with Hitboxes and Nametags switched
//! on through `mc-rs-mods.json` in the scratch game dir, exactly as a user's settings file would.
//! This is the wiring check the unit and phase tests cannot give: settings file -> snapshot ->
//! the hooks in `Client::frame` / `upload_entities` -> the render phases.
//!
//! `MC_X6_E2E_PNG=<dir>` saves both frames. Skips (loudly) without an adapter, the vanilla jar or
//! the capture.

mod common;

use std::path::{Path, PathBuf};

use common::{config, gpu, pump};
use mc_client::app::campath::CameraScript;
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, PhaseList};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
const FRAMES: usize = 120;

fn capture() -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .join("optimization")
        .join("fixtures")
        .join("generated-light-v2")
        .join("e1-50.mccap");
    path.is_file().then_some(path)
}

/// The last of [`FRAMES`] frames of the capture, with `mods_json` as the settings file.
fn render(gpu: &mut Gpu, capture: &Path, mods_json: &str) -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join(mc_mods::SETTINGS_FILE_NAME), mods_json)
        .expect("write the scratch mod settings");
    let mut config = config(dir.path(), "E1Viewer")?;
    config.replay = Some(capture.to_path_buf());
    config.replay_step_micros = 50_000;
    config.replay_camera = CameraScript::Fixed;
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(gpu, &mut phases, WIDTH, HEIGHT);
    assert!(
        phases.names().any(|name| name == "hitboxes"),
        "the Hitboxes phase is registered by Client::install"
    );
    let mut camera = Camera::default();
    client.connect("replay", 0);
    let input = MoveInput::default();
    for _ in 0..FRAMES {
        pump(&mut client, gpu, &mut phases, &mut camera, &input);
    }
    // F3+B was never pressed: the vanilla debug lines stay off either way.
    assert!(!client
        .render()
        .expect("installed")
        .entities
        .with(|phase| phase.debug_bounding_box()));
    Some(gpu.read_pixels().expect("read back"))
}

/// Entity vertices queued after the capture's frames plus one more, with `mods_json` as the
/// settings file, after giving the player nearest the middle of the view six stuck arrows
/// (watched index 9) and, when `invisible`, the invisibility flag (index 0 bit 5).
fn entity_vertices_with_arrows(
    gpu: &mut Gpu,
    capture: &Path,
    mods_json: &str,
    invisible: bool,
) -> Option<usize> {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join(mc_mods::SETTINGS_FILE_NAME), mods_json)
        .expect("write the scratch mod settings");
    let mut config = config(dir.path(), "E1Viewer")?;
    config.replay = Some(capture.to_path_buf());
    config.replay_step_micros = 50_000;
    config.replay_camera = CameraScript::Fixed;
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();
    client.connect("replay", 0);
    let input = MoveInput::default();
    for _ in 0..FRAMES {
        pump(&mut client, gpu, &mut phases, &mut camera, &input);
    }
    let session = client.session_mut().expect("a replay session");
    let target = session
        .state()
        .entities
        .iter()
        .filter(|(_, tracked)| tracked.uuid.is_some())
        .map(|(_, tracked)| {
            let at = tracked.render_position(&session.state().world, 0.0);
            let off_axis = (at.x - 0.5).abs() + (at.z + 8.0).abs();
            (off_axis, tracked.entity)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, entity)| entity)
        .expect("the capture's players are spawned");
    {
        let world = &mut session.state_mut().world;
        let mut metadata = world
            .ecs
            .get::<&mut mc_world::Metadata>(target)
            .expect("a watched player");
        metadata.set(9, mc_world::MetadataValue::Byte(6));
        if invisible {
            metadata.set(0, mc_world::MetadataValue::Byte(0x20));
        }
    }
    pump(&mut client, gpu, &mut phases, &mut camera, &input);
    Some(
        client
            .render()
            .expect("installed")
            .entities
            .with(|phase| phase.vertex_count()),
    )
}

/// The Stuck Arrows switches through the real `upload_entities`: on (the default) draws the six
/// arrows on a visible player and hides them on an invisible one; off hides them everywhere;
/// "Only invisible players" selects invisible players and hides arrows on visible players.
#[test]
fn stuck_arrows_follow_the_mod_through_the_real_entity_pass() {
    let Some(capture) = capture() else {
        eprintln!("SKIP: the E1-50 capture is not in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let base = r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#;
    let off = r#"{"format":1,"mods":{"aero_title":{"enabled":false},
        "stuck_arrows":{"enabled":false}}}"#;
    let invisible_only = r#"{"format":1,"mods":{"aero_title":{"enabled":false},
        "stuck_arrows":{"enabled":true,"options":{"on_invisible":true}}}}"#;
    let Some(visible_on) = entity_vertices_with_arrows(&mut gpu, &capture, base, false) else {
        eprintln!("SKIP: no vanilla jar in this checkout");
        return;
    };
    let visible_off = entity_vertices_with_arrows(&mut gpu, &capture, off, false).unwrap();
    let invisible_default = entity_vertices_with_arrows(&mut gpu, &capture, base, true).unwrap();
    let invisible_vanilla =
        entity_vertices_with_arrows(&mut gpu, &capture, invisible_only, true).unwrap();
    let visible_filtered =
        entity_vertices_with_arrows(&mut gpu, &capture, invisible_only, false).unwrap();
    assert_eq!(
        visible_filtered, visible_off,
        "invisible-only hides arrows on visible players"
    );
    assert!(
        visible_on > visible_off,
        "arrows drawn by default, hidden with the mod off: {visible_on} vs {visible_off}"
    );
    assert!(
        invisible_vanilla > invisible_default,
        "on an invisible player: hidden by default, drawn with 'On invisible players': \
         {invisible_default} vs {invisible_vanilla}"
    );
    // The six arrows are the same geometry on the visible and the invisible player.
    assert_eq!(
        visible_on - visible_off,
        invisible_vanilla - invisible_default,
        "the difference is exactly the arrows"
    );
}

#[test]
fn hitboxes_and_nametags_reach_the_frame_through_the_settings_file() {
    let Some(capture) = capture() else {
        eprintln!("SKIP: the E1-50 capture is not in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let defaults = r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#;
    let mods_on = r##"{"format":1,"mods":{"aero_title":{"enabled":false},
        "hitboxes":{"enabled":true,"options":{"color":"#FFFF00FF","width":3.0}},
        "nametags":{"enabled":true}}}"##;
    let Some(off) = render(&mut gpu, &capture, defaults) else {
        eprintln!("SKIP: no vanilla jar in this checkout");
        return;
    };
    let on = render(&mut gpu, &capture, mods_on).expect("the same checkout");
    if let Some(dir) = std::env::var_os("MC_X6_E2E_PNG") {
        let dir = PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("output dir");
        for (name, pixels) in [("mods-off.png", &off), ("mods-on.png", &on)] {
            image::save_buffer(
                dir.join(name),
                pixels,
                WIDTH,
                HEIGHT,
                image::ExtendedColorType::Rgba8,
            )
            .expect("write the PNG");
        }
    }
    // Magenta (#FF00FF) hitbox lines: red and blue high, green low. Nothing in the lobby scene
    // is that colour with the mods at their defaults.
    let magenta = |px: &[u8]| {
        px.chunks_exact(4)
            .filter(|p| p[0] > 200 && p[1] < 80 && p[2] > 200)
            .count()
    };
    let (before, after) = (magenta(&off), magenta(&on));
    assert!(before < 20, "no magenta without the mod: {before}");
    assert!(
        after > 500,
        "the players' hitboxes are drawn: {after} magenta pixels"
    );
    // The nametag shadow: the labels gain dark grey pixels next to their white ink.
    let grey = |px: &[u8]| {
        px.chunks_exact(4)
            .filter(|p| (50..=75).contains(&p[0]) && p[0] == p[1] && p[1] == p[2])
            .count()
    };
    assert!(
        grey(&on) > grey(&off),
        "the nametag shadow adds grey: {} -> {}",
        grey(&off),
        grey(&on)
    );
}
