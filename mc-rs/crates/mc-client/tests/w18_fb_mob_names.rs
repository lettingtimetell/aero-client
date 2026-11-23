//! Nametags' "Always show mob names" through the real client.
//!
//! Replays the synthetic V1 capture (`optimization/fixtures/generated-light-v2/v1.mccap`: a
//! grass floor and the viewer at `(0.5, 64, -18)` looking south) through `Client`'s full frame
//! path on a headless GPU, removes its players, and puts a row of mobs six blocks in front of the
//! viewer:
//!
//! | x    | entity                                                                  |
//! |------|-------------------------------------------------------------------------|
//! | -5.5 | zombie named "Guard"                                                    |
//! | -3.5 | pig named "Wilbur", CustomNameVisible set (1.8.9 ignores it on a mob)   |
//! | -1.5 | zombie without a name                                                   |
//! |  0.5 | villager named "Shopkeeper" + an invisible hologram stand on the same spot |
//! |  2.5 | zombie named "Bob" + a hologram above it that says "Bob" with his health |
//! |  4.5 | invisible zombie named "Ghost"                                          |
//! |  6.5 | cow named "Daisy" carrying a hologram stand as its rider                |
//!
//! The labels are counted from the nameplate phase's backdrops (one quad, six vertices, per
//! label). Vanilla (the defaults, and the Nametags mod on with the option off) labels the three
//! holograms. The option adds Guard and Wilbur only: the villager's and Bob's labels would
//! double their holograms, Ghost is invisible and Daisy has a rider. Without the holograms the
//! same option labels all five named, visible mobs (the control that shows the three were
//! dropped for the hologram, not for some other reason).
//!
//! `MC_W18_FB_PNG=<dir>` saves the frames. Skips (loudly) without an adapter, the vanilla jar or
//! the capture.

mod common;

use std::path::{Path, PathBuf};

use common::{config, gpu, pump};
use mc_client::app::campath::CameraScript;
use mc_client::app::Client;
use mc_client::sync::entities::TrackedKind;
use mc_data::EntityType;
use mc_math::DVec3;
use mc_physics::state::MoveInput;
use mc_render::{Camera, Gpu, PhaseList};
use mc_world::{Metadata, MetadataValue};

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
/// Frames of the capture before the scene is built (it spawns the world and the viewer).
const WARMUP: usize = 100;
/// Frames after the scene is built, so the new entities are ticked and drawn.
const SETTLE: usize = 4;
/// The row's distance in front of the viewer (z).
const ROW_Z: f64 = -12.0;

const DEFAULTS: &str = r#"{"format":1,"mods":{"aero_title":{"enabled":false}}}"#;
const NAMETAGS_ONLY: &str = r#"{"format":1,"mods":{"aero_title":{"enabled":false},
    "nametags":{"enabled":true}}}"#;
const ALWAYS: &str = r#"{"format":1,"mods":{"aero_title":{"enabled":false},
    "nametags":{"enabled":true,"options":{"always_mob_names":true}}}}"#;

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

/// Watched metadata: flags (index 0), custom name (2) and CustomNameVisible (3).
fn watched(flags: i8, name: &str, visible: bool) -> Metadata {
    let mut metadata = Metadata::new();
    metadata.set(0, MetadataValue::Byte(flags));
    metadata.set(2, MetadataValue::Str(name.to_owned()));
    metadata.set(3, MetadataValue::Byte(i8::from(visible)));
    metadata
}

const INVISIBLE: i8 = 0x20;

/// One entity of the scene: id, type, feet, metadata.
type Spawn = (i32, EntityType, (f64, f64, f64), Metadata);

fn scene(holograms: bool) -> Vec<Spawn> {
    let at = |x: f64, y: f64| (x, 64.0 + y, ROW_Z);
    let mut spawns = vec![
        (
            500,
            EntityType::Zombie,
            at(-5.5, 0.0),
            watched(0, "\u{a7}eGuard", false),
        ),
        (
            501,
            EntityType::Pig,
            at(-3.5, 0.0),
            watched(0, "Wilbur", true),
        ),
        (
            502,
            EntityType::Zombie,
            at(-1.5, 0.0),
            watched(0, "", false),
        ),
        (
            503,
            EntityType::Villager,
            at(0.5, 0.0),
            watched(0, "Shopkeeper", false),
        ),
        (
            504,
            EntityType::Zombie,
            at(2.5, 0.0),
            watched(0, "Bob", false),
        ),
        (
            505,
            EntityType::Zombie,
            at(4.5, 0.0),
            watched(INVISIBLE, "Ghost", false),
        ),
        (
            506,
            EntityType::Cow,
            at(6.5, 0.0),
            watched(0, "Daisy", false),
        ),
    ];
    if holograms {
        spawns.extend([
            // On the villager's spot: its label would print over this one.
            (
                510,
                EntityType::ArmorStand,
                at(0.5, -0.025),
                watched(INVISIBLE, "\u{a7}a\u{a7}lITEM SHOP", true),
            ),
            // Above Bob, already saying his name.
            (
                511,
                EntityType::ArmorStand,
                at(2.5, 0.6),
                watched(INVISIBLE, "\u{a7}cBob \u{a7}a20\u{a7}c\u{2764}", true),
            ),
            // Daisy's rider (attached below).
            (
                512,
                EntityType::ArmorStand,
                at(6.5, 0.0),
                watched(INVISIBLE, "\u{a7}bDaisy the cow", true),
            ),
        ]);
    }
    spawns
}

/// The frame and the label count after [`WARMUP`] capture frames, the scene and [`SETTLE`] more
/// frames, with `mods_json` as the settings file.
fn render(
    gpu: &mut Gpu,
    capture: &Path,
    mods_json: &str,
    holograms: bool,
) -> Option<(Vec<u8>, usize)> {
    let dir = tempfile::tempdir().expect("tempdir");
    std::fs::write(dir.path().join(mc_mods::SETTINGS_FILE_NAME), mods_json)
        .expect("write the scratch mod settings");
    let mut config = config(dir.path(), "FbViewer")?;
    config.replay = Some(capture.to_path_buf());
    config.replay_step_micros = 50_000;
    config.replay_camera = CameraScript::Fixed;
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    let mut phases = PhaseList::new();
    client.install(gpu, &mut phases, WIDTH, HEIGHT);
    let mut camera = Camera::default();
    client.connect("replay", 0);
    let input = MoveInput::default();
    for _ in 0..WARMUP {
        pump(&mut client, gpu, &mut phases, &mut camera, &input);
    }
    {
        let session = client.session_mut().expect("a replay session");
        let state = session.state_mut();
        let players: Vec<i32> = state
            .entities
            .iter()
            .filter(|(_, tracked)| tracked.uuid.is_some())
            .map(|(id, _)| id)
            .collect();
        assert!(!players.is_empty(), "the capture's players are spawned");
        for id in players {
            state.entities.despawn(&mut state.world, id);
        }
        for (id, kind, (x, y, z), metadata) in scene(holograms) {
            state.entities.spawn(
                &mut state.world,
                id,
                TrackedKind::Type(kind),
                None,
                [
                    (x * 32.0).round() as i32,
                    (y * 32.0).round() as i32,
                    (z * 32.0).round() as i32,
                ],
                180.0,
                0.0,
                DVec3::ZERO,
                metadata,
            );
        }
        if holograms {
            let attached = state.entities.apply_entity_attach(
                &mut state.world,
                &mc_protocol::play::s2c::EntityAttach {
                    entity_id: 512,
                    vehicle_entity_id: 506,
                    attach_type: mc_protocol::play::s2c::AttachType::Vehicle,
                },
            );
            assert!(attached, "the hologram rides the cow");
        }
    }
    for _ in 0..SETTLE {
        pump(&mut client, gpu, &mut phases, &mut camera, &input);
    }
    let (through_walls, in_front) = client
        .render()
        .expect("installed")
        .nameplates
        .with(|phase| phase.vertex_counts());
    let backdrop_vertices = through_walls + in_front;
    assert_eq!(backdrop_vertices % 6, 0, "whole backdrop quads");
    Some((gpu.read_pixels().expect("read back"), backdrop_vertices / 6))
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
fn always_show_mob_names_labels_named_mobs_without_doubling_holograms() {
    let Some(capture) = capture() else {
        eprintln!("SKIP: the V1 capture is not in this checkout");
        return;
    };
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let Some((vanilla, vanilla_labels)) = render(&mut gpu, &capture, DEFAULTS, true) else {
        eprintln!("SKIP: no vanilla jar in this checkout");
        return;
    };
    let (styled, styled_labels) = render(&mut gpu, &capture, NAMETAGS_ONLY, true).unwrap();
    let (always, always_labels) = render(&mut gpu, &capture, ALWAYS, true).unwrap();
    let (_, bare_vanilla_labels) = render(&mut gpu, &capture, DEFAULTS, false).unwrap();
    let (bare, bare_labels) = render(&mut gpu, &capture, ALWAYS, false).unwrap();
    save("mob-names-vanilla.png", &vanilla);
    save("mob-names-nametags-on-option-off.png", &styled);
    save("mob-names-always.png", &always);
    save("mob-names-always-no-holograms.png", &bare);

    assert_eq!(vanilla_labels, 3, "vanilla: the three holograms only");
    assert_eq!(
        styled_labels, vanilla_labels,
        "the Nametags mod alone changes the style, not which labels exist"
    );
    assert_eq!(
        always_labels,
        vanilla_labels + 2,
        "the option adds Guard and Wilbur; the villager and Bob would double a hologram, \
         Ghost is invisible and Daisy is ridden"
    );
    assert_eq!(
        bare_vanilla_labels, 0,
        "vanilla never labels an unpointed mob"
    );
    assert_eq!(
        bare_labels, 5,
        "without holograms the option labels all five visible named mobs"
    );
    assert_ne!(vanilla, always, "the new labels are drawn");
}
