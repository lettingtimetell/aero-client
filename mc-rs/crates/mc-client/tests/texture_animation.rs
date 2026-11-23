//! Animated atlas sprites reach the screen.
//!
//! The bug report: "Water and lava dont have animated textures". java: `Minecraft.runTick`
//! -> `TextureManager.tick` -> `TextureMap.updateAnimations`, which re-uploads each animated
//! sprite's rectangle (every mip level) once per client tick.
//!
//! This drives the real client against the loopback double (`tests/common/mod.rs`), turns the
//! far part of the stone floor into a still fluid, freezes the day/night cycle at noon, hides the
//! HUD and hand (F1) and holds the camera still, then compares frames a few client ticks apart.
//!
//! Two startups are covered, because they differ in exactly the way the bug needed:
//! * [`Client::new_deferred`] is what the game binary runs. The shell keeps running 20 Hz ticks
//!   while `step_loading` walks the asset stages one per frame, so ticks land after the model
//!   bake is installed but before the assets are `ready`.
//! * [`Client::new`] loads everything before the first tick; every older test used it, which is
//!   why the defect never showed up in one.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK, FLOOR_Y};
use mc_client::app::Client;
use mc_data::{block_by_name, BlockState};
use mc_physics::state::MoveInput;
use mc_protocol::play::s2c;
use mc_render::{Camera, Gpu, PhaseList};
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const LOCAL_ENTITY_ID: i32 = 42;

/// Replace the floor's far rows (`z = 9..16`, what the 45-degree spawn camera looks at) with
/// `fluid` at meta 0: a source block, which a client never flows on its own.
fn fluid_floor(fluid: &str) -> s2c::MultiBlockChange {
    // S22 carries `Block.BLOCK_STATE_IDS` (`id << 4 | meta`), which is `BlockState::id`.
    let state = i32::from(BlockState::new(block_by_name(fluid).expect(fluid), 0).id);
    let mut changes = Vec::new();
    for z in 9..16u8 {
        for x in 0..16u8 {
            changes.push(s2c::MultiBlockChangeEntry {
                rel_x: x,
                y: FLOOR_Y as u8,
                rel_z: z,
                block_state_id: state,
            });
        }
    }
    s2c::MultiBlockChange {
        chunk_x: CHUNK.x,
        chunk_z: CHUNK.z,
        changes,
    }
}

#[derive(Clone, Copy, Debug)]
enum Startup {
    /// `Client::new_deferred` + `step_loading`, with a tick before every frame (the binary).
    Deferred,
    /// `Client::new` (every older integration test).
    Eager,
}

struct Scene {
    client: Client,
    gpu: Gpu,
    phases: PhaseList,
    camera: Camera,
    input: MoveInput,
}

impl Scene {
    fn pump(&mut self) {
        pump(
            &mut self.client,
            &mut self.gpu,
            &mut self.phases,
            &mut self.camera,
            &self.input,
        );
    }

    fn frame(&mut self) -> Vec<u8> {
        self.gpu.read_pixels().expect("readback")
    }
}

/// Start the client, join, turn the floor into `fluid`, and let the re-mesh land.
fn fluid_scene(startup: Startup, fluid: &'static str) -> Scene {
    let mut gpu = gpu(WIDTH, HEIGHT).expect("this test needs the real GPU");
    let dir = Box::leak(Box::new(tempfile::tempdir().expect("tempdir")));
    let config = config(dir.path(), "Dev").expect("this test needs the real 1.8.9 assets");
    let mut phases = PhaseList::new();
    let mut camera = mc_client::app::driver::menu_camera();
    let mut client = match startup {
        Startup::Eager => {
            let mut client =
                Client::new(config, &mut |_, _| {}).expect("the real assets must load");
            client.install(&gpu, &mut phases, WIDTH, HEIGHT);
            client
        }
        Startup::Deferred => {
            let mut client = Client::new_deferred(config).expect("boot");
            client.install_loading_ui(&gpu, &mut phases, WIDTH, HEIGHT);
            let mut frames = 0u64;
            // `Shell::draw`: every elapsed 20 Hz tick, then one frame, which steps the loader.
            // A model bake (or cache read) takes far longer than 50 ms, so in the real binary
            // ticks always run between the model stage and the end of the load.
            while client.is_loading() && frames < 32 {
                frames += 1;
                client.tick(&MoveInput::default());
                client.step_loading(&gpu, &mut phases, WIDTH, HEIGHT, frames * 200);
                client.frame(&gpu, &mut camera, 0.0);
                let frame = gpu.acquire().expect("acquire offscreen frame");
                phases.render(&gpu, frame.view(), &camera, 0.0);
            }
            assert!(!client.is_loading(), "the deferred load must finish");
            assert!(client.assets.ready && client.assets.models_installed);
            client
        }
    };
    // F1: no HUD or hand in the compared frames (the hunger bar shakes every tick).
    client.set_hide_gui_for_capture(true);

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let floor = fluid_floor(fluid);
    client.runtime().handle().spawn(async move {
        let mut peer = accept_and_join(listener, LOCAL_ENTITY_ID).await;
        // Noon, frozen: a negative time of day is vanilla's `doDaylightCycle=false`, so the sky
        // and lightmap cannot drift between the frames compared below.
        peer.send(&s2c::TimeUpdate {
            total_world_time: 6000,
            world_time: -6000,
        })
        .await;
        peer.send(&floor).await;
        peer.record_until_closed().await;
        peer
    });
    client.handle_action(mc_ui::screens::Action::Connect(Box::new(
        mc_ui::screens::multiplayer::ServerEntry::new("Loopback", format!("127.0.0.1:{port}")),
    )));

    let input = MoveInput {
        pitch: common::SPAWN_PITCH,
        ..MoveInput::default()
    };
    let fluid_block = block_by_name(fluid).expect(fluid);
    let probe = mc_math::BlockPos::new(8, FLOOR_Y as i32, 12);
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &input,
        Duration::from_secs(30),
        "the fluid floor",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|s| s.world().block_at(probe).block() == fluid_block)
        },
    );
    let mut scene = Scene {
        client,
        gpu,
        phases,
        camera,
        input,
    };
    // Let the re-mesh, the relight and the fog-brightness smoothing settle.
    for _ in 0..100 {
        scene.pump();
        std::thread::sleep(Duration::from_millis(2));
    }
    // Nothing is compared until the whole floor is on screen. In a few runs of the deferred
    // startup the floor section stayed missing (15 of 16 sections uploaded, a black frame), with
    // or without this lane's fix; a black frame would turn a negative result into noise.
    let mut extra = 0;
    while !floor_on_screen(&scene.frame()) {
        assert!(
            extra < 400,
            "{startup:?}-{fluid}: the floor never rendered ({} sections uploaded)",
            scene.client.uploaded_section_count()
        );
        scene.pump();
        std::thread::sleep(Duration::from_millis(2));
        extra += 1;
    }
    if extra > 0 {
        eprintln!("{startup:?}-{fluid}: the floor appeared {extra} pumps late");
    }
    scene
}

/// Whether both bands [`changes`] reads are lit. No fluid or stone texel is near black, so 95%
/// of each band must not be (lava's dark smoke particles may be).
fn floor_on_screen(pixels: &[u8]) -> bool {
    let h = HEIGHT as usize;
    let lit = |rows: std::ops::Range<usize>| {
        let row_bytes = WIDTH as usize * 4;
        let band = &pixels[rows.start * row_bytes..rows.end * row_bytes];
        let lit = band
            .chunks_exact(4)
            .filter(|p| p[..3].iter().any(|&c| c > 24))
            .count();
        lit * 20 >= band.len() / 4 * 19
    };
    lit(h * 2 / 5..h * 2 / 3) && lit(h * 17 / 20..h)
}

/// `(fluid band, stone band)`: the pixels whose largest channel moved by at least `threshold`
/// between `a` and `b`, in rows 40%..67% of the frame (the fluid, `z >= 9`) and in rows
/// 85%..100% (the stone the player stands on, `z <= 8`).
fn changes(a: &[u8], b: &[u8], threshold: u8) -> (usize, usize) {
    let count = |rows: std::ops::Range<usize>| {
        rows.flat_map(|y| (0..WIDTH as usize).map(move |x| (y * WIDTH as usize + x) * 4))
            .filter(|&at| (0..3).any(|c| a[at + c].abs_diff(b[at + c]) >= threshold))
            .count()
    };
    let h = HEIGHT as usize;
    (count(h * 2 / 5..h * 2 / 3), count(h * 17 / 20..h))
}

/// The installed bake is process-global: a second client's `install`/`reinstall` bumps the
/// generation every client watches, which would rebuild a dropped animator and hide the defect.
/// The tests in this binary therefore run one at a time.
static ONE_CLIENT_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The mean of the stone band's red channel, to watch the scene's overall brightness.
fn stone_mean(pixels: &[u8]) -> f64 {
    let h = HEIGHT as usize;
    let start = h * 17 / 20 * WIDTH as usize * 4;
    let reds: Vec<u8> = pixels[start..].chunks_exact(4).map(|p| p[0]).collect();
    reds.iter().map(|&r| f64::from(r)).sum::<f64>() / reds.len() as f64
}

/// `(fluid band, stone band)` changed-pixel counts, see [`changes`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Moved {
    /// Every capture against the reference frame, in order.
    captures: Vec<(usize, usize)>,
    /// The pair of consecutive captures (6 or 10 ticks apart) that differs most.
    consecutive: (usize, usize),
}

/// Frames 6, 16, 26 and 36 client ticks after a reference frame (java: both still fluids have
/// `frametime: 2`; `lava_still` plays 0..19 and back, so a single short gap can land on two
/// equal frames around a turn), with a no-tick control first.
///
/// The whole scene also brightens slowly over ticks (about 0.15 units per tick on the stone,
/// whatever the fluid and the startup, so not part of this lane's fix); consecutive captures
/// keep that drift near one unit for subtle sprites like water.
fn animate(startup: Startup, fluid: &'static str, threshold: u8) -> Moved {
    let _one_client = ONE_CLIENT_AT_A_TIME
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let name = format!("{startup:?}-{fluid}");
    let mut scene = fluid_scene(startup, fluid);
    scene.pump();
    let before = scene.frame();
    let player = scene
        .client
        .session()
        .and_then(|session| session.predictor())
        .map(|predictor| (predictor.state.pos, predictor.state.pitch));
    eprintln!(
        "{name}: player {player:?}, camera {:?} pitch {}, uploaded sections {}, phases {}",
        scene.camera.pos,
        scene.camera.pitch,
        scene.client.uploaded_section_count(),
        scene.phases.names().count(),
    );
    // Hold-still control: one frame later without a tick boundary, nothing may move.
    common::pump_no_tick(
        &mut scene.client,
        &mut scene.gpu,
        &mut scene.phases,
        &mut scene.camera,
    );
    let same = scene.frame();
    // Two renders of one tick state differ by at most one unit (rounding), never by three.
    let still = changes(&before, &same, 3);
    let mut most = Moved::default();
    let mut most_frame = before.clone();
    let mut previous = before.clone();
    let mut ticks = 0;
    for target in [6, 16, 26, 36] {
        while ticks < target {
            scene.pump();
            ticks += 1;
        }
        let after = scene.frame();
        let from_reference = changes(&before, &after, threshold);
        let consecutive = changes(&previous, &after, threshold);
        eprintln!(
            "{name}: +{ticks} ticks (fluid, stone) from the reference {from_reference:?}, \
             from the previous capture {consecutive:?}, stone mean {:.2} -> {:.2}",
            stone_mean(&before),
            stone_mean(&after)
        );
        if most
            .captures
            .iter()
            .all(|capture| from_reference.0 > capture.0)
        {
            most_frame = after.clone();
        }
        most.captures.push(from_reference);
        if consecutive.0 > most.consecutive.0 {
            most.consecutive = consecutive;
        }
        previous = after;
    }
    eprintln!("{name}: no-tick control {still:?}; most changed {most:?}");
    // `TEXANIM_DUMP=<dir>` keeps the reference and the most-changed frame for a look by eye.
    if let Some(dir) = std::env::var_os("TEXANIM_DUMP") {
        let dir = std::path::PathBuf::from(dir);
        for (label, pixels) in [("before", &before), ("after", &most_frame)] {
            image::RgbaImage::from_raw(WIDTH, HEIGHT, pixels.clone())
                .expect("frame size")
                .save(dir.join(format!("{name}-{label}.png")))
                .expect("write frame");
        }
    }
    assert_eq!(
        still,
        (0, 0),
        "{name}: nothing may change without a client tick"
    );
    most
}

/// Lava's frames differ strongly (java `lava_still.png`: 155 of 256 texels move by 16 or more
/// in some channel between frame 0 and frame 6), while lighting and drift move a pixel by a few
/// units, so a 16-unit threshold sees the animation. Its playback (0..19 and back, two ticks a
/// frame) puts at least three of the four captures three or more frames from the reference
/// whatever the starting phase; two are required, because one capture of about 1900 pixels was
/// seen without any animation (a strip along the chunk edge changed once; not investigated).
fn assert_lava_animates(startup: Startup) {
    let captures = animate(startup, "lava", 16).captures;
    let animated = captures.iter().filter(|(fluid, _)| *fluid > 2000).count();
    assert!(
        animated >= 2,
        "{startup:?}: still lava must animate between client ticks: {captures:?}"
    );
}

/// The shipping startup. Negative control: before the 2026-11-06 fix the atlas animator was
/// dropped for good by the first tick that ran between the model stage and `ready`.
#[test]
fn lava_animates_after_the_deferred_startup_the_binary_uses() {
    assert_lava_animates(Startup::Deferred);
}

/// Water is subtle (java `water_still.png` moves few texels by more than a few units after the
/// biome tint), so it is judged on consecutive captures, between which the scene's slow
/// brightness drift stays below the 4-unit threshold: the plain stone band must stay still while
/// the water moves.
#[test]
fn water_animates_after_the_deferred_startup_the_binary_uses() {
    let (fluid, stone) = animate(Startup::Deferred, "water", 4).consecutive;
    assert!(
        fluid > 300,
        "still water must animate between client ticks, only {fluid} pixels moved"
    );
    assert!(
        stone * 10 < fluid,
        "the stone band ({stone}) must not move like the water ({fluid})"
    );
}

/// Control: the eager startup every older test used animates with or without the fix, which
/// isolates the defect to the deferred startup's tick ordering.
#[test]
fn lava_animates_after_an_eager_startup() {
    assert_lava_animates(Startup::Eager);
}
