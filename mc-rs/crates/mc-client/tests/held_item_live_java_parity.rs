//! "Held items should be a bit more to the left (recommend
//! comparing held item renders from java to rust)" — the live-client half of the comparison.
//!
//! `crates/mc-render/tests/held_item_java_parity.rs` checks the first-person renderer against the
//! real vanilla 1.8.9 `ItemRenderer` (captured by `crates/mc-client/javaref/HeldItemRef.java`) with
//! the renderer's inputs set by hand. This test closes the remaining gap: the *driver's* inputs.
//! A real client joins the loopback peer, which puts a diamond sword in hotbar slot 0; the player
//! stands still with the user's own settings (FOV 90) until the sword is fully raised, and then
//! only the client's live `first-person` phase is rendered at 1920x1080 against a magenta clear.
//! Whatever the driver feeds it (the held model, the hand FOV, the aspect, the equip progress,
//! the arm sway and the bob) must put the sword on exactly the pixels vanilla covers.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_protocol::play::s2c;
use mc_render::{PhaseList, PhaseOrder};
use tokio::net::TcpListener;

const WIDTH: u32 = 1920;
const HEIGHT: u32 = 1080;
const PLAYER: i32 = 55;

/// Java's `sword_idle` coverage mask from `HeldItemRef.java` (the same row as the `REFS` table
/// in `crates/mc-render/tests/held_item_java_parity.rs`): pixel count, bounds and centroid.
const JAVA_SWORD_IDLE: (u64, [u32; 4], [f64; 2]) =
    (177_607, [1473, 1919, 336, 1079], [1756.2761, 770.9449]);

#[test]
fn a_live_client_holds_its_sword_on_vanillas_pixels() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    // The user's own options.txt: FOV 90 and view bobbing off. Vanilla's hand ignores the FOV.
    client.settings.fov = 90.0;
    client.settings.view_bobbing = false;
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
        let mut peer = accept_and_join(listener, PLAYER).await;
        let mut items: Vec<Option<mc_protocol::ItemStack>> = vec![None; 45];
        items[36] = Some(mc_protocol::ItemStack {
            item_id: 276, // minecraft:diamond_sword
            count: 1,
            damage: 0,
            nbt: None,
        });
        peer.send(&s2c::WindowItems {
            window_id: 0,
            items,
        })
        .await;
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
        "the world and the sword",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
                && client.player_hud.hotbar[0].is_some()
        },
    );

    // Look level, due south, and stand still until the sword is fully raised, the arm has
    // caught up with the look and the walk sway has decayed.
    let still = MoveInput {
        yaw: 0.0,
        pitch: 0.0,
        ..MoveInput::default()
    };
    if let Some(predictor) = client
        .session_mut()
        .and_then(mc_client::Session::predictor_mut)
    {
        predictor.state.yaw = 0.0;
        predictor.state.pitch = 0.0;
    }
    camera.yaw = 0.0;
    camera.pitch = 0.0;
    for _ in 0..30 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    // The frame whose hand pass is measured: the shell's order, partial ticks zero as Java's.
    client.frame(&gpu, &mut camera, 0.0);

    let hand = client.render().expect("render stack").first_person.clone();
    let (equip, sway) = hand.with(|phase| {
        (
            phase.render_state.equip_amount(phase.partial_ticks),
            [
                phase.prev_render_arm_yaw,
                phase.render_arm_yaw,
                phase.prev_render_arm_pitch,
                phase.render_arm_pitch,
            ],
        )
    });
    assert!(
        equip.abs() < 1e-6,
        "the sword must be fully raised, equip {equip}"
    );
    assert!(
        sway.iter().all(|angle| angle.abs() < 1e-3),
        "standing still, the arm has caught up with the look: {sway:?}"
    );
    assert!(
        camera.effective_fov() > 80.0 && (camera.hand_fov - 70.0).abs() < 1e-6,
        "the world uses the FOV setting ({}), the hand vanilla's fixed 70 ({})",
        camera.effective_fov(),
        camera.hand_fov
    );

    let mut only_hand = PhaseList::new();
    only_hand.clear_color = wgpu::Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    };
    only_hand.insert(PhaseOrder::FIRST_PERSON, Box::new(hand));
    let frame = gpu.acquire().expect("acquire offscreen frame");
    only_hand.render(&gpu, frame.view(), &camera, 0.0);
    let pixels = gpu.read_pixels().expect("readback");
    if let Some(dir) = std::env::var_os("MC_HELD_ITEM_DUMP").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&dir).expect("dump dir");
        image::RgbaImage::from_raw(WIDTH, HEIGHT, pixels.clone())
            .expect("rgba frame")
            .save(dir.join("live_sword_idle.png"))
            .expect("write png");
    }

    let (mut count, mut sum_x, mut sum_y) = (0_u64, 0.0_f64, 0.0_f64);
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (WIDTH, 0, HEIGHT, 0);
    for (index, pixel) in pixels.chunks_exact(4).enumerate() {
        if pixel[0] == 255 && pixel[1] == 0 && pixel[2] == 255 {
            continue;
        }
        let (x, y) = (index as u32 % WIDTH, index as u32 / WIDTH);
        count += 1;
        sum_x += f64::from(x) + 0.5;
        sum_y += f64::from(y) + 0.5;
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let (java_count, [jx0, jx1, jy0, jy1], [jcx, jcy]) = JAVA_SWORD_IDLE;
    assert!(count > 0, "the live hand pass drew no sword at all");
    let (cx, cy) = (sum_x / count as f64, sum_y / count as f64);
    eprintln!(
        "live sword: n {count} x {min_x}..{max_x} y {min_y}..{max_y} c ({cx:.2}, {cy:.2}); java: \
         n {java_count} x {jx0}..{jx1} y {jy0}..{jy1} c ({jcx:.2}, {jcy:.2}); centroid off by \
         ({:+.2}, {:+.2}) px",
        cx - jcx,
        cy - jcy
    );
    assert!(
        (cx - jcx).abs() <= 1.0 && (cy - jcy).abs() <= 1.0,
        "the live sword's centroid ({cx:.2}, {cy:.2}) must be within a pixel of vanilla's \
         ({jcx:.2}, {jcy:.2})"
    );
    assert!(
        min_x.abs_diff(jx0) <= 1
            && max_x.abs_diff(jx1) <= 1
            && min_y.abs_diff(jy0) <= 1
            && max_y.abs_diff(jy1) <= 1,
        "the live sword's bounds x {min_x}..{max_x} y {min_y}..{max_y} must be within a pixel \
         of vanilla's x {jx0}..{jx1} y {jy0}..{jy1}"
    );
    assert!(
        (count as f64 - java_count as f64).abs() <= java_count as f64 * 0.01,
        "the live sword covers {count} pixels, vanilla {java_count}"
    );

    client.disconnect();
    for _ in 0..10 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    client.runtime().handle().block_on(async {
        tokio::time::timeout(Duration::from_secs(10), server)
            .await
            .expect("the peer must finish")
            .expect("the peer task must not panic");
    });
}
