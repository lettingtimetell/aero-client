//! A graphics change made while the world is still being meshed must not lose sections.
//!
//! A mesher setting (Fancy/Fast, smooth lighting, Connected Textures, Tall Grass, a shader pack's
//! mesher mode) re-meshes every section on screen: the queue cancels the meshes it is building,
//! finished meshes waiting for upload are dropped, and the sections that have GPU buffers are
//! queued again. A section that was being meshed for the first time, or again after a resource
//! reload (which drops every buffer), had no buffer yet, so nothing queued it again: it stayed
//! invisible, a hole in the world, until a block in it changed. Seen with Connected Textures
//! switched on (a reload) and its mode changed a moment later: the far chunks never came back.
//! java: `RenderGlobal.loadRenderers` rebuilds every `RenderChunk` of the view area.

mod common;

use std::time::{Duration, Instant};

use common::{config, gpu, pump, pump_until, serve_join, CHUNK};
use mc_client::app::Client;
use mc_math::ChunkPos;
use mc_physics::state::MoveInput;
use mc_render::PhaseList;
use mc_ui::screens::Action;
use tokio::net::TcpListener;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
const PLAYER: i32 = 55;
/// Chunks around spawn, each with a stone floor in section 0: one section to mesh per chunk.
const RADIUS: i32 = 6;

fn idle(client: &Client) -> bool {
    client.mesh_queue_len() == 0
        && client.mesh_queue_in_flight_len() == 0
        && client.mesh_queue_deferred_len() == 0
}

/// Resource reloads one after another while the world is being meshed again (the Mipmap Levels
/// slider moved twice, a pack switched while chunks load after joining, F3+T pressed twice): a
/// reload drops every buffer and the old terrain renderer with the finished meshes still waiting
/// for upload in it. Those sections had no buffer to be queued again from: holes in the world
/// (seen after a Mipmap Levels 4 -> 0 -> 2 -> 4 round trip: everything past the nearest chunks
/// was gone). java: `RenderGlobal.loadRenderers` rebuilds every `RenderChunk` of the view area.
#[test]
fn reloads_in_a_row_keep_every_section() {
    meshed_world(|client, gpu, phases, camera, expected| {
        let still = MoveInput::default();
        let mut progress = |_: &str, _: i32| {};
        for frames in [0, 1, 2, 3, 1, 0, 2] {
            client
                .reload_resource_packs(Some(gpu), Some(phases), &mut progress)
                .expect("the reload succeeds");
            for _ in 0..frames {
                pump(client, gpu, phases, camera, &still);
            }
        }
        settle(client, gpu, phases, camera);
        assert_eq!(
            client.uploaded_section_count(),
            expected,
            "every floor section is meshed again after the reloads"
        );
    });
}

fn settle(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut mc_render::Camera,
) {
    let still = MoveInput::default();
    let deadline = Instant::now() + Duration::from_secs(60);
    while !idle(client) {
        pump(client, gpu, phases, camera, &still);
        assert!(Instant::now() < deadline, "the mesh queue never went idle");
    }
    for _ in 0..60 {
        pump(client, gpu, phases, camera, &still);
    }
}

/// Join a loopback server with a floor of chunks, wait until every section is meshed, run
/// `scenario` with the number of meshed sections, then leave.
fn meshed_world(
    scenario: impl FnOnce(
        &mut Client,
        &mut mc_render::Gpu,
        &mut PhaseList,
        &mut mc_render::Camera,
        usize,
    ),
) {
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
    let chunks: Vec<ChunkPos> = (-RADIUS..=RADIUS)
        .flat_map(|x| (-RADIUS..=RADIUS).map(move |z| ChunkPos { x, z }))
        .collect();
    let server = client.runtime().handle().spawn(async move {
        let mut peer = serve_join(&listener, PLAYER, &chunks, 0b1).await;
        peer.drain_until_closed().await;
    });
    client.connect("127.0.0.1", port);
    let still = MoveInput::default();
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &still,
        Duration::from_secs(60),
        "every floor section meshed",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
                && idle(client)
                && client.uploaded_section_count() > 0
        },
    );
    for _ in 0..60 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    let expected = client.uploaded_section_count();
    assert!(expected > 100, "most floor sections are meshed: {expected}");
    scenario(&mut client, &mut gpu, &mut phases, &mut camera, expected);
    client.disconnect();
    for _ in 0..5 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    let _ = client
        .runtime()
        .handle()
        .block_on(async { tokio::time::timeout(Duration::from_secs(10), server).await });
}

#[test]
fn a_graphics_change_during_meshing_keeps_every_section() {
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
    let chunks: Vec<ChunkPos> = (-RADIUS..=RADIUS)
        .flat_map(|x| (-RADIUS..=RADIUS).map(move |z| ChunkPos { x, z }))
        .collect();
    let server = client.runtime().handle().spawn(async move {
        let mut peer = serve_join(&listener, PLAYER, &chunks, 0b1).await;
        peer.drain_until_closed().await;
    });

    client.connect("127.0.0.1", port);
    let still = MoveInput::default();
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &still,
        Duration::from_secs(60),
        "every floor section meshed",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|session| session.world().is_chunk_loaded(CHUNK))
                && idle(client)
                && client.uploaded_section_count() > 0
        },
    );
    for _ in 0..60 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    let expected = client.uploaded_section_count();
    assert!(expected > 100, "most floor sections are meshed: {expected}");

    // A reload (every buffer dropped, every section queued again), then graphics changes while
    // that re-mesh is under way, a few times.
    let mut progress = |_: &str, _: i32| {};
    for round in 0..4 {
        client
            .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut progress)
            .expect("the reload succeeds");
        for _ in 0..=round {
            pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
        }
        client.settings.ambient_occlusion = if client.settings.ambient_occlusion > 0 {
            0
        } else {
            2
        };
        client.handle_action(Action::SaveSettings);
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
        client.settings.fancy_graphics = !client.settings.fancy_graphics;
        client.handle_action(Action::SaveSettings);
    }

    let deadline = Instant::now() + Duration::from_secs(60);
    while !idle(&client) {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
        assert!(Instant::now() < deadline, "the mesh queue never went idle");
    }
    for _ in 0..60 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    assert_eq!(
        client.uploaded_section_count(),
        expected,
        "every floor section is meshed again after the graphics changes"
    );

    client.disconnect();
    for _ in 0..5 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &still);
    }
    let _ = client
        .runtime()
        .handle()
        .block_on(async { tokio::time::timeout(Duration::from_secs(10), server).await });
}
