//! Regression test for the `S21e -> S14` §13 cross-section row: F5's third-person camera used to
//! orbit an invisible body, because nothing ever drew the local player's own model — only remote
//! players get a `TrackedEntity` (S14's tracker), and the ECS never held the local player either.
//!
//! `Client::upload_entities` (`app/driver.rs`) now draws the local player straight from
//! `mc_physics::Predictor::state` — the same source `place_camera` already trusts for the camera
//! itself — whenever `third_person_view > 0`, using the exact `player_body_renderer` +
//! `RenderManager::emit_with` pair already proven for a remote player (the `None` arm of the
//! entity loop). No ECS entity and no synthetic `TrackedEntity` are involved.
//!
//! This is deliberately geometry-level (`EntityPhase::vertex_count`), not a pixel comparison:
//! `vertex_count` is exactly the phase the fix touches, so a regression here cannot be masked by
//! terrain or sky pixels drawn behind the body. Against a real headless GPU adapter per
//! `tests/common/mod.rs`'s disclosure — a skip, not a stub, on a machine with none.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_render::PhaseList;
use tokio::net::TcpListener;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;

/// Vertices currently queued in the `entities` render phase, after a frame has been driven.
fn entity_vertex_count(client: &Client) -> usize {
    client
        .render()
        .expect("a live render stack")
        .entities
        .with(|phase| phase.vertex_count())
}

#[test]
fn third_person_draws_the_local_players_body_and_first_person_stays_empty() {
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

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    // No remote players are ever scripted here on purpose: with `EntityTracker` empty, every
    // vertex the `entities` phase carries can only have come from the local player's own body,
    // so `vertex_count` is an unambiguous signal.
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
    assert!(
        client.session().unwrap().predictor().is_some(),
        "a live predictor is what the fix reads position/yaw/pitch from"
    );

    // ---- first person (the default): unaffected -----------------------------------------
    assert_eq!(client.third_person_view, 0, "first person is the default");
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    assert_eq!(
        entity_vertex_count(&client),
        0,
        "first person must draw no local-player body — vanilla hides it there too, and with no \
         remote players tracked nothing else could contribute a vertex either"
    );

    // ---- F5 once: third-person back view --------------------------------------------------
    client.third_person_view = 1;
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    let back_view_verts = entity_vertex_count(&client);
    assert!(
        back_view_verts > 0,
        "third-person back view must draw the local player's body"
    );

    // ---- F5 again: third-person front (mirrored) view --------------------------------------
    client.third_person_view = 2;
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    assert_eq!(
        entity_vertex_count(&client),
        back_view_verts,
        "the front view draws the same body from the other side: same geometry, same vertex count"
    );

    // ---- F5 a third time: back to first person, and the body must vanish again -------------
    client.third_person_view = 0;
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    assert_eq!(
        entity_vertex_count(&client),
        0,
        "cycling back to first person must stop drawing the body"
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
