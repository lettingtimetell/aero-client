//! A resource reload (F3+T, a pack change, Connected Textures switched on) must not take the
//! held item out of the player's hand.
//!
//! The reload rebuilds every render phase, the first-person one included, but the client's equip
//! latch outlived it: the hotbar still held the latched sword, so no equip edge came and the new
//! phase never got the sword's model. The hand stayed empty until the player changed items.
//! java: `ItemRenderer` outlives `refreshResources` and looks the model up from `itemToRender`
//! every frame, so the sword stays in the hand, fully raised, with no equip animation.

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_protocol::play::s2c;
use mc_render::PhaseList;
use tokio::net::TcpListener;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
const PLAYER: i32 = 55;

#[test]
fn a_resource_reload_keeps_the_held_sword_in_the_hand() {
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
    // Until the sword is fully raised.
    for _ in 0..30 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    let held = |client: &Client| {
        let hand = client.render().expect("render stack").first_person.clone();
        hand.with(|phase| {
            (
                phase.has_item_model(),
                phase.render_state.equipped_progress,
                phase.render_state.prev_equipped_progress,
            )
        })
    };
    assert_eq!(held(&client), (true, 1.0, 1.0), "the sword is in the hand");

    // F3+T's reload: every phase is rebuilt.
    let mut progress = |_: &str, _: i32| {};
    client
        .reload_resource_packs(Some(&gpu), Some(&mut phases), &mut progress)
        .expect("the reload succeeds");
    for tick in 1..=10 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
        assert_eq!(
            held(&client),
            (true, 1.0, 1.0),
            "tick {tick} after the reload: the sword stays in the hand, fully raised"
        );
    }

    client.disconnect();
    for _ in 0..5 {
        pump(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            &MoveInput::default(),
        );
    }
    let _ = client
        .runtime()
        .handle()
        .block_on(async { tokio::time::timeout(Duration::from_secs(10), server).await });
}
