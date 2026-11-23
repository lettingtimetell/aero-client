//! A fishing cast seen in first person, end to end.
//!
//! The loopback double hands the player a fishing rod; the test right-clicks with the shipping
//! input path, then plays the client exactly the packet stream a vanilla 1.8.9 server sends for
//! that cast. The stream (spawn `S0E`, its immediate `S12`, then the five-tick tracker updates)
//! was produced by the real `EntityFishHook` + `EntityTrackerEntry` classes
//! (`crates/mc-client/javaref/FishHookRef.java`, scenario names below), so the frames show what a
//! real server makes the client draw.
//!
//! The assertion is the angler link: `RenderItem.renderItemModelForEntity` (`:284-287`) draws the
//! held rod as `fishing_rod_cast` exactly while `thePlayer.fishEntity` is set, which the hook's
//! client constructor sets and `EntityFishHook.setDead` (the S13 destroy) clears. The uncast
//! model's white hook used to sweep through the view during the cast swing (the reported "small
//! blue line": a white texel under the sky lightmap).
//!
//! Set `MC_RS_FISH_FRAMES=<dir>` to dump every rendered frame as a PNG (four partial ticks per
//! game tick).

mod common;

use std::time::Duration;

use common::{accept_and_join, config, gpu, pump, pump_until, CHUNK};
use mc_client::app::Client;
use mc_physics::state::MoveInput;
use mc_protocol::play::s2c;
use mc_protocol::Angle;
use mc_render::input::Input;
use mc_render::{Camera, InputState, KeyBindings, PhaseList};
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::screens::Action;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use winit::event::MouseButton;

const WIDTH: u32 = 854;
const HEIGHT: u32 = 480;
const PLAYER: i32 = 55;
const HOOK: i32 = 900;
const FRAMES_PER_TICK: u32 = 4;

#[derive(Clone, Copy, Debug)]
enum Pkt {
    Vel([i16; 3]),
    Rel([i8; 3], Option<(i8, i8)>, bool),
}

struct Cast {
    name: &'static str,
    yaw: f32,
    pitch: f32,
    spawn: [i32; 3],
    spawn_look: (i8, i8),
    spawn_velocity: [i16; 3],
    packets: &'static [(u32, Pkt)],
}

/// Java `FishHookRef` scenarios whose world is exactly the loopback floor (stone at y = 4).
const CASTS: &[Cast] = &[
    Cast {
        name: "floor_in_front",
        yaw: 0.0,
        pitch: 60.0,
        spawn: [266, 208, 272],
        spawn_look: (0, -43),
        spawn_velocity: [63, -10313, 5921],
        packets: &[
            (2, Pkt::Vel([58, -9782, 5448])),
            (7, Pkt::Vel([0, -582, 58])),
            (7, Pkt::Rel([0, -7, 3], None, false)),
            (12, Pkt::Vel([0, -298, 0])),
            (12, Pkt::Rel([0, 0, 0], Some((0, -49)), false)),
            (17, Pkt::Rel([0, 0, 0], Some((0, -54)), false)),
            (22, Pkt::Rel([0, 0, 0], Some((0, -58)), false)),
            (27, Pkt::Vel([0, -42, 0])),
        ],
    },
    Cast {
        name: "down_at_feet",
        yaw: 30.0,
        pitch: 82.0,
        spawn: [267, 208, 269],
        spawn_look: (-19, -59),
        spawn_velocity: [-708, -11951, 1432],
        packets: &[
            (2, Pkt::Vel([-651, -11289, 1318])),
            (7, Pkt::Vel([0, -332, 0])),
            (7, Pkt::Rel([0, 0, 0], Some((-18, -52)), false)),
            (12, Pkt::Rel([0, 0, 0], Some((-15, -56)), false)),
            (22, Pkt::Rel([0, 0, 0], Some((-15, -61)), false)),
            (27, Pkt::Vel([0, -54, 0])),
            (27, Pkt::Rel([0, 0, 0], Some((-24, -62)), false)),
        ],
    },
];

/// One scripted server packet for the peer to forward.
enum Out {
    Spawn(s2c::SpawnObject),
    Packet(Pkt),
    Destroy,
}

async fn serve(listener: TcpListener, mut rx: mpsc::UnboundedReceiver<Out>) -> common::RefPeer {
    let mut peer = accept_and_join(listener, PLAYER).await;
    let mut items: Vec<Option<mc_protocol::ItemStack>> = vec![None; 45];
    items[36] = Some(mc_protocol::ItemStack {
        item_id: 346, // minecraft:fishing_rod
        count: 1,
        damage: 0,
        nbt: None,
    });
    peer.send(&s2c::WindowItems {
        window_id: 0,
        items,
    })
    .await;
    while let Some(out) = rx.recv().await {
        match out {
            Out::Spawn(spawn) => peer.send(&spawn).await,
            Out::Packet(Pkt::Vel(motion)) => {
                peer.send(&s2c::EntityVelocity {
                    entity_id: HOOK,
                    motion_x: motion[0],
                    motion_y: motion[1],
                    motion_z: motion[2],
                })
                .await
            }
            Out::Packet(Pkt::Rel(delta, None, on_ground)) => {
                peer.send(&s2c::EntityRelMove {
                    entity_id: HOOK,
                    delta_x: delta[0],
                    delta_y: delta[1],
                    delta_z: delta[2],
                    on_ground,
                })
                .await
            }
            Out::Packet(Pkt::Rel(delta, Some((yaw, pitch)), on_ground)) => {
                peer.send(&s2c::EntityLookAndRelMove {
                    entity_id: HOOK,
                    delta_x: delta[0],
                    delta_y: delta[1],
                    delta_z: delta[2],
                    yaw: Angle(yaw as u8),
                    pitch: Angle(pitch as u8),
                    on_ground,
                })
                .await
            }
            Out::Destroy => {
                peer.send(&s2c::DestroyEntities {
                    entity_ids: vec![HOOK],
                })
                .await
            }
        }
    }
    peer
}

fn quad_count(model: &mc_assets::BakedModel) -> usize {
    model.general_quads.len() + model.face_quads.iter().map(Vec::len).sum::<usize>()
}

/// Quads in the model the first-person renderer currently holds.
fn held_quads(client: &Client) -> Option<usize> {
    client.render().and_then(|render| {
        render
            .first_person
            .with(|phase| phase.item_model().map(quad_count))
    })
}

fn render_tick(
    client: &mut Client,
    gpu: &mut mc_render::Gpu,
    phases: &mut PhaseList,
    camera: &mut Camera,
    dump: Option<&(std::path::PathBuf, &str)>,
    tick: u32,
) {
    for frame_index in 0..FRAMES_PER_TICK {
        let partial = frame_index as f32 / FRAMES_PER_TICK as f32;
        client.frame(gpu, camera, partial);
        let frame = gpu.acquire().expect("acquire offscreen frame");
        phases.render(gpu, frame.view(), camera, partial);
        drop(frame);
        if let Some((dir, name)) = dump {
            let pixels = gpu.read_pixels().expect("readback");
            let path = dir.join(format!("{name}_t{tick:02}_f{frame_index}.png"));
            image::save_buffer(&path, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
                .expect("write frame");
        }
    }
}

fn run_cast(cast: &Cast) {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
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
    let (tx, rx) = mpsc::unbounded_channel();
    let peer = client.runtime().handle().spawn(serve(listener, rx));
    client.handle_action(Action::Connect(Box::new(ServerEntry::new(
        "Loopback",
        format!("127.0.0.1:{port}"),
    ))));
    let look = MoveInput {
        yaw: cast.yaw,
        pitch: cast.pitch,
        ..MoveInput::default()
    };
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the world and the rod",
        |client| {
            client.ui.name() == "in_game"
                && client
                    .session()
                    .is_some_and(|s| s.world().is_chunk_loaded(CHUNK))
                && client.player_hud.hotbar[0].is_some()
        },
    );
    if let Some(predictor) = client
        .session_mut()
        .and_then(mc_client::Session::predictor_mut)
    {
        predictor.state.yaw = cast.yaw;
        predictor.state.pitch = cast.pitch;
    }
    for _ in 0..5 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &look);
    }
    // `RenderItem.registerItems`: variant 0 is `fishing_rod`, variant 1 `fishing_rod_cast`. The
    // uncast texture's string and hook pixels give its generated model extra edge quads.
    let rod = mc_data::ItemId(346);
    let uncast_quads = quad_count(mc_assets::item_model_for_variant(rod, 0));
    let cast_quads = quad_count(mc_assets::item_model_for_variant(rod, 1));
    assert_ne!(uncast_quads, cast_quads, "the two rod models must differ");
    assert_eq!(
        held_quads(&client),
        Some(uncast_quads),
        "{}: an idle rod is drawn uncast",
        cast.name
    );

    let dump_dir = std::env::var_os("MC_RS_FISH_FRAMES").map(std::path::PathBuf::from);
    if let Some(dir) = &dump_dir {
        std::fs::create_dir_all(dir).expect("frame dir");
    }
    let dump = dump_dir.map(|dir| (dir, cast.name));

    // Right-click with the rod: the shipping input path swings and sends the use-item C08.
    let mut state = InputState::new();
    let mut keys = KeyBindings::vanilla();
    client.handle_game_input(&state, &mut keys);
    state.on_mouse_button(MouseButton::Right, true);
    keys.set_state(Input::MOUSE_RIGHT, true);
    client.handle_game_input(&state, &mut keys);
    // The server answers the C08 in its next tick: the spawn pair arrives before client tick 1.
    tx.send(Out::Spawn(s2c::SpawnObject {
        entity_id: HOOK,
        object_type: 90,
        x: cast.spawn[0],
        y: cast.spawn[1],
        z: cast.spawn[2],
        pitch: Angle(cast.spawn_look.1 as u8),
        yaw: Angle(cast.spawn_look.0 as u8),
        data: PLAYER,
        velocity: Some((0, 0, 0)),
    }))
    .expect("peer alive");
    tx.send(Out::Packet(Pkt::Vel(cast.spawn_velocity)))
        .expect("peer alive");
    std::thread::sleep(Duration::from_millis(60));
    keys.set_state(Input::MOUSE_RIGHT, false);
    state.on_mouse_button(MouseButton::Right, false);

    let mut saw_hook = false;
    for tick in 1..=30 {
        for &(when, packet) in cast.packets {
            if when == tick {
                tx.send(Out::Packet(packet)).expect("peer alive");
            }
        }
        std::thread::sleep(Duration::from_millis(30));
        client.poll();
        client.tick(&look);
        client.handle_game_input(&state, &mut keys);
        saw_hook |= client
            .session()
            .is_some_and(|session| session.state().entities.get(HOOK).is_some());
        assert_eq!(
            held_quads(&client),
            Some(cast_quads),
            "{} t={tick}: the rod is drawn cast while its hook exists",
            cast.name
        );
        render_tick(
            &mut client,
            &mut gpu,
            &mut phases,
            &mut camera,
            dump.as_ref(),
            tick,
        );
    }
    assert!(
        saw_hook,
        "{}: the scripted hook must reach the tracker",
        cast.name
    );

    // S13: `removeEntityFromWorld` -> `EntityFishHook.setDead` clears `thePlayer.fishEntity`.
    tx.send(Out::Destroy).expect("peer alive");
    std::thread::sleep(Duration::from_millis(60));
    for _ in 0..2 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &look);
    }
    assert_eq!(
        held_quads(&client),
        Some(uncast_quads),
        "{}: destroying the hook restores the uncast rod",
        cast.name
    );
    drop(tx);
    client.disconnect();
    for _ in 0..5 {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &look);
    }
    let handle = client.runtime().handle().clone();
    let _ = handle.block_on(async { tokio::time::timeout(Duration::from_secs(10), peer).await });
}

#[test]
fn a_fishing_cast_renders_through_the_whole_vanilla_packet_stream() {
    for cast in CASTS {
        run_cast(cast);
    }
}
