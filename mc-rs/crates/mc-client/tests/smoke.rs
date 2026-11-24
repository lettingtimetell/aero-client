//! S21's smoke test: launch → sign in → server list → join → walk, mine, chat → quit.
//!
//! # Read `tests/common/mod.rs` first
//!
//! It states plainly what this is and is not. In one line: **the server is a loopback double,
//! not a public 1.8.9 server, and the Microsoft login is mocked.** Everything else — the assets,
//! the atlas, the mesher, the physics, the screens, the GPU — is real.
//!
//! java: this is `Minecraft.run` driven by a script instead of by a human, with
//! `runGameLoop`'s three phases (`processReceivedPackets`, `runTick`, render) called explicitly
//! so each step can be asserted between them.

mod common;

use std::time::Duration;

use common::{
    accept_and_join, config, decode, drawn_pixels, gpu, pump, pump_until, CHUNK, CLIENT_CHAT,
    DIG_TARGET, FLOOR_Y, SERVER_CHAT, SPAWN, STONE,
};
use mc_client::app::state::GameState;
use mc_client::app::Client;
use mc_client::sync::ClientEvent;
use mc_math::{BlockPos, DVec3, Facing};
use mc_physics::state::MoveInput;
use mc_protocol::packet::Packet;
use mc_protocol::play::{c2s, s2c};
use mc_render::{Camera, PhaseList};
use mc_ui::screens::multiplayer::ServerEntry;
use mc_ui::screens::Action;
use tokio::net::TcpListener;
use uuid::Uuid;

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
/// How many ticks the player walks forward.
///
/// Long enough for S13's acceleration curve to cover a visible distance, and short enough to
/// stay on the floor: vanilla walking settles at about 0.216 blocks/tick, spawn is at z = 8.5,
/// and the single chunk the peer sends ends at z = 16. Forty ticks walks the player off the edge
/// and the fall is the *correct* physics answer — which the first run of this test duly caught.
const WALK_TICKS: usize = 20;

/// What the scripted peer observed.
struct Script {
    chat: String,
    dig_start: c2s::PlayerDigging,
    dig_stop: c2s::PlayerDigging,
    movements: Vec<c2s::PlayerPositionAndLook>,
    positions: Vec<c2s::PlayerPosition>,
    settings_locale: String,
    /// Every C17 the client sent: the join's `MC|Brand`, and nothing else.
    plugin_messages: Vec<c2s::PluginMessage>,
}

#[allow(clippy::too_many_lines)]
#[test]
fn launch_sign_in_join_walk_mine_chat_and_quit() {
    let Some(mut gpu) = gpu(WIDTH, HEIGHT) else {
        return;
    };
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };

    // ---- launch: config load + asset load with a real progress report -------------------
    let mut reports: Vec<(String, i32)> = Vec::new();
    let mut client = Client::new(config, &mut |message, percent| {
        reports.push((message.to_owned(), percent));
    })
    .expect("the client must start against the real 1.8.9 assets");

    assert!(
        reports.iter().any(|(m, _)| m == "Baking block models"),
        "the loading report must name the stage that dominates startup: {reports:?}"
    );
    assert_eq!(reports.last().unwrap().1, 100);
    assert!(matches!(client.machine.state(), GameState::Menu));
    assert_eq!(client.ui.name(), "main_menu");

    // ---- "MS login": mocked. This is offline mode, and it is the honest limit of this test.
    let auth = client.auth.clone().expect("offline mode signs in locally");
    assert_eq!(auth.username, "Dev");
    assert!(
        auth.expose().is_empty(),
        "offline mode has no token; nothing here has ever contacted Microsoft"
    );

    // ---- render phases, against a real headless adapter ---------------------------------
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, WIDTH, HEIGHT);
    let names: Vec<&str> = phases.names().collect();
    for expected in [
        "sky/dome",
        "sky/clouds",
        // The clouds' second point, after the world, for an eye at or above y=128.
        "sky/clouds-above",
        "terrain/solid",
        "terrain/cutout",
        "terrain/translucent",
        "entity",
        "sky/weather",
        "sky/fog",
        "ui",
    ] {
        assert!(
            names.contains(&expected),
            "missing phase {expected}: {names:?}"
        );
    }

    // The main menu must draw something before anything else happens.
    let mut camera = mc_client::app::driver::menu_camera();
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    let menu_pixels = gpu.read_pixels().expect("readback");
    assert!(
        drawn_pixels(&menu_pixels, common::CLEAR_RGB) > 1_000,
        "the main menu must put something on the screen"
    );

    // ---- the server list ----------------------------------------------------------------
    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let address = format!("127.0.0.1:{port}");
    client
        .servers
        .add(ServerEntry::new("Loopback", address.clone()));

    client.handle_action(Action::OpenMultiplayer);
    assert_eq!(client.ui.name(), "multiplayer");
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );

    let server = client.runtime().handle().spawn(script(listener));

    let entry = ServerEntry::new("Loopback", address);
    client.handle_action(Action::Connect(Box::new(entry)));
    // The connecting screen is shown while Java's address parsing/SRV lookup is still pending;
    // the explicit Rust state is `Resolving` until that work produces the socket endpoint.
    assert!(
        matches!(
            client.machine.state(),
            GameState::Resolving { .. } | GameState::Connecting { .. }
        ),
        "{:?}",
        client.machine.state()
    );
    assert_eq!(client.ui.name(), "connecting");

    // ---- join ---------------------------------------------------------------------------
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the client to reach the world",
        // `JoinGame` alone is not "in the world": vanilla holds `GuiDownloadTerrain` up until
        // the first `S08PacketPlayerPosLook`, which is also when the chunks have arrived.
        |client| {
            client.machine.state().is_in_world()
                && client.ui.name() == "in_game"
                && client.session().is_some_and(|session| {
                    session.world().is_chunk_loaded(CHUNK)
                        && session
                            .predictor()
                            .is_some_and(|predictor| predictor.state.pos.y == SPAWN.1)
                })
        },
    );
    assert_eq!(client.machine.joins(), 1);

    // Terrain arrived and decodes to the floor the server built.
    let session = client.session().expect("a live session");
    assert!(session.world().is_chunk_loaded(CHUNK));
    assert_eq!(
        session
            .world()
            .block_at(BlockPos::new(8, FLOOR_Y as i32, 8))
            .id,
        STONE
    );
    assert_eq!(
        session
            .world()
            .block_at(BlockPos::new(8, FLOOR_Y as i32 + 1, 8)),
        mc_data::BlockState::AIR
    );
    // The position correction placed the player, and it survived the physics tick that follows
    // (a physics bug, since fixed at the source).
    let start = session.predictor().expect("predictor").state.pos;
    assert_eq!(start, DVec3::new(SPAWN.0, SPAWN.1, SPAWN.2));

    // The remote player the server spawned must land in the ECS and the tab list, and the chunk
    // must reach the GPU rather than only the world.
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(20),
        "the remote player and the meshed chunk",
        |client| {
            client.uploaded_section_count() > 0
                && client
                    .session()
                    .is_some_and(|session| session.state().players.len() == 1)
        },
    );
    let session = client.session().expect("a live session");
    assert_eq!(
        session.state().entities.len(),
        1,
        "the remote player is tracked"
    );
    let (_, tracked) = session.state().entities.iter().next().expect("one entity");
    assert_eq!(tracked.uuid, Some(Uuid::from_u128(0xABCD)));
    assert_eq!(
        session
            .state()
            .players
            .get(&Uuid::from_u128(0xABCD))
            .expect("tab list row")
            .username,
        "Notch"
    );

    // ---- walk ---------------------------------------------------------------------------
    let walk = MoveInput {
        forward: true,
        yaw: 0.0,
        pitch: common::SPAWN_PITCH,
        ..MoveInput::default()
    };
    for _ in 0..WALK_TICKS {
        pump(&mut client, &mut gpu, &mut phases, &mut camera, &walk);
        std::thread::sleep(Duration::from_millis(1));
    }
    let walked = client.session().unwrap().predictor().unwrap().state.pos;
    // Yaw 0 faces +Z in Minecraft, and the floor is solid, so the player must have moved along
    // +Z and stayed on top of it.
    assert!(
        walked.z > start.z + 0.5,
        "the player must have walked forward: {start:?} -> {walked:?}"
    );
    assert!(
        walked.z < 16.0,
        "the walk must stay on the floor the peer sent: {walked:?}"
    );
    assert_eq!(
        walked.y, start.y,
        "the player must not have fallen or risen"
    );
    assert!(
        client
            .session()
            .unwrap()
            .predictor()
            .unwrap()
            .state
            .on_ground,
        "the player must be standing on the stone floor"
    );

    // The camera follows the player's eyes, which is what makes the frame show the world.
    assert!(
        (camera.pos.y - (walked.y + 1.62)).abs() < 1e-6,
        "{:?}",
        camera.pos
    );
    let world_pixels = gpu.read_pixels().expect("readback");
    assert!(
        drawn_pixels(&world_pixels, common::CLEAR_RGB) > 10_000,
        "the world must be drawn, not just simulated"
    );

    // ---- mine ---------------------------------------------------------------------------
    // java: `PlayerControllerMP.clickBlock` then `onPlayerDamageBlock`'s STOP.
    client.session_mut().unwrap().player_action(|out, packets| {
        out.start_digging(DIG_TARGET, Facing::Up, packets);
    });
    client.session_mut().unwrap().player_action(|out, packets| {
        out.finish_digging(DIG_TARGET, Facing::Up, packets);
    });

    // ---- chat ---------------------------------------------------------------------------
    client.handle_action(Action::SendChat(CLIENT_CHAT.to_owned()));
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(20),
        "the server's chat message to arrive",
        // The server's own line, found by its text.
        |client| {
            client
                .chat
                .lines()
                .iter()
                .any(|line| line.text.contains("<Notch> welcome"))
        },
    );
    let heard = client.chat.lines()[0].text.clone();
    assert!(
        heard.contains("<Notch> welcome"),
        "the chat log must hold the server's message, got {heard:?}"
    );

    // ---- the peer's side of the story ---------------------------------------------------
    let script = client
        .runtime()
        .handle()
        .block_on(server)
        .expect("the scripted peer must run to completion");

    assert_eq!(script.settings_locale, "en_US");
    // The client always tells servers its name: the join's one C17 is vanilla's `MC|Brand` (a
    // protocol string) saying `Aero`, and no other plugin message was sent.
    assert_eq!(script.plugin_messages.len(), 1, "one C17");
    assert_eq!(script.plugin_messages[0].channel, "MC|Brand");
    assert_eq!(script.plugin_messages[0].data, b"\x04Aero");
    assert_eq!(script.chat, CLIENT_CHAT);
    assert_eq!(script.dig_start.position, DIG_TARGET);
    assert_eq!(script.dig_start.face, Facing::Up);
    assert_eq!(
        script.dig_start.status,
        c2s::PlayerDiggingStatus::StartDestroyBlock
    );
    assert_eq!(
        script.dig_stop.status,
        c2s::PlayerDiggingStatus::StopDestroyBlock
    );

    // The walk must be visible *on the wire*, not only in local state — that is the half a
    // server actually sees, and hazard H11 is about exactly this traffic.
    let furthest = script
        .positions
        .iter()
        .map(|packet| packet.z)
        .chain(script.movements.iter().map(|packet| packet.z))
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        furthest > SPAWN.2 + 0.5,
        "the server must have seen the player walk: furthest z {furthest}"
    );
    // Every packet sent *while walking* must report standing on the stone — a client that
    // reports `on_ground = false` while walking is the classic anticheat flag (hazard H11).
    let walking: Vec<bool> = script
        .positions
        .iter()
        .filter(|packet| packet.z > SPAWN.2 + 0.1)
        .map(|packet| packet.on_ground)
        .chain(
            script
                .movements
                .iter()
                .filter(|packet| packet.z > SPAWN.2 + 0.1)
                .map(|packet| packet.on_ground),
        )
        .collect();
    assert!(
        !walking.is_empty(),
        "no movement packets were sent while walking"
    );
    assert!(
        walking.iter().all(|on_ground| *on_ground),
        "every movement packet sent while walking on stone must say on_ground: {walking:?}"
    );
    // java: `handlePlayerPosLook` (`:717`) hard-codes `onGround = false` in the confirmation,
    // regardless of the real ground state. That is the one movement packet that must *not*
    // claim the ground, and it is easy to "fix" by accident.
    let confirmation = script
        .movements
        .iter()
        .find(|packet| (packet.x - SPAWN.0).abs() < 1e-9 && (packet.z - SPAWN.2).abs() < 1e-9)
        .expect("the position confirmation");
    assert!(!confirmation.on_ground);

    // ---- quit ---------------------------------------------------------------------------
    client.quit();
    assert!(!client.running());
    assert!(client.session().is_none(), "quitting unloads the world");
    assert_eq!(
        client.uploaded_section_count(),
        0,
        "quitting releases every chunk mesh"
    );

    let report = client.shutdown();
    assert!(
        report.is_clean(),
        "shutdown must not fail: {:?}",
        report.failures
    );
    assert_eq!(report.steps, mc_client::app::shutdown::ORDER.to_vec());
    assert!(
        dir.path().join("options.txt").is_file(),
        "settings must be saved"
    );
    assert!(
        dir.path().join("servers.dat").is_file(),
        "the server list must be saved"
    );
    // The saved list must round-trip, which is the only proof it is a real `servers.dat`.
    let bytes = std::fs::read(dir.path().join("servers.dat")).expect("read back");
    let reloaded = mc_ui::screens::multiplayer::ServerList::read(&bytes).expect("parse");
    assert_eq!(reloaded.len(), 1);
    assert_eq!(reloaded.get(0).unwrap().name, "Loopback");
    assert!(client.render().is_none(), "the GPU stack must be released");
}

/// The scripted peer: join, then read what the player does.
async fn script(listener: TcpListener) -> Script {
    let mut peer = accept_and_join(listener, 42).await;
    let settings: c2s::ClientSettings = peer
        .all::<c2s::ClientSettings>()
        .pop()
        .expect("the client settings packet");

    // A remote player, so nameplates and the tab list have something to work with.
    peer.send(&s2c::SpawnPlayer {
        entity_id: 100,
        player_uuid: Uuid::from_u128(0xABCD),
        x: (SPAWN.0 * 32.0) as i32 + 96,
        y: (SPAWN.1 * 32.0) as i32,
        z: (SPAWN.2 * 32.0) as i32,
        yaw: mc_protocol::Angle::from_degrees(180.0),
        pitch: mc_protocol::Angle(0),
        current_item: 0,
        metadata: Vec::new(),
    })
    .await;
    peer.send(&s2c::PlayerListItem::AddPlayer(vec![s2c::AddPlayerEntry {
        uuid: Uuid::from_u128(0xABCD),
        username: "Notch".into(),
        properties: vec![],
        game_mode: s2c::GameMode::Survival,
        ping: 42,
        display_name: None,
    }]))
    .await;
    peer.send(&s2c::Chat {
        component: SERVER_CHAT.into(),
        position: 0,
    })
    .await;

    // Read strictly in wire order: `read_until` discards what it scans past, so the digging
    // packets (sent before the chat message) must be read first.
    let dig_start: c2s::PlayerDigging = decode(
        &peer
            .read_until(c2s::PlayerDigging::ID, |body| {
                let packet: c2s::PlayerDigging = decode(body);
                packet.status == c2s::PlayerDiggingStatus::StartDestroyBlock
            })
            .await,
    );
    let dig_stop: c2s::PlayerDigging = decode(
        &peer
            .read_until(c2s::PlayerDigging::ID, |body| {
                let packet: c2s::PlayerDigging = decode(body);
                packet.status == c2s::PlayerDiggingStatus::StopDestroyBlock
            })
            .await,
    );
    let chat: c2s::ChatMessage = decode(&peer.read_until(c2s::ChatMessage::ID, |_| true).await);

    Script {
        chat: chat.message,
        dig_start,
        dig_stop,
        movements: peer.all::<c2s::PlayerPositionAndLook>(),
        positions: peer.all::<c2s::PlayerPosition>(),
        settings_locale: settings.locale,
        plugin_messages: peer.all::<c2s::PluginMessage>(),
    }
}

/// A kick must reach the disconnect screen with its reason intact, and going back to the menu
/// must leave no world behind. java: `NetHandlerPlayClient.handleDisconnect` → `GuiDisconnected`.
#[test]
fn a_kick_reaches_the_disconnect_screen_and_back_to_menu_clears_the_world() {
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
    let mut camera = Camera::default();

    let listener = client
        .runtime()
        .handle()
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind");
    let port = listener.local_addr().expect("addr").port();
    let server = client.runtime().handle().spawn(async move {
        let mut peer = accept_and_join(listener, 7).await;
        peer.send(&s2c::Disconnect {
            reason: r#"{"translate":"multiplayer.disconnect.server_shutdown"}"#.into(),
        })
        .await;
        // Hold the socket open so the kick is what ends the session, not the hang-up.
        tokio::time::sleep(Duration::from_secs(2)).await;
    });

    client.connect("127.0.0.1", port);
    pump_until(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
        Duration::from_secs(30),
        "the kick to reach the disconnect screen",
        |client| client.ui.name() == "disconnected",
    );
    let _ = client.runtime().handle().block_on(server);

    assert!(matches!(
        client.machine.state(),
        GameState::Disconnected { .. }
    ));
    assert!(
        client
            .last_events
            .iter()
            .any(|event| matches!(event, ClientEvent::Disconnected(_))),
        "the disconnect must have surfaced as an event"
    );
    // The screen renders — a disconnect screen that cannot draw is not a disconnect screen.
    pump(
        &mut client,
        &mut gpu,
        &mut phases,
        &mut camera,
        &MoveInput::default(),
    );
    let pixels = gpu.read_pixels().expect("readback");
    assert!(drawn_pixels(&pixels, common::CLEAR_RGB) > 1_000);

    client.handle_action(Action::Back);
    assert!(matches!(client.machine.state(), GameState::Menu));
    assert_eq!(client.ui.name(), "main_menu");
    assert!(client.session().is_none());
    assert_eq!(client.uploaded_section_count(), 0);
}

/// The device-code login seam S16 asked for in §13, exercised end to end on its failure path.
///
/// With no Azure client id configured — which is this workspace's permanent state
/// S03 hazard — `mc_auth` returns `AuthError::MissingClientId` and the screen must show it. This
/// proves the *wiring* (screen → runtime → `mc_auth` → screen), not the flow, which needs a
/// human and an app registration nobody here has.
#[test]
fn the_login_screen_is_driven_by_the_client_and_shows_a_typed_failure() {
    let dir = tempfile::tempdir().expect("tempdir");
    let Some(mut config) = config(dir.path(), "Dev") else {
        eprintln!("skip: no vanilla jar in this checkout");
        return;
    };
    config.login = mc_client::app::config::LoginMode::Microsoft;
    // The skip has to ask the same question `login_device_code` does. It used to test only
    // `MC_RS_MSA_CLIENT_ID`, but `AuthConfig::load` falls back to `%APPDATA%\mc-rs\config\
    // auth.json` — and once `mc-launcher` wrote a real client id there (2026-08-16) this test
    // stopped hitting `MissingClientId` and started making a genuine device-code request to
    // Microsoft, which does not resolve inside the 20 s below. Checking the loaded configuration
    // rather than one of its two sources is the assertion this test always meant to make.
    let configured = mc_auth::AuthConfig::load()
        .ok()
        .is_some_and(|auth| auth.require_client_id().is_ok());
    if configured {
        eprintln!("skip: a real MSA client id is configured; this test asserts its absence");
        return;
    }

    let mut client = Client::new(config, &mut |_, _| {}).expect("start");
    assert!(client.auth.is_none(), "Microsoft mode starts signed out");

    client.ui = mc_client::app::driver::Ui::Login(Box::default());
    client.handle_action(Action::BeginLogin);

    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    loop {
        client.poll();
        if let mc_client::app::driver::Ui::Login(screen) = &client.ui {
            if let mc_ui::screens::login::LoginState::Failed(message) = &screen.state {
                assert!(!message.is_empty(), "a failure must say something");
                assert!(!screen.advice.is_empty(), "and must offer advice");
                break;
            }
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the login attempt never resolved"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(
        client.auth.is_none(),
        "a failed sign-in must not sign anyone in"
    );
}
