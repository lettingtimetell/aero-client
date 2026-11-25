//! The client's 1.7.10 screen, HUD and camera hooks on the real client (real assets, offline
//! session on channels the test holds), each through the client's own entry point and next to a
//! 1.8.9 session as the negative control. Every test skips (prints "skip:") without the vanilla
//! jar, like the driver's own tests, and the camera tests also without a GPU adapter.

use mc_math::BlockPos;
use mc_protocol::connection::LoginOutcome;
use mc_protocol::play::{s2c as s2c47, C2SPlay, S2CPlay};
use mc_protocol::v5::{s2c as s2c5, S2CPlay5};
use mc_render::Camera;
use mc_ui::font::component::{ChatComponent, ChatStyle, ClickAction, ClickEvent};
use mc_ui::screens::chat::{ChatLayout, ChatScreen};
use mc_ui::screens::container::v5::open_window_v47;
use mc_ui::screens::container::ContainerKind;
use mc_ui::screens::sign::EditSignScreen;
use mc_ui::widget::ScaledResolution;
use tokio::sync::mpsc::{Sender, UnboundedReceiver};

use super::*;
use crate::app::config::{ClientConfig, LoginMode};
use crate::error::ClientError;
use crate::net::{NetHandle, ToGame, ToNet};
use crate::session::JoinSettings;
use crate::sync::event::ScoreboardEvent;
use crate::sync::{ClientEvent, WindowEvent};

type Harness = (
    Client,
    tempfile::TempDir,
    Sender<ToGame>,
    UnboundedReceiver<ToNet>,
);

/// A client with an offline `version` session that has joined a world (`Dev`, entity 27).
fn joined(version: GameVersion) -> Option<Harness> {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClientConfig {
        game_dir: dir.path().to_path_buf(),
        login: LoginMode::Offline("Dev".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout");
        return None;
    }
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    let (handle, to_game, to_net) = NetHandle::new_pair();
    client.session = Some(Session::with_version(
        handle,
        JoinSettings::default(),
        version,
    ));
    to_game
        .try_send(ToGame::Connected(LoginOutcome {
            uuid: uuid::Uuid::from_u128(7),
            username: "Dev".to_owned(),
            compression_threshold: None,
            encrypted: false,
        }))
        .expect("room");
    let join = match version {
        GameVersion::V1_7_10 => ToGame::PacketV5(Box::new(S2CPlay5::JoinGame(s2c5::JoinGame {
            entity_id: 27,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 60,
            level_type: "flat".to_owned(),
        }))),
        GameVersion::V1_8_9 => ToGame::Packet(S2CPlay::JoinGame(s2c47::JoinGame {
            entity_id: 27,
            hardcore_mode: false,
            game_mode: 0,
            dimension: 0,
            difficulty: 2,
            max_players: 60,
            level_type: "flat".to_owned(),
            reduced_debug_info: false,
        })),
    };
    to_game.try_send(join).expect("room");
    client.poll_in_place();
    client.drain_v5_before_world(&mut Camera::default());
    assert!(
        client
            .session
            .as_ref()
            .is_some_and(|session| session.phase().is_live()),
        "joined"
    );
    Some((client, dir, to_game, to_net))
}

/// Every serverbound packet queued so far.
fn sent(to_net: &mut UnboundedReceiver<ToNet>) -> Vec<C2SPlay> {
    let mut packets = Vec::new();
    while let Ok(message) = to_net.try_recv() {
        if let ToNet::Send(packet) = message {
            packets.push(packet);
        }
    }
    packets
}

fn ctx() -> ScreenCtx {
    ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
}

/// Open the chat over one line, `<a> ` then a `/spawn` RUN_COMMAND name, pointer on the name.
fn chat_over_command(client: &mut Client) {
    let name = ChatComponent::text("Steve").with_style(ChatStyle {
        click_event: Some(ClickEvent {
            action: ClickAction::RunCommand,
            value: "/spawn".to_owned(),
        }),
        ..ChatStyle::default()
    });
    let layout = ChatLayout::from_settings(&client.settings);
    client.chat.add_message(
        ChatComponent::text("<a> ").with_extra(name),
        0,
        0,
        &layout,
        &client.font,
        &client.assets.language,
        true,
    );
    client.ui = Ui::Chat(Box::new(ChatScreen::new("", 0, &client.font)));
    let x = 3 + client.font.string_width("<a> ") + 2;
    client.cursor = (x, ctx().height - 30);
}

/// A press on a chat command, through the client's `mouse_clicked` (both chat hooks): 1.7.10
/// runs it only with Chat Links on and records it in the sent history; 1.8.9 runs it either
/// way and records nothing.
#[test]
fn a_chat_command_click_follows_the_sessions_rules() {
    for (version, chat_links, runs, recorded) in [
        (GameVersion::V1_7_10, true, true, true),
        (GameVersion::V1_7_10, false, false, false),
        (GameVersion::V1_8_9, true, true, false),
        (GameVersion::V1_8_9, false, true, false),
    ] {
        let Some((mut client, _dir, _to_game, mut to_net)) = joined(version) else {
            return;
        };
        sent(&mut to_net);
        client.settings.chat_links = chat_links;
        chat_over_command(&mut client);
        client.mouse_clicked(0, ctx());
        let commands: Vec<String> = sent(&mut to_net)
            .into_iter()
            .filter_map(|packet| match packet {
                C2SPlay::ChatMessage(message) => Some(message.message),
                _ => None,
            })
            .collect();
        let case = format!("{version}, chat links {chat_links}");
        let expected: &[&str] = if runs { &["/spawn"] } else { &[] };
        assert_eq!(commands, expected, "{case}");
        let history: &[&str] = if recorded { &["/spawn"] } else { &[] };
        assert_eq!(client.chat.sent_messages(), history, "{case}");
    }
}

#[test]
fn a_1_7_10_window_opens_1_7_10_containers_and_the_inventory_follows_the_session() {
    for (version, kind) in [
        (GameVersion::V1_7_10, ContainerKind::EnchantingV5),
        (GameVersion::V1_8_9, ContainerKind::Enchanting),
    ] {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(version) else {
            return;
        };
        assert_eq!(client.player_container.game_version(), version);
        let lowered = open_window_v47(&s2c5::OpenWindow {
            window_id: 3,
            inventory_type: 4,
            title: String::new(),
            slot_count: 9,
            use_provided_title: false,
            entity_id: 0,
        })
        .expect("an enchanting table");
        client.react_to_window(&WindowEvent::Open(lowered));
        let Ui::Container(screen) = &client.ui else {
            panic!("a container screen ({version})");
        };
        assert_eq!(screen.container.kind, kind, "{version}");
    }
}

#[test]
fn the_1_7_10_resource_pack_prompt_enables_the_server_on_either_answer_and_never_answers() {
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let Some((mut client, _dir, _to_game, mut to_net)) = joined(version) else {
            return;
        };
        sent(&mut to_net);
        client.react(&ClientEvent::ResourcePack {
            url: "http://127.0.0.1:9/pack.zip".to_owned(),
            hash: String::new(),
        });
        assert!(
            matches!(client.ui, Ui::ConfirmServerResourcePack(_, _, _)),
            "{version} prompts"
        );
        client.finish_server_resource_pack_choice(false);
        assert!(client.pending_server_resource_pack_hash.is_none());
        let statuses = sent(&mut to_net)
            .into_iter()
            .filter(|packet| matches!(packet, C2SPlay::ResourcePackStatus(_)))
            .count();
        if version.is_1_7_10() {
            assert_eq!(client.current_server_resource_mode, ResourceMode::Enabled);
            assert_eq!(statuses, 0);
        } else {
            assert_eq!(client.current_server_resource_mode, ResourceMode::Disabled);
            assert_eq!(statuses, 1, "1.8.9 declines with a C19");
        }
    }
}

#[test]
fn only_a_zip_file_name_is_fetched_in_1_7_10() {
    let Some((mut client, _dir, _to_game, mut to_net)) = joined(GameVersion::V1_7_10) else {
        return;
    };
    client.current_server_resource_mode = ResourceMode::Enabled;
    for url in [
        "http://127.0.0.1:9/pack.txt",
        "http://127.0.0.1:9/zip?pack.zip",
    ] {
        client.react(&ClientEvent::ResourcePack {
            url: url.to_owned(),
            hash: String::new(),
        });
        assert!(client.pending_server_resource_pack_hash.is_none(), "{url}");
    }
    client.react(&ClientEvent::ResourcePack {
        url: "http://127.0.0.1:9/a/pack.zip?token=1".to_owned(),
        hash: String::new(),
    });
    assert_eq!(
        client.pending_server_resource_pack_hash.as_deref(),
        Some("")
    );
    client.cancel_server_resource_pack_download();
    assert!(sent(&mut to_net)
        .iter()
        .all(|packet| !matches!(packet, C2SPlay::ResourcePackStatus(_))));
}

/// java17 forgets the server's current pack without a reload when a new one is requested
/// (`func_148529_f`): the old pack stays drawn until the new one is installed, with one reload.
#[test]
fn a_new_1_7_10_pack_replaces_the_old_one_with_one_reload() {
    let Some((mut client, _dir, _to_game, _to_net)) = joined(GameVersion::V1_7_10) else {
        return;
    };
    client.current_server_resource_mode = ResourceMode::Enabled;
    let marker = assets::server_resource_pack_dir(&client.config).join(".active");
    std::fs::create_dir_all(marker.parent().expect("a folder")).expect("the pack folder");
    std::fs::write(&marker, "legacy").expect("an active server pack");
    client.pending_pack_reload = false;
    client.react(&ClientEvent::ResourcePack {
        url: "http://127.0.0.1:9/pack.zip".to_owned(),
        hash: String::new(),
    });
    assert!(!marker.exists(), "the old pack is forgotten");
    assert!(
        !client.pending_pack_reload,
        "and stays drawn: no reload yet"
    );
    client.cancel_server_resource_pack_download();
}

/// The held list key draws 1.7.10's grid from the name-keyed list (the hook in
/// `draw_tab_list`), and the Ping HUD finds the player there by name (the hook in the mods'
/// HUD data); a 1.8.9 session reads neither.
#[test]
fn the_1_7_10_player_list_is_a_grid_by_name_and_the_ping_hud_finds_the_player_by_name() {
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(version) else {
            return;
        };
        {
            let state = client.session.as_mut().unwrap().state_mut();
            state.players_v5.handle_player_list_item("Zed", true, 90);
            state.players_v5.handle_player_list_item("Dev", true, 42);
            state.players_v5.handle_player_list_item("Amy", true, 700);
        }
        client.tab_list_held = true;
        client.draw_tab_list(&mut DrawList::new(), &ctx(), 0);
        let names: Vec<&str> = client
            .tab_list
            .cells_v5()
            .iter()
            .map(|cell| cell.name.as_str())
            .collect();
        let expected: &[&str] = if version.is_1_7_10() {
            // Arrival order, unsorted.
            &["Zed", "Dev", "Amy"]
        } else {
            &[]
        };
        assert_eq!(names, expected, "{version}");

        client.mods.settings.set_enabled("ping", true);
        client.draw_mod_hud(&mut DrawList::new(), &ctx());
        let ping = version.is_1_7_10().then_some(42);
        assert_eq!(client.mods.hud_data.ping_ms, ping, "{version}");
    }
}

/// The sidebar the HUD draws (the hook in `sidebar_snapshot`): 1.7.10 shows a `#` name and
/// nothing at all above 15 scores; 1.8.9 hides `#` names and shows the top 15.
#[test]
fn the_1_7_10_sidebar_shows_hash_names_and_nothing_above_fifteen_scores() {
    let score = |player: &str, value: i32| {
        ScoreboardEvent::Score(s2c47::UpdateScore {
            player_name: player.to_owned(),
            objective_name: "kills".to_owned(),
            action: s2c47::UpdateScoreAction::Change { value },
        })
    };
    let rows = |client: &Client| {
        client.sidebar_snapshot().map(|sidebar| {
            sidebar
                .visible_rows()
                .iter()
                .map(|(name, _)| name.clone())
                .collect::<Vec<_>>()
        })
    };
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(version) else {
            return;
        };
        let v17 = version.is_1_7_10();
        let board = &mut client.session.as_mut().unwrap().state_mut().scoreboard;
        board.apply(&ScoreboardEvent::Objective(s2c47::ScoreboardObjective {
            objective_name: "kills".to_owned(),
            mode: s2c47::ScoreboardObjectiveMode::Create {
                display_name: "Kills".to_owned(),
                render_type: s2c47::ScoreboardRenderType::Integer,
            },
        }));
        board.apply(&ScoreboardEvent::Display(s2c47::DisplayScoreboard {
            position: 1,
            objective_name: "kills".to_owned(),
        }));
        board.apply(&score("#tie", 50));
        board.apply(&score("Amy", 3));
        let expected: &[&str] = if v17 { &["Amy", "#tie"] } else { &["Amy"] };
        assert_eq!(
            rows(&client),
            Some(expected.iter().map(|s| (*s).to_owned()).collect())
        );

        let board = &mut client.session.as_mut().unwrap().state_mut().scoreboard;
        for points in 0..14 {
            board.apply(&score(&format!("p{points}"), points));
        }
        let sixteen = rows(&client);
        if v17 {
            assert_eq!(sixteen, None, "16 scores: no sidebar");
        } else {
            assert_eq!(sixteen.map(|rows| rows.len()), Some(15), "the top 15");
        }
    }
}

/// A finished (here: failed) pack download answers nothing in a 1.7.10 session, which has no
/// status packet (the hook in `send_server_resource_pack_status`); 1.8.9 sends C19.
#[test]
fn a_finished_1_7_10_pack_download_sends_no_status() {
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let Some((mut client, _dir, _to_game, mut to_net)) = joined(version) else {
            return;
        };
        sent(&mut to_net);
        client.pending_server_resource_pack_hash = Some(String::new());
        let refused = ClientError::Io(std::io::Error::other("connection refused"));
        client.finish_server_resource_pack_download("", Err(refused));
        assert!(client.pending_server_resource_pack_hash.is_none());
        let statuses = sent(&mut to_net)
            .into_iter()
            .filter(|packet| matches!(packet, C2SPlay::ResourcePackStatus(_)))
            .count();
        let expected = usize::from(!version.is_1_7_10());
        assert_eq!(statuses, expected, "{version}");
    }
}

#[test]
fn a_1_7_10_sign_editor_stops_at_fifteen_characters() {
    for (version, length) in [(GameVersion::V1_7_10, 15), (GameVersion::V1_8_9, 20)] {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(version) else {
            return;
        };
        let editor = EditSignScreen::new(BlockPos::new(0, 64, 0), Default::default());
        client.show_screen(Some(Ui::EditSign(Box::new(editor))));
        let Ui::EditSign(editor) = &mut client.ui else {
            panic!("the sign editor");
        };
        for _ in 0..20 {
            editor.key_typed(Some('i'), 0, &client.font);
        }
        // 20 narrow `i`s fit 1.8.9's 90 pixels.
        assert_eq!(editor.line(0).unwrap().len(), length, "{version}");
    }
}

/// The shell around one joined client: the real `ShellApp::tick` and `ShellApp::frame` (whose
/// look and smooth camera steps are the ones under test) on a headless GPU, with the mouse fed
/// in as raw counts and the Zoom mod's key as a key.
struct Shell<'a> {
    gpu: &'a mc_render::Gpu,
    phases: mc_render::PhaseList,
    camera: Camera,
    input: mc_render::InputState,
    settings: mc_render::GameSettings,
    stats: mc_render::FrameStats,
    ticks: u64,
}

impl<'a> Shell<'a> {
    fn new(gpu: &'a mc_render::Gpu, client: &mut Client) -> Self {
        let mut phases = mc_render::PhaseList::new();
        client.install(gpu, &mut phases, 64, 64);
        client.ui = Ui::InGame;
        client.mods.settings.set_enabled("zoom", true);
        let mut input = mc_render::InputState::new();
        input.request_cursor(mc_render::CursorMode::Grabbed);
        Self {
            gpu,
            phases,
            camera: Camera::default(),
            input,
            settings: mc_render::GameSettings::default(),
            stats: mc_render::FrameStats::new(),
            ticks: 0,
        }
    }

    /// The yaw mouse look turns: the local player's, or the camera's before there is one.
    fn yaw(&self, client: &Client) -> f32 {
        client
            .session
            .as_ref()
            .and_then(Session::predictor)
            .map_or(self.camera.yaw, |predictor| predictor.state.yaw)
    }

    /// One frame with `counts` of sideways mouse motion: how far it turned the player.
    fn frame(&mut self, client: &mut Client, counts: f64, partial_ticks: f32) -> f32 {
        assert!(matches!(client.ui, Ui::InGame), "in game, the mouse looks");
        let before = self.yaw(client);
        self.input.on_raw_motion(counts, 0.0);
        mc_render::ShellApp::frame(
            client,
            &mut mc_render::FrameCtx {
                gpu: self.gpu,
                phases: &mut self.phases,
                camera: &mut self.camera,
                input: &mut self.input,
                settings: &self.settings,
                partial_ticks,
                stats: &self.stats,
            },
        );
        self.yaw(client) - before
    }

    fn tick(&mut self, client: &mut Client) {
        self.ticks += 1;
        mc_render::ShellApp::tick(
            client,
            &mut mc_render::TickCtx {
                input: &mut self.input,
                settings: &mut self.settings,
                camera: &mut self.camera,
                tick: self.ticks,
            },
        );
    }

    /// Hold (or release) the Zoom mod's key; it takes effect at the end of the next frame.
    fn zoom_key(&mut self, held: bool) {
        self.input
            .on_key(winit::keyboard::KeyCode::KeyC, held, false);
    }
}

/// The smooth camera's whole state (filters, their output, the frame accumulators).
fn smoothing(client: &Client) -> String {
    format!("{:?}", client.smooth_camera)
}

/// A headless GPU, or `None` (and "skip:") without an adapter.
fn headless_gpu() -> Option<mc_render::Gpu> {
    match mc_render::Gpu::headless(64, 64, mc_render::GpuOptions::default()) {
        Ok(gpu) => Some(gpu),
        Err(error) => {
            eprintln!("skip: no GPU adapter: {error}");
            None
        }
    }
}

/// Smooth camera on, mouse flicks over frames and ticks, one more flick the tick has not yet
/// filtered: the state the tests below then switch off.
fn flick_with_smoothing(shell: &mut Shell<'_>, client: &mut Client) {
    for _ in 0..3 {
        shell.frame(client, 40.0, 0.5);
        shell.tick(client);
    }
    shell.frame(client, 40.0, 0.5);
}

/// The vanilla smooth camera setting, through the client's own tick and frame (both hooks):
/// 1.7.10 leaves the filters and the unfiltered frame deltas exactly as they were while the
/// setting is off, so the camera turns on its own when it is turned on again (java17); 1.8.9
/// clears them and starts from rest.
#[test]
fn the_1_7_10_smooth_camera_setting_keeps_its_state_while_off() {
    let Some(gpu) = headless_gpu() else {
        return;
    };
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let Some((mut client, _dir, _to_game, _to_net)) = joined(version) else {
            return;
        };
        let v17 = version.is_1_7_10();
        let mut shell = Shell::new(&gpu, &mut client);
        assert_ne!(shell.frame(&mut client, 40.0, 0.5), 0.0, "the mouse looks");
        client.settings.smooth_camera = true;
        flick_with_smoothing(&mut shell, &mut client);
        let on = smoothing(&client);

        client.settings.smooth_camera = false;
        // Off, the delta passes through in both versions.
        assert_ne!(shell.frame(&mut client, 8.0, 0.5), 0.0, "{version}");
        assert_eq!(shell.frame(&mut client, 0.0, 0.5), 0.0, "{version}");
        // The frame step: 1.7.10 keeps the unfiltered deltas.
        assert_eq!(smoothing(&client) == on, v17, "{version}: after off frames");
        for _ in 0..3 {
            shell.tick(&mut client);
        }
        // The tick step: 1.7.10 keeps the filters.
        assert_eq!(smoothing(&client) == on, v17, "{version}: after off ticks");

        // On again, with the mouse still: the kept state turns 1.7.10's camera on its own.
        client.settings.smooth_camera = true;
        let first = shell.frame(&mut client, 0.0, 0.75);
        shell.tick(&mut client);
        let second = shell.frame(&mut client, 0.0, 0.5);
        assert_eq!(first != 0.0, v17, "{version}: {first}");
        assert_eq!(second != 0.0, v17, "{version}: {second}");
    }
}

/// One script for both versions: every frame's turn and the smooth camera's state after every
/// step.
fn zoom_script(
    gpu: &mc_render::Gpu,
    version: GameVersion,
    script: impl Fn(&mut Shell<'_>, &mut Client, &mut dyn FnMut(&Client, f32)),
) -> Option<Vec<(f32, String)>> {
    let (mut client, _dir, _to_game, _to_net) = joined(version)?;
    let mut shell = Shell::new(gpu, &mut client);
    let mut trace = Vec::new();
    script(
        &mut shell,
        &mut client,
        &mut |client: &Client, turn: f32| {
            trace.push((turn, smoothing(client)));
        },
    );
    Some(trace)
}

/// The Zoom mod's "smooth camera while zoomed" is not vanilla and behaves exactly as in a 1.8.9
/// session: zoom, flick, release, zoom again (it must not turn on its own: those turns would
/// reach the server), and a zoom right after the vanilla setting was switched off.
#[test]
fn the_zoom_mods_smoothing_is_the_same_in_both_versions() {
    let Some(gpu) = headless_gpu() else {
        return;
    };
    // Zoom, flick, release, zoom again with the mouse still.
    let rezoom = |shell: &mut Shell<'_>, client: &mut Client, log: &mut dyn FnMut(&Client, f32)| {
        shell.zoom_key(true);
        let turn = shell.frame(client, 0.0, 0.5);
        log(client, turn);
        for _ in 0..3 {
            let turn = shell.frame(client, 40.0, 0.5);
            log(client, turn);
            shell.tick(client);
            log(client, 0.0);
        }
        let turn = shell.frame(client, 40.0, 0.5);
        log(client, turn);
        shell.zoom_key(false);
        for _ in 0..3 {
            let turn = shell.frame(client, 0.0, 0.5);
            log(client, turn);
            shell.tick(client);
            log(client, 0.0);
        }
        shell.zoom_key(true);
        for _ in 0..4 {
            let turn = shell.frame(client, 0.0, 0.5);
            log(client, turn);
            shell.tick(client);
            log(client, 0.0);
        }
    };
    // The vanilla setting on, a flick, the setting off, then a zoom with the mouse still.
    let after_setting =
        |shell: &mut Shell<'_>, client: &mut Client, log: &mut dyn FnMut(&Client, f32)| {
            client.settings.smooth_camera = true;
            flick_with_smoothing(shell, client);
            client.settings.smooth_camera = false;
            let turn = shell.frame(client, 0.0, 0.5);
            log(client, turn);
            shell.tick(client);
            shell.zoom_key(true);
            for _ in 0..4 {
                let turn = shell.frame(client, 0.0, 0.5);
                log(client, turn);
                shell.tick(client);
                log(client, 0.0);
            }
        };

    let Some(v17) = zoom_script(&gpu, GameVersion::V1_7_10, rezoom) else {
        return;
    };
    let v18 = zoom_script(&gpu, GameVersion::V1_8_9, rezoom).expect("the same assets");
    assert_eq!(v17, v18, "zoom, flick, release, zoom again");
    // The second zoom, the last eight steps, starts from rest: no turn without the mouse.
    assert!(v17[v17.len() - 8..].iter().all(|(turn, _)| *turn == 0.0));
    assert!(
        v17.iter().any(|(turn, _)| *turn != 0.0),
        "the flicks turned"
    );

    let v17 = zoom_script(&gpu, GameVersion::V1_7_10, after_setting).expect("assets");
    let v18 = zoom_script(&gpu, GameVersion::V1_8_9, after_setting).expect("assets");
    // The same turns all the way (none: the mouse is still); the same state from the zoom's
    // first tick on (the third step: before it, 1.7.10 still holds what the setting left).
    let turns = |trace: &[(f32, String)]| trace.iter().map(|(turn, _)| *turn).collect::<Vec<_>>();
    assert_eq!(turns(&v17), turns(&v18), "a zoom after the setting");
    assert!(turns(&v17).iter().all(|turn| *turn == 0.0));
    assert_ne!(
        v17[1], v18[1],
        "the setting's state, kept in 1.7.10 until the zoom"
    );
    assert_eq!(v17[2..], v18[2..], "from the zoom's first tick");
}
