//! Checks of replay fidelity against real recordings and of the
//! screen and inventory records (`gui.rs`). Test-only.

use std::collections::HashMap;

use mc_math::DVec3;
use mc_protocol::play::S2CPlay;
use mc_replay::{Record, ReplayMeta, StreamId};

use super::super::super::{Client, Ui};
use super::codec::{TickRecord, KIND_FRAME, KIND_TICK};
use super::gui::{ScreenState, KIND_INVENTORY, KIND_SCREEN};
use super::{decode_packet, FrameRecord, ReplayCameraMode};
use crate::app::config::{ClientConfig, LoginMode};

/// An offline client with its game dir in a temp folder (`None`: no vanilla jar here).
fn offline_client() -> Option<(Client, tempfile::TempDir)> {
    let dir = tempfile::tempdir().expect("tempdir");
    let config = ClientConfig {
        game_dir: dir.path().to_path_buf(),
        login: LoginMode::Offline("Dev".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
        return None;
    }
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    // These checks play real replays; never ask for their skins.
    client.set_skin_downloads(false);
    Some((client, dir))
}

// ---------------------------------------------------------------------------------------------
// A remote player's position: playback against vanilla's own arithmetic
// ---------------------------------------------------------------------------------------------

/// One remote player as vanilla 1.8.9's client moves it: `NetHandlerPlayClient` keeps the
/// fixed-point `serverPosX/Y/Z` and hands `setPositionAndRotation2(serverPos / 32, .., 3)` to
/// `EntityOtherPlayerMP`, which eases a third, a half, then all of the way there over the next
/// three ticks (`onLivingUpdate`), with no collision at all.
#[derive(Debug, Clone, Copy)]
struct VanillaRemote {
    server: [i32; 3],
    pos: DVec3,
    last: DVec3,
    target: DVec3,
    steps: i32,
    ticked: bool,
}

fn fixed(server: [i32; 3]) -> DVec3 {
    DVec3::new(
        f64::from(server[0]) / 32.0,
        f64::from(server[1]) / 32.0,
        f64::from(server[2]) / 32.0,
    )
}

/// `handleSpawnPlayer`, `handleEntityMovement` (S14 to S17), `handleEntityTeleport` (with its
/// dead zone), `handleDestroyEntities`.
fn vanilla_packet(model: &mut HashMap<i32, VanillaRemote>, packet: &S2CPlay) {
    let mut moved = |id: i32, delta: [i8; 3]| {
        if let Some(remote) = model.get_mut(&id) {
            for (axis, step) in remote.server.iter_mut().zip(delta) {
                *axis = axis.wrapping_add(i32::from(step));
            }
            remote.target = fixed(remote.server);
            remote.steps = 3;
        }
    };
    match packet {
        S2CPlay::JoinGame(_) | S2CPlay::Respawn(_) => model.clear(),
        S2CPlay::SpawnPlayer(spawn) => {
            let server = [spawn.x, spawn.y, spawn.z];
            let pos = fixed(server);
            model.insert(
                spawn.entity_id,
                VanillaRemote {
                    server,
                    pos,
                    last: pos,
                    target: pos,
                    steps: 0,
                    ticked: false,
                },
            );
        }
        S2CPlay::Entity(p) => moved(p.entity_id, [0, 0, 0]),
        S2CPlay::EntityLook(p) => moved(p.entity_id, [0, 0, 0]),
        S2CPlay::EntityRelMove(p) => moved(p.entity_id, [p.delta_x, p.delta_y, p.delta_z]),
        S2CPlay::EntityLookAndRelMove(p) => {
            moved(p.entity_id, [p.delta_x, p.delta_y, p.delta_z]);
        }
        S2CPlay::EntityTeleport(p) => {
            if let Some(remote) = model.get_mut(&p.entity_id) {
                remote.server = [p.x, p.y, p.z];
                let to = fixed(remote.server);
                let near = (remote.pos.x - to.x).abs() < 0.03125
                    && (remote.pos.y - to.y).abs() < 0.015_625
                    && (remote.pos.z - to.z).abs() < 0.03125;
                remote.target = if near { remote.pos } else { to };
                remote.steps = 3;
            }
        }
        S2CPlay::DestroyEntities(p) => {
            for id in &p.entity_ids {
                model.remove(id);
            }
        }
        _ => {}
    }
}

/// `World.updateEntityWithOptionalForce` (`lastTickPos = pos`) then
/// `EntityOtherPlayerMP.onLivingUpdate`'s ease.
fn vanilla_tick(model: &mut HashMap<i32, VanillaRemote>) {
    for remote in model.values_mut() {
        remote.last = remote.pos;
        remote.ticked = true;
        if remote.steps > 0 {
            remote.pos += (remote.target - remote.pos) / f64::from(remote.steps);
            remote.steps -= 1;
        }
    }
}

/// The named remote player in the replayed world: `(network id, pos, last tick pos)`.
fn replayed_player(client: &Client, name: &str) -> Option<(i32, DVec3, DVec3)> {
    let session = client.session.as_ref()?;
    let state = session.state();
    let (id, tracked) = state.entities.iter().find(|(_, tracked)| {
        tracked
            .player_profile
            .as_ref()
            .is_some_and(|profile| profile.username == name)
    })?;
    let pos = state
        .world
        .ecs
        .get::<&mc_world::entity::Position>(tracked.entity)
        .ok()?
        .0;
    Some((id, pos, tracked.last_tick_pos))
}

/// The user's "why does that player go fully into the wooden log?": replay a real recording
/// through the real playback engine, and after every replayed tick compare the named player's
/// position with vanilla's arithmetic on the same packets (`VanillaRemote`). Every tick must
/// match bit for bit. `MC_RB_REPLAY=<file> MC_RB_PLAYER=<name> [MC_RB_CSV=<out.csv>]`.
#[test]
#[ignore = "needs a real replay (MC_RB_REPLAY) and the vanilla jar (MC_JARS_DIR)"]
fn a_remote_player_is_exactly_where_vanilla_puts_him() {
    let Some(path) = std::env::var_os("MC_RB_REPLAY") else {
        eprintln!("skip: set MC_RB_REPLAY");
        return;
    };
    let Ok(name) = std::env::var("MC_RB_PLAYER") else {
        eprintln!("skip: set MC_RB_PLAYER to the player to follow");
        return;
    };
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let reader = mc_replay::Reader::open(&path).expect("the replay opens");
    let records: Vec<Record> = reader.records().map(|r| r.expect("record")).collect();
    client
        .replay_open(std::path::Path::new(&path))
        .expect("replay_open");
    let mut model: HashMap<i32, VanillaRemote> = HashMap::new();
    let mut next = 0usize;
    let mut compared = 0u64;
    let mut differed = 0u64;
    let mut last_tick = None;
    let mut csv = String::from(
        "time_s,client_tick,id,client_x,client_y,client_z,vanilla_x,vanilla_y,vanilla_z,\
         client_last_x,client_last_z\n",
    );
    for frame_index in 0..records.len() {
        let record = &records[frame_index];
        if record.stream != StreamId::Client(KIND_FRAME) {
            continue;
        }
        let Some(frame) = FrameRecord::decode(&record.payload) else {
            continue;
        };
        if last_tick == Some(frame.client_tick) {
            continue;
        }
        last_tick = Some(frame.client_tick);
        let time = record.time;
        client.replay_render_at(time);
        // The model takes every record up to this frame's time, in order.
        while next < records.len() && records[next].time <= time {
            let r = &records[next];
            match r.stream {
                StreamId::Packet => {
                    if let Ok(packet) = decode_packet(&r.payload) {
                        vanilla_packet(&mut model, &packet);
                    }
                }
                StreamId::Client(KIND_TICK) if TickRecord::decode(&r.payload).is_some() => {
                    vanilla_tick(&mut model);
                }
                _ => {}
            }
            next += 1;
        }
        let Some((id, pos, last)) = replayed_player(&client, &name) else {
            continue;
        };
        let Some(vanilla) = model.get(&id) else {
            continue;
        };
        compared += 1;
        let same = pos == vanilla.pos && (!vanilla.ticked || last == vanilla.last);
        if !same {
            differed += 1;
            if differed <= 10 {
                eprintln!(
                    "differs at {:.3} s (tick {}): client {pos:?} last {last:?}, vanilla {:?} \
                     last {:?}",
                    time as f64 / 1e6,
                    frame.client_tick,
                    vanilla.pos,
                    vanilla.last
                );
            }
        }
        csv.push_str(&format!(
            "{:.3},{},{id},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            time as f64 / 1e6,
            frame.client_tick,
            pos.x,
            pos.y,
            pos.z,
            vanilla.pos.x,
            vanilla.pos.y,
            vanilla.pos.z,
            last.x,
            last.z
        ));
        // The moment in question: what the replayed world has where he stands.
        let seconds = time as f64 / 1e6;
        if (7.5..10.6).contains(&seconds) && compared.is_multiple_of(4) {
            let world = client.session.as_ref().map(|session| session.world());
            let block = |x: i32, y: i32, z: i32| {
                world.map_or(0, |world| {
                    world.block_at(mc_math::BlockPos::new(x, y, z)).id
                })
            };
            let (bx, by, bz) = (
                pos.x.floor() as i32,
                pos.y.floor() as i32,
                pos.z.floor() as i32,
            );
            eprintln!(
                "{seconds:7.3} s  {name} at ({:.4}, {:.4}, {:.4})  feet block ({bx},{by},{bz}) \
                 = state {} (block id {}), head block = state {}",
                pos.x,
                pos.y,
                pos.z,
                block(bx, by, bz),
                block(bx, by, bz) >> 4,
                block(bx, by + 1, bz),
            );
        }
    }
    if let Some(out) = std::env::var_os("MC_RB_CSV") {
        std::fs::write(out, csv).expect("write the CSV");
    }
    eprintln!("{name}: {compared} ticks compared, {differed} differ from vanilla's arithmetic");
    let stats = client.replay_stats().expect("stats");
    eprintln!(
        "playback: {} packets, {} decode errors, {} ticks, {} player mismatches",
        stats.packets, stats.packet_errors, stats.ticks, stats.player_mismatches
    );
    assert!(compared > 100, "the player was found and followed");
    assert_eq!(
        differed, 0,
        "playback places {name} exactly where vanilla does"
    );
    client.replay_close();
}

// ---------------------------------------------------------------------------------------------
// Pictures: a real replay in the Player and free cameras, and screens drawn from records
// ---------------------------------------------------------------------------------------------

/// Insert a client record at `time`, after every record already at or before it.
fn insert_record(records: &mut Vec<Record>, time: u64, kind: u8, payload: Vec<u8>) {
    let index = records.partition_point(|record| record.time <= time);
    records.insert(
        index,
        Record {
            time,
            stream: StreamId::Client(kind),
            payload,
        },
    );
}

fn screen_payload(record: &super::gui::ScreenRecord) -> Vec<u8> {
    let mut out = Vec::new();
    record.encode_into(&mut out);
    out
}

/// The slots of window `window` as the replay's packets left them by `time`.
fn window_slots(records: &[Record], window: u8, time: u64) -> Vec<Option<mc_protocol::ItemStack>> {
    let mut slots = Vec::new();
    for record in records.iter().take_while(|record| record.time <= time) {
        if record.stream != StreamId::Packet {
            continue;
        }
        match decode_packet(&record.payload) {
            Ok(S2CPlay::WindowItems(items)) if items.window_id == window => {
                slots.clone_from(&items.items);
            }
            Ok(S2CPlay::SetSlot(set)) if set.window_id == window as i8 => {
                let index = usize::try_from(set.slot).unwrap_or(usize::MAX);
                if index < slots.len() {
                    slots[index].clone_from(&set.item);
                }
            }
            _ => {}
        }
    }
    slots
}

/// One picture: `(file, time in seconds, camera, free camera pose)`.
type Shot = (
    &'static str,
    f64,
    ReplayCameraMode,
    Option<(DVec3, f32, f32)>,
);

/// Render `shots` (in time order) of `records` and write them to `out`.
fn render_shots(
    records: Vec<Record>,
    meta: ReplayMeta,
    shots: &[Shot],
    out: &std::path::Path,
    size: (u32, u32),
) {
    use mc_render::{Gpu, GpuOptions, PhaseList};
    let mut gpu = match Gpu::headless(
        size.0,
        size.1,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: no headless GPU ({error})");
            return;
        }
    };
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, size.0, size.1);
    let mut camera = crate::app::driver::menu_camera();
    camera.set_viewport(size.0, size.1);
    client
        .replay_open_records(meta, records)
        .expect("the replay opens");
    client.set_replay_deterministic_mesh(true);
    for (name, seconds, mode, free) in shots {
        client.replay_set_camera_mode(*mode);
        if let Some((pos, yaw, pitch)) = free {
            client.replay_set_free_camera(*pos, *yaw, *pitch);
        }
        let time = (seconds * 1e6) as u64;
        let partial = client.replay_render_at(time);
        for warm in 0..900 {
            client.frame(&gpu, &mut camera, partial);
            if warm >= 6 && client.replay_terrain_settled() {
                break;
            }
        }
        client.frame(&gpu, &mut camera, partial);
        let target = gpu.acquire().expect("offscreen frame");
        phases.render(&gpu, target.view(), &camera, partial);
        let pixels = gpu.read_pixels().expect("readback");
        let path = out.join(name);
        image::save_buffer(
            &path,
            &pixels,
            size.0,
            size.1,
            image::ExtendedColorType::Rgba8,
        )
        .expect("write the PNG");
        eprintln!("wrote {}", path.display());
    }
    client.replay_close();
}

/// Pictures for the report: the followed player's moments in the Player camera and two free
/// cameras, the eating at 27.3 s with the packet-only inventory and with an inventory record, and
/// screens drawn from SCREEN records injected into the same real replay (the replay itself
/// predates them). `MC_RB_REPLAY=<file> MC_RB_PNG_DIR=<dir>`.
#[test]
#[ignore = "writes PNGs; needs MC_RB_REPLAY, MC_RB_PNG_DIR, a GPU and MC_JARS_DIR"]
fn replay_pictures() {
    let (Some(path), Some(out)) = (
        std::env::var_os("MC_RB_REPLAY"),
        std::env::var_os("MC_RB_PNG_DIR"),
    ) else {
        eprintln!("skip: set MC_RB_REPLAY and MC_RB_PNG_DIR");
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("png dir");
    let reader = mc_replay::Reader::open(&path).expect("the replay opens");
    let meta = reader.meta().clone();
    let records: Vec<Record> = reader.records().map(|r| r.expect("record")).collect();
    let size = (1280, 720);
    let pov = ReplayCameraMode::Pov;
    let free = ReplayCameraMode::Free;
    // As recorded.
    let shots: Vec<Shot> = vec![
        ("player_8.5s_player_camera.png", 8.5, pov, None),
        (
            "player_8.5s_free_from_bridge.png",
            8.5,
            free,
            Some((DVec3::new(4087.0, 63.6, -3938.5), -154.9, 9.6)),
        ),
        (
            "player_7.70s_walking_into_the_log_free.png",
            7.70,
            free,
            Some((DVec3::new(4088.6, 63.5, -3946.2), 0.0, 14.0)),
        ),
        (
            "player_7.80s_inside_the_log_free.png",
            7.80,
            free,
            Some((DVec3::new(4088.6, 63.5, -3946.2), 0.0, 14.0)),
        ),
        (
            "player_31.3s_stopped_by_the_same_log_free.png",
            31.3,
            free,
            Some((DVec3::new(4089.4, 63.6, -3946.0), 0.0, 14.0)),
        ),
        ("eat_27.3s_packet_only_inventory.png", 27.3, pov, None),
        (
            "player_89.0s_against_log_free.png",
            89.0,
            free,
            Some((DVec3::new(4060.4, 64.6, -3959.6), 90.0, 10.0)),
        ),
    ];
    render_shots(records.clone(), meta.clone(), &shots, &out, size);

    // With records of the new kinds injected (what a new recording carries).
    let mut injected = records.clone();
    let chest = window_slots(&records, 1, 1_600_000);
    let mut chest_record = super::gui::ScreenRecord {
        state: Some(ScreenState::Container { window_id: 1 }),
        pointer: Some((0, -67)),
        slots: Vec::new(),
        layout: None,
    };
    for (index, stack) in chest.into_iter().enumerate() {
        chest_record.slots.push((index as u16, stack));
    }
    insert_record(
        &mut injected,
        1_600_000,
        KIND_SCREEN,
        screen_payload(&chest_record),
    );
    let none = super::gui::ScreenRecord {
        state: Some(ScreenState::None),
        ..Default::default()
    };
    insert_record(&mut injected, 2_400_000, KIND_SCREEN, screen_payload(&none));
    let inventory = super::gui::ScreenRecord {
        state: Some(ScreenState::Inventory),
        pointer: Some((-72, 67)),
        slots: Vec::new(),
        layout: None,
    };
    insert_record(
        &mut injected,
        20_000_000,
        KIND_SCREEN,
        screen_payload(&inventory),
    );
    insert_record(
        &mut injected,
        20_600_000,
        KIND_SCREEN,
        screen_payload(&none),
    );
    // The sounds prove the recording client ate from the first hotbar slot at 26.9 s; an
    // inventory record says what it held (golden apples, as an illustration).
    let mut apples = Vec::new();
    super::gui::InventoryRecord {
        slots: vec![(
            36,
            Some(mc_protocol::ItemStack {
                item_id: 322,
                count: 19,
                damage: 0,
                nbt: None,
            }),
        )],
    }
    .encode_into(&mut apples);
    insert_record(&mut injected, 26_600_000, KIND_INVENTORY, apples);
    let chat = super::gui::ScreenRecord {
        state: Some(ScreenState::Chat {
            text: "gg wp".to_owned(),
        }),
        pointer: Some((40, 60)),
        slots: Vec::new(),
        layout: None,
    };
    insert_record(
        &mut injected,
        30_000_000,
        KIND_SCREEN,
        screen_payload(&chat),
    );
    let pause = super::gui::ScreenRecord {
        state: Some(ScreenState::Pause),
        pointer: Some((10, -20)),
        slots: Vec::new(),
        layout: None,
    };
    insert_record(
        &mut injected,
        35_000_000,
        KIND_SCREEN,
        screen_payload(&pause),
    );
    let menu = super::gui::ScreenRecord {
        state: Some(ScreenState::ModMenu),
        pointer: Some((-30, 10)),
        slots: Vec::new(),
        layout: None,
    };
    insert_record(
        &mut injected,
        40_000_000,
        KIND_SCREEN,
        screen_payload(&menu),
    );
    insert_record(
        &mut injected,
        41_000_000,
        KIND_SCREEN,
        screen_payload(&none),
    );
    let shots: Vec<Shot> = vec![
        ("screen_chest_1.8s.png", 1.8, pov, None),
        ("screen_inventory_20.2s.png", 20.2, pov, None),
        ("eat_27.3s_with_inventory_record.png", 27.3, pov, None),
        ("screen_chat_30.2s.png", 30.2, pov, None),
        ("screen_pause_35.2s.png", 35.2, pov, None),
        ("screen_mod_menu_40.2s.png", 40.2, pov, None),
        (
            "screen_chat_30.2s_free_camera_hides_it.png",
            30.2,
            free,
            None,
        ),
    ];
    render_shots(injected, meta, &shots, &out, size);
}

// ---------------------------------------------------------------------------------------------
// Screens and inventory clicks: recorded, then played back
// ---------------------------------------------------------------------------------------------

fn sword() -> mc_protocol::ItemStack {
    mc_protocol::ItemStack {
        item_id: 276,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

fn kinds(records: &[Record], kind: u8) -> Vec<&Record> {
    records
        .iter()
        .filter(|record| record.stream == StreamId::Client(kind))
        .collect()
}

/// Recording: the first frame writes the whole inventory and "no screen"; a click that changed
/// a slot writes that slot; an open chat writes its (privacy-masked) text and the pointer; an
/// unchanged frame writes nothing. Playback: the clicked item is in the hotbar and the chat is
/// the recorded screen. Negative control: the same replay without these records (an older
/// recording) keeps the packet-only inventory, which is the bug the records fix.
#[test]
fn screens_and_inventory_clicks_are_recorded_and_played_back() {
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    // In a world, with no screen up (an offline client starts on the title screen).
    client.ui = Ui::InGame;
    client.replay_debug_start_capture();
    client.nw_rb_gui_begin(1280, 720);
    {
        let records = client.replay_debug_captured();
        let inventory = kinds(records, KIND_INVENTORY);
        assert_eq!(
            inventory.len(),
            1,
            "a full inventory snapshot starts the recording"
        );
        let first = super::gui::InventoryRecord::decode(&inventory[0].payload).expect("decodes");
        assert_eq!(first.slots.len(), 45);
        let screen = kinds(records, KIND_SCREEN);
        assert_eq!(screen.len(), 1);
        let screen = super::gui::ScreenRecord::decode(&screen[0].payload).expect("decodes");
        assert_eq!(screen.state, Some(ScreenState::None));
        assert_eq!(screen.pointer, None, "no pointer without a screen");
    }
    // A click put a sword in the first hotbar slot (window 0, slot 36): only that is written.
    client.player_container.set_slot_contents(36, Some(sword()));
    client.nw_rb_gui_begin(1280, 720);
    {
        let inventory = kinds(client.replay_debug_captured(), KIND_INVENTORY);
        assert_eq!(inventory.len(), 2);
        let second = super::gui::InventoryRecord::decode(&inventory[1].payload).expect("decodes");
        assert_eq!(second.slots, vec![(36, Some(sword()))]);
    }
    // The chat, with a command being typed: the default keeps only the command's name.
    let chat = mc_ui::screens::chat::ChatScreen::new("/login hunter2", 0, &client.font);
    client.ui = Ui::Chat(Box::new(chat));
    client.cursor = (100, 50);
    client.nw_rb_gui_begin(1280, 720);
    let written = client.replay_debug_captured().len();
    {
        let screens = kinds(client.replay_debug_captured(), KIND_SCREEN);
        let last =
            super::gui::ScreenRecord::decode(&screens[screens.len() - 1].payload).expect("decodes");
        assert_eq!(
            last.state,
            Some(ScreenState::Chat {
                text: "/login *******".to_owned()
            })
        );
        assert!(last.pointer.is_some());
    }
    // Nothing changed: nothing written.
    client.nw_rb_gui_begin(1280, 720);
    assert_eq!(client.replay_debug_captured().len(), written);
    let records = client.replay_debug_finish_capture();
    client.ui = Ui::InGame;

    // Playback, from an empty inventory (a replay starts from nothing but its records).
    let play = |client: &mut Client, records: Vec<Record>| {
        client.player_container = mc_ui::screens::container::player_container(0);
        client.refresh_hotbar();
        let meta = ReplayMeta {
            duration: 1,
            ..ReplayMeta::default()
        };
        client.replay_open_records(meta, records).expect("opens");
        client.replay_render_at(1);
    };
    play(&mut client, records.clone());
    assert_eq!(client.player_container.slot_stack(36), Some(&sword()));
    assert_eq!(
        client.player_hud.hotbar[0],
        Some(sword()),
        "the hand holds the sword"
    );
    assert_eq!(
        client
            .tw_play
            .playback
            .as_deref()
            .map(|playback| playback.gui.screen.clone()),
        Some(ScreenState::Chat {
            text: "/login *******".to_owned()
        })
    );
    client.replay_close();

    // Negative control: without this lane's records the click is lost.
    let old: Vec<Record> = records
        .into_iter()
        .filter(|record| {
            !matches!(
                record.stream,
                StreamId::Client(KIND_INVENTORY | KIND_SCREEN)
            )
        })
        .collect();
    let meta_only_has_packets = old.is_empty();
    if meta_only_has_packets {
        // An empty replay cannot be opened; one harmless record stands in for the rest.
        let old = vec![Record {
            time: 0,
            stream: StreamId::Marker,
            payload: b"x".to_vec(),
        }];
        play(&mut client, old);
    } else {
        play(&mut client, old);
    }
    assert_eq!(
        client.player_container.slot_stack(36),
        None,
        "an older replay keeps the packet-only inventory"
    );
    client.replay_close();
}

// ---------------------------------------------------------------------------------------------
// The website's mod-menu picture: the menu as it opens in game, over a real recording
// ---------------------------------------------------------------------------------------------

/// The Aero mod menu as Right Shift first opens it in game (My setup, its first mod open) over
/// a real recording in the Player camera, drawn by the real stack: the world, the vanilla and
/// mod HUD (both stay up under the menu, as in game), then the menu's glass frosting that frame.
/// A SCREEN record puts the recorded menu up, as a recording made with the menu open does.
/// Five mods are switched on (FPS, Keystrokes, Crosshair, Toggle Sprint, Zoom).
///
/// Writes `menu-<w>x<h>.png` and `clean-<w>x<h>.png` (the same frame without the menu) at
/// 1920x1080 with GUI scale 3 and at 2560x1440 with GUI scale 4 (the same proportions), FOV 90.
/// `MC_SITE_REPLAY=<file> MC_SITE_PNG_DIR=<dir>`, optionally `MC_SITE_SECONDS` (default 5) and
/// `MC_SITE_FPS` (the number the FPS readout shows; 0 without it).
#[test]
#[ignore = "writes PNGs; needs MC_SITE_REPLAY, MC_SITE_PNG_DIR, a GPU and MC_JARS_DIR"]
fn site_menu_over_replay() {
    let (Some(path), Some(out)) = (
        std::env::var_os("MC_SITE_REPLAY"),
        std::env::var_os("MC_SITE_PNG_DIR"),
    ) else {
        eprintln!("skip: set MC_SITE_REPLAY and MC_SITE_PNG_DIR");
        return;
    };
    let seconds: f64 = std::env::var("MC_SITE_SECONDS")
        .ok()
        .and_then(|text| text.parse().ok())
        .unwrap_or(5.0);
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("png dir");
    let reader = mc_replay::Reader::open(&path).expect("the replay opens");
    let meta = reader.meta().clone();
    let records: Vec<Record> = reader.records().map(|r| r.expect("record")).collect();
    let time = (seconds * 1e6) as u64;
    let mut with_menu = records.clone();
    let menu = super::gui::ScreenRecord {
        state: Some(ScreenState::ModMenu),
        ..Default::default()
    };
    insert_record(
        &mut with_menu,
        time.saturating_sub(500_000),
        KIND_SCREEN,
        screen_payload(&menu),
    );
    for (size, gui_scale) in [((1920, 1080), 3), ((2560, 1440), 4)] {
        for (name, records) in [("menu", &with_menu), ("clean", &records)] {
            let file = out.join(format!("{name}-{}x{}.png", size.0, size.1));
            site_shot(records.clone(), meta.clone(), time, size, gui_scale, &file);
        }
    }
}

/// One frame of [`site_menu_over_replay`], in a fresh client with its game dir in a temp folder.
fn site_shot(
    records: Vec<Record>,
    meta: ReplayMeta,
    time: u64,
    size: (u32, u32),
    gui_scale: i32,
    file: &std::path::Path,
) {
    use mc_render::{Gpu, GpuOptions, PhaseList};
    let mut gpu = match Gpu::headless(
        size.0,
        size.1,
        GpuOptions {
            vsync: false,
            sample_count: 1,
            low_power: false,
        },
    ) {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skip: no headless GPU ({error})");
            return;
        }
    };
    let dir = tempfile::tempdir().expect("tempdir");
    // The player's own view: FOV 90 (stored as (90 - 70) / 40), the GUI scale, 8 chunks, fancy
    // graphics and full smooth lighting. Chat hidden: a short recording that starts at the join
    // keeps the join's chat lines up throughout (they stay for 10 s), and the picture shows the
    // game as it looks once they have gone.
    std::fs::write(
        dir.path().join("options.txt"),
        format!(
            "fov:0.5\nguiScale:{gui_scale}\nrenderDistance:8\nfancyGraphics:true\nao:2\n\
             chatVisibility:2\n"
        ),
    )
    .expect("options.txt");
    let config = ClientConfig {
        game_dir: dir.path().to_path_buf(),
        login: LoginMode::Offline("Dev".to_owned()),
        ..ClientConfig::default()
    }
    .with_discovered_assets();
    if !config.version_jar.as_ref().is_some_and(|p| p.is_file()) {
        eprintln!("skip: no vanilla jar in this checkout (set MC_JARS_DIR)");
        return;
    }
    let mut client = Client::new(config, &mut |_, _| {}).expect("the real assets must load");
    client.set_skin_downloads(false);
    for id in ["fps", "keystrokes", "crosshair", "toggle_sprint", "zoom"] {
        assert!(
            client.mods.settings.catalog().iter().any(|d| d.id == id),
            "{id}"
        );
        client.mods.settings.set_enabled(id, true);
    }
    let mut phases = PhaseList::new();
    client.install(&gpu, &mut phases, size.0, size.1);
    let mut camera = crate::app::driver::menu_camera();
    camera.set_viewport(size.0, size.1);
    client
        .replay_open_records(meta, records)
        .expect("the replay opens");
    client.set_replay_deterministic_mesh(true);
    client.replay_set_camera_mode(ReplayCameraMode::Pov);
    let partial = client.replay_render_at(time);
    for warm in 0..900 {
        client.frame(&gpu, &mut camera, partial);
        if warm >= 6 && client.replay_terrain_settled() {
            break;
        }
    }
    // These frames bypass the shell's frame counter, so the FPS readout would read 0;
    // `MC_SITE_FPS` sets the number it shows.
    if let Some(fps) = std::env::var("MC_SITE_FPS")
        .ok()
        .and_then(|text| text.parse().ok())
    {
        client.debug_fps.fps = fps;
    }
    // The menu's switches, scroll and fades run on wall time: let them settle.
    for _ in 0..40 {
        client.frame(&gpu, &mut camera, partial);
    }
    let target = gpu.acquire().expect("offscreen frame");
    phases.render(&gpu, target.view(), &camera, partial);
    let pixels = gpu.read_pixels().expect("readback");
    image::save_buffer(
        file,
        &pixels,
        size.0,
        size.1,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write the PNG");
    eprintln!("wrote {}", file.display());
    client.replay_close();
}
