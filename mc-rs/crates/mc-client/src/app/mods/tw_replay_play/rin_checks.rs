//! The inventory a replay shows against the one the live game showed,
//! with the `centered_inventory` mod on and off. Test-only.
//!
//! * [`recorded_inventory_draws_like_the_live_one`]: the live client opens its inventory with an
//!   effect active and records that frame; the replay of it, in the Player camera, must draw
//!   exactly what the live inventory drew (headless: the two draw lists are compared).
//! * [`scan_replay`] (ignored): a real replay's recorded screens and the player's own effects,
//!   by time, to find the moments worth a picture.
//! * [`pictures`] (ignored): a real replay drawn on the GPU at such moments, next to the live
//!   inventory built from the same replayed state.

use mc_replay::{Record, ReplayMeta, StreamId};
use mc_ui::screen_stack::{Screen as _, ScreenCtx};
use mc_ui::screens::hud::ActiveEffect;
use mc_ui::widget::DrawList;

use super::super::super::{Client, Ui};
use super::gui::{EffectLayout, ScreenRecord, ScreenState, KIND_INVENTORY, KIND_SCREEN};
use super::{decode_packet, ReplayCameraMode};
use crate::app::config::{ClientConfig, LoginMode};

/// An offline client with its game dir in a temp folder, skin and cape downloads off (`None`:
/// no vanilla jar here).
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
    client.set_skin_downloads(false);
    Some((client, dir))
}

fn effect(name: &str, icon_index: i32) -> ActiveEffect {
    ActiveEffect {
        icon_index,
        name: name.to_owned(),
        duration: "1:00".to_owned(),
        beneficial: true,
    }
}

/// Init and draw the open container screen (`client.ui`) the way `build_ui` does, with the
/// pointer at `mouse`: its `gui_left` and everything it drew.
fn draw_container(client: &mut Client, ctx: &ScreenCtx, mouse: (i32, i32)) -> (i32, DrawList) {
    let Ui::Container(screen) = &mut client.ui else {
        panic!("a container screen is up, not {}", client.ui.name());
    };
    screen.init(ctx);
    let mut list = DrawList::new();
    screen.draw(&mut list, &mut client.font, ctx, mouse);
    (screen.gui_left(), list)
}

/// The mod settings one case runs with.
fn set_layout(client: &mut Client, centered: bool, show_effects: bool) {
    client
        .mods
        .settings
        .set_enabled("centered_inventory", centered);
    client
        .mods
        .settings
        .set_bool("centered_inventory", "show_effects", show_effects);
}

/// The live inventory with `effects`, as the live client draws it: opened, laid out by the
/// frame's `m2_frame`, then drawn by `build_ui`. Records that frame and returns the records,
/// the screen context and pointer it was drawn at, and what it drew.
fn live_inventory_frame(
    client: &mut Client,
    effects: &[ActiveEffect],
    size: (u32, u32),
) -> (Vec<Record>, ScreenCtx, (i32, i32), i32, DrawList) {
    client.player_hud.effects = effects.to_vec();
    client.ui = Ui::Container(Box::new(client.new_player_inventory_screen()));
    // `Client::frame`: the mod hooks (`m2_frame`), then `build_ui` (which records first).
    client.m2_frame();
    let ctx = ScreenCtx::new(client.screen_resolution(size.0, size.1), 0.0);
    // The pointer over the first hotbar slot of a centred inventory: the slot it hovers
    // depends on where the inventory is.
    client.cursor = (ctx.width / 2 - 88 + 8 + 8, ctx.height / 2 - 83 + 142 + 8);
    // Every in-memory recording is generation 1: start this one's screen records afresh, as a
    // file recording's new generation does.
    client.tw_play.capture.gui = Default::default();
    client.replay_debug_start_capture();
    let swap = client.nw_rb_gui_begin(size.0, size.1);
    assert!(swap.is_none(), "recording swaps nothing in");
    let mouse = client.cursor;
    let (left, list) = draw_container(client, &ctx, mouse);
    let records = client.replay_debug_finish_capture();
    client.ui = Ui::InGame;
    (records, ctx, mouse, left, list)
}

/// The replayed frame of `records` in the Player camera, with the viewer's game in the same
/// state (`effects`): what the recorded inventory draws.
fn replayed_inventory_frame(
    client: &mut Client,
    records: Vec<Record>,
    effects: &[ActiveEffect],
    size: (u32, u32),
    ctx: &ScreenCtx,
) -> ((i32, i32), i32, DrawList) {
    let meta = ReplayMeta {
        duration: 1,
        ..ReplayMeta::default()
    };
    client.replay_open_records(meta, records).expect("opens");
    client.replay_set_camera_mode(ReplayCameraMode::Pov);
    client.replay_render_at(1);
    client.player_hud.effects = effects.to_vec();
    client.ui = Ui::InGame;
    let swap = client.nw_rb_gui_begin(size.0, size.1);
    assert!(
        swap.is_some(),
        "the recorded inventory is swapped in for the drawing"
    );
    let mouse = client.cursor;
    let (left, list) = draw_container(client, ctx, mouse);
    client.nw_rb_gui_end(swap, &mut DrawList::new(), ctx);
    assert!(
        matches!(client.ui, Ui::InGame),
        "the viewer's own screen is back"
    );
    client.replay_close();
    (mouse, left, list)
}

/// The bug report: "centered inventory not showing accurately in replays". The live client
/// keeps its inventory centred while an effect is active (the `centered_inventory` mod); the
/// replay of that moment must draw it exactly where and how the live one was drawn: the same
/// `gui_left`, the effect panels beside it or hidden, the recorded pointer over the same slot.
/// With no effect active every layout is vanilla's centring (the control case).
#[test]
fn recorded_inventory_draws_like_the_live_one() {
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let size = (1280, 720);
    let one = [effect("Speed", 0)];
    let cases: [(bool, bool, &[ActiveEffect]); 5] = [
        (false, true, &[]),
        (true, true, &[]),
        (false, true, &one),
        (true, true, &one),
        (true, false, &one),
    ];
    for (centered, show_effects, effects) in cases {
        set_layout(&mut client, centered, show_effects);
        let (records, ctx, live_mouse, live_left, live_list) =
            live_inventory_frame(&mut client, effects, size);
        let (replay_mouse, replay_left, replay_list) =
            replayed_inventory_frame(&mut client, records, effects, size, &ctx);
        let case = format!(
            "centered_inventory {centered}, show_effects {show_effects}, {} effect(s)",
            effects.len()
        );
        eprintln!(
            "{case}: live gui_left {live_left}, replay gui_left {replay_left}; pointer live \
             {live_mouse:?}, replay {replay_mouse:?}"
        );
        assert_eq!(replay_mouse, live_mouse, "{case}: the recorded pointer");
        assert_eq!(replay_left, live_left, "{case}: where the inventory is");
        assert!(
            replay_list == live_list,
            "{case}: the replayed inventory draws what the live one drew ({} vs {} vertices)",
            replay_list.vertices().len(),
            live_list.vertices().len()
        );
    }
}

/// The SCREEN records of `records` with their layout taken out: a replay recorded before they
/// carried it.
fn without_layout(records: Vec<Record>) -> Vec<Record> {
    records
        .into_iter()
        .map(|mut record| {
            if record.stream == StreamId::Client(KIND_SCREEN) {
                let mut screen = ScreenRecord::decode(&record.payload).expect("decodes");
                screen.layout = None;
                record.payload.clear();
                screen.encode_into(&mut record.payload);
            }
            record
        })
        .collect()
}

/// A replay shows the inventory as the recording client laid it out, whatever the viewer's own
/// setting is now (a friend's replay, or the setting changed since): the SCREEN records carry the
/// layout. A replay recorded before they did follows the viewer's setting.
#[test]
fn the_recorded_layout_wins_and_older_replays_follow_the_viewer() {
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let size = (1280, 720);
    let one = [effect("Strength", 4)];
    for (recorded, viewer) in [(true, false), (false, true), (true, true), (false, false)] {
        set_layout(&mut client, recorded, true);
        let (records, ctx, _, live_left, live_list) = live_inventory_frame(&mut client, &one, size);
        // The other layout, drawn live, for the older replay's expectation.
        set_layout(&mut client, viewer, true);
        let (_, _, _, viewer_left, viewer_list) = live_inventory_frame(&mut client, &one, size);
        let (_, left, list) =
            replayed_inventory_frame(&mut client, records.clone(), &one, size, &ctx);
        let case = format!("recorded centred {recorded}, viewer centred {viewer}");
        assert_eq!(left, live_left, "{case}: as recorded");
        assert!(list == live_list, "{case}: draws as recorded");
        let (_, left, list) =
            replayed_inventory_frame(&mut client, without_layout(records), &one, size, &ctx);
        assert_eq!(
            left, viewer_left,
            "{case}: an older replay follows the viewer"
        );
        assert!(
            list == viewer_list,
            "{case}: an older replay draws as the viewer's game would"
        );
    }
}

/// The layout is written with a recording's first SCREEN record, then only when it changes; an
/// unchanged frame writes nothing.
#[test]
fn the_layout_is_recorded_first_and_on_every_change() {
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let screens = |client: &Client| -> Vec<ScreenRecord> {
        client
            .replay_debug_captured()
            .iter()
            .filter(|record| record.stream == StreamId::Client(KIND_SCREEN))
            .map(|record| ScreenRecord::decode(&record.payload).expect("decodes"))
            .collect()
    };
    set_layout(&mut client, true, false);
    client.ui = Ui::InGame;
    client.tw_play.capture.gui = Default::default();
    client.replay_debug_start_capture();
    client.nw_rb_gui_begin(1280, 720);
    let first = screens(&client);
    assert_eq!(first.len(), 1);
    assert_eq!(
        first[0].layout,
        Some(EffectLayout {
            centered: true,
            show_effects: false
        })
    );
    client.nw_rb_gui_begin(1280, 720);
    assert_eq!(screens(&client).len(), 1, "unchanged: nothing written");
    set_layout(&mut client, false, false);
    client.nw_rb_gui_begin(1280, 720);
    let after = screens(&client);
    assert_eq!(after.len(), 2);
    let vanilla = EffectLayout {
        centered: false,
        show_effects: true,
    };
    assert_eq!(after[1].layout, Some(vanilla), "off is vanilla's layout");
    assert_eq!(after[1].state, None, "only the layout changed");
    client.replay_debug_finish_capture();
}

/// With the independent GUI scale mod's inventory scale (GUI scale 3, inventories at 1), the
/// recorded pointer is measured in the inventory's scale, and the replay puts it back in that
/// scale, on the same slot: not in the global scale the HUD is drawn at.
#[test]
fn the_pointer_comes_back_in_the_inventorys_own_gui_scale() {
    let Some((mut client, _dir)) = offline_client() else {
        return;
    };
    let size = (1280, 720);
    client.settings.gui_scale = 3;
    client.mods.settings.set_enabled("independent_gui", true);
    client
        .mods
        .settings
        .set_choice("independent_gui", "inventory_scale", 2);
    set_layout(&mut client, true, true);
    let one = [effect("Speed", 0)];
    let (records, ctx, live_mouse, live_left, live_list) =
        live_inventory_frame(&mut client, &one, size);
    assert_eq!(
        ctx.resolution.scale_factor(),
        1,
        "inventories at GUI scale 1"
    );
    let (replay_mouse, replay_left, replay_list) =
        replayed_inventory_frame(&mut client, records, &one, size, &ctx);
    assert_eq!(
        replay_mouse, live_mouse,
        "the pointer in the inventory's scale"
    );
    assert_eq!(replay_left, live_left);
    assert!(
        replay_list == live_list,
        "the same slot is hovered and drawn"
    );
}

// ---------------------------------------------------------------------------------------------
// Real replays: where the inventory was open, and with which effects
// ---------------------------------------------------------------------------------------------

/// The player's own effects as the replay's packets leave them: `(id, amplifier)`.
#[derive(Debug, Default)]
struct OwnEffects {
    entity_id: Option<i32>,
    active: Vec<(i8, i8)>,
}

impl OwnEffects {
    fn packet(&mut self, packet: &mc_protocol::play::S2CPlay) -> bool {
        use mc_protocol::play::S2CPlay;
        match packet {
            S2CPlay::JoinGame(join) => {
                self.entity_id = Some(join.entity_id);
                self.active.clear();
                true
            }
            S2CPlay::Respawn(_) => {
                self.active.clear();
                true
            }
            S2CPlay::EntityEffect(add) if Some(add.entity_id) == self.entity_id => {
                self.active.retain(|(id, _)| *id != add.effect_id);
                self.active.push((add.effect_id, add.amplifier));
                true
            }
            S2CPlay::RemoveEntityEffect(remove) if Some(remove.entity_id) == self.entity_id => {
                self.active.retain(|(id, _)| *id as u8 != remove.effect_id);
                true
            }
            _ => false,
        }
    }
}

/// A real replay's recorded screens (state changes) and the recording player's own effects, by
/// time; and the spans with the inventory open while an effect was active.
/// `MC_RIN_REPLAY=<file>`.
#[test]
#[ignore = "needs a real replay (MC_RIN_REPLAY)"]
fn scan_replay() {
    let Some(path) = std::env::var_os("MC_RIN_REPLAY") else {
        eprintln!("skip: set MC_RIN_REPLAY");
        return;
    };
    let reader = mc_replay::Reader::open(&path).expect("the replay opens");
    let duration = reader.meta().duration;
    let mut effects = OwnEffects::default();
    let mut screen = ScreenState::None;
    let mut opened_at = 0u64;
    let mut spans = Vec::new();
    let mut pointer_updates = 0usize;
    let mut inventory_records = 0usize;
    for record in reader.records() {
        let record = record.expect("record");
        let seconds = record.time as f64 / 1e6;
        match record.stream {
            StreamId::Packet => {
                let Ok(packet) = decode_packet(&record.payload) else {
                    continue;
                };
                if let mc_protocol::play::S2CPlay::OpenWindow(open) = &packet {
                    eprintln!(
                        "{seconds:9.3} s  OpenWindow {} ({})",
                        open.window_id, open.inventory_type
                    );
                }
                if effects.packet(&packet) {
                    eprintln!("{seconds:9.3} s  own effects {:?}", effects.active);
                }
            }
            StreamId::Client(KIND_INVENTORY) => inventory_records += 1,
            StreamId::Client(KIND_SCREEN) => {
                let Some(decoded) = ScreenRecord::decode(&record.payload) else {
                    eprintln!("{seconds:9.3} s  SCREEN record that does not decode");
                    continue;
                };
                if decoded.pointer.is_some() {
                    pointer_updates += 1;
                }
                if let Some(state) = decoded.state {
                    if screen == ScreenState::Inventory && state != ScreenState::Inventory {
                        spans.push((opened_at, record.time));
                    }
                    if state != screen {
                        if state == ScreenState::Inventory {
                            opened_at = record.time;
                        }
                        eprintln!(
                            "{seconds:9.3} s  screen {state:?} (pointer {:?}), own effects {:?}",
                            decoded.pointer, effects.active
                        );
                    }
                    screen = state;
                }
            }
            _ => {}
        }
    }
    eprintln!(
        "duration {:.1} s; {inventory_records} INVENTORY records, {pointer_updates} pointer \
         updates; inventory open: {}",
        duration as f64 / 1e6,
        spans
            .iter()
            .map(|(from, to)| format!("{:.2}-{:.2} s", *from as f64 / 1e6, *to as f64 / 1e6))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

// ---------------------------------------------------------------------------------------------
// Pictures: a real replay against the live inventory for the same replayed state
// ---------------------------------------------------------------------------------------------

/// Save `pixels` (RGBA8) as a PNG.
fn save_png(path: &std::path::Path, pixels: &[u8], size: (u32, u32)) {
    image::save_buffer(
        path,
        pixels,
        size.0,
        size.1,
        image::ExtendedColorType::Rgba8,
    )
    .expect("write the PNG");
    eprintln!("wrote {}", path.display());
}

/// A `side` x `side` crop of an RGBA frame (`frame_width` wide) from `(x, y)`, magnified
/// `zoom` times (nearest), outside pixels black.
fn crop_zoom(
    frame: &[u8],
    frame_width: usize,
    (x, y): (i64, i64),
    side: usize,
    zoom: usize,
) -> Vec<u8> {
    let frame_height = frame.len() / 4 / frame_width;
    let out_side = side * zoom;
    let mut out = vec![0u8; out_side * out_side * 4];
    for oy in 0..out_side {
        for ox in 0..out_side {
            let (sx, sy) = (x + (ox / zoom) as i64, y + (oy / zoom) as i64);
            if sx < 0 || sy < 0 || sx as usize >= frame_width || sy as usize >= frame_height {
                continue;
            }
            let from = (sy as usize * frame_width + sx as usize) * 4;
            let to = (oy * out_side + ox) * 4;
            out[to..to + 4].copy_from_slice(&frame[from..from + 4]);
        }
    }
    out
}

/// The recorded pointer before and after (the desktop's standard arrow, drawn at the target's
/// own pixels) over a real replay's recorded inventory (`MC_RIN_AT`, the first time given), at
/// GUI scales 1 to 4 (1920x1080, display scaling 100%) and at GUI scale 3 with 150%: the full
/// frames, and a sheet of crops around the tip magnified 4x (rows: GUI scale 1, 2, 3, 4, then 3
/// at 150%; columns: before, after).
/// `MC_RIN_REPLAY=<file> MC_RIN_AT=<seconds> MC_RIN_PNG_DIR=<dir>`.
#[test]
#[ignore = "writes PNGs; needs MC_RIN_REPLAY, MC_RIN_AT, MC_RIN_PNG_DIR, a GPU and MC_JARS_DIR"]
fn pointer_pictures() {
    use mc_render::{Gpu, GpuOptions, PhaseList, ShellApp};
    let (Some(path), Some(at), Some(out)) = (
        std::env::var_os("MC_RIN_REPLAY"),
        std::env::var("MC_RIN_AT").ok(),
        std::env::var_os("MC_RIN_PNG_DIR"),
    ) else {
        eprintln!("skip: set MC_RIN_REPLAY, MC_RIN_AT and MC_RIN_PNG_DIR");
        return;
    };
    let Some(seconds) = at
        .split(',')
        .find_map(|text| text.trim().parse::<f64>().ok())
    else {
        eprintln!("skip: MC_RIN_AT has no time");
        return;
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("png dir");
    let size = (1920u32, 1080u32);
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
        .replay_open(std::path::Path::new(&path))
        .expect("the replay opens");
    client.set_replay_deterministic_mesh(true);
    client.replay_set_camera_mode(ReplayCameraMode::Pov);
    set_layout(&mut client, true, true);
    let time = (seconds * 1e6) as u64;
    let (side, zoom) = (64usize, 4usize);
    let cell = side * zoom;
    let rows = [(1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (3, 1.5)];
    let mut sheet = vec![0u8; (2 * cell) * (rows.len() * cell) * 4];
    for (row, (gui_scale, display)) in rows.into_iter().enumerate() {
        client.settings.gui_scale = gui_scale;
        <Client as ShellApp>::window_scale_changed(&mut client, display);
        let partial = client.replay_render_at(time);
        for warm in 0..900 {
            client.frame(&gpu, &mut camera, partial);
            if warm >= 6 && client.replay_terrain_settled() {
                break;
            }
        }
        let Some(playback) = client.tw_play.playback.as_deref() else {
            panic!("the replay is open");
        };
        assert_eq!(
            playback.gui.screen,
            ScreenState::Inventory,
            "the inventory is up"
        );
        let (dx, dy) = playback.gui.pointer;
        let ctx = ScreenCtx::new(client.screen_resolution(size.0, size.1), 0.0);
        let scale = i64::from(ctx.resolution.scale_factor());
        let tip = (
            i64::from(ctx.width / 2 + i32::from(dx)) * scale,
            i64::from(ctx.height / 2 + i32::from(dy)) * scale,
        );
        for (column, legacy) in [true, false].into_iter().enumerate() {
            client.tw_play.pointer.legacy = legacy;
            for _ in 0..3 {
                client.frame(&gpu, &mut camera, partial);
            }
            let target = gpu.acquire().expect("offscreen frame");
            phases.render(&gpu, target.view(), &camera, partial);
            let pixels = gpu.read_pixels().expect("readback");
            let name = format!(
                "pointer_gui{gui_scale}_display{:.0}_{}.png",
                display * 100.0,
                if legacy { "before" } else { "after" }
            );
            save_png(&out.join(name), &pixels, size);
            let crop = crop_zoom(&pixels, size.0 as usize, (tip.0 - 8, tip.1 - 8), side, zoom);
            for y in 0..cell {
                let from = y * cell * 4;
                let to = ((row * cell + y) * 2 * cell + column * cell) * 4;
                sheet[to..to + cell * 4].copy_from_slice(&crop[from..from + cell * 4]);
            }
        }
    }
    client.tw_play.pointer.legacy = false;
    assert!(!client.skin_worker_running(), "skin downloads stay off");
    client.replay_close();
    let sheet_size = ((2 * cell) as u32, (rows.len() * cell) as u32);
    save_png(
        &out.join("pointer_sheet_before_after.png"),
        &sheet,
        sheet_size,
    );
}

/// The pixels that differ between two frames (any channel).
fn differing_pixels(a: &[u8], b: &[u8]) -> usize {
    a.chunks_exact(4)
        .zip(b.chunks_exact(4))
        .filter(|(x, y)| x != y)
        .count()
}

/// A real replay at moments with the inventory open (`MC_RIN_AT`, seconds, comma-separated),
/// drawn on the GPU in the Player camera with the viewer's `centered_inventory` on and off. For
/// each, the live inventory built from the same replayed state (the live client's own
/// `new_player_inventory_screen`, `m2_frame` and draw, aged as the recorded one is, with the
/// pointer where the recording had it) is drawn too, and the two frames are compared pixel for
/// pixel. The recorded pointer is hidden in the compared frames (a live frame never contains it)
/// and shown in one more picture per moment.
///
/// `MC_RIN_RECORDED=vanilla|centered` adds one SCREEN record at the start carrying that layout,
/// as every replay recorded from this wave on carries the recording client's: the live inventory
/// is then drawn with the recorded layout, whatever the viewer's setting, and the replay must
/// match it.
/// `MC_RIN_REPLAY=<file> MC_RIN_AT=<s,s,..> MC_RIN_PNG_DIR=<dir> [MC_RIN_TAG=<prefix>]`.
#[test]
#[ignore = "writes PNGs; needs MC_RIN_REPLAY, MC_RIN_AT, MC_RIN_PNG_DIR, a GPU and MC_JARS_DIR"]
fn pictures() {
    use mc_render::{Gpu, GpuOptions, PhaseList};
    let (Some(path), Some(at), Some(out)) = (
        std::env::var_os("MC_RIN_REPLAY"),
        std::env::var("MC_RIN_AT").ok(),
        std::env::var_os("MC_RIN_PNG_DIR"),
    ) else {
        eprintln!("skip: set MC_RIN_REPLAY, MC_RIN_AT and MC_RIN_PNG_DIR");
        return;
    };
    let tag = std::env::var("MC_RIN_TAG").unwrap_or_else(|_| "rin".to_owned());
    let recorded_centered = match std::env::var("MC_RIN_RECORDED").ok().as_deref() {
        Some("vanilla") => Some(false),
        Some("centered") => Some(true),
        _ => None,
    };
    let out = std::path::PathBuf::from(out);
    std::fs::create_dir_all(&out).expect("png dir");
    let size = (1280, 720);
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
    if let Some(centered) = recorded_centered {
        let reader = mc_replay::Reader::open(&path).expect("the replay opens");
        let meta = reader.meta().clone();
        let mut records: Vec<Record> = reader.records().map(|r| r.expect("record")).collect();
        let mut payload = Vec::new();
        ScreenRecord {
            layout: Some(EffectLayout {
                centered,
                show_effects: true,
            }),
            ..ScreenRecord::default()
        }
        .encode_into(&mut payload);
        records.insert(
            0,
            Record {
                time: 0,
                stream: StreamId::Client(KIND_SCREEN),
                payload,
            },
        );
        client
            .replay_open_records(meta, records)
            .expect("the replay opens");
    } else {
        client
            .replay_open(std::path::Path::new(&path))
            .expect("the replay opens");
    }
    client.set_replay_deterministic_mesh(true);
    client.replay_set_camera_mode(ReplayCameraMode::Pov);
    let mut summary = Vec::new();
    for seconds in at
        .split(',')
        .filter_map(|text| text.trim().parse::<f64>().ok())
    {
        let time = (seconds * 1e6) as u64;
        for centered in [true, false] {
            set_layout(&mut client, centered, true);
            let mode = if centered { "on" } else { "off" };
            let name = format!("{tag}_{seconds:.2}s_centered_{mode}");
            let partial = client.replay_render_at(time);
            for warm in 0..900 {
                client.frame(&gpu, &mut camera, partial);
                if warm >= 6 && client.replay_terrain_settled() {
                    break;
                }
            }
            let mut shot = |client: &mut Client, gpu: &mut Gpu, phases: &mut PhaseList| {
                for _ in 0..3 {
                    client.frame(gpu, &mut camera, partial);
                }
                let target = gpu.acquire().expect("offscreen frame");
                phases.render(gpu, target.view(), &camera, partial);
                gpu.read_pixels().expect("readback")
            };
            // What the viewer sees: the recorded inventory and pointer.
            client.replay_set_pointer_visible(true);
            let viewer = shot(&mut client, &mut gpu, &mut phases);
            save_png(&out.join(format!("{name}_replay.png")), &viewer, size);
            // The comparable replayed frame: without the pointer.
            client.replay_set_pointer_visible(false);
            let replayed = shot(&mut client, &mut gpu, &mut phases);
            // The live inventory over the same replayed state, as old as the recorded one, the
            // pointer where the recording had it.
            let Some(playback) = client.tw_play.playback.as_deref() else {
                panic!("the replay is open");
            };
            let recorded = playback.gui.screen.clone();
            let opened_tick = playback.gui.opened_tick;
            let (dx, dy) = playback.gui.pointer;
            if recorded != ScreenState::Inventory {
                eprintln!("{name}: the recorded screen is {recorded:?}, not the inventory");
                summary.push(format!("{name}: no inventory ({recorded:?})"));
                continue;
            }
            let mut live = client.new_player_inventory_screen();
            let steps = client.tick.saturating_sub(opened_tick).min(1_200);
            for _ in 0..steps {
                live.set_active_effects(client.player_hud.effects.clone());
                mc_ui::screen_stack::Screen::tick(&mut live);
            }
            // The live game as the recording client had it set.
            set_layout(&mut client, recorded_centered.unwrap_or(centered), true);
            client.ui = Ui::Container(Box::new(live));
            let ctx = ScreenCtx::new(client.screen_resolution(size.0, size.1), 0.0);
            let cursor = client.cursor;
            client.cursor = (
                ctx.width / 2 + i32::from(dx),
                ctx.height / 2 + i32::from(dy),
            );
            let live_pixels = shot(&mut client, &mut gpu, &mut phases);
            client.cursor = cursor;
            client.ui = Ui::InGame;
            save_png(&out.join(format!("{name}_live.png")), &live_pixels, size);
            let differ = differing_pixels(&replayed, &live_pixels);
            let line = format!(
                "{name}: {} effect(s); {differ} of {} pixels differ between the replayed and the \
                 live inventory",
                client.player_hud.effects.len(),
                size.0 * size.1
            );
            eprintln!("{line}");
            summary.push(line);
        }
    }
    assert!(!client.skin_worker_running(), "skin downloads stay off");
    client.replay_close();
    let report = summary.join("\n") + "\n";
    std::fs::write(out.join(format!("{tag}_summary.txt")), report).expect("summary");
}
