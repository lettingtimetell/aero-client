//! Headless tests of the replay screens (the browser, the
//! viewer's controls, the export window): layout, hit testing, clicks, keys and the events they
//! send, with a deterministic 5-pixel test font and no GPU.
//!
//! An integration test, so it builds against the mc-ui library alone (the crate's own unit-test
//! target also compiles every other module's tests). Controls are found the way a player finds
//! them: by their tooltip.

use std::time::{Duration, Instant};

use chrono::FixedOffset;

use mc_ui::font::{FontData, FontRenderer};
use mc_ui::screens::modmenu::glass::GlassFrame;
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::screens::replay::browser::{Tab, ARM_MS, TOOLBAR_H};
use mc_ui::screens::replay::export::ExportState;
use mc_ui::screens::replay::{
    key, BrowserEvent, CameraMode, ExportDialog, ExportProgress, ExportSettings, LibraryView,
    PlaybackStatus, ReplayBrowserScreen, ReplayEntry, ReplayEnv, ReplayInfo, ReplayViewerOverlay,
    ViewerEvent,
};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::DrawList;

const SCREEN: (f32, f32) = (1920.0, 1080.0);
const NONE: Modifiers = Modifiers::NONE;

/// The Minecraft font with every glyph 5 pixels wide (the crate's own test font, rebuilt from
/// public parts).
fn uniform_font() -> FontRenderer {
    let mut img = image::RgbaImage::new(128, 128);
    for cell in 0..256u32 {
        let cx = (cell % 16) * 8;
        let cy = (cell / 16) * 8;
        for col in 0..4 {
            img.put_pixel(cx + col, cy + 2, image::Rgba([255, 255, 255, 255]));
        }
    }
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img)
        .write_to(&mut png, image::ImageFormat::Png)
        .expect("encode");
    let data = FontData::from_parts(&png.into_inner(), &[0u8; 0x1_0000]).expect("font data");
    FontRenderer::new(data, false, false)
}

struct Env {
    text: MenuText,
    mc: FontRenderer,
    t0: Instant,
}

impl Env {
    fn new() -> Env {
        Env {
            text: MenuText::minecraft_only(),
            mc: uniform_font(),
            t0: Instant::now(),
        }
    }

    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn env(&mut self, ms: u64) -> ReplayEnv<'_> {
        ReplayEnv {
            text: &mut self.text,
            mc_font: &mut self.mc,
            screen: SCREEN,
            now: self.t0 + Duration::from_millis(ms),
            time_s: ms as f32 / 1000.0,
        }
    }
}

fn entry(id: u64, title: &str, clip: bool) -> ReplayEntry {
    ReplayEntry {
        id,
        title: title.to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms: 1_794_537_660_000 - id * 3_600_000,
        duration_us: 754_000_000,
        size_bytes: 8_600_000,
        clip,
        has_mic: id == 2,
        problem: None,
        bookmarks: 0,
        folder: String::new(),
        player: String::new(),
    }
}

fn library() -> LibraryView {
    LibraryView {
        recordings: vec![
            entry(1, "SkyWars on mc.example.net", false),
            entry(2, "BedWars on mc.example.net", false),
            entry(3, "Duels on mc.example.net", false),
        ],
        clips: vec![entry(10, "Clip 9:41 PM", true)],
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: true,
        ..LibraryView::default()
    }
}

fn browser(env: &mut Env) -> ReplayBrowserScreen {
    let mut screen = ReplayBrowserScreen::new(env.t0);
    let est = FixedOffset::west_opt(5 * 3600).unwrap();
    screen.set_library_in_zone(library(), 1_794_537_660_000, &est);
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(1000), &mut list);
    screen
}

/// The first point (on a 4-pixel grid) whose control's tooltip says `label`.
fn find_tip(hits: &dyn Fn(f32, f32) -> Option<&'static str>, label: &str) -> Option<(f32, f32)> {
    let mut y = 0.0;
    while y < SCREEN.1 {
        let mut x = 0.0;
        while x < SCREEN.0 {
            if hits(x, y) == Some(label) {
                return Some((x, y));
            }
            x += 4.0;
        }
        y += 4.0;
    }
    None
}

fn press_at(screen: &mut ReplayBrowserScreen, (x, y): (f32, f32), now: Instant) {
    screen.pointer_moved(x, y);
    screen.mouse_pressed(x, y, 0);
    screen.mouse_released(x, y, 0, now);
}

#[test]
fn the_browser_lists_recordings_and_watch_sends_the_entry() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    assert_eq!(screen.tab(), Tab::Recordings);
    assert_eq!(screen.selected(), Some(1), "the newest is selected");
    let watch = find_tip(&|x, y| screen.tooltip_at(x, y), "Watch").expect("a Watch button");
    press_at(&mut screen, watch, env.at(1100));
    let events: Vec<_> = screen.drain_events().collect();
    assert_eq!(events, vec![BrowserEvent::Watch(1)]);
    // Enter watches the selection.
    assert!(screen.key(key::ENTER, None, NONE, env.at(1200)));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Watch(1)]
    );
}

#[test]
fn a_double_click_on_a_row_watches_it_and_arrows_move_the_selection() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    screen.key(key::DOWN, None, NONE, env.at(1100));
    assert_eq!(screen.selected(), Some(2));
    screen.key(key::END, None, NONE, env.at(1100));
    assert_eq!(screen.selected(), Some(3));
    screen.key(key::HOME, None, NONE, env.at(1100));
    assert_eq!(screen.selected(), Some(1));
    // The row's title area is a plain row target (its tile is a Watch button).
    let window = ReplayBrowserScreen::window_rect(SCREEN);
    // The second row's middle, under the toolbar and its day's 34 px header.
    let row_y = window.y + (78.0 + TOOLBAR_H + 12.0 + 40.0) + 34.0 + 86.0;
    let point = (window.x + 400.0, row_y);
    press_at(&mut screen, point, env.at(2000));
    assert_eq!(screen.selected(), Some(2));
    assert!(
        screen.drain_events().next().is_none(),
        "one click only selects"
    );
    press_at(&mut screen, point, env.at(2200));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Watch(2)]
    );
    // Too slow is two single clicks.
    press_at(&mut screen, point, env.at(4000));
    press_at(&mut screen, point, env.at(4800));
    assert!(screen.drain_events().next().is_none());
    // One click on a row's tile watches it, as on a video thumbnail.
    let tile = (window.x + 60.0, row_y);
    press_at(&mut screen, tile, env.at(6000));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Watch(2)]
    );
}

#[test]
fn in_game_watch_and_export_need_a_second_click_because_they_leave_the_server() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    screen.set_in_game("mc.example.net");
    screen.key(key::ENTER, None, NONE, env.at(1100));
    assert!(
        screen.drain_events().next().is_none(),
        "the first Enter only warns"
    );
    screen.key(key::ENTER, None, NONE, env.at(1500));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Watch(1)]
    );
    // The arm runs out.
    screen.key(key::ENTER, None, NONE, env.at(2000));
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(2000 + ARM_MS as u64 + 50), &mut list);
    screen.key(key::ENTER, None, NONE, env.at(2000 + ARM_MS as u64 + 100));
    assert!(
        screen.drain_events().next().is_none(),
        "expired: warns again"
    );
}

#[test]
fn delete_needs_a_second_click_within_three_seconds() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    let delete = find_tip(&|x, y| screen.tooltip_at(x, y), "Delete").expect("a Delete button");
    press_at(&mut screen, delete, env.at(1100));
    assert!(
        screen.drain_events().next().is_none(),
        "the first click arms it"
    );
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(1200), &mut list);
    let armed = find_tip(&|x, y| screen.tooltip_at(x, y), "Click again to delete").expect("armed");
    press_at(&mut screen, armed, env.at(1300));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Delete(1)]
    );
    assert_eq!(screen.selected(), Some(2), "the next row is selected");
    // The keyboard: Delete twice, but the arm runs out after three seconds.
    screen.key(key::DELETE, None, NONE, env.at(2000));
    let _ = screen.build(&mut env.env(2000 + ARM_MS as u64 + 100), &mut list);
    screen.key(key::DELETE, None, NONE, env.at(2000 + ARM_MS as u64 + 200));
    assert!(
        screen.drain_events().next().is_none(),
        "the arm expired: this re-arms"
    );
    screen.key(key::DELETE, None, NONE, env.at(2000 + ARM_MS as u64 + 400));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Delete(2)]
    );
}

#[test]
fn renaming_types_a_new_title_and_enter_saves_it() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    screen.key(key::F2, None, NONE, env.at(1100));
    assert!(screen.renaming());
    // The whole title is selected: typing replaces it.
    for c in "  Best game  ".chars() {
        screen.key(0, Some(c), NONE, env.at(1200));
    }
    screen.key(key::BACKSPACE, None, NONE, env.at(1200));
    screen.key(key::ENTER, None, NONE, env.at(1300));
    assert!(!screen.renaming());
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Rename {
            id: 1,
            title: "Best game".to_owned()
        }]
    );
    // Esc cancels; an empty title is never sent.
    screen.key(key::F2, None, NONE, env.at(1400));
    screen.key(key::ESCAPE, None, NONE, env.at(1500));
    screen.key(key::F2, None, NONE, env.at(1600));
    screen.key(key::BACKSPACE, None, NONE, env.at(1600));
    screen.key(key::ENTER, None, NONE, env.at(1700));
    assert!(screen.drain_events().next().is_none());
    // Paste goes into the field while renaming (and only then).
    screen.key(key::F2, None, NONE, env.at(1750));
    assert!(screen.paste("Pasted"));
    screen.key(key::ENTER, None, NONE, env.at(1760));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Rename {
            id: 1,
            title: "Pasted".to_owned()
        }]
    );
    assert!(!screen.paste("ignored"));
    // Esc out of the browser.
    screen.key(key::ESCAPE, None, NONE, env.at(1800));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::Close]
    );
}

#[test]
fn the_clips_tab_has_a_one_click_export() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    let recordings_export = find_tip(
        &|x, y| screen.tooltip_at(x, y),
        "Export with your clip settings",
    );
    assert!(
        recordings_export.is_none(),
        "recordings have no one-click export"
    );
    screen.key(key::TAB, None, NONE, env.at(1100));
    assert_eq!(screen.tab(), Tab::Clips);
    assert_eq!(screen.selected(), Some(10));
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(1200), &mut list);
    let export = find_tip(
        &|x, y| screen.tooltip_at(x, y),
        "Export with your clip settings",
    )
    .expect("clips export in one click");
    press_at(&mut screen, export, env.at(1300));
    assert_eq!(
        screen.drain_events().collect::<Vec<_>>(),
        vec![BrowserEvent::ExportClip(10)]
    );
    // Clicking the other tab goes back.
    let tab = find_tip(&|x, y| screen.tooltip_at(x, y), "Full recordings").expect("tab");
    press_at(&mut screen, tab, env.at(1400));
    assert_eq!(screen.tab(), Tab::Recordings);
}

#[test]
fn an_unreadable_replay_explains_itself_instead_of_opening() {
    let mut env = Env::new();
    let mut screen = ReplayBrowserScreen::new(env.t0);
    let mut lib = library();
    lib.recordings[0].problem = Some("made by a newer Aero Client".to_owned());
    screen.set_library(lib, 1_794_537_660_000);
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(1000), &mut list);
    screen.key(key::ENTER, None, NONE, env.at(1100));
    assert!(screen.drain_events().next().is_none());
}

#[test]
fn an_empty_tab_and_a_loading_library_still_draw() {
    let mut env = Env::new();
    let mut screen = ReplayBrowserScreen::new(env.t0);
    let mut list = DrawList::new();
    let glass = screen.build(&mut env.env(1000), &mut list);
    assert!(!list.is_empty(), "Loading\u{2026}");
    assert_eq!(glass.rects.len(), 1, "one glass window");
    screen.set_library(LibraryView::default(), 0);
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(1100), &mut list);
    assert!(!list.is_empty(), "the empty state");
    assert_eq!(screen.selected(), None);
    // Keys on an empty list do nothing harmful.
    screen.key(key::DOWN, None, NONE, env.at(1200));
    screen.key(key::ENTER, None, NONE, env.at(1200));
    assert!(screen.drain_events().next().is_none());
}

#[test]
fn the_browser_fits_small_and_large_windows() {
    for screen_size in [
        (1280.0, 720.0),
        (854.0, 480.0),
        (3840.0, 2160.0),
        (1024.0, 768.0),
    ] {
        let rect = ReplayBrowserScreen::window_rect(screen_size);
        assert!(rect.x >= 0.0 && rect.y >= 0.0, "{screen_size:?}: {rect:?}");
        assert!(rect.right() <= screen_size.0 && rect.bottom() <= screen_size.1);
    }
}

// ---- the viewer ---------------------------------------------------------------------------

fn info() -> ReplayInfo {
    ReplayInfo {
        title: "SkyWars on mc.example.net".to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms: 0,
        clip: false,
        has_mic: false,
        markers: vec![(30_000_000, "Kill".to_owned())],
        window_size: (1920, 1080),
    }
}

fn status(time_s: f32, playing: bool) -> PlaybackStatus {
    PlaybackStatus {
        duration_us: 120_000_000,
        time_us: (time_s * 1e6) as u64,
        playing,
        ..PlaybackStatus::default()
    }
}

fn viewer(env: &mut Env) -> ReplayViewerOverlay {
    let mut overlay = ReplayViewerOverlay::new(info(), env.t0);
    overlay.set_status(status(10.0, false));
    let mut list = DrawList::new();
    for ms in (0..=600).step_by(16) {
        list.clear();
        let _ = overlay.build(&mut env.env(ms), &mut list);
    }
    overlay
}

fn frame(overlay: &mut ReplayViewerOverlay, env: &mut Env, ms: u64) -> GlassFrame {
    let mut list = DrawList::new();
    overlay.build(&mut env.env(ms), &mut list)
}

fn press_viewer(overlay: &mut ReplayViewerOverlay, (x, y): (f32, f32), now: Instant) {
    overlay.pointer_moved(x, y, now);
    overlay.mouse_pressed(x, y, 0, now);
    overlay.mouse_released(x, y, 0, now);
}

#[test]
fn space_and_the_play_button_toggle_playback() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.key(key::SPACE, Some(' '), NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Play]
    );
    overlay.set_status(status(10.0, true));
    let _ = frame(&mut overlay, &mut env, 716);
    let pause = find_tip(&|x, y| overlay.tooltip_at(x, y), "Pause").expect("a Pause button");
    press_viewer(&mut overlay, pause, env.at(800));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Pause]
    );
    // At the very end, Play starts over.
    overlay.set_status(status(120.0, false));
    overlay.key(key::K, Some('k'), NONE, env.at(900));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(0), ViewerEvent::Play]
    );
}

#[test]
fn arrows_jump_five_seconds_and_stay_inside_the_replay() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.key(key::RIGHT, None, NONE, env.at(700));
    overlay.key(key::LEFT, None, NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(15_000_000), ViewerEvent::Seek(5_000_000)]
    );
    overlay.set_status(status(2.0, false));
    overlay.key(key::LEFT, None, NONE, env.at(700));
    overlay.set_status(status(118.0, false));
    overlay.key(key::RIGHT, None, NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(0), ViewerEvent::Seek(120_000_000)]
    );
    // Number keys jump to tenths (LWJGL 6 is the 5 key: half way).
    overlay.key(6, None, NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(60_000_000)]
    );
    // Step pauses first.
    overlay.set_status(status(50.0, true));
    overlay.key(key::PERIOD, None, NONE, env.at(800));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Pause, ViewerEvent::Step(1)]
    );
}

#[test]
fn in_and_out_make_a_selection_that_previews_on_a_loop() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.set_status(status(20.0, false));
    overlay.key(key::I, None, NONE, env.at(700));
    overlay.set_status(status(32.5, false));
    overlay.key(key::O, None, NONE, env.at(700));
    assert_eq!(overlay.selection(), Some((20_000_000, 32_500_000)));
    overlay.key(key::P, None, NONE, env.at(700));
    assert!(overlay.previewing());
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(20_000_000), ViewerEvent::Play]
    );
    // Playing past Out loops back to In.
    overlay.set_status(status(31.0, true));
    assert!(overlay.drain_events().next().is_none());
    overlay.set_status(status(32.6, true));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Seek(20_000_000)]
    );
    // An In after the Out clears the Out.
    overlay.set_status(status(40.0, false));
    overlay.key(key::I, None, NONE, env.at(800));
    assert_eq!(overlay.selection(), None);
    // Delete clears everything.
    overlay.key(key::DELETE, None, NONE, env.at(900));
    assert!(!overlay.previewing());
}

#[test]
fn dragging_the_timeline_scrubs_then_resumes_playback() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.set_status(status(10.0, true));
    let _ = frame(&mut overlay, &mut env, 700);
    let bar = ReplayViewerOverlay::bar_rect(SCREEN);
    let track_y = bar.y + 34.0 + 3.0;
    let x_at = |fraction: f32| bar.x + 26.0 + (bar.w - 52.0) * fraction;
    let now = env.at(720);
    overlay.pointer_moved(x_at(0.25), track_y, now);
    assert!(overlay.mouse_pressed(x_at(0.25), track_y, 0, now));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Pause]
    );
    let _ = frame(&mut overlay, &mut env, 736);
    overlay.pointer_moved(x_at(0.5), track_y, env.at(740));
    let _ = frame(&mut overlay, &mut env, 752);
    overlay.mouse_released(x_at(0.75), track_y, 0, env.at(760));
    let events: Vec<_> = overlay.drain_events().collect();
    let seeks: Vec<u64> = events
        .iter()
        .filter_map(|e| match e {
            ViewerEvent::Seek(t) => Some(*t / 1_000_000),
            _ => None,
        })
        .collect();
    assert_eq!(
        seeks,
        vec![30, 60, 90],
        "one seek per frame while dragging, then the drop"
    );
    assert_eq!(events.last(), Some(&ViewerEvent::Play));
}

#[test]
fn the_controls_fade_while_playing_idle_and_h_hides_them() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    assert!(overlay.controls_visible());
    overlay.set_status(status(10.0, true));
    for ms in (700..4000).step_by(16) {
        let _ = frame(&mut overlay, &mut env, ms);
    }
    assert!(!overlay.controls_visible(), "idle while playing");
    let glass = frame(&mut overlay, &mut env, 4016);
    assert!(
        !glass.is_visible(),
        "no glass pass while the controls are gone"
    );
    overlay.pointer_moved(10.0, 10.0, env.at(4020));
    for ms in (4032..4400).step_by(16) {
        let _ = frame(&mut overlay, &mut env, ms);
    }
    assert!(overlay.controls_visible(), "the mouse brings them back");
    overlay.key(key::H, None, NONE, env.at(4400));
    for ms in (4416..5200).step_by(16) {
        let _ = frame(&mut overlay, &mut env, ms);
    }
    assert!(!overlay.controls_visible());
    overlay.pointer_moved(20.0, 20.0, env.at(5200));
    for ms in (5216..5600).step_by(16) {
        let _ = frame(&mut overlay, &mut env, ms);
    }
    assert!(!overlay.controls_visible(), "H keeps them hidden");
    // A click on the picture still pauses, hidden or not, as in a video player.
    press_viewer(&mut overlay, (960.0, 300.0), env.at(5590));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Pause]
    );
    overlay.key(key::ESCAPE, None, NONE, env.at(5600));
    assert!(
        overlay.drain_events().next().is_none(),
        "Esc first shows them again"
    );
    overlay.key(key::ESCAPE, None, NONE, env.at(5700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Exit]
    );
}

#[test]
fn speed_camera_and_render_settings_buttons_send_their_events() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.key(key::EQUALS, None, NONE, env.at(700));
    overlay.key(key::MINUS, None, NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Speed(2.0), ViewerEvent::Speed(0.75)]
    );
    overlay.key(key::C, None, NONE, env.at(700));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Camera(CameraMode::Follow)]
    );
    let free = find_tip(
        &|x, y| overlay.tooltip_at(x, y),
        "Free camera: hold right mouse to fly",
    )
    .expect("the free camera button");
    press_viewer(&mut overlay, free, env.at(800));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Camera(CameraMode::Free)]
    );
    let settings = find_tip(
        &|x, y| overlay.tooltip_at(x, y),
        "Shaders and mods, live over the replay",
    )
    .expect("Render settings");
    press_viewer(&mut overlay, settings, env.at(900));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::RenderSettings]
    );
    // The speed list: open it, pick 0.5x.
    let speed = find_tip(&|x, y| overlay.tooltip_at(x, y), "Speed").expect("speed");
    press_viewer(&mut overlay, speed, env.at(1000));
    let _ = frame(&mut overlay, &mut env, 1016);
    let bar = ReplayViewerOverlay::bar_rect(SCREEN);
    // The list floats above the bar: 8 items of 32 px, fastest on top, 0.1x at the bottom.
    let (sx, _) = speed;
    let half = (sx + 20.0, bar.y - 10.0 - 6.0 - 32.0 * 2.5);
    press_viewer(&mut overlay, half, env.at(1100));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Speed(0.5)]
    );
    // Open again; a click on the world only closes the list (no play/pause on its release).
    press_viewer(&mut overlay, speed, env.at(1200));
    let _ = frame(&mut overlay, &mut env, 1216);
    press_viewer(&mut overlay, (960.0, 300.0), env.at(1300));
    assert!(
        overlay.drain_events().next().is_none(),
        "closing the list is all that click does"
    );
    // With the list closed, a click on the world plays or pauses, as in a video player.
    let _ = frame(&mut overlay, &mut env, 1316);
    press_viewer(&mut overlay, (960.0, 300.0), env.at(1400));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Play]
    );
}

#[test]
fn export_opens_a_form_whose_request_follows_the_choices() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.set_selection(Some(20_000_000), Some(32_700_000));
    overlay.key(key::E, None, NONE, env.at(700));
    assert!(overlay.exporting());
    let _ = frame(&mut overlay, &mut env, 1000);
    // Choose 1500 fps and 10 MB, then Enter starts it.
    let dialog = overlay.export_dialog().expect("dialog");
    assert!(
        dialog.settings.selection,
        "a selection exports the selection"
    );
    dialog.settings.max_size = 2;
    dialog.settings.fps = 1500;
    overlay.key(key::ENTER, None, NONE, env.at(1100));
    let events: Vec<_> = overlay.drain_events().collect();
    let request = events
        .iter()
        .find_map(|e| match e {
            ViewerEvent::Export(request) => Some(request.clone()),
            _ => None,
        })
        .expect("an export request");
    assert_eq!((request.start_us, request.end_us), (20_000_000, 32_700_000));
    assert_eq!(request.fps, 1500);
    assert_eq!(request.max_bytes, Some(10_000_000));
    assert_eq!((request.width, request.height), (1920, 1080));
    assert!(!request.mic, "no microphone track, no microphone");
    assert_eq!(request.file_name, "SkyWars on mc.example.net.mp4");
    assert!(overlay.export_running());
    // Esc never throws a running export away.
    overlay.key(key::ESCAPE, None, NONE, env.at(1200));
    assert!(overlay.exporting());
    // Done: Esc (or Done) closes the window.
    overlay
        .export_dialog()
        .unwrap()
        .set_progress(ExportProgress {
            frames_done: 19_050,
            frames_total: 19_050,
            elapsed_s: 40.0,
            state: ExportState::Done {
                path: "C:/Videos/x.mp4".to_owned(),
                bytes: 9_700_000,
            },
        });
    assert!(!overlay.export_running());
    overlay.key(key::ESCAPE, None, NONE, env.at(1300));
    assert!(!overlay.exporting(), "Esc closes a finished export");
}

#[test]
fn a_typed_frame_rate_is_any_number_up_to_the_limit() {
    let mut env = Env::new();
    let now = env.at(0);
    let mut dialog = ExportDialog::new(
        ExportSettings::default(),
        None,
        10_000_000,
        (1600, 901),
        true,
        false,
        now,
    );
    let mut list = DrawList::new();
    let mut glass = GlassFrame::default();
    dialog.build(&mut env.env(500), &mut list, &mut glass);
    assert_eq!(glass.rects.len(), 1);
    assert!(glass.dim > 0.0, "the form dims the replay behind");
    // Click the box, type 1337, Enter.
    let field = dialog.fps_field_rect().expect("the fps box");
    let (x, y) = field.center();
    dialog.pointer_moved(x, y);
    dialog.mouse_pressed(x, y, 0);
    dialog.mouse_released(x, y, 0, env.at(600));
    assert!(dialog.typing());
    for c in "1337".chars() {
        dialog.key(0, Some(c), NONE, env.at(700));
    }
    dialog.key(key::ENTER, None, NONE, env.at(800));
    assert!(!dialog.typing());
    assert_eq!(dialog.settings.fps, 1337);
    // "Window" resolution rounds the window down to even numbers.
    dialog.settings.resolution = 0;
    assert_eq!(
        (dialog.request().width, dialog.request().height),
        (1600, 900)
    );
    // Over the limit clamps.
    dialog.mouse_pressed(x, y, 0);
    dialog.mouse_released(x, y, 0, env.at(900));
    for c in "99999".chars() {
        dialog.key(0, Some(c), NONE, env.at(900));
    }
    dialog.key(key::ENTER, None, NONE, env.at(950));
    assert_eq!(dialog.settings.fps, 9999);
    assert_eq!(dialog.request().fps, 9999);
    // 2000 fps (the brief's example) goes through untouched.
    dialog.settings.fps = 2000;
    assert_eq!(dialog.request().fps, 2000);
}

#[test]
fn a_one_click_clip_export_starts_at_once_and_leaves_when_closed() {
    let mut env = Env::new();
    let mut overlay = viewer(&mut env);
    overlay.export_now(ExportSettings::default(), env.at(700));
    let events: Vec<_> = overlay.drain_events().collect();
    assert!(
        matches!(events.as_slice(), [ViewerEvent::Export(_)]),
        "{events:?}"
    );
    for ms in (700..1200).step_by(16) {
        let _ = frame(&mut overlay, &mut env, ms);
    }
    assert!(
        !overlay.controls_visible(),
        "the progress panel takes the bar's place"
    );
    overlay
        .export_dialog()
        .unwrap()
        .set_progress(ExportProgress {
            frames_done: 10,
            frames_total: 10,
            elapsed_s: 1.0,
            state: ExportState::Failed("The GPU encoder stopped".to_owned()),
        });
    overlay.key(key::ESCAPE, None, NONE, env.at(1300));
    assert_eq!(
        overlay.drain_events().collect::<Vec<_>>(),
        vec![ViewerEvent::Exit]
    );
}

#[test]
fn the_viewer_lays_out_at_common_sizes_without_overlap() {
    for size in [
        (1280.0, 720.0),
        (1920.0, 1080.0),
        (1024.0, 768.0),
        (2560.0, 1080.0),
        (854.0, 480.0),
    ] {
        let mut env = Env::new();
        let mut overlay = ReplayViewerOverlay::new(info(), env.t0);
        overlay.set_status(status(10.0, false));
        overlay.set_selection(Some(1_000_000), Some(9_000_000));
        let mut list = DrawList::new();
        let mut e = env.env(800);
        e.screen = size;
        let _ = overlay.build(&mut e, &mut list);
        let rects = overlay.control_rects();
        assert!(rects.len() >= 15, "{size:?}: {}", rects.len());
        for (i, a) in rects.iter().enumerate() {
            assert!(
                a.x >= 0.0 && a.right() <= size.0,
                "{size:?}: {a:?} off screen"
            );
            for b in &rects[i + 1..] {
                let overlap = a.intersect(b);
                assert!(
                    overlap.w <= 1.0 || overlap.h <= 1.0,
                    "{size:?}: {a:?} overlaps {b:?}"
                );
            }
        }
    }
}
