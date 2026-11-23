//! The replay browser's day groups, short recordings folded
//! away (and shown with S or the footer button), the bookmark count, and the viewer's `[` and
//! `]` between bookmarks. Headless, with a 5-pixel test font (as `tw_replay_ui_logic.rs`).

use std::time::{Duration, Instant};

use chrono::{FixedOffset, TimeZone};

use mc_ui::font::{FontData, FontRenderer};
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::screens::replay::{
    key, LibraryView, PlaybackStatus, ReplayBrowserScreen, ReplayEntry, ReplayEnv, ReplayInfo,
    ReplayViewerOverlay, ViewerEvent,
};
use mc_ui::widget::keys::Modifiers;
use mc_ui::widget::DrawList;

const SCREEN: (f32, f32) = (1920.0, 1080.0);
const NONE: Modifiers = Modifiers::NONE;

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

fn est() -> FixedOffset {
    FixedOffset::west_opt(5 * 3600).unwrap()
}

fn at(d: u32, h: u32, mi: u32) -> u64 {
    est()
        .with_ymd_and_hms(2026, 11, d, h, mi, 0)
        .unwrap()
        .timestamp_millis() as u64
}

fn entry(id: u64, started: u64, seconds: u64, bookmarks: u32) -> ReplayEntry {
    ReplayEntry {
        id,
        title: format!("game {id}"),
        server: "mc.example.net".to_owned(),
        started_unix_ms: started,
        duration_us: seconds * 1_000_000,
        size_bytes: 1_000_000,
        clip: false,
        has_mic: false,
        problem: None,
        bookmarks,
        folder: String::new(),
        player: String::new(),
    }
}

/// Like the user's library on the night of Nov 13: games, lobby hops of a few seconds, and one
/// short recording with a bookmark (kept in view).
fn library() -> LibraryView {
    LibraryView {
        recordings: vec![
            entry(1, at(13, 23, 26), 1798, 3),
            entry(2, at(13, 21, 30), 162, 0),
            entry(3, at(13, 21, 30) - 14_000, 10, 0), // lobby hop
            entry(4, at(13, 21, 29), 9, 0),           // lobby hop
            entry(5, at(13, 20, 51), 165, 0),
            entry(6, at(13, 20, 48), 1, 0),  // reconnect
            entry(7, at(13, 20, 47), 12, 1), // short, but bookmarked
            entry(8, at(12, 9, 33), 13, 0),  // lobby hop, the day before
        ],
        clips: Vec::new(),
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: true,
        ..LibraryView::default()
    }
}

fn browser(env: &mut Env) -> ReplayBrowserScreen {
    let mut screen = ReplayBrowserScreen::new(env.t0);
    // "Now" is 01:30 on the 14th: the 13th is yesterday.
    screen.set_library_in_zone(library(), at(14, 1, 30), &est());
    draw(&mut screen, env, 1000);
    screen
}

fn draw(screen: &mut ReplayBrowserScreen, env: &mut Env, ms: u64) {
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(ms), &mut list);
}

/// The ids in list order, walked with the arrow keys from the top.
fn listed(screen: &mut ReplayBrowserScreen, env: &Env) -> Vec<u64> {
    screen.key(key::HOME, None, NONE, env.at(1100));
    let mut ids = Vec::new();
    for _ in 0..20 {
        let id = screen.selected().expect("a selection");
        if ids.last() == Some(&id) {
            break;
        }
        ids.push(id);
        screen.key(key::DOWN, None, NONE, env.at(1100));
    }
    ids
}

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

/// The rows' vertical centres (their Watch buttons), top to bottom.
fn row_centres(screen: &ReplayBrowserScreen) -> Vec<f32> {
    let mut ys: Vec<f32> = Vec::new();
    let mut y = 0.0;
    while y < SCREEN.1 {
        let mut x = 0.0;
        let mut hit = false;
        while x < SCREEN.0 {
            if screen.tooltip_at(x, y) == Some("Watch") {
                hit = true;
                break;
            }
            x += 8.0;
        }
        // A row's tile and its Watch button both say "Watch": one entry per row.
        if hit && ys.last().is_none_or(|last| y - last > 70.0) {
            ys.push(y);
        }
        y += 2.0;
    }
    ys
}

#[test]
fn short_recordings_are_folded_away_until_shown() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    assert!(!screen.showing_short());
    // Under 20 s with no bookmark: 3, 4, 6 and 8 are folded; 7 has a bookmark and stays.
    assert_eq!(listed(&mut screen, &env), vec![1, 2, 5, 7]);
    // The footer offers them, and S shows them (and again hides them).
    draw(&mut screen, &mut env, 1200);
    let toggle = find_tip(
        &|x, y| screen.tooltip_at(x, y),
        "Recordings under 20 s with no bookmark",
    );
    assert!(toggle.is_some(), "the footer's Show 4 short button");
    assert!(screen.key(key::S, Some('s'), NONE, env.at(1300)));
    assert!(screen.showing_short());
    draw(&mut screen, &mut env, 1400);
    assert_eq!(listed(&mut screen, &env), vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // The button works like the key.
    draw(&mut screen, &mut env, 1500);
    let (x, y) = find_tip(
        &|x, y| screen.tooltip_at(x, y),
        "Recordings under 20 s with no bookmark",
    )
    .expect("the Hide short button");
    screen.pointer_moved(x, y);
    screen.mouse_pressed(x, y, 0);
    screen.mouse_released(x, y, 0, env.at(1600));
    assert!(!screen.showing_short());
    draw(&mut screen, &mut env, 1700);
    assert_eq!(listed(&mut screen, &env), vec![1, 2, 5, 7]);
    // Coming back from watching a short one shows them again, with it selected.
    screen.select(4);
    assert!(screen.showing_short());
    assert_eq!(screen.selected(), Some(4));
}

#[test]
fn rows_sit_under_a_header_per_day() {
    let mut env = Env::new();
    let mut screen = browser(&mut env);
    screen.key(key::S, Some('s'), NONE, env.at(1100));
    draw(&mut screen, &mut env, 1200);
    let ys = row_centres(&screen);
    assert!(ys.len() >= 6, "rows on screen: {ys:?}");
    // Rows of one day are a row apart; the first row of the next day (8, the 12th) has the
    // day's header above it as well. The first 7 rows are the 13th.
    let gaps: Vec<f32> = ys.windows(2).map(|w| w[1] - w[0]).collect();
    let row = gaps[0];
    assert!(
        gaps[..gaps.len().min(6)]
            .iter()
            .all(|g| (g - row).abs() < 3.0),
        "{gaps:?}"
    );
    if ys.len() >= 8 {
        assert!(gaps[6] > row + 20.0, "a header between the days: {gaps:?}");
    }
}

// ---- the viewer -----------------------------------------------------------------------------

fn viewer(env: &mut Env, markers: Vec<(u64, String)>) -> ReplayViewerOverlay {
    let info = ReplayInfo {
        title: "SkyWars".to_owned(),
        server: "mc.example.net".to_owned(),
        started_unix_ms: 0,
        clip: false,
        has_mic: false,
        markers,
        window_size: (1920, 1080),
    };
    let mut overlay = ReplayViewerOverlay::new(info, env.t0);
    overlay.set_status(status(10.0));
    let mut list = DrawList::new();
    for ms in (0..=600).step_by(16) {
        list.clear();
        let _ = overlay.build(&mut env.env(ms), &mut list);
    }
    overlay
}

fn status(time_s: f32) -> PlaybackStatus {
    PlaybackStatus {
        duration_us: 120_000_000,
        time_us: (time_s * 1e6) as u64,
        playing: true,
        ..PlaybackStatus::default()
    }
}

fn seeks(overlay: &mut ReplayViewerOverlay) -> Vec<u64> {
    overlay
        .drain_events()
        .filter_map(|e| match e {
            ViewerEvent::Seek(t) => Some(t),
            _ => None,
        })
        .collect()
}

#[test]
fn brackets_step_between_bookmarks() {
    let mut env = Env::new();
    let mut overlay = viewer(
        &mut env,
        vec![
            (90_000_000, "Bookmark".to_owned()),
            (30_000_000, "Bookmark".to_owned()),
            (60_000_000, "Clip".to_owned()),
        ],
    );
    let press = |overlay: &mut ReplayViewerOverlay, code: i32, t: f32, ms: u64| {
        overlay.set_status(status(t));
        overlay.key(code, None, NONE, env.at(ms));
        seeks(overlay)
    };
    assert_eq!(
        press(&mut overlay, key::RBRACKET, 10.0, 700),
        vec![30_000_000]
    );
    assert_eq!(
        press(&mut overlay, key::RBRACKET, 30.0, 710),
        vec![60_000_000]
    );
    assert_eq!(
        press(&mut overlay, key::RBRACKET, 60.0, 720),
        vec![90_000_000]
    );
    assert!(
        press(&mut overlay, key::RBRACKET, 95.0, 730).is_empty(),
        "none after"
    );
    // Previous: just after a marker, the one before; later on, back to its start.
    assert_eq!(
        press(&mut overlay, key::LBRACKET, 60.5, 740),
        vec![30_000_000]
    );
    assert_eq!(
        press(&mut overlay, key::LBRACKET, 63.0, 750),
        vec![60_000_000]
    );
    assert!(
        press(&mut overlay, key::LBRACKET, 30.5, 760).is_empty(),
        "none before"
    );
    // The diamonds' tooltips name the keys.
    let mut list = DrawList::new();
    let _ = overlay.build(&mut env.env(800), &mut list);
    assert!(find_tip(&|x, y| overlay.tooltip_at(x, y), "Bookmark").is_some());
    // No markers: nothing to seek.
    let mut plain = viewer(&mut env, Vec::new());
    plain.key(key::RBRACKET, None, NONE, env.at(900));
    assert!(seeks(&mut plain).is_empty());
}

#[test]
fn a_library_of_only_short_recordings_says_so_and_still_offers_them() {
    let mut env = Env::new();
    let mut screen = ReplayBrowserScreen::new(env.t0);
    let library = LibraryView {
        recordings: vec![
            entry(1, at(13, 20, 48), 1, 0),
            entry(2, at(13, 20, 47), 9, 0),
        ],
        clips: Vec::new(),
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: false,
        ..LibraryView::default()
    };
    screen.set_library_in_zone(library, at(14, 1, 30), &est());
    draw(&mut screen, &mut env, 1000);
    assert_eq!(screen.selected(), None, "nothing listed");
    assert!(find_tip(
        &|x, y| screen.tooltip_at(x, y),
        "Recordings under 20 s with no bookmark"
    )
    .is_some());
    screen.key(key::S, Some('s'), NONE, env.at(1100));
    draw(&mut screen, &mut env, 1200);
    assert_eq!(listed(&mut screen, &env), vec![1, 2]);
}
