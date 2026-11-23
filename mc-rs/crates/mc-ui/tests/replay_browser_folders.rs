//! The replay browser's Folders view and sort orders, headless with a 5-pixel test font (as
//! `tw_replay_ui_logic.rs`). Folders as on disk with their replay counts,
//! opened and left by keyboard and mouse, the path and the Up button, odd folder names, every
//! order and its headers, the choices sent to the client to remember, and a big library that is
//! laid out once per change, never per frame. Server and account names are made up.

use std::time::{Duration, Instant};

use chrono::{FixedOffset, TimeZone};

use mc_ui::font::{FontData, FontRenderer};
use mc_ui::screens::modmenu::text::MenuText;
use mc_ui::screens::replay::arrange::{Arrangement, ListView, SortOrder};
use mc_ui::screens::replay::browser::{Tab, TOOLBAR_H};
use mc_ui::screens::replay::{
    key, BrowserEvent, LibraryView, ReplayBrowserScreen, ReplayEntry, ReplayEnv,
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

#[allow(clippy::too_many_arguments)]
fn entry(
    id: u64,
    title: &str,
    folder: &str,
    started: u64,
    server: &str,
    player: &str,
    seconds: u64,
    mb: u64,
) -> ReplayEntry {
    ReplayEntry {
        id,
        title: title.to_owned(),
        server: server.to_owned(),
        started_unix_ms: started,
        duration_us: seconds * 1_000_000,
        size_bytes: mb * 1_000_000,
        clip: false,
        has_mic: false,
        problem: None,
        bookmarks: 0,
        folder: folder.to_owned(),
        player: player.to_owned(),
    }
}

const ODD: &str = "odd\u{202e}name";

/// Date folders, a folder of the player's own, an empty one, a replay left in the root and an
/// oddly named folder (a direction override in its name).
fn library() -> LibraryView {
    let day13 = "2026/11 November/13";
    let mut recordings = vec![
        entry(
            1,
            "Game A",
            day13,
            at(13, 23, 0),
            "a.example.net",
            "PlayerOne",
            300,
            5,
        ),
        entry(
            2,
            "Game B",
            day13,
            at(13, 21, 0),
            "b.example.net",
            "PlayerTwo",
            600,
            9,
        ),
        entry(
            3,
            "Game C",
            "2026/11 November/12",
            at(12, 20, 0),
            "A.example.net:25565",
            "PlayerOne",
            100,
            2,
        ),
        entry(
            4,
            "Ace",
            "Best games",
            at(11, 19, 0),
            "a.example.net",
            "PlayerOne",
            900,
            3,
        ),
        entry(
            5,
            "Flat",
            "",
            at(10, 18, 0),
            "c.example.net",
            "PlayerTwo",
            200,
            1,
        ),
        entry(
            6,
            "Odd one",
            &format!("{ODD}/inner"),
            at(9, 17, 0),
            "",
            "",
            400,
            4,
        ),
    ];
    recordings[3].bookmarks = 2;
    let mut clips = vec![
        entry(
            10,
            "Clip X",
            day13,
            at(13, 22, 0),
            "a.example.net",
            "PlayerOne",
            30,
            1,
        ),
        entry(
            11,
            "Clip Y",
            "",
            at(12, 8, 0),
            "b.example.net",
            "PlayerOne",
            30,
            1,
        ),
    ];
    for clip in &mut clips {
        clip.clip = true;
    }
    let keys = |list: &[&str]| list.iter().map(|k| (*k).to_owned()).collect::<Vec<_>>();
    LibraryView {
        recordings,
        clips,
        recording_folders: keys(&[
            "2026",
            "2026/11 November",
            "2026/11 November/13",
            "2026/11 November/12",
            "Best games",
            "Empty one",
            ODD,
            &format!("{ODD}/inner"),
        ]),
        clip_folders: keys(&["2026", "2026/11 November", "2026/11 November/13"]),
        cap_bytes: 20_000_000_000,
        recording_on: true,
        clip_key_set: true,
    }
}

fn draw(screen: &mut ReplayBrowserScreen, env: &mut Env, ms: u64) {
    let mut list = DrawList::new();
    let _ = screen.build(&mut env.env(ms), &mut list);
}

/// A browser on `library()` in `view` (as the client opens it: the remembered choices first).
fn browser(env: &mut Env, view: ListView) -> ReplayBrowserScreen {
    let mut screen = ReplayBrowserScreen::new(env.t0);
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Newest,
        view,
    });
    // "Now" is 01:30 on the 14th: the 13th is yesterday.
    screen.set_library_in_zone(library(), at(14, 1, 30), &est());
    draw(&mut screen, env, 1000);
    screen
}

fn rows(screen: &mut ReplayBrowserScreen) -> Vec<String> {
    screen
        .rows()
        .into_iter()
        .map(|(folder, name)| if folder { format!("[{name}]") } else { name })
        .collect()
}

fn find_tip(screen: &ReplayBrowserScreen, label: &str) -> Option<(f32, f32)> {
    let mut y = 0.0;
    while y < SCREEN.1 {
        let mut x = 0.0;
        while x < SCREEN.0 {
            if screen.tooltip_at(x, y) == Some(label) {
                return Some((x, y));
            }
            x += 4.0;
        }
        y += 4.0;
    }
    None
}

fn click(screen: &mut ReplayBrowserScreen, (x, y): (f32, f32), now: Instant) {
    screen.pointer_moved(x, y);
    screen.mouse_pressed(x, y, 0);
    screen.mouse_released(x, y, 0, now);
}

fn events(screen: &mut ReplayBrowserScreen) -> Vec<BrowserEvent> {
    screen.drain_events().collect()
}

/// The middle of row `n` (0 = first) when no header is above it, in its title area.
fn row_point(n: usize) -> (f32, f32) {
    let window = ReplayBrowserScreen::window_rect(SCREEN);
    (
        window.x + 400.0,
        window.y + 78.0 + TOOLBAR_H + 12.0 + 40.0 + 86.0 * n as f32,
    )
}

#[test]
fn the_folders_view_shows_folders_first_with_counts_and_the_keys_walk_them() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::Folders);
    // Folders first (the newest replay inside first, the empty one last), then the root's own.
    assert_eq!(
        rows(&mut screen),
        [
            "[2026]",
            "[Best games]",
            "[odd?name]",
            "[Empty one]",
            "Flat"
        ]
    );
    assert_eq!(screen.folder_path(), ["Recordings"]);
    assert_eq!(screen.selected_folder(), Some("2026"));
    assert_eq!(screen.selected(), None);
    // Enter opens, Right opens, Enter again: the day's replays.
    assert!(screen.key(key::ENTER, None, NONE, env.at(1100)));
    assert_eq!(rows(&mut screen), ["[11 November]"]);
    assert_eq!(screen.folder_path(), ["Recordings", "2026"]);
    screen.key(key::RIGHT, None, NONE, env.at(1110));
    assert_eq!(rows(&mut screen), ["[13]", "[12]"]);
    screen.key(key::ENTER, None, NONE, env.at(1120));
    assert_eq!(rows(&mut screen), ["Game A", "Game B"]);
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November/13");
    assert_eq!(screen.selected(), Some(1));
    assert_eq!(screen.headers(), ["Yesterday"]);
    // Backspace goes up with the folder it left selected; Left too.
    screen.key(key::BACKSPACE, None, NONE, env.at(1130));
    assert_eq!(screen.selected_folder(), Some("2026/11 November/13"));
    screen.key(key::DOWN, None, NONE, env.at(1140));
    assert_eq!(screen.selected_folder(), Some("2026/11 November/12"));
    screen.key(key::LEFT, None, NONE, env.at(1150));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026");
    assert_eq!(screen.selected_folder(), Some("2026/11 November"));
    screen.key(key::BACKSPACE, None, NONE, env.at(1160));
    screen.key(key::BACKSPACE, None, NONE, env.at(1170));
    assert_eq!(screen.folder_key(Tab::Recordings), "", "the root stops it");
    assert_eq!(screen.selected_folder(), Some("2026"));
    // The arrows run over folders and replays alike; Enter on a replay watches it.
    screen.key(key::END, None, NONE, env.at(1180));
    assert_eq!(screen.selected(), Some(5));
    screen.key(key::ENTER, None, NONE, env.at(1190));
    assert_eq!(events(&mut screen), [BrowserEvent::Watch(5)]);
    // Rename and Delete keys do nothing to a folder.
    screen.key(key::HOME, None, NONE, env.at(1200));
    screen.key(key::R, Some('r'), NONE, env.at(1210));
    assert!(!screen.renaming());
    screen.key(key::DELETE, None, NONE, env.at(1220));
    screen.key(key::DELETE, None, NONE, env.at(1230));
    assert!(events(&mut screen).is_empty());
    assert_eq!(rows(&mut screen).len(), 5);
    // An empty folder says so, and still draws.
    for _ in 0..3 {
        screen.key(key::DOWN, None, NONE, env.at(1240));
    }
    assert_eq!(screen.selected_folder(), Some("Empty one"));
    screen.key(key::ENTER, None, NONE, env.at(1250));
    assert!(rows(&mut screen).is_empty());
    draw(&mut screen, &mut env, 1260);
    assert_eq!(screen.selected(), None);
}

#[test]
fn the_mouse_opens_folders_and_the_path_up_and_back_buttons_go_back() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::Folders);
    // One click on a folder's row selects it; a second, quickly, opens it.
    click(&mut screen, row_point(1), env.at(1100));
    assert_eq!(screen.selected_folder(), Some("Best games"));
    assert_eq!(screen.folder_key(Tab::Recordings), "");
    click(&mut screen, row_point(1), env.at(1300));
    assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
    assert_eq!(rows(&mut screen), ["Ace"]);
    draw(&mut screen, &mut env, 1400);
    // The path's first part is the tab: a click goes back there, the folder selected.
    let root = find_tip(&screen, "Go to this folder").expect("the path's Recordings");
    click(&mut screen, root, env.at(1500));
    assert_eq!(screen.folder_key(Tab::Recordings), "");
    assert_eq!(screen.selected_folder(), Some("Best games"));
    draw(&mut screen, &mut env, 1600);
    assert!(
        find_tip(&screen, "Up one folder").is_none(),
        "no Up in the root"
    );
    // A folder's tile and Open button open it.
    let open = find_tip(&screen, "Open").expect("the first folder's Open");
    click(&mut screen, open, env.at(1700));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026");
    draw(&mut screen, &mut env, 1800);
    let open = find_tip(&screen, "Open").expect("11 November's Open");
    click(&mut screen, open, env.at(1900));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November");
    draw(&mut screen, &mut env, 2000);
    // Up one folder.
    let up = find_tip(&screen, "Up one folder").expect("Up");
    click(&mut screen, up, env.at(2100));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026");
    // The mouse's back button, as in Explorer.
    screen.mouse_pressed(row_point(0).0, row_point(0).1, 3);
    assert_eq!(screen.folder_key(Tab::Recordings), "");
    // In the List view the back button does nothing.
    screen.key(key::F, Some('f'), NONE, env.at(2200));
    screen.mouse_pressed(row_point(0).0, row_point(0).1, 3);
    assert_eq!(rows(&mut screen).len(), 6, "every replay, no folder");
}

#[test]
fn explorer_is_asked_for_a_folder_by_its_key_only() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::Folders);
    let explorer = find_tip(&screen, "Open in Explorer").expect("a folder's Explorer button");
    click(&mut screen, explorer, env.at(1100));
    assert_eq!(
        events(&mut screen),
        [BrowserEvent::OpenFolderAt {
            clips: false,
            folder: "2026".to_owned()
        }]
    );
    // The footer's Open folder opens the folder showing.
    screen.key(key::ENTER, None, NONE, env.at(1200));
    draw(&mut screen, &mut env, 1300);
    let footer = find_tip(&screen, "Open this folder in Explorer").expect("footer");
    click(&mut screen, footer, env.at(1400));
    assert_eq!(
        events(&mut screen),
        [BrowserEvent::OpenFolderAt {
            clips: false,
            folder: "2026".to_owned()
        }]
    );
    // The List view: the replays folder, and on the clips tab the clips folder.
    screen.key(key::F, Some('f'), NONE, env.at(1500));
    let _ = events(&mut screen);
    draw(&mut screen, &mut env, 1600);
    let footer = find_tip(&screen, "Open the replays folder").expect("footer, recordings");
    click(&mut screen, footer, env.at(1700));
    assert_eq!(events(&mut screen), [BrowserEvent::OpenFolder]);
    screen.key(key::TAB, None, NONE, env.at(1800));
    draw(&mut screen, &mut env, 1900);
    let footer = find_tip(&screen, "Open the clips folder").expect("footer, clips");
    click(&mut screen, footer, env.at(2000));
    assert_eq!(
        events(&mut screen),
        [BrowserEvent::OpenFolderAt {
            clips: true,
            folder: String::new()
        }]
    );
}

#[test]
fn the_view_switch_keeps_the_replay_and_shows_where_it_is() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::List);
    assert_eq!(screen.arrangement(), Arrangement::default());
    screen.select(3);
    // F: the Folders view opens on the replay's folder, still selected, and the client hears it.
    screen.key(key::F, Some('f'), NONE, env.at(1100));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November/12");
    assert_eq!(screen.selected(), Some(3));
    assert_eq!(
        events(&mut screen),
        [BrowserEvent::Arrange(Arrangement {
            sort: SortOrder::Newest,
            view: ListView::Folders
        })]
    );
    // The switch's segments work too.
    draw(&mut screen, &mut env, 1200);
    let list = find_tip(&screen, "Every replay in one list").expect("List");
    click(&mut screen, list, env.at(1300));
    assert_eq!(screen.arrangement().view, ListView::List);
    assert_eq!(screen.selected(), Some(3));
    assert_eq!(rows(&mut screen).len(), 6);
    let _ = events(&mut screen);
    // Watching a replay from another folder brings its folder back in the Folders view.
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Newest,
        view: ListView::Folders,
    });
    assert!(
        events(&mut screen).is_empty(),
        "set by the client: no event"
    );
    screen.select(4);
    assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
    assert_eq!(screen.selected(), Some(4));
    // A clip: its tab and its folder.
    screen.select(10);
    assert_eq!(screen.tab(), Tab::Clips);
    assert_eq!(screen.folder_key(Tab::Clips), "2026/11 November/13");
    assert_eq!(rows(&mut screen), ["Clip X"]);
    // Each tab keeps its own folder.
    screen.key(key::TAB, None, NONE, env.at(1400));
    assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
}

#[test]
fn every_order_sorts_and_groups_the_list() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::List);
    let sorted = |screen: &mut ReplayBrowserScreen, order: SortOrder| {
        screen.set_arrangement(Arrangement {
            sort: order,
            view: ListView::List,
        });
        (rows(screen), screen.headers())
    };
    let (order, headers) = sorted(&mut screen, SortOrder::Newest);
    assert_eq!(
        order,
        ["Game A", "Game B", "Game C", "Ace", "Flat", "Odd one"]
    );
    assert_eq!(
        headers,
        ["Yesterday", "Thursday", "Wednesday", "Tuesday", "Monday"]
    );
    let (order, headers) = sorted(&mut screen, SortOrder::Oldest);
    assert_eq!(
        order,
        ["Odd one", "Flat", "Ace", "Game C", "Game B", "Game A"]
    );
    assert_eq!(headers[0], "Monday");
    // The default port and case are the same server; an unknown server goes last.
    let (order, headers) = sorted(&mut screen, SortOrder::Server);
    assert_eq!(
        order,
        ["Game A", "Game C", "Ace", "Game B", "Flat", "Odd one"]
    );
    assert_eq!(
        headers,
        [
            "a.example.net",
            "b.example.net",
            "c.example.net",
            "Unknown server"
        ]
    );
    let (order, headers) = sorted(&mut screen, SortOrder::Name);
    assert_eq!(
        order,
        ["Ace", "Flat", "Game A", "Game B", "Game C", "Odd one"]
    );
    assert!(headers.is_empty());
    let (order, _) = sorted(&mut screen, SortOrder::Longest);
    assert_eq!(
        order,
        ["Ace", "Game B", "Odd one", "Game A", "Flat", "Game C"]
    );
    let (order, _) = sorted(&mut screen, SortOrder::Largest);
    assert_eq!(
        order,
        ["Game B", "Game A", "Odd one", "Ace", "Game C", "Flat"]
    );
    let (order, _) = sorted(&mut screen, SortOrder::Bookmarks);
    assert_eq!(
        order,
        ["Ace", "Game A", "Game B", "Game C", "Flat", "Odd one"]
    );
    let (order, headers) = sorted(&mut screen, SortOrder::Account);
    assert_eq!(
        order,
        ["Game A", "Game C", "Ace", "Game B", "Flat", "Odd one"]
    );
    assert_eq!(headers, ["PlayerOne", "PlayerTwo", "Unknown account"]);
    // Folders follow the order too: by size, the biggest first.
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Largest,
        view: ListView::Folders,
    });
    assert_eq!(
        rows(&mut screen),
        [
            "[2026]",
            "[odd?name]",
            "[Best games]",
            "[Empty one]",
            "Flat"
        ]
    );
}

#[test]
fn the_sort_menu_works_by_keyboard_and_mouse_and_a_click_outside_only_closes_it() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::List);
    // O opens it on the current order; arrows choose; Enter picks; the client hears it.
    screen.key(key::O, Some('o'), NONE, env.at(1100));
    for _ in 0..3 {
        screen.key(key::DOWN, None, NONE, env.at(1110));
    }
    screen.key(key::ENTER, None, NONE, env.at(1120));
    assert_eq!(screen.arrangement().sort, SortOrder::Server);
    assert_eq!(
        events(&mut screen),
        [BrowserEvent::Arrange(Arrangement {
            sort: SortOrder::Server,
            view: ListView::List
        })]
    );
    // Esc closes it without leaving the browser.
    screen.key(key::O, Some('o'), NONE, env.at(1200));
    screen.key(key::ESCAPE, None, NONE, env.at(1210));
    assert!(events(&mut screen).is_empty());
    // The mouse: the Sort button opens it under itself; an item picks.
    draw(&mut screen, &mut env, 1300);
    let (bx, by) = find_tip(&screen, "Sort order").expect("the Sort button");
    let mut bottom = by;
    while screen.tooltip_at(bx, bottom + 1.0) == Some("Sort order") {
        bottom += 1.0;
    }
    click(&mut screen, (bx, by), env.at(1400));
    draw(&mut screen, &mut env, 1500);
    // Item 2 ("Name"): 6 px below the button, 6 px of padding, 32 px items.
    let item = (bx + 4.0, bottom + 1.0 + 6.0 + 6.0 + 32.0 * 2.0 + 16.0);
    click(&mut screen, item, env.at(1600));
    assert_eq!(screen.arrangement().sort, SortOrder::Name);
    let _ = events(&mut screen);
    // A click outside closes it and does nothing else (no selection change, no watch).
    screen.key(key::HOME, None, NONE, env.at(1650));
    assert_eq!(screen.selected(), Some(4), "Ace, first by name");
    screen.key(key::O, Some('o'), NONE, env.at(1700));
    draw(&mut screen, &mut env, 1800);
    // Row 2 is Game A (id 1), outside the menu.
    click(&mut screen, row_point(2), env.at(1900));
    assert!(
        events(&mut screen).is_empty(),
        "the outside click was eaten"
    );
    assert_eq!(screen.selected(), Some(4));
    assert_eq!(screen.arrangement().sort, SortOrder::Name);
    // Closed: the arrows move the selection again, not the menu's highlight.
    screen.key(key::DOWN, None, NONE, env.at(2000));
    assert_eq!(screen.selected(), Some(5), "Flat");
    // The same click with the menu closed selects the row.
    draw(&mut screen, &mut env, 2100);
    click(&mut screen, row_point(2), env.at(2200));
    assert_eq!(screen.selected(), Some(1));
}

#[test]
fn the_open_folder_survives_a_refresh_and_a_start_folder_opens() {
    let env = Env::new();
    // Where the player was earlier this session.
    let mut screen = ReplayBrowserScreen::new(env.t0);
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Newest,
        view: ListView::Folders,
    });
    screen.set_start_folders(Some("Best games".to_owned()), Some("2026".to_owned()));
    screen.set_library_in_zone(library(), at(14, 1, 30), &est());
    assert_eq!(screen.folder_key(Tab::Recordings), "Best games");
    assert_eq!(screen.folder_key(Tab::Clips), "2026");
    assert_eq!(rows(&mut screen), ["Ace"]);
    // The day folder empties and goes (a refresh): the nearest folder above it opens.
    screen.key(key::BACKSPACE, None, NONE, env.at(1000));
    screen.key(key::HOME, None, NONE, env.at(1005));
    screen.key(key::ENTER, None, NONE, env.at(1010));
    screen.key(key::ENTER, None, NONE, env.at(1020));
    screen.key(key::ENTER, None, NONE, env.at(1030));
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November/13");
    let mut refreshed = library();
    refreshed.recordings.retain(|e| e.id != 1 && e.id != 2);
    refreshed
        .recording_folders
        .retain(|k| k != "2026/11 November/13");
    screen.set_library_in_zone(refreshed, at(14, 1, 30), &est());
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November");
    assert_eq!(rows(&mut screen), ["[12]"]);
    // A start folder that is not there opens the root.
    let mut screen = ReplayBrowserScreen::new(env.t0);
    screen.set_start_folders(Some("Gone/Deeper".to_owned()), None);
    screen.set_library_in_zone(library(), at(14, 1, 30), &est());
    assert_eq!(screen.folder_key(Tab::Recordings), "");
}

#[test]
fn deleting_and_renaming_inside_a_folder_keep_it_open() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::Folders);
    for _ in 0..3 {
        screen.key(key::ENTER, None, NONE, env.at(1100));
    }
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November/13");
    screen.key(key::DELETE, None, NONE, env.at(1200));
    screen.key(key::DELETE, None, NONE, env.at(1300));
    assert_eq!(events(&mut screen), [BrowserEvent::Delete(1)]);
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/11 November/13");
    assert_eq!(rows(&mut screen), ["Game B"]);
    assert_eq!(screen.selected(), Some(2));
    // The counts above follow.
    screen.key(key::BACKSPACE, None, NONE, env.at(1400));
    draw(&mut screen, &mut env, 1500);
    // Renamed under the Name order, a replay moves to its new place.
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Name,
        view: ListView::List,
    });
    assert_eq!(
        rows(&mut screen),
        ["Ace", "Flat", "Game B", "Game C", "Odd one"]
    );
    screen.select(6);
    screen.key(key::R, Some('r'), NONE, env.at(1600));
    for c in "A first".chars() {
        screen.key(0, Some(c), NONE, env.at(1610));
    }
    screen.key(key::ENTER, None, NONE, env.at(1620));
    assert_eq!(
        rows(&mut screen),
        ["A first", "Ace", "Flat", "Game B", "Game C"]
    );
    assert_eq!(screen.selected(), Some(6));
}

#[test]
fn odd_folder_names_show_safely() {
    let mut env = Env::new();
    let mut screen = browser(&mut env, ListView::Folders);
    screen.key(key::DOWN, None, NONE, env.at(1100));
    screen.key(key::DOWN, None, NONE, env.at(1110));
    assert_eq!(screen.selected_folder(), Some(ODD), "the key stays exact");
    screen.key(key::ENTER, None, NONE, env.at(1120));
    assert_eq!(screen.folder_path(), ["Recordings", "odd?name"]);
    assert_eq!(rows(&mut screen), ["[inner]"]);
    draw(&mut screen, &mut env, 1200);
    // Keys a listing could never make are refused: those replays are listed in the root.
    let mut view = library();
    view.recordings[4].folder = "../../outside".to_owned();
    view.recording_folders.push("a//b".to_owned());
    view.recording_folders.push("..".to_owned());
    let mut screen = browser(&mut env, ListView::Folders);
    screen.set_library_in_zone(view, at(14, 1, 30), &est());
    assert_eq!(
        rows(&mut screen),
        [
            "[2026]",
            "[Best games]",
            "[odd?name]",
            "[Empty one]",
            "Flat"
        ]
    );
}

#[test]
fn a_big_library_is_laid_out_once_per_change_never_per_frame() {
    let mut env = Env::new();
    let mut recordings = Vec::with_capacity(20_000);
    for i in 0..20_000u64 {
        let folder = format!("2026/{:02} Month/{:02}", 1 + i % 12, 1 + (i / 12) % 28);
        let started = at(1, 0, 0) - i * 60_000;
        let title = format!("Game {i}");
        let server = format!("s{}.example.net", i % 40);
        recordings.push(entry(
            i + 1,
            &title,
            &folder,
            started,
            &server,
            "PlayerOne",
            60,
            2,
        ));
    }
    let view = LibraryView {
        recordings,
        cap_bytes: 20_000_000_000,
        ..LibraryView::default()
    };
    let mut screen = ReplayBrowserScreen::new(env.t0);
    let start = Instant::now();
    screen.set_library_in_zone(view, at(14, 1, 30), &est());
    draw(&mut screen, &mut env, 1000);
    let loaded = start.elapsed();
    let laid_out = screen.layouts_made();
    let start = Instant::now();
    for frame in 0..200 {
        draw(&mut screen, &mut env, 1000 + frame * 16);
    }
    let frames = start.elapsed();
    assert_eq!(
        screen.layouts_made(),
        laid_out,
        "200 frames laid nothing out"
    );
    // A change lays out once: the Folders view, on the selected (newest) replay's day folder.
    screen.key(key::F, Some('f'), NONE, env.at(5000));
    draw(&mut screen, &mut env, 5016);
    draw(&mut screen, &mut env, 5032);
    assert_eq!(screen.layouts_made(), laid_out + 1);
    assert_eq!(screen.folder_key(Tab::Recordings), "2026/01 Month/01");
    assert_eq!(
        rows(&mut screen).len(),
        60,
        "one replay in 336 is on this day"
    );
    screen.key(key::BACKSPACE, None, NONE, env.at(5100));
    screen.key(key::BACKSPACE, None, NONE, env.at(5110));
    assert_eq!(rows(&mut screen).len(), 12, "twelve month folders");
    assert_eq!(screen.selected_folder(), Some("2026/01 Month"));
    screen.key(key::BACKSPACE, None, NONE, env.at(5120));
    assert_eq!(rows(&mut screen), ["[2026]"]);
    let start = Instant::now();
    screen.set_arrangement(Arrangement {
        sort: SortOrder::Server,
        view: ListView::List,
    });
    let resorted = start.elapsed();
    assert_eq!(screen.headers().len(), 40);
    eprintln!(
        "20,000 replays: listed and drawn in {loaded:?}; 200 frames in {frames:?} ({:?} each); \
         sorted by server in {resorted:?} (debug build, timings provisional)",
        frames / 200
    );
}
