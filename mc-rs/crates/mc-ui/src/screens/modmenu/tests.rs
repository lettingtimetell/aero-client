//! Headless menu tests: layout, clicks, keys, the HUD editor and the animation clock, with the
//! deterministic 5-pixel test font and no GPU.
//!
//! Rewritten for the three windows (Where, What, How). Every
//! behaviour the seven-window tests covered keeps a test here in its new form; new tests cover
//! the views, the last place, My setup and pins, per-setting reset, the rules, the keyboard and
//! a menu without a world.

use std::time::{Duration, Instant};

use mc_mods::menu_layout::CATEGORIES;
use mc_mods::menu_rules::{self, Cond, Heading, OptionState, Unmet};
use mc_mods::{HudPlacement, OptionKind, OptionValue, RowButton, RowIcon};

use super::details::nav_option;
use super::glass::GlassFrame;
use super::input::{
    KEY_BACK, KEY_DOWN, KEY_ESCAPE, KEY_LEFT, KEY_RETURN, KEY_RIGHT, KEY_SPACE, KEY_TAB,
    SEARCH_MAX_CHARS,
};
use super::layout::{MENU_DIM, SIDE_MARGIN, WINDOW_GAP};
use super::list::MenuOrder;
use super::paint::{css, Rect};
use super::state::{Mode, Pane, Target, View};
use super::*;
use crate::widget::keys::{self, Clipboard, Modifiers};
use crate::widget::test_support::uniform_font;
use crate::widget::DrawList;

const SCREEN: (f32, f32) = (1920.0, 1080.0);
const GUI_SCALE: f32 = 3.0;
const GUI_SCREEN: (f32, f32) = (640.0, 360.0);

/// The window-1 entry of the category with this id.
fn category(id: &str) -> View {
    let c = CATEGORIES
        .iter()
        .position(|category| category.id == id)
        .expect("a category with this id");
    View::Category(u8::try_from(c).unwrap())
}

struct Harness {
    settings: ModSettings,
    hud: ModHud,
    rows: Vec<DynamicRows>,
    text: MenuText,
    mc: FontRenderer,
    events: Vec<ModEvent>,
    screen: ModMenuScreen,
    t0: Instant,
    /// The window size in physical pixels (1920x1080 unless a test changes it).
    size: (f32, f32),
    /// Shift held for the next key or click.
    shift: bool,
    /// No world is loaded (the title screen).
    no_world: bool,
}

impl Harness {
    fn new() -> Harness {
        let settings = ModSettings::new();
        let t0 = Instant::now();
        Harness {
            rows: vec![DynamicRows::default(); settings.catalog().len()],
            screen: ModMenuScreen::new(settings.catalog(), t0),
            settings,
            hud: ModHud::new(),
            text: MenuText::minecraft_only(),
            mc: uniform_font(),
            events: Vec::new(),
            t0,
            size: SCREEN,
            shift: false,
            no_world: false,
        }
    }

    fn at(&self, ms: u64) -> Instant {
        self.t0 + Duration::from_millis(ms)
    }

    fn build(&mut self, ms: u64) -> (DrawList, GlassFrame) {
        let mut list = DrawList::new();
        let now = self.at(ms);
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.mc,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: self.size,
            gui_scale: GUI_SCALE,
            now,
            time_s: 0.0,
            game: MenuGame {
                no_world: self.no_world,
                ..MenuGame::default()
            },
        };
        let glass = self.screen.build(&mut env, &mut list);
        (list, glass)
    }

    fn input(&mut self, ms: u64) -> MenuInput<'_> {
        let now = self.t0 + Duration::from_millis(ms);
        MenuInput {
            settings: &mut self.settings,
            hud: &mut self.hud,
            events: &mut self.events,
            gui_screen: GUI_SCREEN,
            gui_scale: GUI_SCALE,
            now,
            alt: false,
            shift: self.shift,
        }
    }

    fn click(&mut self, x: f32, y: f32, button: u8, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.mouse_pressed(x, y, button, &mut input);
            screen.mouse_released(x, y, button, &mut input);
        }
        self.screen = screen;
    }

    fn key(&mut self, lwjgl: i32, ms: u64) -> KeyOutcome {
        self.key_char(lwjgl, None, ms)
    }

    fn key_char(&mut self, lwjgl: i32, typed: Option<char>, ms: u64) -> KeyOutcome {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        let outcome = {
            let mut input = self.input(ms);
            screen.key(lwjgl, typed, &mut input)
        };
        self.screen = screen;
        outcome
    }

    fn clipboard_key(
        &mut self,
        lwjgl: i32,
        modifiers: Modifiers,
        clipboard: &mut dyn Clipboard,
    ) -> bool {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        let handled = screen.clipboard_shortcut(lwjgl, modifiers, clipboard, &mut self.input(1000));
        self.screen = screen;
        handled
    }

    /// Type `text` as key presses with characters (the LWJGL code does not matter for typing;
    /// 30 is A).
    fn type_text(&mut self, text: &str, ms: u64) {
        for c in text.chars() {
            let code = if c == ' ' { 57 } else { 30 };
            assert_eq!(
                self.key_char(code, Some(c), ms),
                KeyOutcome::Handled,
                "{c:?}"
            );
        }
    }

    /// The first hit region matching `pick`, from the last build.
    fn hit_where(&self, pick: impl Fn(&Target) -> bool) -> Option<Rect> {
        self.screen
            .hits
            .iter()
            .find_map(|(rect, target)| pick(target).then_some(*rect))
    }

    fn click_target(&mut self, pick: impl Fn(&Target) -> bool, ms: u64) {
        let rect = self.hit_where(pick).expect("the target is on screen");
        let (x, y) = rect.center();
        self.click(x, y, 0, ms);
    }

    /// Show `mod_id` in window 3 (as a click on its row does) and lay the menu out.
    fn open(&mut self, mod_id: &str, ms: u64) -> usize {
        let m = self.settings.index_of(mod_id).expect("in the catalog");
        self.screen.select(m);
        self.build(ms);
        m
    }
}

/// Frames for half a second from `from_ms`: animations and eased scrolling settle.
fn settle(h: &mut Harness, from_ms: u64) {
    for ms in (from_ms..from_ms + 480).step_by(16) {
        h.build(ms);
    }
}

fn paste(h: &mut Harness, text: &str, ms: u64) -> bool {
    let mut screen = std::mem::replace(&mut h.screen, ModMenuScreen::new(&[], h.t0));
    let took = {
        let mut input = h.input(ms);
        screen.paste(text, &mut input)
    };
    h.screen = screen;
    took
}

// ---- the windows and the animation ---------------------------------------------------------

#[test]
fn three_glass_windows_and_the_taskbar_once_open_and_the_dim_ramps_with_the_animation() {
    let mut h = Harness::new();
    let (_, start) = h.build(0);
    assert!(start.dim < 0.01, "the dim starts at zero: {}", start.dim);
    assert!(start.rects.iter().all(|r| r.alpha < 0.05));
    let (list, open) = h.build(1000);
    assert!(!list.is_empty());
    // Three windows (Where, What, How) plus the taskbar's "Edit HUD".
    assert_eq!(open.rects.len(), 4, "{:?}", open.rects);
    assert!(
        (open.dim - 0.35).abs() < 1e-4,
        "glass.js uDim: {}",
        open.dim
    );
    assert!((open.strength - 1.0).abs() < 1e-4);
    assert_eq!(open.frost, 1.5);
    // Windows sit side by side, left to right, without overlapping, inside the screen; each is
    // wider than the one before (220, 330, 440 design px).
    let windows: Vec<_> = open.rects.iter().take(3).map(|r| r.rect).collect();
    for pair in windows.windows(2) {
        assert!(pair[0][0] + pair[0][2] <= pair[1][0], "{pair:?}");
        assert!(pair[0][2] < pair[1][2], "wider to the right: {pair:?}");
    }
    assert!(windows[0][0] >= 0.0 && windows[2][0] + windows[2][2] <= SCREEN.0);
    // Mid-animation (50 ms): windows start 35 ms apart, so 0 and 1 are on their way and 2 has
    // not started (it is not drawn at all); the later one lags the first.
    let (_, mid) = h.build(50);
    assert_eq!(
        mid.rects.len(),
        2 + 1,
        "two windows plus the taskbar: {:?}",
        mid.rects
    );
    assert!(mid.rects[0].alpha > mid.rects[1].alpha);
    assert!(
        mid.rects[1].rect[1] > open.rects[1].rect[1],
        "still sliding up"
    );
    assert_eq!(
        mid.rects[1].rect[0], open.rects[1].rect[0],
        "the slide is vertical only"
    );
}

/// Item 11: Esc and the menu key (Right Shift) close; Tab does not (it moves the keyboard
/// between the windows now, and it is the player list's key in game).
#[test]
fn escape_and_the_menu_key_close_but_tab_does_not_and_the_close_animation_finishes() {
    let mut h = Harness::new();
    h.build(500);
    assert_eq!(h.key(KEY_ESCAPE, 500), KeyOutcome::Close);
    assert_eq!(h.key(mc_mods::DEFAULT_MENU_KEY, 500), KeyOutcome::Close);
    assert_eq!(h.key(KEY_TAB, 500), KeyOutcome::Handled);
    let close_at = h.at(600);
    h.screen.begin_close(close_at);
    assert!(!h.screen.finished(close_at));
    let (_, closing) = h.build(650);
    assert!(closing.dim < 0.35 && closing.dim > 0.0);
    assert!(h.screen.finished(h.at(600 + 250)));
}

#[test]
fn the_menu_scale_fits_three_windows_and_the_height_decides_at_16_9() {
    assert_eq!(
        design_width(),
        220.0 + 330.0 + 440.0 + 2.0 * 16.0 + 2.0 * 32.0
    );
    for (size, s) in [
        ((1280.0, 720.0), 0.8),
        ((1920.0, 1080.0), 1.2),
        ((2560.0, 1440.0), 1.6),
    ] {
        assert!((px_per_dip(size) - s).abs() < 1e-4, "{size:?}");
    }
    // About 20% larger text than the seven windows had (0.98 px per dip at 1080p).
    let seven = 1920.0 / (7.0 * 256.0 + 6.0 * 16.0 + 2.0 * 32.0);
    assert!(px_per_dip((1920.0, 1080.0)) / seven > 1.2);
    assert!(px_per_dip((854.0, 480.0)) >= 0.55);
    assert!(px_per_dip((3840.0, 2160.0)) > 1.9);
}

/// The layout at 1280x720, 1920x1080 and 2560x1440: three windows of 220, 330 and 440 design
/// pixels, equal gaps, the same margin on both sides, inside the screen; the taskbar pane under
/// them, centred and on screen; the same design-pixel layout at every 16:9 size.
#[test]
fn the_three_window_layout_is_the_same_design_at_720p_1080p_and_1440p() {
    let mut layouts = Vec::new();
    for size in [(1280.0, 720.0), (1920.0, 1080.0), (2560.0, 1440.0)] {
        let mut h = Harness::new();
        h.size = size;
        let (_, glass) = h.build(1000);
        assert_eq!(glass.rects.len(), 4, "{size:?}");
        let s = px_per_dip(size);
        let windows: Vec<[f32; 4]> = glass.rects[..3].iter().map(|r| r.rect).collect();
        let taskbar = glass.rects[3].rect;
        let left = windows[0][0];
        let right = size.0 - (windows[2][0] + windows[2][2]);
        assert!(
            (left - right).abs() < 1.0,
            "{size:?}: margins {left} vs {right}"
        );
        assert!(left >= SIDE_MARGIN * s - 1.0, "{size:?}: a margin");
        for (w, rect) in windows.iter().enumerate() {
            assert!(
                (rect[2] - WINDOW_WIDTHS[w] * s).abs() < 0.01,
                "{size:?}: width"
            );
        }
        for pair in windows.windows(2) {
            let gap = pair[1][0] - (pair[0][0] + pair[0][2]);
            assert!((gap - WINDOW_GAP * s).abs() < 0.01, "{size:?}: gap {gap}");
        }
        let windows_bottom = windows[0][1] + windows[0][3];
        assert!(
            taskbar[1] > windows_bottom,
            "{size:?}: the taskbar is below"
        );
        assert!(taskbar[1] + taskbar[3] <= size.1, "{size:?}: on screen");
        let centre = taskbar[0] + taskbar[2] * 0.5;
        assert!((centre - size.0 * 0.5).abs() < 1.0, "{size:?}: centred");
        layouts.push(
            windows
                .iter()
                .chain(std::iter::once(&taskbar))
                .map(|r| [r[0] / s, r[1] / s, r[2] / s, r[3] / s])
                .collect::<Vec<_>>(),
        );
        // Readable: a mod's name is at least 10 px even at 720p (it was 9 in seven windows).
        assert!(13.5 * s >= 10.0, "{size:?}: {s}");
    }
    for other in &layouts[1..] {
        for (a, b) in layouts[0].iter().zip(other) {
            for (p, q) in a.iter().zip(b) {
                assert!((p - q).abs() < 0.5, "{a:?} vs {b:?}");
            }
        }
    }
}

#[test]
fn colour_conversions_round_trip() {
    for argb in [
        0xFFFF_FFFF,
        0xFF00_0000,
        0x80FF_5555,
        0xFF12_3456,
        0x0055_FF55,
    ] {
        assert_eq!(hsva_to_argb(argb_to_hsva(argb)), argb, "{argb:08X}");
    }
}

// ---- views, selection and the last place ---------------------------------------------------

/// Every catalog mod is in the menu order exactly once, at home in the first category that
/// lists it; HUD comes first with FPS at its top; All mods is A to Z.
#[test]
fn every_mod_is_in_the_menu_order_once_at_home_in_its_first_category() {
    let catalog = mc_mods::catalog::all();
    let order = MenuOrder::new(catalog);
    let mut seen = vec![0; catalog.len()];
    for &(m, home) in &order.mods {
        seen[m] += 1;
        let first = CATEGORIES
            .iter()
            .position(|category| {
                category
                    .sections
                    .iter()
                    .any(|section| section.mods.contains(&catalog[m].id))
            })
            .map(|c| u8::try_from(c).unwrap());
        assert_eq!(Some(home), first, "{}", catalog[m].id);
    }
    assert!(seen.iter().all(|&n| n == 1), "{seen:?}");
    assert_eq!(catalog[order.mods[0].0].id, "fps");
    let names: Vec<String> = order
        .by_name
        .iter()
        .map(|&m| catalog[m].name.to_ascii_lowercase())
        .collect();
    assert!(names.windows(2).all(|pair| pair[0] <= pair[1]), "A to Z");
}

/// The first open of a session shows My setup, or HUD while My setup is empty, with
/// the list's first mod in window 3; a reopen returns to the same list and mod with the search
/// cleared.
#[test]
fn the_first_open_is_my_setup_or_hud_and_a_reopen_returns_to_the_last_place() {
    let mut h = Harness::new();
    let (fps, zoom, grading) = (
        h.settings.index_of("fps").unwrap(),
        h.settings.index_of("zoom").unwrap(),
        h.settings.index_of("color_grading").unwrap(),
    );
    h.build(1000);
    assert_eq!(
        h.screen.view(),
        Some(category("hud")),
        "nothing changed: HUD"
    );
    assert_eq!(h.screen.selected(), Some(fps), "HUD's first mod");
    assert!(
        h.settings.enabled("dynamic_fov"),
        "vanilla features start on"
    );
    // Something changed: My setup, with it in window 3.
    h.settings.set_enabled("zoom", true);
    h.screen = ModMenuScreen::new(h.settings.catalog(), h.at(2000));
    h.build(3000);
    assert_eq!(h.screen.view(), Some(View::MySetup));
    assert_eq!(h.screen.selected(), Some(zoom));
    // Graphics, Color Grading and a search; the next menu opens there with the search cleared.
    h.click_target(|t| *t == Target::Nav(category("graphics")), 3000);
    h.screen.select(grading);
    h.type_text("fog", 3000);
    let place = h.screen.place();
    assert_eq!(
        place,
        MenuPlace {
            view: category("graphics"),
            selected: Some(grading)
        }
    );
    h.screen = ModMenuScreen::reopen(h.settings.catalog(), h.at(4000), place);
    h.build(4500);
    assert_eq!(h.screen.view(), Some(category("graphics")));
    assert_eq!(h.screen.selected(), Some(grading));
    assert_eq!(h.screen.search_query(), "");
    assert!(
        h.screen.row_rect(grading).is_some(),
        "window 2 shows its list"
    );
}

/// D2: a click on a row shows the mod's settings (it does not toggle it); the switch toggles;
/// a right click pins the mod to My setup, and again unpins it.
#[test]
fn clicking_the_row_selects_the_switch_toggles_and_a_right_click_pins() {
    let mut h = Harness::new();
    h.build(1000);
    let zoom = h.settings.index_of("zoom").unwrap();
    h.click_target(|t| *t == Target::Nav(category("controls")), 1000);
    h.build(1016);
    let row = h.screen.row_rect(zoom).expect("Zoom's row in Controls");
    let (x, y) = (row.x + 40.0, row.y + row.h * 0.5);
    h.click(x, y, 0, 1016);
    assert!(!h.settings.enabled("zoom"), "the row click does not toggle");
    assert_eq!(h.screen.selected(), Some(zoom), "it shows Zoom's settings");
    h.build(1032);
    let switch = h.screen.switch_rect(zoom).expect("its switch");
    h.click(switch.center().0, switch.center().1, 0, 1032);
    assert!(h.settings.enabled("zoom"), "the switch toggles");
    h.settings.set_enabled("zoom", false);
    h.build(1048);
    h.click(x, y, 1, 1048);
    assert!(h.settings.pinned_at(zoom), "a right click pins");
    // My setup lists the pinned mod although nothing in it changed.
    h.build(1064);
    h.click_target(|t| *t == Target::Nav(View::MySetup), 1064);
    h.build(1080);
    let row = h.screen.row_rect(zoom).expect("pinned: in My setup");
    h.click(row.x + 40.0, row.y + row.h * 0.5, 1, 1080);
    assert!(!h.settings.pinned_at(zoom), "and again unpins");
}

#[test]
fn clicking_a_mods_switch_flips_the_setting() {
    let mut h = Harness::new();
    h.build(1000);
    let fps = h.settings.index_of("fps").unwrap();
    let switch = h
        .screen
        .switch_rect(fps)
        .expect("the FPS switch is on screen");
    assert!(!h.settings.enabled("fps"));
    let (x, y) = switch.center();
    h.click(x, y, 0, 1000);
    assert!(h.settings.enabled("fps"), "the switch turned FPS on");
    h.build(1016);
    h.click(x, y, 0, 1016);
    assert!(!h.settings.enabled("fps"), "and off again");
}

/// My setup = mods changed from a fresh install (switch or any setting, not a HUD position)
/// plus pinned ones, each changed row with the dot; the header's Pin adds a mod too.
#[test]
fn my_setup_lists_changed_and_pinned_mods_and_the_header_pin_adds_one() {
    let mut h = Harness::new();
    let id = |h: &Harness, id: &str| h.settings.index_of(id).unwrap();
    let (fps, zoom, fov, cps, ping) = (
        id(&h, "fps"),
        id(&h, "zoom"),
        id(&h, "dynamic_fov"),
        id(&h, "cps"),
        id(&h, "ping"),
    );
    h.settings.set_enabled("fps", true);
    h.settings.set_pinned_at(zoom, true);
    h.settings.set_enabled("dynamic_fov", false);
    let moved = HudPlacement::new(mc_mods::HudAnchor::TopLeft, 30.0, 40.0);
    h.settings.set_placement("cps", "cps", moved);
    h.build(1000);
    assert_eq!(h.screen.view(), Some(View::MySetup));
    for m in [fps, zoom, fov] {
        assert!(h.screen.row_rect(m).is_some(), "{m} is in My setup");
    }
    assert!(
        h.screen.row_rect(cps).is_none(),
        "a moved HUD element is not a change"
    );
    assert!(h.screen.is_changed(fps) && h.screen.is_changed(fov));
    assert!(!h.screen.is_changed(zoom), "pinned, not changed");
    assert_eq!(h.screen.my_setup_len(&h.settings), 3);
    // The header's Pin puts Ping there too; a click again takes it out.
    h.open("ping", 1016);
    h.click_target(|t| *t == Target::Pin(ping), 1016);
    assert!(h.settings.pinned_at(ping));
    h.build(1032);
    assert!(h.screen.row_rect(ping).is_some());
    h.click_target(|t| *t == Target::Pin(ping), 1032);
    assert!(!h.settings.pinned_at(ping));
}

/// While searching, window 1 counts the results per entry and a click narrows them to that
/// entry (All mods: every result); Esc clears the search and window 2 is back on its list.
#[test]
fn window_one_counts_results_and_a_click_narrows_them() {
    let mut h = Harness::new();
    h.build(1000);
    let controls = category("controls");
    h.type_text("zoom", 1000);
    h.build(1016);
    assert!(h.screen.found_count_in(&h.settings, controls) >= 1);
    assert!(h.screen.found_total() >= h.screen.found_count_in(&h.settings, controls));
    h.click_target(|t| *t == Target::Nav(controls), 1016);
    assert_eq!(h.screen.narrow, Some(controls));
    assert_eq!(
        h.screen.view(),
        Some(category("hud")),
        "the list underneath stays"
    );
    h.build(1032);
    let shown: Vec<usize> = h
        .screen
        .hits
        .iter()
        .filter_map(|(_, target)| match target {
            Target::ModRow(m) | Target::Found(m) => Some(*m),
            _ => None,
        })
        .collect();
    assert!(!shown.is_empty());
    for m in shown {
        assert!(
            h.screen.view_lists(&h.settings, controls, m),
            "{m} is listed in Controls"
        );
    }
    h.click_target(|t| *t == Target::Nav(View::AllMods), 1032);
    assert_eq!(h.screen.narrow, None, "All mods: every result");
    assert_eq!(h.key(KEY_ESCAPE, 1048), KeyOutcome::Handled);
    assert!(!h.screen.searching());
    h.build(1064);
    let fps = h.settings.index_of("fps").unwrap();
    assert!(h.screen.row_rect(fps).is_some(), "HUD is back");
}

// ---- window 3: settings ---------------------------------------------------------------------

#[test]
fn choices_cycle_and_keybinds_capture() {
    let mut h = Harness::new();
    let zoom = h.open("zoom", 1000);
    let key_hit = h
        .hit_where(|t| matches!(t, Target::Key(m, _) if *m == zoom))
        .expect("zoom key badge");
    let (x, y) = key_hit.center();
    h.click(x, y, 0, 1000);
    assert!(h.screen.capture.is_some(), "capture armed");
    // LWJGL 44 = Z.
    assert_eq!(h.key(44, 1020), KeyOutcome::Handled);
    assert_eq!(h.settings.key("zoom", "key"), 44);
    assert!(h.screen.capture.is_none());
    // Capture again, Backspace unbinds.
    h.build(1030);
    h.click(x, y, 0, 1030);
    h.key(KEY_BACK, 1031);
    assert_eq!(h.settings.key("zoom", "key"), 0);
    // A choice cycles forward on a click and back on a right click.
    let armor = h.open("armor_status", 1040);
    let layout = h.settings.catalog()[armor].option_index("layout").unwrap();
    let badge = h
        .hit_where(|t| *t == Target::Choice(armor, layout))
        .expect("the Layout badge");
    h.click(badge.center().0, badge.center().1, 0, 1040);
    assert_eq!(h.settings.choice("armor_status", "layout"), 1);
    h.build(1050);
    h.click(badge.center().0, badge.center().1, 1, 1050);
    assert_eq!(h.settings.choice("armor_status", "layout"), 0);
}

/// Item 12: "Reset settings" in window 3's header needs a second click within 3 s, resets the
/// options only, and is greyed out (not clickable) when there is nothing to reset. (FPS's own
/// "Show" setting: its text style follows the shared HUD style by default.)
#[test]
fn reset_settings_needs_two_clicks_and_resets_only_the_options() {
    let mut h = Harness::new();
    let fps = h.open("fps", 1000);
    let reset = |t: &Target| matches!(t, Target::ResetMod(m) if *m == fps);
    assert!(
        h.hit_where(reset).is_none(),
        "greyed out: everything is at its default"
    );
    h.settings.set_enabled("fps", true);
    h.settings.set_choice("fps", "display", 2);
    h.build(1032);
    h.click_target(reset, 1032);
    assert_eq!(h.settings.choice("fps", "display"), 2, "armed only");
    // Too late: the arm has expired, so this click arms again instead of resetting.
    h.build(5000);
    h.click_target(reset, 5000);
    assert_eq!(h.settings.choice("fps", "display"), 2);
    h.build(5100);
    h.click_target(reset, 5100);
    assert!(h.settings.options_are_default_at(fps), "reset");
    assert!(h.settings.enabled("fps"), "the switch stays on");
    h.build(5200);
    assert!(h.hit_where(reset).is_none(), "greyed out again");
}

/// A changed setting gets the dot; hovering its row turns the dot into a reset arrow
/// that resets just that setting, and a right click on its label does the same.
#[test]
fn a_changed_setting_resets_from_its_arrow_or_a_right_click_on_its_label() {
    let mut h = Harness::new();
    let fps = h.open("fps", 1000);
    let show = h.settings.catalog()[fps].option_index("display").unwrap();
    h.settings.set_choice("fps", "display", 2);
    h.build(1016);
    let reset = |t: &Target| *t == Target::ResetOption(fps, show);
    assert!(
        h.hit_where(reset).is_none(),
        "a dot until the row is hovered"
    );
    let label = h
        .hit_where(|t| *t == Target::Label(fps, show))
        .expect("the label");
    h.screen.cursor = label.center();
    h.build(1032);
    h.click_target(reset, 1032);
    assert_eq!(h.settings.choice("fps", "display"), 0, "the arrow");
    assert!(h.settings.options_are_default_at(fps));
    h.settings.set_choice("fps", "display", 2);
    h.build(1048);
    let label = h.hit_where(|t| *t == Target::Label(fps, show)).unwrap();
    h.click(label.center().0, label.center().1, 1, 1048);
    assert_eq!(h.settings.choice("fps", "display"), 0, "a right click");
}

/// Open every block that starts collapsed, as a click on its heading does, and the folded
/// shared-style rows.
fn open_every_group(h: &mut Harness, m: usize) {
    let layout = menu_rules::settings_layout(h.settings.catalog()[m]);
    for (b, block) in layout.blocks.iter().enumerate() {
        if block.collapsed {
            h.screen.toggled_groups.push((m, u8::try_from(b).unwrap()));
        }
    }
    h.screen.extras.style_open = true;
}

/// Window 3 draws every setting the layout places (MB's tests check that the layout holds
/// every visible setting once), unless a rule hides it right now: nothing is silently dropped.
/// Keys go under KEYS and the shared HUD text style under STYLE.
#[test]
fn window_three_draws_every_setting_the_layout_places() {
    let mut h = Harness::new();
    for (m, descriptor) in mc_mods::catalog::all().iter().enumerate() {
        open_every_group(&mut h, m);
        h.open(descriptor.id, 1000);
        let drawn: Vec<usize> = h.screen.nav[Pane::Details.index()]
            .iter()
            .filter_map(|row| nav_option(row.enter))
            .filter(|&(mm, _)| mm == m)
            .map(|(_, o)| o)
            .collect();
        for o in menu_rules::settings_layout(descriptor).options() {
            let hidden = matches!(
                menu_rules::option_state(&h.settings, m, o),
                OptionState::Hidden(_)
            );
            assert!(
                hidden || drawn.contains(&o),
                "{}.{} is not drawn",
                descriptor.id,
                descriptor.options[o].id
            );
        }
    }
    let heading_of = |id: &str, option: &str| {
        let descriptor = mc_mods::catalog::find(id).unwrap();
        let o = descriptor.option_index(option).unwrap();
        let layout = menu_rules::settings_layout(descriptor);
        layout.blocks[layout.block_of(o).unwrap()].heading
    };
    assert_eq!(heading_of("zoom", "key"), Heading::Keys);
    assert_eq!(heading_of("fps", "text_shadow"), Heading::Style);
}

/// Every hide/dim rule in `menu_rules::RULES` (the data): with its condition unmet, the
/// setting is in window 3 exactly when the `option_state` does not hide it (a hide rule
/// wins over a dim rule on the same setting), and the rule itself is in effect.
#[test]
fn rules_hide_or_dim_settings_whose_condition_is_unmet() {
    for rule in menu_rules::RULES {
        let mut h = Harness::new();
        let m = h.settings.index_of(rule.mod_id).expect(rule.mod_id);
        let descriptor = h.settings.catalog()[m];
        let o = descriptor.option_index(rule.option).expect(rule.option);
        match rule.when {
            Cond::On(toggle) => {
                h.settings.set_bool(rule.mod_id, toggle, false);
            }
            Cond::Off(toggle) => {
                h.settings.set_bool(rule.mod_id, toggle, true);
            }
            Cond::ChoiceIn(choice, allowed) => {
                let OptionKind::Choice { choices } = descriptor.option(choice).expect(choice).kind
                else {
                    panic!("{}.{choice} is not a choice", rule.mod_id);
                };
                let other = (0..choices.len())
                    .find(|i| !allowed.contains(i))
                    .expect("a choice outside the rule");
                h.settings.set_choice(rule.mod_id, choice, other);
            }
            Cond::ModOn => {
                h.settings.set_enabled(rule.mod_id, false);
            }
        }
        // Open the groups that start collapsed, so every setting that shows is laid out.
        open_every_group(&mut h, m);
        h.open(rule.mod_id, 1000);
        let shown = h.screen.nav[Pane::Details.index()]
            .iter()
            .any(|row| nav_option(row.enter).is_some_and(|(_, opt)| opt == o));
        let state = menu_rules::option_state(&h.settings, m, o);
        assert_ne!(
            state,
            OptionState::Shown,
            "{}.{}: the rule is in effect",
            rule.mod_id,
            rule.option
        );
        if rule.unmet == Unmet::Hide {
            assert!(matches!(state, OptionState::Hidden(_)));
        }
        assert_eq!(
            shown,
            !matches!(state, OptionState::Hidden(_)),
            "{}.{}: window 3 follows the rules ({state:?})",
            rule.mod_id,
            rule.option
        );
    }
}

/// The special layouts: a tabbed picker draws one tab per choice and a click picks it; a
/// paired row has its key, then its text, on one row.
#[test]
fn tabbed_pickers_and_paired_rows_follow_the_tables() {
    for tabbed in menu_rules::TABBED {
        let mut h = Harness::new();
        let m = h.open(tabbed.mod_id, 1000);
        let picker = h.settings.catalog()[m].option_index(tabbed.picker).unwrap();
        let OptionKind::Choice { choices } = h.settings.catalog()[m].options[picker].kind else {
            panic!("a tabbed picker is a choice");
        };
        for i in 0..choices.len() {
            assert!(
                h.hit_where(|t| *t == Target::Tab(m, picker, i)).is_some(),
                "{} tab {i}",
                tabbed.mod_id
            );
        }
        let last = choices.len() - 1;
        h.click_target(|t| *t == Target::Tab(m, picker, last), 1000);
        assert_eq!(
            h.settings.value_at(m, picker),
            Some(&OptionValue::Choice(last))
        );
    }
    for paired in menu_rules::PAIRED {
        let mut h = Harness::new();
        let m = h.open(paired.mod_id, 1000);
        let descriptor = h.settings.catalog()[m];
        let rows = &h.screen.nav[Pane::Details.index()];
        for (left, right) in paired.pairs {
            let (l, r) = (
                descriptor.option_index(left).unwrap(),
                descriptor.option_index(right).unwrap(),
            );
            let at = rows
                .iter()
                .position(|row| row.enter == Target::Key(m, l))
                .expect("the key");
            assert_eq!(rows[at + 1].enter, Target::Text(m, r), "then its text");
            assert_eq!(rows[at].span, rows[at + 1].span, "on one row");
        }
    }
}

/// Items 12/43: colour grading profiles in window 3: type a name, Enter saves; Load puts the
/// values back; Delete needs a second click.
#[test]
fn colour_grading_profiles_save_load_and_delete_from_the_menu() {
    let mut h = Harness::new();
    let grading = h.open("color_grading", 1000);
    h.settings.set_f32("color_grading", "saturation", 1.3);
    settle(&mut h, 1016);
    assert!(
        h.hit_where(|t| matches!(t, Target::ProfileSave(m) if *m == grading))
            .is_some(),
        "the profiles block is drawn for colour grading"
    );
    assert!(
        h.hit_where(|t| matches!(t, Target::ResetMod(m) if *m == grading))
            .is_some(),
        "the header's reset (a slider moved)"
    );
    h.click_target(
        |t| matches!(t, Target::ProfileName(m) if *m == grading),
        1500,
    );
    assert!(h.screen.wants_key_repeat());
    h.type_text("Warm", 1500);
    assert_eq!(
        h.screen.search_query(),
        "",
        "the name field took the typing"
    );
    assert_eq!(h.key(KEY_RETURN, 1500), KeyOutcome::Handled);
    let profiles = h.settings.profiles_at(grading);
    assert_eq!(profiles.len(), 1);
    assert_eq!(profiles[0].name(), "Warm");
    assert!(h.settings.profile_is_current_at(grading, 0));
    // Save without a name: "Profile 2".
    h.settings.set_f32("color_grading", "saturation", 0.5);
    h.build(1516);
    h.click_target(
        |t| matches!(t, Target::ProfileSave(m) if *m == grading),
        1516,
    );
    assert_eq!(h.settings.profiles_at(grading)[1].name(), "Profile 2");
    // Load "Warm".
    h.build(1532);
    h.click_target(|t| *t == Target::ProfileLoad(grading, 0), 1532);
    assert_eq!(h.settings.f32("color_grading", "saturation"), 1.3);
    // Delete it: armed first.
    h.build(1548);
    h.click_target(|t| *t == Target::ProfileDelete(grading, 0), 1548);
    assert_eq!(h.settings.profiles_at(grading).len(), 2, "armed only");
    h.build(1564);
    h.click_target(|t| *t == Target::ProfileDelete(grading, 0), 1564);
    assert_eq!(h.settings.profiles_at(grading).len(), 1);
    assert_eq!(h.settings.profiles_at(grading)[0].name(), "Profile 2");
    // Another mod has no profiles block.
    h.open("zoom", 1580);
    assert!(h
        .hit_where(|t| matches!(t, Target::ProfileSave(_)))
        .is_none());
}

/// Item 33: Right/Left step the last-dragged (or hovered) slider exactly one step, Shift ten,
/// so every value of a 0..2 slider with 0.01 steps is reachable.
#[test]
fn arrow_keys_step_the_last_dragged_slider_through_every_value() {
    let mut h = Harness::new();
    let grading = h.open("color_grading", 1000);
    let saturation = h.settings.catalog()[grading]
        .option_index("saturation")
        .unwrap();
    settle(&mut h, 1016);
    let Some(Target::Slider(_, _, track)) = h
        .screen
        .hits
        .iter()
        .map(|(_, t)| *t)
        .find(|t| matches!(t, Target::Slider(m, o, _) if *m == grading && *o == saturation))
    else {
        panic!("the saturation slider is on screen");
    };
    // Press at the far left: 0.00, then step right through every value.
    h.click(track.x - 2.0, track.y + track.h * 0.5, 0, 1016);
    assert_eq!(h.settings.f32("color_grading", "saturation"), 0.0);
    // Move the pointer away: the last-dragged slider still takes the arrows.
    h.screen.cursor = (1.0, 1.0);
    for k in 1..=200 {
        assert_eq!(h.key(KEY_RIGHT, 1020), KeyOutcome::Handled);
        let v = h.settings.f32("color_grading", "saturation");
        assert!((v - k as f32 * 0.01).abs() < 1e-5, "step {k}: {v}");
    }
    h.key(KEY_RIGHT, 1020);
    assert_eq!(
        h.settings.f32("color_grading", "saturation"),
        2.0,
        "clamped at the top"
    );
    h.shift = true;
    h.key(KEY_LEFT, 1020);
    assert!((h.settings.f32("color_grading", "saturation") - 1.9).abs() < 1e-5);
    h.shift = false;
    h.key(KEY_LEFT, 1020);
    assert!((h.settings.f32("color_grading", "saturation") - 1.89).abs() < 1e-5);
    // The pointer itself resolves physical pixels: a press half-way lands half-way.
    h.click(track.x + track.w * 0.5, track.y + track.h * 0.5, 0, 1030);
    assert!((h.settings.f32("color_grading", "saturation") - 1.0).abs() < 0.011);
}

/// Window 2 and window 3 scroll with the wheel when their content is taller than they are.
#[test]
fn windows_two_and_three_scroll() {
    let mut h = Harness::new();
    h.build(1000);
    h.click_target(|t| *t == Target::Nav(View::AllMods), 1000);
    // Window 3 with a long list of dynamic rows (waypoints).
    let waypoints = h.settings.index_of("waypoints").unwrap();
    let rows = &mut h.rows[waypoints];
    rows.begin();
    for key in 0..40 {
        rows.push(key).label.push_str("Waypoint");
    }
    h.screen.select(waypoints);
    h.build(1016);
    h.build(1032);
    for pane in [Pane::List, Pane::Details] {
        let w = pane.index();
        assert!(h.screen.max_scroll[w] > 0.0, "window {w} overflows");
        let body = h
            .hit_where(|t| *t == Target::Window(w))
            .expect("the window body");
        let (x, y) = body.center();
        let mut screen = std::mem::replace(&mut h.screen, ModMenuScreen::new(&[], h.t0));
        {
            let mut input = h.input(1032);
            screen.scrolled(x, y, -3.0, &mut input);
        }
        h.screen = screen;
    }
    for ms in (1048..=1600).step_by(16) {
        h.build(ms);
    }
    assert!(h.screen.scroll[1] > 1.0, "window 2: {}", h.screen.scroll[1]);
    assert!(h.screen.scroll[2] > 1.0, "window 3: {}", h.screen.scroll[2]);
}

/// Requested: "Having the Color grading tab open should undim the background so the person can
/// see the raw color they are configuring". The dim eases out while Color Grading is in window
/// 3, comes back for another mod, and never flashes back while the menu closes.
#[test]
fn color_grading_open_fades_the_dim_out_and_another_mod_fades_it_back() {
    let mut h = Harness::new();
    let (_, open) = h.build(1000);
    assert!((open.dim - MENU_DIM).abs() < 1e-4);
    let grading = h.settings.index_of("color_grading").unwrap();
    let blur = h.settings.index_of("motion_blur").unwrap();
    h.click_target(|t| *t == Target::Nav(category("graphics")), 1000);
    h.build(1000);
    let row = h
        .screen
        .row_rect(grading)
        .expect("Color Grading in Graphics");
    h.click(row.x + 40.0, row.y + row.h * 0.5, 0, 1000);
    assert_eq!(h.screen.selected(), Some(grading));

    // Part way after one 60 Hz frame, then monotone down to none within 300 ms.
    let (_, first) = h.build(1016);
    assert!(first.dim > 0.0 && first.dim < open.dim, "{}", first.dim);
    let mut last = first.dim;
    for ms in (1032..=1300).step_by(16) {
        let (_, frame) = h.build(ms);
        assert!(frame.dim <= last, "monotone: {} after {last}", frame.dim);
        last = frame.dim;
    }
    assert_eq!(last, 0.0, "fully undimmed");
    let (_, frame) = h.build(1316);
    assert_eq!(frame.rects.len(), 4, "the windows and the taskbar stay");
    assert_eq!(frame.strength, 1.0);

    // Another mod: the dim comes back.
    let row = h.screen.row_rect(blur).expect("Motion Blur in Graphics");
    h.click(row.x + 40.0, row.y + row.h * 0.5, 0, 1316);
    let (_, partial) = h.build(1332);
    assert!(
        partial.dim > 0.0 && partial.dim < MENU_DIM,
        "{}",
        partial.dim
    );
    for ms in (1348..=1700).step_by(16) {
        h.build(ms);
    }
    let (_, back) = h.build(1716);
    assert!((back.dim - MENU_DIM).abs() < 1e-4, "{}", back.dim);

    // Back to Color Grading, then closing: the dim never comes back.
    let row = h.screen.row_rect(grading).unwrap();
    h.click(row.x + 40.0, row.y + row.h * 0.5, 0, 2100);
    for ms in (2116..=2500).step_by(16) {
        h.build(ms);
    }
    h.screen.begin_close(h.at(2500));
    for ms in (2500..=2700).step_by(8) {
        let (_, frame) = h.build(ms);
        assert_eq!(frame.dim, 0.0, "closing, {ms} ms");
    }
    // A new menu (first open) starts with the normal dim.
    h.screen = ModMenuScreen::new(h.settings.catalog(), h.at(3000));
    let (_, reopened) = h.build(3400);
    assert!((reopened.dim - MENU_DIM).abs() < 1e-4);
}

// ---- the search -------------------------------------------------------------------------------

/// Item 45: typing goes to the search; window 2 lists the results instead of its list; the
/// windows stay; Backspace edits; Esc clears the search first and closes only after that.
#[test]
fn typing_searches_lists_results_and_esc_clears_before_closing() {
    let mut h = Harness::new();
    h.build(1000);
    let (zoom, fps) = (
        h.settings.index_of("zoom").unwrap(),
        h.settings.index_of("fps").unwrap(),
    );
    assert!(h.screen.row_rect(fps).is_some());
    assert!(!h.screen.searching());
    h.type_text("ZOO", 1000);
    assert_eq!(h.screen.search_query(), "ZOO");
    assert!(h.screen.searching());
    assert!(h.screen.wants_key_repeat(), "held Backspace repeats");
    let (_, glass) = h.build(1016);
    assert_eq!(glass.rects.len(), 4, "every window and the taskbar stay");
    assert!(
        h.screen.row_rect(zoom).is_some(),
        "Zoom is a result (any case)"
    );
    assert!(h.screen.row_rect(fps).is_none(), "FPS is not");
    assert!(h.screen.search_found >= 1);
    // Backspace edits the search; a leading space is ignored.
    assert_eq!(h.key(KEY_BACK, 1016), KeyOutcome::Handled);
    assert_eq!(h.screen.search_query(), "ZO");
    h.screen.search.clear();
    h.type_text(" ", 1016);
    assert_eq!(h.screen.search_query(), "");
    // Nothing matches: no rows, the windows still draw.
    h.type_text("qqqxx", 1016);
    let (_, glass) = h.build(1032);
    assert_eq!(glass.rects.len(), 4);
    assert_eq!(h.screen.search_found, 0);
    assert!(h.screen.row_rect(zoom).is_none() && h.screen.row_rect(fps).is_none());
    // Esc clears, then closes.
    assert_eq!(h.key(KEY_ESCAPE, 1032), KeyOutcome::Handled);
    assert!(!h.screen.searching());
    h.build(1048);
    assert!(h.screen.row_rect(fps).is_some(), "the list is back");
    assert_eq!(h.key(KEY_ESCAPE, 1048), KeyOutcome::Close);
}

/// A search for a setting lists it under its mod in window 2's SETTINGS; a click opens the mod
/// in window 3 with that setting flashed, where it works like any other ("zoom key").
#[test]
fn a_settings_result_opens_its_mod_at_the_setting() {
    let mut h = Harness::new();
    h.build(1000);
    let zoom = h.settings.index_of("zoom").unwrap();
    let key = h.settings.catalog()[zoom].option_index("key").unwrap();
    h.type_text("zoom key", 1000);
    h.build(1016);
    assert!(
        h.hit_where(|t| *t == Target::Found(zoom)).is_some(),
        "Zoom › Zoom key is a settings result"
    );
    h.click_target(|t| *t == Target::Found(zoom), 1016);
    assert_eq!(h.screen.selected(), Some(zoom));
    assert!(h.screen.flash_options.contains(&key), "the key is flashed");
    h.build(1032);
    // The badge works like any other: click, press a key.
    h.click_target(|t| *t == Target::Key(zoom, key), 1032);
    assert!(h.screen.capture.is_some());
    assert_eq!(h.key(44, 1040), KeyOutcome::Handled);
    assert_eq!(h.settings.key("zoom", "key"), 44);
}

/// The search also looks at choice labels and at dynamic rows (a waypoint's name), and a
/// dynamic row found that way works in window 3.
#[test]
fn a_search_matches_choice_labels_and_dynamic_row_labels() {
    let mut h = Harness::new();
    let armor = h.settings.index_of("armor_status").unwrap();
    let waypoints = h.settings.index_of("waypoints").unwrap();
    let rows = &mut h.rows[waypoints];
    rows.begin();
    let row = rows.push(7);
    row.label.push_str("Spawn point");
    row.buttons = [Some(RowButton::Icon(RowIcon::Delete)), None];
    rows.push(8).label.push_str("Diamonds");
    h.build(1000);
    h.type_text("horizontal", 1000);
    h.build(1016);
    assert!(
        h.hit_where(|t| *t == Target::Found(armor)).is_some(),
        "Layout: Horizontal"
    );
    h.screen.search.clear();
    h.type_text("spawn", 1016);
    h.build(1032);
    h.click_target(|t| *t == Target::Found(waypoints), 1032);
    h.build(1048);
    h.click_target(
        |t| matches!(t, Target::DynAction(m, 7, 0) if *m == waypoints),
        1048,
    );
    assert_eq!(
        h.events,
        [ModEvent::RowAction {
            mod_id: "waypoints",
            row: 7,
            action: 0
        }]
    );
}

/// A dynamic row's badge with `cycle` set is a choice (the title screen's Game version row): a
/// left click asks for the next value, a right click for the one before. Without `cycle` (that
/// row during a session, greyed) the badge takes no click.
#[test]
fn a_cycle_badge_asks_for_the_next_value_and_a_fixed_badge_takes_no_click() {
    let fill = |rows: &mut DynamicRows, choice: bool| {
        rows.begin();
        let row = rows.push(1);
        row.label.push_str("Game version");
        row.badge.push_str("1.8.9");
        row.cycle = choice;
        row.muted = !choice;
    };
    let mut h = Harness::new();
    let title = h.settings.index_of("aero_title").unwrap();
    fill(&mut h.rows[title], true);
    h.build(1000);
    h.type_text("game version", 1000);
    h.build(1016);
    h.click_target(|t| *t == Target::Found(title), 1016);
    h.build(1032);
    let badge = |t: &Target| matches!(t, Target::DynCycle(m, 1) if *m == title);
    let (x, y) = h.hit_where(badge).expect("the badge is a choice").center();
    h.click(x, y, 0, 1032);
    h.click(x, y, 1, 1048);
    let cycle = |forward| ModEvent::RowCycle {
        mod_id: "aero_title",
        row: 1,
        forward,
    };
    assert_eq!(h.events, [cycle(true), cycle(false)]);

    fill(&mut h.rows[title], false);
    h.build(1064);
    assert!(h.hit_where(badge).is_none(), "a fixed badge is no choice");
    h.click(x, y, 0, 1064);
    assert_eq!(h.events.len(), 2, "the click asked for nothing");
}

/// A printable menu key (M) closes the menu, except while a search is being typed: then it
/// types (searching "zoom" with M as the menu key must work).
#[test]
fn a_printable_menu_key_types_into_a_search_in_progress() {
    let mut h = Harness::new();
    h.settings.set_menu_key(50);
    h.build(1000);
    h.type_text("zoo", 1000);
    assert_eq!(h.key_char(50, Some('m'), 1000), KeyOutcome::Handled);
    assert_eq!(h.screen.search_query(), "zoom");
    h.key(KEY_ESCAPE, 1000);
    assert_eq!(h.key_char(50, Some('m'), 1000), KeyOutcome::Close);
}

#[test]
fn search_clipboard_selects_copies_and_replaces_with_filtered_paste() {
    let mut h = Harness::new();
    let mut clipboard = crate::widget::keys::MemoryClipboard::default();
    clipboard.set("keep existing clipboard");
    h.type_text("zoom", 1000);
    assert!(h.clipboard_key(keys::C, Modifiers::CTRL, &mut clipboard));
    assert_eq!(clipboard.get(), "keep existing clipboard", "no selection");
    assert!(h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    assert_eq!(h.screen.search_query(), "zoom", "Ctrl+A does not type A");
    let (selected, _) = h.build(1000);
    assert!(
        selected
            .vertices()
            .iter()
            .any(|v| { v.color[..3] == css(70, 130, 230, 0.65)[..3] && v.color[3] > 0.0 }),
        "the selected query has a visible highlight"
    );
    assert!(h.clipboard_key(keys::C, Modifiers::CTRL, &mut clipboard));
    assert_eq!(clipboard.get(), "zoom");
    clipboard.set("  capes\r\nphysics\té🐱\0\n");
    assert!(h.clipboard_key(keys::V, Modifiers::CTRL, &mut clipboard));
    assert_eq!(h.screen.search_query(), "capes  physics é🐱");
    assert!(!h.screen.search_selected);
    // Pasting again without a selection retains the menu's append behavior.
    clipboard.set("!");
    assert!(h.clipboard_key(keys::V, Modifiers::CTRL, &mut clipboard));
    assert_eq!(h.screen.search_query(), "capes  physics é🐱!");
}

#[test]
fn search_clipboard_replacement_preserves_unicode_limits_and_empty_reads() {
    let mut h = Harness::new();
    let mut clipboard = crate::widget::keys::MemoryClipboard::default();
    h.type_text(&"x".repeat(SEARCH_MAX_CHARS), 1000);
    assert!(h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    // A busy/unavailable clipboard reads empty. It must neither erase nor deselect the query.
    assert!(h.clipboard_key(keys::V, Modifiers::CTRL, &mut clipboard));
    assert_eq!(h.screen.search_query(), "x".repeat(SEARCH_MAX_CHARS));
    assert!(h.screen.search_selected);
    clipboard.set(&"🐱".repeat(SEARCH_MAX_CHARS + 2));
    assert!(h.clipboard_key(keys::V, Modifiers::CTRL, &mut clipboard));
    assert_eq!(h.screen.search_query(), "🐱".repeat(SEARCH_MAX_CHARS));
    h.key(KEY_BACK, 1000);
    assert_eq!(h.screen.search_query(), "🐱".repeat(SEARCH_MAX_CHARS - 1));
    assert!(h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    h.type_text("é", 1000);
    assert_eq!(h.screen.search_query(), "é", "typing replaces selection");
    assert!(h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    h.key(KEY_BACK, 1000);
    assert_eq!(h.screen.search_query(), "", "Backspace deletes selection");
    h.type_text("zoom", 1000);
    assert!(h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    h.key(keys::DELETE, 1000);
    assert_eq!(h.screen.search_query(), "", "Delete deletes selection");
    h.type_text(" ", 1000);
    assert_eq!(
        h.screen.search_query(),
        "",
        "leading-space rule is retained"
    );
}

#[test]
fn search_clipboard_shortcuts_respect_focus_captures_and_modifiers() {
    let mut h = Harness::new();
    let mut clipboard = crate::widget::keys::MemoryClipboard::default();
    clipboard.set("untouched");
    h.type_text("zoom", 1000);
    for modifiers in [
        Modifiers::NONE,
        Modifiers::SHIFT,
        Modifiers {
            ctrl: true,
            shift: false,
            alt: true,
        },
    ] {
        for key in [keys::A, keys::C, keys::V] {
            assert!(!h.clipboard_key(key, modifiers, &mut clipboard));
        }
    }
    // Option/profile text fields retain priority; A/C cannot select or copy the hidden query.
    h.screen.text_focus = Some((0, 0));
    assert!(!h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    assert!(!h.clipboard_key(keys::C, Modifiers::CTRL, &mut clipboard));
    h.screen.text_focus = None;
    h.screen.name_focus = Some(0);
    assert!(!h.clipboard_key(keys::A, Modifiers::CTRL, &mut clipboard));
    assert!(!h.clipboard_key(keys::C, Modifiers::CTRL, &mut clipboard));
    h.screen.name_focus = None;
    for mode in 0..4 {
        h.screen.capture = (mode == 0).then_some((0, 0));
        h.screen.menu_key_capture = mode == 1;
        h.screen.mode = if mode == 2 {
            Mode::HudEditor
        } else {
            Mode::Menu
        };
        h.screen.closing_since = (mode == 3).then_some(h.t0);
        for key in [keys::A, keys::C, keys::V] {
            assert!(!h.clipboard_key(key, Modifiers::CTRL, &mut clipboard));
        }
    }
    assert_eq!(h.screen.search_query(), "zoom");
    assert_eq!(clipboard.get(), "untouched");
}

/// For X2's chat macros: Ctrl+V pastes into the focused text option (line breaks become
/// spaces, the option's own limit applies), into the profile name, or else into the search; a
/// key capture takes Ctrl+V as a key instead.
#[test]
fn paste_goes_to_the_focused_field_or_else_the_search() {
    let mut h = Harness::new();
    let macros = h.settings.index_of("chat_macros").unwrap();
    let text_option = h.settings.catalog()[macros]
        .options
        .iter()
        .position(|o| matches!(o.kind, OptionKind::Text { .. }))
        .expect("a macro text option");
    h.build(1000);
    assert!(h.screen.takes_paste());
    assert!(paste(&mut h, "zoom\r\n", 1000));
    assert_eq!(
        h.screen.search_query(),
        "zoom",
        "nothing focused: the search"
    );
    h.screen.search.clear();

    h.screen.select(macros);
    settle(&mut h, 1016);
    h.click_target(|t| *t == Target::Text(macros, text_option), 1500);
    let long = "/msg friend gg\n".repeat(10);
    assert!(paste(&mut h, &long, 1500));
    let Some(OptionValue::Text(stored)) = h.settings.value_at(macros, text_option).cloned() else {
        panic!("a text value");
    };
    assert!(
        stored.starts_with("/msg friend gg /msg friend gg"),
        "{stored}"
    );
    assert!(
        !stored.contains('\n') && !stored.ends_with(' '),
        "{stored:?}"
    );
    assert_eq!(
        h.screen.search_query(),
        "",
        "the field took it, not the search"
    );
    // The field draws the end of a text longer than itself; a frame with it builds fine.
    h.build(1520);

    // A key capture takes Ctrl+V as a key: paste does nothing.
    h.screen.capture = Some((macros, 0));
    assert!(!h.screen.takes_paste());
    assert!(!paste(&mut h, "x", 1530));
}

// ---- the keyboard --------------------------------------------------------------

/// ↓ moves through window 2 and window 3 follows; Space toggles the mod; Enter opens it in
/// window 3, where ←/→ switch it and Esc returns to the list before it closes the menu.
#[test]
fn the_keyboard_walks_the_list_toggles_opens_and_steps_back() {
    let mut h = Harness::new();
    // The shared HUD style folds a readout's style rows; unfolded, CPS shows toggles.
    h.screen.extras.style_open = true;
    h.build(1000);
    let (fps, cps) = (
        h.settings.index_of("fps").unwrap(),
        h.settings.index_of("cps").unwrap(),
    );
    assert_eq!(h.screen.pane, Pane::List);
    assert_eq!(h.key(KEY_DOWN, 1000), KeyOutcome::Handled);
    assert_eq!(h.screen.selected(), Some(fps), "lands on the first row");
    h.build(1016);
    h.key(KEY_DOWN, 1016);
    assert_eq!(h.screen.selected(), Some(cps), "window 3 follows");
    h.build(1032);
    h.key(KEY_SPACE, 1032);
    assert!(h.settings.enabled("cps"), "Space toggles the row's mod");
    h.key(KEY_RETURN, 1032);
    assert_eq!(h.screen.pane, Pane::Details, "Enter opens it in window 3");
    h.build(1048);
    // Its switch has the keyboard: ← turns it off, → on.
    h.key(KEY_LEFT, 1048);
    assert!(!h.settings.enabled("cps"));
    h.key(KEY_RIGHT, 1048);
    assert!(h.settings.enabled("cps"));
    // Down to a toggle: Space flips it, ←/→ set it.
    let toggle = h.screen.nav[Pane::Details.index()]
        .iter()
        .position(|row| matches!(row.enter, Target::Toggle(..)))
        .expect("CPS has a toggle");
    let Target::Toggle(m, o) = h.screen.nav[Pane::Details.index()][toggle].enter else {
        unreachable!();
    };
    while h.screen.cursors[Pane::Details.index()] != Some(toggle) {
        h.key(KEY_DOWN, 1048);
        h.build(1048);
    }
    let before = h.settings.value_at(m, o).cloned();
    h.key(KEY_SPACE, 1048);
    assert_ne!(h.settings.value_at(m, o).cloned(), before, "Space flips it");
    h.key(KEY_RIGHT, 1048);
    assert_eq!(h.settings.value_at(m, o), Some(&OptionValue::Bool(true)));
    h.key(KEY_LEFT, 1048);
    assert_eq!(h.settings.value_at(m, o), Some(&OptionValue::Bool(false)));
    // Esc steps back to the list, then closes.
    assert_eq!(h.key(KEY_ESCAPE, 1048), KeyOutcome::Handled);
    assert_eq!(h.screen.pane, Pane::List);
    assert_eq!(h.key(KEY_ESCAPE, 1048), KeyOutcome::Close);
}

/// Tab moves the keyboard between the windows (Shift+Tab back); in window 1 ↓ opens the next
/// entry's list.
#[test]
fn tab_moves_between_the_windows_and_window_one_follows_the_arrows() {
    let mut h = Harness::new();
    h.build(1000);
    h.key(KEY_TAB, 1000);
    assert_eq!(h.screen.pane, Pane::Details);
    h.key(KEY_TAB, 1000);
    assert_eq!(h.screen.pane, Pane::Sidebar);
    h.build(1016);
    let hud = category("hud");
    let row = h.screen.cursors[Pane::Sidebar.index()].expect("on an entry");
    assert_eq!(
        h.screen.nav[Pane::Sidebar.index()][row].enter,
        Target::Nav(hud),
        "on the open list"
    );
    h.key(KEY_DOWN, 1016);
    assert_eq!(
        h.screen.view(),
        Some(category("pvp")),
        "the next entry opens"
    );
    assert_eq!(
        h.screen.pane,
        Pane::Sidebar,
        "the keyboard stays in window 1"
    );
    h.shift = true;
    h.key(KEY_TAB, 1016);
    assert_eq!(h.screen.pane, Pane::Details);
}

// ---- no world, the HUD editor ---------------------------------------------------------------

/// The title screen (no world): Edit HUD is greyed out and does nothing; everything else
/// works.
#[test]
fn without_a_world_edit_hud_is_greyed_out() {
    let mut h = Harness::new();
    h.no_world = true;
    let (_, glass) = h.build(1000);
    assert_eq!(glass.rects.len(), 4, "the pill's glass still shows");
    assert!(h.screen.edit_hud_rect().is_none(), "no click target");
    let [x, y, w, hh] = glass.rects[3].rect;
    h.click(x + w * 0.5, y + hh * 0.5, 0, 1000);
    assert!(!h.screen.hud_editing());
    let fps = h.settings.index_of("fps").unwrap();
    let switch = h.screen.switch_rect(fps).unwrap();
    h.click(switch.center().0, switch.center().1, 0, 1000);
    assert!(h.settings.enabled("fps"), "the rest works");
}

#[test]
fn the_hud_editor_opens_from_the_pill_and_esc_returns() {
    let mut h = Harness::new();
    h.build(1000);
    let pill = h.screen.edit_hud_rect().expect("Edit HUD pill");
    let (x, y) = pill.center();
    h.click(x, y, 0, 1000);
    assert!(h.screen.hud_editing());
    let (_, glass) = h.build(1300);
    assert!(glass.rects.is_empty(), "windows hidden in the editor");
    assert_eq!(glass.dim, 0.0);
    assert_eq!(h.key(KEY_ESCAPE, 1300), KeyOutcome::Handled);
    assert!(!h.screen.hud_editing());
    // Back in the menu the windows play their opening animation again.
    let (_, glass) = h.build(1300);
    assert!(glass.dim < 0.01, "re-opening from zero: {}", glass.dim);
    // The menu key closes the whole menu straight from the editor (Tab does not).
    h.build(2000);
    let pill = h.screen.edit_hud_rect().expect("Edit HUD pill");
    h.click(pill.center().0, pill.center().1, 0, 2000);
    assert!(h.screen.hud_editing());
    assert!(
        h.screen.wants_key_repeat(),
        "held arrows nudge in the editor"
    );
    assert_eq!(h.key(KEY_TAB, 2000), KeyOutcome::Handled);
    assert!(h.screen.hud_editing());
    assert_eq!(h.key(mc_mods::DEFAULT_MENU_KEY, 2000), KeyOutcome::Close);
}

/// Item 11: the menu key is rebound from the Keybinds view: click, then the new key;
/// Backspace restores Right Shift (never unbound), Esc and clicks cancel, and the new key
/// closes the menu while the old one does not.
#[test]
fn the_menu_key_rebinds_from_keybinds_and_can_never_be_unbound() {
    let mut h = Harness::new();
    h.build(1000);
    h.click_target(|t| *t == Target::Nav(View::Keybinds), 1000);
    // The key's badge is sized to its name, so it is looked up again after every frame.
    let click_key = |h: &mut Harness, ms: u64| {
        h.build(ms);
        let pill = h
            .screen
            .menu_key_rect()
            .expect("the menu key is in the Keybinds view");
        h.click(pill.center().0, pill.center().1, 0, ms);
    };
    assert_eq!(h.settings.menu_key(), 54);
    click_key(&mut h, 1016);
    assert!(h.screen.capturing_menu_key());
    // LWJGL 50 = M.
    assert_eq!(h.key_char(50, Some('m'), 1020), KeyOutcome::Handled);
    assert!(!h.screen.capturing_menu_key());
    assert_eq!(h.settings.menu_key(), 50);
    assert_eq!(h.screen.search_query(), "", "the captured key did not type");
    assert_eq!(
        h.key(54, 1020),
        KeyOutcome::Handled,
        "Right Shift is free now"
    );
    // Backspace restores Right Shift rather than unbinding; Esc cancels; a click cancels.
    click_key(&mut h, 1030);
    assert!(h.screen.capturing_menu_key());
    assert_eq!(h.key(KEY_BACK, 1031), KeyOutcome::Handled);
    assert_eq!(h.settings.menu_key(), mc_mods::DEFAULT_MENU_KEY);
    click_key(&mut h, 1040);
    assert_eq!(
        h.key(KEY_ESCAPE, 1041),
        KeyOutcome::Handled,
        "Esc only cancels"
    );
    assert_eq!(h.settings.menu_key(), 54);
    click_key(&mut h, 1050);
    h.click(5.0, 5.0, 1, 1051);
    assert!(!h.screen.capturing_menu_key(), "a click backs out");
    assert_eq!(h.settings.menu_key(), 54);
    assert_eq!(h.key(54, 1060), KeyOutcome::Close);
}

#[test]
fn hud_settings_cog_opens_the_module_without_dragging_and_returns_to_the_editor() {
    let mut h = Harness::new();
    h.settings.set_enabled("fps", true);
    h.screen.enter_editor(h.at(1000));
    h.draw_hud();
    h.build(1300);
    let index = h.settings.index_of("fps").unwrap();
    let before = h.settings.placement("fps", "fps");
    let gear = h
        .screen
        .hits
        .iter()
        .find_map(|(r, target)| (*target == Target::HudSettings(index)).then_some(*r))
        .expect("FPS cog");
    assert!(gear.x >= 0.0 && gear.right() <= SCREEN.0);
    h.click(gear.center().0, gear.center().1, 0, 1300);
    assert_eq!(h.screen.selected(), Some(index));
    assert!(!h.screen.hud_editing());
    assert!(h.screen.return_to_hud);
    assert!(h.settings.enabled("fps"));
    assert_eq!(h.settings.placement("fps", "fps"), before);
    h.build(1700);
    assert!(
        h.screen.hits.iter().any(|(_, target)| matches!(target,
        Target::Toggle(m, _) | Target::Choice(m, _) | Target::Slider(m, _, _)
            | Target::Swatch(m, _) if *m == index)),
        "real module settings are visible"
    );
    assert_eq!(h.key(KEY_ESCAPE, 1700), KeyOutcome::Handled);
    assert!(h.screen.hud_editing());
    h.draw_hud();
    h.build(2000);
    h.click(gear.center().0, gear.center().1, 0, 2000);
    h.build(2400);
    let back = h.screen.edit_hud_rect().expect("Back to HUD pill");
    h.click(back.center().0, back.center().1, 0, 2400);
    assert!(h.screen.hud_editing());
}

// ---- The HUD editor's wheel grip ---------------------------

impl Harness {
    /// Lay the mod HUD out as the client does every frame while the editor is open; that fills
    /// the slot rectangles the editor hit-tests.
    fn draw_hud(&mut self) {
        use crate::modhud::{HudCtx, HudData};
        self.hud.sync(&self.settings);
        let data = HudData::default();
        let ctx = HudCtx {
            data: &data,
            settings: &self.settings,
            screen: GUI_SCREEN,
            pixel_scale: GUI_SCALE,
            editing: true,
        };
        let mut list = DrawList::new();
        self.hud.draw(&ctx, &mut self.mc, &mut list);
    }

    /// The pointer at the GUI point `(gx, gy)`, as the client reports it once per frame.
    fn hover_gui(&mut self, gx: f32, gy: f32, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.pointer_moved(gx * GUI_SCALE, gy * GUI_SCALE, &mut input);
        }
        self.screen = screen;
    }

    /// One wheel notch at the GUI point `(gx, gy)`, then a frame: the HUD is laid out again
    /// and the pointer (which has not moved) is reported again.
    fn wheel_gui(&mut self, gx: f32, gy: f32, delta: f32, ms: u64) {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        {
            let mut input = self.input(ms);
            screen.scrolled(gx * GUI_SCALE, gy * GUI_SCALE, delta, &mut input);
        }
        self.screen = screen;
        self.draw_hud();
        self.hover_gui(gx, gy, ms);
    }

    fn slot_of(&self, mod_id: &str) -> usize {
        let m = self.settings.index_of(mod_id).unwrap();
        self.hud
            .slots()
            .iter()
            .position(|slot| slot.mod_index == m)
            .expect("the mod has a HUD slot")
    }
}

/// Requested: "When im scaling a hud element if it gets too small and goes off my mouse popinter
/// i can no longer scroll". Scale an element down until it has shrunk away from a pointer that
/// never moves (smaller than its distance to the pointer), keep scrolling, then scroll back up:
/// every notch must keep resizing it, and it ends up under the pointer again.
#[test]
fn the_wheel_keeps_resizing_an_element_that_shrank_off_the_pointer() {
    let mut h = Harness::new();
    h.settings.set_enabled("fps", true);
    // Big and pinned top-left: scaling about the anchor shrinks it towards the corner, away
    // from a pointer near its bottom-right.
    let big = HudPlacement::new(mc_mods::HudAnchor::TopLeft, 2.0, 2.0).scaled(3.0);
    h.settings.set_placement("fps", "fps", big);
    h.screen.enter_editor(h.at(1000));
    h.draw_hud();
    let slot = h.slot_of("fps");
    let start = h.hud.slots()[slot].rect.expect("drawn in the editor");
    let (px, py) = (start.x + start.w * 0.9, start.y + start.h * 0.9);
    h.hover_gui(px, py, 1000);
    let scale = |h: &Harness| h.settings.placement("fps", "fps").scale;
    assert_eq!(scale(&h), 3.0);

    // Down: 25 notches reach the minimum, 5 more push against it.
    let mut left_at = None;
    for notch in 0..30_u64 {
        let before = scale(&h);
        h.wheel_gui(px, py, -1.0, 1000 + notch * 16);
        let rect = h.hud.slots()[slot].rect.expect("still drawn");
        if left_at.is_none() && !rect.contains(px, py) {
            left_at = Some(notch);
        }
        if before > mc_mods::HUD_MIN_SCALE {
            assert!(
                scale(&h) < before,
                "notch {notch}: still shrinking ({before} -> {})",
                scale(&h)
            );
        }
    }
    let left_at = left_at.expect("the element shrank off the pointer");
    assert!(left_at < 5, "it left the pointer early: notch {left_at}");
    assert_eq!(scale(&h), mc_mods::HUD_MIN_SCALE, "a sane minimum: 0.5");
    let small = h.hud.slots()[slot].rect.unwrap();
    assert!(
        px - small.right() > small.w && py - small.bottom() > small.h,
        "now smaller than its distance to the pointer: {small:?} vs ({px}, {py})"
    );
    // Off the pointer it keeps its hover outline and scale label for a second after the last
    // notch, so the user sees what the wheel holds.
    let last_notch = h.at(1000 + 29 * 16);
    assert!(h.screen.hud_hover.is_none(), "nothing is under the pointer");
    assert!(h
        .screen
        .wheel_holds(slot, last_notch + Duration::from_millis(900)));
    assert!(!h
        .screen
        .wheel_holds(slot, last_notch + Duration::from_millis(1100)));

    // Up again: every notch grows it, until it covers the pointer once more.
    for notch in 0..25_u64 {
        let before = scale(&h);
        h.wheel_gui(px, py, 1.0, 2000 + notch * 16);
        assert!(
            scale(&h) > before,
            "notch {notch}: growing again ({before} -> {})",
            scale(&h)
        );
    }
    assert_eq!(scale(&h), 3.0);
    assert!(
        h.hud.slots()[slot].rect.unwrap().contains(px, py),
        "back under the pointer"
    );
}

/// The control for the test above: the grip is what keeps the wheel on an element that left
/// the pointer. Without it (the hit-only targeting before W1) the wheel stops the moment the
/// element shrinks off the pointer, which is the bug the user hit.
#[test]
fn without_the_grip_the_wheel_stops_where_the_element_left_the_pointer() {
    let mut h = Harness::new();
    h.settings.set_enabled("fps", true);
    let big = HudPlacement::new(mc_mods::HudAnchor::TopLeft, 2.0, 2.0).scaled(3.0);
    h.settings.set_placement("fps", "fps", big);
    h.screen.enter_editor(h.at(1000));
    h.draw_hud();
    let slot = h.slot_of("fps");
    let start = h.hud.slots()[slot].rect.unwrap();
    let (px, py) = (start.x + start.w * 0.9, start.y + start.h * 0.9);
    h.hover_gui(px, py, 1000);
    let mut ms = 1000;
    while h.hud.slots()[slot].rect.unwrap().contains(px, py) {
        ms += 16;
        h.wheel_gui(px, py, -1.0, ms);
        assert!(ms < 2000, "the element never left the pointer");
    }
    let stuck = h.settings.placement("fps", "fps").scale;
    assert!(
        stuck > mc_mods::HUD_MIN_SCALE,
        "left the pointer at {stuck}"
    );
    for _ in 0..5 {
        h.screen.hud_grip = None;
        ms += 16;
        h.wheel_gui(px, py, -1.0, ms);
    }
    assert_eq!(h.settings.placement("fps", "fps").scale, stuck);
}

/// The grip follows the element the user works with: hovering another element moves it there,
/// and a pointer that has gone away lets go (the wheel then resizes nothing).
#[test]
fn the_wheel_grip_moves_to_a_hovered_element_and_lets_go_far_away() {
    let mut h = Harness::new();
    for id in ["fps", "cps"] {
        h.settings.set_enabled(id, true);
    }
    let top_left = |y: f32| HudPlacement::new(mc_mods::HudAnchor::TopLeft, 2.0, y);
    h.settings
        .set_placement("fps", "fps", top_left(2.0).scaled(2.0));
    h.settings.set_placement("cps", "cps", top_left(120.0));
    h.screen.enter_editor(h.at(1000));
    h.draw_hud();
    let (fps, cps) = (h.slot_of("fps"), h.slot_of("cps"));
    let fps_scale = |h: &Harness| h.settings.placement("fps", "fps").scale;
    let cps_scale = |h: &Harness| h.settings.placement("cps", "cps").scale;

    // One notch at FPS's bottom-right corner shrinks it off the pointer.
    let rect = h.hud.slots()[fps].rect.unwrap();
    let corner = (rect.right() - 1.0, rect.bottom() - 1.0);
    h.hover_gui(corner.0, corner.1, 1000);
    h.wheel_gui(corner.0, corner.1, -1.0, 1016);
    assert_eq!(fps_scale(&h), 1.9);
    let shrunk = h.hud.slots()[fps].rect.unwrap();
    assert!(!shrunk.contains(corner.0, corner.1), "{shrunk:?}");
    // Still gripped: the next notch reaches it although nothing is under the pointer.
    h.wheel_gui(corner.0, corner.1, -1.0, 1032);
    assert_eq!(fps_scale(&h), 1.8);

    // Hovering CPS moves the grip there; back at the corner, the wheel no longer reaches FPS
    // (and CPS is far from it).
    let over_cps = h.hud.slots()[cps].rect.unwrap().center();
    h.hover_gui(over_cps.0, over_cps.1, 1048);
    assert!(
        !h.screen.wheel_holds(fps, h.at(1048)),
        "FPS's highlight left with the grip"
    );
    h.hover_gui(corner.0, corner.1, 1064);
    h.wheel_gui(corner.0, corner.1, -1.0, 1080);
    assert_eq!(fps_scale(&h), 1.8, "the grip moved to CPS");
    assert_eq!(cps_scale(&h), 1.0, "and CPS is not near the pointer");

    // Over FPS again it takes the wheel; far from everything nothing does, and the grip is gone.
    let over_fps = h.hud.slots()[fps].rect.unwrap().center();
    h.hover_gui(over_fps.0, over_fps.1, 1096);
    h.wheel_gui(over_fps.0, over_fps.1, 1.0, 1112);
    assert_eq!(fps_scale(&h), 1.9);
    h.hover_gui(400.0, 300.0, 1128);
    h.wheel_gui(400.0, 300.0, -1.0, 1144);
    assert_eq!((fps_scale(&h), cps_scale(&h)), (1.9, 1.0));
    assert!(h.screen.hud_grip.is_none());
}
