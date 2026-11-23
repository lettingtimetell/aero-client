//! The headless tests: the Keybinds view and its clash check, the
//! Profiles view on a real folder, the shared HUD style's STYLE block, live warnings, "Move on
//! screen" and the Edit HUD tray. Same harness idea as `tests.rs` (the 5-pixel test font, no
//! GPU), with the vanilla controls, `options.txt` settings and a game folder plugged in.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use mc_mods::{OptionValue, DEFAULT_MENU_KEY};
use mc_render::input::KeyBindings;
use mc_render::GameSettings;

use super::extras::ExtrasTarget;
use super::input::{KEY_BACK, KEY_ESCAPE, KEY_RETURN};
use super::keybinds::KeysTarget;
use super::paint::Rect;
use super::profiles::ProfilesTarget;
use super::state::{Target, View};
use super::*;
use crate::widget::test_support::uniform_font;
use crate::widget::DrawList;

const SCREEN: (f32, f32) = (1920.0, 1080.0);
const GUI_SCALE: f32 = 3.0;

struct Harness {
    settings: ModSettings,
    hud: ModHud,
    rows: Vec<DynamicRows>,
    text: MenuText,
    mc: FontRenderer,
    events: Vec<ModEvent>,
    screen: ModMenuScreen,
    t0: Instant,
    vanilla: KeyBindings,
    options: GameSettings,
    game_dir: Option<PathBuf>,
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
            vanilla: KeyBindings::vanilla(),
            options: GameSettings::default(),
            game_dir: None,
            no_world: false,
        }
    }

    fn build(&mut self, ms: u64) -> DrawList {
        let mut list = DrawList::new();
        let now = self.t0 + Duration::from_millis(ms);
        // The HUD pipeline learns which mods are on (the client does this every frame).
        self.hud.sync(&self.settings);
        let mut env = MenuEnv {
            text: &mut self.text,
            mc_font: &mut self.mc,
            settings: &self.settings,
            rows: &self.rows,
            hud: &self.hud,
            screen: SCREEN,
            gui_scale: GUI_SCALE,
            now,
            time_s: 0.0,
            game: MenuGame {
                no_world: self.no_world,
                vanilla_keys: Some(&self.vanilla),
                options: Some(&self.options),
                game_dir: self.game_dir.as_deref(),
            },
        };
        self.screen.build(&mut env, &mut list);
        list
    }

    fn with_input<R>(
        &mut self,
        ms: u64,
        f: impl FnOnce(&mut ModMenuScreen, &mut MenuInput<'_>) -> R,
    ) -> R {
        let mut screen = std::mem::replace(&mut self.screen, ModMenuScreen::new(&[], self.t0));
        let result = {
            let mut input = MenuInput {
                settings: &mut self.settings,
                hud: &mut self.hud,
                events: &mut self.events,
                gui_screen: (SCREEN.0 / GUI_SCALE, SCREEN.1 / GUI_SCALE),
                gui_scale: GUI_SCALE,
                now: self.t0 + Duration::from_millis(ms),
                alt: false,
                shift: false,
            };
            f(&mut screen, &mut input)
        };
        self.screen = screen;
        result
    }

    fn click_at(&mut self, (x, y): (f32, f32), button: u8, ms: u64) {
        self.with_input(ms, |screen, input| {
            screen.mouse_pressed(x, y, button, input);
            screen.mouse_released(x, y, button, input);
        });
    }

    fn key(&mut self, lwjgl: i32, typed: Option<char>, ms: u64) -> KeyOutcome {
        self.with_input(ms, |screen, input| screen.key(lwjgl, typed, input))
    }

    fn type_text(&mut self, text: &str, ms: u64) {
        for c in text.chars() {
            self.key(30, Some(c), ms);
        }
    }

    fn hit_where(&self, pick: impl Fn(&Target) -> bool) -> Option<Rect> {
        self.screen
            .hits
            .iter()
            .rev()
            .find_map(|(rect, target)| pick(target).then_some(*rect))
    }

    fn click_target(&mut self, pick: impl Fn(&Target) -> bool, button: u8, ms: u64) {
        let rect = self.hit_where(pick).expect("the target is on screen");
        self.click_at(rect.center(), button, ms);
    }

    fn open_view(&mut self, view: View, ms: u64) {
        self.build(ms);
        self.click_target(|t| *t == Target::Nav(view), 0, ms);
        self.build(ms + 16);
    }

    fn select(&mut self, mod_id: &str, ms: u64) -> usize {
        let m = self.settings.index_of(mod_id).expect("in the catalog");
        self.screen.select(m);
        self.build(ms);
        m
    }
}

fn temp_game_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "mc-rs-menu-profiles-{tag}-{}-{:?}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---- Keybinds ------------------------------------------------------------------------------

/// Every key is listed (the menu key and each key setting), a click captures and a key binds,
/// a right click puts the default back, and a clash with a Minecraft control shows in window 1's
/// badge only while the mod is on.
#[test]
fn keybinds_list_rebind_reset_and_warn() {
    let mut h = Harness::new();
    h.open_view(View::Keybinds, 1000);
    assert_eq!(h.screen.view(), Some(View::Keybinds));
    let zoom = h.settings.index_of("zoom").unwrap();
    let key = h.settings.catalog()[zoom].option_index("key").unwrap();
    // Every key setting has a row, and the menu key's shell target is still there.
    let keys: usize = h
        .settings
        .catalog()
        .iter()
        .map(|d| {
            d.options
                .iter()
                .filter(|o| matches!(o.kind, mc_mods::OptionKind::Keybind))
                .count()
        })
        .sum();
    // (Rows below the fold have no hit region, but every row is a keyboard row.)
    let rows = h.screen.nav[super::state::Pane::List.index()]
        .iter()
        .filter(|row| matches!(row.enter, Target::Keys(KeysTarget::Key(..))))
        .count();
    assert_eq!(rows, keys, "a row per key setting");
    assert!(h.screen.menu_key_rect().is_some());
    let badge = h.screen.keybinds_badge(&MenuEnv {
        text: &mut h.text,
        mc_font: &mut h.mc,
        settings: &h.settings,
        rows: &h.rows,
        hud: &h.hud,
        screen: SCREEN,
        gui_scale: GUI_SCALE,
        now: h.t0,
        time_s: 0.0,
        game: MenuGame {
            vanilla_keys: Some(&h.vanilla),
            ..MenuGame::default()
        },
    });
    assert_eq!(badge.count, Some(keys + 1));
    assert!(!badge.warn, "a fresh install clashes with nothing");
    // Rebind zoom to Q (Minecraft's Drop Item): click, then the key.
    h.reveal_in_list(Target::Keys(KeysTarget::Key(zoom, key)), 1050);
    h.click_target(|t| *t == Target::Keys(KeysTarget::Key(zoom, key)), 0, 1100);
    assert!(h.screen.capturing_key());
    assert_eq!(h.key(16, Some('q'), 1110), KeyOutcome::Handled);
    assert_eq!(h.settings.key("zoom", "key"), 16);
    assert!(!h.screen.capturing_key());
    // Zoom is off: no warning yet; on: the badge warns.
    let warn = |h: &mut Harness| {
        h.build(1200);
        let vanilla = h.vanilla.clone();
        let env = MenuEnv {
            text: &mut h.text,
            mc_font: &mut h.mc,
            settings: &h.settings,
            rows: &h.rows,
            hud: &h.hud,
            screen: SCREEN,
            gui_scale: GUI_SCALE,
            now: h.t0,
            time_s: 0.0,
            game: MenuGame {
                vanilla_keys: Some(&vanilla),
                ..MenuGame::default()
            },
        };
        h.screen.keybinds_badge(&env).warn
    };
    assert!(!warn(&mut h), "zoom is off: its key does nothing yet");
    h.settings.set_enabled_at(zoom, true);
    assert!(warn(&mut h), "zoom on Q clashes with Drop Item");
    // Right-click puts C back.
    h.build(1300);
    h.click_target(|t| *t == Target::Keys(KeysTarget::Key(zoom, key)), 1, 1300);
    assert_eq!(h.settings.key("zoom", "key"), 46);
    assert!(!warn(&mut h));
    // Backspace unbinds.
    h.build(1400);
    h.click_target(|t| *t == Target::Keys(KeysTarget::Key(zoom, key)), 0, 1400);
    h.key(KEY_BACK, None, 1410);
    assert_eq!(h.settings.key("zoom", "key"), 0);
    // The menu key: right-click restores Right Shift.
    h.settings.set_menu_key(50);
    h.build(1500);
    h.reveal_in_list(Target::Keys(KeysTarget::MenuKey), 1500);
    h.click_target(|t| *t == Target::Keys(KeysTarget::MenuKey), 1, 1500);
    assert_eq!(h.settings.menu_key(), DEFAULT_MENU_KEY);
    // Window 3 is Minecraft's controls, read-only: no mod (and no switch) is shown there.
    assert!(h.hit_where(|t| matches!(t, Target::Switch(_))).is_none());
}

// ---- Profiles ------------------------------------------------------------------------------

/// Save a profile by name, change things, load it back (armed: two clicks), the setup before
/// the load is kept, rename it, and delete it (two clicks), all through the view on a real
/// folder.
#[test]
fn profiles_save_load_with_confirm_rename_delete() {
    let mut h = Harness::new();
    let game = temp_game_dir("view");
    h.game_dir = Some(game.clone());
    h.settings.set_enabled("fps", true);
    h.settings.set_enabled("cps", true);
    h.open_view(View::Profiles, 1000);
    assert_eq!(h.screen.view(), Some(View::Profiles));
    let wait = |h: &mut Harness| {
        if let Some(library) = &h.screen.profiles.library {
            library.wait_idle();
        }
    };
    wait(&mut h);
    h.build(1050);
    // Name it and press Enter.
    h.click_target(
        |t| *t == Target::Profiles(ProfilesTarget::Field(super::profiles::FieldId::SaveName)),
        0,
        1100,
    );
    h.type_text("Bedwars", 1110);
    assert_eq!(
        h.screen.search_query(),
        "",
        "typing went to the field, not the search"
    );
    assert_eq!(h.key(KEY_RETURN, None, 1120), KeyOutcome::Handled);
    wait(&mut h);
    h.build(1200);
    assert_eq!(h.screen.profiles.entries.len(), 1);
    assert_eq!(h.screen.profiles.entries[0].name, "Bedwars");
    assert!(game
        .join("mc-rs-mod-profiles")
        .join("Bedwars.json")
        .exists());
    assert_eq!(h.screen.profiles.current, [true], "it is the current setup");
    // Change the setup, then load the profile back.
    h.settings.set_enabled("fps", false);
    h.settings.set_enabled("zoom", true);
    h.build(1300);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Row(0)), 0, 1300);
    h.build(1310);
    let before = h.settings.generation();
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 1320);
    assert_eq!(h.settings.generation(), before, "the first click only arms");
    h.build(1330);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 1340);
    assert!(h.settings.enabled("fps") && h.settings.enabled("cps"));
    assert!(!h.settings.enabled("zoom"));
    assert_eq!(h.settings.generation(), before + 1, "one step");
    // The setup before the load is kept and can be loaded back.
    let backup = h
        .screen
        .profiles
        .library
        .as_ref()
        .unwrap()
        .backup()
        .unwrap();
    assert!(backup.setup.as_ref().unwrap().enabled("zoom"));
    wait(&mut h);
    assert!(game
        .join("mc-rs-mod-profiles")
        .join(mc_mods::profiles::BACKUP_FILE)
        .exists());
    // An armed Load expires: a click after three seconds only arms again.
    h.build(1400);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Backup), 0, 1400);
    h.build(1410);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 1420);
    h.build(5000);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 5000);
    assert!(
        !h.settings.enabled("zoom"),
        "the arm expired: nothing loaded"
    );
    // Rename.
    h.build(5100);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Row(0)), 0, 5100);
    h.build(5110);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Rename), 0, 5120);
    for _ in 0..7 {
        h.key(KEY_BACK, None, 5130);
    }
    h.type_text("SkyWars", 5140);
    h.key(KEY_RETURN, None, 5150);
    wait(&mut h);
    h.build(5200);
    assert_eq!(h.screen.profiles.entries[0].name, "SkyWars");
    assert!(game
        .join("mc-rs-mod-profiles")
        .join("SkyWars.json")
        .exists());
    assert!(!game
        .join("mc-rs-mod-profiles")
        .join("Bedwars.json")
        .exists());
    // Delete needs two clicks.
    h.build(5300);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Delete), 0, 5300);
    wait(&mut h);
    h.build(5310);
    assert_eq!(h.screen.profiles.entries.len(), 1, "armed only");
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Delete), 0, 5320);
    wait(&mut h);
    h.build(5400);
    assert!(h.screen.profiles.entries.is_empty());
    // Esc leaves a focused field before it closes anything.
    h.click_target(
        |t| *t == Target::Profiles(ProfilesTarget::Field(super::profiles::FieldId::SaveName)),
        0,
        5500,
    );
    assert_eq!(h.key(KEY_ESCAPE, None, 5510), KeyOutcome::Handled);
    assert_eq!(h.key(KEY_ESCAPE, None, 5520), KeyOutcome::Close);
    let _ = std::fs::remove_dir_all(&game);
}

/// Renaming with the mouse: Rename opens the field, typing fills it, and clicking "Done" renames
/// (the click takes the keyboard away from the field on press; the name must survive that).
#[test]
fn rename_commits_with_the_done_button() {
    let mut h = Harness::new();
    let game = temp_game_dir("rename");
    h.game_dir = Some(game.clone());
    h.open_view(View::Profiles, 1000);
    let wait = |h: &mut Harness| {
        if let Some(library) = &h.screen.profiles.library {
            library.wait_idle();
        }
    };
    if let Some(library) = &h.screen.profiles.library {
        library.save("Old name", &h.settings, None);
    }
    wait(&mut h);
    h.build(1100);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Row(0)), 0, 1100);
    h.build(1110);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Rename), 0, 1120);
    for _ in 0.."Old name".len() {
        h.key(KEY_BACK, None, 1130);
    }
    h.type_text("New name", 1140);
    h.build(1150);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Rename), 0, 1160);
    wait(&mut h);
    h.build(1200);
    assert_eq!(h.screen.profiles.entries.len(), 1);
    assert_eq!(h.screen.profiles.entries[0].name, "New name");
    assert!(game
        .join("mc-rs-mod-profiles")
        .join("New name.json")
        .exists());
    let _ = std::fs::remove_dir_all(&game);
}

/// "Fresh install" loads every default (after a confirm) and keeps the menu key.
#[test]
fn fresh_install_resets_the_whole_setup_but_not_the_menu_key() {
    let mut h = Harness::new();
    let game = temp_game_dir("fresh");
    h.game_dir = Some(game.clone());
    h.settings.set_enabled("fps", true);
    h.settings.set_menu_key(50);
    let zoom = h.settings.index_of("zoom").unwrap();
    h.settings.set_pinned_at(zoom, true);
    h.open_view(View::Profiles, 1000);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Fresh), 0, 1010);
    h.build(1020);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 1030);
    h.build(1040);
    h.click_target(|t| *t == Target::Profiles(ProfilesTarget::Load), 0, 1050);
    assert!(h
        .settings
        .setup_eq(&ModSettings::with_catalog(h.settings.catalog())));
    assert_eq!(h.settings.menu_key(), 50);
    assert!(!h.settings.pinned_at(zoom));
    h.build(1100);
    assert!(
        h.hit_where(|t| *t == Target::Profiles(ProfilesTarget::Load))
            .is_none(),
        "the current setup cannot be loaded again"
    );
    if let Some(library) = &h.screen.profiles.library {
        library.wait_idle();
    }
    let _ = std::fs::remove_dir_all(&game);
}

// ---- the shared HUD style, live warnings ---------------------------------------------------

/// A readout's STYLE block: the choice, folded shared rows that unfold, Custom shows its own.
#[test]
fn the_style_block_switches_and_folds() {
    let mut h = Harness::new();
    let fps = h.select("fps", 1000);
    let color = h.settings.catalog()[fps]
        .option_index("text_color")
        .unwrap();
    assert!(h.settings.uses_hud_style_at(fps));
    assert!(h
        .hit_where(|t| *t == Target::Extras(ExtrasTarget::StyleMode(fps, false)))
        .is_some());
    assert!(
        h.hit_where(|t| *t == Target::Swatch(fps, color)).is_none(),
        "folded on the HUD style"
    );
    h.click_target(|t| *t == Target::Extras(ExtrasTarget::StyleOpen), 0, 1010);
    h.build(1020);
    assert!(
        h.hit_where(|t| *t == Target::Swatch(fps, color)).is_some(),
        "unfolded"
    );
    // Editing it there edits the shared style: CPS follows.
    h.settings
        .set_value_at(fps, color, OptionValue::Color(0xFFFF_5555));
    assert_eq!(h.settings.color("cps", "text_color"), 0xFFFF_5555);
    // Custom: its own rows, the look kept.
    h.click_target(
        |t| *t == Target::Extras(ExtrasTarget::StyleMode(fps, false)),
        0,
        1030,
    );
    assert!(!h.settings.uses_hud_style_at(fps));
    assert_eq!(h.settings.color("fps", "text_color"), 0xFFFF_5555);
    h.extras_fold(1040);
    assert!(
        h.hit_where(|t| *t == Target::Swatch(fps, color)).is_some(),
        "Custom shows its own"
    );
    // Keystrokes is not a readout: no choice.
    let keys = h.select("keystrokes", 1100);
    assert!(h
        .hit_where(|t| matches!(t, Target::Extras(ExtrasTarget::StyleMode(m, _)) if *m == keys))
        .is_none());
}

impl Harness {
    /// Scroll window 2 so the keyboard row whose Enter is `target` is in view, and lay out.
    fn reveal_in_list(&mut self, target: Target, ms: u64) {
        let list = super::state::Pane::List.index();
        let span = self.screen.nav[list]
            .iter()
            .find(|row| row.enter == target)
            .map(|row| row.span)
            .expect("a row for the target");
        let top = (span.0 - 40.0).max(0.0);
        self.screen.scroll[list] = top;
        self.screen.scroll_target[list] = top;
        self.build(ms);
    }

    /// Fold the HUD style's rows again and lay out.
    fn extras_fold(&mut self, ms: u64) {
        self.screen.extras.style_open = false;
        self.build(ms);
    }
}

/// A live warning shows only while true: Anti-Aliasing with MSAA on pushes its settings down.
#[test]
fn live_warnings_show_only_while_true() {
    let mut h = Harness::new();
    h.options.msaa_samples = 1;
    let aa = h.select("anti_aliasing", 1000);
    let first = |h: &Harness| {
        h.screen
            .hits
            .iter()
            .filter_map(|(r, t)| match t {
                Target::Choice(m, _) | Target::Toggle(m, _) | Target::Slider(m, _, _)
                    if *m == aa =>
                {
                    Some(r.y)
                }
                _ => None,
            })
            .fold(f32::MAX, f32::min)
    };
    let without = first(&h);
    h.options.msaa_samples = 4;
    h.build(1100);
    let with = first(&h);
    assert!(
        with > without + 10.0,
        "the warning box pushed the rows down: {without} -> {with}"
    );
}

// ---- Move on screen and the tray ------------------------------------------------------------

/// "Move on screen" opens Edit HUD with the element picked; Esc comes back to the mod. It is
/// greyed out while the mod is off and without a world.
#[test]
fn move_on_screen_round_trip() {
    let mut h = Harness::new();
    let fps = h.select("fps", 1000);
    assert!(
        h.hit_where(|t| *t == Target::MoveOnScreen(fps)).is_none(),
        "off: greyed out"
    );
    h.settings.set_enabled_at(fps, true);
    h.build(1010);
    h.click_target(|t| *t == Target::MoveOnScreen(fps), 0, 1020);
    assert!(h.screen.hud_editing());
    let slot = h
        .hud
        .slots()
        .iter()
        .position(|s| s.mod_index == fps)
        .unwrap();
    assert_eq!(h.screen.hud_last, Some(slot), "the arrows nudge it");
    assert!(h.screen.extras.picked.is_some_and(|(p, _)| p == slot));
    h.build(1030);
    assert_eq!(h.key(KEY_ESCAPE, None, 1040), KeyOutcome::Handled);
    assert!(!h.screen.hud_editing());
    assert_eq!(h.screen.selected(), Some(fps), "back at its settings");
    // No world: greyed out.
    h.no_world = true;
    h.build(1100);
    assert!(h.hit_where(|t| *t == Target::MoveOnScreen(fps)).is_none());
    // A mod with nothing on the HUD has no button.
    h.no_world = false;
    let zoom = h.select("zoom", 1200);
    h.settings.set_enabled_at(zoom, true);
    h.build(1210);
    assert!(h.hit_where(|t| *t == Target::MoveOnScreen(zoom)).is_none());
}

/// The tray lists HUD mods that are off; a click turns one on and picks it.
#[test]
fn the_tray_adds_a_hud_mod() {
    let mut h = Harness::new();
    h.build(1000);
    h.click_target(|t| *t == Target::EditHud, 0, 1000);
    assert!(h.screen.hud_editing());
    h.build(1500);
    h.click_target(|t| *t == Target::Extras(ExtrasTarget::Tray), 0, 1500);
    h.build(1510);
    let cps = h.settings.index_of("cps").unwrap();
    assert!(!h.settings.enabled_at(cps));
    h.click_target(
        |t| *t == Target::Extras(ExtrasTarget::TrayAdd(cps)),
        0,
        1520,
    );
    assert!(h.settings.enabled_at(cps), "added");
    assert!(h.screen.hud_editing(), "still in the editor");
    h.build(1530);
    assert!(
        h.hit_where(|t| *t == Target::Extras(ExtrasTarget::TrayAdd(cps)))
            .is_none(),
        "no longer in the tray"
    );
    // A click on the panel does not start a drag or close anything.
    let panel = h
        .hit_where(|t| *t == Target::Extras(ExtrasTarget::TrayPanel))
        .expect("the panel");
    h.click_at((panel.x + 4.0, panel.y + 4.0), 0, 1540);
    assert!(h.screen.hud_editing() && h.screen.drag.is_none());
}
