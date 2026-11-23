//! The menu's state: [`ModMenuScreen`]'s fields, what a click can hit ([`Target`]), drags, the
//! three windows' views and selection, and where the menu reopens ([`MenuPlace`]).
//!
//! Window 1 picks a [`View`], window 2 lists its mods, window 3
//! shows the selected mod's settings. The views keep their own state in `keybinds.rs` and
//! `profiles.rs` (`KeysState`, `ProfilesState`) and their own hit targets (`KeysTarget`,
//! `ProfilesTarget`), so they never need to edit this file.

use std::time::Instant;

use mc_mods::menu_rules::SettingsLayout;
use mc_mods::{HudPlacement, ModDescriptor};

use super::anim;
// The extras' state and targets (`extras.rs`).
use super::extras::{ExtrasState, ExtrasTarget};
use super::keybinds::{KeysState, KeysTarget};
use super::list::{Found, MenuOrder};
use super::paint::Rect;
use super::profiles::{ProfilesState, ProfilesTarget};
use super::widgets::ms_since;

/// What window 2 lists, picked in window 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    /// Every mod changed from a fresh install (its switch or any setting), plus the pinned mods.
    MySetup,
    /// Every key a mod uses.
    Keybinds,
    /// Saved setups, to switch between whole configurations.
    Profiles,
    /// One category: an index into [`mc_mods::menu_layout::CATEGORIES`].
    Category(u8),
    /// Every mod, A to Z.
    AllMods,
}

/// Where the menu is: the list in window 2 and the mod in window 3. The client keeps the last
/// one in memory so the menu reopens there; it is never written to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MenuPlace {
    /// The list in window 2.
    pub view: View,
    /// The mod in window 3, by catalog index.
    pub selected: Option<usize>,
}

/// The window that has the keyboard (Tab moves it). Its [`Pane::index`] is also the index of
/// that window's scroll state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Pane {
    /// Window 1: where.
    Sidebar,
    /// Window 2: what.
    List,
    /// Window 3: how.
    Details,
}

impl Pane {
    /// Left to right.
    pub(super) const ALL: [Pane; 3] = [Pane::Sidebar, Pane::List, Pane::Details];

    /// The window's position, 0 to 2.
    pub(super) const fn index(self) -> usize {
        match self {
            Pane::Sidebar => 0,
            Pane::List => 1,
            Pane::Details => 2,
        }
    }
}

/// A row the arrow keys can land on (recorded while a window draws, in order). Enter does what a
/// left click on `enter` does; Space what a click on `space` does (a mod row: its switch).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct NavRow {
    /// The left-click target.
    pub(super) enter: Target,
    /// What Space toggles, if anything.
    pub(super) space: Option<Target>,
    /// The row's top and bottom in design pixels from the top of the window's content, so the
    /// window can scroll it into view.
    pub(super) span: (f32, f32),
}

/// What a click can hit. Recorded while drawing (immediate mode), read by the next input event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Target {
    /// A window body (scrolling): 0 where, 1 what, 2 how.
    Window(usize),
    /// A window-1 entry.
    Nav(View),
    /// A mod's row in window 2: a click selects it, a right click pins or unpins it.
    ModRow(usize),
    /// A search result's settings under one mod: a click opens the mod at those settings.
    Found(usize),
    /// A mod's switch (window 2's rows, window 3's title bar).
    Switch(usize),
    /// Window 3's pin button.
    Pin(usize),
    /// A setting's label: a right click resets the setting.
    Label(usize, usize),
    /// The reset arrow beside a changed setting.
    ResetOption(usize, usize),
    /// One tab of a tabbed picker: (mod, picker option, choice).
    Tab(usize, usize, usize),
    /// A named group's heading (collapse or expand): (mod, group index).
    Group(usize, u8),
    Toggle(usize, usize),
    Choice(usize, usize),
    Key(usize, usize),
    Text(usize, usize),
    Button(usize, usize),
    Swatch(usize, usize),
    Preset(usize, usize, u32),
    /// A colour slider: channel 0 hue, 1 saturation, 2 value, 3 alpha.
    ColorSlider(usize, usize, u8, Rect),
    Slider(usize, usize, Rect),
    DynCycle(usize, u64),
    DynAction(usize, u64, u8),
    EditHud,
    HudSettings(usize),
    Done,
    ResetAll,
    /// The search field: a click gives typing back to the search.
    SearchField,
    /// The search field's clear button.
    SearchClear,
    /// The menu-key capture (drawn in the Keybinds view): click, then press a key.
    MenuKey,
    /// A mod's "Reset" (armed: needs a second click).
    ResetMod(usize),
    /// A profile mod's name field.
    ProfileName(usize),
    ProfileSave(usize),
    ProfileLoad(usize, usize),
    /// Armed like "Reset".
    ProfileDelete(usize, usize),
    /// Window 3's "Move on screen" (`editor.rs`).
    #[allow(dead_code)] // pushed by `editor.rs`
    MoveOnScreen(usize),
    /// The Keybinds view's own targets (`keybinds.rs`).
    #[allow(dead_code)] // pushed by `keybinds.rs`
    Keys(KeysTarget),
    /// The Profiles view's own targets (`profiles.rs`).
    #[allow(dead_code)] // pushed by `profiles.rs`
    Profiles(ProfilesTarget),
    /// The extras: the HUD style choice in window 3, the HUD editor's tray
    /// (`extras.rs`).
    Extras(ExtrasTarget),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Drag {
    Slider(usize, usize, Rect),
    ColorSlider(usize, usize, u8, Rect),
    /// The HUD editor: a press on an element, armed until it moves 2 GUI px.
    Hud {
        slot: usize,
        start: (f32, f32),
        grab: (f32, f32),
        size: (f32, f32),
        moving: bool,
        placement: Option<HudPlacement>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Mode {
    Menu,
    HudEditor,
}

/// The menu screen. `Ui::ModMenu` in the client holds one.
#[derive(Debug)]
pub struct ModMenuScreen {
    pub(super) opened_at: Instant,
    pub(super) closing_since: Option<Instant>,
    pub(super) mode: Mode,
    pub(super) mode_since: Instant,
    /// Every catalog mod once, in menu order, with its home category.
    pub(super) order: MenuOrder,
    /// What window 2 lists. `None` until the first frame picks My setup (or HUD when My setup
    /// is empty).
    pub(super) view: Option<View>,
    /// The mod window 3 shows.
    pub(super) selected: Option<usize>,
    /// While searching: the window-1 entry that narrows the results (`None`: all of them).
    pub(super) narrow: Option<View>,
    /// The window with the keyboard.
    pub(super) pane: Pane,
    /// The keyboard moved the focus since the pointer last moved: draw the focus ring.
    pub(super) keyboard: bool,
    /// The arrow keys' row in each window (an index into that window's `*_rows`).
    pub(super) cursors: [Option<usize>; 3],
    /// Each window's rows in order, refilled while it draws.
    pub(super) nav: [Vec<NavRow>; 3],
    /// Window 2 scrolls this mod's row into view once.
    pub(super) reveal: Option<usize>,
    /// The arrow keys just moved this window's row: it scrolls the row into view once.
    pub(super) reveal_nav: Option<Pane>,
    /// Window 3's blocks for the selected mod (the `menu_rules::settings_layout`, worked
    /// out when the selection changes).
    pub(super) settings_layout: Option<(usize, SettingsLayout)>,
    /// Window 3 scrolls this setting of the selected mod into view once.
    pub(super) reveal_option: Option<usize>,
    /// Settings of the selected mod found by a search: shown even when a rule would hide them,
    /// and flashed from `flash_since`.
    pub(super) flash_options: Vec<usize>,
    pub(super) flash_since: Option<Instant>,
    /// Changed-from-a-fresh-install per catalog index, refreshed when the settings generation
    /// moves (`changed_gen`).
    pub(super) changed: Vec<bool>,
    pub(super) changed_gen: u64,
    /// Named groups the user opened or closed against their default: (mod, group index).
    pub(super) toggled_groups: Vec<(usize, u8)>,
    /// This menu's search and its results.
    pub(super) found: Found,
    /// The search's best result in window 2 this frame (what Enter opens), `None` when not
    /// searching.
    pub(super) best: Option<Target>,
    /// No world is loaded (the title screen, a replay browser): Edit HUD and Move on screen
    /// are greyed out. From [`super::MenuGame::no_world`] each frame.
    pub(super) no_world: bool,
    /// Hover text drawn last, on top of everything (`tooltip_at` set while something is
    /// hovered this frame).
    pub(super) tooltip: String,
    pub(super) tooltip_at: Option<(f32, f32)>,
    pub(super) scroll: [f32; 3],
    pub(super) scroll_target: [f32; 3],
    pub(super) max_scroll: [f32; 3],
    pub(super) switch_anim: Vec<f32>,
    pub(super) last_frame: Option<Instant>,
    pub(super) hits: Vec<(Rect, Target)>,
    pub(super) cursor: (f32, f32),
    pub(super) pressed: Option<(Target, u8)>,
    pub(super) drag: Option<Drag>,
    pub(super) color_editor: Option<(usize, usize)>,
    pub(super) color_hsv: [f32; 4],
    pub(super) capture: Option<(usize, usize)>,
    pub(super) text_focus: Option<(usize, usize)>,
    pub(super) hud_hover: Option<usize>,
    pub(super) hud_last: Option<usize>,
    /// Opened settings from the HUD editor: Esc or the taskbar returns there.
    pub(super) return_to_hud: bool,
    /// The HUD element the wheel keeps resizing (the last one hovered, pressed or scrolled) and
    /// the GUI point where the pointer last worked on it. See `ModMenuScreen::scroll_target`.
    pub(super) hud_grip: Option<(usize, (f32, f32))>,
    /// When the wheel last resized the gripped element: it keeps its hover outline and scale
    /// label for `WHEEL_HOLD_MS`, also once it has shrunk off the pointer.
    pub(super) hud_wheel_at: Option<Instant>,
    /// How far the menu dim has faded out while Color Grading's settings are open: 0 is the
    /// normal dim, 1 none.
    pub(super) undim: f32,
    pub(super) guides: mc_mods::SnapGuides,
    pub(super) reset_all_armed: Option<Instant>,
    /// What the user typed into the search field (as typed; matching ignores case).
    pub(super) search: String,
    /// Ctrl+A selected the complete query. Search retains its existing end-of-text editing.
    pub(super) search_selected: bool,
    /// The search that the list's scroll position belongs to (a new search scrolls to the top).
    pub(super) search_seen: String,
    /// The menu-key capture is waiting for a key.
    pub(super) menu_key_capture: bool,
    /// The profile name field of this mod has focus.
    pub(super) name_focus: Option<usize>,
    /// The profile name being typed.
    pub(super) profile_name: String,
    /// "Reset" of this mod was clicked once at this time.
    pub(super) reset_armed: Option<(usize, Instant)>,
    /// Delete of this mod's profile was clicked once at this time.
    pub(super) delete_armed: Option<(usize, usize, Instant)>,
    /// The slider the arrow keys move when none is hovered: the last one pressed.
    pub(super) last_slider: Option<(usize, usize)>,
    /// How many results the search shows this frame.
    pub(super) search_found: usize,
    /// The Keybinds view's state.
    #[allow(dead_code)] // read by `keybinds.rs`
    pub(super) keys: KeysState,
    /// The Profiles view's state.
    #[allow(dead_code)] // read by `profiles.rs`
    pub(super) profiles: ProfilesState,
    /// The extras' state (the HUD style's fold, the HUD editor's tray).
    pub(super) extras: ExtrasState,
}

impl ModMenuScreen {
    /// A menu that starts opening at `now`, on My setup (HUD when My setup is empty): the first
    /// open of a session.
    #[must_use]
    pub fn new(catalog: &[&'static ModDescriptor], now: Instant) -> ModMenuScreen {
        ModMenuScreen {
            opened_at: now,
            closing_since: None,
            mode: Mode::Menu,
            mode_since: now,
            order: MenuOrder::new(catalog),
            view: None,
            selected: None,
            narrow: None,
            pane: Pane::List,
            keyboard: false,
            cursors: [None; 3],
            nav: [Vec::new(), Vec::new(), Vec::new()],
            reveal: None,
            reveal_nav: None,
            settings_layout: None,
            reveal_option: None,
            flash_options: Vec::new(),
            flash_since: None,
            changed: Vec::new(),
            changed_gen: 0,
            toggled_groups: Vec::new(),
            found: Found::default(),
            best: None,
            no_world: false,
            tooltip: String::new(),
            tooltip_at: None,
            scroll: [0.0; 3],
            scroll_target: [0.0; 3],
            max_scroll: [0.0; 3],
            switch_anim: Vec::new(),
            last_frame: None,
            hits: Vec::new(),
            cursor: (-1.0, -1.0),
            pressed: None,
            drag: None,
            color_editor: None,
            color_hsv: [0.0, 0.0, 1.0, 1.0],
            capture: None,
            text_focus: None,
            hud_hover: None,
            hud_last: None,
            return_to_hud: false,
            hud_grip: None,
            hud_wheel_at: None,
            undim: 0.0,
            guides: mc_mods::SnapGuides::default(),
            reset_all_armed: None,
            search: String::new(),
            search_selected: false,
            search_seen: String::new(),
            menu_key_capture: false,
            name_focus: None,
            profile_name: String::new(),
            reset_armed: None,
            delete_armed: None,
            last_slider: None,
            search_found: 0,
            keys: KeysState::default(),
            profiles: ProfilesState::default(),
            extras: ExtrasState::default(),
        }
    }

    /// A menu that reopens where the last one was ([`ModMenuScreen::place`]), with the search
    /// cleared.
    #[must_use]
    pub fn reopen(
        catalog: &[&'static ModDescriptor],
        now: Instant,
        place: MenuPlace,
    ) -> ModMenuScreen {
        let mut screen = ModMenuScreen::new(catalog, now);
        screen.view = Some(place.view);
        screen.selected = place.selected.filter(|&m| m < catalog.len());
        screen.reveal = screen.selected;
        screen
    }

    /// Where the menu is now, for [`ModMenuScreen::reopen`].
    #[must_use]
    pub fn place(&self) -> MenuPlace {
        MenuPlace {
            view: self.view.unwrap_or(View::MySetup),
            selected: self.selected,
        }
    }

    /// Start the closing animation (the client keeps drawing the screen until
    /// [`ModMenuScreen::finished`]).
    pub fn begin_close(&mut self, now: Instant) {
        if self.closing_since.is_none() {
            self.closing_since = Some(now);
            self.drag = None;
            self.capture = None;
            self.text_focus = None;
            self.menu_key_capture = false;
            self.name_focus = None;
        }
    }

    /// The closing animation is over.
    #[must_use]
    pub fn finished(&self, now: Instant) -> bool {
        self.closing_since.is_some_and(|since| {
            anim::close_finished(ms_since(since, now), Pane::ALL.len())
                || (self.mode == Mode::HudEditor && ms_since(since, now) > anim::CLOSE_MS)
        })
    }

    /// The catalog index of the mod whose settings window 3 shows (the client builds that mod's
    /// dynamic rows). Same as [`ModMenuScreen::selected`]; the name is the seven-window menu's.
    #[must_use]
    pub fn expanded(&self) -> Option<usize> {
        self.selected
    }

    /// The catalog index of the mod whose settings window 3 shows.
    #[must_use]
    pub fn selected(&self) -> Option<usize> {
        self.selected
    }

    /// What window 2 lists (`None` before the first frame).
    #[must_use]
    pub fn view(&self) -> Option<View> {
        self.view
    }

    /// The HUD editor is open (HUD elements draw samples and the client dims under them).
    #[must_use]
    pub fn hud_editing(&self) -> bool {
        self.mode == Mode::HudEditor && self.closing_since.is_none()
    }

    /// A text field is focused, a search is being typed (held Backspace), or the HUD editor is
    /// open (held arrows nudge): the client then delivers key repeats. (The client drops
    /// repeats of the menu key itself, so holding it never closes the menu it just opened.)
    #[must_use]
    pub fn wants_key_repeat(&self) -> bool {
        self.text_focus.is_some()
            || self.name_focus.is_some()
            || self.profiles_typing()
            || !self.search.is_empty()
            || self.mode == Mode::HudEditor
    }

    /// The search as typed (empty when not searching). Lanes whose dynamic rows are long lists
    /// may filter them by it; the menu matches row labels itself either way.
    #[must_use]
    pub fn search_query(&self) -> &str {
        &self.search
    }

    /// Whether a search is filtering window 2.
    #[must_use]
    pub fn searching(&self) -> bool {
        self.search.split_whitespace().next().is_some()
    }

    /// The menu-key capture is waiting for the new key (the client must then deliver every key
    /// to the menu, the menu key included).
    #[must_use]
    pub fn capturing_menu_key(&self) -> bool {
        self.menu_key_capture
    }

    /// A keybind row or the menu-key capture is waiting for the next key: the client
    /// then delivers it here before the screenshot/fullscreen bindings, so F2 or F11 can be
    /// bound like any other key.
    #[must_use]
    pub fn capturing_key(&self) -> bool {
        self.capture.is_some() || self.menu_key_capture
    }

    /// Where the search field was drawn last frame.
    #[must_use]
    pub fn search_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::SearchField)
    }

    /// Where the menu-key capture was drawn last frame (in the Keybinds view).
    #[must_use]
    pub fn menu_key_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::MenuKey)
    }

    /// Where a mod's switch was drawn last frame (window 2's row first, else window 3's).
    #[must_use]
    pub fn switch_rect(&self, mod_index: usize) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Switch(mod_index))
    }

    /// Where a mod's row was drawn last frame in window 2.
    #[must_use]
    pub fn row_rect(&self, mod_index: usize) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::ModRow(mod_index))
    }

    /// Where a window-1 entry was drawn last frame.
    #[must_use]
    pub fn nav_rect(&self, view: View) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Nav(view))
    }

    /// Where window 3's pin button was drawn last frame.
    #[must_use]
    pub fn pin_rect(&self, mod_index: usize) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::Pin(mod_index))
    }

    /// Where the "Edit HUD" pill was drawn last frame (`None` while it is greyed out).
    #[must_use]
    pub fn edit_hud_rect(&self) -> Option<Rect> {
        self.hit_rect(|target| *target == Target::EditHud)
    }

    pub(super) fn hit_rect(&self, pick: impl Fn(&Target) -> bool) -> Option<Rect> {
        self.hits
            .iter()
            .find_map(|(rect, target)| pick(target).then_some(*rect))
    }

    pub(super) fn hit(&self, x: f32, y: f32) -> Option<Target> {
        self.hits
            .iter()
            .rev()
            .find(|(rect, _)| rect.contains(x, y))
            .map(|(_, target)| *target)
    }

    /// Show mod `m` in window 3 (window 2 scrolls its row into view). Picking another mod drops
    /// the old one's editors, captures and armed buttons, and window 3 starts at its top.
    pub(super) fn select(&mut self, m: usize) {
        if self.selected != Some(m) {
            self.selected = Some(m);
            let details = Pane::Details.index();
            self.scroll[details] = 0.0;
            self.scroll_target[details] = 0.0;
            self.cursors[details] = None;
            self.color_editor = None;
            self.capture = None;
            self.text_focus = None;
            self.name_focus = None;
            self.reset_armed = None;
            self.delete_armed = None;
            self.flash_options.clear();
            self.flash_since = None;
            self.reveal_option = None;
        }
        self.reveal = Some(m);
    }

    /// Switch window 2 to `view` (the list starts at its top).
    pub(super) fn set_view(&mut self, view: View) {
        if self.view != Some(view) {
            self.view = Some(view);
            let list = Pane::List.index();
            self.scroll[list] = 0.0;
            self.scroll_target[list] = 0.0;
            self.cursors[list] = None;
        }
        self.reveal = self.selected;
    }

    /// Whether mod `m` differs from a fresh install (its switch or any setting; not where its
    /// HUD elements sit). Refreshed once per settings generation.
    pub(super) fn is_changed(&self, m: usize) -> bool {
        self.changed.get(m).copied().unwrap_or(false)
    }
}
