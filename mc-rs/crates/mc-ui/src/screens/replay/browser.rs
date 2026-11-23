//! The replay browser: one Aero glass window with two tabs, **Recordings** and **Clips**, a row
//! per file (a tile with its length, the title, then server · date · size), and on every row
//! **Watch** (and for clips a one-click **Export** with the clip settings), **Rename**, **Show in
//! folder** and **Delete** (two clicks: the first arms it, red, for three seconds). Opened from
//! the title screen's Replays button or the `replay` mod's "Replay viewer: Open" button.
//!
//! Simple first: the main action is always visible and blue, the others are icons with
//! tooltips that name their key; a double click on a row watches it; the keyboard works without
//! the mouse (arrows choose, Enter watches, R renames, Delete deletes, Tab switches tabs, Esc
//! goes back). An empty tab says how to fill it.
//!
//! **Folders and order** (the model is [`super::arrange`]): a toolbar under the title bar
//! switches between **List** (every replay of the tab) and **Folders** (the folders on disk, each
//! with its replay count, then the open folder's replays), shows the open folder's path (each
//! part clickable) with an Up button, and holds the **Sort** menu (newest, oldest, name, server,
//! length, size, bookmarks, account). Keys: F switches the view, O opens the Sort menu, Enter or
//! Right opens a folder, Backspace or Left goes up (so does the mouse's back button). The view
//! and the order come back next time ([`BrowserEvent::Arrange`]). The rows are laid out once per
//! change and only the rows on screen are drawn.

use std::collections::HashMap;
use std::time::Instant;

use super::arrange::{
    self, Arrangement, EntryKeys, FolderTree, Headers, Layout, ListView, Row, SortOrder,
};
use super::format;
use super::groups;
use super::icons::{self, Icon};
use super::style::{self, fade, ms_since, styles, Fonts, Hits, Look, Tip, TipClock, ACCENT_TEXT};
use super::{key, LibraryView, ReplayEntry, ReplayEnv};
use crate::screens::modmenu::anim;
use crate::screens::modmenu::glass::{GlassFrame, GlassRect};
use crate::screens::modmenu::paint::{black, css, white, Fill, Painter, Rect};
use crate::screens::modmenu::text::{TextStyle, Weight};
use crate::widget::keys::Modifiers;
use crate::widget::DrawList;

/// The window's design size (1920x1080): 720 tall before the toolbar came, and taller by just
/// the toolbar, so the list kept its height.
const WINDOW_W: f32 = 1040.0;
const WINDOW_H: f32 = 766.0;
const RADIUS: f32 = 25.0;
const TITLE_H: f32 = 78.0;
/// The toolbar under the title bar: the view, the open folder's path and the Sort button.
pub const TOOLBAR_H: f32 = 46.0;
/// A Sort menu item.
const MENU_ITEM_H: f32 = 32.0;
/// The mouse's back button (`mc_render::Input::Mouse(3)`): up one folder, as in Explorer.
const BACK_BUTTON: u8 = 3;
const FOOTER_H: f32 = 66.0;
const PAD: f32 = 18.0;
const ROW_H: f32 = 80.0;
const ROW_GAP: f32 = 6.0;
/// A day's header above its rows.
const HEADER_H: f32 = 34.0;
const THUMB_W: f32 = 112.0;
const THUMB_H: f32 = 63.0;
const BUTTON_H: f32 = 34.0;
/// The longest title a rename keeps.
pub const TITLE_MAX_CHARS: usize = 64;
/// How long an armed Delete waits for its second click.
pub const ARM_MS: f32 = 3000.0;
const DOUBLE_CLICK_MS: f32 = 450.0;
const TOAST_MS: f32 = 2800.0;
/// The browser dims what is behind it (the greenhouse, or the game) a little, as the menu does.
const DIM: f32 = 0.28;
/// Frostier than the menu's 1.5: the busy, bright greenhouse calms behind long rows of text.
const FROST: f32 = 2.4;

/// Which list is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    /// Full recordings.
    Recordings,
    /// Clips.
    Clips,
}

impl Tab {
    /// Its shelf: 0 recordings, 1 clips.
    fn index(self) -> usize {
        match self {
            Tab::Recordings => 0,
            Tab::Clips => 1,
        }
    }
}

/// What the browser asks the client to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BrowserEvent {
    /// Open this replay in the viewer.
    Watch(u64),
    /// Export this clip with the clip settings, now.
    ExportClip(u64),
    /// Give this replay a new title.
    Rename {
        /// Which.
        id: u64,
        /// The new title (trimmed, never empty).
        title: String,
    },
    /// Delete this file (the second click of Delete).
    Delete(u64),
    /// Show this file in Explorer.
    ShowInFolder(u64),
    /// Open the replays folder.
    OpenFolder,
    /// Show this file (an exported video) in Explorer: the message's Show button.
    ShowFile(String),
    /// Leave the browser.
    Close,
    /// The player picked another view or order: remember it (the client keeps it in the
    /// `replay` mod's settings).
    Arrange(Arrangement),
    /// Open one of the library's folders in Explorer: `folder` is its key
    /// ([`ReplayEntry::folder`]) under the clips' root (`clips`) or the recordings' root. The
    /// client opens only a folder its own listing found, never a path made from the key.
    OpenFolderAt {
        /// Under the clips' root.
        clips: bool,
        /// The folder's key (`""`: the root itself).
        folder: String,
    },
}

/// What is selected: a replay (by id) or a folder (by its index in the tab's tree).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pick {
    Entry(u64),
    Folder(usize),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Target {
    Tab(Tab),
    List,
    Row(u64),
    Watch(u64),
    Export(u64),
    Rename(u64),
    Folder(u64),
    Delete(u64),
    RenameField(u64),
    OpenFolder,
    Back,
    /// The message's Show button.
    ToastAction,
    /// Show or hide the short recordings.
    ShortToggle,
    /// A folder's row, its tile and Open button, its Explorer button.
    FolderRow(usize),
    FolderOpen(usize),
    FolderExplorer(usize),
    /// The List / Folders switch.
    View(ListView),
    /// Up one folder.
    Up,
    /// A folder in the open folder's path.
    Crumb(usize),
    /// The Sort button, the open menu's body, one of its items.
    Sort,
    SortMenu,
    SortItem(usize),
}

/// The library's settings (its replays are in the shelves).
#[derive(Debug, Clone, Copy, Default)]
struct Facts {
    cap_bytes: u64,
    recording_on: bool,
    clip_key_set: bool,
}

/// One tab's replays as the browser arranges them.
#[derive(Debug, Default)]
struct Shelf {
    /// Every replay of the tab (short ones too), in the order the client sent them.
    entries: Vec<ReplayEntry>,
    /// In step with `entries`: what the orders compare.
    keys: Vec<EntryKeys>,
    /// In step with `entries`: the time of day (`9:41 PM`, under a day header) and the date
    /// (`Today, 9:41 PM`, in the other orders), made when the library arrives.
    times: Vec<String>,
    whens: Vec<String>,
    /// Entry id to index.
    index: HashMap<u64, usize>,
    /// Every folder the listing found (the tree is rebuilt from these and `entries`).
    folder_keys: Vec<String>,
    tree: FolderTree,
    /// The folder the Folders view shows.
    folder: usize,
}

impl Shelf {
    fn entry(&self, id: u64) -> Option<&ReplayEntry> {
        self.index.get(&id).map(|&i| &self.entries[i])
    }

    /// The index and the tree again after `entries` changed; the open folder stays open (or the
    /// nearest folder above it, when it is gone).
    fn rebuild(&mut self, clips: bool) {
        let open = self.tree.folder(self.folder).key.clone();
        self.index = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id, i))
            .collect();
        self.tree = FolderTree::build(&self.entries, &self.folder_keys, clips);
        self.folder = self.tree.nearest(&open);
    }

    /// The open folder.
    fn open(&self) -> &arrange::Folder {
        self.tree.folder(self.folder)
    }
}

#[derive(Debug, Clone)]
struct Rename {
    id: u64,
    text: String,
    /// Ctrl+A (or the start of the rename) selected the whole title: typing replaces it.
    all: bool,
}

/// The browser's state. `Ui::ReplayBrowser` in the client holds one.
#[derive(Debug)]
pub struct ReplayBrowserScreen {
    opened_at: Instant,
    last_frame: Option<Instant>,
    tab: Tab,
    /// The library's settings; `None` while the client is still reading the library.
    library: Option<Facts>,
    /// Recordings and clips, by [`Tab::index`].
    shelves: [Shelf; 2],
    /// The view and the order.
    arrangement: Arrangement,
    /// The folder keys each tab opens on when the library arrives (the client's memory of
    /// where the player was this session).
    start_folders: [Option<String>; 2],
    /// The rows of the tab showing, laid out for `layout_for` (`None`: laid out again before
    /// the next use).
    layout: Layout,
    layout_for: Option<Tab>,
    /// Short recordings folded away in the rows showing (the empty state's hint).
    hidden_short_here: usize,
    /// How many times the rows were laid out (tests: never per frame).
    layouts_made: u64,
    /// The Sort menu is open, with this item highlighted.
    sort_menu: Option<usize>,
    selected: Option<Pick>,
    scroll: f32,
    scroll_target: f32,
    max_scroll: f32,
    /// The list's visible height in design pixels (last frame), for keyboard scrolling.
    list_h: f32,
    renaming: Option<Rename>,
    delete_armed: Option<(u64, Instant)>,
    hits: Hits<Target>,
    tips: TipClock,
    cursor: (f32, f32),
    pressed: Option<Target>,
    last_click: Option<(Pick, Instant)>,
    toast: Option<(String, bool, Instant)>,
    /// The message's file (its Show button shows it in Explorer).
    toast_file: Option<String>,
    events: Vec<BrowserEvent>,
    /// The selection moved by key: scroll it into view on the next frame.
    reveal: bool,
    /// The Back button's label ("Back", or "Back to game" when opened in game).
    back_label: &'static str,
    /// Opened while connected to this server: Watch and Export leave it, so they need a second
    /// click (a stray click never drops the player out of a game).
    leave_first: Option<String>,
    /// Watch or Export of this entry was clicked once at this time.
    watch_armed: Option<(Target, Instant)>,
    /// Short recordings are folded away unless `show_short`;
    /// `short_count` of them in the recordings tab.
    show_short: bool,
    short_count: usize,
    /// Today, from when the library arrived (the day headers).
    today: Option<chrono::NaiveDate>,
}

impl ReplayBrowserScreen {
    /// A browser whose window pops in from `now`, showing `Loading…` until
    /// [`ReplayBrowserScreen::set_library`].
    #[must_use]
    pub fn new(now: Instant) -> ReplayBrowserScreen {
        ReplayBrowserScreen {
            opened_at: now,
            last_frame: None,
            tab: Tab::Recordings,
            library: None,
            shelves: [Shelf::default(), Shelf::default()],
            arrangement: Arrangement::default(),
            start_folders: [None, None],
            layout: Layout::default(),
            layout_for: None,
            hidden_short_here: 0,
            layouts_made: 0,
            sort_menu: None,
            selected: None,
            scroll: 0.0,
            scroll_target: 0.0,
            max_scroll: 0.0,
            list_h: 400.0,
            renaming: None,
            delete_armed: None,
            hits: Hits::default(),
            tips: TipClock::default(),
            cursor: (-1.0, -1.0),
            pressed: None,
            last_click: None,
            toast: None,
            toast_file: None,
            events: Vec::new(),
            reveal: false,
            back_label: "Back",
            leave_first: None,
            watch_armed: None,
            show_short: false,
            short_count: 0,
            today: None,
        }
    }

    /// Opened in game, connected to `server`: watching or exporting leaves it, so those buttons
    /// ask for a second click, and the Back button says "Back to game".
    pub fn set_in_game(&mut self, server: impl Into<String>) {
        self.leave_first = Some(server.into());
        self.back_label = "Back to game";
    }

    /// Open on the clips tab instead.
    #[must_use]
    pub fn on_tab(mut self, tab: Tab) -> ReplayBrowserScreen {
        self.tab = tab;
        self
    }

    /// The Back button's label (for example "Back to game").
    pub fn set_back_label(&mut self, label: &'static str) {
        self.back_label = label;
    }

    /// The tab showing.
    #[must_use]
    pub fn tab(&self) -> Tab {
        self.tab
    }

    /// The selected entry (`None` when nothing, or a folder, is selected).
    #[must_use]
    pub fn selected(&self) -> Option<u64> {
        match self.selected {
            Some(Pick::Entry(id)) => Some(id),
            _ => None,
        }
    }

    /// Select `id` (and show its tab; in the Folders view, the folder it is in), for example the
    /// replay the viewer just closed.
    pub fn select(&mut self, id: u64) {
        // A folded short recording (the one just watched, say) is shown again.
        if !self.show_short && self.shelves[0].entry(id).is_some_and(groups::is_short) {
            self.set_show_short(true);
        }
        let tab = if self.shelves[1].index.contains_key(&id) {
            Some(Tab::Clips)
        } else if self.shelves[0].index.contains_key(&id) {
            Some(Tab::Recordings)
        } else {
            None
        };
        if let Some(tab) = tab {
            if self.tab != tab {
                self.tab = tab;
                self.sort_menu = None;
            }
            let shelf = &mut self.shelves[tab.index()];
            if let Some(folder) = shelf.entry(id).map(|e| shelf.tree.nearest(&e.folder)) {
                shelf.folder = folder;
            }
            self.layout_for = None;
        }
        self.selected = Some(Pick::Entry(id));
        self.reveal = true;
    }

    /// The view and the order showing.
    #[must_use]
    pub fn arrangement(&self) -> Arrangement {
        self.arrangement
    }

    /// Show this view and order (what the player picked last time). Sends no event.
    pub fn set_arrangement(&mut self, arrangement: Arrangement) {
        if self.arrangement == arrangement {
            return;
        }
        self.arrangement = arrangement;
        self.layout_for = None;
        if self.library.is_some() {
            self.ensure_layout();
            if !self.selection_listed() {
                self.selected = self.first_pick();
            }
        }
    }

    /// Open the tabs on these folders (keys, see [`ReplayEntry::folder`]) when the library
    /// arrives: where the player was earlier this session. A folder that is gone opens the
    /// nearest one above it.
    pub fn set_start_folders(&mut self, recordings: Option<String>, clips: Option<String>) {
        self.start_folders = [recordings, clips];
    }

    /// The key of the folder the Folders view shows in `tab` (`""`: its root).
    #[must_use]
    pub fn folder_key(&self, tab: Tab) -> &str {
        &self.shelves[tab.index()].open().key
    }

    /// The selected folder's key, when a folder is selected.
    #[must_use]
    pub fn selected_folder(&self) -> Option<&str> {
        match self.selected {
            Some(Pick::Folder(f)) => self.shelf().tree.get(f).map(|f| f.key.as_str()),
            _ => None,
        }
    }

    /// The open folder's path as the toolbar shows it: the tab's name, then each folder's.
    #[must_use]
    pub fn folder_path(&self) -> Vec<String> {
        let shelf = self.shelf();
        shelf
            .tree
            .chain(shelf.folder)
            .into_iter()
            .map(|f| match shelf.tree.folder(f).parent {
                None => self.tab_name().to_owned(),
                Some(_) => shelf.tree.folder(f).name.clone(),
            })
            .collect()
    }

    /// The rows showing, top to bottom, as `(is a folder, its name or the replay's title)`
    /// (tests; lays the list out first if it changed).
    pub fn rows(&mut self) -> Vec<(bool, String)> {
        self.ensure_layout();
        let shelf = self.shelf();
        self.layout
            .picks()
            .filter_map(|row| match row {
                Row::Folder(f) => Some((true, shelf.tree.folder(f).name.clone())),
                Row::Entry(i) => shelf.entries.get(i).map(|e| (false, e.title.clone())),
                Row::Header(_) => None,
            })
            .collect()
    }

    /// The header labels showing, top to bottom (tests).
    pub fn headers(&mut self) -> Vec<String> {
        self.ensure_layout();
        self.layout
            .headers
            .iter()
            .map(|h| h.label.clone())
            .collect()
    }

    fn tab_name(&self) -> &'static str {
        match self.tab {
            Tab::Recordings => "Recordings",
            Tab::Clips => "Clips",
        }
    }

    fn shelf(&self) -> &Shelf {
        &self.shelves[self.tab.index()]
    }

    /// Whether a title is being renamed (the client sends typed text here).
    #[must_use]
    pub fn renaming(&self) -> bool {
        self.renaming.is_some()
    }

    /// The events since the last call.
    pub fn drain_events(&mut self) -> std::vec::Drain<'_, BrowserEvent> {
        self.events.drain(..)
    }

    /// Show the library, dates in the computer's own time zone.
    pub fn set_library(&mut self, library: LibraryView, now_unix_ms: u64) {
        self.set_library_in_zone(library, now_unix_ms, &chrono::Local);
    }

    /// [`ReplayBrowserScreen::set_library`] with an explicit zone (tests).
    pub fn set_library_in_zone<Tz: chrono::TimeZone>(
        &mut self,
        library: LibraryView,
        now_unix_ms: u64,
        tz: &Tz,
    ) {
        let LibraryView {
            recordings,
            clips,
            recording_folders,
            clip_folders,
            cap_bytes,
            recording_on,
            clip_key_set,
        } = library;
        self.today = groups::day_of(now_unix_ms, tz);
        // A selected folder is found again by its key in the new tree.
        let selected_folder = self.selected_folder().map(str::to_owned);
        for (tab, entries, folder_keys) in [
            (Tab::Recordings, recordings, recording_folders),
            (Tab::Clips, clips, clip_folders),
        ] {
            let t = tab.index();
            // Where the tab was (or where the client asked it to open).
            let open = self.start_folders[t]
                .take()
                .unwrap_or_else(|| self.shelves[t].open().key.clone());
            // Under a day header a row says only its time; in an order without day headers it
            // says the date too.
            let keys = entries
                .iter()
                .map(|e| EntryKeys::new(e, groups::day_of(e.started_unix_ms, tz)))
                .collect();
            let times = entries
                .iter()
                .map(|e| groups::time_of_day(e.started_unix_ms, tz))
                .collect();
            let whens = entries
                .iter()
                .map(|e| format::when(e.started_unix_ms, now_unix_ms, tz))
                .collect();
            let mut shelf = Shelf {
                entries,
                keys,
                times,
                whens,
                folder_keys,
                ..Shelf::default()
            };
            shelf.rebuild(tab == Tab::Clips);
            shelf.folder = shelf.tree.nearest(&open);
            self.shelves[t] = shelf;
        }
        self.short_count = self.shelves[0]
            .entries
            .iter()
            .filter(|e| groups::is_short(e))
            .count();
        self.library = Some(Facts {
            cap_bytes,
            recording_on,
            clip_key_set,
        });
        self.last_click = None;
        self.layout_for = None;
        self.selected = match self.selected {
            Some(Pick::Folder(_)) => selected_folder
                .and_then(|key| self.shelf().tree.find(&key))
                .map(Pick::Folder),
            other => other,
        };
        self.ensure_layout();
        if !self.selection_listed() {
            self.selected = self.first_pick();
        }
        if let Some(rename) = &self.renaming {
            if !self.listed(Pick::Entry(rename.id)) {
                self.renaming = None;
            }
        }
    }

    /// Whether short recordings (under 20 s, no bookmark) are shown; folded away by default.
    #[must_use]
    pub fn showing_short(&self) -> bool {
        self.show_short
    }

    /// Show or fold away the short recordings ([`groups::is_short`]). The selection stays when
    /// it is still in the list.
    pub fn set_show_short(&mut self, show: bool) {
        self.show_short = show;
        self.layout_for = None;
        if self.tab == Tab::Recordings {
            self.ensure_layout();
            if !self.selection_listed() {
                self.selected = self.first_pick();
            }
            self.reveal = true;
        }
    }

    // ---- the rows ----------------------------------------------------------------------------

    /// Lay the tab's rows out again if anything changed since the last time: the open folder's
    /// sub-folders (Folders view), then its replays (or, in the List view, every replay of the
    /// tab), short ones folded away unless shown, in the chosen order under its headers.
    fn ensure_layout(&mut self) {
        if self.layout_for == Some(self.tab) {
            return;
        }
        let order = self.arrangement.sort;
        let show_short = self.show_short;
        let shelf = &self.shelves[self.tab.index()];
        let shown = |i: &usize| show_short || !groups::is_short(&shelf.entries[*i]);
        let (mut folders, candidates): (Vec<usize>, Vec<usize>) = match self.arrangement.view {
            ListView::List => (Vec::new(), (0..shelf.entries.len()).collect()),
            ListView::Folders => (shelf.open().children.clone(), shelf.open().entries.clone()),
        };
        let mut sorted: Vec<usize> = candidates.iter().copied().filter(shown).collect();
        self.hidden_short_here = candidates.len() - sorted.len();
        arrange::sort_folders(&mut folders, &shelf.tree, order);
        arrange::sort_entries(&mut sorted, &shelf.entries, &shelf.keys, order);
        self.layout = arrange::layout(
            &folders,
            &sorted,
            &shelf.entries,
            &shelf.keys,
            order,
            self.tab == Tab::Clips,
            self.today,
            (HEADER_H, ROW_H, ROW_GAP),
        );
        self.layout_for = Some(self.tab);
        self.layouts_made += 1;
    }

    /// How many times the rows have been laid out (tests: a frame lays out nothing, only a
    /// change does).
    #[must_use]
    pub fn layouts_made(&self) -> u64 {
        self.layouts_made
    }

    /// The selection a row stands for.
    fn pick_of(&self, row: Row) -> Option<Pick> {
        match row {
            Row::Entry(i) => self.shelf().entries.get(i).map(|e| Pick::Entry(e.id)),
            Row::Folder(f) => Some(Pick::Folder(f)),
            Row::Header(_) => None,
        }
    }

    /// The row a selection would be (in the tab showing).
    fn row_of(&self, pick: Pick) -> Option<Row> {
        match pick {
            Pick::Entry(id) => self.shelf().index.get(&id).map(|&i| Row::Entry(i)),
            Pick::Folder(f) => Some(Row::Folder(f)),
        }
    }

    /// Whether `pick` is one of the rows laid out.
    fn listed(&self, pick: Pick) -> bool {
        self.row_of(pick)
            .is_some_and(|row| self.layout.rows.iter().any(|(r, _, _)| *r == row))
    }

    /// Whether the selection is one of the rows laid out.
    fn selection_listed(&self) -> bool {
        self.selected.is_some_and(|pick| self.listed(pick))
    }

    /// The first row that can be selected.
    fn first_pick(&self) -> Option<Pick> {
        self.layout.picks().find_map(|row| self.pick_of(row))
    }

    /// The rows that can be selected, top to bottom.
    fn picks(&self) -> Vec<Pick> {
        self.layout
            .picks()
            .filter_map(|row| self.pick_of(row))
            .collect()
    }

    /// Open folder `index` of the tab showing (the Folders view); its first row is selected.
    fn open_folder(&mut self, index: usize) {
        let t = self.tab.index();
        if self.shelves[t].tree.get(index).is_none() {
            return;
        }
        self.commit_rename();
        self.shelves[t].folder = index;
        self.layout_for = None;
        self.scroll = 0.0;
        self.scroll_target = 0.0;
        self.delete_armed = None;
        self.watch_armed = None;
        self.ensure_layout();
        self.selected = self.first_pick();
        self.reveal = true;
    }

    /// Open `index`, an ancestor of the open folder (a click on the path), with the folder the
    /// player came from selected, as Explorer does.
    fn open_ancestor(&mut self, index: usize) {
        let shelf = self.shelf();
        let chain = shelf.tree.chain(shelf.folder);
        let came_from = chain
            .iter()
            .position(|&f| f == index)
            .and_then(|at| chain.get(at + 1))
            .copied();
        self.open_folder(index);
        if let Some(child) = came_from {
            if self.listed(Pick::Folder(child)) {
                self.selected = Some(Pick::Folder(child));
                self.reveal = true;
            }
        }
    }

    /// Up one folder (the Folders view). Returns whether there was one above.
    fn folder_up(&mut self) -> bool {
        if self.arrangement.view != ListView::Folders {
            return false;
        }
        let Some(parent) = self.shelf().open().parent else {
            return false;
        };
        self.open_ancestor(parent);
        true
    }

    /// Switch between the List and Folders views. A selected replay stays selected: in the
    /// Folders view its folder opens, so the switch also shows where a replay is on disk.
    fn set_view(&mut self, view: ListView) {
        self.sort_menu = None;
        if self.arrangement.view == view {
            return;
        }
        self.commit_rename();
        self.arrangement.view = view;
        if let (ListView::Folders, Some(Pick::Entry(id))) = (view, self.selected) {
            let t = self.tab.index();
            let shelf = &mut self.shelves[t];
            if let Some(folder) = shelf.entry(id).map(|e| shelf.tree.nearest(&e.folder)) {
                shelf.folder = folder;
            }
        }
        self.layout_for = None;
        self.ensure_layout();
        if !self.selection_listed() {
            self.selected = self.first_pick();
            self.scroll = 0.0;
            self.scroll_target = 0.0;
        }
        self.reveal = true;
        self.events.push(BrowserEvent::Arrange(self.arrangement));
    }

    /// Sort by `order`; the selection stays (and in view).
    fn set_sort(&mut self, order: SortOrder) {
        self.sort_menu = None;
        if self.arrangement.sort == order {
            return;
        }
        self.commit_rename();
        self.arrangement.sort = order;
        self.layout_for = None;
        self.ensure_layout();
        if !self.selection_listed() {
            self.selected = self.first_pick();
        }
        self.reveal = true;
        self.events.push(BrowserEvent::Arrange(self.arrangement));
    }

    /// Open the Sort menu (on the current order), or close it.
    fn toggle_sort_menu(&mut self) {
        self.sort_menu = match self.sort_menu {
            Some(_) => None,
            None => Some(self.arrangement.sort.index()),
        };
    }

    /// Ask the client to open folder `index` of the tab showing in Explorer.
    fn open_in_explorer(&mut self, index: usize) {
        let Some(folder) = self.shelf().tree.get(index) else {
            return;
        };
        let event = BrowserEvent::OpenFolderAt {
            clips: self.tab == Tab::Clips,
            folder: folder.key.clone(),
        };
        self.events.push(event);
    }

    /// A short message at the bottom of the window (an error is red).
    pub fn show_message(&mut self, text: impl Into<String>, error: bool, now: Instant) {
        self.toast = Some((text.into(), error, now));
        self.toast_file = None;
    }

    /// A message about a file just written (an exported video), with a Show button that shows
    /// it in Explorer. It stays a little longer than a plain message.
    pub fn show_file_message(
        &mut self,
        text: impl Into<String>,
        file: impl Into<String>,
        now: Instant,
    ) {
        self.toast = Some((text.into(), false, now));
        self.toast_file = Some(file.into());
    }

    /// The tooltip label of the control under `(x, y)` in the last frame (what the pointer
    /// would explain there; the tests find controls by it).
    #[must_use]
    pub fn tooltip_at(&self, x: f32, y: f32) -> Option<&'static str> {
        self.hits
            .entry_at(x, y)
            .and_then(|(_, _, tip)| tip.map(|t| t.label))
    }

    fn entry(&self, id: u64) -> Option<&ReplayEntry> {
        self.shelves.iter().find_map(|shelf| shelf.entry(id))
    }

    fn set_tab(&mut self, tab: Tab) {
        if self.tab != tab {
            self.commit_rename();
            self.tab = tab;
            self.layout_for = None;
            self.scroll = 0.0;
            self.scroll_target = 0.0;
            self.ensure_layout();
            self.selected = self.first_pick();
            self.delete_armed = None;
            self.sort_menu = None;
        }
    }

    // ---- input ----------------------------------------------------------------------------

    /// The pointer moved to `(x, y)` (physical pixels).
    pub fn pointer_moved(&mut self, x: f32, y: f32) {
        self.cursor = (x, y);
    }

    /// A mouse button went down. Returns whether it landed on the window.
    pub fn mouse_pressed(&mut self, x: f32, y: f32, button: u8) -> bool {
        self.cursor = (x, y);
        self.tips.suppress();
        let target = self.hits.at(x, y);
        // The mouse's back button goes up a folder, as in Explorer.
        if button == BACK_BUTTON && self.renaming.is_none() {
            self.sort_menu = None;
            self.folder_up();
            return target.is_some();
        }
        if button != 0 {
            return target.is_some();
        }
        // An open Sort menu closes on a click anywhere else, and that click does nothing more.
        if self.sort_menu.is_some()
            && !matches!(
                target,
                Some(Target::Sort | Target::SortMenu | Target::SortItem(_))
            )
        {
            self.sort_menu = None;
            self.pressed = None;
            return target.is_some();
        }
        // A click anywhere but the field ends a rename (saved, as Explorer does).
        if let Some(rename) = &self.renaming {
            if target != Some(Target::RenameField(rename.id)) {
                self.commit_rename();
            }
        }
        if let Some((armed, _)) = self.delete_armed {
            if target != Some(Target::Delete(armed)) {
                self.delete_armed = None;
            }
        }
        if let Some((armed, _)) = self.watch_armed {
            if target != Some(armed) {
                self.watch_armed = None;
            }
        }
        self.pressed = target;
        target.is_some()
    }

    /// A mouse button went up: a press released over the same control activates it.
    pub fn mouse_released(&mut self, x: f32, y: f32, button: u8, now: Instant) {
        self.cursor = (x, y);
        if button != 0 {
            return;
        }
        let Some(pressed) = self.pressed.take() else {
            return;
        };
        if self.hits.at(x, y) == Some(pressed) {
            self.activate(pressed, now);
        }
    }

    /// The wheel turned (`delta` > 0 is up) with the pointer at `(x, y)`.
    pub fn scrolled(&mut self, x: f32, y: f32, delta: f32) {
        if self.sort_menu.is_none()
            && self
                .hits
                .rect_of(Target::List)
                .is_some_and(|r| r.contains(x, y))
        {
            self.scroll_target = (self.scroll_target - delta * 3.0 * (ROW_H + ROW_GAP) * 0.5)
                .clamp(0.0, self.max_scroll);
        }
    }

    /// Paste text into the rename field. Returns whether it was taken.
    pub fn paste(&mut self, text: &str) -> bool {
        let Some(rename) = self.renaming.as_mut() else {
            return false;
        };
        if rename.all {
            rename.text.clear();
            rename.all = false;
        }
        for c in text.chars().filter(|c| !c.is_control()) {
            if rename.text.chars().count() >= TITLE_MAX_CHARS {
                break;
            }
            rename.text.push(c);
        }
        true
    }

    /// A key went down (`typed` is its character). Returns whether the browser used it.
    pub fn key(&mut self, code: i32, typed: Option<char>, mods: Modifiers, now: Instant) -> bool {
        if let Some(rename) = self.renaming.as_mut() {
            match code {
                key::ESCAPE => self.renaming = None,
                key::ENTER | key::NUMPAD_ENTER => self.commit_rename(),
                key::BACKSPACE => {
                    if rename.all {
                        rename.text.clear();
                        rename.all = false;
                    } else {
                        rename.text.pop();
                    }
                }
                key::A if mods.ctrl => rename.all = true,
                _ => {
                    if let Some(c) = typed.filter(|c| !c.is_control()) {
                        if !mods.ctrl {
                            if rename.all {
                                rename.text.clear();
                                rename.all = false;
                            }
                            if rename.text.chars().count() < TITLE_MAX_CHARS {
                                rename.text.push(c);
                            }
                        }
                    }
                }
            }
            return true;
        }
        // The open Sort menu takes the keys (arrows choose, Enter or Space picks, anything else
        // closes it).
        if let Some(at) = self.sort_menu {
            let last = SortOrder::ALL.len() - 1;
            match code {
                key::UP => self.sort_menu = Some(at.saturating_sub(1)),
                key::DOWN => self.sort_menu = Some((at + 1).min(last)),
                key::HOME | key::PAGE_UP => self.sort_menu = Some(0),
                key::END | key::PAGE_DOWN => self.sort_menu = Some(last),
                key::ENTER | key::NUMPAD_ENTER | key::SPACE => {
                    self.set_sort(SortOrder::ALL[at.min(last)]);
                }
                _ => self.sort_menu = None,
            }
            return true;
        }
        let folders = self.arrangement.view == ListView::Folders;
        match code {
            key::ESCAPE => {
                if self.delete_armed.take().is_none() {
                    self.events.push(BrowserEvent::Close);
                }
            }
            key::TAB => self.set_tab(match self.tab {
                Tab::Recordings => Tab::Clips,
                Tab::Clips => Tab::Recordings,
            }),
            key::F => self.set_view(match self.arrangement.view {
                ListView::List => ListView::Folders,
                ListView::Folders => ListView::List,
            }),
            key::O => self.toggle_sort_menu(),
            key::BACKSPACE | key::LEFT if folders => {
                self.folder_up();
            }
            key::RIGHT if folders => {
                if let Some(Pick::Folder(f)) = self.selected {
                    self.open_folder(f);
                }
            }
            key::UP | key::DOWN | key::HOME | key::END | key::PAGE_UP | key::PAGE_DOWN => {
                let page = (self.list_h / (ROW_H + ROW_GAP)).floor().max(1.0) as isize;
                let delta = match code {
                    key::UP => -1,
                    key::DOWN => 1,
                    key::PAGE_UP => -page,
                    key::PAGE_DOWN => page,
                    key::HOME => isize::MIN / 2,
                    _ => isize::MAX / 2,
                };
                self.move_selection(delta);
            }
            key::ENTER | key::NUMPAD_ENTER => match self.selected {
                Some(Pick::Entry(id)) => self.activate(Target::Watch(id), now),
                Some(Pick::Folder(f)) => self.open_folder(f),
                None => {}
            },
            key::F2 | key::R => {
                if let Some(id) = self.selected() {
                    self.activate(Target::Rename(id), now);
                }
            }
            key::DELETE => {
                if let Some(id) = self.selected() {
                    self.activate(Target::Delete(id), now);
                }
            }
            key::E => {
                if let Some(id) = self.selected() {
                    if self.entry(id).is_some_and(|e| e.clip) {
                        self.activate(Target::Export(id), now);
                    }
                }
            }
            key::S if self.tab == Tab::Recordings => self.activate(Target::ShortToggle, now),
            _ => return false,
        }
        true
    }

    fn move_selection(&mut self, delta: isize) {
        self.ensure_layout();
        let picks = self.picks();
        if picks.is_empty() {
            return;
        }
        let current = self
            .selected
            .and_then(|pick| picks.iter().position(|p| *p == pick));
        let next = match current {
            None => 0,
            Some(i) => (i as isize)
                .saturating_add(delta)
                .clamp(0, picks.len() as isize - 1) as usize,
        };
        self.selected = Some(picks[next]);
        self.delete_armed = None;
        self.reveal = true;
    }

    fn commit_rename(&mut self) {
        let Some(rename) = self.renaming.take() else {
            return;
        };
        let title: String = rename.text.trim().to_owned();
        if title.is_empty() {
            return;
        }
        for shelf in &mut self.shelves {
            let Some(&i) = shelf.index.get(&rename.id) else {
                continue;
            };
            if shelf.entries[i].title != title {
                shelf.entries[i].title.clone_from(&title);
                shelf.keys[i].title = title.to_lowercase();
                // The Name order may move it.
                self.layout_for = None;
                self.reveal = true;
                self.events.push(BrowserEvent::Rename {
                    id: rename.id,
                    title,
                });
            }
            return;
        }
    }

    fn activate(&mut self, target: Target, now: Instant) {
        match target {
            Target::Tab(tab) => self.set_tab(tab),
            Target::List | Target::RenameField(_) => {}
            Target::Row(id) => {
                let pick = Pick::Entry(id);
                let double = self
                    .last_click
                    .is_some_and(|(last, at)| last == pick && ms_since(at, now) <= DOUBLE_CLICK_MS);
                self.selected = Some(pick);
                if double {
                    self.last_click = None;
                    self.activate(Target::Watch(id), now);
                } else {
                    self.last_click = Some((pick, now));
                }
            }
            // Folders, the view, the path and the Sort menu.
            Target::FolderRow(f) => {
                let pick = Pick::Folder(f);
                let double = self
                    .last_click
                    .is_some_and(|(last, at)| last == pick && ms_since(at, now) <= DOUBLE_CLICK_MS);
                self.selected = Some(pick);
                if double {
                    self.last_click = None;
                    self.open_folder(f);
                } else {
                    self.last_click = Some((pick, now));
                }
            }
            Target::FolderOpen(f) => self.open_folder(f),
            Target::FolderExplorer(f) => {
                self.selected = Some(Pick::Folder(f));
                self.open_in_explorer(f);
            }
            Target::View(view) => self.set_view(view),
            Target::Up => {
                self.folder_up();
            }
            Target::Crumb(f) => self.open_ancestor(f),
            Target::Sort => self.toggle_sort_menu(),
            Target::SortMenu => {}
            Target::SortItem(i) => {
                if let Some(order) = SortOrder::ALL.get(i) {
                    self.set_sort(*order);
                }
            }
            Target::Watch(id) => {
                self.selected = Some(Pick::Entry(id));
                match self.entry(id) {
                    Some(entry) if entry.problem.is_none() => {
                        if self.confirm_leave(target, now) {
                            self.events.push(BrowserEvent::Watch(id));
                        }
                    }
                    Some(entry) => {
                        let text = format!(
                            "Can't open this replay: {}",
                            entry.problem.clone().unwrap_or_default()
                        );
                        self.show_message(text, true, now);
                    }
                    None => {}
                }
            }
            Target::Export(id) => {
                self.selected = Some(Pick::Entry(id));
                if self
                    .entry(id)
                    .is_some_and(|e| e.clip && e.problem.is_none())
                    && self.confirm_leave(target, now)
                {
                    self.events.push(BrowserEvent::ExportClip(id));
                }
            }
            Target::Rename(id) => {
                self.selected = Some(Pick::Entry(id));
                if let Some(entry) = self.entry(id) {
                    self.renaming = Some(Rename {
                        id,
                        text: entry.title.clone(),
                        all: true,
                    });
                }
            }
            Target::Folder(id) => {
                self.selected = Some(Pick::Entry(id));
                self.events.push(BrowserEvent::ShowInFolder(id));
            }
            Target::Delete(id) => {
                self.selected = Some(Pick::Entry(id));
                let armed = self
                    .delete_armed
                    .is_some_and(|(armed, at)| armed == id && ms_since(at, now) < ARM_MS);
                if armed {
                    self.delete_armed = None;
                    self.delete_now(id, now);
                } else {
                    self.delete_armed = Some((id, now));
                }
            }
            // The footer's Open folder: the open folder in the Folders view, else the tab's
            // root (the replays folder, or the clips folder).
            Target::OpenFolder => match (self.arrangement.view, self.tab) {
                (ListView::Folders, _) => self.open_in_explorer(self.shelf().folder),
                (ListView::List, Tab::Clips) => self.open_in_explorer(0),
                (ListView::List, Tab::Recordings) => self.events.push(BrowserEvent::OpenFolder),
            },
            Target::Back => self.events.push(BrowserEvent::Close),
            Target::ToastAction => {
                if let Some(file) = self.toast_file.clone() {
                    self.events.push(BrowserEvent::ShowFile(file));
                }
            }
            Target::ShortToggle => {
                if self.short_count > 0 {
                    self.set_show_short(!self.show_short);
                    let n = self.short_count;
                    let text = if self.show_short {
                        format!(
                            "Showing {n} short recording{}",
                            if n == 1 { "" } else { "s" }
                        )
                    } else {
                        format!(
                            "{n} short recording{} hidden (under 20 s)",
                            if n == 1 { "" } else { "s" }
                        )
                    };
                    self.show_message(text, false, now);
                }
            }
        }
    }

    /// How long the current message stays: longer for a written video (it has a button).
    fn toast_ms(&self) -> f32 {
        if self.toast_file.is_some() {
            TOAST_MS * 2.5
        } else {
            TOAST_MS
        }
    }

    /// In game, Watch and Export leave the server: the first click arms the button (and says
    /// so), the second within [`ARM_MS`] goes ahead. Returns whether to go ahead. Out of a game,
    /// always.
    fn confirm_leave(&mut self, target: Target, now: Instant) -> bool {
        let Some(server) = self.leave_first.clone() else {
            return true;
        };
        let armed = self
            .watch_armed
            .is_some_and(|(armed, at)| armed == target && ms_since(at, now) < ARM_MS);
        if armed {
            self.watch_armed = None;
            return true;
        }
        self.watch_armed = Some((target, now));
        let place = if server.is_empty() {
            "the server".to_owned()
        } else {
            server
        };
        self.show_message(
            format!("This leaves {place} \u{b7} click again to go ahead"),
            false,
            now,
        );
        false
    }

    /// The second Delete click: the row goes at once (the client deletes the file and refreshes;
    /// a failure comes back as a red message and the row returns with the refresh).
    fn delete_now(&mut self, id: u64, now: Instant) {
        let title = self.entry(id).map(|e| e.title.clone()).unwrap_or_default();
        self.ensure_layout();
        let index = self.picks().iter().position(|p| *p == Pick::Entry(id));
        // Its shelf's tree is rebuilt (folder indices may change), so the selection is picked
        // again below, by its place in the list.
        for (t, shelf) in self.shelves.iter_mut().enumerate() {
            let Some(&i) = shelf.index.get(&id) else {
                continue;
            };
            shelf.entries.remove(i);
            shelf.keys.remove(i);
            shelf.times.remove(i);
            shelf.whens.remove(i);
            shelf.rebuild(t == 1);
            break;
        }
        self.short_count = self.shelves[0]
            .entries
            .iter()
            .filter(|e| groups::is_short(e))
            .count();
        self.last_click = None;
        self.layout_for = None;
        self.events.push(BrowserEvent::Delete(id));
        self.ensure_layout();
        let picks = self.picks();
        self.selected = index
            .and_then(|i| picks.get(i).or_else(|| picks.last()))
            .copied();
        self.show_message(format!("Deleted \u{201c}{title}\u{201d}"), false, now);
    }

    // ---- drawing --------------------------------------------------------------------------

    /// The window's rectangle for a `screen`-sized window, before the pop-in's rise.
    #[must_use]
    pub fn window_rect(screen: (f32, f32)) -> Rect {
        let s = style::scale(screen);
        let dip = (screen.0 / s, screen.1 / s);
        let w = WINDOW_W.min(dip.0 - 64.0).max(520.0) * s;
        let h = WINDOW_H.min(dip.1 - 72.0).max(340.0) * s;
        Rect::new(
            ((screen.0 - w) * 0.5).round(),
            ((screen.1 - h) * 0.5).round(),
            w.round(),
            h.round(),
        )
    }

    /// Draw one frame into `list` and return its glass.
    #[allow(clippy::too_many_lines)]
    pub fn build(&mut self, env: &mut ReplayEnv<'_>, list: &mut DrawList) -> GlassFrame {
        self.hits.clear();
        let now = env.now;
        let dt = self.last_frame.map_or(0.0, |last| {
            now.saturating_duration_since(last).as_secs_f32()
        });
        self.last_frame = Some(now);
        if self
            .delete_armed
            .is_some_and(|(_, at)| ms_since(at, now) >= ARM_MS)
        {
            self.delete_armed = None;
        }
        if self
            .watch_armed
            .is_some_and(|(_, at)| ms_since(at, now) >= ARM_MS)
        {
            self.watch_armed = None;
        }
        let toast_ms = self.toast_ms();
        if self
            .toast
            .as_ref()
            .is_some_and(|(_, _, at)| ms_since(*at, now) > toast_ms)
        {
            self.toast = None;
            self.toast_file = None;
        }
        let s = style::scale(env.screen);
        let open = anim::window_open_progress(ms_since(self.opened_at, now), 0);
        let k = anim::approach(dt, 18.0);
        self.scroll += (self.scroll_target - self.scroll) * k;
        if (self.scroll_target - self.scroll).abs() < 0.05 {
            self.scroll = self.scroll_target;
        }

        let base = Self::window_rect(env.screen);
        let slide = ((1.0 - open) * anim::SLIDE_DIP * s).round();
        let rect = base.offset(0.0, slide);
        let mut glass = GlassFrame {
            px_per_dip: s,
            time: env.time_s,
            dim: DIM * open,
            strength: open,
            frost: FROST,
            rects: Vec::with_capacity(1),
        };
        if open <= 0.0 {
            return glass;
        }
        let radius = RADIUS * s;
        glass.rects.push(GlassRect {
            rect: [rect.x, rect.y, rect.w, rect.h],
            radius,
            alpha: open,
        });
        let mut fonts = Fonts {
            text: &mut *env.text,
            mc: &mut *env.mc_font,
        };
        let mut painter = Painter::new(list);
        painter.alpha = open;
        style::window(&mut painter, rect, radius, TITLE_H * s, s);

        self.draw_title_bar(&mut painter, &mut fonts, rect, s);
        // The toolbar (view, path, Sort) between the title bar and the list.
        let bar = Rect::new(
            rect.x + PAD * s,
            rect.y + (TITLE_H + 8.0) * s,
            rect.w - 2.0 * PAD * s,
            (TOOLBAR_H - 4.0) * s,
        );
        let sort_button = self.draw_toolbar(&mut painter, &mut fonts, bar, s);
        let body = Rect::new(
            rect.x + PAD * s,
            rect.y + (TITLE_H + TOOLBAR_H + 12.0) * s,
            rect.w - 2.0 * PAD * s,
            rect.h - (TITLE_H + TOOLBAR_H + FOOTER_H + 12.0 + 6.0) * s,
        );
        self.draw_list(&mut painter, &mut fonts, body, s, now);
        self.draw_footer(&mut painter, &mut fonts, rect, s);

        // The toast, over the bottom of the list (with a Show button for a written video).
        if let Some((text, error, at)) = &self.toast {
            let age = ms_since(*at, now);
            let alpha = (age / 150.0).min(1.0) * ((toast_ms - age) / 400.0).clamp(0.0, 1.0);
            let icon = if *error {
                (Icon::Warning, style::DANGER)
            } else {
                (Icon::Check, style::SUCCESS)
            };
            let y = rect.bottom() - (FOOTER_H + 54.0) * s;
            let show_w = 84.0 * s;
            let shift = if self.toast_file.is_some() {
                (show_w + 8.0 * s) * 0.5
            } else {
                0.0
            };
            let toast = style::toast(
                &mut painter,
                &mut fonts,
                text,
                Some(icon),
                rect.x + rect.w * 0.5 - shift,
                y,
                alpha,
                s,
            );
            if self.toast_file.is_some() {
                let show = Rect::new(
                    toast.right() + 8.0 * s,
                    toast.y + 2.0 * s,
                    show_w,
                    toast.h - 4.0 * s,
                );
                let saved = painter.alpha;
                painter.alpha = saved * alpha;
                let look = Look {
                    primary: true,
                    ..self.look(show, Target::ToastAction)
                };
                style::pill(
                    &mut painter,
                    &mut fonts,
                    show,
                    Some(Icon::Folder),
                    "Show",
                    look,
                    s,
                );
                painter.alpha = saved;
                self.hits.push(
                    show,
                    Target::ToastAction,
                    Some(Tip::new("Show the video in its folder", None)),
                );
            }
        }

        // The Sort menu, over the list.
        if let Some(highlight) = self.sort_menu {
            self.draw_sort_menu(&mut painter, &mut fonts, sort_button, highlight, s);
        }

        // The tooltip of the control under the pointer.
        let hovered = self
            .hits
            .entry_at(self.cursor.0, self.cursor.1)
            .and_then(|(r, _, tip)| tip.map(|tip| (r, tip)));
        if let Some((anchor, tip, alpha)) = self.tips.update(hovered, now) {
            style::tooltip(
                &mut painter,
                &mut fonts,
                anchor,
                tip.label,
                tip.key,
                env.screen,
                alpha,
                s,
            );
        }
        glass
    }

    fn hover(&self, rect: Rect) -> bool {
        rect.contains(self.cursor.0, self.cursor.1)
    }

    fn look(&self, rect: Rect, target: Target) -> Look {
        let hover = self.hover(rect);
        Look::new(hover, hover && self.pressed == Some(target))
    }

    fn draw_title_bar(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        rect: Rect,
        s: f32,
    ) {
        let pad = 24.0 * s;
        let cy = rect.y + TITLE_H * s * 0.5;
        icons::draw(
            painter,
            Icon::Replay,
            (rect.x + pad + 12.0 * s, cy),
            26.0 * s,
            1.9 * s,
            white(0.95),
        );
        let x = rect.x + pad + 38.0 * s;
        fonts.shadowed(
            painter,
            "Replays",
            x,
            cy - 21.0 * s,
            TextStyle::new(19.0 * s).weight(Weight::Semibold),
            white(1.0),
            s,
        );
        fonts.shadowed(
            painter,
            "Watch, cut and export your games",
            x,
            cy + 3.0 * s,
            styles::subtitle(s),
            white(0.62),
            s,
        );
        // The tabs.
        // Every recording counts, the folded short ones too, in every folder.
        let (recordings, clips) = (self.shelves[0].entries.len(), self.shelves[1].entries.len());
        let labels = [
            format!("Recordings  {recordings}"),
            format!("Clips  {clips}"),
        ];
        let tabs_w = 340.0 * s;
        let tabs = Rect::new(rect.right() - pad - tabs_w, cy - 19.0 * s, tabs_w, 38.0 * s);
        let hover = style::segment_at(tabs, 2, s, self.cursor);
        let selected = match self.tab {
            Tab::Recordings => 0,
            Tab::Clips => 1,
        };
        let segments = style::segmented(
            painter,
            fonts,
            tabs,
            &[
                (Some(Icon::Film), labels[0].as_str()),
                (Some(Icon::Scissors), labels[1].as_str()),
            ],
            selected,
            hover,
            &[],
            s,
        );
        let tips = [
            Tip::new("Full recordings", Some("Tab")),
            Tip::new("Saved clips", Some("Tab")),
        ];
        for ((segment, tab), tip) in segments
            .into_iter()
            .zip([Tab::Recordings, Tab::Clips])
            .zip(tips)
        {
            self.hits.push(segment, Target::Tab(tab), Some(tip));
        }
    }

    #[allow(clippy::too_many_lines)]
    fn draw_list(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        body: Rect,
        s: f32,
        now: Instant,
    ) {
        self.list_h = body.h / s;
        self.hits.push(body, Target::List, None);
        let row_h = ROW_H * s;
        // Folders first, then the rows under the chosen order's headers (a day, a server, an
        // account). Laid out only when something changed.
        self.ensure_layout();
        let count = self.layout.rows.len();
        let content_h = self.layout.height;
        self.max_scroll = (content_h - self.list_h).max(0.0);
        // Keep the keyboard's selection in view (its header too, when it is the first under one).
        if std::mem::take(&mut self.reveal) {
            if let Some(row_top) = self
                .selected
                .and_then(|pick| self.row_of(pick))
                .and_then(|row| self.layout.top_of(row))
            {
                let top = if self.layout.header_above(row_top) {
                    row_top - HEADER_H
                } else {
                    row_top
                };
                let bottom = row_top + ROW_H;
                if top < self.scroll_target {
                    self.scroll_target = top;
                } else if bottom > self.scroll_target + self.list_h {
                    self.scroll_target = bottom - self.list_h;
                }
            }
        }
        self.scroll_target = self.scroll_target.clamp(0.0, self.max_scroll);
        self.scroll = self
            .scroll
            .clamp(0.0, self.max_scroll.max(self.scroll_target));

        let Some(library) = self.library else {
            let style = styles::body(s);
            let text = "Loading replays\u{2026}";
            let w = fonts.width(text, style);
            fonts.shadowed_mid(
                painter,
                text,
                body.x + (body.w - w) * 0.5,
                body.y + body.h * 0.45,
                style,
                white(0.75),
                s,
            );
            return;
        };
        if count == 0 {
            // A folder of its own (not a tab's root) that is empty.
            let in_folder = self.arrangement.view == ListView::Folders && self.shelf().folder != 0;
            let (icon, title, hint) = match self.tab {
                // Only short ones, folded away.
                Tab::Recordings if self.hidden_short_here > 0 && !self.show_short => (
                    Icon::Film,
                    "Only short recordings",
                    "Recordings under 20 s are folded away: press S or Show short to list them.",
                ),
                _ if in_folder => (
                    Icon::Folder,
                    "This folder is empty",
                    "Replays you move into this folder in Explorer show up here.",
                ),
                Tab::Recordings if self.short_count > 0 && !self.show_short => (
                    Icon::Film,
                    "Only short recordings",
                    "Recordings under 20 s are folded away: press S or Show short to list them.",
                ),
                Tab::Recordings if library.recording_on => (
                    Icon::Film,
                    "No recordings yet",
                    "Your next game will be recorded and show up here.",
                ),
                Tab::Recordings => (
                    Icon::Film,
                    "No recordings yet",
                    "Turn on Replay in the mod menu (Right Shift) to record every game.",
                ),
                Tab::Clips if library.clip_key_set => (
                    Icon::Scissors,
                    "No clips yet",
                    "Press your clip key in a game to save the last moments as a clip.",
                ),
                Tab::Clips => (
                    Icon::Scissors,
                    "No clips yet",
                    "Set a clip key in the Replay mod, then press it in a game to save a clip.",
                ),
            };
            let cx = body.x + body.w * 0.5;
            let cy = body.y + body.h * 0.42;
            icons::draw(
                painter,
                icon,
                (cx, cy - 46.0 * s),
                44.0 * s,
                2.0 * s,
                white(0.5),
            );
            let ts = TextStyle::new(18.0 * s).weight(Weight::Semibold);
            let tw = fonts.width(title, ts);
            fonts.shadowed_mid(
                painter,
                title,
                cx - tw * 0.5,
                cy + 4.0 * s,
                ts,
                white(0.92),
                s,
            );
            let hs = styles::body(s);
            let hw = fonts.width(hint, hs);
            fonts.shadowed_mid(
                painter,
                hint,
                cx - hw * 0.5,
                cy + 34.0 * s,
                hs,
                white(0.66),
                s,
            );
            return;
        }

        let saved_clip = painter.clip();
        painter.set_clip(Some(body));
        let scroll = self.scroll * s;
        // The rows draw from the shelf without copying it (it is put back below).
        let t = self.tab.index();
        let shelf = std::mem::take(&mut self.shelves[t]);
        let by_day = self.layout.headers_by == Some(Headers::Day);
        let by_server = self.layout.headers_by == Some(Headers::Server);
        // Only the rows on screen (the rows are in top order).
        let first = self
            .layout
            .rows
            .partition_point(|(_, top, h)| (top + h) * s <= scroll);
        for at in first..self.layout.rows.len() {
            let (item, top, _) = self.layout.rows[at];
            let y = body.y + top * s - scroll;
            if y >= body.bottom() {
                break;
            }
            match item {
                Row::Header(header) => {
                    let rect = Rect::new(body.x, y, body.w - 10.0 * s, HEADER_H * s);
                    if let Some(header) = self.layout.headers.get(header) {
                        draw_day_header(painter, fonts, &header.label, &header.summary, rect, s);
                    }
                }
                Row::Folder(f) => {
                    let row = Rect::new(body.x, y, body.w - 10.0 * s, row_h);
                    let visible = row.intersect(&body);
                    if visible.h > 0.0 {
                        if let Some(folder) = shelf.tree.get(f) {
                            self.draw_folder_row(painter, fonts, f, folder, row, visible, s);
                        }
                    }
                }
                Row::Entry(index) => {
                    let Some(entry) = shelf.entries.get(index) else {
                        continue;
                    };
                    let row = Rect::new(body.x, y, body.w - 10.0 * s, row_h);
                    let visible = row.intersect(&body);
                    if visible.h <= 0.0 {
                        continue;
                    }
                    let when = if by_day {
                        &shelf.times[index]
                    } else {
                        &shelf.whens[index]
                    };
                    self.draw_row(painter, fonts, entry, when, by_server, row, visible, s, now);
                }
            }
        }
        self.shelves[t] = shelf;
        painter.set_clip(saved_clip);

        if self.max_scroll > 0.0 {
            let track = Rect::new(
                body.right() - 4.0 * s,
                body.y + 6.0 * s,
                3.0 * s,
                body.h - 12.0 * s,
            );
            let thumb_h = (track.h * self.list_h / content_h).max(28.0 * s);
            let t = (self.scroll / self.max_scroll).clamp(0.0, 1.0);
            painter.round_rect(
                Rect::new(track.x, track.y + (track.h - thumb_h) * t, track.w, thumb_h),
                1.5 * s,
                Fill::Solid(white(0.32)),
            );
        }
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn draw_row(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        entry: &ReplayEntry,
        when: &str,
        server_in_header: bool,
        row: Rect,
        visible: Rect,
        s: f32,
        now: Instant,
    ) {
        let id = entry.id;
        let selected = self.selected == Some(Pick::Entry(id));
        let hover = self.hover(visible);
        let radius = 14.0 * s;
        if selected {
            painter.round_rect(
                row,
                radius,
                Fill::css_angle(row, 90.0, white(0.15), white(0.06)),
            );
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.34));
            painter.round_rect(
                Rect::new(row.x + 1.5 * s, row.y + 14.0 * s, 3.0 * s, row.h - 28.0 * s),
                1.5 * s,
                Fill::Solid(fade(ACCENT_TEXT, 0.95)),
            );
        } else if hover {
            painter.round_rect(row, radius, Fill::Solid(white(0.08)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.14));
        }
        self.hits.push(visible, Target::Row(id), None);

        // ---- the tile: kind icon and length ----
        let tile = Rect::new(
            row.x + 10.0 * s,
            row.y + (row.h - THUMB_H * s) * 0.5,
            THUMB_W * s,
            THUMB_H * s,
        );
        let (top, bottom) = if entry.clip {
            (css(150, 96, 220, 0.55), css(70, 36, 130, 0.62))
        } else {
            (css(70, 140, 230, 0.55), css(24, 58, 120, 0.62))
        };
        painter.round_rect(tile, 10.0 * s, Fill::css_angle(tile, 160.0, top, bottom));
        painter.round_rect(
            tile,
            10.0 * s,
            Fill::css_angle(tile, 135.0, white(0.16), white(0.0)),
        );
        painter.round_rect_stroke(tile, 10.0 * s, 1.0, 0.5, white(0.30));
        // Hovered, the tile turns into a play button (a click on it watches), as a video
        // thumbnail does.
        let tile_hover = self.hover(tile.intersect(&visible)) && entry.problem.is_none();
        if tile_hover {
            let (cx, cy) = (tile.x + tile.w * 0.5, tile.y + tile.h * 0.45);
            painter.circle(cx, cy + 1.0 * s, 17.0 * s, black(0.25));
            painter.circle(cx, cy, 16.0 * s, white(0.92));
            icons::draw(
                painter,
                Icon::Play,
                (cx + 1.5 * s, cy),
                17.0 * s,
                1.0,
                css(30, 60, 110, 1.0),
            );
        } else {
            icons::draw(
                painter,
                if entry.clip {
                    Icon::Scissors
                } else {
                    Icon::Film
                },
                (tile.x + tile.w * 0.5, tile.y + tile.h * 0.42),
                26.0 * s,
                1.8 * s,
                white(0.9),
            );
        }
        if entry.problem.is_none() {
            self.hits.push(
                tile.intersect(&visible),
                Target::Watch(id),
                Some(Tip::new("Watch", Some("Enter"))),
            );
        }
        let len = format::clock(entry.duration_us);
        let ls = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let lw = fonts.width(&len, ls);
        let lr = Rect::new(
            tile.right() - lw - 12.0 * s,
            tile.bottom() - 21.0 * s,
            lw + 8.0 * s,
            16.0 * s,
        );
        painter.round_rect(lr, 4.0 * s, Fill::Solid(black(0.62)));
        fonts.plain_mid(
            painter,
            &len,
            lr.x + 4.0 * s,
            lr.y + lr.h * 0.5,
            ls,
            white(0.95),
        );

        // ---- the buttons, right to left ----
        let bh = BUTTON_H * s;
        let by = row.y + (row.h - bh) * 0.5;
        let mut right = row.right() - 12.0 * s;
        let quiet = !(hover || selected);
        let armed = self.delete_armed.is_some_and(|(a, _)| a == id);
        // Delete (armed: a red "Delete?" pill).
        let delete = if armed {
            let w = 96.0 * s;
            Rect::new(right - w, by, w, bh)
        } else {
            Rect::new(right - bh, by, bh, bh)
        };
        if armed {
            let look = Look {
                danger: true,
                ..self.look(delete, Target::Delete(id))
            };
            style::pill(
                painter,
                fonts,
                delete,
                Some(Icon::Trash),
                "Delete?",
                look,
                s,
            );
        } else {
            let look = self.look(delete, Target::Delete(id));
            self.icon_button(painter, delete, Icon::Trash, look, quiet, s);
        }
        self.hits.push(
            delete.intersect(&visible),
            Target::Delete(id),
            Some(if armed {
                Tip::new("Click again to delete", Some("Delete"))
            } else {
                Tip::new("Delete", Some("Delete"))
            }),
        );
        right = delete.x - 6.0 * s;
        let folder = Rect::new(right - bh, by, bh, bh);
        let look = self.look(folder, Target::Folder(id));
        self.icon_button(painter, folder, Icon::Folder, look, quiet, s);
        self.hits.push(
            folder.intersect(&visible),
            Target::Folder(id),
            Some(Tip::new("Show in folder", None)),
        );
        right = folder.x - 6.0 * s;
        let rename = Rect::new(right - bh, by, bh, bh);
        let look = self.look(rename, Target::Rename(id));
        self.icon_button(painter, rename, Icon::Pencil, look, quiet, s);
        self.hits.push(
            rename.intersect(&visible),
            Target::Rename(id),
            // R, not F2: F2 is the game's screenshot key everywhere.
            Some(Tip::new("Rename", Some("R"))),
        );
        right = rename.x - 12.0 * s;
        let unreadable = entry.problem.is_some();
        let armed_watch = self.watch_armed.map(|(t, _)| t);
        if entry.clip {
            let armed = armed_watch == Some(Target::Export(id));
            let w = if armed { 140.0 * s } else { 100.0 * s };
            let export = Rect::new(right - w, by, w, bh);
            let look = Look {
                disabled: unreadable,
                danger: armed,
                ..self.look(export, Target::Export(id))
            };
            let label = if armed { "Leave & export" } else { "Export" };
            style::pill(painter, fonts, export, Some(Icon::Export), label, look, s);
            self.hits.push(
                export.intersect(&visible),
                Target::Export(id),
                Some(Tip::new("Export with your clip settings", Some("E"))),
            );
            right = export.x - 8.0 * s;
        }
        let armed = armed_watch == Some(Target::Watch(id));
        let w = if armed { 140.0 * s } else { 104.0 * s };
        let watch = Rect::new(right - w, by, w, bh);
        let look = Look {
            primary: !unreadable && !armed,
            danger: armed,
            disabled: unreadable,
            ..self.look(watch, Target::Watch(id))
        };
        let label = if armed { "Leave & watch" } else { "Watch" };
        style::pill(painter, fonts, watch, Some(Icon::Play), label, look, s);
        self.hits.push(
            watch.intersect(&visible),
            Target::Watch(id),
            Some(Tip::new("Watch", Some("Enter"))),
        );

        // ---- title and details ----
        let tx = tile.right() + 16.0 * s;
        let room = (watch.x - 16.0 * s - tx).max(40.0 * s);
        let title_style = TextStyle::new(15.5 * s).weight(Weight::Semibold);
        let title_y = row.y + row.h * 0.5 - 12.0 * s;
        let meta_y = row.y + row.h * 0.5 + 13.0 * s;
        let renaming = self.renaming.as_ref().filter(|r| r.id == id).cloned();
        if let Some(rename) = renaming {
            let field = Rect::new(tx - 8.0 * s, title_y - 16.0 * s, room + 8.0 * s, 32.0 * s);
            let caret = (ms_since(self.opened_at, now) as u64 / 530).is_multiple_of(2);
            if rename.all {
                // The whole title is selected: an accent wash behind it.
                let tw = fonts
                    .width(&rename.text, styles::body(s))
                    .min(field.w - 20.0 * s);
                painter.rect(
                    Rect::new(
                        field.x + 10.0 * s,
                        field.y + 7.0 * s,
                        tw,
                        field.h - 14.0 * s,
                    ),
                    fade(style::ACCENT, 0.55),
                );
            }
            style::text_field(
                painter,
                fonts,
                field,
                &rename.text,
                "Title",
                true,
                caret && !rename.all,
                s,
            );
            self.hits
                .push(field.intersect(&visible), Target::RenameField(id), None);
            fonts.shadowed_mid(
                painter,
                "Enter to save \u{b7} Esc to cancel",
                tx,
                meta_y + 3.0 * s,
                styles::small(s),
                white(0.6),
                s,
            );
        } else {
            let title = fonts.ellipsize(&entry.title, title_style, room);
            fonts.shadowed_mid(painter, &title, tx, title_y, title_style, white(0.98), s);
            let small = styles::small(s);
            if let Some(problem) = &entry.problem {
                icons::draw(
                    painter,
                    Icon::Warning,
                    (tx + 7.0 * s, meta_y),
                    14.0 * s,
                    1.5 * s,
                    css(255, 190, 90, 0.95),
                );
                let text = fonts.ellipsize(problem, small, room - 20.0 * s);
                fonts.shadowed_mid(
                    painter,
                    &text,
                    tx + 20.0 * s,
                    meta_y,
                    small,
                    css(255, 205, 130, 0.95),
                    s,
                );
            } else {
                let mut meta = String::with_capacity(64);
                // A new recording is titled with its server ("mc.example.net", "Clip,
                // mc.example.net"): then the server is not said twice (nor under its server's
                // header).
                if !entry.server.is_empty()
                    && !server_in_header
                    && !entry.title.contains(entry.server.as_str())
                {
                    meta.push_str(&entry.server);
                    meta.push_str("  \u{b7}  ");
                }
                meta.push_str(when);
                meta.push_str("  \u{b7}  ");
                meta.push_str(&format::size(entry.size_bytes));
                let mic_w = if entry.has_mic { 20.0 * s } else { 0.0 };
                // The bookmarks set while recording, after an amber
                // diamond like the viewer's timeline markers.
                let marks = (entry.bookmarks > 0).then(|| {
                    let n = entry.bookmarks;
                    format!("{n} bookmark{}", if n == 1 { "" } else { "s" })
                });
                let marks_w = marks
                    .as_deref()
                    .map_or(0.0, |m| fonts.width(m, small) + 30.0 * s);
                let text = fonts.ellipsize(&meta, small, room - mic_w - marks_w);
                let w = fonts.shadowed_mid(painter, &text, tx, meta_y, small, white(0.64), s);
                if entry.has_mic {
                    icons::draw(
                        painter,
                        Icon::Mic,
                        (tx + w + 12.0 * s, meta_y),
                        14.0 * s,
                        1.4 * s,
                        white(0.64),
                    );
                }
                if let Some(marks) = marks {
                    let x = tx + w + mic_w + 16.0 * s;
                    let r = 4.0 * s;
                    style::fill_convex(
                        painter,
                        &[
                            [x, meta_y - r],
                            [x + r, meta_y],
                            [x, meta_y + r],
                            [x - r, meta_y],
                        ],
                        css(255, 206, 110, 0.95),
                    );
                    fonts.shadowed_mid(
                        painter,
                        &marks,
                        x + 9.0 * s,
                        meta_y,
                        small,
                        css(255, 222, 160, 0.92),
                        s,
                    );
                }
            }
        }
    }

    fn icon_button(
        &self,
        painter: &mut Painter<'_>,
        rect: Rect,
        icon: Icon,
        look: Look,
        quiet: bool,
        s: f32,
    ) {
        let saved = painter.alpha;
        if quiet && !look.hover {
            painter.alpha = saved * 0.55;
        }
        style::icon_button(painter, rect, icon, 17.0 * s, look, s);
        painter.alpha = saved;
    }

    fn draw_footer(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        rect: Rect,
        s: f32,
    ) {
        let top = rect.bottom() - FOOTER_H * s;
        painter.rect(
            Rect::new(rect.x + 1.0, top.round(), rect.w - 2.0, 1.0),
            white(0.10),
        );
        let cy = top + FOOTER_H * s * 0.5;
        let pad = 24.0 * s;
        // Right: Back, then Open folder.
        let bh = 36.0 * s;
        let back_w = fonts.width(self.back_label, styles::button(s)) + 58.0 * s;
        let back = Rect::new(rect.right() - pad - back_w, cy - bh * 0.5, back_w, bh);
        let look = self.look(back, Target::Back);
        style::pill(
            painter,
            fonts,
            back,
            Some(Icon::Back),
            self.back_label,
            look,
            s,
        );
        self.hits
            .push(back, Target::Back, Some(Tip::new("Back", Some("Esc"))));
        let folder_w = 148.0 * s;
        let folder = Rect::new(back.x - 10.0 * s - folder_w, back.y, folder_w, bh);
        let look = self.look(folder, Target::OpenFolder);
        style::pill(
            painter,
            fonts,
            folder,
            Some(Icon::Folder),
            "Open folder",
            look,
            s,
        );
        let tip = match (self.arrangement.view, self.tab) {
            (ListView::Folders, _) => "Open this folder in Explorer",
            (ListView::List, Tab::Clips) => "Open the clips folder",
            (ListView::List, Tab::Recordings) => "Open the replays folder",
        };
        self.hits
            .push(folder, Target::OpenFolder, Some(Tip::new(tip, None)));
        // Short recordings, folded away or shown.
        if self.tab == Tab::Recordings && self.short_count > 0 {
            let label = if self.show_short {
                "Hide short".to_owned()
            } else {
                format!("Show {} short", self.short_count)
            };
            let w = fonts.width(&label, styles::button(s)) + 40.0 * s;
            let toggle = Rect::new(folder.x - 10.0 * s - w, back.y, w, bh);
            let look = self.look(toggle, Target::ShortToggle);
            style::pill(painter, fonts, toggle, None, &label, look, s);
            self.hits.push(
                toggle,
                Target::ShortToggle,
                Some(Tip::new(
                    "Recordings under 20 s with no bookmark",
                    Some("S"),
                )),
            );
        }

        // Left: what is stored.
        let Some(library) = self.library else {
            return;
        };
        let small = styles::small(s);
        let x = rect.x + pad;
        match self.tab {
            Tab::Recordings => {
                // Every recording on disk in every folder (the root's totals), the folded short
                // ones too.
                let used = self.shelves[0].tree.folder(0).bytes;
                let n = self.shelves[0].entries.len();
                let text = format!(
                    "{n} recording{}  \u{b7}  {} of {}",
                    if n == 1 { "" } else { "s" },
                    format::size(used),
                    format::size(library.cap_bytes)
                );
                let w = fonts.shadowed_mid(painter, &text, x, cy - 7.0 * s, small, white(0.72), s);
                let bar = Rect::new(x, cy + 8.0 * s, w.max(160.0 * s), 4.0 * s);
                let t = if library.cap_bytes == 0 {
                    0.0
                } else {
                    used as f32 / library.cap_bytes as f32
                };
                let color = if t > 0.9 {
                    css(255, 170, 80, 1.0)
                } else {
                    style::ACCENT
                };
                style::progress_bar(painter, bar, t, color);
            }
            Tab::Clips => {
                let used = self.shelves[1].tree.folder(0).bytes;
                let n = self.shelves[1].entries.len();
                let text = format!(
                    "{n} clip{}  \u{b7}  {}",
                    if n == 1 { "" } else { "s" },
                    format::size(used)
                );
                fonts.shadowed_mid(painter, &text, x, cy - 7.0 * s, small, white(0.72), s);
                fonts.shadowed_mid(
                    painter,
                    "Clips are never deleted automatically",
                    x,
                    cy + 10.0 * s,
                    small,
                    white(0.48),
                    s,
                );
            }
        }
    }

    // ---- the toolbar, folder rows and the Sort menu ------------------------------------------

    /// The toolbar: List | Folders on the left, then (in the Folders view) Up and the open
    /// folder's path, each part a button; the Sort button on the right. Returns the Sort
    /// button's rectangle (its menu drops from it).
    fn draw_toolbar(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        bar: Rect,
        s: f32,
    ) -> Rect {
        let cy = bar.y + bar.h * 0.5;
        let h = 34.0 * s;
        // The view switch.
        let switch = Rect::new(bar.x, cy - h * 0.5, 236.0 * s, h);
        let hover = style::segment_at(switch, 2, s, self.cursor);
        let selected = match self.arrangement.view {
            ListView::List => 0,
            ListView::Folders => 1,
        };
        let segments = style::segmented(
            painter,
            fonts,
            switch,
            &[(Some(Icon::List), "List"), (Some(Icon::Folder), "Folders")],
            selected,
            hover,
            &[],
            s,
        );
        let tips = [
            Tip::new("Every replay in one list", Some("F")),
            Tip::new("Folders, as on disk", Some("F")),
        ];
        for ((segment, view), tip) in segments
            .into_iter()
            .zip([ListView::List, ListView::Folders])
            .zip(tips)
        {
            self.hits.push(segment, Target::View(view), Some(tip));
        }
        // The Sort button, on the right.
        let label = self.arrangement.sort.label();
        let label_style = styles::button(s);
        let w = fonts.width(label, label_style) + 76.0 * s;
        let sort = Rect::new(bar.right() - w, cy - h * 0.5, w, h);
        let look = Look {
            active: self.sort_menu.is_some(),
            ..self.look(sort, Target::Sort)
        };
        style::pill_body(painter, sort, h * 0.5, look, s);
        icons::draw(
            painter,
            Icon::Sort,
            (sort.x + 22.0 * s, cy),
            16.0 * s,
            1.6 * s,
            white(0.92),
        );
        fonts.shadowed_mid(
            painter,
            label,
            sort.x + 38.0 * s,
            cy,
            label_style,
            white(0.97),
            s,
        );
        style::chevron_down(
            painter,
            (sort.right() - 18.0 * s, cy),
            3.5 * s,
            1.4 * s,
            white(0.8),
        );
        self.hits
            .push(sort, Target::Sort, Some(Tip::new("Sort order", Some("O"))));
        // The Folders view: Up, and where it is.
        if self.arrangement.view == ListView::Folders && self.library.is_some() {
            let up = Rect::new(switch.right() + 14.0 * s, cy - h * 0.5, h, h);
            let at_root = self.shelf().open().parent.is_none();
            let look = Look {
                disabled: at_root,
                ..self.look(up, Target::Up)
            };
            style::pill_body(painter, up, h * 0.5, look, s);
            icons::draw(
                painter,
                Icon::Up,
                up.center(),
                17.0 * s,
                1.7 * s,
                white(if at_root { 0.35 } else { 0.92 }),
            );
            if !at_root {
                self.hits.push(
                    up,
                    Target::Up,
                    Some(Tip::new("Up one folder", Some("Backspace"))),
                );
            }
            self.draw_path(
                painter,
                fonts,
                up.right() + 10.0 * s,
                sort.x - 16.0 * s,
                cy,
                s,
            );
        }
        sort
    }

    /// The open folder's path from the tab's name down, between `left` and `right`: each part a
    /// button that opens it, the last (the open folder) in bold. When it does not fit: the tab's
    /// name, `…`, then as many of the last parts as fit.
    #[allow(clippy::too_many_arguments)]
    fn draw_path(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        left: f32,
        right: f32,
        cy: f32,
        s: f32,
    ) {
        let normal = styles::body(s);
        let bold = normal.weight(Weight::Semibold);
        let pad = 7.0 * s;
        let sep = 16.0 * s;
        let shelf = &self.shelves[self.tab.index()];
        let chain = shelf.tree.chain(shelf.folder);
        let last = chain.len() - 1;
        let parts: Vec<(usize, String, f32)> = chain
            .iter()
            .enumerate()
            .map(|(i, &f)| {
                let name = if i == 0 {
                    self.tab_name()
                } else {
                    shelf.tree.folder(f).name.as_str()
                };
                let style = if i == last { bold } else { normal };
                let text = fonts.ellipsize(name, style, 240.0 * s);
                let w = fonts.width(&text, style) + 2.0 * pad;
                (f, text, w)
            })
            .collect();
        let avail = (right - left).max(0.0);
        let total: f32 = parts.iter().map(|p| p.2).sum::<f32>() + sep * (parts.len() - 1) as f32;
        let ellipsis_w = fonts.width("\u{2026}", normal) + 2.0 * pad;
        // Which parts show (`None`: the ellipsis).
        let shown: Vec<Option<usize>> = if total <= avail {
            (0..parts.len()).map(Some).collect()
        } else {
            let mut used = parts[0].2 + sep + ellipsis_w;
            let mut tail = Vec::new();
            for i in (1..parts.len()).rev() {
                let w = parts[i].2 + sep;
                if used + w > avail && !tail.is_empty() {
                    break;
                }
                used += w;
                tail.push(Some(i));
            }
            tail.reverse();
            [Some(0), None].into_iter().chain(tail).collect()
        };
        let area = Rect::new(left, cy - 18.0 * s, avail, 36.0 * s);
        let saved = painter.clip();
        painter.set_clip(Some(saved.map_or(area, |c| c.intersect(&area))));
        let mut x = left;
        for (n, part) in shown.iter().enumerate() {
            if n > 0 {
                style::chevron(
                    painter,
                    (x + sep * 0.5, cy),
                    1.0,
                    3.2 * s,
                    1.3 * s,
                    white(0.42),
                );
                x += sep;
            }
            let Some(i) = *part else {
                fonts.shadowed_mid(painter, "\u{2026}", x + pad, cy, normal, white(0.55), s);
                x += ellipsis_w;
                continue;
            };
            let (folder, text, w) = &parts[i];
            let current = i == last;
            if !current {
                let rect = Rect::new(x, cy - 14.0 * s, *w, 28.0 * s);
                if self.hover(rect) {
                    painter.round_rect(rect, 8.0 * s, Fill::Solid(white(0.12)));
                }
                self.hits.push(
                    rect.intersect(&area),
                    Target::Crumb(*folder),
                    Some(Tip::new("Go to this folder", None)),
                );
            }
            let (style, alpha) = if current { (bold, 1.0) } else { (normal, 0.70) };
            fonts.shadowed_mid(painter, text, x + pad, cy, style, white(alpha), s);
            x += w;
        }
        painter.set_clip(saved);
    }

    /// A folder's row: an amber tile with the folder sign and its replay count (where a
    /// replay's tile shows its length), the name, what it holds, then Open (blue, as Watch is
    /// for a replay) and Open in Explorer. A click selects it; a double click, the tile or Open
    /// opens it.
    #[allow(clippy::too_many_arguments)]
    fn draw_folder_row(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        index: usize,
        folder: &arrange::Folder,
        row: Rect,
        visible: Rect,
        s: f32,
    ) {
        let selected = self.selected == Some(Pick::Folder(index));
        let hover = self.hover(visible);
        let radius = 14.0 * s;
        if selected {
            painter.round_rect(
                row,
                radius,
                Fill::css_angle(row, 90.0, white(0.15), white(0.06)),
            );
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.34));
            painter.round_rect(
                Rect::new(row.x + 1.5 * s, row.y + 14.0 * s, 3.0 * s, row.h - 28.0 * s),
                1.5 * s,
                Fill::Solid(fade(ACCENT_TEXT, 0.95)),
            );
        } else if hover {
            painter.round_rect(row, radius, Fill::Solid(white(0.08)));
            painter.round_rect_stroke(row, radius, 1.0, 0.5, white(0.14));
        }
        self.hits.push(visible, Target::FolderRow(index), None);

        let tile = Rect::new(
            row.x + 10.0 * s,
            row.y + (row.h - THUMB_H * s) * 0.5,
            THUMB_W * s,
            THUMB_H * s,
        );
        painter.round_rect(
            tile,
            10.0 * s,
            Fill::css_angle(tile, 160.0, css(232, 176, 70, 0.52), css(140, 88, 20, 0.62)),
        );
        painter.round_rect(
            tile,
            10.0 * s,
            Fill::css_angle(tile, 135.0, white(0.16), white(0.0)),
        );
        painter.round_rect_stroke(tile, 10.0 * s, 1.0, 0.5, white(0.30));
        icons::draw(
            painter,
            Icon::Folder,
            (tile.x + tile.w * 0.5, tile.y + tile.h * 0.42),
            28.0 * s,
            1.8 * s,
            white(0.92),
        );
        self.hits.push(
            tile.intersect(&visible),
            Target::FolderOpen(index),
            Some(Tip::new("Open", Some("Enter"))),
        );
        let count = format::count(folder.count as u64);
        let cs = TextStyle::new(11.5 * s).weight(Weight::Semibold);
        let cw = fonts.width(&count, cs);
        let badge = Rect::new(
            tile.right() - cw - 12.0 * s,
            tile.bottom() - 21.0 * s,
            cw + 8.0 * s,
            16.0 * s,
        );
        painter.round_rect(badge, 4.0 * s, Fill::Solid(black(0.62)));
        fonts.plain_mid(
            painter,
            &count,
            badge.x + 4.0 * s,
            badge.y + badge.h * 0.5,
            cs,
            white(0.95),
        );

        // The buttons, right to left.
        let bh = BUTTON_H * s;
        let by = row.y + (row.h - bh) * 0.5;
        let quiet = !(hover || selected);
        let explorer = Rect::new(row.right() - 12.0 * s - bh, by, bh, bh);
        let look = self.look(explorer, Target::FolderExplorer(index));
        self.icon_button(painter, explorer, Icon::Folder, look, quiet, s);
        self.hits.push(
            explorer.intersect(&visible),
            Target::FolderExplorer(index),
            Some(Tip::new("Open in Explorer", None)),
        );
        let open_w = 104.0 * s;
        let open = Rect::new(explorer.x - 12.0 * s - open_w, by, open_w, bh);
        let look = Look {
            primary: true,
            ..self.look(open, Target::FolderOpen(index))
        };
        style::pill(painter, fonts, open, None, "Open", look, s);
        self.hits.push(
            open.intersect(&visible),
            Target::FolderOpen(index),
            Some(Tip::new("Open", Some("Enter"))),
        );

        // The name and what it holds.
        let tx = tile.right() + 16.0 * s;
        let room = (open.x - 16.0 * s - tx).max(40.0 * s);
        let title_style = TextStyle::new(15.5 * s).weight(Weight::Semibold);
        let name = fonts.ellipsize(&folder.name, title_style, room);
        fonts.shadowed_mid(
            painter,
            &name,
            tx,
            row.y + row.h * 0.5 - 12.0 * s,
            title_style,
            white(0.98),
            s,
        );
        let small = styles::small(s);
        let summary = fonts.ellipsize(&folder.summary, small, room);
        fonts.shadowed_mid(
            painter,
            &summary,
            tx,
            row.y + row.h * 0.5 + 13.0 * s,
            small,
            white(0.64),
            s,
        );
    }

    /// The Sort menu under the Sort button: every order, the current one ticked, the keyboard's
    /// `highlight` and the hovered item lit. Drawn over the list on a nearly opaque backing.
    fn draw_sort_menu(
        &mut self,
        painter: &mut Painter<'_>,
        fonts: &mut Fonts<'_>,
        button: Rect,
        highlight: usize,
        s: f32,
    ) {
        let item_h = MENU_ITEM_H * s;
        let pad = 6.0 * s;
        let label_style = styles::button(s);
        let widest = SortOrder::ALL
            .iter()
            .map(|o| fonts.width(o.label(), label_style))
            .fold(0.0, f32::max);
        let w = (widest + 64.0 * s).max(button.w);
        let h = item_h * SortOrder::ALL.len() as f32 + 2.0 * pad;
        let menu = Rect::new(button.right() - w, button.bottom() + 6.0 * s, w, h);
        let radius = 14.0 * s;
        painter.round_rect(menu, radius, Fill::Solid(css(16, 20, 30, 0.90)));
        style::panel(painter, menu, radius, 0.0, s);
        self.hits.push(menu, Target::SortMenu, None);
        for (i, order) in SortOrder::ALL.iter().enumerate() {
            let item = Rect::new(
                menu.x + pad,
                menu.y + pad + item_h * i as f32,
                menu.w - 2.0 * pad,
                item_h,
            );
            let current = *order == self.arrangement.sort;
            let lit = self.hover(item) || i == highlight;
            if lit || current {
                painter.round_rect(
                    item,
                    9.0 * s,
                    Fill::Solid(white(if lit { 0.16 } else { 0.08 })),
                );
            }
            if current {
                icons::draw(
                    painter,
                    Icon::Check,
                    (item.x + 15.0 * s, item.y + item_h * 0.5),
                    14.0 * s,
                    1.8 * s,
                    ACCENT_TEXT,
                );
            }
            fonts.shadowed_mid(
                painter,
                order.label(),
                item.x + 32.0 * s,
                item.y + item_h * 0.5,
                label_style,
                white(if current { 1.0 } else { 0.86 }),
                s,
            );
            self.hits.push(item, Target::SortItem(i), None);
        }
    }
}

/// A day's header in the list: its name on the left ("Today",
/// "Friday", "November 13"), what it holds on the right ("3 recordings · 48 MB"), a hairline
/// under both, like the date folders on disk. A server's or an account's header looks the same.
fn draw_day_header(
    painter: &mut Painter<'_>,
    fonts: &mut Fonts<'_>,
    label: &str,
    summary: &str,
    rect: Rect,
    s: f32,
) {
    let cy = rect.y + rect.h * 0.55;
    let label_style = TextStyle::new(13.5 * s).weight(Weight::Semibold);
    let small = styles::small(s);
    let w = fonts.width(summary, small);
    let label = fonts.ellipsize(label, label_style, (rect.w - w - 48.0 * s).max(40.0 * s));
    fonts.shadowed_mid(
        painter,
        &label,
        rect.x + 12.0 * s,
        cy,
        label_style,
        white(0.9),
        s,
    );
    fonts.shadowed_mid(
        painter,
        summary,
        rect.right() - 14.0 * s - w,
        cy,
        small,
        white(0.5),
        s,
    );
    painter.rect(
        Rect::new(
            rect.x + 12.0 * s,
            (rect.bottom() - 4.0 * s).round(),
            rect.w - 26.0 * s,
            1.0,
        ),
        white(0.12),
    );
}
