//! How the replay browser arranges a tab. **List** shows every replay
//! of the tab in one list (as before); **Folders** shows the folders on disk (the date folders
//! `2026\11 November\13` the recorder files replays in, and any the player makes), each with
//! its replay count, then the replays in the folder. Either way the player picks the order:
//! newest or oldest first, name, server, length, size, bookmarks or account, under a header per
//! day, server or account when the order has one.
//!
//! All of it is plain data computed when the library or a choice changes, never per frame:
//! [`FolderTree::build`] once per listing, [`sort_entries`], [`sort_folders`] and [`layout`]
//! once per change. The browser draws only the rows on screen.
//!
//! **Folder names are untrusted**: a folder can come from anywhere (a friend's shared archive,
//! say). The client makes each key from a folder its own listing found (links and junctions are
//! never followed). Here a key is only split into names and checked ([`FolderTree::build`] drops
//! keys with empty, `.`, `..` or over-long names, and caps depth and count), and a name is only
//! ever shown, through [`display_name`]. Nothing here turns a name back into a path: opening a
//! folder in Explorer goes back to the client as its key, and the client opens only a folder its
//! listing found.

use std::cmp::Ordering;
use std::collections::HashMap;

use chrono::NaiveDate;

use super::{format, groups, ReplayEntry};

/// The deepest folder key the browser accepts (the listing itself stops at four levels; this
/// only bounds what a key can make).
pub const MAX_FOLDER_DEPTH: usize = 16;
/// The most folders one tab holds (a pathological library stays quick; the listing caps too).
pub const MAX_FOLDERS: usize = 20_000;
/// The longest folder name kept, in characters (NTFS allows 255 UTF-16 units).
pub const MAX_NAME_CHARS: usize = 255;

/// The orders the list can be sorted in (the Sort menu, top to bottom).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SortOrder {
    /// Newest first, under a header per day (the default).
    #[default]
    Newest,
    /// Oldest first, under a header per day.
    Oldest,
    /// By title, A to Z, numbers by value ("Game 2" before "Game 10").
    Name,
    /// By server, A to Z, under a header per server; newest first within each.
    Server,
    /// Longest first.
    Longest,
    /// Largest file first.
    Largest,
    /// Most bookmarks first.
    Bookmarks,
    /// By the account that recorded it, under a header per account; newest first within each.
    Account,
}

/// What the rows sit under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Headers {
    /// No headers.
    None,
    /// A header per local day.
    Day,
    /// A header per server.
    Server,
    /// A header per account.
    Account,
}

impl SortOrder {
    /// The Sort menu's items, top to bottom.
    pub const ALL: [SortOrder; 8] = [
        SortOrder::Newest,
        SortOrder::Oldest,
        SortOrder::Name,
        SortOrder::Server,
        SortOrder::Longest,
        SortOrder::Largest,
        SortOrder::Bookmarks,
        SortOrder::Account,
    ];

    /// Its label in the Sort menu and on the Sort button.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            SortOrder::Newest => "Newest first",
            SortOrder::Oldest => "Oldest first",
            SortOrder::Name => "Name",
            SortOrder::Server => "Server",
            SortOrder::Longest => "Longest first",
            SortOrder::Largest => "Largest first",
            SortOrder::Bookmarks => "Most bookmarks",
            SortOrder::Account => "Account",
        }
    }

    /// The word the settings file keeps for it (`replay.browser_sort`).
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            SortOrder::Newest => "newest",
            SortOrder::Oldest => "oldest",
            SortOrder::Name => "name",
            SortOrder::Server => "server",
            SortOrder::Longest => "longest",
            SortOrder::Largest => "largest",
            SortOrder::Bookmarks => "bookmarks",
            SortOrder::Account => "account",
        }
    }

    /// The order a settings word names; `None` for anything else (the default then applies).
    #[must_use]
    pub fn from_token(token: &str) -> Option<SortOrder> {
        let token = token.trim();
        SortOrder::ALL.into_iter().find(|o| o.token() == token)
    }

    /// Its place in [`SortOrder::ALL`].
    #[must_use]
    pub fn index(self) -> usize {
        SortOrder::ALL.iter().position(|o| *o == self).unwrap_or(0)
    }

    /// What its rows sit under.
    #[must_use]
    pub fn headers(self) -> Headers {
        match self {
            SortOrder::Newest | SortOrder::Oldest => Headers::Day,
            SortOrder::Server => Headers::Server,
            SortOrder::Account => Headers::Account,
            SortOrder::Name | SortOrder::Longest | SortOrder::Largest | SortOrder::Bookmarks => {
                Headers::None
            }
        }
    }
}

/// How the tab's replays are shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ListView {
    /// Every replay of the tab in one list, whatever folder it is in (the default).
    #[default]
    List,
    /// The folders as on disk: the open folder's sub-folders, then its replays.
    Folders,
}

impl ListView {
    /// The word the settings file keeps for it (`replay.browser_view`).
    #[must_use]
    pub fn token(self) -> &'static str {
        match self {
            ListView::List => "list",
            ListView::Folders => "folders",
        }
    }

    /// The view a settings word names; `None` for anything else.
    #[must_use]
    pub fn from_token(token: &str) -> Option<ListView> {
        match token.trim() {
            "list" => Some(ListView::List),
            "folders" => Some(ListView::Folders),
            _ => None,
        }
    }
}

/// The player's choices, remembered between sessions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Arrangement {
    /// The order.
    pub sort: SortOrder,
    /// The view.
    pub view: ListView,
}

impl Arrangement {
    /// The choices the two settings words name; each unknown word falls back to its default.
    #[must_use]
    pub fn from_tokens(sort: &str, view: &str) -> Arrangement {
        Arrangement {
            sort: SortOrder::from_token(sort).unwrap_or_default(),
            view: ListView::from_token(view).unwrap_or_default(),
        }
    }
}

/// Whether `c` would hide or disguise a name: control characters (line breaks, tabs, escape
/// codes), direction marks and overrides, invisible formatting characters, and the replacement
/// character a name that is not valid Unicode decodes to.
fn is_odd(c: char) -> bool {
    c.is_control()
        || matches!(
            c,
            '\u{00AD}'
                | '\u{034F}'
                | '\u{061C}'
                | '\u{115F}'
                | '\u{1160}'
                | '\u{17B4}'
                | '\u{17B5}'
                | '\u{180B}'..='\u{180F}'
                | '\u{200B}'..='\u{200F}'
                | '\u{2028}'..='\u{202E}'
                | '\u{2060}'..='\u{206F}'
                | '\u{3164}'
                | '\u{FE00}'..='\u{FE0F}'
                | '\u{FEFF}'
                | '\u{FFA0}'
                | '\u{FFF0}'..='\u{FFFD}'
                | '\u{1D173}'..='\u{1D17A}'
                | '\u{E0000}'..='\u{E0FFF}'
        )
}

/// A folder name as the browser shows it: every control, direction or invisible character
/// becomes `?` (a character Windows never allows in a name, so a `?` always means "something
/// odd here"), at most [`MAX_NAME_CHARS`] characters, and a blank name shows as `?`.
#[must_use]
pub fn display_name(name: &str) -> String {
    let out: String = name
        .chars()
        .take(MAX_NAME_CHARS)
        .map(|c| if is_odd(c) { '?' } else { c })
        .collect();
    if out.trim().is_empty() {
        "?".to_owned()
    } else {
        out
    }
}

/// A server address as the Server order groups it: lower case, without surrounding blanks, a
/// trailing dot or the default port (`MC.Example.net:25565` is `mc.example.net`). `""` when
/// unknown.
#[must_use]
pub fn server_key(server: &str) -> String {
    let lower = server.trim().to_lowercase();
    let host = lower.strip_suffix(":25565").unwrap_or(&lower);
    host.trim_end_matches('.').to_owned()
}

/// The ASCII digits at the start of `s`.
fn digit_run(s: &str) -> &str {
    let end = s.bytes().take_while(u8::is_ascii_digit).count();
    &s[..end]
}

/// `a` and `b` (already lower-cased) as people sort names: a run of digits by its value, so
/// "game 2" comes before "game 10" (and "007" after "7"). Allocates nothing.
#[must_use]
pub fn natural_cmp(a: &str, b: &str) -> Ordering {
    let (mut a, mut b) = (a, b);
    loop {
        let (Some(x), Some(y)) = (a.chars().next(), b.chars().next()) else {
            return a.len().cmp(&b.len());
        };
        if x.is_ascii_digit() && y.is_ascii_digit() {
            let (da, db) = (digit_run(a), digit_run(b));
            let (ta, tb) = (da.trim_start_matches('0'), db.trim_start_matches('0'));
            let by_value = ta
                .len()
                .cmp(&tb.len())
                .then_with(|| ta.cmp(tb))
                .then_with(|| da.len().cmp(&db.len()));
            if by_value != Ordering::Equal {
                return by_value;
            }
            a = &a[da.len()..];
            b = &b[db.len()..];
        } else {
            if x != y {
                return x.cmp(&y);
            }
            a = &a[x.len_utf8()..];
            b = &b[y.len_utf8()..];
        }
    }
}

/// Known before unknown (`""`), then [`natural_cmp`].
fn known_first(a: &str, b: &str) -> Ordering {
    match (a.is_empty(), b.is_empty()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => natural_cmp(a, b),
    }
}

/// What the orders compare for one replay, made once per library (lower-cased, so sorting
/// allocates nothing).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EntryKeys {
    /// The title, lower-cased.
    pub title: String,
    /// [`server_key`] of the server.
    pub server: String,
    /// The account that recorded it, lower-cased (`""`: unknown).
    pub account: String,
    /// Its local day (the day headers).
    pub day: Option<NaiveDate>,
}

impl EntryKeys {
    /// The keys of `entry`, whose local day is `day`.
    #[must_use]
    pub fn new(entry: &ReplayEntry, day: Option<NaiveDate>) -> EntryKeys {
        EntryKeys {
            title: entry.title.trim().to_lowercase(),
            server: server_key(&entry.server),
            account: entry.player.trim().to_lowercase(),
            day,
        }
    }
}

/// Newest first; an unknown start (0) last.
fn newest_first(a: u64, b: u64) -> Ordering {
    match (a == 0, b == 0) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => b.cmp(&a),
    }
}

/// Oldest first; an unknown start (0) last.
fn oldest_first(a: u64, b: u64) -> Ordering {
    match (a == 0, b == 0) {
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.cmp(&b),
    }
}

/// Put `indices` (into `entries` and `keys`, which are in step) in `order`. Ties go newest
/// first (oldest first for [`SortOrder::Oldest`]), then by title, then by id, so the order is
/// the same every time.
pub fn sort_entries(
    indices: &mut [usize],
    entries: &[ReplayEntry],
    keys: &[EntryKeys],
    order: SortOrder,
) {
    indices.sort_by(|&a, &b| {
        let (ea, eb) = (&entries[a], &entries[b]);
        let (ka, kb) = (&keys[a], &keys[b]);
        let first = match order {
            SortOrder::Newest => newest_first(ea.started_unix_ms, eb.started_unix_ms),
            SortOrder::Oldest => oldest_first(ea.started_unix_ms, eb.started_unix_ms),
            SortOrder::Name => natural_cmp(&ka.title, &kb.title),
            SortOrder::Server => known_first(&ka.server, &kb.server),
            SortOrder::Longest => eb.duration_us.cmp(&ea.duration_us),
            SortOrder::Largest => eb.size_bytes.cmp(&ea.size_bytes),
            SortOrder::Bookmarks => eb.bookmarks.cmp(&ea.bookmarks),
            SortOrder::Account => known_first(&ka.account, &kb.account),
        };
        first
            .then_with(|| {
                if order == SortOrder::Oldest {
                    oldest_first(ea.started_unix_ms, eb.started_unix_ms)
                } else {
                    newest_first(ea.started_unix_ms, eb.started_unix_ms)
                }
            })
            .then_with(|| natural_cmp(&ka.title, &kb.title))
            .then_with(|| ea.id.cmp(&eb.id))
    });
}

/// One folder of a tab (the tab's root is folder 0).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Folder {
    /// Its name as shown ([`display_name`]); `""` for the root.
    pub name: String,
    /// Its key: the names on disk from the tab's root down to it, joined by `/` (`""` for the
    /// root). What the browser hands the client to open it in Explorer.
    pub key: String,
    /// The folder it is in (`None` for the root).
    pub parent: Option<usize>,
    /// Its sub-folders, in no particular order ([`sort_folders`] orders them for the list).
    pub children: Vec<usize>,
    /// The replays directly in it (indices into the tab's entries).
    pub entries: Vec<usize>,
    /// Replays in it and all its sub-folders.
    pub count: usize,
    /// Their size, bytes.
    pub bytes: u64,
    /// Their total length, microseconds.
    pub length_us: u64,
    /// Their bookmarks.
    pub bookmarks: u64,
    /// The newest and oldest start among them (0 when it holds none with a known start).
    pub newest_ms: u64,
    /// See [`Folder::newest_ms`].
    pub oldest_ms: u64,
    /// "12 recordings  ·  3 folders  ·  340 MB", or "Empty".
    pub summary: String,
    /// The name lower-cased (the order by name).
    sort_name: String,
}

impl Folder {
    fn new(name: &str, key: String, parent: Option<usize>) -> Folder {
        Folder {
            name: if parent.is_some() {
                display_name(name)
            } else {
                String::new()
            },
            key,
            parent,
            children: Vec::new(),
            entries: Vec::new(),
            count: 0,
            bytes: 0,
            length_us: 0,
            bookmarks: 0,
            newest_ms: 0,
            oldest_ms: 0,
            summary: String::new(),
            sort_name: name.to_lowercase(),
        }
    }
}

/// The folders of one tab, as on disk.
#[derive(Debug, Clone, PartialEq)]
pub struct FolderTree {
    folders: Vec<Folder>,
    by_key: HashMap<String, usize>,
}

/// The root of a tree not built yet (a tab before its library arrives).
static EMPTY_ROOT: Folder = Folder {
    name: String::new(),
    key: String::new(),
    parent: None,
    children: Vec::new(),
    entries: Vec::new(),
    count: 0,
    bytes: 0,
    length_us: 0,
    bookmarks: 0,
    newest_ms: 0,
    oldest_ms: 0,
    summary: String::new(),
    sort_name: String::new(),
};

impl Default for FolderTree {
    /// An empty tree (only the root), allocating nothing: the browser swaps a tab's shelf out
    /// for a default one while it draws the rows.
    fn default() -> Self {
        FolderTree {
            folders: Vec::new(),
            by_key: HashMap::new(),
        }
    }
}

/// A part of a folder key that can be a folder's name.
fn valid_name(name: &str) -> bool {
    !name.is_empty() && name != "." && name != ".." && name.chars().count() <= MAX_NAME_CHARS
}

impl FolderTree {
    /// The tree of a tab whose replays are `entries` (each in the folder its
    /// [`ReplayEntry::folder`] names) plus the folders `extra` (every folder the listing found,
    /// empty ones too). `clips` picks the summaries' noun. A replay whose key is refused (see the
    /// module docs) is listed in the root.
    #[must_use]
    pub fn build(entries: &[ReplayEntry], extra: &[String], clips: bool) -> FolderTree {
        let mut tree = FolderTree {
            folders: vec![Folder::new("", String::new(), None)],
            by_key: HashMap::new(),
        };
        for key in extra {
            let _ = tree.ensure(key);
        }
        for (index, entry) in entries.iter().enumerate() {
            let at = tree.ensure(&entry.folder).unwrap_or(0);
            tree.folders[at].entries.push(index);
            let mut node = Some(at);
            while let Some(n) = node {
                let folder = &mut tree.folders[n];
                folder.count += 1;
                folder.bytes = folder.bytes.saturating_add(entry.size_bytes);
                folder.length_us = folder.length_us.saturating_add(entry.duration_us);
                folder.bookmarks = folder.bookmarks.saturating_add(u64::from(entry.bookmarks));
                let started = entry.started_unix_ms;
                if started > 0 {
                    folder.newest_ms = folder.newest_ms.max(started);
                    folder.oldest_ms = if folder.oldest_ms == 0 {
                        started
                    } else {
                        folder.oldest_ms.min(started)
                    };
                }
                node = folder.parent;
            }
        }
        for folder in &mut tree.folders {
            folder.summary = summary(folder.count, folder.children.len(), folder.bytes, clips);
        }
        tree
    }

    /// The folder `key` names, made (with the folders above it) if new. `None` when the key is
    /// refused: an empty, `.`, `..` or over-long name, deeper than [`MAX_FOLDER_DEPTH`], or the
    /// tree already holds [`MAX_FOLDERS`].
    fn ensure(&mut self, key: &str) -> Option<usize> {
        if key.is_empty() {
            return Some(0);
        }
        if let Some(&index) = self.by_key.get(key) {
            return Some(index);
        }
        if key.split('/').count() > MAX_FOLDER_DEPTH || !key.split('/').all(valid_name) {
            return None;
        }
        let mut parent = 0;
        let mut end = 0;
        for (depth, name) in key.split('/').enumerate() {
            end += name.len() + usize::from(depth > 0);
            let prefix = &key[..end];
            if let Some(&index) = self.by_key.get(prefix) {
                parent = index;
                continue;
            }
            if self.folders.len() >= MAX_FOLDERS {
                return None;
            }
            let index = self.folders.len();
            self.folders
                .push(Folder::new(name, prefix.to_owned(), Some(parent)));
            self.folders[parent].children.push(index);
            self.by_key.insert(prefix.to_owned(), index);
            parent = index;
        }
        Some(parent)
    }

    /// The root (always folder 0).
    #[must_use]
    pub fn root(&self) -> usize {
        0
    }

    /// How many folders, the root included.
    #[must_use]
    pub fn len(&self) -> usize {
        self.folders.len().max(1)
    }

    /// Only the root.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.folders.len() <= 1
    }

    /// Folder `index`.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&Folder> {
        self.folders
            .get(index)
            .or_else(|| (index == 0).then_some(&EMPTY_ROOT))
    }

    /// Folder `index`, or the root when there is no such folder.
    #[must_use]
    pub fn folder(&self, index: usize) -> &Folder {
        self.folders
            .get(index)
            .or_else(|| self.folders.first())
            .unwrap_or(&EMPTY_ROOT)
    }

    /// The folder whose key is `key`.
    #[must_use]
    pub fn find(&self, key: &str) -> Option<usize> {
        if key.is_empty() {
            Some(0)
        } else {
            self.by_key.get(key).copied()
        }
    }

    /// The folder `key` names, or the nearest folder above it that still exists (the root at
    /// worst): where the browser goes back to after a refresh.
    #[must_use]
    pub fn nearest(&self, key: &str) -> usize {
        let mut key = key;
        loop {
            if let Some(index) = self.find(key) {
                return index;
            }
            match key.rfind('/') {
                Some(cut) => key = &key[..cut],
                None => return 0,
            }
        }
    }

    /// The folders from the root down to `index` (the path the browser shows).
    #[must_use]
    pub fn chain(&self, index: usize) -> Vec<usize> {
        let mut chain = Vec::new();
        let mut node = self.folders.get(index).map(|_| index);
        while let Some(n) = node {
            chain.push(n);
            node = self.folders[n].parent;
            if chain.len() > MAX_FOLDER_DEPTH + 1 {
                break;
            }
        }
        if chain.last() != Some(&0) {
            chain.push(0);
        }
        chain.reverse();
        chain
    }

    /// Whether `inner` is `outer` or inside it.
    #[must_use]
    pub fn contains(&self, outer: usize, inner: usize) -> bool {
        self.chain(inner).contains(&outer)
    }
}

/// "12 recordings  ·  3 folders  ·  340 MB" (only what is there), or "Empty".
fn summary(count: usize, folders: usize, bytes: u64, clips: bool) -> String {
    let mut parts: Vec<String> = Vec::with_capacity(3);
    if count > 0 {
        parts.push(format!("{count} {}", noun(clips, count)));
    }
    if folders > 0 {
        parts.push(format!(
            "{folders} folder{}",
            if folders == 1 { "" } else { "s" }
        ));
    }
    if count > 0 {
        parts.push(format::size(bytes));
    }
    if parts.is_empty() {
        "Empty".to_owned()
    } else {
        parts.join("  \u{b7}  ")
    }
}

/// "recording", "recordings", "clip", "clips".
#[must_use]
pub fn noun(clips: bool, count: usize) -> &'static str {
    match (clips, count) {
        (true, 1) => "clip",
        (true, _) => "clips",
        (false, 1) => "recording",
        (false, _) => "recordings",
    }
}

/// Put `children` (folders of `tree`) in `order`: the dated orders by the newest (oldest)
/// replay inside, an empty folder last; length, size and bookmarks by the folder's totals; the
/// others by name. Ties by name.
pub fn sort_folders(children: &mut [usize], tree: &FolderTree, order: SortOrder) {
    children.sort_by(|&a, &b| {
        let (x, y) = (tree.folder(a), tree.folder(b));
        let first = match order {
            SortOrder::Newest => newest_first(x.newest_ms, y.newest_ms),
            SortOrder::Oldest => oldest_first(x.oldest_ms, y.oldest_ms),
            SortOrder::Longest => y.length_us.cmp(&x.length_us),
            SortOrder::Largest => y.bytes.cmp(&x.bytes),
            SortOrder::Bookmarks => y.bookmarks.cmp(&x.bookmarks),
            SortOrder::Name | SortOrder::Server | SortOrder::Account => Ordering::Equal,
        };
        first
            .then_with(|| natural_cmp(&x.sort_name, &y.sort_name))
            .then_with(|| x.key.cmp(&y.key))
    });
}

/// One row of the list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Row {
    /// A header: its index in [`Layout::headers`].
    Header(usize),
    /// A folder: its index in the tab's [`FolderTree`].
    Folder(usize),
    /// A replay: its index in the tab's entries.
    Entry(usize),
}

/// A header above rows: a day, a server or an account.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// "Today", "mc.example.net", ...
    pub label: String,
    /// "3 recordings  ·  48 MB".
    pub summary: String,
}

/// The list laid out top to bottom.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Layout {
    /// The headers, in list order.
    pub headers: Vec<Header>,
    /// The rows: `(row, top, height)` in design pixels, top to bottom.
    pub rows: Vec<(Row, f32, f32)>,
    /// The content's height (no gap after the last row).
    pub height: f32,
    /// What the replays sit under: a row under a day header shows only its time of day, and a
    /// row under a server header leaves its server out.
    pub headers_by: Option<Headers>,
}

impl Layout {
    /// The top of `row`, if it is in the list.
    #[must_use]
    pub fn top_of(&self, row: Row) -> Option<f32> {
        self.rows
            .iter()
            .find(|(r, _, _)| *r == row)
            .map(|(_, top, _)| *top)
    }

    /// Whether a header sits right above the row at `top`.
    #[must_use]
    pub fn header_above(&self, top: f32) -> bool {
        self.rows
            .iter()
            .any(|(row, t, h)| matches!(row, Row::Header(_)) && (t + h - top).abs() < 0.01)
    }

    /// The rows that can be selected (folders and replays), top to bottom.
    pub fn picks(&self) -> impl Iterator<Item = Row> + '_ {
        self.rows
            .iter()
            .map(|(row, _, _)| *row)
            .filter(|row| !matches!(row, Row::Header(_)))
    }
}

/// Lay out the folder rows `folders` (already in order), then the replays `sorted` (indices into
/// `entries`/`keys`, already in `order`) under the order's headers. Headers are `header_h`
/// tall, rows `row_h` with `gap` after each. `today` names the days.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub fn layout(
    folders: &[usize],
    sorted: &[usize],
    entries: &[ReplayEntry],
    keys: &[EntryKeys],
    order: SortOrder,
    clips: bool,
    today: Option<NaiveDate>,
    (header_h, row_h, gap): (f32, f32, f32),
) -> Layout {
    let mut out = Layout {
        headers_by: Some(order.headers()),
        ..Layout::default()
    };
    let mut y = 0.0;
    for &folder in folders {
        out.rows.push((Row::Folder(folder), y, row_h));
        y += row_h + gap;
    }
    let headers = order.headers();
    let same = |a: usize, b: usize| match headers {
        Headers::None => true,
        Headers::Day => keys[a].day == keys[b].day,
        Headers::Server => keys[a].server == keys[b].server,
        Headers::Account => keys[a].account == keys[b].account,
    };
    let mut start = 0;
    while start < sorted.len() {
        let first = sorted[start];
        let end = start
            + sorted[start..]
                .iter()
                .take_while(|&&i| same(first, i))
                .count();
        if headers != Headers::None {
            let bytes: u64 = sorted[start..end]
                .iter()
                .map(|&i| entries[i].size_bytes)
                .fold(0, u64::saturating_add);
            let count = end - start;
            let label = match headers {
                Headers::Day => keys[first].day.map_or_else(
                    || "Unknown date".to_owned(),
                    |d| groups::day_label(d, today),
                ),
                Headers::Server if keys[first].server.is_empty() => "Unknown server".to_owned(),
                Headers::Server => display_name(entries[first].server.trim()),
                Headers::Account if keys[first].account.is_empty() => "Unknown account".to_owned(),
                Headers::Account => display_name(entries[first].player.trim()),
                Headers::None => String::new(),
            };
            out.headers.push(Header {
                label,
                summary: format!(
                    "{count} {}  \u{b7}  {}",
                    noun(clips, count),
                    format::size(bytes)
                ),
            });
            out.rows
                .push((Row::Header(out.headers.len() - 1), y, header_h));
            y += header_h;
        }
        for &index in &sorted[start..end] {
            out.rows.push((Row::Entry(index), y, row_h));
            y += row_h + gap;
        }
        start = end;
    }
    out.height = (y - gap).max(0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: u64, title: &str, folder: &str) -> ReplayEntry {
        ReplayEntry {
            id,
            title: title.to_owned(),
            server: "mc.example.net".to_owned(),
            started_unix_ms: 1_000_000 + id * 1000,
            duration_us: 60_000_000,
            size_bytes: 1_000_000,
            clip: false,
            has_mic: false,
            problem: None,
            bookmarks: 0,
            folder: folder.to_owned(),
            player: "Tester".to_owned(),
        }
    }

    fn keys(entries: &[ReplayEntry]) -> Vec<EntryKeys> {
        entries.iter().map(|e| EntryKeys::new(e, None)).collect()
    }

    fn sorted(entries: &[ReplayEntry], order: SortOrder) -> Vec<u64> {
        let keys = keys(entries);
        let mut indices: Vec<usize> = (0..entries.len()).collect();
        sort_entries(&mut indices, entries, &keys, order);
        indices.into_iter().map(|i| entries[i].id).collect()
    }

    #[test]
    fn tokens_round_trip_and_unknown_words_fall_back() {
        for order in SortOrder::ALL {
            assert_eq!(SortOrder::from_token(order.token()), Some(order));
            assert_eq!(SortOrder::ALL[order.index()], order);
        }
        for view in [ListView::List, ListView::Folders] {
            assert_eq!(ListView::from_token(view.token()), Some(view));
        }
        assert_eq!(
            Arrangement::from_tokens(" server ", "folders"),
            Arrangement {
                sort: SortOrder::Server,
                view: ListView::Folders
            }
        );
        assert_eq!(
            Arrangement::from_tokens("../../etc", "\u{202e}list"),
            Arrangement::default()
        );
        assert_eq!(Arrangement::from_tokens("", ""), Arrangement::default());
    }

    #[test]
    fn natural_order_reads_numbers_by_value() {
        let mut names = vec![
            "game 10", "game 2", "game 1", "game 02", "alpha", "", "game", "beta 3b", "beta 3a",
        ];
        names.sort_by(|a, b| natural_cmp(a, b));
        assert_eq!(
            names,
            ["", "alpha", "beta 3a", "beta 3b", "game", "game 1", "game 2", "game 02", "game 10"]
        );
        assert_eq!(
            natural_cmp("x99999999999999999999999", "x1"),
            Ordering::Greater
        );
        assert_eq!(natural_cmp("été", "ete"), Ordering::Greater);
        assert_eq!(natural_cmp("a", "a"), Ordering::Equal);
    }

    #[test]
    fn server_keys_fold_case_port_and_trailing_dot() {
        assert_eq!(server_key(" MC.Example.net:25565 "), "mc.example.net");
        assert_eq!(server_key("mc.example.net."), "mc.example.net");
        assert_eq!(server_key("mc.example.net:25566"), "mc.example.net:25566");
        assert_eq!(server_key("   "), "");
    }

    #[test]
    fn every_order_sorts_and_ties_are_stable() {
        let mut list = vec![
            entry(1, "Game 10", ""),
            entry(2, "game 2", ""),
            entry(3, "Bridge", ""),
            entry(4, "alpha", ""),
        ];
        list[0].server = "b.example.net".to_owned();
        list[1].server = "A.example.net:25565".to_owned();
        list[2].server = String::new();
        list[3].server = "a.example.net".to_owned();
        list[0].duration_us = 5;
        list[2].duration_us = 500_000_000;
        list[1].size_bytes = 9_000_000;
        list[3].bookmarks = 2;
        list[0].bookmarks = 1;
        list[2].player = "Zed".to_owned();
        list[3].player = String::new();
        assert_eq!(sorted(&list, SortOrder::Newest), [4, 3, 2, 1]);
        assert_eq!(sorted(&list, SortOrder::Oldest), [1, 2, 3, 4]);
        assert_eq!(sorted(&list, SortOrder::Name), [4, 3, 2, 1]);
        // a.example.net twice (newest first), then b, then the unknown server last.
        assert_eq!(sorted(&list, SortOrder::Server), [4, 2, 1, 3]);
        assert_eq!(sorted(&list, SortOrder::Longest), [3, 4, 2, 1]);
        assert_eq!(sorted(&list, SortOrder::Largest), [2, 4, 3, 1]);
        assert_eq!(sorted(&list, SortOrder::Bookmarks), [4, 1, 3, 2]);
        // "tester" twice, "zed", then the unknown account last.
        assert_eq!(sorted(&list, SortOrder::Account), [2, 1, 3, 4]);
        // An unknown start goes last both ways.
        list[3].started_unix_ms = 0;
        assert_eq!(sorted(&list, SortOrder::Newest), [3, 2, 1, 4]);
        assert_eq!(sorted(&list, SortOrder::Oldest), [1, 2, 3, 4]);
    }

    #[test]
    fn the_tree_counts_replays_in_each_folder_and_below() {
        let list = vec![
            entry(1, "a", "2026/11 November/13"),
            entry(2, "b", "2026/11 November/13"),
            entry(3, "c", "2026/11 November/12"),
            entry(4, "d", ""),
            entry(5, "e", "Best games"),
        ];
        let extra = vec!["Empty one".to_owned(), "2026/11 November".to_owned()];
        let tree = FolderTree::build(&list, &extra, false);
        let root = tree.folder(0);
        assert_eq!(root.count, 5);
        assert_eq!(root.entries, [3]);
        assert_eq!(root.children.len(), 3, "2026, Best games, Empty one");
        let year = tree.find("2026").expect("2026");
        assert_eq!(tree.folder(year).count, 3);
        assert_eq!(
            tree.folder(year).summary,
            "3 recordings  \u{b7}  1 folder  \u{b7}  3 MB"
        );
        let day = tree.find("2026/11 November/13").expect("the 13th");
        assert_eq!(tree.folder(day).entries, [0, 1]);
        assert_eq!(tree.folder(day).name, "13");
        assert_eq!(tree.folder(day).newest_ms, 1_002_000);
        assert_eq!(tree.folder(day).oldest_ms, 1_001_000);
        let empty = tree.find("Empty one").expect("an empty folder is listed");
        assert_eq!(tree.folder(empty).summary, "Empty");
        assert_eq!(
            tree.chain(day)
                .iter()
                .map(|&i| tree.folder(i).name.as_str())
                .collect::<Vec<_>>(),
            ["", "2026", "11 November", "13"]
        );
        assert!(tree.contains(year, day) && !tree.contains(day, year));
        // Gone after a refresh: the nearest folder above that is still there.
        assert_eq!(
            tree.nearest("2026/11 November/14"),
            tree.find("2026/11 November").unwrap()
        );
        assert_eq!(tree.nearest("Gone/Deeper"), 0);
        assert_eq!(tree.nearest(""), 0);
    }

    #[test]
    fn odd_folder_keys_never_make_a_folder_and_names_show_safely() {
        let list = vec![
            entry(1, "dots", "../outside"),
            entry(2, "dot", "./here"),
            entry(3, "empty part", "a//b"),
            entry(4, "lead", "/rooted"),
            entry(5, "trail", "trail/"),
            entry(6, "deep", &["d"; MAX_FOLDER_DEPTH + 1].join("/")),
            entry(7, "long", &"x".repeat(MAX_NAME_CHARS + 1)),
            entry(8, "odd but fine", "new\nline/\u{202e}evil.exe/tab\there"),
        ];
        let tree = FolderTree::build(&list, &["..".to_owned(), "a/../b".to_owned()], false);
        // The seven refused keys are listed in the root; only the odd-but-valid key made folders.
        assert_eq!(tree.folder(0).entries, [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(tree.len(), 4, "root + three folders of the valid key");
        let leaf = tree
            .find("new\nline/\u{202e}evil.exe/tab\there")
            .expect("a valid key");
        let names: Vec<&str> = tree
            .chain(leaf)
            .iter()
            .map(|&i| tree.folder(i).name.as_str())
            .collect();
        assert_eq!(names, ["", "new?line", "?evil.exe", "tab?here"]);
        assert_eq!(display_name("\u{200b}"), "?");
        assert_eq!(display_name("   "), "?");
        assert_eq!(display_name("ok \u{e9}t\u{e9}"), "ok \u{e9}t\u{e9}");
        assert_eq!(
            display_name(&"y".repeat(400)).chars().count(),
            MAX_NAME_CHARS
        );
    }

    #[test]
    fn the_folder_count_is_capped() {
        let extra: Vec<String> = (0..MAX_FOLDERS + 50).map(|i| format!("f{i}")).collect();
        let tree = FolderTree::build(&[entry(1, "late", "zz/last")], &extra, false);
        assert_eq!(tree.len(), MAX_FOLDERS);
        assert_eq!(
            tree.folder(0).entries,
            [0],
            "no room left: listed in the root"
        );
    }

    #[test]
    fn folders_follow_the_order_by_what_they_hold() {
        let mut list = vec![
            entry(1, "a", "old"),
            entry(2, "b", "new"),
            entry(3, "c", "big"),
        ];
        list[2].size_bytes = 50_000_000;
        list[2].started_unix_ms = 1_001_500;
        list[0].duration_us = 999_000_000;
        let tree = FolderTree::build(&list, &["empty".to_owned()], false);
        let names = |order: SortOrder| {
            let mut children = tree.folder(0).children.clone();
            sort_folders(&mut children, &tree, order);
            children
                .into_iter()
                .map(|i| tree.folder(i).name.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(names(SortOrder::Newest), ["new", "big", "old", "empty"]);
        assert_eq!(names(SortOrder::Oldest), ["old", "big", "new", "empty"]);
        assert_eq!(names(SortOrder::Name), ["big", "empty", "new", "old"]);
        assert_eq!(names(SortOrder::Largest), ["big", "new", "old", "empty"]);
        assert_eq!(names(SortOrder::Longest), ["old", "big", "new", "empty"]);
    }

    #[test]
    fn the_layout_puts_folders_first_then_rows_under_their_headers() {
        let mut list = vec![entry(1, "a", ""), entry(2, "b", ""), entry(3, "c", "")];
        list[0].server = "b.example.net".to_owned();
        list[1].server = "a.example.net".to_owned();
        list[2].server = "A.example.net".to_owned();
        let keys = keys(&list);
        let mut order: Vec<usize> = vec![0, 1, 2];
        sort_entries(&mut order, &list, &keys, SortOrder::Server);
        let out = layout(
            &[7, 8],
            &order,
            &list,
            &keys,
            SortOrder::Server,
            false,
            None,
            (30.0, 80.0, 6.0),
        );
        let rows: Vec<Row> = out.rows.iter().map(|r| r.0).collect();
        assert_eq!(
            rows,
            [
                Row::Folder(7),
                Row::Folder(8),
                Row::Header(0),
                Row::Entry(2),
                Row::Entry(1),
                Row::Header(1),
                Row::Entry(0),
            ]
        );
        // The newest spelling names the group.
        assert_eq!(out.headers[0].label, "A.example.net");
        assert_eq!(out.headers[0].summary, "2 recordings  \u{b7}  2 MB");
        assert_eq!(out.headers[1].label, "b.example.net");
        assert_eq!(out.rows[2].1, 172.0);
        assert!(out.header_above(out.top_of(Row::Entry(2)).unwrap()));
        assert!(!out.header_above(out.top_of(Row::Entry(1)).unwrap()));
        assert_eq!(out.height, 2.0 * 86.0 + 30.0 + 2.0 * 86.0 + 30.0 + 80.0);
        assert_eq!(out.picks().count(), 5);
        // No headers for the name order.
        let flat = layout(
            &[],
            &order,
            &list,
            &keys,
            SortOrder::Name,
            true,
            None,
            (30.0, 80.0, 6.0),
        );
        assert!(flat.headers.is_empty());
        assert_eq!(flat.height, 3.0 * 86.0 - 6.0);
        assert_eq!(flat.headers_by, Some(Headers::None));
    }
}
