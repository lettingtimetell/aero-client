//! The replay library: `<gameDir>/replays/` (full recordings) and `<gameDir>/replays/clips/`
//! (clips). Listing, renaming, deleting, the storage cap, and finishing files a crash left
//! without a footer.
//!
//! # Date folders
//!
//! New recordings and clips go into a folder per day, in this computer's local time:
//!
//! ```text
//! replays\2026\11 November\13\2026-11-13 21.30.39 sky.example.org.aeroreplay
//! replays\clips\2026\11 November\13\2026-11-13 21.33.02 sky.example.org.aeroreplay
//! ```
//!
//! The month's number comes first so every sort order is right, its name so it reads; the file
//! name keeps the whole date and time, so a file shared or moved on its own still says when it is
//! from. The date is that of [`ReplayMeta::title_unix_ms`] (where the part to watch begins).
//!
//! Listing walks the folders (up to [`MAX_DEPTH`] levels, so folders the player makes list too),
//! and files left flat in `replays\` by older versions keep listing until [`Library::organize`]
//! moves them into their date folder. The storage cap deletes only recordings in the places the
//! library writes (flat in `replays\` or in a date folder), never a folder the player made, and
//! never a clip.
//!
//! # Folders in the browser
//!
//! [`Library::list_tree`] lists a tab's replays together with every folder under its root (empty
//! ones too) for the browser's Folders view, and [`folder_key`] names each folder by its path
//! under the root. Folder names are untrusted (a shared archive can hold anything): below the two
//! roots (`replays\` and `replays\clips\`, which are the player's own and may be links to
//! another drive, as the recorder writes through them too) no listing goes through a symbolic
//! link or a junction, and [`is_safe_folder`] checks a folder again before Explorer opens it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, SystemTime};

use chrono::{Datelike as _, TimeZone};

use crate::reader::{read_layout, repair};
use crate::{ReplayError, ReplayKind, ReplayMeta, EXTENSION};

/// The folder under the game directory that holds full recordings.
pub const REPLAYS_DIR: &str = "replays";
/// The folder under [`REPLAYS_DIR`] that holds clips.
pub const CLIPS_DIR: &str = "clips";

/// How many folder levels below `replays\` (and `replays\clips\`) the library looks: the three of
/// a date folder (`2026\11 November\13`), inside folders the player makes (four before the
/// browser's Folders view, which invites folders of one's own, date folders moved into them
/// included).
pub const MAX_DEPTH: usize = 8;

/// The most folders [`Library::list_tree`] lists under one root; below that many, it stops going
/// into new folders (a pathological library stays quick to list and to show).
pub const MAX_TREE_FOLDERS: usize = 10_000;

/// Month folder names (`11 November`).
const MONTHS: [&str; 12] = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
];

/// The label of a bookmark: a marker the player set with the Bookmark key while recording.
pub const BOOKMARK_LABEL: &str = "Bookmark";

/// One [`Library::organize`] at a time in this process (the recorder's helper thread and the
/// browser's listing thread may both start one).
static ORGANIZING: Mutex<()> = Mutex::new(());

/// The local time zone could not be read once already (logged once).
static ZONE_WARNED: AtomicBool = AtomicBool::new(false);

/// One tab of the library as the browser's Folders view shows it
/// ([`Library::list_tree`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LibraryTree {
    /// The replays, as [`Library::list_dir`] lists them.
    pub entries: Vec<LibraryEntry>,
    /// Every folder under the tab's root that the listing went into, empty ones too, as the
    /// real paths it found (never made from a name), in no particular order.
    pub folders: Vec<PathBuf>,
}

/// One replay or clip on disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibraryEntry {
    /// Where it is.
    pub path: PathBuf,
    /// What it says about itself.
    pub meta: ReplayMeta,
    /// Its size on disk.
    pub size_bytes: u64,
    /// It is in the clips folder.
    pub is_clip: bool,
    /// The game stopped while recording it; it plays up to its last complete block.
    pub recovered: bool,
    /// When the file last changed.
    pub modified: Option<SystemTime>,
}

/// How much the library holds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Usage {
    /// Full recordings: bytes.
    pub full_bytes: u64,
    /// Full recordings: files.
    pub full_count: usize,
    /// Clips: bytes.
    pub clip_bytes: u64,
    /// Clips: files.
    pub clip_count: usize,
}

/// The two replay folders of one game directory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    replays: PathBuf,
    clips: PathBuf,
}

/// A file a live recording (this client's or another running client's) may still be writing:
/// changed less than this long ago. Never deleted or repaired.
const LIVE_GRACE: Duration = Duration::from_secs(120);

impl Library {
    /// The library of `game_dir`. Nothing is created until something is written.
    #[must_use]
    pub fn new(game_dir: &Path) -> Library {
        let replays = game_dir.join(REPLAYS_DIR);
        let clips = replays.join(CLIPS_DIR);
        Library { replays, clips }
    }

    /// `<gameDir>/replays`.
    #[must_use]
    pub fn replays_dir(&self) -> &Path {
        &self.replays
    }

    /// `<gameDir>/replays/clips`.
    #[must_use]
    pub fn clips_dir(&self) -> &Path {
        &self.clips
    }

    /// Every replay and clip, newest first. Unreadable files are skipped (and logged).
    #[must_use]
    pub fn list(&self) -> Vec<LibraryEntry> {
        let mut entries = self.list_dir(false);
        entries.extend(self.list_dir(true));
        entries.sort_by(|a, b| {
            b.meta
                .started_unix_ms
                .cmp(&a.meta.started_unix_ms)
                .then_with(|| b.modified.cmp(&a.modified))
        });
        entries
    }

    /// The full recordings (`clips == false`) or the clips, in no particular order: the files
    /// flat in the folder and those in its sub-folders (date folders, or the player's own), up to
    /// [`MAX_DEPTH`] levels down. Recordings leave the clips folder out.
    #[must_use]
    pub fn list_dir(&self, clips: bool) -> Vec<LibraryEntry> {
        let mut out = Vec::new();
        for (path, metadata) in self.files(clips) {
            let modified = metadata.modified().ok();
            match read_layout(&path) {
                Ok(layout) => out.push(LibraryEntry {
                    path,
                    meta: layout.meta,
                    size_bytes: layout.size,
                    is_clip: clips,
                    recovered: layout.recovered,
                    modified,
                }),
                Err(error) => {
                    tracing::warn!(file = %file_name(&path), %error, "unreadable replay skipped");
                }
            }
        }
        out
    }

    /// Change a replay's title (stored in the file; the file name does not change). A file a
    /// crash left unfinished is finished first.
    ///
    /// # Errors
    ///
    /// The path is not a replay in this library, or it cannot be read or written.
    pub fn rename(&self, path: &Path, title: &str) -> Result<(), ReplayError> {
        self.check_inside(path)?;
        let layout = read_layout(path)?;
        check_not_recording(path, layout.recovered)?;
        let mut meta = layout.meta;
        meta.title = title.trim().to_owned();
        crate::writer::rewrite_footer(path, &layout.entries, layout.blocks_end, &meta)
    }

    /// Delete a replay or clip.
    ///
    /// # Errors
    ///
    /// The path is not a replay in this library, it is still being recorded, or the OS refuses.
    pub fn delete(&self, path: &Path) -> Result<(), ReplayError> {
        self.check_inside(path)?;
        let recovered = read_layout(path).is_ok_and(|layout| layout.recovered);
        check_not_recording(path, recovered)?;
        std::fs::remove_file(path)?;
        self.prune_empty_date_dirs(path);
        Ok(())
    }

    /// How much the library holds (sub-folders included).
    #[must_use]
    pub fn usage(&self) -> Usage {
        let mut usage = Usage::default();
        for (clips, bytes, count) in [
            (false, &mut usage.full_bytes, &mut usage.full_count),
            (true, &mut usage.clip_bytes, &mut usage.clip_count),
        ] {
            for (_, metadata) in self.files(clips) {
                *bytes += metadata.len();
                *count += 1;
            }
        }
        usage
    }

    /// Delete the oldest full recordings until they use at most `max_full_bytes`. Clips are
    /// never deleted, nor is `keep` (the recording in progress) or anything changed in the last
    /// two minutes (another running client may be writing it). Only recordings where the library
    /// puts them count and can go (flat in `replays\` or in a date folder): a folder the player
    /// made is theirs. A date folder left empty is removed. Returns what was deleted.
    pub fn enforce_cap(&self, max_full_bytes: u64, keep: Option<&Path>) -> Vec<PathBuf> {
        let mut entries = self.list_dir(false);
        entries.retain(|entry| {
            entry
                .path
                .parent()
                .is_some_and(|dir| is_managed_dir(&self.replays, dir))
        });
        let mut total: u64 = entries.iter().map(|e| e.size_bytes).sum();
        if total <= max_full_bytes {
            return Vec::new();
        }
        // Oldest first.
        entries.sort_by(|a, b| {
            a.meta
                .started_unix_ms
                .cmp(&b.meta.started_unix_ms)
                .then_with(|| a.modified.cmp(&b.modified))
        });
        let now = SystemTime::now();
        let mut deleted = Vec::new();
        for entry in entries {
            if total <= max_full_bytes {
                break;
            }
            if keep.is_some_and(|keep| same_file(keep, &entry.path)) || is_live(entry.modified, now)
            {
                continue;
            }
            match std::fs::remove_file(&entry.path) {
                Ok(()) => {
                    tracing::info!(
                        file = %file_name(&entry.path),
                        bytes = entry.size_bytes,
                        "oldest replay deleted (storage limit)"
                    );
                    total = total.saturating_sub(entry.size_bytes);
                    self.prune_empty_date_dirs(&entry.path);
                    deleted.push(entry.path);
                }
                Err(error) => {
                    tracing::warn!(file = %file_name(&entry.path), %error, "replay not deleted");
                }
            }
        }
        deleted
    }

    /// Give every replay and clip that a crash left without a footer its footer, so it lists
    /// and seeks like any other. Files changed in the last two minutes are left alone (a running
    /// client may still be writing them). Returns what was repaired.
    pub fn repair_unfinished(&self) -> Vec<PathBuf> {
        let now = SystemTime::now();
        let mut repaired = Vec::new();
        for entry in self.list_dir(false).into_iter().chain(self.list_dir(true)) {
            if !entry.recovered || is_live(entry.modified, now) {
                continue;
            }
            match repair(&entry.path) {
                Ok(true) => {
                    tracing::info!(file = %file_name(&entry.path), "unfinished replay repaired");
                    repaired.push(entry.path);
                }
                Ok(false) => {}
                Err(error) => {
                    tracing::warn!(file = %file_name(&entry.path), %error, "replay not repaired");
                }
            }
        }
        repaired
    }

    /// A new, unused path for a recording (`clip == false`) or a clip named after `stem` (for
    /// example `2026-11-12 21.41 mc.example.net`). Characters Windows does not allow in file
    /// names are replaced; `(2)`, `(3)`... is added if the name is taken.
    #[must_use]
    pub fn unique_path(&self, clip: bool, stem: &str) -> PathBuf {
        unique_in(if clip { &self.clips } else { &self.replays }, stem)
    }

    /// [`Library::unique_path`] in the date folder of `unix_ms` (this computer's local date), where
    /// the recorder puts new recordings and clips:
    /// `replays\2026\11 November\13\<stem>.aeroreplay`. The folder is made when the file is.
    #[must_use]
    pub fn unique_path_at(&self, clip: bool, stem: &str, unix_ms: u64) -> PathBuf {
        let root = if clip { &self.clips } else { &self.replays };
        unique_in(&date_folder(root, unix_ms), stem)
    }

    /// Move the recordings and clips that older versions left flat in `replays\` and
    /// `replays\clips\` into their date folders. Returns `(from, to)` for each file moved.
    ///
    /// Nothing can be lost or overwritten: each move is a rename on the same disk onto an empty
    /// placeholder this call creates with `create_new` (so an existing file is never touched);
    /// if the rename fails, the placeholder is removed (only while it is still empty) and the
    /// file stays where it was, and keeps listing. Left alone: files changed in the last two
    /// minutes (a recording in progress, perhaps by another running client), files a crash left
    /// unfinished ([`Library::repair_unfinished`] finishes them first), unreadable files and
    /// files with no start time. A name stamped in another time zone (a friend's recording,
    /// `2026-11-14 02.05.23 ...` for 21:05 here) gets this computer's stamp.
    ///
    /// It reads every flat file's header, so call it off the game thread.
    pub fn organize(&self) -> Vec<(PathBuf, PathBuf)> {
        let _one_at_a_time = ORGANIZING
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let now = SystemTime::now();
        let mut moved = Vec::new();
        for root in [&self.replays, &self.clips] {
            let Ok(read) = std::fs::read_dir(root) else {
                continue;
            };
            let flat: Vec<PathBuf> = read
                .flatten()
                .filter(|item| item.file_type().is_ok_and(|kind| kind.is_file()))
                .map(|item| item.path())
                .filter(|path| has_replay_extension(path))
                .collect();
            for path in flat {
                let modified = std::fs::metadata(&path)
                    .ok()
                    .and_then(|m| m.modified().ok());
                if is_live(modified, now) {
                    continue;
                }
                let Ok(layout) = read_layout(&path) else {
                    continue;
                };
                let instant = layout.meta.title_unix_ms();
                if layout.recovered || instant == 0 {
                    continue;
                }
                let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
                    continue;
                };
                let name = restamp(name, instant).unwrap_or_else(|| name.to_owned());
                let dir = date_folder(root, instant);
                // Logged relative to the replays folder: no local path in a log people share.
                let relative = |p: &Path| p.strip_prefix(root).unwrap_or(p).display().to_string();
                match move_without_overwrite(&path, &dir, &name) {
                    Ok(to) => {
                        tracing::info!(
                            from = %relative(&path),
                            to = %relative(&to),
                            "replay moved into its date folder"
                        );
                        moved.push((path, to));
                    }
                    Err(error) => {
                        tracing::warn!(file = %relative(&path), %error, "replay left where it is");
                    }
                }
            }
        }
        moved
    }

    /// The `.aeroreplay` files of recordings (`clips == false`) or clips, with their metadata.
    fn files(&self, clips: bool) -> Vec<(PathBuf, std::fs::Metadata)> {
        let mut out = Vec::new();
        self.walk_tab(clips, &mut out, None);
        out
    }

    /// Walk a tab's root ([`walk`]); `dirs` collects the folders gone into. The
    /// two roots are the player's own (either may be a link to another drive, as the recorder
    /// writes through them too); below them no link or junction is ever followed.
    fn walk_tab(
        &self,
        clips: bool,
        out: &mut Vec<(PathBuf, std::fs::Metadata)>,
        dirs: Option<&mut Vec<PathBuf>>,
    ) {
        if clips {
            walk(&self.clips, 0, false, out, dirs);
        } else {
            walk(&self.replays, 0, true, out, dirs);
        }
    }

    /// The recordings (`clips == false`) or the clips with every folder under
    /// their root, empty ones too, from one walk (the browser's Folders view): the folders and
    /// files [`Library::list_dir`] looks at, links and junctions never followed, at most
    /// [`MAX_TREE_FOLDERS`] folders. Unreadable files are skipped (and logged).
    #[must_use]
    pub fn list_tree(&self, clips: bool) -> LibraryTree {
        let mut files = Vec::new();
        let mut folders = Vec::new();
        self.walk_tab(clips, &mut files, Some(&mut folders));
        let mut entries = Vec::with_capacity(files.len());
        for (path, metadata) in files {
            let modified = metadata.modified().ok();
            match read_layout(&path) {
                Ok(layout) => entries.push(LibraryEntry {
                    path,
                    meta: layout.meta,
                    size_bytes: layout.size,
                    is_clip: clips,
                    recovered: layout.recovered,
                    modified,
                }),
                Err(error) => {
                    tracing::warn!(file = %file_name(&path), %error, "unreadable replay skipped");
                }
            }
        }
        LibraryTree { entries, folders }
    }

    /// The root of the recordings (`clips == false`) or of the clips.
    #[must_use]
    pub fn tab_root(&self, clips: bool) -> &Path {
        if clips {
            &self.clips
        } else {
            &self.replays
        }
    }

    /// After `path` went: remove its date folder, then the month's and the year's, while they
    /// are empty (`remove_dir` refuses a folder with anything in it).
    fn prune_empty_date_dirs(&self, path: &Path) {
        for root in [&self.clips, &self.replays] {
            if !path.starts_with(root) {
                continue;
            }
            let mut dir = path.parent();
            while let Some(d) = dir {
                if d == root.as_path() || !is_date_path(root, d) || std::fs::remove_dir(d).is_err()
                {
                    break;
                }
                dir = d.parent();
            }
            return;
        }
    }
}

/// A new, unused path for `stem` in `dir` (see [`Library::unique_path`]).
fn unique_in(dir: &Path, stem: &str) -> PathBuf {
    let mut clean: String = stem
        .chars()
        .map(|c| match c {
            '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*' => '-',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    clean = clean.trim().trim_end_matches('.').to_owned();
    if clean.chars().count() > 96 {
        clean = clean.chars().take(96).collect();
    }
    if clean.is_empty() {
        clean = "replay".to_owned();
    }
    let mut path = dir.join(format!("{clean}.{EXTENSION}"));
    let mut n = 2;
    while path.exists() {
        path = dir.join(format!("{clean} ({n}).{EXTENSION}"));
        n += 1;
    }
    path
}

impl Library {
    /// `path` is a replay file somewhere in this library (sub-folders included).
    fn check_inside(&self, path: &Path) -> Result<(), ReplayError> {
        let inside = match (self.replays.canonicalize(), path.canonicalize()) {
            (Ok(root), Ok(path)) => path
                .strip_prefix(&root)
                .is_ok_and(|rest| rest.components().count() <= MAX_DEPTH + 2),
            _ => false,
        };
        if inside && is_replay_file(path) {
            Ok(())
        } else {
            Err(ReplayError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{} is not a replay in this library", file_name(path)),
            )))
        }
    }
}

impl ReplayMeta {
    /// How many bookmarks the player set while recording (markers labelled
    /// [`BOOKMARK_LABEL`]); read from the footer's metadata, no block is opened.
    #[must_use]
    pub fn bookmark_count(&self) -> usize {
        self.markers
            .iter()
            .filter(|marker| marker.label == BOOKMARK_LABEL)
            .count()
    }

    /// The moment a file's name, default title and date folder refer to, in milliseconds since
    /// the Unix epoch: where its part to watch begins (the start key's press for a recording
    /// started mid-game, the clip key's press for a clip). 0 when the start is unknown.
    #[must_use]
    pub fn title_unix_ms(&self) -> u64 {
        if self.started_unix_ms == 0 {
            return 0;
        }
        let into = match self.kind {
            ReplayKind::Full => self.view_start,
            ReplayKind::Clip { end, .. } => end,
        };
        self.started_unix_ms.saturating_add(into / 1000)
    }
}

/// The date folder under `root` for this computer's local date of `unix_ms`:
/// `root\2026\11 November\13`. UTC if the local zone is unknown.
#[must_use]
pub fn date_folder(root: &Path, unix_ms: u64) -> PathBuf {
    let date = local_datetime(unix_ms).date();
    date_folder_of(root, date)
}

/// [`date_folder`] in the zone `tz` (tests).
#[must_use]
pub fn date_folder_in<Tz: TimeZone>(root: &Path, unix_ms: u64, tz: &Tz) -> PathBuf {
    let seconds = i64::try_from(unix_ms / 1000).unwrap_or(0);
    let date = tz
        .timestamp_opt(seconds, 0)
        .single()
        .map_or_else(|| utc_datetime(seconds), |at| at.naive_local())
        .date();
    date_folder_of(root, date)
}

fn date_folder_of(root: &Path, date: chrono::NaiveDate) -> PathBuf {
    root.join(format!("{:04}", date.year()))
        .join(format!(
            "{:02} {}",
            date.month(),
            MONTHS[date.month0() as usize % 12]
        ))
        .join(format!("{:02}", date.day()))
}

/// Whether `dir` is `root` or one of its date folders (`2026\11 November\13`): where the
/// library itself puts recordings.
fn is_managed_dir(root: &Path, dir: &Path) -> bool {
    match dir.strip_prefix(root) {
        Ok(rest) => {
            let depth = rest.components().count();
            depth == 0 || (depth == 3 && is_date_path(root, dir))
        }
        Err(_) => false,
    }
}

/// Whether `dir` is a year (`2026`), month (`2026\11 November`) or day (`2026\11 November\13`)
/// folder under `root`.
fn is_date_path(root: &Path, dir: &Path) -> bool {
    let Ok(rest) = dir.strip_prefix(root) else {
        return false;
    };
    let parts: Vec<&str> = rest
        .components()
        .map(|part| part.as_os_str().to_str().unwrap_or(""))
        .collect();
    let two_digits = |part: &str| {
        let bytes = part.as_bytes();
        bytes.len() >= 2
            && bytes[0].is_ascii_digit()
            && bytes[1].is_ascii_digit()
            && (bytes.len() == 2 || bytes[2] == b' ')
    };
    match parts.as_slice() {
        [year, rest @ ..] if rest.len() <= 2 => {
            year.len() == 4
                && year.bytes().all(|b| b.is_ascii_digit())
                && rest.iter().all(|part| two_digits(part))
        }
        _ => false,
    }
}

/// Every `.aeroreplay` file under `dir` with its metadata, up to [`MAX_DEPTH`] folders down
/// (`skip_clips`: leave the top-level clips folder out). Symbolic links are not followed, nor
/// junctions: `file_type` reports a name-surrogate reparse point as a link, neither a folder nor
/// a file. With `dirs`, every folder gone into is collected too (at most [`MAX_TREE_FOLDERS`];
/// then no new folder is gone into).
fn walk(
    dir: &Path,
    depth: usize,
    skip_clips: bool,
    out: &mut Vec<(PathBuf, std::fs::Metadata)>,
    mut dirs: Option<&mut Vec<PathBuf>>,
) {
    let Ok(read) = std::fs::read_dir(dir) else {
        return;
    };
    for item in read.flatten() {
        let Ok(kind) = item.file_type() else {
            continue;
        };
        if kind.is_dir() {
            let clips = skip_clips && item.file_name().eq_ignore_ascii_case(CLIPS_DIR);
            if depth < MAX_DEPTH && !clips {
                let path = item.path();
                if let Some(dirs) = dirs.as_deref_mut() {
                    if dirs.len() >= MAX_TREE_FOLDERS {
                        continue;
                    }
                    dirs.push(path.clone());
                }
                walk(&path, depth + 1, false, out, dirs.as_deref_mut());
            }
        } else if kind.is_file() {
            let path = item.path();
            if has_replay_extension(&path) {
                if let Ok(metadata) = item.metadata() {
                    out.push((path, metadata));
                }
            }
        }
    }
}

/// Move `from` to `dir\name` (or `dir\name (2)`...) without ever replacing a file: an empty
/// placeholder is created with `create_new` first (so the name is free and now ours), then
/// `from` is renamed onto it. If the rename fails, the placeholder goes (while it is still
/// empty) and `from` is untouched.
fn move_without_overwrite(from: &Path, dir: &Path, name: &str) -> std::io::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let (stem, extension) = match name.rsplit_once('.') {
        Some((stem, extension)) if !stem.is_empty() => (stem, Some(extension)),
        _ => (name, None),
    };
    for n in 1..1000 {
        let candidate = match (n, extension) {
            (1, _) => dir.join(name),
            (_, Some(extension)) => dir.join(format!("{stem} ({n}).{extension}")),
            (_, None) => dir.join(format!("{stem} ({n})")),
        };
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(placeholder) => {
                drop(placeholder);
                return match std::fs::rename(from, &candidate) {
                    Ok(()) => Ok(candidate),
                    Err(error) => {
                        if std::fs::metadata(&candidate).is_ok_and(|m| m.len() == 0) {
                            let _ = std::fs::remove_file(&candidate);
                        }
                        Err(error)
                    }
                };
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error),
        }
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::AlreadyExists,
        "no free name in the date folder",
    ))
}

/// A file name stamped `2026-11-14 02.05.23 ...` in another time zone than this computer's (a
/// recording made on a friend's computer): the same name with this computer's stamp of
/// `unix_ms`. `None` if the name has no stamp, already has the local one, or differs from it by
/// anything but a whole time-zone offset (a name the player chose stays).
#[must_use]
pub fn restamp(name: &str, unix_ms: u64) -> Option<String> {
    restamp_in(name, unix_ms, &chrono::Local)
}

/// [`restamp`] for a computer in the zone `tz` (tests).
#[must_use]
pub fn restamp_in<Tz: TimeZone>(name: &str, unix_ms: u64, tz: &Tz) -> Option<String> {
    const STAMP: &str = "%Y-%m-%d %H.%M.%S";
    let stamp = name.get(..19)?;
    let rest = &name[19..];
    if !(rest.is_empty() || rest.starts_with(' ') || rest.starts_with('.')) {
        return None;
    }
    let named = chrono::NaiveDateTime::parse_from_str(stamp, STAMP).ok()?;
    let seconds = i64::try_from(unix_ms / 1000).ok()?;
    let here = tz.timestamp_opt(seconds, 0).single()?;
    let offset = zone_offset(named, utc_datetime(seconds), 2)?;
    let local_offset = (here.naive_local() - utc_datetime(seconds)).num_seconds();
    if offset == local_offset {
        return None;
    }
    Some(format!("{}{rest}", here.naive_local().format(STAMP)))
}

/// The time-zone offset (seconds) that turns `utc` into `named`, if `named` is `utc` in some
/// zone: a whole number of quarter hours, at most 14 hours, give or take `slack` seconds (the
/// name and the metadata read the clock separately).
fn zone_offset(
    named: chrono::NaiveDateTime,
    utc: chrono::NaiveDateTime,
    slack: i64,
) -> Option<i64> {
    const QUARTER: i64 = 15 * 60;
    let diff = (named - utc).num_seconds();
    let offset = (diff + diff.signum() * QUARTER / 2) / QUARTER * QUARTER;
    ((diff - offset).abs() <= slack && offset.abs() <= 14 * 3600).then_some(offset)
}

/// A default title ending in a time written on a computer in another time zone
/// ("sky.example.org · Nov 14, 2:05 AM" for 21:05 here), with this computer's local time
/// instead. `None` if the title does not end in such a time or it is already local. For
/// display: the file keeps its title.
#[must_use]
pub fn localize_title(title: &str, unix_ms: u64) -> Option<String> {
    localize_title_in(title, unix_ms, &chrono::Local)
}

/// [`localize_title`] for a computer in the zone `tz` (tests).
#[must_use]
pub fn localize_title_in<Tz: TimeZone>(title: &str, unix_ms: u64, tz: &Tz) -> Option<String> {
    let (head, when) = title.rsplit_once(" \u{b7} ")?;
    // "Nov 14, 2:05 AM"
    let (date, time) = when.split_once(", ")?;
    let (month, day) = date.split_once(' ')?;
    let month = MONTHS
        .iter()
        .position(|name| name.get(..3) == Some(month))? as u32
        + 1;
    let day: u32 = day.parse().ok()?;
    let (clock, half) = time.split_once(' ')?;
    let (hour, minute) = clock.split_once(':')?;
    if minute.len() != 2 {
        return None;
    }
    let (hour, minute): (u32, u32) = (hour.parse().ok()?, minute.parse().ok()?);
    if !(1..=12).contains(&hour) || minute > 59 {
        return None;
    }
    let hour = match half {
        "AM" => hour % 12,
        "PM" => hour % 12 + 12,
        _ => return None,
    };
    let seconds = i64::try_from(unix_ms / 1000).ok()?;
    let utc = utc_datetime(seconds);
    // To the minute, like the title.
    let utc_minute = utc - chrono::Duration::seconds(i64::from(chrono::Timelike::second(&utc)));
    let here = tz.timestamp_opt(seconds, 0).single()?;
    // The title has no year: the one that puts it nearest the start.
    let offset = [utc.year() - 1, utc.year(), utc.year() + 1]
        .into_iter()
        .filter_map(|year| {
            chrono::NaiveDate::from_ymd_opt(year, month, day)?.and_hms_opt(hour, minute, 0)
        })
        .find_map(|named| zone_offset(named, utc_minute, 60))?;
    let local_offset = (here.naive_local() - utc).num_seconds();
    if offset == local_offset {
        return None;
    }
    Some(format!("{head} \u{b7} {}", title_time(here.naive_local())))
}

/// `Nov 12, 9:41 PM`.
fn title_time(at: chrono::NaiveDateTime) -> String {
    at.format("%b %-d, %-I:%M %p").to_string()
}

/// `seconds` since the Unix epoch as a UTC date and time.
fn utc_datetime(seconds: i64) -> chrono::NaiveDateTime {
    chrono::DateTime::from_timestamp(seconds, 0)
        .map(|utc| utc.naive_utc())
        .unwrap_or_default()
}

/// `unix_ms` in this computer's local time; UTC (logged once) if the zone cannot be read.
fn local_datetime(unix_ms: u64) -> chrono::NaiveDateTime {
    let seconds = i64::try_from(unix_ms / 1000).unwrap_or(0);
    match chrono::Local.timestamp_opt(seconds, 0).single() {
        Some(local) => local.naive_local(),
        None => {
            if !ZONE_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!("the local time zone could not be read; replay names use UTC");
            }
            utc_datetime(seconds)
        }
    }
}

/// `unix_ms` as local date and time for a file name: `2026-11-12 21.41.05` (Windows file names
/// cannot hold `:`). UTC if the local zone is unknown.
#[must_use]
pub fn local_stamp(unix_ms: u64) -> String {
    local_datetime(unix_ms)
        .format("%Y-%m-%d %H.%M.%S")
        .to_string()
}

/// `unix_ms` as a short local date and time for a title: `Nov 12, 9:41 PM`. UTC if the local
/// zone is unknown.
#[must_use]
pub fn local_title_time(unix_ms: u64) -> String {
    title_time(local_datetime(unix_ms))
}

/// An unfinished file changed in the last two minutes is a recording in progress.
fn check_not_recording(path: &Path, unfinished: bool) -> Result<(), ReplayError> {
    let modified = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());
    if unfinished && is_live(modified, SystemTime::now()) {
        return Err(ReplayError::Io(std::io::Error::new(
            std::io::ErrorKind::ResourceBusy,
            "this replay is still being recorded",
        )));
    }
    Ok(())
}

fn is_replay_file(path: &Path) -> bool {
    path.is_file() && has_replay_extension(path)
}

fn has_replay_extension(path: &Path) -> bool {
    path.extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case(EXTENSION))
}

fn is_live(modified: Option<SystemTime>, now: SystemTime) -> bool {
    modified.is_some_and(|m| now.duration_since(m).map_or(true, |age| age < LIVE_GRACE))
}

/// A replay's file name for logs and messages: never the folder it is in,
/// which holds the player's account name (players share logs).
pub(crate) fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || String::from("(no name)"),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Whether `path` is a symbolic link or a junction (on Windows, any
/// name-surrogate reparse point): what the library never goes through below its two roots.
fn is_link(path: &Path) -> bool {
    std::fs::symlink_metadata(path).is_ok_and(|m| m.file_type().is_symlink())
}

/// A folder's key under `root` (what the browser calls it): its names from `root`
/// down, joined by `/`; `""` for `root` itself. `None` when `dir` is not under `root` or a part is
/// not a plain name (`..`, `.`, a drive or a root) or holds a `/`. A name that is not valid
/// Unicode is decoded lossily (two such names may then share a key: the browser shows them as one
/// folder; nothing is ever opened by its key alone).
#[must_use]
pub fn folder_key(root: &Path, dir: &Path) -> Option<String> {
    let rest = dir.strip_prefix(root).ok()?;
    let mut key = String::new();
    for part in rest.components() {
        let std::path::Component::Normal(name) = part else {
            return None;
        };
        let name = name.to_string_lossy();
        if name.is_empty() || name.contains('/') {
            return None;
        }
        if !key.is_empty() {
            key.push('/');
        }
        key.push_str(&name);
    }
    Some(key)
}

/// Whether `dir` is `root` or a real folder inside it, safe to open in Explorer:
/// it is a folder, its canonical path is under `root`'s, and no part below `root` is a link or a
/// junction (a folder swapped for a link since the listing is refused).
#[must_use]
pub fn is_safe_folder(root: &Path, dir: &Path) -> bool {
    let Ok(rest) = dir.strip_prefix(root) else {
        return false;
    };
    let mut at = root.to_path_buf();
    for part in rest.components() {
        let std::path::Component::Normal(name) = part else {
            return false;
        };
        at.push(name);
        if is_link(&at) {
            return false;
        }
    }
    match (root.canonicalize(), dir.canonicalize()) {
        (Ok(root), Ok(dir)) => dir.starts_with(&root) && dir.is_dir(),
        _ => false,
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}
