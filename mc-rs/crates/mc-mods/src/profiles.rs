//! Whole-setup profiles (a request for "an option for
//! saving and loading entire mod profiles").
//!
//! A profile is a whole mod setup: every mod's switch and settings, where its HUD elements sit,
//! the mods pinned to My setup and the shared HUD style ([`ModSettings::setup_json`]). The menu
//! key and Color Grading's own saved looks are not part of it: loading a profile never moves the
//! key that opens the menu and never deletes a saved look.
//!
//! # Files
//!
//! One file per profile in `<gameDir>/mc-rs-mod-profiles/` ([`PROFILES_DIR`]), in the settings
//! file's own format plus a `"profile"` header (`{"name": "Bedwars", "saved": 1762990000}`). So
//! a profile is easy to share, any build reads it like a settings file (unknown mods and
//! options are kept, missing ones load as their defaults), and one bad file never hides the
//! others. [`BACKUP_FILE`] holds the setup from before the last load.
//!
//! # Safety
//!
//! * **Loading is atomic.** A profile is parsed completely when the folder is read; a file
//!   that is not a settings object is listed as unreadable and cannot be loaded. Loading is then
//!   one [`ModSettings::apply_setup`]: every value changes in the same step, or none does.
//! * **Nothing is lost without a confirm.** The menu asks before loading, before saving over a
//!   profile and before deleting one, and keeps the setup from before a load as "Before last
//!   load" ([`ProfileLibrary::keep_backup`]): in memory at once, on disk soon after.
//! * **File work never blocks the game.** Every read and write runs on the library's worker
//!   thread (the user's game folder is on a USB disk where one write took up to 1.2 s); the menu
//!   reads the list and the results without waiting. Files are written to a temporary file and
//!   renamed into place, so a crash never leaves half a profile.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Number, Value};

use crate::descriptor::ModDescriptor;
use crate::settings::ModSettings;

/// The profiles folder, inside the game directory (next to `mc-rs-mods.json`).
pub const PROFILES_DIR: &str = "mc-rs-mod-profiles";
/// The setup from before the last load (shown as "Before last load").
pub const BACKUP_FILE: &str = "_before-last-load.json";
/// The longest profile name, in characters.
pub const NAME_MAX: usize = 32;
/// The largest profile file read: a whole setup is tens of
/// kilobytes.
pub const MAX_PROFILE_BYTES: u64 = 4 << 20;
/// The name the backup's header carries.
pub const BACKUP_NAME: &str = "Before last load";
/// The root key of a profile file's header.
const HEADER: &str = "profile";

/// What loading a setup would change, mod by mod (catalog indices, in catalog order).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SetupDiff {
    /// Mods it turns on.
    pub turns_on: Vec<usize>,
    /// Mods it turns off.
    pub turns_off: Vec<usize>,
    /// Mods whose settings differ (any value, or the choice of HUD style).
    pub settings: Vec<usize>,
    /// Mods with a HUD element somewhere else.
    pub moved: Vec<usize>,
    /// The pinned mods differ.
    pub pins: bool,
    /// The shared HUD style differs.
    pub hud_style: bool,
}

impl SetupDiff {
    /// Nothing would change.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.turns_on.is_empty()
            && self.turns_off.is_empty()
            && self.settings.is_empty()
            && self.moved.is_empty()
            && !self.pins
            && !self.hud_style
    }

    /// How many mods change in any way.
    #[must_use]
    pub fn mods_changed(&self) -> usize {
        let mut all: Vec<usize> = self
            .turns_on
            .iter()
            .chain(&self.turns_off)
            .chain(&self.settings)
            .chain(&self.moved)
            .copied()
            .collect();
        all.sort_unstable();
        all.dedup();
        all.len()
    }
}

/// `name` as a profile name: control characters removed, spaces trimmed, at most [`NAME_MAX`]
/// characters. May be empty.
#[must_use]
pub fn clean_name(name: &str) -> String {
    let clean: String = name.chars().filter(|c| !c.is_control()).collect();
    let short: String = clean.trim().chars().take(NAME_MAX).collect();
    short.trim_end().to_owned()
}

/// The file name (without `.json`) for profile `name`: letters, digits, spaces, `-` and `_`
/// kept, anything else `_`. Never empty, never a name Windows reserves (`CON`, `COM1`...), and
/// never starting with `_` or `.` (the library's own files start with `_`).
#[must_use]
pub fn file_stem(name: &str) -> String {
    let mut stem: String = clean_name(name)
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || matches!(c, ' ' | '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect();
    stem = stem.trim().trim_start_matches(['_', '.']).to_owned();
    let upper = stem.to_ascii_uppercase();
    let reserved = matches!(upper.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || ((upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.len() == 4
            && upper.as_bytes()[3].is_ascii_digit());
    if stem.is_empty() || reserved {
        stem = format!("profile {stem}").trim().to_owned();
    }
    stem
}

/// The profile file for `settings` called `name`: its whole setup plus the header.
#[must_use]
pub fn profile_json(name: &str, settings: &ModSettings, saved_unix: u64) -> Value {
    let mut root = match settings.setup_json() {
        Value::Object(map) => map,
        _ => Map::new(),
    };
    let mut header = Map::new();
    header.insert("name".into(), Value::String(clean_name(name)));
    header.insert("saved".into(), Value::Number(Number::from(saved_unix)));
    root.insert(HEADER.into(), Value::Object(header));
    Value::Object(root)
}

/// Seconds since 1970 now (0 if the clock is before it).
#[must_use]
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// One profile in the library.
#[derive(Debug, Clone)]
pub struct ProfileEntry {
    /// The name the user gave it.
    pub name: String,
    /// Its file.
    pub file: PathBuf,
    /// When it was saved (seconds since 1970), if the file says.
    pub saved_unix: Option<u64>,
    /// The setup, parsed when the folder was read. `None`: the file is not a settings file (it
    /// cannot be loaded; it can be deleted).
    pub setup: Option<Arc<ModSettings>>,
    /// Lines from parsing it (values out of range, unreadable entries), for the log.
    pub warnings: Vec<String>,
}

impl ProfileEntry {
    /// Parse the profile file text `text` read from `file`.
    #[must_use]
    pub fn parse(file: &Path, text: &str, catalog: &'static [&'static ModDescriptor]) -> Self {
        let json: Option<Value> = serde_json::from_str(text).ok();
        let header = json
            .as_ref()
            .and_then(|root| root.get(HEADER))
            .and_then(Value::as_object);
        let name = header
            .and_then(|header| header.get("name"))
            .and_then(Value::as_str)
            .map(clean_name)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| {
                file.file_stem()
                    .map(|stem| clean_name(&stem.to_string_lossy()))
                    .unwrap_or_default()
            });
        let saved_unix = header
            .and_then(|header| header.get("saved"))
            .and_then(Value::as_u64);
        let (settings, report) = ModSettings::from_json_str_with_catalog(text, catalog);
        let setup = (!report.corrupt).then(|| Arc::new(settings));
        ProfileEntry {
            name,
            file: file.to_path_buf(),
            saved_unix,
            setup,
            warnings: report.warnings,
        }
    }

    /// Read and parse the profile file at `file`.
    ///
    /// # Errors
    ///
    /// The file cannot be read.
    pub fn read(file: &Path, catalog: &'static [&'static ModDescriptor]) -> std::io::Result<Self> {
        let text = read_profile_text(file)?;
        Ok(ProfileEntry::parse(file, &text, catalog))
    }
}

/// The text of the profile file at `file`, read up to [`MAX_PROFILE_BYTES`].
///
/// Profiles are made to be shared: a file far larger than any setup (tens
/// of kilobytes) is listed as unreadable rather than read whole.
/// Renaming reads through this too: the menu offers Rename on every listed row, unreadable ones
/// included.
fn read_profile_text(file: &Path) -> std::io::Result<String> {
    let mut text = String::new();
    std::io::Read::read_to_string(
        &mut std::io::Read::take(std::fs::File::open(file)?, MAX_PROFILE_BYTES + 1),
        &mut text,
    )?;
    if text.len() as u64 > MAX_PROFILE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "the profile file is too large",
        ));
    }
    Ok(text)
}

/// Something the library finished, for the menu to show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Notice {
    /// A profile was saved (its name).
    Saved(String),
    /// A profile was renamed (its new name).
    Renamed(String),
    /// A profile was deleted (its name).
    Deleted(String),
    /// Something failed (what, in words).
    Failed(String),
}

#[derive(Debug, Default)]
struct State {
    /// The folder has been read once.
    scanned: bool,
    /// Saved profiles, by name (case-insensitive).
    entries: Vec<ProfileEntry>,
    /// The setup from before the last load.
    backup: Option<ProfileEntry>,
    /// Jobs queued and not finished.
    pending: usize,
    notices: Vec<Notice>,
}

#[derive(Debug, Default)]
struct Inner {
    state: Mutex<State>,
    idle: Condvar,
    /// Moves whenever the list, the backup or the notices change.
    generation: AtomicU64,
}

impl Inner {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn changed(&self) {
        self.generation.fetch_add(1, Ordering::Release);
    }
}

#[derive(Debug)]
enum Job {
    Scan,
    Save {
        name: String,
        json: Value,
        replace: Option<PathBuf>,
    },
    Rename {
        file: PathBuf,
        name: String,
    },
    Delete {
        file: PathBuf,
    },
    Backup {
        text: String,
    },
}

/// The profiles folder and its worker thread. The menu opens one while it is up; dropping it
/// lets the worker finish what is queued and stop.
#[derive(Debug)]
pub struct ProfileLibrary {
    dir: PathBuf,
    catalog: &'static [&'static ModDescriptor],
    inner: Arc<Inner>,
    /// `None` when no thread could be started: jobs then run on the caller's thread.
    jobs: Option<Sender<Job>>,
}

impl ProfileLibrary {
    /// Open the profiles folder of `game_dir` (it is created on the first save) and start
    /// reading it in the background.
    #[must_use]
    pub fn open(game_dir: &Path, catalog: &'static [&'static ModDescriptor]) -> ProfileLibrary {
        let dir = game_dir.join(PROFILES_DIR);
        let inner = Arc::new(Inner::default());
        let (tx, rx) = mpsc::channel::<Job>();
        let worker = {
            let (dir, inner) = (dir.clone(), Arc::clone(&inner));
            std::thread::Builder::new()
                .name("mod-profiles".into())
                .spawn(move || work(&rx, &dir, catalog, &inner))
        };
        let library = ProfileLibrary {
            dir,
            catalog,
            inner,
            jobs: worker.is_ok().then_some(tx),
        };
        library.submit(Job::Scan);
        library
    }

    /// The profiles folder.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Moves whenever the list, the backup or the notices change: compare it to refresh a copy.
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.inner.generation.load(Ordering::Acquire)
    }

    /// The folder has been read once.
    #[must_use]
    pub fn scanned(&self) -> bool {
        self.inner.lock().scanned
    }

    /// Some file work is still queued or running.
    #[must_use]
    pub fn busy(&self) -> bool {
        self.inner.lock().pending > 0
    }

    /// The saved profiles, by name (case-insensitive).
    #[must_use]
    pub fn entries(&self) -> Vec<ProfileEntry> {
        self.inner.lock().entries.clone()
    }

    /// The setup from before the last load, if there is one.
    #[must_use]
    pub fn backup(&self) -> Option<ProfileEntry> {
        self.inner.lock().backup.clone()
    }

    /// The saved profile called `name` (compared trimmed and case-insensitively).
    #[must_use]
    pub fn find(&self, name: &str) -> Option<ProfileEntry> {
        let wanted = clean_name(name).to_lowercase();
        self.inner
            .lock()
            .entries
            .iter()
            .find(|entry| entry.name.to_lowercase() == wanted)
            .cloned()
    }

    /// Save the whole setup of `settings` as profile `name` (cleaned; an empty name becomes
    /// "Setup N"). `replace`: the file of the profile it overwrites (the menu confirmed); a new
    /// name gets a new file. The JSON is built now, the file written in the background.
    pub fn save(&self, name: &str, settings: &ModSettings, replace: Option<&Path>) {
        let mut name = clean_name(name);
        if name.is_empty() {
            let state = self.inner.lock();
            let mut n = state.entries.len() + 1;
            loop {
                let candidate = format!("Setup {n}");
                let taken = state
                    .entries
                    .iter()
                    .any(|entry| entry.name.eq_ignore_ascii_case(&candidate));
                if !taken {
                    name = candidate;
                    break;
                }
                n += 1;
            }
        }
        let json = profile_json(&name, settings, now_unix());
        self.submit(Job::Save {
            name,
            json,
            replace: replace.map(Path::to_path_buf),
        });
    }

    /// Rename the profile in `file` to `name` (cleaned; an empty name does nothing).
    pub fn rename(&self, file: &Path, name: &str) {
        let name = clean_name(name);
        if !name.is_empty() {
            self.submit(Job::Rename {
                file: file.to_path_buf(),
                name,
            });
        }
    }

    /// Delete the profile in `file`.
    pub fn delete(&self, file: &Path) {
        self.submit(Job::Delete {
            file: file.to_path_buf(),
        });
    }

    /// Keep `settings`' whole setup as "Before last load", before loading another one: in
    /// memory at once (it can be loaded back right away), in [`BACKUP_FILE`] soon after.
    pub fn keep_backup(&self, settings: &ModSettings) {
        let saved = now_unix();
        let json = profile_json(BACKUP_NAME, settings, saved);
        let text = serde_json::to_string_pretty(&json).unwrap_or_default();
        let entry = ProfileEntry {
            name: BACKUP_NAME.to_owned(),
            file: self.dir.join(BACKUP_FILE),
            saved_unix: Some(saved),
            setup: Some(Arc::new(settings.clone())),
            warnings: Vec::new(),
        };
        self.inner.lock().backup = Some(entry);
        self.inner.changed();
        self.submit(Job::Backup { text });
    }

    /// What finished since the last call, oldest first.
    pub fn take_notices(&self) -> Vec<Notice> {
        let notices = std::mem::take(&mut self.inner.lock().notices);
        if !notices.is_empty() {
            self.inner.changed();
        }
        notices
    }

    /// Wait until every queued job has finished (tests; the menu never waits).
    pub fn wait_idle(&self) {
        let mut state = self.inner.lock();
        while state.pending > 0 {
            state = self
                .inner
                .idle
                .wait(state)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }

    fn submit(&self, job: Job) {
        self.inner.lock().pending += 1;
        let job = match &self.jobs {
            Some(tx) => match tx.send(job) {
                Ok(()) => return,
                Err(mpsc::SendError(job)) => job,
            },
            None => job,
        };
        // No worker: do it here.
        run(job, &self.dir, self.catalog, &self.inner);
        finish(&self.inner);
    }
}

/// The worker thread: run jobs in order until the library is dropped.
fn work(
    jobs: &Receiver<Job>,
    dir: &Path,
    catalog: &'static [&'static ModDescriptor],
    inner: &Inner,
) {
    for job in jobs {
        run(job, dir, catalog, inner);
        finish(inner);
    }
}

/// A job is done: the generation moves first, then waiters wake, so anyone woken by
/// [`ProfileLibrary::wait_idle`] already sees the change.
fn finish(inner: &Inner) {
    inner.changed();
    let mut state = inner.lock();
    state.pending = state.pending.saturating_sub(1);
    drop(state);
    inner.idle.notify_all();
}

/// One job. File work happens with the lock released; only the list update takes it.
fn run(job: Job, dir: &Path, catalog: &'static [&'static ModDescriptor], inner: &Inner) {
    match job {
        Job::Scan => {
            let (entries, backup) = scan(dir, catalog);
            let mut state = inner.lock();
            state.entries = entries;
            // A backup kept in memory this session is the newest one: the file only catches up.
            if state.backup.is_none() {
                state.backup = backup;
            }
            state.scanned = true;
        }
        Job::Save {
            name,
            mut json,
            replace,
        } => {
            let file = replace.unwrap_or_else(|| {
                let state = inner.lock();
                match state
                    .entries
                    .iter()
                    .find(|entry| entry.name.eq_ignore_ascii_case(&name))
                {
                    Some(entry) => entry.file.clone(),
                    None => unique_file(dir, &file_stem(&name), &state.entries),
                }
            });
            if let Value::Object(root) = &mut json {
                if let Some(Value::Object(header)) = root.get_mut(HEADER) {
                    header.insert("name".into(), Value::String(name.clone()));
                }
            }
            let text = serde_json::to_string_pretty(&json).unwrap_or_default();
            match write_file(&file, &text) {
                Ok(()) => {
                    let entry = ProfileEntry::parse(&file, &text, catalog);
                    let mut state = inner.lock();
                    state.entries.retain(|e| e.file != file);
                    state.entries.push(entry);
                    sort(&mut state.entries);
                    state.notices.push(Notice::Saved(name));
                }
                Err(error) => inner.lock().notices.push(Notice::Failed(format!(
                    "Could not save \u{201c}{name}\u{201d}: {error}"
                ))),
            }
        }
        Job::Rename { file, name } => {
            let notice = match rename_file(dir, &file, &name, inner) {
                Ok(new_file) => {
                    let mut state = inner.lock();
                    if let Some(entry) = state.entries.iter_mut().find(|e| e.file == file) {
                        entry.name.clone_from(&name);
                        entry.file = new_file;
                    }
                    sort(&mut state.entries);
                    Notice::Renamed(name)
                }
                Err(error) => Notice::Failed(format!(
                    "Could not rename to \u{201c}{name}\u{201d}: {error}"
                )),
            };
            inner.lock().notices.push(notice);
        }
        Job::Delete { file } => {
            let result = match std::fs::remove_file(&file) {
                Err(error) if error.kind() != std::io::ErrorKind::NotFound => Err(error),
                _ => Ok(()),
            };
            let mut state = inner.lock();
            let name = state
                .entries
                .iter()
                .find(|e| e.file == file)
                .map(|e| e.name.clone())
                .unwrap_or_default();
            match result {
                Ok(()) => {
                    state.entries.retain(|e| e.file != file);
                    state.notices.push(Notice::Deleted(name));
                }
                Err(error) => state.notices.push(Notice::Failed(format!(
                    "Could not delete \u{201c}{name}\u{201d}: {error}"
                ))),
            }
        }
        Job::Backup { text } => {
            if let Err(error) = write_file(&dir.join(BACKUP_FILE), &text) {
                inner.lock().notices.push(Notice::Failed(format!(
                    "Could not keep a copy of your last setup on disk ({error}). It stays \
                     loadable until the menu closes."
                )));
            }
        }
    }
}

/// Read every profile in `dir` (a missing folder is an empty library), and the backup.
fn scan(
    dir: &Path,
    catalog: &'static [&'static ModDescriptor],
) -> (Vec<ProfileEntry>, Option<ProfileEntry>) {
    let mut entries = Vec::new();
    let mut backup = None;
    let Ok(listing) = std::fs::read_dir(dir) else {
        return (entries, backup);
    };
    for item in listing.flatten() {
        let path = item.path();
        let Some(file_name) = path.file_name().map(|n| n.to_string_lossy().into_owned()) else {
            continue;
        };
        let is_json = path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("json"));
        if !is_json || !path.is_file() {
            continue;
        }
        if file_name.eq_ignore_ascii_case(BACKUP_FILE) {
            backup = ProfileEntry::read(&path, catalog).ok().map(|mut entry| {
                entry.name = BACKUP_NAME.to_owned();
                entry
            });
            continue;
        }
        if file_name.starts_with(['_', '.']) {
            continue;
        }
        match ProfileEntry::read(&path, catalog) {
            Ok(entry) => entries.push(entry),
            Err(error) => entries.push(ProfileEntry {
                name: path
                    .file_stem()
                    .map(|stem| clean_name(&stem.to_string_lossy()))
                    .unwrap_or_default(),
                file: path.clone(),
                saved_unix: None,
                setup: None,
                warnings: vec![format!("could not read {}: {error}", path.display())],
            }),
        }
    }
    sort(&mut entries);
    (entries, backup)
}

fn sort(entries: &mut [ProfileEntry]) {
    entries.sort_by_cached_key(|entry| entry.name.to_lowercase());
}

/// A file in `dir` for a new profile: `<stem>.json`, or `<stem> 2.json`... when another
/// profile or file has it.
fn unique_file(dir: &Path, stem: &str, entries: &[ProfileEntry]) -> PathBuf {
    let taken = |path: &Path| {
        path.exists()
            || entries.iter().any(|entry| {
                entry
                    .file
                    .to_string_lossy()
                    .eq_ignore_ascii_case(&path.to_string_lossy())
            })
    };
    let mut path = dir.join(format!("{stem}.json"));
    let mut n = 2;
    while taken(&path) {
        path = dir.join(format!("{stem} {n}.json"));
        n += 1;
    }
    path
}

/// Write `text` to `path` through `<path>.tmp`, flushed, and a rename (the folder is created
/// first; [`mc_core::durable::write_atomic`]).
fn write_file(path: &Path, text: &str) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = text.to_owned();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    mc_core::durable::write_atomic(path, text.as_bytes())
}

/// Rename the profile in `file` to `name`: the name in its header changes, and the file gets the
/// new name's file name unless that is the same name (letter case aside). Returns the file.
fn rename_file(dir: &Path, file: &Path, name: &str, inner: &Inner) -> std::io::Result<PathBuf> {
    let text = read_profile_text(file)?;
    let mut json: Value = serde_json::from_str(&text)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
    let Value::Object(root) = &mut json else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "not a profile",
        ));
    };
    let header = root
        .entry(HEADER)
        .or_insert_with(|| Value::Object(Map::new()));
    if let Value::Object(header) = header {
        header.insert("name".into(), Value::String(name.to_owned()));
    }
    let text = serde_json::to_string_pretty(&json).unwrap_or_default();
    let old_stem = file
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_default();
    let new_stem = file_stem(name);
    if old_stem.eq_ignore_ascii_case(&new_stem) {
        write_file(file, &text)?;
        return Ok(file.to_path_buf());
    }
    let others: Vec<ProfileEntry> = inner
        .lock()
        .entries
        .iter()
        .filter(|entry| entry.file != file)
        .cloned()
        .collect();
    let new_file = unique_file(dir, &new_stem, &others);
    write_file(&new_file, &text)?;
    std::fs::remove_file(file)?;
    Ok(new_file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog;

    fn temp_game_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "mc-rs-profiles-{tag}-{}-{}",
            std::process::id(),
            now_unix()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn names_and_file_names_are_safe() {
        assert_eq!(clean_name("  Bed\nwars  "), "Bedwars");
        assert_eq!(clean_name(&"x".repeat(80)).chars().count(), NAME_MAX);
        assert_eq!(file_stem("PvP: 1.8 / Bed Wars"), "PvP_ 1_8 _ Bed Wars");
        assert_eq!(file_stem("con"), "profile con");
        assert_eq!(file_stem("COM1"), "profile COM1");
        assert_eq!(file_stem("COMPACT"), "COMPACT");
        assert_eq!(file_stem("_hidden"), "hidden");
        assert_eq!(file_stem("..."), "profile");
        assert_eq!(file_stem(""), "profile");
        assert_eq!(file_stem("Ünïcode ok"), "Ünïcode ok");
    }

    /// The whole flow on a real folder: save, list, find, load back exactly (atomic, one
    /// generation), rename (the file follows the name), save over, delete, and a broken file
    /// that is listed but can never be loaded.
    #[test]
    fn save_list_load_rename_replace_delete_through_real_files() {
        let game = temp_game_dir("flow");
        let library = ProfileLibrary::open(&game, catalog::all());
        library.wait_idle();
        assert!(library.scanned() && library.entries().is_empty());
        // A setup worth saving: switches, a value, a HUD position, a pin, the HUD style.
        let mut mine = ModSettings::new();
        let fps = mine.index_of("fps").unwrap();
        mine.set_enabled_at(fps, true);
        mine.set_f32("zoom", "sensitivity", 0.4);
        mine.set_placement(
            "fps",
            "fps",
            crate::hud::HudPlacement::new(crate::hud::HudAnchor::TopRight, -4.0, 8.0),
        );
        mine.set_pinned_at(mine.index_of("zoom").unwrap(), true);
        mine.set_hud_style_value("text_color", crate::OptionValue::Color(0xFFFF_5555));
        library.save("Bedwars", &mine, None);
        library.wait_idle();
        let entries = library.entries();
        assert_eq!(entries.len(), 1, "{entries:?}");
        let saved = &entries[0];
        assert_eq!(saved.name, "Bedwars");
        assert!(saved
            .file
            .ends_with(Path::new(PROFILES_DIR).join("Bedwars.json")));
        assert!(saved.saved_unix.is_some());
        assert!(saved.warnings.is_empty(), "{:?}", saved.warnings);
        let setup = saved.setup.clone().expect("parsed");
        assert!(mine.setup_eq(&setup), "the file holds exactly this setup");
        assert_eq!(library.take_notices(), [Notice::Saved("Bedwars".into())]);
        // Loading it into a fresh store: everything in one generation step.
        let mut other = ModSettings::new();
        let menu_key = 88;
        other.set_menu_key(menu_key);
        let before = other.generation();
        assert!(!other.setup_eq(&setup));
        assert!(other.apply_setup(&setup));
        assert_eq!(other.generation(), before + 1, "one step");
        assert!(other.setup_eq(&mine));
        assert_eq!(
            other.menu_key(),
            menu_key,
            "the menu key is not part of a setup"
        );
        assert!(!other.apply_setup(&setup), "nothing left to change");
        // A second library sees the same list (it reads the folder).
        let again = ProfileLibrary::open(&game, catalog::all());
        again.wait_idle();
        assert_eq!(again.entries().len(), 1);
        assert!(
            again.find(" bedwars ").is_some(),
            "names compare case-insensitively"
        );
        // Rename: the header and the file follow.
        library.rename(&saved.file, "Bed Wars 2");
        library.wait_idle();
        let renamed = library.find("Bed Wars 2").expect("renamed");
        assert!(renamed.file.ends_with("Bed Wars 2.json"));
        assert!(!saved.file.exists(), "the old file is gone");
        let reread = ProfileEntry::read(&renamed.file, catalog::all()).unwrap();
        assert_eq!(reread.name, "Bed Wars 2");
        // Letter case only: the same file.
        library.rename(&renamed.file, "bed wars 2");
        library.wait_idle();
        let recased = library.find("bed wars 2").unwrap();
        assert_eq!(recased.name, "bed wars 2");
        assert!(recased.file.exists());
        // Save over it (the menu confirmed) with another setup.
        let fresh = ModSettings::new();
        library.save("bed wars 2", &fresh, Some(&recased.file));
        library.wait_idle();
        assert_eq!(library.entries().len(), 1);
        let over = library.find("bed wars 2").unwrap();
        assert!(fresh.setup_eq(over.setup.as_ref().unwrap()));
        // An empty name gets one.
        library.save("   ", &mine, None);
        library.wait_idle();
        assert!(library.find("Setup 2").is_some(), "{:?}", library.entries());
        // A broken file is listed, never loadable; it can be deleted.
        let broken = game.join(PROFILES_DIR).join("Broken.json");
        std::fs::write(&broken, "{ not json").unwrap();
        let third = ProfileLibrary::open(&game, catalog::all());
        third.wait_idle();
        let bad = third.find("Broken").expect("listed");
        assert!(bad.setup.is_none(), "cannot be loaded");
        third.delete(&bad.file);
        third.wait_idle();
        assert!(!broken.exists());
        assert!(third.find("Broken").is_none());
        let _ = std::fs::remove_dir_all(&game);
    }

    /// The backup is loadable at once, written to disk soon after, and read back by the next
    /// library; it is not listed with the saved profiles.
    #[test]
    fn the_setup_before_a_load_is_kept() {
        let game = temp_game_dir("backup");
        let library = ProfileLibrary::open(&game, catalog::all());
        let mut mine = ModSettings::new();
        mine.set_enabled("cps", true);
        library.keep_backup(&mine);
        let kept = library.backup().expect("in memory at once");
        assert!(mine.setup_eq(kept.setup.as_ref().unwrap()));
        library.wait_idle();
        assert!(game.join(PROFILES_DIR).join(BACKUP_FILE).exists());
        let next = ProfileLibrary::open(&game, catalog::all());
        next.wait_idle();
        let read = next.backup().expect("read back");
        assert_eq!(read.name, BACKUP_NAME);
        assert!(mine.setup_eq(read.setup.as_ref().unwrap()));
        assert!(next.entries().is_empty(), "not a saved profile");
        let _ = std::fs::remove_dir_all(&game);
    }

    /// A profile saved by another build: unknown mods and options are fine, missing mods load
    /// as their defaults, and the header is not mistaken for a setting.
    #[test]
    fn a_profile_from_another_build_loads() {
        let text = r##"{
            "format": 1,
            "profile": {"name": "Old", "saved": 5},
            "mods": {
                "fps": {"enabled": true, "options": {"text_color": "#FF00FF00", "future": 1}},
                "a_future_mod": {"enabled": true}
            }
        }"##;
        let entry = ProfileEntry::parse(Path::new("Old.json"), text, catalog::all());
        assert_eq!(entry.name, "Old");
        assert_eq!(entry.saved_unix, Some(5));
        assert!(entry.warnings.is_empty(), "{:?}", entry.warnings);
        let setup = entry.setup.unwrap();
        assert!(setup.enabled("fps"));
        assert_eq!(setup.color("fps", "text_color"), 0xFF00_FF00);
        assert!(
            !setup.uses_hud_style_at(setup.index_of("fps").unwrap()),
            "styled: Custom"
        );
        assert!(!setup.enabled("zoom"), "a mod it does not mention: default");
        let diff = ModSettings::new().setup_diff(&setup);
        assert_eq!(diff.turns_on, [setup.index_of("fps").unwrap()]);
        assert!(diff.settings.contains(&setup.index_of("fps").unwrap()));
        assert_eq!(diff.mods_changed(), 1);
        // Not JSON at all: listed, not loadable, named after its file.
        let bad = ProfileEntry::parse(Path::new("x/Junk.json"), "nope", catalog::all());
        assert!(bad.setup.is_none());
        assert_eq!(bad.name, "Junk");
    }

    /// A shared profile far larger than any setup is refused without
    /// being read whole; one at the cap still reads.
    #[test]
    fn an_oversized_profile_file_is_refused_not_read_whole() {
        let dir = temp_game_dir("oversized");
        let huge = dir.join("Huge.json");
        let file = std::fs::File::create(&huge).expect("create");
        file.set_len(1 << 30).expect("1 GiB");
        drop(file);
        let error = ProfileEntry::read(&huge, catalog::all()).expect_err("refused");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);

        let ok = dir.join("Fine.json");
        let text = profile_json("Fine", &ModSettings::new(), 7).to_string();
        let padded = format!(
            "{text}{}",
            " ".repeat(MAX_PROFILE_BYTES as usize - text.len())
        );
        std::fs::write(&ok, padded).expect("write");
        let entry = ProfileEntry::read(&ok, catalog::all()).expect("at the cap: read");
        assert_eq!(entry.name, "Fine");
        assert!(entry.setup.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A profile over the cap is listed as unreadable, and the
    /// menu offers Rename on every row: renaming it is refused without reading it whole, and
    /// the file stays as it is.
    #[test]
    fn an_oversized_profile_is_not_read_whole_to_rename_it() {
        let game = temp_game_dir("rename-oversized");
        let dir = game.join(PROFILES_DIR);
        std::fs::create_dir_all(&dir).expect("profiles folder");
        // A real profile padded one byte past the cap: still valid JSON, so only the cap
        // stands between it and a full read, parse and rewrite.
        let text = profile_json("Big", &ModSettings::new(), 7).to_string();
        let padded = format!(
            "{text}{}",
            " ".repeat(MAX_PROFILE_BYTES as usize + 1 - text.len())
        );
        let big = dir.join("Big.json");
        std::fs::write(&big, &padded).expect("write");
        let library = ProfileLibrary::open(&game, catalog::all());
        library.wait_idle();
        let listed = library.find("Big").expect("listed");
        assert!(listed.setup.is_none(), "listed as unreadable");
        library.take_notices();

        library.rename(&listed.file, "Renamed");
        library.wait_idle();
        let notices = library.take_notices();
        assert!(
            matches!(notices.as_slice(), [Notice::Failed(why)] if why.contains("too large")),
            "{notices:?}"
        );
        assert!(!dir.join("Renamed.json").exists(), "nothing written");
        assert_eq!(
            std::fs::metadata(&big).expect("still there").len(),
            padded.len() as u64,
            "the file is untouched"
        );
        let _ = std::fs::remove_dir_all(&game);
    }
}
