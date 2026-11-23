//! The waypoint list and its file, `<gameDir>/mc-rs-waypoints.json`.
//!
//! Pure data plus `serde_json`, so every rule here is unit-testable without a client:
//!
//! * **Per server.** Waypoints are kept under a server key, [`server_key`]`(host, port)`, the
//!   lowercased `host:port`. Keys starting with `~` (`~replay`, `~local`) are session-only: they
//!   live in memory and are never written ([`is_session_only`]).
//! * **Auto name and colour.** [`WaypointStore::create`] names a waypoint `Waypoint N`, with `N`
//!   the smallest number not already used on that server, coloured
//!   [`AUTO_COLORS`]`[(N - 1) % 7]` (the readable chat colours; red is kept for deaths).
//! * **One death waypoint per server.** [`WaypointStore::set_death`] replaces the previous one.
//! * **Loading never loses data.** A missing file is an empty store. Unknown fields are ignored,
//!   and a bad entry is skipped and counted. A file that will not parse at all is renamed to
//!   `<file>.bad`, so the next save cannot overwrite it. A file that cannot be read, or cannot
//!   be moved aside, makes [`LoadReport::read_only`] true: the caller must not write it.
//! * **Atomic writes.** [`write_atomic`] writes `<file>.tmp` and renames it over the file. The
//!   text is written by hand, one waypoint per line, so the file is easy to edit.
//!
//! ```json
//! {
//!   "version": 1,
//!   "servers": {
//!     "play.example.net:25565": [
//!       {"name": "Waypoint 1", "x": 120, "y": 64, "z": -30, "dim": 0, "color": "FFAA00", "visible": true},
//!       {"name": "Death", "x": 10, "y": 70, "z": 5, "dim": 0, "color": "FF5555", "visible": true, "death": true}
//!     ]
//!   }
//! }
//! ```

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The waypoint file, relative to the game directory (`--gameDir`).
pub const FILE_NAME: &str = "mc-rs-waypoints.json";

/// The `"version"` written to the file.
pub const FORMAT_VERSION: u32 = 1;

/// Colours new waypoints get in turn: gold, yellow, green, aqua, blue, light purple, white.
/// These are vanilla's chat colours (`§6 §e §a §b §9 §d §f`), so the chat line that announces a
/// waypoint can show its name in the same colour. Red (`§c`) is kept for the death waypoint.
pub const AUTO_COLORS: [u32; 7] = [
    0xFF_AA00, 0xFF_FF55, 0x55_FF55, 0x55_FFFF, 0x55_55FF, 0xFF_55FF, 0xFF_FFFF,
];

/// The `§` code of each of [`AUTO_COLORS`], in the same order.
pub const AUTO_COLOR_CODES: [char; 7] = ['6', 'e', 'a', 'b', '9', 'd', 'f'];

/// The death waypoint's colour: vanilla's red, `§c`.
pub const DEATH_COLOR: u32 = 0xFF_5555;

/// The death waypoint's name.
pub const DEATH_NAME: &str = "Death";

/// Automatic names are this prefix plus a number.
pub const AUTO_NAME_PREFIX: &str = "Waypoint ";

/// Most waypoints [`WaypointStore::create`] makes on one server (the death waypoint is not
/// counted). Bounds the per-frame label work, and stops a held-down key from flooding the list.
pub const MAX_PER_SERVER: usize = 100;
/// The most waypoints loaded for one server: ten times what the
/// client creates, so a hand-edited file keeps its extras and a crafted one stays drawable.
pub const MAX_LOADED_PER_SERVER: usize = 10 * MAX_PER_SERVER;

/// Longest name kept from the file, in characters.
pub const MAX_NAME_CHARS: usize = 48;

/// The largest waypoint file read. The per-server cap above applies
/// after parsing, and parsing builds a JSON tree of about ten times the text, so a crafted file
/// in a shared game folder (200 MB of empty entries zip to a few hundred KB) used to take
/// gigabytes at the first join. What the client writes stays far below this (at most
/// [`MAX_PER_SERVER`] short entries per server). A larger file is left alone, as one that
/// cannot be read: nothing is loaded and nothing is written over it this session.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// `path`'s bytes, or an error when it is longer than `cap` (without reading past it).
fn read_capped(path: &Path, cap: u64) -> std::io::Result<Vec<u8>> {
    use std::io::Read as _;
    let file = std::fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(cap + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > cap {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("larger than {cap} bytes"),
        ));
    }
    Ok(bytes)
}

/// One saved place.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Waypoint {
    /// Runtime id, unique within a [`WaypointStore`] for the session. Never written; the menu
    /// echoes it back in its row events.
    pub id: u64,
    /// Display name. May hold `§` formatting codes (the file is hand-editable).
    pub name: String,
    /// Block X.
    pub x: i32,
    /// Block Y: the block the player's feet were in.
    pub y: i32,
    /// Block Z.
    pub z: i32,
    /// Dimension: `-1` Nether, `0` Overworld, `1` End.
    pub dim: i32,
    /// Colour, `0xRRGGBB`.
    pub color: u32,
    /// Drawn in the world. Hidden waypoints stay in the list, greyed out.
    pub visible: bool,
    /// The death waypoint: at most one per server, replaced by the next death.
    pub death: bool,
}

impl Waypoint {
    /// The centre of the marked block, in world coordinates: where labels and distances
    /// measure to.
    #[must_use]
    pub fn center(&self) -> [f64; 3] {
        [
            f64::from(self.x) + 0.5,
            f64::from(self.y) + 0.5,
            f64::from(self.z) + 0.5,
        ]
    }
}

/// The key a server's waypoints are stored under: lowercased `host:port`.
#[must_use]
pub fn server_key(host: &str, port: u16) -> String {
    let mut key = host.trim().to_ascii_lowercase();
    let _ = write!(key, ":{port}");
    key
}

/// Whether a server key is kept in memory only (`~replay`, `~local`).
#[must_use]
pub fn is_session_only(key: &str) -> bool {
    key.starts_with('~')
}

/// The chat colour code (`§x`) for a waypoint colour: exact for [`AUTO_COLORS`] and
/// [`DEATH_COLOR`], white for anything else.
#[must_use]
pub fn chat_color_code(rgb: u32) -> char {
    if rgb == DEATH_COLOR {
        return 'c';
    }
    AUTO_COLORS
        .iter()
        .position(|&color| color == rgb)
        .map_or('f', |i| AUTO_COLOR_CODES[i])
}

/// What [`WaypointStore::load`] found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// There was no file: a first run.
    pub file_missing: bool,
    /// The file would not parse, and was renamed to this path before anything could overwrite
    /// it. The store starts empty.
    pub moved_aside: Option<PathBuf>,
    /// The file exists but could not be read, or could not be moved aside. Writing would risk
    /// the user's data, so the caller must not save this session.
    pub read_only: bool,
    /// Entries skipped because they were unusable: a waypoint without a name or coordinates,
    /// or a server whose value is not a list.
    pub skipped: usize,
    /// What went wrong, for the log.
    pub error: Option<String>,
}

/// Every waypoint on every server, plus the id counter and a change counter.
#[derive(Debug, Clone, Default)]
pub struct WaypointStore {
    servers: BTreeMap<String, Vec<Waypoint>>,
    next_id: u64,
    revision: u64,
}

impl WaypointStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> WaypointStore {
        WaypointStore::default()
    }

    /// Bumped by every change. Callers compare it to rebuild what they derive from the list
    /// (labels, menu rows) only when the list changed.
    #[must_use]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// One server's waypoints, oldest first. Empty for an unknown key.
    #[must_use]
    pub fn list(&self, key: &str) -> &[Waypoint] {
        self.servers.get(key).map_or(&[], Vec::as_slice)
    }

    /// Every server key with its waypoints, in key order.
    pub fn servers(&self) -> impl Iterator<Item = (&str, &[Waypoint])> + '_ {
        self.servers
            .iter()
            .map(|(key, list)| (key.as_str(), list.as_slice()))
    }

    /// The waypoint with this runtime id, on any server.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Waypoint> {
        self.servers
            .values()
            .flat_map(|list| list.iter())
            .find(|waypoint| waypoint.id == id)
    }

    /// Waypoints across all servers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.servers.values().map(Vec::len).sum()
    }

    /// Whether there are no waypoints at all.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn fresh_id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// The `N` the next automatic name on this server gets: the smallest positive number no
    /// `Waypoint N` there already uses.
    #[must_use]
    pub fn next_auto_number(&self, key: &str) -> u32 {
        let mut used: Vec<u32> = self
            .list(key)
            .iter()
            .filter_map(|waypoint| auto_number(&waypoint.name))
            .collect();
        used.sort_unstable();
        used.dedup();
        let mut next = 1;
        for n in used {
            if n == next {
                next += 1;
            } else if n > next {
                break;
            }
        }
        next
    }

    /// Add a waypoint at `pos` with the next automatic name and colour. `None` once the server
    /// already holds [`MAX_PER_SERVER`] waypoints.
    pub fn create(&mut self, key: &str, pos: [i32; 3], dim: i32) -> Option<&Waypoint> {
        let count = self.list(key).iter().filter(|w| !w.death).count();
        if count >= MAX_PER_SERVER {
            return None;
        }
        let n = self.next_auto_number(key);
        let id = self.fresh_id();
        let waypoint = Waypoint {
            id,
            name: format!("{AUTO_NAME_PREFIX}{n}"),
            x: pos[0],
            y: pos[1],
            z: pos[2],
            dim,
            color: AUTO_COLORS[(n.saturating_sub(1) % 7) as usize],
            visible: true,
            death: false,
        };
        let list = self.servers.entry(key.to_owned()).or_default();
        list.push(waypoint);
        self.revision += 1;
        list.last()
    }

    /// Mark where the player died: removes this server's previous death waypoint and adds a
    /// new one, named [`DEATH_NAME`] and coloured [`DEATH_COLOR`].
    pub fn set_death(&mut self, key: &str, pos: [i32; 3], dim: i32) -> &Waypoint {
        let id = self.fresh_id();
        let list = self.servers.entry(key.to_owned()).or_default();
        list.retain(|waypoint| !waypoint.death);
        list.push(Waypoint {
            id,
            name: DEATH_NAME.to_owned(),
            x: pos[0],
            y: pos[1],
            z: pos[2],
            dim,
            color: DEATH_COLOR,
            visible: true,
            death: true,
        });
        self.revision += 1;
        &list[list.len() - 1]
    }

    /// Delete a waypoint. Returns whether it existed.
    pub fn remove(&mut self, id: u64) -> bool {
        for list in self.servers.values_mut() {
            if let Some(index) = list.iter().position(|waypoint| waypoint.id == id) {
                list.remove(index);
                self.revision += 1;
                return true;
            }
        }
        false
    }

    /// Show or hide a waypoint in the world. Returns its new visibility, or `None` if it does
    /// not exist.
    pub fn toggle_visible(&mut self, id: u64) -> Option<bool> {
        let waypoint = self
            .servers
            .values_mut()
            .flat_map(|list| list.iter_mut())
            .find(|waypoint| waypoint.id == id)?;
        waypoint.visible = !waypoint.visible;
        let visible = waypoint.visible;
        self.revision += 1;
        Some(visible)
    }

    /// The file text: every server except session-only ones and empty lists, one waypoint per
    /// line.
    #[must_use]
    pub fn to_json_string(&self) -> String {
        let mut out = String::with_capacity(64 + 128 * self.len());
        let _ = write!(
            out,
            "{{\n  \"version\": {FORMAT_VERSION},\n  \"servers\": {{"
        );
        let mut any = false;
        for (key, list) in &self.servers {
            if is_session_only(key) || list.is_empty() {
                continue;
            }
            out.push_str(if any { ",\n    " } else { "\n    " });
            any = true;
            push_json_string(&mut out, key);
            out.push_str(": [");
            for (i, waypoint) in list.iter().enumerate() {
                out.push_str(if i == 0 { "\n      " } else { ",\n      " });
                out.push_str("{\"name\": ");
                push_json_string(&mut out, &waypoint.name);
                let _ = write!(
                    out,
                    ", \"x\": {}, \"y\": {}, \"z\": {}, \"dim\": {}, \"color\": \"{:06X}\", \
                     \"visible\": {}",
                    waypoint.x,
                    waypoint.y,
                    waypoint.z,
                    waypoint.dim,
                    waypoint.color & 0xFF_FFFF,
                    waypoint.visible
                );
                if waypoint.death {
                    out.push_str(", \"death\": true");
                }
                out.push('}');
            }
            out.push_str("\n    ]");
        }
        out.push_str(if any { "\n  }\n}\n" } else { "}\n}\n" });
        out
    }

    /// Parse file text. Returns the store and the number of skipped entries.
    ///
    /// # Errors
    ///
    /// The text is not JSON, not an object, or has no `"servers"` object. The caller keeps such
    /// a file aside rather than overwriting it.
    pub fn from_json_str(text: &str) -> Result<(WaypointStore, usize), String> {
        // Windows editors may save UTF-8 with a byte-order mark, which JSON does not allow.
        let text = text.strip_prefix('\u{feff}').unwrap_or(text);
        let root: Value = serde_json::from_str(text).map_err(|error| error.to_string())?;
        let Value::Object(root) = root else {
            return Err("the file is not a JSON object".to_owned());
        };
        let Some(Value::Object(servers)) = root.get("servers") else {
            return Err("\"servers\" is missing or not an object".to_owned());
        };
        let mut store = WaypointStore::new();
        let mut skipped = 0;
        for (key, entries) in servers {
            let key = key.trim().to_ascii_lowercase();
            let Value::Array(entries) = entries else {
                skipped += 1;
                continue;
            };
            if key.is_empty() || is_session_only(&key) {
                skipped += entries.len().max(1);
                continue;
            }
            for entry in entries {
                // A shared or crafted file cannot make the world draw
                // hundreds of thousands of beams every frame: ten times what the client makes
                // for one server is kept, the rest is counted as skipped.
                let full = store
                    .servers
                    .get(&key)
                    .is_some_and(|list| list.len() >= MAX_LOADED_PER_SERVER);
                if full {
                    skipped += 1;
                    continue;
                }
                match parse_entry(entry) {
                    Some(mut waypoint) => {
                        waypoint.id = store.fresh_id();
                        store.servers.entry(key.clone()).or_default().push(waypoint);
                    }
                    None => skipped += 1,
                }
            }
        }
        Ok((store, skipped))
    }

    /// Load `path`. Never fails: see [`LoadReport`] for what happened.
    #[must_use]
    pub fn load(path: &Path) -> (WaypointStore, LoadReport) {
        let mut report = LoadReport::default();
        let bytes = match read_capped(path, MAX_FILE_BYTES) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                report.file_missing = true;
                return (WaypointStore::new(), report);
            }
            Err(error) => {
                report.read_only = true;
                report.error = Some(format!("could not read {}: {error}", path.display()));
                return (WaypointStore::new(), report);
            }
        };
        let parsed = std::str::from_utf8(&bytes)
            .map_err(|error| format!("not UTF-8: {error}"))
            .and_then(WaypointStore::from_json_str);
        match parsed {
            Ok((store, skipped)) => {
                report.skipped = skipped;
                (store, report)
            }
            Err(error) => {
                let bad = sibling(path, ".bad");
                match std::fs::rename(path, &bad) {
                    Ok(()) => {
                        report.moved_aside = Some(bad);
                        report.error = Some(error);
                    }
                    Err(rename_error) => {
                        report.read_only = true;
                        report.error = Some(format!(
                            "{error}; could not move it aside to {}: {rename_error}",
                            bad.display()
                        ));
                    }
                }
                (WaypointStore::new(), report)
            }
        }
    }

    /// Write the store to `path` atomically (see [`write_atomic`]).
    ///
    /// # Errors
    ///
    /// The directory is not writable.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        write_atomic(path, &self.to_json_string())
    }
}

/// Write `text` to `<path>.tmp`, flush it to the disk, then rename it over `path`, so neither a
/// crash nor a power cut mid-write leaves a truncated or zero-filled file
/// ([`mc_core::durable::write_atomic`]).
///
/// # Errors
///
/// The directory is not writable (the old file is then untouched).
pub fn write_atomic(path: &Path, text: &str) -> std::io::Result<()> {
    mc_core::durable::write_atomic(path, text.as_bytes())
}

/// `path` with `suffix` appended to its file name.
fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(suffix);
    PathBuf::from(name)
}

/// `N` if `name` is exactly an automatic name, `Waypoint N`.
fn auto_number(name: &str) -> Option<u32> {
    let digits = name.strip_prefix(AUTO_NAME_PREFIX)?;
    if digits.is_empty() || digits.starts_with('0') || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn push_json_string(out: &mut String, text: &str) {
    match serde_json::to_string(text) {
        Ok(quoted) => out.push_str(&quoted),
        Err(_) => out.push_str("\"\""),
    }
}

/// Control characters dropped, trimmed, at most [`MAX_NAME_CHARS`] characters.
fn clean_name(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .chars()
        .take(MAX_NAME_CHARS)
        .collect()
}

fn json_i32(value: &Value) -> Option<i32> {
    let Value::Number(number) = value else {
        return None;
    };
    if let Some(n) = number.as_i64() {
        return i32::try_from(n).ok();
    }
    let f = number.as_f64()?;
    if !f.is_finite() {
        return None;
    }
    let floored = f.floor();
    (f64::from(i32::MIN)..=f64::from(i32::MAX))
        .contains(&floored)
        .then_some(floored as i32)
}

fn json_color(value: &Value) -> Option<u32> {
    match value {
        Value::String(text) => crate::settings::parse_color(text).map(|argb| argb & 0xFF_FFFF),
        Value::Number(number) => number
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .map(|v| v & 0xFF_FFFF),
        _ => None,
    }
}

/// One waypoint from the file. The name and coordinates are required; anything else that is
/// missing or unusable falls back to its default rather than losing the waypoint.
fn parse_entry(entry: &Value) -> Option<Waypoint> {
    let Value::Object(map) = entry else {
        return None;
    };
    let name = clean_name(map.get("name")?.as_str()?);
    let x = json_i32(map.get("x")?)?;
    let y = json_i32(map.get("y")?)?;
    let z = json_i32(map.get("z")?)?;
    let dim = map.get("dim").and_then(json_i32).unwrap_or(0);
    let death = map.get("death").and_then(Value::as_bool).unwrap_or(false);
    let color = map.get("color").and_then(json_color).unwrap_or(if death {
        DEATH_COLOR
    } else {
        AUTO_COLORS[0]
    });
    let visible = map.get("visible").and_then(Value::as_bool).unwrap_or(true);
    Some(Waypoint {
        id: 0,
        name,
        x,
        y,
        z,
        dim,
        color,
        visible,
        death,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "play.example.net:25565";
    const B: &str = "other.example.net:25566";

    /// A fresh directory under the system temp dir, removed by the test.
    fn temp_dir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("mc-rs-waypoints-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    /// Everything that is persisted, without the runtime ids.
    fn persisted(list: &[Waypoint]) -> Vec<(String, [i32; 4], u32, bool, bool)> {
        list.iter()
            .map(|w| {
                (
                    w.name.clone(),
                    [w.x, w.y, w.z, w.dim],
                    w.color,
                    w.visible,
                    w.death,
                )
            })
            .collect()
    }

    #[test]
    fn server_keys_are_lowercase_host_and_port() {
        assert_eq!(server_key("Play.Example.NET", 25565), A);
        assert_eq!(server_key(" mc.local ", 7), "mc.local:7");
        assert!(is_session_only("~replay"));
        assert!(!is_session_only(A));
    }

    #[test]
    fn names_take_the_smallest_free_number_and_colours_cycle() {
        let mut store = WaypointStore::new();
        for _ in 0..3 {
            store.create(A, [0, 64, 0], 0).expect("room");
        }
        let names: Vec<_> = store.list(A).iter().map(|w| w.name.as_str()).collect();
        assert_eq!(names, ["Waypoint 1", "Waypoint 2", "Waypoint 3"]);
        assert_eq!(store.list(A)[1].color, 0xFF_FF55);

        let second = store.list(A)[1].id;
        assert!(store.remove(second));
        let refill = store.create(A, [1, 2, 3], -1).expect("room").clone();
        assert_eq!(refill.name, "Waypoint 2", "the gap is reused");
        assert_eq!(refill.color, AUTO_COLORS[1]);
        assert_eq!((refill.x, refill.y, refill.z, refill.dim), (1, 2, 3, -1));

        // A hand-edited name that only looks automatic does not take a number.
        let mut store = WaypointStore::new();
        for _ in 0..7 {
            store.create(A, [0, 0, 0], 0);
        }
        let eighth = store.create(A, [0, 0, 0], 0).expect("room");
        assert_eq!(eighth.name, "Waypoint 8");
        assert_eq!(eighth.color, AUTO_COLORS[0], "colours wrap after seven");
        assert_eq!(auto_number("Waypoint 08"), None);
        assert_eq!(auto_number("Waypoint 9x"), None);
        assert_eq!(auto_number("Waypoint "), None);
        assert_eq!(auto_number("Base"), None);
    }

    #[test]
    fn servers_are_kept_separate() {
        let mut store = WaypointStore::new();
        store.create(A, [1, 1, 1], 0);
        store.create(A, [2, 2, 2], 0);
        let on_b = store.create(B, [3, 3, 3], 0).expect("room").clone();
        assert_eq!(on_b.name, "Waypoint 1", "numbering is per server");
        assert_eq!(store.list(A).len(), 2);
        assert_eq!(store.list(B).len(), 1);
        assert!(store.list("nowhere:1").is_empty());

        // Deleting on B leaves A alone.
        assert!(store.remove(on_b.id));
        assert!(store.list(B).is_empty());
        assert_eq!(store.list(A).len(), 2);
    }

    #[test]
    fn a_death_replaces_the_previous_death_on_that_server_only() {
        let mut store = WaypointStore::new();
        store.create(A, [5, 5, 5], 0);
        let first = store.set_death(A, [10, 70, 5], 0).clone();
        store.set_death(B, [-1, -2, -3], -1);
        let revision = store.revision();
        let second = store.set_death(A, [20, 71, 6], 1).clone();
        assert!(store.revision() > revision);
        assert_ne!(first.id, second.id);

        let deaths: Vec<_> = store.list(A).iter().filter(|w| w.death).collect();
        assert_eq!(deaths.len(), 1, "one death waypoint per server");
        assert_eq!((deaths[0].x, deaths[0].y, deaths[0].z), (20, 71, 6));
        assert_eq!(deaths[0].dim, 1);
        assert_eq!(deaths[0].name, DEATH_NAME);
        assert_eq!(deaths[0].color, DEATH_COLOR);
        assert!(store.get(first.id).is_none(), "the old death is gone");
        assert_eq!(store.list(A).len(), 2, "ordinary waypoints are untouched");
        assert_eq!(
            store.list(B).iter().filter(|w| w.death).count(),
            1,
            "another server's death waypoint survives"
        );
    }

    #[test]
    fn the_file_round_trips_through_disk() {
        let dir = temp_dir("roundtrip");
        let path = dir.join(FILE_NAME);
        let mut store = WaypointStore::new();
        store.create(A, [120, 64, -30], 0);
        let hidden = store.create(A, [0, 5, 0], -1).expect("room").id;
        store.set_death(A, [10, 70, 5], 0);
        store.create(B, [-7, 200, 9], 1);
        assert_eq!(store.toggle_visible(hidden), Some(false));
        store.save(&path).expect("save");
        assert!(
            !dir.join(format!("{FILE_NAME}.tmp")).exists(),
            "tmp renamed away"
        );

        let text = std::fs::read_to_string(&path).expect("written");
        // One waypoint per line, so the file is easy to edit by hand.
        assert_eq!(text.lines().filter(|l| l.contains("\"name\"")).count(), 4);
        assert!(text.contains(
            "{\"name\": \"Waypoint 1\", \"x\": 120, \"y\": 64, \"z\": -30, \"dim\": 0, \
             \"color\": \"FFAA00\", \"visible\": true}"
        ));

        let (loaded, report) = WaypointStore::load(&path);
        assert_eq!(report, LoadReport::default(), "a clean load");
        assert_eq!(persisted(loaded.list(A)), persisted(store.list(A)));
        assert_eq!(persisted(loaded.list(B)), persisted(store.list(B)));
        assert_eq!(loaded.len(), 4);
        // Ids are runtime-only but unique after a load.
        let mut ids: Vec<_> = loaded
            .servers()
            .flat_map(|(_, l)| l.iter().map(|w| w.id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), 4);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn session_only_servers_are_never_written() {
        let mut store = WaypointStore::new();
        store.create("~replay", [1, 2, 3], 0);
        store.set_death("~local", [1, 2, 3], 0);
        store.create(A, [4, 5, 6], 0);
        let text = store.to_json_string();
        assert!(!text.contains('~'), "{text}");
        let (loaded, skipped) = WaypointStore::from_json_str(&text).expect("parses");
        assert_eq!(skipped, 0);
        assert_eq!(loaded.len(), 1);

        let empty = WaypointStore::new().to_json_string();
        assert_eq!(empty, "{\n  \"version\": 1,\n  \"servers\": {}\n}\n");
        assert!(WaypointStore::from_json_str(&empty).is_ok());
    }

    #[test]
    fn unknown_fields_are_ignored_and_bad_entries_skipped() {
        let text = "\u{feff}{\"version\": 1, \"future\": true, \"servers\": {\
            \"Play.Example.NET:25565\": [\
              {\"name\": \"Base\", \"x\": 1, \"y\": 2, \"z\": 3, \"icon\": \"star\", \
               \"color\": \"#55FF55\"},\
              {\"name\": \"No x\", \"y\": 2, \"z\": 3},\
              \"not an object\",\
              {\"name\": \"Defaults\", \"x\": 4.7, \"y\": -1, \"z\": 0, \"dim\": \"nether\", \
               \"color\": \"pink\", \"visible\": \"yes\"},\
              {\"name\": \"Too far\", \"x\": 9999999999, \"y\": 0, \"z\": 0}\
            ],\
            \"broken:1\": {\"name\": \"not a list\"},\
            \"~replay:1\": [{\"name\": \"session\", \"x\": 0, \"y\": 0, \"z\": 0}]\
        }}";
        let (store, skipped) = WaypointStore::from_json_str(text).expect("parses");
        assert_eq!(
            skipped, 5,
            "no x, not an object, out of range, not a list, ~ key"
        );
        let list = store.list(A);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].name, "Base");
        assert_eq!(list[0].color, 0x55_FF55);
        assert!(list[0].visible && !list[0].death && list[0].dim == 0);
        // Unusable optional values fall back to defaults instead of losing the waypoint.
        assert_eq!((list[1].x, list[1].dim), (4, 0));
        assert_eq!(list[1].color, AUTO_COLORS[0]);
        assert!(list[1].visible);
    }

    #[test]
    fn a_corrupt_file_is_moved_aside_and_never_overwritten() {
        let dir = temp_dir("corrupt");
        let path = dir.join(FILE_NAME);
        std::fs::write(&path, "{ this is not json").expect("write");
        let (store, report) = WaypointStore::load(&path);
        assert!(store.is_empty());
        let bad = dir.join(format!("{FILE_NAME}.bad"));
        assert_eq!(report.moved_aside.as_deref(), Some(bad.as_path()));
        assert!(
            !report.read_only,
            "the original is safe, so saving is allowed"
        );
        assert!(report.error.is_some());
        assert!(!path.exists());
        assert_eq!(
            std::fs::read_to_string(&bad).expect("kept"),
            "{ this is not json"
        );

        // A valid JSON file without a servers object is kept aside too.
        std::fs::write(&path, "[1, 2, 3]").expect("write");
        let (_, report) = WaypointStore::load(&path);
        assert!(report.moved_aside.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_is_a_first_run_and_an_unreadable_one_is_read_only() {
        let dir = temp_dir("missing");
        let (store, report) = WaypointStore::load(&dir.join(FILE_NAME));
        assert!(store.is_empty());
        assert!(report.file_missing && !report.read_only);

        // A directory where the file should be: an IO error that is not "not found".
        let path = dir.join("as-dir").join(FILE_NAME);
        std::fs::create_dir_all(&path).expect("dir");
        let (store, report) = WaypointStore::load(&path);
        assert!(store.is_empty());
        assert!(report.read_only, "{report:?}");
        assert!(report.moved_aside.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A crafted file with 50,000 waypoints for one server loads the
    /// first [`MAX_LOADED_PER_SERVER`] and reports the rest as skipped; other servers keep theirs.
    #[test]
    fn a_crafted_file_cannot_load_unbounded_waypoints_for_one_server() {
        let entry = r#"{"name":"x","x":1,"y":64,"z":1,"dim":0,"color":16711680}"#;
        let text = format!(
            r#"{{"servers":{{"{A}":[{many}],"{B}":[{entry}]}}}}"#,
            many = vec![entry; 50_000].join(","),
        );
        let (store, skipped) = WaypointStore::from_json_str(&text).expect("parse");
        assert_eq!(store.list(A).len(), MAX_LOADED_PER_SERVER);
        assert_eq!(skipped, 50_000 - MAX_LOADED_PER_SERVER);
        assert_eq!(store.list(B).len(), 1);
    }

    /// A file over [`MAX_FILE_BYTES`] is not read (let alone parsed
    /// into a tree ten times its size): nothing loads, the session must not save, and the file
    /// stays exactly where and what it was. Under the cap the same content parses (and, being
    /// no JSON object, is moved aside as before): the control.
    #[test]
    fn a_waypoint_file_over_the_byte_cap_is_left_alone() {
        let dir = temp_dir("bytes");
        let path = dir.join(FILE_NAME);
        let len = usize::try_from(MAX_FILE_BYTES).expect("cap fits") + 1;
        std::fs::write(&path, vec![b' '; len]).expect("write");
        let (store, report) = WaypointStore::load(&path);
        assert!(store.is_empty());
        assert!(report.read_only, "{report:?}");
        assert!(report.moved_aside.is_none());
        assert_eq!(std::fs::metadata(&path).expect("kept").len(), len as u64);

        std::fs::write(&path, vec![b' '; 1024]).expect("write");
        let (_, report) = WaypointStore::load(&path);
        assert!(
            !report.read_only && report.moved_aside.is_some(),
            "{report:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn creation_stops_at_the_per_server_limit_but_deaths_still_land() {
        let mut store = WaypointStore::new();
        for _ in 0..MAX_PER_SERVER {
            assert!(store.create(A, [0, 0, 0], 0).is_some());
        }
        assert!(store.create(A, [0, 0, 0], 0).is_none());
        store.set_death(A, [1, 1, 1], 0);
        assert_eq!(store.list(A).len(), MAX_PER_SERVER + 1);
        assert!(
            store.create(B, [0, 0, 0], 0).is_some(),
            "other servers are unaffected"
        );
    }

    #[test]
    fn toggling_and_removing_unknown_ids_change_nothing() {
        let mut store = WaypointStore::new();
        let id = store.create(A, [0, 0, 0], 0).expect("room").id;
        let revision = store.revision();
        assert_eq!(store.toggle_visible(id + 100), None);
        assert!(!store.remove(id + 100));
        assert_eq!(store.revision(), revision);
        assert_eq!(store.toggle_visible(id), Some(false));
        assert_eq!(store.toggle_visible(id), Some(true));
        assert!(store.revision() > revision);
    }

    #[test]
    fn chat_codes_match_the_waypoint_colours() {
        for (color, code) in AUTO_COLORS.iter().zip(AUTO_COLOR_CODES) {
            assert_eq!(chat_color_code(*color), code);
        }
        assert_eq!(chat_color_code(DEATH_COLOR), 'c');
        assert_eq!(chat_color_code(0x12_3456), 'f');
    }

    #[test]
    fn names_are_cleaned() {
        assert_eq!(clean_name("  Base\u{7}  "), "Base");
        assert_eq!(clean_name(&"x".repeat(100)).chars().count(), MAX_NAME_CHARS);
        assert_eq!(
            clean_name("§aGreen"),
            "§aGreen",
            "formatting codes are kept"
        );
    }
}
