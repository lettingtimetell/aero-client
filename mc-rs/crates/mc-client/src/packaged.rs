//! The packaged start: what `app\mc-client.exe` does when it is started with no arguments and
//! the packaging marker `app\aero-install.json` is beside it (by the portable folder's `Aero
//! Client.exe`, a shortcut, the updater, or a double-click). It does, in
//! this process, what the portable package's PowerShell launcher did before it started the game,
//! so starting Aero Client shows the game's window and nothing else:
//!
//! 1. The game's output goes to `game\logs\latest-launch.log` and its errors (the crash
//!    messages) to `game\logs\latest-launch.errors.log`, replacing the last start's
//!    ([`Packaged::open_logs`], before the log writer starts).
//! 2. The player's Minecraft 1.8.9 files are found ([`find_game_files_dir`]): the folder chosen
//!    last time, then the official launcher's (`%APPDATA%\.minecraft`), else the player is asked
//!    once, with a folder picker, and the answer is remembered for this Windows user in
//!    `%LOCALAPPDATA%\Aero Client\launcher-settings.json` (key `gameFilesDir`: the launcher's own
//!    file and key, so a choice made with it carries over). Nothing from Minecraft ships in the
//!    package.
//! 3. The game starts with `--gameDir <root>\game --assetsDir <assets> --assetIndex 1.8
//!    --versionJar <jar>` ([`game_args`]), the pipelined submit on and `game\` as its working
//!    folder ([`apply_process_settings`]). No account and no sign-in: the player signs in inside
//!    the game, on its Accounts screen.
//! 4. A copy installed by `Aero-Client-Setup.exe` keeps the version Settings > Apps shows in step
//!    with `app\aero-install.json`, which an in-game update replaces
//!    ([`update_installed_version`]).
//! 5. Every problem is a plain message that says what to do next, and a crash offers to open
//!    `game\logs` ([`report_exit`]).
//!
//! # What the game files folder may be
//!
//! Players zip and send their `game` folder to each other, and a network path (`\\server\share`)
//! makes Windows sign in to that server. So the folder is remembered per Windows user and never
//! in `game\` (a `game\launcher-settings.json` is never read), and only a folder on a local drive
//! given by its full path, outside the `Aero Client` folder, that holds the 1.8.9 files is used
//! ([`game_files_dir`]). The path is checked before anything on it is touched.
//!
//! # What it can't do
//!
//! The launcher waited for the game, so it could report any way the game ended. A process can't
//! report its own hard crash (ended by Windows, a stack overflow, an abort inside a driver): what
//! Rust prints on the way out still reaches `latest-launch.errors.log`, but no message is shown.
//! Every failure the game catches (a panic on any thread, a failed start, a lost GPU) is reported
//! with the launcher's message.
//!
//! Nothing here runs for any other start: a command line with arguments (a developer's client,
//! the benchmarks and the profile training, a launcher that passes options) or a client without
//! the marker beside it (every development build) starts exactly as before.

use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use mc_shell::dialog::{Answer, Buttons, Icon};
use mc_update::install::{InstallMarker, APP_DIR, MARKER_FILE};

use crate::app::config::DEFAULT_ASSET_INDEX;
use crate::app::ClientConfig;

/// The title of every message.
pub const TITLE: &str = "Aero Client";

/// Test hook: the game closes itself after this many frames (the launcher's own hook; nothing
/// the player starts sets it). With `AERO_LAUNCHER_NOGUI=1` ([`mc_shell::dialog::NO_GUI_ENV`])
/// messages are printed instead of shown, and questions answered with Cancel.
pub const FRAMES_ENV: &str = "AERO_LAUNCHER_FRAMES";

/// The folder in `%LOCALAPPDATA%` where the game files folder is remembered.
pub const SETTINGS_FOLDER: &str = "Aero Client";
/// The file in [`SETTINGS_FOLDER`].
pub const SETTINGS_FILE: &str = "launcher-settings.json";
/// The key in [`SETTINGS_FILE`].
pub const SETTINGS_KEY: &str = "gameFilesDir";

/// The Settings > Apps entry `Aero-Client-Setup.exe` writes, under `HKEY_CURRENT_USER` (its
/// AppId; keep in step with `packaging\aero\installer\aero-client.iss`). A portable copy has none.
pub const UNINSTALL_KEY: &str = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{33D5C40D-DF3B-44A8-89C8-B29ABEFBF5B1}_is1";

/// The player's folder next to `app\`: settings, worlds, logs.
pub const GAME_DIR: &str = "game";
/// The game's output, in `game\logs\`.
pub const LOG_FILE: &str = "latest-launch.log";
/// The game's errors and crash messages, in `game\logs\`.
pub const ERROR_LOG_FILE: &str = "latest-launch.errors.log";

/// What the launcher set in the game's environment: frame encoding and presenting on a second
/// thread, paced (the build's tuned default).
pub const PROCESS_ENV: [(&str, &str); 1] = [(mc_render::shell::PIPELINED_SUBMIT_ENV, "1")];

/// No game files were found: the question (OK opens the folder picker).
pub const MISSING_FILES: &str = "Aero Client uses your own Minecraft 1.8.9 game files, but \
    couldn't find them.\n\nIf you haven't set them up yet: in the official Minecraft Launcher, add \
    an installation of version 1.8.9, press Play once and close the game when it opens. Then start \
    Aero Client again.\n\nIf your game files are somewhere else, click OK and select their folder \
    (the one that contains the assets folder).";

/// The folder picker's title.
pub const PICKER_TITLE: &str = "Select the folder with your Minecraft 1.8.9 game files (the one \
    that contains the assets folder)";

/// The picked folder doesn't have the files.
pub const WRONG_FOLDER: &str = "That folder doesn't have the Minecraft 1.8.9 game files.\n\n\
    Select the folder that contains the assets folder and the 1.8.9 game, or start version 1.8.9 \
    once in your launcher so it downloads them, then start Aero Client again.";

/// The message after a run that ended with exit code `code`.
#[must_use]
pub fn crashed_message(code: u8) -> String {
    format!(
        "Aero Client closed because of an error (code {code}).\n\nLogs were saved in the \
         game\\logs folder. If you ask for help, send latest-launch.log and \
         latest-launch.errors.log from there.\n\nOpen the logs folder now?"
    )
}

/// The message when the start fails for `reason`.
#[must_use]
pub fn could_not_start_message(reason: &str) -> String {
    format!("Aero Client couldn't start:\n\n{reason}")
}

/// A packaged client's folders.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Packaged {
    /// The `Aero Client` folder, which holds `app\`.
    pub root: PathBuf,
}

impl Packaged {
    /// Whether a process started with `args` (the program first) from `exe` makes the packaged
    /// start: no argument after the program, and `exe` is `<root>\app\<exe>` with the marker
    /// beside it. Anything else is a normal start.
    pub fn detect(args: impl IntoIterator<Item = OsString>, exe: &Path) -> Option<Packaged> {
        let mut args = args.into_iter().skip(1);
        if args.next().is_some() {
            return None;
        }
        let app = exe.parent()?;
        let in_app = app
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.eq_ignore_ascii_case(APP_DIR));
        if !in_app || !app.join(MARKER_FILE).is_file() {
            return None;
        }
        Some(Packaged {
            root: plain_path(app.parent()?),
        })
    }

    /// [`Packaged::detect`] for this process.
    #[must_use]
    pub fn from_process() -> Option<Packaged> {
        let exe = std::env::current_exe().ok()?;
        Packaged::detect(std::env::args_os(), &exe)
    }

    /// `app\`.
    #[must_use]
    pub fn app(&self) -> PathBuf {
        self.root.join(APP_DIR)
    }

    /// `app\aero-install.json`.
    #[must_use]
    pub fn marker(&self) -> PathBuf {
        self.app().join(MARKER_FILE)
    }

    /// `game\`, the game's folder.
    #[must_use]
    pub fn game(&self) -> PathBuf {
        self.root.join(GAME_DIR)
    }

    /// `game\logs\`.
    #[must_use]
    pub fn logs(&self) -> PathBuf {
        self.game().join("logs")
    }

    /// Make `game\logs\` and the two log files in it, empty.
    ///
    /// # Errors
    ///
    /// A folder or file that can't be made.
    pub fn create_logs(&self) -> io::Result<(File, File)> {
        let logs = self.logs();
        std::fs::create_dir_all(&logs)?;
        Ok((
            File::create(logs.join(LOG_FILE))?,
            File::create(logs.join(ERROR_LOG_FILE))?,
        ))
    }

    /// Step 1: this process's standard output and error go to the two log files from here on.
    /// Call before the log writer starts.
    ///
    /// # Errors
    ///
    /// The message to show when `game\logs` can't be made (the game couldn't keep its settings
    /// there either). Log files that can't be opened only lose the logs.
    pub fn open_logs(&self) -> Result<(), String> {
        let logs = self.logs();
        if let Err(error) = std::fs::create_dir_all(&logs) {
            return Err(format!(
                "The folder {} can't be created ({error}).",
                logs.display()
            ));
        }
        if let Ok((out, err)) = self.create_logs() {
            // Without the redirect the game still starts; only its output is lost.
            let _ = mc_shell::stdio::redirect(out, err);
        }
        Ok(())
    }
}

/// `path` without the `\\?\` prefix Windows sometimes reports a program's own path with, so it
/// compares with the paths a player gives.
fn plain_path(path: &Path) -> PathBuf {
    let Some(text) = path.to_str() else {
        return path.to_path_buf();
    };
    match text.strip_prefix(r"\\?\") {
        Some(rest) if is_drive_path(rest, &['\\']) => PathBuf::from(rest),
        _ => path.to_path_buf(),
    }
}

/// What the packaged start reads from this process's environment (a value, so tests never read
/// or change the real one).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StartEnv {
    /// `%LOCALAPPDATA%`, where the game files folder is remembered.
    pub local_app_data: Option<PathBuf>,
    /// `%APPDATA%`, where the official launcher keeps `.minecraft`.
    pub app_data: Option<PathBuf>,
    /// [`FRAMES_ENV`].
    pub frames: Option<u64>,
}

impl StartEnv {
    /// This process's environment. A folder variable that is not a full path counts as unset.
    #[must_use]
    pub fn from_process() -> StartEnv {
        let folder = |name: &str| {
            std::env::var_os(name)
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
        };
        StartEnv {
            local_app_data: folder("LOCALAPPDATA"),
            app_data: folder("APPDATA"),
            frames: std::env::var(FRAMES_ENV)
                .ok()
                .and_then(|text| frames_from(&text)),
        }
    }

    /// `%LOCALAPPDATA%\Aero Client\launcher-settings.json`.
    #[must_use]
    pub fn settings_path(&self) -> Option<PathBuf> {
        Some(
            self.local_app_data
                .as_ref()?
                .join(SETTINGS_FOLDER)
                .join(SETTINGS_FILE),
        )
    }

    /// `%APPDATA%\.minecraft`, the official launcher's folder.
    #[must_use]
    pub fn official_dir(&self) -> Option<PathBuf> {
        Some(self.app_data.as_ref()?.join(".minecraft"))
    }
}

/// [`FRAMES_ENV`]'s value: digits only.
#[must_use]
pub fn frames_from(text: &str) -> Option<u64> {
    if text.is_empty() || !text.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// Minecraft 1.8.9's own files in a game files folder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VanillaFiles {
    /// The 1.8.9 client jar.
    pub jar: PathBuf,
    /// The assets folder that holds the 1.8 index.
    pub assets: PathBuf,
}

/// The 1.8.9 files in `dir`: the assets folder with `indexes\1.8.json`, and the 1.8.9 client jar
/// where the official launcher keeps it (`versions\1.8.9\1.8.9.jar`) or under `libraries\`
/// (`com\mojang\minecraft\1.8.9`).
#[must_use]
pub fn find_vanilla_files(dir: &Path) -> Option<VanillaFiles> {
    let assets = dir.join("assets");
    if !assets.join("indexes").join("1.8.json").is_file() {
        return None;
    }
    let jars = [
        dir.join("versions").join("1.8.9").join("1.8.9.jar"),
        dir.join("libraries")
            .join("com")
            .join("mojang")
            .join("minecraft")
            .join("1.8.9")
            .join("minecraft-1.8.9-client.jar"),
    ];
    let jar = jars.into_iter().find(|jar| jar.is_file())?;
    Some(VanillaFiles { jar, assets })
}

/// Whether `text` starts with a drive letter, a colon and one of `separators`.
fn is_drive_path(text: &str, separators: &[char]) -> bool {
    let mut chars = text.chars();
    match (chars.next(), chars.next(), chars.next()) {
        (Some(letter), Some(':'), Some(separator)) => {
            letter.is_ascii_alphabetic() && separators.contains(&separator)
        }
        _ => false,
    }
}

/// `path` in lower case, ending in exactly one backslash: two folders compare as text.
fn folder_key(path: &Path) -> Option<String> {
    Some(format!("{}\\", path.to_str()?.trim_end_matches('\\')).to_lowercase())
}

/// A usable game files folder, as its full path: a folder on a local drive given by its full
/// path (never a network path, which Windows would sign in to, a device path or a relative one),
/// outside the `Aero Client` folder `root` (which other players' files can come in with), that
/// holds the Minecraft 1.8.9 files. The path is checked before anything on it is touched.
#[must_use]
pub fn game_files_dir(dir: &Path, root: &Path) -> Option<PathBuf> {
    if !is_drive_path(dir.to_str()?, &['\\', '/']) {
        return None;
    }
    // `..` and `.` resolved and `/` turned into `\`, without touching the disk.
    let full = std::path::absolute(dir).ok()?;
    if !is_drive_path(full.to_str()?, &['\\']) {
        return None;
    }
    let inside = folder_key(&std::path::absolute(root).ok()?)?;
    if folder_key(&full)?.starts_with(&inside) {
        return None;
    }
    find_vanilla_files(&full)?;
    Some(full)
}

/// The largest settings file read.
const MAX_SETTINGS_BYTES: u64 = 64 * 1024;

/// The settings file's JSON object, when it has one (a UTF-8 byte order mark, which Windows
/// PowerShell writes, is allowed).
fn read_settings(settings: &Path) -> Option<serde_json::Map<String, serde_json::Value>> {
    use std::io::Read as _;
    let mut bytes = Vec::new();
    File::open(settings)
        .ok()?
        .take(MAX_SETTINGS_BYTES)
        .read_to_end(&mut bytes)
        .ok()?;
    let json = bytes.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&bytes);
    match serde_json::from_slice(json).ok()? {
        serde_json::Value::Object(object) => Some(object),
        _ => None,
    }
}

/// The game files folder remembered in `settings`.
#[must_use]
pub fn read_saved_dir(settings: &Path) -> Option<PathBuf> {
    let object = read_settings(settings)?;
    object.get(SETTINGS_KEY)?.as_str().map(PathBuf::from)
}

/// Remember `dir` in `settings`, keeping anything else in the file. Written only when it
/// changes, whole (a temporary file renamed over it), with the byte order mark the launcher
/// wrote.
///
/// # Errors
///
/// The folder or file can't be written, or `dir` isn't valid Unicode.
pub fn save_game_files_dir(settings: &Path, dir: &Path) -> io::Result<()> {
    let text = dir
        .to_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "not a Unicode path"))?;
    let mut object = read_settings(settings).unwrap_or_default();
    if object.get(SETTINGS_KEY).and_then(serde_json::Value::as_str) == Some(text) {
        return Ok(());
    }
    object.insert(
        SETTINGS_KEY.to_owned(),
        serde_json::Value::String(text.to_owned()),
    );
    if let Some(folder) = settings.parent() {
        std::fs::create_dir_all(folder)?;
    }
    let mut bytes = b"\xEF\xBB\xBF".to_vec();
    bytes.extend(serde_json::to_vec_pretty(&serde_json::Value::Object(
        object,
    ))?);
    let partial = settings.with_extension("json.partial");
    std::fs::write(&partial, bytes)?;
    std::fs::rename(&partial, settings)
}

/// The pop-ups, the folder picker and Explorer: [`ShellUi`] in the game, a script in tests.
pub trait StartUi {
    /// Show `text` and wait for the player's answer.
    fn message(&self, text: &str, icon: Icon, buttons: Buttons) -> Answer;
    /// The folder picker titled `title`; `None` when the player cancels.
    fn pick_folder(&self, title: &str) -> Option<PathBuf>;
    /// Open `folder` in Explorer.
    fn open_folder(&self, folder: &Path);
}

/// The real pop-ups ([`mc_shell::dialog`]) and Explorer.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShellUi;

impl StartUi for ShellUi {
    fn message(&self, text: &str, icon: Icon, buttons: Buttons) -> Answer {
        mc_shell::dialog::message(TITLE, text, icon, buttons)
    }

    fn pick_folder(&self, title: &str) -> Option<PathBuf> {
        mc_shell::dialog::pick_folder(title)
    }

    fn open_folder(&self, folder: &Path) {
        let Some(explorer) = explorer_path() else {
            return;
        };
        if let Err(error) = Command::new(explorer).arg(folder).spawn() {
            tracing::warn!(%error, "Explorer could not be opened");
        }
    }
}

/// Explorer by its full path (a bare name would also be looked for in the current folder).
#[must_use]
pub fn explorer_path() -> Option<PathBuf> {
    let windows = std::env::var_os("SystemRoot").or_else(|| std::env::var_os("windir"))?;
    let windows = PathBuf::from(windows);
    windows.is_absolute().then(|| windows.join("explorer.exe"))
}

/// Step 2: where the game files are. The folder chosen last time while it still has them, then
/// the official launcher's, else the player is asked (OK opens the folder picker). `None` when
/// the player cancels or picks a folder without the files (they are told what to do).
pub fn find_game_files_dir(root: &Path, env: &StartEnv, ui: &dyn StartUi) -> Option<PathBuf> {
    let saved = env.settings_path().and_then(|path| read_saved_dir(&path));
    if let Some(dir) = saved.and_then(|saved| game_files_dir(&saved, root)) {
        tracing::info!("Minecraft 1.8.9 files: the folder chosen before");
        return Some(dir);
    }
    if let Some(dir) = env
        .official_dir()
        .and_then(|official| game_files_dir(&official, root))
    {
        tracing::info!("Minecraft 1.8.9 files: the official launcher's folder");
        return Some(dir);
    }
    tracing::info!("Minecraft 1.8.9 files not found; asking");
    if ui.message(MISSING_FILES, Icon::Question, Buttons::OkCancel) != Answer::Ok {
        return None;
    }
    let picked = ui.pick_folder(PICKER_TITLE)?;
    match game_files_dir(&picked, root) {
        Some(dir) => {
            tracing::info!("Minecraft 1.8.9 files: the folder the player chose");
            Some(dir)
        }
        None => {
            ui.message(WRONG_FOLDER, Icon::Warning, Buttons::Ok);
            None
        }
    }
}

/// Step 3: the game's command line. Its folder and the game files, nothing else (no account),
/// and the frame count of the test hook.
///
/// # Errors
///
/// A path that isn't valid Unicode (the game's options are text).
pub fn game_args(
    game: &Path,
    files: &VanillaFiles,
    frames: Option<u64>,
) -> Result<Vec<String>, String> {
    let text = |path: &Path| {
        path.to_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("The folder name {} can't be used.", path.display()))
    };
    let mut args = vec![
        "--gameDir".to_owned(),
        text(game)?,
        "--assetsDir".to_owned(),
        text(&files.assets)?,
        "--assetIndex".to_owned(),
        DEFAULT_ASSET_INDEX.to_owned(),
        "--versionJar".to_owned(),
        text(&files.jar)?,
    ];
    if let Some(frames) = frames {
        args.push("--frames".to_owned());
        args.push(frames.to_string());
    }
    Ok(args)
}

/// The Settings > Apps entry of an installed copy (see [`UNINSTALL_KEY`]).
pub trait UninstallEntry {
    /// The string value `name`, if the entry and the value exist.
    fn read(&self, name: &str) -> Option<String>;
    /// Set the string value `name` of the existing entry; whether it was written.
    fn write(&self, name: &str, value: &str) -> bool;
}

/// The entry in this Windows user's registry (`HKEY_CURRENT_USER\<key>`), never created here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CurrentUserEntry {
    key: String,
}

impl CurrentUserEntry {
    /// The entry Setup writes ([`UNINSTALL_KEY`]).
    #[must_use]
    pub fn setup() -> CurrentUserEntry {
        CurrentUserEntry::at(UNINSTALL_KEY)
    }

    /// The entry at `HKEY_CURRENT_USER\<key>`.
    #[must_use]
    pub fn at(key: &str) -> CurrentUserEntry {
        CurrentUserEntry {
            key: key.to_owned(),
        }
    }
}

impl UninstallEntry for CurrentUserEntry {
    fn read(&self, name: &str) -> Option<String> {
        mc_shell::registry::read_current_user_string(&self.key, name)
            .ok()
            .flatten()
    }

    fn write(&self, name: &str, value: &str) -> bool {
        mc_shell::registry::set_current_user_string(&self.key, name, value).is_ok()
    }
}

/// Whether `text` is a plain `x.y.z` version (one to nine digits each).
#[must_use]
pub fn plain_version(text: &str) -> bool {
    let parts: Vec<&str> = text.split('.').collect();
    parts.len() == 3
        && parts.iter().all(|part| {
            (1..=9).contains(&part.len()) && part.bytes().all(|byte| byte.is_ascii_digit())
        })
}

/// Step 4: Settings > Apps shows the version Setup installed, and an in-game update replaces
/// `app\` without Setup, so the entry would keep naming the old one. Its `DisplayVersion`
/// becomes the version in the marker `marker`, but only for the entry whose `InstallLocation` is
/// this very folder `root` (one installed copy per Windows user; a portable copy elsewhere has
/// nothing to do with it), and only to a plain `x.y.z`. A convenience: anything unexpected
/// leaves the entry as it is, silently. Returns whether it changed the entry.
pub fn update_installed_version(entry: &dyn UninstallEntry, root: &Path, marker: &Path) -> bool {
    let Some(location) = entry
        .read("InstallLocation")
        .filter(|location| !location.is_empty())
    else {
        return false;
    };
    let key = |path: &Path| {
        std::path::absolute(path)
            .ok()
            .and_then(|full| folder_key(&full))
    };
    match (key(Path::new(&location)), key(root)) {
        (Some(installed), Some(here)) if installed == here => {}
        _ => return false,
    }
    let Ok(marker) = InstallMarker::read(marker) else {
        return false;
    };
    if !plain_version(&marker.version) {
        return false;
    }
    if entry.read("DisplayVersion").as_deref() == Some(marker.version.as_str()) {
        return false;
    }
    entry.write("DisplayVersion", &marker.version)
}

/// Steps 2 to 4: the game's configuration, or `None` when the start stops here (the player was
/// told why). Changes nothing in this process; see [`apply_process_settings`].
pub fn prepare(
    packaged: &Packaged,
    env: &StartEnv,
    ui: &dyn StartUi,
    entry: &dyn UninstallEntry,
) -> Option<ClientConfig> {
    let dir = find_game_files_dir(&packaged.root, env, ui)?;
    if let Some(settings) = env.settings_path() {
        // Only a convenience: the folder is looked for again next time.
        if let Err(error) = save_game_files_dir(&settings, &dir) {
            tracing::warn!(%error, "the game files folder could not be remembered");
        }
    }
    let config = find_vanilla_files(&dir)
        .ok_or_else(|| WRONG_FOLDER.to_owned())
        .and_then(|files| game_args(&packaged.game(), &files, env.frames))
        .and_then(|args| ClientConfig::parse_args(&args));
    match config {
        Ok(config) => {
            update_installed_version(entry, &packaged.root, &packaged.marker());
            Some(config)
        }
        Err(reason) => {
            report_start_failure(ui, &reason);
            None
        }
    }
}

/// What the launcher set for the game's process: [`PROCESS_ENV`], and `game\` as the working
/// folder. Once, before the game starts and before any thread reads the environment.
pub fn apply_process_settings(packaged: &Packaged) {
    for (name, value) in PROCESS_ENV {
        std::env::set_var(name, value);
    }
    if let Err(error) = std::env::set_current_dir(packaged.game()) {
        tracing::warn!(%error, "the game folder could not be made the working folder");
    }
}

/// A start that failed for `reason`: the launcher's message.
pub fn report_start_failure(ui: &dyn StartUi, reason: &str) {
    ui.message(&could_not_start_message(reason), Icon::Error, Buttons::Ok);
}

/// Step 5, after the game: a run that ended with a non-zero exit code `code` (a crash report
/// was written, or the game failed to start) gets the launcher's message, which offers to open
/// `game\logs`.
pub fn report_exit(packaged: &Packaged, code: u8, ui: &dyn StartUi) {
    if code == 0 {
        return;
    }
    if ui.message(&crashed_message(code), Icon::Error, Buttons::YesNo) == Answer::Yes {
        ui.open_folder(&packaged.logs());
    }
}

#[cfg(all(test, windows))]
mod tests {
    //! The rules of the PowerShell launcher's trust test (`test-launcher-trust.ps1`), ported:
    //! the folder rules, where the choice is remembered, what the game gets, the questions, and
    //! the upkeep of an installed copy's Settings > Apps entry. Everything happens in a
    //! temporary folder (and a temporary registry key); no real game files, settings or
    //! accounts are read.

    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::{HashMap, VecDeque};

    /// A stand-in game files folder: the two files the start looks for, with stand-in content.
    fn game_files(dir: &Path, libraries: bool, jar: bool, index: bool) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        if index {
            let indexes = dir.join("assets").join("indexes");
            std::fs::create_dir_all(&indexes).unwrap();
            std::fs::write(indexes.join("1.8.json"), r#"{"objects":{}}"#).unwrap();
        }
        if jar {
            let jar = if libraries {
                dir.join(r"libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar")
            } else {
                dir.join(r"versions\1.8.9\1.8.9.jar")
            };
            std::fs::create_dir_all(jar.parent().unwrap()).unwrap();
            std::fs::write(jar, "a stand-in, not a jar").unwrap();
        }
        dir.to_path_buf()
    }

    /// A stand-in `Aero Client` folder: `app\mc-client.exe` and the marker.
    fn package(dir: &Path, version: &str) -> Packaged {
        let root = dir.join("Aero Client");
        std::fs::create_dir_all(root.join("app")).unwrap();
        std::fs::write(root.join(r"app\mc-client.exe"), "exe").unwrap();
        std::fs::write(
            root.join(r"app\aero-install.json"),
            InstallMarker::new(version).to_json(),
        )
        .unwrap();
        Packaged { root }
    }

    /// The pop-ups, answered from a script: what was shown, the answers given, the folder the
    /// picker returns, what Explorer was asked to open.
    #[derive(Default)]
    struct ScriptedUi {
        answers: RefCell<VecDeque<Answer>>,
        picked: RefCell<Option<PathBuf>>,
        shown: RefCell<Vec<(String, Icon, Buttons)>>,
        pickers: Cell<u32>,
        opened: RefCell<Vec<PathBuf>>,
    }

    impl ScriptedUi {
        fn answering(answers: &[Answer], picked: Option<PathBuf>) -> ScriptedUi {
            ScriptedUi {
                answers: RefCell::new(answers.iter().copied().collect()),
                picked: RefCell::new(picked),
                ..ScriptedUi::default()
            }
        }

        fn texts(&self) -> Vec<String> {
            self.shown
                .borrow()
                .iter()
                .map(|(text, ..)| text.clone())
                .collect()
        }
    }

    impl StartUi for ScriptedUi {
        fn message(&self, text: &str, icon: Icon, buttons: Buttons) -> Answer {
            self.shown
                .borrow_mut()
                .push((text.to_owned(), icon, buttons));
            self.answers
                .borrow_mut()
                .pop_front()
                .unwrap_or(Answer::Cancel)
        }

        fn pick_folder(&self, title: &str) -> Option<PathBuf> {
            assert_eq!(title, PICKER_TITLE);
            self.pickers.set(self.pickers.get() + 1);
            self.picked.borrow_mut().take()
        }

        fn open_folder(&self, folder: &Path) {
            self.opened.borrow_mut().push(folder.to_path_buf());
        }
    }

    /// A Settings > Apps entry in memory (`None`: no entry at all, a portable copy).
    struct FakeEntry {
        values: RefCell<Option<HashMap<String, String>>>,
        writes: Cell<u32>,
    }

    impl FakeEntry {
        fn none() -> FakeEntry {
            FakeEntry {
                values: RefCell::new(None),
                writes: Cell::new(0),
            }
        }

        fn with(values: &[(&str, &str)]) -> FakeEntry {
            let map = values
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect();
            FakeEntry {
                values: RefCell::new(Some(map)),
                writes: Cell::new(0),
            }
        }

        fn value(&self, name: &str) -> Option<String> {
            self.values.borrow().as_ref()?.get(name).cloned()
        }
    }

    impl UninstallEntry for FakeEntry {
        fn read(&self, name: &str) -> Option<String> {
            self.value(name)
        }

        fn write(&self, name: &str, value: &str) -> bool {
            let mut values = self.values.borrow_mut();
            let Some(values) = values.as_mut() else {
                return false;
            };
            values.insert(name.to_owned(), value.to_owned());
            self.writes.set(self.writes.get() + 1);
            true
        }
    }

    fn env_in(dir: &Path) -> StartEnv {
        StartEnv {
            local_app_data: Some(dir.join(r"user\Local")),
            app_data: Some(dir.join(r"user\Roaming")),
            frames: None,
        }
    }

    // ---- the decision -----------------------------------------------------------------------

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn only_a_start_without_arguments_beside_the_marker_is_packaged() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let exe = packaged.app().join("mc-client.exe");
        assert_eq!(
            Packaged::detect(args(&["mc-client.exe"]), &exe),
            Some(packaged.clone()),
            "no arguments, the marker beside it"
        );
        assert_eq!(
            Packaged::detect(Vec::new(), &exe),
            Some(packaged.clone()),
            "not even the program's name"
        );
        // Any argument is a normal start: the user's own client, the benchmarks, the profile
        // training, a launcher that passes options, the update helper's command lines.
        for arguments in [
            &["mc-client.exe", "--frames", "5"][..],
            &["mc-client.exe", "--gameDir", "x"],
            &["mc-client.exe", ""],
            &["mc-client.exe", "--aero-update-self-test"],
        ] {
            assert_eq!(
                Packaged::detect(args(arguments), &exe),
                None,
                "{arguments:?}"
            );
        }
        // No marker beside it: every development build.
        std::fs::remove_file(packaged.marker()).unwrap();
        assert_eq!(Packaged::detect(args(&["mc-client.exe"]), &exe), None);
        // The marker, but not in an app folder (target\debug, say).
        let debug = dir.path().join(r"target\debug");
        std::fs::create_dir_all(&debug).unwrap();
        std::fs::write(
            debug.join(MARKER_FILE),
            InstallMarker::new("1.0.0").to_json(),
        )
        .unwrap();
        assert_eq!(
            Packaged::detect(args(&["mc-client.exe"]), &debug.join("mc-client.exe")),
            None
        );
    }

    #[test]
    fn the_package_folders_hang_off_the_folder_above_app() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let root = dir.path().join("Aero Client");
        assert_eq!(packaged.game(), root.join("game"));
        assert_eq!(packaged.logs(), root.join(r"game\logs"));
        assert_eq!(packaged.marker(), root.join(r"app\aero-install.json"));
        // A program path Windows reports with `\\?\` compares like the plain one.
        let verbatim = PathBuf::from(format!(
            r"\\?\{}",
            root.join(r"app\mc-client.exe").display()
        ));
        assert_eq!(
            Packaged::detect(args(&["x"]), &verbatim).map(|found| found.root),
            Some(root)
        );
    }

    #[test]
    fn the_logs_are_made_in_game_logs_and_emptied_at_each_start() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let (mut out, _err) = packaged.create_logs().unwrap();
        std::io::Write::write_all(&mut out, b"the last start").unwrap();
        drop(out);
        let _ = packaged.create_logs().unwrap();
        for name in [LOG_FILE, ERROR_LOG_FILE] {
            let path = packaged.logs().join(name);
            assert_eq!(std::fs::read(&path).unwrap(), b"", "{name} emptied");
        }
    }

    // ---- the folder rules ------------------------------------------------------------------

    #[test]
    fn only_a_local_folder_outside_the_package_with_the_files_is_used() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let root = &packaged.root;
        // Game files a shared "game" folder could bring along.
        let planted = game_files(&root.join(r"game\evil"), false, true, true);
        let official = game_files(&dir.path().join(r"Roaming\.minecraft"), false, true, true);
        let libraries = game_files(&dir.path().join(r"Games\Minecraft"), true, true, true);
        let no_jar = game_files(&dir.path().join("NoJar"), false, false, true);
        let no_index = game_files(&dir.path().join("NoIndex"), false, true, false);

        assert_eq!(
            game_files_dir(&official, root),
            Some(official.clone()),
            "a local folder with the 1.8.9 jar under versions\\ is accepted"
        );
        assert!(
            game_files_dir(&libraries, root).is_some(),
            "and one with it under libraries\\"
        );
        assert!(
            game_files_dir(&no_jar, root).is_none(),
            "a folder without the 1.8.9 jar is not"
        );
        assert!(
            game_files_dir(&no_index, root).is_none(),
            "nor one without the 1.8 asset index"
        );
        assert!(
            game_files_dir(&dir.path().join("missing"), root).is_none(),
            "nor a missing folder"
        );
        assert!(
            game_files_dir(&planted, root).is_none(),
            "nor a folder inside the Aero Client folder"
        );
        assert!(
            game_files_dir(&root.join(r"game\..\game\evil"), root).is_none(),
            "nor the same through .."
        );
        assert!(
            game_files_dir(&PathBuf::from(format!("{}\\", root.display())), root).is_none(),
            "nor the Aero Client folder itself"
        );
        assert!(
            game_files_dir(Path::new(r"game\evil"), root).is_none(),
            "nor a relative path"
        );
        assert!(
            game_files_dir(Path::new(r"\\localhost\aero-no-such-share\games"), root).is_none(),
            "nor a network (UNC) path"
        );
        assert!(
            game_files_dir(Path::new(r"\\?\C:\Windows"), root).is_none(),
            "nor a device path"
        );
        assert!(
            game_files_dir(Path::new(""), root).is_none(),
            "nor an empty path"
        );
        assert!(
            game_files_dir(Path::new("C:"), root).is_none(),
            "nor a drive-relative one"
        );
        // A forward slash after the drive is a full path too, and the result is normalised.
        let slashed = PathBuf::from(official.to_str().unwrap().replace('\\', "/"));
        assert_eq!(game_files_dir(&slashed, root), Some(official.clone()));
        // A folder next to the Aero Client folder whose name only begins the same is outside it.
        let sibling = game_files(&dir.path().join("Aero Client 2"), false, true, true);
        assert_eq!(game_files_dir(&sibling, root), Some(sibling.clone()));
    }

    #[test]
    fn the_jar_and_the_assets_folder_are_found_in_both_layouts() {
        let dir = tempfile::tempdir().unwrap();
        let official = game_files(&dir.path().join(".minecraft"), false, true, true);
        assert_eq!(
            find_vanilla_files(&official),
            Some(VanillaFiles {
                jar: official.join(r"versions\1.8.9\1.8.9.jar"),
                assets: official.join("assets"),
            })
        );
        let libraries = game_files(&dir.path().join("other-launcher"), true, true, true);
        assert_eq!(
            find_vanilla_files(&libraries).unwrap().jar,
            libraries.join(r"libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar")
        );
        // Both: the official launcher's place first.
        game_files(&libraries, false, true, true);
        assert_eq!(
            find_vanilla_files(&libraries).unwrap().jar,
            libraries.join(r"versions\1.8.9\1.8.9.jar")
        );
    }

    // ---- remembering -----------------------------------------------------------------------

    #[test]
    fn the_choice_is_remembered_per_user_never_in_the_game_folder() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let env = env_in(dir.path());
        let settings = env.settings_path().unwrap();
        assert_eq!(
            settings,
            dir.path()
                .join(r"user\Local\Aero Client\launcher-settings.json")
        );
        let official = game_files(
            &dir.path().join(r"user\Roaming\.minecraft"),
            false,
            true,
            true,
        );

        // Anything else in the file is kept (written by PowerShell, with a byte order mark).
        std::fs::create_dir_all(settings.parent().unwrap()).unwrap();
        std::fs::write(&settings, b"\xEF\xBB\xBF{\r\n    \"keep\":  \"me\"\r\n}").unwrap();
        save_game_files_dir(&settings, &official).unwrap();
        assert!(settings.is_file(), "the choice is saved under LocalAppData");
        assert!(
            !packaged.game().join(SETTINGS_FILE).exists(),
            "and not in the game folder"
        );
        assert_eq!(
            read_saved_dir(&settings),
            Some(official.clone()),
            "and read back"
        );
        let object = read_settings(&settings).unwrap();
        assert_eq!(object["keep"], "me", "the rest of the file is kept");
        assert!(std::fs::read(&settings)
            .unwrap()
            .starts_with(b"\xEF\xBB\xBF"));

        // A settings file that came in with a shared game folder is never read.
        std::fs::create_dir_all(packaged.game()).unwrap();
        let planted = game_files(&packaged.game().join("evil"), false, true, true);
        std::fs::write(
            packaged.game().join(SETTINGS_FILE),
            format!("{{\"gameFilesDir\": {:?}}}", planted.to_str().unwrap()),
        )
        .unwrap();
        let ui = ScriptedUi::default();
        assert_eq!(
            find_game_files_dir(&packaged.root, &env, &ui),
            Some(official.clone()),
            "a game\\launcher-settings.json is ignored"
        );
        // A damaged file reads as nothing remembered, and is replaced by the next choice.
        std::fs::write(&settings, "{not json").unwrap();
        assert_eq!(read_saved_dir(&settings), None);
        save_game_files_dir(&settings, &official).unwrap();
        assert_eq!(read_saved_dir(&settings), Some(official));
        // Without %LOCALAPPDATA% nothing is read or written.
        let nowhere = StartEnv {
            local_app_data: None,
            ..env
        };
        assert_eq!(nowhere.settings_path(), None);
    }

    // ---- the whole start ---------------------------------------------------------------------

    #[test]
    fn the_game_gets_its_folder_and_the_game_files_and_nothing_else() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let env = StartEnv {
            frames: Some(5),
            ..env_in(dir.path())
        };
        let official = game_files(
            &dir.path().join(r"user\Roaming\.minecraft"),
            false,
            true,
            true,
        );
        let ui = ScriptedUi::default();
        let config = prepare(&packaged, &env, &ui, &FakeEntry::none()).expect("the game starts");
        assert!(
            ui.shown.borrow().is_empty(),
            "no question when the files are there"
        );

        let files = find_vanilla_files(&official).unwrap();
        let game = packaged.game();
        let expected = [
            "--gameDir",
            game.to_str().unwrap(),
            "--assetsDir",
            files.assets.to_str().unwrap(),
            "--assetIndex",
            "1.8",
            "--versionJar",
            files.jar.to_str().unwrap(),
            "--frames",
            "5",
        ];
        assert_eq!(
            game_args(&packaged.game(), &files, Some(5)).unwrap(),
            expected
        );
        assert_eq!(config.game_dir, packaged.game());
        assert_eq!(config.assets_dir, files.assets);
        assert_eq!(config.version_jar.as_deref(), Some(files.jar.as_path()));
        assert_eq!(config.asset_index, "1.8");
        assert_eq!(config.max_frames, Some(5));
        assert_eq!(
            config.resource_pack_dir,
            packaged.game().join("resourcepacks")
        );
        // No account and no sign-in: the player signs in inside the game.
        assert_eq!(config.login, crate::app::config::LoginMode::Microsoft);
        for arg in game_args(&packaged.game(), &files, None).unwrap() {
            for forbidden in [
                "--username",
                "--uuid",
                "--accessToken",
                "--userType",
                "--offline",
            ] {
                assert_ne!(arg, forbidden);
            }
        }
        // The game files folder is remembered under LocalAppData, and nothing in game\.
        assert_eq!(
            read_saved_dir(&env.settings_path().unwrap()),
            Some(official)
        );
        assert!(!packaged.game().join(SETTINGS_FILE).exists());
        // What the process gets besides: the pipelined submit, as the launcher set it.
        assert_eq!(PROCESS_ENV, [("MC_PIPELINED_SUBMIT", "1")]);
    }

    #[test]
    fn the_folder_chosen_last_time_wins_while_it_still_has_the_files() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let env = env_in(dir.path());
        game_files(
            &dir.path().join(r"user\Roaming\.minecraft"),
            false,
            true,
            true,
        );
        let other = game_files(&dir.path().join(r"user\Games\Minecraft"), true, true, true);
        save_game_files_dir(&env.settings_path().unwrap(), &other).unwrap();
        let config = prepare(&packaged, &env, &ScriptedUi::default(), &FakeEntry::none()).unwrap();
        assert_eq!(
            config.version_jar.unwrap(),
            other.join(r"libraries\com\mojang\minecraft\1.8.9\minecraft-1.8.9-client.jar")
        );
        // Once its files are gone, the official launcher's folder is used again.
        std::fs::remove_dir_all(&other).unwrap();
        let config = prepare(&packaged, &env, &ScriptedUi::default(), &FakeEntry::none()).unwrap();
        assert!(config
            .version_jar
            .unwrap()
            .ends_with(r"versions\1.8.9\1.8.9.jar"));
    }

    #[test]
    fn without_game_files_the_player_is_told_and_the_game_is_not_started() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        let env = env_in(dir.path());
        let entry = FakeEntry::with(&[("InstallLocation", packaged.root.to_str().unwrap())]);

        // Cancel: nothing more is asked and nothing starts.
        let ui = ScriptedUi::answering(&[Answer::Cancel], None);
        assert!(prepare(&packaged, &env, &ui, &entry).is_none());
        assert_eq!(ui.texts(), [MISSING_FILES]);
        assert!(MISSING_FILES.contains("couldn't find them"));
        assert_eq!(ui.shown.borrow()[0].1, Icon::Question);
        assert_eq!(ui.shown.borrow()[0].2, Buttons::OkCancel);
        assert_eq!(ui.pickers.get(), 0);
        assert_eq!(
            entry.writes.get(),
            0,
            "the entry is only kept up to date for a game start"
        );

        // OK, then the picker is closed.
        let ui = ScriptedUi::answering(&[Answer::Ok], None);
        assert!(prepare(&packaged, &env, &ui, &entry).is_none());
        assert_eq!(ui.pickers.get(), 1);
        assert_eq!(ui.texts(), [MISSING_FILES]);

        // OK, then a folder without the files: told so, nothing starts or is remembered.
        let empty = dir.path().join("Empty");
        std::fs::create_dir_all(&empty).unwrap();
        let ui = ScriptedUi::answering(&[Answer::Ok], Some(empty));
        assert!(prepare(&packaged, &env, &ui, &entry).is_none());
        assert_eq!(ui.texts(), [MISSING_FILES, WRONG_FOLDER]);
        assert_eq!(ui.shown.borrow()[1].1, Icon::Warning);
        assert!(!env.settings_path().unwrap().exists());

        // OK, then a folder inside the Aero Client folder: refused the same way.
        let planted = game_files(&packaged.game().join("evil"), false, true, true);
        let ui = ScriptedUi::answering(&[Answer::Ok], Some(planted));
        assert!(prepare(&packaged, &env, &ui, &entry).is_none());
        assert_eq!(ui.texts(), [MISSING_FILES, WRONG_FOLDER]);

        // OK, then the right folder: the game starts with it, and it is remembered.
        let chosen = game_files(
            &dir.path().join(r"D Games\Some Launcher\minecraft"),
            true,
            true,
            true,
        );
        let ui = ScriptedUi::answering(&[Answer::Ok], Some(chosen.clone()));
        let config = prepare(&packaged, &env, &ui, &entry).expect("started");
        assert!(config.version_jar.unwrap().starts_with(&chosen));
        assert_eq!(read_saved_dir(&env.settings_path().unwrap()), Some(chosen));
    }

    /// Files with Windows' "downloaded from the internet" mark: the package's own, and one a
    /// player put in the game folder. The start changes no file's mark.
    #[test]
    fn the_start_leaves_download_marks_alone() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");
        std::fs::create_dir_all(packaged.game().join("resourcepacks")).unwrap();
        let marked = [
            packaged.root.join("README.txt"),
            packaged.app().join("NOTICE.txt"),
            packaged.game().join(r"resourcepacks\pack.zip"),
        ];
        for file in &marked {
            std::fs::write(file, "stand-in").unwrap();
            let stream = format!("{}:Zone.Identifier", file.display());
            if std::fs::write(&stream, "[ZoneTransfer]\r\nZoneId=3").is_err() {
                eprintln!("skip: this file system has no alternate data streams");
                return;
            }
        }
        let env = env_in(dir.path());
        game_files(
            &dir.path().join(r"user\Roaming\.minecraft"),
            false,
            true,
            true,
        );
        let _ = packaged.create_logs().unwrap();
        assert!(prepare(&packaged, &env, &ScriptedUi::default(), &FakeEntry::none()).is_some());
        for file in &marked {
            let stream = format!("{}:Zone.Identifier", file.display());
            assert!(
                std::fs::read_to_string(&stream).is_ok_and(|text| text.contains("ZoneId=3")),
                "{} keeps its mark",
                file.display()
            );
        }
    }

    // ---- an installed copy -------------------------------------------------------------------

    #[test]
    fn an_installed_copys_settings_entry_follows_the_marker() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "0.2.0");
        let root = &packaged.root;
        let marker = packaged.marker();

        let portable = FakeEntry::none();
        assert!(!update_installed_version(&portable, root, &marker));
        assert_eq!(portable.writes.get(), 0, "a portable copy writes nothing");

        let other_location = format!("{}\\", dir.path().join(r"Other\Aero Client").display());
        let other = FakeEntry::with(&[
            ("DisplayVersion", "0.1.0"),
            ("InstallLocation", &other_location),
        ]);
        assert!(!update_installed_version(&other, root, &marker));
        assert_eq!(
            other.value("DisplayVersion").as_deref(),
            Some("0.1.0"),
            "another folder's"
        );

        // This folder, in other letter case and with a trailing backslash.
        let here = format!("{}\\", root.display()).to_uppercase();
        let entry = FakeEntry::with(&[("DisplayVersion", "0.1.0"), ("InstallLocation", &here)]);
        assert!(update_installed_version(&entry, root, &marker));
        assert_eq!(entry.value("DisplayVersion").as_deref(), Some("0.2.0"));
        assert!(
            !update_installed_version(&entry, root, &marker),
            "nothing to do twice"
        );
        assert_eq!(entry.writes.get(), 1);

        // Never anything but a plain x.y.z version.
        for version in ["v0.3.0", "0.3.0\r\nInjected: yes"] {
            let text = InstallMarker::new("0.3.0")
                .to_json()
                .replace("0.3.0", version);
            std::fs::write(&marker, text).unwrap();
            assert!(
                !update_installed_version(&entry, root, &marker),
                "{version:?}"
            );
        }
        std::fs::write(&marker, r#"{"version": "0.3.0\r\nInjected: yes"}"#).unwrap();
        assert!(!update_installed_version(&entry, root, &marker));
        // A missing marker changes nothing, silently.
        std::fs::remove_file(&marker).unwrap();
        assert!(!update_installed_version(&entry, root, &marker));
        assert_eq!(entry.value("DisplayVersion").as_deref(), Some("0.2.0"));

        assert!(plain_version("1.0.0") && plain_version("123456789.0.10"));
        for bad in [
            "1.0",
            "1.0.0.0",
            "1.0.0-rc.1",
            "1..0",
            "1234567890.0.0",
            " 1.0.0",
            "",
        ] {
            assert!(!plain_version(bad), "{bad:?}");
        }
    }

    /// The same against the real registry: a temporary key under `HKEY_CURRENT_USER\Software`
    /// stands in for the entry Setup writes (made and removed with Windows' own `reg.exe`).
    #[test]
    fn the_real_registry_entry_is_kept_in_step_and_never_created() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "0.2.0");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |since| since.as_nanos());
        let key = format!(
            r"Software\AeroPackagedStartTest-{}-{nanos}",
            std::process::id()
        );
        let reg = |args: &[&str]| {
            let system = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
            Command::new(Path::new(&system).join(r"System32\reg.exe"))
                .arg(args[0])
                .arg(format!(r"HKCU\{key}"))
                .args(&args[1..])
                .output()
                .is_ok_and(|output| output.status.success())
        };
        let entry = CurrentUserEntry::at(&key);
        assert!(!update_installed_version(
            &entry,
            &packaged.root,
            &packaged.marker()
        ));
        assert!(
            !reg(&["query"]),
            "a portable copy writes nothing to the registry"
        );

        let location = format!("{}\\", packaged.root.display());
        assert!(reg(&[
            "add",
            "/v",
            "InstallLocation",
            "/t",
            "REG_SZ",
            "/d",
            &location,
            "/f"
        ]));
        assert!(reg(&[
            "add",
            "/v",
            "DisplayVersion",
            "/t",
            "REG_SZ",
            "/d",
            "0.1.0",
            "/f"
        ]));
        let updated = update_installed_version(&entry, &packaged.root, &packaged.marker());
        let version = entry.read("DisplayVersion");
        let _ = reg(&["delete", "/f"]);
        assert!(updated);
        assert_eq!(
            version.as_deref(),
            Some("0.2.0"),
            "this folder's entry gets the marker's"
        );
        assert_eq!(
            CurrentUserEntry::setup(),
            CurrentUserEntry::at(UNINSTALL_KEY)
        );
    }

    // ---- after the game ----------------------------------------------------------------------

    #[test]
    fn a_failed_run_is_reported_and_offers_the_logs_folder() {
        let dir = tempfile::tempdir().unwrap();
        let packaged = package(dir.path(), "1.0.0");

        let ui = ScriptedUi::default();
        report_exit(&packaged, 0, &ui);
        assert!(ui.shown.borrow().is_empty(), "a normal exit says nothing");

        let ui = ScriptedUi::answering(&[Answer::Yes], None);
        report_exit(&packaged, 255, &ui);
        let shown = ui.shown.borrow();
        assert_eq!(shown.len(), 1);
        assert!(shown[0]
            .0
            .starts_with("Aero Client closed because of an error (code 255)."));
        assert!(shown[0]
            .0
            .contains("latest-launch.log and latest-launch.errors.log"));
        assert!(shown[0].0.ends_with("Open the logs folder now?"));
        assert_eq!((shown[0].1, shown[0].2), (Icon::Error, Buttons::YesNo));
        assert_eq!(*ui.opened.borrow(), [packaged.logs()]);

        let ui = ScriptedUi::answering(&[Answer::No], None);
        report_exit(&packaged, 254, &ui);
        assert!(ui.opened.borrow().is_empty(), "No opens nothing");

        let ui = ScriptedUi::default();
        report_start_failure(&ui, "The folder X can't be created.");
        assert_eq!(
            ui.texts(),
            ["Aero Client couldn't start:\n\nThe folder X can't be created."]
        );
    }

    #[test]
    fn programs_are_started_by_their_full_path_and_hooks_parse_strictly() {
        let explorer = explorer_path().expect("SystemRoot is set on Windows");
        assert!(explorer.is_absolute());
        assert!(explorer.ends_with("explorer.exe"));
        assert_eq!(frames_from("5"), Some(5));
        for bad in ["", "5x", "-1", " 5", "+5"] {
            assert_eq!(frames_from(bad), None, "{bad:?}");
        }
    }
}
