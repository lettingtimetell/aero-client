//! Applying a staged update on restart, rolling back a version that does not start, and the
//! client's own start and health hooks.
//!
//! # The restart, step by step
//!
//! 1. The player clicks "Restart to update" on the title screen (no world loaded). The client
//!    copies its own exe to `update\helper\aero-update-helper.exe`, starts that copy with
//!    [`APPLY_FLAG`] (detached, no window, its working folder `update\`), and quits normally.
//! 2. The helper ([`apply`]) waits until nothing runs from `app\mc-client.exe` (up to a minute),
//!    checks the staged exe's SHA-256 against the one recorded at staging, and runs the new exe
//!    with [`SELF_TEST_FLAG`]: it must print [`SELF_TEST_OK`] and its version and exit 0 (this
//!    catches a binary that cannot even load on this PC before anything is swapped).
//! 3. [`crate::swap::swap_in`] renames the old `app\` (and root files) to `update\previous\` and
//!    the staged ones into place; the state records a *trial* of the new version.
//! 4. The helper starts `app\mc-client.exe` with no arguments, its packaged start (what the
//!    player's `Aero Client.exe` does), and watches the trial: the new client counts its start
//!    ([`on_client_start`]) and ends the trial when its title screen shows
//!    ([`title_screen_reached`]). If the new client exits before that, the helper rolls back
//!    ([`crate::swap::roll_back`]), marks the version as failing on this PC, and starts the old
//!    version again.
//! 5. If the helper stops watching first (the start stopped before the game, say because the
//!    player closed the question about the game files), the trial stays open: the client keeps
//!    counting its starts, and a version
//!    that has not reached its title screen in [`MAX_TRIAL_STARTS`] starts hands itself back to
//!    the helper's [`ROLLBACK_FLAG`] and exits (on at most [`MAX_ROLLBACK_TRIES`] starts: a
//!    rollback that keeps failing never stops the client from starting for good).
//!
//! The helper is the **old** version's code (a copy of the client that was running), so a broken
//! new version cannot break its own rollback.

use std::ffi::OsString;
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::install::{portable_install, InstallPaths, APP_DIR, EXE_NAME};
use crate::state::{self, Failure, Trial};
use crate::swap;

/// The helper's command-line flag: apply the staged update (`--root <dir> --version <v>`).
pub const APPLY_FLAG: &str = "--aero-apply-update";
/// The helper's command-line flag: roll the version on trial back (`--root <dir>`).
pub const ROLLBACK_FLAG: &str = "--aero-rollback-update";
/// The self-test's command-line flag: print [`SELF_TEST_OK`] and the version, exit 0.
pub const SELF_TEST_FLAG: &str = "--aero-update-self-test";
/// What the self-test prints before the version.
pub const SELF_TEST_OK: &str = "aero-update-self-test ok";
/// Starts of a version on trial without reaching the title screen before it is rolled back.
pub const MAX_TRIAL_STARTS: u32 = 3;
/// Starts after [`MAX_TRIAL_STARTS`] that hand over to the rollback
/// and exit. A rollback that keeps failing (its `update\previous\` gone, files held open) would
/// otherwise make every later start exit at once, with nothing on screen, for good: past these,
/// the client starts whatever `app\` holds and the helper's log says why the rollback failed.
pub const MAX_ROLLBACK_TRIES: u32 = 3;

const WAIT_FOR_EXIT: Duration = Duration::from_secs(60);
const WATCH_FOR: Duration = Duration::from_secs(180);
const SELF_TEST_TIMEOUT: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(500);
const MAX_LOG_BYTES: u64 = 256 * 1024;

/// What the helper did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The new version reached its title screen.
    Applied,
    /// Swapped and started; the helper stopped watching before the title screen showed (the
    /// trial stays open, see the module docs).
    Unconfirmed,
    /// The new version did not start; the old one is back and was started again.
    RolledBack,
    /// Nothing was swapped (the old version was started again, unless it was still running).
    NotApplied,
    /// A swap or rollback step failed and could not be fully undone: see `update\update.log`.
    Broken,
}

impl Outcome {
    /// The helper's exit code.
    #[must_use]
    pub fn exit_code(&self) -> u8 {
        match self {
            Outcome::Applied => 0,
            Outcome::Unconfirmed => 10,
            Outcome::RolledBack => 11,
            Outcome::NotApplied => 12,
            Outcome::Broken => 13,
        }
    }
}

/// What the helper does to the world, so tests can stand in for processes and time.
pub trait Hooks {
    /// Whether a client runs from `exe`.
    fn client_running(&self, exe: &Path) -> bool;
    /// Whether the client at `exe` passes its self-test for `version`.
    fn self_test(&self, exe: &Path, version: &str) -> bool;
    /// Start the client: `app\mc-client.exe` with no arguments (see [`relaunch_command`]).
    fn launch(&self, paths: &InstallPaths) -> std::io::Result<()>;
    /// Wait.
    fn sleep(&self, duration: Duration);
    /// The time now.
    fn now(&self) -> Instant;
    /// How long to wait for the old client to exit.
    fn wait_for_exit(&self) -> Duration {
        WAIT_FOR_EXIT
    }
    /// How long to watch the new version.
    fn watch_for(&self) -> Duration {
        WATCH_FOR
    }
}

/// The real processes and clock.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemHooks;

impl Hooks for SystemHooks {
    fn client_running(&self, exe: &Path) -> bool {
        client_running(exe)
    }

    fn self_test(&self, exe: &Path, version: &str) -> bool {
        run_self_test(exe, version)
    }

    fn launch(&self, paths: &InstallPaths) -> std::io::Result<()> {
        // The client runs for the whole session: started and left running.
        relaunch_command(paths).spawn().map(drop)
    }

    fn sleep(&self, duration: Duration) {
        std::thread::sleep(duration);
    }

    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// The helper's log, `update\update.log` (best effort; started over past 256 KiB). Lines name
/// no folder outside the install.
pub struct Log {
    file: Option<Mutex<File>>,
}

impl Log {
    /// Open (append) the install's log.
    #[must_use]
    pub fn open(paths: &InstallPaths) -> Log {
        let path = paths.log_file();
        let _ = std::fs::create_dir_all(&paths.update);
        if std::fs::metadata(&path).is_ok_and(|meta| meta.len() > MAX_LOG_BYTES) {
            let _ = std::fs::remove_file(&path);
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .ok()
            .map(Mutex::new);
        Log { file }
    }

    /// No log (tests).
    #[must_use]
    pub fn none() -> Log {
        Log { file: None }
    }

    /// Append one line.
    pub fn line(&self, text: &str) {
        tracing::info!("{text}");
        if let Some(file) = &self.file {
            let mut file = file
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let _ = writeln!(file, "{} {text}", crate::now_unix());
        }
    }
}

/// Handle the helper and self-test command lines: `Some(exit code)` when this process is one of
/// them (`main` returns it at once, before anything of the game starts), `None` otherwise.
pub fn dispatch(args: impl IntoIterator<Item = OsString>) -> Option<u8> {
    let mut args = args.into_iter().skip(1);
    let first = args.next()?;
    let rollback = match first.to_str() {
        Some(SELF_TEST_FLAG) => {
            println!("{SELF_TEST_OK} {}", crate::CURRENT_VERSION);
            return Some(0);
        }
        Some(APPLY_FLAG) => false,
        Some(ROLLBACK_FLAG) => true,
        _ => return None,
    };
    let (mut root, mut version) = (None, None);
    while let Some(flag) = args.next() {
        match (flag.to_str(), args.next()) {
            (Some("--root"), Some(value)) => root = Some(PathBuf::from(value)),
            (Some("--version"), Some(value)) => version = value.into_string().ok(),
            _ => return Some(2),
        }
    }
    // The client passes its own install folder, an absolute path; nothing relative to wherever
    // this process happens to run.
    let root = root.filter(|root| root.is_absolute());
    let Some(paths) = root.and_then(|root| InstallPaths::at_root(&root).ok()) else {
        return Some(3);
    };
    let log = Log::open(&paths);
    let hooks = SystemHooks;
    let outcome = if rollback {
        log.line("rollback requested by a version that did not reach its title screen");
        roll_back_trial(&paths, &hooks, &log)
    } else {
        match version.filter(|v| crate::version::parse_release_tag(v).is_some()) {
            Some(version) => apply(&paths, &version, &hooks, &log),
            None => return Some(2),
        }
    };
    log.line(&format!("helper finished: {outcome:?}"));
    Some(outcome.exit_code())
}

/// Wait until `done` holds or `limit` passes. Returns whether it held.
fn wait_until(hooks: &dyn Hooks, limit: Duration, mut done: impl FnMut() -> bool) -> bool {
    let deadline = hooks.now() + limit;
    loop {
        if done() {
            return true;
        }
        if hooks.now() >= deadline {
            return false;
        }
        hooks.sleep(POLL);
    }
}

/// Apply the staged `version` (see the module docs). Runs in the helper process.
pub fn apply(paths: &InstallPaths, version: &str, hooks: &dyn Hooks, log: &Log) -> Outcome {
    log.line(&format!(
        "applying {version} over {}",
        crate::CURRENT_VERSION
    ));
    if !wait_until(hooks, hooks.wait_for_exit(), || {
        !hooks.client_running(&paths.exe())
    }) {
        log.line("the old client is still running; nothing applied");
        return Outcome::NotApplied;
    }
    // Never a downgrade, whatever the state file says: the version must be newer than this
    // helper (the version being replaced).
    let newer = crate::version::parse_release_tag(version)
        .is_some_and(|v| crate::version::is_newer(&v, &crate::version::current()));
    if !newer {
        return not_applied(paths, hooks, log, version, "it is not newer", false);
    }
    let staged_root = paths.staged_root(version);
    let staged_exe = staged_root.join(APP_DIR).join(EXE_NAME);
    let recorded = state::read(paths)
        .staged
        .filter(|staged| staged.version == version);
    let Some(recorded) = recorded else {
        return not_applied(paths, hooks, log, version, "nothing staged for it", false);
    };
    match crate::verify::sha256_file(&staged_exe) {
        Ok(hash) if hash == recorded.exe_sha256 => {}
        _ => {
            return not_applied(
                paths,
                hooks,
                log,
                version,
                "the staged files changed",
                false,
            );
        }
    }
    if !hooks.self_test(&staged_exe, version) {
        return not_applied(
            paths,
            hooks,
            log,
            version,
            "it doesn't run on this PC",
            true,
        );
    }
    if let Err(error) = swap::swap_in(paths, &staged_root) {
        let reason = format!("the swap failed: {error}");
        return not_applied(paths, hooks, log, version, &reason, false);
    }
    log.line(&format!("{version} swapped in"));
    let recorded = state::update(paths, |state| {
        state.staged = None;
        state.last_failure = None;
        state.trial = Some(Trial {
            version: version.to_owned(),
            previous_version: crate::CURRENT_VERSION.to_owned(),
            attempts: 0,
            healthy: false,
            started_unix: crate::now_unix(),
        });
    });
    if let Err(error) = recorded {
        log.line(&format!("can't record the trial ({error}); rolling back"));
        return roll_back_trial_of(
            paths,
            hooks,
            log,
            version,
            "its trial could not be recorded",
        );
    }
    if let Err(error) = hooks.launch(paths) {
        log.line(&format!("can't start {version}: {error}"));
        return roll_back_trial_of(paths, hooks, log, version, "it could not be started");
    }
    watch(paths, version, hooks, log)
}

/// Watch a version on trial until it reaches its title screen, exits without doing so, or the
/// watch ends.
fn watch(paths: &InstallPaths, version: &str, hooks: &dyn Hooks, log: &Log) -> Outcome {
    let deadline = hooks.now() + hooks.watch_for();
    let open = |trial: &Option<Trial>| {
        trial
            .as_ref()
            .filter(|trial| trial.version == version && !trial.healthy)
            .map(|trial| trial.attempts)
    };
    loop {
        hooks.sleep(POLL);
        let Some(attempts) = open(&state::read(paths).trial) else {
            log.line(&format!("{version} reached its title screen"));
            return Outcome::Applied;
        };
        if attempts > 0 && !hooks.client_running(&paths.exe()) {
            // It may have reached its title screen just before it exited.
            if open(&state::read(paths).trial).is_none() {
                log.line(&format!("{version} reached its title screen"));
                return Outcome::Applied;
            }
            return roll_back_trial_of(
                paths,
                hooks,
                log,
                version,
                "it closed before its title screen",
            );
        }
        if hooks.now() >= deadline {
            log.line(&format!(
                "stopped watching {version} (started {attempts} times, no title screen yet)"
            ));
            return Outcome::Unconfirmed;
        }
    }
}

/// Nothing was swapped: record why, drop the staged files, and start the old version again.
fn not_applied(
    paths: &InstallPaths,
    hooks: &dyn Hooks,
    log: &Log,
    version: &str,
    reason: &str,
    bad: bool,
) -> Outcome {
    log.line(&format!("{version} not applied: {reason}"));
    let _ = state::update(paths, |state| {
        state.staged = None;
        if bad && !state.bad_versions.iter().any(|v| v == version) {
            state.bad_versions.push(version.to_owned());
        }
        state.last_failure = Some(Failure {
            version: version.to_owned(),
            kept_version: crate::CURRENT_VERSION.to_owned(),
            reason: reason.to_owned(),
            shown: false,
        });
    });
    let _ = std::fs::remove_dir_all(paths.staging_root());
    relaunch(paths, hooks, log);
    Outcome::NotApplied
}

/// Roll back the open trial (the rollback command line).
fn roll_back_trial(paths: &InstallPaths, hooks: &dyn Hooks, log: &Log) -> Outcome {
    let Some(trial) = state::read(paths).trial else {
        log.line("no version on trial; nothing to roll back");
        return Outcome::NotApplied;
    };
    if !wait_until(hooks, hooks.wait_for_exit(), || {
        !hooks.client_running(&paths.exe())
    }) {
        log.line("the client is still running; not rolled back");
        return Outcome::NotApplied;
    }
    roll_back_trial_of(
        paths,
        hooks,
        log,
        &trial.version,
        "it didn't reach its title screen",
    )
}

/// Put the previous version back, record `version` as failing on this PC, and start the old
/// version.
fn roll_back_trial_of(
    paths: &InstallPaths,
    hooks: &dyn Hooks,
    log: &Log,
    version: &str,
    reason: &str,
) -> Outcome {
    let kept = state::read(paths)
        .trial
        .map_or_else(|| crate::CURRENT_VERSION.to_owned(), |t| t.previous_version);
    if let Err(error) = swap::roll_back(paths) {
        log.line(&format!("ROLLBACK FAILED for {version}: {error}"));
        return Outcome::Broken;
    }
    log.line(&format!("{version} rolled back to {kept}: {reason}"));
    let _ = state::update(paths, |state| {
        state.trial = None;
        state.staged = None;
        if !state.bad_versions.iter().any(|v| v == version) {
            state.bad_versions.push(version.to_owned());
        }
        state.last_failure = Some(Failure {
            version: version.to_owned(),
            kept_version: kept.clone(),
            reason: reason.to_owned(),
            shown: false,
        });
    });
    relaunch(paths, hooks, log);
    Outcome::RolledBack
}

fn relaunch(paths: &InstallPaths, hooks: &dyn Hooks, log: &Log) {
    if let Err(error) = hooks.launch(paths) {
        log.line(&format!("can't start Aero Client again: {error}"));
    }
}

/// What `main` does after [`on_client_start`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartAction {
    /// Start the game.
    Run,
    /// Exit at once: the helper is rolling this version back.
    Exit,
}

/// The client starts (called from `main` right after [`dispatch`]). In a portable install,
/// counts a start of a version on trial, and hands a version that has not reached its title
/// screen in [`MAX_TRIAL_STARTS`] starts to the helper's rollback. Elsewhere, nothing (one file
/// check, once).
pub fn on_client_start() -> StartAction {
    match portable_install() {
        Some(paths) => count_start(paths, crate::CURRENT_VERSION, &|paths| {
            spawn_helper(paths, ROLLBACK_FLAG, None)
        }),
        None => StartAction::Run,
    }
}

/// [`on_client_start`] for the client of `version`.
fn count_start(
    paths: &InstallPaths,
    version: &str,
    spawn_rollback: &dyn Fn(&InstallPaths) -> std::io::Result<()>,
) -> StartAction {
    let attempts = state::update(paths, |state| match &mut state.trial {
        Some(trial) if trial.version == version && !trial.healthy => {
            trial.attempts += 1;
            Some(trial.attempts)
        }
        _ => None,
    });
    match attempts {
        Ok(Some(attempts)) if attempts > MAX_TRIAL_STARTS + MAX_ROLLBACK_TRIES => {
            tracing::warn!(
                attempts,
                "the update rollback did not happen; starting anyway"
            );
            StartAction::Run
        }
        Ok(Some(attempts)) if attempts > MAX_TRIAL_STARTS && paths.helper_exe().is_file() => {
            match spawn_rollback(paths) {
                Ok(()) => StartAction::Exit,
                Err(error) => {
                    tracing::warn!(%error, "can't start the update rollback");
                    StartAction::Run
                }
            }
        }
        _ => StartAction::Run,
    }
}

/// The title screen showed: a version on trial is good (its trial ends, and the update's
/// leftover folders go). Once per process: one relaxed atomic load per call after the first.
/// The file work runs on a short-lived thread, off the frame.
pub fn title_screen_reached() {
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.load(Ordering::Relaxed) || DONE.swap(true, Ordering::Relaxed) {
        return;
    }
    let Some(paths) = portable_install() else {
        return;
    };
    let paths = paths.clone();
    let _ = std::thread::Builder::new()
        .name("aero-update-health".to_owned())
        .spawn(move || mark_healthy(&paths));
}

/// End this version's trial, if it has one, and clean up after the update.
pub fn mark_healthy(paths: &InstallPaths) {
    end_trial(paths, crate::CURRENT_VERSION);
}

/// [`mark_healthy`] for the client of `version`.
fn end_trial(paths: &InstallPaths, version: &str) {
    let ended = state::update(paths, |state| {
        let ours = state
            .trial
            .as_ref()
            .is_some_and(|trial| trial.version == version);
        if ours {
            state.trial = None;
        }
        ours
    });
    if matches!(ended, Ok(true)) {
        tracing::info!(version, "update confirmed");
        swap::clean_after_update(paths);
    } else if state::read(paths).trial.is_none() && paths.helper_dir().exists() {
        // A helper from an earlier update that has finished: its copy is no longer needed.
        let _ = std::fs::remove_dir_all(paths.helper_dir());
    }
}

/// Whether a client process runs from `exe`: Windows refuses to open a running executable for
/// writing (a sharing violation). Opening it for writing changes nothing.
#[must_use]
pub fn client_running(exe: &Path) -> bool {
    if !cfg!(windows) {
        return false;
    }
    match OpenOptions::new().write(true).open(exe) {
        Ok(_) => false,
        Err(error) => error.raw_os_error() == Some(32),
    }
}

#[cfg(windows)]
fn no_window(command: &mut Command, detached: bool) {
    use std::os::windows::process::CommandExt as _;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    const DETACHED_PROCESS: u32 = 0x0000_0008;
    const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
    command.creation_flags(if detached {
        DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    } else {
        CREATE_NO_WINDOW
    });
}

#[cfg(not(windows))]
fn no_window(_command: &mut Command, _detached: bool) {}

/// Run the client at `exe` with [`SELF_TEST_FLAG`]: it must exit 0 within 30 seconds having
/// printed [`SELF_TEST_OK`] and `version`.
fn run_self_test(exe: &Path, version: &str) -> bool {
    let mut command = Command::new(exe);
    command
        .arg(SELF_TEST_FLAG)
        .current_dir(exe.parent().unwrap_or(exe))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    no_window(&mut command, false);
    let Ok(mut child) = command.spawn() else {
        return false;
    };
    let deadline = Instant::now() + SELF_TEST_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut out = String::new();
                if let Some(stdout) = child.stdout.take() {
                    let _ = stdout.take(4096).read_to_string(&mut out);
                }
                return status.success() && out.contains(&format!("{SELF_TEST_OK} {version}"));
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(50)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

/// How the helper starts the client after an update or a rollback:
/// `app\mc-client.exe` itself, with no arguments, so it makes its packaged start (finds the
/// Minecraft files, writes `game\logs`, reports problems) exactly as when the player starts it
/// with `Aero Client.exe` or a shortcut. Working folder the install folder, no console window
/// (a packaged client has none anyway). It needs nothing in the install folder but `app\`, so an
/// install from before `Aero Client.exe` starts the same way.
#[must_use]
pub fn relaunch_command(paths: &InstallPaths) -> Command {
    let mut command = Command::new(paths.exe());
    command
        .current_dir(&paths.root)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    no_window(&mut command, false);
    command
}

/// Start the helper copy (`update\helper\aero-update-helper.exe`) with `flag`, detached, its
/// working folder `update\` (so nothing holds `app\` open).
pub(crate) fn spawn_helper(
    paths: &InstallPaths,
    flag: &str,
    version: Option<&str>,
) -> std::io::Result<()> {
    let mut command = Command::new(paths.helper_exe());
    command.arg(flag).arg("--root").arg(&paths.root);
    if let Some(version) = version {
        command.arg("--version").arg(version);
    }
    command
        .current_dir(&paths.update)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    no_window(&mut command, true);
    command.spawn().map(drop)
}

/// Copy the running client to the helper's place, replacing an older copy.
pub(crate) fn install_helper(paths: &InstallPaths) -> Result<(), crate::UpdateError> {
    let exe = std::env::current_exe()
        .map_err(|error| crate::UpdateError::io("finding the running client", error))?;
    std::fs::create_dir_all(paths.helper_dir())
        .map_err(|error| crate::UpdateError::io("creating the helper folder", error))?;
    std::fs::copy(&exe, paths.helper_exe())
        .map_err(|error| crate::UpdateError::io("copying the helper", error))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::{fake_install, fake_staged_recorded, FakeHooks};

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    /// The version installed in these tests: this build's own, since `apply` only swaps in a
    /// release newer than the helper running it and records this build as the kept version.
    const OLD: &str = crate::CURRENT_VERSION;

    /// A release newer than this build (the next minor version).
    fn new_version() -> String {
        let current = crate::version::current();
        format!("{}.{}.0", current.major, current.minor + 1)
    }

    #[test]
    fn a_new_version_that_reaches_its_title_screen_stays() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, &new);
        let hooks = FakeHooks::new(&paths);
        // The new client starts once and reaches its title screen.
        let launched = new.clone();
        hooks.on_launch(move |paths| {
            assert_eq!(
                read(&paths.exe()),
                format!("exe {launched}"),
                "launched after the swap"
            );
            let _ = count_start_as(paths, &launched);
            mark_healthy_as(paths, &launched);
        });
        let outcome = apply(&paths, &new, &hooks, &Log::none());
        assert_eq!(outcome, Outcome::Applied);
        assert_eq!(read(&paths.exe()), format!("exe {new}"));
        assert_eq!(hooks.launches(), 1);
        let state = state::read(&paths);
        assert_eq!(state.trial, None);
        assert_eq!(state.staged, None);
        assert!(state.bad_versions.is_empty());
    }

    #[test]
    fn a_new_version_that_closes_before_its_title_screen_is_rolled_back() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, &new);
        let hooks = FakeHooks::new(&paths);
        // The new client starts and dies at once (it is not running by the next poll).
        let launched = new.clone();
        hooks.on_launch(move |paths| {
            if read(&paths.exe()) == format!("exe {launched}") {
                let _ = count_start_as(paths, &launched);
            }
        });
        let outcome = apply(&paths, &new, &hooks, &Log::none());
        assert_eq!(outcome, Outcome::RolledBack);
        assert_eq!(
            read(&paths.exe()),
            format!("exe {OLD}"),
            "the old version is back"
        );
        assert_eq!(read(&paths.launcher_cmd()), format!("@rem cmd {OLD}"));
        assert_eq!(read(&paths.launcher_exe()), format!("starter {OLD}"));
        assert_eq!(
            hooks.launches(),
            2,
            "the new version, then the old one again"
        );
        let state = state::read(&paths);
        assert_eq!(state.trial, None);
        assert_eq!(state.bad_versions, [new.as_str()]);
        let failure = state.last_failure.as_ref().expect("recorded");
        assert_eq!(
            (failure.version.as_str(), failure.kept_version.as_str()),
            (new.as_str(), OLD)
        );
        assert!(!state.may_offer(&new), "never offered again");
    }

    #[test]
    fn a_version_that_fails_its_self_test_is_never_swapped_in() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, &new);
        let hooks = FakeHooks::new(&paths);
        hooks.self_test_passes(false);
        let outcome = apply(&paths, &new, &hooks, &Log::none());
        assert_eq!(outcome, Outcome::NotApplied);
        assert_eq!(read(&paths.exe()), format!("exe {OLD}"));
        assert_eq!(hooks.launches(), 1, "the old version is started again");
        let state = state::read(&paths);
        assert_eq!(state.bad_versions, [new.as_str()]);
        assert_eq!(state.staged, None);
        assert!(!paths.staging_root().exists(), "the staged files are gone");
    }

    #[test]
    fn staged_files_that_changed_since_staging_are_not_used() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        let staged = fake_staged_recorded(&paths, &new);
        std::fs::write(staged.join(APP_DIR).join(EXE_NAME), "tampered").unwrap();
        let hooks = FakeHooks::new(&paths);
        assert_eq!(
            apply(&paths, &new, &hooks, &Log::none()),
            Outcome::NotApplied
        );
        assert_eq!(read(&paths.exe()), format!("exe {OLD}"));
        assert_eq!(hooks.self_tests(), 0, "not even run");
    }

    #[test]
    fn an_older_or_equal_version_is_never_applied() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, OLD);
        let hooks = FakeHooks::new(&paths);
        assert_eq!(
            apply(&paths, OLD, &hooks, &Log::none()),
            Outcome::NotApplied
        );
        assert_eq!(hooks.self_tests(), 0, "refused before the self-test");
        assert_eq!(read(&paths.exe()), format!("exe {OLD}"));
    }

    #[test]
    fn nothing_is_applied_while_the_old_client_still_runs() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, &new);
        let hooks = FakeHooks::new(&paths);
        hooks.old_client_never_exits();
        assert_eq!(
            apply(&paths, &new, &hooks, &Log::none()),
            Outcome::NotApplied
        );
        assert_eq!(read(&paths.exe()), format!("exe {OLD}"));
        assert_eq!(hooks.launches(), 0, "it is still running: nothing to start");
        assert!(
            state::read(&paths).staged.is_some(),
            "still staged for next time"
        );
    }

    #[test]
    fn a_trial_the_helper_stops_watching_is_rolled_back_after_three_failed_starts() {
        let new = new_version();
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), OLD);
        fake_staged_recorded(&paths, &new);
        let hooks = FakeHooks::new(&paths);
        // The start stops before the game (say, the player closed the question about the game
        // files): no start is counted.
        hooks.on_launch(|_| {});
        assert_eq!(
            apply(&paths, &new, &hooks, &Log::none()),
            Outcome::Unconfirmed
        );
        assert_eq!(
            read(&paths.exe()),
            format!("exe {new}"),
            "swapped, on trial"
        );
        std::fs::create_dir_all(paths.helper_dir()).unwrap();
        std::fs::write(paths.helper_exe(), "old exe").unwrap();
        // Three starts that never reach the title screen run; the fourth hands over.
        let spawned = std::cell::Cell::new(0);
        let spawn = |_: &InstallPaths| {
            spawned.set(spawned.get() + 1);
            Ok(())
        };
        for _ in 0..MAX_TRIAL_STARTS {
            assert_eq!(count_start_with(&paths, &new, &spawn), StartAction::Run);
        }
        assert_eq!(count_start_with(&paths, &new, &spawn), StartAction::Exit);
        assert_eq!(spawned.get(), 1);
        // A rollback that keeps failing (here it never runs) is
        // handed over a bounded number of times; after that the client starts again instead of
        // exiting at every start for good.
        for _ in 1..MAX_ROLLBACK_TRIES {
            assert_eq!(count_start_with(&paths, &new, &spawn), StartAction::Exit);
        }
        assert_eq!(spawned.get(), MAX_ROLLBACK_TRIES);
        for _ in 0..3 {
            assert_eq!(count_start_with(&paths, &new, &spawn), StartAction::Run);
        }
        assert_eq!(spawned.get(), MAX_ROLLBACK_TRIES, "no more hand-overs");
        // The helper's rollback then puts the old version back.
        let outcome = roll_back_trial(&paths, &hooks, &Log::none());
        assert_eq!(outcome, Outcome::RolledBack);
        assert_eq!(read(&paths.exe()), format!("exe {OLD}"));
        assert_eq!(state::read(&paths).bad_versions, [new.as_str()]);
    }

    #[test]
    fn the_running_test_binary_counts_as_a_running_client() {
        let exe = std::env::current_exe().unwrap();
        if cfg!(windows) {
            assert!(client_running(&exe), "this test binary is running");
        }
        let dir = tempfile::tempdir().unwrap();
        let copy = dir.path().join("copy.exe");
        std::fs::copy(&exe, &copy).unwrap();
        assert!(!client_running(&copy), "a copy nobody runs");
        assert!(!client_running(&dir.path().join("missing.exe")));
    }

    /// The helper starts the client itself: `app\mc-client.exe` by its full
    /// path, no arguments (its packaged start), in the install folder, never a file next to
    /// `app\`; an install from before `Aero Client.exe` starts the same way.
    #[test]
    fn the_relaunch_starts_the_client_itself_with_no_arguments() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        let command = relaunch_command(&paths);
        assert_eq!(command.get_program(), paths.exe().as_os_str());
        assert_eq!(
            command.get_args().count(),
            0,
            "no arguments: the packaged start"
        );
        assert_eq!(command.get_current_dir(), Some(paths.root.as_path()));
        assert!(paths.exe().is_absolute());
        std::fs::remove_file(paths.launcher_exe()).unwrap();
        std::fs::remove_file(paths.launcher_cmd()).unwrap();
        let without = InstallPaths::at_root(&paths.root).expect("still an install");
        assert_eq!(
            relaunch_command(&without).get_program(),
            paths.exe().as_os_str()
        );
    }

    /// The real launch: a stand-in client (Windows' own `hostname.exe`, which prints the PC's
    /// name and exits) in a folder with a space in its name ("Aero Client") is started and left
    /// to run, as the helper starts the real one.
    #[cfg(windows)]
    #[test]
    fn the_real_launch_starts_the_client_in_a_folder_with_spaces() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        assert!(paths.root.to_string_lossy().contains("Aero Client"));
        let system = std::env::var_os("SystemRoot").unwrap_or_else(|| r"C:\Windows".into());
        let stand_in = Path::new(&system).join("System32").join("hostname.exe");
        std::fs::copy(stand_in, paths.exe()).unwrap();
        SystemHooks.launch(&paths).expect("Windows started it");
        // It exits on its own; wait for that so the folder can be deleted.
        let deadline = Instant::now() + Duration::from_secs(30);
        while client_running(&paths.exe()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!client_running(&paths.exe()), "the stand-in has exited");
    }

    #[test]
    fn the_self_test_flag_answers_without_starting_anything() {
        let code = dispatch([
            OsString::from("mc-client.exe"),
            OsString::from(SELF_TEST_FLAG),
        ]);
        assert_eq!(code, Some(0));
        assert_eq!(dispatch([OsString::from("mc-client.exe")]), None);
        assert_eq!(
            dispatch(["mc-client.exe", "--username", "Steve"].map(OsString::from)),
            None,
            "a normal command line is the game's"
        );
        assert_eq!(
            dispatch(["x", APPLY_FLAG, "--root"].map(OsString::from)),
            Some(2),
            "a malformed helper command line"
        );
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_string_lossy().into_owned();
        assert_eq!(
            dispatch(["x", APPLY_FLAG, "--root", &root, "--version", "0.2.0"].map(OsString::from)),
            Some(3),
            "not an install"
        );
    }

    fn count_start_with(
        paths: &InstallPaths,
        version: &str,
        spawn: &dyn Fn(&InstallPaths) -> std::io::Result<()>,
    ) -> StartAction {
        count_start(paths, version, spawn)
    }

    fn count_start_as(paths: &InstallPaths, version: &str) -> StartAction {
        count_start(paths, version, &|_| Ok(()))
    }

    fn mark_healthy_as(paths: &InstallPaths, version: &str) {
        end_trial(paths, version);
    }

    /// The start and title-screen hooks of this build's own version.
    #[test]
    fn this_builds_start_and_title_screen_hooks() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), crate::CURRENT_VERSION);
        state::update(&paths, |state| {
            state.trial = Some(Trial {
                version: crate::CURRENT_VERSION.to_owned(),
                previous_version: "0.0.1".to_owned(),
                attempts: 0,
                healthy: false,
                started_unix: 1,
            });
        })
        .unwrap();
        std::fs::create_dir_all(paths.staging_root().join("x")).unwrap();
        let start = || count_start(&paths, crate::CURRENT_VERSION, &|_| Ok(()));
        assert_eq!(start(), StartAction::Run);
        assert_eq!(state::read(&paths).trial.map(|t| t.attempts), Some(1));
        mark_healthy(&paths);
        assert_eq!(state::read(&paths).trial, None);
        assert!(
            !paths.staging_root().exists(),
            "cleaned up after the update"
        );
        // No trial: a start changes nothing.
        assert_eq!(start(), StartAction::Run);
        assert_eq!(state::read(&paths), state::UpdateState::default());
    }
}
