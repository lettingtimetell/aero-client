//! The update helper's command lines against the real client binary:
//! the self-test the helper runs on a staged version before swapping it in, and a helper command
//! line pointed at a folder that is not an install. Neither opens a window or touches a game
//! folder: `mc_update::helper::dispatch` answers before anything of the game starts.
//!
//! After an update (or not), the helper starts `app\mc-client.exe` itself with
//! no arguments: the real client's packaged start. These tests run it in test mode
//! (`AERO_LAUNCHER_NOGUI`) with the per-user folders in the test's temporary folder, where there
//! are no Minecraft files: the client writes the question it would have asked to
//! `game\logs\latest-launch.log` and exits without starting the game.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use mc_update::helper::{
    Hooks as _, SystemHooks, APPLY_FLAG, ROLLBACK_FLAG, SELF_TEST_FLAG, SELF_TEST_OK,
};
use mc_update::install::InstallMarker;
use mc_update::state::{StagedUpdate, Trial, UpdateState};

fn client() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_mc-client"))
}

/// An `Aero Client` folder in `dir` whose `app\` holds a copy of the real client with a marker
/// naming `version`, next to an `Aero Client.exe` and a `Play Aero Client.cmd` that name `tag`
/// (stand-ins: the helper starts neither).
fn install_folder(dir: &Path, version: &str, tag: &str) -> PathBuf {
    let app = dir.join("app");
    std::fs::create_dir_all(&app).unwrap();
    std::fs::copy(client(), app.join("mc-client.exe")).unwrap();
    std::fs::write(
        app.join("aero-install.json"),
        InstallMarker::new(version).to_json(),
    )
    .unwrap();
    std::fs::write(dir.join("Aero Client.exe"), format!("starter {tag}")).unwrap();
    std::fs::write(dir.join("Play Aero Client.cmd"), format!("@rem {tag}\r\n")).unwrap();
    dir.to_path_buf()
}

/// Run the helper: a copy of the real client in `update\helper\`, started as the client starts
/// it (working folder `update\`), waited for. The client it starts afterwards inherits test
/// mode and per-user folders inside `root`'s parent, where there are no Minecraft files.
fn run_helper(root: &Path, args: &[&str]) -> (Option<i32>, String) {
    let update = root.join("update");
    let helper = update.join("helper").join("aero-update-helper.exe");
    std::fs::create_dir_all(helper.parent().unwrap()).unwrap();
    std::fs::copy(client(), &helper).unwrap();
    let user = root.parent().unwrap().join("user");
    for folder in ["Roaming", "Local", "config"] {
        std::fs::create_dir_all(user.join(folder)).unwrap();
    }
    let status = Command::new(&helper)
        .args(args)
        .current_dir(&update)
        .env("MC_NO_AUDIO", "1")
        .env("MC_BACKGROUND_WINDOW", "1")
        .env(mc_shell::dialog::NO_GUI_ENV, "1")
        .env("AERO_LAUNCHER_FRAMES", "1")
        .env("APPDATA", user.join("Roaming"))
        .env("LOCALAPPDATA", user.join("Local"))
        .env("MC_RS_CONFIG_DIR", user.join("config"))
        .status()
        .expect("run the helper");
    let log = std::fs::read_to_string(update.join("update.log")).unwrap_or_default();
    (status.code(), log)
}

/// The packaged start of the client the helper started from `root\app\`: wait (up to two
/// minutes) until it has written the game files question to `game\logs\latest-launch.log` and
/// exited. Returns that log.
fn packaged_start_log(root: &Path) -> String {
    let log = root.join("game").join("logs").join("latest-launch.log");
    let exe = root.join("app").join("mc-client.exe");
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let text = std::fs::read_to_string(&log).unwrap_or_default();
        if text.contains("couldn't find them") && !mc_update::helper::client_running(&exe) {
            return text;
        }
        assert!(
            Instant::now() < deadline,
            "the client did not make its packaged start; log:\n{text}"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn marker_version(app: &Path) -> String {
    InstallMarker::read(&app.join("aero-install.json"))
        .unwrap()
        .version
}

/// The whole apply path with real processes, up to the self-test: the staged "99.0.0" is the
/// real client, whose self-test answers with its own version, so the helper must refuse it,
/// leave `app\` alone, forget the staged files and start the old version again: the real
/// `app\mc-client.exe`, which makes its packaged start.
#[cfg(windows)]
#[test]
fn the_real_helper_refuses_a_staged_version_that_fails_its_self_test() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = install_folder(
        &dir.path().join("Aero Client"),
        mc_update::CURRENT_VERSION,
        "old",
    );
    let staged = install_folder(
        &root
            .join("update")
            .join("staging")
            .join("99.0.0")
            .join("Aero Client"),
        "99.0.0",
        "new",
    );
    let exe_sha256 =
        mc_update::verify::sha256_file(&staged.join("app").join("mc-client.exe")).unwrap();
    UpdateState {
        staged: Some(StagedUpdate {
            version: "99.0.0".to_owned(),
            exe_sha256,
            zip_sha256: "0".repeat(64),
        }),
        ..UpdateState::default()
    }
    .save(&root.join("update").join("state.json"))
    .unwrap();
    let before = mc_update::verify::sha256_file(&root.join("app").join("mc-client.exe")).unwrap();

    let root_text = root.to_str().unwrap();
    let (code, log) = run_helper(
        &root,
        &[APPLY_FLAG, "--root", root_text, "--version", "99.0.0"],
    );
    assert_eq!(code, Some(12), "not applied; log:\n{log}");
    assert!(
        log.contains("99.0.0 not applied: it doesn't run on this PC"),
        "{log}"
    );
    assert_eq!(
        mc_update::verify::sha256_file(&root.join("app").join("mc-client.exe")).unwrap(),
        before,
        "app is untouched"
    );
    assert_eq!(
        marker_version(&root.join("app")),
        mc_update::CURRENT_VERSION
    );
    let started = packaged_start_log(&root);
    assert!(
        started.contains("[Question] Aero Client uses your own Minecraft 1.8.9 game files"),
        "the old version was started again, with no arguments: {started}"
    );
    assert!(
        !root.join("update").join("staging").exists(),
        "the staged files are gone"
    );
    let state = UpdateState::load(&root.join("update").join("state.json"));
    assert_eq!(state.staged, None);
    assert_eq!(state.bad_versions, ["99.0.0"]);
    assert_eq!(
        state.last_failure.map(|f| f.version).as_deref(),
        Some("99.0.0")
    );
}

/// The rollback command line with real folders and processes: a "99.0.0" on trial that never
/// reached its title screen goes to `update\failed\`, the previous `app\` and root files come
/// back, and the old version is started (its packaged start).
#[cfg(windows)]
#[test]
fn the_real_helper_rolls_back_a_trial() {
    let dir = tempfile::tempdir().expect("tempdir");
    let root = install_folder(&dir.path().join("Aero Client"), "99.0.0", "new");
    let previous = root.join("update").join("previous");
    install_folder(&previous, mc_update::CURRENT_VERSION, "old");
    UpdateState {
        trial: Some(Trial {
            version: "99.0.0".to_owned(),
            previous_version: mc_update::CURRENT_VERSION.to_owned(),
            attempts: 4,
            healthy: false,
            started_unix: 1,
        }),
        ..UpdateState::default()
    }
    .save(&root.join("update").join("state.json"))
    .unwrap();

    let root_text = root.to_str().unwrap();
    let (code, log) = run_helper(&root, &[ROLLBACK_FLAG, "--root", root_text]);
    assert_eq!(code, Some(11), "rolled back; log:\n{log}");
    assert_eq!(
        marker_version(&root.join("app")),
        mc_update::CURRENT_VERSION
    );
    let failed = root.join("update").join("failed");
    assert_eq!(marker_version(&failed.join("app")), "99.0.0");
    for name in ["Aero Client.exe", "Play Aero Client.cmd"] {
        assert!(
            std::fs::read_to_string(root.join(name))
                .unwrap()
                .contains("old"),
            "the old {name} is back"
        );
    }
    let started = packaged_start_log(&root);
    assert!(
        started.contains("[Question] Aero Client uses your own Minecraft 1.8.9 game files"),
        "the old version was started: {started}"
    );
    assert!(
        !failed.join("game").exists(),
        "the version that failed was not started"
    );
    let state = UpdateState::load(&root.join("update").join("state.json"));
    assert_eq!(state.trial, None);
    assert_eq!(state.bad_versions, ["99.0.0"]);
    let failure = state.last_failure.expect("recorded");
    assert_eq!(
        (failure.version.as_str(), failure.kept_version.as_str()),
        ("99.0.0", mc_update::CURRENT_VERSION)
    );
}

#[test]
fn the_client_binary_answers_the_update_self_test() {
    let output = Command::new(client())
        .arg(SELF_TEST_FLAG)
        .env("MC_NO_AUDIO", "1")
        .output()
        .expect("run the client");
    assert!(output.status.success(), "{:?}", output.status);
    assert_eq!(
        String::from_utf8_lossy(&output.stdout).trim(),
        format!("{SELF_TEST_OK} {}", mc_update::CURRENT_VERSION)
    );
    // The helper's own check: passes for this version, fails for any other.
    assert!(SystemHooks.self_test(client(), mc_update::CURRENT_VERSION));
    assert!(!SystemHooks.self_test(client(), "99.0.0"));
}

#[test]
fn a_helper_command_line_outside_an_install_changes_nothing() {
    let dir = tempfile::tempdir().expect("tempdir");
    let status = Command::new(client())
        .args([APPLY_FLAG, "--root"])
        .arg(dir.path())
        .args(["--version", "99.0.0"])
        .env("MC_NO_AUDIO", "1")
        .status()
        .expect("run the client");
    assert_eq!(status.code(), Some(3), "not an install");
    assert_eq!(
        std::fs::read_dir(dir.path()).expect("list").count(),
        0,
        "nothing written"
    );
    // A relative root is refused the same way.
    let status = Command::new(client())
        .args([APPLY_FLAG, "--root", "relative", "--version", "99.0.0"])
        .env("MC_NO_AUDIO", "1")
        .status()
        .expect("run the client");
    assert_eq!(status.code(), Some(3));
}
