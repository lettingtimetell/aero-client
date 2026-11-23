//! The update helper's command lines against the real client binary:
//! the self-test the helper runs on a staged version before swapping it in, and a helper command
//! line pointed at a folder that is not an install. Neither opens a window or touches a game
//! folder: `mc_update::helper::dispatch` answers before anything of the game starts.

use std::path::{Path, PathBuf};
use std::process::Command;

use mc_update::helper::{
    Hooks as _, SystemHooks, APPLY_FLAG, ROLLBACK_FLAG, SELF_TEST_FLAG, SELF_TEST_OK,
};
use mc_update::install::InstallMarker;
use mc_update::state::{StagedUpdate, Trial, UpdateState};

fn client() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_mc-client"))
}

/// An `Aero Client` folder in `dir` whose `app\` holds a copy of the real client with a marker
/// naming `version`, and whose `Play Aero Client.cmd` writes `<tag>.txt` next to itself.
fn install_folder(dir: &Path, version: &str, tag: &str) -> PathBuf {
    let app = dir.join("app");
    std::fs::create_dir_all(&app).unwrap();
    std::fs::copy(client(), app.join("mc-client.exe")).unwrap();
    std::fs::write(app.join("launcher.ps1"), "# launcher").unwrap();
    std::fs::write(
        app.join("aero-install.json"),
        InstallMarker::new(version).to_json(),
    )
    .unwrap();
    std::fs::write(
        dir.join("Play Aero Client.cmd"),
        format!("@echo off\r\necho {tag}> \"%~dp0{tag}.txt\"\r\n"),
    )
    .unwrap();
    dir.to_path_buf()
}

/// Run the helper: a copy of the real client in `update\helper\`, started as the client starts
/// it (working folder `update\`), waited for.
fn run_helper(root: &Path, args: &[&str]) -> (Option<i32>, String) {
    let update = root.join("update");
    let helper = update.join("helper").join("aero-update-helper.exe");
    std::fs::create_dir_all(helper.parent().unwrap()).unwrap();
    std::fs::copy(client(), &helper).unwrap();
    let status = Command::new(&helper)
        .args(args)
        .current_dir(&update)
        .env("MC_NO_AUDIO", "1")
        .status()
        .expect("run the helper");
    let log = std::fs::read_to_string(update.join("update.log")).unwrap_or_default();
    (status.code(), log)
}

fn marker_version(app: &Path) -> String {
    InstallMarker::read(&app.join("aero-install.json"))
        .unwrap()
        .version
}

/// The whole apply path with real processes, up to the self-test: the staged "99.0.0" is the
/// real client, whose self-test answers with its own version, so the helper must refuse it,
/// leave `app\` alone, forget the staged files and start the old version again through the real
/// `Play Aero Client.cmd`.
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
    assert!(
        root.join("old.txt").is_file(),
        "the old version was started again"
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
/// reached its title screen goes to `update\failed\`, the previous `app\` and launcher come back,
/// and the old version is started.
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
    assert!(
        std::fs::read_to_string(root.join("Play Aero Client.cmd"))
            .unwrap()
            .contains("old"),
        "the old launcher is back"
    );
    assert!(
        root.join("old.txt").is_file(),
        "the old version was started"
    );
    assert!(!root.join("new.txt").exists());
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
