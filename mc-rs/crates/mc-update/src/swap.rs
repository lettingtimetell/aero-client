//! Swapping `app\` for a staged one, and back.
//!
//! Every step is a rename inside the install folder (one volume), so each is atomic: at any
//! moment `app\` is either the whole old version or the whole new one. The old `app\` (and the
//! old `Play Aero Client.cmd` and `README.txt`) go to `update\previous\`; when a later step fails,
//! the steps already done are undone in reverse. [`roll_back`] puts `update\previous\` back and
//! moves the version that failed to `update\failed\`. Renames are retried for a few seconds,
//! because a virus scanner looking at a new executable briefly holds it open.
//!
//! The caller guarantees no client runs from `app\` (the helper waits for that).

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::error::UpdateError;
use crate::install::{InstallPaths, APP_DIR};
use crate::stage::ROOT_FILES;

/// How long a rename is retried while something holds the file open (short in unit tests, which
/// hold files open on purpose).
const RENAME_PATIENCE: Duration = if cfg!(test) {
    Duration::from_millis(400)
} else {
    Duration::from_secs(10)
};
const RENAME_RETRY: Duration = Duration::from_millis(200);

/// Rename `from` to `to`, retrying while the OS reports the file as in use.
fn rename_retry(from: &Path, to: &Path, what: &'static str) -> Result<(), UpdateError> {
    let started = Instant::now();
    loop {
        match std::fs::rename(from, to) {
            Ok(()) => return Ok(()),
            Err(error) if in_use(&error) && started.elapsed() < RENAME_PATIENCE => {
                std::thread::sleep(RENAME_RETRY);
            }
            Err(error) => return Err(UpdateError::io(what, error)),
        }
    }
}

/// Access denied (5) and sharing violation (32): what Windows reports while another process
/// holds a file open without sharing deletion.
fn in_use(error: &std::io::Error) -> bool {
    error.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(error.raw_os_error(), Some(5 | 32))
}

/// Delete a folder tree if it exists; if it can't be deleted, move it out of the way.
fn clear(path: &Path, what: &'static str) -> Result<(), UpdateError> {
    if !path.exists() {
        return Ok(());
    }
    if std::fs::remove_dir_all(path).is_ok() {
        return Ok(());
    }
    let aside = path.with_extension(format!("old-{}", crate::now_unix()));
    rename_retry(path, &aside, what)
}

/// The renames done so far, to undo in reverse.
#[derive(Default)]
struct Moves(Vec<(PathBuf, PathBuf)>);

impl Moves {
    fn rename(&mut self, from: &Path, to: &Path, what: &'static str) -> Result<(), UpdateError> {
        rename_retry(from, to, what)?;
        self.0.push((from.to_path_buf(), to.to_path_buf()));
        Ok(())
    }

    /// Undo every rename, newest first. Returns whether all were undone.
    fn undo(self) -> bool {
        let mut ok = true;
        for (from, to) in self.0.into_iter().rev() {
            if let Err(error) = rename_retry(&to, &from, "undoing a swap step") {
                tracing::error!(%error, "a swap step could not be undone");
                ok = false;
            }
        }
        ok
    }
}

/// Swap the staged `Aero Client` folder at `staged_root` in: its `app\` replaces `paths.app`,
/// and its root files replace the install's. On error everything done is undone and the
/// install is as it was (the error says so if even that failed).
pub fn swap_in(paths: &InstallPaths, staged_root: &Path) -> Result<(), UpdateError> {
    let staged_app = staged_root.join(APP_DIR);
    if !staged_app.is_dir() {
        return Err(UpdateError::Apply("nothing staged".to_owned()));
    }
    let previous = paths.previous_dir();
    clear(&previous, "clearing the previous backup")?;
    std::fs::create_dir_all(&previous)
        .map_err(|error| UpdateError::io("creating the previous backup", error))?;

    let mut moves = Moves::default();
    let result: Result<(), UpdateError> = (|| {
        moves.rename(&paths.app, &previous.join(APP_DIR), "moving app aside")?;
        moves.rename(&staged_app, &paths.app, "moving the new app in")?;
        for name in ROOT_FILES {
            let new = staged_root.join(name);
            if !new.is_file() {
                continue;
            }
            let current = paths.root.join(name);
            if current.exists() {
                moves.rename(&current, &previous.join(name), "moving a root file aside")?;
            }
            moves.rename(&new, &current, "moving a new root file in")?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            if moves.undo() {
                Err(error)
            } else {
                Err(UpdateError::Apply(format!(
                    "{error}; undoing it failed too, the old version is in update\\previous"
                )))
            }
        }
    }
}

/// Put `update\previous\` back: the current `app\` (the version that failed) and its root files
/// go to `update\failed\`.
pub fn roll_back(paths: &InstallPaths) -> Result<(), UpdateError> {
    let previous = paths.previous_dir();
    if !previous.join(APP_DIR).is_dir() {
        return Err(UpdateError::Apply(
            "no previous version to go back to".to_owned(),
        ));
    }
    let failed = paths.failed_dir();
    clear(&failed, "clearing the failed folder")?;
    std::fs::create_dir_all(&failed)
        .map_err(|error| UpdateError::io("creating the failed folder", error))?;

    let mut moves = Moves::default();
    let result: Result<(), UpdateError> = (|| {
        if paths.app.exists() {
            moves.rename(
                &paths.app,
                &failed.join(APP_DIR),
                "moving the failed app aside",
            )?;
        }
        moves.rename(
            &previous.join(APP_DIR),
            &paths.app,
            "moving the previous app back",
        )?;
        for name in ROOT_FILES {
            let old = previous.join(name);
            if !old.is_file() {
                continue;
            }
            let current = paths.root.join(name);
            if current.exists() {
                moves.rename(&current, &failed.join(name), "moving a root file aside")?;
            }
            moves.rename(&old, &current, "moving a root file back")?;
        }
        Ok(())
    })();
    match result {
        Ok(()) => Ok(()),
        Err(error) => {
            let undone = moves.undo();
            Err(UpdateError::Apply(format!(
                "rolling back: {error}{}",
                if undone {
                    ""
                } else {
                    " (and undoing it failed)"
                }
            )))
        }
    }
}

/// Folders an applied update no longer needs: the extracted release, the download, a failed
/// version and the helper. `update\previous\` stays (one version to go back to by hand). Best
/// effort: something still open is left for the next start.
pub fn clean_after_update(paths: &InstallPaths) {
    for dir in [
        paths.staging_root(),
        paths.download_dir(),
        paths.failed_dir(),
        paths.helper_dir(),
    ] {
        if dir.exists() {
            if let Err(error) = std::fs::remove_dir_all(&dir) {
                tracing::debug!(%error, "an update folder is still in use; left for later");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::install::{EXE_NAME, LAUNCHER_CMD};
    use crate::tests::{fake_install, fake_staged};

    fn read(path: &Path) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn the_staged_app_and_root_files_replace_the_old_ones() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        std::fs::create_dir_all(paths.root.join("game")).unwrap();
        std::fs::write(paths.root.join("game").join("options.txt"), "mine").unwrap();
        let staged = fake_staged(&paths, "0.2.0");

        swap_in(&paths, &staged).expect("swapped");
        assert_eq!(read(&paths.exe()), "exe 0.2.0");
        assert_eq!(read(&paths.launcher_cmd()), "@rem cmd 0.2.0");
        assert_eq!(read(&paths.root.join("README.txt")), "readme 0.2.0");
        let previous = paths.previous_dir();
        assert_eq!(read(&previous.join(APP_DIR).join(EXE_NAME)), "exe 0.1.0");
        assert_eq!(read(&previous.join(LAUNCHER_CMD)), "@rem cmd 0.1.0");
        assert!(!staged.join(APP_DIR).exists(), "moved, not copied");
        assert_eq!(
            read(&paths.root.join("game").join("options.txt")),
            "mine",
            "game is never touched"
        );

        roll_back(&paths).expect("rolled back");
        assert_eq!(read(&paths.exe()), "exe 0.1.0");
        assert_eq!(read(&paths.launcher_cmd()), "@rem cmd 0.1.0");
        assert_eq!(read(&paths.root.join("README.txt")), "readme 0.1.0");
        let failed = paths.failed_dir();
        assert_eq!(read(&failed.join(APP_DIR).join(EXE_NAME)), "exe 0.2.0");
        assert_eq!(read(&paths.root.join("game").join("options.txt")), "mine");
    }

    #[test]
    fn a_failed_swap_leaves_the_install_as_it_was() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        let staged = fake_staged(&paths, "0.2.0");
        // The old README is held open without delete sharing, so moving it aside fails after
        // `app\` and `Play Aero Client.cmd` were already swapped: both must be undone.
        let held = hold_open(&paths.root.join("README.txt"));
        let result = swap_in(&paths, &staged);
        drop(held);
        if cfg!(windows) {
            assert!(result.is_err(), "the held file blocks the swap");
            assert_eq!(read(&paths.exe()), "exe 0.1.0", "app undone");
            assert_eq!(read(&paths.launcher_cmd()), "@rem cmd 0.1.0", "cmd undone");
            assert_eq!(read(&paths.root.join("README.txt")), "readme 0.1.0");
            assert_eq!(
                read(&staged.join(APP_DIR).join(EXE_NAME)),
                "exe 0.2.0",
                "the staged app is back in staging"
            );
            // Nothing held: the same swap now works.
            swap_in(&paths, &staged).expect("swapped");
            assert_eq!(read(&paths.exe()), "exe 0.2.0");
        }
    }

    /// Hold `path` open so it can't be renamed (Windows: no delete sharing).
    fn hold_open(path: &Path) -> std::fs::File {
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .read(true)
                .share_mode(1) // FILE_SHARE_READ only
                .open(path)
                .unwrap()
        }
        #[cfg(not(windows))]
        {
            std::fs::File::open(path).unwrap()
        }
    }

    #[test]
    fn nothing_to_roll_back_to_is_an_error_that_changes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let paths = fake_install(dir.path(), "0.1.0");
        assert!(roll_back(&paths).is_err());
        assert_eq!(read(&paths.exe()), "exe 0.1.0");
        assert!(
            swap_in(&paths, &paths.staged_root("9.9.9")).is_err(),
            "nothing staged"
        );
        assert_eq!(read(&paths.exe()), "exe 0.1.0");
    }
}
