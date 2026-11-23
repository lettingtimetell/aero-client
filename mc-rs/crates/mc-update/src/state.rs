//! `update\state.json`: what the updater checked, offered, staged and tried, kept across
//! launches. Written atomically (a temporary file renamed over the old one); a file that cannot
//! be read is kept as `state.json.bad` and the updater starts over from an empty state.

use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::error::UpdateError;
use crate::github::ReleaseInfo;
use crate::install::InstallPaths;

/// The largest state file read.
const MAX_STATE_BYTES: u64 = 256 * 1024;

/// Every load-modify-save in this process goes through this lock (the title screen's thread and
/// the updater's thread both write). Between processes (the client and the update helper) the
/// writes are ordered by the protocol in [`crate::helper`].
static STATE_LOCK: Mutex<()> = Mutex::new(());

/// The updater's state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UpdateState {
    /// When GitHub was last asked (Unix seconds).
    pub last_check_unix: Option<u64>,
    /// The newest release the last check found (newer than the build that checked).
    pub known_release: Option<ReleaseInfo>,
    /// The version the player said "Later" to: not offered again.
    pub dismissed_version: Option<String>,
    /// A verified release extracted into the staging folder, waiting for a restart.
    pub staged: Option<StagedUpdate>,
    /// A version just swapped in that has not shown its title screen yet.
    pub trial: Option<Trial>,
    /// The last update that failed to apply or to start, for the title screen to say so once.
    pub last_failure: Option<Failure>,
    /// Versions that failed to start on this PC: never offered again.
    pub bad_versions: Vec<String>,
}

/// A staged release.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StagedUpdate {
    /// Its version.
    pub version: String,
    /// The SHA-256 of its `app\mc-client.exe`, checked again just before the swap.
    pub exe_sha256: String,
    /// The SHA-256 of the zip it came from.
    pub zip_sha256: String,
}

/// A version on trial after a swap.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trial {
    /// The version swapped in.
    pub version: String,
    /// The version it replaced (kept in `update\previous\`).
    pub previous_version: String,
    /// How many times the new version's client process started.
    pub attempts: u32,
    /// Whether it reached its title screen.
    pub healthy: bool,
    /// When the swap happened (Unix seconds).
    pub started_unix: u64,
}

/// An update that did not work out.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Failure {
    /// The version that failed.
    pub version: String,
    /// The version kept (running now).
    pub kept_version: String,
    /// What happened, for the log.
    pub reason: String,
    /// Whether the title screen has told the player.
    pub shown: bool,
}

impl UpdateState {
    /// The state in `path`: empty if there is none; a file that cannot be read is moved aside
    /// to `state.json.bad` (never silently deleted) and the state starts empty.
    #[must_use]
    pub fn load(path: &Path) -> UpdateState {
        let bytes = match crate::install::read_small(path, MAX_STATE_BYTES) {
            Ok(bytes) => bytes,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return UpdateState::default();
            }
            Err(error) => {
                tracing::warn!(%error, "the update state can't be read; starting over");
                set_aside(path);
                return UpdateState::default();
            }
        };
        match serde_json::from_slice(&bytes) {
            Ok(state) => state,
            Err(error) => {
                tracing::warn!(%error, "the update state is malformed; starting over");
                set_aside(path);
                UpdateState::default()
            }
        }
    }

    /// Write the state to `path` atomically: `state.json.tmp`, flushed, then renamed over.
    pub fn save(&self, path: &Path) -> Result<(), UpdateError> {
        use std::io::Write as _;
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| UpdateError::io("creating the update folder", error))?;
        }
        let json = serde_json::to_vec_pretty(self)
            .map_err(|error| UpdateError::Apply(format!("encoding the update state: {error}")))?;
        let temporary = path.with_extension("json.tmp");
        let mut file = std::fs::File::create(&temporary)
            .map_err(|error| UpdateError::io("writing the update state", error))?;
        file.write_all(&json)
            .and_then(|()| file.sync_all())
            .map_err(|error| UpdateError::io("writing the update state", error))?;
        drop(file);
        std::fs::rename(&temporary, path)
            .map_err(|error| UpdateError::io("replacing the update state", error))
    }

    /// Whether `version` may be offered: not dismissed, not known to fail on this PC.
    #[must_use]
    pub fn may_offer(&self, version: &str) -> bool {
        self.dismissed_version.as_deref() != Some(version)
            && !self.bad_versions.iter().any(|bad| bad == version)
    }
}

/// Move an unreadable state file aside, replacing an older `.bad` one.
fn set_aside(path: &Path) {
    let aside = path.with_extension("json.bad");
    let _ = std::fs::remove_file(&aside);
    let _ = std::fs::rename(path, &aside);
}

/// Load the state of `paths`, let `change` modify it, and save it (only if it changed), under
/// this process's state lock. Returns what `change` returned.
pub fn update<R>(
    paths: &InstallPaths,
    change: impl FnOnce(&mut UpdateState) -> R,
) -> Result<R, UpdateError> {
    let _guard = STATE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let file = paths.state_file();
    let before = UpdateState::load(&file);
    let mut state = before.clone();
    let result = change(&mut state);
    if state != before {
        state.save(&file)?;
    }
    Ok(result)
}

/// The state of `paths`, read under this process's state lock.
#[must_use]
pub fn read(paths: &InstallPaths) -> UpdateState {
    let _guard = STATE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    UpdateState::load(&paths.state_file())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_state_round_trips_and_saves_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("update").join("state.json");
        assert_eq!(UpdateState::load(&path), UpdateState::default(), "none yet");
        let state = UpdateState {
            last_check_unix: Some(1_800_000_000),
            dismissed_version: Some("0.2.0".to_owned()),
            bad_versions: vec!["0.1.5".to_owned()],
            staged: Some(StagedUpdate {
                version: "0.3.0".to_owned(),
                exe_sha256: "a".repeat(64),
                zip_sha256: "b".repeat(64),
            }),
            ..UpdateState::default()
        };
        state.save(&path).unwrap();
        assert_eq!(UpdateState::load(&path), state);
        assert!(
            !path.with_extension("json.tmp").exists(),
            "renamed into place"
        );
        assert!(!state.may_offer("0.2.0"), "dismissed");
        assert!(!state.may_offer("0.1.5"), "failed before");
        assert!(state.may_offer("0.3.0"));
    }

    #[test]
    fn a_corrupt_state_is_set_aside_not_deleted() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("state.json");
        std::fs::write(&path, b"{ not json").unwrap();
        assert_eq!(UpdateState::load(&path), UpdateState::default());
        assert!(!path.exists());
        assert_eq!(
            std::fs::read(path.with_extension("json.bad")).unwrap(),
            b"{ not json"
        );
        // Unknown fields from a newer build are ignored, known ones kept.
        std::fs::write(&path, br#"{"last_check_unix": 5, "from_the_future": true}"#).unwrap();
        assert_eq!(UpdateState::load(&path).last_check_unix, Some(5));
    }
}
