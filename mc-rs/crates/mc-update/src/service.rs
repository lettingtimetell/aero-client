//! The updater as the client runs it: one background thread that checks, downloads, verifies and
//! stages, and a [`Status`] the title screen reads.
//!
//! Nothing here touches the frame: the title screen compares [`Updater::generation`] (one atomic
//! load) and takes the status lock only when it changed; the download's progress is two relaxed
//! atomic loads while it is shown. With "Check for updates" off, or outside a portable install,
//! the client never creates an [`Updater`], so no thread exists and nothing is read or sent.
//!
//! The thread never panics on bad input (every failure is a [`UpdateError`] turned into a status
//! and a log line), because the client's panic hook would report any panic as a game crash.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use crate::error::UpdateError;
use crate::github::{self, ReleaseInfo};
use crate::helper;
use crate::install::{InstallPaths, APP_DIR, EXE_NAME};
use crate::net::{self, HttpPolicy, Progress};
use crate::stage::{self, Limits};
use crate::state::{self, StagedUpdate, UpdateState};
use crate::verify;
use crate::version::{self, Version};

/// What the title screen's update card shows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Status {
    /// Nothing.
    Idle,
    /// A newer release can be downloaded: "Aero Client 0.2.0 is available".
    Available {
        /// Its version.
        version: String,
        /// Its release page.
        notes_url: Option<String>,
    },
    /// Downloading ([`Updater::progress`]).
    Downloading {
        /// Its version.
        version: String,
    },
    /// Checking and unpacking the download.
    Installing {
        /// Its version.
        version: String,
    },
    /// Staged: "Restart to update".
    Ready {
        /// Its version.
        version: String,
        /// Its release page.
        notes_url: Option<String>,
    },
    /// The download or its checks failed.
    Failed {
        /// Its version.
        version: String,
        /// [`UpdateError::player_message`].
        message: &'static str,
    },
    /// The last restart did not apply an update, or rolled it back.
    RolledBack {
        /// The version that did not work.
        version: String,
        /// The version running now.
        kept: String,
        /// Whether it can be offered again: it was not applied for a
        /// passing reason (a file held open during the swap, staged files that changed) rather
        /// than marked as failing on this PC (it did not start, or failed its self-test).
        offered_again: bool,
    },
}

/// How the updater is set up.
#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// `owner/name`.
    pub repo: String,
    /// Which URLs may be fetched.
    pub policy: HttpPolicy,
    /// The Ed25519 key releases must be signed with (`None`: no signature needed).
    pub public_key: Option<[u8; 32]>,
    /// This build's version.
    pub current: Version,
    /// The least time between checks, across launches.
    pub check_interval: Duration,
    /// Install releases verified by their SHA-256 alone when there is no `public_key`. Never in
    /// the shipped configuration: both hashes come from the same GitHub release, so whoever can
    /// publish a release could ship code to every player. Without a key the shipped updater
    /// only tells the player about a new version and links its release page.
    pub allow_unsigned_install: bool,
}

impl ServiceConfig {
    /// The shipped configuration: [`crate::REPO`] on GitHub, the compiled-in key (without one,
    /// notify only). An unusable repository name or a malformed compiled-in key is an error, and
    /// the client then never starts the updater (it fails closed).
    pub fn shipped() -> Result<ServiceConfig, UpdateError> {
        if !github::valid_repo(crate::REPO) {
            return Err(UpdateError::BadRelease(
                "the build's repository name is not owner/name".to_owned(),
            ));
        }
        Ok(ServiceConfig {
            repo: crate::REPO.to_owned(),
            policy: HttpPolicy::github(),
            public_key: verify::compiled_public_key()?,
            current: version::current(),
            check_interval: crate::CHECK_INTERVAL,
            allow_unsigned_install: false,
        })
    }

    /// Whether this configuration downloads and installs updates (a key is compiled in), or
    /// only tells the player about them.
    #[must_use]
    pub fn can_install(&self) -> bool {
        self.public_key.is_some() || self.allow_unsigned_install
    }
}

struct Shared {
    generation: AtomicU64,
    status: Mutex<Status>,
    progress: Progress,
    cancel: AtomicBool,
    /// Commands the worker has finished (tests wait on it).
    processed: AtomicU64,
}

impl Shared {
    fn set(&self, status: Status) {
        let mut current = self
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if *current != status {
            *current = status;
            self.generation.fetch_add(1, Ordering::Release);
        }
    }

    fn get(&self) -> Status {
        self.status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

enum Command {
    Check,
    Download,
    Later,
    DismissFailure,
    Shutdown,
}

/// The client's handle on the updater thread. Dropping it stops the thread (a download in
/// progress is cancelled).
pub struct Updater {
    shared: Arc<Shared>,
    commands: mpsc::Sender<Command>,
    paths: InstallPaths,
    can_install: bool,
}

impl std::fmt::Debug for Updater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Updater")
            .field("status", &self.shared.get())
            .finish_non_exhaustive()
    }
}

impl Updater {
    /// Start the updater thread for the install at `paths`. It first shows what the saved state
    /// holds (a rolled-back update, a staged one, one found by an earlier check) and then waits
    /// for [`Updater::check`] or [`Updater::download`].
    pub fn start(paths: InstallPaths, config: ServiceConfig) -> Result<Updater, UpdateError> {
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(1),
            status: Mutex::new(Status::Idle),
            progress: Progress::default(),
            cancel: AtomicBool::new(false),
            processed: AtomicU64::new(0),
        });
        let (commands, receiver) = mpsc::channel();
        let can_install = config.can_install();
        let worker = Worker {
            paths: paths.clone(),
            config,
            shared: Arc::clone(&shared),
            runtime: None,
            http: None,
            checked: false,
        };
        std::thread::Builder::new()
            .name("aero-updater".to_owned())
            .spawn(move || worker.run(&receiver))
            .map_err(|error| UpdateError::io("starting the updater thread", error))?;
        Ok(Updater {
            shared,
            commands,
            paths,
            can_install,
        })
    }

    /// Whether Update can download and install (a key is compiled in), or the card only links
    /// the release page ([`ServiceConfig::can_install`]).
    #[must_use]
    pub fn can_install(&self) -> bool {
        self.can_install
    }

    /// Changes whenever [`Updater::status`] does (one atomic load).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.shared.generation.load(Ordering::Acquire)
    }

    /// What to show now.
    #[must_use]
    pub fn status(&self) -> Status {
        self.shared.get()
    }

    /// Commands the updater thread has finished (tests).
    #[cfg(test)]
    pub(crate) fn processed(&self) -> u64 {
        self.shared.processed.load(Ordering::Acquire)
    }

    /// `(bytes received, bytes expected)` of the download.
    #[must_use]
    pub fn progress(&self) -> (u64, u64) {
        (
            self.shared.progress.done.load(Ordering::Relaxed),
            self.shared.progress.total.load(Ordering::Relaxed),
        )
    }

    /// Ask GitHub, unless this launch already did or the last check is under the interval old.
    pub fn check(&self) {
        let _ = self.commands.send(Command::Check);
    }

    /// Download, verify and stage the available release (the player clicked Update).
    pub fn download(&self) {
        self.shared.cancel.store(false, Ordering::Relaxed);
        let _ = self.commands.send(Command::Download);
    }

    /// Stop the download in progress.
    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    /// The player said "Later": this version is not offered again.
    pub fn later(&self) {
        let _ = self.commands.send(Command::Later);
    }

    /// The player read the "couldn't update" message.
    pub fn dismiss_failure(&self) {
        let _ = self.commands.send(Command::DismissFailure);
    }

    /// "Restart to update": start the helper (a copy of this client) that applies the staged
    /// update once this process has exited. The caller quits the game right after `Ok`, and
    /// must only call this with no world loaded.
    pub fn restart_to_update(&self) -> Result<(), UpdateError> {
        let Status::Ready { version, .. } = self.status() else {
            return Err(UpdateError::Apply("no update is ready".to_owned()));
        };
        let staged = self
            .paths
            .staged_root(&version)
            .join(APP_DIR)
            .join(EXE_NAME);
        if !staged.is_file() {
            return Err(UpdateError::Apply("the staged update is gone".to_owned()));
        }
        helper::install_helper(&self.paths)?;
        helper::spawn_helper(&self.paths, helper::APPLY_FLAG, Some(&version))
            .map_err(|error| UpdateError::io("starting the update helper", error))
    }
}

impl Drop for Updater {
    fn drop(&mut self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
        let _ = self.commands.send(Command::Shutdown);
    }
}

struct Worker {
    paths: InstallPaths,
    config: ServiceConfig,
    shared: Arc<Shared>,
    runtime: Option<tokio::runtime::Runtime>,
    http: Option<reqwest::Client>,
    checked: bool,
}

/// A release read back from `state.json` is checked again before it is used (the file is in the
/// install folder, and anyone can edit it): the zip's name must be the one its version gives
/// (it becomes a file name), and its size within the cap. URLs are checked again when fetched.
fn saved_release_ok(release: &ReleaseInfo) -> bool {
    let name = version::parse_release_tag(&release.version).map(|v| github::asset_name(&v));
    name.as_deref() == Some(release.zip.name.as_str())
        && (1..=github::MAX_ZIP_BYTES).contains(&release.zip.size)
}

/// The worker's current-thread runtime, built on first use (a field-level borrow, so the rest of
/// the worker stays usable while it runs a request).
fn ensure_runtime(
    slot: &mut Option<tokio::runtime::Runtime>,
) -> Result<&tokio::runtime::Runtime, UpdateError> {
    if slot.is_none() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| UpdateError::io("starting the updater's runtime", error))?;
        *slot = Some(runtime);
    }
    slot.as_ref()
        .ok_or_else(|| UpdateError::Apply("no runtime".to_owned()))
}

impl Worker {
    fn run(mut self, commands: &mpsc::Receiver<Command>) {
        self.show_saved();
        self.shared.processed.fetch_add(1, Ordering::Release);
        while let Ok(command) = commands.recv() {
            match command {
                Command::Check => self.check(),
                Command::Download => self.download(),
                Command::Later => self.later(),
                Command::DismissFailure => self.dismiss_failure(),
                Command::Shutdown => break,
            }
            // Commands are rare (a check per launch, a click): between them the thread holds no
            // runtime, connection pool or TLS state, only its stack. Rebuilt in milliseconds.
            self.http = None;
            self.runtime = None;
            self.shared.processed.fetch_add(1, Ordering::Release);
        }
    }

    fn newer(&self, version: &str) -> bool {
        version::parse_release_tag(version)
            .is_some_and(|version| version::is_newer(&version, &self.config.current))
    }

    /// The saved state, before any network: a failed update to report once, else an offer.
    fn show_saved(&mut self) {
        let failure = state::update(&self.paths, |state| {
            let failure = state
                .last_failure
                .as_mut()
                .filter(|failure| !failure.shown)?;
            failure.shown = true;
            let bad = state.bad_versions.contains(&failure.version);
            Some((failure.clone(), !bad))
        });
        if let Ok(Some((failure, offered_again))) = failure {
            self.shared.set(Status::RolledBack {
                version: failure.version,
                kept: failure.kept_version,
                offered_again,
            });
            return;
        }
        self.show_offer(&state::read(&self.paths));
    }

    /// The staged update ("Restart to update"), else the release found earlier, else nothing.
    fn show_offer(&self, state: &UpdateState) {
        let notes = |version: &str| {
            state
                .known_release
                .as_ref()
                .filter(|release| release.version == version)
                .and_then(|release| release.notes_url.clone())
        };
        if let Some(staged) = &state.staged {
            let exe = self
                .paths
                .staged_root(&staged.version)
                .join(APP_DIR)
                .join(EXE_NAME);
            if self.newer(&staged.version) && state.may_offer(&staged.version) && exe.is_file() {
                self.shared.set(Status::Ready {
                    version: staged.version.clone(),
                    notes_url: notes(&staged.version),
                });
                return;
            }
        }
        match &state.known_release {
            Some(release) if self.newer(&release.version) && state.may_offer(&release.version) => {
                self.shared.set(Status::Available {
                    version: release.version.clone(),
                    notes_url: release.notes_url.clone(),
                });
            }
            _ => self.shared.set(Status::Idle),
        }
    }

    fn http(&mut self) -> Result<reqwest::Client, UpdateError> {
        if self.http.is_none() {
            self.http = Some(net::client(&self.config.policy)?);
        }
        self.http
            .clone()
            .ok_or_else(|| UpdateError::Apply("no HTTP client".to_owned()))
    }

    fn check(&mut self) {
        if self.checked {
            return;
        }
        self.checked = true;
        let now = crate::now_unix();
        let last = state::read(&self.paths).last_check_unix;
        let interval = self.config.check_interval.as_secs();
        if last.is_some_and(|last| last <= now && now - last < interval) {
            return;
        }
        match self.fetch_release() {
            Ok(found) => {
                let saved = state::update(&self.paths, |state| {
                    state.last_check_unix = Some(now);
                    state.known_release = found;
                    state.clone()
                });
                let showing_offer =
                    matches!(self.shared.get(), Status::Idle | Status::Available { .. });
                if let (Ok(state), true) = (saved, showing_offer) {
                    self.show_offer(&state);
                }
            }
            Err(UpdateError::RateLimited) => {
                tracing::info!("update check rate limited; next try after the interval");
                let _ = state::update(&self.paths, |state| state.last_check_unix = Some(now));
            }
            Err(error) => tracing::info!(%error, "update check failed"),
        }
    }

    fn fetch_release(&mut self) -> Result<Option<ReleaseInfo>, UpdateError> {
        let http = self.http()?;
        let policy = self.config.policy.clone();
        let repo = self.config.repo.clone();
        let json = ensure_runtime(&mut self.runtime)?
            .block_on(net::fetch_latest_release(&http, &policy, &repo))?;
        let Some(json) = json else {
            return Ok(None);
        };
        let release = github::parse_release(&json)?;
        github::choose_release(
            &release,
            &self.config.current,
            &policy,
            self.config.public_key.is_some(),
        )
    }

    fn download(&mut self) {
        if !self.config.can_install() {
            tracing::warn!("no update signing key in this build: updates are announced only");
            return;
        }
        let state = state::read(&self.paths);
        let Some(release) = state
            .known_release
            .clone()
            .filter(|release| self.newer(&release.version) && saved_release_ok(release))
        else {
            return;
        };
        let version = release.version.clone();
        self.shared.set(Status::Downloading {
            version: version.clone(),
        });
        match self.download_and_stage(&release) {
            Ok(staged) => {
                let _ = state::update(&self.paths, |state| state.staged = Some(staged));
                tracing::info!(version = %version, "update ready");
                self.shared.set(Status::Ready {
                    version,
                    notes_url: release.notes_url.clone(),
                });
            }
            Err(error) => {
                let _ = std::fs::remove_dir_all(self.paths.download_dir());
                let _ = std::fs::remove_dir_all(self.paths.staging_root());
                if matches!(error, UpdateError::Cancelled) {
                    self.shared.set(Status::Available {
                        version,
                        notes_url: release.notes_url.clone(),
                    });
                } else {
                    tracing::warn!(%error, version = %version, "update download failed");
                    self.shared.set(Status::Failed {
                        version,
                        message: error.player_message(),
                    });
                }
            }
        }
    }

    fn download_and_stage(&mut self, release: &ReleaseInfo) -> Result<StagedUpdate, UpdateError> {
        let http = self.http()?;
        let policy = self.config.policy.clone();
        let key = self.config.public_key;
        let shared = Arc::clone(&self.shared);
        let download_dir = self.paths.download_dir();
        let _ = std::fs::remove_dir_all(&download_dir);
        std::fs::create_dir_all(&download_dir)
            .map_err(|error| UpdateError::io("creating the download folder", error))?;
        let part = download_dir.join(format!("{}.part", release.zip.name));
        let runtime = ensure_runtime(&mut self.runtime)?;
        let sha256 = runtime.block_on(net::download(
            &http,
            &policy,
            &release.zip.url,
            release.zip.size,
            &part,
            &shared.progress,
            &shared.cancel,
        ))?;
        let expected = match (&release.sha256, &release.sha256_asset) {
            (Some(hash), _) => hash.clone(),
            (None, Some(asset)) => {
                let bytes = runtime.block_on(net::fetch_small(&http, &policy, &asset.url))?;
                verify::parse_sha256_sidecar(&bytes, &release.zip.name)?
            }
            (None, None) => {
                return Err(UpdateError::BadRelease("no published SHA-256".to_owned()));
            }
        };
        if sha256 != expected {
            return Err(UpdateError::ChecksumMismatch);
        }
        if let Some(key) = key {
            let asset = release
                .signature_asset
                .as_ref()
                .ok_or_else(|| UpdateError::Signature("the .sig asset is missing".to_owned()))?;
            let bytes = runtime.block_on(net::fetch_small(&http, &policy, &asset.url))?;
            let signature = verify::parse_signature(&bytes)?;
            let message = verify::signed_message(&release.zip.name, &release.version, &sha256);
            verify::verify_signature(&key, message.as_bytes(), &signature)?;
        }
        shared.set(Status::Installing {
            version: release.version.clone(),
        });
        let zip = download_dir.join(&release.zip.name);
        std::fs::rename(&part, &zip)
            .map_err(|error| UpdateError::io("keeping the verified download", error))?;
        let _ = std::fs::remove_dir_all(self.paths.staging_root());
        let dest = self.paths.staging_dir(&release.version);
        let staged = stage::stage_release(&zip, &dest, &release.version, &Limits::default())?;
        let _ = std::fs::remove_dir_all(&download_dir);
        tracing::info!(
            version = %release.version,
            files = staged.files,
            bytes = staged.bytes,
            "update staged"
        );
        Ok(StagedUpdate {
            version: release.version.clone(),
            exe_sha256: staged.exe_sha256,
            zip_sha256: sha256,
        })
    }

    fn later(&mut self) {
        let version = match self.shared.get() {
            Status::Available { version, .. }
            | Status::Ready { version, .. }
            | Status::Failed { version, .. } => version,
            _ => return,
        };
        let _ = state::update(&self.paths, |state| {
            state.dismissed_version = Some(version);
        });
        self.shared.set(Status::Idle);
    }

    fn dismiss_failure(&mut self) {
        let _ = state::update(&self.paths, |state| {
            if let Some(failure) = &mut state.last_failure {
                failure.shown = true;
            }
        });
        self.show_offer(&state::read(&self.paths));
    }
}
