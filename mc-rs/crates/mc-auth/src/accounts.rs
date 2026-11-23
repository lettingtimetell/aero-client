//! The client's own account list: the Microsoft accounts added with the in-client sign-in, and
//! which account is in use (one of those, or the session a launcher passed on the command line).
//!
//! # Where and how it is stored
//!
//! One file, [`ACCOUNTS_FILE_NAME`], in the per-user config directory
//! ([`crate::config::config_dir`]: `%APPDATA%\mc-rs\config` on Windows, `~/.config/mc-rs` on
//! Linux; overridable with `MC_RS_CONFIG_DIR`). Never in the game folder: players zip and share
//! that folder.
//!
//! Each account keeps its public details in plain JSON (UUID, name, skin URL and model, when it
//! was added and last used, when its Minecraft token expires) so the list can be shown, and a
//! sign-in that cannot be opened can still be named and removed. Its secrets (the Microsoft
//! refresh token and the Minecraft access token) are one [`AccountSecrets`] value sealed by
//! [`crate::seal`]: Windows DPAPI for the current user on Windows, the AES key-file scheme
//! elsewhere. A sealed value that will not open (the file was copied from another PC or another
//! Windows user, or damaged) makes only that account need a new sign-in; the others are kept.
//!
//! Every change rewrites the whole file atomically (a uniquely named temporary file in the same
//! folder, flushed to disk, then renamed over the old one), so a crash or a full disk never
//! leaves a half-written list. Two clients writing at the same moment: the last write wins.
//! Removing an account removes its sealed secrets from the file at once.
//!
//! A file that is not valid JSON is never overwritten blind: the next change first renames it to
//! `accounts.json.unreadable-<unix time>` and then starts a new list.
//!
//! # Secrets
//!
//! Only the account in use is ever unsealed ([`AccountStore::unseal`]): on start and on a
//! switch. Unsealed tokens are registered with the process-wide secret scrubber, and no token or
//! sealed value is ever logged, shown, or put in an error message; `Debug` redacts them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::client::AuthClient;
use crate::config;
use crate::error::AuthError;
use crate::flow::REFRESH_SKEW;
use crate::seal::Sealer;
use crate::session::{unix_now, Authenticated, Redacted, Session, SkinVariant};

/// The account list's file name inside the config directory.
pub const ACCOUNTS_FILE_NAME: &str = "accounts.json";
/// The key file of the key-file sealing scheme (platforms without DPAPI only).
pub const ACCOUNTS_KEY_FILE_NAME: &str = "accounts.key";

/// The file format this code writes.
const FORMAT_VERSION: u32 = 1;
/// DPAPI's optional entropy for this store: fixed and not secret; it names the purpose, so a
/// blob sealed for something else is not opened as an account's secrets.
const ENTROPY: &[u8] = b"mc-auth account store, format 1";

/// Which account the client uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ActiveAccount {
    /// The session a launcher passed on the command line (`--accessToken`).
    Launcher,
    /// An account added in the client.
    Stored {
        /// The account's Minecraft UUID.
        uuid: Uuid,
    },
}

/// One account added with the in-client Microsoft sign-in.
#[derive(Clone, Serialize, Deserialize)]
pub struct StoredAccount {
    /// The Minecraft UUID: the account's identity in the list.
    pub uuid: Uuid,
    /// The Minecraft name at the last sign-in or refresh.
    pub username: String,
    /// The skin worn at the last sign-in or refresh (a `textures.minecraft.net` URL).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skin_url: Option<String>,
    /// Whether that skin uses the slim (3-pixel arm) model.
    #[serde(default)]
    pub slim: bool,
    /// Unix seconds when the account was first added.
    #[serde(default)]
    pub added_at: u64,
    /// Unix seconds when the account was last chosen; `0` if never.
    #[serde(default)]
    pub last_used_at: u64,
    /// Unix seconds when the stored Minecraft token stops working. Not a secret: it lets the
    /// client tell, without unsealing anything, whether a join has to refresh first.
    #[serde(default)]
    pub expires_at: u64,
    /// The sealed [`AccountSecrets`], base64. Never shown or logged.
    sealed: String,
}

impl StoredAccount {
    /// Whether a join at `now` (unix seconds) must refresh the Minecraft token first: it expires
    /// within [`REFRESH_SKEW`].
    #[must_use]
    pub fn needs_refresh(&self, now: u64) -> bool {
        self.expires_at <= now.saturating_add(REFRESH_SKEW.as_secs())
    }
}

impl std::fmt::Debug for StoredAccount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StoredAccount")
            .field("uuid", &self.uuid)
            .field("username", &self.username)
            .field("skin_url", &self.skin_url)
            .field("slim", &self.slim)
            .field("added_at", &self.added_at)
            .field("last_used_at", &self.last_used_at)
            .field("expires_at", &self.expires_at)
            .field("sealed", &Redacted)
            .finish()
    }
}

/// An account's secrets, sealed together. Both tokens are redacted in `Debug`.
#[derive(Clone, Serialize, Deserialize)]
pub struct AccountSecrets {
    /// The Microsoft refresh token: silent re-sign-in. **Secret.**
    pub refresh_token: String,
    /// The Minecraft services access token used to join servers. **Secret.**
    pub access_token: String,
    /// Unix seconds when `access_token` stops working.
    pub expires_at: u64,
}

impl std::fmt::Debug for AccountSecrets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AccountSecrets")
            .field("refresh_token", &Redacted)
            .field("access_token", &Redacted)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// The list as read from disk: public details only, nothing unsealed.
#[derive(Debug, Clone, Default)]
pub struct AccountList {
    /// The account chosen last time, if any.
    pub active: Option<ActiveAccount>,
    /// The added accounts, in the order they were first added.
    pub accounts: Vec<StoredAccount>,
}

impl AccountList {
    /// The added account with this UUID.
    #[must_use]
    pub fn get(&self, uuid: Uuid) -> Option<&StoredAccount> {
        self.accounts.iter().find(|account| account.uuid == uuid)
    }
}

/// The JSON file.
#[derive(Serialize, Deserialize)]
struct FileFormat {
    version: u32,
    #[serde(default)]
    active: Option<ActiveAccount>,
    #[serde(default)]
    accounts: Vec<StoredAccount>,
}

/// A session ready for a server join, from [`AccountStore::session_for_join`].
#[derive(Debug, Clone)]
pub struct JoinSession {
    /// What the network layer needs.
    pub session: Session,
    /// The account's stored details after the call (updated when it refreshed).
    pub account: StoredAccount,
    /// Whether the Minecraft token was refreshed on the way.
    pub refreshed: bool,
}

/// The account list on disk. See the module documentation.
#[derive(Debug, Clone)]
pub struct AccountStore {
    dir: PathBuf,
    sealer: Sealer,
}

impl AccountStore {
    /// The store in the per-user config directory.
    pub fn discover() -> Result<AccountStore, AuthError> {
        Ok(AccountStore::at(config::config_dir()?))
    }

    /// A store in `dir`, sealed with the platform's scheme (DPAPI on Windows).
    #[must_use]
    pub fn at(dir: impl Into<PathBuf>) -> AccountStore {
        let dir = dir.into();
        let sealer = Sealer::platform(dir.join(ACCOUNTS_KEY_FILE_NAME), ENTROPY);
        AccountStore { dir, sealer }
    }

    /// A store in `dir` sealed with the key-file scheme on every platform (the format platforms
    /// without DPAPI use).
    #[must_use]
    pub fn at_with_key_file(dir: impl Into<PathBuf>) -> AccountStore {
        let dir = dir.into();
        let sealer = Sealer::key_file(dir.join(ACCOUNTS_KEY_FILE_NAME));
        AccountStore { dir, sealer }
    }

    /// The directory the store lives in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The list file.
    #[must_use]
    pub fn file_path(&self) -> PathBuf {
        self.dir.join(ACCOUNTS_FILE_NAME)
    }

    /// Read the list. A missing file is an empty list; nothing is unsealed.
    ///
    /// # Errors
    ///
    /// [`AuthError::Io`] if the file cannot be read, [`AuthError::Config`] if it is not a list
    /// this code understands (the next change backs it up and starts a new one).
    pub fn load(&self) -> Result<AccountList, AuthError> {
        let path = self.file_path();
        let text = match std::fs::read(&path) {
            Ok(text) => text,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(AccountList::default());
            }
            Err(source) => return Err(AuthError::Io { path, source }),
        };
        let file: FileFormat =
            serde_json::from_slice(&text).map_err(|source| AuthError::Config { path, source })?;
        Ok(AccountList {
            active: file.active,
            accounts: file.accounts,
        })
    }

    /// Read, change and write the list back atomically. An unreadable file is renamed aside
    /// first (never overwritten blind).
    fn modify<T>(&self, change: impl FnOnce(&mut AccountList) -> T) -> Result<T, AuthError> {
        let mut list = match self.load() {
            Ok(list) => list,
            Err(AuthError::Config { path, .. }) => {
                let backup =
                    path.with_file_name(format!("{ACCOUNTS_FILE_NAME}.unreadable-{}", unix_now()));
                std::fs::rename(&path, &backup).map_err(|source| AuthError::Io {
                    path: path.clone(),
                    source,
                })?;
                // The file name only: the folder is under the user's own profile path.
                tracing::warn!(
                    backup = %backup.file_name().unwrap_or_default().to_string_lossy(),
                    "the account list was unreadable; it was kept aside and a new one started"
                );
                AccountList::default()
            }
            Err(error) => return Err(error),
        };
        let result = change(&mut list);
        self.save(&list)?;
        Ok(result)
    }

    fn save(&self, list: &AccountList) -> Result<(), AuthError> {
        std::fs::create_dir_all(&self.dir).map_err(|source| AuthError::Io {
            path: self.dir.clone(),
            source,
        })?;
        let file = FileFormat {
            version: FORMAT_VERSION,
            active: list.active,
            accounts: list.accounts.clone(),
        };
        let path = self.file_path();
        let text = serde_json::to_vec_pretty(&file).map_err(|source| AuthError::Config {
            path: path.clone(),
            source,
        })?;
        write_atomically(&path, &text).map_err(|source| AuthError::Io { path, source })
    }

    /// Open an account's secrets. Only for the account in use (on start, on a switch, before a
    /// join).
    ///
    /// # Errors
    ///
    /// [`AuthError::CacheCorrupt`] when the sealed value will not open here (another PC or
    /// Windows user, or damaged): that account needs a new sign-in.
    pub fn unseal(&self, account: &StoredAccount) -> Result<AccountSecrets, AuthError> {
        let sealed = base64::engine::general_purpose::STANDARD
            .decode(account.sealed.as_bytes())
            .map_err(|_| AuthError::CacheCorrupt)?;
        let plain = self
            .sealer
            .unseal(&sealed)
            .map_err(|_| AuthError::CacheCorrupt)?;
        let secrets: AccountSecrets =
            serde_json::from_slice(&plain).map_err(|_| AuthError::CacheCorrupt)?;
        register_secrets(&secrets);
        Ok(secrets)
    }

    /// The account's session from its stored token, without refreshing (see
    /// [`AccountStore::session_for_join`] for joins).
    ///
    /// # Errors
    ///
    /// As [`AccountStore::unseal`].
    pub fn session(&self, account: &StoredAccount) -> Result<Session, AuthError> {
        let secrets = self.unseal(account)?;
        Ok(Session::new(
            account.uuid,
            account.username.clone(),
            secrets.access_token,
        ))
    }

    fn seal(&self, secrets: &AccountSecrets) -> Result<String, AuthError> {
        let plain = serde_json::to_vec(secrets).map_err(|source| AuthError::Config {
            path: self.file_path(),
            source,
        })?;
        let sealed = self.sealer.seal(&plain).map_err(|source| AuthError::Io {
            path: self.file_path(),
            source,
        })?;
        Ok(base64::engine::general_purpose::STANDARD.encode(sealed))
    }

    /// Add a completed sign-in, or update the account with the same UUID in place (signing in
    /// again with the same account, or a refresh: the rotated refresh token replaces the old one).
    ///
    /// # Errors
    ///
    /// [`AuthError::Io`] when sealing or writing failed; nothing changed on disk then.
    pub fn add(&self, auth: &Authenticated) -> Result<StoredAccount, AuthError> {
        let secrets = AccountSecrets {
            refresh_token: auth.credentials.msa_refresh_token.clone(),
            access_token: auth.credentials.mc_access_token.clone(),
            expires_at: auth.credentials.expires_at,
        };
        register_secrets(&secrets);
        let sealed = self.seal(&secrets)?;
        let skin = auth.profile.active_skin();
        let now = unix_now();
        self.modify(|list| {
            let existing = list
                .accounts
                .iter()
                .position(|account| account.uuid == auth.profile.uuid);
            let entry = StoredAccount {
                uuid: auth.profile.uuid,
                username: auth.profile.username.clone(),
                skin_url: skin.map(|skin| skin.url.clone()),
                slim: skin.is_some_and(|skin| skin.variant == SkinVariant::Slim),
                added_at: existing.map_or(now, |i| list.accounts[i].added_at),
                last_used_at: existing.map_or(0, |i| list.accounts[i].last_used_at),
                expires_at: secrets.expires_at,
                sealed,
            };
            match existing {
                Some(i) => list.accounts[i] = entry.clone(),
                None => list.accounts.push(entry.clone()),
            }
            entry
        })
    }

    /// Remove an account and its sealed secrets. Returns whether it was there. If it was the
    /// account in use, no account is in use afterwards.
    ///
    /// # Errors
    ///
    /// [`AuthError::Io`] when the list could not be written; the account is still there then.
    pub fn remove(&self, uuid: Uuid) -> Result<bool, AuthError> {
        self.modify(|list| {
            let before = list.accounts.len();
            list.accounts.retain(|account| account.uuid != uuid);
            if list.active == Some(ActiveAccount::Stored { uuid }) {
                list.active = None;
            }
            list.accounts.len() != before
        })
    }

    /// Remember the account in use (and when an added account was last used).
    ///
    /// # Errors
    ///
    /// [`AuthError::Io`] when the list could not be written.
    pub fn set_active(&self, active: Option<ActiveAccount>) -> Result<(), AuthError> {
        let now = unix_now();
        self.modify(|list| {
            if let Some(ActiveAccount::Stored { uuid }) = active {
                if let Some(account) = list.accounts.iter_mut().find(|a| a.uuid == uuid) {
                    account.last_used_at = now;
                }
            }
            list.active = active;
        })
    }

    /// The session for joining a server with an added account, refreshed first when its
    /// Minecraft token expires within [`REFRESH_SKEW`]. A refresh stores the rotated refresh
    /// token and the new access token **before** the session is handed out, so a crash right
    /// after cannot leave a used-up refresh token on disk.
    ///
    /// # Errors
    ///
    /// [`AuthError::CacheCorrupt`] (cannot be opened here), [`AuthError::TokenRejected`] (no
    /// refresh token, or the account was removed), an [`AuthError::Oauth`] such as
    /// `invalid_grant` (expired or revoked: sign in again), [`AuthError::MissingClientId`] (this
    /// build cannot refresh), or a network error. [`AuthError::needs_new_sign_in`] tells which
    /// ones a new sign-in fixes.
    pub async fn session_for_join(
        &self,
        client: &AuthClient,
        uuid: Uuid,
    ) -> Result<JoinSession, AuthError> {
        let list = self.load()?;
        let account = list.get(uuid).cloned().ok_or(AuthError::TokenRejected)?;
        let secrets = self.unseal(&account)?;
        let now = unix_now();
        if secrets.expires_at > now.saturating_add(REFRESH_SKEW.as_secs()) {
            let session =
                Session::new(account.uuid, account.username.clone(), secrets.access_token);
            return Ok(JoinSession {
                session,
                account,
                refreshed: false,
            });
        }
        if secrets.refresh_token.is_empty() {
            return Err(AuthError::TokenRejected);
        }
        let auth = client.refresh(&secrets.refresh_token).await?;
        let account = self.add(&auth)?;
        Ok(JoinSession {
            session: auth.session,
            account,
            refreshed: true,
        })
    }
}

/// `error` for a log line, without the absolute path an [`AuthError::Io`] or
/// [`AuthError::Config`] carries (it is under the user's own profile folder, so it names the
/// user's account on the PC).
#[must_use]
pub fn loggable(error: &AuthError) -> std::borrow::Cow<'_, str> {
    match error {
        AuthError::Io { path, source } => format!(
            "i/o error on {}: {source}",
            path.file_name().unwrap_or_default().to_string_lossy()
        )
        .into(),
        AuthError::Config { path, source } => format!(
            "{} is not valid JSON: {source}",
            path.file_name().unwrap_or_default().to_string_lossy()
        )
        .into(),
        other => other.to_string().into(),
    }
}

/// Register both tokens with the process-wide secret scrubber (`mc_core::secrets`) so logs, crash
/// reports and panic output never show them.
fn register_secrets(secrets: &AccountSecrets) {
    mc_core::secrets::register(&secrets.refresh_token);
    mc_core::secrets::register(&secrets.access_token);
}

/// Write `bytes` to `path` atomically: a uniquely named temporary file in the same folder
/// (process id and a counter, created new, owner-only on Unix), flushed to disk, then renamed
/// over `path`. On failure the temporary file is removed and `path` is untouched.
pub(crate) fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .map_or_else(|| "file".into(), |name| name.to_string_lossy());
    let temp = dir.join(format!(
        ".{name}.{}-{}.tmp",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        rename_over(&temp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

/// `rename(from, to)`, retried for up to a second while Windows reports
/// the target as in use: a virus scanner (or the other client) reading the list it just saw
/// written holds it without delete sharing for a moment, and the rename over it fails with
/// "access denied". Without this, that moment cost a refresh its rotated tokens (the join then
/// said to check the connection) or a sign-in its account ("couldn't save").
fn rename_over(from: &Path, to: &Path) -> std::io::Result<()> {
    const PATIENCE: std::time::Duration = std::time::Duration::from_secs(1);
    const STEP: std::time::Duration = std::time::Duration::from_millis(50);
    let started = std::time::Instant::now();
    loop {
        match std::fs::rename(from, to) {
            Err(error) if held_by_another(&error) && started.elapsed() < PATIENCE => {
                std::thread::sleep(STEP);
            }
            result => return result,
        }
    }
}

/// Access denied (5) or a sharing violation (32): another process has the file open.
fn held_by_another(error: &std::io::Error) -> bool {
    cfg!(windows)
        && (error.kind() == std::io::ErrorKind::PermissionDenied
            || matches!(error.raw_os_error(), Some(5 | 32)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Credentials, Profile, Skin};

    const REFRESH: &str = "fake-refresh-token-for-tests-0123456789";
    const ACCESS: &str = "fake-access-token-for-tests-0123456789";

    fn uuid(n: u8) -> Uuid {
        Uuid::from_bytes([n; 16])
    }

    fn signed_in(n: u8, name: &str, expires_at: u64) -> Authenticated {
        let profile = Profile {
            uuid: uuid(n),
            username: name.to_owned(),
            skins: vec![Skin {
                id: "s".to_owned(),
                state: "ACTIVE".to_owned(),
                url: format!("http://textures.minecraft.net/texture/{n}"),
                variant: SkinVariant::Slim,
                alias: None,
            }],
            capes: Vec::new(),
        };
        Authenticated {
            session: Session::new(uuid(n), name, format!("{ACCESS}-{n}")),
            credentials: Credentials {
                msa_refresh_token: format!("{REFRESH}-{n}"),
                mc_access_token: format!("{ACCESS}-{n}"),
                expires_at,
                uuid: uuid(n),
                username: name.to_owned(),
            },
            profile,
        }
    }

    fn stores(dir: &Path) -> [AccountStore; 2] {
        [
            AccountStore::at(dir.join("platform")),
            AccountStore::at_with_key_file(dir.join("key-file")),
        ]
    }

    #[test]
    fn a_fresh_directory_has_no_accounts_and_nothing_is_written_by_reading() {
        let dir = tempfile::tempdir().expect("tempdir");
        for store in stores(dir.path()) {
            let list = store.load().expect("load");
            assert!(list.accounts.is_empty());
            assert_eq!(list.active, None);
            assert!(!store.file_path().exists());
        }
    }

    #[test]
    fn added_accounts_round_trip_and_their_tokens_are_sealed() {
        let dir = tempfile::tempdir().expect("tempdir");
        for store in stores(dir.path()) {
            let added = store
                .add(&signed_in(1, "Alpha", 2_000_000_000))
                .expect("add");
            assert_eq!(added.username, "Alpha");
            assert!(added.slim);
            assert_eq!(
                added.skin_url.as_deref(),
                Some("http://textures.minecraft.net/texture/1")
            );
            store
                .add(&signed_in(2, "Beta", 2_000_000_000))
                .expect("add");

            let list = store.load().expect("load");
            let names: Vec<_> = list.accounts.iter().map(|a| a.username.as_str()).collect();
            assert_eq!(names, ["Alpha", "Beta"], "listed in the order added");

            let secrets = store
                .unseal(list.get(uuid(2)).expect("beta"))
                .expect("unseal");
            assert_eq!(secrets.refresh_token, format!("{REFRESH}-2"));
            assert_eq!(secrets.access_token, format!("{ACCESS}-2"));
            assert_eq!(secrets.expires_at, 2_000_000_000);

            // On disk the names are readable and the tokens are not.
            let raw = std::fs::read_to_string(store.file_path()).expect("read");
            assert!(raw.contains("Alpha") && raw.contains("Beta"));
            assert!(!raw.contains(REFRESH), "a refresh token is in the file");
            assert!(!raw.contains(ACCESS), "an access token is in the file");
        }
    }

    #[test]
    fn the_platform_store_keeps_no_key_file_on_windows() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        store.add(&signed_in(1, "Alpha", 1)).expect("add");
        let key_exists = dir.path().join(ACCOUNTS_KEY_FILE_NAME).exists();
        assert_eq!(key_exists, !cfg!(windows));
    }

    #[test]
    fn signing_in_again_updates_the_account_in_place() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        let first = store.add(&signed_in(1, "Alpha", 10)).expect("add");
        store
            .set_active(Some(ActiveAccount::Stored { uuid: uuid(1) }))
            .expect("active");
        let mut again = signed_in(1, "AlphaRenamed", 20);
        again.credentials.msa_refresh_token = "fake-rotated-refresh-token-0123456789".into();
        store.add(&again).expect("add again");
        let list = store.load().expect("load");
        assert_eq!(list.accounts.len(), 1);
        let account = &list.accounts[0];
        assert_eq!(account.username, "AlphaRenamed");
        assert_eq!(account.added_at, first.added_at, "first added time kept");
        assert!(account.last_used_at > 0, "last-used time kept");
        assert_eq!(account.expires_at, 20);
        let secrets = store.unseal(account).expect("unseal");
        assert_eq!(
            secrets.refresh_token,
            "fake-rotated-refresh-token-0123456789"
        );
    }

    #[test]
    fn the_choice_persists_and_removing_deletes_the_sealed_tokens() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        store.add(&signed_in(1, "Alpha", 10)).expect("add");
        store.add(&signed_in(2, "Beta", 10)).expect("add");
        store
            .set_active(Some(ActiveAccount::Stored { uuid: uuid(2) }))
            .expect("active");
        // A new store over the same folder (the next start) sees the same choice.
        let reopened = AccountStore::at(dir.path());
        assert_eq!(
            reopened.load().expect("load").active,
            Some(ActiveAccount::Stored { uuid: uuid(2) })
        );
        let sealed_beta = reopened.load().expect("load").accounts[1].sealed.clone();

        assert!(reopened.remove(uuid(2)).expect("remove"));
        assert!(!reopened.remove(uuid(2)).expect("remove twice"));
        let list = reopened.load().expect("load");
        assert_eq!(list.active, None, "the removed account is no longer in use");
        assert_eq!(list.accounts.len(), 1);
        let raw = std::fs::read_to_string(reopened.file_path()).expect("read");
        assert!(
            !raw.contains(&sealed_beta),
            "the sealed tokens left the file"
        );
        assert!(!raw.contains("Beta"));

        reopened
            .set_active(Some(ActiveAccount::Launcher))
            .expect("launcher");
        assert_eq!(
            reopened.load().expect("load").active,
            Some(ActiveAccount::Launcher)
        );
    }

    #[test]
    fn one_account_that_will_not_open_leaves_the_others_alone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        store.add(&signed_in(1, "Alpha", 10)).expect("add");
        store.add(&signed_in(2, "Beta", 10)).expect("add");
        // Damage Alpha's sealed value, as a copy from another PC would be to DPAPI.
        let path = store.file_path();
        let mut json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read")).expect("json");
        json["accounts"][0]["sealed"] = serde_json::Value::String("AAAA".into());
        std::fs::write(&path, serde_json::to_vec(&json).expect("json")).expect("write");

        let list = store.load().expect("load");
        assert_eq!(list.accounts.len(), 2, "still listed, so it can be removed");
        let error = store.unseal(&list.accounts[0]).expect_err("damaged");
        assert!(matches!(error, AuthError::CacheCorrupt));
        assert!(error.needs_new_sign_in());
        assert!(store.unseal(&list.accounts[1]).is_ok(), "Beta still opens");
        // Signing Alpha in again repairs it.
        store.add(&signed_in(1, "Alpha", 10)).expect("add");
        assert!(store
            .unseal(&store.load().expect("load").accounts[0])
            .is_ok());
    }

    #[test]
    fn a_blob_from_the_other_scheme_or_another_store_does_not_open() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = AccountStore::at_with_key_file(dir.path().join("a"));
        let b = AccountStore::at_with_key_file(dir.path().join("b"));
        let account = a.add(&signed_in(1, "Alpha", 10)).expect("add");
        b.add(&signed_in(2, "Beta", 10)).expect("add");
        assert!(matches!(b.unseal(&account), Err(AuthError::CacheCorrupt)));
    }

    #[test]
    fn an_unreadable_file_is_kept_aside_not_overwritten() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        std::fs::write(store.file_path(), b"{ this is not json").expect("write");
        assert!(matches!(store.load(), Err(AuthError::Config { .. })));
        store.add(&signed_in(1, "Alpha", 10)).expect("add");
        assert_eq!(store.load().expect("load").accounts.len(), 1);
        let kept: Vec<_> = std::fs::read_dir(dir.path())
            .expect("dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.starts_with("accounts.json.unreadable-"))
            .collect();
        assert_eq!(kept.len(), 1, "{kept:?}");
        let backup = std::fs::read(dir.path().join(&kept[0])).expect("backup");
        assert_eq!(backup, b"{ this is not json");
    }

    #[test]
    fn writes_are_atomic_and_leave_no_temporary_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        for n in 0..5 {
            store.add(&signed_in(n, "Name", 10)).expect("add");
        }
        let names: Vec<_> = std::fs::read_dir(dir.path())
            .expect("dir")
            .filter_map(Result::ok)
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            names.iter().all(|name| !name.ends_with(".tmp")),
            "{names:?}"
        );
        // A failed write (the target is a directory) leaves no temporary file behind.
        let blocked = dir.path().join("blocked");
        std::fs::create_dir(&blocked).expect("dir");
        assert!(write_atomically(&blocked, b"x").is_err());
        let leftovers = std::fs::read_dir(dir.path())
            .expect("dir")
            .filter_map(Result::ok)
            .filter(|entry| entry.file_name().to_string_lossy().ends_with(".tmp"))
            .count();
        assert_eq!(leftovers, 0);
    }

    /// The list held open by another program (as a virus scanner does
    /// right after a write: no delete sharing) only delays a save; it saves once let go.
    #[cfg(windows)]
    #[test]
    fn a_save_waits_briefly_for_a_list_held_open_by_another_program() {
        use std::os::windows::fs::OpenOptionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(ACCOUNTS_FILE_NAME);
        std::fs::write(&path, b"old").expect("write");
        let held = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(1) // FILE_SHARE_READ only
            .open(&path)
            .expect("hold");
        let release = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(300));
            drop(held);
        });
        write_atomically(&path, b"new").expect("saved once the file is let go");
        release.join().expect("release");
        assert_eq!(std::fs::read(&path).expect("read"), b"new");
    }

    #[test]
    fn debug_output_never_shows_a_token_or_the_sealed_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        let account = store.add(&signed_in(1, "Alpha", 10)).expect("add");
        let secrets = store.unseal(&account).expect("unseal");
        let shown = format!("{account:?} {secrets:?} {:?}", store.load().expect("load"));
        assert!(
            !shown.contains(REFRESH) && !shown.contains(ACCESS),
            "{shown}"
        );
        assert!(!shown.contains(&account.sealed[..16]), "{shown}");
        assert!(shown.contains("Alpha"));
    }

    #[test]
    fn log_lines_name_the_file_but_never_its_folder() {
        let dir = tempfile::tempdir().expect("tempdir");
        let folder = dir.path().join("SomeUserName").join("mc-rs");
        let io = AuthError::Io {
            path: folder.join(ACCOUNTS_FILE_NAME),
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        };
        let line = loggable(&io);
        assert!(line.contains(ACCOUNTS_FILE_NAME), "{line}");
        assert!(!line.contains("SomeUserName"), "{line}");
        let bad = serde_json::from_str::<serde_json::Value>("{").expect_err("bad json");
        let config = AuthError::Config {
            path: folder.join(ACCOUNTS_FILE_NAME),
            source: bad,
        };
        assert!(!loggable(&config).contains("SomeUserName"));
        assert_eq!(
            loggable(&AuthError::TokenRejected),
            AuthError::TokenRejected.to_string()
        );
    }

    #[test]
    fn needs_refresh_allows_for_the_skew() {
        let dir = tempfile::tempdir().expect("tempdir");
        let store = AccountStore::at(dir.path());
        let account = store.add(&signed_in(1, "Alpha", 1_000_000)).expect("add");
        assert!(!account.needs_refresh(1_000_000 - REFRESH_SKEW.as_secs() - 1));
        assert!(account.needs_refresh(1_000_000 - REFRESH_SKEW.as_secs()));
        assert!(account.needs_refresh(2_000_000));
    }
}
