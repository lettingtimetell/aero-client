//! Encrypted-at-rest token cache.
//!
//! # Threat model — read this before trusting it
//!
//! Credentials are sealed with AES-256-GCM under a random 32-byte key that lives in a sibling
//! file (`session.key`, mode `0600` on Unix). **That key is readable by anything that can read
//! the ciphertext**, so this is obfuscation, not protection:
//!
//! * It **does** stop a Minecraft access token appearing in plain text in a backup, a synced
//!   config folder, a screen-shared file browser, a support-request zip, or a `grep -r` across
//!   the home directory.
//! * It does **not** protect against a compromised user account, malware running as the user,
//!   or anyone with unattended access to an unlocked machine. If your account is compromised,
//!   so is the token.
//!
//! # Why not the OS keyring
//!
//! The honest reason is verifiability. A keyring backend (`keyring`/`secret-service`) would
//! genuinely raise the bar on Windows and macOS, but on Linux it depends on a running D-Bus
//! Secret Service, which is absent in headless CI and in some desktop setups, and this section
//! was implemented without the ability to test that path on any of the three platforms. The
//! sealing key is deliberately isolated in [`TokenCache::key`] so that swapping the keyfile for
//! an OS keyring entry is a contained change; the on-disk format needs no revision for it.
//!
//! Mitigation that *is* in place: tokens are short-lived (the Minecraft access token expires in
//! about a day), the cache is written with restrictive permissions, and nothing here is ever
//! logged.

use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};

use crate::config;
use crate::error::AuthError;
use crate::session::Credentials;

/// File name of the sealed credential blob.
pub const CACHE_FILE_NAME: &str = "session.bin";
/// File name of the sealing key.
pub const KEY_FILE_NAME: &str = "session.key";

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;
/// Format byte, so a future scheme change can be detected instead of decrypting garbage.
const FORMAT_V1: u8 = 1;

/// A directory holding the sealed credentials and their key.
#[derive(Debug, Clone)]
pub struct TokenCache {
    dir: PathBuf,
}

impl TokenCache {
    /// The cache in the platform's per-user config directory.
    pub fn discover() -> Result<Self, AuthError> {
        Ok(Self::at(config::config_dir()?))
    }

    /// A cache rooted at a specific directory.
    #[must_use]
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// The directory this cache uses.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn cache_path(&self) -> PathBuf {
        self.dir.join(CACHE_FILE_NAME)
    }

    fn key_path(&self) -> PathBuf {
        self.dir.join(KEY_FILE_NAME)
    }

    /// Read and decrypt the stored credentials.
    ///
    /// `Ok(None)` means "nothing cached", which is a normal first-run state.
    /// [`AuthError::CacheCorrupt`] means something is there but unusable — callers should fall
    /// back to an interactive login rather than aborting.
    pub fn load(&self) -> Result<Option<Credentials>, AuthError> {
        let path = self.cache_path();
        let blob = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(AuthError::Io { path, source }),
        };
        if blob.len() < 1 + NONCE_LEN || blob[0] != FORMAT_V1 {
            return Err(AuthError::CacheCorrupt);
        }
        let key = self.key()?.ok_or(AuthError::CacheCorrupt)?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
        let nonce = Nonce::from_slice(&blob[1..1 + NONCE_LEN]);
        let plaintext = cipher
            .decrypt(nonce, &blob[1 + NONCE_LEN..])
            .map_err(|_| AuthError::CacheCorrupt)?;
        let credentials: Credentials =
            serde_json::from_slice(&plaintext).map_err(|_| AuthError::CacheCorrupt)?;
        // Tokens read back from disk are as secret as fresh ones.
        mc_core::secrets::register(&credentials.msa_refresh_token);
        mc_core::secrets::register(&credentials.mc_access_token);
        Ok(Some(credentials))
    }

    /// Encrypt and store credentials, creating the directory and key on first use.
    pub fn save(&self, creds: &Credentials) -> Result<(), AuthError> {
        std::fs::create_dir_all(&self.dir).map_err(|source| AuthError::Io {
            path: self.dir.clone(),
            source,
        })?;
        let key = match self.key()? {
            Some(k) => k,
            None => self.create_key()?,
        };
        let plaintext = serde_json::to_vec(creds).map_err(|_| AuthError::CacheCorrupt)?;

        let mut nonce_bytes = [0u8; NONCE_LEN];
        getrandom::getrandom(&mut nonce_bytes).map_err(|_| AuthError::CacheCorrupt)?;
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&key));
        let ciphertext = cipher
            .encrypt(Nonce::from_slice(&nonce_bytes), plaintext.as_slice())
            .map_err(|_| AuthError::CacheCorrupt)?;

        let mut blob = Vec::with_capacity(1 + NONCE_LEN + ciphertext.len());
        blob.push(FORMAT_V1);
        blob.extend_from_slice(&nonce_bytes);
        blob.extend_from_slice(&ciphertext);
        write_private(&self.cache_path(), &blob)
    }

    /// Delete the cached credentials (a "sign out"). The sealing key is removed too.
    pub fn clear(&self) -> Result<(), AuthError> {
        for path in [self.cache_path(), self.key_path()] {
            match std::fs::remove_file(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => return Err(AuthError::Io { path, source }),
            }
        }
        Ok(())
    }

    /// Read the sealing key, if one has been created.
    ///
    /// This is the single seam an OS-keyring backend would replace.
    fn key(&self) -> Result<Option<[u8; KEY_LEN]>, AuthError> {
        let path = self.key_path();
        match std::fs::read(&path) {
            Ok(bytes) if bytes.len() == KEY_LEN => {
                let mut key = [0u8; KEY_LEN];
                key.copy_from_slice(&bytes);
                Ok(Some(key))
            }
            Ok(_) => Err(AuthError::CacheCorrupt),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(source) => Err(AuthError::Io { path, source }),
        }
    }

    fn create_key(&self) -> Result<[u8; KEY_LEN], AuthError> {
        let mut key = [0u8; KEY_LEN];
        getrandom::getrandom(&mut key).map_err(|_| AuthError::CacheCorrupt)?;
        write_private(&self.key_path(), &key)?;
        Ok(key)
    }
}

/// Write a file that only the owning user can read.
///
/// On Unix the mode is set to `0600` on creation, so there is no window in which the file exists
/// world-readable. On Windows the inherited ACL of the per-user config directory is what
/// restricts access; there is no portable `std` API to tighten it further.
///
/// Written atomically (a new temporary file beside it, synced, then renamed
/// over it, as the account store writes), so a crash or a power cut mid-save leaves the old
/// credentials, never a truncated file.
fn write_private(path: &Path, bytes: &[u8]) -> Result<(), AuthError> {
    crate::accounts::write_atomically(path, bytes).map_err(|source| AuthError::Io {
        path: path.to_owned(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn creds() -> Credentials {
        Credentials {
            msa_refresh_token: "M.R3_BAY.the-refresh-token".to_owned(),
            mc_access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.the-access-token".to_owned(),
            expires_at: 1_800_000_000,
            uuid: Uuid::parse_str("986dec87-b7ec-47ff-89ff-033fdb95c4b5").expect("uuid"),
            username: "HowDoesAuthWork".to_owned(),
        }
    }

    #[test]
    fn round_trips() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path().join("nested"));
        assert!(cache.load().expect("load").is_none());

        cache.save(&creds()).expect("save");
        let got = cache.load().expect("load").expect("some");
        assert_eq!(got.msa_refresh_token, creds().msa_refresh_token);
        assert_eq!(got.mc_access_token, creds().mc_access_token);
        assert_eq!(got.uuid, creds().uuid);
        assert_eq!(got.username, creds().username);
        assert_eq!(got.expires_at, creds().expires_at);
    }

    #[test]
    fn tokens_are_not_plaintext_on_disk() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        let raw = std::fs::read(dir.path().join(CACHE_FILE_NAME)).expect("read");
        let haystack = String::from_utf8_lossy(&raw);
        assert!(!haystack.contains("the-refresh-token"));
        assert!(!haystack.contains("the-access-token"));
        assert!(!haystack.contains("HowDoesAuthWork"));
    }

    #[test]
    fn overwriting_uses_a_fresh_nonce() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        let first = std::fs::read(dir.path().join(CACHE_FILE_NAME)).expect("read");
        cache.save(&creds()).expect("save");
        let second = std::fs::read(dir.path().join(CACHE_FILE_NAME)).expect("read");
        assert_ne!(first, second, "identical ciphertext means a reused nonce");
    }

    #[test]
    fn tampering_is_detected() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        let path = dir.path().join(CACHE_FILE_NAME);
        let mut raw = std::fs::read(&path).expect("read");
        let last = raw.len() - 1;
        raw[last] ^= 0xff;
        std::fs::write(&path, &raw).expect("write");
        assert!(matches!(cache.load(), Err(AuthError::CacheCorrupt)));
    }

    #[test]
    fn a_wrong_key_does_not_decrypt() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        std::fs::write(dir.path().join(KEY_FILE_NAME), [7u8; KEY_LEN]).expect("write");
        assert!(matches!(cache.load(), Err(AuthError::CacheCorrupt)));
    }

    #[test]
    fn truncated_and_misversioned_blobs_are_corrupt_not_panics() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        let path = dir.path().join(CACHE_FILE_NAME);

        std::fs::write(&path, [FORMAT_V1, 0, 0]).expect("write");
        assert!(matches!(cache.load(), Err(AuthError::CacheCorrupt)));

        let mut raw = std::fs::read(&path).expect("read");
        raw[0] = 0xfe;
        std::fs::write(&path, &raw).expect("write");
        assert!(matches!(cache.load(), Err(AuthError::CacheCorrupt)));
    }

    /// Saves go through a temporary file and a rename: after several saves the
    /// folder holds exactly the cache and its key, and the last save is what loads.
    #[test]
    fn saves_are_atomic_and_leave_no_temporary_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        let mut latest = creds();
        for round in 0..3 {
            latest.username = format!("Player{round}");
            cache.save(&latest).expect("save");
        }
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .expect("read_dir")
            .map(|entry| {
                entry
                    .expect("entry")
                    .file_name()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        names.sort();
        assert_eq!(names, [CACHE_FILE_NAME, KEY_FILE_NAME]);
        assert_eq!(
            cache.load().expect("load").expect("some").username,
            "Player2"
        );
    }

    #[test]
    fn clear_removes_everything() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        cache.clear().expect("clear");
        assert!(cache.load().expect("load").is_none());
        assert!(!dir.path().join(KEY_FILE_NAME).exists());
        // Clearing twice is not an error.
        cache.clear().expect("clear again");
    }

    #[cfg(unix)]
    #[test]
    fn files_are_owner_only() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().expect("tempdir");
        let cache = TokenCache::at(dir.path());
        cache.save(&creds()).expect("save");
        for name in [CACHE_FILE_NAME, KEY_FILE_NAME] {
            let mode = std::fs::metadata(dir.path().join(name))
                .expect("metadata")
                .permissions()
                .mode();
            assert_eq!(mode & 0o077, 0, "{name} is group/world accessible");
        }
    }
}
