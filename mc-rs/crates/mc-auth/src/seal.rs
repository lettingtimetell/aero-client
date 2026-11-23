//! Sealing small secrets at rest for the account store ([`crate::accounts`]).
//!
//! A sealed blob starts with one byte naming its scheme, so a store written on one platform, or
//! by a later scheme, is recognised instead of decrypted into garbage:
//!
//! * [`SCHEME_DPAPI`] (Windows): Windows DPAPI for the current user ([`crate::dpapi`]). Only the
//!   same Windows user on the same PC can open it; there is no key file to steal.
//! * [`SCHEME_KEY_FILE`] (every other platform): AES-256-GCM under a random 32-byte key kept in a
//!   key file beside the store (mode `0600` on Unix). That key is readable by anything that can
//!   read the store, so this is obfuscation rather than protection, exactly as
//!   [`crate::cache`] documents for the older single-account cache it shares its format with.
//!
//! Failures never carry the plaintext or the sealed bytes, only the OS error.

use std::io;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Key, Nonce};

/// Windows DPAPI, current user.
pub const SCHEME_DPAPI: u8 = 1;
/// AES-256-GCM with a key file.
pub const SCHEME_KEY_FILE: u8 = 2;

const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;

/// How secrets are sealed.
#[derive(Debug, Clone)]
pub struct Sealer {
    kind: Kind,
    /// The key file for [`SCHEME_KEY_FILE`] (created on first use); also lets a Windows build
    /// open a key-file blob, for tests.
    key_path: PathBuf,
    /// DPAPI's optional entropy: a fixed, non-secret value naming the purpose.
    #[cfg_attr(not(windows), allow(dead_code))]
    entropy: &'static [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    #[cfg(windows)]
    Dpapi,
    KeyFile,
}

impl Sealer {
    /// The platform's best scheme: DPAPI on Windows, the key file at `key_path` elsewhere.
    #[must_use]
    pub fn platform(key_path: PathBuf, entropy: &'static [u8]) -> Sealer {
        #[cfg(windows)]
        let kind = Kind::Dpapi;
        #[cfg(not(windows))]
        let kind = Kind::KeyFile;
        Sealer {
            kind,
            key_path,
            entropy,
        }
    }

    /// The key-file scheme on every platform (the non-Windows format; tests use it on Windows).
    #[must_use]
    pub fn key_file(key_path: PathBuf) -> Sealer {
        Sealer {
            kind: Kind::KeyFile,
            key_path,
            entropy: b"",
        }
    }

    /// The scheme byte new blobs get.
    #[must_use]
    pub fn scheme(&self) -> u8 {
        match self.kind {
            #[cfg(windows)]
            Kind::Dpapi => SCHEME_DPAPI,
            Kind::KeyFile => SCHEME_KEY_FILE,
        }
    }

    /// Seal `plain`: the scheme byte, then the scheme's payload.
    pub fn seal(&self, plain: &[u8]) -> io::Result<Vec<u8>> {
        let payload = match self.kind {
            #[cfg(windows)]
            Kind::Dpapi => crate::dpapi::protect(plain, self.entropy)?,
            Kind::KeyFile => {
                let key = match read_key(&self.key_path)? {
                    Some(key) => key,
                    None => create_key(&self.key_path)?,
                };
                aes_seal(&key, plain)?
            }
        };
        let mut blob = Vec::with_capacity(1 + payload.len());
        blob.push(self.scheme());
        blob.extend_from_slice(&payload);
        Ok(blob)
    }

    /// Open a blob [`Sealer::seal`] made, whichever scheme made it (as long as this platform
    /// has that scheme).
    pub fn unseal(&self, blob: &[u8]) -> io::Result<Vec<u8>> {
        let Some((&scheme, payload)) = blob.split_first() else {
            return Err(unreadable("empty sealed blob"));
        };
        match scheme {
            #[cfg(windows)]
            SCHEME_DPAPI => crate::dpapi::unprotect(payload, self.entropy),
            #[cfg(not(windows))]
            SCHEME_DPAPI => Err(unreadable("sealed with Windows DPAPI on another system")),
            SCHEME_KEY_FILE => {
                let key = read_key(&self.key_path)?.ok_or_else(|| unreadable("no key file"))?;
                aes_open(&key, payload)
            }
            _ => Err(unreadable("unknown sealing scheme")),
        }
    }

    /// Delete the key file, if this scheme has one (nothing for DPAPI).
    pub fn forget_key(&self) -> io::Result<()> {
        match std::fs::remove_file(&self.key_path) {
            Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
            _ => Ok(()),
        }
    }
}

fn unreadable(what: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, what)
}

fn aes_seal(key: &[u8; KEY_LEN], plain: &[u8]) -> io::Result<Vec<u8>> {
    let mut nonce = [0u8; NONCE_LEN];
    getrandom::getrandom(&mut nonce).map_err(|_| io::Error::other("no random numbers"))?;
    let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key));
    let ciphertext = cipher
        .encrypt(Nonce::from_slice(&nonce), plain)
        .map_err(|_| io::Error::other("encryption failed"))?;
    let mut out = Vec::with_capacity(NONCE_LEN + ciphertext.len());
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

fn aes_open(key: &[u8; KEY_LEN], payload: &[u8]) -> io::Result<Vec<u8>> {
    if payload.len() < NONCE_LEN {
        return Err(unreadable("truncated sealed blob"));
    }
    let (nonce, ciphertext) = payload.split_at(NONCE_LEN);
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| unreadable("the sealed blob does not open with this key"))
}

fn read_key(path: &Path) -> io::Result<Option<[u8; KEY_LEN]>> {
    match std::fs::read(path) {
        Ok(bytes) if bytes.len() == KEY_LEN => {
            let mut key = [0u8; KEY_LEN];
            key.copy_from_slice(&bytes);
            Ok(Some(key))
        }
        Ok(_) => Err(unreadable("the key file has the wrong length")),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn create_key(path: &Path) -> io::Result<[u8; KEY_LEN]> {
    let mut key = [0u8; KEY_LEN];
    getrandom::getrandom(&mut key).map_err(|_| io::Error::other("no random numbers"))?;
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::accounts::write_atomically(path, &key)?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SECRET: &[u8] = b"fake-access-token-0123456789abcdef";

    #[test]
    fn the_platform_scheme_round_trips_without_showing_the_secret() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sealer = Sealer::platform(dir.path().join("k.key"), b"test purpose");
        let blob = sealer.seal(SECRET).expect("seal");
        assert_eq!(blob[0], sealer.scheme());
        assert!(!blob.windows(SECRET.len()).any(|w| w == SECRET));
        assert_eq!(sealer.unseal(&blob).expect("unseal"), SECRET);
        #[cfg(windows)]
        {
            assert_eq!(sealer.scheme(), SCHEME_DPAPI);
            assert!(
                !dir.path().join("k.key").exists(),
                "DPAPI keeps no key file next to the store"
            );
        }
    }

    #[test]
    fn the_key_file_scheme_round_trips_and_needs_its_key() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("nested").join("k.key");
        let sealer = Sealer::key_file(key_path.clone());
        let blob = sealer.seal(SECRET).expect("seal");
        assert_eq!(blob[0], SCHEME_KEY_FILE);
        assert!(key_path.exists());
        assert_eq!(sealer.unseal(&blob).expect("unseal"), SECRET);
        // A fresh nonce per seal.
        assert_ne!(sealer.seal(SECRET).expect("seal"), blob);
        // Without the key, or with another key, it stays shut.
        std::fs::write(&key_path, [7u8; KEY_LEN]).expect("write");
        assert!(sealer.unseal(&blob).is_err());
        sealer.forget_key().expect("forget");
        assert!(sealer.unseal(&blob).is_err());
        sealer.forget_key().expect("forgetting twice is fine");
    }

    #[test]
    fn damaged_or_unknown_blobs_are_errors_not_panics() {
        let dir = tempfile::tempdir().expect("tempdir");
        for sealer in [
            Sealer::platform(dir.path().join("a.key"), b"p"),
            Sealer::key_file(dir.path().join("b.key")),
        ] {
            let mut blob = sealer.seal(SECRET).expect("seal");
            let last = blob.len() - 1;
            blob[last] ^= 0xff;
            assert!(sealer.unseal(&blob).is_err());
            assert!(sealer.unseal(&[]).is_err());
            assert!(sealer.unseal(&[SCHEME_KEY_FILE, 1, 2]).is_err());
            assert!(sealer.unseal(&[0xEE, 1, 2, 3]).is_err());
        }
    }

    #[cfg(windows)]
    #[test]
    fn a_windows_build_still_opens_a_key_file_blob() {
        let dir = tempfile::tempdir().expect("tempdir");
        let key_path = dir.path().join("k.key");
        let blob = Sealer::key_file(key_path.clone())
            .seal(SECRET)
            .expect("seal");
        let platform = Sealer::platform(key_path, b"p");
        assert_eq!(platform.unseal(&blob).expect("unseal"), SECRET);
    }
}
