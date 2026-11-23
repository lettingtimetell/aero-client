//! Checking a download: its SHA-256 against the published one, and its Ed25519 signature
//! against the public key compiled into this build.
//!
//! # What is signed
//!
//! Not the zip's bytes directly but a short text naming them ([`signed_message`]):
//!
//! ```text
//! aero-client-release-v1
//! asset: Aero-Client-0.2.0-windows-x64.zip
//! version: 0.2.0
//! sha256: <64 lower-case hex digits>
//! ```
//!
//! Binding the version and the asset name means a validly signed old release cannot be
//! re-published under a newer tag to push players back to it, and the hash ties the text to the
//! exact bytes. The `.sig` asset is the 64-byte Ed25519 signature, in standard base64 on one line
//! (the raw 64 bytes are accepted too). `AERO_UPDATE_PUBKEY` at build time is the 32-byte public
//! key in standard base64.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use base64::Engine as _;

use crate::error::UpdateError;

/// The first line of every [`signed_message`]: a domain separator, so a signature made for
/// anything else can never pass as a release signature.
pub const SIGNATURE_CONTEXT: &str = "aero-client-release-v1";

/// Lower-case hex of `bytes`.
#[must_use]
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        text.push(char::from(DIGITS[usize::from(byte >> 4)]));
        text.push(char::from(DIGITS[usize::from(byte & 0x0f)]));
    }
    text
}

/// `text` if it is exactly 64 hex digits (either case), in lower case.
#[must_use]
pub fn sha256_hex(text: &str) -> Option<String> {
    (text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| text.to_ascii_lowercase())
}

/// The SHA-256 in a release asset's `digest` field (`sha256:<hex>`), in lower case.
pub fn parse_api_digest(digest: &str) -> Result<String, UpdateError> {
    digest
        .strip_prefix("sha256:")
        .and_then(sha256_hex)
        .ok_or_else(|| UpdateError::BadRelease("the asset's digest is not a SHA-256".to_owned()))
}

/// The SHA-256 in a `.sha256` asset: the `sha256sum` line `<hex>  <name>` (or `<hex> *<name>`),
/// or the hex alone. When a name is given it must be `asset_name`.
pub fn parse_sha256_sidecar(bytes: &[u8], asset_name: &str) -> Result<String, UpdateError> {
    let bad = || UpdateError::BadRelease("the .sha256 asset is malformed".to_owned());
    let text = std::str::from_utf8(bytes).map_err(|_| bad())?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let line = lines.next().ok_or_else(bad)?;
    if lines.next().is_some() {
        return Err(bad());
    }
    let mut fields = line.split_whitespace();
    let hash = fields.next().and_then(sha256_hex).ok_or_else(bad)?;
    if let Some(name) = fields.next() {
        if name.trim_start_matches('*') != asset_name || fields.next().is_some() {
            return Err(bad());
        }
    }
    Ok(hash)
}

/// The text a release signature covers (see the module docs).
#[must_use]
pub fn signed_message(asset_name: &str, version: &str, sha256: &str) -> String {
    format!("{SIGNATURE_CONTEXT}\nasset: {asset_name}\nversion: {version}\nsha256: {sha256}\n")
}

/// A `.sig` asset's signature: base64 text of 64 bytes (surrounding whitespace allowed), or the
/// raw 64 bytes.
pub fn parse_signature(bytes: &[u8]) -> Result<[u8; 64], UpdateError> {
    if let Ok(raw) = <[u8; 64]>::try_from(bytes) {
        if !raw.iter().all(|byte| byte.is_ascii_graphic()) {
            return Ok(raw);
        }
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|_| UpdateError::Signature("the .sig asset is malformed".to_owned()))?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(text.trim())
        .map_err(|_| UpdateError::Signature("the .sig asset is not base64".to_owned()))?;
    <[u8; 64]>::try_from(decoded.as_slice())
        .map_err(|_| UpdateError::Signature("the .sig asset is not 64 bytes".to_owned()))
}

/// Verify `signature` over `message` with the Ed25519 `public_key`.
pub fn verify_signature(
    public_key: &[u8; 32],
    message: &[u8],
    signature: &[u8; 64],
) -> Result<(), UpdateError> {
    ring::signature::UnparsedPublicKey::new(&ring::signature::ED25519, public_key)
        .verify(message, signature)
        .map_err(|_| UpdateError::Signature("the signature does not verify".to_owned()))
}

/// A public key in standard base64 (32 bytes).
pub fn parse_public_key(text: &str) -> Result<[u8; 32], UpdateError> {
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(text.trim())
        .map_err(|_| UpdateError::Signature("the public key is not base64".to_owned()))?;
    <[u8; 32]>::try_from(decoded.as_slice())
        .map_err(|_| UpdateError::Signature("the public key is not 32 bytes".to_owned()))
}

/// The public key compiled in through `AERO_UPDATE_PUBKEY` (`None` when the build did not set
/// it, or set it empty). A malformed key is an error, and the updater then refuses to install
/// anything rather than skip the check.
pub fn compiled_public_key() -> Result<Option<[u8; 32]>, UpdateError> {
    match option_env!("AERO_UPDATE_PUBKEY") {
        Some(text) if !text.trim().is_empty() => parse_public_key(text).map(Some),
        _ => Ok(None),
    }
}

/// The SHA-256 of a file, in lower-case hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut reader = BufReader::with_capacity(1 << 16, File::open(path)?);
    let mut hasher = ring::digest::Context::new(&ring::digest::SHA256);
    let mut buffer = vec![0_u8; 1 << 16];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex(hasher.finish().as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ring::signature::{Ed25519KeyPair, KeyPair};

    pub(crate) fn test_key(seed: u8) -> Ed25519KeyPair {
        Ed25519KeyPair::from_seed_unchecked(&[seed; 32]).expect("a test key")
    }

    #[test]
    fn hex_round_trips() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        let hash = "A".repeat(64);
        assert_eq!(sha256_hex(&hash), Some("a".repeat(64)));
        assert_eq!(sha256_hex(&"a".repeat(63)), None);
        assert_eq!(sha256_hex(&"g".repeat(64)), None);
    }

    #[test]
    fn api_digests_must_be_sha256() {
        let hex64 = "ab".repeat(32);
        assert_eq!(parse_api_digest(&format!("sha256:{hex64}")).unwrap(), hex64);
        assert_eq!(
            parse_api_digest(&format!("sha256:{}", hex64.to_uppercase())).unwrap(),
            hex64
        );
        for bad in [
            String::new(),
            hex64.clone(),
            format!("sha1:{hex64}"),
            format!("sha256:{}", &hex64[2..]),
            format!("SHA256:{hex64}"),
            format!("sha256: {hex64}"),
        ] {
            assert!(parse_api_digest(&bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn sha256_sidecars_in_the_usual_shapes() {
        let hex64 = "cd".repeat(32);
        let name = "Aero-Client-0.2.0-windows-x64.zip";
        for good in [
            hex64.clone(),
            format!("{hex64}\n"),
            format!("{hex64}  {name}\n"),
            format!("{hex64} *{name}\r\n"),
            format!("\u{feff}{}  {name}", hex64.to_uppercase()),
        ] {
            assert_eq!(
                parse_sha256_sidecar(good.as_bytes(), name).unwrap(),
                hex64,
                "{good:?}"
            );
        }
        for bad in [
            String::new(),
            "not a hash".to_owned(),
            format!("{hex64}  other.zip"),
            format!("{hex64}  {name}\n{hex64}  {name}"),
            format!("{hex64}  {name} extra"),
        ] {
            assert!(
                parse_sha256_sidecar(bad.as_bytes(), name).is_err(),
                "{bad:?}"
            );
        }
        assert!(parse_sha256_sidecar(&[0xff, 0xfe], name).is_err());
    }

    #[test]
    fn signatures_verify_only_for_the_signed_text_and_key() {
        let key = test_key(7);
        let public: [u8; 32] = key.public_key().as_ref().try_into().unwrap();
        let hash = "ef".repeat(32);
        let message = signed_message("Aero-Client-0.2.0-windows-x64.zip", "0.2.0", &hash);
        assert!(message.starts_with("aero-client-release-v1\n"));
        let signature: [u8; 64] = key.sign(message.as_bytes()).as_ref().try_into().unwrap();
        verify_signature(&public, message.as_bytes(), &signature).expect("good signature");

        // Another version, asset or hash: refused.
        for other in [
            signed_message("Aero-Client-0.2.0-windows-x64.zip", "0.3.0", &hash),
            signed_message("Aero-Client-0.3.0-windows-x64.zip", "0.2.0", &hash),
            signed_message(
                "Aero-Client-0.2.0-windows-x64.zip",
                "0.2.0",
                &"00".repeat(32),
            ),
        ] {
            assert!(verify_signature(&public, other.as_bytes(), &signature).is_err());
        }
        // A flipped bit, or another key: refused.
        let mut flipped = signature;
        flipped[10] ^= 1;
        assert!(verify_signature(&public, message.as_bytes(), &flipped).is_err());
        let other: [u8; 32] = test_key(8).public_key().as_ref().try_into().unwrap();
        assert!(verify_signature(&other, message.as_bytes(), &signature).is_err());
    }

    #[test]
    fn signature_files_are_base64_or_raw() {
        let signature = [0x5a_u8; 64];
        let text = base64::engine::general_purpose::STANDARD.encode(signature);
        assert_eq!(parse_signature(text.as_bytes()).unwrap(), signature);
        assert_eq!(
            parse_signature(format!("  {text}\r\n").as_bytes()).unwrap(),
            signature
        );
        let raw = [0x01_u8; 64];
        assert_eq!(parse_signature(&raw).unwrap(), raw);
        assert!(parse_signature(b"").is_err());
        assert!(parse_signature(b"not base64!").is_err());
        let short = base64::engine::general_purpose::STANDARD.encode([1_u8; 63]);
        assert!(parse_signature(short.as_bytes()).is_err());
    }

    #[test]
    fn public_keys_are_32_bytes_of_base64() {
        let key = base64::engine::general_purpose::STANDARD.encode([9_u8; 32]);
        assert_eq!(parse_public_key(&key).unwrap(), [9_u8; 32]);
        assert_eq!(parse_public_key(&format!(" {key}\n")).unwrap(), [9_u8; 32]);
        assert!(parse_public_key("").is_err());
        assert!(
            parse_public_key(&base64::engine::general_purpose::STANDARD.encode([9_u8; 31]))
                .is_err()
        );
    }

    /// Whatever this build was given, it is either absent or a valid key: a malformed
    /// `AERO_UPDATE_PUBKEY` fails here, at test time, as well as at run time.
    #[test]
    fn the_compiled_key_is_absent_or_valid() {
        let key = compiled_public_key().expect("AERO_UPDATE_PUBKEY, when set, is a valid key");
        if let Some(key) = key {
            assert_eq!(key.len(), 32);
        }
    }

    #[test]
    fn file_hashes_match_known_vectors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("abc");
        std::fs::write(&path, b"abc").unwrap();
        assert_eq!(
            sha256_file(&path).unwrap(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
