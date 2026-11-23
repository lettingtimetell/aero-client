//! Vanilla offline UUIDs and development-only offline sessions.
//!
//! [`offline_uuid`] is also the production-compatible fallback used by vanilla render paths when
//! a [`GameProfile`](crate::GameProfile) has a name but no UUID. Constructing an actual offline
//! session remains development-only behind the `offline` feature: online-mode servers verify the
//! join through `sessionserver.mojang.com` and reject it.

use uuid::Uuid;

#[cfg(feature = "offline")]
use crate::session::Session;

/// The prefix vanilla hashes to derive an offline player's UUID.
const OFFLINE_PREFIX: &str = "OfflinePlayer:";

/// Vanilla's offline-mode UUID: `UUID.nameUUIDFromBytes("OfflinePlayer:<name>")`.
///
/// That is a name-based (version 3, MD5) UUID over the UTF-8 bytes of the prefixed name, with
/// the version and variant bits forced. Servers compute the same value, so a client that
/// derives it any other way gets a different identity than the server assigns it.
///
/// // java: EntityPlayer.getOfflineUUID / PreYggdrasilConverter
#[must_use]
pub fn offline_uuid(username: &str) -> Uuid {
    use md5::{Digest, Md5};

    let mut hasher = Md5::new();
    hasher.update(OFFLINE_PREFIX.as_bytes());
    hasher.update(username.as_bytes());
    let mut bytes: [u8; 16] = hasher.finalize().into();
    bytes[6] = (bytes[6] & 0x0f) | 0x30; // version 3
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // IETF variant
    Uuid::from_bytes(bytes)
}

/// Build a session for an offline-mode server.
///
/// The access token is empty; [`join_server`](crate::join_server) must not be called with it.
#[must_use]
#[cfg(feature = "offline")]
pub fn offline_session(username: &str) -> Session {
    Session::new(offline_uuid(username), username, String::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_vanilla_offline_uuids() {
        // Golden values: version-3 MD5 UUIDs of "OfflinePlayer:<name>", the same derivation the
        // vanilla server uses when it assigns an identity in offline mode.
        for (name, expected) in [
            ("Notch", "b50ad385-829d-3141-a216-7e7d7539ba7f"),
            ("jeb_", "a762f560-4fce-3236-812a-b80efff0b62b"),
            ("Player", "a01e3843-e521-3998-958a-f459800e4d11"),
            ("Dinnerbone", "4d258a81-2358-3084-8166-05b9faccad80"),
        ] {
            assert_eq!(offline_uuid(name).to_string(), expected, "for {name}");
        }
    }

    #[test]
    fn version_and_variant_bits_are_forced() {
        let u = offline_uuid("SomeoneElse");
        assert_eq!(u.get_version_num(), 3);
        assert_eq!(u.as_bytes()[8] & 0xc0, 0x80);
    }

    #[cfg(feature = "offline")]
    #[test]
    fn session_carries_no_token() {
        let s = offline_session("Notch");
        assert_eq!(s.username, "Notch");
        assert!(s.expose().is_empty());
        assert_eq!(s.uuid, offline_uuid("Notch"));
    }
}
