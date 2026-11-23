//! The `Login` state: authentication handshake. 2 serverbound + 4 clientbound packets.
//!
//! java: `network/login/**`. The actual crypto (RSA encrypt/decrypt, AES-CFB8 setup) is S07's
//! job; this module only carries the wire fields.

use uuid::Uuid;

use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError, State};

/// `0x00 Login Start` (serverbound): the username the client wants to join as.
/// java: `C00PacketLoginStart`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginStart {
    /// The player's username (not yet a verified profile at this point in the handshake).
    pub username: String,
}

impl Packet for LoginStart {
    const STATE: State = State::Login;
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(LoginStart {
            username: buf.read_string(16)?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.username);
    }
}

/// `0x01 Encryption Response` (serverbound): the client's shared secret and verify token,
/// each RSA-encrypted (PKCS#1 v1.5) under the server's public key from [`EncryptionRequest`].
/// java: `C01PacketEncryptionResponse`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionResponse {
    /// The AES shared secret, RSA-encrypted.
    pub shared_secret_encrypted: Vec<u8>,
    /// [`EncryptionRequest::verify_token`], echoed back RSA-encrypted, proving possession of
    /// the shared secret without a plaintext round trip.
    pub verify_token_encrypted: Vec<u8>,
}

impl Packet for EncryptionResponse {
    const STATE: State = State::Login;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EncryptionResponse {
            shared_secret_encrypted: buf.read_byte_array()?,
            verify_token_encrypted: buf.read_byte_array()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_byte_array(&self.shared_secret_encrypted);
        buf.write_byte_array(&self.verify_token_encrypted);
    }
}

/// `0x00 Disconnect` (clientbound): login was rejected. `reason` is a raw chat-component JSON
/// string, as elsewhere in this crate.
/// java: `S00PacketDisconnect`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disconnect {
    /// The kick reason, as chat-component JSON.
    pub reason: String,
}

impl Packet for Disconnect {
    const STATE: State = State::Login;
    const ID: i32 = 0x00;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(Disconnect {
            reason: buf.read_chat()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_chat(&self.reason);
    }
}

/// `0x01 Encryption Request` (clientbound): asks the client to generate an AES shared secret,
/// RSA-encrypt it under `public_key_der`, and authenticate `server_id` with the session server
/// (hazard H3 — `server_id` here is the *unhashed* id; `mc-auth::server_hash` does the signed
/// SHA-1 rendering before the join-server call). java: `S01PacketEncryptionRequest`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionRequest {
    /// An (often empty) server-chosen id, combined with the shared secret and public key to
    /// form the session-server join hash.
    pub server_id: String,
    /// The server's RSA public key, X.509 `SubjectPublicKeyInfo` DER — parsing it is S07's job.
    pub public_key_der: Vec<u8>,
    /// A random token the client must echo back encrypted, proving it used the right key.
    pub verify_token: Vec<u8>,
}

impl Packet for EncryptionRequest {
    const STATE: State = State::Login;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EncryptionRequest {
            server_id: buf.read_string(20)?,
            public_key_der: buf.read_byte_array()?,
            verify_token: buf.read_byte_array()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.server_id);
        buf.write_byte_array(&self.public_key_der);
        buf.write_byte_array(&self.verify_token);
    }
}

/// `0x02 Login Success` (clientbound): login is complete, switch to `Play`.
///
/// **Not** the packed-`i64` [`PacketBuffer::read_uuid`] encoding used everywhere else in the
/// protocol — this one lone field is `UUID.toString()` (dashed, `readStringFromBuffer(36)`).
/// java: `S02PacketLoginSuccess`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoginSuccess {
    /// The joining player's UUID.
    pub uuid: Uuid,
    /// The joining player's username.
    pub username: String,
}

impl Packet for LoginSuccess {
    const STATE: State = State::Login;
    const ID: i32 = 0x02;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        let uuid_str = buf.read_string(36)?;
        let username = buf.read_string(16)?;
        // java: `UUID.fromString`, with Java 8's lenient component widths.
        let uuid =
            mc_auth::java_uuid_from_string(&uuid_str).ok_or(ProtoError::InvalidUuid(uuid_str))?;
        Ok(LoginSuccess { uuid, username })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.uuid.hyphenated().to_string());
        buf.write_string(&self.username);
    }
}

/// `0x03 Set Compression` (clientbound): every packet from here on with a decompressed size
/// `>= threshold` is zlib-compressed on the wire (S07's job to apply). A negative or absent
/// threshold means compression stays off.
/// java: `S03PacketEnableCompression`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SetCompression {
    /// The compression threshold, in bytes.
    pub threshold: i32,
}

impl Packet for SetCompression {
    const STATE: State = State::Login;
    const ID: i32 = 0x03;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(SetCompression {
            threshold: buf.read_varint()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_varint(self.threshold);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip<P: Packet + PartialEq + std::fmt::Debug>(pkt: P) {
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(P::read(&mut read_buf).unwrap(), pkt);
        assert!(read_buf.is_empty());
    }

    #[test]
    fn login_start_round_trips() {
        round_trip(LoginStart {
            username: "Notch".into(),
        });
    }

    #[test]
    fn encryption_response_round_trips() {
        round_trip(EncryptionResponse {
            shared_secret_encrypted: vec![1, 2, 3, 4],
            verify_token_encrypted: vec![5, 6, 7, 8],
        });
    }

    #[test]
    fn disconnect_round_trips() {
        round_trip(Disconnect {
            reason: r#"{"text":"kicked"}"#.into(),
        });
    }

    #[test]
    fn encryption_request_round_trips() {
        round_trip(EncryptionRequest {
            server_id: String::new(),
            public_key_der: vec![0xAA; 162],
            verify_token: vec![1, 2, 3, 4],
        });
    }

    #[test]
    fn login_success_round_trips_dashed_uuid() {
        let uuid = Uuid::from_u128(0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10);
        let pkt = LoginSuccess {
            uuid,
            username: "Notch".into(),
        };
        let mut buf = PacketBuffer::new();
        pkt.write(&mut buf);
        // Confirm it's really the dashed-string form, not the packed-i64 form.
        let mut peek = PacketBuffer::from_slice(buf.as_slice());
        let s = peek.read_string(36).unwrap();
        assert_eq!(s, uuid.hyphenated().to_string());

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(LoginSuccess::read(&mut read_buf).unwrap(), pkt);
    }

    #[test]
    fn login_success_rejects_malformed_uuid() {
        let mut buf = PacketBuffer::new();
        buf.write_string("not-a-uuid");
        buf.write_string("Notch");
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert!(matches!(
            LoginSuccess::read(&mut read_buf),
            Err(ProtoError::InvalidUuid(_))
        ));
    }

    fn read_login_success(uuid: &str) -> Result<LoginSuccess, ProtoError> {
        let mut buf = PacketBuffer::new();
        buf.write_string(uuid);
        buf.write_string("Notch");
        LoginSuccess::read(&mut PacketBuffer::from_slice(buf.as_slice()))
    }

    #[test]
    fn login_success_accepts_java_8_lenient_uuid_components() {
        let probes = [
            ("1-2-3-4-5", "00000001-0002-0003-0004-000000000005"),
            ("0-0-0-0-0", "00000000-0000-0000-0000-000000000000"),
            (
                "01020304-0506-0708-090a-0b0c0d0e0f1",
                "01020304-0506-0708-090a-00b0c0d0e0f1",
            ),
            // Java 8 shifts components into place without first enforcing their usual widths.
            ("123456789-2-3-4-5", "23456789-0002-0003-0004-000000000005"),
            // `Long.decode` consumes hex digits through Java 8's Unicode-aware Character.digit.
            ("１-２-３-４-５", "00000001-0002-0003-0004-000000000005"),
            ("Ａ-Ｂ-Ｃ-Ｄ-Ｅ", "0000000a-000b-000c-000d-00000000000e"),
            // Java's one-argument String.split discards trailing empty components.
            ("1-2-3-4-5---", "00000001-0002-0003-0004-000000000005"),
        ];

        for (wire, expected) in probes {
            let packet = read_login_success(wire).unwrap();
            assert_eq!(
                packet.uuid,
                Uuid::parse_str(expected).unwrap(),
                "wire UUID {wire}"
            );
        }
    }

    #[test]
    fn login_success_rejects_non_java_uuid_spellings() {
        for wire in [
            "0102030405060708090a0b0c0d0e0f10",
            "1-2-3-4-5-6",
            "1-2-3-4-",
            "8000000000000000-2-3-4-5",
        ] {
            assert!(
                matches!(read_login_success(wire), Err(ProtoError::InvalidUuid(_))),
                "wire UUID {wire} must be rejected"
            );
        }
    }

    #[test]
    fn login_success_reads_username_before_parsing_uuid() {
        let mut buf = PacketBuffer::new();
        buf.write_string("not-a-uuid");
        buf.write_string("seventeen-letters!");

        assert!(matches!(
            LoginSuccess::read(&mut PacketBuffer::from_slice(buf.as_slice())),
            Err(ProtoError::StringTooLong { max: 16, .. })
        ));
    }

    #[test]
    fn set_compression_round_trips() {
        round_trip(SetCompression { threshold: 256 });
        round_trip(SetCompression { threshold: -1 });
    }
}
