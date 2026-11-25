//! Protocol-5 login packets that differ from 1.8.9: the two encryption packets, whose byte
//! arrays carry an `i16` length (`Packet.readBlob`/`writeBlob`) instead of a VarInt.
//!
//! Everything else in the login state is byte-identical and reused from [`crate::login`]:
//! `C00PacketLoginStart` (`string(16)`), `S00PacketDisconnect` (chat JSON) and
//! `S02PacketLoginSuccess` (`string(36)` dashed UUID read with `UUID.fromString`, then
//! `string(16)`). Login has no `0x03`: protocol 5 has no compression.
//!
//! The conversions to and from [`crate::login::EncryptionRequest`] and
//! [`crate::login::EncryptionResponse`] carry the same fields, so [`crate::crypt`] runs the key
//! exchange unchanged.
//!
//! java17: `network/login/server/S01PacketEncryptionRequest.java`,
//! `network/login/client/C01PacketEncryptionResponse.java`

use super::buffer::PacketBufferV5;
use crate::login as login47;
use crate::packet::Packet;
use crate::{PacketBuffer, ProtoError, State};

/// `0x01 Encryption Request` (clientbound). java17: `S01PacketEncryptionRequest`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionRequest {
    /// `readStringFromBuffer(20)`, hashed into the session-server join.
    pub server_id: String,
    /// The server's RSA public key, X.509 `SubjectPublicKeyInfo` DER (`readBlob`, then
    /// `CryptManager.decodePublicKey`).
    pub public_key_der: Vec<u8>,
    /// The token to echo back encrypted (`readBlob`).
    pub verify_token: Vec<u8>,
}

impl Packet for EncryptionRequest {
    const STATE: State = State::Login;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EncryptionRequest {
            server_id: buf.read_string(20)?,
            public_key_der: buf.read_blob()?,
            verify_token: buf.read_blob()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_string(&self.server_id);
        buf.write_blob(&self.public_key_der);
        buf.write_blob(&self.verify_token);
    }
}

impl From<EncryptionRequest> for login47::EncryptionRequest {
    fn from(packet: EncryptionRequest) -> login47::EncryptionRequest {
        login47::EncryptionRequest {
            server_id: packet.server_id,
            public_key_der: packet.public_key_der,
            verify_token: packet.verify_token,
        }
    }
}

/// `0x01 Encryption Response` (serverbound). java17: `C01PacketEncryptionResponse`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptionResponse {
    /// The AES shared secret, RSA-encrypted (`writeBlob`).
    pub shared_secret_encrypted: Vec<u8>,
    /// The request's verify token, RSA-encrypted (`writeBlob`).
    pub verify_token_encrypted: Vec<u8>,
}

impl Packet for EncryptionResponse {
    const STATE: State = State::Login;
    const ID: i32 = 0x01;

    fn read(buf: &mut PacketBuffer) -> Result<Self, ProtoError> {
        Ok(EncryptionResponse {
            shared_secret_encrypted: buf.read_blob()?,
            verify_token_encrypted: buf.read_blob()?,
        })
    }

    fn write(&self, buf: &mut PacketBuffer) {
        buf.write_blob(&self.shared_secret_encrypted);
        buf.write_blob(&self.verify_token_encrypted);
    }
}

impl From<login47::EncryptionResponse> for EncryptionResponse {
    fn from(packet: login47::EncryptionResponse) -> EncryptionResponse {
        EncryptionResponse {
            shared_secret_encrypted: packet.shared_secret_encrypted,
            verify_token_encrypted: packet.verify_token_encrypted,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blobs_are_short_prefixed() {
        let packet = EncryptionResponse {
            shared_secret_encrypted: vec![0xAB; 128],
            verify_token_encrypted: vec![0xCD; 128],
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        // 128 is `00 80` here; 1.8.9's VarInt writes `80 01`.
        assert_eq!(&buf.as_slice()[..3], &[0x00, 0x80, 0xAB]);
        assert_eq!(&buf.as_slice()[130..133], &[0x00, 0x80, 0xCD]);
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(EncryptionResponse::read(&mut read).unwrap(), packet);
        assert!(read.is_empty());
    }

    #[test]
    fn request_round_trips_and_converts() {
        let packet = EncryptionRequest {
            server_id: String::new(),
            public_key_der: vec![0x30, 0x81, 0x9F],
            verify_token: vec![1, 2, 3, 4],
        };
        let mut buf = PacketBuffer::new();
        packet.write(&mut buf);
        assert_eq!(
            buf.as_slice(),
            &[0x00, 0x00, 0x03, 0x30, 0x81, 0x9F, 0x00, 0x04, 1, 2, 3, 4]
        );
        let mut read = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(EncryptionRequest::read(&mut read).unwrap(), packet);
        let v47: login47::EncryptionRequest = packet.clone().into();
        assert_eq!(v47.public_key_der, packet.public_key_der);
        assert_eq!(v47.verify_token, packet.verify_token);
    }

    #[test]
    fn negative_blob_length_is_rejected() {
        let mut read = PacketBuffer::from_slice(&[0x00, 0xFF, 0xFF]);
        assert!(EncryptionRequest::read(&mut read).is_err());
    }
}
