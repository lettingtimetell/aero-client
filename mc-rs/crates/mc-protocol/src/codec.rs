//! The length-prefixed frame codec: `tokio_util::codec::{Decoder, Encoder}` over VarInt-length
//! frames.
//!
//! This is *only* the outer framing layer — each frame's payload is `packet_id: VarInt` then
//! the packet body, undecoded. Splitting the id out and dispatching to a concrete packet type
//! is [`crate::play::S2CPlay`]/[`crate::play::C2SPlay`]'s job; applying encryption (AES-CFB8)
//! and compression (S07) happens *around* this codec, not inside it — see hazard H4 (CFB8 must
//! be one continuous stream, so it cannot be re-derived per frame) and this crate's own doc
//! comment on layering (compression sits inside encryption).
//!
//! java: `NetworkManager`'s pipeline setup (the length-prefix framer has no single dedicated
//! file in the decompiled source; the format itself — one leading VarInt byte count, matching
//! [`PacketBuffer::read_varint`]/[`PacketBuffer::write_varint`] — is protocol 47's framing and
//! is what every packet capture on the wire actually shows).

use bytes::{Buf, BufMut, BytesMut};
use tokio_util::codec::{Decoder, Encoder};

use crate::ProtoError;

/// Largest frame encodable by vanilla's three-byte (`21`-bit) length prefix.
///
/// java: `MessageDeserializer2.decode` reads exactly three prefix bytes before throwing
/// `CorruptedFrameException("length wider than 21-bit")`; `MessageSerializer2.encode` rejects
/// lengths for which `PacketBuffer.getVarIntSize` returns more than three.
pub const DEFAULT_MAX_FRAME_LEN: usize = (1 << 21) - 1;

/// Peek a VarInt at the front of `src` without consuming it.
///
/// Returns `Ok(None)` when `src` is a prefix of a would-be-valid VarInt (need more bytes —
/// the [`Decoder`] contract's signal to wait for more data), `Ok(Some((value, byte_len)))`
/// once a complete one is present, and `Err` only for a VarInt that is already malformed
/// regardless of what bytes might follow (three bytes, all with the continuation bit set).
fn peek_varint(src: &[u8]) -> Result<Option<(i32, usize)>, ProtoError> {
    let mut result: i32 = 0;
    for i in 0..3usize {
        let Some(&byte) = src.get(i) else {
            return Ok(None);
        };
        result |= i32::from(byte & 0x7F) << (i * 7);
        if byte & 0x80 == 0 {
            return Ok(Some((result, i + 1)));
        }
    }
    Err(ProtoError::VarIntTooBig)
}

/// A length-prefixed frame `Decoder`/`Encoder`. One frame in ↔ one frame out; a frame's
/// contents are opaque bytes (`packet_id` varint + body), so this codec has no state beyond
/// the configured size ceiling.
#[derive(Debug, Clone, Copy)]
pub struct PacketFrameCodec {
    max_frame_len: usize,
}

impl PacketFrameCodec {
    /// A codec with [`DEFAULT_MAX_FRAME_LEN`].
    #[must_use]
    pub fn new() -> PacketFrameCodec {
        PacketFrameCodec {
            max_frame_len: DEFAULT_MAX_FRAME_LEN,
        }
    }

    /// A codec with an explicit frame-size ceiling, capped at vanilla's 21-bit protocol limit.
    #[must_use]
    pub fn with_max_frame_len(max_frame_len: usize) -> PacketFrameCodec {
        PacketFrameCodec {
            max_frame_len: max_frame_len.min(DEFAULT_MAX_FRAME_LEN),
        }
    }
}

impl Default for PacketFrameCodec {
    fn default() -> PacketFrameCodec {
        PacketFrameCodec::new()
    }
}

impl Decoder for PacketFrameCodec {
    type Item = BytesMut;
    type Error = ProtoError;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<BytesMut>, ProtoError> {
        let (len, prefix_len) = match peek_varint(&src[..])? {
            Some(v) => v,
            None => return Ok(None),
        };
        if len < 0 {
            return Err(ProtoError::FrameTooLong {
                len: 0,
                max: self.max_frame_len,
            });
        }
        let len = len as usize;
        if len > self.max_frame_len {
            return Err(ProtoError::FrameTooLong {
                len,
                max: self.max_frame_len,
            });
        }
        if src.len() < prefix_len + len {
            // Not a full frame yet; reserve room for the rest so the next `read` can fill it
            // in one shot, and wait for more bytes without disturbing `src`.
            src.reserve(prefix_len + len - src.len());
            return Ok(None);
        }
        src.advance(prefix_len);
        Ok(Some(src.split_to(len)))
    }
}

impl Encoder<BytesMut> for PacketFrameCodec {
    type Error = ProtoError;

    fn encode(&mut self, item: BytesMut, dst: &mut BytesMut) -> Result<(), ProtoError> {
        self.encode(&item[..], dst)
    }
}

impl Encoder<&[u8]> for PacketFrameCodec {
    type Error = ProtoError;

    fn encode(&mut self, item: &[u8], dst: &mut BytesMut) -> Result<(), ProtoError> {
        let len = item.len();
        if len > self.max_frame_len {
            return Err(ProtoError::FrameTooLong {
                len,
                max: self.max_frame_len,
            });
        }
        let prefix_len = crate::PacketBuffer::varint_size(len as i32);
        dst.reserve(prefix_len + len);
        let mut prefix = crate::PacketBuffer::with_capacity(prefix_len);
        prefix.write_varint(len as i32);
        dst.put_slice(prefix.as_slice());
        dst.put_slice(item);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_a_single_frame() {
        let mut codec = PacketFrameCodec::new();
        let mut wire = BytesMut::new();
        codec.encode(&b"hello"[..], &mut wire).unwrap();
        assert_eq!(&wire[..], &[5, b'h', b'e', b'l', b'l', b'o']);

        let frame = codec.decode(&mut wire).unwrap().unwrap();
        assert_eq!(&frame[..], b"hello");
        assert!(wire.is_empty());
    }

    #[test]
    fn waits_for_a_split_frame() {
        let mut codec = PacketFrameCodec::new();
        let mut wire = BytesMut::new();
        codec.encode(&b"hello world"[..], &mut wire).unwrap();

        // Feed it back in two pieces, as a real socket read would.
        let mut partial = wire.split_to(4);
        assert!(codec.decode(&mut partial).unwrap().is_none());
        partial.extend_from_slice(&wire);
        let frame = codec.decode(&mut partial).unwrap().unwrap();
        assert_eq!(&frame[..], b"hello world");
    }

    #[test]
    fn waits_when_even_the_length_prefix_is_split() {
        let mut codec = PacketFrameCodec::new();
        let big = [0xABu8; 200]; // needs a 2-byte VarInt length prefix
        let mut wire = BytesMut::new();
        codec.encode(&big[..], &mut wire).unwrap();

        let mut one_byte = wire.split_to(1);
        assert!(codec.decode(&mut one_byte).unwrap().is_none());
        one_byte.extend_from_slice(&wire);
        let frame = codec.decode(&mut one_byte).unwrap().unwrap();
        assert_eq!(&frame[..], &big[..]);
    }

    #[test]
    fn oversized_frame_is_rejected_without_buffering_it() {
        let mut codec = PacketFrameCodec::with_max_frame_len(10);
        let mut wire = BytesMut::new();
        wire.put_u8(20); // claims a 20-byte frame, over the 10-byte cap
        assert!(matches!(
            codec.decode(&mut wire),
            Err(ProtoError::FrameTooLong { len: 20, max: 10 })
        ));
    }

    #[test]
    fn malformed_length_prefix_errors_not_hangs() {
        let mut codec = PacketFrameCodec::new();
        let mut wire = BytesMut::from(&[0xFF, 0xFF, 0xFF][..]);
        assert!(matches!(
            codec.decode(&mut wire),
            Err(ProtoError::VarIntTooBig)
        ));
    }

    #[test]
    fn waits_for_an_incomplete_valid_three_byte_prefix() {
        let mut codec = PacketFrameCodec::new();
        let mut wire = BytesMut::from(&[0x80, 0x80][..]);
        assert!(codec.decode(&mut wire).unwrap().is_none());
        assert_eq!(&wire[..], &[0x80, 0x80]);
    }

    #[test]
    fn rejects_a_four_byte_length_prefix_even_when_its_value_is_small() {
        let mut codec = PacketFrameCodec::new();
        // An overlong encoding of zero. PacketBuffer's general VarInt reader accepts it, but
        // vanilla's outer frame splitter rejects after the first three continuation bytes.
        let mut wire = BytesMut::from(&[0x80, 0x80, 0x80, 0x00][..]);
        assert!(matches!(
            codec.decode(&mut wire),
            Err(ProtoError::VarIntTooBig)
        ));
    }

    #[test]
    fn explicit_ceiling_cannot_exceed_vanillas_three_byte_limit() {
        let mut codec = PacketFrameCodec::with_max_frame_len(usize::MAX);
        let oversized = vec![0_u8; DEFAULT_MAX_FRAME_LEN + 1];
        let mut wire = BytesMut::new();
        assert!(matches!(
            codec.encode(&oversized[..], &mut wire),
            Err(ProtoError::FrameTooLong {
                len,
                max: DEFAULT_MAX_FRAME_LEN
            }) if len == DEFAULT_MAX_FRAME_LEN + 1
        ));
        assert!(wire.is_empty());
    }
}
