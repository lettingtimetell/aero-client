//! The zlib compression layer that `S03PacketEnableCompression` switches on.
//!
//! java: `network/NettyCompressionEncoder.java`, `network/NettyCompressionDecoder.java`,
//! `network/NetworkManager.java` (`setCompressionTreshold`),
//! `network/login/server/S03PacketEnableCompression.java`
//!
//! # Where this sits
//!
//! Compression is applied to the frame *payload* and encryption to the framed bytes, so
//! compression is the **inner** layer. See [`crate::connection`] for the full ordering and the
//! `NetworkManager` pipeline it is derived from.
//!
//! # The format
//!
//! Once enabled, every frame payload gains a leading VarInt:
//!
//! | Leading VarInt | Rest of the frame | When |
//! |---|---|---|
//! | `0` | the payload verbatim | uncompressed payload is **shorter than** the threshold |
//! | `n` (the uncompressed length) | zlib stream inflating to exactly `n` bytes | payload is `>=` the threshold |
//!
//! Before it is enabled there is no leading VarInt at all — the handler simply is not in the
//! pipeline. That is why [`CompressionCodec::disabled`] is a distinct state and not "threshold
//! = 0": a threshold of zero means *compress everything, still with the VarInt header*, which is
//! a different wire format from *no compression layer*.

use bytes::BytesMut;
use flate2::{Compress, Compression, Decompress, FlushCompress, FlushDecompress, Status};

use crate::{PacketBuffer, ProtoError};

/// The largest decompressed payload the client will accept, matching vanilla's own ceiling.
/// java: `NettyCompressionDecoder.decode` — "larger than protocol maximum of 2097152"
pub const MAX_DECOMPRESSED_LEN: i32 = 2 * 1024 * 1024;

/// java: `NettyCompressionEncoder` — `private final byte[] buffer = new byte[8192]`
const DEFLATE_SCRATCH: usize = 8192;

/// The compression half of the pipeline: absent, or enabled with a threshold.
///
/// Holds the zlib streams so they are reused across packets, exactly as the Java handlers keep
/// one `Deflater`/`Inflater` each and `reset()` between packets. Unlike the CFB8 ciphers
/// (hazard H4) zlib state here is deliberately *not* carried across packets — each packet is an
/// independent zlib stream — so `reset()` between them is correct, not a bug.
pub struct CompressionCodec {
    /// `None` = the handler is not in the pipeline at all.
    threshold: Option<usize>,
    deflate: Compress,
    inflate: Decompress,
}

impl std::fmt::Debug for CompressionCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompressionCodec")
            .field("threshold", &self.threshold)
            .finish_non_exhaustive()
    }
}

impl Default for CompressionCodec {
    fn default() -> CompressionCodec {
        CompressionCodec::disabled()
    }
}

impl CompressionCodec {
    /// A codec with no compression layer — payloads pass through untouched, with no length
    /// header. This is the state every connection starts in.
    #[must_use]
    pub fn disabled() -> CompressionCodec {
        CompressionCodec {
            threshold: None,
            // `true` selects the zlib container (header + Adler-32 trailer), which is what
            // java.util.zip.Deflater/Inflater produce by default (`nowrap = false`).
            deflate: Compress::new(Compression::default(), true),
            inflate: Decompress::new(true),
        }
    }

    /// A codec that compresses payloads of `threshold` bytes or more.
    #[must_use]
    pub fn with_threshold(threshold: usize) -> CompressionCodec {
        let mut codec = CompressionCodec::disabled();
        codec.threshold = Some(threshold);
        codec
    }

    /// Apply a `SetCompression` threshold. A negative value removes the layer entirely.
    /// java: `NetworkManager.setCompressionTreshold` — `treshold >= 0` adds or updates the two
    /// handlers, anything else removes them.
    pub fn set_threshold(&mut self, threshold: i32) {
        self.threshold = usize::try_from(threshold).ok();
    }

    /// The active threshold, or `None` when the layer is not installed.
    #[must_use]
    pub fn threshold(&self) -> Option<usize> {
        self.threshold
    }

    /// Whether the compression layer is installed at all.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.threshold.is_some()
    }

    /// Turn a plaintext frame payload (`packet id` VarInt + body) into the bytes that go inside
    /// the length prefix.
    ///
    /// java: `NettyCompressionEncoder.encode`
    ///
    /// # Errors
    ///
    /// [`ProtoError::Zlib`] if deflate fails, which in practice means allocation failure.
    pub fn encode(&mut self, payload: &[u8], out: &mut BytesMut) -> Result<(), ProtoError> {
        let Some(threshold) = self.threshold else {
            out.extend_from_slice(payload);
            return Ok(());
        };

        let mut header = PacketBuffer::with_capacity(5);
        if payload.len() < threshold {
            // java: `packetbuffer.writeVarIntToBuffer(0); packetbuffer.writeBytes(...)`
            header.write_varint(0);
            out.extend_from_slice(header.as_slice());
            out.extend_from_slice(payload);
            return Ok(());
        }

        let len = i32::try_from(payload.len()).map_err(|_| ProtoError::FrameTooLong {
            len: payload.len(),
            max: MAX_DECOMPRESSED_LEN as usize,
        })?;
        header.write_varint(len);
        out.extend_from_slice(header.as_slice());
        self.deflate_into(payload, out)
    }

    /// Recover the plaintext frame payload from the bytes inside the length prefix.
    ///
    /// java: `NettyCompressionDecoder.decode`
    ///
    /// # Errors
    ///
    /// [`ProtoError::Eof`] for an empty frame, [`ProtoError::BadlyCompressed`] if the declared
    /// size is under the negotiated threshold, [`ProtoError::DecompressedTooLarge`] if it is over
    /// the 2 MiB protocol ceiling, and [`ProtoError::Zlib`] for malformed compressed data.
    pub fn decode(&mut self, frame: &[u8]) -> Result<BytesMut, ProtoError> {
        let Some(threshold) = self.threshold else {
            return Ok(BytesMut::from(frame));
        };
        if frame.is_empty() {
            // java drops a zero-length frame silently here (`if (readableBytes() != 0)`), but it
            // carries no packet id and so is undecodable either way; a typed error is more
            // useful than a stalled connection.
            return Err(ProtoError::Eof("compressed frame header"));
        }

        let (declared, rest) = compressed_header(frame)?;

        if declared == 0 {
            return Ok(BytesMut::from(rest));
        }
        if declared < 0 || (declared as usize) < threshold {
            return Err(ProtoError::BadlyCompressed {
                len: declared,
                threshold: threshold as i32,
            });
        }
        if declared > MAX_DECOMPRESSED_LEN {
            return Err(ProtoError::DecompressedTooLarge {
                len: declared,
                max: MAX_DECOMPRESSED_LEN,
            });
        }
        self.inflate_java_sized(rest, declared as usize)
    }

    fn deflate_into(&mut self, input: &[u8], out: &mut BytesMut) -> Result<(), ProtoError> {
        self.deflate.reset();
        let mut scratch = [0u8; DEFLATE_SCRATCH];
        loop {
            let consumed = self.deflate.total_in() as usize;
            let before_out = self.deflate.total_out();
            let status = self
                .deflate
                .compress(&input[consumed..], &mut scratch, FlushCompress::Finish)
                .map_err(|e| ProtoError::Zlib(e.to_string()))?;
            let produced = (self.deflate.total_out() - before_out) as usize;
            out.extend_from_slice(&scratch[..produced]);
            match status {
                Status::StreamEnd => return Ok(()),
                Status::Ok | Status::BufError => {
                    if produced == 0 && self.deflate.total_in() as usize == consumed {
                        // No input consumed and no output produced: without this guard a zlib
                        // stall would spin forever.
                        return Err(ProtoError::Zlib("deflate made no progress".to_owned()));
                    }
                }
            }
        }
    }

    fn inflate_java_sized(
        &mut self,
        input: &[u8],
        declared: usize,
    ) -> Result<BytesMut, ProtoError> {
        self.inflate.reset(true);
        // Java allocates `new byte[declared]`, invokes `Inflater.inflate` once, and ignores the
        // returned byte count. The zero-filled tail therefore survives a short stream, while a
        // stream longer than the fixed output slice is truncated.
        let mut out = BytesMut::zeroed(declared);
        self.inflate
            .decompress(input, &mut out, FlushDecompress::None)
            .map_err(|e| ProtoError::Zlib(e.to_string()))?;
        Ok(out)
    }
}

// Same five-group decoding, signed truncation and error as PacketBuffer::read_varint,
// with a borrowed remainder instead of two payload-sized intermediate allocations.
fn compressed_header(frame: &[u8]) -> Result<(i32, &[u8]), ProtoError> {
    let mut value = 0i32;
    for index in 0..5 {
        let byte = *frame.get(index).ok_or(ProtoError::Eof("a u8"))?;
        value |= i32::from(byte & 0x7f) << (index * 7);
        if byte & 0x80 == 0 {
            return Ok((value, &frame[index + 1..]));
        }
    }
    Err(ProtoError::VarIntTooBig)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_header_matches_owned_parser_for_noncanonical_and_truncated_inputs() {
        let mut random = 0x19275439u64;
        for len in 0..32 {
            for _ in 0..512 {
                let bytes: Vec<_> = (0..len)
                    .map(|_| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        random as u8
                    })
                    .collect();
                let mut owned = PacketBuffer::from_slice(&bytes);
                match (owned.read_varint(), compressed_header(&bytes)) {
                    (Ok(a), Ok((b, rest))) => {
                        assert_eq!(a, b);
                        assert_eq!(owned.read_remaining(), rest);
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string()),
                    pair => panic!("different header interpretation: {pair:?}"),
                }
            }
        }
    }

    fn encode(codec: &mut CompressionCodec, payload: &[u8]) -> BytesMut {
        let mut out = BytesMut::new();
        codec.encode(payload, &mut out).expect("encode");
        out
    }

    #[test]
    fn disabled_is_a_pass_through_with_no_header() {
        let mut codec = CompressionCodec::disabled();
        assert!(!codec.is_enabled());
        let framed = encode(&mut codec, b"\x00hello");
        assert_eq!(&framed[..], b"\x00hello");
        assert_eq!(&codec.decode(&framed).unwrap()[..], b"\x00hello");
    }

    #[test]
    fn below_threshold_is_a_zero_varint_then_the_payload() {
        let mut codec = CompressionCodec::with_threshold(256);
        let framed = encode(&mut codec, b"short");
        assert_eq!(&framed[..], b"\x00short");
        assert_eq!(&codec.decode(&framed).unwrap()[..], b"short");
    }

    #[test]
    fn marker_only_zero_varint_decodes_to_an_empty_payload() {
        let mut codec = CompressionCodec::with_threshold(256);
        assert!(codec.decode(&[0]).unwrap().is_empty());
    }

    #[test]
    fn at_the_threshold_it_compresses() {
        // java: `if (i < this.treshold)` — so exactly-at-threshold compresses.
        let mut codec = CompressionCodec::with_threshold(8);
        let framed = encode(&mut codec, b"AAAAAAAA");
        assert_eq!(
            framed[0], 8,
            "leading VarInt must be the uncompressed length"
        );
        assert_ne!(&framed[1..], b"AAAAAAAA", "payload should be deflated");
        assert_eq!(&codec.decode(&framed).unwrap()[..], b"AAAAAAAA");

        let framed = encode(&mut codec, b"AAAAAAA");
        assert_eq!(framed[0], 0, "one byte under the threshold stays raw");
    }

    #[test]
    fn round_trips_a_large_payload() {
        let mut codec = CompressionCodec::with_threshold(256);
        let payload: Vec<u8> = (0..200_000u32).map(|i| (i % 97) as u8).collect();
        let framed = encode(&mut codec, &payload);
        assert!(
            framed.len() < payload.len() / 4,
            "repetitive data should compress well, got {} bytes",
            framed.len()
        );
        assert_eq!(&codec.decode(&framed).unwrap()[..], &payload[..]);
    }

    /// The zlib streams are reused between packets; `reset()` must actually reset them.
    #[test]
    fn reused_codec_round_trips_many_packets() {
        let mut codec = CompressionCodec::with_threshold(4);
        for n in 0..50usize {
            let payload: Vec<u8> = (0..n * 37).map(|i| (i % 251) as u8).collect();
            let framed = encode(&mut codec, &payload);
            assert_eq!(
                &codec.decode(&framed).unwrap()[..],
                &payload[..],
                "packet {n}"
            );
        }
    }

    /// Threshold changes mid-connection are legal — a server may send `SetCompression` more than
    /// once, and vanilla just updates the existing handlers.
    #[test]
    fn threshold_can_be_changed_and_removed() {
        let mut codec = CompressionCodec::disabled();
        codec.set_threshold(64);
        assert_eq!(codec.threshold(), Some(64));
        codec.set_threshold(0);
        assert_eq!(codec.threshold(), Some(0));
        // Threshold 0 compresses even an empty payload — and still writes the header.
        let framed = encode(&mut codec, b"");
        assert_eq!(
            framed[0], 0,
            "an empty payload has length 0, the raw marker"
        );
        codec.set_threshold(-1);
        assert_eq!(codec.threshold(), None);
        assert!(!codec.is_enabled());
    }

    #[test]
    fn declared_size_below_the_threshold_is_rejected() {
        // A server that compresses something it should have sent raw. java throws
        // "Badly compressed packet - size of N is below server threshold of T".
        let mut sender = CompressionCodec::with_threshold(4);
        let framed = encode(&mut sender, b"AAAAAAAAAAAA");
        let mut strict = CompressionCodec::with_threshold(64);
        assert!(matches!(
            strict.decode(&framed),
            Err(ProtoError::BadlyCompressed {
                len: 12,
                threshold: 64
            })
        ));
    }

    #[test]
    fn absurd_declared_size_is_rejected_before_allocating() {
        let mut codec = CompressionCodec::with_threshold(4);
        let mut frame = PacketBuffer::new();
        frame.write_varint(MAX_DECOMPRESSED_LEN + 1);
        frame.write_bytes(&[0x78, 0x9c, 0x03, 0x00, 0x00, 0x00, 0x00, 0x01]);
        assert!(matches!(
            codec.decode(frame.as_slice()),
            Err(ProtoError::DecompressedTooLarge { .. })
        ));
    }

    #[test]
    fn negative_declared_size_is_rejected() {
        let mut codec = CompressionCodec::with_threshold(4);
        let mut frame = PacketBuffer::new();
        frame.write_varint(-5);
        frame.write_bytes(&[0x78, 0x9c]);
        assert!(matches!(
            codec.decode(frame.as_slice()),
            Err(ProtoError::BadlyCompressed { len: -5, .. })
        ));
    }

    #[test]
    fn a_declared_size_larger_than_the_stream_zero_pads_like_java() {
        let mut codec = CompressionCodec::with_threshold(4);
        let mut real = BytesMut::new();
        codec.encode(b"AAAAAAAAAAAAAAAA", &mut real).unwrap();
        // Rewrite the leading VarInt to claim one byte more than the stream really holds.
        let mut forged = PacketBuffer::new();
        forged.write_varint(17);
        forged.write_bytes(&real[1..]);
        assert_eq!(
            &codec.decode(forged.as_slice()).unwrap()[..],
            b"AAAAAAAAAAAAAAAA\0"
        );
    }

    #[test]
    fn a_declared_size_smaller_than_the_stream_truncates_like_java() {
        let mut codec = CompressionCodec::with_threshold(4);
        let mut real = BytesMut::new();
        codec.encode(b"ABCDEFGHIJKLMNOP", &mut real).unwrap();
        // Rewrite the leading VarInt to provide a fixed output buffer four bytes too short.
        let mut forged = PacketBuffer::new();
        forged.write_varint(12);
        forged.write_bytes(&real[1..]);
        assert_eq!(
            &codec.decode(forged.as_slice()).unwrap()[..],
            b"ABCDEFGHIJKL"
        );
    }

    #[test]
    fn garbage_zlib_stream_is_an_error_not_a_panic() {
        let mut codec = CompressionCodec::with_threshold(4);
        let mut frame = PacketBuffer::new();
        frame.write_varint(64);
        frame.write_bytes(&[0xFF; 32]);
        assert!(matches!(
            codec.decode(frame.as_slice()),
            Err(ProtoError::Zlib(_))
        ));
    }

    #[test]
    fn empty_frame_is_an_error_when_enabled() {
        let mut codec = CompressionCodec::with_threshold(4);
        assert!(matches!(codec.decode(&[]), Err(ProtoError::Eof(_))));
    }

    #[test]
    fn truncated_varint_header_is_an_error() {
        let mut codec = CompressionCodec::with_threshold(4);
        assert!(codec.decode(&[0x80]).is_err());
    }
}
