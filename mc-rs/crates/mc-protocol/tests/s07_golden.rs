//! Golden wire tests for S07: the two encryption packets S06's `golden.rs` did not cover, the
//! compressed-frame format, and — the reason this file exists — the **order** the compression
//! and encryption layers are applied in.
//!
//! Every expectation here is derived from the protocol-47 format independently of the code
//! under test: packet bytes are written out literally, CFB8 comes from
//! [`common::RefCfb8`] (the textbook shift-register loop over raw AES), and zlib goes through
//! `flate2`'s `Read`/`Write` adapters rather than the `Compress`/`Decompress` API
//! `mc_protocol::compress` uses.
//!
//! **What this file does not prove:** that a real vanilla 1.8.9 server accepts these bytes. No
//! server and no network are available in this environment (same limitation S06 recorded). It
//! proves the bytes match the documented format and that the layers compose in the order
//! `NetworkManager`'s pipeline composes them.

mod common;

use bytes::BytesMut;
use common::{read_varint, write_varint, zlib_compress, zlib_decompress, RefCfb8};
use mc_protocol::compress::CompressionCodec;
use mc_protocol::crypt::{EncryptStream, SharedSecret};
use mc_protocol::login::{EncryptionRequest, EncryptionResponse};
use mc_protocol::packet::Packet;
use mc_protocol::{Connection, PacketBuffer};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

fn encode<P: Packet>(packet: &P) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    packet.write(&mut buf);
    buf.as_slice().to_vec()
}

/// `S01PacketEncryptionRequest` = String(20) + VarInt-prefixed bytes + VarInt-prefixed bytes.
/// java: `S01PacketEncryptionRequest.writePacketData`
#[test]
fn encryption_request_matches_hand_derived_bytes() {
    let packet = EncryptionRequest {
        server_id: "ServerId".to_owned(),
        public_key_der: vec![0x30, 0x81, 0x9F, 0x00],
        verify_token: vec![0xDE, 0xAD, 0xBE, 0xEF],
    };

    let mut expected = Vec::new();
    expected.push(8); // VarInt string length
    expected.extend_from_slice(b"ServerId");
    expected.push(4); // VarInt array length
    expected.extend_from_slice(&[0x30, 0x81, 0x9F, 0x00]);
    expected.push(4);
    expected.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

    assert_eq!(encode(&packet), expected);

    let mut buf = PacketBuffer::from_slice(&expected);
    assert_eq!(EncryptionRequest::read(&mut buf).unwrap(), packet);
    assert!(buf.is_empty());
}

/// An empty server id — what vanilla 1.8.9 servers usually send — is a single zero byte, not an
/// omitted field.
#[test]
fn encryption_request_with_an_empty_server_id() {
    let packet = EncryptionRequest {
        server_id: String::new(),
        public_key_der: vec![0xAA],
        verify_token: vec![],
    };
    assert_eq!(encode(&packet), vec![0x00, 0x01, 0xAA, 0x00]);
}

/// `C01PacketEncryptionResponse` = two VarInt-prefixed byte arrays, secret first.
/// java: `C01PacketEncryptionResponse.writePacketData`
#[test]
fn encryption_response_matches_hand_derived_bytes() {
    let packet = EncryptionResponse {
        shared_secret_encrypted: vec![0x11; 128],
        verify_token_encrypted: vec![0x22; 128],
    };

    let mut expected = Vec::new();
    // 128 needs a two-byte VarInt: 0x80 0x01
    expected.extend_from_slice(&[0x80, 0x01]);
    expected.extend_from_slice(&[0x11; 128]);
    expected.extend_from_slice(&[0x80, 0x01]);
    expected.extend_from_slice(&[0x22; 128]);

    assert_eq!(encode(&packet), expected);
    let mut buf = PacketBuffer::from_slice(&expected);
    assert_eq!(EncryptionResponse::read(&mut buf).unwrap(), packet);
}

/// With no compression layer, a frame payload is the packet bytes verbatim — there is no
/// leading VarInt at all. java: the `compress`/`decompress` handlers are simply absent.
#[test]
fn uncompressed_payload_has_no_extra_header() {
    let mut codec = CompressionCodec::disabled();
    let mut out = BytesMut::new();
    codec.encode(&[0x00, 0x2A], &mut out).unwrap();
    assert_eq!(&out[..], &[0x00, 0x2A]);
}

/// Below the threshold: `VarInt(0)` then the payload verbatim.
/// java: `NettyCompressionEncoder.encode` — `if (i < this.treshold)`
#[test]
fn compressed_frame_below_threshold_matches_hand_derived_bytes() {
    let mut codec = CompressionCodec::with_threshold(256);
    let payload = [0x00u8, 0x2A]; // KeepAlive(42)
    let mut out = BytesMut::new();
    codec.encode(&payload, &mut out).unwrap();
    assert_eq!(&out[..], &[0x00, 0x00, 0x2A]);
    assert_eq!(&codec.decode(&out).unwrap()[..], &payload);
}

/// At or above the threshold: `VarInt(uncompressed length)` then a zlib stream. The stream is
/// checked by inflating it with an implementation that shares no code with the encoder.
#[test]
fn compressed_frame_above_threshold_is_a_zlib_stream() {
    let mut codec = CompressionCodec::with_threshold(16);
    let payload: Vec<u8> = std::iter::repeat_n(b'A', 300).collect();
    let mut out = BytesMut::new();
    codec.encode(&payload, &mut out).unwrap();

    let (declared, used) = read_varint(&out);
    assert_eq!(declared, 300, "leading VarInt is the *uncompressed* length");
    // 0x78 is zlib's CMF byte for a 32 KiB window with the deflate method: the container java's
    // `Deflater` produces by default (`nowrap = false`).
    assert_eq!(
        out[used], 0x78,
        "expected a zlib header, not a raw deflate stream"
    );
    assert_eq!(zlib_decompress(&out[used..]), payload);
    assert_eq!(&codec.decode(&out).unwrap()[..], &payload[..]);
}

/// Our CFB8 stream must agree, byte for byte, with the mode written out from its definition —
/// including across arbitrary chunk boundaries (hazard H4).
#[test]
fn cfb8_matches_a_from_scratch_implementation() {
    let key = [0x5Au8; 16];
    let secret = SharedSecret::from_bytes(key);
    let plaintext: Vec<u8> = (0..777u32)
        .map(|i| (i.wrapping_mul(37) % 256) as u8)
        .collect();

    let mut ours = plaintext.clone();
    let mut stream = EncryptStream::new(&secret);
    for chunk in ours.chunks_mut(13) {
        stream.apply(chunk);
    }

    let mut reference = plaintext.clone();
    RefCfb8::new(&key).encrypt(&mut reference);

    assert_eq!(ours, reference);
}

/// **Task 6, at the byte level.** Outbound order is compress → length-prefix → encrypt, so the
/// length prefix is *inside* the ciphertext and the compression header is inside that again.
///
/// Below the threshold there is no zlib in play, so the whole wire image is deterministic and
/// can be asserted byte for byte.
#[tokio::test]
async fn outbound_layers_are_byte_exact_below_the_threshold() {
    let key = [0x11u8; 16];
    let (client_side, mut peer) = tokio::io::duplex(1 << 16);
    let mut conn = Connection::new(client_side);
    conn.enable_encryption(&SharedSecret::from_bytes(key))
        .unwrap();
    conn.set_compression_threshold(256);

    let payload = [0x00u8, 0x2A];
    conn.send_payload(&payload).await.unwrap();

    // Independently: compression header (0, below threshold) + payload, then the frame length
    // prefix over that, then CFB8 over the whole thing.
    let mut inner = Vec::new();
    write_varint(&mut inner, 0);
    inner.extend_from_slice(&payload);
    let mut expected = Vec::new();
    write_varint(&mut expected, inner.len() as i32);
    expected.extend_from_slice(&inner);
    assert_eq!(expected, vec![0x03, 0x00, 0x00, 0x2A]);
    RefCfb8::new(&key).encrypt(&mut expected);

    let mut wire = vec![0u8; expected.len()];
    peer.read_exact(&mut wire).await.unwrap();
    assert_eq!(wire, expected);

    // Sanity: the length prefix really is encrypted, not left in the clear.
    assert_ne!(wire[0], 0x03, "the frame length prefix was not encrypted");
}

/// The same ordering, above the threshold: decrypt the wire bytes with the reference cipher and
/// confirm what falls out is `VarInt(frame len) || VarInt(uncompressed len) || zlib(payload)`.
#[tokio::test]
async fn outbound_layers_put_compression_inside_encryption() {
    let key = [0x33u8; 16];
    let (client_side, mut peer) = tokio::io::duplex(1 << 16);
    let mut conn = Connection::new(client_side);
    conn.enable_encryption(&SharedSecret::from_bytes(key))
        .unwrap();
    conn.set_compression_threshold(16);

    let payload: Vec<u8> = (0..500u32).map(|i| (i % 7) as u8).collect();
    conn.send_payload(&payload).await.unwrap();

    let compressed = zlib_compress(&payload);
    let mut declared = Vec::new();
    write_varint(&mut declared, payload.len() as i32);
    let inner_len = declared.len() + compressed.len();
    let mut prefix = Vec::new();
    write_varint(&mut prefix, inner_len as i32);
    let total = prefix.len() + inner_len;

    let mut wire = vec![0u8; total];
    peer.read_exact(&mut wire).await.unwrap();

    let mut plain = wire.clone();
    RefCfb8::new(&key).decrypt(&mut plain);

    let (frame_len, used) = read_varint(&plain);
    assert_eq!(frame_len as usize, inner_len);
    let frame = &plain[used..];
    let (uncompressed_len, used) = read_varint(frame);
    assert_eq!(uncompressed_len as usize, payload.len());
    assert_eq!(zlib_decompress(&frame[used..]), payload);
}

/// The inbound mirror: bytes built by the reference implementation must decode.
#[tokio::test]
async fn inbound_layers_unwrap_in_the_reverse_order() {
    let key = [0x77u8; 16];
    let (client_side, mut peer) = tokio::io::duplex(1 << 16);
    let mut conn = Connection::new(client_side);
    conn.enable_encryption(&SharedSecret::from_bytes(key))
        .unwrap();
    conn.set_compression_threshold(16);

    let mut cipher = RefCfb8::new(&key);
    let payloads: Vec<Vec<u8>> = vec![
        vec![0x00, 0x2A],                              // below the threshold
        (0..400u32).map(|i| (i % 11) as u8).collect(), // above it
        vec![0x01, 0x02, 0x03],                        // below again
    ];

    // Write all three back to back through one continuous cipher, in a single socket write, so
    // the client has to handle several frames out of one decrypt call.
    let mut wire = Vec::new();
    for payload in &payloads {
        let mut inner = Vec::new();
        if payload.len() < 16 {
            write_varint(&mut inner, 0);
            inner.extend_from_slice(payload);
        } else {
            write_varint(&mut inner, payload.len() as i32);
            inner.extend_from_slice(&zlib_compress(payload));
        }
        let mut framed = Vec::new();
        write_varint(&mut framed, inner.len() as i32);
        framed.extend_from_slice(&inner);
        cipher.encrypt(&mut framed);
        wire.extend_from_slice(&framed);
    }
    peer.write_all(&wire).await.unwrap();
    peer.flush().await.unwrap();

    for payload in &payloads {
        let got = conn.read_payload().await.unwrap();
        assert_eq!(got.as_slice(), &payload[..]);
    }
}

/// A frame split across three socket reads at awkward offsets — mid-length-prefix, mid-body —
/// must reassemble identically. This is the shape TCP actually delivers.
#[tokio::test]
async fn a_frame_split_across_reads_reassembles() {
    let key = [0x99u8; 16];
    let (client_side, mut peer) = tokio::io::duplex(1 << 16);
    let mut conn = Connection::new(client_side);
    conn.enable_encryption(&SharedSecret::from_bytes(key))
        .unwrap();

    let payload: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
    let mut framed = Vec::new();
    write_varint(&mut framed, payload.len() as i32);
    framed.extend_from_slice(&payload);
    RefCfb8::new(&key).encrypt(&mut framed);

    // The length prefix is two bytes here; split inside it.
    let reader = tokio::spawn(async move {
        let got = conn.read_payload().await.unwrap();
        got.as_slice().to_vec()
    });
    for piece in [&framed[..1], &framed[1..5], &framed[5..]] {
        peer.write_all(piece).await.unwrap();
        peer.flush().await.unwrap();
        tokio::task::yield_now().await;
    }
    assert_eq!(reader.await.unwrap(), payload);
}
