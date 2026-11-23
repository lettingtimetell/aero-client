//! A deliberately **independent** implementation of the three transport layers S07 owns, for
//! use as the far side of an integration test.
//!
//! The point of this file is that it shares no code with `mc_protocol::{crypt, compress,
//! connection, codec}`:
//!
//! * **CFB8** is the textbook shift-register loop written directly over `aes::Aes128`'s raw
//!   block-encrypt primitive, not the `cfb8` crate the client uses.
//! * **VarInt framing** is hand-written here rather than reusing `PacketBuffer`.
//! * **zlib** goes through `flate2`'s `Read`/`Write` adapters, not the low-level
//!   `Compress`/`Decompress` API `mc_protocol::compress` drives.
//! * **RSA** is a real 1024-bit keypair, and the private half really decrypts what the client
//!   sends.
//!
//! What is *not* independent: the packet bodies themselves (`Handshake`, `LoginStart`,
//! `EncryptionRequest`, ...) are S06's types on both sides. S06's own `tests/golden.rs` pins
//! most of those against hand-derived wire bytes, and `tests/s07_golden.rs` in this crate pins
//! the two encryption packets S06 did not cover, so this is not a blind spot — but it does mean
//! a test passing here is evidence about *layering and cipher continuity*, not about whether a
//! real vanilla server would accept the packet bodies.

#![allow(dead_code)]

use std::io::{Read, Write};

use aes::cipher::{BlockEncrypt, KeyInit};
use aes::{Aes128, Block};
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use flate2::Compression;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// AES-128 CFB8, written from the mode's definition rather than from a crate that implements it.
///
/// Encrypting one byte: run the 16-byte shift register through the block cipher, XOR the
/// plaintext byte with the first output byte, then shift the resulting *ciphertext* byte in at
/// the end of the register. The register is the state, and it persists — that is the whole
/// content of hazard H4.
pub struct RefCfb8 {
    cipher: Aes128,
    register: [u8; 16],
}

impl RefCfb8 {
    /// Minecraft uses the key as the IV as well.
    pub fn new(key: &[u8; 16]) -> RefCfb8 {
        RefCfb8 {
            cipher: Aes128::new(key.into()),
            register: *key,
        }
    }

    fn keystream_byte(&mut self) -> u8 {
        let mut block = Block::clone_from_slice(&self.register);
        self.cipher.encrypt_block(&mut block);
        block[0]
    }

    fn shift_in(&mut self, ciphertext_byte: u8) {
        self.register.copy_within(1..16, 0);
        self.register[15] = ciphertext_byte;
    }

    /// Encrypt in place.
    pub fn encrypt(&mut self, buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            *byte ^= self.keystream_byte();
            self.shift_in(*byte);
        }
    }

    /// Decrypt in place.
    pub fn decrypt(&mut self, buf: &mut [u8]) {
        for byte in buf.iter_mut() {
            let ciphertext = *byte;
            *byte ^= self.keystream_byte();
            self.shift_in(ciphertext);
        }
    }
}

/// Hand-written VarInt writer (protocol-47 framing).
pub fn write_varint(out: &mut Vec<u8>, value: i32) {
    let mut bits = value as u32;
    loop {
        let mut byte = (bits & 0x7F) as u8;
        bits >>= 7;
        if bits != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if bits == 0 {
            return;
        }
    }
}

/// Hand-written VarInt reader; returns the value and how many bytes it used.
pub fn read_varint(bytes: &[u8]) -> (i32, usize) {
    let mut value: i32 = 0;
    for (i, &byte) in bytes.iter().take(5).enumerate() {
        value |= i32::from(byte & 0x7F) << (i * 7);
        if byte & 0x80 == 0 {
            return (value, i + 1);
        }
    }
    panic!("VarInt too long");
}

/// zlib-compress, through `flate2`'s `Write` adapter.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("deflate");
    encoder.finish().expect("deflate finish")
}

/// zlib-decompress, through `flate2`'s `Read` adapter.
pub fn zlib_decompress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    ZlibDecoder::new(data)
        .read_to_end(&mut out)
        .expect("inflate");
    out
}

/// The far end of a connection, speaking the wire format through the independent primitives
/// above. Used as the "server" in the loopback integration test.
pub struct RefPeer {
    stream: TcpStream,
    encrypt: Option<RefCfb8>,
    decrypt: Option<RefCfb8>,
    threshold: Option<usize>,
}

impl RefPeer {
    /// Wrap an accepted socket. Starts unencrypted and uncompressed, like a fresh connection.
    pub fn new(stream: TcpStream) -> RefPeer {
        RefPeer {
            stream,
            encrypt: None,
            decrypt: None,
            threshold: None,
        }
    }

    /// Switch both directions to CFB8 under `secret`, from this point in the byte stream on.
    pub fn enable_encryption(&mut self, secret: &[u8; 16]) {
        self.encrypt = Some(RefCfb8::new(secret));
        self.decrypt = Some(RefCfb8::new(secret));
    }

    /// Install the compression layer at `threshold`.
    pub fn set_compression(&mut self, threshold: usize) {
        self.threshold = Some(threshold);
    }

    async fn read_exact_n(&mut self, n: usize) -> Vec<u8> {
        let mut buf = vec![0u8; n];
        self.stream.read_exact(&mut buf).await.expect("read_exact");
        if let Some(cipher) = self.decrypt.as_mut() {
            cipher.decrypt(&mut buf);
        }
        buf
    }

    async fn read_frame_len(&mut self) -> usize {
        // One byte at a time, because the length prefix's own length is not known in advance.
        // CFB8 is byte-granular, so this is a legal way to consume the stream.
        let mut value: i32 = 0;
        for i in 0..5 {
            let byte = self.read_exact_n(1).await[0];
            value |= i32::from(byte & 0x7F) << (i * 7);
            if byte & 0x80 == 0 {
                return usize::try_from(value).expect("non-negative frame length");
            }
        }
        panic!("frame length VarInt too long");
    }

    /// Read one packet: `(id, body)`, after decryption, framing, and decompression.
    pub async fn read_packet(&mut self) -> (i32, Vec<u8>) {
        let len = self.read_frame_len().await;
        let frame = self.read_exact_n(len).await;

        let payload = match self.threshold {
            None => frame,
            Some(threshold) => {
                let (declared, used) = read_varint(&frame);
                let rest = &frame[used..];
                if declared == 0 {
                    rest.to_vec()
                } else {
                    assert!(
                        declared as usize >= threshold,
                        "peer compressed a {declared}-byte packet under a threshold of {threshold}"
                    );
                    let out = zlib_decompress(rest);
                    assert_eq!(out.len(), declared as usize, "declared size did not match");
                    out
                }
            }
        };
        let (id, used) = read_varint(&payload);
        (id, payload[used..].to_vec())
    }

    /// Send one packet through compress → frame → encrypt, in that order.
    pub async fn send_packet(&mut self, id: i32, body: &[u8]) {
        let mut payload = Vec::new();
        write_varint(&mut payload, id);
        payload.extend_from_slice(body);

        let inner = match self.threshold {
            None => payload,
            Some(threshold) => {
                let mut wrapped = Vec::new();
                if payload.len() < threshold {
                    write_varint(&mut wrapped, 0);
                    wrapped.extend_from_slice(&payload);
                } else {
                    write_varint(&mut wrapped, payload.len() as i32);
                    wrapped.extend_from_slice(&zlib_compress(&payload));
                }
                wrapped
            }
        };

        let mut framed = Vec::new();
        write_varint(&mut framed, inner.len() as i32);
        framed.extend_from_slice(&inner);
        if let Some(cipher) = self.encrypt.as_mut() {
            cipher.encrypt(&mut framed);
        }
        self.stream.write_all(&framed).await.expect("write");
        self.stream.flush().await.expect("flush");
    }
}
