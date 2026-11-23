//! [`PacketBuffer`]: the wire codec every packet type is built from.
//!
//! java: `network/PacketBuffer.java` (a thin decorator over Netty's `ByteBuf`). This port keeps
//! the same primitive set — VarInt/VarLong, length-prefixed UTF-8 strings, UUIDs, the packed
//! `Position` (hazard H5), and length-delimited NBT/`ItemStack` — but backs it with `bytes`
//! instead of Netty, and returns [`ProtoError`] instead of throwing, since every read is
//! driven by untrusted server bytes (§11).

use bytes::{Buf, BufMut, BytesMut};
use mc_math::BlockPos;
use mc_nbt::NbtCompound;
use uuid::Uuid;

use crate::item::ItemStack;
use crate::ProtoError;

/// A yaw/pitch-style angle, stored on the wire as a single byte: `256` steps per full turn.
/// java: e.g. `S18PacketEntityTeleport` — `(byte)((int)(yaw * 256.0F / 360.0F))`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Angle(pub u8);

impl Angle {
    /// Encode a degree value the way vanilla does: truncate (not round) to a step, then narrow
    /// to a byte (wrapping, matching Java's `(byte)` cast).
    #[must_use]
    pub fn from_degrees(degrees: f32) -> Angle {
        // java: float->int cast truncates toward zero; Rust's `as i32` on a finite f32 does the
        // same. A `(byte)` narrowing conversion in Java keeps the low 8 bits, which is exactly
        // what `as u8` does when going through `i32` first.
        let truncated = if degrees.is_finite() {
            (degrees * 256.0 / 360.0) as i32
        } else {
            0
        };
        Angle(truncated as u8)
    }

    /// Decode back to degrees with vanilla's signed-byte semantics, in `[-180, 180)`.
    ///
    /// Java packet angle accessors return `byte`, so consumers such as
    /// `NetHandlerPlayClient.handleEntityTeleport` calculate
    /// `(float)(packetIn.getYaw() * 360) / 256.0F` from a signed value. Treating the wire octet as
    /// an unsigned integer instead shifts every value from `0x80..=0xff` by 360 degrees. Yaw can
    /// often hide that behind wrapping, but vanilla deliberately does not wrap pitch while easing
    /// it, making the distinction observable.
    #[must_use]
    pub fn to_degrees(self) -> f32 {
        f32::from(self.0 as i8) * 360.0 / 256.0
    }
}

/// A growable, cursor-based buffer for encoding/decoding one packet's payload.
///
/// Reading consumes from the front (mirroring `ByteBuf`'s `readerIndex`); writing appends to
/// the end. All multi-byte integers are big-endian, matching Java's `DataOutput`/`DataInput`
/// (and Netty's default `ByteBuf` order).
#[derive(Debug, Default, Clone)]
pub struct PacketBuffer {
    inner: BytesMut,
    write_error: Option<DeferredWriteError>,
}

#[derive(Debug, Clone)]
enum DeferredWriteError {
    Nbt(String),
    PayloadTooLong { len: usize, max: usize },
    StringTooLong { len: i32, max: i32 },
}

/// A minimal `std::io::Read` adapter over the unread tail of a [`PacketBuffer`], for handing to
/// `mc_nbt::read*`. Never returns an `Err`; a short read (including a zero-byte read at EOF) is
/// reported the ordinary `Read` way, which `mc_nbt` already treats as `UnexpectedEof`.
struct TailReader<'a>(&'a mut BytesMut);

impl std::io::Read for TailReader<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        let n = out.len().min(self.0.remaining());
        self.0.copy_to_slice(&mut out[..n]);
        Ok(n)
    }
}

/// Decode UTF-8 with the replacement grouping used by Java's `new String(bytes, UTF_8)`.
///
/// `String::from_utf8_lossy` is almost identical, but it treats an encoded UTF-16 surrogate
/// (`ED A0 80` through `ED BF BF`) as three independent malformed bytes. Java's JDK UTF-8
/// decoder recognizes the complete three-byte shape and emits one replacement character. The
/// distinction matters because [`PacketBuffer::read_string`] applies its limit after decoding:
/// a one-UTF-16-unit Java result must not become a three-unit Rust rejection.
///
/// The malformed-prefix lengths below mirror `sun.nio.cs.UTF_8$Decoder`: a valid lead plus one
/// or more valid continuation bytes is replaced as one unit when the next byte is malformed;
/// invalid lead/range bytes are left for the next iteration. This is intentionally a fallback
/// only; valid input takes `str::from_utf8`'s optimized path.
fn java_utf8_lossy(bytes: &[u8]) -> String {
    if let Ok(value) = std::str::from_utf8(bytes) {
        return value.to_owned();
    }

    const REPLACEMENT: char = '\u{FFFD}';
    let mut out = String::with_capacity(bytes.len());
    let mut index = 0;

    while index < bytes.len() {
        let first = bytes[index];

        if first < 0x80 {
            // The fast path above already rejected the whole input, but ASCII bytes around a
            // malformed span remain ordinary one-byte Java code points.
            out.push(char::from(first));
            index += 1;
            continue;
        }

        // Java's two-byte branch recognizes only C2..DF. C0/C1 are overlong lead bytes and are
        // each replaced independently rather than consuming their following continuation.
        if (0xC2..=0xDF).contains(&first) {
            if index + 1 >= bytes.len() {
                out.push(REPLACEMENT);
                index += 1;
                continue;
            }
            let second = bytes[index + 1];
            if !is_utf8_continuation(second) {
                out.push(REPLACEMENT);
                index += 1;
                continue;
            }
            let code_point = (u32::from(first & 0x1F) << 6) | u32::from(second & 0x3F);
            // C2..DF plus a continuation is always a Unicode scalar.
            out.push(char::from_u32(code_point).unwrap_or(REPLACEMENT));
            index += 2;
            continue;
        }

        // Java's three-byte decoder checks the second and third byte separately, then reports a
        // malformed prefix length of one or two. A complete surrogate-shaped sequence is handled
        // after composition and therefore reports one replacement for all three bytes.
        if (0xE0..=0xEF).contains(&first) {
            let remaining = bytes.len() - index;
            if remaining < 3 {
                if remaining > 1
                    && is_utf8_continuation(bytes[index + 1])
                    && !(first == 0xE0 && bytes[index + 1] < 0xA0)
                {
                    // Java's decoder reaches end-of-input with a valid two-byte prefix and
                    // replaces the entire underflowing prefix.
                    out.push(REPLACEMENT);
                    index += remaining;
                } else if remaining > 1
                    && (first == 0xE0 && bytes[index + 1] < 0xA0
                        || !is_utf8_continuation(bytes[index + 1]))
                {
                    // The second byte itself is malformed; only the lead belongs to this
                    // malformed result. The next iteration handles the second byte.
                    out.push(REPLACEMENT);
                    index += 1;
                } else {
                    out.push(REPLACEMENT);
                    index += remaining;
                }
                continue;
            }

            let second = bytes[index + 1];
            let third = bytes[index + 2];
            let second_is_valid = is_utf8_continuation(second) && !(first == 0xE0 && second < 0xA0);
            if !second_is_valid {
                out.push(REPLACEMENT);
                index += 1;
                continue;
            }
            if !is_utf8_continuation(third) {
                out.push(REPLACEMENT);
                index += 2;
                continue;
            }

            let code_point = (u32::from(first & 0x0F) << 12)
                | (u32::from(second & 0x3F) << 6)
                | u32::from(third & 0x3F);
            if (0xD800..=0xDFFF).contains(&code_point) {
                // Java's decoder calls malformedForLength(3) after detecting the surrogate.
                out.push(REPLACEMENT);
            } else {
                out.push(char::from_u32(code_point).unwrap_or(REPLACEMENT));
            }
            index += 3;
            continue;
        }

        // Java's four-byte decoder has the same valid-prefix grouping, with F0/F4 boundary
        // checks and a four-byte malformed prefix reduced to one, two, or three bytes when a
        // later byte is the first malformed member.
        if (0xF0..=0xF4).contains(&first) {
            let remaining = bytes.len() - index;
            if remaining < 4 {
                if remaining > 1 && !valid_utf8_four_second(first, bytes[index + 1]) {
                    out.push(REPLACEMENT);
                    index += 1;
                } else if remaining > 2 && !is_utf8_continuation(bytes[index + 2]) {
                    out.push(REPLACEMENT);
                    index += 2;
                } else {
                    out.push(REPLACEMENT);
                    index += remaining;
                }
                continue;
            }

            let second = bytes[index + 1];
            let third = bytes[index + 2];
            let fourth = bytes[index + 3];
            if !valid_utf8_four_second(first, second) {
                out.push(REPLACEMENT);
                index += 1;
                continue;
            }
            if !is_utf8_continuation(third) {
                out.push(REPLACEMENT);
                index += 2;
                continue;
            }
            if !is_utf8_continuation(fourth) {
                out.push(REPLACEMENT);
                index += 3;
                continue;
            }

            let code_point = (u32::from(first & 0x07) << 18)
                | (u32::from(second & 0x3F) << 12)
                | (u32::from(third & 0x3F) << 6)
                | u32::from(fourth & 0x3F);
            out.push(char::from_u32(code_point).unwrap_or(REPLACEMENT));
            index += 4;
            continue;
        }

        // C0/C1, F5..FF, and stray continuation bytes are malformed one byte at a time in the
        // JDK decoder.
        out.push(REPLACEMENT);
        index += 1;
    }

    out
}

#[inline]
fn is_utf8_continuation(byte: u8) -> bool {
    (byte & 0xC0) == 0x80
}

#[inline]
fn valid_utf8_four_second(first: u8, second: u8) -> bool {
    is_utf8_continuation(second)
        && !(first == 0xF0 && second < 0x90)
        && !(first == 0xF4 && second > 0x8F)
}

impl PacketBuffer {
    /// An empty buffer, ready to write into.
    #[must_use]
    pub fn new() -> PacketBuffer {
        PacketBuffer::default()
    }

    /// An empty buffer with room reserved for `capacity` bytes before it needs to reallocate.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> PacketBuffer {
        PacketBuffer {
            inner: BytesMut::with_capacity(capacity),
            write_error: None,
        }
    }

    /// Wrap already-framed bytes for decoding (e.g. one packet's payload, split out by
    /// [`crate::codec`]).
    #[must_use]
    pub fn from_bytes(bytes: BytesMut) -> PacketBuffer {
        PacketBuffer {
            inner: bytes,
            write_error: None,
        }
    }

    /// Copy a byte slice into a fresh decodable buffer. Convenience for tests and golden
    /// fixtures; hot paths should prefer [`PacketBuffer::from_bytes`] to avoid the copy.
    #[must_use]
    pub fn from_slice(data: &[u8]) -> PacketBuffer {
        PacketBuffer {
            inner: BytesMut::from(data),
            write_error: None,
        }
    }

    /// Bytes not yet consumed by a `read_*` call.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.inner.remaining()
    }

    /// `true` once every written byte has been read back.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The unread tail, without consuming it.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.inner[..]
    }

    /// Everything written so far, unread and read alike. Mainly for the frame codec, which
    /// needs the fully-encoded payload to prefix with a length.
    #[must_use]
    pub fn into_inner(self) -> BytesMut {
        self.inner
    }

    /// Report a deferred write-side encoding error before a packet is framed or sent.
    ///
    /// [`Packet::write`](crate::Packet::write) is intentionally infallible, so compound NBT
    /// encoding and packet-owned payload ceilings record the equivalent of Netty's encoder
    /// exception here. Every production packet-send path checks this method before observing
    /// or writing the payload.
    pub fn finish_write(&self) -> Result<(), ProtoError> {
        match &self.write_error {
            Some(DeferredWriteError::Nbt(message)) => Err(ProtoError::NbtEncode(message.clone())),
            Some(DeferredWriteError::PayloadTooLong { len, max }) => {
                Err(ProtoError::PayloadTooLong {
                    len: *len,
                    max: *max,
                })
            }
            Some(DeferredWriteError::StringTooLong { len, max }) => {
                Err(ProtoError::StringTooLong {
                    len: *len,
                    max: *max,
                })
            }
            None => Ok(()),
        }
    }

    /// Append an unprefixed packet-owned payload, deferring a typed error when it exceeds its
    /// field-specific ceiling. This preserves [`Packet::write`](crate::Packet::write)'s
    /// infallible signature while ensuring every production send rejects the packet at
    /// [`Self::finish_write`] before framing any partial bytes.
    pub(crate) fn write_bounded_payload(&mut self, data: &[u8], max: usize) {
        if self.write_error.is_some() {
            return;
        }
        if data.len() > max {
            self.write_error = Some(DeferredWriteError::PayloadTooLong {
                len: data.len(),
                max,
            });
            return;
        }
        self.write_bytes(data);
    }

    fn need(&self, n: usize, what: &'static str) -> Result<(), ProtoError> {
        if self.inner.remaining() < n {
            Err(ProtoError::Eof(what))
        } else {
            Ok(())
        }
    }

    // ---- fixed-width primitives ------------------------------------------------------------

    /// Read one byte as a `bool`. java: `readBoolean` — any nonzero byte is `true`, matching
    /// `ByteBuf.readBoolean`'s `readByte() != 0`; unlike some formats, a byte other than 0/1 is
    /// *not* an error.
    pub fn read_bool(&mut self) -> Result<bool, ProtoError> {
        Ok(self.read_u8()? != 0)
    }

    /// Write a `bool` as one byte (`0`/`1`).
    pub fn write_bool(&mut self, v: bool) {
        self.write_u8(u8::from(v));
    }

    /// Read one unsigned byte.
    pub fn read_u8(&mut self) -> Result<u8, ProtoError> {
        self.need(1, "a u8")?;
        Ok(self.inner.get_u8())
    }

    /// Write one unsigned byte.
    pub fn write_u8(&mut self, v: u8) {
        self.inner.put_u8(v);
    }

    /// Read one signed byte.
    pub fn read_i8(&mut self) -> Result<i8, ProtoError> {
        Ok(self.read_u8()? as i8)
    }

    /// Write one signed byte.
    pub fn write_i8(&mut self, v: i8) {
        self.write_u8(v as u8);
    }

    /// Read a big-endian unsigned 16-bit integer.
    pub fn read_u16(&mut self) -> Result<u16, ProtoError> {
        self.need(2, "a u16")?;
        Ok(self.inner.get_u16())
    }

    /// Write a big-endian unsigned 16-bit integer.
    pub fn write_u16(&mut self, v: u16) {
        self.inner.put_u16(v);
    }

    /// Read a big-endian signed 16-bit integer.
    pub fn read_i16(&mut self) -> Result<i16, ProtoError> {
        self.need(2, "an i16")?;
        Ok(self.inner.get_i16())
    }

    /// Write a big-endian signed 16-bit integer.
    pub fn write_i16(&mut self, v: i16) {
        self.inner.put_i16(v);
    }

    /// Read a big-endian unsigned 32-bit integer.
    pub fn read_u32(&mut self) -> Result<u32, ProtoError> {
        self.need(4, "a u32")?;
        Ok(self.inner.get_u32())
    }

    /// Write a big-endian unsigned 32-bit integer.
    pub fn write_u32(&mut self, v: u32) {
        self.inner.put_u32(v);
    }

    /// Read a big-endian signed 32-bit integer.
    pub fn read_i32(&mut self) -> Result<i32, ProtoError> {
        self.need(4, "an i32")?;
        Ok(self.inner.get_i32())
    }

    /// Write a big-endian signed 32-bit integer.
    pub fn write_i32(&mut self, v: i32) {
        self.inner.put_i32(v);
    }

    /// Read a big-endian unsigned 64-bit integer.
    pub fn read_u64(&mut self) -> Result<u64, ProtoError> {
        self.need(8, "a u64")?;
        Ok(self.inner.get_u64())
    }

    /// Write a big-endian unsigned 64-bit integer.
    pub fn write_u64(&mut self, v: u64) {
        self.inner.put_u64(v);
    }

    /// Read a big-endian signed 64-bit integer.
    pub fn read_i64(&mut self) -> Result<i64, ProtoError> {
        self.need(8, "an i64")?;
        Ok(self.inner.get_i64())
    }

    /// Write a big-endian signed 64-bit integer.
    pub fn write_i64(&mut self, v: i64) {
        self.inner.put_i64(v);
    }

    /// Read a big-endian IEEE-754 `f32`.
    pub fn read_f32(&mut self) -> Result<f32, ProtoError> {
        self.need(4, "an f32")?;
        Ok(self.inner.get_f32())
    }

    /// Write a big-endian IEEE-754 `f32`.
    pub fn write_f32(&mut self, v: f32) {
        self.inner.put_f32(v);
    }

    /// Read a big-endian IEEE-754 `f64`.
    pub fn read_f64(&mut self) -> Result<f64, ProtoError> {
        self.need(8, "an f64")?;
        Ok(self.inner.get_f64())
    }

    /// Write a big-endian IEEE-754 `f64`.
    pub fn write_f64(&mut self, v: f64) {
        self.inner.put_f64(v);
    }

    // ---- VarInt / VarLong ---------------------------------------------------------------

    /// The number of bytes a VarInt encoding of `input` would take (1-5).
    /// java: `PacketBuffer.getVarIntSize`
    #[must_use]
    pub fn varint_size(input: i32) -> usize {
        for i in 1..5 {
            if (input & (-1i32 << (i * 7))) == 0 {
                return i as usize;
            }
        }
        5
    }

    /// Read a VarInt: up to 5 little-endian-ordered 7-bit groups, high bit of each byte marking
    /// continuation. java: `readVarIntFromBuffer` — a 6th continuation byte is a hard error
    /// even before its payload bits are looked at (5 groups already cover all 32 bits).
    pub fn read_varint(&mut self) -> Result<i32, ProtoError> {
        let mut result: i32 = 0;
        for i in 0..5u32 {
            let byte = self.read_u8()?;
            result |= i32::from(byte & 0x7F) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(ProtoError::VarIntTooBig)
    }

    /// Write a VarInt. java: `writeVarIntToBuffer`
    pub fn write_varint(&mut self, value: i32) {
        let mut v = value as u32;
        loop {
            if v & !0x7Fu32 == 0 {
                self.write_u8(v as u8);
                return;
            }
            self.write_u8(((v & 0x7F) | 0x80) as u8);
            v >>= 7;
        }
    }

    /// Read a VarLong: up to 10 groups. java: `readVarLong`
    pub fn read_varlong(&mut self) -> Result<i64, ProtoError> {
        let mut result: i64 = 0;
        for i in 0..10u32 {
            let byte = self.read_u8()?;
            result |= i64::from(byte & 0x7F) << (i * 7);
            if byte & 0x80 == 0 {
                return Ok(result);
            }
        }
        Err(ProtoError::VarLongTooBig)
    }

    /// Write a VarLong. java: `writeVarLong`
    pub fn write_varlong(&mut self, value: i64) {
        let mut v = value as u64;
        loop {
            if v & !0x7Fu64 == 0 {
                self.write_u8(v as u8);
                return;
            }
            self.write_u8(((v & 0x7F) | 0x80) as u8);
            v >>= 7;
        }
    }

    // ---- strings, chat, byte arrays -------------------------------------------------------

    /// Read a VarInt-length-prefixed UTF-8 string (**not** NBT's Modified UTF-8 — hazard H2
    /// doesn't apply to the play protocol, only to NBT). `max_chars` bounds the *decoded*
    /// length in UTF-16 code units, matching Java's `String.length()`; the declared byte
    /// length is bounded at `max_chars * 4` before a single byte is read, so a hostile length
    /// prefix costs a rejected read, never an oversized allocation.
    ///
    /// Malformed UTF-8 is replaced with `U+FFFD` rather than rejected, matching
    /// `new String(bytes, StandardCharsets.UTF_8)`'s non-throwing decode.
    /// java: `readStringFromBuffer`
    pub fn read_string(&mut self, max_chars: i32) -> Result<String, ProtoError> {
        let len = self.read_varint()?;
        if len < 0 {
            return Err(ProtoError::NegativeLength(len));
        }
        let byte_cap = max_chars.saturating_mul(4);
        if len > byte_cap {
            return Err(ProtoError::StringTooLong { len, max: byte_cap });
        }
        let bytes = self.read_bytes(len as usize)?;
        let s = java_utf8_lossy(&bytes);
        let utf16_len = i32::try_from(s.encode_utf16().count()).unwrap_or(i32::MAX);
        if utf16_len > max_chars {
            return Err(ProtoError::StringTooLong {
                len: utf16_len,
                max: max_chars,
            });
        }
        Ok(s)
    }

    /// Write a VarInt-length-prefixed UTF-8 string. `write` is infallible by contract
    /// ([`crate::Packet`]), so vanilla's encoder exception for a value over `32767` UTF-8 bytes
    /// is deferred to [`Self::finish_write`]. No truncated packet can reach the wire.
    /// java: `writeString`
    pub fn write_string(&mut self, s: &str) {
        const MAX_BYTES: usize = 32767;
        let bytes = s.as_bytes();
        if self.write_error.is_some() {
            return;
        }
        if bytes.len() > MAX_BYTES {
            self.write_error = Some(DeferredWriteError::StringTooLong {
                len: i32::try_from(bytes.len()).unwrap_or(i32::MAX),
                max: MAX_BYTES as i32,
            });
            return;
        }
        self.write_varint(bytes.len() as i32);
        self.inner.put_slice(bytes);
    }

    /// Read a chat component, still as raw JSON text (component-tree parsing is S15/S16's
    /// concern). Shares `readStringFromBuffer`'s 32767-char ceiling.
    /// java: `readChatComponent` (minus the `IChatComponent.Serializer` JSON parse)
    pub fn read_chat(&mut self) -> Result<String, ProtoError> {
        self.read_string(32767)
    }

    /// Write a chat component from its already-serialized JSON text.
    /// java: `writeChatComponent`
    pub fn write_chat(&mut self, json: &str) {
        self.write_string(json);
    }

    /// Read a VarInt-length-prefixed raw byte array. java: `readByteArray`
    pub fn read_byte_array(&mut self) -> Result<Vec<u8>, ProtoError> {
        let len = self.read_varint()?;
        if len < 0 {
            return Err(ProtoError::InvalidByteArrayLen(len));
        }
        self.read_bytes(len as usize)
    }

    /// Write a VarInt-length-prefixed raw byte array. java: `writeByteArray`
    pub fn write_byte_array(&mut self, data: &[u8]) {
        self.write_varint(data.len() as i32);
        self.inner.put_slice(data);
    }

    /// Read exactly `n` raw bytes with no length prefix (used by packets — chunk data among
    /// them — whose length is implied by other fields). Never allocates more than is already
    /// buffered: `n` is checked against [`PacketBuffer::remaining`] before anything is
    /// allocated, so a hostile `n` costs a rejected read, not a multi-gigabyte `Vec`.
    pub fn read_bytes(&mut self, n: usize) -> Result<Vec<u8>, ProtoError> {
        self.need(n, "a raw byte span")?;
        let mut out = vec![0u8; n];
        self.inner.copy_to_slice(&mut out);
        Ok(out)
    }

    /// Write raw bytes with no length prefix.
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.inner.put_slice(data);
    }

    /// Consume and return every remaining byte (used by e.g. plugin-channel payloads, whose
    /// length is implicit in the outer frame rather than self-delimited).
    pub fn read_remaining(&mut self) -> Vec<u8> {
        let n = self.inner.remaining();
        let mut out = vec![0u8; n];
        self.inner.copy_to_slice(&mut out);
        out
    }

    // ---- UUID, Position (H5), Angle -------------------------------------------------------

    /// Read a UUID as two big-endian `i64` halves. java: `readUuid`
    pub fn read_uuid(&mut self) -> Result<Uuid, ProtoError> {
        let hi = self.read_i64()? as u64;
        let lo = self.read_i64()? as u64;
        Ok(Uuid::from_u64_pair(hi, lo))
    }

    /// Write a UUID as two big-endian `i64` halves. java: `writeUuid`
    pub fn write_uuid(&mut self, uuid: Uuid) {
        let (hi, lo) = uuid.as_u64_pair();
        self.write_i64(hi as i64);
        self.write_i64(lo as i64);
    }

    /// Read the packed `Position` type: hazard **H5**, 26/12/26 bits sign-extended out of a
    /// single `i64`. Delegates to `mc_math::BlockPos::from_long`, which carries this exact
    /// packing (shared with `BlockPos.toLong`/`fromLong`) and is covered by its own
    /// round-trip test in `mc-math`, including negative coordinates away from the origin.
    /// java: `readBlockPos`
    pub fn read_block_pos(&mut self) -> Result<BlockPos, ProtoError> {
        Ok(BlockPos::from_long(self.read_i64()?))
    }

    /// Write the packed `Position` type. java: `writeBlockPos`
    pub fn write_block_pos(&mut self, pos: BlockPos) {
        self.write_i64(pos.to_long());
    }

    /// Read a yaw/pitch [`Angle`] byte.
    pub fn read_angle(&mut self) -> Result<Angle, ProtoError> {
        Ok(Angle(self.read_u8()?))
    }

    /// Write a yaw/pitch [`Angle`] byte.
    pub fn write_angle(&mut self, angle: Angle) {
        self.write_u8(angle.0);
    }

    // ---- NBT and ItemStack ------------------------------------------------------------------

    /// Read an optional NBT compound: a single `0x00` byte means `None` (vanilla's null-tag
    /// sentinel); any other leading byte means a full named compound follows, decoded through
    /// `mc-nbt` with [`mc_nbt::NbtLimits::PROTOCOL`] (2 MiB), matching vanilla's own
    /// `NBTSizeTracker(2097152L)` budget for this exact call site.
    /// java: `readNBTTagCompoundFromBuffer`
    pub fn read_nbt_compound_opt(&mut self) -> Result<Option<NbtCompound>, ProtoError> {
        self.need(1, "an NBT tag-type byte")?;
        let tag_byte = self.inner.chunk()[0];
        if tag_byte == 0 {
            self.inner.advance(1);
            return Ok(None);
        }
        let mut reader = TailReader(&mut self.inner);
        let (_name, compound) =
            mc_nbt::read_compound_with_limits(&mut reader, mc_nbt::NbtLimits::PROTOCOL)?;
        Ok(Some(compound))
    }

    /// Write an optional NBT compound: `None` as a bare `0x00`, `Some` as a full named
    /// compound (empty name, matching vanilla). An invalid locally-constructed tree records a
    /// deferred error which [`Self::finish_write`] surfaces before the packet reaches the wire.
    /// java: `writeNBTTagCompoundToBuffer`
    pub fn write_nbt_compound_opt(&mut self, nbt: Option<&NbtCompound>) {
        match nbt {
            None => self.write_u8(0),
            Some(c) => {
                if self.write_error.is_some() {
                    return;
                }
                // Encode atomically: an overlong modified-UTF-8 key or illegal list must not
                // leave a partial NBT payload that a caller could accidentally frame.
                let mut encoded = Vec::new();
                match mc_nbt::write_compound(&mut encoded, "", c) {
                    Ok(()) => self.inner.put_slice(&encoded),
                    Err(error) => {
                        self.write_error = Some(DeferredWriteError::Nbt(error.to_string()))
                    }
                }
            }
        }
    }

    /// Read an optional [`ItemStack`] (a `null` stack is `id == -1`).
    /// java: `readItemStackFromBuffer`
    pub fn read_item_stack(&mut self) -> Result<Option<ItemStack>, ProtoError> {
        ItemStack::read(self)
    }

    /// Write an optional [`ItemStack`]. java: `writeItemStackToBuffer`
    pub fn write_item_stack(&mut self, stack: Option<&ItemStack>) {
        ItemStack::write(stack, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_matches_known_vectors() {
        // Protocol-47 VarInt test vectors: the known encodings at each byte-length boundary.
        let cases: &[(i32, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (2, &[0x02]),
            (127, &[0x7f]),
            (128, &[0x80, 0x01]),
            (255, &[0xff, 0x01]),
            (2_147_483_647, &[0xff, 0xff, 0xff, 0xff, 0x07]),
            (-1, &[0xff, 0xff, 0xff, 0xff, 0x0f]),
            (-2_147_483_648, &[0x80, 0x80, 0x80, 0x80, 0x08]),
        ];
        for &(value, encoded) in cases {
            let mut buf = PacketBuffer::new();
            buf.write_varint(value);
            assert_eq!(buf.as_slice(), encoded, "encoding {value}");
            assert_eq!(PacketBuffer::varint_size(value), encoded.len());
            let mut read_buf = PacketBuffer::from_slice(encoded);
            assert_eq!(
                read_buf.read_varint().unwrap(),
                value,
                "decoding {encoded:?}"
            );
        }
    }

    #[test]
    fn varint_six_continuation_bytes_errors() {
        let mut buf = PacketBuffer::from_slice(&[0xff, 0xff, 0xff, 0xff, 0xff, 0xff]);
        assert!(matches!(buf.read_varint(), Err(ProtoError::VarIntTooBig)));
    }

    #[test]
    fn varlong_round_trips() {
        for value in [0i64, 1, -1, i64::MAX, i64::MIN, 123_456_789_012_345] {
            let mut buf = PacketBuffer::new();
            buf.write_varlong(value);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            assert_eq!(read_buf.read_varlong().unwrap(), value);
        }
    }

    #[test]
    fn eof_is_an_error_not_a_panic() {
        let mut buf = PacketBuffer::from_slice(&[0x01]);
        assert!(buf.read_i64().is_err());
        let mut empty = PacketBuffer::new();
        assert!(matches!(empty.read_varint(), Err(ProtoError::Eof(_))));
        assert!(matches!(empty.read_byte_array(), Err(ProtoError::Eof(_))));
        // A byte array claiming to be huge, but with nothing behind it, must not allocate.
        let mut lying = PacketBuffer::new();
        lying.write_varint(i32::MAX);
        assert!(matches!(lying.read_byte_array(), Err(ProtoError::Eof(_))));
    }

    #[test]
    fn string_round_trips_utf8() {
        let mut buf = PacketBuffer::new();
        buf.write_string("hello \u{1F600}");
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(read_buf.read_string(32767).unwrap(), "hello \u{1F600}");
    }

    #[test]
    fn string_declared_length_over_cap_is_rejected_before_reading() {
        let mut buf = PacketBuffer::new();
        buf.write_varint(1000); // no payload follows
        assert!(matches!(
            buf.read_string(10),
            Err(ProtoError::StringTooLong { .. })
        ));
    }

    #[test]
    fn string_invalid_utf8_uses_java_replacement_grouping() {
        // Java's `new String(bytes, UTF_8)` delegates malformed input to the JDK decoder. In
        // particular, a structurally complete UTF-16-surrogate encoding is one malformed
        // three-byte sequence, not three malformed bytes. The one-unit limit is the negative
        // control: std::str::from_utf8_lossy would produce three UTF-16 units and reject it.
        let surrogate = [0xed, 0xa0, 0x80];
        assert_eq!(
            String::from_utf8_lossy(&surrogate).encode_utf16().count(),
            3
        );

        let cases: &[(&[u8], i32, &str)] = &[
            (&[0xed, 0xa0, 0x80], 1, "\u{fffd}"),
            (&[0xed, 0xbf, 0xbf], 1, "\u{fffd}"),
            (&[0xe2, 0x80, 0x41], 2, "\u{fffd}A"),
            (&[0xf0, 0x90, 0x80], 1, "\u{fffd}"),
            (&[0xf0, 0x90, 0x80, 0x41], 2, "\u{fffd}A"),
            (&[0xc0, 0x80], 2, "\u{fffd}\u{fffd}"),
            (
                &[0xf4, 0x90, 0x80, 0x80],
                4,
                "\u{fffd}\u{fffd}\u{fffd}\u{fffd}",
            ),
        ];

        for &(bytes, max_chars, expected) in cases {
            let mut encoded = PacketBuffer::new();
            encoded.write_varint(bytes.len() as i32);
            encoded.write_bytes(bytes);
            let mut decoded = PacketBuffer::from_slice(encoded.as_slice());
            assert_eq!(decoded.read_string(max_chars).unwrap(), expected);
            assert!(decoded.is_empty(), "decoder left bytes for {bytes:02x?}");
        }
    }

    #[test]
    fn outgoing_string_over_32767_bytes_is_rejected_not_truncated() {
        let mut buf = PacketBuffer::new();
        buf.write_string(&"x".repeat(32_768));
        assert!(matches!(
            buf.finish_write(),
            Err(ProtoError::StringTooLong {
                len: 32_768,
                max: 32_767
            })
        ));
        assert!(
            buf.is_empty(),
            "an invalid prefix must not be partially encoded"
        );
    }

    #[test]
    fn outgoing_string_accepts_exactly_32767_utf8_bytes() {
        // java: PacketBuffer.writeString measures `string.getBytes(UTF_8).length`, not Java
        // String.length(). Exercise both plain ASCII and supplementary scalars so a UTF-16-unit
        // or Unicode-scalar ceiling cannot accidentally satisfy this boundary test.
        for value in [
            "x".repeat(32_767),
            format!("{}abc", "\u{1f600}".repeat(8_191)),
        ] {
            assert_eq!(value.len(), 32_767);
            let mut buf = PacketBuffer::new();
            buf.write_string(&value);
            assert!(buf.finish_write().is_ok());

            let mut encoded = PacketBuffer::from_slice(buf.as_slice());
            assert_eq!(encoded.read_varint().unwrap(), 32_767);
            assert_eq!(encoded.read_bytes(32_767).unwrap(), value.as_bytes());
            assert!(encoded.is_empty());
        }

        let mut four_byte_overflow = PacketBuffer::new();
        four_byte_overflow.write_string(&"\u{1f600}".repeat(8_192));
        assert!(matches!(
            four_byte_overflow.finish_write(),
            Err(ProtoError::StringTooLong {
                len: 32_768,
                max: 32_767
            })
        ));
        assert!(four_byte_overflow.is_empty());
    }

    #[test]
    fn uuid_round_trips() {
        let id = Uuid::from_u128(0x0123_4567_89ab_cdef_0123_4567_89ab_cdef);
        let mut buf = PacketBuffer::new();
        buf.write_uuid(id);
        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(read_buf.read_uuid().unwrap(), id);
    }

    #[test]
    fn block_pos_sign_extends_away_from_origin() {
        // Hazard H5: the packed 26/12/26 i64 must sign-extend each field on decode.
        for p in [
            BlockPos::new(0, 0, 0),
            BlockPos::new(-1, -1, -1),
            BlockPos::new(-30_000_000, -64, -30_000_000),
            BlockPos::new(30_000_000, 255, 30_000_000),
            BlockPos::new(-1234, 70, 5678),
        ] {
            let mut buf = PacketBuffer::new();
            buf.write_block_pos(p);
            let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
            assert_eq!(
                read_buf.read_block_pos().unwrap(),
                p,
                "round trip for {p:?}"
            );
        }
    }

    #[test]
    fn angle_round_trips_at_representable_steps() {
        for step in 0u8..=255 {
            let angle = Angle(step);
            let degrees = angle.to_degrees();
            assert_eq!(Angle::from_degrees(degrees).0, step, "step {step}");
        }
    }

    /// Java oracle: packet angle accessors return a signed `byte`, and the client multiplies that
    /// value by 360 before dividing by 256. Exercise the complete one-byte input domain rather
    /// than relying on the round-trip test above, whose encoder and decoder could share a bug.
    #[test]
    fn angle_decode_matches_java_signed_byte_for_all_wire_values() {
        for raw in 0u8..=u8::MAX {
            let java = f32::from(raw as i8) * 360.0 / 256.0;
            assert_eq!(Angle(raw).to_degrees(), java, "wire byte {raw:#04x}");
        }
    }

    /// Negative control for the former unsigned decoder. It agrees with Java on the low half of
    /// the domain, then differs by exactly one full turn for all 128 high-bit values.
    #[test]
    fn unsigned_angle_decode_is_rejected_on_exactly_the_high_bit_domain() {
        let divergent = (0u8..=u8::MAX)
            .filter(|&raw| {
                let java = f32::from(raw as i8) * 360.0 / 256.0;
                let unsigned_mutant = f32::from(raw) * 360.0 / 256.0;
                unsigned_mutant != java
            })
            .count();

        assert_eq!(divergent, 128);
        assert_eq!(Angle(0x7f).to_degrees(), 178.593_75);
        assert_eq!(Angle(0x80).to_degrees(), -180.0);
        assert_eq!(Angle(0xff).to_degrees(), -1.406_25);
    }

    #[test]
    fn nbt_optional_round_trips_none_and_some() {
        let mut buf = PacketBuffer::new();
        buf.write_nbt_compound_opt(None);
        let mut c = NbtCompound::new();
        c.insert("x", mc_nbt::Nbt::Int(7));
        buf.write_nbt_compound_opt(Some(&c));

        let mut read_buf = PacketBuffer::from_slice(buf.as_slice());
        assert_eq!(read_buf.read_nbt_compound_opt().unwrap(), None);
        let round_tripped = read_buf.read_nbt_compound_opt().unwrap().unwrap();
        assert_eq!(round_tripped.get_i32("x"), Some(7));
    }

    #[test]
    fn invalid_outgoing_nbt_is_a_deferred_encoder_error_not_a_panic() {
        let mut invalid = NbtCompound::new();
        invalid.insert("x".repeat(65_536), mc_nbt::Nbt::Byte(1));

        let mut buf = PacketBuffer::new();
        buf.write_nbt_compound_opt(Some(&invalid));
        assert!(matches!(buf.finish_write(), Err(ProtoError::NbtEncode(_))));
        assert!(buf.as_slice().is_empty(), "failed NBT encoding is atomic");

        let mut valid = NbtCompound::new();
        valid.insert("x", mc_nbt::Nbt::Byte(1));
        let mut control = PacketBuffer::new();
        control.write_nbt_compound_opt(Some(&valid));
        assert!(control.finish_write().is_ok());
        assert!(!control.as_slice().is_empty());
    }
}
