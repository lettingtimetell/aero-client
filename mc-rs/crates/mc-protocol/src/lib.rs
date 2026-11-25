#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Protocol 47 (Minecraft 1.8.9): packet types, codec, encryption, compression.
//!
//! Two sections share this crate: **S06** (packets and [`PacketBuffer`]) and **S07**
//! ([`connection`], [`crypt`], [`compress`] — the handshake, AES-CFB8 and zlib layers).
//!
//! The protocol-47 rule makes the wire format non-negotiable, so parity bugs here are project-fatal.
//! Hazard H5 (the protocol's packed `Position` type is sign-extended 26/12/26) is handled by
//! re-using `mc_math::BlockPos::to_long`/`from_long`, which already carries that exact packing
//! and is covered by its own round-trip test in `mc-math`.
//!
//! # Layout
//!
//! | Module | Contents | java source |
//! |---|---|---|
//! | [`buffer`] | [`PacketBuffer`]: varint/varlong, string, UUID, angle, position, chat, NBT | `network/PacketBuffer.java` |
//! | [`item`] | [`item::ItemStack`] codec | `PacketBuffer.writeItemStackToBuffer`/`readItemStackFromBuffer` |
//! | [`metadata`] | Entity metadata (`DataWatcher`) codec | `entity/DataWatcher.java` |
//! | [`packet`] | The [`Packet`] trait and [`State`] enum | `network/Packet.java`, `EnumConnectionState.java` |
//! | [`handshake`] | The one handshake-state packet | `network/handshake/**` |
//! | [`status`] | Server-list-ping packets | `network/status/**` |
//! | [`login`] | Login-state packets | `network/login/**` |
//! | [`chunk`] | Chunk data / map chunk bulk (raw payload only — S08 unpacks it) | `S21PacketChunkData`, `S26PacketMapChunkBulk` |
//! | [`play`] | [`play::S2CPlay`] (74 variants) and [`play::C2SPlay`] (26 variants) | `network/play/**` |
//! | [`codec`] | Length-prefixed frame [`Decoder`](tokio_util::codec::Decoder)/[`Encoder`](tokio_util::codec::Encoder) | `NetworkManager` framing |
//! | [`crypt`] | RSA key exchange + the AES-CFB8 stream (hazards H3, H4) | `util/CryptManager.java`, `NettyEncryptionTranslator` |
//! | [`compress`] | The zlib layer `SetCompression` switches on | `NettyCompression{En,De}coder` |
//! | [`connection`] | Socket, layer ordering, state machine, handshake/status/login drivers | `NetworkManager`, `NetHandlerLoginClient` |
//!
//! # Layer order
//!
//! Outbound: **packet → compress → length-prefix → encrypt → socket**; inbound is the mirror.
//! Compression therefore sits *inside* encryption, and encryption covers the length prefix.
//! [`connection`]'s module documentation derives this from `NetworkManager`'s Netty pipeline.
//!
//! # A note on "77 / 27" packet counts
//!
//! `EnumConnectionState`'s packet registration (the actual source of wire IDs — vanilla assigns
//! IDs by registration order, not by the `S`/`C` number embedded in a class's *file* name) yields
//! **74** clientbound and **26** serverbound Play packets, not 77/27. Two things explain the
//! gap: `S14PacketEntity` has three nested subclasses (`S15`/`S16`/`S17`, IDs `0x15`-`0x17`) that
//! are not separate files, and `C03PacketPlayer` has three of its own (`C04`-`C06`). Conversely,
//! two *files* (`S19PacketEntityHeadLook.java`, `S19PacketEntityStatus.java`) share an `S19`
//! prefix but register as `0x19` and `0x1A` respectively — a naming quirk of the decompiled source,
//! not a duplicate ID. `play::S2CPlay`/`play::C2SPlay` match the registration order exactly;
//! see their module docs for the full ID table.

pub mod buffer;
pub mod chunk;
pub mod codec;
pub mod compress;
pub mod connection;
pub mod connection_v5;
pub mod crypt;
pub mod handshake;
pub mod item;
pub mod login;
pub mod metadata;
pub mod packet;
pub mod play;
pub mod status;
pub mod v5;

pub use buffer::{Angle, PacketBuffer};
pub use compress::CompressionCodec;
pub use connection::{
    Connection, ConnectionReader, ConnectionWriter, Direction, LoginOutcome, ServerStatus,
    PROTOCOL_VERSION,
};
pub use crypt::{DecryptStream, EncryptStream, SharedSecret};
pub use item::ItemStack;
pub use metadata::{Metadata, MetadataEntry, MetadataValue};
pub use packet::{Packet, State};

use thiserror::Error;

/// Errors returned by `mc-protocol`.
///
/// Every variant that can be produced while decoding is reachable from untrusted server bytes
/// (§11: no `unwrap()` on anything derived from network input), so this crate never panics on
/// malformed input — it returns one of these instead.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ProtoError {
    /// The buffer ran out of bytes before a fixed-size or length-prefixed field was satisfied.
    #[error("mc-protocol: unexpected end of buffer while reading {0}")]
    Eof(&'static str),
    /// A VarInt used more than the 5 bytes a 32-bit value can ever need.
    /// java: `PacketBuffer.readVarIntFromBuffer` — "VarInt too big"
    #[error("mc-protocol: VarInt is too big (more than 5 bytes)")]
    VarIntTooBig,
    /// A VarLong used more than the 10 bytes a 64-bit value can ever need.
    /// java: `PacketBuffer.readVarLong` — "VarLong too big"
    #[error("mc-protocol: VarLong is too big (more than 10 bytes)")]
    VarLongTooBig,
    /// A length-prefixed string's declared byte length exceeded the protocol ceiling before a
    /// single character was decoded. java: `PacketBuffer.readStringFromBuffer`
    #[error("mc-protocol: string byte length {len} exceeds the allowed maximum {max}")]
    StringTooLong {
        /// The declared length, in bytes.
        len: i32,
        /// The maximum allowed, in bytes (`max_chars * 4`).
        max: i32,
    },
    /// A length-prefixed string's declared length was negative.
    #[error("mc-protocol: string length {0} is negative")]
    NegativeLength(i32),
    /// A string field that vanilla encodes as `UUID.toString()` (login success, and a few
    /// others) was not a parseable UUID.
    #[error("mc-protocol: `{0}` is not a valid UUID")]
    InvalidUuid(String),
    /// A byte array's declared length was negative or absurd for the remaining buffer.
    #[error("mc-protocol: byte array length {0} is invalid")]
    InvalidByteArrayLen(i32),
    /// An `EnumConnectionState`-style bounded enum value (e.g. a data-watcher type tag) was out
    /// of range.
    #[error("mc-protocol: {what} value {value} is out of range")]
    InvalidEnum {
        /// What was being decoded.
        what: &'static str,
        /// The out-of-range value seen.
        value: i32,
    },
    /// No packet is registered for this (state, direction, id) triple.
    #[error("mc-protocol: unknown packet id {id:#04x} in state {state:?}")]
    UnknownPacketId {
        /// The connection state the id was read in.
        state: State,
        /// The raw id.
        id: i32,
    },
    /// A packet's declared length prefix (frame codec) exceeded the configured ceiling.
    #[error("mc-protocol: frame length {len} exceeds the maximum {max}")]
    FrameTooLong {
        /// The declared frame length.
        len: usize,
        /// The configured ceiling.
        max: usize,
    },
    /// A packet-owned byte payload exceeded the field-specific vanilla ceiling.
    ///
    /// Unlike [`ProtoError::FrameTooLong`], this is checked after the outer packet frame has
    /// already been accepted (for example `S3FPacketCustomPayload`'s 1 MiB body).
    #[error("mc-protocol: payload length {len} exceeds the allowed maximum {max}")]
    PayloadTooLong {
        /// The payload length, in bytes.
        len: usize,
        /// The field-specific maximum, in bytes.
        max: usize,
    },
    /// A registered packet decoder returned before consuming its complete framed body.
    /// java: `MessageDeserializer` rejects this as "larger than I expected".
    #[error("mc-protocol: packet id {packet_id:#04x} left {remaining} unread trailing bytes")]
    TrailingBytes {
        /// The state-scoped packet id that was decoded.
        packet_id: i32,
        /// Bytes remaining in the packet body after `Packet::read` returned.
        remaining: usize,
    },
    /// NBT decoding failed while reading an item stack's tag compound or another NBT-bearing
    /// field.
    #[error("mc-protocol: NBT error: {0}")]
    Nbt(#[from] mc_nbt::NbtError),
    /// Outgoing NBT could not be represented on the wire. Vanilla wraps the corresponding
    /// `IOException` in Netty's `EncoderException` and fails the send rather than panicking.
    #[error("mc-protocol: NBT encoder error: {0}")]
    NbtEncode(String),
    /// An I/O error surfaced by an `impl Read`/`Write` adapter (in practice, only allocation
    /// failure paths inside `bytes`/`mc_nbt` — sockets are S07/S14's concern, not this crate's).
    #[error("mc-protocol: I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A code path that S06/S07 have not implemented yet.
    #[error("mc-protocol: {0} is not implemented yet")]
    Unimplemented(&'static str),

    // ---- S07: handshake, encryption, compression ----
    /// The peer closed the connection, possibly mid-frame.
    #[error("mc-protocol: the connection was closed by the peer")]
    ConnectionClosed,
    /// The server's RSA public key was not well-formed X.509 `SubjectPublicKeyInfo` DER.
    /// java: `CryptManager.decodePublicKey` logs "Public key reconstitute failed!" and returns
    /// `null`, which then NPEs; a typed error keeps a hostile server from crashing us.
    #[error("mc-protocol: could not parse the server's RSA public key: {0}")]
    PublicKey(String),
    /// RSA PKCS#1 v1.5 encryption of the shared secret or verify token failed.
    #[error("mc-protocol: RSA encryption failed: {0}")]
    RsaEncrypt(String),
    /// Encryption was already active. Vanilla installs the CFB8 ciphers exactly once per
    /// connection (hazard H4); a second `EncryptionRequest` would reset the stream.
    #[error("mc-protocol: encryption is already enabled on this connection")]
    AlreadyEncrypted,
    /// A compressed packet declared a decompressed size below the negotiated threshold, which
    /// means it should have been sent uncompressed.
    /// java: `NettyCompressionDecoder` — "Badly compressed packet - size of N is below server
    /// threshold of T"
    #[error("mc-protocol: badly compressed packet: declared size {len} is below the server threshold {threshold}")]
    BadlyCompressed {
        /// The declared decompressed size.
        len: i32,
        /// The negotiated threshold.
        threshold: i32,
    },
    /// A compressed packet declared a decompressed size over the protocol's 2 MiB ceiling.
    /// java: `NettyCompressionDecoder` — "larger than protocol maximum of 2097152"
    #[error("mc-protocol: decompressed size {len} exceeds the protocol maximum {max}")]
    DecompressedTooLarge {
        /// The declared decompressed size.
        len: i32,
        /// The 2 MiB ceiling.
        max: i32,
    },
    /// zlib deflate or inflate failed, or a stream did not inflate to the size it declared.
    #[error("mc-protocol: zlib error: {0}")]
    Zlib(String),
    /// A second non-negative `SetCompression` arrived while compression was already installed.
    /// java: `NetworkManager.setCompressionTreshold` (`:454-469`) updates the existing decoder,
    /// then casts the `decompress` handler to `NettyCompressionEncoder`; that
    /// `ClassCastException` reaches `NetworkManager.exceptionCaught` and closes the channel.
    #[error("mc-protocol: compression was reconfigured while already enabled")]
    CompressionReconfigured,
    /// A state change `EnumConnectionState` does not permit was attempted.
    #[error("mc-protocol: illegal state transition {from:?} -> {to:?}")]
    IllegalStateTransition {
        /// The state the connection was in.
        from: State,
        /// The state that was requested.
        to: State,
    },
    /// A packet id arrived that has no meaning at this point in the sequence.
    #[error("mc-protocol: expected {expected} in state {state:?}, got packet id {got:#04x}")]
    UnexpectedPacket {
        /// The state the id was read in.
        state: State,
        /// What the driver was waiting for.
        expected: &'static str,
        /// The id actually received.
        got: i32,
    },
    /// The server kicked us. `reason` is chat-component JSON, ready for S16 to render.
    /// java: `S00PacketDisconnect` in `Login`, `S40PacketDisconnect` in `Play`.
    #[error("mc-protocol: disconnected by the server: {reason}")]
    Disconnected {
        /// The kick reason, as chat-component JSON.
        reason: String,
    },
    /// A status `Pong` did not echo the payload its `Ping` sent.
    #[error("mc-protocol: status pong echoed {got} instead of {sent}")]
    PingMismatch {
        /// What was sent.
        sent: i64,
        /// What came back.
        got: i64,
    },
    /// The server sent this many `Login`-state packets without ever finishing the login.
    #[error("mc-protocol: the server sent {0} login packets without completing the login")]
    LoginStalled(usize),
    /// The Mojang session server rejected, or could not process, the join.
    #[error("mc-protocol: session-server join failed: {0}")]
    Auth(#[from] mc_auth::AuthError),
}
