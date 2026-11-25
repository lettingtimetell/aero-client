//! Protocol 5 (Minecraft 1.7.10): an exact model of the wire.
//!
//! Every packet the 1.7.10 client sends or receives in the Play state, plus the two login
//! packets whose encoding differs, decoded and encoded the way the 1.7.10 classes'
//! `readPacketData` / `writePacketData` do. Nothing here is lowered into the 1.8.9 types; that
//! happens where the client applies a packet, and only where the handlers are the same.
//!
//! # What is shared with protocol 47
//!
//! The frame (VarInt length, at most three bytes), VarInt packet ids, VarInt and string
//! encodings, the handshake, status and most of the login packets, AES/CFB8 and the packet id of
//! every packet both versions have. [`crate::codec`], [`crate::crypt`], [`crate::handshake`],
//! [`crate::status`] and [`crate::login`] serve both protocols unchanged.
//!
//! # What is not
//!
//! | Module | Protocol 5 |
//! |---|---|
//! | [`buffer`] | `i16`-prefixed blobs, gzip NBT, item stacks with gzip NBT |
//! | [`nbt`] | the gzip framing and 1.7.10's own tag reader and size accounting |
//! | [`metadata`] | watcher types 0-6; type 7 is Java's `null` entry |
//! | [`login`] | the encryption request/response with `i16` byte-array lengths |
//! | [`s2c`], [`chunk`] | the 65 clientbound Play packets (`0x00`-`0x40`) |
//! | [`c2s`] | the 24 serverbound Play packets (`0x00`-`0x17`) and the 1.8.9 conversion |
//! | [`play`] | [`S2CPlay5`] / [`C2SPlay5`] |
//! | [`tables`] | world types and Java's case-insensitive name lookup |
//!
//! There is no compression in protocol 5 (no login `0x03`, no play `0x46`), and the login loop
//! is [`crate::connection_v5`].
//!
//! java17: `network/**` of 1.7.10

pub mod buffer;
pub mod c2s;
pub mod chunk;
pub mod login;
pub mod metadata;
pub mod nbt;
pub mod play;
pub mod s2c;
pub mod tables;

pub use buffer::PacketBufferV5;
pub use c2s::{NotInProtocol5, V5Ctx};
pub use metadata::{Metadata5, MetadataEntry5, MetadataValue5};
pub use nbt::NbtBudget;
pub use play::{C2SPlay5, S2CPlay5};
