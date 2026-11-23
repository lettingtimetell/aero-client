#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Named Binary Tag read/write, matching Java's `DataInput`/`DataOutput` semantics.
//!
//! Ported from `net/minecraft/nbt/` (18 files).
//!
//! ```
//! use mc_nbt::{Nbt, NbtCompound};
//!
//! let mut root = NbtCompound::new();
//! root.insert("LevelName", Nbt::String("mcpworld".into()));
//! root.insert("Time", Nbt::Long(1620));
//!
//! let bytes = mc_nbt::to_vec("", &Nbt::Compound(root)).unwrap();
//! let (name, value) = mc_nbt::from_slice(&bytes).unwrap();
//!
//! assert_eq!(name, "");
//! assert_eq!(value.as_compound().unwrap().get_str("LevelName"), Some("mcpworld"));
//! assert_eq!(mc_nbt::to_vec(&name, &value).unwrap(), bytes); // byte-identical
//! ```
//!
//! # Three things that will bite you
//!
//! **Strings are Modified UTF-8, not UTF-8** (hazard H2). `NUL` is written as the two bytes
//! `0xC0 0x80` and characters above `U+FFFF` become CESU-8 surrogate pairs — six bytes, not
//! four. [`modified_utf8`] handles both directions; `String::from_utf8` does not.
//!
//! **Everything is big-endian and signed**, matching Java's `DataInput`. The one exception
//! is [`Nbt::ByteArray`], which we store as `u8` for ergonomics.
//!
//! **This parses untrusted server input.** [`read`] therefore defaults to
//! [`NbtLimits::PROTOCOL`] — vanilla's 2 MiB `NBTSizeTracker` budget and its depth cap of
//! 512. No allocation is ever sized from a length field in the input, so a hostile header
//! costs a failed read rather than memory. Pass [`NbtLimits::UNLIMITED`] via
//! [`read_with_limits`] for files you already trust, as vanilla does for saves on disk.
//!
//! # Layout
//!
//! | Module | Java original |
//! |---|---|
//! | [`tag`] | `NBTBase`, `NBTTagCompound`, `NBTTagList`, the ten scalar tags |
//! | [`read`] / [`write`] | each `NBTTag*.read` / `.write` |
//! | [`compress`] | `CompressedStreamTools` |
//! | [`limits`] | `NBTSizeTracker` |
//! | [`modified_utf8`] | `java.io.DataInput/OutputStream` UTF handling |
//!
//! With the `serde` feature, [`to_nbt`] and [`from_nbt`] map ordinary Rust types to and
//! from [`Nbt`], and [`Nbt`] itself implements `Serialize`/`Deserialize`.

pub mod compress;
pub mod error;
pub mod limits;
pub mod modified_utf8;
pub mod read;
pub mod snbt;
pub mod tag;
pub mod write;

#[cfg(feature = "serde")]
pub mod serde_support;

#[doc(inline)]
pub use crate::compress::{
    read_any, read_gzip, read_gzip_with_limits, read_zlib, read_zlib_with_limits, safe_write,
    sniff, write_gzip, write_zlib, Framing,
};
#[doc(inline)]
pub use crate::error::NbtError;
#[doc(inline)]
pub use crate::limits::{NbtLimits, SizeTracker};
#[doc(inline)]
pub use crate::read::{
    from_slice, from_slice_with_limits, read, read_compound, read_compound_with_limits,
    read_with_limits,
};
#[doc(inline)]
pub use crate::snbt::{parse_snbt, parse_snbt_compound, SnbtError};
#[doc(inline)]
pub use crate::tag::{Nbt, NbtCompound, NbtList, TagType};
#[doc(inline)]
pub use crate::write::{to_vec, write, write_compound};

#[cfg(feature = "serde")]
#[doc(inline)]
pub use crate::serde_support::{from_nbt, to_nbt};
