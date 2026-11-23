//! The crate error type.
//!
//! Every failure mode a hostile server can trigger is a typed variant here. Nothing in
//! this crate panics on input: §11 forbids `unwrap()` on anything derived from the network,
//! and NBT is the first thing a server gets to hand us.

use thiserror::Error;

use crate::modified_utf8::ModifiedUtf8Error;
use crate::tag::TagType;

/// Errors returned by `mc-nbt`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NbtError {
    /// The underlying reader or writer failed. A truncated stream shows up as
    /// [`std::io::ErrorKind::UnexpectedEof`].
    #[error("nbt i/o: {0}")]
    Io(#[from] std::io::Error),

    /// A tag type byte outside `0..=11`.
    ///
    /// java: `NBTBase.createNewByType` returns `null` here and vanilla then throws NPE.
    #[error("unknown NBT tag type {0}")]
    UnknownTagType(u8),

    /// Modified UTF-8 that could not be decoded, or a string too long to encode.
    #[error("nbt string: {0}")]
    String(#[from] ModifiedUtf8Error),

    /// Nesting deeper than the configured limit.
    ///
    /// java: `NBTTagCompound.read` / `NBTTagList.read`, "too high complexity, depth > 512"
    #[error("NBT nesting deeper than {limit}")]
    DepthLimit {
        /// The configured maximum.
        limit: u32,
    },

    /// The [`crate::limits::SizeTracker`] budget ran out.
    ///
    /// java: `NBTSizeTracker.read`, "Tried to read NBT tag that was too big"
    #[error("NBT exceeds the {limit}-byte budget (wanted at least {requested})")]
    SizeLimit {
        /// The configured budget, in `NBTSizeTracker` accounting bytes.
        limit: u64,
        /// The running total that broke it.
        requested: u64,
    },

    /// A `TAG_List`, `TAG_Byte_Array` or `TAG_Int_Array` declared a negative length.
    ///
    /// Vanilla passes these straight to `new byte[i]` / `newArrayListWithCapacity(i)`.
    #[error("negative NBT {kind} length {len}")]
    NegativeLength {
        /// Which container declared it.
        kind: TagType,
        /// The declared length.
        len: i32,
    },

    /// A `TAG_List` declared element type `TAG_End` but a non-zero element count.
    ///
    /// java: `NBTTagList.read`, "Missing type on ListTag"
    #[error("TAG_List declared TAG_End elements but a count of {count}")]
    MissingListType {
        /// The declared element count.
        count: i32,
    },

    /// A `TAG_List` being written holds more than one element type. NBT lists are
    /// homogeneous; vanilla writes the first element's type and would silently corrupt.
    #[error("TAG_List is heterogeneous: element {index} is {found}, expected {expected}")]
    HeterogeneousList {
        /// Position of the offending element.
        index: usize,
        /// The type the list is being written as (the first element's).
        expected: TagType,
        /// The type actually found.
        found: TagType,
    },

    /// A [`crate::Nbt::End`] turned up inside a compound or list.
    ///
    /// Vanilla's `writeEntry` would emit a bare `0x00`, silently truncating the compound;
    /// we refuse instead.
    #[error("TAG_End cannot be stored inside a container")]
    EndInContainer,

    /// Something claimed to be a compound but was not.
    ///
    /// java: `CompressedStreamTools.read`, "Root tag must be a named compound tag"
    #[error("expected {expected}, found {found}")]
    UnexpectedType {
        /// What the caller asked for.
        expected: TagType,
        /// What was actually there.
        found: TagType,
    },

    /// Free-form message. Produced by the `serde` integration's `Error::custom`.
    #[error("{0}")]
    Message(String),
}

impl NbtError {
    /// Convenience constructor for the "stream ended mid-tag" case.
    pub(crate) fn eof(what: &'static str) -> Self {
        NbtError::Io(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            format!("NBT stream ended while reading {what}"),
        ))
    }
}

#[cfg(feature = "serde")]
impl serde::ser::Error for NbtError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        NbtError::Message(msg.to_string())
    }
}

#[cfg(feature = "serde")]
impl serde::de::Error for NbtError {
    fn custom<T: std::fmt::Display>(msg: T) -> Self {
        NbtError::Message(msg.to_string())
    }
}
