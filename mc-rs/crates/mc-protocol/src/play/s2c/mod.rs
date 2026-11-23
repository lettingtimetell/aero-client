//! Clientbound Play packets, split into four roughly-equal batches for parallel authorship
//! (the own suggested split). Each batch module owns a contiguous slice
//! of the `EnumConnectionState` clientbound-Play registration order; nothing here reorders or
//! renumbers them.
//!
//! `0x21 ChunkData` and `0x26 MapChunkBulk` are **not** here — see [`crate::chunk`].

pub mod batch1;
pub mod batch2;
pub mod batch3;
pub mod batch4;

pub use batch1::*;
pub use batch2::*;
pub use batch3::*;
pub use batch4::*;
