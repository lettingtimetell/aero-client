#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! The client-side world: chunk storage, block access, lighting, and the entity arena.
//!
//! This is where the arena-index rule is established:
//! entities are `hecs` arena indices ([`Entity`]), behaviour is systems over `&mut World`, and
//! **no `Rc<RefCell<_>>` appears anywhere in this crate** — `tests::world_is_send_and_sync` in
//! `world.rs` makes that a type-level check rather than only a doc comment. Vanilla's
//! `World.java` is the cyclic mutable graph I3 exists to avoid — read it for the rules of the
//! simulation, not for its ownership shape.
//!
//! # Layout
//!
//! | module | covers |
//! |---|---|
//! | [`chunk`] | [`ChunkSection`]/[`Chunk`] storage and the chunk-packet byte unpacker (H6) |
//! | [`world`] | [`World`]: chunk load/unload, block/light access, the dirty set, the `hecs` arena |
//! | [`entity`] | ECS components and [`EntityBundle`] |
//! | [`collision`] | the `mc_data::Aabb` -> `mc_math::Aabb` boundary S13 will use |
//!
//! # Hazard H6
//!
//! Chunk section block arrays are **little-endian `u16`**, and sky light is **absent outside
//! the overworld**. [`chunk::Chunk::fill_from_packet`] takes `has_sky_light` as an explicit
//! parameter rather than assuming it, precisely so a Nether/End chunk cannot desync the reader
//! by skipping a region that vanilla's server never sent. [`World`] tracks the flag itself
//! ([`World::has_sky_light`]) so callers only have to set it once, at dimension change.
//!
//! # `mc-protocol` (S06) is a dependency for exactly one type
//!
//! Per the plan's dependency graph, S08 depends only on S01 (`mc-data`) and S05 (`mc-math`) —
//! not S06. The chunk/block-change *unpack* functions in [`chunk`] and [`world`] still honour
//! that: they are written against the **raw byte layout** described in `S21PacketChunkData`/
//! `S22PacketMultiBlockChange`/`S26PacketMapChunkBulk` (fields, not packet structs) rather than
//! against `mc-protocol`'s packet structs, so a chunk X/Z pair, a section bitmask, a byte slice
//! and plain integers are all [`World::apply_chunk_data`]/[`World::apply_chunk_bulk`]/
//! [`World::apply_multi_block_change`] ever ask for.
//!
//! [`entity::MetadataValue`]'s type-5 arm is the one exception, added once `mc-protocol` (S06)
//! existed: it carries `mc_protocol::ItemStack` directly rather than the `Vec<u8>` placeholder
//! S08 originally left there (the S14 -> S08 §13 cross-section request), because `ItemStack` is
//! DataWatcher's real wire shape for that slot and re-encoding an already-decoded stack back to
//! bytes just to fit a byte-blob placeholder had no upside. That single `use` is this crate's
//! entire `mc-protocol` surface — [`chunk`] and [`world`] still know nothing about it.

pub mod chunk;
pub mod collision;
pub mod entity;
mod genlayer;
pub mod light;
#[cfg(test)]
mod light_equivalence;
pub mod map;
pub mod tileent;
pub mod v5;
pub mod world;

pub use chunk::ChunkSetOutcome;
pub use light::SkyBlock;
pub use map::{MapData, MapDecoration, MapUpdateError, MAP_PIXELS, MAP_SIZE};
pub use tileent::{
    chest_lid_inputs, dye_rgb, BannerState, BeaconState, BeamSegment, BlockActionEffect,
    ChestNeighbours, ChestState, EnchantState, FlowerPotState, PistonWorldBorder, SignState,
    SkullState, SpawnerState, TileData, TileEntity, TileEntitySound, TileKind,
};

pub use chunk::{expected_chunk_data_len, Chunk, ChunkSection};
pub use collision::{collision_boxes_at, to_world_aabb};
pub use entity::{
    Acceleration, ArrowShooter, EntityBundle, EntityKind, ItemFrameFacing, Metadata, MetadataValue,
    Position, Rotation, SplashPotionDamage, Velocity,
};
pub use world::{multi_block_change_pos, BulkChunkEntry, Entity, World};
// Exact world copies for replay checkpoints (`world/snapshot.rs`).
pub use world::snapshot::EcsCloner;

use thiserror::Error;

/// Errors returned by `mc-world`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum WorldError {
    /// A chunk packet's byte payload did not match the length its bitmask/flags implied
    /// ([`chunk::expected_chunk_data_len`]). Always a malformed or truncated packet, never a
    /// panic path — untrusted server input must not be trusted to be well-formed.
    #[error("mc-world: chunk data length mismatch: expected {expected} bytes, got {actual}")]
    ChunkDataLength {
        /// The length `expected_chunk_data_len` computed from the bitmask and flags.
        expected: usize,
        /// The length actually supplied.
        actual: usize,
    },

    /// [`World::despawn`] was called with an `Entity` that does not exist in this `World`
    /// (already despawned, or never spawned here).
    #[error("mc-world: {0}")]
    NoSuchEntity(#[from] hecs::NoSuchEntity),
}

#[cfg(test)]
mod tests {
    /// The crate-wide half of the arena-index rule: no *implementation* file in this crate names
    /// `Rc` or `RefCell`. `world.rs`'s `world_is_send_and_sync` test is the type-level
    /// complement — this one catches an `Rc`/`RefCell` that would technically still be
    /// `Send + Sync` (say, behind a `Mutex`) but that the plan forbids outright.
    ///
    /// `lib.rs` itself is deliberately not in the scanned list: it is where this test's own
    /// needle strings live, which would otherwise trivially match themselves, and it carries
    /// no logic beyond re-exports and the error enum.
    #[test]
    fn no_rc_refcell_in_source() {
        let src_dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        for name in ["chunk.rs", "collision.rs", "entity.rs", "world.rs"] {
            let text = std::fs::read_to_string(format!("{src_dir}/{name}")).unwrap();
            for needle in ["Rc<", "RefCell<", "std::rc::Rc", "std::cell::RefCell"] {
                assert!(
                    !text.contains(needle),
                    "{name} contains `{needle}` — the arena rule forbids Rc/RefCell in mc-world"
                );
            }
        }
    }
}
