#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Player physics, collision, and client-side prediction — section **S13**.
//!
//! Parity here is non-negotiable: errors surface as rubber-banding and anticheat kicks, not as
//! visual glitches. Every constant, every order of operations and every `f32`/`f64` boundary in
//! this crate is transcribed from 1.8.9's `Entity.moveEntity`, `EntityLivingBase
//! .moveEntityWithHeading`, `EntityPlayer` and `EntityPlayerSP`, and is checked against a trace
//! produced by the *real recompiled game* — see `tests/parity.rs` and `javaref/PhysicsRefGen.java`.
//!
//! # Hazards this crate exists to get right
//!
//! **H1 — trig is a table, not libm.** [`mc_math::sin`]/[`mc_math::cos`] read the JVM's own
//! 65,536-entry `SIN_TABLE`. `f32::sin` is *close* but not equal, and the error compounds:
//! after a few hundred ticks of running the position is metres out and the server disagrees.
//! Nothing in this crate may call `f32::sin`, `f64::sin`, or `f64::floor` — use
//! [`mc_math::floor_f64`], whose semantics are `MathHelper.floor_double`'s.
//!
//! **H7 — `f64` throughout, applied Y then X then Z, step-up retried after the first sweep.**
//! [`move_entity`] does exactly that. The one place `f32` is *correct* is inside
//! `Entity.moveFlying`, whose whole body is single-precision before being widened into
//! `motionX`/`motionZ`; [`player::move_flying`] reproduces that width for width. A stray
//! `f64` there is as much a parity bug as a stray `f32` anywhere else.
//!
//! # Layout
//!
//! | module | covers |
//! |---|---|
//! | [`state`] | [`PhysicsState`], [`MoveInput`] and the player's constants |
//! | [`collision`] | gathering world collision boxes, including the 195 neighbour-dependent states |
//! | [`movement`] | [`move_entity`]: the sweep, the sneak edge-guard, step-up, block-collision effects |
//! | [`fluid`] | water/lava detection, `handleMaterialAcceleration` and the flow vector |
//! | [`player`] | [`tick_player`]: one client tick of the local player |
//! | [`prediction`] | client prediction and `S08PacketPlayerPosLook` reconciliation |
//! | [`scene`] | building synthetic worlds for tests and tooling |
//!
//! # Example
//!
//! ```
//! use mc_physics::{tick_player, MoveInput, PhysicsState};
//! use mc_math::DVec3;
//! use mc_world::World;
//!
//! let stone = mc_data::BlockState::new(mc_data::block_by_name("stone").unwrap(), 0);
//! let w = mc_physics::scene::flat(stone, 63, 8);
//! let mut s = PhysicsState::new_player(DVec3::new(0.5, 64.0, 0.5));
//! let input = MoveInput { forward: true, ..MoveInput::default() };
//! for _ in 0..20 {
//!     tick_player(&w, &mut s, &input);
//! }
//! assert!(s.on_ground);
//! assert!(s.pos.z > 0.5, "walking forward at yaw 0 moves +Z");
//! ```

pub mod attribute;
pub mod collision;
pub mod entity_collision;
pub mod fluid;
pub mod movement;
pub mod opaque;
pub mod player;
pub mod prediction;
pub mod scene;
pub mod state;
pub mod v5;

pub use collision::{
    collision_boxes_at, collision_boxes_in, collision_boxes_in_with_entities,
    fence_selection_bounds, EntityCollisionCandidate, EntityCollisionRelation,
};
pub use entity_collision::{
    entity_collision_delta, EntityCollisionBody, EntityCollisionDelta,
    EntityCollisionRelation as PairwiseEntityCollisionRelation,
};
pub use movement::{
    move_entity, move_entity_full, move_entity_full_with_entities, move_entity_with,
    move_entity_with_entities, move_entity_with_entities_full, MoveResult,
};
pub use opaque::{is_entity_inside_opaque_block, physics_state_inside_opaque_block};
pub use player::{tick_player, tick_player_with_entities};
pub use prediction::{PositionCorrection, Predictor};
pub use state::{MoveInput, PhysicsState, StepSound, WaterEntry};

use thiserror::Error;

/// `World.isAreaLoaded(from, to)` — the guard `doBlockCollisions` and
/// `handleMaterialAcceleration` are wrapped in.
///
/// **What this actually evaluates to on a client, stated rather than transcribed.**
/// `World.isAreaLoaded(from, to)` (`world/World.java:274`) passes `allowEmpty = true`, and the
/// private seven-argument form (`:294`) is
///
/// ```text
/// if (yEnd >= 0 && yStart < 256) { for each chunk in the x/z range: isChunkLoaded(i, j, true) }
/// else return false;
/// ```
///
/// with `isChunkLoaded(x, z, true)` reducing to `chunkProvider.chunkExists(x, z)` — a hard-coded
/// `return true` in `ChunkProviderClient` (`client/multiplayer/ChunkProviderClient.java:44`). So
/// on the client the whole call collapses to the **Y-bounds test**, and that is what this
/// reproduces: a box entirely below `y = 0` or entirely at or above `y = 256` skips the scan.
/// The chunk loop is deliberately not reproduced, because reproducing it would be *stricter*
/// than vanilla and would silently disable block effects at a chunk seam.
#[must_use]
pub fn is_area_loaded(
    _w: &mc_world::World,
    from: mc_math::BlockPos,
    to: mc_math::BlockPos,
) -> bool {
    to.y >= 0 && from.y < 256
}

/// Errors returned by `mc-physics`.
///
/// Deliberately small. Almost everything here is **total**: a sweep against an unloaded chunk
/// finds air, an unregistered block state resolves to air, and a tick can always be simulated —
/// so [`tick_player`] and [`move_entity`] cannot fail, and giving them a `Result` would be
/// noise at every call site. Only the prediction history, which is a bounded ring buffer, can
/// be asked for something it no longer holds.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PhysicsError {
    /// A rewind or correction referred to a tick the predictor no longer has history for.
    /// The caller should snap to the server position rather than replaying.
    #[error("mc-physics: no prediction history for tick {0}")]
    HistoryExpired(u64),
}
