//! Entity data for the 1.7.10 (protocol 5) game version.
//!
//! Protocol-5 packet handling turns 1.7.10 entity data into the client's 1.8.9 entity model with
//! these: which mob and object ids exist, each entity's 1.7.10 box, eye height and `yOffset`,
//! the 1.7.10 `DataWatcher` registrations, and the remap of 1.7.10 metadata onto the 1.8.9
//! indices. Nothing here is reachable from a 1.8.9 session. The metadata remap is the second
//! place this crate uses an `mc-protocol` type (the decoded metadata list), next to
//! [`crate::entity::MetadataValue`]'s item stack.

mod entity;
mod metadata;

pub use entity::{
    ageable_scale_v5, collision_border_v5, exists_in_v5, geometry_v5, mob_type_v5, object_type_v5,
    slime_dimensions_v5, EntityGeometry5, LOCAL_PLAYER_GEOMETRY_V5, REMOTE_PLAYER_GEOMETRY_V5,
    WITHER_SKELETON_DIMENSIONS_V5,
};
pub use metadata::{
    growing_age_v5, player_skin_parts_v5, registered_type_tag_v5, remap_metadata_v5,
    PLAYER_DEFAULT_SKIN_PARTS_V5,
};
