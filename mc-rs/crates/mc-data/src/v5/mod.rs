//! Tables for the 1.7.10 (protocol 5) game version.
//!
//! The client runs 1.7.10 sessions on its 1.8.9 representations: 1.8.9 block states, 1.8.9
//! particle ids, 1.8.9 item and statistic tables. This module holds what turns 1.7.10 data into
//! those representations exactly, and the 1.7.10 values that differ from 1.8.9's. Only
//! protocol-5 code calls it: nothing in the 1.8.9 lookups (`block_props`, `item_props`, ...)
//! reads these tables, so a 1.8.9 session neither changes nor pays for them.
//!
//! | module | covers |
//! |---|---|
//! | block | chunk cells, block changes, falling and carried blocks: [`state_from_v5`] |
//! | item | the 1.7.10 item registry: [`try_item_props_v5`] |
//! | particle | `S2APacketParticles` names: [`particle_from_v5_name`] |
//! | sound | events 1.7.10 knows and its door file lists: [`sound_event_v5`] |
//! | stat | `S37PacketStatistics` ids: [`stat_from_v5`] |
//! | potion | potion and enchantment ids: [`potion_v5`], [`enchantment_v5`] |
//! | placement | materials and placement rules: [`material_v5`] |
//! | recipe | the crafting manager: [`find_matching_recipe_v5`] |
//! | creative | the creative tabs' lists: [`creative_tab_v5`] |
//!
//! Every value cites the 1.7.10 source it reproduces (`java17: Class.method`).

mod block;
mod creative;
mod generated;
mod item;
mod particle;
mod placement;
mod potion;
mod recipe;
mod sound;
mod stat;

pub use block::{
    is_block_registered_v5, state_from_v5, state_from_v5_enderman, state_from_v5_falling_block,
    state_from_v5_id_meta, write_section_blob_v5,
};
pub use creative::{
    creative_search_pool_v5, creative_tab_labels_v5, creative_tab_v5, CreativeStackV5,
};
pub use item::{is_item_registered_v5, try_item_props_v5};
pub use particle::{ignores_range_v5, particle_from_v5_name, Particle5, ParticleV5};
pub use placement::{has_solid_top_surface_v5, item_block_replaces_clicked_v5, material_v5};
pub use potion::{enchantment_v5, potion_v5};
pub use recipe::{
    find_matching_recipe_v5, recipes_v5, repair_result_v5, RecipeResultV5, RecipeTransformV5,
    RecipeV5, SlotTagV5, TagEntryV5,
};
pub use sound::{sound_event_v5, sound_files_v5};
pub use stat::stat_from_v5;
