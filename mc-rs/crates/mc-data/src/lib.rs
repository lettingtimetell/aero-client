#![forbid(unsafe_code)]
#![warn(missing_docs)]

//! Static game tables for Minecraft 1.8.9: block states, item properties, entity types,
//! biomes, potions, enchantments and sound events.
//!
//! It is a leaf crate with no dependencies on the
//! rest of the workspace, and every lookup is an array index into `static` data - no
//! allocation, no hashing, no lazy initialisation - because S08 (world) and S11 (mesher)
//! call [`block_props`] once per block in their hot loops.
//!
//! # Where the data comes from
//!
//! Nothing here is hand-transcribed. `crates/mc-data/extractor/` holds a one-shot Java
//! harness that loads the recompiled vanilla 1.8.9 classes, runs
//! `net.minecraft.init.Bootstrap.register()` so the real registries populate, and reflects
//! them into JSON; `codegen.py` turns that JSON into `src/generated/`. Collision shapes in
//! particular are obtained by *calling* `Block.addCollisionBoxesToList` for every legal
//! state, because 1.8.9 assigns block bounds imperatively and a text parse of the Java
//! misses most of them.
//!
//! Both the extractor and its output are committed, so a rebuild needs neither a JDK nor
//! the Minecraft jars.
//!
//! # Totality
//!
//! A server - hostile or merely odd - can put any `u16` in a chunk section and any id in an
//! `ItemStack`. Wire-facing lookups are total or explicitly optional:
//!
//! * [`block_props`] resolves all 65536 `u16` values, returning air for the ~63000 that are
//!   not registered states. That mirrors `ExtendedBlockStorage.get`, which falls back to
//!   `Blocks.air.getDefaultState()`.
//! * [`try_item_props`] mirrors Java's nullable item-registry lookup; [`item_props`] retains an
//!   explicit compatibility sentinel for wire-facing callers that need a total lookup.
//! * [`biome`] covers all 256 ids.
//! * [`entity_type`], [`potion`] and [`enchantment`] return `Option`.
//!
//! # Example
//!
//! ```
//! use mc_data::{block_props, BlockState, CollisionShape};
//!
//! let state = BlockState::from_raw(1 << 4); // minecraft:stone
//! let props = block_props(state);
//! assert_eq!(props.collision, CollisionShape::Full);
//! assert_eq!(props.light_opacity, 255);
//!
//! // Garbage from the wire resolves to air instead of panicking.
//! assert!(block_props(BlockState::from_raw(0xBEEF)).is_air());
//! ```

use thiserror::Error;

pub mod achievement;
pub mod biome;
pub mod block;
pub mod entity;
pub mod item;
pub mod misc;
pub mod recipe;
pub mod stat;
pub mod v5;

mod generated;

pub use crate::achievement::{
    achievement_by_name, achievement_by_stat_id, achievement_roots, achievements, Achievement,
    AchievementInfo,
};
pub use crate::biome::{biome, biomes, Biome};
pub use crate::block::{
    block_by_name, block_info, block_of, block_props, can_catch_fire, fire_encouragement,
    fire_flammability, fire_info, java_state_properties, java_state_variant_key, legal_states,
    meta_of, state_properties, state_variant_key, Aabb, BlockId, BlockInfo, BlockProps, BlockState,
    CollisionShape, DebugWorldState, FireInfo, RenderLayer, RenderType, MAX_BLOCK_ID, STATE_COUNT,
};
pub use crate::entity::{
    entity_by_name, entity_type, entity_types, object_type, EntityInfo, EntityType, ObjectType,
};
pub use crate::generated::materials::{Material, MaterialInfo};
pub use crate::item::{
    item_by_name, item_props, items, record_info, try_item_props, ArmorMaterial, ArmorMaterialInfo,
    ItemId, ItemKind, ItemProps, RecordInfo, ToolMaterial, ToolMaterialInfo, MAX_ITEM_ID, RECORDS,
};
pub use crate::misc::{
    enchantment, enchantments, potion, potion_effects_from_damage, potion_liquid_color_from_damage,
    potions, sound_event, sound_events, EnchantmentInfo, EnchantmentTarget, PotionEffect,
    PotionStackEffect, SoundEvent,
};
pub use crate::recipe::{
    dynamic_remaining_items, evaluate_dynamic_recipe, find_matching_recipe,
    find_matching_vanilla_recipe, find_matching_vanilla_recipe_dynamic, remaining_items,
    repair_result, smelting_result, sort_static_recipes, vanilla_recipes, vanilla_static_recipes,
    CraftingGrid, DynamicRecipeEvaluation, DynamicRecipeKind, DynamicRecipeMatch,
    DynamicRecipeResult, DynamicRecipeTransform, DynamicRemainingItem, DynamicSlotData, Ingredient,
    RecipeStack, ShapedRecipe, ShapelessRecipe, SmeltingRecipe, StaticRecipe, VanillaRecipe,
    SMELTING_RECIPES, VANILLA_RECIPES, WILDCARD_DAMAGE,
};
pub use crate::stat::{
    all_stats, block_stats, general_stats, item_stats, mob_stats, stat_by_id, BlockStatAssociation,
    ItemStatAssociation, MobStatAssociation, StatFormat, StatInfo, StatKind,
};

/// Errors returned by `mc-data`.
///
/// The lookup API is deliberately total, so this is only produced by the fallible name
/// parsers.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum DataError {
    /// A registry name that does not resolve, e.g. `minecraft:not_a_block`.
    #[error("mc-data: unknown {registry} name `{name}`")]
    UnknownName {
        /// Which registry was searched (`block`, `item`, ...).
        registry: &'static str,
        /// The name that failed to resolve.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The S01 acceptance test: **every** one of the 4096 packed state ids resolves
    /// without panicking, and the illegal ones come back as air.
    ///
    /// The loop actually covers the whole `u16` range, since that is what a chunk section
    /// can physically contain.
    #[test]
    fn every_state_id_resolves_without_panic() {
        let mut legal = 0usize;
        for id in 0..=u16::MAX {
            let state = BlockState::from_raw(id);
            let props = block_props(state);

            // Consistency: the decoded halves always agree with the raw packing.
            assert_eq!(block_of(state).0, id >> 4);
            assert_eq!(meta_of(state), (id & 0x0F) as u8);

            if props.legal {
                legal += 1;
                assert_eq!(props.block.0, id >> 4, "state {id:#06x}");
                assert!(id < STATE_COUNT as u16, "legal state outside the table");
            } else {
                assert!(
                    props.is_air(),
                    "illegal state {id:#06x} must resolve to air"
                );
                assert_eq!(props.block, BlockId::AIR);
            }

            // Touching every derived accessor proves none of them can panic either.
            let _ = props.collision.boxes();
            let _ = props.material.blocks_movement();
            let _ = props.bounds.is_empty();
            let _ = state_variant_key(state);
            let _ = state_properties(state);
        }
        assert_eq!(legal, 1394, "1.8.9 registers 1394 block states");
    }

    #[test]
    fn frozen_api_shapes() {
        // The signatures §7 froze for S01.
        let s: BlockState = BlockState { id: 1 << 4 };
        let _: BlockId = block_of(s);
        let _: u8 = meta_of(s);
        let _: &'static BlockProps = block_props(s);
        let _: &'static ItemProps = item_props(ItemId(1));
        let _: Option<EntityType> = entity_type(54);
        let _: &'static Biome = biome(1);
    }

    #[test]
    fn air_is_the_sentinel_for_everything_unregistered() {
        for id in [0u16, 0x0FF0, 0xF00F, 0xFFFF] {
            let p = block_props(BlockState::from_raw(id));
            if !p.legal {
                assert!(p.is_air());
                assert_eq!(p.collision, CollisionShape::Empty);
                assert_eq!(p.material, Material::Air);
            }
        }
    }

    #[test]
    fn block_fire_init_has_every_named_1_8_9_entry_and_nothing_material_derived() {
        // `BlockFire.init`, in source order (with oak's actual 1.8.9 registry names).  Keeping
        // this enumeration explicit makes a missing entry or a tempting "all wood burns"
        // generalisation fail visibly.
        let entries = [
            ("planks", (5, 20)),
            ("double_wooden_slab", (5, 20)),
            ("wooden_slab", (5, 20)),
            ("fence_gate", (5, 20)),
            ("spruce_fence_gate", (5, 20)),
            ("birch_fence_gate", (5, 20)),
            ("jungle_fence_gate", (5, 20)),
            ("dark_oak_fence_gate", (5, 20)),
            ("acacia_fence_gate", (5, 20)),
            ("fence", (5, 20)),
            ("spruce_fence", (5, 20)),
            ("birch_fence", (5, 20)),
            ("jungle_fence", (5, 20)),
            ("dark_oak_fence", (5, 20)),
            ("acacia_fence", (5, 20)),
            ("oak_stairs", (5, 20)),
            ("birch_stairs", (5, 20)),
            ("spruce_stairs", (5, 20)),
            ("jungle_stairs", (5, 20)),
            ("log", (5, 5)),
            ("log2", (5, 5)),
            ("leaves", (30, 60)),
            ("leaves2", (30, 60)),
            ("bookshelf", (30, 20)),
            ("tnt", (15, 100)),
            ("tallgrass", (60, 100)),
            ("double_plant", (60, 100)),
            ("yellow_flower", (60, 100)),
            ("red_flower", (60, 100)),
            ("deadbush", (60, 100)),
            ("wool", (30, 60)),
            ("vine", (15, 100)),
            ("coal_block", (5, 5)),
            ("hay_block", (60, 20)),
            ("carpet", (60, 20)),
        ];
        assert_eq!(entries.len(), 35, "each BlockFire.init call is represented");

        for (name, expected) in entries {
            let block = block_by_name(name).expect("BlockFire.init registry block");
            assert_eq!(
                fire_info(block),
                FireInfo {
                    encouragement: expected.0,
                    flammability: expected.1,
                },
                "{name}"
            );
            assert_eq!(fire_encouragement(block), expected.0, "{name}");
            assert_eq!(fire_flammability(block), expected.1, "{name}");
            assert!(can_catch_fire(block), "{name}");
        }

        // Identity map semantics: maps are keyed by `Block`, not metadata.  All six plank
        // species and all states of log have their block-wide values.
        for meta in 0..=5 {
            assert_eq!(
                fire_info(BlockState::new(BlockId(5), meta).block()),
                FireInfo {
                    encouragement: 5,
                    flammability: 20,
                }
            );
        }
        for meta in 0..=15 {
            assert_eq!(
                fire_info(BlockState::new(BlockId(17), meta).block()),
                FireInfo {
                    encouragement: 5,
                    flammability: 5,
                }
            );
        }

        // These are the important negative controls: properties/materials do not fill the
        // maps.  1.8.9 notably omitted its two newer wooden-stair blocks from `init`.
        for name in ["stone", "sapling", "acacia_stairs", "dark_oak_stairs"] {
            let block = block_by_name(name).unwrap();
            assert_eq!(
                fire_info(block),
                FireInfo::default(),
                "{name} is absent from BlockFire's identity maps"
            );
            assert!(
                !can_catch_fire(block),
                "{name} must not be inferred from its material/class"
            );
        }
        assert_eq!(fire_info(BlockId(u16::MAX)), FireInfo::default());
        assert!(!can_catch_fire(BlockId(u16::MAX)));
    }

    #[test]
    fn tables_are_self_consistent() {
        for state in legal_states() {
            let p = block_props(state);
            assert!(p.legal);
            assert!(p.block.is_registered(), "state {:#06x}", state.id);
            assert_eq!(p.block.info().id, p.block);
            assert!(p.light_emission <= 15);
            assert!(!p.block.info().name.is_empty());
            for b in p.collision.boxes() {
                assert!(b.min_x <= b.max_x && b.min_y <= b.max_y && b.min_z <= b.max_z);
            }
        }
    }
}
