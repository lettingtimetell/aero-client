//! Which blocks the Tall Grass mod hides when it is turned off.
//!
//! The mod is named after the vanilla feature and defaults **on** (grass visible, vanilla). Off,
//! the grass foliage is neither meshed ([`super::MeshSettings::hide_tall_grass`]) nor hit by the
//! F5 camera ray (`mc-client`'s `w5_grass`). The world itself never changes, so targeting,
//! breaking, collisions and packets stay vanilla.
//!
//! "Grass" is exactly vanilla's grass and fern foliage:
//!
//! | block | stored state | vanilla variant |
//! |---|---|---|
//! | 31 `tallgrass` | meta 1, 2 | `tall_grass`, `fern` |
//! | 175 `double_plant`, lower half | meta 2, 3 | `double_grass`, `double_fern` |
//! | 175 `double_plant`, upper half | meta bit 8, over a lower half of meta 2 or 3 | the same |
//!
//! Kept visible on purpose: the flowers (37, 38, and `double_plant` 0, 1, 4, 5), the
//! `tallgrass` shrub (meta 0) and the dead bush (32). The shrub and the dead bush draw the same
//! `dead_bush` model, so hiding only the one that shares grass's block id would leave an
//! identical-looking plant behind; neither is the green foliage the mod is about.
//!
//! java: `BlockTallGrass.EnumType` (`DEAD_BUSH(0)`, `GRASS(1)`, `FERN(2)`),
//! `BlockDoublePlant.EnumPlantType` (`GRASS(2)`, `FERN(3)`), and
//! `BlockDoublePlant.getActualState`, which gives the upper half the variant of the block below
//! it when that block is a double plant.

use mc_data::{BlockId, BlockState};
use mc_math::BlockPos;
use mc_world::World;

/// `tallgrass`, `type=tall_grass`.
const TALL_GRASS: BlockState = BlockState::new(BlockId(31), 1);
/// `tallgrass`, `type=fern`.
const FERN: BlockState = BlockState::new(BlockId(31), 2);
/// `double_plant`.
const DOUBLE_PLANT: BlockId = BlockId(175);
/// `double_plant`, `half=lower`, `variant=double_grass`.
const DOUBLE_GRASS_LOWER: BlockState = BlockState::new(DOUBLE_PLANT, 2);
/// `double_plant`, `half=lower`, `variant=double_fern`.
const DOUBLE_FERN_LOWER: BlockState = BlockState::new(DOUBLE_PLANT, 3);
/// `BlockDoublePlant.getStateFromMeta`: metadata bit 8 is the upper half.
const UPPER_HALF_BIT: u16 = 8;

/// Whether the block at `pos` (whose stored state is `state`) is grass the mod hides.
///
/// Two packed-id compares for every block; one extra world read only for the upper half of a
/// double plant, whose own metadata has no room for its variant.
#[inline]
#[must_use]
pub fn is_hidden_tall_grass(world: &World, state: BlockState, pos: BlockPos) -> bool {
    if state == TALL_GRASS || state == FERN || is_double_grass_lower(state) {
        return true;
    }
    state.block() == DOUBLE_PLANT
        && state.id & UPPER_HALF_BIT != 0
        && is_double_grass_lower(world.block_at(pos.down()))
}

/// The lower half of double tall grass or a large fern. An upper half (bit 8) never matches,
/// which is `getActualState`'s rule: an upper half over an upper half keeps its own stored
/// variant, the last registered tuple (the peony), so it is not grass.
#[inline]
const fn is_double_grass_lower(state: BlockState) -> bool {
    state.id == DOUBLE_GRASS_LOWER.id || state.id == DOUBLE_FERN_LOWER.id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::fixture;
    use mc_data::block::ActualBlockState;

    fn property(state: BlockState, name: &str) -> Option<&'static str> {
        ActualBlockState::from_stored(state).get(name)
    }

    #[test]
    fn the_constants_are_the_vanilla_grass_and_fern_states() {
        assert_eq!(mc_data::block_by_name("tallgrass"), Some(BlockId(31)));
        assert_eq!(mc_data::block_by_name("double_plant"), Some(DOUBLE_PLANT));
        assert_eq!(mc_data::block_by_name("deadbush"), Some(BlockId(32)));
        assert_eq!(property(TALL_GRASS, "type"), Some("tall_grass"));
        assert_eq!(property(FERN, "type"), Some("fern"));
        assert_eq!(
            property(BlockState::new(BlockId(31), 0), "type"),
            Some("dead_bush"),
            "meta 0 is the shrub, which stays"
        );
        assert_eq!(
            property(DOUBLE_GRASS_LOWER, "variant"),
            Some("double_grass")
        );
        assert_eq!(property(DOUBLE_FERN_LOWER, "variant"), Some("double_fern"));
        for lower in [DOUBLE_GRASS_LOWER, DOUBLE_FERN_LOWER] {
            assert_eq!(property(lower, "half"), Some("lower"));
        }
        for meta in 8..12 {
            assert_eq!(
                property(BlockState::new(DOUBLE_PLANT, meta), "half"),
                Some("upper"),
                "meta {meta}"
            );
        }
    }

    #[test]
    fn grass_and_ferns_are_hidden_and_every_other_plant_stays() {
        let mut world = fixture::blank_world(0);
        let pos = BlockPos::new(8, 64, 8);
        let hidden = |world: &World, state: BlockState| is_hidden_tall_grass(world, state, pos);
        assert!(hidden(&world, TALL_GRASS));
        assert!(hidden(&world, FERN));
        assert!(!hidden(&world, BlockState::new(BlockId(31), 0)), "shrub");
        assert!(!hidden(&world, fixture::state_of("minecraft:deadbush")));
        assert!(!hidden(
            &world,
            fixture::state_of("minecraft:yellow_flower")
        ));
        for meta in 0..9 {
            assert!(
                !hidden(&world, BlockState::new(BlockId(38), meta)),
                "red_flower {meta}"
            );
        }
        for meta in 0..6 {
            let lower = BlockState::new(DOUBLE_PLANT, meta);
            assert_eq!(
                hidden(&world, lower),
                matches!(meta, 2 | 3),
                "lower half {meta}"
            );
        }
        assert!(!hidden(&world, fixture::state_of("minecraft:sapling")));
        assert!(
            !hidden(&world, fixture::state_of("minecraft:grass")),
            "the grass block"
        );
        assert!(!hidden(&world, BlockState::AIR));

        // Upper halves follow the block below, as `BlockDoublePlant.getActualState` does.
        let upper = BlockState::new(DOUBLE_PLANT, 10);
        for meta in 0..6 {
            world.set_block(pos.down(), BlockState::new(DOUBLE_PLANT, meta));
            assert_eq!(
                hidden(&world, upper),
                matches!(meta, 2 | 3),
                "over lower {meta}"
            );
        }
        world.set_block(pos.down(), BlockState::new(DOUBLE_PLANT, 10));
        assert!(
            !hidden(&world, upper),
            "an upper half over an upper half is a peony top"
        );
        world.set_block(pos.down(), TALL_GRASS);
        assert!(!hidden(&world, upper), "an orphan upper half is not grass");
        world.set_block(pos.down(), BlockState::AIR);
        assert!(!hidden(&world, upper));
        // Every upper-half facing bit pattern behaves the same.
        world.set_block(pos.down(), DOUBLE_FERN_LOWER);
        for meta in 8..16 {
            assert!(
                hidden(&world, BlockState::new(DOUBLE_PLANT, meta)),
                "upper {meta}"
            );
        }
    }
}
