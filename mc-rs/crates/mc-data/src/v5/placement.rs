//! Block data 1.7.10's placement rules read where 1.8.9's differ.
//!
//! All three predicates take the stored (decoded) state. Its meta is the raw 1.7.10 meta for
//! every meta these rules read: stairs, slabs and snow layers keep their bits through the
//! decode (`super::state_from_v5`).

use crate::block::{block_props, BlockId, BlockState};
use crate::Material;

/// `minecraft:double_plant`.
const DOUBLE_PLANT: BlockId = BlockId(175);

/// A block's material in 1.7.10: 1.8.9's, except the double plant, which is `plants` (not
/// replaceable) in 1.7.10 and `vine` (replaceable) in 1.8.9 (java17: `BlockDoublePlant.<init>`,
/// `block/BlockDoublePlant.java:30`; java18: `block/BlockDoublePlant.java:34`). Every other
/// shared material is the same, flag for flag (both `Material` classes compared).
///
/// `World.canPlaceEntityOnSide` (java17: `world/World.java:3104-3109`) refuses a placement into a
/// block whose material is not replaceable, so in 1.7.10 nothing can be placed into a double
/// plant.
#[inline]
#[must_use]
pub fn material_v5(state: BlockState) -> Material {
    let props = block_props(state);
    if props.block == DOUBLE_PLANT {
        Material::Plants
    } else {
        props.material
    }
}

/// Whether 1.7.10's `ItemBlock.onItemUse` places into the clicked block itself rather than next
/// to it: a snow layer with `(meta & 7) < 1`, vine, tall grass or a dead bush, hard-coded
/// (java17: `ItemBlock.onItemUse`, `item/ItemBlock.java:42-75`).
///
/// 1.8.9 asks `Block.isReplaceable` instead (java18: `ItemBlock.onItemUse`,
/// `item/ItemBlock.java:38-46`), which adds the double-plant grass and fern. Other item placers
/// (slabs, snow, reeds, doors, beds, signs, skulls) have their own rules.
#[inline]
#[must_use]
pub fn item_block_replaces_clicked_v5(state: BlockState) -> bool {
    let props = block_props(state);
    match props.block.0 {
        78 => state.meta() & 7 < 1,
        106 | 31 | 32 => true,
        _ => false,
    }
}

/// 1.7.10's `World.doesBlockHaveSolidTopSurface` (java17: `world/World.java:2438-2443`): an
/// opaque full cube, the top half of stairs (`meta & 4`) or a slab (`meta & 8`), any hopper, or
/// a snow layer with all 8 layers (`(meta & 7) == 7`; java18 `World.doesBlockHaveSolidTopSurface`
/// takes `LAYERS == 7`, i.e. 7 layers, `world/World.java:2419-2423`).
///
/// Decides torch, rail, redstone, pressure plate, lever, door, bed, flower pot, fire and
/// tripwire placement, and dismount spots. java17's `renderAsNormalBlock` returns what java18's
/// `isFullCube` does for every block both versions register (dumps of both block registries
/// compared), so the 1.8.9 `full_cube` column stands in for it.
#[must_use]
pub fn has_solid_top_surface_v5(state: BlockState) -> bool {
    let props = block_props(state);
    if material_v5(state).is_opaque() && props.full_cube {
        return true;
    }
    let meta = state.meta();
    match props.block.0 {
        // The twelve BlockStairs blocks 1.7.10 registers.
        53 | 67 | 108 | 109 | 114 | 128 | 134 | 135 | 136 | 156 | 163 | 164 => meta & 4 == 4,
        // BlockSlab: the single and double stone and wooden slabs (the doubles are full cubes
        // and already returned above).
        43 | 44 | 125 | 126 => meta & 8 == 8,
        154 => true,
        78 => meta & 7 == 7,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v5::state_from_v5;

    fn s(block: u16, meta: u8) -> BlockState {
        state_from_v5((block << 4) | u16::from(meta))
    }

    #[test]
    fn double_plants_are_plants() {
        for meta in 0..16 {
            let state = s(175, meta);
            assert_eq!(material_v5(state), Material::Plants);
            assert!(!material_v5(state).is_replaceable());
            // 1.8.9's material is the replaceable vine.
            assert_eq!(block_props(state).material, Material::Vine);
            assert!(block_props(state).material.is_replaceable());
        }
        assert_eq!(material_v5(s(1, 0)), block_props(s(1, 0)).material);
        assert_eq!(material_v5(s(31, 1)), Material::Vine);
    }

    #[test]
    fn replace_in_place_list() {
        assert!(item_block_replaces_clicked_v5(s(78, 0)));
        assert!(item_block_replaces_clicked_v5(s(78, 8)), "meta & 7 is 0");
        assert!(!item_block_replaces_clicked_v5(s(78, 1)));
        for (block, meta) in [(106, 0), (106, 15), (31, 0), (31, 1), (31, 2), (32, 0)] {
            assert!(
                item_block_replaces_clicked_v5(s(block, meta)),
                "{block}:{meta}"
            );
        }
        // Double-plant grass and fern are replaced in place only in 1.8.9; air is never clicked.
        for meta in [2, 3, 10] {
            assert!(!item_block_replaces_clicked_v5(s(175, meta)));
        }
        assert!(!item_block_replaces_clicked_v5(s(1, 0)));
    }

    #[test]
    fn solid_top_surface() {
        assert!(has_solid_top_surface_v5(s(1, 0)));
        assert!(!has_solid_top_surface_v5(s(20, 0)), "glass is not opaque");
        assert!(has_solid_top_surface_v5(s(53, 4)) && !has_solid_top_surface_v5(s(53, 0)));
        // A stair meta 1.8.9 does not register keeps its upside-down bit through the decode.
        assert!(has_solid_top_surface_v5(s(53, 12)));
        assert!(has_solid_top_surface_v5(s(44, 8)) && !has_solid_top_surface_v5(s(44, 0)));
        assert!(
            has_solid_top_surface_v5(s(126, 14)),
            "wooden slab 14 decodes to a top slab"
        );
        assert!(has_solid_top_surface_v5(s(43, 0)));
        assert!(has_solid_top_surface_v5(s(154, 0)));
        // Snow: 8 layers in 1.7.10, not 7.
        assert!(has_solid_top_surface_v5(s(78, 7)));
        assert!(has_solid_top_surface_v5(s(78, 15)));
        assert!(!has_solid_top_surface_v5(s(78, 6)));
        assert!(!has_solid_top_surface_v5(s(0, 0)));
    }
}
