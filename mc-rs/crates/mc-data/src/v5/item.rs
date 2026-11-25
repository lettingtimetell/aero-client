//! The 1.7.10 item registry, as the client's item logic reads it.
//!
//! 1.7.10 has twelve block items 1.8.9 lacks (water, lava, fire, portal, cocoa, ...: java17
//! `Item.registerItems` gives every block an item outside a fixed exclusion list), lacks the 34
//! items 1.8.9 added, and differs from 1.8.9 on nine shared items: doors stack to 1 (java17:
//! `ItemDoor.<init>`), stone, grass and sponge are plain `ItemBlock`s without subtypes, item 350
//! is `cooked_fished`, the blaze rod is held flat, carrots heal 4 and baked potatoes 6 (java17:
//! `Item.registerItems`; `extractor/codegen_v5.py` cites each line). Everything else is the 1.8.9
//! row.

use crate::item::{try_item_props, ItemId, ItemProps};

use super::generated::{V5_ITEMS_ONLY_IN_1_8_9, V5_ITEM_OVERLAY};

/// 1.7.10's `Item.getItemById`: the item's properties as 1.7.10 has them, `None` where 1.7.10
/// registers no item.
///
/// Every field a 1.8.9 caller reads means the same here, including
/// [`ItemProps::food_stats`]: carrots and baked potatoes return their 1.7.10 heal amounts.
///
/// # Example
/// ```
/// use mc_data::v5::try_item_props_v5;
/// use mc_data::{try_item_props, ItemId};
///
/// // Doors stack to one in 1.7.10.
/// assert_eq!(try_item_props_v5(ItemId(324)).unwrap().max_stack_size, 1);
/// // 1.7.10 has a water item; 1.8.9 does not.
/// assert!(try_item_props_v5(ItemId(9)).is_some() && try_item_props(ItemId(9)).is_none());
/// // Prismarine shards are 1.8.9 only.
/// assert!(try_item_props_v5(ItemId(409)).is_none());
/// ```
#[must_use]
pub fn try_item_props_v5(id: ItemId) -> Option<&'static ItemProps> {
    if let Ok(i) = V5_ITEM_OVERLAY.binary_search_by_key(&id.0, |props| props.id.0) {
        return Some(&V5_ITEM_OVERLAY[i]);
    }
    if V5_ITEMS_ONLY_IN_1_8_9.binary_search(&id.0).is_ok() {
        return None;
    }
    try_item_props(id)
}

/// Whether 1.7.10 registers item `id`.
#[inline]
#[must_use]
pub fn is_item_registered_v5(id: ItemId) -> bool {
    try_item_props_v5(id).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item::{items, MAX_ITEM_ID};

    fn v5(id: u16) -> &'static ItemProps {
        try_item_props_v5(ItemId(id)).expect("1.7.10 item")
    }

    #[test]
    fn registry_size_and_membership() {
        let count = (0..=2300u16)
            .filter(|&id| is_item_registered_v5(ItemId(id)))
            .count();
        // java17 Item.itemRegistry: 315 items (1.8.9's 337, minus 34, plus 12).
        assert_eq!(count, 315);
        assert_eq!(items().len(), 337);
        for id in [8, 9, 10, 11, 43, 51, 90, 119, 125, 127, 141, 142] {
            assert!(
                try_item_props(ItemId(id)).is_none(),
                "{id} is not a 1.8.9 item"
            );
            let props = v5(id);
            assert_eq!(props.id, ItemId(id));
            assert_eq!(props.block.map(|b| b.0), Some(id));
            assert_eq!(props.max_stack_size, 64);
            assert_eq!(props.has_subtypes, id == 43 || id == 125, "{id}");
        }
        for id in [165, 166, 179, 192, 409, 416, 423, 425, 427, 431] {
            assert!(try_item_props(ItemId(id)).is_some(), "{id} is a 1.8.9 item");
            assert!(!is_item_registered_v5(ItemId(id)), "{id}");
        }
        assert!(!is_item_registered_v5(ItemId(MAX_ITEM_ID + 1)));
    }

    #[test]
    fn shared_items_with_1_7_10_values() {
        assert_eq!(v5(324).max_stack_size, 1);
        assert_eq!(v5(330).max_stack_size, 1);
        assert_eq!(v5(350).name, "minecraft:cooked_fished");
        assert!(!v5(369).full_3d);
        assert!(!v5(1).has_subtypes && !v5(19).has_subtypes);
        let carrot = v5(391).food_stats(0);
        let potato = v5(393).food_stats(0);
        assert_eq!((carrot.heal, potato.heal), (4, 6));
        // Negative controls: the 1.8.9 rows hold the other values.
        let props18 = |id| try_item_props(ItemId(id)).unwrap();
        assert_eq!(props18(324).max_stack_size, 64);
        assert_eq!(props18(391).food_stats(0).heal, 3);
        assert_eq!(props18(393).food_stats(0).heal, 5);
        assert!(props18(369).full_3d);
        // The fish rows still read the metadata-specific food values.
        assert_eq!(
            v5(350).food_stats(1),
            props18(350).food_stats(1),
            "cooked salmon is unchanged"
        );
    }

    #[test]
    fn every_other_shared_item_is_the_1_8_9_row() {
        let patched = [1, 2, 19, 324, 330, 350, 369, 391, 393];
        for props in items() {
            let id = props.id.0;
            match try_item_props_v5(props.id) {
                Some(v5) if patched.contains(&id) => assert_ne!(v5, props, "{id}"),
                Some(v5) => assert!(std::ptr::eq(v5, props), "{id} must be the 1.8.9 row"),
                None => assert!(V5_ITEMS_ONLY_IN_1_8_9.contains(&id), "{id}"),
            }
        }
    }

    /// Each patched row is the 1.8.9 row with exactly the 1.7.10 fields changed, so a newer 1.8.9
    /// extraction cannot leave the overlay with stale copies of the other fields.
    #[test]
    fn patched_rows_differ_from_1_8_9_only_in_the_patched_fields() {
        let row18 = |id| *try_item_props(ItemId(id)).unwrap();
        let block_item = "net.minecraft.item.ItemBlock";
        let mut expected = Vec::new();
        for id in [1, 19] {
            let mut row = row18(id);
            row.java_class = block_item;
            row.has_subtypes = false;
            expected.push(row);
        }
        let mut grass = row18(2);
        grass.java_class = block_item;
        expected.push(grass);
        for id in [324, 330] {
            let mut door = row18(id);
            door.max_stack_size = 1;
            expected.push(door);
        }
        let mut fish = row18(350);
        fish.name = "minecraft:cooked_fished";
        expected.push(fish);
        let mut rod = row18(369);
        rod.full_3d = false;
        expected.push(rod);
        for (id, heal) in [(391, 4), (393, 6)] {
            let mut food = row18(id);
            food.food_heal = heal;
            expected.push(food);
        }
        for row in expected {
            assert_eq!(v5(row.id.0), &row, "{}", row.id.0);
        }
    }

    /// The twelve 1.7.10-only block items are plain block items of their block.
    #[test]
    fn block_items_only_1_7_10_has_mirror_their_block() {
        for id in [8, 9, 10, 11, 43, 51, 90, 119, 125, 127, 141, 142] {
            let props = v5(id);
            let block = crate::block::BlockId(id).info();
            assert_eq!(props.name, block.name, "{id}");
            assert_eq!(props.unlocalized, block.unlocalized, "{id}");
            assert_eq!(props.display_name, block.display_name, "{id}");
            assert_eq!(props.kind, crate::item::ItemKind::Block, "{id}");
            assert_eq!(
                (
                    props.max_damage,
                    props.damageable,
                    props.full_3d,
                    props.enchantability
                ),
                (0, false, false, 0),
                "{id}"
            );
            assert_eq!(props.creative_tab, None, "{id}");
        }
    }
}
