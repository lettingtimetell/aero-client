//! The 1.7.10 creative inventory's lists: what each tab shows and the pool the search tab
//! filters, in 1.7.10's order (java17: `CreativeTabs.displayAllReleventItems`,
//! `GuiContainerCreative.updateCreativeSearch`).
//!
//! They come from 1.7.10's own registry (`creative_table.rs`, generated from the 1.7.10 classes):
//! only items 1.7.10 registers, with 1.7.10's sub-items (stone without granite, dirt with podzol,
//! no banners or prismarine), so a stack picked from a tab, and sent in a creative inventory
//! action, is one a 1.7.10 server knows.

use crate::item::ItemId;

include!("creative_table.rs");

/// One stack a 1.7.10 creative list shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeStackV5 {
    /// One item of this id and damage, without a tag.
    Item {
        /// The item.
        id: ItemId,
        /// Its damage (the sub-item).
        damage: i16,
    },
    /// An enchanted book storing one enchantment at one level
    /// (java17 `ItemEnchantedBook.getEnchantedItemStack`).
    EnchantedBook {
        /// The enchantment id.
        enchantment: u8,
        /// Its level.
        level: u8,
    },
}

const fn item(id: u16, damage: i16) -> CreativeStackV5 {
    CreativeStackV5::Item {
        id: ItemId(id),
        damage,
    }
}

const fn book(enchantment: u8, level: u8) -> CreativeStackV5 {
    CreativeStackV5::EnchantedBook { enchantment, level }
}

/// What the 1.7.10 creative tab `label` lists (`"buildingBlocks"`, `"tools"`, ...), in its
/// order. Empty for the search and inventory tabs, whose grids are not lists of their own, and
/// for a label 1.7.10 has no tab for.
#[must_use]
pub fn creative_tab_v5(label: &str) -> &'static [CreativeStackV5] {
    CREATIVE_TABS_V5
        .iter()
        .find(|(tab, _)| *tab == label)
        .map_or(&[], |(_, stacks)| stacks)
}

/// The 1.7.10 creative tabs' labels, by tab index.
pub fn creative_tab_labels_v5() -> impl Iterator<Item = &'static str> {
    CREATIVE_TABS_V5.iter().map(|(label, _)| *label)
}

/// The stacks the 1.7.10 search tab filters by their tooltips, in its order: every item with a
/// creative tab, with its sub-items, then an enchanted book for every level of every enchantment.
#[must_use]
pub fn creative_search_pool_v5() -> &'static [CreativeStackV5] {
    &CREATIVE_SEARCH_V5
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::item::try_item_props;
    use crate::v5::{enchantment_v5, is_item_registered_v5};

    fn all() -> impl Iterator<Item = &'static CreativeStackV5> {
        CREATIVE_TABS_V5
            .iter()
            .flat_map(|(_, stacks)| stacks.iter())
            .chain(creative_search_pool_v5())
    }

    #[test]
    fn the_tabs_are_1_7_10s_twelve_in_order() {
        assert_eq!(
            creative_tab_labels_v5().collect::<Vec<_>>(),
            [
                "buildingBlocks",
                "decorations",
                "redstone",
                "transportation",
                "misc",
                "search",
                "food",
                "tools",
                "combat",
                "brewing",
                "materials",
                "inventory"
            ]
        );
        assert!(creative_tab_v5("search").is_empty() && creative_tab_v5("inventory").is_empty());
        assert!(creative_tab_v5("hotbar").is_empty());
    }

    #[test]
    fn every_stack_is_an_item_or_enchantment_1_7_10_registers() {
        for stack in all() {
            match *stack {
                CreativeStackV5::Item { id, .. } => {
                    assert!(is_item_registered_v5(id), "item {}", id.0);
                }
                CreativeStackV5::EnchantedBook { enchantment, level } => {
                    let info = enchantment_v5(enchantment).expect("a 1.7.10 enchantment");
                    assert!(
                        level >= 1 && level <= info.max_level,
                        "book {enchantment} {level}"
                    );
                }
            }
        }
        // Items 1.8.9 added are nowhere.
        for id in [165, 166, 168, 169, 179, 409, 410, 411, 423, 425, 427] {
            assert!(
                all().all(|stack| *stack != item(id, 0)),
                "{id} is a 1.8.9 item"
            );
            assert!(try_item_props(ItemId(id)).is_some());
        }
    }

    #[test]
    fn sub_items_are_1_7_10s() {
        let building = creative_tab_v5("buildingBlocks");
        // Stone has no variants in 1.7.10; dirt has podzol (2) but no coarse dirt (1).
        assert_eq!(
            &building[..4],
            &[item(1, 0), item(2, 0), item(3, 0), item(3, 2)]
        );
        // The tools tab ends with its enchanted books, at their highest level.
        let tools = creative_tab_v5("tools");
        assert!(matches!(
            tools.last(),
            Some(CreativeStackV5::EnchantedBook { .. })
        ));
        // The search pool lists every level of a book: efficiency (32) I to V.
        let efficiency: Vec<u8> = creative_search_pool_v5()
            .iter()
            .filter_map(|stack| match *stack {
                CreativeStackV5::EnchantedBook {
                    enchantment: 32,
                    level,
                } => Some(level),
                _ => None,
            })
            .collect();
        assert_eq!(efficiency, [1, 2, 3, 4, 5]);
    }
}
