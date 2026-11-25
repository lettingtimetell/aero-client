//! The creative screen's lists for a game version: in a 1.7.10 session, 1.7.10's own tab lists
//! and search pool ([`mc_data::v5::creative_tab_v5`], generated from the 1.7.10 classes), so the
//! grid only ever offers, and a creative inventory action only ever sends, items a 1.7.10 server
//! registers, in the order 1.7.10 shows them. A child of `creative`; `creative_screen.rs` asks it
//! for every list it shows.

use mc_assets::Language;
use mc_data::v5::{creative_search_pool_v5, creative_tab_v5, CreativeStackV5};
use mc_version::GameVersion;

use super::{enchanted_book, plain_stack, search_matches, search_stacks, stacks_for_tab};
use crate::screens::container::stack_ext::ItemStack;

/// [`stacks_for_tab`] for a session of `version`. java17: `CreativeTabs.displayAllReleventItems`.
#[must_use]
pub fn stacks_for_tab_in(version: GameVersion, label: &str) -> Vec<ItemStack> {
    match version {
        GameVersion::V1_8_9 => stacks_for_tab(label),
        GameVersion::V1_7_10 => creative_tab_v5(label).iter().map(stack_v5).collect(),
    }
}

/// [`search_stacks`] for a session of `version`: the version's pool, filtered by its tooltip
/// test. java17: `GuiContainerCreative.updateCreativeSearch`, which, unlike 1.8.9, matches each
/// tooltip line with its formatting codes (`:358`), so a query of `9` finds blue lines.
#[must_use]
pub fn search_stacks_in(
    version: GameVersion,
    query: &str,
    advanced_tooltips: bool,
    language: Option<&Language>,
) -> Vec<ItemStack> {
    match version {
        GameVersion::V1_8_9 => search_stacks(query, advanced_tooltips, language),
        GameVersion::V1_7_10 => {
            let needle = query.to_lowercase();
            creative_search_pool_v5()
                .iter()
                .map(stack_v5)
                .filter(|stack| search_matches(stack, &needle, advanced_tooltips, language, false))
                .collect()
        }
    }
}

/// The stack a 1.7.10 list entry stands for: one item, or a book whose `StoredEnchantments` holds
/// the one enchantment (java17 `ItemEnchantedBook.addEnchantment`, the same tag as 1.8.9's).
fn stack_v5(entry: &CreativeStackV5) -> ItemStack {
    match *entry {
        CreativeStackV5::Item { id, damage } => plain_stack(id, damage),
        CreativeStackV5::EnchantedBook { enchantment, level } => enchanted_book(enchantment, level),
    }
}
