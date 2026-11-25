//! Creative-mode inventory: tabs, search, and the click rules specific to picking items out of
//! a tab (as opposed to a real server-synced slot). java: `GuiContainerCreative.java`,
//! `CreativeTabs.java`.
//!
//! # Why this is separate from [`super::model::Container::slot_click`]
//!
//! `GuiContainerCreative.handleMouseClick` does not call `Container.slotClick` for most clicks
//! — it overrides `handleMouseClick` entirely and only *falls back* to the real container's
//! `slotClick` for the "Inventory" tab (which is the player's ordinary [`super::model::ContainerKind::Player`]
//! [`super::model::Container`], just with `creative: true` passed through — already handled by
//! the generic state machine's `mode == 3`/drag-mode-`2` creative gates). Every *other* tab is a
//! read-only, **client-only** virtual grid built from [`mc_data::items()`] — there is no real
//! backing slot, no server sync, and no `ClickWindow` packet for picking an item out of it
//! (vanilla's `CreativeInventoryAction`, sent afterward, carries the *result* of the pick, not
//! the pick itself). So the click rule for that grid is its own small, pure function
//! ([`click_tab_source`]) rather than a `Container` variant.
//!
//! java: `ContainerKind::Player` (this module) — [`super::model::ContainerKind::Player`] is the
//! real Rust type for the always-open inventory.

use super::stack_ext::{self, ItemStack};
use super::tooltip;
use mc_assets::Language;
use mc_data::{EnchantmentTarget, ItemId};
use mc_nbt::{Nbt, NbtCompound};
use mc_version::GameVersion;

/// One creative-inventory tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeTab {
    /// A real item-holding tab, keyed by `ItemProps::creative_tab` (e.g. `"buildingBlocks"`).
    Items(&'static str),
    /// The magnifying-glass tab: shows every item matching the current search text instead of a
    /// fixed set. java: `CreativeTabs.tabAllSearch`.
    Search,
    /// Mirrors the player's real, server-synced inventory (crafting grid, armour, main,
    /// hotbar) rather than a static item list. java: `CreativeTabs.tabInventory`.
    Inventory,
}

/// Every tab, **indexed by vanilla's own `tabIndex`**: `TABS[i].index() == i`.
///
/// This is `CreativeTabs.java`'s static-initialiser order, transcribed from the `new
/// CreativeTabs(<index>, "<label>")` calls, and it is load-bearing rather than cosmetic —
/// `getTabColumn()` is `tabIndex % 6` and `isTabInFirstRow()` is `tabIndex < 6`, so the index
/// *is* the on-screen position. The ten content-tab labels are also exactly `mc_data`'s
/// `ItemProps::creative_tab` values.
// java: CreativeTabs.java:15 onwards.
pub const TABS: &[CreativeTab] = &[
    CreativeTab::Items("buildingBlocks"), // 0
    CreativeTab::Items("decorations"),    // 1
    CreativeTab::Items("redstone"),       // 2
    CreativeTab::Items("transportation"), // 3
    CreativeTab::Items("misc"),           // 4
    CreativeTab::Search,                  // 5 — top row, rightmost
    CreativeTab::Items("food"),           // 6
    CreativeTab::Items("tools"),          // 7
    CreativeTab::Items("combat"),         // 8
    CreativeTab::Items("brewing"),        // 9
    CreativeTab::Items("materials"),      // 10
    CreativeTab::Inventory,               // 11 — bottom row, rightmost
];

/// How many tabs there are. Vanilla's `creativeTabArray` is a fixed `new CreativeTabs[12]`.
pub const TAB_COUNT: u8 = 12;

impl CreativeTab {
    /// This tab's vanilla `tabIndex` — its position in [`TABS`].
    ///
    /// Returns `None` for a hand-built `Items(label)` whose label is not one vanilla registers,
    /// rather than guessing a position for it.
    #[must_use]
    pub fn index(self) -> Option<u8> {
        TABS.iter()
            .position(|t| *t == self)
            .map(|i| u8::try_from(i).unwrap_or(0))
    }

    /// The tab's untranslated label, e.g. `"buildingBlocks"`. java: `CreativeTabs.tabLabel`.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            CreativeTab::Items(label) => label,
            CreativeTab::Search => "search",
            CreativeTab::Inventory => "inventory",
        }
    }

    /// The translation key the tab strip's hover tooltip shows. java:
    /// `CreativeTabs.getTranslatedTabLabel`.
    #[must_use]
    pub fn translation_key(self) -> String {
        format!("itemGroup.{}", self.label())
    }

    /// The panel texture behind this tab's item grid, as a
    /// `textures/gui/container/creative_inventory/` file name. java:
    /// `CreativeTabs.getBackgroundImageName` with `"tab_"` prepended by
    /// `GuiContainerCreative.drawGuiContainerBackgroundLayer`.
    #[must_use]
    pub const fn background_image(self) -> &'static str {
        match self {
            CreativeTab::Search => "tab_item_search.png",
            CreativeTab::Inventory => "tab_inventory.png",
            CreativeTab::Items(_) => "tab_items.png",
        }
    }

    /// Whether this tab replaces the player's inventory with a scrollable item grid. java:
    /// `CreativeTabs.shouldHidePlayerInventory` (`hasScrollbar`) — only `tabInventory` clears it.
    #[must_use]
    pub const fn has_item_grid(self) -> bool {
        !matches!(self, CreativeTab::Inventory)
    }

    /// Whether the tab's own name is drawn over its grid. java:
    /// `CreativeTabs.drawInForegroundOfTab` (`drawTitle`) — only `tabInventory` clears it.
    #[must_use]
    pub const fn draws_title(self) -> bool {
        !matches!(self, CreativeTab::Inventory)
    }

    /// The registry name and metadata of the item drawn on the tab itself. java:
    /// `CreativeTabs.getTabIconItem` / `getIconItemDamage`.
    #[must_use]
    pub fn icon_item(self) -> (&'static str, i16) {
        match self {
            CreativeTab::Items("buildingBlocks") => ("brick_block", 0),
            // `BlockDoublePlant.EnumPlantType.PAEONIA.getMeta()` is 5.
            CreativeTab::Items("decorations") => ("double_plant", 5),
            CreativeTab::Items("redstone") => ("redstone", 0),
            CreativeTab::Items("transportation") => ("golden_rail", 0),
            CreativeTab::Items("misc") => ("lava_bucket", 0),
            CreativeTab::Items("food") => ("apple", 0),
            CreativeTab::Items("tools") => ("iron_axe", 0),
            CreativeTab::Items("combat") => ("golden_sword", 0),
            // java: `Items.potionitem`, whose registry name is `potion`.
            CreativeTab::Items("brewing") => ("potion", 0),
            CreativeTab::Items("materials") => ("stick", 0),
            CreativeTab::Search => ("compass", 0),
            CreativeTab::Inventory => ("chest", 0),
            CreativeTab::Items(_) => ("stone", 0),
        }
    }
}

/// A tab's column in the strip, `0..=5`. java: `CreativeTabs.getTabColumn` — `tabIndex % 6`.
#[must_use]
pub const fn tab_column(index: u8) -> i32 {
    (index % 6) as i32
}

/// Whether a tab sits in the strip *above* the panel rather than below it. java:
/// `CreativeTabs.isTabInFirstRow` — `tabIndex < 6`.
#[must_use]
pub const fn tab_is_first_row(index: u8) -> bool {
    index < 6
}

/// Every stack belonging to `label`, in registry order and then the item's own sub-item order.
/// java: `CreativeTabs.displayAllReleventItems`.
///
/// Vanilla calls `Item.getSubItems` rather than adding one metadata-zero representative. That
/// distinction is nearly half of the menu: wool contributes sixteen stacks, potions contribute
/// sixty-three, spawn eggs twenty-seven, and so on. Enchanted books are appended after the
/// registry walk for Tools and Combat, exactly like `addEnchantmentBooksToList`.
#[must_use]
pub fn stacks_for_tab(label: &str) -> Vec<ItemStack> {
    let mut stacks = Vec::new();
    for props in mc_data::items()
        .iter()
        .filter(|p| p.registered && p.creative_tab == Some(label))
    {
        stacks.extend(sub_items(props.id));
    }

    let targets = relevant_enchantment_targets(label);
    stacks.extend(
        mc_data::enchantments()
            .filter(|enchantment| targets.contains(&enchantment.target))
            .map(|enchantment| enchanted_book(enchantment.id, enchantment.max_level)),
    );
    stacks
}

fn relevant_enchantment_targets(label: &str) -> &'static [EnchantmentTarget] {
    match label {
        "tools" => &[
            EnchantmentTarget::Digger,
            EnchantmentTarget::FishingRod,
            EnchantmentTarget::Breakable,
        ],
        "combat" => &[
            EnchantmentTarget::Armor,
            EnchantmentTarget::ArmorFeet,
            EnchantmentTarget::ArmorHead,
            EnchantmentTarget::ArmorTorso,
            EnchantmentTarget::Bow,
            EnchantmentTarget::Weapon,
        ],
        // `tabMisc` declares ALL in Java, but no enchantment's concrete target is ALL, so the
        // identity comparison appends no books. There is intentionally no catch-all here.
        _ => &[],
    }
}

/// Compatibility view of [`stacks_for_tab`] for callers that only need item ids. Metadata
/// variants therefore appear as repeated ids; the screen itself consumes the stack API.
#[must_use]
pub fn items_for_tab(label: &str) -> Vec<ItemId> {
    stacks_for_tab(label)
        .into_iter()
        .filter(|stack| stack.item_id >= 0)
        .map(|stack| ItemId(stack.item_id as u16))
        .collect()
}

/// The pool the search tab filters. java: `updateCreativeSearch`'s first loop --
/// `for (Item item : Item.itemRegistry) if (item != null && item.getCreativeTab() != null)`.
///
/// The `getCreativeTab() != null` test is load-bearing and easy to drop: 13 registered items
/// carry no tab (`mob_spawner`, `farmland`, `lit_furnace`, both mushroom blocks, `dragon_egg`,
/// `command_block`, `barrier`, `filled_map`, `written_book`, `fireworks`, `enchanted_book`,
/// `command_block_minecart`) and vanilla's search never lists any of them.
fn searchable_pool() -> Vec<ItemStack> {
    let mut stacks = Vec::new();
    for props in mc_data::items()
        .iter()
        .filter(|p| p.registered && p.creative_tab.is_some())
    {
        stacks.extend(sub_items(props.id));
    }
    // `ItemEnchantedBook.getAll`: every level, unlike a content tab's max-level-only books.
    for enchantment in mc_data::enchantments() {
        for level in enchantment.min_level..=enchantment.max_level {
            stacks.push(enchanted_book(enchantment.id, level));
        }
    }
    stacks
}

/// Every searchable item whose `en_US` display name contains `query`, case-insensitively.
///
/// java: `GuiContainerCreative.updateCreativeSearch` (`:341-383`).
///
/// # An empty query lists everything
///
/// Vanilla lowercases the field text and keeps any stack with a tooltip line *containing* it.
/// `"".contains("")` is `true`, so the instant the magnifying-glass tab is clicked -- with
/// `searchField` still empty -- the grid shows the **whole** pool. This used to return
/// `Vec::new()`, which opened the tab blank, and the in-tree test asserting that
/// (`empty_search_returns_nothing_rather_than_everything`) pinned it as if it were the spec.
///
#[must_use]
pub fn search_items(query: &str) -> Vec<ItemId> {
    search_stacks(query, false, None)
        .into_iter()
        .filter(|stack| stack.item_id >= 0)
        .map(|stack| ItemId(stack.item_id as u16))
        .collect()
}

/// Vanilla's creative-search stack expansion and full-tooltip filter.
#[must_use]
pub fn search_stacks(
    query: &str,
    advanced_tooltips: bool,
    language: Option<&Language>,
) -> Vec<ItemStack> {
    let needle = query.to_lowercase();
    searchable_pool()
        .into_iter()
        .filter(|stack| search_matches(stack, &needle, advanced_tooltips, language, true))
        .collect()
}

/// Whether one of `stack`'s tooltip lines contains `needle` (already lowercased): the filter of
/// `GuiContainerCreative.updateCreativeSearch`, which strips each line's formatting codes first
/// when `strip_formatting` (java18 `:372`; java17 `:358` matches the raw line).
fn search_matches(
    stack: &ItemStack,
    needle: &str,
    advanced_tooltips: bool,
    language: Option<&Language>,
    strip_formatting: bool,
) -> bool {
    let mut lines = language.map_or_else(
        || tooltip::build(stack, advanced_tooltips),
        |language| tooltip::build_localized(stack, advanced_tooltips, language),
    );
    if let Some((id, level)) = stored_enchantment(stack) {
        if let Some(info) = mc_data::enchantment(id) {
            lines.push(format!("{} {}", info.display_name, level));
        }
    }
    lines.into_iter().any(|line| {
        let line = if strip_formatting {
            strip_formatting_codes(&line)
        } else {
            line
        };
        line.to_lowercase().contains(needle)
    })
}

// The lists of a 1.7.10 session's creative screen (1.7.10's own items and order).
#[path = "creative_v5.rs"]
mod v5;
pub use v5::{search_stacks_in, stacks_for_tab_in};

/// The content tab named by a search-result stack, including vanilla's single-enchantment book
/// inference. Tabs are checked in array order, so Unbreaking names Tools before Combat.
#[must_use]
pub fn source_tab(stack: &ItemStack) -> Option<CreativeTab> {
    let props = stack_ext::props(stack);
    if let Some(label) = props.creative_tab {
        return TABS
            .iter()
            .copied()
            .find(|tab| matches!(tab, CreativeTab::Items(candidate) if *candidate == label));
    }
    let (id, _) = stored_enchantment(stack)?;
    let target = mc_data::enchantment(id)?.target;
    TABS.iter().copied().find(|tab| {
        matches!(tab, CreativeTab::Items(label) if relevant_enchantment_targets(label).contains(&target))
    })
}

/// The one-of-each representative stack a tab shows for `id` (count `1`, no NBT, default
/// metadata `0` — a resource-pack/model concern for which *variant* icon to draw, not this
/// data model's job).
#[must_use]
pub fn representative_stack(id: ItemId) -> ItemStack {
    ItemStack {
        item_id: id.0 as i16,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

fn plain_stack(id: ItemId, damage: i16) -> ItemStack {
    ItemStack {
        item_id: id.0 as i16,
        count: 1,
        damage,
        nbt: None,
    }
}

/// `Item.getSubItems` plus every 1.8.9 override, keyed by the stable numeric item/block id.
fn sub_items(id: ItemId) -> Vec<ItemStack> {
    const SPAWN_EGGS: &[i16] = &[
        50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 90, 91, 92, 93, 94, 95, 96,
        98, 100, 101, 120,
    ];

    let damages: &[i16] = match id.0 {
        1 => &[0, 1, 2, 3, 4, 5, 6],
        3 => &[0, 1, 2],
        5 | 6 | 126 | 175 => &[0, 1, 2, 3, 4, 5],
        12 | 19 | 139 | 161 | 162 | 263 | 322 | 350 => &[0, 1],
        17 | 18 | 98 | 349 => &[0, 1, 2, 3],
        24 | 145 | 155 | 168 | 179 => &[0, 1, 2],
        31 => &[1, 2],
        35 | 95 | 159 | 160 | 171 | 351 => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        38 => &[0, 1, 2, 3, 4, 5, 6, 7, 8],
        // Stone slabs exclude the obsolete WOOD variant at metadata 2.
        44 => &[0, 1, 3, 4, 5, 6, 7],
        97 => &[0, 1, 2, 3, 4, 5],
        397 => &[0, 1, 2, 3, 4],
        383 => SPAWN_EGGS,
        _ => &[0],
    };

    if id.0 == 373 {
        return std::iter::once(plain_stack(id, 0))
            .chain(
                mc_data::misc::potion_creative_damage_values()
                    .into_iter()
                    .map(|damage| plain_stack(id, damage)),
            )
            .collect();
    }
    if id.0 == 425 {
        return (0..16)
            .rev()
            .map(|damage| {
                let mut banner = plain_stack(id, damage);
                let mut block_entity = NbtCompound::new();
                block_entity.insert("Base", Nbt::Int(i32::from(damage)));
                let mut root = NbtCompound::new();
                root.insert("BlockEntityTag", Nbt::Compound(block_entity));
                banner.nbt = Some(root);
                banner
            })
            .collect();
    }
    damages
        .iter()
        .copied()
        .map(|damage| plain_stack(id, damage))
        .collect()
}

fn enchanted_book(id: u8, level: u8) -> ItemStack {
    let book = mc_data::item_by_name("enchanted_book").expect("1.8.9 enchanted book registry row");
    let mut enchantment = NbtCompound::new();
    enchantment.insert("id", Nbt::Short(i16::from(id)));
    enchantment.insert("lvl", Nbt::Short(i16::from(level)));
    let mut root = NbtCompound::new();
    root.insert(
        "StoredEnchantments",
        Nbt::list(vec![Nbt::Compound(enchantment)]),
    );
    let mut stack = plain_stack(book, 0);
    stack.nbt = Some(root);
    stack
}

fn stored_enchantment(stack: &ItemStack) -> Option<(u8, u8)> {
    let first = stack
        .nbt
        .as_ref()?
        .get_list("StoredEnchantments")?
        .first()?
        .as_compound()?;
    let id = u8::try_from(first.get_i16("id")?).ok()?;
    let level = u8::try_from(first.get_i16("lvl")?).ok()?;
    Some((id, level))
}

fn strip_formatting_codes(text: &str) -> String {
    let mut output = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(ch) = chars.next() {
        if ch == '\u{a7}' {
            chars.next();
        } else {
            output.push(ch);
        }
    }
    output
}

/// The ordinary click rule for the read-only tab-source grid (every tab except `Inventory`).
/// `shift` is vanilla's `clickType == 1` (shift held); `clicked_button` is passed through because
/// vanilla increments only for button `0` and decrements for every other value. Returns the new
/// cursor stack. java: the `slotIn.inventory == field_147060_v` branch of
/// `GuiContainerCreative.handleMouseClick`, restricted to its core pick/increment/decrement
/// logic. The caller handles middle-click's earlier `clickType == 3` return separately.
#[must_use]
pub fn click_tab_source(
    cursor: Option<ItemStack>,
    tab_item: &ItemStack,
    clicked_button: i32,
    shift: bool,
) -> Option<ItemStack> {
    click_tab_source_in(GameVersion::V1_8_9, cursor, tab_item, clicked_button, shift)
}

/// [`click_tab_source`] with `version`'s item registry: its stack sizes (a full stack of doors is
/// one door in 1.7.10) and the items it knows.
#[must_use]
pub fn click_tab_source_in(
    version: GameVersion,
    cursor: Option<ItemStack>,
    tab_item: &ItemStack,
    clicked_button: i32,
    shift: bool,
) -> Option<ItemStack> {
    // Network-owned unknown stacks remain opaque to local container simulation. The creative
    // tab grid is normally built exclusively from the 1.8.9 registry, but preserving this gate
    // at the public boundary prevents the zeroed fallback properties from fabricating a cursor
    // stack (or clearing an authoritative unknown cursor) if a caller supplies an unknown id.
    if stack_ext::try_props_in(version, tab_item).is_none()
        || cursor
            .as_ref()
            .is_some_and(|held| stack_ext::try_props_in(version, held).is_none())
    {
        return cursor;
    }

    match cursor {
        Some(mut held) if stack_ext::is_item_equal(&held, tab_item) => {
            if clicked_button == 0 {
                if shift {
                    held.count = stack_ext::max_stack_size_in(version, &held);
                } else if held.count < stack_ext::max_stack_size_in(version, &held) {
                    held.count += 1;
                }
                Some(held)
            } else if held.count <= 1 {
                None
            } else {
                held.count -= 1;
                Some(held)
            }
        }
        Some(_) => None, // a different item was held: vanilla drops it (clears the cursor here)
        None => {
            let mut fresh = tab_item.clone();
            if shift {
                fresh.count = stack_ext::max_stack_size_in(version, &fresh);
            }
            Some(fresh)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tab_label_used_here_actually_appears_in_the_item_table() {
        for tab in TABS {
            if let CreativeTab::Items(label) = tab {
                assert!(
                    !items_for_tab(label).is_empty(),
                    "tab {label:?} has no items — a stale/misspelled label"
                );
            }
        }
    }

    #[test]
    fn the_tab_table_is_indexed_by_vanillas_own_tab_index() {
        assert_eq!(TABS.len(), TAB_COUNT as usize);
        for (position, tab) in TABS.iter().enumerate() {
            assert_eq!(tab.index(), Some(position as u8), "{tab:?}");
        }
        // The two special tabs are the rightmost of each row, which is why they are 5 and 11.
        assert_eq!(CreativeTab::Search.index(), Some(5));
        assert_eq!(CreativeTab::Inventory.index(), Some(11));
        assert_eq!(
            CreativeTab::Items("not-a-real-tab").index(),
            None,
            "an unregistered label must not be given a position"
        );
    }

    #[test]
    fn columns_and_rows_come_straight_from_the_index() {
        // java: `tabIndex % 6` and `tabIndex < 6`.
        assert_eq!(
            (0..12u8).map(tab_column).collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5]
        );
        assert!((0..6u8).all(tab_is_first_row));
        assert!(!(6..12u8).any(tab_is_first_row));
    }

    #[test]
    fn only_the_inventory_tab_drops_the_grid_and_the_title() {
        for tab in TABS {
            let inventory = *tab == CreativeTab::Inventory;
            assert_eq!(tab.has_item_grid(), !inventory, "{tab:?}");
            assert_eq!(tab.draws_title(), !inventory, "{tab:?}");
        }
    }

    #[test]
    fn each_tab_names_a_background_image_the_vanilla_jar_actually_ships() {
        // The jar has exactly these three under creative_inventory/.
        for tab in TABS {
            assert!(
                matches!(
                    tab.background_image(),
                    "tab_items.png" | "tab_item_search.png" | "tab_inventory.png"
                ),
                "{tab:?} -> {}",
                tab.background_image()
            );
        }
        assert_eq!(
            CreativeTab::Search.background_image(),
            "tab_item_search.png"
        );
        assert_eq!(
            CreativeTab::Inventory.background_image(),
            "tab_inventory.png"
        );
    }

    #[test]
    fn every_tab_icon_is_a_real_registered_item() {
        for tab in TABS {
            let (name, _meta) = tab.icon_item();
            assert!(
                mc_data::item_by_name(name).is_some(),
                "{tab:?} icon {name:?} is not a 1.8.9 item"
            );
        }
        assert_eq!(
            CreativeTab::Items("decorations").icon_item(),
            ("double_plant", 5)
        );
    }

    #[test]
    fn search_is_case_insensitive_and_substring_based() {
        let hits = search_items("DIAMOND SWORD");
        let sword = mc_data::item_by_name("diamond_sword").unwrap();
        assert!(hits.contains(&sword));
    }

    /// java: `updateCreativeSearch` keeps a stack when a tooltip line `.contains(s1)`, and
    /// `"".contains("")` is `true` -- so clicking the search tab, whose field starts empty,
    /// shows the whole pool.
    ///
    /// This replaces `empty_search_returns_nothing_rather_than_everything`, which asserted the
    /// divergence as if it were the spec and therefore could never fail while the bug was there.
    #[test]
    fn an_empty_search_lists_the_whole_pool() {
        let all = search_items("");
        assert_eq!(all.len(), searchable_pool().len());
        assert_eq!(all.len(), 678, "full 1.8.9 JVM-oracle count");
        let sword = mc_data::item_by_name("diamond_sword").unwrap();
        assert!(all.contains(&sword));
    }

    /// java: the registry pass requires a non-null creative tab. Enchanted books are the one
    /// apparent exception because `getAll` appends 79 tagged stacks in a separate second pass.
    #[test]
    fn the_search_pool_excludes_items_with_no_creative_tab() {
        let all = search_items("");
        for name in ["mob_spawner", "barrier", "command_block", "dragon_egg"] {
            let Some(id) = mc_data::item_by_name(name) else {
                continue;
            };
            assert!(!all.contains(&id), "{name} has no creative tab in 1.8.9");
        }
        let book = mc_data::item_by_name("enchanted_book").unwrap();
        assert_eq!(all.iter().filter(|id| **id == book).count(), 79);
    }

    #[test]
    fn content_tab_counts_match_the_full_vanilla_registry_oracle() {
        let expected = [158, 126, 37, 12, 59, 32, 33, 46, 73, 48];
        let actual: Vec<_> = [
            "buildingBlocks",
            "decorations",
            "redstone",
            "transportation",
            "misc",
            "food",
            "tools",
            "combat",
            "brewing",
            "materials",
        ]
        .into_iter()
        .map(|label| stacks_for_tab(label).len())
        .collect();
        assert_eq!(actual, expected);
        assert_eq!(actual.iter().sum::<usize>(), 624);
    }

    #[test]
    fn metadata_sub_items_are_distinct_pickable_stacks() {
        let building = stacks_for_tab("buildingBlocks");
        let decorations = stacks_for_tab("decorations");
        let materials = stacks_for_tab("materials");
        let wool = mc_data::item_by_name("wool").unwrap().0 as i16;
        assert_eq!(
            building
                .iter()
                .filter(|stack| stack.item_id == wool)
                .map(|stack| stack.damage)
                .collect::<Vec<_>>(),
            (0..16).collect::<Vec<_>>()
        );

        for (tab, name, expected) in [
            (&building, "prismarine", 0..3),
            (&decorations, "red_flower", 0..9),
            (&building, "red_sandstone", 0..3),
            (&materials, "coal", 0..2),
        ] {
            let id = mc_data::item_by_name(name).unwrap().0 as i16;
            let variants: Vec<_> = tab
                .iter()
                .filter(|stack| stack.item_id == id)
                .map(|stack| stack.damage)
                .collect();
            assert_eq!(
                variants,
                expected.collect::<Vec<_>>(),
                "{name} must expose every metadata variant, not one representative"
            );
        }

        let brewing = stacks_for_tab("brewing");
        let potion = mc_data::item_by_name("potion").unwrap().0 as i16;
        assert_eq!(
            brewing
                .iter()
                .filter(|stack| stack.item_id == potion)
                .count(),
            63
        );

        let misc = stacks_for_tab("misc");
        let egg = mc_data::item_by_name("spawn_egg").unwrap().0 as i16;
        assert_eq!(misc.iter().filter(|stack| stack.item_id == egg).count(), 27);
    }

    #[test]
    fn class_specific_items_keep_vanilla_creative_sub_items_and_banner_base_nbt() {
        let decorations = stacks_for_tab("decorations");
        for (tab, name, expected) in [
            (stacks_for_tab("food"), "golden_apple", 0..2),
            (decorations.clone(), "anvil", 0..3),
        ] {
            let id = mc_data::item_by_name(name).unwrap().0 as i16;
            let variants: Vec<_> = tab
                .iter()
                .filter(|stack| stack.item_id == id)
                .map(|stack| stack.damage)
                .collect();
            assert_eq!(
                variants,
                expected.collect::<Vec<_>>(),
                "{name} must retain every class-provided creative entry"
            );
        }

        let banner = mc_data::item_by_name("banner").unwrap().0 as i16;
        let banners: Vec<_> = decorations
            .iter()
            .filter(|stack| stack.item_id == banner)
            .collect();
        assert_eq!(banners.len(), 16, "ItemBanner contributes every dye colour");
        assert_eq!(
            banners.iter().map(|stack| stack.damage).collect::<Vec<_>>(),
            (0i16..16).rev().collect::<Vec<_>>(),
            "EnumDyeColor.values() supplies banner entries in vanilla order"
        );
        assert!(banners.iter().all(|stack| {
            stack
                .nbt
                .as_ref()
                .and_then(|root| root.get_compound("BlockEntityTag"))
                .and_then(|tag| tag.get_i32("Base"))
                == Some(i32::from(stack.damage))
        }));
    }

    #[test]
    fn potion_catalog_preserves_java_order_and_reaches_name_and_tint_helpers() {
        let potion_id = mc_data::item_by_name("potion").unwrap().0 as i16;
        let brewing = stacks_for_tab("brewing");
        let potions: Vec<_> = brewing
            .iter()
            .filter(|stack| stack.item_id == potion_id)
            .collect();

        let mut expected = vec![0];
        expected.extend(mc_data::misc::potion_creative_damage_values());
        assert_eq!(
            potions.iter().map(|stack| stack.damage).collect::<Vec<_>>(),
            expected
        );

        for stack in potions.iter().skip(1) {
            let damage = i32::from(stack.damage);
            assert!(!mc_data::potion_effects_from_damage(damage).is_empty());
            assert_eq!(
                tooltip::potion_liquid_color(stack),
                Some(mc_data::potion_liquid_color_from_damage(damage)),
                "metadata {damage}"
            );
        }

        let first_name = |damage| {
            let stack = potions
                .iter()
                .find(|stack| stack.damage == damage)
                .expect("potion metadata from the creative catalog");
            tooltip::build(stack, false)
                .into_iter()
                .next()
                .expect("potion tooltip title")
        };
        assert_eq!(first_name(8193), "Potion of Regeneration§r");
        assert_eq!(first_name(16420), "Splash Potion of Poison§r");
    }

    #[test]
    fn enchanted_books_use_stored_enchantments_and_vanillas_tab_counts() {
        let book = mc_data::item_by_name("enchanted_book").unwrap().0 as i16;
        let tools = stacks_for_tab("tools");
        let combat = stacks_for_tab("combat");
        assert_eq!(
            tools.iter().filter(|stack| stack.item_id == book).count(),
            6
        );
        assert_eq!(
            combat.iter().filter(|stack| stack.item_id == book).count(),
            19
        );
        assert!(tools
            .iter()
            .filter(|stack| stack.item_id == book)
            .all(|stack| stack
                .nbt
                .as_ref()
                .and_then(|tag| tag.get_list("StoredEnchantments"))
                .is_some_and(|list| list.len() == 1)));
    }

    #[test]
    fn search_matches_enchantment_tooltip_lines() {
        let hits = search_stacks("Efficiency", false, None);
        let book = mc_data::item_by_name("enchanted_book").unwrap().0 as i16;
        assert_eq!(hits.iter().filter(|stack| stack.item_id == book).count(), 5);
    }

    #[test]
    fn enchanted_book_search_results_keep_vanilla_source_tabs_without_ordinary_book_duplication() {
        let ordinary_book = representative_stack(mc_data::item_by_name("book").unwrap());
        assert_eq!(
            source_tab(&ordinary_book),
            Some(CreativeTab::Items("misc")),
            "ordinary books retain their registry creative tab"
        );

        let efficiency = enchanted_book(32, 1);
        assert_eq!(
            source_tab(&efficiency),
            Some(CreativeTab::Items("tools")),
            "Digger enchantments attribute to Tools"
        );
        let sharpness = enchanted_book(16, 1);
        assert_eq!(
            source_tab(&sharpness),
            Some(CreativeTab::Items("combat")),
            "Weapon enchantments attribute to Combat"
        );

        let search = search_stacks("Efficiency", false, None);
        assert_eq!(
            search
                .iter()
                .filter(|stack| stack.item_id == efficiency.item_id)
                .count(),
            5,
            "search shows each Efficiency level exactly once"
        );
        assert!(search
            .iter()
            .filter(|stack| stack.item_id == efficiency.item_id)
            .all(|stack| source_tab(stack) == Some(CreativeTab::Items("tools"))));

        let untagged = representative_stack(mc_data::item_by_name("enchanted_book").unwrap());
        assert_eq!(
            source_tab(&untagged),
            None,
            "an untagged enchanted book has no inferred source tab"
        );
    }

    #[test]
    fn search_matches_substrings_not_just_whole_names() {
        // "sword" should surface every sword, not just an exact "Sword" name.
        let hits = search_items("sword");
        assert!(
            hits.len() >= 5,
            "expected every tool-tier sword, got {}",
            hits.len()
        );
    }

    #[test]
    fn picking_from_an_empty_cursor_gives_a_single_item() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let result = click_tab_source(None, &stone, 0, false).unwrap();
        assert_eq!(result.count, 1);
        assert_eq!(result.item_id, stone.item_id);
    }

    #[test]
    fn shift_picking_gives_a_full_stack() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let result = click_tab_source(None, &stone, 0, true).unwrap();
        assert_eq!(result.count, 64);
    }

    #[test]
    fn left_clicking_the_same_item_again_increments_by_one() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held = ItemStack {
            count: 5,
            ..stone.clone()
        };
        let result = click_tab_source(Some(held), &stone, 0, false).unwrap();
        assert_eq!(result.count, 6);
    }

    #[test]
    fn left_click_does_not_exceed_max_stack_size() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held = ItemStack {
            count: 64,
            ..stone.clone()
        };
        let result = click_tab_source(Some(held), &stone, 0, false).unwrap();
        assert_eq!(
            result.count, 64,
            "already at cap, left click must not overflow it"
        );
    }

    #[test]
    fn right_clicking_decrements_by_one() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held = ItemStack {
            count: 5,
            ..stone.clone()
        };
        let result = click_tab_source(Some(held), &stone, 1, false).unwrap();
        assert_eq!(result.count, 4);
    }

    #[test]
    fn right_clicking_a_single_item_clears_the_cursor() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held = ItemStack {
            count: 1,
            ..stone.clone()
        };
        assert_eq!(click_tab_source(Some(held), &stone, 1, false), None);
    }

    #[test]
    fn every_non_left_button_takes_the_decrement_side() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held = ItemStack {
            count: 5,
            ..stone.clone()
        };
        let result = click_tab_source(Some(held), &stone, 7, false).unwrap();
        assert_eq!(
            result.count, 4,
            "java tests clickedButton == 0, not clickedButton == 1"
        );
    }

    #[test]
    fn clicking_a_different_item_while_holding_something_clears_the_cursor() {
        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let dirt = representative_stack(mc_data::item_by_name("dirt").unwrap());
        let held = ItemStack { count: 3, ..dirt };
        assert_eq!(click_tab_source(Some(held), &stone, 0, false), None);
    }

    #[test]
    fn unknown_tab_items_and_cursors_are_opaque_to_local_simulation() {
        let unknown = ItemStack {
            item_id: i16::MAX,
            count: 5,
            damage: 0,
            nbt: None,
        };
        assert_eq!(
            click_tab_source(None, &unknown, 0, true),
            None,
            "an unknown virtual source must not fabricate a zero-count cursor stack"
        );

        let stone = representative_stack(mc_data::item_by_name("stone").unwrap());
        let held_stone = ItemStack {
            count: 3,
            ..stone.clone()
        };
        assert_eq!(
            click_tab_source(Some(held_stone.clone()), &unknown, 0, false),
            Some(held_stone),
            "an unknown source leaves the authoritative cursor unchanged"
        );
        assert_eq!(
            click_tab_source(Some(unknown.clone()), &stone, 0, false),
            Some(unknown),
            "an authoritative unknown cursor is likewise inert"
        );
    }
}
