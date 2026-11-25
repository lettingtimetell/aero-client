//! Minecraft 1.7.10's container prediction: the parts of `Container.slotClick` and its helpers
//! a 1.7.10 client computes differently from 1.8.9. A child of `model`, for [`Container`]'s
//! private state; `model.rs` reaches each function from one line that runs only for a container
//! built with [`Container::with_game_version`]`(GameVersion::V1_7_10)`.
//!
//! What differs (java17 vs java18, `inventory/*.java`):
//!
//! - **Drags**: `Container.isValidDragMode` takes modes 0 and 1 only, creative or not; the
//!   pick-block button's drag reuses the last left/right mode (`GuiContainer.mouseClicked`, see
//!   `interaction.rs`), so it splits the stack like a left or right drag.
//! - **Enchanting table**: one slot and no lapis ([`ContainerKind::EnchantingV5`]).
//! - **Anvil**: `ContainerRepair.updateRepairOutput` charges per material, per point of
//!   durability, per enchantment level gained, per enchantment on the result and per name, and
//!   writes `RepairCost` its own way ([`Container::update_repair_output_v5`]).
//! - **Crafting**: `SlotCrafting.onPickupFromSlot` puts a container item (the empty bucket of a
//!   water, lava or milk bucket) in the player's inventory first, then in the emptied grid slot,
//!   else drops it ([`Container::pickup_craft_result_v5`]).
//!
//! The predicted results matter on the wire: they are the `clicked_item` of the next `C0E` and
//! decide whether the server accepts it.

use mc_nbt::Nbt;
use mc_protocol::play::s2c;
use mc_version::GameVersion;

use super::{
    add_stored_enchantment, all_registered, clear_custom_name, enchantment_can_apply,
    enchantment_entry, enchantment_map, input_has_custom_name, java_is_blank, known_or_empty,
    registered_stack, set_custom_name, stack_repair_cost, Container, ContainerKind, MerchantRecipe,
    ENCHANTED_BOOK_ID,
};
use crate::screens::container::stack_ext::{self, ItemStack};

// The crafting grid's result: 1.7.10's recipe list and special recipes.
#[path = "crafting_v5.rs"]
mod crafting;

impl Container {
    /// This container with `version`'s click, anvil and crafting rules. Containers start with
    /// 1.8.9's; a 1.7.10 session builds its windows (and its window 0) with 1.7.10's.
    #[must_use]
    pub fn with_game_version(mut self, version: GameVersion) -> Self {
        self.version = version;
        self
    }

    /// Switch this container to `version`'s rules (the session's window 0, which outlives a
    /// screen).
    pub fn set_game_version(&mut self, version: GameVersion) {
        self.version = version;
    }

    /// The game version whose rules this container predicts with.
    #[must_use]
    pub const fn game_version(&self) -> GameVersion {
        self.version
    }

    /// Whether a `S2FPacketSetSlot` writes the item slot of a 1.7.10 enchanting table, the one
    /// slot whose change re-runs `ContainerEnchantment.onCraftMatrixChanged` (java17: the table
    /// inventory is `new InventoryBasic("Enchant", true, 1)`).
    pub(crate) fn table_slot_v5(&self, pkt: &s2c::SetSlot) -> bool {
        self.kind == ContainerKind::EnchantingV5
            && pkt.window_id >= 0
            && pkt.window_id as u8 == self.window_id
            && pkt.slot == 0
    }

    /// The 1.7.10 enchanting table's half of `transferStackInSlot`: `false` where java17 returns
    /// `null`. Slot 0 goes back to the player's slots (`1..37`, last first); a player stack moves
    /// into the empty item slot whole when it is one tagged item, else as one plain item of the
    /// same kind and metadata. java17: `ContainerEnchantment.transferStackInSlot`
    /// (`inventory/ContainerEnchantment.java:233-280`).
    pub(super) fn transfer_enchanting_v5(&mut self, index: usize, working: &mut ItemStack) -> bool {
        if index == 0 {
            let end = self.slots.len();
            return self.merge_item_stack(working, 1, end, true);
        }
        // The item slot's `isItemValid` is always true.
        if self.items[0].is_some() {
            return false;
        }
        if working.nbt.is_some() && working.count == 1 {
            self.items[0] = Some(working.clone());
            working.count = 0;
        } else if working.count >= 1 {
            self.items[0] = Some(ItemStack {
                item_id: working.item_id,
                count: 1,
                damage: working.damage,
                nbt: None,
            });
            working.count -= 1;
        }
        // `putStack(copy)` / `putStack(new ItemStack(..))`: a new object in the item slot.
        self.renew_item_identity(0);
        true
    }

    /// `TileEntityFurnace.isItemFuel` over this container's version's items, for the furnace's
    /// shift-click routing. java17 and java18 run the same code (`getItemBurnTime`: wooden
    /// blocks, coal, sticks, ...); the items differ: 1.7.10 has a double wooden slab item, a
    /// wooden block, so it burns. Every item both versions have is fuel in both or in neither
    /// (`tests/v5_item_rules_oracle.rs`), and a 1.8.9-only item never reaches this (the 1.7.10
    /// registry guard stops the click first).
    pub(super) fn is_furnace_fuel(&self, stack: &ItemStack) -> bool {
        const DOUBLE_WOODEN_SLAB: i16 = 125;
        match self.version {
            GameVersion::V1_8_9 => super::super::slot::is_furnace_fuel(stack),
            GameVersion::V1_7_10 => {
                stack.item_id == DOUBLE_WOODEN_SLAB || super::super::slot::is_furnace_fuel(stack)
            }
        }
    }

    /// java17: `ContainerRepair.updateRepairOutput` (`inventory/ContainerRepair.java:140-426`),
    /// line for line. `base` is java's `var19` (the inputs' `RepairCost`s plus the per-result
    /// enchantment charge), `extra` its `var2` (this repair's own levels), `rename_cost` its
    /// `var4`; the output's cost is `base + extra`.
    ///
    /// Two places where vanilla 1.7.10 throws a `NullPointerException` (an enchantment id it does
    /// not register, read from either input's NBT) do not crash here: such an id is skipped where
    /// its `Enchantment` would be read, and kept in the output's list as `setEnchantments`
    /// writes every id.
    pub(super) fn update_repair_output_v5(&mut self) {
        let Some(input) = self.items.first().and_then(Option::as_ref).cloned() else {
            self.items[2] = None;
            self.maximum_cost = Some(0);
            return;
        };
        let second = self.items.get(1).and_then(Option::as_ref).cloned();
        // As in 1.8.9 mode: a stack the item table does not know is shown and echoed, never
        // simulated.
        if !registered_stack(self.version, &input) || !known_or_empty(self.version, second.as_ref())
        {
            self.items[2] = None;
            self.anvil_material_cost = 0;
            self.maximum_cost = Some(0);
            return;
        }

        let mut output = input.clone();
        let mut enchantments = enchantment_map(&output);
        let mut extra = 0i32;
        let mut base =
            stack_repair_cost(&input).wrapping_add(second.as_ref().map_or(0, stack_repair_cost));
        let mut rename_cost = 0i32;
        self.anvil_material_cost = 0;
        let mut book = false;

        if let Some(material) = second.as_ref() {
            // `Items.enchanted_book.func_92110_g(var6).tagCount() > 0`.
            book = material.item_id == ENCHANTED_BOOK_ID
                && material
                    .nbt
                    .as_ref()
                    .and_then(|root| root.get_list("StoredEnchantments"))
                    .is_some_and(|list| !list.is_empty());
            let output_props = stack_ext::props_in(self.version, &output);
            let max_damage = i32::from(output_props.max_damage);

            if stack_ext::is_stack_damageable_in(self.version, &output)
                && output_props.is_repairable_with(mc_data::ItemId(material.item_id as u16))
            {
                let mut repair = i32::from(output.damage).min(max_damage / 4);
                if repair <= 0 {
                    self.items[2] = None;
                    self.maximum_cost = Some(0);
                    return;
                }
                let mut used = 0i32;
                while repair > 0 && used < material.count {
                    let repaired = i32::from(output.damage) - repair;
                    set_metadata(&mut output, repaired);
                    extra = extra
                        .wrapping_add((repair / 100).max(1))
                        .wrapping_add(enchantments.len() as i32);
                    repair = i32::from(output.damage).min(max_damage / 4);
                    used += 1;
                }
                self.anvil_material_cost = used;
            } else {
                if !book
                    && (output.item_id != material.item_id
                        || !stack_ext::is_stack_damageable_in(self.version, &output))
                {
                    self.items[2] = None;
                    self.maximum_cost = Some(0);
                    return;
                }

                if stack_ext::is_stack_damageable_in(self.version, &output) && !book {
                    let first_left = max_damage - i32::from(input.damage);
                    let second_left =
                        i32::from(stack_ext::props_in(self.version, material).max_damage)
                            - i32::from(material.damage);
                    let second_share = second_left + max_damage * 12 / 100;
                    let durability = first_left + second_share;
                    let damage = (max_damage - durability).max(0);
                    if damage < i32::from(output.damage) {
                        set_metadata(&mut output, damage);
                        extra = extra.wrapping_add((second_share / 100).max(1));
                    }
                }

                for (id, incoming) in enchantment_map(material) {
                    let Some(info) = enchantment_v5(id) else {
                        continue;
                    };
                    let existing = level_of(&enchantments, id);
                    let incoming = i32::from(incoming);
                    let mut level = if existing == incoming {
                        incoming + 1
                    } else {
                        incoming.max(existing)
                    };
                    let gained = level - existing;
                    let mut applies = enchantment_can_apply(info, &input)
                        || self.anvil_creative
                        || input.item_id == ENCHANTED_BOOK_ID;
                    for &(other, _) in &enchantments {
                        if other != id && !can_apply_together_v5(id, other) {
                            applies = false;
                            extra = extra.wrapping_add(gained);
                        }
                    }
                    if applies {
                        level = level.min(i32::from(info.max_level));
                        put_level(&mut enchantments, id, level as i16);
                        let mut cost = weight_cost(info.weight);
                        if book {
                            cost = (cost / 2).max(1);
                        }
                        extra = extra.wrapping_add(cost.wrapping_mul(gained));
                    }
                }
            }
        }

        let name = self.anvil_name.clone();
        if java_is_blank(&name) {
            if input_has_custom_name(&input) {
                rename_cost = rename_charge(&input);
                extra = extra.wrapping_add(rename_cost);
                clear_custom_name(&mut output);
            }
        } else if name != self.anvil_input_display_name_of(&input) {
            rename_cost = rename_charge(&input);
            extra = extra.wrapping_add(rename_cost);
            if input_has_custom_name(&input) {
                base = base.wrapping_add(rename_cost / 2);
            }
            set_custom_name(&mut output, &name);
        }

        // Every enchantment the result keeps: its position in the list (from 1) plus its level
        // times its weight cost.
        let mut position = 0i32;
        for &(id, level) in &enchantments {
            let Some(info) = enchantment_v5(id) else {
                continue;
            };
            position += 1;
            let mut cost = weight_cost(info.weight);
            if book {
                cost = (cost / 2).max(1);
            }
            base = base.wrapping_add(position.wrapping_add(i32::from(level) * cost));
        }
        if book {
            base = (base / 2).max(1);
        }

        let mut maximum_cost = base.wrapping_add(extra);
        let mut keep = extra > 0;
        if rename_cost == extra && rename_cost > 0 && maximum_cost >= 40 {
            maximum_cost = 39;
        }
        if maximum_cost >= 40 && !self.anvil_creative {
            keep = false;
        }
        self.maximum_cost = Some(maximum_cost);
        self.items[2] = keep.then(|| {
            let mut repair_cost = stack_repair_cost(&output);
            if let Some(material) = second.as_ref() {
                repair_cost = repair_cost.max(stack_repair_cost(material));
            }
            if input_has_custom_name(&output) {
                repair_cost = repair_cost.wrapping_sub(9);
            }
            repair_cost = repair_cost.max(0).wrapping_add(2);
            Self::ensure_root(&mut output).insert("RepairCost", Nbt::Int(repair_cost));
            set_enchantments_v5(&mut output, &enchantments);
            output
        });
    }

    /// java17: `SlotCrafting.onPickupFromSlot` (`inventory/SlotCrafting.java:104-134`). Each
    /// occupied grid slot loses one item; when that item has a container item, a new one goes to
    /// the player's inventory, else into the grid slot if it is now empty, else on the ground
    /// (nothing to show here). `addItemStackToInventory` also "succeeds" in creative when
    /// nothing fits; the creative flag is the one the screen supplies before each click
    /// ([`Container::set_creative_mode`]).
    pub(super) fn pickup_craft_result_v5(&mut self) {
        let Some((start, end, _, _)) = self.crafting_grid_shape() else {
            return;
        };
        self.predicted_craft = None;
        for grid in start..end {
            let Some(stack) = self.items[grid].as_mut() else {
                continue;
            };
            let item = stack.item_id;
            // `InventoryCrafting.decrStackSize(i, 1)`.
            if stack.count <= 1 {
                self.items[grid] = None;
            } else {
                stack.count -= 1;
            }
            let Some(container) = u16::try_from(item)
                .ok()
                .and_then(|id| Self::crafting_container_item(mc_data::ItemId(id)))
            else {
                continue;
            };
            let empty = ItemStack {
                item_id: container.0 as i16,
                count: 1,
                damage: 0,
                nbt: None,
            };
            let left = self.add_stack_to_player_inventory(empty.clone());
            let added = left.count < empty.count || self.anvil_creative;
            if !added && self.items[grid].is_none() {
                self.items[grid] = Some(empty);
                self.renew_item_identity(grid);
            }
        }
        self.recompute_craft_result();
    }

    /// java17: `InventoryPlayer.storePartialItemStack` (`entity/player/InventoryPlayer.java:222-245`)
    /// for an item that stacks to one, in a 1.7.10 container: the whole stack goes into the first
    /// empty slot (`ItemStack.copyItemStack`) and nothing is left, or nothing moves when there is
    /// no empty slot. 1.8.9 has no such branch and adds one item per slot. `None` where the 1.8.9
    /// algorithm applies.
    pub(super) fn store_unstackable_v5(
        &mut self,
        stack: &ItemStack,
        start: usize,
        end: usize,
    ) -> Option<i32> {
        if !self.version.is_1_7_10() || stack_ext::max_stack_size_in(self.version, stack) != 1 {
            return None;
        }
        let Some(slot) = self.find_empty_slot(start, end) else {
            return Some(stack.count);
        };
        self.items[slot] = Some(stack.clone());
        self.renew_item_identity(slot);
        Some(0)
    }
}

/// java17: `MerchantRecipeList.canRecipeBeUsed` (`village/MerchantRecipeList.java:21-42`, lines
/// 26 and 34): an offered stack meets a wanted one by item and count only, with no damage and no
/// tag compared (java18's `func_181078_a` also wants the damage and a tag subset). Every 1.7.10
/// buy offer has damage 0 (`EntityVillager.func_146088_a:566-569`), so a villager takes any wool
/// colour for its wool offer.
pub(super) fn merchant_stack_matches_v5(input: &ItemStack, required: &ItemStack) -> bool {
    input.item_id == required.item_id && input.count >= required.count
}

/// A 1.7.10 merchant window's trade state: the client's `MerchantRecipeList`, the selected offer
/// (`InventoryMerchant.currentRecipeIndex`) and the offer the last reset found
/// (`InventoryMerchant.currentRecipe`).
#[derive(Debug, Clone, PartialEq, Default)]
pub(super) struct MerchantV5 {
    /// The offers `MC|TrList` delivered; `None` before any (`getRecipes` is null then).
    offers: Option<Vec<MerchantRecipe>>,
    /// The selected offer's index.
    selected: usize,
    /// The offer whose result the output slot shows, as it was found: `currentRecipe` is that
    /// list's object, which a new list leaves in place.
    current: Option<MerchantRecipe>,
}

impl MerchantV5 {
    /// java17 `MerchantRecipeList.canRecipeBeUsed` (`:21-42`): with an offer past the first
    /// selected, that offer alone; with the first selected (or an index past the list), the
    /// first offer of the whole list that `first` and `second` pay for, disabled or not.
    fn can_recipe_be_used(&self, first: &ItemStack, second: Option<&ItemStack>) -> Option<usize> {
        let offers = self.offers.as_deref()?;
        let pays = |offer: &MerchantRecipe| {
            merchant_stack_matches_v5(first, &offer.buy_a)
                && match (&offer.buy_b, second) {
                    (None, None) => true,
                    (Some(wanted), Some(second)) => merchant_stack_matches_v5(second, wanted),
                    _ => false,
                }
        };
        if self.selected > 0 && self.selected < offers.len() {
            return pays(&offers[self.selected]).then_some(self.selected);
        }
        offers.iter().position(pays)
    }
}

/// `MerchantRecipe.isRecipeDisabled`: used up (the decoded `MC|TrList` marks a disabled offer).
fn offer_disabled(offer: &MerchantRecipe) -> bool {
    offer.disabled || offer.uses >= offer.max_uses
}

impl Container {
    /// The whole offer list and the selected offer of a 1.7.10 merchant window, then the reset a
    /// page change runs (`InventoryMerchant.setCurrentRecipeIndex:207-211`). 1.7.10 does not
    /// test the selected offer alone: with the first one selected it takes the first offer of
    /// the list the inputs pay for (`canRecipeBeUsed`), so a later page's trade shows on the
    /// first page.
    pub fn set_merchant_offers_v5(&mut self, offers: &[MerchantRecipe], selected: usize) {
        if self.kind != ContainerKind::Merchant {
            return;
        }
        self.merchant_v5 = MerchantV5 {
            offers: Some(offers.to_vec()),
            selected,
            current: None,
        };
        self.merchant_recipe = offers.get(selected).cloned();
        self.recompute_merchant_result_v5();
    }

    /// A new offer list for a 1.7.10 merchant window, taken as java17 takes one: `MC|TrList`
    /// only replaces the list (`NetHandlerPlayClient.handleCustomPayload:1407-1421` ->
    /// `NpcMerchant.setRecipes`, `entity/NpcMerchant.java:34-37`), so the output and the offer
    /// it came from stay until the next slot or page change resets them. `selected` is the page
    /// the screen keeps.
    pub fn receive_merchant_offers_v5(&mut self, offers: &[MerchantRecipe], selected: usize) {
        if self.kind != ContainerKind::Merchant {
            return;
        }
        self.merchant_v5.offers = Some(offers.to_vec());
        self.merchant_v5.selected = selected;
        self.merchant_recipe = offers.get(selected).cloned();
    }

    /// java17: `InventoryMerchant.resetRecipeAndSlots` (`inventory/InventoryMerchant.java:149-
    /// 200`): the inputs in slot order (the second alone when the first is empty), then swapped;
    /// an offer found but disabled counts as none. Before any offer list the output stays.
    pub(super) fn recompute_merchant_result_v5(&mut self) {
        // As in 1.8.9 mode: a stack the item table does not know is shown, never simulated.
        if !all_registered(self.version, &self.items[..2]) {
            return;
        }
        let (first, second) = match (self.items[0].clone(), self.items[1].clone()) {
            (None, second) => (second, None),
            inputs => inputs,
        };
        self.merchant_v5.current = None;
        let Some(first) = first else {
            self.items[2] = None;
            return;
        };
        let Some(offers) = self.merchant_v5.offers.as_deref() else {
            return;
        };
        let usable = |a: &ItemStack, b: Option<&ItemStack>| {
            self.merchant_v5
                .can_recipe_be_used(a, b)
                .filter(|&index| !offer_disabled(&offers[index]))
        };
        let found = usable(&first, second.as_ref()).or_else(|| {
            second
                .as_ref()
                .and_then(|second| usable(second, Some(&first)))
        });
        let current = found.map(|index| offers[index].clone());
        self.items[2] = current.as_ref().map(|offer| offer.sell.clone());
        self.merchant_v5.current = current;
    }

    /// java17: `SlotMerchantResult.onPickupFromSlot` (`:51-79`): the offer the output came from
    /// (`getCurrentRecipe`) is paid in either orientation (`doTrade`, the same as 1.8.9's) and
    /// the emptied inputs re-run the reset. The trade uses nothing up: the client's merchant
    /// ignores `useRecipe` (`entity/NpcMerchant.java:39`, where 1.8.9 calls
    /// `incrementToolUses`), so an offer is used up only when the server's next list says so.
    pub(super) fn pickup_merchant_result_v5(&mut self) {
        let Some(offer) = self.merchant_v5.current.clone() else {
            return;
        };
        if !(self.merchant_do_trade(&offer, 0, 1) || self.merchant_do_trade(&offer, 1, 0)) {
            return;
        }
        for input in 0..2 {
            if self.items[input]
                .as_ref()
                .is_some_and(|stack| stack.count <= 0)
            {
                self.items[input] = None;
            }
        }
        self.recompute_merchant_result_v5();
    }
}

/// An enchantment 1.7.10 registers (`Enchantment.enchantmentsList[id] != null`).
fn enchantment_v5(id: i16) -> Option<&'static mc_data::EnchantmentInfo> {
    u8::try_from(id).ok().and_then(mc_data::v5::enchantment_v5)
}

/// `ItemStack.setMetadata`: a negative value is stored as 0.
fn set_metadata(stack: &mut ItemStack, damage: i32) {
    stack.damage = damage.max(0) as i16;
}

/// The anvil's charge for a name: 7 for a damageable item, else 5 per item in the stack.
fn rename_charge(input: &ItemStack) -> i32 {
    if stack_ext::is_stack_damageable_in(GameVersion::V1_7_10, input) {
        7
    } else {
        input.count.wrapping_mul(5)
    }
}

/// `updateRepairOutput`'s weight switch: rarer enchantments cost more per level.
fn weight_cost(weight: u8) -> i32 {
    match weight {
        1 => 8,
        2 => 4,
        5 => 2,
        10 => 1,
        _ => 0,
    }
}

fn level_of(map: &[(i16, i16)], id: i16) -> i32 {
    map.iter()
        .find_map(|&(key, level)| (key == id).then_some(i32::from(level)))
        .unwrap_or(0)
}

/// `LinkedHashMap.put`: a present key keeps its place.
fn put_level(map: &mut Vec<(i16, i16)>, id: i16, level: i16) {
    match map.iter_mut().find(|(key, _)| *key == id) {
        Some(entry) => entry.1 = level,
        None => map.push((id, level)),
    }
}

/// `Enchantment.canApplyTogether` as 1.7.10 registers it: damage enchantments exclude each
/// other; protections exclude each other unless one is feather falling; looting, fortune and
/// luck of the sea exclude silk touch, which excludes fortune; and nothing combines with itself.
/// An id 1.7.10 does not register is compatible (vanilla would throw on it in two of these).
/// java17: `enchantment/Enchantment.java:96`, `EnchantmentDamage.java:51`,
/// `EnchantmentProtection.java:60`, `EnchantmentLootBonus.java:40`,
/// `EnchantmentUntouching.java:31`.
fn can_apply_together_v5(new: i16, old: i16) -> bool {
    if enchantment_v5(old).is_none() {
        return true;
    }
    if new == old {
        return false;
    }
    match (new, old) {
        (16..=18, 16..=18) => false,
        (0..=4, 0..=4) => new == 2 || old == 2,
        (21 | 35 | 61, 33) => false,
        (33, 35) => false,
        _ => true,
    }
}

/// java17: `EnchantmentHelper.setEnchantments`: every id is written, in map order; a book
/// stores them through `ItemEnchantedBook.addEnchantment` (registered ids only: vanilla throws
/// on the others) and never gets an `ench` list; an empty map removes `ench`.
fn set_enchantments_v5(stack: &mut ItemStack, map: &[(i16, i16)]) {
    let book = stack.item_id == ENCHANTED_BOOK_ID;
    if book {
        for &(id, level) in map {
            if enchantment_v5(id).is_some() {
                add_stored_enchantment(stack, id, level);
            }
        }
    }
    if !map.is_empty() {
        if !book {
            let list = map
                .iter()
                .map(|&(id, level)| enchantment_entry(id, level))
                .collect();
            Container::ensure_root(stack).insert("ench", Nbt::list(list));
        }
    } else if let Some(root) = stack.nbt.as_mut() {
        root.remove("ench");
    }
}

#[cfg(test)]
#[path = "model_v5_tests.rs"]
mod tests;
