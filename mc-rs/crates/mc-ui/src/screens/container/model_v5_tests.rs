//! 1.7.10 container prediction, each test with its 1.8.9 negative control.

use super::super::stack_custom_name;
use super::*;
use crate::screens::container::layouts;
use crate::screens::container::v5::{enchanting_table_container_v5, furnace_container_v5};

const STONE: i16 = 1;
const DIRT: i16 = 3;

fn stack(item_id: i16, count: i32) -> ItemStack {
    ItemStack {
        item_id,
        count,
        damage: 0,
        nbt: None,
    }
}

fn v5(container: Container) -> Container {
    container.with_game_version(GameVersion::V1_7_10)
}

/// The three-phase drag `GuiContainer.mouseReleased` sends for `buttons = (start, add, end)`
/// over `slots`, applied in order.
fn drag(c: &mut Container, buttons: (u8, u8, u8), slots: &[i32], creative: bool) {
    c.click(-999, buttons.0, 5, creative);
    for &slot in slots {
        c.click(slot, buttons.1, 5, creative);
    }
    c.click(-999, buttons.2, 5, creative);
}

#[test]
fn a_drag_with_the_left_mode_splits_in_survival_in_both_versions() {
    for version in [GameVersion::V1_8_9, GameVersion::V1_7_10] {
        let mut c = layouts::player_container(0).with_game_version(version);
        c.set_cursor(Some(stack(STONE, 4)));
        drag(&mut c, (0, 1, 2), &[9, 10], false);
        assert_eq!(c.slot_stack(9).map(|s| s.count), Some(2), "{version}");
        assert_eq!(c.slot_stack(10).map(|s| s.count), Some(2), "{version}");
        assert!(c.cursor().is_none(), "{version}");
    }
}

#[test]
fn the_creative_fill_mode_is_refused_by_1_7_10_even_in_creative() {
    // 1.8.9 creative: mode 2 fills each slot to a full stack (negative control).
    let mut c = layouts::player_container(0);
    c.set_cursor(Some(stack(STONE, 4)));
    drag(&mut c, (8, 9, 10), &[9, 10], true);
    assert_eq!(c.slot_stack(9).map(|s| s.count), Some(64));
    // java17 `isValidDragMode`: modes 0 and 1 only.
    let mut c = v5(layouts::player_container(0));
    c.set_cursor(Some(stack(STONE, 4)));
    drag(&mut c, (8, 9, 10), &[9, 10], true);
    assert!(c.slot_stack(9).is_none() && c.slot_stack(10).is_none());
    assert_eq!(c.cursor().map(|s| s.count), Some(4));
}

#[test]
fn the_1_7_10_table_has_one_slot_and_its_player_slots_one_lower() {
    let mut c = enchanting_table_container_v5(3);
    let sword = ItemStack {
        item_id: 276,
        count: 1,
        damage: 0,
        nbt: None,
    };
    // Window slot 1 is the first main-inventory slot (2 in 1.8.9).
    c.set_slot_contents(1, Some(sword.clone()));
    let packet = c.click(1, 0, 1, false);
    assert_eq!(packet.slot_id, 1);
    assert_eq!(packet.clicked_item.as_ref(), Some(&sword));
    assert_eq!(c.slot_stack(0), Some(&sword));
    assert!(c.slot_stack(1).is_none());
    // Shift-clicking it back fills the player's slots from the hotbar end (`1..37`, reversed).
    c.click(0, 0, 1, false);
    assert!(c.slot_stack(0).is_none());
    assert_eq!(c.slot_stack(36), Some(&sword));
}

#[test]
fn a_shift_clicked_stack_puts_one_plain_item_in_the_table() {
    let mut c = enchanting_table_container_v5(3);
    c.set_slot_contents(5, Some(stack(340, 3)));
    c.click(5, 0, 1, false);
    assert_eq!(c.slot_stack(0), Some(&stack(340, 1)));
    assert_eq!(c.slot_stack(5).map(|s| s.count), Some(2));
    // A second shift-click finds the slot taken.
    let packet = c.click(5, 0, 1, false);
    assert!(packet.clicked_item.is_none());
    assert_eq!(c.slot_stack(5).map(|s| s.count), Some(2));
}

#[test]
fn the_1_7_10_fuel_slot_takes_any_item() {
    for (container, takes) in [
        (layouts::furnace_container(2), false),
        (furnace_container_v5(2), true),
    ] {
        let mut c = container;
        c.set_cursor(Some(stack(DIRT, 1)));
        let packet = c.click(1, 0, 0, false);
        assert_eq!(packet.slot_id, 1);
        assert!(packet.clicked_item.is_none());
        assert_eq!(c.slot_stack(1).is_some(), takes);
        assert_eq!(c.cursor().is_none(), takes);
    }
}

#[test]
fn stone_bricks_shift_click_into_the_player_half_of_a_1_7_10_furnace() {
    let brick = ItemStack {
        item_id: 98,
        count: 5,
        damage: 0,
        nbt: None,
    };
    // 1.8.9 smelts stone bricks: the input slot (negative control).
    let mut c = layouts::furnace_container(2);
    c.set_slot_contents(3, Some(brick.clone()));
    c.click(3, 0, 1, false);
    assert_eq!(c.slot_stack(0), Some(&brick));
    // 1.7.10 does not: a main-inventory stack goes to the hotbar.
    let mut c = furnace_container_v5(2);
    c.set_slot_contents(3, Some(brick.clone()));
    c.click(3, 0, 1, false);
    assert!(c.slot_stack(0).is_none());
    assert_eq!(c.slot_stack(30), Some(&brick));
}

// ---- anvil: java17 `ContainerRepair.updateRepairOutput` (`inventory/ContainerRepair.java`) ----
//
// Each expected value is worked out from the java17 source: `base` is its `var19` (the inputs'
// `RepairCost` plus the result's per-enchantment charge), `extra` its `var2`.

const IRON_SWORD: i16 = 267;
const DIAMOND_SWORD: i16 = 276;
const DIAMOND: i16 = 264;
const IRON_INGOT: i16 = 265;
const BOOK: i16 = 403;
const SHARPNESS: i16 = 16;
const SMITE: i16 = 17;

fn item(item_id: i16, count: i32, damage: i16) -> ItemStack {
    ItemStack {
        item_id,
        count,
        damage,
        nbt: None,
    }
}

fn enchanted(mut stack: ItemStack, key: &str, list: &[(i16, i16)]) -> ItemStack {
    let entries = list
        .iter()
        .map(|&(id, level)| enchantment_entry(id, level))
        .collect();
    stack
        .nbt
        .get_or_insert_with(mc_nbt::NbtCompound::new)
        .insert(key, Nbt::list(entries));
    stack
}

fn with_repair_cost(mut stack: ItemStack, repair_cost: i32) -> ItemStack {
    stack
        .nbt
        .get_or_insert_with(mc_nbt::NbtCompound::new)
        .insert("RepairCost", Nbt::Int(repair_cost));
    stack
}

fn named(mut stack: ItemStack, name: &str, repair_cost: i32) -> ItemStack {
    set_custom_name(&mut stack, name);
    with_repair_cost(stack, repair_cost)
}

/// An anvil holding `left` and `right` with `name` in the name field, under `version`'s rules.
fn anvil(version: GameVersion, left: ItemStack, right: Option<ItemStack>, name: &str) -> Container {
    let mut c = layouts::anvil_container(1).with_game_version(version);
    c.set_slot_contents(0, Some(left));
    c.set_slot_contents(1, right);
    c.set_anvil_name(name);
    c
}

fn output(c: &Container) -> &ItemStack {
    c.slot_stack(2).expect("an output")
}

fn output_repair_cost(c: &Container) -> i32 {
    stack_repair_cost(output(c))
}

fn output_enchantments(c: &Container) -> Vec<(i16, i16)> {
    enchantment_map(output(c))
}

#[test]
fn renaming_a_tool_costs_seven_and_writes_repair_cost_two() {
    // extra = 7 (damageable); base 0; RepairCost max(0, 0) - 9 -> 0, + 2.
    let c = anvil(GameVersion::V1_7_10, item(IRON_SWORD, 1, 0), None, "Blade");
    assert_eq!(c.maximum_cost, Some(7));
    assert_eq!(stack_custom_name(output(&c)), Some("Blade"));
    assert_eq!(output_repair_cost(&c), 2);
    // 1.8.9: one level, RepairCost 0 * 2 + 1.
    let c = anvil(GameVersion::V1_8_9, item(IRON_SWORD, 1, 0), None, "Blade");
    assert_eq!(c.maximum_cost, Some(1));
    assert_eq!(output_repair_cost(&c), 1);
}

#[test]
fn renaming_a_stack_costs_five_per_item_capped_at_thirty_nine() {
    // extra = 10 * 5 = 50, the rename's own cost, and >= 40: shown as 39, still takeable.
    let c = anvil(GameVersion::V1_7_10, item(DIRT, 10, 0), None, "Mud");
    assert_eq!(c.maximum_cost, Some(39));
    assert_eq!(output(&c).count, 10);
    assert_eq!(output_repair_cost(&c), 2);
    let c = anvil(GameVersion::V1_7_10, item(DIRT, 5, 0), None, "Mud");
    assert_eq!(c.maximum_cost, Some(25));
}

#[test]
fn renaming_a_named_item_adds_half_the_rename_to_the_base() {
    // base = RepairCost 35 + 7 / 2 = 38; extra 7; 45 -> 39 (only a rename); RepairCost
    // 35 - 9 + 2 = 28.
    let left = named(item(IRON_SWORD, 1, 0), "Old", 35);
    let c = anvil(GameVersion::V1_7_10, left.clone(), None, "New");
    assert_eq!(c.maximum_cost, Some(39));
    assert_eq!(stack_custom_name(output(&c)), Some("New"));
    assert_eq!(output_repair_cost(&c), 28);
    // Clearing the name: extra 7, no base addition, 42 -> 39 (the name is all this repair
    // does); the output loses its name, RepairCost 35 + 2.
    let c = anvil(GameVersion::V1_7_10, left, None, "");
    assert_eq!(c.maximum_cost, Some(39));
    assert_eq!(stack_custom_name(output(&c)), None);
    assert_eq!(output_repair_cost(&c), 37);
}

#[test]
fn material_repair_charges_per_hundred_points_plus_one_per_enchantment() {
    // Diamond sword, damage 1000, Sharpness I, diamonds. Quarter = 1561 / 4 = 390:
    // 1000 -> 610 (+3 +1), 610 -> 220 (+3 +1), 220 -> 0 (+2 +1): extra 11, three diamonds.
    // Result charge: Sharpness at position 1, weight 10 -> 1: base 1 + 1 * 1 = 2. Total 13.
    let left = enchanted(item(DIAMOND_SWORD, 1, 1000), "ench", &[(SHARPNESS, 1)]);
    let mut c = anvil(
        GameVersion::V1_7_10,
        left.clone(),
        Some(item(DIAMOND, 5, 0)),
        "",
    );
    assert_eq!(c.maximum_cost, Some(13));
    assert_eq!(output(&c).damage, 0);
    assert_eq!(output_enchantments(&c), [(SHARPNESS, 1)]);
    assert_eq!(output_repair_cost(&c), 2);
    // Taking it uses exactly three diamonds.
    c.set_player_experience_level(30);
    c.click(2, 0, 0, false);
    assert_eq!(c.slot_stack(1).map(|s| s.count), Some(2));
    // 1.8.9: one level per diamond.
    let c = anvil(GameVersion::V1_8_9, left, Some(item(DIAMOND, 5, 0)), "");
    assert_eq!(c.maximum_cost, Some(3));
}

#[test]
fn a_repair_at_forty_levels_is_too_expensive_outside_creative() {
    // Iron sword damage 100, RepairCost 40, one ingot: 100 -> 38 (+1). 40 + 1 = 41.
    let left = with_repair_cost(item(IRON_SWORD, 1, 100), 40);
    let ingot = item(IRON_INGOT, 1, 0);
    let c = anvil(GameVersion::V1_7_10, left.clone(), Some(ingot.clone()), "");
    assert_eq!(c.maximum_cost, Some(41));
    assert!(c.slot_stack(2).is_none());
    let mut c = layouts::anvil_container(1).with_game_version(GameVersion::V1_7_10);
    c.set_creative_mode(true);
    c.set_slot_contents(0, Some(left));
    c.set_slot_contents(1, Some(ingot));
    assert_eq!(output(&c).damage, 38);
}

#[test]
fn combining_two_tools_charges_for_the_second_tools_share() {
    // Left damage 200, right 100, max 250: 50 + (150 + 30) = 230 -> damage 20; extra
    // max(1, 180 / 100) = 1.
    let right = Some(item(IRON_SWORD, 1, 100));
    let c = anvil(
        GameVersion::V1_7_10,
        item(IRON_SWORD, 1, 200),
        right.clone(),
        "",
    );
    assert_eq!(c.maximum_cost, Some(1));
    assert_eq!(output(&c).damage, 20);
    let c = anvil(GameVersion::V1_8_9, item(IRON_SWORD, 1, 200), right, "");
    assert_eq!(c.maximum_cost, Some(2));
}

#[test]
fn a_book_halves_the_weight_cost_and_the_base() {
    // Sharpness II book on a plain sword: gained 2 at max(1, 1 / 2) = 1 -> extra 2; result
    // charge 1 + 2 * 1 = 3, halved -> 1. Total 3.
    let book = enchanted(item(BOOK, 1, 0), "StoredEnchantments", &[(SHARPNESS, 2)]);
    let sword = item(IRON_SWORD, 1, 0);
    let c = anvil(GameVersion::V1_7_10, sword.clone(), Some(book.clone()), "");
    assert_eq!(c.maximum_cost, Some(3));
    assert_eq!(output_enchantments(&c), [(SHARPNESS, 2)]);
    assert_eq!(output_repair_cost(&c), 2);
    // Equal levels gain one: II + II -> III, gained 1 -> extra 1; base 1 + 3 = 4 -> 2. Total 3.
    let left = enchanted(sword.clone(), "ench", &[(SHARPNESS, 2)]);
    let c = anvil(GameVersion::V1_7_10, left, Some(book.clone()), "");
    assert_eq!(c.maximum_cost, Some(3));
    assert_eq!(output_enchantments(&c), [(SHARPNESS, 3)]);
    // 1.8.9: weight 1, halved to at least 1, times level 2 -> 2; RepairCost 1.
    let c = anvil(GameVersion::V1_8_9, sword, Some(book), "");
    assert_eq!(c.maximum_cost, Some(2));
    assert_eq!(output_repair_cost(&c), 1);
}

#[test]
fn a_level_past_the_maximum_is_still_charged() {
    // V + V -> VI, gained 1, clamped back to V: extra 1; base 1 + 5 = 6 -> 3. Total 4.
    let left = enchanted(item(IRON_SWORD, 1, 0), "ench", &[(SHARPNESS, 5)]);
    let book = enchanted(item(BOOK, 1, 0), "StoredEnchantments", &[(SHARPNESS, 5)]);
    let c = anvil(GameVersion::V1_7_10, left, Some(book), "");
    assert_eq!(c.maximum_cost, Some(4));
    assert_eq!(output_enchantments(&c), [(SHARPNESS, 5)]);
}

#[test]
fn a_conflicting_enchantment_is_charged_but_not_applied() {
    // Sharpness IV onto Smite III: conflict -> extra += gained 4, not applied. Base: Smite at
    // position 1, weight 5 -> 2, halved for the book -> 1: 1 + 3 = 4 -> 2. Total 6.
    let left = enchanted(item(IRON_SWORD, 1, 0), "ench", &[(SMITE, 3)]);
    let book = enchanted(item(BOOK, 1, 0), "StoredEnchantments", &[(SHARPNESS, 4)]);
    let c = anvil(GameVersion::V1_7_10, left, Some(book), "");
    assert_eq!(c.maximum_cost, Some(6));
    assert_eq!(output_enchantments(&c), [(SMITE, 3)]);
}

#[test]
fn an_enchantment_1_7_10_lacks_is_kept_and_costs_nothing() {
    // Depth strider (8) is 1.8.9's: vanilla 1.7.10 would throw. The rename alone: 7.
    let left = enchanted(item(IRON_SWORD, 1, 0), "ench", &[(8, 3)]);
    let c = anvil(GameVersion::V1_7_10, left, None, "Blade");
    assert_eq!(c.maximum_cost, Some(7));
    assert_eq!(output_enchantments(&c), [(8, 3)]);
}

#[test]
fn luck_of_the_sea_and_silk_touch_exclude_each_other_in_1_7_10() {
    // java17 `EnchantmentLootBonus.canApplyTogether` covers looting, fortune and luck of the
    // sea alike.
    assert!(!can_apply_together_v5(61, 33));
    assert!(!can_apply_together_v5(35, 33) && !can_apply_together_v5(33, 35));
    assert!(can_apply_together_v5(33, 21));
    assert!(!can_apply_together_v5(0, 1) && can_apply_together_v5(0, 2));
    assert!(!can_apply_together_v5(16, 18) && !can_apply_together_v5(16, 16));
    // An id 1.7.10 does not register combines with anything.
    assert!(can_apply_together_v5(21, 8));
}

// ---- crafting: java17 `SlotCrafting.onPickupFromSlot` (`inventory/SlotCrafting.java`) ----

const MILK: i16 = 335;
const BUCKET: i16 = 325;
const CAKE: i16 = 354;

/// A workbench holding the cake recipe: three milk buckets, sugar, egg, sugar, three wheat.
fn cake_table(version: GameVersion) -> Container {
    let mut c = layouts::crafting_table_container(4).with_game_version(version);
    let grid = [MILK, MILK, MILK, 353, 344, 353, 296, 296, 296];
    for (slot, item_id) in grid.into_iter().enumerate() {
        c.set_slot_contents(slot + 1, Some(stack(item_id, 1)));
    }
    assert_eq!(c.slot_stack(0).map(|s| s.item_id), Some(CAKE));
    c
}

#[test]
fn crafting_puts_the_empty_buckets_in_the_inventory_first_in_1_7_10() {
    let mut c = cake_table(GameVersion::V1_7_10);
    c.click(0, 0, 0, false);
    assert_eq!(c.cursor().map(|s| s.item_id), Some(CAKE));
    for grid in 1..=9 {
        assert!(c.slot_stack(grid).is_none(), "grid slot {grid}");
    }
    // `addItemStackToInventory` fills the hotbar first: three buckets in its first slot.
    assert_eq!(c.slot_stack(37), Some(&stack(BUCKET, 3)));
    // 1.8.9: the buckets stay in the grid.
    let mut c = cake_table(GameVersion::V1_8_9);
    c.click(0, 0, 0, false);
    for grid in 1..=3 {
        assert_eq!(
            c.slot_stack(grid),
            Some(&stack(BUCKET, 1)),
            "grid slot {grid}"
        );
    }
    assert!(c.slot_stack(37).is_none());
}

#[test]
fn with_a_full_inventory_the_buckets_go_back_to_the_grid_or_vanish_in_creative() {
    for creative in [false, true] {
        let mut c = cake_table(GameVersion::V1_7_10);
        for slot in 10..46 {
            c.set_slot_contents(slot, Some(stack(STONE, 64)));
        }
        c.set_creative_mode(creative);
        c.click(0, 0, 0, creative);
        let expected = (!creative).then(|| stack(BUCKET, 1));
        for grid in 1..=3 {
            assert_eq!(c.slot_stack(grid), expected.as_ref(), "creative {creative}");
        }
    }
}

/// An S2F for the 1.7.10 table's one slot re-evaluates the offers (java17
/// `ContainerEnchantment.onCraftMatrixChanged`), as one for either of 1.8.9's table slots does;
/// slot 1 is the player's inventory in 1.7.10.
#[test]
fn a_set_slot_on_the_1_7_10_tables_slot_re_evaluates_the_offers() {
    for (slot, changes) in [(0, true), (1, false)] {
        let mut c = enchanting_table_container_v5(5);
        c.apply_set_slot(&s2c::SetSlot {
            window_id: 5,
            slot,
            item: Some(item(276, 1, 0)),
        });
        assert_eq!(c.enchanting_inputs_changed, changes, "slot {slot}");
    }
}

// ---- the anvil against the real java17 `ContainerRepair` ----
//
// `tests/fixtures/v17_anvil/anvil_ref.txt` is written by `tests/javaref/V17AnvilRef.java`
// (`generate-v17-anvil.ps1`): one case per line, `name, creative, left, right, the name field,
// maximumCost, materialCost, output`, stacks as `id,count,damage` plus `;ench=id.lvl,...`,
// `;stored=...`, `;name=...`, `;rc=N`, or `-`; then the whole tag of the left input, the right
// input and the output in the oracle's canonical text ([`show_nbt`]).

fn parse_levels(list: &str) -> Vec<(i16, i16)> {
    if list.is_empty() {
        return Vec::new();
    }
    list.split(',')
        .map(|entry| {
            let (id, level) = entry.split_once('.').expect("id.lvl");
            (id.parse().expect("id"), level.parse().expect("lvl"))
        })
        .collect()
}

fn parse_stack(text: &str) -> Option<ItemStack> {
    if text == "-" {
        return None;
    }
    let mut parts = text.split(';');
    let head: Vec<i32> = parts
        .next()
        .expect("id,count,damage")
        .split(',')
        .map(|n| n.parse().expect("number"))
        .collect();
    let mut stack = item(head[0] as i16, head[1], head[2] as i16);
    for part in parts {
        let (key, value) = part.split_once('=').expect("key=value");
        match key {
            "ench" => stack = enchanted(stack, "ench", &parse_levels(value)),
            "stored" => stack = enchanted(stack, "StoredEnchantments", &parse_levels(value)),
            "name" => set_custom_name(&mut stack, value),
            "rc" => stack = with_repair_cost(stack, value.parse().expect("rc")),
            other => panic!("unknown field {other}"),
        }
    }
    Some(stack)
}

fn show_levels(stack: &ItemStack, key: &str) -> Option<String> {
    let list = stack.nbt.as_ref()?.get_list(key)?;
    let levels: Vec<String> = list
        .iter()
        .map(|entry| {
            let entry = entry.as_compound().expect("a compound");
            let id = entry.get_i16("id").unwrap_or(0);
            let level = entry.get_i16("lvl").unwrap_or(0);
            format!("{id}.{level}")
        })
        .collect();
    Some(levels.join(","))
}

/// The fixture's stack format, from the Rust stack.
fn show_stack(stack: Option<&ItemStack>) -> String {
    let Some(stack) = stack else {
        return "-".to_owned();
    };
    let mut out = format!("{},{},{}", stack.item_id, stack.count, stack.damage);
    if let Some(levels) = show_levels(stack, "ench") {
        out.push_str(&format!(";ench={levels}"));
    }
    if let Some(levels) = show_levels(stack, "StoredEnchantments") {
        out.push_str(&format!(";stored={levels}"));
    }
    if let Some(name) = stack_custom_name(stack) {
        out.push_str(&format!(";name={name}"));
    }
    if let Some(Nbt::Int(cost)) = stack.nbt.as_ref().and_then(|root| root.get("RepairCost")) {
        out.push_str(&format!(";rc={cost}"));
    }
    out
}

/// The oracle's canonical text for a stack's whole tag (`V17AnvilRef.nbt`), `-` for none: the
/// server compares the anvil output a click sends with its own by the whole tag.
fn show_nbt(stack: Option<&ItemStack>) -> String {
    match stack.and_then(|stack| stack.nbt.as_ref()) {
        Some(tag) => show_compound(tag),
        None => "-".to_owned(),
    }
}

/// Keys in Java's `String` order (UTF-16 units).
fn show_compound(tag: &mc_nbt::NbtCompound) -> String {
    let mut keys: Vec<&str> = tag.keys().collect();
    keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    let entries: Vec<String> = keys
        .into_iter()
        .map(|key| format!("{key}:{}", show_node(tag.get(key).expect("a key"))))
        .collect();
    format!("{{{}}}", entries.join(","))
}

fn show_node(node: &Nbt) -> String {
    let join = |items: Vec<String>| items.join(",");
    match node {
        Nbt::End => panic!("TAG_End inside a tag"),
        Nbt::Byte(n) => format!("{n}b"),
        Nbt::Short(n) => format!("{n}s"),
        Nbt::Int(n) => n.to_string(),
        Nbt::Long(n) => format!("{n}L"),
        Nbt::Float(n) => format!("{:x}f", n.to_bits()),
        Nbt::Double(n) => format!("{:x}d", n.to_bits()),
        Nbt::ByteArray(bytes) => {
            let bytes = bytes.iter().map(|&b| (b as i8).to_string()).collect();
            format!("[B;{}]", join(bytes))
        }
        Nbt::String(text) => {
            let mut out = String::from("\"");
            for unit in text.encode_utf16() {
                match char::from_u32(u32::from(unit)) {
                    Some(c) if (' '..='~').contains(&c) && c != '"' && c != '\\' => out.push(c),
                    _ => out.push_str(&format!("\\u{unit:04x}")),
                }
            }
            out.push('"');
            out
        }
        Nbt::List(list) => format!("[{}]", join(list.iter().map(show_node).collect())),
        Nbt::Compound(tag) => show_compound(tag),
        Nbt::IntArray(ints) => format!("[I;{}]", join(ints.iter().map(i32::to_string).collect())),
    }
}

#[test]
fn the_anvil_matches_java17_container_repair_case_by_case() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/v17_anvil/anvil_ref.txt"
    );
    let fixture = std::fs::read_to_string(path).expect("the java17 anvil fixture");
    let mut cases = 0;
    for line in fixture.lines().filter(|line| !line.is_empty()) {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 11, "{line}");
        let (name, creative) = (fields[0], fields[1] == "true");
        let left = parse_stack(fields[2]).expect("a left input");
        let right = parse_stack(fields[3]);
        // The fixture echoes its inputs: the parser must rebuild them exactly, tags included.
        assert_eq!(show_stack(Some(&left)), fields[2], "{name}: left input");
        assert_eq!(show_stack(right.as_ref()), fields[3], "{name}: right input");
        assert_eq!(show_nbt(Some(&left)), fields[8], "{name}: left tag");
        assert_eq!(show_nbt(right.as_ref()), fields[9], "{name}: right tag");
        let mut c = layouts::anvil_container(1).with_game_version(GameVersion::V1_7_10);
        c.set_creative_mode(creative);
        c.set_slot_contents(0, Some(left));
        c.set_slot_contents(1, right);
        c.set_anvil_name(fields[4]);
        let cost: i32 = fields[5].parse().expect("maximumCost");
        let material: i32 = fields[6].parse().expect("materialCost");
        assert_eq!(c.maximum_cost, Some(cost), "{name}: maximumCost");
        assert_eq!(c.anvil_material_cost, material, "{name}: materialCost");
        assert_eq!(show_stack(c.slot_stack(2)), fields[7], "{name}: output");
        assert_eq!(show_nbt(c.slot_stack(2)), fields[10], "{name}: output tag");
        cases += 1;
    }
    assert_eq!(cases, 46);
}
