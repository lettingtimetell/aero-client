//! The 1.7.10 item rules oracle: what a 1.7.10 container's click prediction reads about each item
//! and where its shift-click sends the item, against the real 1.7.10 containers.
//!
//! The fixture comes from `tests/javaref/V17ItemRulesRef.java` (run by
//! `tests/javaref/generate-v17-item-rules.ps1` against the deobfuscated 1.7.10 client classes):
//! every item 1.7.10 registers, at eight damages, with its stack size, subtypes and
//! damageability, which armour, brewing and beacon slots take it, and the slots after a
//! shift-click in the player's inventory, a furnace, a brewing stand and a beacon, plus a merge
//! onto a stack of the same item and onto one of the next damage. The containers here are the ones
//! a 1.7.10 session builds; each must end exactly as java17's.

use std::path::PathBuf;

use mc_protocol::play::s2c;
use mc_ui::screens::container::{
    container_for_open_window, layouts, stack_ext, v5, Container, ItemStack, SlotRole,
};
use mc_version::GameVersion;

const V17: GameVersion = GameVersion::V1_7_10;

/// One fixture line: the item, the properties Java reported, and the shift-click results.
struct Row {
    line: usize,
    id: i16,
    damage: i16,
    max_stack: i32,
    subtypes: bool,
    damageable: bool,
    stackable: bool,
    smelts: bool,
    armor: [bool; 4],
    ingredient: bool,
    bottle: bool,
    beacon: bool,
    /// `(scenario, slots)`: `P`, `F`, `B`, `E`, `M`, `S` as the Java file documents.
    shifts: Vec<(String, String)>,
}

fn flag(text: &str) -> bool {
    match text {
        "1" => true,
        "0" => false,
        other => panic!("a flag, got {other}"),
    }
}

fn read_rows() -> Vec<Row> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v17_item_rules/item_rules_ref.txt");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 19, "line {}", index + 1);
            let armor: Vec<bool> = fields[8].chars().map(|c| c == '1').collect();
            Row {
                line: index + 1,
                id: fields[0].parse().unwrap(),
                damage: fields[1].parse().unwrap(),
                max_stack: fields[2].parse().unwrap(),
                subtypes: flag(fields[3]),
                damageable: flag(fields[4]),
                stackable: flag(fields[5]),
                // fields[6] is `TileEntityFurnace.isItemFuel`: the furnace shift-click shows it.
                smelts: flag(fields[7]),
                armor: armor.try_into().unwrap(),
                ingredient: flag(fields[9]),
                bottle: flag(fields[10]),
                beacon: flag(fields[11]),
                shifts: fields[12..]
                    .iter()
                    .map(|field| {
                        let (name, slots) = field.split_once('=').unwrap();
                        (name.to_owned(), slots.to_owned())
                    })
                    .collect(),
            }
        })
        .collect()
}

fn stack(id: i16, count: i32, damage: i16) -> ItemStack {
    ItemStack {
        item_id: id,
        count,
        damage,
        nbt: None,
    }
}

/// Every non-empty slot as `index:count:damage`, `-` when none (the Java file's format).
fn slots(container: &Container) -> String {
    let list: Vec<String> = (0..container.slots().len())
        .filter_map(|index| {
            let stack = container.slot_stack(index)?;
            Some(format!("{index}:{}:{}", stack.count, stack.damage))
        })
        .collect();
    if list.is_empty() {
        "-".to_owned()
    } else {
        list.join(",")
    }
}

/// Puts `stack` into `slot`, shift-clicks it and lists the container's slots.
fn shift(mut container: Container, slot: usize, stack: ItemStack) -> String {
    container.set_slot_contents(slot, Some(stack));
    container.slot_click(slot as i32, 0, 1, false);
    slots(&container)
}

fn player(version: GameVersion) -> Container {
    layouts::player_container(0).with_game_version(version)
}

/// The shift-click scenario `name` in containers of `version`.
fn scenario(version: GameVersion, name: &str, id: i16, damage: i16) -> String {
    match name {
        "P" => shift(player(version), 9, stack(id, 1, damage)),
        "F" => {
            let furnace = match version {
                GameVersion::V1_7_10 => v5::furnace_container_v5(1),
                GameVersion::V1_8_9 => layouts::furnace_container(1),
            };
            shift(furnace, 3, stack(id, 1, damage))
        }
        "B" => {
            let brewing = layouts::brewing_stand_container(1).with_game_version(version);
            shift(brewing, 4, stack(id, 1, damage))
        }
        "E" => shift(
            layouts::beacon_container(1).with_game_version(version),
            1,
            stack(id, 1, damage),
        ),
        "M" => {
            let mut merge = player(version);
            merge.set_slot_contents(36, Some(stack(id, 1, damage)));
            shift(merge, 9, stack(id, 2, damage))
        }
        "S" => {
            let mut subtype = player(version);
            subtype.set_slot_contents(36, Some(stack(id, 1, damage)));
            shift(subtype, 9, stack(id, 1, damage + 1))
        }
        "K" => key_swap(version, stack(id, 3, damage)),
        other => panic!("no scenario {other}"),
    }
}

/// The number key 1 over a chest's first slot (one stone) with `held` in the first hotbar slot
/// (54): the stone takes the hotbar slot and `held` goes back into the player's inventory.
fn key_swap(version: GameVersion, held: ItemStack) -> String {
    let mut chest = layouts::chest_container(1, 3).with_game_version(version);
    chest.set_slot_contents(54, Some(held));
    chest.set_slot_contents(0, Some(stack(1, 1, 0)));
    chest.slot_click(0, 0, 2, false);
    slots(&chest)
}

#[test]
fn every_1_7_10_item_has_its_1_7_10_properties_and_shift_clicks() {
    let rows = read_rows();
    assert_eq!(rows.len() % 8, 0);
    assert!(rows.len() / 8 > 300, "every item 1.7.10 registers");
    let player = player(V17);
    let brewing = layouts::brewing_stand_container(1).with_game_version(V17);
    let beacon = layouts::beacon_container(1).with_game_version(V17);
    let mut wrong = Vec::new();
    for row in &rows {
        let s = stack(row.id, 1, row.damage);
        let armor: Vec<bool> = (0..4)
            .map(|k| player.slots()[5 + k].is_item_valid(&s))
            .collect();
        let properties = [
            (
                "max stack",
                i64::from(stack_ext::max_stack_size_in(V17, &s)),
                i64::from(row.max_stack),
            ),
            (
                "subtypes",
                stack_ext::has_subtypes_in(V17, &s).into(),
                row.subtypes.into(),
            ),
            (
                "damageable",
                stack_ext::is_stack_damageable_in(V17, &s).into(),
                row.damageable.into(),
            ),
            (
                "stackable",
                stack_ext::is_stackable_in(V17, &s).into(),
                row.stackable.into(),
            ),
            ("smelts", v5::is_smeltable_v5(&s).into(), row.smelts.into()),
            (
                "ingredient",
                brewing.slots()[3].is_item_valid(&s).into(),
                row.ingredient.into(),
            ),
            (
                "bottle",
                brewing.slots()[0].is_item_valid(&s).into(),
                row.bottle.into(),
            ),
            (
                "beacon",
                beacon.slots()[0].is_item_valid(&s).into(),
                row.beacon.into(),
            ),
        ];
        for (name, got, want) in properties {
            if got != want {
                wrong.push(format!(
                    "line {} item {}:{} {name}: rust {got}, java {want}",
                    row.line, row.id, row.damage
                ));
            }
        }
        if armor != row.armor {
            wrong.push(format!(
                "line {} item {}:{} armour slots: rust {armor:?}, java {:?}",
                row.line, row.id, row.damage, row.armor
            ));
        }
        for (name, want) in &row.shifts {
            let got = scenario(V17, name, row.id, row.damage);
            if &got != want {
                wrong.push(format!(
                    "line {} item {}:{} shift {name}: rust {got}, java {want}",
                    row.line, row.id, row.damage
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} differences:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(30)].join("\n")
    );
}

/// The negative control: with 1.8.9's rules the same scenarios differ, doors and stone among
/// them.
#[test]
fn the_1_8_9_rules_shift_click_differently() {
    let rows = read_rows();
    let differing: Vec<&Row> = rows
        .iter()
        .filter(|row| {
            row.shifts.iter().any(|(name, want)| {
                &scenario(GameVersion::V1_8_9, name, row.id, row.damage) != want
            })
        })
        .collect();
    let has = |id: i16| differing.iter().any(|row| row.id == id);
    assert!(has(324) && has(330), "doors stack to 64 in 1.8.9");
    assert!(has(1), "stone has subtypes in 1.8.9");
    assert!(has(9), "1.8.9 has no water item");
    // A door merged onto a door: 1.7.10 leaves both stacks, 1.8.9 merges them.
    assert_eq!(scenario(V17, "M", 324, 0), "36:1:0,37:2:0");
    assert_eq!(scenario(GameVersion::V1_8_9, "M", 324, 0), "36:3:0");
}

const DOOR: i16 = 324;

/// java17 `Container.slotClick` mode 0 on a slot holding the cursor's item moves
/// `min(cursor, slot limit - n, getMaxStackSize() - n)` items: nothing for a door on a door, as
/// 1.7.10 doors stack to one. 1.8.9 moves the door.
#[test]
fn a_door_clicked_onto_a_door_stays_on_the_cursor_in_1_7_10() {
    for (version, slot, cursor) in [(V17, 1, Some(1)), (GameVersion::V1_8_9, 2, None)] {
        let mut container = player(version);
        container.set_slot_contents(9, Some(stack(DOOR, 1, 0)));
        container.set_cursor(Some(stack(DOOR, 1, 0)));
        container.slot_click(9, 0, 0, false);
        assert_eq!(
            container.slot_stack(9).map(|s| s.count),
            Some(slot),
            "{version:?}"
        );
        assert_eq!(container.cursor().map(|s| s.count), cursor, "{version:?}");
    }
}

/// Shift-clicking a crafting table's result crafts until the grid runs out: 1.7.10 makes one door
/// per craft and each goes to its own slot; 1.8.9 makes three and stacks them.
#[test]
fn shift_clicking_the_door_recipe_fills_one_slot_per_door_in_1_7_10() {
    for version in [V17, GameVersion::V1_8_9] {
        let mut table = layouts::crafting_table_container(1).with_game_version(version);
        for slot in [1, 2, 4, 5, 7, 8] {
            table.set_slot_contents(slot, Some(stack(5, 2, 0)));
        }
        let first = table.slot_stack(0).cloned();
        table.slot_click(0, 0, 1, false);
        let doors: Vec<(usize, i32)> = (10..46)
            .filter_map(|slot| table.slot_stack(slot).map(|s| (slot, s.count)))
            .collect();
        match version {
            GameVersion::V1_7_10 => {
                assert_eq!(first, Some(stack(DOOR, 1, 0)));
                assert_eq!(doors, [(44, 1), (45, 1)]);
            }
            GameVersion::V1_8_9 => {
                assert_eq!(first, Some(stack(DOOR, 3, 0)));
                assert_eq!(doors, [(45, 6)]);
            }
        }
        assert!(
            (1..10).all(|slot| table.slot_stack(slot).is_none()),
            "{version:?}: grid used up"
        );
        assert_eq!(
            table.slot_stack(0),
            None,
            "{version:?}: nothing left to craft"
        );
    }
}

/// The creative pick (mode 3) copies the stack at its maximum size, and a double-click (mode 6)
/// gathers up to the cursor's maximum: one door in 1.7.10.
#[test]
fn a_full_stack_of_doors_is_one_door_in_1_7_10() {
    for (version, full) in [(V17, 1), (GameVersion::V1_8_9, 64)] {
        let mut container = player(version);
        container.set_slot_contents(9, Some(stack(DOOR, 1, 0)));
        container.slot_click(9, 2, 3, true);
        assert_eq!(
            container.cursor().map(|s| s.count),
            Some(full),
            "{version:?} pick"
        );

        let mut container = player(version);
        container.set_slot_contents(10, Some(stack(DOOR, 1, 0)));
        container.set_slot_contents(11, Some(stack(DOOR, 1, 0)));
        container.set_cursor(Some(stack(DOOR, 1, 0)));
        container.slot_click(9, 0, 6, false);
        let gathered = container.cursor().map_or(0, |s| s.count);
        assert_eq!(gathered, full.min(3), "{version:?} double-click");
    }
}

/// java17 `InventoryPlayer.storePartialItemStack` (`:228-244`): a stack of an item that stacks to
/// one goes back whole into the first empty slot. 1.8.9 puts one item into each empty slot.
#[test]
fn a_swapped_out_stack_of_book_and_quills_stays_whole_in_1_7_10() {
    let quills = stack(386, 3, 0);
    assert_eq!(key_swap(V17, quills.clone()), "54:1:0,55:3:0");
    assert_eq!(
        key_swap(GameVersion::V1_8_9, quills),
        "54:1:0,55:1:0,56:1:0,57:1:0"
    );
}

fn horse_window() -> s2c::OpenWindow {
    s2c::OpenWindow {
        window_id: 3,
        inventory_type: "EntityHorse".to_owned(),
        window_title: r#"{"text":"Horse"}"#.to_owned(),
        slot_count: 2,
        entity_id: Some(42),
    }
}

/// java17 `ContainerHorseInventory` slot 1 (`:32-34`) takes, from a horse that can wear armour,
/// exactly the three horse armours (`EntityHorse.func_146085_a`, `:1647-1650`), and a shift-click
/// from the player's slots sends one there (`transferStackInSlot`, `:91-97`).
#[test]
fn a_1_7_10_horse_takes_the_three_horse_armours_and_nothing_else() {
    let mut horse = v5::container_for_open_window_v5(&horse_window()).expect("a horse window");
    let horse_armour = [417, 418, 419].map(|id| stack(id, 1, 0));
    let others = [306, 307, 310, 314, 329, 1].map(|id| stack(id, 1, 0));
    // Until the entity's type is known the slot takes nothing and is not hovered.
    assert!(horse_armour
        .iter()
        .all(|s| !horse.slots()[1].is_item_valid(s)));
    assert!(!horse.slots()[1].can_be_hovered());

    horse.set_horse_can_wear_armor(true);
    assert!(horse_armour
        .iter()
        .all(|s| horse.slots()[1].is_item_valid(s)));
    assert!(others.iter().all(|s| !horse.slots()[1].is_item_valid(s)));
    assert!(horse.slots()[1].can_be_hovered());
    horse.set_slot_contents(2, Some(stack(419, 1, 0)));
    horse.slot_click(2, 0, 1, false);
    assert_eq!(horse.slot_stack(1).map(|s| s.item_id), Some(419));

    // A donkey or a mule wears none.
    horse.set_horse_can_wear_armor(false);
    assert!(!horse.slots()[1].is_item_valid(&horse_armour[0]));
    assert!(!horse.slots()[1].can_be_hovered());
}

/// The 1.8.9 control: a 1.8.9 horse window keeps today's 1.8.9 rule (the slot takes an item
/// with an armour material, see `container/net.rs`'s horse tests), untouched by the 1.7.10 one.
#[test]
fn a_1_8_9_horse_keeps_its_own_armour_rule() {
    let mut horse = container_for_open_window(&horse_window()).expect("a horse window");
    horse.set_horse_can_wear_armor(true);
    assert_eq!(
        horse.slots()[1].role,
        SlotRole::HorseArmor { can_wear: true }
    );
    for id in [417, 418, 419, 306, 307, 310, 314, 329, 1] {
        let s = stack(id, 1, 0);
        assert_eq!(
            horse.slots()[1].is_item_valid(&s),
            stack_ext::props(&s).armor_material.is_some(),
            "item {id}"
        );
    }
}
