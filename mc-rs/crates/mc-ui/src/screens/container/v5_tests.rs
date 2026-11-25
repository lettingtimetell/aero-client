//! 1.7.10 window lowering and container shapes, against java17 `handleOpenWindow`.

use super::*;

fn open(inventory_type: u8, title: &str, slot_count: u8, provided: bool) -> s2c5::OpenWindow {
    s2c5::OpenWindow {
        window_id: 7,
        inventory_type,
        title: title.to_owned(),
        slot_count,
        use_provided_title: provided,
        entity_id: 0,
    }
}

#[test]
fn every_window_number_lowers_to_its_family_and_numbers_past_eleven_open_nothing() {
    for (number, name) in WINDOW_TYPES_V5.iter().enumerate() {
        let lowered = open_window_v47(&open(number as u8, "x", 9, false)).expect("a 1.7.10 type");
        assert_eq!(lowered.inventory_type, *name);
        assert_eq!(lowered.window_id, 7);
        assert_eq!(lowered.entity_id.is_some(), number == 11);
    }
    for number in [12, 13, 200, 255] {
        assert!(open_window_v47(&open(number, "x", 9, false)).is_none());
    }
}

#[test]
fn titles_follow_the_1_7_10_screens() {
    let title = |t: u8, s: &str, provided: bool| {
        open_window_v47(&open(t, s, 27, provided))
            .unwrap()
            .window_title
    };
    // A chest's (and a horse's) unprovided title is a translation key.
    assert_eq!(
        title(0, "container.chest", false),
        translated("container.chest")
    );
    assert_eq!(
        title(0, "Shop", true),
        ChatComponent::text("Shop").to_json()
    );
    // Tile-entity windows fall back to their own key; most also for an empty provided name.
    assert_eq!(title(2, "", true), translated("container.furnace"));
    assert_eq!(
        title(2, "Smelter", true),
        ChatComponent::text("Smelter").to_json()
    );
    assert_eq!(title(9, "ignored", false), translated("container.hopper"));
    assert_eq!(title(6, "", true), translated("entity.Villager.name"));
    // The enchanting table, the dispenser and the dropper keep a provided name even when it is
    // empty.
    let empty = ChatComponent::text("").to_json();
    for window in [4, 3, 10] {
        assert_eq!(title(window, "", true), empty, "window {window}");
    }
    assert_eq!(title(4, "x", false), translated("container.enchant"));
    assert_eq!(title(3, "x", false), translated("container.dispenser"));
    assert_eq!(title(10, "x", false), translated("container.dropper"));
}

#[test]
fn client_built_windows_lower_with_no_slots_and_fixed_sizes_elsewhere() {
    let count = |t: u8, sent: u8| {
        open_window_v47(&open(t, "x", sent, false))
            .unwrap()
            .slot_count
    };
    // The 1.7.10 server sends 9 for the crafting table, the enchanting table and the anvil.
    assert_eq!(count(1, 9), 0);
    assert_eq!(count(4, 9), 0);
    assert_eq!(count(8, 9), 0);
    assert_eq!(count(0, 54), 54);
    assert_eq!(count(11, 17), 17);
    assert_eq!(count(9, 9), 5);
    assert_eq!(count(2, 0), 3);
    assert_eq!(count(5, 0), 4);
}

#[test]
fn the_1_7_10_containers_have_1_7_10_shapes_and_rules() {
    let build = |t: u8, sent: u8| {
        let lowered = open_window_v47(&open(t, "x", sent, false)).unwrap();
        container_for_open_window_v5(&lowered).expect("a 1.7.10 window")
    };
    let enchanting = build(4, 9);
    assert_eq!(enchanting.kind, ContainerKind::EnchantingV5);
    assert_eq!(enchanting.slots().len(), 37);
    assert_eq!(enchanting.game_version(), GameVersion::V1_7_10);
    let furnace = build(2, 3);
    assert_eq!(furnace.kind, ContainerKind::Furnace);
    assert_eq!(furnace.slots()[1].role, SlotRole::Normal);
    assert_eq!(build(1, 9).kind, ContainerKind::Workbench);
    assert_eq!(build(8, 9).kind, ContainerKind::Anvil);
    assert_eq!(build(9, 9).slots().len(), 5 + 36);
    assert_eq!(build(0, 54).slots().len(), 54 + 36);
    for t in 0..11u8 {
        assert_eq!(
            build(t, 27).game_version(),
            GameVersion::V1_7_10,
            "type {t}"
        );
    }
    // Negative control: the 1.8.9 builder on the same lowered enchanting window has the lapis
    // slot.
    let lowered = open_window_v47(&open(4, "x", 9, false)).unwrap();
    let v47 = super::super::net::container_for_open_window(&lowered).unwrap();
    assert_eq!(v47.kind, ContainerKind::Enchanting);
    assert_eq!(v47.slots().len(), 38);
}

#[test]
fn a_horse_window_needs_its_entity_and_a_valid_size() {
    let mut pkt = open(11, "entity.horse.name", 2, false);
    pkt.entity_id = 42;
    let lowered = open_window_v47(&pkt).unwrap();
    assert_eq!(lowered.entity_id, Some(42));
    let horse = container_for_open_window_v5(&lowered).unwrap();
    assert!(matches!(horse.kind, ContainerKind::Horse { top: 2 }));
}

#[test]
fn stone_bricks_and_wet_sponge_do_not_smelt_in_1_7_10() {
    let stack = |item_id: i16, damage: i16| ItemStack {
        item_id,
        count: 1,
        damage,
        nbt: None,
    };
    for (item, damage) in [(98, 0), (19, 1)] {
        assert!(
            super::super::slot::is_smeltable(&stack(item, damage)),
            "1.8.9 smelts it"
        );
        assert!(!is_smeltable_v5(&stack(item, damage)));
    }
    // Cracked stone bricks and dry sponge smelt in neither; cobblestone and iron ore in both.
    assert!(!is_smeltable_v5(&stack(98, 2)));
    assert!(is_smeltable_v5(&stack(4, 0)));
    assert!(is_smeltable_v5(&stack(15, 0)));
}
