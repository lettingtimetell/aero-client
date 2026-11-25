//! The creative inventory oracle: what each creative tab lists, and the pool the search tab
//! filters, in a session of each game version, against the real creative tabs of that version.
//!
//! The fixtures come from `tests/javaref/CreativeTabsRef.java`, run by
//! `tests/javaref/generate-creative-tabs.ps1` against the deobfuscated 1.7.10 and 1.8.9 client
//! classes: every tab's `displayAllReleventItems` and the search tab's pool, every stack in full
//! (item, count, damage, tag). The creative screen picks its stacks from these lists and sends
//! them in creative inventory actions, so a 1.7.10 session must list exactly 1.7.10's, in its
//! order: never an item a 1.7.10 server does not register.

use std::path::PathBuf;

use mc_nbt::Nbt;
use mc_ui::screens::container::creative::{search_stacks_in, stacks_for_tab_in, CreativeTab, TABS};
use mc_ui::screens::container::ItemStack;
use mc_version::GameVersion;

struct Fixture {
    /// `(tab index, label, stacks)` in tab order.
    tabs: Vec<(usize, String, Vec<ItemStack>)>,
    search: Vec<ItemStack>,
}

fn decode_hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn parse_stack(text: &str) -> ItemStack {
    let mut parts = text.split(',');
    let item_id = parts.next().unwrap().parse().unwrap();
    let count = parts.next().unwrap().parse().unwrap();
    let damage = parts.next().unwrap().parse().unwrap();
    let nbt =
        parts.next().map(
            |hex| match mc_nbt::from_slice(&decode_hex(hex)).expect("an NBT tag").1 {
                Nbt::Compound(compound) => compound,
                other => panic!("a compound tag, got {other:?}"),
            },
        );
    ItemStack {
        item_id,
        count,
        damage,
        nbt,
    }
}

fn read_fixture(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/creative")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines();
    let mut tabs = Vec::new();
    let mut search = None;
    while let Some(line) = lines.next() {
        let fields: Vec<&str> = line.split(' ').collect();
        let count: usize = fields.last().unwrap().parse().expect("a list length");
        let stacks: Vec<ItemStack> = (&mut lines).take(count).map(parse_stack).collect();
        assert_eq!(stacks.len(), count, "{line}");
        match fields[0] {
            "TAB" => tabs.push((fields[1].parse().unwrap(), fields[2].to_owned(), stacks)),
            "SEARCH" => search = Some(stacks),
            other => panic!("unexpected line {other}"),
        }
    }
    Fixture {
        tabs,
        search: search.expect("the search pool"),
    }
}

/// What a creative screen of `version` lists on the tab with this label: a content tab's list,
/// the search tab's whole pool (its box starts empty), nothing for the inventory tab.
fn listed(version: GameVersion, label: &str) -> Vec<ItemStack> {
    match TABS.iter().find(|tab| tab.label() == label) {
        Some(CreativeTab::Items(label)) => stacks_for_tab_in(version, label),
        Some(CreativeTab::Search) | Some(CreativeTab::Inventory) => Vec::new(),
        None => panic!("no tab {label}"),
    }
}

/// Every tab's list and the search pool of a `version` session against the fixture: the names
/// of the lists that differ.
fn differing(version: GameVersion, fixture: &Fixture) -> Vec<String> {
    let mut wrong = Vec::new();
    for (index, label, stacks) in &fixture.tabs {
        assert_eq!(TABS[*index].label(), label, "tab {index}");
        if &listed(version, label) != stacks {
            wrong.push(label.clone());
        }
    }
    if search_stacks_in(version, "", false, None) != fixture.search {
        wrong.push("search pool".to_owned());
    }
    wrong
}

/// The first difference between two lists, for a readable failure.
fn first_difference(got: &[ItemStack], want: &[ItemStack]) -> String {
    let index = got
        .iter()
        .zip(want)
        .position(|(got, want)| got != want)
        .unwrap_or(got.len().min(want.len()));
    format!(
        "lengths {} / {}, first difference at {index}: rust {:?}, java {:?}",
        got.len(),
        want.len(),
        got.get(index),
        want.get(index)
    )
}

#[test]
fn every_1_7_10_tab_lists_what_the_1_7_10_creative_tabs_list() {
    let fixture = read_fixture("creative_1_7_10.txt");
    assert_eq!(fixture.tabs.len(), 12);
    for (_, label, stacks) in &fixture.tabs {
        let got = listed(GameVersion::V1_7_10, label);
        assert!(
            &got == stacks,
            "{label}: {}",
            first_difference(&got, stacks)
        );
    }
    let pool = search_stacks_in(GameVersion::V1_7_10, "", false, None);
    assert!(
        pool == fixture.search,
        "search: {}",
        first_difference(&pool, &fixture.search)
    );
}

/// The 1.8.9 control: a 1.8.9 session's lists are the ones it had (`stacks_for_tab`,
/// `search_stacks`), and they are java18's.
#[test]
fn every_1_8_9_tab_lists_what_the_1_8_9_creative_tabs_list() {
    let fixture = read_fixture("creative_1_8_9.txt");
    assert_eq!(fixture.tabs.len(), 12);
    for (_, label, stacks) in &fixture.tabs {
        let got = listed(GameVersion::V1_8_9, label);
        if let Some(CreativeTab::Items(label)) = TABS.iter().find(|tab| tab.label() == label) {
            assert_eq!(
                got,
                mc_ui::screens::container::creative::stacks_for_tab(label)
            );
        }
        assert!(
            &got == stacks,
            "{label}: {}",
            first_difference(&got, stacks)
        );
    }
    let pool = search_stacks_in(GameVersion::V1_8_9, "", false, None);
    assert_eq!(
        pool,
        mc_ui::screens::container::creative::search_stacks("", false, None)
    );
    assert!(
        pool == fixture.search,
        "search: {}",
        first_difference(&pool, &fixture.search)
    );
}

/// The negative control: 1.8.9's lists are not 1.7.10's, and they hold items 1.7.10 lacks.
#[test]
fn the_1_8_9_lists_miss_the_1_7_10_ones() {
    let fixture = read_fixture("creative_1_7_10.txt");
    let wrong = differing(GameVersion::V1_8_9, &fixture);
    assert!(wrong.len() >= 8, "only {wrong:?} differ");
    assert!(wrong.contains(&"search pool".to_owned()));
    let unknown_to_1_7_10 = |stack: &ItemStack| {
        mc_data::v5::try_item_props_v5(mc_data::ItemId(stack.item_id as u16)).is_none()
    };
    assert!(stacks_for_tab_in(GameVersion::V1_8_9, "buildingBlocks")
        .iter()
        .any(unknown_to_1_7_10));
    assert!(differing(GameVersion::V1_7_10, &fixture).is_empty());
}

/// Every stack a 1.7.10 list offers is an item 1.7.10 registers.
#[test]
fn every_stack_a_1_7_10_session_lists_is_a_1_7_10_item() {
    let stacks: Vec<ItemStack> = TABS
        .iter()
        .flat_map(|tab| listed(GameVersion::V1_7_10, tab.label()))
        .chain(search_stacks_in(GameVersion::V1_7_10, "", false, None))
        .collect();
    assert!(stacks.len() > 1000);
    for stack in stacks {
        assert!(
            mc_data::v5::try_item_props_v5(mc_data::ItemId(stack.item_id as u16)).is_some(),
            "{stack:?}"
        );
    }
}

/// java17 `GuiContainerCreative.updateCreativeSearch` (`:358`) matches each tooltip line with its
/// formatting codes, java18 (`:372`) strips them first: a query of `9` finds the diamond sword by
/// its blue attack-damage line (`§9+7 Attack Damage`) in a 1.7.10 session only.
#[test]
fn a_1_7_10_search_reads_formatting_codes_and_a_1_8_9_search_does_not() {
    let finds_sword = |version| {
        search_stacks_in(version, "9", false, None)
            .iter()
            .any(|stack| stack.item_id == 276)
    };
    assert!(finds_sword(GameVersion::V1_7_10));
    assert!(!finds_sword(GameVersion::V1_8_9));
    let with_code = |version| search_stacks_in(version, "\u{a7}", false, None).len();
    assert!(with_code(GameVersion::V1_7_10) > 100);
    assert_eq!(with_code(GameVersion::V1_8_9), 0);
}
