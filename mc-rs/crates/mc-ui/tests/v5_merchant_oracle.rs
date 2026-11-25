//! The villager trade oracle: the output a merchant window predicts for the stacks a player
//! offers, against what the real `InventoryMerchant` of each game version shows
//! (`resetRecipeAndSlots`, `MerchantRecipeList.canRecipeBeUsed`).
//!
//! The fixtures come from `tests/javaref/MerchantRef.java`, run by
//! `tests/javaref/generate-merchant.ps1` against the deobfuscated 1.7.10 and 1.8.9 client
//! classes:
//!
//! - one offer at a time (a wool, a cooked-fish and a coal offer, a book-and-emeralds offer, an
//!   offer for damaged wool and one for a named sword), and the offer's item at damages 0-3, at
//!   three counts, with and without a tag, in either slot, with the second item present, missing
//!   or wrong: 1.7.10 compares the item and the count only, 1.8.9 also the damage and a tag
//!   subset;
//! - lists of three offers (one starting with a used-up offer), each offer selected and an index
//!   past the list: with the first offer selected both versions take the first offer of the
//!   whole list the inputs pay for, used up or not; with a later one, that offer alone.
//!
//! Each line also records what the output shows when the list holds only the selected offer,
//! the rule of a client that tests the selected offer alone, which is this client's 1.8.9 rule.
//!
//! The hand tests at the end follow a trade through the screen (the list as the server sends
//! it, page changes, a new list, the output click) and check that a 1.7.10 trade never uses an
//! offer up.

use std::path::PathBuf;

use mc_nbt::Nbt;
use mc_protocol::play::s2c;
use mc_protocol::PacketBuffer;
use mc_ui::screens::container::model::MerchantRecipe;
use mc_ui::screens::container::{
    container_for_open_window, v5, Container, ContainerArt, ContainerScreen, ItemStack,
};
use mc_version::GameVersion;

struct Case {
    line: usize,
    index: usize,
    offers: Vec<MerchantRecipe>,
    slots: [Option<ItemStack>; 2],
    /// What the version's real `InventoryMerchant` shows.
    output: Option<ItemStack>,
    /// What it shows with only the selected offer in the list.
    selected_only: Option<ItemStack>,
}

fn decode_hex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex"))
        .collect()
}

fn parse_stack(text: &str) -> Option<ItemStack> {
    if text == "-" {
        return None;
    }
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
    Some(ItemStack {
        item_id,
        count,
        damage,
        nbt,
    })
}

fn parse_offer(text: &str) -> MerchantRecipe {
    let fields: Vec<&str> = text.split(' ').collect();
    assert_eq!(fields.len(), 4, "{text}");
    MerchantRecipe {
        buy_a: parse_stack(fields[0]).expect("an offer buys something"),
        buy_b: parse_stack(fields[1]),
        sell: parse_stack(fields[2]).expect("an offer sells something"),
        uses: fields[3].parse().unwrap(),
        max_uses: 7,
        disabled: false,
    }
}

fn read_cases(name: &str) -> Vec<Case> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/merchant")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            let parts: Vec<&str> = line.splitn(3, " ; ").collect();
            let (slots, outputs) = parts[2].split_once(" => ").expect("slots => output");
            let (output, selected_only) = outputs.split_once(" ~ ").expect("output ~ selected");
            let slots: Vec<&str> = slots.split(' ').collect();
            Case {
                line: index + 1,
                index: parts[0].parse().unwrap(),
                offers: parts[1].split(" | ").map(parse_offer).collect(),
                slots: [parse_stack(slots[0]), parse_stack(slots[1])],
                output: parse_stack(output),
                selected_only: parse_stack(selected_only),
            }
        })
        .collect()
}

/// A villager window of `version`, as a session of that version opens one.
fn villager(version: GameVersion) -> Container {
    let packet = s2c::OpenWindow {
        window_id: 4,
        inventory_type: "minecraft:villager".to_owned(),
        window_title: r#"{"text":"Villager"}"#.to_owned(),
        slot_count: 3,
        entity_id: None,
    };
    match version {
        GameVersion::V1_7_10 => v5::container_for_open_window_v5(&packet),
        GameVersion::V1_8_9 => container_for_open_window(&packet),
    }
    .expect("a villager window")
}

/// The output slot once the window holds the case's offers as a session of `version` hands them
/// over (1.7.10: the whole list and the selected index; 1.8.9: the selected offer), then the two
/// inputs.
fn predict(version: GameVersion, case: &Case) -> Option<ItemStack> {
    let mut window = villager(version);
    match version {
        GameVersion::V1_7_10 => window.set_merchant_offers_v5(&case.offers, case.index),
        GameVersion::V1_8_9 => window.set_merchant_recipe(case.offers.get(case.index).cloned()),
    }
    window.set_slot_contents(0, case.slots[0].clone());
    window.set_slot_contents(1, case.slots[1].clone());
    window.slot_stack(2).cloned()
}

fn describe(case: &Case, got: &Option<ItemStack>, want: &Option<ItemStack>) -> String {
    format!(
        "line {}: index {} offers {:?} slots {:?}: java {want:?}, rust {got:?}",
        case.line, case.index, case.offers, case.slots
    )
}

#[test]
fn every_1_7_10_trade_shows_what_the_1_7_10_merchant_shows() {
    let cases = read_cases("merchant_1_7_10.txt");
    assert!(cases.len() > 600);
    let wrong: Vec<String> = cases
        .iter()
        .filter_map(|case| {
            let got = predict(GameVersion::V1_7_10, case);
            (got != case.output).then(|| describe(case, &got, &case.output))
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{} mismatches:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(10)].join("\n")
    );
}

/// java17 `MerchantRecipeList.canRecipeBeUsed` (`:28-41`): with the first offer selected, coal
/// on the first page buys the second offer's diamond; a used-up first offer that the inputs pay
/// for hides the next one that would take them.
#[test]
fn the_first_page_of_a_1_7_10_merchant_trades_any_offer() {
    let cases = read_cases("merchant_1_7_10.txt");
    let scanned: Vec<&Case> = cases
        .iter()
        .filter(|case| case.output != case.selected_only)
        .collect();
    assert!(scanned.len() >= 10, "{} cases", scanned.len());
    assert!(scanned
        .iter()
        .all(|case| case.offers.len() > 1 && (case.index == 0 || case.index >= case.offers.len())));
    let diamond_for_coal = scanned.iter().any(|case| {
        case.index == 0
            && case.slots[0].as_ref().is_some_and(|s| s.item_id == 263)
            && case.output.as_ref().is_some_and(|s| s.item_id == 264)
    });
    assert!(diamond_for_coal);
    let hidden = cases.iter().any(|case| {
        case.index == 0
            && case.offers[0].uses == 7
            && case.slots[0]
                .as_ref()
                .is_some_and(|s| s.item_id == 35 && s.count >= 16)
            && case.output.is_none()
    });
    assert!(hidden, "a used-up first offer hides the second wool offer");
}

/// The 1.8.9 control, documenting today's 1.8.9 client: it tests the selected offer alone, which
/// is java18's rule wherever an offer past the first is selected or the list has one offer. With
/// the first offer selected (or an index past the list), java18 scans the whole list; there the
/// client keeps its own rule, and the cases where the two differ are listed here.
#[test]
fn every_1_8_9_trade_shows_what_the_1_8_9_client_has_always_shown() {
    let cases = read_cases("merchant_1_8_9.txt");
    let mut wrong = Vec::new();
    let mut differs_from_java18 = 0;
    for case in &cases {
        let got = predict(GameVersion::V1_8_9, case);
        if got != case.selected_only {
            wrong.push(describe(case, &got, &case.selected_only));
        }
        if case.selected_only != case.output {
            differs_from_java18 += 1;
            assert!(
                case.offers.len() > 1 && (case.index == 0 || case.index >= case.offers.len()),
                "line {}: java18 and the selected offer differ outside the first-page scan",
                case.line
            );
        }
    }
    assert!(
        wrong.is_empty(),
        "{} mismatches:\n{}",
        wrong.len(),
        wrong[..wrong.len().min(10)].join("\n")
    );
    // The documented differences from java18 (the first-page scan), still exercised.
    assert!(differs_from_java18 >= 5, "{differs_from_java18}");
}

/// The negative control: with 1.8.9's matching rule, other wool colours and tagged stacks find
/// no trade that 1.7.10 makes.
#[test]
fn the_1_8_9_rule_refuses_trades_1_7_10_makes() {
    let cases = read_cases("merchant_1_7_10.txt");
    let differing = cases
        .iter()
        .filter(|case| case.offers.len() == 1)
        .filter(|case| predict(GameVersion::V1_8_9, case) != case.output)
        .count();
    assert!(differing > 100, "only {differing} differ");
    let orange_wool = cases
        .iter()
        .find(|case| {
            case.offers.len() == 1
                && case.offers[0].buy_a.item_id == 35
                && case.offers[0].buy_a.damage == 0
                && case.slots[0]
                    .as_ref()
                    .is_some_and(|s| s.damage == 1 && s.count >= 16)
                && case.slots[1].is_none()
        })
        .expect("orange wool for the white-wool offer");
    assert!(predict(GameVersion::V1_7_10, orange_wool).is_some());
    assert!(predict(GameVersion::V1_8_9, orange_wool).is_none());
}

fn item(item_id: i16, count: i32) -> ItemStack {
    ItemStack {
        item_id,
        count,
        damage: 0,
        nbt: None,
    }
}

fn offer(buy_a: ItemStack, buy_b: Option<ItemStack>, sell: ItemStack) -> MerchantRecipe {
    MerchantRecipe {
        buy_a,
        buy_b,
        sell,
        uses: 0,
        max_uses: 7,
        disabled: false,
    }
}

/// Wool for an emerald, coal for a diamond, a book and five emeralds for an enchanted book.
fn three_offers() -> Vec<MerchantRecipe> {
    vec![
        offer(item(35, 16), None, item(388, 1)),
        offer(item(263, 18), None, item(264, 1)),
        offer(item(340, 1), Some(item(388, 5)), item(403, 1)),
    ]
}

/// The `MC|TrList` payload a session hands the screen for window 4 (the 1.8.9 layout, which a
/// 1.7.10 session's list is lowered to).
fn offer_list(offers: &[MerchantRecipe]) -> Vec<u8> {
    let mut buf = PacketBuffer::new();
    buf.write_i32(4);
    buf.write_u8(u8::try_from(offers.len()).unwrap());
    for offer in offers {
        ItemStack::write(Some(&offer.buy_a), &mut buf);
        ItemStack::write(Some(&offer.sell), &mut buf);
        buf.write_bool(offer.buy_b.is_some());
        if let Some(second) = &offer.buy_b {
            ItemStack::write(Some(second), &mut buf);
        }
        buf.write_bool(offer.disabled);
        buf.write_i32(offer.uses);
        buf.write_i32(offer.max_uses);
    }
    buf.as_slice().to_vec()
}

/// A villager screen of `version` holding [`three_offers`], as the session opens it.
fn villager_screen(version: GameVersion) -> ContainerScreen {
    let packet = s2c::OpenWindow {
        window_id: 4,
        inventory_type: "minecraft:villager".to_owned(),
        window_title: r#"{"text":"Villager"}"#.to_owned(),
        slot_count: 3,
        entity_id: None,
    };
    let mut screen = match version {
        GameVersion::V1_7_10 => ContainerScreen::for_open_window_v5(&packet, "Villager"),
        GameVersion::V1_8_9 => ContainerScreen::for_open_window(&packet, "Villager"),
    }
    .expect("a villager screen");
    assert!(screen
        .apply_merchant_plugin(&offer_list(&three_offers()))
        .expect("an offer list"));
    screen
}

/// java17 `MerchantRecipeList.canRecipeBeUsed` through the screen: on the first page coal
/// shows the second offer's diamond and the output click carries it; on the third page the
/// third offer alone is tested and coal shows nothing; on the second, the diamond again.
#[test]
fn a_1_7_10_first_page_trades_the_later_offer_the_inputs_pay_for() {
    let mut screen = villager_screen(GameVersion::V1_7_10);
    screen.container.set_slot_contents(0, Some(item(263, 18)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
    assert!(screen.select_merchant_recipe(2));
    assert_eq!(screen.container.slot_stack(2), None);
    assert!(screen.select_merchant_recipe(1));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
    assert!(screen.select_merchant_recipe(0));
    assert_eq!(
        screen.container.slot_click(2, 0, 0, false),
        Some(item(264, 1)),
        "the click's clicked_item"
    );
    assert_eq!(screen.container.cursor(), Some(&item(264, 1)));
    assert_eq!(screen.container.slot_stack(0), None, "the coal paid");
    assert_eq!(screen.container.slot_stack(2), None);
}

/// The 1.8.9 control, documenting today's 1.8.9 client (java18 would scan the list here, as
/// 1.7.10 does): the first page tests the wool offer alone, so coal shows nothing and the click
/// carries nothing.
#[test]
fn a_1_8_9_first_page_tests_the_selected_offer_alone() {
    let mut screen = villager_screen(GameVersion::V1_8_9);
    screen.container.set_slot_contents(0, Some(item(263, 18)));
    assert_eq!(screen.container.slot_stack(2), None);
    assert_eq!(screen.container.slot_click(2, 0, 0, false), None);
    assert_eq!(screen.container.slot_stack(0), Some(&item(263, 18)));
    assert!(screen.select_merchant_recipe(1));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
}

/// java17's client merchant ignores `useRecipe` (`entity/NpcMerchant.java:39`): a 1.7.10 trade
/// never uses an offer up on the client, only the server's next list does. The 1.8.9 control
/// (java18 `NpcMerchant.useRecipe` calls `incrementToolUses`) loses the offer after its last use.
#[test]
fn a_1_7_10_trade_never_uses_an_offer_up_on_the_client() {
    let wool = offer(item(35, 16), None, item(388, 1));
    let wool = MerchantRecipe {
        max_uses: 2,
        ..wool
    };
    for version in [GameVersion::V1_7_10, GameVersion::V1_8_9] {
        let mut window = villager(version);
        match version {
            GameVersion::V1_7_10 => window.set_merchant_offers_v5(std::slice::from_ref(&wool), 0),
            GameVersion::V1_8_9 => window.set_merchant_recipe(Some(wool.clone())),
        }
        for _ in 0..2 {
            window.set_slot_contents(0, Some(item(35, 16)));
            assert_eq!(window.slot_click(2, 0, 0, false), Some(item(388, 1)));
        }
        assert_eq!(window.cursor(), Some(&item(388, 2)));
        window.set_slot_contents(0, Some(item(35, 16)));
        let after = window.slot_stack(2).cloned();
        match version {
            GameVersion::V1_7_10 => assert_eq!(after, Some(item(388, 1)), "{version:?}"),
            GameVersion::V1_8_9 => assert_eq!(after, None, "{version:?}"),
        }
    }
}

/// The page the merchant screen shows.
fn page(screen: &ContainerScreen) -> usize {
    match &screen.art {
        ContainerArt::Merchant(merchant) => merchant.selected,
        _ => panic!("a merchant screen"),
    }
}

/// [`three_offers`] with the coal offer wanting 20 coal.
fn repriced_offers() -> Vec<MerchantRecipe> {
    let mut offers = three_offers();
    offers[1].buy_a.count = 20;
    offers
}

/// java17 only stores a new `MC|TrList` (`NetHandlerPlayClient.handleCustomPayload:1407-1421`
/// -> `NpcMerchant.setRecipes`): the page, the output and the offer the output came from stay
/// until the next slot or page change. The server reprices the coal offer to 20 while 18 coal
/// sit on the second page: the diamond still shows and its click pays the old 18; then wool on
/// the kept page (the coal offer alone) shows nothing, and 20 coal buys the diamond.
#[test]
fn a_1_7_10_offer_list_keeps_the_page_the_output_and_its_offer() {
    let mut screen = villager_screen(GameVersion::V1_7_10);
    assert!(screen.select_merchant_recipe(1));
    assert_eq!(
        screen.take_plugin_outgoing().len(),
        1,
        "the page's MC|TrSel"
    );
    screen.container.set_slot_contents(0, Some(item(263, 18)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
    assert!(screen
        .apply_merchant_plugin(&offer_list(&repriced_offers()))
        .expect("an offer list"));
    assert_eq!(page(&screen), 1);
    assert!(screen.take_plugin_outgoing().is_empty(), "no MC|TrSel");
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
    assert_eq!(
        screen.container.slot_click(2, 0, 0, false),
        Some(item(264, 1))
    );
    assert_eq!(screen.container.slot_stack(0), None, "the old 18 coal paid");
    screen.container.set_slot_contents(0, Some(item(35, 16)));
    assert_eq!(screen.container.slot_stack(2), None, "the coal offer alone");
    screen.container.set_slot_contents(0, Some(item(263, 20)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
}

/// A page past a shorter new list becomes the first: java17 keeps that index and its screen
/// would throw (`GuiMerchant.drawGuiContainerBackgroundLayer:117-118`), and the first page
/// matches the same offers as that index, the whole list (`MerchantRecipeList.canRecipeBeUsed:23`).
#[test]
fn a_1_7_10_page_past_a_shorter_list_becomes_the_first() {
    let mut screen = villager_screen(GameVersion::V1_7_10);
    assert!(screen.select_merchant_recipe(2));
    assert!(screen
        .apply_merchant_plugin(&offer_list(&three_offers()[..2]))
        .expect("an offer list"));
    assert_eq!(page(&screen), 0);
    screen.container.set_slot_contents(0, Some(item(263, 18)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
}

/// The 1.8.9 control, documenting today's 1.8.9 client (java18 also only stores the list,
/// `NpcMerchant.setRecipes:44-47`, and keeps the page): a new list goes back to the first page
/// and recomputes the output there, so the coal on the second page loses its diamond and wool
/// (the first offer) shows the emerald.
#[test]
fn a_1_8_9_offer_list_goes_back_to_the_first_page() {
    let mut screen = villager_screen(GameVersion::V1_8_9);
    assert!(screen.select_merchant_recipe(1));
    screen.container.set_slot_contents(0, Some(item(263, 18)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(264, 1)));
    assert!(screen
        .apply_merchant_plugin(&offer_list(&repriced_offers()))
        .expect("an offer list"));
    assert_eq!(page(&screen), 0);
    assert_eq!(screen.container.slot_stack(2), None);
    screen.container.set_slot_contents(0, Some(item(35, 16)));
    assert_eq!(screen.container.slot_stack(2), Some(&item(388, 1)));
}
