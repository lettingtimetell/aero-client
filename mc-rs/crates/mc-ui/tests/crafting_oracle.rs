//! The crafting oracle: the result a container predicts for a crafting grid, against what the
//! real `CraftingManager.findMatchingRecipe` of each game version returns for it.
//!
//! The fixtures come from `tests/javaref/CraftingRef.java`, run against the deobfuscated 1.7.10
//! and 1.8.9 client classes by `tests/javaref/generate-crafting.ps1`: every recipe of both
//! versions laid into 2x2 and 3x3 grids in several places, mirrored and with other damages for
//! the recipes that take any, a mutation of each of those grids (an item changed, removed, added
//! or tagged), the special recipes with their tags (dyed leather, fireworks, written books, maps,
//! repairs), and random grids. A container of the version must show exactly Java's result in its
//! result slot: item, count, damage and tag. Where the 1.7.10 manager throws, a 1.7.10 container
//! shows no result.

use std::path::PathBuf;

use mc_nbt::Nbt;
use mc_ui::screens::container::{layouts, Container, ItemStack};
use mc_version::GameVersion;

/// What the Java manager returned for a grid.
#[derive(Debug, Clone, PartialEq)]
enum Expected {
    Result(Option<ItemStack>),
    Throws(String),
}

#[derive(Debug, Clone)]
struct Case {
    line: usize,
    side: u8,
    slots: Vec<Option<ItemStack>>,
    expected: Expected,
}

struct Fixture {
    /// Map ids the Java world held data for, with their scales.
    maps: Vec<(i32, i8)>,
    cases: Vec<Case>,
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
    let nbt = parts.next().map(|hex| {
        let (_, tag) = mc_nbt::from_slice(&decode_hex(hex)).expect("an NBT tag");
        match tag {
            Nbt::Compound(compound) => compound,
            other => panic!("a compound tag, got {other:?}"),
        }
    });
    Some(ItemStack {
        item_id,
        count,
        damage,
        nbt,
    })
}

fn read_fixture(name: &str) -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/crafting")
        .join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = text.lines().enumerate();
    let (_, header) = lines.next().expect("a MAPS line");
    let maps = header
        .strip_prefix("MAPS")
        .expect("a MAPS line")
        .split_whitespace()
        .map(|pair| {
            let (id, scale) = pair.split_once(':').unwrap();
            (id.parse().unwrap(), scale.parse().unwrap())
        })
        .collect();
    let cases = lines
        .map(|(index, line)| {
            let (grid, result) = line.split_once(" => ").expect("grid => result");
            let mut fields = grid.split(' ');
            let side: u8 = fields.next().unwrap().parse().unwrap();
            let height: u8 = fields.next().unwrap().parse().unwrap();
            assert_eq!(side, height, "square grids only");
            let slots: Vec<_> = fields.map(parse_stack).collect();
            assert_eq!(slots.len(), usize::from(side * side), "line {}", index + 1);
            let expected = match result.strip_prefix("THROWS ") {
                Some(exception) => Expected::Throws(exception.to_owned()),
                None => Expected::Result(parse_stack(result)),
            };
            Case {
                line: index + 1,
                side,
                slots,
                expected,
            }
        })
        .collect();
    Fixture { maps, cases }
}

/// The result slot of a container of `version` holding `case`'s grid: the player's own 2x2
/// grid or a crafting table's 3x3 one, filled slot by slot as the server's packets would.
fn predict(version: GameVersion, maps: &[(i32, i8)], case: &Case) -> Option<ItemStack> {
    let container: Container = match case.side {
        2 => layouts::player_container(0),
        _ => layouts::crafting_table_container(1),
    };
    let mut container = container.with_game_version(version);
    for &(id, scale) in maps {
        container.set_map_scale(id, Some(scale));
    }
    for (index, stack) in case.slots.iter().enumerate() {
        container.set_slot_contents(1 + index, stack.clone());
    }
    container.slot_stack(0).cloned()
}

/// What a container of `version` must show for `case`: Java's result, or nothing where Java
/// throws.
fn wanted(case: &Case) -> Option<ItemStack> {
    match &case.expected {
        Expected::Result(stack) => stack.clone(),
        Expected::Throws(_) => None,
    }
}

/// The cases whose prediction differs from Java's, with the prediction.
fn mismatches(version: GameVersion, fixture: &Fixture) -> Vec<(&Case, Option<ItemStack>)> {
    fixture
        .cases
        .iter()
        .filter_map(|case| {
            let got = predict(version, &fixture.maps, case);
            (got != wanted(case)).then_some((case, got))
        })
        .collect()
}

fn report(mismatches: &[(&Case, Option<ItemStack>)]) -> String {
    let shown: Vec<_> = mismatches
        .iter()
        .take(20)
        .map(|(case, got)| {
            format!(
                "line {}: grid {:?}\n    java {:?}\n    rust {:?}",
                case.line, case.slots, case.expected, got
            )
        })
        .collect();
    format!(
        "{} mismatches, first:\n{}",
        mismatches.len(),
        shown.join("\n")
    )
}

/// The three ways this client's 1.8.9 prediction differs from java18 today, on grids no vanilla
/// item set makes. They are 1.8.9 behaviour and stay as they are here; each is named by the
/// java18 rule the prediction misses.
fn known_1_8_9_difference(case: &Case, got: &Option<ItemStack>) -> Option<&'static str> {
    let has = |item_id: i16, rule: &dyn Fn(&ItemStack) -> bool| {
        case.slots
            .iter()
            .flatten()
            .any(|stack| stack.item_id == item_id && rule(stack))
    };
    match (&case.expected, got) {
        // `RecipeFireworks.matches`: a fade's result is a copy of the charge, damage included.
        (Expected::Result(Some(want)), Some(got))
            if want.item_id == 402
                && want.damage != 0
                && *got
                    == (ItemStack {
                        damage: 0,
                        ..want.clone()
                    }) =>
        {
            Some("a fade keeps the charge's damage")
        }
        // `RecipesMapExtending`: the pattern's paper has damage 0.
        (Expected::Result(None), Some(got))
            if got.item_id == 358 && has(339, &|paper| paper.damage != 0) =>
        {
            Some("map extending wants paper of damage 0")
        }
        // `RecipeBookCloning.getCraftingResult`: `getTagCompound().copy()` throws without a tag.
        (Expected::Throws(thrown), Some(got))
            if thrown == "NullPointerException"
                && got.item_id == 387
                && has(387, &|book| book.nbt.is_none()) =>
        {
            Some("a written book without a tag throws")
        }
        _ => None,
    }
}

#[test]
fn every_1_7_10_grid_crafts_what_the_1_7_10_crafting_manager_returns() {
    let fixture = read_fixture("crafting_1_7_10.txt");
    assert!(fixture.cases.len() > 5000, "the whole fixture");
    let wrong = mismatches(GameVersion::V1_7_10, &fixture);
    assert!(wrong.is_empty(), "{}", report(&wrong));
}

/// 1.8.9 mode matches java18 on every grid except three known 1.8.9 differences (see
/// [`known_1_8_9_difference`]), each of which the fixture still exercises.
#[test]
fn every_1_8_9_grid_crafts_what_the_1_8_9_crafting_manager_returns() {
    let fixture = read_fixture("crafting_1_8_9.txt");
    assert!(fixture.cases.len() > 5000, "the whole fixture");
    let wrong = mismatches(GameVersion::V1_8_9, &fixture);
    let (known, unknown): (Vec<_>, Vec<_>) = wrong
        .into_iter()
        .partition(|(case, got)| known_1_8_9_difference(case, got).is_some());
    assert!(unknown.is_empty(), "{}", report(&unknown));
    let kinds: std::collections::BTreeSet<_> = known
        .iter()
        .filter_map(|(case, got)| known_1_8_9_difference(case, got))
        .collect();
    assert_eq!(kinds.len(), 3, "the known differences seen: {kinds:?}");
}

/// The negative control: 1.8.9's rules miss the 1.7.10 results on hundreds of grids, the door
/// among them, so the 1.7.10 test above tells the two apart.
#[test]
fn the_1_8_9_rules_miss_the_1_7_10_results() {
    let fixture = read_fixture("crafting_1_7_10.txt");
    let wrong = mismatches(GameVersion::V1_8_9, &fixture);
    assert!(wrong.len() > 300, "only {} grids differ", wrong.len());
    let planks = Some(ItemStack {
        item_id: 5,
        count: 1,
        damage: 0,
        nbt: None,
    });
    let door_grid = Case {
        line: 0,
        side: 3,
        slots: (0..9)
            .map(|slot| (slot % 3 != 2).then(|| planks.clone()).flatten())
            .collect(),
        expected: Expected::Result(None),
    };
    let door =
        |version| predict(version, &[], &door_grid).map(|stack| (stack.item_id, stack.count));
    assert_eq!(door(GameVersion::V1_7_10), Some((324, 1)));
    assert_eq!(door(GameVersion::V1_8_9), Some((324, 3)));
}

/// The fixtures exercise what they claim: special recipes with tags, results with tags, grids
/// where the 1.7.10 manager throws, and both grid sizes.
#[test]
fn the_fixtures_cover_tags_throws_and_both_grid_sizes() {
    for (name, throws) in [("crafting_1_7_10.txt", 10), ("crafting_1_8_9.txt", 1)] {
        let fixture = read_fixture(name);
        let tagged_results = fixture
            .cases
            .iter()
            .filter(|case| {
                matches!(&case.expected, Expected::Result(Some(stack)) if stack.nbt.is_some())
            })
            .count();
        let thrown = fixture
            .cases
            .iter()
            .filter(|case| matches!(case.expected, Expected::Throws(_)))
            .count();
        let small = fixture.cases.iter().filter(|case| case.side == 2).count();
        assert!(
            tagged_results > 300,
            "{name}: {tagged_results} tagged results"
        );
        assert!(thrown >= throws, "{name}: {thrown} throws");
        assert!(small > 500, "{name}: {small} 2x2 grids");
    }
}

/// java17 `ContainerWorkbench.onCraftMatrixChanged` always writes the manager's result, nothing
/// included, so a result a server wrote into the slot goes at the next grid change that matches
/// nothing. 1.8.9 mode keeps such a result until a recipe of its own matches.
#[test]
fn a_1_7_10_grid_change_clears_a_result_the_server_wrote() {
    let dirt = |count| {
        Some(ItemStack {
            item_id: 3,
            count,
            damage: 0,
            nbt: None,
        })
    };
    let diamond = ItemStack {
        item_id: 264,
        count: 1,
        damage: 0,
        nbt: None,
    };
    for (version, kept) in [
        (GameVersion::V1_7_10, None),
        (GameVersion::V1_8_9, Some(&diamond)),
    ] {
        let mut table = layouts::crafting_table_container(1).with_game_version(version);
        table.set_slot_contents(1, dirt(1));
        table.set_slot_contents(0, Some(diamond.clone()));
        assert_eq!(table.slot_stack(0), Some(&diamond), "{version:?}: written");
        table.set_slot_contents(5, dirt(2));
        assert_eq!(
            table.slot_stack(0),
            kept,
            "{version:?}: after a grid change"
        );
    }
}
