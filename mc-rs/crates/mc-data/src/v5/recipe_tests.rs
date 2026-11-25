use super::*;
use crate::recipe::{
    find_matching_vanilla_recipe_dynamic, DynamicRecipeEvaluation, DynamicSlotData,
};

fn stack(item: u16, damage: i16) -> Option<RecipeStack> {
    Some(RecipeStack::with_damage(item, 1, damage))
}

/// The 1.7.10 result for a 3x3 grid whose stacks carry no tag.
fn craft3(slots: [Option<RecipeStack>; 9]) -> Option<RecipeResultV5> {
    let grid = CraftingGrid::new(3, 3, &slots).unwrap();
    find_matching_recipe_v5(grid, &[SlotTagV5::default(); 9])
}

/// The 1.7.10 result for a 3x3 grid with the given slot tags.
fn craft3_tagged(slots: [Option<RecipeStack>; 9], tags: [SlotTagV5; 9]) -> Option<RecipeResultV5> {
    let grid = CraftingGrid::new(3, 3, &slots).unwrap();
    find_matching_recipe_v5(grid, &tags)
}

/// The 1.8.9 result for the same tagless 3x3 grid (the negative control).
fn craft3_1_8_9(slots: [Option<RecipeStack>; 9]) -> Option<RecipeStack> {
    let grid = CraftingGrid::new(3, 3, &slots).unwrap();
    match find_matching_vanilla_recipe_dynamic(grid, &[DynamicSlotData::default(); 9]) {
        DynamicRecipeEvaluation::Match(Some(result)) => Some(result.stack),
        _ => None,
    }
}

fn plain_stack(result: Option<RecipeResultV5>) -> Option<RecipeStack> {
    let result = result?;
    assert_eq!(
        result.transform,
        RecipeTransformV5::Shared(DynamicRecipeTransform::Plain),
        "a plain result"
    );
    Some(result.stack)
}

#[test]
fn the_list_is_1_7_10s_with_its_five_special_recipes_in_runtime_order() {
    let recipes = recipes_v5();
    assert_eq!(recipes.len(), 317);
    let dynamic: Vec<(usize, DynamicRecipeKind)> = recipes
        .iter()
        .enumerate()
        .filter_map(|(index, recipe)| match recipe {
            RecipeV5::Dynamic(kind) => Some((index, *kind)),
            _ => None,
        })
        .collect();
    assert_eq!(
        dynamic,
        [
            (0, DynamicRecipeKind::ArmorDyes),
            (1, DynamicRecipeKind::Fireworks),
            (65, DynamicRecipeKind::BookCloning),
            (66, DynamicRecipeKind::MapCloning),
            (67, DynamicRecipeKind::MapExtending),
        ]
    );
    let copies: Vec<(usize, RecipeStack)> = recipes
        .iter()
        .enumerate()
        .filter_map(|(index, recipe)| match recipe {
            RecipeV5::CopiesIngredientTag(recipe) => Some((index, recipe.output())),
            _ => None,
        })
        .collect();
    assert_eq!(
        copies,
        [(178, RecipeStack::new(398, 1))],
        "the carrot on a stick"
    );
    // Every shaped recipe precedes every shapeless one (the special recipes sort by size alone).
    let first_shapeless = recipes
        .iter()
        .position(|recipe| matches!(recipe, RecipeV5::Static(StaticRecipe::Shapeless(_))))
        .unwrap();
    assert!(recipes[first_shapeless..]
        .iter()
        .all(|recipe| !matches!(recipe, RecipeV5::Static(StaticRecipe::Shaped(_)))));
    assert_eq!(
        recipes
            .iter()
            .filter(|recipe| matches!(recipe, RecipeV5::Static(StaticRecipe::Shapeless(_))))
            .count(),
        54
    );
}

#[test]
fn a_wooden_door_is_one_door_from_planks_of_any_wood() {
    let planks = |damage| [stack(5, damage), stack(5, damage), None].repeat(3);
    for damage in [0, 1, 5] {
        let slots: [Option<RecipeStack>; 9] = planks(damage).try_into().unwrap();
        assert_eq!(
            plain_stack(craft3(slots)),
            Some(RecipeStack::new(324, 1)),
            "planks {damage}"
        );
    }
    // 1.8.9: three doors, and a spruce door from spruce planks.
    let oak: [Option<RecipeStack>; 9] = planks(0).try_into().unwrap();
    let spruce: [Option<RecipeStack>; 9] = planks(1).try_into().unwrap();
    assert_eq!(craft3_1_8_9(oak), Some(RecipeStack::new(324, 3)));
    assert_eq!(craft3_1_8_9(spruce), Some(RecipeStack::new(427, 3)));
}

#[test]
fn shared_recipes_take_1_7_10_outputs_and_ingredients() {
    let ingot = stack(265, 0);
    let iron_door = [ingot, ingot, None, ingot, ingot, None, ingot, ingot, None];
    assert_eq!(
        plain_stack(craft3(iron_door)),
        Some(RecipeStack::new(330, 1))
    );
    assert_eq!(craft3_1_8_9(iron_door), Some(RecipeStack::new(330, 3)));

    // Six sticks make two fences.
    let stick = stack(280, 0);
    let fence = [stick, stick, stick, stick, stick, stick, None, None, None];
    assert_eq!(plain_stack(craft3(fence)), Some(RecipeStack::new(85, 2)));
    assert_eq!(craft3_1_8_9(fence), None);

    // Any stone makes stone bricks; 1.8.9 wants plain stone.
    let granite = stack(1, 1);
    let bricks = [
        granite, granite, None, granite, granite, None, None, None, None,
    ];
    assert_eq!(plain_stack(craft3(bricks)), Some(RecipeStack::new(98, 4)));
    assert_eq!(
        craft3_1_8_9(bricks),
        Some(RecipeStack::with_damage(1, 4, 2))
    );

    // Chainmail from fire, a recipe only 1.7.10 has.
    let fire = stack(51, 0);
    let helmet = [fire, fire, fire, fire, None, fire, None, None, None];
    assert_eq!(plain_stack(craft3(helmet)), Some(RecipeStack::new(302, 1)));
}

#[test]
fn two_damaged_tools_combine_before_the_list_is_tried() {
    // Iron pickaxes: 250 durability, a five percent bonus of 12.
    let slots = [
        stack(257, 200),
        None,
        None,
        None,
        stack(257, 150),
        None,
        None,
        None,
        None,
    ];
    // 50 + 100 + 12 of 250 left: damage 88.
    assert_eq!(
        plain_stack(craft3(slots)),
        Some(RecipeStack::with_damage(257, 1, 88))
    );
    // Two items, not one stack of two; the same item; an item with a durability.
    let mut two = slots;
    two[0] = Some(RecipeStack::with_damage(257, 2, 200));
    assert_eq!(craft3(two), None);
    let mixed = [
        stack(257, 200),
        stack(278, 150),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    assert_eq!(craft3(mixed), None);
    let dyes = [
        stack(351, 1),
        stack(351, 1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    assert_eq!(craft3(dyes), None);
}

#[test]
fn a_cloned_book_is_one_more_copy_of_the_whole_tag() {
    let mut tags = [SlotTagV5::default(); 9];
    tags[4].has_tag = true;
    let slots = [
        None,
        stack(386, 0),
        None,
        None,
        stack(387, 3),
        None,
        stack(386, 0),
        None,
        None,
    ];
    assert_eq!(
        craft3_tagged(slots, tags),
        Some(RecipeResultV5 {
            stack: RecipeStack::new(387, 3),
            transform: RecipeTransformV5::CopyTag { source_slot: 4 },
        })
    );
    // java17 throws on a written book without a tag: no result.
    assert_eq!(craft3(slots), None);
    // 1.8.9 makes one copy per blank book.
    assert_eq!(craft3_1_8_9(slots), Some(RecipeStack::new(387, 2)));
}

#[test]
fn firework_dyes_past_fifteen_throw_where_1_8_9_wraps_them() {
    let star = [
        stack(289, 0),
        stack(351, 16),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    assert_eq!(craft3(star), None, "ItemDye.dyeColors[16] throws");
    assert_eq!(craft3_1_8_9(star), Some(RecipeStack::new(402, 1)));
    let mut fine = star;
    fine[1] = stack(351, 15);
    let Some(result) = craft3(fine) else {
        panic!("a white firework star");
    };
    assert_eq!(
        result.transform,
        RecipeTransformV5::Shared(DynamicRecipeTransform::FireworkStar {
            colors: vec![15_790_320],
            flicker: false,
            trail: false,
            shape: 0,
        })
    );
}

#[test]
fn a_fade_needs_a_tagged_charge_and_throws_on_a_non_compound_explosion() {
    let slots = [
        stack(402, 0),
        stack(351, 1),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    assert_eq!(craft3(slots), None, "a charge without a tag does not match");
    let mut tags = [SlotTagV5::default(); 9];
    tags[0].has_tag = true;
    let fade = |explosion_exists| {
        Some(RecipeResultV5 {
            stack: RecipeStack::new(402, 1),
            transform: RecipeTransformV5::Shared(DynamicRecipeTransform::FireworkFade {
                source_slot: 0,
                colors: vec![11_743_532],
                explosion_exists,
            }),
        })
    };
    assert_eq!(craft3_tagged(slots, tags), fade(false));
    tags[0].explosion = TagEntryV5::Compound;
    assert_eq!(craft3_tagged(slots, tags), fade(true));
    tags[0].explosion = TagEntryV5::NotCompound;
    assert_eq!(craft3_tagged(slots, tags), None);
    // The result is a copy of the charge: its damage stays.
    let mut damaged = slots;
    damaged[0] = stack(402, 2);
    tags[0].explosion = TagEntryV5::Absent;
    assert_eq!(
        craft3_tagged(damaged, tags).map(|result| result.stack),
        Some(RecipeStack::with_damage(402, 1, 2))
    );
}

#[test]
fn a_dye_past_fifteen_dyes_leather_by_its_low_bits() {
    let dye = |damage| {
        let slots = [
            stack(299, 0),
            stack(351, damage),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ];
        match craft3(slots).map(|result| result.transform) {
            Some(RecipeTransformV5::Shared(DynamicRecipeTransform::DyeLeather {
                source_slot: 0,
                color,
            })) => color,
            other => panic!("dyed leather, got {other:?}"),
        }
    };
    // Dye 1 is red: 0.6, 0.2, 0.2 of 255.
    assert_eq!(dye(1), (153 << 16) | (51 << 8) | 51);
    assert_eq!(dye(17), dye(1), "~17 & 15 is red's wool colour");
    // 1.8.9 reads damage 17 as black.
    let slots = [
        stack(299, 0),
        stack(351, 17),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let grid = CraftingGrid::new(3, 3, &slots).unwrap();
    let DynamicRecipeEvaluation::Match(Some(result)) =
        find_matching_vanilla_recipe_dynamic(grid, &[DynamicSlotData::default(); 9])
    else {
        panic!("1.8.9 dyes it too");
    };
    assert_ne!(
        result.transform,
        DynamicRecipeTransform::DyeLeather {
            source_slot: 0,
            color: dye(17)
        }
    );
    // `ItemArmor.func_82813_b` throws on a `display` that is not a compound.
    let mut tags = [SlotTagV5::default(); 9];
    tags[0].has_tag = true;
    tags[0].display_not_compound = true;
    assert_eq!(craft3_tagged(slots, tags), None);
}

#[test]
fn the_carrot_on_a_stick_takes_the_last_tag_in_the_grid() {
    // The recipe wants an undamaged rod (`new ItemStack(Items.fishing_rod)`).
    let damaged = [
        stack(346, 5),
        None,
        None,
        None,
        stack(391, 0),
        None,
        None,
        None,
        None,
    ];
    assert_eq!(craft3(damaged), None);
    let slots = [
        stack(346, 0),
        None,
        None,
        None,
        stack(391, 0),
        None,
        None,
        None,
        None,
    ];
    assert_eq!(plain_stack(craft3(slots)), Some(RecipeStack::new(398, 1)));
    let mut tags = [SlotTagV5::default(); 9];
    tags[0].has_tag = true;
    let copy = |source_slot| {
        Some(RecipeResultV5 {
            stack: RecipeStack::new(398, 1),
            transform: RecipeTransformV5::CopyTag { source_slot },
        })
    };
    assert_eq!(craft3_tagged(slots, tags), copy(0));
    tags[4].has_tag = true;
    assert_eq!(craft3_tagged(slots, tags), copy(4));
}

#[test]
fn map_extending_needs_its_exact_pattern_and_a_known_scale_below_four() {
    let paper = stack(339, 0);
    let mut slots = [paper; 9];
    slots[4] = stack(358, 7);
    let mut tags = [SlotTagV5::default(); 9];
    assert_eq!(craft3_tagged(slots, tags), None, "no map data");
    tags[4].map_scale = Some(4);
    assert_eq!(craft3_tagged(slots, tags), None, "already the largest");
    tags[4].map_scale = Some(3);
    assert_eq!(
        craft3_tagged(slots, tags),
        Some(RecipeResultV5 {
            stack: RecipeStack::with_damage(358, 1, 7),
            transform: RecipeTransformV5::Shared(DynamicRecipeTransform::ScaleMap {
                source_slot: 4
            }),
        })
    );
    slots[0] = stack(339, 1);
    assert_eq!(
        craft3_tagged(slots, tags),
        None,
        "paper of damage 1 is not the pattern's"
    );
}

#[test]
fn a_birch_log_in_a_two_by_two_grid_makes_birch_planks() {
    let log = Some(RecipeStack::with_damage(17, 1, 2));
    let slots = [None, None, log, None];
    let grid = CraftingGrid::new(2, 2, &slots).unwrap();
    assert_eq!(
        plain_stack(find_matching_recipe_v5(grid, &[SlotTagV5::default(); 4])),
        Some(RecipeStack::with_damage(5, 4, 2))
    );
}
