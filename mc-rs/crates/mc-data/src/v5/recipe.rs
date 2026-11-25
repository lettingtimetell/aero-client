//! The 1.7.10 crafting manager (java17: `item/crafting/CraftingManager.java`): its recipe list and
//! `findMatchingRecipe`, which a 1.7.10 client runs on every change of a crafting grid. The
//! predicted result is what the next click on the result slot sends as `clicked_item`, so it has
//! to be 1.7.10's own result, not 1.8.9's.
//!
//! What 1.7.10 does differently from [`crate::find_matching_vanilla_recipe_dynamic`]:
//!
//! - **Its own list** ([`recipes_v5`], generated from the 1.7.10 classes): 317 recipes against
//!   1.8.9's 373. Shared recipes can differ in their output (one iron door, not three) or their
//!   ingredients (stone of any damage makes stone bricks).
//! - **Repair first**: two single damageable items of one kind are combined before the list is
//!   tried at all (java17: `CraftingManager.findMatchingRecipe:277-316`), where 1.8.9 sorts a
//!   `RecipeRepairItem` into its list. The list has no banner recipes either.
//! - **Book cloning** makes one book more than the blank books used and copies the written
//!   book's tag unchanged, with no `generation` (java17: `RecipeBookCloning`).
//! - **Fireworks** read `ItemDye.dyeColors[damage]` without 1.8.9's `& 15` (java17:
//!   `RecipeFireworks.matches:132,186`).
//! - **Leather dyes** take a dye's colour from `EntitySheep.fleeceColorTable[~damage & 15]`
//!   (java17: `RecipesArmorDyes.getCraftingResult:104`), so a damage past 15 counts by its low
//!   four bits where 1.8.9 reads it as black.
//! - **The carrot on a stick** takes the tag of an ingredient (java17:
//!   `ShapedRecipes.func_92100_c`, set on that one recipe by `CraftingManager.<init>:116`).
//!
//! Where vanilla 1.7.10 throws while it evaluates a grid (a written book without a tag, a
//! firework dye damage past 15, a `display` or `Explosion` entry that is not a compound), the
//! client predicts no result: 1.7.10 never computes one for such a grid.

use crate::recipe::{
    CraftingGrid, DynamicRecipeKind, DynamicRecipeTransform, Ingredient, RecipeStack, ShapedRecipe,
    ShapelessRecipe, StaticRecipe,
};

use super::item::try_item_props_v5;

include!("recipe_table.rs");

/// One entry of the 1.7.10 crafting manager's list.
#[derive(Debug, Clone, Copy)]
pub enum RecipeV5 {
    /// A shaped or shapeless recipe whose result is a copy of its output.
    Static(StaticRecipe),
    /// A shaped recipe whose result also takes the tag of the last ingredient that has one
    /// (java17: `ShapedRecipes.getCraftingResult` with `func_92100_c` set).
    CopiesIngredientTag(ShapedRecipe),
    /// One of 1.7.10's five special recipes: armour dyes, fireworks, book cloning, map cloning
    /// and map extending. [`find_matching_recipe_v5`] evaluates them the 1.7.10 way.
    Dynamic(DynamicRecipeKind),
}

/// What 1.7.10's special recipes read from the stack in one grid slot besides its item, count
/// and damage. The container builds one per slot from the stack's tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotTagV5 {
    /// `ItemStack.hasTagCompound()`.
    pub has_tag: bool,
    /// `ItemArmor.getColor` where `ItemArmor.hasColor` holds: an int `display.color` inside a
    /// compound `display`.
    pub leather_color: Option<u32>,
    /// The tag has a `display` entry that is not a compound, on which 1.7.10's
    /// `NBTTagCompound.getCompoundTag` throws.
    pub display_not_compound: bool,
    /// The tag's `Explosion` entry.
    pub explosion: TagEntryV5,
    /// The client's `MapData.scale` for this filled map, `None` when the client has no map data
    /// for it (java17: `ItemMap.getMapData` on a client world).
    pub map_scale: Option<i8>,
}

/// How a tag holds one named entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TagEntryV5 {
    /// No entry of that name (or no tag).
    #[default]
    Absent,
    /// A compound entry.
    Compound,
    /// An entry of another type, on which 1.7.10's `getCompoundTag` throws.
    NotCompound,
}

/// The tag operation that turns a [`RecipeResultV5`]'s stack into the crafted item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecipeTransformV5 {
    /// An operation 1.7.10 performs exactly as 1.8.9 does: the plain copy of a recipe's output,
    /// or the tag a shared special recipe writes (dyed leather, fireworks, map cloning and
    /// extending).
    Shared(DynamicRecipeTransform),
    /// Give the result a copy of the whole tag of grid slot `source_slot`.
    CopyTag {
        /// The grid slot whose tag is copied.
        source_slot: usize,
    },
}

/// The result 1.7.10's `findMatchingRecipe` returns for a grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecipeResultV5 {
    /// Item, count and damage of the result.
    pub stack: RecipeStack,
    /// What the result's tag is.
    pub transform: RecipeTransformV5,
}

/// One recipe's verdict on a grid.
enum Evaluation {
    /// The recipe does not match; the search goes on.
    NoMatch,
    /// The recipe matches with this result; the search stops.
    Result(RecipeResultV5),
    /// 1.7.10 throws while evaluating the recipe; the search stops without a result.
    Throws,
}

/// The 1.7.10 crafting manager's recipe list in its runtime order (java17:
/// `CraftingManager.<init>`, which sorts the list once: shaped before shapeless, larger before
/// smaller, registration order kept for ties).
#[inline]
#[must_use]
pub fn recipes_v5() -> &'static [RecipeV5] {
    &RECIPES_V5
}

/// java17: `CraftingManager.findMatchingRecipe` (`item/crafting/CraftingManager.java:275-331`):
/// the repair of two damaged items first, then the first recipe of [`recipes_v5`] that matches.
///
/// `tags` has one entry per grid slot, in slot order. `None` when nothing matches, and also
/// where 1.7.10 throws while evaluating the grid (see the module documentation). A `tags` of
/// another length is a caller's mistake: debug builds stop on it, release builds find nothing
/// rather than read past the grid.
#[must_use]
pub fn find_matching_recipe_v5(
    grid: CraftingGrid<'_>,
    tags: &[SlotTagV5],
) -> Option<RecipeResultV5> {
    debug_assert_eq!(tags.len(), grid.len(), "one tag entry per grid slot");
    if tags.len() != grid.len() {
        return None;
    }
    if let Some(stack) = repair_result_v5(grid) {
        return Some(plain(stack));
    }
    for recipe in recipes_v5() {
        let evaluation = match recipe {
            RecipeV5::Static(recipe) if recipe.matches(grid) => {
                Evaluation::Result(plain(recipe.output()))
            }
            RecipeV5::CopiesIngredientTag(recipe) if recipe.matches(grid) => {
                Evaluation::Result(copies_ingredient_tag(recipe.output(), grid, tags))
            }
            RecipeV5::Static(_) | RecipeV5::CopiesIngredientTag(_) => Evaluation::NoMatch,
            RecipeV5::Dynamic(kind) => evaluate_dynamic(*kind, grid, tags),
        };
        match evaluation {
            Evaluation::NoMatch => {}
            Evaluation::Result(result) => return Some(result),
            Evaluation::Throws => return None,
        }
    }
    None
}

/// The repair `findMatchingRecipe` tries before its list (java17:
/// `CraftingManager.findMatchingRecipe:277-316`): exactly two stacks in the grid, one item each,
/// of the same damageable item (`Item.isDamageable`: a durability and no subtypes). The result
/// has both remaining durabilities plus five percent of the maximum, and no tag.
#[must_use]
pub fn repair_result_v5(grid: CraftingGrid<'_>) -> Option<RecipeStack> {
    let mut stacks = occupied(grid).map(|(_, stack)| stack);
    let first = stacks.next()?;
    let second = stacks.next()?;
    if stacks.next().is_some() || first.item != second.item || first.count != 1 || second.count != 1
    {
        return None;
    }
    let props = try_item_props_v5(first.item)?;
    if !props.damageable {
        return None;
    }
    let max_damage = i32::from(props.max_damage);
    let remaining = max_damage - i32::from(first.damage) + max_damage - i32::from(second.damage);
    let repaired = remaining + max_damage * 5 / 100;
    let damage = (max_damage - repaired).max(0) as i16;
    Some(RecipeStack::with_damage(first.item.0, 1, damage))
}

fn plain(stack: RecipeStack) -> RecipeResultV5 {
    RecipeResultV5 {
        stack,
        transform: RecipeTransformV5::Shared(DynamicRecipeTransform::Plain),
    }
}

fn result(stack: RecipeStack, transform: DynamicRecipeTransform) -> Evaluation {
    Evaluation::Result(RecipeResultV5 {
        stack,
        transform: RecipeTransformV5::Shared(transform),
    })
}

/// The occupied grid slots in slot order (`InventoryCrafting` index `x + y * width`).
fn occupied(grid: CraftingGrid<'_>) -> impl Iterator<Item = (usize, RecipeStack)> + '_ {
    let width = usize::from(grid.width());
    (0..grid.len()).filter_map(move |slot| {
        let (x, y) = ((slot % width) as u8, (slot / width) as u8);
        grid.get(x, y).map(|stack| (slot, stack))
    })
}

/// java17: `ShapedRecipes.getCraftingResult` with `func_92100_c` set: every ingredient with a
/// tag in turn replaces the result's tag with a copy of its own, so the last one wins.
fn copies_ingredient_tag(
    output: RecipeStack,
    grid: CraftingGrid<'_>,
    tags: &[SlotTagV5],
) -> RecipeResultV5 {
    match occupied(grid)
        .filter(|(slot, _)| tags[*slot].has_tag)
        .last()
    {
        Some((source_slot, _)) => RecipeResultV5 {
            stack: output,
            transform: RecipeTransformV5::CopyTag { source_slot },
        },
        None => plain(output),
    }
}

fn evaluate_dynamic(
    kind: DynamicRecipeKind,
    grid: CraftingGrid<'_>,
    tags: &[SlotTagV5],
) -> Evaluation {
    match kind {
        DynamicRecipeKind::ArmorDyes => armor_dyes(grid, tags),
        DynamicRecipeKind::Fireworks => fireworks(grid, tags),
        DynamicRecipeKind::BookCloning => book_cloning(grid, tags),
        DynamicRecipeKind::MapCloning => map_cloning(grid),
        DynamicRecipeKind::MapExtending => map_extending(grid, tags),
        // 1.8.9 additions 1.7.10's list never holds (`RECIPES_V5` is generated from it).
        DynamicRecipeKind::BannerAddPattern
        | DynamicRecipeKind::RepairItem
        | DynamicRecipeKind::BannerDuplicatePattern => Evaluation::NoMatch,
    }
}

const LEATHER_ARMOR: std::ops::RangeInclusive<u16> = 298..=301;
const DYE: u16 = 351;

/// java17: `RecipesArmorDyes` (`matches:16-51`, `getCraftingResult:53-136`). One leather armour
/// piece and at least one dye, nothing else. The colour averages the armour's own colour (when
/// it has one) and every dye's sheep colour, rescaled to their average brightness; the arithmetic
/// below keeps Java's float and int steps.
fn armor_dyes(grid: CraftingGrid<'_>, tags: &[SlotTagV5]) -> Evaluation {
    // `EntitySheep.fleeceColorTable`, indexed by wool colour.
    const FLEECE_RGB: [[f32; 3]; 16] = [
        [1.0, 1.0, 1.0],
        [0.85, 0.5, 0.2],
        [0.7, 0.3, 0.85],
        [0.4, 0.6, 0.85],
        [0.9, 0.9, 0.2],
        [0.5, 0.8, 0.1],
        [0.95, 0.5, 0.65],
        [0.3, 0.3, 0.3],
        [0.6, 0.6, 0.6],
        [0.3, 0.5, 0.6],
        [0.5, 0.25, 0.7],
        [0.2, 0.3, 0.7],
        [0.4, 0.3, 0.2],
        [0.4, 0.5, 0.2],
        [0.6, 0.2, 0.2],
        [0.1, 0.1, 0.1],
    ];

    let mut armor = None;
    let mut dyes = 0;
    let mut channels = [0i32; 3];
    let mut brightness = 0i32;
    let mut contributors = 0i32;
    for (slot, stack) in occupied(grid) {
        if LEATHER_ARMOR.contains(&stack.item.0) {
            if armor.is_some() {
                return Evaluation::NoMatch;
            }
            armor = Some((slot, stack));
            if let Some(color) = tags[slot].leather_color {
                let rgb = [
                    ((color >> 16) & 255) as f32 / 255.0,
                    ((color >> 8) & 255) as f32 / 255.0,
                    (color & 255) as f32 / 255.0,
                ];
                let max = rgb[0].max(rgb[1].max(rgb[2]));
                brightness = (brightness as f32 + max * 255.0) as i32;
                for (channel, value) in channels.iter_mut().zip(rgb) {
                    *channel = (*channel as f32 + value * 255.0) as i32;
                }
                contributors += 1;
            }
        } else if stack.item.0 == DYE {
            dyes += 1;
            // `BlockColored.func_150032_b`: wool colour `~damage & 15`.
            let rgb = FLEECE_RGB[(!stack.damage & 15) as usize];
            let scaled = rgb.map(|value| (value * 255.0) as i32);
            brightness += scaled[0].max(scaled[1].max(scaled[2]));
            for (channel, value) in channels.iter_mut().zip(scaled) {
                *channel += value;
            }
            contributors += 1;
        } else {
            return Evaluation::NoMatch;
        }
    }
    let Some((source_slot, armor)) = armor else {
        return Evaluation::NoMatch;
    };
    if dyes == 0 {
        return Evaluation::NoMatch;
    }
    // `ItemArmor.func_82813_b` reads `display` with `getCompoundTag`.
    if tags[source_slot].display_not_compound {
        return Evaluation::Throws;
    }
    let mut average = channels.map(|channel| channel / contributors);
    let average_brightness = brightness as f32 / contributors as f32;
    let max = average[0].max(average[1].max(average[2])) as f32;
    for channel in &mut average {
        *channel = (*channel as f32 * average_brightness / max) as i32;
    }
    let color = (((average[0] << 8) + average[1]) << 8) + average[2];
    result(
        RecipeStack::with_damage(armor.item.0, 1, armor.damage),
        DynamicRecipeTransform::DyeLeather {
            source_slot,
            color: color as u32,
        },
    )
}

/// java17: `RecipeFireworks.matches` (`:17-231`; `getCraftingResult` copies what it built). A
/// rocket from gunpowder and paper (with the charges' explosions), a charge from gunpowder and
/// dyes (with at most one shape item), or a charge's fade colours from dyes.
fn fireworks(grid: CraftingGrid<'_>, tags: &[SlotTagV5]) -> Evaluation {
    // `ItemDye.dyeColors`.
    const DYE_COLORS: [u32; 16] = [
        1_973_019, 11_743_532, 3_887_386, 5_320_730, 2_437_522, 8_073_150, 2_651_799, 11_250_603,
        4_408_131, 14_188_952, 4_312_372, 14_602_026, 6_719_955, 12_801_229, 15_435_844,
        15_790_320,
    ];
    let mut paper = 0u8;
    let mut gunpowder = 0u8;
    let mut dye_damages = Vec::new();
    // The firework charges' slots and damages.
    let mut charges = Vec::new();
    // Glowstone dust and diamonds, each counted (`var7`), and whether either is present.
    let mut effects = 0usize;
    let mut flicker = false;
    let mut trail = false;
    let mut shapes = Vec::new();
    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            289 => gunpowder += 1,
            402 => charges.push((slot, stack.damage)),
            DYE => dye_damages.push(stack.damage),
            339 => paper += 1,
            348 => {
                effects += 1;
                flicker = true;
            }
            264 => {
                effects += 1;
                trail = true;
            }
            385 => shapes.push(1),
            288 => shapes.push(4),
            371 => shapes.push(2),
            397 => shapes.push(3),
            _ => return Evaluation::NoMatch,
        }
    }
    // `var7 += var5 + var8`: effects, dyes and shape items together.
    let modifiers = effects + dye_damages.len() + shapes.len();
    if gunpowder > 3 || paper > 1 {
        return Evaluation::NoMatch;
    }
    // `ItemDye.dyeColors[damage]` throws past the end of the table.
    let colors = || -> Option<Vec<u32>> {
        dye_damages
            .iter()
            .map(|damage| DYE_COLORS.get(usize::try_from(*damage).ok()?).copied())
            .collect()
    };

    if gunpowder >= 1 && paper == 1 && modifiers == 0 {
        let explosion_slots = charges
            .iter()
            .map(|(slot, _)| *slot)
            .filter(|slot| tags[*slot].explosion == TagEntryV5::Compound)
            .collect();
        return result(
            RecipeStack::new(401, 1),
            DynamicRecipeTransform::FireworkRocket {
                flight: gunpowder,
                has_charges: !charges.is_empty(),
                explosion_slots,
            },
        );
    }
    if gunpowder == 1
        && paper == 0
        && charges.is_empty()
        && !dye_damages.is_empty()
        && shapes.len() <= 1
    {
        let Some(colors) = colors() else {
            return Evaluation::Throws;
        };
        return result(
            RecipeStack::new(402, 1),
            DynamicRecipeTransform::FireworkStar {
                colors,
                flicker,
                trail,
                shape: shapes.first().copied().unwrap_or(0),
            },
        );
    }
    if gunpowder == 0
        && paper == 0
        && charges.len() == 1
        && !dye_damages.is_empty()
        && dye_damages.len() == modifiers
    {
        let Some(colors) = colors() else {
            return Evaluation::Throws;
        };
        let (source_slot, damage) = charges[0];
        let tag = tags[source_slot];
        if !tag.has_tag {
            return Evaluation::NoMatch;
        }
        return match tag.explosion {
            TagEntryV5::NotCompound => Evaluation::Throws,
            // A copy of the charge, one item: its damage stays.
            explosion => result(
                RecipeStack::with_damage(402, 1, damage),
                DynamicRecipeTransform::FireworkFade {
                    source_slot,
                    colors,
                    explosion_exists: explosion == TagEntryV5::Compound,
                },
            ),
        };
    }
    Evaluation::NoMatch
}

const WRITABLE_BOOK: u16 = 386;
const WRITTEN_BOOK: u16 = 387;

/// java17: `RecipeBookCloning` (`matches:13-46`, `getCraftingResult:48-96`). One written book and
/// at least one book and quill make the written book plus one copy per book and quill, all with
/// the written book's tag (`getTagCompound().copy()`, which throws when it has none).
fn book_cloning(grid: CraftingGrid<'_>, tags: &[SlotTagV5]) -> Evaluation {
    let mut source = None;
    let mut blanks = 0i32;
    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            WRITTEN_BOOK if source.is_none() => source = Some(slot),
            WRITTEN_BOOK => return Evaluation::NoMatch,
            WRITABLE_BOOK => blanks += 1,
            _ => return Evaluation::NoMatch,
        }
    }
    let Some(source_slot) = source else {
        return Evaluation::NoMatch;
    };
    if blanks == 0 {
        return Evaluation::NoMatch;
    }
    if !tags[source_slot].has_tag {
        return Evaluation::Throws;
    }
    // A new written book: damage 0 whatever the original's.
    Evaluation::Result(RecipeResultV5 {
        stack: RecipeStack::new(WRITTEN_BOOK, blanks + 1),
        transform: RecipeTransformV5::CopyTag { source_slot },
    })
}

const FILLED_MAP: u16 = 358;
const EMPTY_MAP: u16 = 395;

/// java17: `RecipesMapCloning` (`matches:12-45`, `getCraftingResult:47-94`), the same as 1.8.9's:
/// one filled map and at least one empty map make that many copies plus one, with the filled
/// map's damage and its custom name only.
fn map_cloning(grid: CraftingGrid<'_>) -> Evaluation {
    let mut source = None;
    let mut blanks = 0i32;
    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            FILLED_MAP if source.is_none() => source = Some((slot, stack.damage)),
            FILLED_MAP => return Evaluation::NoMatch,
            EMPTY_MAP => blanks += 1,
            _ => return Evaluation::NoMatch,
        }
    }
    let Some((source_slot, damage)) = source else {
        return Evaluation::NoMatch;
    };
    if blanks == 0 {
        return Evaluation::NoMatch;
    }
    result(
        RecipeStack::with_damage(FILLED_MAP, blanks + 1, damage),
        DynamicRecipeTransform::CloneMapDisplayName { source_slot },
    )
}

/// java17: `RecipesMapExtending` (`matches:19-49`, `getCraftingResult:51-75`): its shaped pattern
/// (paper of damage 0 around a filled map), then the client's map data for the first filled map
/// must exist with a scale below 4. The result is that map, one item, marked `map_is_scaling`.
fn map_extending(grid: CraftingGrid<'_>, tags: &[SlotTagV5]) -> Evaluation {
    if !MAP_EXTENDING_PATTERN_V5.matches(grid) {
        return Evaluation::NoMatch;
    }
    let Some((source_slot, map)) = occupied(grid).find(|(_, stack)| stack.item.0 == FILLED_MAP)
    else {
        return Evaluation::NoMatch;
    };
    if !tags[source_slot].map_scale.is_some_and(|scale| scale < 4) {
        return Evaluation::NoMatch;
    }
    result(
        RecipeStack::with_damage(FILLED_MAP, 1, map.damage),
        DynamicRecipeTransform::ScaleMap { source_slot },
    )
}

#[cfg(test)]
#[path = "recipe_tests.rs"]
mod tests;
