//! Client-side crafting and smelting recipe primitives for Minecraft 1.8.9.
//!
//! Vanilla evaluates crafting recipes on the client whenever its crafting matrix changes. The
//! predicted result is later copied into `C0EPacketClickWindow`, so this table is part of the wire
//! contract rather than merely a GUI convenience. This module deliberately stores only registry
//! id, count, and metadata: NBT-bearing dynamic recipes belong in the protocol-facing adapter.

use crate::item::{try_item_props, ItemId};

/// Metadata used by vanilla recipe ingredients to match every subtype.
pub const WILDCARD_DAMAGE: i16 = 32_767;

/// The NBT-free portion of an item stack used by static recipe tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecipeStack {
    /// Numeric item registry id.
    pub item: ItemId,
    /// Number of items in the stack.
    pub count: i32,
    /// Item damage or subtype metadata.
    pub damage: i16,
}

impl RecipeStack {
    /// Constructs a stack with no subtype metadata.
    #[inline]
    #[must_use]
    pub const fn new(item: u16, count: i32) -> Self {
        Self {
            item: ItemId(item),
            count,
            damage: 0,
        }
    }

    /// Constructs a stack with explicit subtype metadata.
    #[inline]
    #[must_use]
    pub const fn with_damage(item: u16, count: i32, damage: i16) -> Self {
        Self {
            item: ItemId(item),
            count,
            damage,
        }
    }
}

/// One item accepted by a recipe slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Ingredient {
    /// Numeric item registry id.
    pub item: ItemId,
    /// Required metadata, or [`WILDCARD_DAMAGE`] for every subtype.
    pub damage: i16,
}

impl Ingredient {
    /// Constructs an exact-metadata ingredient.
    #[inline]
    #[must_use]
    pub const fn exact(item: u16, damage: i16) -> Self {
        Self {
            item: ItemId(item),
            damage,
        }
    }

    /// Constructs an ingredient accepting every metadata value.
    #[inline]
    #[must_use]
    pub const fn any_damage(item: u16) -> Self {
        Self::exact(item, WILDCARD_DAMAGE)
    }

    #[inline]
    fn accepts(self, stack: RecipeStack) -> bool {
        self.item == stack.item && (self.damage == WILDCARD_DAMAGE || self.damage == stack.damage)
    }
}

/// A row-major crafting inventory containing either a 2x2 or 3x3 matrix.
#[derive(Debug, Clone, Copy)]
pub struct CraftingGrid<'a> {
    width: u8,
    height: u8,
    slots: &'a [Option<RecipeStack>],
}

impl<'a> CraftingGrid<'a> {
    /// Makes a grid when its dimensions and slice length describe a valid vanilla matrix.
    #[must_use]
    pub fn new(width: u8, height: u8, slots: &'a [Option<RecipeStack>]) -> Option<Self> {
        if !(1..=3).contains(&width)
            || !(1..=3).contains(&height)
            || slots.len() != usize::from(width) * usize::from(height)
        {
            return None;
        }
        Some(Self {
            width,
            height,
            slots,
        })
    }

    /// Grid width.
    #[inline]
    #[must_use]
    pub const fn width(self) -> u8 {
        self.width
    }

    /// Grid height.
    #[inline]
    #[must_use]
    pub const fn height(self) -> u8 {
        self.height
    }

    /// Stack at a coordinate, or `None` for an empty or out-of-bounds slot.
    #[inline]
    #[must_use]
    pub fn get(self, x: u8, y: u8) -> Option<RecipeStack> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.slots[usize::from(x) + usize::from(y) * usize::from(self.width)]
    }

    /// Number of physical slots in this matrix.
    #[inline]
    #[must_use]
    pub fn len(self) -> usize {
        self.slots.len()
    }

    /// Whether the matrix has no physical slots.
    #[inline]
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.slots.is_empty()
    }
}

/// A vanilla `ShapedRecipes` entry.
#[derive(Debug, Clone, Copy)]
pub struct ShapedRecipe {
    width: u8,
    height: u8,
    ingredients: &'static [Option<Ingredient>],
    output: RecipeStack,
}

impl ShapedRecipe {
    /// Creates a shaped recipe. Invalid table dimensions make the recipe permanently nonmatching.
    #[must_use]
    pub const fn new(
        width: u8,
        height: u8,
        ingredients: &'static [Option<Ingredient>],
        output: RecipeStack,
    ) -> Self {
        Self {
            width,
            height,
            ingredients,
            output,
        }
    }

    /// Returns whether the matrix matches at any vanilla offset, mirrored or unmirrored.
    #[must_use]
    pub fn matches(&self, grid: CraftingGrid<'_>) -> bool {
        if self.width == 0
            || self.height == 0
            || self.width > 3
            || self.height > 3
            || self.ingredients.len() != usize::from(self.width) * usize::from(self.height)
        {
            return false;
        }

        for offset_x in 0..=3 - self.width {
            for offset_y in 0..=3 - self.height {
                // Vanilla checks mirrored first. It does not affect a boolean result, but keeping
                // the order exact makes this safe to extend with diagnostic match information.
                if self.matches_at(grid, offset_x, offset_y, true)
                    || self.matches_at(grid, offset_x, offset_y, false)
                {
                    return true;
                }
            }
        }
        false
    }

    fn matches_at(
        &self,
        grid: CraftingGrid<'_>,
        offset_x: u8,
        offset_y: u8,
        mirrored: bool,
    ) -> bool {
        for x in 0..3 {
            for y in 0..3 {
                let local_x = i16::from(x) - i16::from(offset_x);
                let local_y = i16::from(y) - i16::from(offset_y);
                let expected = if local_x >= 0
                    && local_y >= 0
                    && local_x < i16::from(self.width)
                    && local_y < i16::from(self.height)
                {
                    let local_x = local_x as usize;
                    let local_y = local_y as usize;
                    let recipe_x = if mirrored {
                        usize::from(self.width) - local_x - 1
                    } else {
                        local_x
                    };
                    self.ingredients[recipe_x + local_y * usize::from(self.width)]
                } else {
                    None
                };

                match (grid.get(x, y), expected) {
                    (None, None) => {}
                    (Some(stack), Some(ingredient)) if ingredient.accepts(stack) => {}
                    _ => return false,
                }
            }
        }
        true
    }

    /// Recipe output. Static shaped recipes return a plain copy of this value.
    #[inline]
    #[must_use]
    pub const fn output(&self) -> RecipeStack {
        self.output
    }

    /// Width times height, matching `ShapedRecipes.getRecipeSize`.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> u8 {
        self.width * self.height
    }
}

/// A vanilla `ShapelessRecipes` entry.
#[derive(Debug, Clone, Copy)]
pub struct ShapelessRecipe {
    ingredients: &'static [Ingredient],
    output: RecipeStack,
}

impl ShapelessRecipe {
    /// Creates a shapeless recipe.
    #[must_use]
    pub const fn new(ingredients: &'static [Ingredient], output: RecipeStack) -> Self {
        Self {
            ingredients,
            output,
        }
    }

    /// Matches using vanilla's consume-from-a-copy list walk.
    #[must_use]
    pub fn matches(&self, grid: CraftingGrid<'_>) -> bool {
        if self.ingredients.len() > 9 {
            return false;
        }

        let mut consumed = 0u16;
        for y in 0..grid.height() {
            for x in 0..grid.width() {
                let Some(stack) = grid.get(x, y) else {
                    continue;
                };
                let Some(index) = self
                    .ingredients
                    .iter()
                    .enumerate()
                    .find(|(index, ingredient)| {
                        consumed & (1 << index) == 0 && ingredient.accepts(stack)
                    })
                    .map(|(index, _)| index)
                else {
                    return false;
                };
                consumed |= 1 << index;
            }
        }

        consumed.count_ones() as usize == self.ingredients.len()
    }

    /// Recipe output.
    #[inline]
    #[must_use]
    pub const fn output(&self) -> RecipeStack {
        self.output
    }

    /// Number of ingredients, matching `ShapelessRecipes.getRecipeSize`.
    #[inline]
    #[must_use]
    pub const fn size(&self) -> usize {
        self.ingredients.len()
    }
}

/// A static shaped or shapeless recipe.
#[derive(Debug, Clone, Copy)]
pub enum StaticRecipe {
    /// Position-sensitive recipe.
    Shaped(ShapedRecipe),
    /// Position-independent recipe.
    Shapeless(ShapelessRecipe),
}

/// One of vanilla's dynamic crafting recipes whose match/result depends on NBT or world-facing
/// item behavior that deliberately does not live in `mc-data`.
///
/// The variants are in registration order. Their final positions among static recipes are stored
/// in [`VANILLA_RECIPES`] after vanilla's stable comparator has run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DynamicRecipeKind {
    /// Leather armour plus dyes.
    ArmorDyes = 0,
    /// Firework stars, rockets, and fade colors.
    Fireworks = 1,
    /// Adds one of the vanilla banner patterns.
    BannerAddPattern = 2,
    /// Copies a written book while incrementing its generation.
    BookCloning = 3,
    /// Copies a filled map.
    MapCloning = 4,
    /// Surrounds a filled map with paper when the map can be scaled.
    MapExtending = 5,
    /// Combines the remaining durability of two identical tools.
    RepairItem = 6,
    /// Copies a patterned banner onto a blank banner.
    BannerDuplicatePattern = 7,
}

impl DynamicRecipeKind {
    /// Value returned by the Java recipe's `getRecipeSize` implementation.
    #[must_use]
    pub const fn size(self) -> u8 {
        match self {
            Self::ArmorDyes | Self::Fireworks | Self::BannerAddPattern => 10,
            Self::BookCloning | Self::MapCloning | Self::MapExtending => 9,
            Self::RepairItem => 4,
            Self::BannerDuplicatePattern => 2,
        }
    }
}

/// One entry in the complete, runtime-sorted vanilla 1.8.9 crafting manager.
#[derive(Debug, Clone, Copy)]
pub enum VanillaRecipe {
    /// A fully represented NBT-free recipe.
    Static(StaticRecipe),
    /// A recipe that must be evaluated by a protocol-facing adapter.
    Dynamic(DynamicRecipeKind),
}

/// Result of asking a protocol-facing adapter to evaluate one dynamic recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicRecipeMatch {
    /// The recipe does not match, so manager lookup should continue.
    NoMatch,
    /// The recipe matches and manager lookup must stop. The result may deliberately be absent,
    /// as with cloning a generation-two written book.
    Match(Option<RecipeStack>),
}

/// NBT/world facts needed by vanilla's dynamic recipes, aligned with the physical crafting slots.
///
/// `mc-data` intentionally does not depend on the protocol NBT representation.  A container
/// adapter extracts these few observable facts from its real item stacks and applies the
/// [`DynamicRecipeTransform`] returned by [`evaluate_dynamic_recipe`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DynamicSlotData {
    /// Existing leather-armour `display.color`, when present.
    pub leather_color: Option<u32>,
    /// Written-book generation. A missing integer has vanilla's value of zero.
    pub book_generation: i32,
    /// Whether a firework charge has any root tag compound.
    pub firework_has_tag: bool,
    /// Whether that root contains an `Explosion` compound.
    pub firework_has_explosion: bool,
    /// Banner `BlockEntityTag.Base`; absent means use the stack metadata.
    ///
    /// Java keeps this as an `int` all the way through `TileEntityBanner.getBaseColor`.
    /// Do not narrow it to the item metadata's `short`: malformed-but-wire-representable
    /// values outside the metadata range still participate in the equality check.
    pub banner_base_color: Option<i32>,
    /// Number of compound entries in the banner's `Patterns` list.
    pub banner_patterns: u8,
    /// Client map-data scale, or `None` when the map data is unavailable.
    ///
    /// `MapData.scale` is a signed Java byte and `RecipesMapExtending` checks only `< 4`;
    /// negative values are therefore valid matches, even though normal maps use `0..=4`.
    pub map_scale: Option<i8>,
}

/// Exact NBT operation required after producing the NBT-free [`RecipeStack`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicRecipeTransform {
    /// No tag operation (ordinary static recipes and repair).
    Plain,
    /// Copy the armour stack and replace `display.color`.
    DyeLeather {
        /// Slot whose stack/NBT is copied.
        source_slot: usize,
        /// Replacement 24-bit RGB color.
        color: u32,
    },
    /// Create a rocket tag. Only charge slots that contain an `Explosion` compound are copied.
    FireworkRocket {
        /// Number of gunpowder slots, serialized as the `Flight` byte.
        flight: u8,
        /// Whether vanilla creates the root `Fireworks` tag at all.
        has_charges: bool,
        /// Charge slots whose `Explosion` compounds are appended in matrix order.
        explosion_slots: Vec<usize>,
    },
    /// Create a new firework-star `Explosion` compound.
    FireworkStar {
        /// `ItemDye.dyeColors` values in matrix order.
        colors: Vec<u32>,
        /// Whether glowstone dust was present.
        flicker: bool,
        /// Whether a diamond was present.
        trail: bool,
        /// Vanilla explosion-shape byte.
        shape: u8,
    },
    /// Copy a star and set `Explosion.FadeColors` when `Explosion` already exists.
    FireworkFade {
        /// Star slot whose stack/NBT is copied.
        source_slot: usize,
        /// Fade colors in matrix order.
        colors: Vec<u32>,
        /// Whether `FadeColors` mutates an attached compound rather than a detached empty value.
        explosion_exists: bool,
    },
    /// Copy all written-book NBT, increment generation, and preserve its display name.
    CloneWrittenBook {
        /// Written-book source slot.
        source_slot: usize,
        /// Replacement generation value.
        generation: i32,
    },
    /// Preserve only the filled map's custom display name, as vanilla does.
    CloneMapDisplayName {
        /// Filled-map source slot.
        source_slot: usize,
    },
    /// Copy the filled map and set `map_is_scaling` to true.
    ScaleMap {
        /// Filled-map source slot.
        source_slot: usize,
    },
    /// Copy the banner and append one `Patterns` compound.
    AddBannerPattern {
        /// Banner slot whose stack/NBT is copied.
        source_slot: usize,
        /// Vanilla short pattern id.
        pattern: &'static str,
        /// Dye metadata, or zero for vanilla's special-item/no-dye edge case.
        color: i16,
    },
    /// Copy the patterned banner's complete stack NBT.
    DuplicateBanner {
        /// Patterned banner slot whose stack/NBT is copied.
        source_slot: usize,
    },
}

/// Full result of a dynamic recipe evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicRecipeResult {
    /// NBT-free output fields.
    pub stack: RecipeStack,
    /// Operation for the protocol-facing item-stack adapter.
    pub transform: DynamicRecipeTransform,
}

/// Match state preserving vanilla recipes which match but deliberately return `null`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DynamicRecipeEvaluation {
    /// Lookup continues to the next recipe.
    NoMatch,
    /// Lookup stops, even when the crafting result is absent.
    Match(Option<DynamicRecipeResult>),
}

/// One dynamic recipe's remaining stack and how its NBT is obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DynamicRemainingItem {
    /// NBT-free remaining stack.
    pub stack: RecipeStack,
    /// Source slot to copy NBT from; `None` means a fresh container-item stack.
    pub copy_nbt_from: Option<usize>,
}

include!("recipe_table_1_8_9.rs");

/// The complete 373-entry vanilla 1.8.9 recipe-manager table in final lookup order.
#[inline]
#[must_use]
pub fn vanilla_recipes() -> &'static [VanillaRecipe] {
    &VANILLA_RECIPES
}

/// Iterates the 365 ordinary shaped and shapeless registrations in final relative order.
///
/// This iterator is useful for inspection, but a caller performing crafting lookup must use
/// [`find_matching_vanilla_recipe`] so it cannot accidentally skip dynamic recipes.
pub fn vanilla_static_recipes() -> impl Iterator<Item = &'static StaticRecipe> {
    VANILLA_RECIPES.iter().filter_map(|recipe| match recipe {
        VanillaRecipe::Static(recipe) => Some(recipe),
        VanillaRecipe::Dynamic(_) => None,
    })
}

/// Performs exact first-match lookup over vanilla's complete sorted registration table.
///
/// `match_dynamic` is the dependency boundary for NBT-bearing recipes. Returning
/// [`DynamicRecipeMatch::Match`] stops lookup even when its result is `None`, matching
/// `CraftingManager.findMatchingRecipe` and the generation-two book-cloning edge case.
#[must_use]
pub fn find_matching_vanilla_recipe(
    grid: CraftingGrid<'_>,
    mut match_dynamic: impl FnMut(DynamicRecipeKind, CraftingGrid<'_>) -> DynamicRecipeMatch,
) -> Option<RecipeStack> {
    for recipe in vanilla_recipes() {
        match recipe {
            VanillaRecipe::Static(recipe) if recipe.matches(grid) => return Some(recipe.output()),
            VanillaRecipe::Static(_) => {}
            VanillaRecipe::Dynamic(kind) => match match_dynamic(*kind, grid) {
                DynamicRecipeMatch::NoMatch => {}
                DynamicRecipeMatch::Match(output) => return output,
            },
        }
    }
    None
}

/// Evaluates the complete sorted vanilla table, including all eight dynamic registrations.
///
/// `slot_data` must have one entry per physical matrix slot. The only world-derived value is a
/// filled map's already-known scale; a missing map datum reproduces vanilla's non-match.
#[must_use]
pub fn find_matching_vanilla_recipe_dynamic(
    grid: CraftingGrid<'_>,
    slot_data: &[DynamicSlotData],
) -> DynamicRecipeEvaluation {
    if slot_data.len() != grid.len() {
        return DynamicRecipeEvaluation::NoMatch;
    }

    for recipe in vanilla_recipes() {
        match recipe {
            VanillaRecipe::Static(recipe) if recipe.matches(grid) => {
                return DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                    stack: recipe.output(),
                    transform: DynamicRecipeTransform::Plain,
                }));
            }
            VanillaRecipe::Static(_) => {}
            VanillaRecipe::Dynamic(DynamicRecipeKind::RepairItem) => {
                if let Some(stack) = repair_result(grid) {
                    return DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                        stack,
                        transform: DynamicRecipeTransform::Plain,
                    }));
                }
            }
            VanillaRecipe::Dynamic(kind) => {
                if let DynamicRecipeEvaluation::Match(result) =
                    evaluate_dynamic_recipe(*kind, grid, slot_data)
                {
                    return DynamicRecipeEvaluation::Match(result);
                }
            }
        }
    }
    DynamicRecipeEvaluation::NoMatch
}

/// Evaluates one of vanilla 1.8.9's dynamic crafting recipes.
#[must_use]
pub fn evaluate_dynamic_recipe(
    kind: DynamicRecipeKind,
    grid: CraftingGrid<'_>,
    slot_data: &[DynamicSlotData],
) -> DynamicRecipeEvaluation {
    if slot_data.len() != grid.len() {
        return DynamicRecipeEvaluation::NoMatch;
    }
    match kind {
        DynamicRecipeKind::ArmorDyes => armor_dye_result(grid, slot_data),
        DynamicRecipeKind::Fireworks => firework_result(grid, slot_data),
        DynamicRecipeKind::BannerAddPattern => banner_add_pattern_result(grid, slot_data),
        DynamicRecipeKind::BookCloning => book_clone_result(grid, slot_data),
        DynamicRecipeKind::MapCloning => map_clone_result(grid),
        DynamicRecipeKind::MapExtending => map_extend_result(grid, slot_data),
        DynamicRecipeKind::RepairItem => match repair_result(grid) {
            Some(stack) => DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack,
                transform: DynamicRecipeTransform::Plain,
            })),
            None => DynamicRecipeEvaluation::NoMatch,
        },
        DynamicRecipeKind::BannerDuplicatePattern => banner_duplicate_result(grid, slot_data),
    }
}

/// Produces vanilla's remaining-item array for a dynamic recipe.
///
/// Book cloning keeps the first editable book, and banner duplication keeps the patterned banner.
/// Every other slot follows ordinary container-item behavior.
#[must_use]
pub fn dynamic_remaining_items(
    kind: DynamicRecipeKind,
    grid: CraftingGrid<'_>,
    slot_data: &[DynamicSlotData],
    mut container_item: impl FnMut(ItemId) -> Option<ItemId>,
) -> Vec<Option<DynamicRemainingItem>> {
    if slot_data.len() != grid.len() {
        return vec![None; grid.len()];
    }
    let mut remaining: Vec<_> = grid
        .slots
        .iter()
        .enumerate()
        .map(|(slot, stack)| {
            stack
                .and_then(|stack| container_item(stack.item))
                .map(|item| DynamicRemainingItem {
                    stack: RecipeStack::new(item.0, 1),
                    copy_nbt_from: None,
                })
                .or_else(|| {
                    if kind == DynamicRecipeKind::BannerDuplicatePattern
                        && stack.is_some()
                        && slot_data[slot].banner_patterns > 0
                    {
                        let mut stack = stack.unwrap();
                        stack.count = 1;
                        Some(DynamicRemainingItem {
                            stack,
                            copy_nbt_from: Some(slot),
                        })
                    } else {
                        None
                    }
                })
        })
        .collect();

    if kind == DynamicRecipeKind::BookCloning {
        for (slot, stack) in grid.slots.iter().enumerate() {
            if stack.is_some_and(|stack| matches!(stack.item.0, 386 | 387)) {
                remaining[slot] = Some(DynamicRemainingItem {
                    stack: stack.unwrap(),
                    copy_nbt_from: Some(slot),
                });
                break;
            }
        }
    }
    remaining
}

fn result(stack: RecipeStack, transform: DynamicRecipeTransform) -> DynamicRecipeEvaluation {
    DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult { stack, transform }))
}

fn occupied(grid: CraftingGrid<'_>) -> impl Iterator<Item = (usize, RecipeStack)> + '_ {
    grid.slots
        .iter()
        .enumerate()
        .filter_map(|(slot, stack)| stack.map(|stack| (slot, stack)))
}

fn armor_dye_result(grid: CraftingGrid<'_>, data: &[DynamicSlotData]) -> DynamicRecipeEvaluation {
    const LEATHER_ARMOR: std::ops::RangeInclusive<u16> = 298..=301;
    // EnumDyeColor.byDyeDamage, in dye-damage order. The f32 multiplication and truncation are
    // intentional: RecipesArmorDyes performs these exact operations rather than using dyeColors.
    const DYE_RGB: [[f32; 3]; 16] = [
        [0.1, 0.1, 0.1],
        [0.6, 0.2, 0.2],
        [0.4, 0.5, 0.2],
        [0.4, 0.3, 0.2],
        [0.2, 0.3, 0.7],
        [0.5, 0.25, 0.7],
        [0.3, 0.5, 0.6],
        [0.6, 0.6, 0.6],
        [0.3, 0.3, 0.3],
        [0.95, 0.5, 0.65],
        [0.5, 0.8, 0.1],
        [0.9, 0.9, 0.2],
        [0.4, 0.6, 0.85],
        [0.7, 0.3, 0.85],
        [0.85, 0.5, 0.2],
        [1.0, 1.0, 1.0],
    ];

    let mut armor = None;
    let mut dye_count = 0;
    let mut channel = [0i32; 3];
    let mut brightness = 0i32;
    let mut contributors = 0i32;

    for (slot, stack) in occupied(grid) {
        if LEATHER_ARMOR.contains(&stack.item.0) {
            if armor.is_some() {
                return DynamicRecipeEvaluation::NoMatch;
            }
            armor = Some((slot, stack));
            if let Some(color) = data[slot].leather_color {
                let rgb = [
                    ((color >> 16) & 255) as f32 / 255.0,
                    ((color >> 8) & 255) as f32 / 255.0,
                    (color & 255) as f32 / 255.0,
                ];
                let scaled = [
                    (rgb[0] * 255.0) as i32,
                    (rgb[1] * 255.0) as i32,
                    (rgb[2] * 255.0) as i32,
                ];
                brightness += *scaled.iter().max().unwrap();
                for index in 0..3 {
                    channel[index] += scaled[index];
                }
                contributors += 1;
            }
        } else if stack.item.0 == 351 {
            dye_count += 1;
            let damage = if (0..16).contains(&stack.damage) {
                stack.damage as usize
            } else {
                0
            };
            let rgb = DYE_RGB[damage];
            let scaled = [
                (rgb[0] * 255.0) as i32,
                (rgb[1] * 255.0) as i32,
                (rgb[2] * 255.0) as i32,
            ];
            brightness += *scaled.iter().max().unwrap();
            for index in 0..3 {
                channel[index] += scaled[index];
            }
            contributors += 1;
        } else {
            return DynamicRecipeEvaluation::NoMatch;
        }
    }

    let Some((source_slot, armor)) = armor else {
        return DynamicRecipeEvaluation::NoMatch;
    };
    if dye_count == 0 {
        return DynamicRecipeEvaluation::NoMatch;
    }

    let mut averaged = [
        channel[0] / contributors,
        channel[1] / contributors,
        channel[2] / contributors,
    ];
    let average_brightness = brightness as f32 / contributors as f32;
    let max_channel = *averaged.iter().max().unwrap() as f32;
    for value in &mut averaged {
        *value = (*value as f32 * average_brightness / max_channel) as i32;
    }
    let color = ((averaged[0] << 16) | (averaged[1] << 8) | averaged[2]) as u32;
    result(
        RecipeStack::with_damage(armor.item.0, 1, armor.damage),
        DynamicRecipeTransform::DyeLeather { source_slot, color },
    )
}

fn firework_result(grid: CraftingGrid<'_>, data: &[DynamicSlotData]) -> DynamicRecipeEvaluation {
    const DYE_COLORS: [u32; 16] = [
        1_973_019, 11_743_532, 3_887_386, 5_320_730, 2_437_522, 8_073_150, 2_651_799, 11_250_603,
        4_408_131, 14_188_952, 4_312_372, 14_602_026, 6_719_955, 12_801_229, 15_435_844,
        15_790_320,
    ];
    let mut paper = 0u8;
    let mut gunpowder = 0u8;
    let mut dyes = Vec::new();
    let mut charges = Vec::new();
    let mut flicker = false;
    let mut trail = false;
    let mut shapes = Vec::new();

    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            289 => gunpowder += 1,
            339 => paper += 1,
            351 => dyes.push(DYE_COLORS[(stack.damage & 15) as usize]),
            402 => charges.push(slot),
            348 => flicker = true,
            264 => trail = true,
            385 => shapes.push(1),
            288 => shapes.push(4),
            371 => shapes.push(2),
            397 => shapes.push(3),
            _ => return DynamicRecipeEvaluation::NoMatch,
        }
    }
    let modifiers = dyes.len() + usize::from(flicker) + usize::from(trail) + shapes.len();
    if gunpowder > 3 || paper > 1 {
        return DynamicRecipeEvaluation::NoMatch;
    }

    if (1..=3).contains(&gunpowder) && paper == 1 && modifiers == 0 {
        let explosion_slots = charges
            .iter()
            .copied()
            .filter(|slot| data[*slot].firework_has_explosion)
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

    if gunpowder == 1 && paper == 0 && charges.is_empty() && !dyes.is_empty() && shapes.len() <= 1 {
        return result(
            RecipeStack::new(402, 1),
            DynamicRecipeTransform::FireworkStar {
                colors: dyes,
                flicker,
                trail,
                shape: shapes.first().copied().unwrap_or(0),
            },
        );
    }

    if gunpowder == 0
        && paper == 0
        && charges.len() == 1
        && !dyes.is_empty()
        && !flicker
        && !trail
        && shapes.is_empty()
        && data[charges[0]].firework_has_tag
    {
        let source_slot = charges[0];
        return result(
            RecipeStack::new(402, 1),
            DynamicRecipeTransform::FireworkFade {
                source_slot,
                colors: dyes,
                explosion_exists: data[source_slot].firework_has_explosion,
            },
        );
    }
    DynamicRecipeEvaluation::NoMatch
}

fn book_clone_result(grid: CraftingGrid<'_>, data: &[DynamicSlotData]) -> DynamicRecipeEvaluation {
    let mut source = None;
    let mut blanks = 0u8;
    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            387 if source.is_none() => source = Some(slot),
            387 => return DynamicRecipeEvaluation::NoMatch,
            386 => blanks += 1,
            _ => return DynamicRecipeEvaluation::NoMatch,
        }
    }
    let Some(source_slot) = source else {
        return DynamicRecipeEvaluation::NoMatch;
    };
    if blanks == 0 {
        return DynamicRecipeEvaluation::NoMatch;
    }
    let generation = data[source_slot].book_generation;
    if generation >= 2 {
        return DynamicRecipeEvaluation::Match(None);
    }
    result(
        RecipeStack::new(387, i32::from(blanks)),
        DynamicRecipeTransform::CloneWrittenBook {
            source_slot,
            generation: generation + 1,
        },
    )
}

fn map_clone_result(grid: CraftingGrid<'_>) -> DynamicRecipeEvaluation {
    let mut source = None;
    let mut blanks = 0u8;
    for (slot, stack) in occupied(grid) {
        match stack.item.0 {
            358 if source.is_none() => source = Some((slot, stack.damage)),
            358 => return DynamicRecipeEvaluation::NoMatch,
            395 => blanks += 1,
            _ => return DynamicRecipeEvaluation::NoMatch,
        }
    }
    let Some((source_slot, damage)) = source else {
        return DynamicRecipeEvaluation::NoMatch;
    };
    if blanks == 0 {
        return DynamicRecipeEvaluation::NoMatch;
    }
    result(
        RecipeStack::with_damage(358, i32::from(blanks) + 1, damage),
        DynamicRecipeTransform::CloneMapDisplayName { source_slot },
    )
}

fn map_extend_result(grid: CraftingGrid<'_>, data: &[DynamicSlotData]) -> DynamicRecipeEvaluation {
    if grid.width() != 3 || grid.height() != 3 {
        return DynamicRecipeEvaluation::NoMatch;
    }
    let mut source_slot = None;
    let mut source = None;
    for slot in 0..9 {
        let Some(stack) = grid.slots[slot] else {
            return DynamicRecipeEvaluation::NoMatch;
        };
        if slot == 4 {
            if stack.item.0 != 358 {
                return DynamicRecipeEvaluation::NoMatch;
            }
            source_slot = Some(slot);
            source = Some(stack);
        } else if stack.item.0 != 339 {
            return DynamicRecipeEvaluation::NoMatch;
        }
    }
    let source_slot = source_slot.unwrap();
    if !data[source_slot].map_scale.is_some_and(|scale| scale < 4) {
        return DynamicRecipeEvaluation::NoMatch;
    }
    let source = source.unwrap();
    result(
        RecipeStack::with_damage(358, 1, source.damage),
        DynamicRecipeTransform::ScaleMap { source_slot },
    )
}

#[derive(Clone, Copy)]
struct BannerPattern {
    id: &'static str,
    layers: Option<[&'static str; 3]>,
    item: Option<Ingredient>,
}

const BANNER_PATTERNS: [BannerPattern; 38] = [
    pattern("bl", ["   ", "   ", "#  "]),
    pattern("br", ["   ", "   ", "  #"]),
    pattern("tl", ["#  ", "   ", "   "]),
    pattern("tr", ["  #", "   ", "   "]),
    pattern("bs", ["   ", "   ", "###"]),
    pattern("ts", ["###", "   ", "   "]),
    pattern("ls", ["#  ", "#  ", "#  "]),
    pattern("rs", ["  #", "  #", "  #"]),
    pattern("cs", [" # ", " # ", " # "]),
    pattern("ms", ["   ", "###", "   "]),
    pattern("drs", ["#  ", " # ", "  #"]),
    pattern("dls", ["  #", " # ", "#  "]),
    pattern("ss", ["# #", "# #", "   "]),
    pattern("cr", ["# #", " # ", "# #"]),
    pattern("sc", [" # ", "###", " # "]),
    pattern("bt", ["   ", " # ", "# #"]),
    pattern("tt", ["# #", " # ", "   "]),
    pattern("bts", ["   ", "# #", " # "]),
    pattern("tts", [" # ", "# #", "   "]),
    pattern("ld", ["## ", "#  ", "   "]),
    pattern("rd", ["   ", "  #", " ##"]),
    pattern("lud", ["   ", "#  ", "## "]),
    pattern("rud", [" ##", "  #", "   "]),
    pattern("mc", ["   ", " # ", "   "]),
    pattern("mr", [" # ", "# #", " # "]),
    pattern("vh", ["## ", "## ", "## "]),
    pattern("hh", ["###", "###", "   "]),
    pattern("vhr", [" ##", " ##", " ##"]),
    pattern("hhb", ["   ", "###", "###"]),
    pattern("bo", ["###", "# #", "###"]),
    item_pattern("cbo", 106, 0),
    item_pattern("cre", 397, 4),
    pattern("gra", ["# #", " # ", " # "]),
    pattern("gru", [" # ", " # ", "# #"]),
    item_pattern("bri", 45, 0),
    item_pattern("sku", 397, 1),
    item_pattern("flo", 38, 8),
    item_pattern("moj", 322, 1),
];

const fn pattern(id: &'static str, layers: [&'static str; 3]) -> BannerPattern {
    BannerPattern {
        id,
        layers: Some(layers),
        item: None,
    }
}

const fn item_pattern(id: &'static str, item: u16, damage: i16) -> BannerPattern {
    BannerPattern {
        id,
        layers: None,
        item: Some(Ingredient::exact(item, damage)),
    }
}

fn banner_add_pattern_result(
    grid: CraftingGrid<'_>,
    data: &[DynamicSlotData],
) -> DynamicRecipeEvaluation {
    let banners: Vec<_> = occupied(grid)
        .filter(|(_, stack)| stack.item.0 == 425)
        .collect();
    if banners.len() != 1 || data[banners[0].0].banner_patterns >= 6 {
        return DynamicRecipeEvaluation::NoMatch;
    }
    let Some((pattern, color)) = matching_banner_pattern(grid) else {
        return DynamicRecipeEvaluation::NoMatch;
    };
    let (source_slot, banner) = banners[0];
    result(
        RecipeStack::with_damage(425, 1, banner.damage),
        DynamicRecipeTransform::AddBannerPattern {
            source_slot,
            pattern: pattern.id,
            color,
        },
    )
}

fn matching_banner_pattern(grid: CraftingGrid<'_>) -> Option<(&'static BannerPattern, i16)> {
    'patterns: for pattern in &BANNER_PATTERNS {
        if let Some(item) = pattern.item {
            let mut found_item = false;
            let mut found_dye = false;
            let mut dye_color = 0;
            for (_, stack) in occupied(grid) {
                if stack.item.0 == 425 {
                    continue;
                }
                if stack.item.0 == 351 {
                    if found_dye {
                        continue 'patterns;
                    }
                    found_dye = true;
                    dye_color = stack.damage;
                } else if found_item || !item.accepts(stack) {
                    continue 'patterns;
                } else {
                    found_item = true;
                }
            }
            if found_item {
                return Some((pattern, dye_color));
            }
        } else {
            if grid.len() != 9 {
                continue;
            }
            let layers = pattern.layers.unwrap();
            let mut dye_color = None;
            for slot in 0..9 {
                let expects_dye = layers[slot / 3].as_bytes()[slot % 3] != b' ';
                match (grid.slots[slot], expects_dye) {
                    (Some(stack), false) if stack.item.0 == 425 => {}
                    (Some(stack), true) if stack.item.0 == 351 => {
                        if dye_color.is_some_and(|color| color != stack.damage) {
                            continue 'patterns;
                        }
                        dye_color = Some(stack.damage);
                    }
                    (None, false) => {}
                    _ => continue 'patterns,
                }
            }
            return Some((pattern, dye_color.unwrap_or(0)));
        }
    }
    None
}

fn banner_duplicate_result(
    grid: CraftingGrid<'_>,
    data: &[DynamicSlotData],
) -> DynamicRecipeEvaluation {
    let mut patterned = None;
    let mut blank = None;
    for (slot, stack) in occupied(grid) {
        if stack.item.0 != 425 || (patterned.is_some() && blank.is_some()) {
            return DynamicRecipeEvaluation::NoMatch;
        }
        let base = data[slot]
            .banner_base_color
            .unwrap_or_else(|| i32::from(stack.damage));
        if data[slot].banner_patterns > 0 {
            if patterned.is_some() || blank.is_some_and(|(_, other)| other != base) {
                return DynamicRecipeEvaluation::NoMatch;
            }
            patterned = Some((slot, base, stack.damage));
        } else {
            if blank.is_some() || patterned.is_some_and(|(_, other, _)| other != base) {
                return DynamicRecipeEvaluation::NoMatch;
            }
            blank = Some((slot, base));
        }
    }
    let (source_slot, _, damage) = match (patterned, blank) {
        (Some(patterned), Some(_)) => patterned,
        _ => return DynamicRecipeEvaluation::NoMatch,
    };
    result(
        RecipeStack::with_damage(425, 1, damage),
        DynamicRecipeTransform::DuplicateBanner { source_slot },
    )
}

impl StaticRecipe {
    /// Whether this recipe matches the matrix.
    #[must_use]
    pub fn matches(&self, grid: CraftingGrid<'_>) -> bool {
        match self {
            Self::Shaped(recipe) => recipe.matches(grid),
            Self::Shapeless(recipe) => recipe.matches(grid),
        }
    }

    /// Recipe output.
    #[must_use]
    pub const fn output(&self) -> RecipeStack {
        match self {
            Self::Shaped(recipe) => recipe.output(),
            Self::Shapeless(recipe) => recipe.output(),
        }
    }

    /// Recipe size used by `CraftingManager`'s ordering comparator.
    #[must_use]
    pub fn size(&self) -> usize {
        match self {
            Self::Shaped(recipe) => usize::from(recipe.size()),
            Self::Shapeless(recipe) => recipe.size(),
        }
    }
}

/// Finds the first matching recipe. Callers must preserve vanilla's sorted registration order.
#[must_use]
pub fn find_matching_recipe(
    recipes: &[StaticRecipe],
    grid: CraftingGrid<'_>,
) -> Option<RecipeStack> {
    recipes
        .iter()
        .find(|recipe| recipe.matches(grid))
        .map(StaticRecipe::output)
}

/// Applies vanilla's stable `CraftingManager` recipe comparator.
///
/// Every shaped recipe precedes every shapeless recipe, even when the shapeless recipe is larger.
/// Recipes of the same kind are ordered by descending size, and registration order is retained for
/// ties. The last detail is observable because [`find_matching_recipe`] stops at the first match.
pub fn sort_static_recipes(recipes: &mut [StaticRecipe]) {
    recipes.sort_by(|left, right| match (left, right) {
        (StaticRecipe::Shapeless(_), StaticRecipe::Shaped(_)) => std::cmp::Ordering::Greater,
        (StaticRecipe::Shaped(_), StaticRecipe::Shapeless(_)) => std::cmp::Ordering::Less,
        _ => right.size().cmp(&left.size()),
    });
}

/// Produces the ordinary shaped/shapeless remaining-item array.
///
/// The callback is the protocol-facing item's `hasContainerItem`/`getContainerItem` knowledge.
/// Dynamic recipes can override this behavior in their own adapter.
#[must_use]
pub fn remaining_items(
    grid: CraftingGrid<'_>,
    mut container_item: impl FnMut(ItemId) -> Option<ItemId>,
) -> Vec<Option<RecipeStack>> {
    grid.slots
        .iter()
        .map(|slot| {
            slot.and_then(|stack| container_item(stack.item))
                .map(|item| RecipeStack {
                    item,
                    count: 1,
                    damage: 0,
                })
        })
        .collect()
}

/// Evaluates vanilla's dynamic two-item repair recipe.
///
/// Both inputs must be single-item stacks of the same damageable item. Their remaining durability
/// is combined with an integer-truncated five-percent bonus, and the new stack intentionally has
/// no ingredient NBT. This recipe reports a nominal size of four in Java, despite consuming two
/// stacks, so a full manager must order it with the other size-four dynamic recipes.
#[must_use]
pub fn repair_result(grid: CraftingGrid<'_>) -> Option<RecipeStack> {
    let mut inputs = grid.slots.iter().filter_map(|slot| *slot);
    let first = inputs.next()?;
    let second = inputs.next()?;
    if inputs.next().is_some() || first.item != second.item || first.count != 1 || second.count != 1
    {
        return None;
    }

    // Java's ItemStack carries the result of Item.getItemById. A malformed wire id is null,
    // not a zeroed damageable item, and therefore cannot satisfy ItemStack.isItemStackDamageable.
    let props = try_item_props(first.item)?;
    if !props.damageable {
        return None;
    }

    let max_damage = i32::from(props.max_damage);
    let first_remaining = max_damage - i32::from(first.damage);
    let second_remaining = max_damage - i32::from(second.damage);
    let repaired = first_remaining + second_remaining + max_damage * 5 / 100;
    let damage = (max_damage - repaired).max(0) as i16;
    Some(RecipeStack::with_damage(first.item.0, 1, damage))
}

/// One entry in vanilla's furnace recipe map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmeltingRecipe {
    /// Input item and metadata predicate.
    pub input: Ingredient,
    /// Produced stack.
    pub output: RecipeStack,
}

/// Vanilla 1.8.9's complete 26-entry `FurnaceRecipes.smeltingList`.
///
/// The source Java uses a `HashMap`, so lookup order is intentionally irrelevant: no two input
/// predicates overlap except by item id, and those exact-metadata entries have no wildcard sibling.
pub static SMELTING_RECIPES: [SmeltingRecipe; 26] = [
    smelt_any(15, 265, 0),       // iron_ore -> iron_ingot
    smelt_any(14, 266, 0),       // gold_ore -> gold_ingot
    smelt_any(56, 264, 0),       // diamond_ore -> diamond
    smelt_any(12, 20, 0),        // sand -> glass
    smelt_any(319, 320, 0),      // porkchop -> cooked_porkchop
    smelt_any(363, 364, 0),      // beef -> cooked_beef
    smelt_any(365, 366, 0),      // chicken -> cooked_chicken
    smelt_any(411, 412, 0),      // rabbit -> cooked_rabbit
    smelt_any(423, 424, 0),      // mutton -> cooked_mutton
    smelt_any(4, 1, 0),          // cobblestone -> stone
    smelt_exact(98, 0, 98, 2),   // stonebrick -> cracked stonebrick
    smelt_any(337, 336, 0),      // clay_ball -> brick
    smelt_any(82, 172, 0),       // clay -> hardened_clay
    smelt_any(81, 351, 2),       // cactus -> green dye
    smelt_any(17, 263, 1),       // log -> charcoal
    smelt_any(162, 263, 1),      // log2 -> charcoal
    smelt_any(129, 388, 0),      // emerald_ore -> emerald
    smelt_any(392, 393, 0),      // potato -> baked_potato
    smelt_any(87, 405, 0),       // netherrack -> netherbrick
    smelt_exact(19, 1, 19, 0),   // wet sponge -> sponge
    smelt_exact(349, 0, 350, 0), // cod
    smelt_exact(349, 1, 350, 1), // salmon
    smelt_any(16, 263, 0),       // coal_ore -> coal
    smelt_any(73, 331, 0),       // redstone_ore -> redstone
    smelt_any(21, 351, 4),       // lapis_ore -> blue dye
    smelt_any(153, 406, 0),      // quartz_ore -> quartz
];

const fn smelt_any(input: u16, output: u16, output_damage: i16) -> SmeltingRecipe {
    SmeltingRecipe {
        input: Ingredient::any_damage(input),
        output: RecipeStack::with_damage(output, 1, output_damage),
    }
}

const fn smelt_exact(
    input: u16,
    input_damage: i16,
    output: u16,
    output_damage: i16,
) -> SmeltingRecipe {
    SmeltingRecipe {
        input: Ingredient::exact(input, input_damage),
        output: RecipeStack::with_damage(output, 1, output_damage),
    }
}

/// Returns the first matching smelting result, including vanilla's `32767` wildcard semantics.
#[must_use]
pub fn smelting_result(input: RecipeStack) -> Option<RecipeStack> {
    SMELTING_RECIPES
        .iter()
        .find(|recipe| recipe.input.accepts(input))
        .map(|recipe| recipe.output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STICK: Ingredient = Ingredient::exact(280, 0);
    const PLANK: Ingredient = Ingredient::any_damage(5);

    fn stack(item: u16, damage: i16) -> Option<RecipeStack> {
        Some(RecipeStack::with_damage(item, 1, damage))
    }

    #[test]
    fn shaped_recipes_shift_and_mirror_inside_the_full_three_by_three_scan() {
        static AXE: [Option<Ingredient>; 6] = [
            Some(PLANK),
            Some(PLANK),
            Some(PLANK),
            Some(STICK),
            None,
            Some(STICK),
        ];
        let recipe = ShapedRecipe::new(2, 3, &AXE, RecipeStack::new(271, 1));

        let left = [
            stack(5, 4),
            stack(5, 1),
            None,
            stack(5, 2),
            stack(280, 0),
            None,
            None,
            stack(280, 0),
            None,
        ];
        let right = [
            None,
            stack(5, 1),
            stack(5, 4),
            None,
            stack(280, 0),
            stack(5, 2),
            None,
            stack(280, 0),
            None,
        ];
        assert!(recipe.matches(CraftingGrid::new(3, 3, &left).unwrap()));
        assert!(recipe.matches(CraftingGrid::new(3, 3, &right).unwrap()));

        let mut with_extra = left;
        with_extra[8] = stack(280, 0);
        assert!(!recipe.matches(CraftingGrid::new(3, 3, &with_extra).unwrap()));
    }

    #[test]
    fn a_three_wide_recipe_cannot_match_the_player_two_by_two_grid() {
        static ROW: [Option<Ingredient>; 3] = [Some(PLANK), Some(PLANK), Some(PLANK)];
        let recipe = ShapedRecipe::new(3, 1, &ROW, RecipeStack::new(5, 6));
        let slots = [stack(5, 0), stack(5, 0), None, None];
        assert!(!recipe.matches(CraftingGrid::new(2, 2, &slots).unwrap()));
    }

    #[test]
    fn shapeless_matching_consumes_duplicate_ingredients_once() {
        static TWO_PLANKS: [Ingredient; 2] = [PLANK, PLANK];
        let recipe = ShapelessRecipe::new(&TWO_PLANKS, RecipeStack::new(280, 4));
        let two = [stack(5, 0), None, None, stack(5, 5)];
        let one = [stack(5, 0), None, None, None];
        assert!(recipe.matches(CraftingGrid::new(2, 2, &two).unwrap()));
        assert!(!recipe.matches(CraftingGrid::new(2, 2, &one).unwrap()));
    }

    #[test]
    fn first_match_wins_without_continuing_after_an_output() {
        static ONE: [Ingredient; 1] = [Ingredient::exact(1, 0)];
        let recipes = [
            StaticRecipe::Shapeless(ShapelessRecipe::new(&ONE, RecipeStack::new(2, 1))),
            StaticRecipe::Shapeless(ShapelessRecipe::new(&ONE, RecipeStack::new(3, 1))),
        ];
        let slots = [stack(1, 0)];
        assert_eq!(
            find_matching_recipe(&recipes, CraftingGrid::new(1, 1, &slots).unwrap()),
            Some(RecipeStack::new(2, 1))
        );
    }

    #[test]
    fn vanilla_sort_is_shaped_first_descending_size_and_stable_for_ties() {
        static ONE: [Ingredient; 1] = [Ingredient::exact(1, 0)];
        static TWO: [Ingredient; 2] = [Ingredient::exact(1, 0), Ingredient::exact(2, 0)];
        static SHAPED_ONE: [Option<Ingredient>; 1] = [Some(Ingredient::exact(1, 0))];
        static SHAPED_TWO: [Option<Ingredient>; 2] =
            [Some(Ingredient::exact(1, 0)), Some(Ingredient::exact(2, 0))];
        let mut recipes = [
            StaticRecipe::Shapeless(ShapelessRecipe::new(&TWO, RecipeStack::new(12, 1))),
            StaticRecipe::Shaped(ShapedRecipe::new(
                1,
                1,
                &SHAPED_ONE,
                RecipeStack::new(21, 1),
            )),
            StaticRecipe::Shapeless(ShapelessRecipe::new(&ONE, RecipeStack::new(11, 1))),
            StaticRecipe::Shaped(ShapedRecipe::new(
                1,
                2,
                &SHAPED_TWO,
                RecipeStack::new(22, 1),
            )),
            StaticRecipe::Shaped(ShapedRecipe::new(
                2,
                1,
                &SHAPED_TWO,
                RecipeStack::new(23, 1),
            )),
        ];
        sort_static_recipes(&mut recipes);
        assert_eq!(
            recipes.map(|recipe| recipe.output().item.0),
            [22, 23, 21, 12, 11],
            "equal-size shaped recipes retain registration order"
        );
    }

    #[test]
    fn remaining_items_uses_one_fresh_container_per_occupied_slot() {
        let slots = [stack(335, 0), None, stack(326, 0), stack(5, 0)];
        let remaining = remaining_items(CraftingGrid::new(2, 2, &slots).unwrap(), |item| {
            matches!(item.0, 325 | 326 | 327 | 335).then_some(ItemId(325))
        });
        assert_eq!(remaining, vec![stack(325, 0), None, stack(325, 0), None]);
    }

    #[test]
    fn repair_recipe_uses_two_single_damageable_stacks_and_integer_bonus() {
        // Iron tools have maxDamage=250, so the bonus is floor(250*5/100)=12.
        let inputs = [
            Some(RecipeStack::with_damage(257, 1, 200)),
            None,
            Some(RecipeStack::with_damage(257, 1, 200)),
            None,
        ];
        assert_eq!(
            repair_result(CraftingGrid::new(2, 2, &inputs).unwrap()),
            Some(RecipeStack::with_damage(257, 1, 138))
        );

        let stacked = [
            Some(RecipeStack::with_damage(257, 2, 200)),
            Some(RecipeStack::with_damage(257, 1, 200)),
            None,
            None,
        ];
        assert_eq!(
            repair_result(CraftingGrid::new(2, 2, &stacked).unwrap()),
            None
        );

        let nondamageable = [stack(5, 0), stack(5, 0), None, None];
        assert_eq!(
            repair_result(CraftingGrid::new(2, 2, &nondamageable).unwrap()),
            None
        );

        let unknown = [stack(60_000, 0), stack(60_000, 0), None, None];
        assert_eq!(
            repair_result(CraftingGrid::new(2, 2, &unknown).unwrap()),
            None,
            "an unregistered Item.getItemById result cannot enter ItemStack repair"
        );
    }

    #[test]
    fn furnace_table_is_complete_and_uses_exact_and_wildcard_damage() {
        assert_eq!(SMELTING_RECIPES.len(), 26);
        assert_eq!(
            smelting_result(RecipeStack::with_damage(17, 1, 13)),
            Some(RecipeStack::with_damage(263, 1, 1))
        );
        assert_eq!(
            smelting_result(RecipeStack::with_damage(19, 1, 1)),
            Some(RecipeStack::new(19, 1))
        );
        assert_eq!(smelting_result(RecipeStack::new(19, 1)), None);
        assert_eq!(smelting_result(RecipeStack::with_damage(349, 1, 2)), None);
    }

    #[test]
    fn invalid_grid_and_recipe_shapes_never_match_or_panic() {
        assert!(CraftingGrid::new(0, 2, &[]).is_none());
        assert!(CraftingGrid::new(2, 2, &[None; 3]).is_none());
        static WRONG: [Option<Ingredient>; 1] = [Some(STICK)];
        let recipe = ShapedRecipe::new(2, 2, &WRONG, RecipeStack::new(1, 1));
        assert!(!recipe.matches(CraftingGrid::new(1, 1, &[stack(280, 0)]).unwrap()));
    }

    #[test]
    fn generated_manager_table_has_every_static_and_dynamic_registration() {
        assert_eq!(vanilla_recipes().len(), 373);
        assert_eq!(vanilla_static_recipes().count(), 365);

        let dynamic: Vec<_> = vanilla_recipes()
            .iter()
            .enumerate()
            .filter_map(|(index, recipe)| match recipe {
                VanillaRecipe::Dynamic(kind) => Some((index, *kind, kind.size())),
                VanillaRecipe::Static(_) => None,
            })
            .collect();
        assert_eq!(
            dynamic,
            vec![
                (0, DynamicRecipeKind::ArmorDyes, 10),
                (1, DynamicRecipeKind::Fireworks, 10),
                (2, DynamicRecipeKind::BannerAddPattern, 10),
                (70, DynamicRecipeKind::BookCloning, 9),
                (71, DynamicRecipeKind::MapCloning, 9),
                (72, DynamicRecipeKind::MapExtending, 9),
                (216, DynamicRecipeKind::RepairItem, 4),
                (280, DynamicRecipeKind::BannerDuplicatePattern, 2),
            ]
        );
    }

    #[test]
    fn complete_lookup_crafts_from_the_generated_table() {
        let slots = [
            stack(5, 4),
            stack(5, 4),
            stack(5, 4),
            None,
            stack(280, 0),
            None,
            None,
            stack(280, 0),
            None,
        ];
        let grid = CraftingGrid::new(3, 3, &slots).unwrap();
        assert_eq!(
            find_matching_vanilla_recipe(grid, |_, _| DynamicRecipeMatch::NoMatch),
            Some(RecipeStack::new(270, 1))
        );
    }

    #[test]
    fn recipes_armor_all_sixteen_3x3_entries_match_the_exact_material_and_output() {
        // java: RecipesArmor.addRecipes. Its material row is leather, iron, diamond, gold;
        // each of those supplies helmet/chestplate/leggings/boots in that order.  Chainmail is
        // deliberately absent because vanilla has no chainmail crafting material.
        const HELMET: [usize; 5] = [0, 1, 2, 3, 5];
        const CHESTPLATE: [usize; 8] = [0, 2, 3, 4, 5, 6, 7, 8];
        const LEGGINGS: [usize; 7] = [0, 1, 2, 3, 5, 6, 8];
        const BOOTS: [usize; 4] = [0, 2, 3, 5];
        type ArmorCase<'a> = (u16, [(u16, &'a [usize]); 4]);
        let cases: [ArmorCase<'_>; 4] = [
            (
                334,
                [
                    (298, &HELMET),
                    (299, &CHESTPLATE),
                    (300, &LEGGINGS),
                    (301, &BOOTS),
                ],
            ),
            (
                265,
                [
                    (306, &HELMET),
                    (307, &CHESTPLATE),
                    (308, &LEGGINGS),
                    (309, &BOOTS),
                ],
            ),
            (
                264,
                [
                    (310, &HELMET),
                    (311, &CHESTPLATE),
                    (312, &LEGGINGS),
                    (313, &BOOTS),
                ],
            ),
            (
                266,
                [
                    (314, &HELMET),
                    (315, &CHESTPLATE),
                    (316, &LEGGINGS),
                    (317, &BOOTS),
                ],
            ),
        ];

        for (material, pieces) in cases {
            for (output, occupied) in pieces {
                let mut slots = [None; 9];
                for slot in occupied {
                    slots[*slot] = stack(material, 0);
                }
                let grid = CraftingGrid::new(3, 3, &slots).unwrap();
                assert_eq!(
                    find_matching_vanilla_recipe(grid, |_, _| DynamicRecipeMatch::NoMatch),
                    Some(RecipeStack::new(output, 1)),
                    "material {material}, output {output}"
                );
            }
        }

        let wrong_damage = [
            stack(334, 1),
            stack(334, 0),
            stack(334, 0),
            stack(334, 0),
            None,
            stack(334, 0),
            None,
            None,
            None,
        ];
        assert_eq!(
            find_matching_vanilla_recipe(
                CraftingGrid::new(3, 3, &wrong_damage).unwrap(),
                |_, _| DynamicRecipeMatch::NoMatch
            ),
            None,
            "RecipesArmor uses exact material-stack metadata"
        );
    }

    #[test]
    fn a_matching_dynamic_recipe_with_no_output_still_stops_lookup() {
        let slots = [
            stack(5, 0),
            stack(5, 0),
            stack(5, 0),
            None,
            stack(280, 0),
            None,
            None,
            stack(280, 0),
            None,
        ];
        let grid = CraftingGrid::new(3, 3, &slots).unwrap();
        let mut calls = 0;
        let result = find_matching_vanilla_recipe(grid, |kind, _| {
            calls += 1;
            if kind == DynamicRecipeKind::ArmorDyes {
                DynamicRecipeMatch::Match(None)
            } else {
                DynamicRecipeMatch::NoMatch
            }
        });
        assert_eq!(result, None);
        assert_eq!(calls, 1, "lookup must stop on the matched null result");
    }

    fn dynamic(
        kind: DynamicRecipeKind,
        width: u8,
        height: u8,
        slots: &[Option<RecipeStack>],
        data: &[DynamicSlotData],
    ) -> DynamicRecipeEvaluation {
        evaluate_dynamic_recipe(kind, CraftingGrid::new(width, height, slots).unwrap(), data)
    }

    #[test]
    fn armor_dye_uses_sheep_rgb_brightness_and_copies_the_armor_slot() {
        let slots = [stack(298, 37), stack(351, 1), None, None];
        let data = [DynamicSlotData::default(); 4];
        assert_eq!(
            dynamic(DynamicRecipeKind::ArmorDyes, 2, 2, &slots, &data),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack: RecipeStack::with_damage(298, 1, 37),
                transform: DynamicRecipeTransform::DyeLeather {
                    source_slot: 0,
                    color: 0x99_33_33,
                },
            }))
        );

        let mut colored = data;
        colored[0].leather_color = Some(0x33_99_33);
        assert_ne!(
            dynamic(DynamicRecipeKind::ArmorDyes, 2, 2, &slots, &colored),
            dynamic(DynamicRecipeKind::ArmorDyes, 2, 2, &slots, &data),
            "existing leather color must participate in the weighted blend"
        );
    }

    #[test]
    fn fireworks_build_all_three_nbt_transforms_and_reject_two_shapes() {
        let rocket = [stack(339, 0), stack(289, 0), stack(289, 0), stack(402, 0)];
        let mut rocket_data = [DynamicSlotData::default(); 4];
        rocket_data[3].firework_has_explosion = true;
        assert_eq!(
            dynamic(DynamicRecipeKind::Fireworks, 2, 2, &rocket, &rocket_data),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack: RecipeStack::new(401, 1),
                transform: DynamicRecipeTransform::FireworkRocket {
                    flight: 2,
                    has_charges: true,
                    explosion_slots: vec![3],
                },
            }))
        );

        let star = [stack(289, 0), stack(351, 1), stack(348, 0), stack(288, 0)];
        assert_eq!(
            dynamic(
                DynamicRecipeKind::Fireworks,
                2,
                2,
                &star,
                &[DynamicSlotData::default(); 4]
            ),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack: RecipeStack::new(402, 1),
                transform: DynamicRecipeTransform::FireworkStar {
                    colors: vec![11_743_532],
                    flicker: true,
                    trail: false,
                    shape: 4,
                },
            }))
        );

        let fade = [stack(402, 0), stack(351, 15), None, None];
        let mut fade_data = [DynamicSlotData::default(); 4];
        fade_data[0].firework_has_tag = true;
        assert!(matches!(
            dynamic(DynamicRecipeKind::Fireworks, 2, 2, &fade, &fade_data),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                transform: DynamicRecipeTransform::FireworkFade {
                    explosion_exists: false,
                    ..
                },
                ..
            }))
        ));

        let two_shapes = [stack(289, 0), stack(351, 0), stack(288, 0), stack(385, 0)];
        assert_eq!(
            dynamic(
                DynamicRecipeKind::Fireworks,
                2,
                2,
                &two_shapes,
                &[DynamicSlotData::default(); 4]
            ),
            DynamicRecipeEvaluation::NoMatch
        );
    }

    #[test]
    fn book_clone_matches_generation_two_but_deliberately_has_no_result() {
        let slots = [stack(387, 0), stack(386, 0), stack(386, 0), None];
        let mut data = [DynamicSlotData::default(); 4];
        data[0].book_generation = 1;
        assert_eq!(
            dynamic(DynamicRecipeKind::BookCloning, 2, 2, &slots, &data),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack: RecipeStack::new(387, 2),
                transform: DynamicRecipeTransform::CloneWrittenBook {
                    source_slot: 0,
                    generation: 2,
                },
            }))
        );
        data[0].book_generation = 2;
        assert_eq!(
            dynamic(DynamicRecipeKind::BookCloning, 2, 2, &slots, &data),
            DynamicRecipeEvaluation::Match(None)
        );
    }

    #[test]
    fn map_clone_counts_blank_slots_and_extending_requires_live_scale_below_four() {
        let clone_slots = [stack(395, 0), stack(358, 12), stack(395, 0), None];
        assert_eq!(
            dynamic(
                DynamicRecipeKind::MapCloning,
                2,
                2,
                &clone_slots,
                &[DynamicSlotData::default(); 4]
            ),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                stack: RecipeStack::with_damage(358, 3, 12),
                transform: DynamicRecipeTransform::CloneMapDisplayName { source_slot: 1 },
            }))
        );

        let extend_slots = [
            stack(339, 0),
            stack(339, 0),
            stack(339, 0),
            stack(339, 0),
            stack(358, 7),
            stack(339, 0),
            stack(339, 0),
            stack(339, 0),
            stack(339, 0),
        ];
        let mut data = [DynamicSlotData::default(); 9];
        assert_eq!(
            dynamic(DynamicRecipeKind::MapExtending, 3, 3, &extend_slots, &data),
            DynamicRecipeEvaluation::NoMatch,
            "vanilla mismatches when client MapData is missing"
        );
        data[4].map_scale = Some(3);
        assert!(matches!(
            dynamic(DynamicRecipeKind::MapExtending, 3, 3, &extend_slots, &data),
            DynamicRecipeEvaluation::Match(Some(_))
        ));
        data[4].map_scale = Some(-1);
        assert!(
            matches!(
                dynamic(DynamicRecipeKind::MapExtending, 3, 3, &extend_slots, &data),
                DynamicRecipeEvaluation::Match(Some(_))
            ),
            "RecipesMapExtending checks only MapData.scale < 4; signed negative bytes match"
        );
        data[4].map_scale = Some(4);
        assert_eq!(
            dynamic(DynamicRecipeKind::MapExtending, 3, 3, &extend_slots, &data),
            DynamicRecipeEvaluation::NoMatch
        );
    }

    #[test]
    fn banners_preserve_pattern_order_and_the_special_item_no_dye_edge() {
        let shaped = [
            stack(351, 5),
            None,
            None,
            None,
            stack(425, 2),
            None,
            None,
            None,
            None,
        ];
        assert!(matches!(
            dynamic(
                DynamicRecipeKind::BannerAddPattern,
                3,
                3,
                &shaped,
                &[DynamicSlotData::default(); 9]
            ),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                transform: DynamicRecipeTransform::AddBannerPattern {
                    pattern: "tl",
                    color: 5,
                    ..
                },
                ..
            }))
        ));

        let special = [stack(425, 0), stack(106, 0), None, None];
        assert!(matches!(
            dynamic(
                DynamicRecipeKind::BannerAddPattern,
                2,
                2,
                &special,
                &[DynamicSlotData::default(); 4]
            ),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                transform: DynamicRecipeTransform::AddBannerPattern {
                    pattern: "cbo",
                    color: 0,
                    ..
                },
                ..
            }))
        ));

        let duplicate = [stack(425, 4), stack(425, 4), None, None];
        let mut duplicate_data = [DynamicSlotData::default(); 4];
        duplicate_data[0].banner_patterns = 1;
        assert!(matches!(
            dynamic(
                DynamicRecipeKind::BannerDuplicatePattern,
                2,
                2,
                &duplicate,
                &duplicate_data
            ),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                transform: DynamicRecipeTransform::DuplicateBanner { source_slot: 0 },
                ..
            }))
        ));
        duplicate_data[1].banner_base_color = Some(3);
        assert_eq!(
            dynamic(
                DynamicRecipeKind::BannerDuplicatePattern,
                2,
                2,
                &duplicate,
                &duplicate_data
            ),
            DynamicRecipeEvaluation::NoMatch
        );
    }

    #[test]
    fn banner_base_comparison_keeps_java_integer_width() {
        let slots = [stack(425, 4), stack(425, 4), None, None];
        let mut data = [DynamicSlotData::default(); 4];
        data[0].banner_patterns = 1;
        data[0].banner_base_color = Some(65_536);
        data[1].banner_base_color = Some(0);

        assert_eq!(
            dynamic(
                DynamicRecipeKind::BannerDuplicatePattern,
                2,
                2,
                &slots,
                &data
            ),
            DynamicRecipeEvaluation::NoMatch,
            "TileEntityBanner.getBaseColor returns an int; 65536 must not wrap to 0"
        );

        data[1].banner_base_color = Some(65_536);
        assert!(matches!(
            dynamic(
                DynamicRecipeKind::BannerDuplicatePattern,
                2,
                2,
                &slots,
                &data
            ),
            DynamicRecipeEvaluation::Match(Some(_))
        ));
    }

    #[test]
    fn complete_dynamic_lookup_keeps_sorted_first_match_semantics() {
        let slots = [stack(298, 0), stack(351, 0), None, None];
        let grid = CraftingGrid::new(2, 2, &slots).unwrap();
        let data = [DynamicSlotData::default(); 4];
        assert!(matches!(
            find_matching_vanilla_recipe_dynamic(grid, &data),
            DynamicRecipeEvaluation::Match(Some(DynamicRecipeResult {
                transform: DynamicRecipeTransform::DyeLeather { .. },
                ..
            }))
        ));
        assert_eq!(
            find_matching_vanilla_recipe_dynamic(grid, &data[..3]),
            DynamicRecipeEvaluation::NoMatch,
            "misaligned NBT sidecars must never be indexed or partially evaluated"
        );
    }

    #[test]
    fn dynamic_remaining_items_keep_book_and_patterned_banner_nbt_sources() {
        let books = [
            Some(RecipeStack::with_damage(387, 1, 2)),
            stack(386, 0),
            None,
            None,
        ];
        let grid = CraftingGrid::new(2, 2, &books).unwrap();
        let remaining = dynamic_remaining_items(
            DynamicRecipeKind::BookCloning,
            grid,
            &[DynamicSlotData::default(); 4],
            |_| None,
        );
        assert_eq!(
            remaining,
            vec![
                Some(DynamicRemainingItem {
                    stack: RecipeStack::with_damage(387, 1, 2),
                    copy_nbt_from: Some(0),
                }),
                None,
                None,
                None,
            ]
        );

        let banners = [stack(425, 3), stack(425, 3), None, None];
        let mut data = [DynamicSlotData::default(); 4];
        data[1].banner_patterns = 2;
        let remaining = dynamic_remaining_items(
            DynamicRecipeKind::BannerDuplicatePattern,
            CraftingGrid::new(2, 2, &banners).unwrap(),
            &data,
            |_| None,
        );
        assert_eq!(
            remaining[1],
            Some(DynamicRemainingItem {
                stack: RecipeStack::with_damage(425, 1, 3),
                copy_nbt_from: Some(1),
            })
        );
    }

    #[test]
    fn generated_recipe_table_matches_the_runtime_oracle_fingerprint() {
        fn hash(hash: &mut u64, bytes: impl IntoIterator<Item = u8>) {
            for byte in bytes {
                *hash ^= u64::from(byte);
                *hash = hash.wrapping_mul(0x100000001b3);
            }
        }
        fn hash_stack(hash_value: &mut u64, stack: RecipeStack) {
            hash(hash_value, stack.item.0.to_le_bytes());
            hash(hash_value, [(stack.count as i8) as u8]);
            hash(hash_value, stack.damage.to_le_bytes());
        }
        fn hash_ingredient(hash_value: &mut u64, ingredient: Option<Ingredient>) {
            match ingredient {
                None => hash(hash_value, [0]),
                Some(ingredient) => {
                    hash(hash_value, [1]);
                    hash(hash_value, ingredient.item.0.to_le_bytes());
                    hash(hash_value, ingredient.damage.to_le_bytes());
                }
            }
        }

        let mut fingerprint = 0xcbf29ce484222325;
        for recipe in vanilla_recipes() {
            match recipe {
                VanillaRecipe::Dynamic(kind) => {
                    hash(&mut fingerprint, [b'D', *kind as u8, kind.size()]);
                }
                VanillaRecipe::Static(StaticRecipe::Shaped(recipe)) => {
                    hash(&mut fingerprint, [b'S', recipe.width, recipe.height]);
                    hash_stack(&mut fingerprint, recipe.output);
                    hash(&mut fingerprint, [recipe.ingredients.len() as u8]);
                    for ingredient in recipe.ingredients {
                        hash_ingredient(&mut fingerprint, *ingredient);
                    }
                }
                VanillaRecipe::Static(StaticRecipe::Shapeless(recipe)) => {
                    hash(&mut fingerprint, [b'L', recipe.ingredients.len() as u8]);
                    hash_stack(&mut fingerprint, recipe.output);
                    for ingredient in recipe.ingredients {
                        hash_ingredient(&mut fingerprint, Some(*ingredient));
                    }
                }
            }
        }
        assert_eq!(fingerprint, 0x635b_6eae_fef7_5edd);
    }
}
