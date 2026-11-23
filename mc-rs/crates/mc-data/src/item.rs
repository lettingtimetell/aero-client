//! Item registry: ids, stack limits, durability, tool/armour/food statistics.
//!
//! Java input: `net/minecraft/item/Item.java` plus its subclasses, read live through the
//! extractor rather than transcribed.

use crate::block::BlockId;
use crate::generated::items::{FOOD_VARIANTS, ITEMS, ITEM_BY_ID, ITEM_NAMES};

pub use crate::generated::items::{
    ArmorMaterial, ArmorMaterialInfo, ItemKind, ToolMaterial, ToolMaterialInfo, MAX_ITEM_ID,
};

/// An item registry id. Block items reuse the block id (`0..=197`); pure items start at 256.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ItemId(pub u16);

impl ItemId {
    /// Properties for this id, or the unregistered compatibility sentinel.
    ///
    /// New registry-sensitive code should prefer [`Self::try_props`], which models Java's
    /// nullable `Item.getItemById` result. This total form exists for established wire-facing
    /// call sites that must not panic while decoding an arbitrary `ItemStack` id.
    #[inline]
    #[must_use]
    pub fn props(self) -> &'static ItemProps {
        item_props(self)
    }

    /// Properties for a registered item id.
    ///
    /// This is the faithful boundary for Java's `Item.getItemById`: an id absent from the
    /// registry produces `None`, rather than a fabricated item row.
    #[inline]
    #[must_use]
    pub fn try_props(self) -> Option<&'static ItemProps> {
        try_item_props(self)
    }

    /// Metadata-aware unlocalized name for an item stack.
    ///
    /// Unlike [`ItemProps::unlocalized`], this follows the
    /// `Item.getUnlocalizedName(ItemStack)` overrides modelled by
    /// [`ItemProps::unlocalized_name`]. The returned value is the key stem; callers append
    /// `.name` before asking the language table to translate it. Negative damage is
    /// normalised to zero, as `ItemStack` does in 1.8.9.
    #[inline]
    #[must_use]
    pub fn unlocalized_name(self, damage: i16) -> &'static str {
        self.props().unlocalized_name(damage)
    }

    /// The `ItemRecord` data for this registry id, if it is one of 1.8.9's twelve music discs.
    ///
    /// This is the extracted registry id paired with the constructor argument passed by
    /// `Items.registerItems`: `new ItemRecord("13")`, `new ItemRecord("cat")`, and so on.
    /// The argument is not recoverable from `Item`'s shared property getters, but it drives
    /// both `ItemRecord.addInformation` and the jukebox sound/now-playing translation key.
    #[inline]
    #[must_use]
    pub fn record_info(self) -> Option<&'static RecordInfo> {
        record_info(self)
    }
}

/// The per-disc constructor data held by Java's `ItemRecord` subclass.
///
/// `record_name` is `ItemRecord.recordName`; Java forms the user-visible line as
/// `item.record.<record_name>.desc`. The generated item table deliberately holds only fields
/// common to `Item`, so the twelve subclass-only rows live here instead of pretending that the
/// shared `item.record` unlocalised name distinguishes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordInfo {
    /// The numeric item id (`2256..=2267`, in registration order).
    pub item: ItemId,
    /// The exact `ItemRecord` constructor argument, e.g. `"mellohi"`.
    pub record_name: &'static str,
    /// Translation key read by `ItemRecord.getRecordNameLocal`.
    pub description_key: &'static str,
    /// Bundled 1.8.9 `en_US.lang` fallback for [`Self::description_key`].
    pub description_en_us: &'static str,
}

/// Every 1.8.9 music disc, in `Items.registerItems` numeric-id order.
///
/// `record_11` is intentionally between `ward` and `wait`, despite `11` sorting before `13`.
pub const RECORDS: [RecordInfo; 12] = [
    RecordInfo {
        item: ItemId(2256),
        record_name: "13",
        description_key: "item.record.13.desc",
        description_en_us: "C418 - 13",
    },
    RecordInfo {
        item: ItemId(2257),
        record_name: "cat",
        description_key: "item.record.cat.desc",
        description_en_us: "C418 - cat",
    },
    RecordInfo {
        item: ItemId(2258),
        record_name: "blocks",
        description_key: "item.record.blocks.desc",
        description_en_us: "C418 - blocks",
    },
    RecordInfo {
        item: ItemId(2259),
        record_name: "chirp",
        description_key: "item.record.chirp.desc",
        description_en_us: "C418 - chirp",
    },
    RecordInfo {
        item: ItemId(2260),
        record_name: "far",
        description_key: "item.record.far.desc",
        description_en_us: "C418 - far",
    },
    RecordInfo {
        item: ItemId(2261),
        record_name: "mall",
        description_key: "item.record.mall.desc",
        description_en_us: "C418 - mall",
    },
    RecordInfo {
        item: ItemId(2262),
        record_name: "mellohi",
        description_key: "item.record.mellohi.desc",
        description_en_us: "C418 - mellohi",
    },
    RecordInfo {
        item: ItemId(2263),
        record_name: "stal",
        description_key: "item.record.stal.desc",
        description_en_us: "C418 - stal",
    },
    RecordInfo {
        item: ItemId(2264),
        record_name: "strad",
        description_key: "item.record.strad.desc",
        description_en_us: "C418 - strad",
    },
    RecordInfo {
        item: ItemId(2265),
        record_name: "ward",
        description_key: "item.record.ward.desc",
        description_en_us: "C418 - ward",
    },
    RecordInfo {
        item: ItemId(2266),
        record_name: "11",
        description_key: "item.record.11.desc",
        description_en_us: "C418 - 11",
    },
    RecordInfo {
        item: ItemId(2267),
        record_name: "wait",
        description_key: "item.record.wait.desc",
        description_en_us: "C418 - wait",
    },
];

/// Java's `itemId -> ItemRecord.recordName` relation, if the id names a registered record.
#[must_use]
pub fn record_info(id: ItemId) -> Option<&'static RecordInfo> {
    RECORDS.iter().find(|record| record.item == id)
}

/// One Java 1.8.9 `EnumDyeColor` row, in block-metadata order.
///
/// `name` is the serialized snake-case value from `IStringSerializable.getName`; the distinct
/// `unlocalized_name` is the legacy camel-case value used to build keys such as
/// `item.fireworksCharge.lightBlue`. Those two values differ for light blue only in 1.8.9.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DyeColor {
    /// Block metadata (`0` white through `15` black).
    pub metadata: u8,
    /// `ItemDye` damage (`0` black through `15` white).
    pub dye_damage: u8,
    /// Serialized block-state name.
    pub name: &'static str,
    /// Legacy translation-key suffix returned by `getUnlocalizedName()`.
    pub unlocalized_name: &'static str,
}

/// Java 1.8.9 `EnumDyeColor.values()`, which is also `META_LOOKUP` order.
pub const DYE_COLORS: [DyeColor; 16] = [
    DyeColor {
        metadata: 0,
        dye_damage: 15,
        name: "white",
        unlocalized_name: "white",
    },
    DyeColor {
        metadata: 1,
        dye_damage: 14,
        name: "orange",
        unlocalized_name: "orange",
    },
    DyeColor {
        metadata: 2,
        dye_damage: 13,
        name: "magenta",
        unlocalized_name: "magenta",
    },
    DyeColor {
        metadata: 3,
        dye_damage: 12,
        name: "light_blue",
        unlocalized_name: "lightBlue",
    },
    DyeColor {
        metadata: 4,
        dye_damage: 11,
        name: "yellow",
        unlocalized_name: "yellow",
    },
    DyeColor {
        metadata: 5,
        dye_damage: 10,
        name: "lime",
        unlocalized_name: "lime",
    },
    DyeColor {
        metadata: 6,
        dye_damage: 9,
        name: "pink",
        unlocalized_name: "pink",
    },
    DyeColor {
        metadata: 7,
        dye_damage: 8,
        name: "gray",
        unlocalized_name: "gray",
    },
    DyeColor {
        metadata: 8,
        dye_damage: 7,
        name: "silver",
        unlocalized_name: "silver",
    },
    DyeColor {
        metadata: 9,
        dye_damage: 6,
        name: "cyan",
        unlocalized_name: "cyan",
    },
    DyeColor {
        metadata: 10,
        dye_damage: 5,
        name: "purple",
        unlocalized_name: "purple",
    },
    DyeColor {
        metadata: 11,
        dye_damage: 4,
        name: "blue",
        unlocalized_name: "blue",
    },
    DyeColor {
        metadata: 12,
        dye_damage: 3,
        name: "brown",
        unlocalized_name: "brown",
    },
    DyeColor {
        metadata: 13,
        dye_damage: 2,
        name: "green",
        unlocalized_name: "green",
    },
    DyeColor {
        metadata: 14,
        dye_damage: 1,
        name: "red",
        unlocalized_name: "red",
    },
    DyeColor {
        metadata: 15,
        dye_damage: 0,
        name: "black",
        unlocalized_name: "black",
    },
];

/// Java `EnumDyeColor.byMetadata`; out-of-range values select metadata `0` (white).
#[must_use]
pub fn dye_color_by_metadata(metadata: i32) -> &'static DyeColor {
    usize::try_from(metadata)
        .ok()
        .and_then(|index| DYE_COLORS.get(index))
        .unwrap_or(&DYE_COLORS[0])
}

/// Java `EnumDyeColor.byDyeDamage`; out-of-range values select dye damage `0` (black).
#[must_use]
pub fn dye_color_by_dye_damage(damage: i32) -> &'static DyeColor {
    usize::try_from(damage)
        .ok()
        .filter(|&index| index < DYE_COLORS.len())
        .map_or(&DYE_COLORS[15], |index| &DYE_COLORS[15 - index])
}

impl ToolMaterial {
    /// Java 1.8.9 `Item.ToolMaterial.getRepairItem()`.
    ///
    /// `Emerald` is vanilla's enum name for the diamond tool tier, so its repair item is a
    /// diamond rather than an emerald.
    #[must_use]
    pub const fn repair_item(self) -> ItemId {
        match self {
            Self::Wood => ItemId(5),      // minecraft:planks
            Self::Stone => ItemId(4),     // minecraft:cobblestone
            Self::Iron => ItemId(265),    // minecraft:iron_ingot
            Self::Emerald => ItemId(264), // minecraft:diamond
            Self::Gold => ItemId(266),    // minecraft:gold_ingot
        }
    }
}

impl ArmorMaterial {
    /// Java 1.8.9 `ItemArmor.ArmorMaterial.getRepairItem()`.
    #[must_use]
    pub const fn repair_item(self) -> ItemId {
        match self {
            Self::Leather => ItemId(334),            // minecraft:leather
            Self::Chain | Self::Iron => ItemId(265), // minecraft:iron_ingot
            Self::Gold => ItemId(266),               // minecraft:gold_ingot
            Self::Diamond => ItemId(264),            // minecraft:diamond
        }
    }
}

/// Everything the inventory, tooltip and item-render code needs about one item.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ItemProps {
    /// The item's numeric registry id.
    pub id: ItemId,
    /// Registry name, e.g. `minecraft:diamond_sword`.
    pub name: &'static str,
    /// Translation key stem, e.g. `item.swordDiamond`.
    pub unlocalized: &'static str,
    /// `en_US` display name.
    pub display_name: &'static str,
    /// False only for the unregistered-id sentinel.
    pub registered: bool,
    /// Maximum stack size, `1..=64`.
    pub max_stack_size: u8,
    /// Durability, or `0` when the item takes no damage.
    pub max_damage: u16,
    /// Whether the item has a durability bar.
    pub damageable: bool,
    /// Whether the damage value selects a subtype (wool colours, dyes, spawn eggs).
    pub has_subtypes: bool,
    /// Enchantability weight; `0` means the item cannot be enchanted at a table.
    pub enchantability: u8,
    /// `Item.isFull3D` - rendered as a held tool rather than a flat sprite.
    pub full_3d: bool,
    /// Broad category, derived from the vanilla `Item` subclass.
    pub kind: ItemKind,
    /// Tool tier for tools and swords.
    pub tool_material: Option<ToolMaterial>,
    /// Armour tier for armour pieces.
    pub armor_material: Option<ArmorMaterial>,
    /// Armour slot, using vanilla's `ItemArmor.armorType` numbering:
    /// `0` helmet, `1` chestplate, `2` leggings, `3` boots.
    pub armor_slot: Option<u8>,
    /// Armour points this piece grants.
    pub armor_reduction: u8,
    /// Bonus melee damage over the 1-damage baseline.
    pub attack_damage: f32,
    /// Break-speed multiplier against the tool's proper material.
    pub efficiency: f32,
    /// Hunger restored, in half-drumsticks.
    pub food_heal: u8,
    /// Saturation modifier for food.
    pub food_saturation: f32,
    /// The block this item places, when it is a block item.
    pub block: Option<BlockId>,
    /// Creative-inventory tab label, e.g. `buildingBlocks`.
    pub creative_tab: Option<&'static str>,
    /// Originating Java class.
    pub java_class: &'static str,
}

/// The virtual food statistics for one concrete item stack.
///
/// Most `ItemFood` instances return the two values stored on [`ItemProps`].
/// Metadata-sensitive subclasses such as 1.8.9's `ItemFishFood` select these through
/// their `ItemStack` overload instead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FoodStats {
    /// Hunger restored, in half-drumsticks.
    pub heal: u8,
    /// Saturation modifier passed to `FoodStats.addStats`.
    pub saturation: f32,
}

/// The part of an `ItemStack` mutated by Java's durability methods.
///
/// `mc-protocol` owns the wire stack shape, while this small value type keeps the item-data
/// crate independent of that protocol crate.  `damage` is deliberately an `i32`: Java stores
/// `ItemStack.itemDamage` as an `int`, even though the network representation is a signed short.
/// [`Self::new`] applies the same negative-damage clamp as the three-argument Java constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ItemStackDamageState {
    /// Java `ItemStack.stackSize`.
    pub count: i32,
    /// Java `ItemStack.itemDamage`.
    pub damage: i32,
}

impl ItemStackDamageState {
    /// Constructs a state using Java `ItemStack(Item, amount, meta)` damage normalization.
    #[must_use]
    pub const fn new(count: i32, damage: i32) -> Self {
        Self {
            count,
            damage: if damage < 0 { 0 } else { damage },
        }
    }
}

/// The random operations used by Java's `EnchantmentDurability.negateDamage`.
///
/// This trait keeps `mc-data` independent of a particular RNG crate.  A caller that owns the
/// entity's `java.util.Random` stream can adapt it directly.  The implementation must provide
/// Java-equivalent `nextFloat()` and `nextInt(bound)` results; the durability code intentionally
/// does not substitute a deterministic or host-language RNG.
pub trait DurabilityRandom {
    /// Java `Random.nextFloat()`, consumed only for armour stacks.
    fn next_f32(&mut self) -> f32;

    /// Java `Random.nextInt(bound)`, with a positive bound.
    fn next_i32_bound(&mut self, bound: i32) -> i32;
}

/// The observable result of one Java `ItemStack.damageItem` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ItemDamageResult {
    /// Durability points that survived Unbreaking and were added to `damage`.
    ///
    /// A non-positive input is retained here because Java adds it directly when `amount <= 0`.
    pub applied_damage: i32,
    /// Whether the resulting damage exceeded `ItemProps.max_damage` and the stack was broken.
    pub broke: bool,
}

/// The enchantment id stored by Java's `Enchantment.unbreaking`.
pub const UNBREAKING_ENCHANTMENT_ID: i16 = 34;

/// Java `EnchantmentHelper.getEnchantmentLevel` over already-decoded `{id, level}` pairs.
///
/// The helper returns the first matching entry and does not take the maximum or validate the
/// id.  NBT callers can decode their list with the numeric widening rules of their owning crate,
/// then pass the pairs here without duplicating that ordering rule.
#[must_use]
pub fn first_enchantment_level(enchantments: &[(i16, i16)], enchantment_id: i16) -> i16 {
    enchantments
        .iter()
        .find_map(|&(id, level)| (id == enchantment_id).then_some(level))
        .unwrap_or(0)
}

/// Returns the first Java Unbreaking level in an enchantment list.
#[must_use]
pub fn unbreaking_level(enchantments: &[(i16, i16)]) -> i16 {
    first_enchantment_level(enchantments, UNBREAKING_ENCHANTMENT_ID)
}

impl ItemProps {
    /// Java `Item.isItemTool(ItemStack)`, including the stack-sensitive `ItemBook` override.
    ///
    /// The registry row alone cannot answer this question: ordinary `Item` uses its maximum
    /// stack size and damageability, while `ItemBook` is a special non-damageable item that is a
    /// tool only when the concrete stack has count one.  Keep this separate from [`Self::is_tool`]
    /// because that older predicate intentionally has no stack/count input and is used for the
    /// broad held-item/tool classification.
    #[must_use]
    pub fn is_stack_tool(&self, stack_count: i32) -> bool {
        if self.name == "minecraft:book" {
            return stack_count == 1;
        }
        if self.name == "minecraft:enchanted_book" {
            return false;
        }
        self.max_stack_size == 1 && self.damageable
    }

    /// Java `ItemStack.isItemEnchantable` for a decoded stack.
    ///
    /// `has_enchantment_data` is the owning NBT layer's result for
    /// `ItemStack.isItemEnchanted()` (`stackTagCompound.hasKey("ench", 9)`).  The item-data
    /// crate deliberately does not depend on the NBT codec, but it still owns the exact virtual
    /// item/count gate needed by enchanting-table clients.  In particular, a single ordinary
    /// book is enchantable, a stacked book is not, and an item carrying enchantment data is not.
    #[must_use]
    pub fn is_stack_enchantable(&self, stack_count: i32, has_enchantment_data: bool) -> bool {
        self.is_stack_tool(stack_count) && !has_enchantment_data
    }

    /// Java `ItemStack.isItemStackDamageable`, using the stack's decoded `Unbreakable` value.
    ///
    /// The item must have a positive maximum damage and the numeric NBT `Unbreakable` value must
    /// be false.  The latter is supplied by the NBT owner because `mc-data` intentionally does
    /// not depend on the NBT codec.
    #[must_use]
    pub fn is_stack_damageable(&self, unbreakable: bool) -> bool {
        self.damageable && self.max_damage > 0 && !unbreakable
    }

    /// Java `ItemStack.attemptDamageItem`, including its exact Unbreaking random loop.
    ///
    /// The state is mutated only when the item is damageable and at least one point survives the
    /// Unbreaking loop.  A positive `amount` performs one `nextInt(level + 1)` per point for
    /// ordinary damageable items.  Armour first consumes one `nextFloat()` per point; a value
    /// below `0.6F` takes damage immediately, while values at or above that threshold then
    /// consume `nextInt(level + 1)`.  The returned boolean is Java's `itemDamage > maxDamage`
    /// result, so equality does not break the stack.
    pub fn attempt_damage_item<R: DurabilityRandom + ?Sized>(
        &self,
        state: &mut ItemStackDamageState,
        amount: i32,
        unbreaking_level: i16,
        unbreakable: bool,
        rng: &mut R,
    ) -> bool {
        self.attempt_damage_item_inner(state, amount, unbreaking_level, unbreakable, rng)
            .1
    }

    /// Java `ItemStack.damageItem`, returning a compact observation for callers that need to
    /// trigger break particles/statistics after the mutation.
    ///
    /// On a break this decrements `count`, clamps a negative count to zero, and resets `damage`
    /// to zero.  Those operations happen after the Java `attemptDamageItem` `>` test.  Creative
    /// mode is an entity/player guard outside this item-data seam and is therefore not inferred
    /// here.
    pub fn damage_item<R: DurabilityRandom + ?Sized>(
        &self,
        state: &mut ItemStackDamageState,
        amount: i32,
        unbreaking_level: i16,
        unbreakable: bool,
        rng: &mut R,
    ) -> ItemDamageResult {
        if !self.is_stack_damageable(unbreakable) {
            return ItemDamageResult::default();
        }

        let (applied_damage, broke) =
            self.attempt_damage_item_inner(state, amount, unbreaking_level, unbreakable, rng);
        if broke {
            // Java uses `--stackSize`, then clamps only when the result is negative.  Wrapping
            // subtraction preserves that int behaviour even for hostile synthetic states.
            state.count = state.count.wrapping_sub(1);
            if state.count < 0 {
                state.count = 0;
            }
            state.damage = 0;
        }
        ItemDamageResult {
            applied_damage,
            broke,
        }
    }

    fn attempt_damage_item_inner<R: DurabilityRandom + ?Sized>(
        &self,
        state: &mut ItemStackDamageState,
        amount: i32,
        unbreaking_level: i16,
        unbreakable: bool,
        rng: &mut R,
    ) -> (i32, bool) {
        if !self.is_stack_damageable(unbreakable) {
            return (0, false);
        }

        let mut effective = amount;
        if amount > 0 {
            let mut negated = 0;
            let level = i32::from(unbreaking_level);
            for _ in 0..amount {
                if level > 0 && unbreaking_negates_damage(self.kind == ItemKind::Armor, level, rng)
                {
                    negated += 1;
                }
            }
            effective -= negated;
            if effective <= 0 {
                return (0, false);
            }
        }

        // ItemStack.itemDamage is a Java int.  Rust's wrapping operation retains the result for
        // malformed values instead of making debug builds panic where the JVM wraps.
        state.damage = state.damage.wrapping_add(effective);
        (effective, state.damage > i32::from(self.max_damage))
    }

    /// Java `ItemFood.getHealAmount(stack)` / `getSaturationModifier(stack)`.
    ///
    /// Values for the normal metadata range are generated by invoking the virtual Java
    /// methods.  Out-of-range damage uses the same metadata-zero fallback that
    /// `ItemFishFood.FishType.byMetadata` applies; ordinary foods retain their row values.
    #[must_use]
    pub fn food_stats(&self, damage: i16) -> FoodStats {
        let damage = damage.max(0) as usize;
        if let Some((_, variants)) = FOOD_VARIANTS.iter().find(|(id, _)| *id == self.id) {
            return variants.get(damage).copied().unwrap_or(variants[0]);
        }
        FoodStats {
            heal: self.food_heal,
            saturation: self.food_saturation,
        }
    }

    /// The item accepted by Java's `ItemTool` / `ItemSword` / `ItemArmor.getIsRepairable`
    /// override, or `None` for item classes which retain base `Item`'s always-false result.
    #[must_use]
    pub fn repair_item(&self) -> Option<ItemId> {
        self.tool_material
            .map(ToolMaterial::repair_item)
            .or_else(|| self.armor_material.map(ArmorMaterial::repair_item))
    }

    /// Java `getIsRepairable(toRepair, repair)` for the material-backed item classes represented
    /// by [`ItemProps`]. Stack metadata and NBT do not participate in these 1.8.9 overrides.
    #[must_use]
    pub fn is_repairable_with(&self, repair: ItemId) -> bool {
        self.repair_item() == Some(repair)
    }

    /// Metadata-aware `Item.getUnlocalizedName(ItemStack)` result for this item and damage.
    ///
    /// This is deliberately separate from [`ItemProps::unlocalized`], which is the
    /// no-stack overload captured by the extractor. This resolver covers every 1.8.9
    /// `ItemMultiTexture` registration, plus the metadata-sensitive coloured-block, slab, wall,
    /// anvil, leaf, carpet, double-plant, coal, fish, dye, and skull families; other items retain
    /// that extracted base stem. These are the families whose generic base key is missing from
    /// the bundled 1.8.9 language table or whose variant name is user-visible in the same
    /// inventory paths.
    #[must_use]
    pub fn unlocalized_name(&self, damage: i16) -> &'static str {
        let damage = damage.max(0) as usize;

        // Every ItemMultiTexture registration in 1.8.9, plus ItemColored's subtype registration,
        // is kept in one data table. This is intentionally checked before the handful of
        // non-MultiTexture overrides below: the registration table is the source of truth for
        // the shared metadata-name functions, including their Java fallbacks.
        if let Some(name) = metadata_name_for(self.id.0, damage) {
            return name;
        }

        match self.id.0 {
            // ItemCloth + EnumDyeColor.byMetadata.  These are block-metadata ordering, unlike
            // ItemDye below, whose item-damage ordering runs in reverse.
            35 => by_metadata(damage, &WOOL_NAMES),

            // ItemLeaves delegates to BlockOldLeaf/BlockNewLeaf.getWoodType. Leaf state
            // flag bits do not participate in the item name.
            18 => OLD_LEAF_NAMES[damage & 3],

            // ItemSlab delegates to the single slab's getUnlocalizedName(meta).
            44 => by_metadata(damage, &STONE_SLAB_NAMES),
            126 => by_metadata(damage, &WOOD_SLAB_NAMES),

            // ItemAnvilBlock uses the damage stage in the item's display-name key.
            145 => by_metadata(damage, &ANVIL_NAMES),

            // BlockNewLeaf maps metadata 0/1 to planks metadata 4/5. Values 2/3
            // reach BlockPlanks.EnumType.byMetadata(6/7), whose vanilla fallback is oak.
            161 => NEW_LEAF_NAMES
                .get(damage & 3)
                .copied()
                .unwrap_or("tile.leaves.oak"),

            171 => by_metadata(damage, &CARPET_NAMES),

            // ItemDoublePlant is an ItemMultiTexture subclass.
            175 => by_metadata(damage, &DOUBLE_PLANT_NAMES),

            // 1.8.9 has only the red-sandstone member; every other metadata falls back
            // to that sole enum value.
            182 => "tile.stoneSlab2.red_sandstone",

            // ItemCoal uses metadata zero for coal and one for charcoal.
            263 => by_metadata(damage, &COAL_NAMES),

            349 => by_metadata(damage, &RAW_FISH_NAMES),
            350 => by_metadata(damage, &COOKED_FISH_NAMES),

            // ItemDye uses EnumDyeColor.byDyeDamage, whose ordering is the reverse of
            // block colour metadata.
            351 => by_metadata(damage, &DYE_NAMES),

            95 => by_metadata(damage, &STAINED_GLASS_NAMES),
            159 => by_metadata(damage, &STAINED_CLAY_NAMES),
            160 => by_metadata(damage, &STAINED_GLASS_PANE_NAMES),

            // ItemSkull.getUnlocalizedName(ItemStack). The player-owner NBT formatting
            // is a later getItemStackDisplayName override, not part of this method.
            397 => by_metadata(damage, &SKULL_NAMES),

            _ => self.unlocalized,
        }
    }

    /// Metadata-aware bundled `en_US` display name for this item.
    ///
    /// The normal item table stores one display name per registry id, which is enough for
    /// ordinary items but loses metadata-selected families such as `ItemCloth` and `ItemSkull`.
    /// Vanilla resolves those names from `getUnlocalizedName(ItemStack)` and the language table at
    /// draw time.  This allocation-free fallback is used by headless/no-language callers; the
    /// localized UI performs the same key selection through [`Self::unlocalized_name`] and its
    /// active language table.  Invalid metadata follows each Java variant table's metadata-zero
    /// fallback.
    #[must_use]
    pub fn display_name_for(&self, damage: i16) -> &'static str {
        let damage = damage.max(0) as usize;
        match self.id.0 {
            38 => by_metadata(damage, &RED_FLOWER_DISPLAY_NAMES),
            168 => by_metadata(damage, &PRISMARINE_DISPLAY_NAMES),
            179 => by_metadata(damage, &RED_SANDSTONE_DISPLAY_NAMES),
            263 => by_metadata(damage, &COAL_DISPLAY_NAMES),
            145 => by_metadata(damage, &ANVIL_DISPLAY_NAMES),
            35 => by_metadata(damage, &WOOL_DISPLAY_NAMES),
            95 => by_metadata(damage, &STAINED_GLASS_DISPLAY_NAMES),
            159 => by_metadata(damage, &STAINED_CLAY_DISPLAY_NAMES),
            160 => by_metadata(damage, &STAINED_GLASS_PANE_DISPLAY_NAMES),
            171 => by_metadata(damage, &CARPET_DISPLAY_NAMES),
            // ItemSkull.getItemStackDisplayName resolves the metadata-specific key selected by
            // getUnlocalizedName(ItemStack).  Player-owner formatting is an NBT concern owned by
            // mc-ui; this table covers the five ordinary skull types and Java's metadata-zero
            // fallback for hostile values.
            397 => by_metadata(damage, &SKULL_DISPLAY_NAMES),
            _ => self.display_name,
        }
    }

    /// True when the item is a tool, sword or shears.
    #[inline]
    #[must_use]
    pub fn is_tool(&self) -> bool {
        matches!(
            self.kind,
            ItemKind::Pickaxe
                | ItemKind::Axe
                | ItemKind::Shovel
                | ItemKind::Hoe
                | ItemKind::Sword
                | ItemKind::Shears
        )
    }

    /// True when the item can be eaten.
    #[inline]
    #[must_use]
    pub fn is_food(&self) -> bool {
        matches!(self.kind, ItemKind::Food)
    }
}

#[inline]
fn by_metadata(damage: usize, names: &'static [&'static str]) -> &'static str {
    names.get(damage).copied().unwrap_or(names[0])
}

/// One metadata-aware item-name registration from Java's `Items.registerItems`.
///
/// `ItemMultiTexture(String[])` clamps an invalid metadata index to zero.  The function-backed
/// registrations in this table use the same shape after extracting their vanilla enum's
/// unlocalized suffixes.  `fallback` is separate from `names[0]` for `log2`: Java adds four to
/// the metadata before looking up the `BlockPlanks` enum, so metadata two and above resolve to
/// oak rather than acacia.
#[derive(Debug, Clone, Copy)]
struct MetadataNameTable {
    item_id: u16,
    names: &'static [&'static str],
    fallback: &'static str,
}

const ITEM_METADATA_NAMES: &[MetadataNameTable] = &[
    MetadataNameTable {
        item_id: 1,
        names: &STONE_NAMES,
        fallback: "tile.stone.stone",
    },
    MetadataNameTable {
        item_id: 3,
        names: &DIRT_NAMES,
        fallback: "tile.dirt.default",
    },
    MetadataNameTable {
        item_id: 5,
        names: &PLANKS_NAMES,
        fallback: "tile.wood.oak",
    },
    MetadataNameTable {
        item_id: 6,
        names: &SAPLING_NAMES,
        fallback: "tile.sapling.oak",
    },
    MetadataNameTable {
        item_id: 12,
        names: &SAND_NAMES,
        fallback: "tile.sand.default",
    },
    MetadataNameTable {
        item_id: 17,
        names: &LOG_NAMES,
        fallback: "tile.log.oak",
    },
    MetadataNameTable {
        item_id: 19,
        names: &SPONGE_NAMES,
        fallback: "tile.sponge.dry",
    },
    MetadataNameTable {
        item_id: 24,
        names: &SANDSTONE_NAMES,
        fallback: "tile.sandStone.default",
    },
    MetadataNameTable {
        item_id: 31,
        names: &TALLGRASS_NAMES,
        // ItemColored returns the base key when its subtype-name array is out of range.
        fallback: "tile.tallgrass",
    },
    MetadataNameTable {
        item_id: 37,
        names: &YELLOW_FLOWER_NAMES,
        fallback: "tile.flower1.dandelion",
    },
    MetadataNameTable {
        item_id: 38,
        names: &RED_FLOWER_NAMES,
        fallback: "tile.flower2.poppy",
    },
    MetadataNameTable {
        item_id: 97,
        names: &MONSTER_EGG_NAMES,
        fallback: "tile.monsterStoneEgg.stone",
    },
    MetadataNameTable {
        item_id: 98,
        names: &STONEBRICK_NAMES,
        fallback: "tile.stonebricksmooth.default",
    },
    MetadataNameTable {
        item_id: 139,
        names: &COBBLESTONE_WALL_NAMES,
        fallback: "tile.cobbleWall.normal",
    },
    MetadataNameTable {
        item_id: 155,
        names: &QUARTZ_NAMES,
        fallback: "tile.quartzBlock.default",
    },
    MetadataNameTable {
        item_id: 162,
        names: &LOG2_NAMES,
        fallback: "tile.log.oak",
    },
    MetadataNameTable {
        item_id: 168,
        names: &PRISMARINE_NAMES,
        fallback: "tile.prismarine.rough",
    },
    MetadataNameTable {
        item_id: 179,
        names: &RED_SANDSTONE_NAMES,
        fallback: "tile.redSandStone.default",
    },
];

#[inline]
fn metadata_name_for(item_id: u16, damage: usize) -> Option<&'static str> {
    ITEM_METADATA_NAMES
        .iter()
        .find(|table| table.item_id == item_id)
        .map(|table| table.names.get(damage).copied().unwrap_or(table.fallback))
}

/// Java `EnchantmentDurability.negateDamage`, with the caller-provided Java RNG stream.
fn unbreaking_negates_damage<R: DurabilityRandom + ?Sized>(
    armor: bool,
    level: i32,
    rng: &mut R,
) -> bool {
    // Java's short-circuiting `?:` matters: an armour roll below 0.6F does not consume the
    // following nextInt call.  For all other cases the result is a uniform bounded integer > 0.
    if armor && rng.next_f32() < 0.6 {
        false
    } else {
        rng.next_i32_bound(level + 1) > 0
    }
}

const STONE_NAMES: [&str; 7] = [
    "tile.stone.stone",
    "tile.stone.granite",
    "tile.stone.graniteSmooth",
    "tile.stone.diorite",
    "tile.stone.dioriteSmooth",
    "tile.stone.andesite",
    "tile.stone.andesiteSmooth",
];
const DIRT_NAMES: [&str; 3] = ["tile.dirt.default", "tile.dirt.coarse", "tile.dirt.podzol"];
const PLANKS_NAMES: [&str; 6] = [
    "tile.wood.oak",
    "tile.wood.spruce",
    "tile.wood.birch",
    "tile.wood.jungle",
    "tile.wood.acacia",
    "tile.wood.big_oak",
];
const SAPLING_NAMES: [&str; 6] = [
    "tile.sapling.oak",
    "tile.sapling.spruce",
    "tile.sapling.birch",
    "tile.sapling.jungle",
    "tile.sapling.acacia",
    "tile.sapling.big_oak",
];
const SAND_NAMES: [&str; 2] = ["tile.sand.default", "tile.sand.red"];
const LOG_NAMES: [&str; 6] = [
    "tile.log.oak",
    "tile.log.spruce",
    "tile.log.birch",
    "tile.log.jungle",
    "tile.log.acacia",
    "tile.log.big_oak",
];
const SPONGE_NAMES: [&str; 2] = ["tile.sponge.dry", "tile.sponge.wet"];
const SANDSTONE_NAMES: [&str; 3] = [
    "tile.sandStone.default",
    "tile.sandStone.chiseled",
    "tile.sandStone.smooth",
];
const TALLGRASS_NAMES: [&str; 3] = [
    "tile.tallgrass.shrub",
    "tile.tallgrass.grass",
    "tile.tallgrass.fern",
];
const WOOL_NAMES: [&str; 16] = [
    "tile.cloth.white",
    "tile.cloth.orange",
    "tile.cloth.magenta",
    "tile.cloth.lightBlue",
    "tile.cloth.yellow",
    "tile.cloth.lime",
    "tile.cloth.pink",
    "tile.cloth.gray",
    "tile.cloth.silver",
    "tile.cloth.cyan",
    "tile.cloth.purple",
    "tile.cloth.blue",
    "tile.cloth.brown",
    "tile.cloth.green",
    "tile.cloth.red",
    "tile.cloth.black",
];
const WOOL_DISPLAY_NAMES: [&str; 16] = [
    "Wool",
    "Orange Wool",
    "Magenta Wool",
    "Light Blue Wool",
    "Yellow Wool",
    "Lime Wool",
    "Pink Wool",
    "Gray Wool",
    "Light Gray Wool",
    "Cyan Wool",
    "Purple Wool",
    "Blue Wool",
    "Brown Wool",
    "Green Wool",
    "Red Wool",
    "Black Wool",
];
const OLD_LEAF_NAMES: [&str; 4] = [
    "tile.leaves.oak",
    "tile.leaves.spruce",
    "tile.leaves.birch",
    "tile.leaves.jungle",
];
const NEW_LEAF_NAMES: [&str; 2] = ["tile.leaves.acacia", "tile.leaves.big_oak"];
const STONE_SLAB_NAMES: [&str; 8] = [
    "tile.stoneSlab.stone",
    "tile.stoneSlab.sand",
    "tile.stoneSlab.wood",
    "tile.stoneSlab.cobble",
    "tile.stoneSlab.brick",
    "tile.stoneSlab.smoothStoneBrick",
    "tile.stoneSlab.netherBrick",
    "tile.stoneSlab.quartz",
];
const WOOD_SLAB_NAMES: [&str; 6] = [
    "tile.woodSlab.oak",
    "tile.woodSlab.spruce",
    "tile.woodSlab.birch",
    "tile.woodSlab.jungle",
    "tile.woodSlab.acacia",
    "tile.woodSlab.big_oak",
];
const COBBLESTONE_WALL_NAMES: [&str; 2] = ["tile.cobbleWall.normal", "tile.cobbleWall.mossy"];
const YELLOW_FLOWER_NAMES: [&str; 1] = ["tile.flower1.dandelion"];
const MONSTER_EGG_NAMES: [&str; 6] = [
    "tile.monsterStoneEgg.stone",
    "tile.monsterStoneEgg.cobble",
    "tile.monsterStoneEgg.brick",
    "tile.monsterStoneEgg.mossybrick",
    "tile.monsterStoneEgg.crackedbrick",
    "tile.monsterStoneEgg.chiseledbrick",
];
const STONEBRICK_NAMES: [&str; 4] = [
    "tile.stonebricksmooth.default",
    "tile.stonebricksmooth.mossy",
    "tile.stonebricksmooth.cracked",
    "tile.stonebricksmooth.chiseled",
];
const QUARTZ_NAMES: [&str; 3] = [
    "tile.quartzBlock.default",
    "tile.quartzBlock.chiseled",
    "tile.quartzBlock.lines",
];
const LOG2_NAMES: [&str; 2] = ["tile.log.acacia", "tile.log.big_oak"];
const STAINED_GLASS_NAMES: [&str; 16] = [
    "tile.stainedGlass.white",
    "tile.stainedGlass.orange",
    "tile.stainedGlass.magenta",
    "tile.stainedGlass.lightBlue",
    "tile.stainedGlass.yellow",
    "tile.stainedGlass.lime",
    "tile.stainedGlass.pink",
    "tile.stainedGlass.gray",
    "tile.stainedGlass.silver",
    "tile.stainedGlass.cyan",
    "tile.stainedGlass.purple",
    "tile.stainedGlass.blue",
    "tile.stainedGlass.brown",
    "tile.stainedGlass.green",
    "tile.stainedGlass.red",
    "tile.stainedGlass.black",
];
const STAINED_GLASS_DISPLAY_NAMES: [&str; 16] = [
    "White Stained Glass",
    "Orange Stained Glass",
    "Magenta Stained Glass",
    "Light Blue Stained Glass",
    "Yellow Stained Glass",
    "Lime Stained Glass",
    "Pink Stained Glass",
    "Gray Stained Glass",
    "Light Gray Stained Glass",
    "Cyan Stained Glass",
    "Purple Stained Glass",
    "Blue Stained Glass",
    "Brown Stained Glass",
    "Green Stained Glass",
    "Red Stained Glass",
    "Black Stained Glass",
];
const ANVIL_NAMES: [&str; 3] = [
    "tile.anvil.intact",
    "tile.anvil.slightlyDamaged",
    "tile.anvil.veryDamaged",
];
const ANVIL_DISPLAY_NAMES: [&str; 3] = ["Anvil", "Slightly Damaged Anvil", "Very Damaged Anvil"];
const STAINED_CLAY_NAMES: [&str; 16] = [
    "tile.clayHardenedStained.white",
    "tile.clayHardenedStained.orange",
    "tile.clayHardenedStained.magenta",
    "tile.clayHardenedStained.lightBlue",
    "tile.clayHardenedStained.yellow",
    "tile.clayHardenedStained.lime",
    "tile.clayHardenedStained.pink",
    "tile.clayHardenedStained.gray",
    "tile.clayHardenedStained.silver",
    "tile.clayHardenedStained.cyan",
    "tile.clayHardenedStained.purple",
    "tile.clayHardenedStained.blue",
    "tile.clayHardenedStained.brown",
    "tile.clayHardenedStained.green",
    "tile.clayHardenedStained.red",
    "tile.clayHardenedStained.black",
];
const STAINED_CLAY_DISPLAY_NAMES: [&str; 16] = [
    "White Stained Clay",
    "Orange Stained Clay",
    "Magenta Stained Clay",
    "Light Blue Stained Clay",
    "Yellow Stained Clay",
    "Lime Stained Clay",
    "Pink Stained Clay",
    "Gray Stained Clay",
    "Light Gray Stained Clay",
    "Cyan Stained Clay",
    "Purple Stained Clay",
    "Blue Stained Clay",
    "Brown Stained Clay",
    "Green Stained Clay",
    "Red Stained Clay",
    "Black Stained Clay",
];
const STAINED_GLASS_PANE_NAMES: [&str; 16] = [
    "tile.thinStainedGlass.white",
    "tile.thinStainedGlass.orange",
    "tile.thinStainedGlass.magenta",
    "tile.thinStainedGlass.lightBlue",
    "tile.thinStainedGlass.yellow",
    "tile.thinStainedGlass.lime",
    "tile.thinStainedGlass.pink",
    "tile.thinStainedGlass.gray",
    "tile.thinStainedGlass.silver",
    "tile.thinStainedGlass.cyan",
    "tile.thinStainedGlass.purple",
    "tile.thinStainedGlass.blue",
    "tile.thinStainedGlass.brown",
    "tile.thinStainedGlass.green",
    "tile.thinStainedGlass.red",
    "tile.thinStainedGlass.black",
];
const STAINED_GLASS_PANE_DISPLAY_NAMES: [&str; 16] = [
    "White Stained Glass Pane",
    "Orange Stained Glass Pane",
    "Magenta Stained Glass Pane",
    "Light Blue Stained Glass Pane",
    "Yellow Stained Glass Pane",
    "Lime Stained Glass Pane",
    "Pink Stained Glass Pane",
    "Gray Stained Glass Pane",
    "Light Gray Stained Glass Pane",
    "Cyan Stained Glass Pane",
    "Purple Stained Glass Pane",
    "Blue Stained Glass Pane",
    "Brown Stained Glass Pane",
    "Green Stained Glass Pane",
    "Red Stained Glass Pane",
    "Black Stained Glass Pane",
];
const PRISMARINE_NAMES: [&str; 3] = [
    "tile.prismarine.rough",
    "tile.prismarine.bricks",
    "tile.prismarine.dark",
];
const PRISMARINE_DISPLAY_NAMES: [&str; 3] = ["Prismarine", "Prismarine Bricks", "Dark Prismarine"];
const RED_FLOWER_NAMES: [&str; 9] = [
    "tile.flower2.poppy",
    "tile.flower2.blueOrchid",
    "tile.flower2.allium",
    "tile.flower2.houstonia",
    "tile.flower2.tulipRed",
    "tile.flower2.tulipOrange",
    "tile.flower2.tulipWhite",
    "tile.flower2.tulipPink",
    "tile.flower2.oxeyeDaisy",
];
const RED_FLOWER_DISPLAY_NAMES: [&str; 9] = [
    "Poppy",
    "Blue Orchid",
    "Allium",
    "Azure Bluet",
    "Red Tulip",
    "Orange Tulip",
    "White Tulip",
    "Pink Tulip",
    "Oxeye Daisy",
];
const RED_SANDSTONE_NAMES: [&str; 3] = [
    "tile.redSandStone.default",
    "tile.redSandStone.chiseled",
    "tile.redSandStone.smooth",
];
const RED_SANDSTONE_DISPLAY_NAMES: [&str; 3] = [
    "Red Sandstone",
    "Chiseled Red Sandstone",
    "Smooth Red Sandstone",
];
const CARPET_NAMES: [&str; 16] = [
    "tile.woolCarpet.white",
    "tile.woolCarpet.orange",
    "tile.woolCarpet.magenta",
    "tile.woolCarpet.lightBlue",
    "tile.woolCarpet.yellow",
    "tile.woolCarpet.lime",
    "tile.woolCarpet.pink",
    "tile.woolCarpet.gray",
    "tile.woolCarpet.silver",
    "tile.woolCarpet.cyan",
    "tile.woolCarpet.purple",
    "tile.woolCarpet.blue",
    "tile.woolCarpet.brown",
    "tile.woolCarpet.green",
    "tile.woolCarpet.red",
    "tile.woolCarpet.black",
];
const CARPET_DISPLAY_NAMES: [&str; 16] = [
    "Carpet",
    "Orange Carpet",
    "Magenta Carpet",
    "Light Blue Carpet",
    "Yellow Carpet",
    "Lime Carpet",
    "Pink Carpet",
    "Gray Carpet",
    "Light Gray Carpet",
    "Cyan Carpet",
    "Purple Carpet",
    "Blue Carpet",
    "Brown Carpet",
    "Green Carpet",
    "Red Carpet",
    "Black Carpet",
];
const DOUBLE_PLANT_NAMES: [&str; 6] = [
    "tile.doublePlant.sunflower",
    "tile.doublePlant.syringa",
    "tile.doublePlant.grass",
    "tile.doublePlant.fern",
    "tile.doublePlant.rose",
    "tile.doublePlant.paeonia",
];
const RAW_FISH_NAMES: [&str; 4] = [
    "item.fish.cod.raw",
    "item.fish.salmon.raw",
    "item.fish.clownfish.raw",
    "item.fish.pufferfish.raw",
];
const COOKED_FISH_NAMES: [&str; 4] = [
    "item.fish.cod.cooked",
    "item.fish.salmon.cooked",
    // Clownfish and pufferfish cannot be cooked. Vanilla deliberately retains their
    // raw keys even in an otherwise-impossible cooked_fish stack.
    "item.fish.clownfish.raw",
    "item.fish.pufferfish.raw",
];
const DYE_NAMES: [&str; 16] = [
    "item.dyePowder.black",
    "item.dyePowder.red",
    "item.dyePowder.green",
    "item.dyePowder.brown",
    "item.dyePowder.blue",
    "item.dyePowder.purple",
    "item.dyePowder.cyan",
    "item.dyePowder.silver",
    "item.dyePowder.gray",
    "item.dyePowder.pink",
    "item.dyePowder.lime",
    "item.dyePowder.yellow",
    "item.dyePowder.lightBlue",
    "item.dyePowder.magenta",
    "item.dyePowder.orange",
    "item.dyePowder.white",
];
const COAL_NAMES: [&str; 2] = ["item.coal", "item.charcoal"];
const COAL_DISPLAY_NAMES: [&str; 2] = ["Coal", "Charcoal"];
const SKULL_NAMES: [&str; 5] = [
    "item.skull.skeleton",
    "item.skull.wither",
    "item.skull.zombie",
    "item.skull.char",
    "item.skull.creeper",
];
const SKULL_DISPLAY_NAMES: [&str; 5] = [
    "Skeleton Skull",
    "Wither Skeleton Skull",
    "Zombie Head",
    "Head",
    "Creeper Head",
];

/// The row returned for ids vanilla does not register. Total lookups need a value to
/// return, and a server can put any id in an `ItemStack`.
static UNKNOWN_ITEM: ItemProps = ItemProps {
    id: ItemId(0),
    name: "",
    unlocalized: "",
    display_name: "",
    registered: false,
    max_stack_size: 64,
    max_damage: 0,
    damageable: false,
    has_subtypes: false,
    enchantability: 0,
    full_3d: false,
    kind: ItemKind::Generic,
    tool_material: None,
    armor_material: None,
    armor_slot: None,
    armor_reduction: 0,
    attack_damage: 0.0,
    efficiency: 1.0,
    food_heal: 0,
    food_saturation: 0.0,
    block: None,
    creative_tab: None,
    java_class: "",
};

/// Properties for an item id. Frozen S01 compatibility API.
///
/// Total and allocation-free: unregistered ids return a sentinel with `registered: false`
/// instead of panicking, because item ids arrive from the network inside `ItemStack`s.
///
/// # Example
/// ```
/// use mc_data::{item_props, ItemId};
/// let sword = item_props(ItemId(276)); // diamond_sword
/// assert_eq!(sword.max_stack_size, 1);
/// assert!(sword.damageable);
/// assert!(!item_props(ItemId(60000)).registered);
/// ```
#[inline]
#[must_use]
pub fn item_props(id: ItemId) -> &'static ItemProps {
    try_item_props(id).unwrap_or(&UNKNOWN_ITEM)
}

/// Looks up properties for a registered item id.
///
/// Java: `Item.getItemById`. The registry returns `null` for an unregistered id; this API
/// represents that faithfully as `None`. Use [`item_props`] only at an explicit wire-safety
/// boundary which deliberately wants the historical unregistered compatibility sentinel.
///
/// # Example
/// ```
/// use mc_data::{try_item_props, ItemId};
///
/// assert_eq!(try_item_props(ItemId(276)).map(|item| item.name), Some("minecraft:diamond_sword"));
/// assert_eq!(try_item_props(ItemId(60000)), None);
/// ```
#[inline]
#[must_use]
pub fn try_item_props(id: ItemId) -> Option<&'static ItemProps> {
    let slot = *ITEM_BY_ID.get(usize::from(id.0))?;
    ITEMS.get(usize::from(slot))
}

/// Looks up an item by registry name, with or without the `minecraft:` namespace, then falls
/// back to a decimal numeric registry id.
///
/// This deliberately consults only the item registry.  A block/item registry name is not a
/// universal identity: sixteen vanilla names resolve in both registries with different numeric
/// ids (for example, `bed` is item 355 but block 26).  Call [`crate::block_by_name`] when the
/// Java call site is resolving a `Block`.
///
/// java: `Item.getByNameOrId` first constructs a `ResourceLocation`; only when that registry
/// lookup misses does it pass the original, unqualified string to `Integer.parseInt` and
/// `getItemById`. Consequently `"1"` resolves to stone while `"minecraft:1"` does not.
#[must_use]
pub fn item_by_name(name: &str) -> Option<ItemId> {
    // `minecraft:` is implied for an unqualified name; compared in place, without a `format!`.
    ITEM_NAMES
        .binary_search_by(|(n, _)| crate::block::cmp_registry_name(n, name))
        .ok()
        .map(|i| ItemId(ITEM_NAMES[i].1))
        .or_else(|| {
            // Parse as Java's signed `int` before narrowing to our non-negative registry-id
            // type. This rejects negative and >i32 values exactly where vanilla either gets a
            // null registry row or catches `NumberFormatException`.
            let numeric = name.parse::<i32>().ok()?;
            let id = ItemId(u16::try_from(numeric).ok()?);
            try_item_props(id).map(|_| id)
        })
}

/// Every registered item, in registry-id order.
#[must_use]
pub fn items() -> &'static [ItemProps] {
    &ITEMS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct ScriptedRandom {
        floats: Vec<f32>,
        ints: Vec<i32>,
        float_calls: usize,
        int_calls: Vec<i32>,
    }

    impl DurabilityRandom for ScriptedRandom {
        fn next_f32(&mut self) -> f32 {
            let value = self.floats[self.float_calls];
            self.float_calls += 1;
            value
        }

        fn next_i32_bound(&mut self, bound: i32) -> i32 {
            self.int_calls.push(bound);
            self.ints[self.int_calls.len() - 1]
        }
    }

    #[test]
    fn every_u16_id_resolves_without_panic() {
        for id in 0..=u16::MAX {
            let p = item_props(ItemId(id));
            assert!(p.max_stack_size > 0);
        }
    }

    #[test]
    fn strict_registry_lookup_does_not_fabricate_unknown_items() {
        let sword = try_item_props(ItemId(276)).expect("diamond sword is registered");
        assert_eq!(sword.id, ItemId(276));
        assert_eq!(sword.name, "minecraft:diamond_sword");

        // 0 is a gap in the 1.8.9 item registry, 8 is an in-range gap, and 60000 is beyond
        // the generated lookup table. All three are Java-null registry lookups.
        for id in [0, 8, 60_000] {
            assert_eq!(try_item_props(ItemId(id)), None, "id {id}");
            assert!(
                !item_props(ItemId(id)).registered,
                "the explicit compatibility wrapper remains safe for wire callers"
            );
        }
    }

    #[test]
    fn known_items() {
        let sword = item_by_name("diamond_sword").expect("diamond_sword");
        let p = item_props(sword);
        assert_eq!(p.max_stack_size, 1);
        assert!(p.damageable);
        assert_eq!(p.max_damage, 1561);
        assert_eq!(p.kind, ItemKind::Sword);
        assert_eq!(p.tool_material, Some(ToolMaterial::Emerald));

        let apple = item_props(item_by_name("apple").expect("apple"));
        assert!(apple.is_food());
        assert_eq!(apple.food_heal, 4);

        let helmet = item_props(item_by_name("diamond_helmet").expect("diamond_helmet"));
        assert_eq!(helmet.kind, ItemKind::Armor);
        assert_eq!(helmet.armor_material, Some(ArmorMaterial::Diamond));
        assert_eq!(helmet.armor_slot, Some(0), "armorType 0 is the helmet");
        assert_eq!(helmet.armor_reduction, 3);
        assert_eq!(helmet.max_damage, 363, "11 * DIAMOND.maxDamageFactor(33)");
        assert_eq!(
            helmet.tool_material, None,
            "armour must not borrow a ToolMaterial"
        );
    }

    #[test]
    fn stack_tool_and_enchantability_preserve_itembook_override() {
        let book = item_props(item_by_name("book").expect("book"));
        assert_eq!(book.enchantability, 1);
        assert!(book.is_stack_tool(1));
        assert!(book.is_stack_enchantable(1, false));
        assert!(!book.is_stack_tool(2));
        assert!(!book.is_stack_enchantable(2, false));
        assert!(!book.is_stack_enchantable(1, true));

        let enchanted_book = item_props(item_by_name("enchanted_book").expect("enchanted_book"));
        assert!(!enchanted_book.is_stack_tool(1));
        assert!(!enchanted_book.is_stack_enchantable(1, false));
    }

    #[test]
    fn stack_tool_and_enchantability_keep_ordinary_item_rules() {
        let sword = item_props(item_by_name("diamond_sword").expect("diamond_sword"));
        assert!(sword.is_stack_tool(1));
        assert!(sword.is_stack_enchantable(1, false));
        assert!(!sword.is_stack_enchantable(1, true));
        assert!(sword.is_stack_tool(2));

        let apple = item_props(item_by_name("apple").expect("apple"));
        assert!(!apple.is_stack_tool(1));
        assert!(!apple.is_stack_enchantable(1, false));
    }

    #[test]
    fn itembook_tool_gate_keeps_stack_counts_and_registry_boundaries_exact() {
        let book = item_props(item_by_name("book").expect("book"));
        assert!(book.is_stack_tool(1));
        for count in [0, -1, 2, 64] {
            assert!(
                !book.is_stack_tool(count),
                "book count {count} must not be a tool"
            );
            assert!(
                !book.is_stack_enchantable(count, false),
                "book count {count} must not be enchantable"
            );
        }

        // ItemEnchantedBook overrides Item.isItemTool to false, while a written book is merely
        // a non-damageable ItemEditableBook and therefore also fails Item.isItemTool.
        for name in ["enchanted_book", "written_book", "iron_horse_armor"] {
            let item = item_props(item_by_name(name).expect("registered item"));
            assert!(!item.is_stack_tool(1), "{name} must not be a tool");
            assert!(
                !item.is_stack_enchantable(1, false),
                "{name} must not be enchantable"
            );
        }

        // Java's registry lookup is nullable; the compatibility sentinel must not accidentally
        // become enchantable through the ALL target or a name-based special case.
        let unknown = item_props(ItemId(u16::MAX));
        assert!(!unknown.registered);
        assert!(!unknown.is_stack_tool(1));
        assert!(!unknown.is_stack_enchantable(1, false));
    }

    #[test]
    fn itemstack_damage_uses_strict_java_break_threshold_and_resets_after_break() {
        let sword = item_props(item_by_name("diamond_sword").expect("diamond sword"));
        let max = i32::from(sword.max_damage);
        let mut rng = ScriptedRandom::default();

        let mut at_limit = ItemStackDamageState::new(1, max - 1);
        let result = sword.damage_item(&mut at_limit, 1, 0, false, &mut rng);
        assert_eq!(
            result,
            ItemDamageResult {
                applied_damage: 1,
                broke: false,
            }
        );
        assert_eq!(at_limit, ItemStackDamageState::new(1, max));

        let mut over_limit = ItemStackDamageState::new(1, max);
        let result = sword.damage_item(&mut over_limit, 1, 0, false, &mut rng);
        assert_eq!(
            result,
            ItemDamageResult {
                applied_damage: 1,
                broke: true,
            }
        );
        assert_eq!(
            over_limit,
            ItemStackDamageState::new(0, 0),
            "Java resets itemDamage after decrementing and clamping stackSize"
        );

        // A plausible `>= maxDamage` mutant would break the first stack, while a missing reset
        // would leave the second stack at max+1.  Both outcomes are intentionally asserted.
    }

    #[test]
    fn itemstack_damage_matches_unbreaking_per_point_rng_and_armor_branch() {
        let pick = item_props(item_by_name("diamond_pickaxe").expect("diamond pickaxe"));
        let mut ordinary_rng = ScriptedRandom {
            ints: vec![0, 1, 2],
            ..ScriptedRandom::default()
        };
        let mut ordinary = ItemStackDamageState::new(1, 0);
        let result = pick.damage_item(&mut ordinary, 3, 2, false, &mut ordinary_rng);
        assert_eq!(
            result,
            ItemDamageResult {
                applied_damage: 1,
                broke: false,
            }
        );
        assert_eq!(ordinary.damage, 1);
        assert_eq!(ordinary_rng.float_calls, 0);
        assert_eq!(ordinary_rng.int_calls, vec![3, 3, 3]);

        let helmet = item_props(item_by_name("diamond_helmet").expect("diamond helmet"));
        let mut armor_rng = ScriptedRandom {
            floats: vec![0.599_999_96, 0.6],
            ints: vec![1],
            ..ScriptedRandom::default()
        };
        let mut armor = ItemStackDamageState::new(1, 0);
        let result = helmet.damage_item(&mut armor, 2, 3, false, &mut armor_rng);
        assert_eq!(
            result,
            ItemDamageResult {
                applied_damage: 1,
                broke: false,
            }
        );
        assert_eq!(armor.damage, 1);
        assert_eq!(armor_rng.float_calls, 2);
        assert_eq!(
            armor_rng.int_calls,
            vec![4],
            "armour's <0.6F branch consumes no nextInt"
        );

        // These controls bite a single-roll implementation, a level-as-bound implementation,
        // the wrong `>= 0.6F` threshold, and an armor path that consumes nextInt too early.
    }

    #[test]
    fn itemstack_damage_preserves_java_nonpositive_and_unbreakable_noop_paths() {
        let pick = item_props(item_by_name("diamond_pickaxe").expect("diamond pickaxe"));
        let mut rng = ScriptedRandom::default();
        let mut state = ItemStackDamageState::new(1, 10);

        let result = pick.damage_item(&mut state, -3, 3, false, &mut rng);
        assert_eq!(
            result,
            ItemDamageResult {
                applied_damage: -3,
                broke: false,
            }
        );
        assert_eq!(state.damage, 7);
        assert_eq!(rng.float_calls, 0);
        assert!(rng.int_calls.is_empty());

        let before = state;
        let result = pick.damage_item(&mut state, 7, 3, true, &mut rng);
        assert_eq!(result, ItemDamageResult::default());
        assert_eq!(state, before, "Unbreakable bypasses both damage and RNG");

        let apple = item_props(item_by_name("apple").expect("apple"));
        assert!(!apple.is_stack_damageable(false));
        let mut food = ItemStackDamageState::new(12, 0);
        assert_eq!(
            apple.damage_item(&mut food, 1, 3, false, &mut rng),
            ItemDamageResult::default()
        );
        assert_eq!(food, ItemStackDamageState::new(12, 0));
    }

    #[test]
    fn unbreaking_lookup_keeps_java_first_matching_entry_and_signed_level() {
        let entries = [(34, 1), (16, 5), (34, 3)];
        assert_eq!(first_enchantment_level(&entries, 34), 1);
        assert_eq!(unbreaking_level(&entries), 1);
        assert_eq!(first_enchantment_level(&entries, 16), 5);
        assert_eq!(first_enchantment_level(&entries, 99), 0);

        let negative = [(34, -1)];
        let pick = item_props(item_by_name("diamond_pickaxe").expect("diamond pickaxe"));
        let mut rng = ScriptedRandom::default();
        let mut state = ItemStackDamageState::new(1, 0);
        pick.damage_item(&mut state, 2, unbreaking_level(&negative), false, &mut rng);
        assert_eq!(
            state.damage, 2,
            "non-positive Java enchantment levels skip RNG"
        );
        assert!(rng.int_calls.is_empty());
    }

    #[test]
    fn fish_food_stats_follow_the_virtual_itemstack_overrides() {
        let raw = item_props(item_by_name("fish").expect("raw fish"));
        let cooked = item_props(item_by_name("cooked_fish").expect("cooked fish"));

        assert_eq!(
            raw.food_stats(0),
            FoodStats {
                heal: 2,
                saturation: 0.1
            }
        );
        assert_eq!(
            raw.food_stats(1),
            FoodStats {
                heal: 2,
                saturation: 0.1
            }
        );
        assert_eq!(
            raw.food_stats(2),
            FoodStats {
                heal: 1,
                saturation: 0.1
            }
        );
        assert_eq!(
            raw.food_stats(3),
            FoodStats {
                heal: 1,
                saturation: 0.1
            }
        );
        assert_eq!(
            cooked.food_stats(0),
            FoodStats {
                heal: 5,
                saturation: 0.6
            }
        );
        assert_eq!(
            cooked.food_stats(1),
            FoodStats {
                heal: 6,
                saturation: 0.8
            }
        );
        // `FishType.byMetadata` falls back to cod, and non-cookable fish retain raw values.
        assert_eq!(
            cooked.food_stats(2),
            FoodStats {
                heal: 1,
                saturation: 0.1
            }
        );
        assert_eq!(
            cooked.food_stats(3),
            FoodStats {
                heal: 1,
                saturation: 0.1
            }
        );
        assert_eq!(raw.food_stats(16), raw.food_stats(0));
        assert_eq!(cooked.food_stats(-1), cooked.food_stats(0));
    }

    /// java: `Item.getByNameOrId`'s fallback covers every registered numeric id, including the
    /// large record-id tail. Removing the fallback makes this oracle fail on the first row.
    #[test]
    fn numeric_item_names_resolve_every_registered_id() {
        for props in items() {
            assert_eq!(
                item_by_name(&props.id.0.to_string()),
                Some(props.id),
                "decimal id {}",
                props.id.0
            );
        }
    }

    #[test]
    fn every_item_record_has_its_exact_constructor_name_and_no_other_item_does() {
        let expected = [
            (2256, "13", "C418 - 13"),
            (2257, "cat", "C418 - cat"),
            (2258, "blocks", "C418 - blocks"),
            (2259, "chirp", "C418 - chirp"),
            (2260, "far", "C418 - far"),
            (2261, "mall", "C418 - mall"),
            (2262, "mellohi", "C418 - mellohi"),
            (2263, "stal", "C418 - stal"),
            (2264, "strad", "C418 - strad"),
            (2265, "ward", "C418 - ward"),
            (2266, "11", "C418 - 11"),
            (2267, "wait", "C418 - wait"),
        ];
        for (id, name, description) in expected {
            let id = ItemId(id);
            let record = record_info(id).expect("registered disc");
            assert_eq!(record.record_name, name, "item {id:?}");
            assert_eq!(record.description_en_us, description, "item {id:?}");
            assert_eq!(record.description_key, format!("item.record.{name}.desc"));
            assert_eq!(id.props().kind, ItemKind::Record);
            assert_eq!(id.props().name, format!("minecraft:record_{name}"));
        }
        for id in [ItemId(1), ItemId(2255), ItemId(2268), ItemId(u16::MAX)] {
            assert_eq!(record_info(id), None, "item {id:?} is not a disc");
        }
    }

    #[test]
    fn numeric_item_name_fallback_preserves_java_parse_and_registry_boundaries() {
        let stone = item_by_name("minecraft:stone").expect("stone is registered");
        assert_eq!(item_by_name("1"), Some(stone));
        assert_eq!(item_by_name("0001"), Some(stone));
        assert_eq!(item_by_name("+1"), Some(stone));

        // The original string, not ResourceLocation's path, is parsed by Java.
        assert_eq!(item_by_name("minecraft:1"), None);
        // Parsed integers still have to name a registered registry row.
        assert_eq!(item_by_name("0"), None);
        assert_eq!(item_by_name("255"), None);
        assert_eq!(item_by_name("65535"), None);
        // Integer.parseInt rejects whitespace, suffixes, negatives outside the registry, and
        // values beyond signed-int range.
        assert_eq!(item_by_name(" 1"), None);
        assert_eq!(item_by_name("1x"), None);
        assert_eq!(item_by_name("-1"), None);
        assert_eq!(item_by_name("2147483648"), None);
    }

    /// Java keeps the Block and Item registries separately.  Most block-items happen to reuse
    /// their block id, which makes a name-only lookup mistake invisible; these are the complete
    /// 1.8.9 counterexamples.  They are *not* `ItemBlock` pairs: Java registers independently
    /// named items such as `ItemDoor`, so neither generated relation may infer a link from text.
    #[test]
    fn differently_numbered_block_item_names_keep_typed_registry_identity() {
        const SHARED_DIFFERENT_IDS: &[(&str, u16, u16)] = &[
            ("minecraft:acacia_door", 196, 430),
            ("minecraft:bed", 26, 355),
            ("minecraft:birch_door", 194, 428),
            ("minecraft:brewing_stand", 117, 379),
            ("minecraft:cake", 92, 354),
            ("minecraft:cauldron", 118, 380),
            ("minecraft:dark_oak_door", 197, 431),
            ("minecraft:flower_pot", 140, 390),
            ("minecraft:iron_door", 71, 330),
            ("minecraft:jungle_door", 195, 429),
            ("minecraft:nether_wart", 115, 372),
            ("minecraft:reeds", 83, 338),
            ("minecraft:skull", 144, 397),
            ("minecraft:spruce_door", 193, 427),
            ("minecraft:wheat", 59, 296),
            ("minecraft:wooden_door", 64, 324),
        ];

        let extracted_different_ids: Vec<_> = crate::generated::blocks::BLOCK_NAMES
            .iter()
            .filter_map(|&(name, block)| {
                let item_index = crate::generated::items::ITEM_NAMES
                    .binary_search_by(|(item_name, _)| (*item_name).cmp(name))
                    .ok()?;
                let item = crate::generated::items::ITEM_NAMES[item_index].1;
                (block != item).then_some((name, block, item))
            })
            .collect();
        assert_eq!(extracted_different_ids, SHARED_DIFFERENT_IDS);

        for &(name, expected_block, expected_item) in SHARED_DIFFERENT_IDS {
            let unqualified = name.strip_prefix("minecraft:").expect("vanilla namespace");
            for key in [unqualified, name] {
                let block = crate::block_by_name(key).expect("registered block");
                let item = item_by_name(key).expect("registered item");
                assert_eq!(block.0, expected_block, "block lookup {key}");
                assert_eq!(item.0, expected_item, "item lookup {key}");
                assert_ne!(block.0, item.0, "the registries must not alias {key}");
                assert_eq!(block.info().item, None, "block must not infer item {key}");
                assert_eq!(item.props().block, None, "item must not infer block {key}");
                assert_ne!(item.props().kind, ItemKind::Block, "non-ItemBlock {key}");
            }
        }

        // Nonshared controls: neither lookup may silently fall through to the other registry.
        assert_eq!(crate::block_by_name("apple"), None);
        assert!(item_by_name("apple").is_some());
        assert_eq!(item_by_name("air"), None);
        assert_eq!(crate::block_by_name("air"), Some(BlockId::AIR));
    }

    /// Exact constructor rows from Java 1.8.9 `EnumDyeColor`, including the one serialized /
    /// unlocalized spelling split that motivated the API (`light_blue` vs `lightBlue`).
    #[test]
    fn dye_color_rows_preserve_metadata_damage_and_unlocalized_names() {
        let expected = [
            (0, 15, "white", "white"),
            (1, 14, "orange", "orange"),
            (2, 13, "magenta", "magenta"),
            (3, 12, "light_blue", "lightBlue"),
            (4, 11, "yellow", "yellow"),
            (5, 10, "lime", "lime"),
            (6, 9, "pink", "pink"),
            (7, 8, "gray", "gray"),
            (8, 7, "silver", "silver"),
            (9, 6, "cyan", "cyan"),
            (10, 5, "purple", "purple"),
            (11, 4, "blue", "blue"),
            (12, 3, "brown", "brown"),
            (13, 2, "green", "green"),
            (14, 1, "red", "red"),
            (15, 0, "black", "black"),
        ];
        for (row, expected) in DYE_COLORS.iter().zip(expected) {
            assert_eq!(
                (row.metadata, row.dye_damage, row.name, row.unlocalized_name),
                expected
            );
        }
    }

    #[test]
    fn dye_color_lookups_reverse_item_damage_and_keep_distinct_java_fallbacks() {
        for row in &DYE_COLORS {
            assert_eq!(dye_color_by_metadata(i32::from(row.metadata)), row);
            assert_eq!(dye_color_by_dye_damage(i32::from(row.dye_damage)), row);
        }

        for invalid in [i32::MIN, -1, 16, 17, i32::MAX] {
            assert_eq!(dye_color_by_metadata(invalid).unlocalized_name, "white");
            assert_eq!(dye_color_by_dye_damage(invalid).unlocalized_name, "black");
        }
    }

    #[test]
    fn tool_and_armor_material_repair_items_match_every_java_enum_variant() {
        for (material, repair_name) in [
            (ToolMaterial::Wood, "planks"),
            (ToolMaterial::Stone, "cobblestone"),
            (ToolMaterial::Iron, "iron_ingot"),
            (ToolMaterial::Emerald, "diamond"),
            (ToolMaterial::Gold, "gold_ingot"),
        ] {
            assert_eq!(
                material.repair_item(),
                item_by_name(repair_name).expect("vanilla repair item")
            );
        }

        for (material, repair_name) in [
            (ArmorMaterial::Leather, "leather"),
            (ArmorMaterial::Chain, "iron_ingot"),
            (ArmorMaterial::Iron, "iron_ingot"),
            (ArmorMaterial::Gold, "gold_ingot"),
            (ArmorMaterial::Diamond, "diamond"),
        ] {
            assert_eq!(
                material.repair_item(),
                item_by_name(repair_name).expect("vanilla repair item")
            );
        }
    }

    #[test]
    fn item_repair_predicate_dispatches_only_material_backed_classes() {
        let cases = [
            ("wooden_pickaxe", Some("planks")),
            ("stone_sword", Some("cobblestone")),
            ("diamond_axe", Some("diamond")),
            ("chainmail_helmet", Some("iron_ingot")),
            ("leather_boots", Some("leather")),
            ("golden_chestplate", Some("gold_ingot")),
            ("apple", None),
            ("shears", None),
        ];
        for (item_name, repair_name) in cases {
            let props = item_props(item_by_name(item_name).expect("registered item"));
            let expected = repair_name.map(|name| item_by_name(name).expect("repair item"));
            assert_eq!(props.repair_item(), expected, "{item_name}");
            for candidate in [
                "planks",
                "cobblestone",
                "iron_ingot",
                "gold_ingot",
                "diamond",
            ] {
                let candidate = item_by_name(candidate).expect("repair candidate");
                assert_eq!(
                    props.is_repairable_with(candidate),
                    expected == Some(candidate),
                    "{item_name} with {}",
                    item_props(candidate).name
                );
            }
        }
    }

    #[test]
    fn tool_and_armour_material_slots_do_not_cross_contaminate() {
        // ItemSword names its ToolMaterial field `material`, exactly like ItemArmor names
        // its ArmorMaterial. IRON and GOLD exist in both enums, so a name-based extraction
        // silently mislabels an iron sword as iron armour.
        let sword = item_props(item_by_name("iron_sword").expect("iron_sword"));
        assert_eq!(sword.tool_material, Some(ToolMaterial::Iron));
        assert_eq!(sword.armor_material, None);
        assert_eq!(sword.armor_slot, None);

        for it in items() {
            if it.kind == ItemKind::Armor {
                assert!(it.armor_material.is_some(), "{}", it.name);
                assert!(it.tool_material.is_none(), "{}", it.name);
            }
            if it.is_tool() {
                assert!(it.armor_material.is_none(), "{}", it.name);
            }
        }
    }

    #[test]
    fn block_items_point_back_at_their_block() {
        let stone = item_by_name("stone").expect("stone item");
        assert_eq!(item_props(stone).block.map(|b| b.0), Some(1));
        assert_eq!(item_props(stone).kind, ItemKind::Block);
    }

    #[test]
    fn metadata_unlocalized_names_follow_vanilla_variant_order() {
        fn assert_variants(name: &str, expected: &[&str]) {
            let id = item_by_name(name).unwrap_or_else(|| panic!("missing item {name}"));
            for (damage, expected) in expected.iter().enumerate() {
                assert_eq!(
                    id.unlocalized_name(damage as i16),
                    *expected,
                    "{name} damage {damage}"
                );
            }
            assert_eq!(
                id.unlocalized_name(-1),
                expected[0],
                "ItemStack clamps negative damage to zero"
            );
        }

        assert_variants(
            "sapling",
            &[
                "tile.sapling.oak",
                "tile.sapling.spruce",
                "tile.sapling.birch",
                "tile.sapling.jungle",
                "tile.sapling.acacia",
                "tile.sapling.big_oak",
            ],
        );
        assert_variants(
            "stone_slab",
            &[
                "tile.stoneSlab.stone",
                "tile.stoneSlab.sand",
                "tile.stoneSlab.wood",
                "tile.stoneSlab.cobble",
                "tile.stoneSlab.brick",
                "tile.stoneSlab.smoothStoneBrick",
                "tile.stoneSlab.netherBrick",
                "tile.stoneSlab.quartz",
            ],
        );
        assert_variants(
            "wooden_slab",
            &[
                "tile.woodSlab.oak",
                "tile.woodSlab.spruce",
                "tile.woodSlab.birch",
                "tile.woodSlab.jungle",
                "tile.woodSlab.acacia",
                "tile.woodSlab.big_oak",
            ],
        );
        assert_variants(
            "cobblestone_wall",
            &["tile.cobbleWall.normal", "tile.cobbleWall.mossy"],
        );
        assert_variants(
            "double_plant",
            &[
                "tile.doublePlant.sunflower",
                "tile.doublePlant.syringa",
                "tile.doublePlant.grass",
                "tile.doublePlant.fern",
                "tile.doublePlant.rose",
                "tile.doublePlant.paeonia",
            ],
        );
        assert_variants(
            "fish",
            &[
                "item.fish.cod.raw",
                "item.fish.salmon.raw",
                "item.fish.clownfish.raw",
                "item.fish.pufferfish.raw",
            ],
        );
        assert_variants(
            "cooked_fish",
            &[
                "item.fish.cod.cooked",
                "item.fish.salmon.cooked",
                "item.fish.clownfish.raw",
                "item.fish.pufferfish.raw",
            ],
        );
        assert_variants(
            "dye",
            &[
                "item.dyePowder.black",
                "item.dyePowder.red",
                "item.dyePowder.green",
                "item.dyePowder.brown",
                "item.dyePowder.blue",
                "item.dyePowder.purple",
                "item.dyePowder.cyan",
                "item.dyePowder.silver",
                "item.dyePowder.gray",
                "item.dyePowder.pink",
                "item.dyePowder.lime",
                "item.dyePowder.yellow",
                "item.dyePowder.lightBlue",
                "item.dyePowder.magenta",
                "item.dyePowder.orange",
                "item.dyePowder.white",
            ],
        );
        assert_variants(
            "skull",
            &[
                "item.skull.skeleton",
                "item.skull.wither",
                "item.skull.zombie",
                "item.skull.char",
                "item.skull.creeper",
            ],
        );

        // ItemLeaves masks off the decay/check flags before selecting its wood type.
        let old_leaves = item_by_name("leaves").expect("old leaves item");
        assert_eq!(old_leaves.unlocalized_name(0), "tile.leaves.oak");
        assert_eq!(old_leaves.unlocalized_name(5), "tile.leaves.spruce");
        assert_eq!(old_leaves.unlocalized_name(14), "tile.leaves.birch");
        assert_eq!(old_leaves.unlocalized_name(15), "tile.leaves.jungle");

        let new_leaves = item_by_name("leaves2").expect("new leaves item");
        assert_eq!(new_leaves.unlocalized_name(0), "tile.leaves.acacia");
        assert_eq!(new_leaves.unlocalized_name(1), "tile.leaves.big_oak");
        assert_eq!(
            new_leaves.unlocalized_name(2),
            "tile.leaves.oak",
            "BlockPlanks.EnumType.byMetadata(6) falls back to oak"
        );
    }

    #[test]
    fn item_multi_texture_and_colored_names_are_table_driven() {
        for item in items()
            .iter()
            .filter(|item| item.java_class == "net.minecraft.item.ItemMultiTexture")
        {
            assert!(
                metadata_name_for(item.id.0, 0).is_some(),
                "missing ItemMultiTexture metadata table row for {}",
                item.name
            );
        }

        let cases: &[(&str, &[&str], &str)] = &[
            (
                "stone",
                &[
                    "tile.stone.stone",
                    "tile.stone.granite",
                    "tile.stone.graniteSmooth",
                    "tile.stone.diorite",
                    "tile.stone.dioriteSmooth",
                    "tile.stone.andesite",
                    "tile.stone.andesiteSmooth",
                ],
                "tile.stone.stone",
            ),
            (
                "dirt",
                &["tile.dirt.default", "tile.dirt.coarse", "tile.dirt.podzol"],
                "tile.dirt.default",
            ),
            (
                "planks",
                &[
                    "tile.wood.oak",
                    "tile.wood.spruce",
                    "tile.wood.birch",
                    "tile.wood.jungle",
                    "tile.wood.acacia",
                    "tile.wood.big_oak",
                ],
                "tile.wood.oak",
            ),
            (
                "sand",
                &["tile.sand.default", "tile.sand.red"],
                "tile.sand.default",
            ),
            (
                "log",
                &[
                    "tile.log.oak",
                    "tile.log.spruce",
                    "tile.log.birch",
                    "tile.log.jungle",
                    "tile.log.acacia",
                    "tile.log.big_oak",
                ],
                "tile.log.oak",
            ),
            (
                "sandstone",
                &[
                    "tile.sandStone.default",
                    "tile.sandStone.chiseled",
                    "tile.sandStone.smooth",
                ],
                "tile.sandStone.default",
            ),
            (
                "tallgrass",
                &[
                    "tile.tallgrass.shrub",
                    "tile.tallgrass.grass",
                    "tile.tallgrass.fern",
                ],
                "tile.tallgrass",
            ),
            (
                "yellow_flower",
                &["tile.flower1.dandelion"],
                "tile.flower1.dandelion",
            ),
            (
                "monster_egg",
                &[
                    "tile.monsterStoneEgg.stone",
                    "tile.monsterStoneEgg.cobble",
                    "tile.monsterStoneEgg.brick",
                    "tile.monsterStoneEgg.mossybrick",
                    "tile.monsterStoneEgg.crackedbrick",
                    "tile.monsterStoneEgg.chiseledbrick",
                ],
                "tile.monsterStoneEgg.stone",
            ),
            (
                "stonebrick",
                &[
                    "tile.stonebricksmooth.default",
                    "tile.stonebricksmooth.mossy",
                    "tile.stonebricksmooth.cracked",
                    "tile.stonebricksmooth.chiseled",
                ],
                "tile.stonebricksmooth.default",
            ),
            (
                "quartz_block",
                &[
                    "tile.quartzBlock.default",
                    "tile.quartzBlock.chiseled",
                    "tile.quartzBlock.lines",
                ],
                "tile.quartzBlock.default",
            ),
            (
                "log2",
                &["tile.log.acacia", "tile.log.big_oak"],
                "tile.log.oak",
            ),
        ];

        for &(name, expected, invalid) in cases {
            let id = item_by_name(name).unwrap_or_else(|| panic!("missing item {name}"));
            for (damage, expected) in expected.iter().enumerate() {
                assert_eq!(
                    id.unlocalized_name(damage as i16),
                    *expected,
                    "{name} valid metadata {damage}"
                );
            }
            assert_eq!(
                id.unlocalized_name(-1),
                expected[0],
                "{name} negative metadata"
            );
            assert_eq!(
                id.unlocalized_name(i16::MAX),
                invalid,
                "{name} high metadata fallback"
            );
        }

        // ItemColored's no-subtype registrations and ordinary items must not inherit a metadata
        // suffix table merely because they share an ItemBlock/base-key family.
        for name in ["grass", "vine", "diamond_sword"] {
            let id = item_by_name(name).unwrap_or_else(|| panic!("missing item {name}"));
            let base = id.props().unlocalized;
            for damage in [-1, 0, 1, i16::MAX] {
                assert_eq!(
                    id.unlocalized_name(damage),
                    base,
                    "{name} metadata {damage}"
                );
            }
        }
    }

    #[test]
    fn prismarine_flower_and_red_sandstone_metadata_names_cover_every_variant() {
        let assert_family = |item: &str, keys: &[&str], display_names: &[&str]| {
            let props = item_by_name(item).unwrap_or_else(|| panic!("missing item {item}"));
            assert!(!keys.is_empty(), "variant table must not be empty");
            for (damage, (key, display)) in keys.iter().zip(display_names).enumerate() {
                assert_eq!(
                    props.unlocalized_name(damage as i16),
                    *key,
                    "{item} metadata {damage}"
                );
                assert_eq!(
                    props.props().display_name_for(damage as i16),
                    *display,
                    "{item} metadata {damage}"
                );
            }
            assert_eq!(
                props.unlocalized_name(i16::MAX),
                keys[0],
                "{item} invalid key fallback"
            );
            assert_eq!(
                props.props().display_name_for(i16::MAX),
                display_names[0],
                "{item} invalid display fallback"
            );
            assert_eq!(
                props.unlocalized_name(-1),
                keys[0],
                "{item} negative key fallback"
            );
            assert_eq!(
                props.props().display_name_for(-1),
                display_names[0],
                "{item} negative display fallback"
            );
        };

        assert_family(
            "prismarine",
            &[
                "tile.prismarine.rough",
                "tile.prismarine.bricks",
                "tile.prismarine.dark",
            ],
            &["Prismarine", "Prismarine Bricks", "Dark Prismarine"],
        );
        assert_family(
            "red_flower",
            &[
                "tile.flower2.poppy",
                "tile.flower2.blueOrchid",
                "tile.flower2.allium",
                "tile.flower2.houstonia",
                "tile.flower2.tulipRed",
                "tile.flower2.tulipOrange",
                "tile.flower2.tulipWhite",
                "tile.flower2.tulipPink",
                "tile.flower2.oxeyeDaisy",
            ],
            &[
                "Poppy",
                "Blue Orchid",
                "Allium",
                "Azure Bluet",
                "Red Tulip",
                "Orange Tulip",
                "White Tulip",
                "Pink Tulip",
                "Oxeye Daisy",
            ],
        );
        assert_family(
            "red_sandstone",
            &[
                "tile.redSandStone.default",
                "tile.redSandStone.chiseled",
                "tile.redSandStone.smooth",
            ],
            &[
                "Red Sandstone",
                "Chiseled Red Sandstone",
                "Smooth Red Sandstone",
            ],
        );
    }

    #[test]
    fn coal_metadata_selects_charcoal_name_and_zero_fallback() {
        let coal = item_by_name("coal").expect("coal item");
        assert_eq!(coal.unlocalized_name(0), "item.coal");
        assert_eq!(coal.unlocalized_name(1), "item.charcoal");
        assert_eq!(coal.props().display_name_for(0), "Coal");
        assert_eq!(coal.props().display_name_for(1), "Charcoal");
        for invalid in [-1, 2, i16::MAX] {
            assert_eq!(
                coal.unlocalized_name(invalid),
                "item.coal",
                "key fallback {invalid}"
            );
            assert_eq!(
                coal.props().display_name_for(invalid),
                "Coal",
                "display fallback {invalid}"
            );
        }
    }

    #[test]
    fn anvil_metadata_selects_damage_stage_name_and_zero_fallback() {
        let anvil = item_by_name("anvil").expect("anvil item");
        let expected = [
            ("tile.anvil.intact", "Anvil"),
            ("tile.anvil.slightlyDamaged", "Slightly Damaged Anvil"),
            ("tile.anvil.veryDamaged", "Very Damaged Anvil"),
        ];
        for (damage, (key, display)) in expected.iter().enumerate() {
            assert_eq!(anvil.unlocalized_name(damage as i16), *key);
            assert_eq!(anvil.props().display_name_for(damage as i16), *display);
        }
        assert_eq!(anvil.unlocalized_name(i16::MAX), expected[0].0);
        assert_eq!(anvil.props().display_name_for(i16::MAX), expected[0].1);
    }

    #[test]
    fn invalid_variant_metadata_uses_each_vanilla_enum_fallback() {
        for (name, expected) in [
            ("sapling", "tile.sapling.oak"),
            ("stone_slab", "tile.stoneSlab.stone"),
            ("wooden_slab", "tile.woodSlab.oak"),
            ("cobblestone_wall", "tile.cobbleWall.normal"),
            ("double_plant", "tile.doublePlant.sunflower"),
            ("stone_slab2", "tile.stoneSlab2.red_sandstone"),
            ("fish", "item.fish.cod.raw"),
            ("cooked_fish", "item.fish.cod.cooked"),
            ("dye", "item.dyePowder.black"),
            ("skull", "item.skull.skeleton"),
        ] {
            let id = item_by_name(name).unwrap_or_else(|| panic!("missing item {name}"));
            assert_eq!(id.unlocalized_name(i16::MAX), expected, "{name}");
        }
    }

    #[test]
    fn formerly_unresolvable_base_keys_now_select_shipped_variant_stems() {
        for (name, damage, expected) in [
            ("stone", 6, "tile.stone.andesiteSmooth"),
            ("sapling", 5, "tile.sapling.big_oak"),
            ("sponge", 1, "tile.sponge.wet"),
            ("wool", 14, "tile.cloth.red"),
            ("cobblestone_wall", 1, "tile.cobbleWall.mossy"),
            ("stained_glass", 14, "tile.stainedGlass.red"),
            ("anvil", 2, "tile.anvil.veryDamaged"),
            ("stained_hardened_clay", 14, "tile.clayHardenedStained.red"),
            ("stained_glass_pane", 14, "tile.thinStainedGlass.red"),
            ("prismarine", 2, "tile.prismarine.dark"),
            ("carpet", 14, "tile.woolCarpet.red"),
            ("stone_slab2", 0, "tile.stoneSlab2.red_sandstone"),
            ("coal", 1, "item.charcoal"),
            ("fish", 3, "item.fish.pufferfish.raw"),
            ("dye", 12, "item.dyePowder.lightBlue"),
            ("skull", 4, "item.skull.creeper"),
        ] {
            let props = item_props(item_by_name(name).unwrap());
            assert_eq!(props.unlocalized_name(damage), expected, "{name}");
            assert_ne!(
                props.unlocalized_name(damage),
                props.unlocalized,
                "{name} must not fall back to its absent base translation key"
            );
        }

        let sword = item_by_name("diamond_sword").unwrap();
        assert_eq!(
            sword.unlocalized_name(900),
            sword.props().unlocalized,
            "ordinary items retain the extracted no-stack key"
        );
    }

    #[test]
    fn cloth_family_display_names_match_every_en_us_variant_and_invalid_fallback() {
        // These are the values of the 1.8.9 en_US.lang entries selected by ItemCloth's
        // `EnumDyeColor.byMetadata(stack.getMetadata())` call.  In particular, white wool and
        // white carpet intentionally use the base names rather than "White Wool"/"White Carpet".
        let cases: &[(&str, &[&str])] = &[
            (
                "wool",
                &[
                    "Wool",
                    "Orange Wool",
                    "Magenta Wool",
                    "Light Blue Wool",
                    "Yellow Wool",
                    "Lime Wool",
                    "Pink Wool",
                    "Gray Wool",
                    "Light Gray Wool",
                    "Cyan Wool",
                    "Purple Wool",
                    "Blue Wool",
                    "Brown Wool",
                    "Green Wool",
                    "Red Wool",
                    "Black Wool",
                ],
            ),
            (
                "stained_glass",
                &[
                    "White Stained Glass",
                    "Orange Stained Glass",
                    "Magenta Stained Glass",
                    "Light Blue Stained Glass",
                    "Yellow Stained Glass",
                    "Lime Stained Glass",
                    "Pink Stained Glass",
                    "Gray Stained Glass",
                    "Light Gray Stained Glass",
                    "Cyan Stained Glass",
                    "Purple Stained Glass",
                    "Blue Stained Glass",
                    "Brown Stained Glass",
                    "Green Stained Glass",
                    "Red Stained Glass",
                    "Black Stained Glass",
                ],
            ),
            (
                "stained_hardened_clay",
                &[
                    "White Stained Clay",
                    "Orange Stained Clay",
                    "Magenta Stained Clay",
                    "Light Blue Stained Clay",
                    "Yellow Stained Clay",
                    "Lime Stained Clay",
                    "Pink Stained Clay",
                    "Gray Stained Clay",
                    "Light Gray Stained Clay",
                    "Cyan Stained Clay",
                    "Purple Stained Clay",
                    "Blue Stained Clay",
                    "Brown Stained Clay",
                    "Green Stained Clay",
                    "Red Stained Clay",
                    "Black Stained Clay",
                ],
            ),
            (
                "stained_glass_pane",
                &[
                    "White Stained Glass Pane",
                    "Orange Stained Glass Pane",
                    "Magenta Stained Glass Pane",
                    "Light Blue Stained Glass Pane",
                    "Yellow Stained Glass Pane",
                    "Lime Stained Glass Pane",
                    "Pink Stained Glass Pane",
                    "Gray Stained Glass Pane",
                    "Light Gray Stained Glass Pane",
                    "Cyan Stained Glass Pane",
                    "Purple Stained Glass Pane",
                    "Blue Stained Glass Pane",
                    "Brown Stained Glass Pane",
                    "Green Stained Glass Pane",
                    "Red Stained Glass Pane",
                    "Black Stained Glass Pane",
                ],
            ),
            (
                "carpet",
                &[
                    "Carpet",
                    "Orange Carpet",
                    "Magenta Carpet",
                    "Light Blue Carpet",
                    "Yellow Carpet",
                    "Lime Carpet",
                    "Pink Carpet",
                    "Gray Carpet",
                    "Light Gray Carpet",
                    "Cyan Carpet",
                    "Purple Carpet",
                    "Blue Carpet",
                    "Brown Carpet",
                    "Green Carpet",
                    "Red Carpet",
                    "Black Carpet",
                ],
            ),
        ];

        for &(name, expected) in cases {
            let id = item_by_name(name).unwrap_or_else(|| panic!("missing item {name}"));
            for (damage, expected) in expected.iter().enumerate() {
                assert_eq!(
                    id.props().display_name_for(damage as i16),
                    *expected,
                    "{name} damage {damage}"
                );
            }
            for invalid in [-1, 16, i16::MAX] {
                assert_eq!(
                    id.props().display_name_for(invalid),
                    expected[0],
                    "{name} invalid damage {invalid} follows white EnumDyeColor fallback"
                );
            }
        }

        let ordinary = item_by_name("diamond_sword").expect("diamond sword");
        assert_eq!(
            ordinary.props().display_name_for(13),
            "Diamond Sword",
            "non-cloth items retain their registry display name"
        );
    }

    #[test]
    fn skull_unlocalized_name_stays_separate_from_owner_nbt_display_override() {
        let skull = item_by_name("skull").unwrap();
        assert_eq!(skull.unlocalized_name(3), "item.skull.char");
        // ItemSkull.getItemStackDisplayName changes this to item.skull.player.name only
        // when a SkullOwner tag is present; mc-data intentionally has no NBT dependency.
        assert_ne!(skull.unlocalized_name(3), "item.skull.player");
    }

    #[test]
    fn skull_display_names_match_every_metadata_variant_and_invalid_fallback() {
        let skull = item_by_name("skull").expect("skull item");
        let expected = [
            "Skeleton Skull",
            "Wither Skeleton Skull",
            "Zombie Head",
            "Head",
            "Creeper Head",
        ];

        for (damage, expected) in expected.iter().enumerate() {
            assert_eq!(
                skull.props().display_name_for(damage as i16),
                *expected,
                "skull damage {damage}"
            );
        }

        // ItemSkull.getUnlocalizedName and getItemStackDisplayName both clamp through the
        // metadata-zero branch when the wire carries a hostile value.  The negative value is
        // normally normalised by ItemStack.setItemDamage, but this direct resolver test keeps the
        // boundary explicit for packet-derived stacks.
        for invalid in [-1, 5, 16, i16::MAX] {
            assert_eq!(
                skull.props().display_name_for(invalid),
                expected[0],
                "invalid skull metadata {invalid} falls back to skeleton"
            );
            assert_eq!(
                skull.unlocalized_name(invalid),
                "item.skull.skeleton",
                "invalid skull metadata {invalid} selects metadata zero's key"
            );
        }
    }

    #[test]
    fn tool_material_stats_match_vanilla() {
        assert_eq!(ToolMaterial::Wood.info().max_uses, 59);
        assert_eq!(ToolMaterial::Emerald.info().max_uses, 1561);
        assert_eq!(ToolMaterial::Gold.info().harvest_level, 0);
        assert_eq!(ToolMaterial::Emerald.info().harvest_level, 3);
    }
}
