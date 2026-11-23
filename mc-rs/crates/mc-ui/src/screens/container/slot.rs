//! [`Slot`]: display position plus validity/limit rules. java: `Slot.java` and its subclasses
//! (`SlotCrafting`, `SlotFurnaceFuel`, `SlotFurnaceOutput`, and the anonymous subclasses inline
//! in each `ContainerXxx` constructor).
//!
//! Rust has no inheritance, so where Java gives each special slot its own `Slot` subclass this
//! is one [`Slot`] struct plus a closed [`SlotRole`] enum covering every distinct override this
//! port needs.
//!
//! # `canTakeStack`
//!
//! `grep -rn canTakeStack src/minecraft` returns exactly **two** definitions in the whole 1.8.9
//! tree: the `return true` base at `Slot.java:150`, and one override — the anonymous anvil
//! output slot at `ContainerRepair.java:71-74`:
//!
//! ```text
//! (playerIn.capabilities.isCreativeMode || playerIn.experienceLevel >= maximumCost)
//!     && maximumCost > 0 && this.getHasStack()
//! ```
//!
//! An earlier revision of this doc claimed `SlotFurnaceOutput` also overrides it and that both
//! overrides were server-only effects. **Both halves were false.** `SlotFurnaceOutput` defines
//! only `isItemValid`/`decrStackSize`/`onPickupFromSlot`/`onCrafting`, and the anvil predicate is
//! fully client-computable once its local inputs have been evaluated: `maximumCost` is
//! `ContainerRepair` field `0`, which the server sends as a one-time open snapshot through
//! `S31PacketWindowProperty` (always `0`, before any inputs are present), while the client then
//! recomputes it in `ContainerRepair.updateRepairOutput` (this port stores that local value on
//! [`super::model::Container::maximum_cost`]); `experienceLevel` is client-side state too.
//!
//! The predicate is load-bearing on the **wire**, not just locally: `Container.slotClick`'s
//! `mode == 1` arm assigns its return value *inside* the gate, so vanilla shift-clicking an
//! unaffordable anvil output sends `clicked_item = <empty>` while an ungated port sends the
//! repaired item — which the server rejects with `S32(accepted = false)` plus a window-wide
//! resync. Modes 0/2/4 keep matching bytes, so their transaction is *accepted* and the server
//! sends nothing back (`isChangingQuantityOnly` suppresses both `sendSlotContents` and
//! `updateHeldItem`): a ghost item stays on the client cursor with nothing to correct it.
//!
//! [`SlotRole::AnvilOutput`] therefore identifies the gate, evaluated by
//! [`super::model::Container::can_take_stack`] only for `ContainerRepair`'s index-2 output.
//! `onPickupFromSlot`'s XP deduction and input consumption are also client-predictable here so
//! the next click and local output match vanilla; world-side anvil damage and sound remain
//! server-authoritative and arrive through `SetSlot`/`WindowItems` when applicable.

use super::stack_ext::{self, ItemStack};

/// The container-wide default a slot falls back to when it has no override. java: every
/// vanilla `IInventory.getInventoryStackLimit()` this port builds a container over returns 64.
pub const DEFAULT_STACK_LIMIT: i32 = 64;

/// What makes one slot behave differently from a plain inventory slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotRole {
    /// Accepts anything, no special limit. The overwhelming majority of slots.
    Normal,
    /// `isItemValid` always false: a *result* slot backed by an ordinary `IInventory`, i.e. the
    /// furnace output. Items can only leave it (via a normal left/right pickup or a
    /// shift-click), never be placed into it, and they leave by the usual split rule
    /// (`TileEntityFurnace.decrStackSize` is a plain `splitStack`).
    /// java: `SlotFurnaceOutput` over `TileEntityFurnace`.
    Output,
    /// `isItemValid` always false like [`SlotRole::Output`], but backed by
    /// `InventoryCraftResult`, whose `decrStackSize(index, count)` **discards `count`** and
    /// returns the whole stack (`inventory/InventoryCraftResult.java:57-69`). So a right-click,
    /// a `Q`, or a double-click sweep on one of these takes *everything*, not a half or a one.
    /// java: `SlotCrafting` over `InventoryCraftResult` — the player's 2x2 result and the
    /// workbench's 3x3 result. (The anvil output is also `InventoryCraftResult`-backed but
    /// carries an extra `canTakeStack` gate, so it is [`SlotRole::AnvilOutput`].)
    CraftResult,
    /// An anvil's output slot: `isItemValid` false like [`SlotRole::Output`], **plus** the one
    /// `canTakeStack` override in 1.8.9 (`ContainerRepair.java:71-74`). See this module's doc.
    AnvilOutput,
    /// One of the four player armour slots. `0` helmet, `1` chestplate, `2` leggings, `3`
    /// boots — vanilla's `ItemArmor.armorType` numbering. java: the anonymous `Slot` in
    /// `ContainerPlayer`'s constructor.
    Armor(u8),
    /// A furnace's fuel slot. java: `SlotFurnaceFuel`.
    FurnaceFuel,
    /// A brewing stand's ingredient slot (top). java: `ContainerBrewingStand.Ingredient`.
    BrewIngredient,
    /// One of a brewing stand's three potion-bottle slots. java:
    /// `ContainerBrewingStand.Potion`.
    BrewBottle,
    /// An enchanting table's lapis slot (must be blue dye). java: the anonymous `Slot` in
    /// `ContainerEnchantment`'s constructor (slot index 1).
    EnchantLapis,
    /// A beacon's payment slot (emerald/diamond/gold ingot/iron ingot). java:
    /// `ContainerBeacon.BeaconSlot`.
    Beacon,
    /// `ContainerHorseInventory` slot 0: saddle-only insertion.
    HorseSaddle,
    /// `ContainerHorseInventory` slot 1. `can_wear` is the tracked entity's
    /// `EntityHorse.canWearArmor()` result.
    HorseArmor {
        /// Whether this mount accepts armour.
        can_wear: bool,
    },
}

impl SlotRole {
    /// `true` when this slot's backing `IInventory` ignores the requested count in
    /// `decrStackSize` and hands back the whole stack.
    ///
    /// java: `InventoryCraftResult.decrStackSize` (`:57-69`) and `InventoryMerchant`'s
    /// `index == 2` arm (`:49-54`) both null the slot and return the entire stack whatever
    /// `count` says. `Container.slotClick` never looks: it computes
    /// `j2 = clickedButton == 0 ? stackSize : (stackSize + 1) / 2` and trusts whatever
    /// `slot.decrStackSize(j2)` returns (`Container.java:325-326`), so a right-click on a craft
    /// result yields all of it. The same is true of mode 4's `Q` (`:452`) and mode 6's collect
    /// sweep (`:475`) — and mode 6 adds only its own `l` to the cursor, so vanilla genuinely
    /// destroys the remainder there.
    ///
    /// `SlotRole::Output` (the furnace) is deliberately **not** in this set:
    /// `TileEntityFurnace.decrStackSize` is an ordinary split.
    #[must_use]
    pub const fn takes_whole_stack(self) -> bool {
        matches!(self, SlotRole::CraftResult | SlotRole::AnvilOutput)
    }

    /// java: `Slot.isItemValid` (or the relevant override).
    #[must_use]
    pub fn is_item_valid(self, stack: &ItemStack) -> bool {
        match self {
            SlotRole::Normal => true,
            SlotRole::Output | SlotRole::CraftResult | SlotRole::AnvilOutput => false,
            SlotRole::Armor(index) => is_valid_armor(stack, index),
            SlotRole::FurnaceFuel => is_furnace_fuel(stack) || is_bucket(stack),
            SlotRole::BrewIngredient => is_potion_ingredient(stack),
            SlotRole::BrewBottle => is_potion_or_bottle(stack),
            SlotRole::EnchantLapis => is_lapis(stack),
            SlotRole::Beacon => is_beacon_payment(stack),
            SlotRole::HorseSaddle => stack_ext::props(stack).name == "minecraft:saddle",
            SlotRole::HorseArmor { can_wear } => {
                can_wear && stack_ext::props(stack).armor_material.is_some()
            }
        }
    }

    /// java: `Slot.canBeHovered`. This is intentionally independent from click hit-testing.
    #[must_use]
    pub const fn can_be_hovered(self) -> bool {
        !matches!(self, SlotRole::HorseArmor { can_wear: false })
    }
}

/// One slot in a [`super::model::Container`]: a display position plus the rules above. The
/// stack it currently holds lives in the container's parallel item array, not here — matching
/// the Java split between `Slot` (a view) and `IInventory` (the storage) closely enough for the
/// click protocol's purposes without reproducing the full `IInventory` graph (see
/// [`super::model`]'s module doc for why a flat array is sufficient here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    /// `Slot.slotNumber` — this slot's index, both in `Container.inventorySlots` and in the
    /// parallel item array.
    pub id: u16,
    /// `Slot.xDisplayPosition`.
    pub x: i32,
    /// `Slot.yDisplayPosition`.
    pub y: i32,
    /// What kind of slot this is.
    pub role: SlotRole,
    /// Overrides [`DEFAULT_STACK_LIMIT`] when `Some`. java: `Slot.getSlotStackLimit`.
    pub stack_limit: Option<i32>,
    /// The atlas sprite drawn in this slot while it is **empty**, without the `minecraft:`
    /// namespace. java: `Slot.getSlotTexture()` (`Slot.java:125`), which returns `null` for
    /// every 1.8.9 slot except `ContainerPlayer`'s four armour slots
    /// (`ContainerPlayer.java:47-51` -> `ItemArmor.EMPTY_SLOT_NAMES`, `ItemArmor.java:24`).
    /// `GuiContainerCreative.java:994` wraps a real slot and delegates, so it adds no new
    /// names.
    pub background_sprite: Option<&'static str>,
}

impl Slot {
    /// A plain slot at a display position, backed by the container's default inventory.
    #[must_use]
    pub const fn new(id: u16, x: i32, y: i32) -> Self {
        Slot {
            id,
            x,
            y,
            role: SlotRole::Normal,
            stack_limit: None,
            background_sprite: None,
        }
    }

    /// Builder: set the role.
    #[must_use]
    pub const fn with_role(mut self, role: SlotRole) -> Self {
        self.role = role;
        self
    }

    /// Builder: pin a fixed stack limit (armour, potions, beacon payment, ...).
    #[must_use]
    pub const fn with_limit(mut self, limit: i32) -> Self {
        self.stack_limit = Some(limit);
        self
    }

    /// Builder: the placeholder sprite drawn while the slot is empty. java:
    /// `Slot.getSlotTexture`.
    #[must_use]
    pub const fn with_background(mut self, sprite: &'static str) -> Self {
        self.background_sprite = Some(sprite);
        self
    }

    /// java: `Slot.isItemValid`.
    #[must_use]
    pub fn is_item_valid(&self, stack: &ItemStack) -> bool {
        self.role.is_item_valid(stack)
    }

    /// The insertion predicate when a slot already has a stack. Horse saddle's anonymous Java
    /// slot adds `!getHasStack()` to `isItemValid`; the other roles are stack-independent.
    #[must_use]
    pub fn is_item_valid_for(&self, existing: Option<&ItemStack>, stack: &ItemStack) -> bool {
        self.is_item_valid(stack)
            && (!matches!(self.role, SlotRole::HorseSaddle) || existing.is_none())
    }

    /// java: `Slot.getItemStackLimit(ItemStack)`. Only `SlotFurnaceFuel` makes this depend on
    /// the stack itself (a bucket caps at 1 even though the fuel slot's own limit is the usual
    /// 64) — every other role's limit is fixed, so this just forwards to `stack_limit` for them.
    #[must_use]
    pub fn item_stack_limit(&self, stack: &ItemStack) -> i32 {
        if self.role == SlotRole::FurnaceFuel && is_bucket(stack) {
            return 1;
        }
        self.stack_limit.unwrap_or(DEFAULT_STACK_LIMIT)
    }

    /// java: `Slot.canBeHovered` for GUI highlight rendering.
    #[must_use]
    pub const fn can_be_hovered(&self) -> bool {
        self.role.can_be_hovered()
    }
}

/// The block-atlas UV rect of a [`Slot::background_sprite`] name.
///
/// java: `GuiContainer.drawSlot` resolves `getSlotTexture()` with
/// `TextureMap.getAtlasSprite(name)`, keyed by the sprite's icon name
/// (`minecraft:items/empty_armor_slot_helmet`). `mc_assets` registers the same stitched sprite
/// under its texture location, `textures/` prefix included (`mc_assets::model::BUILTIN_TEXTURES`
/// goes through the same `texture_location` as every model texture). Looked up without the
/// prefix the name misses, and `mc_assets::sprite` answers with its full-atlas placeholder: the
/// whole block atlas squeezed into the 16x16 slot.
#[must_use]
pub fn background_sprite_uv(name: &str) -> mc_assets::sprite::UvRect {
    mc_assets::sprite(&mc_assets::ResourceLocation::minecraft(format!(
        "textures/{name}"
    )))
    .uv
}

/// java: `SlotFurnaceFuel.isBucket`.
#[must_use]
pub fn is_bucket(stack: &ItemStack) -> bool {
    stack_ext::props(stack).name == "minecraft:bucket"
}

/// java: `TileEntityFurnace.getItemBurnTime(stack) > 0` (`isItemFuel`). The block-material-wood
/// branch (planks/logs/stairs/fences/... all burn at 300 ticks) is a name list, because `mc_data`
/// does not expose block materials to this crate. The list is **not** hand-transcribed: it is the
/// exact set of 1.8.9 blocks that (a) resolve to `Material.wood` and (b) have an `ItemBlock`,
/// re-derived by walking `Block.registerBlocks` (`block/Block.java:1251-1450`) with each block
/// class's constructor chain and intersecting with `Item.registerItems`' `registerItemBlock`
/// calls (`item/Item.java:529-737`), plus `banner`, whose `ItemBanner extends ItemBlock` over
/// `Blocks.standing_banner` (`item/ItemBanner.java:19-27`, `BlockBanner.java:31`).
///
/// Blocks that are wood but whose item is **not** an `ItemBlock` are correctly absent: `sign`
/// (`ItemSign extends Item`) and the six doors (`ItemDoor extends Item`) fail
/// `item instanceof ItemBlock`. `ladder` and `wooden_button` look wooden and are not:
/// `BlockLadder.java:23` and `BlockButton.java:30` are both `super(Material.circuits)`.
///
/// A wrong guess here is *not* self-healing. `Container.slotClick`'s mode-0 `itemstack9 == null`
/// arm never assigns `itemstack`, so a refused placement into an empty fuel slot leaves
/// `clicked_item` empty on both sides, the server accepts the click, and `isChangingQuantityOnly`
/// suppresses the corrective `SetSlot` — contradicting the module doc's "only costs one resync".
#[must_use]
pub fn is_furnace_fuel(stack: &ItemStack) -> bool {
    let p = stack_ext::props(stack);
    match p.name {
        "minecraft:wooden_slab" => return true,
        "minecraft:coal_block" => return true,
        "minecraft:coal" => return true, // covers coal (0) and charcoal (1) alike
        "minecraft:lava_bucket" => return true,
        "minecraft:sapling" => return true,
        "minecraft:blaze_rod" => return true,
        "minecraft:stick" => return true,
        _ => {}
    }
    if p.tool_material == Some(mc_data::ToolMaterial::Wood)
        && matches!(
            p.kind,
            mc_data::ItemKind::Pickaxe
                | mc_data::ItemKind::Axe
                | mc_data::ItemKind::Shovel
                | mc_data::ItemKind::Hoe
                | mc_data::ItemKind::Sword
        )
    {
        return true;
    }
    const WOOD_MATERIAL_BLOCKS: &[&str] = &[
        "minecraft:planks",
        "minecraft:log",
        "minecraft:log2",
        "minecraft:oak_stairs",
        "minecraft:spruce_stairs",
        "minecraft:birch_stairs",
        "minecraft:jungle_stairs",
        "minecraft:acacia_stairs",
        "minecraft:dark_oak_stairs",
        "minecraft:fence",
        "minecraft:spruce_fence",
        "minecraft:birch_fence",
        "minecraft:jungle_fence",
        "minecraft:dark_oak_fence",
        "minecraft:acacia_fence",
        "minecraft:fence_gate",
        "minecraft:spruce_fence_gate",
        "minecraft:birch_fence_gate",
        "minecraft:jungle_fence_gate",
        "minecraft:dark_oak_fence_gate",
        "minecraft:acacia_fence_gate",
        "minecraft:trapdoor",
        "minecraft:wooden_pressure_plate",
        "minecraft:chest",
        "minecraft:trapped_chest",
        "minecraft:crafting_table",
        "minecraft:bookshelf",
        "minecraft:jukebox",
        "minecraft:daylight_detector",
        "minecraft:noteblock",
        // `BlockHugeMushroom(Material.wood, ...)` at `Block.java:1359-1360`, both given an
        // `ItemBlock` at `Item.java:673-674`.
        "minecraft:brown_mushroom_block",
        "minecraft:red_mushroom_block",
        // `ItemBanner extends ItemBlock` over `standing_banner`, which is `Material.wood`.
        "minecraft:banner",
    ];
    WOOD_MATERIAL_BLOCKS.contains(&p.name)
}

/// java: `FurnaceRecipes.getSmeltingResult(stack) != null` — the whole of 1.8.9's smelting
/// table, centralized in `mc-data` from `FurnaceRecipes`' private constructor
/// (`item/crafting/FurnaceRecipes.java:29-64`). Only "does this smelt at all" is needed here;
/// the *result* is the server's business, because `ContainerFurnace` has no client-side output
/// prediction (see the `model` module doc).
///
/// The metadata rule is `compareItemStacks` (`:104-107`): a key registered through
/// `addSmelting`/`addSmeltingRecipeForBlock` gets metadata `32767`, which matches **any**
/// damage; only the three `addSmeltingRecipe` call sites pin a metadata —
/// `stonebrick` at `DEFAULT_META = 0`, `sponge` at `1` (wet), and `fish` at each
/// `FishType.canCook()` metadata, which is `COD = 0` and `SALMON = 1` only
/// (`ItemFishFood.java:81-84`), so clownfish and pufferfish do not smelt.
#[must_use]
pub fn is_smeltable(stack: &ItemStack) -> bool {
    let Ok(item) = u16::try_from(stack.item_id) else {
        return false;
    };
    mc_data::smelting_result(mc_data::RecipeStack::with_damage(
        item,
        stack.count,
        stack.damage,
    ))
    .is_some()
}

/// java: `ContainerBrewingStand.Ingredient.isItemValid` (`:160-163`) is
/// `stack.getItem().isPotionIngredient(stack)`, i.e. `getPotionEffect(stack) != null`
/// (`item/Item.java:394-402`).
///
/// The only writer of that field is the **protected** `Item.setPotionEffect(String)`.
/// `ItemFood.setPotionEffect(int,int,int,float)` is an unrelated overload and does **not** make
/// an item an ingredient, which is why golden apple, chicken, rotten flesh and poisonous potato
/// are absent here even though they call a method of that name. Extracted from
/// `Item.registerItems` by taking every `setPotionEffect(` whose argument is a
/// `PotionHelper` string, the set is exactly these 13.
///
/// The one override is `ItemFishFood.getPotionEffect` (`item/ItemFishFood.java:36-39`), which is
/// non-null only for `FishType.byItemStack(stack) == PUFFERFISH` — metadata `3`
/// (`ItemFishFood.java:84,152-161`) — and `ItemFishFood` backs **both** `fish` (349) and
/// `cooked_fish` (350) (`Item.java:859-860`), so both accept metadata 3 and nothing else.
///
/// Getting this wrong is not a free guess: a refused-locally-but-accepted-by-the-server
/// placement into an empty slot is invisible to `isChangingQuantityOnly`, so no `SetSlot`
/// corrects it — see `is_furnace_fuel`.
const POTION_INGREDIENT_ITEMS: &[&str] = &[
    "minecraft:blaze_powder",
    "minecraft:fermented_spider_eye",
    "minecraft:ghast_tear",
    "minecraft:glowstone_dust",
    "minecraft:golden_carrot",
    "minecraft:gunpowder",
    "minecraft:magma_cream",
    "minecraft:nether_wart",
    "minecraft:rabbit_foot",
    "minecraft:redstone",
    "minecraft:speckled_melon",
    "minecraft:spider_eye",
    "minecraft:sugar",
];

/// Returns whether vanilla 1.8.9 treats `stack` as a brewing-stand ingredient.
#[must_use]
pub fn is_potion_ingredient(stack: &ItemStack) -> bool {
    let name = stack_ext::props(stack).name;
    if name == "minecraft:fish" || name == "minecraft:cooked_fish" {
        return stack.damage == 3;
    }
    POTION_INGREDIENT_ITEMS.contains(&name)
}

/// java: `ContainerBrewingStand.Potion.canHoldPotion` — a potion item or an empty glass bottle.
#[must_use]
pub fn is_potion_or_bottle(stack: &ItemStack) -> bool {
    matches!(
        stack_ext::props(stack).name,
        "minecraft:potion" | "minecraft:glass_bottle"
    )
}

/// java: the `Items.dye`/`EnumDyeColor.BLUE` check in `ContainerEnchantment`'s slot-1
/// constructor. Lapis lazuli is dye metadata `4`.
#[must_use]
pub fn is_lapis(stack: &ItemStack) -> bool {
    stack_ext::props(stack).name == "minecraft:dye" && stack.damage == 4
}

/// java: `ContainerBeacon.BeaconSlot.isItemValid`.
#[must_use]
pub fn is_beacon_payment(stack: &ItemStack) -> bool {
    matches!(
        stack_ext::props(stack).name,
        "minecraft:emerald" | "minecraft:diamond" | "minecraft:gold_ingot" | "minecraft:iron_ingot"
    )
}

/// java: the armour-slot anonymous `Slot.isItemValid` in `ContainerPlayer`'s constructor.
#[must_use]
pub fn is_valid_armor(stack: &ItemStack, armor_index: u8) -> bool {
    let p = stack_ext::props(stack);
    if p.armor_slot.is_some() {
        return p.armor_slot == Some(armor_index);
    }
    (p.name == "minecraft:pumpkin" || p.name == "minecraft:skull") && armor_index == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(name: &str, damage: i16) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("known item").0 as i16;
        ItemStack {
            item_id: id,
            count: 1,
            damage,
            nbt: None,
        }
    }

    #[test]
    fn output_slot_never_accepts_anything() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::Output);
        assert!(!slot.is_item_valid(&stack("diamond", 0)));
    }

    #[test]
    fn horse_saddle_is_one_empty_slot_and_unusable_armour_is_not_hoverable() {
        let saddle = Slot::new(0, 0, 0)
            .with_role(SlotRole::HorseSaddle)
            .with_limit(1);
        let saddle_stack = stack("saddle", 0);
        assert!(saddle.is_item_valid_for(None, &saddle_stack));
        assert!(!saddle.is_item_valid_for(Some(&saddle_stack), &saddle_stack));
        assert!(!saddle.is_item_valid(&stack("diamond_sword", 0)));
        assert_eq!(saddle.item_stack_limit(&saddle_stack), 1);

        let armour = Slot::new(1, 0, 0).with_role(SlotRole::HorseArmor { can_wear: false });
        assert!(!armour.is_item_valid(&stack("diamond_helmet", 0)));
        assert!(!armour.can_be_hovered());
    }

    #[test]
    fn armor_slot_zero_accepts_helmets_and_pumpkins_and_skulls_only() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::Armor(0));
        assert!(slot.is_item_valid(&stack("diamond_helmet", 0)));
        assert!(slot.is_item_valid(&stack("pumpkin", 0)));
        assert!(slot.is_item_valid(&stack("skull", 0)));
        assert!(!slot.is_item_valid(&stack("diamond_chestplate", 0)));
        assert!(!slot.is_item_valid(&stack("diamond_boots", 0)));
    }

    #[test]
    fn armor_slot_two_is_leggings_only() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::Armor(2));
        assert!(slot.is_item_valid(&stack("diamond_leggings", 0)));
        assert!(!slot.is_item_valid(&stack("diamond_helmet", 0)));
        assert!(!slot.is_item_valid(&stack("pumpkin", 0)));
    }

    #[test]
    fn furnace_fuel_accepts_coal_and_lava_bucket_but_not_stone() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::FurnaceFuel);
        assert!(slot.is_item_valid(&stack("coal", 0)));
        assert!(slot.is_item_valid(&stack("coal", 1)), "charcoal");
        assert!(slot.is_item_valid(&stack("lava_bucket", 0)));
        assert!(slot.is_item_valid(&stack("planks", 0)));
        assert!(!slot.is_item_valid(&stack("stone", 0)));
    }

    #[test]
    fn furnace_fuel_slot_caps_the_empty_bucket_left_behind_by_burnt_lava_to_one() {
        // java: `SlotFurnaceFuel.isBucket` checks `Items.bucket` specifically (the *empty*
        // bucket a burnt lava bucket turns into via `getContainerItem()`), not `lava_bucket`
        // itself — which is fuel (`isItemFuel`) but not "a bucket" for this limit override.
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::FurnaceFuel);
        assert_eq!(slot.item_stack_limit(&stack("bucket", 0)), 1);
        assert_eq!(slot.item_stack_limit(&stack("lava_bucket", 0)), 64);
        assert_eq!(slot.item_stack_limit(&stack("coal", 0)), 64);
    }

    #[test]
    fn enchant_lapis_slot_wants_blue_dye_specifically() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::EnchantLapis);
        assert!(slot.is_item_valid(&stack("dye", 4)));
        assert!(!slot.is_item_valid(&stack("dye", 0)));
        assert!(!slot.is_item_valid(&stack("diamond", 0)));
    }

    #[test]
    fn beacon_slot_accepts_only_the_four_payment_items() {
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::Beacon).with_limit(1);
        assert!(slot.is_item_valid(&stack("emerald", 0)));
        assert!(slot.is_item_valid(&stack("diamond", 0)));
        assert!(slot.is_item_valid(&stack("gold_ingot", 0)));
        assert!(slot.is_item_valid(&stack("iron_ingot", 0)));
        assert!(!slot.is_item_valid(&stack("gold_nugget", 0)));
        assert_eq!(slot.item_stack_limit(&stack("emerald", 0)), 1);
    }

    #[test]
    fn the_fuel_slot_takes_material_wood_blocks_and_refuses_the_two_circuits_lookalikes() {
        // The four items the hand-written list got wrong. `ladder` and `wooden_button` are
        // `Material.circuits` (`BlockLadder.java:23`, `BlockButton.java:30`) so
        // `getItemBurnTime` returns 0; the two mushroom blocks are
        // `new BlockHugeMushroom(Material.wood, ...)` (`Block.java:1359-1360`) with an
        // `ItemBlock` (`Item.java:673-674`) so they burn for 300 ticks.
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::FurnaceFuel);
        assert!(
            !slot.is_item_valid(&stack("ladder", 0)),
            "Material.circuits"
        );
        assert!(
            !slot.is_item_valid(&stack("wooden_button", 0)),
            "Material.circuits"
        );
        assert!(slot.is_item_valid(&stack("brown_mushroom_block", 0)));
        assert!(slot.is_item_valid(&stack("red_mushroom_block", 0)));
        // `ItemBanner extends ItemBlock` over `standing_banner` (`Material.wood`).
        assert!(slot.is_item_valid(&stack("banner", 0)));
    }

    #[test]
    fn a_wooden_block_whose_item_is_not_an_item_block_is_not_fuel() {
        // `getItemBurnTime`'s wood branch is guarded by `item instanceof ItemBlock`. The sign
        // and door *blocks* are `Material.wood`, but `ItemSign` and `ItemDoor` both extend
        // `Item` directly, so neither ever reaches the material test. Negative control for the
        // derivation: a list built from block materials alone would wrongly accept these.
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::FurnaceFuel);
        assert!(!slot.is_item_valid(&stack("sign", 0)));
        assert!(!slot.is_item_valid(&stack("wooden_door", 0)));
    }

    #[test]
    fn the_brewing_ingredient_slot_takes_pufferfish_only_out_of_the_whole_fish_item() {
        // `ItemFishFood.getPotionEffect` is non-null only at `FishType.PUFFERFISH`, metadata 3,
        // and backs both `fish` (349) and `cooked_fish` (350).
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::BrewIngredient);
        assert!(!slot.is_item_valid(&stack("fish", 0)), "raw cod");
        assert!(!slot.is_item_valid(&stack("fish", 1)), "raw salmon");
        assert!(!slot.is_item_valid(&stack("fish", 2)), "clownfish");
        assert!(slot.is_item_valid(&stack("fish", 3)), "raw pufferfish");
        assert!(
            slot.is_item_valid(&stack("cooked_fish", 3)),
            "cooked_fish shares ItemFishFood, so metadata 3 is an ingredient there too"
        );
        assert!(!slot.is_item_valid(&stack("cooked_fish", 0)));
    }

    #[test]
    fn the_brewing_ingredient_set_is_exactly_the_java_1_8_registry_set() {
        // These are the thirteen `Item.registerItems` entries which call the String overload of
        // `Item.setPotionEffect`. Keeping the oracle at the name-set seam also proves that later
        // additions such as glistering_melon_slice, pufferfish and dragon_breath did not leak in.
        assert_eq!(
            POTION_INGREDIENT_ITEMS,
            [
                "minecraft:blaze_powder",
                "minecraft:fermented_spider_eye",
                "minecraft:ghast_tear",
                "minecraft:glowstone_dust",
                "minecraft:golden_carrot",
                "minecraft:gunpowder",
                "minecraft:magma_cream",
                "minecraft:nether_wart",
                "minecraft:rabbit_foot",
                "minecraft:redstone",
                "minecraft:speckled_melon",
                "minecraft:spider_eye",
                "minecraft:sugar",
            ]
        );

        // Check the public predicate as well as the backing name oracle: every entry has to
        // produce a locally placeable stack.  This catches a stale registry spelling even when
        // the table itself happens to look right.
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::BrewIngredient);
        for name in POTION_INGREDIENT_ITEMS {
            let bare_name = name
                .strip_prefix("minecraft:")
                .expect("the registry table is namespaced");
            assert!(
                slot.is_item_valid(&stack(bare_name, 0)),
                "{name} is a Java 1.8.9 potion ingredient"
            );
        }
    }

    #[test]
    fn set_potion_effect_with_an_int_does_not_make_a_food_an_ingredient() {
        // `ItemFood.setPotionEffect(int,int,int,float)` is a different method from
        // `Item.setPotionEffect(String)`; only the latter writes `potionEffect`. These four
        // foods call the int overload and are therefore *not* brewing ingredients, while
        // `spider_eye` calls both and is.
        let slot = Slot::new(0, 0, 0).with_role(SlotRole::BrewIngredient);
        assert!(!slot.is_item_valid(&stack("golden_apple", 0)));
        assert!(!slot.is_item_valid(&stack("chicken", 0)));
        assert!(!slot.is_item_valid(&stack("rotten_flesh", 0)));
        assert!(!slot.is_item_valid(&stack("poisonous_potato", 0)));
        assert!(slot.is_item_valid(&stack("spider_eye", 0)));
        assert!(slot.is_item_valid(&stack("nether_wart", 0)));
    }

    #[test]
    fn a_plain_slot_falls_back_to_the_default_limit() {
        let slot = Slot::new(0, 0, 0);
        assert_eq!(
            slot.item_stack_limit(&stack("stone", 0)),
            DEFAULT_STACK_LIMIT
        );
        assert!(slot.is_item_valid(&stack("stone", 0)));
    }
}
