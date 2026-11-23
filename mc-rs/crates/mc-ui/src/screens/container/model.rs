//! [`Container`]: the slot-click state machine. java: `Container.java`'s `slotClick` and its
//! supporting statics, plus every `ContainerXxx.transferStackInSlot` override for the
//! shift-click merge ranges (ported in [`super::layouts`]).
//!
//! # Why a flat item array instead of vanilla's `IInventory` graph
//!
//! Vanilla composes a `Container` out of `Slot`s that each point at an `IInventory` (the
//! player's own inventory, a chest's tile entity, a synthetic 2-slot `InventoryCrafting`, ...)
//! plus an index into it. That indirection exists so several *different* Java objects can be
//! read/written uniformly. This port has no such need: every slot a `Container` owns here is
//! either a real, server-synced slot with a fixed position in the window, or (armour/crafting)
//! still just an index into the same window — there is exactly one `Container` alive per open
//! window and nothing else ever touches its slots. So [`Container`] stores one flat
//! `Vec<Option<ItemStack>>`, indexed by [`crate::screens::container::slot::Slot::id`], which is
//! observably identical to the Java graph for every click scenario below (verified against
//! `Container.java` line by line) while avoiding an `Rc<RefCell<_>>` graph — see the arena-index rule (no `Rc<RefCell<>>` in the "hot" crates; this crate is not one of them, but there is no
//! reason to reach for it when a flat array suffices).
//!
//! # What is *not* reproduced, and why that is still correct
//!
//! - **Most `onPickupFromSlot` side effects**: furnace XP, achievements, and anvil damage are
//!   server-authoritative. `SlotCrafting`'s matrix consumption and `ContainerRepair`'s local XP/
//!   input transitions feed the next click prediction and are reproduced below.
//!
//!   **`canTakeStack` is *not* in that category** and used to be listed here as if it were.
//!   1.8.9 has exactly one override (the anvil output slot) and its inputs are both client-side;
//!   [`Container::can_take_stack`] evaluates it and the five `slotClick` arms below gate on it.
//!   The claim this bullet used to make — that skipping it "only ever costs one extra optimistic
//!   click that the server's resync silently fixes" — is backwards for modes 0/2/4: those send
//!   byte-identical packets either way, so the server *accepts* the transaction and (because
//!   `isChangingQuantityOnly` suppresses `sendSlotContents`/`updateHeldItem` on the accept path)
//!   never sends a correction at all. See [`super::slot`]'s module doc.
//! - **World-side anvil damage and statistics**: vanilla computes these only on the authoritative
//!   side. The client-owned `updateRepairOutput` cost/output and pickup input transitions are
//!   reproduced here; block damage, sounds and achievements remain server work.
//!
//!   **`CraftingManager` is not in that category**, and its full static/dynamic adapter is
//!   reproduced here.
//!   `EntityPlayerMP.sendSlotContents` (`:854-863`) refuses to send **any** slot that is
//!   `instanceof SlotCrafting`, so a craft result is never pushed incrementally at all — not
//!   "a frame or two late", *never*, outside a full `S30PacketWindowItems` resync. A client
//!   without this adapter would show an empty or stale 2x2/3x3 result slot and
//!   sends the wrong `clicked_item`. The full static/dynamic recipe table is evaluated here.
//!
//!   **The smelting *table* is a different thing and is ported** — see
//!   [`super::slot::is_smeltable`]. `ContainerFurnace.transferStackInSlot` branches on
//!   `getSmeltingResult(...) != null` to decide *which slot a shift-click lands in*, which is
//!   client-side routing, not output prediction. Getting it wrong desyncs: the mode-1 packet is
//!   byte-identical either way, so the server accepts it, runs its own routing, and
//!   `isChangingQuantityOnly` suppresses the corrective `SetSlot`.
//! - **`detectAndSendChanges`/`ICrafting` listeners**: the server uses them to push ordinary
//!   state, while client-side `CreativeCrafting` and `GuiRepair` also participate. Their local
//!   effects are represented by the creative-screen diff and repair-output adapters here.

use std::collections::{BTreeMap, BTreeSet};

use mc_nbt::{Nbt, NbtCompound};

use super::slot::{Slot, DEFAULT_STACK_LIMIT};
use super::stack_ext::{self, ItemStack};

/// Whether a network stack names an actual 1.8.9 registry entry. Unknown stacks remain in the
/// authoritative slot/cursor snapshots so the UI can display and echo them, but must never enter
/// the local Java container simulation through the fabricated fallback properties.
fn registered_stack(stack: &ItemStack) -> bool {
    stack_ext::try_props(stack).is_some()
}

fn known_or_empty(stack: Option<&ItemStack>) -> bool {
    stack.is_none_or(registered_stack)
}

fn all_registered(stacks: &[Option<ItemStack>]) -> bool {
    stacks.iter().all(|stack| known_or_empty(stack.as_ref()))
}

/// A locally evaluated crafting result and its recipe provenance. Keeping the provenance is
/// important: when no local recipe matches, an authoritative result received in `S30` must not
/// be erased merely because the client cannot evaluate that matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PredictedCraft {
    /// A static or dynamic entry from the complete 1.8.9 recipe manager table.
    Vanilla {
        /// Dynamic recipe kind, when remaining-item handling must use the dynamic adapter.
        kind: Option<mc_data::DynamicRecipeKind>,
        /// NBT operation applied to the NBT-free registry result.
        transform: mc_data::DynamicRecipeTransform,
    },
}

fn dynamic_kind_for_result(
    transform: &mc_data::DynamicRecipeTransform,
    grid: mc_data::CraftingGrid<'_>,
) -> Option<mc_data::DynamicRecipeKind> {
    use mc_data::DynamicRecipeKind as K;
    use mc_data::DynamicRecipeTransform as T;
    match transform {
        T::Plain if mc_data::repair_result(grid).is_some() => Some(K::RepairItem),
        T::Plain => None,
        T::DyeLeather { .. } => Some(K::ArmorDyes),
        T::FireworkRocket { .. } | T::FireworkStar { .. } | T::FireworkFade { .. } => {
            Some(K::Fireworks)
        }
        T::AddBannerPattern { .. } => Some(K::BannerAddPattern),
        T::CloneWrittenBook { .. } => Some(K::BookCloning),
        T::CloneMapDisplayName { .. } => Some(K::MapCloning),
        T::ScaleMap { .. } => Some(K::MapExtending),
        T::DuplicateBanner { .. } => Some(K::BannerDuplicatePattern),
    }
}

/// Test-side reader of one enchantment list. The anvil itself reads through
/// [`enchantment_map`], which is the exact `EnchantmentHelper.getEnchantments`.
#[cfg(test)]
fn stack_enchantments(stack: &ItemStack, stored: bool) -> Vec<(u8, i16)> {
    let key = if stored { "StoredEnchantments" } else { "ench" };
    stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_list(key))
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let compound = entry.as_compound()?;
            let id = u8::try_from(compound.get_i16("id")?).ok()?;
            let level = compound.get_i16("lvl")?;
            Some((id, level))
        })
        .collect()
}

/// `Items.enchanted_book`. `EnchantmentHelper.getEnchantments`/`setEnchantments` branch on it to
/// use `StoredEnchantments` instead of `ench`.
const ENCHANTED_BOOK_ID: i16 = 403;

/// java: `EnchantmentHelper.getEnchantments` (`:73-90`). A book reads
/// `ItemEnchantedBook.getEnchantments` (`StoredEnchantments`, any list), anything else
/// `getEnchantmentTagList` (`ench`, a compound list). The result is a `LinkedHashMap`: a repeated
/// id keeps its first position and takes the later level, and an id stays in the map whether or
/// not an enchantment is registered for it. `getCompoundTagAt` reads a non-compound element of
/// the untyped book list as an empty compound, i.e. id `0` at level `0`.
fn enchantment_map(stack: &ItemStack) -> Vec<(i16, i16)> {
    let book = stack.item_id == ENCHANTED_BOOK_ID;
    let key = if book { "StoredEnchantments" } else { "ench" };
    let mut map: Vec<(i16, i16)> = Vec::new();
    let list = stack.nbt.as_ref().and_then(|root| root.get(key));
    let entries = match list {
        Some(list) if book || list.list_tag_type() == Some(mc_nbt::TagType::Compound) => {
            list.as_list().unwrap_or(&[])
        }
        _ => &[],
    };
    for entry in entries {
        let (id, level) = entry.as_compound().map_or((0, 0), |compound| {
            (
                compound.get_i16("id").unwrap_or(0),
                compound.get_i16("lvl").unwrap_or(0),
            )
        });
        match map.iter_mut().find(|(existing, _)| *existing == id) {
            Some(slot) => slot.1 = level,
            None => map.push((id, level)),
        }
    }
    map
}

/// `Enchantment.getEnchantmentById`: only ids `0..=255` with a registered enchantment resolve.
fn registered_enchantment(id: i16) -> Option<&'static mc_data::EnchantmentInfo> {
    u8::try_from(id).ok().and_then(mc_data::enchantment)
}

fn enchantment_entry(id: i16, level: i16) -> Nbt {
    Nbt::Compound(
        [
            ("id".to_owned(), Nbt::Short(id)),
            ("lvl".to_owned(), Nbt::Short(level)),
        ]
        .into_iter()
        .collect(),
    )
}

/// java: `EnchantmentHelper.setEnchantments` (`:95-116`). Only registered ids are
/// written, in map order. A book receives each one through
/// `ItemEnchantedBook.addEnchantment` and never gets an `ench` list; any other stack gets the
/// list as its `ench` tag. An empty result removes `ench`.
fn set_enchantments(stack: &mut ItemStack, map: &[(i16, i16)]) {
    let registered: Vec<(i16, i16)> = map
        .iter()
        .copied()
        .filter(|(id, _)| registered_enchantment(*id).is_some())
        .collect();
    let book = stack.item_id == ENCHANTED_BOOK_ID;
    if book {
        for &(id, level) in &registered {
            add_stored_enchantment(stack, id, level);
        }
    }
    if !registered.is_empty() {
        if !book {
            let list = registered
                .iter()
                .map(|&(id, level)| enchantment_entry(id, level))
                .collect();
            stack
                .nbt
                .get_or_insert_with(NbtCompound::new)
                .insert("ench", Nbt::list(list));
        }
    } else if let Some(root) = stack.nbt.as_mut() {
        root.remove("ench");
    }
}

/// java: `ItemEnchantedBook.addEnchantment` (`:69-104`): raise the matching stored entry's level
/// (never lower it), or append a new entry, then store the list back as `StoredEnchantments`.
fn add_stored_enchantment(stack: &mut ItemStack, id: i16, level: i16) {
    let root = stack.nbt.get_or_insert_with(NbtCompound::new);
    // `getEnchantments(stack)`: the existing list when the key holds one, else a fresh list.
    if !matches!(root.get("StoredEnchantments"), Some(Nbt::List(_))) {
        root.insert("StoredEnchantments", Nbt::list(Vec::new()));
    }
    let Some(list) = root.get_mut("StoredEnchantments") else {
        return;
    };
    let mut found = false;
    for entry in list.as_list_mut().into_iter().flatten() {
        // A non-compound element reads as a fresh empty compound: id 0, and a write to that
        // copy is lost.
        let existing_id = entry
            .as_compound()
            .map_or(0, |compound| compound.get_i16("id").unwrap_or(0));
        if existing_id == id {
            if let Some(compound) = entry.as_compound_mut() {
                if compound.get_i16("lvl").unwrap_or(0) < level {
                    compound.insert("lvl", Nbt::Short(level));
                }
            }
            found = true;
            break;
        }
    }
    if !found {
        // `NBTTagList.appendTag` refuses an element of the wrong type, as `append_tag` does.
        let _ = list.append_tag(enchantment_entry(id, level));
    }
}

/// java: `StringUtils.isBlank`: empty, or every char is `Character.isWhitespace` (Java 8's
/// Unicode 6.2 tables: the Zs/Zl/Zp separators minus the three no-break spaces, plus
/// `U+0009..=U+000D` and `U+001C..=U+001F`).
fn java_is_blank(text: &str) -> bool {
    text.chars().all(|c| {
        matches!(c, '\u{1C}'..='\u{1F}' | '\u{180E}')
            || (c.is_whitespace() && !matches!(c, '\u{85}' | '\u{A0}' | '\u{2007}' | '\u{202F}'))
    })
}

fn enchantment_can_apply(info: &mc_data::EnchantmentInfo, stack: &ItemStack) -> bool {
    let props = stack_ext::props(stack);
    // `EnchantmentTarget::can_enchant_item` is the shared Java predicate. Five subclasses widen
    // it in 1.8.9 (`canApply` overrides): damage enchantments accept axes, Efficiency and Silk
    // Touch accept shears, Thorns accepts every `ItemArmor`, and Unbreaking accepts every
    // damageable stack.
    info.target.can_enchant_item(props)
        || (matches!(info.id, 16..=18) && props.kind == mc_data::ItemKind::Axe)
        || (matches!(info.id, 32 | 33) && props.kind == mc_data::ItemKind::Shears)
        || (info.id == 7 && props.kind == mc_data::ItemKind::Armor)
        || (info.id == 34 && stack_ext::is_stack_damageable(stack))
}

/// `Enchantment.canApplyTogether` for the 1.8.9 registrations. The generated data retains the
/// target and weight but not each Java subclass' small incompatibility override; these are the
/// only non-trivial pairs in that version.
fn enchantments_can_apply_together(new_id: u8, old_id: u8) -> bool {
    if new_id == old_id {
        return false;
    }
    if matches!(new_id, 16..=18) && matches!(old_id, 16..=18) {
        return false;
    }
    // `EnchantmentProtection.canApplyTogether` allows Feather Falling (type 2) to coexist with
    // every other protection, but rejects two equal types and the other pairings. The generated
    // table stores ids rather than Java's protectionType, which is identical for ids 0..4.
    if matches!(new_id, 0..=4) && matches!(old_id, 0..=4) {
        return new_id == 2 || old_id == 2;
    }
    // Preserve the tiny directional asymmetry in the 1.8.9 subclasses: LootBonus rejects Silk
    // Touch, while Silk Touch only rejects Fortune (not Looting).
    if matches!(new_id, 21 | 35) && old_id == 33 {
        return false;
    }
    if new_id == 33 && old_id == 35 {
        return false;
    }
    true
}

fn nbt_compound_subset(expected: Option<&NbtCompound>, actual: Option<&NbtCompound>) -> bool {
    let Some(expected) = expected else {
        return true;
    };
    let Some(actual) = actual else {
        return false;
    };
    expected.matches(actual, false)
}

fn merchant_stack_matches(input: &ItemStack, required: &ItemStack) -> bool {
    input.item_id == required.item_id
        && input.damage == required.damage
        && input.count >= required.count
        && nbt_compound_subset(required.nbt.as_ref(), input.nbt.as_ref())
}

/// java: `ItemStack.getRepairCost` -- `hasKey("RepairCost", 3)`, so only a
/// `TAG_Int` counts (no numeric coercion), and its value is used as stored: a negative cost is
/// not clamped.
fn stack_repair_cost(stack: &ItemStack) -> i32 {
    match stack.nbt.as_ref().and_then(|root| root.get("RepairCost")) {
        Some(Nbt::Int(cost)) => *cost,
        _ => 0,
    }
}

fn stack_custom_name(stack: &ItemStack) -> Option<&str> {
    stack
        .nbt
        .as_ref()
        .and_then(|root| root.get_compound("display"))
        .and_then(|display| display.get_str("Name"))
}

fn input_has_custom_name(stack: &ItemStack) -> bool {
    stack_custom_name(stack).is_some()
}

fn stack_display_name(stack: &ItemStack) -> String {
    stack_custom_name(stack)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            stack_ext::props(stack)
                .display_name_for(stack.damage)
                .to_owned()
        })
}

fn clear_custom_name(stack: &mut ItemStack) {
    let Some(root) = stack.nbt.as_mut() else {
        return;
    };
    let remove_display =
        if let Some(display) = root.get_mut("display").and_then(Nbt::as_compound_mut) {
            display.remove("Name");
            display.is_empty()
        } else {
            false
        };
    if remove_display {
        root.remove("display");
    }
    if root.is_empty() {
        stack.nbt = None;
    }
}

fn set_custom_name(stack: &mut ItemStack, name: &str) {
    let root = stack.nbt.get_or_insert_with(NbtCompound::new);
    let display = if !root.contains_type("display", mc_nbt::TagType::Compound) {
        root.insert("display", Nbt::Compound(NbtCompound::new()));
        root.get_mut("display")
            .and_then(Nbt::as_compound_mut)
            .expect("inserted display compound")
    } else {
        root.get_mut("display")
            .and_then(Nbt::as_compound_mut)
            .expect("display is a compound")
    };
    display.insert("Name", Nbt::String(name.to_owned()));
}

/// Which vanilla `ContainerXxx` a [`Container`] was built as — determines
/// [`Container::transfer_stack_in_slot`]'s merge ranges and which slots count as "the player's
/// own inventory" for the number-key hotbar swap (mode `2`). See [`super::layouts`] for the
/// builder functions that produce each shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContainerKind {
    /// The always-open player inventory screen: 1 craft result + 2×2 craft grid + 4 armour +
    /// 27 main + 9 hotbar. java: `ContainerPlayer`.
    Player,
    /// `top` container-specific slots, then the usual 27 main + 9 hotbar. Covers chests (any
    /// row count, including a double chest's 54), the ender chest (27), hoppers (5), and
    /// dispensers/droppers (9, identical GUIs in vanilla). java: `ContainerChest`,
    /// `ContainerHopper`, `ContainerDispenser` — all three have the exact same shift-click
    /// shape, just a different `top`.
    TopThenPlayer {
        /// How many leading slots belong to the block/entity inventory.
        top: usize,
    },
    /// `ContainerHorseInventory`: saddle/armour and optional chest slots, then the player
    /// inventory. The first two slots have entity-dependent validity.
    Horse {
        /// Number of mount-owned slots: 2 for an unchested horse, 17 for a chested mount.
        top: usize,
    },
    /// `ContainerWorkbench`: craft result(0) + 3x3 grid(1-9), then 27 main + 9 hotbar.
    /// 46 slots. The 3x3 sibling of [`ContainerKind::Player`]'s 2x2.
    Workbench,
    /// `ContainerFurnace`: input(0) fuel(1) output(2), then 27 main + 9 hotbar.
    Furnace,
    /// `ContainerBrewingStand`: 3 potion bottles(0-2) + ingredient(3), then 27 main + 9 hotbar.
    BrewingStand,
    /// `ContainerEnchantment`: item(0) lapis(1), then 27 main + 9 hotbar.
    Enchanting,
    /// `ContainerRepair` (anvil): input1(0) input2(1) output(2), then 27 main + 9 hotbar.
    Anvil,
    /// `ContainerBeacon`: payment(0), then 27 main + 9 hotbar.
    Beacon,
    /// `ContainerMerchant`: two buy slots + merchant result, then 27 main + 9 hotbar.
    Merchant,
}

/// The concrete `IInventory` object a slot points at in vanilla.  The flat model does not keep
/// Java's inventory graph, but `GuiContainer.mouseReleased` still compares these object identities
/// during a shift-double-click sweep (`slot2.inventory == slot.inventory`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackingInventory {
    Player,
    CraftResult,
    CraftingMatrix,
    Container,
    AnvilInput,
    AnvilOutput,
}

/// One `MerchantRecipeList` entry received on the legacy `MC|TrList` channel.
#[derive(Debug, Clone, PartialEq)]
pub struct MerchantRecipe {
    /// First required buy stack.
    pub buy_a: ItemStack,
    /// Optional second required buy stack.
    pub buy_b: Option<ItemStack>,
    /// Stack delivered by the villager.
    pub sell: ItemStack,
    /// Number of successful uses reported by the server.
    pub uses: i32,
    /// Maximum uses reported by the server.
    pub max_uses: i32,
    /// Whether the server marked this offer disabled.
    pub disabled: bool,
}

/// A server-synced window: slots, their current contents, the cursor stack, and the drag/
/// transaction state machine. One instance per open window (the always-present player inventory
/// counts as one, `window_id == 0`). java: `Container`.
#[derive(Debug, Clone, PartialEq)]
pub struct Container {
    /// `Container.windowId`. `0` is always the player's own inventory.
    pub window_id: u8,
    /// Which vanilla container shape this is, for shift-click routing.
    pub kind: ContainerKind,
    slots: Vec<Slot>,
    items: Vec<Option<ItemStack>>,
    /// Client-local identities for the `ItemStack` objects stored in each slot. Java compares
    /// the held stack by reference while an item is in use; packet replacements therefore need
    /// to remain observable even when every serialized stack field is unchanged.
    item_identities: Vec<u64>,
    next_item_identity: u64,
    /// `InventoryPlayer.itemStack` — the stack currently picked up by the mouse. Modelled on
    /// `Container` rather than a separate player-inventory type because at most one window (and
    /// therefore at most one live cursor) is ever open at a time in this client.
    cursor: Option<ItemStack>,
    /// The identity the cursor's object had in the slot it was picked up
    /// whole from, so putting it back down keeps it (Java moves the object itself); `0` when the
    /// cursor holds a new object (a split, a copy, a packet).
    cursor_identity: u64,
    /// A server packet touched one of an enchanting table's two backing-inventory slots. The
    /// screen consumes this edge so a direct `Container::apply_set_slot`/
    /// `Container::apply_window_items` call cannot leave stale client-visible offers behind.
    pub(crate) enchanting_inputs_changed: bool,
    drag_event: u8,
    drag_mode: i32,
    drag_slots: BTreeSet<u16>,
    transaction_id: i16,
    /// Which locally evaluated recipe currently owns slot `0`, if any. This lets the local
    /// predictor retract its own stale result without destroying a server-supplied result for a
    /// recipe that is not represented locally yet.
    predicted_craft: Option<PredictedCraft>,
    /// Client-side `map_<id>` scales made available to the dynamic map-extending recipe.
    /// `RecipesMapExtending.matches` reads `MapData.scale` from the client world; keeping this
    /// small sidecar on the container lets a host feed that already-synchronised fact without
    /// making the UI depend on `mc-world` or the network driver.
    map_scales: BTreeMap<i32, i8>,
    /// `ContainerRepair.maximumCost`, or `None` before local inputs/name have been evaluated.
    ///
    /// **It is `None` in every real session today, and feeding it from
    /// `S31PacketWindowProperty` would be a bug, not a fix.** `ContainerRepair` is the one
    /// container in 1.8.9 whose only `sendProgressBarUpdate` call sits in `onCraftGuiOpened`
    /// (`ContainerRepair.java:389-393`) — it does not override `detectAndSendChanges`, so a
    /// vanilla server emits exactly one `S31PacketWindowProperty(window, 0, 0)` when the window
    /// opens and never another. Vanilla's *client* computes `maximumCost` itself in
    /// `updateRepairOutput`, which this container evaluates locally.
    ///
    /// So the value the wire offers is `0` forever, and gating on it would refuse **every**
    /// anvil take — a worse divergence than the one [`Self::can_take_stack`] exists to fix.
    /// While this is `None` that method falls back to `Slot.java:150`'s `return true`. Set it
    /// from the local `updateRepairOutput` evaluator and the gate becomes live with no other
    /// change.
    pub maximum_cost: Option<i32>,
    /// Number of material items consumed by the current local anvil result.
    anvil_material_cost: i32,
    /// Current `ContainerRepair.repairedItemName`, supplied by `GuiRepair`'s text field.
    anvil_name: String,
    /// Creative mode affects the anvil's forty-level cap and enchanted-book applicability.
    anvil_creative: bool,
    /// Slot 0's `ItemStack.getDisplayName()` in the player's language as
    /// the screen resolved it, with the stack it was resolved for. See
    /// [`Self::set_anvil_input_display_name`].
    anvil_input_display_name: Option<(ItemStack, String)>,
    /// The currently selected merchant offer, if the host has delivered `MC|TrList`.
    merchant_recipe: Option<MerchantRecipe>,
    /// `EntityPlayer.experienceLevel`, or `None` when the host has not supplied it.
    ///
    /// [`Self::can_take_stack`]'s anvil predicate needs it. Treating a missing level as `0`
    /// would make **every** survival-mode anvil take fail, which is a worse divergence than the
    /// bug being fixed, so `None` satisfies the level conjunct and leaves the other two
    /// (`maximum_cost > 0`, `has_stack`) doing their work. Set it with
    /// [`Self::set_player_experience_level`] — or, from a screen,
    /// [`super::screen::ContainerScreen::set_player_experience_level`].
    player_experience_level: Option<i32>,
    /// java: `Container.inventoryItemStacks`, the copy of every slot
    /// that `detectAndSendChanges` last compared against. It belongs to the container, so on
    /// window 0 it lives as long as Java's `ContainerPlayer` (until the next join or respawn),
    /// and the creative screen's `CreativeCrafting` listener uploads only what changed since its
    /// last look, also across openings. Server writes (`putStackInSlot`) never touch it.
    listener_mirror: Vec<Option<ItemStack>>,
}

impl Container {
    /// Build an empty container over `slots` (all slots start empty, no cursor stack).
    #[must_use]
    pub fn new(window_id: u8, kind: ContainerKind, slots: Vec<Slot>) -> Self {
        let items = vec![None; slots.len()];
        let item_identities = vec![0; slots.len()];
        let listener_mirror = vec![None; slots.len()];
        Container {
            window_id,
            kind,
            slots,
            items,
            item_identities,
            next_item_identity: 0,
            cursor: None,
            cursor_identity: 0,
            enchanting_inputs_changed: false,
            drag_event: 0,
            drag_mode: -1,
            drag_slots: BTreeSet::new(),
            transaction_id: 0,
            predicted_craft: None,
            map_scales: BTreeMap::new(),
            maximum_cost: None,
            anvil_material_cost: 0,
            anvil_name: String::new(),
            anvil_creative: false,
            anvil_input_display_name: None,
            merchant_recipe: None,
            player_experience_level: None,
            listener_mirror,
        }
    }

    /// java: `Container.detectAndSendChanges` for a container with a
    /// listener: every slot whose stack differs from the mirror (`ItemStack.areItemStacksEqual`)
    /// is copied into the mirror and reported, in slot order.
    pub fn detect_listener_changes(&mut self) -> Vec<(u16, Option<ItemStack>)> {
        let mut changes = Vec::new();
        for (index, mirrored) in self.listener_mirror.iter_mut().enumerate() {
            let now = self.items.get(index).cloned().flatten();
            if stack_ext::stacks_equal(now.as_ref(), mirrored.as_ref()) {
                continue;
            }
            mirrored.clone_from(&now);
            changes.push((index as u16, now));
        }
        changes
    }

    /// Forget what the listener has seen, as the new
    /// `ContainerPlayer` of the `EntityPlayerSP` that `handleRespawn` builds starts with an empty
    /// `inventoryItemStacks`.
    pub fn reset_listener_mirror(&mut self) {
        self.listener_mirror.fill(None);
    }

    /// Every slot, in `slotNumber` order.
    #[must_use]
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// Refresh the tracked horse's `canWearArmor()` predicate on slot 1. The open-window packet
    /// tells us only the chest size, so this must be supplied once the entity watcher snapshot
    /// resolves its horse type (donkeys/llamas expose the slot but reject armour).
    pub fn set_horse_can_wear_armor(&mut self, can_wear: bool) {
        if !matches!(self.kind, ContainerKind::Horse { .. }) {
            return;
        }
        if let Some(slot) = self.slots.get_mut(1) {
            slot.role = super::slot::SlotRole::HorseArmor { can_wear };
        }
    }

    /// The stack a slot currently holds, or `None` for an empty slot / an out-of-range index.
    #[must_use]
    pub fn slot_stack(&self, index: usize) -> Option<&ItemStack> {
        self.items.get(index).and_then(|s| s.as_ref())
    }

    /// Identity of the current stack object in `index`, or `None` for an empty/out-of-range
    /// slot. Every direct inventory replacement receives a new identity even if its value is
    /// byte-for-byte equal to the previous stack.
    #[must_use]
    pub fn slot_stack_identity(&self, index: usize) -> Option<u64> {
        self.items
            .get(index)
            .is_some_and(Option::is_some)
            .then(|| self.item_identities[index])
    }

    /// The stack currently picked up by the mouse.
    #[must_use]
    pub fn cursor(&self) -> Option<&ItemStack> {
        self.cursor.as_ref()
    }

    /// Consume the server-update edge for an enchanting table's input inventory. This is
    /// `pub(crate)` because only [`super::screen::ContainerScreen`] needs to bridge the model's
    /// packet path to `GuiEnchantment`'s client-side invalid-input clear.
    pub(crate) fn take_enchanting_input_change(&mut self) -> bool {
        std::mem::take(&mut self.enchanting_inputs_changed)
    }

    /// Directly overwrite a slot's contents, bypassing every click rule. This is what
    /// `WindowItems`/`SetSlot` application (see [`super::net`]) and test setup use — never the
    /// click state machine itself, which only ever mutates through `self.items[..]` internally.
    ///
    /// java: `Container.putStackInSlot` / `putStacksInSlots` (`Container.java:538-552`) →
    /// `Slot.putStack` → **the backing `IInventory`'s** `setInventorySlotContents`, which is
    /// where the 64-item clamp lives or does not — see [`Self::slot_clamps_to_stack_limit`].
    pub fn set_slot_contents(&mut self, index: usize, stack: Option<ItemStack>) {
        if index < self.items.len() {
            self.next_item_identity = self.next_item_identity.wrapping_add(1);
            if self.next_item_identity == 0 {
                self.next_item_identity = 1;
            }
            self.item_identities[index] = if stack.is_some() {
                self.next_item_identity
            } else {
                0
            };
            self.items[index] = stack;
            if self.is_crafting_result_slot(index) {
                // A direct result-slot write came from WindowItems/SetSlot (or test setup), not
                // from our local evaluator. Preserve it until a locally matched recipe wins.
                self.predicted_craft = None;
            }
            if self.slot_clamps_to_stack_limit(index) {
                if let Some(s) = self.items[index].as_mut() {
                    if s.count > DEFAULT_STACK_LIMIT {
                        s.count = DEFAULT_STACK_LIMIT;
                    }
                }
            }
            if self.is_crafting_grid_slot(index) {
                self.recompute_craft_result();
            } else if self.kind == ContainerKind::Anvil && index < 2 {
                self.update_repair_output();
            } else if self.kind == ContainerKind::Merchant && index < 2 {
                self.recompute_merchant_result();
            }
        }
    }

    /// Write a new value into the stack object already in `index`, the way
    /// Java's client mutates a live `ItemStack` (`--stackSize`, `damageItem`) rather than
    /// replacing it: the slot keeps its [`Self::slot_stack_identity`] while it stays non-empty.
    /// Emptying the slot is still a replacement by `null`, and filling an empty slot creates a
    /// new object. Every other effect is [`Self::set_slot_contents`]'s.
    ///
    /// java: `ItemRenderer.updateEquippedItem` and `EntityPlayer.onUpdate`'s item-in-use check
    /// both compare the held stack by reference, so this is what keeps a placed block or a
    /// sword's hit damage from looking like a new stack in the hand.
    pub fn set_slot_contents_in_place(&mut self, index: usize, stack: Option<ItemStack>) {
        let identity = self.slot_stack_identity(index);
        self.set_slot_contents(index, stack);
        if let Some(identity) = identity {
            if self.slot_stack(index).is_some() {
                self.item_identities[index] = identity;
            }
        }
    }

    /// The click state machine put a *new* Java object in `index` (a split,
    /// a copy, `new ItemStack(..)`), as opposed to the in-place `stackSize` arithmetic of a merge
    /// or a partial pickup, which keeps the slot's identity. `ItemRenderer.updateEquippedItem`
    /// compares the held stack by reference, so without this a stack swapped into the held slot
    /// looked like the old object and the hand never showed it.
    fn renew_item_identity(&mut self, index: usize) {
        self.next_item_identity = self.next_item_identity.wrapping_add(1);
        if self.next_item_identity == 0 {
            self.next_item_identity = 1;
        }
        self.item_identities[index] = if self.items[index].is_some() {
            self.next_item_identity
        } else {
            0
        };
    }

    /// `index` now holds an object that moved there from elsewhere (Java's
    /// `slot.putStack(thatObject)`): it keeps `identity`, or gets a fresh one when the object's
    /// origin is unknown (`0`).
    fn carry_item_identity(&mut self, index: usize, identity: u64) {
        if identity == 0 || self.items[index].is_none() {
            self.renew_item_identity(index);
        } else {
            self.item_identities[index] = identity;
        }
    }

    fn crafting_grid_shape(&self) -> Option<(usize, usize, u8, u8)> {
        match self.kind {
            ContainerKind::Player => Some((1, 5, 2, 2)),
            ContainerKind::Workbench => Some((1, 10, 3, 3)),
            _ => None,
        }
    }

    fn is_crafting_result_slot(&self, index: usize) -> bool {
        index == 0 && self.crafting_grid_shape().is_some()
    }

    fn is_crafting_grid_slot(&self, index: usize) -> bool {
        self.crafting_grid_shape()
            .is_some_and(|(start, end, _, _)| (start..end).contains(&index))
    }

    fn result_inputs_registered(&self, index: usize) -> bool {
        if self.is_crafting_result_slot(index) {
            return self
                .crafting_grid_shape()
                .is_none_or(|(start, end, _, _)| all_registered(&self.items[start..end]));
        }
        if index == 2 && matches!(self.kind, ContainerKind::Anvil | ContainerKind::Merchant) {
            return all_registered(&self.items[..2]);
        }
        true
    }

    fn recipe_stack(stack: &ItemStack) -> Option<mc_data::RecipeStack> {
        let item = u16::try_from(stack.item_id).ok()?;
        Some(mc_data::RecipeStack::with_damage(
            item,
            stack.count,
            stack.damage,
        ))
    }

    fn wire_stack(stack: mc_data::RecipeStack) -> ItemStack {
        ItemStack {
            item_id: stack.item.0 as i16,
            count: stack.count,
            damage: stack.damage,
            nbt: None,
        }
    }

    fn crafting_grid(&self) -> Option<(Vec<Option<mc_data::RecipeStack>>, u8, u8)> {
        let (start, end, width, height) = self.crafting_grid_shape()?;
        let slots = self.items[start..end]
            .iter()
            .map(|stack| stack.as_ref().and_then(Self::recipe_stack))
            .collect();
        Some((slots, width, height))
    }

    fn dynamic_slot_data(&self, stack: Option<&ItemStack>) -> mc_data::DynamicSlotData {
        let Some(stack) = stack else {
            return mc_data::DynamicSlotData::default();
        };
        let map_scale = (stack.item_id == 358)
            .then(|| self.map_scales.get(&i32::from(stack.damage)).copied())
            .flatten();
        let Some(root) = stack.nbt.as_ref() else {
            return mc_data::DynamicSlotData {
                map_scale,
                ..mc_data::DynamicSlotData::default()
            };
        };
        let display = root.get_compound("display");
        // ItemArmor.hasColor uses hasKey("color", 3), so numeric-but-not-Int tags do not
        // contribute to the dye average (NBTTagCompound.getInteger's broad numeric coercion is
        // only used after that exact-type gate).
        let leather_color = display
            .filter(|tag| tag.contains_type("color", mc_nbt::TagType::Int))
            .and_then(|tag| tag.get_i32("color"))
            .map(|v| v as u32);
        let book_generation = root.get_i32("generation").unwrap_or(0);
        let firework_has_explosion = root.get_compound("Explosion").is_some();
        let block_entity = root.get_compound("BlockEntityTag");
        // TileEntityBanner.getBaseColor checks only key presence, then calls getInteger. A
        // malformed nonnumeric value therefore contributes zero rather than falling back to the
        // stack metadata; keep the Option distinction for a genuinely absent key.
        let banner_base_color = block_entity.and_then(|tag| {
            tag.contains_key("Base")
                .then(|| tag.get_i32("Base").unwrap_or(0))
        });
        let banner_patterns = block_entity
            .and_then(|tag| tag.get("Patterns"))
            .and_then(|patterns| {
                let entries = patterns.as_list()?;
                // Java's getTagList("Patterns", 10) returns a fresh empty list when a
                // non-empty list carries another element type. An empty list has no patterns
                // regardless of its stored type.
                (entries.is_empty() || patterns.list_tag_type() == Some(mc_nbt::TagType::Compound))
                    .then_some(entries)
            })
            .map_or(0, |patterns| patterns.len().min(255) as u8);
        mc_data::DynamicSlotData {
            leather_color,
            book_generation,
            firework_has_tag: true,
            firework_has_explosion,
            banner_base_color,
            banner_patterns,
            map_scale,
        }
    }

    fn dynamic_slot_data_for_grid(&self) -> Option<Vec<mc_data::DynamicSlotData>> {
        self.crafting_grid_shape().map(|(start, end, _, _)| {
            self.items[start..end]
                .iter()
                .map(|stack| self.dynamic_slot_data(stack.as_ref()))
                .collect()
        })
    }

    fn source_stack(&self, grid_start: usize, source_slot: usize) -> Option<&ItemStack> {
        self.items
            .get(grid_start + source_slot)
            .and_then(Option::as_ref)
    }

    fn ensure_root(stack: &mut ItemStack) -> &mut NbtCompound {
        stack.nbt.get_or_insert_with(NbtCompound::new)
    }

    fn ensure_compound<'a>(root: &'a mut NbtCompound, key: &str) -> &'a mut NbtCompound {
        if !root.contains_type(key, mc_nbt::TagType::Compound) {
            root.insert(key, Nbt::Compound(NbtCompound::new()));
        }
        root.get_mut(key)
            .and_then(Nbt::as_compound_mut)
            .expect("ensure_compound inserted a compound")
    }

    fn copied_stack(
        &self,
        grid_start: usize,
        source_slot: usize,
        mut result: ItemStack,
    ) -> ItemStack {
        if let Some(source) = self.source_stack(grid_start, source_slot) {
            result.nbt = source.nbt.clone();
        }
        result
    }

    fn apply_dynamic_transform(
        &self,
        grid_start: usize,
        result: ItemStack,
        transform: &mc_data::DynamicRecipeTransform,
    ) -> ItemStack {
        use mc_data::DynamicRecipeTransform as T;
        match transform {
            T::Plain => result,
            T::DyeLeather { source_slot, color } => {
                let mut result = self.copied_stack(grid_start, *source_slot, result);
                let display = Self::ensure_compound(Self::ensure_root(&mut result), "display");
                display.insert("color", Nbt::Int(*color as i32));
                result
            }
            T::FireworkRocket {
                flight,
                has_charges,
                explosion_slots,
            } => {
                let mut result = result;
                // RecipeFireworks only installs the root `Fireworks` compound when at least one
                // charge was present (`l > 0`). Gunpowder still sets Flight, but a plain
                // paper+gunpowder rocket is deliberately an untagged ItemStack.
                if *has_charges {
                    let root = Self::ensure_root(&mut result);
                    let mut fireworks = NbtCompound::new();
                    fireworks.insert("Flight", Nbt::Byte(*flight as i8));
                    let explosions = explosion_slots
                        .iter()
                        .filter_map(|slot| self.source_stack(grid_start, *slot))
                        .filter_map(|stack| stack.nbt.as_ref()?.get("Explosion").cloned())
                        .collect();
                    fireworks.insert("Explosions", Nbt::list(explosions));
                    root.insert("Fireworks", Nbt::Compound(fireworks));
                }
                result
            }
            T::FireworkStar {
                colors,
                flicker,
                trail,
                shape,
            } => {
                let mut result = result;
                let root = Self::ensure_root(&mut result);
                let mut explosion = NbtCompound::new();
                explosion.insert(
                    "Colors",
                    Nbt::IntArray(colors.iter().map(|v| *v as i32).collect()),
                );
                explosion.insert("Type", Nbt::Byte(*shape as i8));
                // RecipeFireworks only writes these boolean tags when their modifier item was
                // present; absent glowstone/diamond means absent keys, not explicit zero bytes.
                if *flicker {
                    explosion.insert("Flicker", Nbt::Byte(1));
                }
                if *trail {
                    explosion.insert("Trail", Nbt::Byte(1));
                }
                root.insert("Explosion", Nbt::Compound(explosion));
                result
            }
            T::FireworkFade {
                source_slot,
                colors,
                explosion_exists,
            } => {
                let mut result = self.copied_stack(grid_start, *source_slot, result);
                if *explosion_exists {
                    let root = Self::ensure_root(&mut result);
                    let explosion = Self::ensure_compound(root, "Explosion");
                    explosion.insert(
                        "FadeColors",
                        Nbt::IntArray(colors.iter().map(|v| *v as i32).collect()),
                    );
                } else {
                    // RecipeFireworks mutates the detached value returned by
                    // NBTTagCompound.getCompoundTag("Explosion") when the key is absent. The
                    // source copy therefore remains byte-for-byte unchanged.
                }
                result
            }
            T::CloneWrittenBook {
                source_slot,
                generation,
            } => {
                let mut result = self.copied_stack(grid_start, *source_slot, result);
                let root = Self::ensure_root(&mut result);
                root.insert("generation", Nbt::Int(*generation));
                result
            }
            T::CloneMapDisplayName { source_slot } => {
                let mut result = result;
                if let Some(source) = self.source_stack(grid_start, *source_slot) {
                    if let Some(name) = stack_custom_name(source) {
                        set_custom_name(&mut result, name);
                    }
                }
                result
            }
            T::ScaleMap { source_slot } => {
                let mut result = self.copied_stack(grid_start, *source_slot, result);
                let root = Self::ensure_root(&mut result);
                root.insert("map_is_scaling", Nbt::Byte(1));
                result
            }
            T::AddBannerPattern {
                source_slot,
                pattern,
                color,
            } => {
                let mut result = self.copied_stack(grid_start, *source_slot, result);
                let root = Self::ensure_root(&mut result);
                let block_entity = Self::ensure_compound(root, "BlockEntityTag");
                let pattern = Nbt::Compound(
                    [
                        ("Pattern".to_owned(), Nbt::String((*pattern).to_owned())),
                        ("Color".to_owned(), Nbt::Int(i32::from(*color))),
                    ]
                    .into_iter()
                    .collect(),
                );
                if block_entity.contains_type("Patterns", mc_nbt::TagType::List) {
                    // NBTTagCompound.getTagList("Patterns", 10) returns a detached empty
                    // list for a non-empty list of another type; appendTag then has no effect.
                    // Nbt::append_tag preserves that typed-list rejection instead of creating a
                    // mixed list that vanilla could never serialize.
                    if let Some(patterns) = block_entity.get_mut("Patterns") {
                        let _ = patterns.append_tag(pattern);
                    }
                } else {
                    block_entity.insert("Patterns", Nbt::list(vec![pattern]));
                }
                result
            }
            T::DuplicateBanner { source_slot } => {
                self.copied_stack(grid_start, *source_slot, result)
            }
        }
    }

    /// Re-evaluate vanilla's complete `CraftingManager` table. Unsupported or unavailable
    /// recipes deliberately leave an authoritative server result untouched; only a result
    /// previously produced here may be retracted.
    fn recompute_craft_result(&mut self) {
        let Some((start, end, _, _)) = self.crafting_grid_shape() else {
            return;
        };
        if !all_registered(&self.items[start..end]) {
            return;
        }
        let Some((slots, width, height)) = self.crafting_grid() else {
            return;
        };
        let grid = mc_data::CraftingGrid::new(width, height, &slots)
            .expect("the container's fixed crafting shape is valid");
        let data = self.dynamic_slot_data_for_grid().unwrap_or_default();
        match mc_data::find_matching_vanilla_recipe_dynamic(grid, &data) {
            mc_data::DynamicRecipeEvaluation::Match(Some(result)) => {
                let mut output = Self::wire_stack(result.stack);
                let origin = self
                    .crafting_grid_shape()
                    .map_or(1, |(start, _, _, _)| start);
                output = self.apply_dynamic_transform(origin, output, &result.transform);
                self.items[0] = Some(output);
                self.predicted_craft = Some(PredictedCraft::Vanilla {
                    kind: dynamic_kind_for_result(&result.transform, grid),
                    transform: result.transform,
                });
            }
            mc_data::DynamicRecipeEvaluation::Match(None) => {
                if self.predicted_craft.take().is_some() {
                    self.items[0] = None;
                }
            }
            mc_data::DynamicRecipeEvaluation::NoMatch => {
                if self.predicted_craft.take().is_some() {
                    self.items[0] = None;
                }
            }
        }
    }

    /// `Item.hasContainerItem` has exactly three registrations in vanilla 1.8.9: the filled
    /// bucket variants all leave an empty bucket. Keeping this as an id mapping also avoids
    /// conflating the ordinary empty bucket (which has no container item of its own).
    fn crafting_container_item(item: mc_data::ItemId) -> Option<mc_data::ItemId> {
        matches!(item.0, 326 | 327 | 335).then_some(mc_data::ItemId(325))
    }

    /// Client-side half of `SlotCrafting.onPickupFromSlot`. It consumes one item from every
    /// occupied matrix slot, restores container items, then immediately recomputes the result so
    /// shift-click retry sees the next craft.
    ///
    /// Most result slots carry [`PredictedCraft`] provenance from the local matrix callback. A
    /// result can also arrive in a `WindowItems`/`SetSlot` snapshot after the matrix was already
    /// populated; those writes deliberately clear provenance so an unsupported authoritative
    /// result is not erased. Vanilla still runs `SlotCrafting.onPickupFromSlot` for that later
    /// pickup, so when provenance is absent we re-evaluate the *current* matrix solely to choose
    /// its remaining-item adapter. A genuine no-match keeps vanilla's fallback (returning each
    /// input to the same slot), which is a net no-op and must not consume an authoritative result.
    fn pickup_craft_result(&mut self, index: usize) {
        if !self.is_crafting_result_slot(index) {
            return;
        }
        let Some((slots, width, height)) = self.crafting_grid() else {
            return;
        };
        let grid = mc_data::CraftingGrid::new(width, height, &slots)
            .expect("the container's fixed crafting shape is valid");
        let (start, end, _, _) = self.crafting_grid_shape().unwrap();
        let kind = match self.predicted_craft.take() {
            Some(PredictedCraft::Vanilla { kind, .. }) => kind,
            None => {
                // `set_slot_contents(0, ...)` may have replaced a locally computed result with
                // an authoritative snapshot. Re-run the same complete ordered evaluator that
                // populates the result slot, but do not mutate that slot here: this hook is
                // reached after the clicked stack has already been removed.
                let data = self.dynamic_slot_data_for_grid().unwrap_or_default();
                let mc_data::DynamicRecipeEvaluation::Match(Some(result)) =
                    mc_data::find_matching_vanilla_recipe_dynamic(grid, &data)
                else {
                    return;
                };
                dynamic_kind_for_result(&result.transform, grid)
            }
        };
        let remaining: Vec<Option<ItemStack>> = if let Some(kind) = kind {
            let data = self.dynamic_slot_data_for_grid().unwrap_or_default();
            mc_data::dynamic_remaining_items(kind, grid, &data, Self::crafting_container_item)
                .into_iter()
                .map(|item| {
                    item.map(|item| {
                        let mut stack = Self::wire_stack(item.stack);
                        if let Some(source_slot) = item.copy_nbt_from {
                            stack.nbt = self
                                .items
                                .get(start + source_slot)
                                .and_then(Option::as_ref)
                                .and_then(|source| source.nbt.clone());
                        }
                        stack
                    })
                })
                .collect()
        } else {
            mc_data::remaining_items(grid, Self::crafting_container_item)
                .into_iter()
                .map(|item| item.map(Self::wire_stack))
                .collect()
        };

        for item in &mut self.items[start..end] {
            if let Some(stack) = item {
                stack.count = stack.count.saturating_sub(1);
                if stack.count == 0 {
                    *item = None;
                }
            }
        }

        for (offset, remainder) in remaining.into_iter().enumerate() {
            let Some(remainder) = remainder else {
                continue;
            };
            let grid_index = start + offset;
            if self.items[grid_index].is_none() {
                self.items[grid_index] = Some(remainder);
            } else {
                // Vanilla tries InventoryPlayer next and drops the remainder only when it cannot
                // fit. Dropped entities are not represented by this UI-only model, but the local
                // inventory mutation is still exact whenever space exists.
                let _dropped = self.add_stack_to_player_inventory(remainder);
            }
        }
        self.recompute_craft_result();
    }

    /// Whether a slot's backing `IInventory` clamps an incoming stack to
    /// `getInventoryStackLimit()` (always `64`).
    ///
    /// java: `InventoryBasic.setInventorySlotContents` (`:160-170`) clamps;
    /// `InventoryPlayer.setInventorySlotContents` (`:539-550`),
    /// `InventoryCrafting.setInventorySlotContents` (`:133-137`) and
    /// `InventoryCraftResult.setInventorySlotContents` (`:91-94`) do not. So which slots clamp
    /// is decided by what the *client* backs each window with:
    ///
    /// * every window that arrives with slots is a `ContainerLocalMenu extends InventoryBasic`
    ///   (`NetHandlerPlayClient.handleOpenWindow:1123-1126`, and `:1099` for
    ///   `minecraft:container`) — chests, hoppers, dispensers, furnaces, brewing stands,
    ///   beacons: **the whole top half clamps**;
    /// * the anvil's two inputs are `new InventoryBasic("Repair", true, 2)`
    ///   (`ContainerRepair.java:51-59`) and clamp, but its output is `InventoryCraftResult` and
    ///   does not; the enchanting table's two slots are `new InventoryBasic("Enchant", true, 2)`
    ///   (`ContainerEnchantment.java:39`) and clamp;
    /// * the player's and the workbench's craft grids and result slots are `InventoryCrafting` /
    ///   `InventoryCraftResult` and **never** clamp;
    /// * every container's trailing 36 slots are `InventoryPlayer` and never clamp.
    ///
    /// This is wire-visible, not cosmetic: a `clicked_item` echo of an unclamped `100` where
    /// vanilla echoes `64` makes the server reject a click it would have accepted.
    fn slot_clamps_to_stack_limit(&self, index: usize) -> bool {
        let index = index as u16;
        if index >= self.non_player_prefix() {
            return false; // InventoryPlayer
        }
        match self.kind {
            // InventoryCrafting + InventoryCraftResult all the way down.
            ContainerKind::Player | ContainerKind::Workbench => false,
            // InventoryBasic inputs, InventoryCraftResult output.
            ContainerKind::Anvil => index < 2,
            ContainerKind::TopThenPlayer { .. }
            | ContainerKind::Horse { .. }
            | ContainerKind::Furnace
            | ContainerKind::BrewingStand
            | ContainerKind::Enchanting
            | ContainerKind::Beacon
            | ContainerKind::Merchant => true,
        }
    }

    /// Directly overwrite the cursor stack. Used by [`super::net`]'s window-open bootstrap and
    /// by tests; the click state machine mutates `self.cursor` internally instead.
    pub fn set_cursor(&mut self, stack: Option<ItemStack>) {
        self.cursor = stack;
        self.cursor_identity = 0;
    }

    /// Run the client-owned half of `Container.onContainerClosed` and return the stack which
    /// vanilla spawns as a local `EntityItem`.
    ///
    /// `GuiContainer.onGuiClosed` calls `inventorySlots.onContainerClosed(thePlayer)` before
    /// `EntityPlayerSP.closeScreenAndDropStack` replaces `openContainer`. The base
    /// `Container` implementation is deliberately small: if `InventoryPlayer.itemStack` (the
    /// carried/cursor stack) is non-null, it calls
    /// `dropPlayerItemWithRandomChoice(stack, false)` and then clears that field. The caller
    /// owns the returned stack's short-lived local drop visual; this model owns the observable
    /// carried-stack transition.
    ///
    /// Do **not** empty ordinary container slots here. `ContainerPlayer` and
    /// `ContainerWorkbench` drop their crafting matrices, and `ContainerRepair` and
    /// `ContainerEnchantment` drop their input inventories, only behind
    /// `if (!worldObj.isRemote)`. A client container must retain those server-synchronised
    /// values until the close/reopen packets supersede them. Likewise, chest-like inventories'
    /// `closeInventory` callback is a no-op for the client's `ContainerLocalMenu`.
    ///
    /// This reset also makes a reused model unable to carry an unfinished mode-5 drag into its
    /// next window. Java normally gets that property by discarding the GUI/container object.
    /// `ContainerMerchant` happens to invoke `super.onContainerClosed` twice in 1.8.9; using
    /// `take` makes the second invocation correctly return no second local drop.
    pub fn on_container_closed(&mut self) -> Option<ItemStack> {
        self.reset_drag();
        self.drag_mode = -1;
        self.cursor_identity = 0;
        self.cursor.take()
    }

    /// java: `Container.getNextTransactionID` — pre-incremented, so the first call returns `1`.
    /// Wraps like Java's `short` on overflow rather than panicking.
    pub fn next_transaction_id(&mut self) -> i16 {
        self.transaction_id = self.transaction_id.wrapping_add(1);
        self.transaction_id
    }

    /// Whether `a` and `b` point at the same backing `IInventory` object. Used by
    /// [`super::interaction`]'s shift-double-click sweep, which in Java compares
    /// `slot2.inventory == slot.inventory`. The flat model has no inventory graph, so keep the
    /// small concrete object mapping here instead. In particular, a player's craft result and
    /// 2x2 matrix are distinct objects even though neither is part of `InventoryPlayer`.
    pub(crate) fn same_backing_region(&self, a: u16, b: u16) -> bool {
        self.backing_inventory(a)
            .zip(self.backing_inventory(b))
            .is_some_and(|(left, right)| left == right)
    }

    fn backing_inventory(&self, slot_id: u16) -> Option<BackingInventory> {
        let index = usize::from(slot_id);
        if index >= self.slots.len() {
            return None;
        }

        match self.kind {
            // java: ContainerPlayer has one InventoryCraftResult, one InventoryCrafting, and
            // one shared InventoryPlayer (including armour, main inventory, and hotbar).
            ContainerKind::Player => match index {
                0 => Some(BackingInventory::CraftResult),
                1..5 => Some(BackingInventory::CraftingMatrix),
                _ if self.slot_is_player_inventory(slot_id) => Some(BackingInventory::Player),
                _ => None,
            },
            // java: ContainerWorkbench has the same result/matrix split with a 3x3 matrix.
            ContainerKind::Workbench => match index {
                0 => Some(BackingInventory::CraftResult),
                1..10 => Some(BackingInventory::CraftingMatrix),
                _ if self.slot_is_player_inventory(slot_id) => Some(BackingInventory::Player),
                _ => None,
            },
            // java: ContainerRepair creates a two-slot InventoryBasic for inputs and a separate
            // InventoryCraftResult for output; its trailing slots still point at InventoryPlayer.
            ContainerKind::Anvil => match index {
                0..2 => Some(BackingInventory::AnvilInput),
                2 => Some(BackingInventory::AnvilOutput),
                _ if self.slot_is_player_inventory(slot_id) => Some(BackingInventory::Player),
                _ => None,
            },
            // Every other supported vanilla container uses one backing inventory for its
            // container-specific prefix and one shared InventoryPlayer for the trailing section.
            _ if self.slot_is_player_inventory(slot_id) => Some(BackingInventory::Player),
            _ if index < usize::from(self.non_player_prefix()) => Some(BackingInventory::Container),
            _ => None,
        }
    }

    /// Whether `stack` could be dropped into `slot_id` right now — combines the three checks
    /// java's `GuiContainer.mouseClickMove` makes while accumulating a drag (`Slot.isItemValid`
    /// plus the static `Container.canAddItemToSlot`/`canDragIntoSlot`). Exposed for
    /// [`super::interaction`]'s drag-splitting preview; [`Self::slot_click`]'s own `mode == 5`
    /// handling re-validates independently and is the actual source of truth (see its doc), so
    /// a caller that skips this check entirely still cannot desync anything — it would just
    /// accumulate a slot that the state machine itself silently drops on the "add" event.
    #[must_use]
    pub fn slot_accepts_for_drag(&self, slot_id: u16, stack: &ItemStack) -> bool {
        let Some(slot) = self.slots.get(slot_id as usize) else {
            return false;
        };
        Self::can_add_item_to_slot(self.items[slot_id as usize].as_ref(), stack, true)
            && slot.is_item_valid_for(self.items[slot_id as usize].as_ref(), stack)
            && self.can_drag_into_slot(slot_id)
    }

    /// java: the static `Container.canAddItemToSlot(slot2, shiftClickedSlot, true)` call in
    /// `GuiContainer.mouseReleased`'s shift-double-click sweep — unlike
    /// [`Self::slot_accepts_for_drag`] this deliberately does **not** also check
    /// `Slot.isItemValid`, so an output-only slot (a furnace's smelted result, say) that
    /// already legitimately holds a matching stack is still swept, exactly as vanilla does.
    #[must_use]
    pub fn slot_stack_would_accept(&self, slot_id: u16, stack: &ItemStack) -> bool {
        Self::can_add_item_to_slot(
            self.items.get(slot_id as usize).and_then(Option::as_ref),
            stack,
            true,
        )
    }

    /// Java `GuiContainer`'s touchscreen release/hold predicate:
    /// `Container.canAddItemToSlot(slot, draggedStack, false)`. Unlike drag splitting, this is
    /// only a compatibility check; the subsequent real mode-0 click remains responsible for
    /// the slot's own validity and stack-limit rules.
    #[must_use]
    pub fn slot_stack_accepts_touch(&self, slot_id: u16, stack: &ItemStack) -> bool {
        Self::can_add_item_to_slot(
            self.items.get(slot_id as usize).and_then(Option::as_ref),
            stack,
            false,
        )
    }

    /// The player's XP level, as `S1FPacketSetExperience` last reported it. See
    /// [`Self::can_take_stack`].
    pub fn set_player_experience_level(&mut self, level: i32) {
        self.player_experience_level = Some(level);
    }

    /// Supply or clear the client-world `MapData.scale` for one filled-map metadata id.
    ///
    /// `RecipesMapExtending.matches` runs on the client and therefore needs the map datum that
    /// vanilla's `ItemMap.getMapData` would resolve. The Java check is only `scale < 4`, so a
    /// signed negative packet value is still a match. Updating a scale immediately
    /// re-evaluates a 2x2/3x3 matrix so a map-extending result appears as soon as the map packet
    /// arrives, rather than waiting for another inventory click.
    pub fn set_map_scale(&mut self, map_id: i32, scale: Option<i8>) {
        match scale {
            Some(value) => {
                self.map_scales.insert(map_id, value);
            }
            None => {
                self.map_scales.remove(&map_id);
            }
        }
        if self.crafting_grid_shape().is_some() {
            self.recompute_craft_result();
        }
    }

    /// Set `ContainerRepair.repairedItemName` and immediately rebuild its client-owned output.
    /// `GuiRepair` calls this whenever the name field changes; keeping the text in the container
    /// also makes direct packet/test setup exercise the same cost gate as the live screen.
    pub fn set_anvil_name(&mut self, name: impl Into<String>) {
        if self.kind != ContainerKind::Anvil {
            return;
        }
        self.anvil_name = name.into();
        self.update_repair_output();
    }

    /// Keep the anvil predictor aligned with the player's capability flag. The screen calls this
    /// before click dispatch; callers that construct a bare container can use it directly.
    pub fn set_creative_mode(&mut self, creative: bool) {
        self.anvil_creative = creative;
        if self.kind == ContainerKind::Anvil {
            self.update_repair_output();
        }
    }

    /// Record slot 0's `ItemStack.getDisplayName()` as resolved in the
    /// player's language, for [`Self::update_repair_output`]'s rename test.
    ///
    /// java: `updateRepairOutput` compares `repairedItemName` with `itemstack.getDisplayName()`,
    /// which `StatCollector` translates into the client's current language: the same text
    /// `GuiRepair.sendSlotContents` puts in the name field. The model has no language table, so
    /// the screen hands its resolution over. It is used only while slot 0 still holds `stack`;
    /// otherwise (and in a bare container) the registry's English name stands in. This does not
    /// recompute: the screen's `set_anvil_name` that follows does, as `renameItem` follows
    /// `sendSlotContents` in vanilla.
    pub fn set_anvil_input_display_name(&mut self, resolved: Option<(ItemStack, String)>) {
        if self.kind == ContainerKind::Anvil {
            self.anvil_input_display_name = resolved;
        }
    }

    /// Slot 0's display name for the rename test: the screen's localized resolution when it was
    /// made for this very stack, else the registry's English name.
    fn anvil_input_display_name_of(&self, input: &ItemStack) -> String {
        match &self.anvil_input_display_name {
            Some((stack, name)) if stack == input => name.clone(),
            _ => stack_display_name(input),
        }
    }

    fn merchant_recipe_matches_slots(&self, recipe: &MerchantRecipe) -> bool {
        if recipe.disabled || recipe.uses >= recipe.max_uses {
            return false;
        }
        let first = self.items[0].as_ref();
        let second = self.items[1].as_ref();
        let matches = |buy_a: Option<&ItemStack>, buy_b: Option<&ItemStack>| {
            let Some(buy_a) = buy_a else {
                return false;
            };
            merchant_stack_matches(buy_a, &recipe.buy_a)
                && match (&recipe.buy_b, buy_b) {
                    (Some(required), Some(actual)) => merchant_stack_matches(actual, required),
                    (None, None) => true,
                    _ => false,
                }
        };
        matches(first, second) || matches(second, first)
    }

    fn recompute_merchant_result(&mut self) {
        if self.kind != ContainerKind::Merchant {
            return;
        }
        if !all_registered(&self.items[..2]) {
            return;
        }
        let Some(recipe) = self.merchant_recipe.clone() else {
            return;
        };
        self.items[2] = self
            .merchant_recipe_matches_slots(&recipe)
            .then_some(recipe.sell);
    }

    /// Install the selected offer decoded from `MC|TrList`; input changes then rebuild the
    /// merchant result using the same item/damage/NBT-subset checks as `MerchantRecipeList`.
    pub fn set_merchant_recipe(&mut self, recipe: Option<MerchantRecipe>) {
        if self.kind != ContainerKind::Merchant {
            return;
        }
        self.merchant_recipe = recipe;
        if self.merchant_recipe.is_some() {
            self.recompute_merchant_result();
        } else {
            self.items[2] = None;
        }
    }

    /// java: `SlotMerchantResult.doTrade` (`:107-123`). Unlike the
    /// `MerchantRecipeList` matcher that chose the offer, it compares **items only** (no
    /// metadata, NBT or counts) and subtracts the offer's counts in the orientation it is called
    /// with. A pair the matcher accepted reversed can therefore be charged forward when both
    /// inputs are the same item, e.g. two colours of wool.
    fn merchant_do_trade(
        &mut self,
        recipe: &MerchantRecipe,
        first_index: usize,
        second_index: usize,
    ) -> bool {
        let Some(first) = self.items[first_index].as_ref() else {
            return false;
        };
        if first.item_id != recipe.buy_a.item_id {
            return false;
        }
        let second = self.items[second_index].as_ref();
        match (&recipe.buy_b, second) {
            (Some(required), Some(second)) if required.item_id == second.item_id => {
                let required = required.count;
                if let Some(stack) = self.items[first_index].as_mut() {
                    stack.count -= recipe.buy_a.count;
                }
                if let Some(stack) = self.items[second_index].as_mut() {
                    stack.count -= required;
                }
                true
            }
            (None, None) => {
                if let Some(stack) = self.items[first_index].as_mut() {
                    stack.count -= recipe.buy_a.count;
                }
                true
            }
            _ => false,
        }
    }

    /// Client-side `SlotMerchantResult.onPickupFromSlot`: consume the selected offer's buy
    /// stacks, including the reversed-input fallback, and immediately expose the next trade.
    fn pickup_merchant_result(&mut self) {
        if self.kind != ContainerKind::Merchant {
            return;
        }
        let Some(recipe) = self.merchant_recipe.clone() else {
            return;
        };
        if recipe.disabled || recipe.uses >= recipe.max_uses {
            return;
        }
        let traded = self.merchant_do_trade(&recipe, 0, 1) || self.merchant_do_trade(&recipe, 1, 0);
        if traded {
            // java: an input whose `stackSize <= 0` is written back as `null`.
            for index in 0..2 {
                if self.items[index]
                    .as_ref()
                    .is_some_and(|stack| stack.count <= 0)
                {
                    self.items[index] = None;
                }
            }
            if let Some(current) = self.merchant_recipe.as_mut() {
                current.uses = current.uses.saturating_add(1);
            }
            self.recompute_merchant_result();
        }
    }

    /// Client-side `ContainerRepair.updateRepairOutput`, including material repairs, durability
    /// combines, enchantment compatibility/costs, renames, the forty-level cap, and the repair
    /// cost escalation written into the output stack.
    pub fn update_repair_output(&mut self) {
        if self.kind != ContainerKind::Anvil {
            return;
        }
        self.anvil_material_cost = 0;
        let Some(input) = self.items.first().and_then(Option::as_ref).cloned() else {
            self.items[2] = None;
            self.maximum_cost = Some(0);
            return;
        };

        let second = self.items.get(1).and_then(Option::as_ref).cloned();
        if !registered_stack(&input) || !known_or_empty(second.as_ref()) {
            self.items[2] = None;
            self.anvil_material_cost = 0;
            self.maximum_cost = Some(0);
            return;
        }
        let mut output = input.clone();
        // java: `EnchantmentHelper.getEnchantments(itemstack1)`, which is a book's
        // `StoredEnchantments`, with `LinkedHashMap` semantics.
        let mut enchantments = enchantment_map(&output);
        let mut level_cost = 0i32;
        // java: `int` arithmetic on two raw `getRepairCost()` values.
        let prior_cost =
            stack_repair_cost(&input).wrapping_add(second.as_ref().map_or(0, stack_repair_cost));
        let mut rename_cost = 0i32;
        // java: `Items.enchanted_book.getEnchantments(itemstack2).tagCount() > 0` counts the raw
        // list, whatever its entries hold.
        let enchanted_book = second.as_ref().is_some_and(|stack| {
            stack.item_id == ENCHANTED_BOOK_ID
                && stack
                    .nbt
                    .as_ref()
                    .and_then(|root| root.get_list("StoredEnchantments"))
                    .is_some_and(|list| !list.is_empty())
        });

        if let Some(material) = second.as_ref() {
            let output_props = stack_ext::props(&output);
            if stack_ext::is_stack_damageable(&output)
                && output_props.is_repairable_with(mc_data::ItemId(material.item_id as u16))
            {
                let mut repair_per_item =
                    i32::from(output.damage).min(i32::from(output_props.max_damage) / 4);
                if repair_per_item <= 0 {
                    self.items[2] = None;
                    self.maximum_cost = Some(0);
                    return;
                }
                let mut used = 0i32;
                while repair_per_item > 0 && used < material.count {
                    output.damage = output.damage.saturating_sub(repair_per_item as i16);
                    level_cost += 1;
                    used += 1;
                    repair_per_item =
                        i32::from(output.damage).min(i32::from(output_props.max_damage) / 4);
                }
                self.anvil_material_cost = used;
            } else {
                if !enchanted_book
                    && (output.item_id != material.item_id
                        || !stack_ext::is_stack_damageable(&output))
                {
                    self.items[2] = None;
                    self.maximum_cost = Some(0);
                    return;
                }

                if stack_ext::is_stack_damageable(&output) && !enchanted_book {
                    let first_remaining =
                        i32::from(output_props.max_damage) - i32::from(input.damage);
                    let second_remaining = i32::from(stack_ext::props(material).max_damage)
                        - i32::from(material.damage);
                    let bonus = i32::from(output_props.max_damage) * 12 / 100;
                    let combined = first_remaining + second_remaining + bonus;
                    let damage = (i32::from(output_props.max_damage) - combined).max(0);
                    if damage < i32::from(input.damage) {
                        output.damage = damage as i16;
                        level_cost += 2;
                    }
                }

                // java: `EnchantmentHelper.getEnchantments(itemstack2)`.
                for (enchantment_id, incoming_level) in enchantment_map(material) {
                    let Some(info) = registered_enchantment(enchantment_id) else {
                        continue;
                    };
                    let existing_level = enchantments
                        .iter()
                        .find_map(|(id, level)| (*id == enchantment_id).then_some(*level))
                        .unwrap_or(0);
                    let mut level = if existing_level == incoming_level {
                        incoming_level.saturating_add(1)
                    } else {
                        existing_level.max(incoming_level)
                    };
                    let mut can_apply = enchantment_can_apply(info, &input)
                        || self.anvil_creative
                        || input.item_id == ENCHANTED_BOOK_ID;
                    for (existing_id, _) in &enchantments {
                        // `canApplyTogether(getEnchantmentById(i4))`: an unregistered id is
                        // `null`, which the base and Damage/Protection overrides accept (the
                        // LootBonus/Untouching overrides would throw on it; not reproduced).
                        let compatible = registered_enchantment(*existing_id)
                            .is_none_or(|old| enchantments_can_apply_together(info.id, old.id));
                        if *existing_id != enchantment_id && !compatible {
                            can_apply = false;
                            level_cost += 1;
                        }
                    }
                    if can_apply {
                        level = level.min(i16::from(info.max_level));
                        if let Some(entry) = enchantments
                            .iter_mut()
                            .find(|(id, _)| *id == enchantment_id)
                        {
                            entry.1 = level;
                        } else {
                            enchantments.push((enchantment_id, level));
                        }
                        let mut weight = match info.weight {
                            1 => 8,
                            2 => 4,
                            5 => 2,
                            10 => 1,
                            _ => 0,
                        };
                        if enchanted_book {
                            weight = weight.max(1) / 2;
                            weight = weight.max(1);
                        }
                        level_cost += weight * i32::from(level);
                    }
                }
            }
        }

        if java_is_blank(&self.anvil_name) {
            if input_has_custom_name(&input) {
                rename_cost = 1;
                level_cost += rename_cost;
                clear_custom_name(&mut output);
            }
        } else if self.anvil_name != self.anvil_input_display_name_of(&input) {
            // `getDisplayName()` in the player's language, not English.
            rename_cost = 1;
            level_cost += rename_cost;
            set_custom_name(&mut output, &self.anvil_name);
        }

        let mut maximum_cost = prior_cost.wrapping_add(level_cost);
        if level_cost <= 0 {
            output = ItemStack {
                item_id: 0,
                count: 0,
                damage: 0,
                nbt: None,
            };
        }
        if rename_cost == level_cost && rename_cost > 0 && maximum_cost >= 40 {
            maximum_cost = 39;
        }
        if maximum_cost >= 40 && !self.anvil_creative {
            output = ItemStack {
                item_id: 0,
                count: 0,
                damage: 0,
                nbt: None,
            };
        }

        self.maximum_cost = Some(maximum_cost);
        self.items[2] = if output.count > 0 {
            // java: `k4 = itemstack1.getRepairCost()`, raised to the second input's
            // only when there is one, then `k4 * 2 + 1` in `int` arithmetic.
            let mut repair_cost = stack_repair_cost(&output);
            if let Some(material) = second.as_ref() {
                repair_cost = repair_cost.max(stack_repair_cost(material));
            }
            Self::ensure_root(&mut output).insert(
                "RepairCost",
                Nbt::Int(repair_cost.wrapping_mul(2).wrapping_add(1)),
            );
            set_enchantments(&mut output, &enchantments);
            Some(output)
        } else {
            None
        };
    }

    /// Client-side `SlotCrafting.onPickupFromSlot` analogue for an anvil output. The server still
    /// authoritatively validates and performs the world-side anvil damage/sound; local input/XP
    /// transitions are needed so the next click packet and GUI immediately match vanilla.
    pub fn pickup_anvil_output(&mut self) {
        if self.kind != ContainerKind::Anvil {
            return;
        }
        let cost = self.maximum_cost.unwrap_or(0);
        if cost <= 0 {
            return;
        }
        if !self.anvil_creative {
            if let Some(level) = self.player_experience_level.as_mut() {
                *level = (*level - cost).max(0);
            }
        }
        self.items[0] = None;
        if self.anvil_material_cost > 0 {
            if let Some(material) = self.items[1].as_mut() {
                material.count = material.count.saturating_sub(self.anvil_material_cost);
                if material.count <= 0 {
                    self.items[1] = None;
                }
            }
        } else {
            self.items[1] = None;
        }
        self.items[2] = None;
        self.anvil_material_cost = 0;
        self.maximum_cost = Some(0);
    }

    fn on_stack_picked_from_slot(&mut self, index: usize) {
        if self.kind == ContainerKind::Merchant && index == 2 {
            self.pickup_merchant_result();
        } else if self.is_crafting_result_slot(index) {
            self.pickup_craft_result(index);
        } else if self.kind == ContainerKind::Anvil && index == 2 {
            self.pickup_anvil_output();
        }
    }

    /// java: `Slot.canTakeStack(EntityPlayer)`.
    ///
    /// `Slot.java:150` returns `true`; the single override in 1.8.9 is the anvil output slot
    /// (`ContainerRepair.java:71-74`):
    ///
    /// ```text
    /// (isCreativeMode || experienceLevel >= maximumCost) && maximumCost > 0 && getHasStack()
    /// ```
    ///
    /// Note that `maximumCost > 0` is **not** exempted for creative: a creative player cannot
    /// take a zero-cost output either.
    ///
    /// Before local inputs are evaluated the cost can be unknown, and the missing XP level falls
    /// back to "conjunct satisfied" rather than to zero so a legitimate take is not refused: see
    /// [`Self::maximum_cost`] and [`Self::set_player_experience_level`].
    #[must_use]
    pub fn can_take_stack(&self, slot_id: usize, creative: bool) -> bool {
        let Some(slot) = self.slots.get(slot_id) else {
            return false;
        };
        // `ContainerRepair` is the only vanilla container that installs this anonymous
        // subclass, and it installs it at slot number 2.  Keep the container/slot identity in
        // the predicate instead of treating the enum marker as a global permission: callers can
        // construct a normal container with an `AnvilOutput` role for layout tests, but Java's
        // ordinary `Slot` in that container still inherits `Slot.canTakeStack`'s unconditional
        // `true`.
        if self.kind != ContainerKind::Anvil
            || slot_id != 2
            || slot.role != super::slot::SlotRole::AnvilOutput
        {
            return true;
        }
        let Some(cost) = self.maximum_cost else {
            return true;
        };
        let affordable = creative
            || self
                .player_experience_level
                .is_none_or(|level| level >= cost);
        affordable && cost > 0 && self.items[slot_id].is_some()
    }

    /// The 36 trailing slots every vanilla container ends with: `InventoryPlayer`'s 27 main
    /// slots and then its 9 hotbar slots.
    ///
    /// java: the identical pair of loops that closes **every** `ContainerXxx` constructor in
    /// 1.8.9 --
    /// `for (k…) for (i1…) addSlotToContainer(new Slot(playerInventory, i1 + k * 9 + 9, …))`
    /// followed by `for (l…) addSlotToContainer(new Slot(playerInventory, l, …))`. Because the
    /// backing indices and their order are the same in every container, slot `n` of one
    /// container's player section is slot `n` of every other's.
    #[must_use]
    pub fn player_section(&self) -> std::ops::Range<usize> {
        self.slots.len().saturating_sub(36)..self.slots.len()
    }

    /// Copy this container's view of `InventoryPlayer` into `dst`'s.
    ///
    /// **In vanilla there is nothing to copy.** Every container's trailing 36 `Slot`s point at
    /// the *same* `InventoryPlayer` object that `thePlayer.inventoryContainer`'s do, so a
    /// shift-click out of a chest is instantly visible in the hotbar, to
    /// `InventoryPlayer.getCurrentItem()`, and therefore to `C08PacketPlayerBlockPlacement`.
    /// This port gives each [`Container`] its own flat array (see the module doc), so the host
    /// has to re-converge them, and the server will not do it for you: `processClickWindow`'s
    /// accept path wraps `detectAndSendChanges()`/`updateHeldItem()` in
    /// `isChangingQuantityOnly = true`, which suppresses both, so an accepted click produces
    /// **no** slot traffic at all.
    ///
    /// Call this after every click on a non-zero window, and after that window's
    /// `WindowItems`/`SetSlot` are applied. Containers whose player sections are different
    /// lengths cannot both be vanilla shapes, so the copy is skipped in that case rather than
    /// guessing.
    pub fn sync_player_section_into(&self, dst: &mut Container) {
        let (src_range, dst_range) = (self.player_section(), dst.player_section());
        if src_range.len() != dst_range.len() {
            return;
        }
        for (src, target) in src_range.zip(dst_range) {
            dst.items[target] = self.items[src].clone();
            dst.item_identities[target] = self.item_identities[src];
        }
        // Keep later direct replacements from reusing an identity that was already allocated by
        // the source view. The copied value itself is intentionally preserved: both flat views are
        // modelling the same Java ItemStack object in InventoryPlayer.
        dst.next_item_identity = dst.next_item_identity.max(self.next_item_identity);
    }

    fn slot_at(&self, slot_id: i32) -> Option<&Slot> {
        if slot_id < 0 {
            return None;
        }
        self.slots.get(slot_id as usize)
    }

    /// java: `Container.canDragIntoSlot` — no container this port builds overrides it, so it is
    /// always `true`.
    fn can_drag_into_slot(&self, _slot_id: u16) -> bool {
        true
    }

    /// java: `Container.canMergeSlot` — only `ContainerPlayer` overrides it, excluding the
    /// crafting-result slot (index `0`) from double-click collection. `pub(crate)` so
    /// [`super::interaction`]'s double-click gate (which mirrors
    /// `GuiContainer.mouseReleased`'s own `canMergeSlot(null, slot)` check) can reuse the exact
    /// same rule.
    pub(crate) fn can_merge_slot(&self, slot_id: u16) -> bool {
        // java: `ContainerPlayer.canMergeSlot` and `ContainerWorkbench.canMergeSlot` are the two
        // overrides, and both say the same thing -- `slotIn.inventory != this.craftResult`.
        !(matches!(self.kind, ContainerKind::Player | ContainerKind::Workbench) && slot_id == 0)
    }

    /// How many leading slots are *not* part of the player's own inventory (armour counts as
    /// part of it, matching `Slot.inventory == InventoryPlayer` in Java — see
    /// [`Self::slot_is_player_inventory`]).
    fn non_player_prefix(&self) -> u16 {
        match self.kind {
            ContainerKind::Player => 5,     // craft result(1) + 2x2 grid(4)
            ContainerKind::Workbench => 10, // craft result(1) + 3x3 grid(9)
            ContainerKind::TopThenPlayer { top } => top as u16,
            ContainerKind::Horse { top } => top as u16,
            ContainerKind::Furnace => 3,
            ContainerKind::BrewingStand => 4,
            ContainerKind::Enchanting => 2,
            ContainerKind::Anvil => 3,
            ContainerKind::Beacon => 1,
            ContainerKind::Merchant => 3,
        }
    }

    /// java: `slot.inventory == inventoryplayer` — true for armour/main/hotbar slots (every
    /// container appends the player's own 27 main + 9 hotbar slots last, and `ContainerPlayer`
    /// additionally backs its 4 armour slots with `InventoryPlayer` itself).
    fn slot_is_player_inventory(&self, slot_id: u16) -> bool {
        slot_id >= self.non_player_prefix() && (slot_id as usize) < self.slots.len()
    }

    /// The slot number of hotbar key `0..=8`. Every container this port builds appends the
    /// player's hotbar as its final 9 slots (`for k in 0..9 { addSlotToContainer(new
    /// Slot(playerInventory, k, ...)) }` in every vanilla `ContainerXxx` constructor), so this
    /// mapping is the same arithmetic regardless of `kind`.
    fn hotbar_slot_id(&self, hotbar_index: u8) -> u16 {
        (self.slots.len() - 9 + hotbar_index as usize) as u16
    }

    /// java: `Slot.decrStackSize` (`Slot.java:134-137`), which forwards to the *backing
    /// inventory's* `decrStackSize(index, count)`. That is why the amount is not always
    /// honoured: `InventoryCraftResult.decrStackSize` discards `count` and returns the whole
    /// stack. See [`super::slot::SlotRole::takes_whole_stack`].
    fn decr_stack_size(&mut self, index: usize, amount: i32) -> Option<ItemStack> {
        self.items[index].as_ref()?;
        let whole = self
            .slots
            .get(index)
            .is_some_and(|s| s.role.takes_whole_stack());
        let take_all = whole || matches!(&self.items[index], Some(s) if s.count <= amount);
        if take_all {
            self.items[index].take()
        } else {
            let s = self.items[index].as_mut().unwrap();
            Some(stack_ext::split_stack(s, amount))
        }
    }

    /// Container slot order is main rows then hotbar, but `InventoryPlayer.mainInventory` is
    /// hotbar then main rows. Java's linear inventory scans therefore visit the trailing nine
    /// container slots before the preceding twenty-seven.
    fn player_inventory_indices(
        &self,
    ) -> std::iter::Chain<std::ops::Range<usize>, std::ops::Range<usize>> {
        let end = self.items.len();
        let start = end.saturating_sub(36);
        let hotbar = end.saturating_sub(9);
        (hotbar..end).chain(start..hotbar)
    }

    fn is_whole_player_inventory_range(&self, start: usize, end: usize) -> bool {
        end == self.items.len() && start == end.saturating_sub(36)
    }

    fn find_empty_slot(&self, start: usize, end: usize) -> Option<usize> {
        let end = end.min(self.items.len());
        if self.is_whole_player_inventory_range(start, end) {
            self.player_inventory_indices()
                .find(|&i| self.items[i].is_none())
        } else {
            (start..end).find(|&i| self.items[i].is_none())
        }
    }

    /// java: `InventoryPlayer.storeItemStack` restricted to an arbitrary `[start, end)` range.
    fn find_partial_stack_slot(
        &self,
        stack: &ItemStack,
        start: usize,
        end: usize,
    ) -> Option<usize> {
        let matches = |&i: &usize| {
            self.items[i].as_ref().is_some_and(|existing| {
                stack_ext::same_item_respecting_subtypes(existing, stack)
                    && stack_ext::is_stackable(existing)
                    && existing.count < stack_ext::max_stack_size(existing)
                    && existing.count < DEFAULT_STACK_LIMIT
                    && stack_ext::tags_equal(Some(existing), Some(stack))
            })
        };
        let end = end.min(self.items.len());
        if self.is_whole_player_inventory_range(start, end) {
            self.player_inventory_indices().find(matches)
        } else {
            (start..end).find(matches)
        }
    }

    /// java: `InventoryPlayer.storePartialItemStack`. Returns the amount of `stack` that did
    /// not fit into the one slot this call chose (`0` if it all fit).
    fn store_partial_item_stack(&mut self, stack: &ItemStack, start: usize, end: usize) -> i32 {
        let i = stack.count;
        let Some(j) = self
            .find_partial_stack_slot(stack, start, end)
            .or_else(|| self.find_empty_slot(start, end))
        else {
            return i;
        };
        if self.items[j].is_none() {
            self.items[j] = Some(ItemStack {
                item_id: stack.item_id,
                count: 0,
                damage: stack.damage,
                nbt: stack.nbt.clone(),
            });
            // java: `mainInventory[j] = new ItemStack(item, 0, meta)`.
            self.renew_item_identity(j);
        }
        let slot_stack = self.items[j].as_mut().unwrap();
        let max = stack_ext::max_stack_size(slot_stack);
        let mut k = i;
        k = k.min(max.saturating_sub(slot_stack.count));
        k = k.min(DEFAULT_STACK_LIMIT.saturating_sub(slot_stack.count));
        if k == 0 {
            return i;
        }
        slot_stack.count += k;
        i - k
    }

    /// java: `InventoryPlayer.addItemStackToInventory`, restricted to this container's trailing
    /// 36-slot main+hotbar range (the only range `mode == 2` ever routes an overflow item into).
    /// Returns the leftover that did not fit (`count == 0` when everything was placed).
    fn add_stack_to_player_inventory(&mut self, stack: ItemStack) -> ItemStack {
        if !registered_stack(&stack) {
            return stack;
        }
        let end = self.slots.len();
        let start = end.saturating_sub(36);
        let mut remaining = stack;
        if stack_ext::is_item_damaged(&remaining) {
            if let Some(empty) = self.find_empty_slot(start, end) {
                // java copies the entire damaged stack in one assignment, even if a malformed
                // server stack carries count > 1, then zeroes the source count.
                self.items[empty] = Some(remaining.clone());
                // `ItemStack.copyItemStack`: a new object.
                self.renew_item_identity(empty);
                remaining.count = 0;
            }
            return remaining;
        }
        loop {
            let before = remaining.count;
            remaining.count = self.store_partial_item_stack(&remaining, start, end);
            if remaining.count == 0 || remaining.count >= before {
                break;
            }
        }
        remaining
    }

    /// java: `Container.canAddItemToSlot` (static). `existing` is the slot's current contents
    /// (`null` slot vs. empty slot collapse to the same case here, since every slot this port
    /// addresses is real).
    fn can_add_item_to_slot(
        existing: Option<&ItemStack>,
        stack: &ItemStack,
        stack_size_matters: bool,
    ) -> bool {
        if !registered_stack(stack) || !known_or_empty(existing) {
            return false;
        }
        let mut flag = existing.is_none();
        if let Some(existing) = existing {
            if stack_ext::is_item_equal(stack, existing)
                && stack_ext::tags_equal(Some(existing), Some(stack))
            {
                let extra = if stack_size_matters { 0 } else { stack.count };
                flag |= existing.count + extra <= stack_ext::max_stack_size(stack);
            }
        }
        flag
    }

    /// java: `Container.computeStackSize` (static). `drag_stack` starts as a copy of the full
    /// cursor stack; this sets its size per the drag mode, then adds `existing_in_slot`.
    fn compute_stack_size(
        drag_mode: i32,
        drag_stack: &mut ItemStack,
        existing_in_slot: i32,
        slot_count: u32,
    ) {
        match drag_mode {
            0 => drag_stack.count /= slot_count.max(1) as i32,
            1 => drag_stack.count = 1,
            2 => drag_stack.count = stack_ext::max_stack_size(drag_stack),
            _ => {}
        }
        drag_stack.count = drag_stack.count.saturating_add(existing_in_slot);
    }

    /// Build the stack vanilla previews over one member of `dragSplittingSlots` and report how
    /// many cursor items that preview consumes. The optional count is the yellow override text
    /// `GuiContainer.drawSlot` supplies when either the item's own maximum or the slot limit
    /// clips the computed stack.
    ///
    /// This is deliberately read-only. The real distribution remains owned by
    /// [`Self::click_drag`]; the GUI calls this before release solely to reproduce
    /// `GuiContainer.drawSlot` / `updateDragSplitting`.
    pub(crate) fn drag_preview_stack(
        &self,
        slot_id: u16,
        drag_mode: i32,
        slot_count: usize,
    ) -> Option<(ItemStack, Option<i32>, i32)> {
        let cursor = self.cursor.as_ref()?;
        if !self.slot_accepts_for_drag(slot_id, cursor) {
            return None;
        }
        let index = usize::from(slot_id);
        let existing_count = self.items[index].as_ref().map_or(0, |stack| stack.count);
        let mut preview = cursor.clone();
        Self::compute_stack_size(drag_mode, &mut preview, existing_count, slot_count as u32);

        let mut override_count = None;
        let item_max = stack_ext::max_stack_size(&preview);
        if preview.count > item_max {
            preview.count = item_max;
            override_count = Some(item_max);
        }
        let slot_max = self.slots[index].item_stack_limit(&preview);
        if preview.count > slot_max {
            preview.count = slot_max;
            override_count = Some(slot_max);
        }

        let consumed = preview.count - existing_count;
        Some((preview, override_count, consumed))
    }

    /// java: `Container.resetDrag`.
    fn reset_drag(&mut self) {
        self.drag_event = 0;
        self.drag_slots.clear();
    }

    fn anvil_inputs_snapshot(&self) -> Option<[Option<ItemStack>; 2]> {
        (self.kind == ContainerKind::Anvil).then(|| {
            [
                self.items.first().cloned().flatten(),
                self.items.get(1).cloned().flatten(),
            ]
        })
    }

    fn recompute_anvil_after_input_change(&mut self, before: Option<[Option<ItemStack>; 2]>) {
        let Some(before) = before else {
            return;
        };
        let after = [
            self.items.first().cloned().flatten(),
            self.items.get(1).cloned().flatten(),
        ];
        if after != before {
            self.update_repair_output();
        }
    }

    /// The shift-click destination range for slot `index`, mirroring the matching
    /// `ContainerXxx.transferStackInSlot`. Returns `None` immediately wherever Java's override
    /// returns `null` outright (deny the transfer). See [`super::layouts`] for how each
    /// `ContainerKind`'s slot count is built so these literal ranges line up with it.
    ///
    /// java: `ContainerPlayer`/`ContainerChest`/`ContainerFurnace`/`ContainerBrewingStand`/
    /// `ContainerEnchantment`/`ContainerRepair`/`ContainerBeacon`/`ContainerDispenser`/
    /// `ContainerHopper`'s `transferStackInSlot`.
    pub fn transfer_stack_in_slot(&mut self, index: usize) -> Option<ItemStack> {
        let anvil_inputs_before = self.anvil_inputs_snapshot();
        let original = self.items.get(index)?.clone()?;
        if !registered_stack(&original) || !self.result_inputs_registered(index) {
            return None;
        }
        let mut working = original.clone();
        let total = self.slots.len();

        match self.kind {
            ContainerKind::Player => {
                if index == 0 {
                    if !self.merge_item_stack(&mut working, 9, 45, true) {
                        return None;
                    }
                } else if (1..9).contains(&index) {
                    if !self.merge_item_stack(&mut working, 9, 45, false) {
                        return None;
                    }
                } else if let Some(armor_index) = stack_ext::props(&working).armor_slot {
                    let target = 5 + armor_index as usize;
                    if self.items[target].is_none() {
                        if !self.merge_item_stack(&mut working, target, target + 1, false) {
                            return None;
                        }
                    } else if (9..36).contains(&index) {
                        if !self.merge_item_stack(&mut working, 36, 45, false) {
                            return None;
                        }
                    } else if (36..45).contains(&index) {
                        if !self.merge_item_stack(&mut working, 9, 36, false) {
                            return None;
                        }
                    } else if !self.merge_item_stack(&mut working, 9, 45, false) {
                        return None;
                    }
                } else if (9..36).contains(&index) {
                    if !self.merge_item_stack(&mut working, 36, 45, false) {
                        return None;
                    }
                } else if (36..45).contains(&index) {
                    if !self.merge_item_stack(&mut working, 9, 36, false) {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 9, 45, false) {
                    return None;
                }
            }
            ContainerKind::TopThenPlayer { top } => {
                if index < top {
                    if !self.merge_item_stack(&mut working, top, total, true) {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 0, top, false) {
                    return None;
                }
            }
            // java: `ContainerHorseInventory.transferStackInSlot` (`:116-151`). A horse
            // prioritises the player's section when moving mount-owned stacks out; when moving
            // from the player it tries armour, saddle, then chest storage in that order.
            ContainerKind::Horse { top } => {
                if index < top {
                    if !self.merge_item_stack(&mut working, top, total, true) {
                        return None;
                    }
                } else if self.items[1].is_none()
                    && self.slots[1].is_item_valid_for(self.items[1].as_ref(), &working)
                {
                    if !self.merge_item_stack(&mut working, 1, 2, false) {
                        return None;
                    }
                } else if self.slots[0].is_item_valid_for(self.items[0].as_ref(), &working) {
                    if !self.merge_item_stack(&mut working, 0, 1, false) {
                        return None;
                    }
                } else if top <= 2 || !self.merge_item_stack(&mut working, 2, top, false) {
                    return None;
                }
            }
            // java: `ContainerWorkbench.transferStackInSlot` (`:88-144`). Slot `0` is the
            // result and pushes into the player half in *reverse*; the 3x3 grid (`1..=9`) is the
            // `else` fall-through, which is why it is not spelled out as its own arm here
            // either.
            ContainerKind::Workbench => {
                if index == 0 {
                    if !self.merge_item_stack(&mut working, 10, 46, true) {
                        return None;
                    }
                } else if (10..37).contains(&index) {
                    if !self.merge_item_stack(&mut working, 37, 46, false) {
                        return None;
                    }
                } else if (37..46).contains(&index) {
                    if !self.merge_item_stack(&mut working, 10, 37, false) {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 10, 46, false) {
                    return None;
                }
            }
            ContainerKind::Furnace => {
                if index == 2 {
                    if !self.merge_item_stack(&mut working, 3, 39, true) {
                        return None;
                    }
                } else if index != 0 && index != 1 {
                    // java tests "would this smelt" **first** and only then "is this fuel"
                    // (`ContainerFurnace.java:114-129`). The order matters for the items that
                    // are both: `log` and `log2` smelt to charcoal *and* burn for 300 ticks, so
                    // vanilla sends a shift-clicked log to the **input** slot, not the fuel
                    // slot.
                    //
                    // This is not a free convenience either way. For mode 1 the `clicked_item`
                    // field is the source slot's pre-click copy on both sides, so the server
                    // accepts the click and runs *its* routing, and `isChangingQuantityOnly`
                    // then suppresses the corrective `SetSlot` -- leaving the item in a
                    // different slot on each side with no resync.
                    if super::slot::is_smeltable(&working) {
                        if !self.merge_item_stack(&mut working, 0, 1, false) {
                            return None;
                        }
                    } else if super::slot::is_furnace_fuel(&working) {
                        if !self.merge_item_stack(&mut working, 1, 2, false) {
                            return None;
                        }
                    } else if (3..30).contains(&index) {
                        if !self.merge_item_stack(&mut working, 30, 39, false) {
                            return None;
                        }
                    } else if (30..39).contains(&index)
                        && !self.merge_item_stack(&mut working, 3, 30, false)
                    {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 3, 39, false) {
                    return None;
                }
            }
            ContainerKind::BrewingStand => {
                let is_top = (0..=2).contains(&index) || index == 3;
                if !is_top {
                    if self.items[3].is_none() && self.slots[3].is_item_valid(&working) {
                        if !self.merge_item_stack(&mut working, 3, 4, false) {
                            return None;
                        }
                    } else if super::slot::is_potion_or_bottle(&working) {
                        if !self.merge_item_stack(&mut working, 0, 3, false) {
                            return None;
                        }
                    } else if (4..31).contains(&index) {
                        if !self.merge_item_stack(&mut working, 31, 40, false) {
                            return None;
                        }
                    } else if (31..40).contains(&index) {
                        if !self.merge_item_stack(&mut working, 4, 31, false) {
                            return None;
                        }
                    } else if !self.merge_item_stack(&mut working, 4, 40, false) {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 4, 40, true) {
                    return None;
                }
            }
            ContainerKind::Enchanting => {
                if index == 0 || index == 1 {
                    if !self.merge_item_stack(&mut working, 2, 38, true) {
                        return None;
                    }
                } else if super::slot::is_lapis(&working) {
                    if !self.merge_item_stack(&mut working, 1, 2, true) {
                        return None;
                    }
                } else {
                    if self.items[0].is_some() || !self.slots[0].is_item_valid(&working) {
                        return None;
                    }
                    if working.nbt.is_some() && working.count == 1 {
                        self.items[0] = Some(working.clone());
                        working.count = 0;
                    } else if working.count >= 1 {
                        self.items[0] = Some(ItemStack {
                            item_id: working.item_id,
                            count: 1,
                            damage: working.damage,
                            nbt: None,
                        });
                        working.count -= 1;
                    }
                    // A copy / `new ItemStack(..)` in the item slot.
                    self.renew_item_identity(0);
                }
            }
            ContainerKind::Anvil => {
                if index == 2 {
                    if !self.merge_item_stack(&mut working, 3, 39, true) {
                        return None;
                    }
                } else if index != 0 && index != 1 {
                    if (3..39).contains(&index) && !self.merge_item_stack(&mut working, 0, 2, false)
                    {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 3, 39, false) {
                    return None;
                }
            }
            ContainerKind::Merchant => {
                if index == 2 {
                    if !self.merge_item_stack(&mut working, 3, 39, true) {
                        return None;
                    }
                } else if index != 0 && index != 1 {
                    if (3..30).contains(&index) {
                        if !self.merge_item_stack(&mut working, 30, 39, false) {
                            return None;
                        }
                    } else if (30..39).contains(&index)
                        && !self.merge_item_stack(&mut working, 3, 30, false)
                    {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 3, 39, false) {
                    return None;
                }
            }
            ContainerKind::Beacon => {
                if index == 0 {
                    if !self.merge_item_stack(&mut working, 1, 37, true) {
                        return None;
                    }
                } else if self.items[0].is_none()
                    && self.slots[0].is_item_valid(&working)
                    && working.count == 1
                {
                    if !self.merge_item_stack(&mut working, 0, 1, false) {
                        return None;
                    }
                } else if (1..28).contains(&index) {
                    if !self.merge_item_stack(&mut working, 28, 37, false) {
                        return None;
                    }
                } else if (28..37).contains(&index) {
                    if !self.merge_item_stack(&mut working, 1, 28, false) {
                        return None;
                    }
                } else if !self.merge_item_stack(&mut working, 1, 37, false) {
                    return None;
                }
            }
        }

        if working.count == 0 {
            self.items[index] = None;
        } else {
            self.items[index] = Some(working.clone());
        }
        // java: every `transferStackInSlot` override ends
        // `if (itemstack1.stackSize == itemstack.stackSize) return null;`. It matters on the
        // wire, not just locally: `Container.slotClick`'s mode-1 arm only assigns `itemstack`
        // when `transferStackInSlot` returned non-null, so a shift-click that moved nothing
        // sends `clicked_item = <empty>` rather than echoing the stack back at the server.
        if working.count == original.count {
            return None;
        }
        self.on_stack_picked_from_slot(index);
        self.recompute_craft_result();
        self.recompute_merchant_result();
        self.recompute_anvil_after_input_change(anvil_inputs_before);
        Some(original)
    }

    /// java: `Container.mergeItemStack`. Tries to absorb `stack` into `[start, end)`, first
    /// topping up compatible existing stacks, then using the first empty slot. Mutates `stack`
    /// in place (its `count` is whatever did not fit) and returns whether *anything* moved.
    fn merge_item_stack(
        &mut self,
        stack: &mut ItemStack,
        start: usize,
        end: usize,
        reverse: bool,
    ) -> bool {
        if !registered_stack(stack) {
            return false;
        }
        let end = end.min(self.items.len());
        if start >= end {
            return false;
        }
        let indices: Vec<usize> = if reverse {
            (start..end).rev().collect()
        } else {
            (start..end).collect()
        };
        let mut changed = false;

        if stack_ext::is_stackable(stack) {
            for &i in &indices {
                if stack.count == 0 {
                    break;
                }
                if let Some(existing) = self.items[i].as_mut() {
                    if stack_ext::same_item_respecting_subtypes(existing, stack)
                        && stack_ext::tags_equal(Some(existing), Some(stack))
                    {
                        let max = stack_ext::max_stack_size(stack);
                        let total = existing.count + stack.count;
                        if total <= max {
                            stack.count = 0;
                            existing.count = total;
                            changed = true;
                        } else if existing.count < max {
                            let room = max - existing.count;
                            stack.count -= room;
                            existing.count = max;
                            changed = true;
                        }
                    }
                }
            }
        }

        if stack.count > 0 {
            for &i in &indices {
                // java: `mergeItemStack` deliberately does not call `Slot.isItemValid` here
                // (`Container.java:593-596,656-667`). Container subclasses validate restricted
                // destinations before invoking this helper; the helper itself fills the first
                // empty slot in the requested range, exactly like vanilla.
                if self.items[i].is_none() {
                    self.items[i] = Some(stack.clone());
                    // java: `slot1.putStack(stack.copy())`.
                    self.renew_item_identity(i);
                    stack.count = 0;
                    changed = true;
                    break;
                }
            }
        }

        changed
    }

    /// java: `Container.slotClick` — the whole click state machine, including drag (H10).
    /// `slot_id` is `i32` because `-999` (clicked outside the window) and `-1` (a couple of
    /// modes' "no slot" sentinel) are both legal, out-of-range-on-purpose values, exactly as in
    /// Java. Returns what the server-bound `ClickWindow` packet's `clicked_item` field should
    /// carry (see [`Self::click`]).
    pub fn slot_click(
        &mut self,
        slot_id: i32,
        clicked_button: u8,
        mode: u8,
        creative: bool,
    ) -> Option<ItemStack> {
        let clicked_stack = self
            .slot_at(slot_id)
            .and_then(|slot| self.items.get(slot.id as usize))
            .and_then(Option::as_ref)
            .cloned();
        let hotbar_known = !(mode == 2 && clicked_button < 9)
            || known_or_empty(
                self.items
                    .get(self.hotbar_slot_id(clicked_button) as usize)
                    .and_then(Option::as_ref),
            );
        let drag_slots_known = mode != 5
            || self.drag_slots.iter().all(|slot| {
                self.items
                    .get(*slot as usize)
                    .and_then(Option::as_ref)
                    .is_none_or(registered_stack)
            });
        let dependent_inputs_known = self
            .slot_at(slot_id)
            .is_none_or(|slot| self.result_inputs_registered(slot.id as usize));
        if !known_or_empty(clicked_stack.as_ref())
            || !known_or_empty(self.cursor.as_ref())
            || !hotbar_known
            || !drag_slots_known
            || !dependent_inputs_known
        {
            if mode == 5 {
                self.reset_drag();
            }
            return (mode == 0).then_some(clicked_stack).flatten();
        }
        let anvil_inputs_before = self.anvil_inputs_snapshot();
        if self.kind == ContainerKind::Anvil && self.anvil_creative != creative {
            self.anvil_creative = creative;
            if self.items[0].is_some() || self.items[1].is_some() {
                self.update_repair_output();
            }
        }
        let result = if mode == 5 {
            self.click_drag(slot_id, clicked_button, creative);
            None
        } else if self.drag_event != 0 {
            self.reset_drag();
            None
        } else if (mode == 0 || mode == 1) && (clicked_button == 0 || clicked_button == 1) {
            self.click_pickup_or_shift(slot_id, clicked_button, mode, creative)
        } else if mode == 2 && clicked_button < 9 {
            self.click_hotbar_swap(slot_id, clicked_button, creative);
            None
        } else if mode == 3 && creative && self.cursor.is_none() && slot_id >= 0 {
            self.click_creative_pick(slot_id);
            None
        } else if mode == 4 && self.cursor.is_none() && slot_id >= 0 {
            self.click_drop_key(slot_id, clicked_button, creative);
            None
        } else if mode == 6 && slot_id >= 0 {
            self.click_double_click_collect(slot_id, clicked_button, creative);
            None
        } else {
            None
        };
        self.recompute_craft_result();
        self.recompute_merchant_result();
        self.recompute_anvil_after_input_change(anvil_inputs_before);
        result
    }

    // Two distinct java conditions both call `resetDrag()` here (a stray phase transition, and
    // a cursor that emptied out mid-drag) — kept as separate `if`/`else if` arms rather than
    // OR-ed together so each still maps to one line of `Container.java:slotClick`'s own
    // mode-5 branch.
    #[allow(clippy::if_same_then_else)]
    fn click_drag(&mut self, slot_id: i32, clicked_button: u8, creative: bool) {
        let prev_event = self.drag_event;
        self.drag_event = drag_event_of(clicked_button);

        if (prev_event != 1 || self.drag_event != 2) && prev_event != self.drag_event {
            self.reset_drag();
        } else if self.cursor.is_none() {
            self.reset_drag();
        } else if self.drag_event == 0 {
            self.drag_mode = extract_drag_mode(clicked_button);
            if is_valid_drag_mode(self.drag_mode, creative) {
                self.drag_event = 1;
                self.drag_slots.clear();
            } else {
                self.reset_drag();
            }
        } else if self.drag_event == 1 {
            if let Some(slot) = self.slot_at(slot_id).map(|s| s.id) {
                let cursor = self.cursor.clone().unwrap();
                let existing = self.items[slot as usize].clone();
                let ok = Self::can_add_item_to_slot(existing.as_ref(), &cursor, true)
                    && self.slots[slot as usize]
                        .is_item_valid_for(self.items[slot as usize].as_ref(), &cursor)
                    && cursor.count > self.drag_slots.len() as i32
                    && self.can_drag_into_slot(slot);
                if ok {
                    self.drag_slots.insert(slot);
                }
            }
        } else if self.drag_event == 2 {
            if !self.drag_slots.is_empty() {
                let mut remaining = self.cursor.clone().unwrap();
                let mut j = remaining.count;
                let n = self.drag_slots.len() as u32;
                for sid in self.drag_slots.clone() {
                    let idx = sid as usize;
                    let cursor = self.cursor.clone().unwrap();
                    let existing_stack = self.items[idx].clone();
                    let ok = Self::can_add_item_to_slot(existing_stack.as_ref(), &cursor, true)
                        && self.slots[idx].is_item_valid_for(self.items[idx].as_ref(), &cursor)
                        && cursor.count >= n as i32
                        && self.can_drag_into_slot(sid);
                    if !ok {
                        continue;
                    }
                    let existing_count = existing_stack.as_ref().map_or(0, |s| s.count);
                    let mut candidate = remaining.clone();
                    Self::compute_stack_size(self.drag_mode, &mut candidate, existing_count, n);
                    let max = stack_ext::max_stack_size(&candidate);
                    if candidate.count > max {
                        candidate.count = max;
                    }
                    let limit = self.slots[idx].item_stack_limit(&candidate);
                    if candidate.count > limit {
                        candidate.count = limit;
                    }
                    j -= candidate.count - existing_count;
                    self.items[idx] = Some(candidate);
                    // java: `slot8.putStack(itemstack3.copy())`, a new
                    // object even where the slot already held the same item.
                    self.renew_item_identity(idx);
                }
                remaining.count = j.max(0);
                self.cursor = if j <= 0 { None } else { Some(remaining) };
                // java: the cursor becomes the copy `itemstack3`, not the object it held.
                self.cursor_identity = 0;
            }
            self.reset_drag();
        } else {
            self.reset_drag();
        }
    }

    fn click_pickup_or_shift(
        &mut self,
        slot_id: i32,
        clicked_button: u8,
        mode: u8,
        creative: bool,
    ) -> Option<ItemStack> {
        if slot_id == -999 {
            if let Some(cursor) = self.cursor.clone() {
                if clicked_button == 0 {
                    self.cursor = None;
                    self.cursor_identity = 0;
                } else if clicked_button == 1 {
                    let mut c = cursor;
                    let _dropped = stack_ext::split_stack(&mut c, 1);
                    self.cursor = if c.count == 0 { None } else { Some(c) };
                }
            }
            return None;
        }

        if mode == 1 {
            if slot_id < 0 {
                return None;
            }
            let slot = self.slot_at(slot_id)?;
            let sid = slot.id as usize;
            // java: `if (slot6 != null && slot6.canTakeStack(playerIn))` -- the return value is
            // assigned *inside* this gate, so a refused shift-click reports an empty
            // `clicked_item` on the wire, not the stack it would have moved.
            if !self.can_take_stack(sid, creative) {
                return None;
            }
            let pre = self.transfer_stack_in_slot(sid)?;
            let result = pre.clone();
            let retry = self.items[sid]
                .as_ref()
                .is_some_and(|s| s.item_id == pre.item_id);
            if retry {
                // java: `retrySlotClick` — re-enters the whole state machine, matching a second
                // physical shift-click, so a crafting/smelting output that produces more than
                // one merge-target's worth keeps draining until nothing more fits.
                self.slot_click(slot_id, clicked_button, 1, creative);
            }
            return Some(result);
        }

        // mode == 0: plain left/right click pickup/place/swap/combine.
        if slot_id < 0 {
            return None;
        }
        let slot = self.slot_at(slot_id).copied()?;
        let sid = slot.id as usize;
        let slot_stack = self.items[sid].clone();
        let cursor_stack = self.cursor.clone();
        // java: `if (itemstack9 != null) itemstack = itemstack9.copy();` -- the packet's
        // `clicked_item` is snapshotted *before* the `canTakeStack` gate below, which is why a
        // refused mode-0 click still puts the slot's stack on the wire and is therefore
        // *accepted* by the server (see this module's doc).
        let result = slot_stack.clone();
        // java: `else if (slot7.canTakeStack(playerIn))` at `Container.java:321` gates the whole
        // `itemstack9 != null` branch -- pickup, merge and swap alike. Only the
        // `itemstack9 == null` place-into-empty-slot arm sits outside it.
        if slot_stack.is_some() && !self.can_take_stack(sid, creative) {
            return result;
        }

        match (slot_stack, cursor_stack) {
            (None, Some(mut cursor)) => {
                if slot.is_item_valid_for(None, &cursor) {
                    let mut amount = if clicked_button == 0 { cursor.count } else { 1 };
                    let limit = slot.item_stack_limit(&cursor);
                    amount = amount.min(limit);
                    if cursor.count >= amount {
                        // java: `putStack(k1 == itemstack4.stackSize ? itemstack4 :
                        // itemstack4.splitStack(k1))`: the cursor's own object, or a new split.
                        let whole = amount == cursor.count;
                        self.items[sid] = Some(stack_ext::split_stack(&mut cursor, amount));
                        if whole {
                            self.carry_item_identity(sid, self.cursor_identity);
                            self.cursor_identity = 0;
                        } else {
                            self.renew_item_identity(sid);
                        }
                    }
                    self.cursor = if cursor.count == 0 {
                        None
                    } else {
                        Some(cursor)
                    };
                }
            }
            (Some(existing), None) => {
                let amount = if clicked_button == 0 {
                    existing.count
                } else {
                    (existing.count + 1) / 2
                };
                let slot_identity = self.item_identities[sid];
                self.cursor = self.decr_stack_size(sid, amount);
                // A whole pickup carries the slot's object to the cursor; a
                // partial one splits a new object off and leaves the slot's own in place.
                self.cursor_identity = if self.items[sid].is_none() {
                    slot_identity
                } else {
                    0
                };
                self.on_stack_picked_from_slot(sid);
            }
            (Some(mut existing), Some(mut cursor)) => {
                if slot.is_item_valid_for(Some(&existing), &cursor) {
                    if existing.item_id == cursor.item_id
                        && existing.damage == cursor.damage
                        && stack_ext::tags_equal(Some(&existing), Some(&cursor))
                    {
                        // java keeps these terms as signed `int`s. For a server-supplied
                        // over-full slot, `limit - stackSize` is negative; `splitStack` with
                        // that negative amount grows the cursor and shrinks the slot. The old
                        // `u8::saturating_sub` silently turned vanilla's bug into a no-op.
                        let desired = if clicked_button == 0 { cursor.count } else { 1 };
                        let amount = desired
                            .min(slot.item_stack_limit(&cursor) - existing.count)
                            .min(stack_ext::max_stack_size(&cursor) - existing.count);
                        cursor.count -= amount;
                        self.cursor = if cursor.count == 0 {
                            None
                        } else {
                            Some(cursor)
                        };
                        existing.count += amount;
                        self.items[sid] = Some(existing);
                    } else if cursor.count <= slot.item_stack_limit(&cursor) {
                        // The two objects trade places.
                        let slot_identity = self.item_identities[sid];
                        self.items[sid] = Some(cursor);
                        self.carry_item_identity(sid, self.cursor_identity);
                        self.cursor = Some(existing);
                        self.cursor_identity = slot_identity;
                    }
                } else if stack_ext::same_item_respecting_subtypes(&existing, &cursor)
                    && stack_ext::max_stack_size(&cursor) > 1
                    && stack_ext::tags_equal(Some(&existing), Some(&cursor))
                {
                    let moving = existing.count;
                    if moving > 0 && moving + cursor.count <= stack_ext::max_stack_size(&cursor) {
                        cursor.count += moving;
                        self.cursor = Some(cursor);
                        self.decr_stack_size(sid, moving);
                        self.on_stack_picked_from_slot(sid);
                    }
                }
            }
            (None, None) => {}
        }

        result
    }

    fn click_hotbar_swap(&mut self, slot_id: i32, hotbar_index: u8, creative: bool) {
        let Some(slot) = self.slot_at(slot_id).copied() else {
            return;
        };
        let sid = slot.id as usize;
        // java: `if (slot5.canTakeStack(playerIn))` -- `Container.java:394`.
        if !self.can_take_stack(sid, creative) {
            return;
        }
        let hotbar_id = self.hotbar_slot_id(hotbar_index) as usize;
        let hotbar_stack = self.items[hotbar_id].clone();
        // The hotbar's object, which moves into the clicked slot below.
        let hotbar_identity = self.item_identities[hotbar_id];
        let slot_is_player_inv = self.slot_is_player_inventory(slot.id);
        let mut flag = match &hotbar_stack {
            None => true,
            Some(s) => slot_is_player_inv && slot.is_item_valid_for(self.items[sid].as_ref(), s),
        };
        let mut found_empty = false;
        if !flag {
            let start = self.slots.len().saturating_sub(36);
            found_empty = self.find_empty_slot(start, self.slots.len()).is_some();
            flag |= found_empty;
        }

        if self.items[sid].is_some() && flag {
            let original = self.items[sid].clone().unwrap();
            self.items[hotbar_id] = Some(original.clone());
            // java: `setInventorySlotContents(clickedButton,
            // itemstack11.copy())`, a new object in the hotbar slot.
            self.renew_item_identity(hotbar_id);
            let cant_place_directly = (!slot_is_player_inv
                || !hotbar_stack
                    .as_ref()
                    .is_some_and(|s| slot.is_item_valid_for(self.items[sid].as_ref(), s)))
                && hotbar_stack.is_some();
            if cant_place_directly {
                if found_empty {
                    let leftover = self.add_stack_to_player_inventory(hotbar_stack.unwrap());
                    debug_assert_eq!(leftover.count, 0, "an empty slot was confirmed to exist");
                    self.items[sid] = None;
                }
            } else {
                // java: `slot4.putStack(itemstack7)`, the hotbar's old object.
                self.items[sid] = hotbar_stack;
                self.carry_item_identity(sid, hotbar_identity);
            }
            self.on_stack_picked_from_slot(sid);
        } else if self.items[sid].is_none() {
            if let Some(s) = hotbar_stack {
                if slot.is_item_valid_for(self.items[sid].as_ref(), &s) {
                    self.items[hotbar_id] = None;
                    self.items[sid] = Some(s);
                    // The hotbar's object moves to the clicked slot.
                    self.carry_item_identity(sid, hotbar_identity);
                }
            }
        }
    }

    fn click_creative_pick(&mut self, slot_id: i32) {
        let Some(slot) = self.slot_at(slot_id) else {
            return;
        };
        if let Some(existing) = self.items[slot.id as usize].clone() {
            let mut copy = existing;
            copy.count = stack_ext::max_stack_size(&copy);
            self.cursor = Some(copy);
            // `itemstack.copy()`: a new object on the cursor.
            self.cursor_identity = 0;
        }
    }

    fn click_drop_key(&mut self, slot_id: i32, clicked_button: u8, creative: bool) {
        let Some(slot) = self.slot_at(slot_id).copied() else {
            return;
        };
        let sid = slot.id as usize;
        if self.items[sid].is_none() {
            return;
        }
        // java: `if (slot3 != null && slot3.getHasStack() && slot3.canTakeStack(playerIn))`
        // -- `Container.java:450`.
        if !self.can_take_stack(sid, creative) {
            return;
        }
        let amount = if clicked_button == 0 {
            1
        } else {
            self.items[sid].as_ref().unwrap().count
        };
        let _dropped = self.decr_stack_size(sid, amount);
        self.on_stack_picked_from_slot(sid);
    }

    fn click_double_click_collect(&mut self, slot_id: i32, clicked_button: u8, creative: bool) {
        let Some(mut cursor) = self.cursor.clone() else {
            return;
        };
        let clicked_id = self.slot_at(slot_id).map(|s| s.id as usize);
        let clicked_blocks_collection = clicked_id
            .is_some_and(|sid| self.items[sid].is_some() && self.can_take_stack(sid, creative));
        // java: `if (itemstack4 != null && (slot2 == null || !slot2.getHasStack() ||
        // !slot2.canTakeStack(playerIn)))` -- the sweep only runs when the clicked slot is *not*
        // itself a valid pickup target.
        if clicked_blocks_collection {
            return;
        }

        let forward = clicked_button == 0;
        let n = self.slots.len();
        for pass in 0..2 {
            let indices: Vec<usize> = if forward {
                (0..n).collect()
            } else {
                (0..n).rev().collect()
            };
            for i in indices {
                if cursor.count >= stack_ext::max_stack_size(&cursor) {
                    break;
                }
                let Some(existing) = self.items[i].clone() else {
                    continue;
                };
                let full = existing.count >= stack_ext::max_stack_size(&existing);
                if pass == 0 && full {
                    continue;
                }
                if !Self::can_add_item_to_slot(Some(&existing), &cursor, true) {
                    continue;
                }
                // java: `&& slot8.canTakeStack(playerIn)` -- `Container.java:473`.
                if !self.can_take_stack(i, creative) {
                    continue;
                }
                if !self.can_merge_slot(i as u16) {
                    continue;
                }
                if existing.item_id != cursor.item_id
                    || existing.damage != cursor.damage
                    || !stack_ext::tags_equal(Some(&existing), Some(&cursor))
                {
                    continue;
                }
                let room = stack_ext::max_stack_size(&cursor) - cursor.count;
                let take = room.min(existing.count);
                if take == 0 {
                    continue;
                }
                let taken = self.decr_stack_size(i, take);
                cursor.count += take;
                if taken.is_some() {
                    self.on_stack_picked_from_slot(i);
                }
            }
        }
        self.cursor = Some(cursor);
    }
}

/// java: `Container.extractDragMode` — `(button >> 2) & 3`.
#[must_use]
pub fn extract_drag_mode(button: u8) -> i32 {
    i32::from((button >> 2) & 3)
}

/// java: `Container.getDragEvent` — `button & 3`.
#[must_use]
pub fn drag_event_of(button: u8) -> u8 {
    button & 3
}

/// java: `Container.func_94534_d` — packs `(event, mode)` into one wire byte for the
/// `used_button` field of a drag-mode `ClickWindow` packet.
#[must_use]
pub fn encode_drag(event: u8, mode: i32) -> u8 {
    (event & 3) | (((mode as u8) & 3) << 2)
}

/// java: `Container.isValidDragMode` — mode `2` (creative "fill to max stack") is gated on
/// creative mode; `0` (even split) and `1` (one per slot) are always legal.
#[must_use]
pub fn is_valid_drag_mode(mode: i32, creative: bool) -> bool {
    mode == 0 || mode == 1 || (mode == 2 && creative)
}

#[cfg(test)]
mod tests {
    //! Hand-derived scenarios against `Container.java`'s `slotClick`, one per click mode, plus
    //! a concentrated block on drag (mode `5`) for hazard **H10**: drag has three protocol
    //! phases (start/add-slot/end) and an approximate implementation loses or duplicates items
    //! only under specific phase sequences, which is exactly what these tests drive by hand.

    use super::super::layouts;
    use super::*;

    fn stack(item_id: i16, count: i32) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage: 0,
            nbt: None,
        }
    }

    fn tagged_stack(item_id: i16, count: i32, damage: i16, nbt: NbtCompound) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage,
            nbt: Some(nbt),
        }
    }

    /// `n` plain slots at arbitrary positions, `ContainerKind::Player`-shaped only insofar as
    /// tests that need real merge-range routing build a real layout instead (see
    /// [`layouts::player_container`] below); this is for isolated pickup/drag/hotbar-swap tests
    /// that only care about slot mechanics, not shift-click routing.
    fn tiny_container(n: u16) -> Container {
        let slots: Vec<Slot> = (0..n).map(|i| Slot::new(i, 0, 0)).collect();
        Container::new(0, ContainerKind::TopThenPlayer { top: 0 }, slots)
    }

    fn total_items(c: &Container, item_id: i16) -> i32 {
        let mut total: i32 = c
            .slots()
            .iter()
            .filter_map(|s| c.slot_stack(s.id as usize))
            .filter(|s| s.item_id == item_id)
            .map(|s| s.count)
            .sum();
        if let Some(cursor) = c.cursor() {
            if cursor.item_id == item_id {
                total += cursor.count;
            }
        }
        total
    }

    /// Java's inventory stores `ItemStack` object references. A server `SetSlot` replacement is
    /// therefore observable to `EntityPlayer.onUpdate` even when the new stack serializes to the
    /// exact same fields as the old one.
    #[test]
    fn equal_direct_slot_replacements_receive_distinct_stack_identities() {
        let mut c = tiny_container(1);
        let stone = stack(1, 3);
        c.set_slot_contents(0, Some(stone.clone()));
        let first = c
            .slot_stack_identity(0)
            .expect("the first stack has an identity");

        c.set_slot_contents(0, Some(stone));
        let second = c
            .slot_stack_identity(0)
            .expect("the equal replacement has an identity");
        assert_ne!(
            first, second,
            "replacement identity must not collapse by value"
        );

        c.set_slot_contents(0, None);
        assert_eq!(c.slot_stack_identity(0), None);
    }

    /// Java's `--stackSize` / `damageItem` change the live object: the slot
    /// keeps its identity until it empties; a later `SetSlot` is a new object again.
    #[test]
    fn in_place_writes_keep_the_stack_identity_until_the_slot_empties() {
        let mut c = tiny_container(1);
        c.set_slot_contents(0, Some(stack(35, 3)));
        let object = c.slot_stack_identity(0).expect("an identity");

        c.set_slot_contents_in_place(0, Some(stack(35, 2)));
        assert_eq!(
            c.slot_stack_identity(0),
            Some(object),
            "--stackSize keeps the object"
        );
        assert_eq!(c.slot_stack(0), Some(&stack(35, 2)));

        c.set_slot_contents(0, Some(stack(35, 2)));
        let echoed = c.slot_stack_identity(0).expect("the echo's identity");
        assert_ne!(echoed, object, "the server's SetSlot is a new object");

        c.set_slot_contents_in_place(0, None);
        assert_eq!(c.slot_stack_identity(0), None, "an emptied slot holds null");
        c.set_slot_contents_in_place(0, Some(stack(35, 1)));
        let refilled = c.slot_stack_identity(0).expect("a new object");
        assert!(refilled != object && refilled != echoed);
    }

    /// Item 6: every click that puts a *different* Java object in a slot
    /// (the held hotbar slot 36 here) gives it a different identity, objects that move keep
    /// theirs, and in-place merges and partial pickups keep the slot's. Before, the click state
    /// machine never touched identities, so a stack swapped into the held slot kept the old
    /// object's identity and `updateEquippedItem` never re-equipped.
    #[test]
    fn clicks_give_the_held_slot_a_new_identity_exactly_when_java_installs_a_new_object() {
        const HELD: usize = 36;
        const SWORD: i16 = 267;
        const BOW: i16 = 261;
        const WOOL: i16 = 35;
        let id = |c: &Container, slot: usize| c.slot_stack_identity(slot);

        // Mode 0: pick the bow up from slot 9 and click it onto the held sword: they swap.
        let mut c = layouts::player_container(0);
        c.set_slot_contents(HELD, Some(stack(SWORD, 1)));
        c.set_slot_contents(9, Some(stack(BOW, 1)));
        let (sword, bow) = (id(&c, HELD).unwrap(), id(&c, 9).unwrap());
        c.slot_click(9, 0, 0, false);
        c.slot_click(HELD as i32, 0, 0, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(BOW, 1)));
        assert_eq!(
            id(&c, HELD),
            Some(bow),
            "the bow object moved into the hand"
        );
        // Put the sword (now on the cursor) back down in slot 9: the same object again.
        c.slot_click(9, 0, 0, false);
        assert_eq!(id(&c, 9), Some(sword));

        // Mode 2: hover slot 9 (sword) and press hotbar key 1 (the held slot).
        let held_before = id(&c, HELD).unwrap();
        c.slot_click(9, 0, 2, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(SWORD, 1)));
        let held_after = id(&c, HELD).unwrap();
        assert!(
            held_after != held_before && held_after != sword,
            "java: `setInventorySlotContents(button, itemstack11.copy())`, a new object"
        );
        assert_eq!(
            id(&c, 9),
            Some(held_before),
            "the bow object moved to slot 9"
        );

        // Mode 1: shift-click wool from slot 10 into an empty held slot (a `copy()`), then
        // shift-click more wool, which merges into the held stack in place.
        let mut c = layouts::player_container(0);
        c.set_slot_contents(10, Some(stack(WOOL, 20)));
        c.slot_click(10, 0, 1, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(WOOL, 20)));
        let wool = id(&c, HELD).expect("a new object in the hand");
        c.set_slot_contents(11, Some(stack(WOOL, 5)));
        c.slot_click(11, 0, 1, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(WOOL, 25)));
        assert_eq!(
            id(&c, HELD),
            Some(wool),
            "a merge is `stackSize +=` on the same object"
        );

        // A right-click splits half onto the cursor: the held object stays, a little smaller.
        c.slot_click(HELD as i32, 1, 0, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(WOOL, 12)));
        assert_eq!(id(&c, HELD), Some(wool));
        // Clicking the half back merges in place.
        c.slot_click(HELD as i32, 0, 0, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(WOOL, 25)));
        assert_eq!(id(&c, HELD), Some(wool));
        // Picking the whole stack up and putting it straight back is the same object.
        c.slot_click(HELD as i32, 0, 0, false);
        assert_eq!(c.slot_stack_identity(HELD), None);
        c.slot_click(HELD as i32, 0, 0, false);
        assert_eq!(id(&c, HELD), Some(wool));

        // Mode 5: a left drag across the held slot installs `itemstack3.copy()` there even
        // though it already holds wool.
        c.set_slot_contents(12, Some(stack(WOOL, 10)));
        c.slot_click(12, 0, 0, false);
        c.slot_click(-999, 0, 5, false);
        c.slot_click(HELD as i32, 1, 5, false);
        c.slot_click(13, 1, 5, false);
        c.slot_click(-999, 2, 5, false);
        assert_eq!(c.slot_stack(HELD), Some(&stack(WOOL, 30)));
        assert_ne!(
            id(&c, HELD),
            Some(wool),
            "a drag puts a copy in every slot it touched"
        );
    }

    // ---------------------------------------------------------------- mode 0: plain click

    #[test]
    fn left_click_picks_up_the_whole_stack() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 20)));
        let ret = c.slot_click(0, 0, 0, false);
        assert_eq!(ret, Some(stack(1, 20)));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.cursor(), Some(&stack(1, 20)));
    }

    #[test]
    fn right_click_picks_up_half_rounded_up() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 7)));
        c.slot_click(0, 1, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 4)), "(7+1)/2 = 4");
        assert_eq!(c.slot_stack(0), Some(&stack(1, 3)));
    }

    #[test]
    fn left_click_places_the_whole_cursor_into_an_empty_slot() {
        let mut c = tiny_container(3);
        c.set_cursor(Some(stack(1, 10)));
        c.slot_click(0, 0, 0, false);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 10)));
        assert_eq!(c.cursor(), None);
    }

    #[test]
    fn right_click_places_one_into_an_empty_slot_and_keeps_the_rest() {
        let mut c = tiny_container(3);
        c.set_cursor(Some(stack(1, 10)));
        c.slot_click(0, 1, 0, false);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 1)));
        assert_eq!(c.cursor(), Some(&stack(1, 9)));
    }

    #[test]
    fn left_click_combines_matching_stacks_up_to_the_max_stack_size() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 60)));
        c.set_cursor(Some(stack(1, 20)));
        c.slot_click(0, 0, 0, false);
        // item 1 = stone, max stack 64: 60 + 20 = 80, clamps to 64, 16 stays on the cursor.
        assert_eq!(c.slot_stack(0), Some(&stack(1, 64)));
        assert_eq!(c.cursor(), Some(&stack(1, 16)));
    }

    /// java: `Container.slotClick:340-359` leaves the transfer amount signed. A stack of 65 in
    /// an unclamped `InventoryPlayer` slot therefore produces `amount = -1`: the slot shrinks
    /// to 64 and the one-item cursor grows to 2. The previous saturating subtraction made this
    /// a no-op. The ordinary 63+1 case is the negative control for accidentally inverting all
    /// same-item transfers while reproducing this hostile-server edge case.
    #[test]
    fn an_overfull_same_item_slot_uses_vanillas_negative_transfer_amount() {
        for count in 65..=127 {
            let mut overfull = layouts::player_container(0);
            overfull.set_slot_contents(9, Some(stack(1, count)));
            overfull.set_cursor(Some(stack(1, 1)));
            assert_eq!(
                overfull.slot_click(9, 0, 0, false),
                Some(stack(1, count)),
                "clicked_item is still the pre-click copy at {count}"
            );
            assert_eq!(overfull.slot_stack(9), Some(&stack(1, 64)), "{count}");
            assert_eq!(overfull.cursor(), Some(&stack(1, count - 63)), "{count}");
        }

        let mut ordinary = layouts::player_container(0);
        ordinary.set_slot_contents(9, Some(stack(1, 63)));
        ordinary.set_cursor(Some(stack(1, 1)));
        ordinary.slot_click(9, 0, 0, false);
        assert_eq!(ordinary.slot_stack(9), Some(&stack(1, 64)));
        assert_eq!(ordinary.cursor(), None);
    }

    #[test]
    fn left_click_swaps_two_different_items() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 5)));
        c.set_cursor(Some(stack(2, 3)));
        let ret = c.slot_click(0, 0, 0, false);
        assert_eq!(
            ret,
            Some(stack(1, 5)),
            "returns the slot's pre-click contents"
        );
        assert_eq!(c.slot_stack(0), Some(&stack(2, 3)));
        assert_eq!(c.cursor(), Some(&stack(1, 5)));
    }

    #[test]
    fn left_click_outside_the_window_drops_the_whole_cursor() {
        let mut c = tiny_container(3);
        c.set_cursor(Some(stack(1, 5)));
        c.slot_click(-999, 0, 0, false);
        assert_eq!(c.cursor(), None);
    }

    #[test]
    fn right_click_outside_the_window_drops_one_and_keeps_the_rest() {
        let mut c = tiny_container(3);
        c.set_cursor(Some(stack(1, 5)));
        c.slot_click(-999, 1, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 4)));
    }

    // ---------------------------------------------------------------- mode 1: shift click

    #[test]
    fn shift_click_moves_a_main_inventory_item_into_an_empty_hotbar_slot() {
        let mut c = layouts::player_container(0);
        // Slot 9 is the first main-inventory slot; the hotbar (36..45) is empty.
        c.set_slot_contents(9, Some(stack(1, 5)));
        let ret = c.slot_click(9, 0, 1, false);
        assert_eq!(ret, Some(stack(1, 5)));
        assert_eq!(c.slot_stack(9), None);
        assert!((36..45).any(|i| c.slot_stack(i) == Some(&stack(1, 5))));
    }

    #[test]
    fn shift_click_merges_metadata_only_for_items_without_subtypes() {
        let mut wool = layouts::player_container(0);
        wool.set_slot_contents(9, Some(named("wool", 5, 0))); // white, main inventory
        wool.set_slot_contents(36, Some(named("wool", 10, 1))); // orange, hotbar
        wool.slot_click(9, 0, 1, false);
        assert_eq!(
            wool.slot_stack(36),
            Some(&named("wool", 10, 1)),
            "different wool subtypes must not merge"
        );
        assert!(
            (37..45).any(|i| wool.slot_stack(i) == Some(&named("wool", 5, 0))),
            "the white wool instead uses an empty hotbar slot"
        );

        // Negative control for blindly comparing metadata: sticks have no subtypes, so their
        // otherwise-meaningless damage values do not prevent the live merge.
        let mut sticks = layouts::player_container(0);
        sticks.set_slot_contents(9, Some(named("stick", 5, 0)));
        sticks.set_slot_contents(36, Some(named("stick", 10, 7)));
        sticks.slot_click(9, 0, 1, false);
        assert_eq!(sticks.slot_stack(36), Some(&named("stick", 15, 7)));
        assert_eq!(sticks.slot_stack(9), None);
    }

    #[test]
    fn shift_click_with_nowhere_to_go_returns_none_and_changes_nothing() {
        let mut c = layouts::player_container(0);
        // Fill every main+hotbar slot with an incompatible item, then shift-click a lone
        // compatible stack in the crafting grid area — nothing can absorb it.
        for i in 9..45 {
            c.set_slot_contents(i, Some(stack(2, 64)));
        }
        c.set_slot_contents(1, Some(stack(1, 1)));
        let ret = c.slot_click(1, 0, 1, false);
        assert_eq!(ret, None);
        assert_eq!(c.slot_stack(1), Some(&stack(1, 1)), "unmoved");
    }

    #[test]
    fn unknown_items_remain_authoritative_but_plain_click_simulation_is_inert() {
        let unknown = stack(32_767, 5);

        let mut clicked = tiny_container(3);
        clicked.set_slot_contents(0, Some(unknown.clone()));
        assert_eq!(
            clicked.slot_click(0, 0, 0, false),
            Some(unknown.clone()),
            "mode zero still echoes the pre-click authoritative slot"
        );
        assert_eq!(clicked.slot_stack(0), Some(&unknown));
        assert_eq!(clicked.cursor(), None);

        let mut carried = tiny_container(3);
        carried.set_cursor(Some(unknown.clone()));
        assert_eq!(carried.slot_click(0, 0, 0, false), None);
        assert_eq!(carried.slot_stack(0), None);
        assert_eq!(carried.cursor(), Some(&unknown));

        let mut registered = tiny_container(3);
        registered.set_slot_contents(0, Some(stack(1, 5)));
        assert_eq!(registered.slot_click(0, 0, 0, false), Some(stack(1, 5)));
        assert_eq!(
            registered.slot_stack(0),
            None,
            "registered stone still simulates"
        );
        assert_eq!(registered.cursor(), Some(&stack(1, 5)));
    }

    #[test]
    fn unknown_items_cannot_shift_drag_merge_or_enter_player_inventory() {
        let unknown = stack(32_767, 5);

        let mut shifted = layouts::player_container(0);
        shifted.set_slot_contents(9, Some(unknown.clone()));
        assert_eq!(shifted.slot_click(9, 0, 1, false), None);
        assert_eq!(shifted.slot_stack(9), Some(&unknown));
        assert!((36..45).all(|i| shifted.slot_stack(i).is_none()));

        let mut dragged = tiny_container(3);
        dragged.set_cursor(Some(unknown.clone()));
        assert_eq!(dragged.slot_click(-999, 0, 5, false), None);
        assert_eq!(
            dragged.drag_event, 0,
            "unknown drag start resets the drag machine"
        );
        assert_eq!(dragged.cursor(), Some(&unknown));

        let mut direct = tiny_container(3);
        let mut incoming = unknown.clone();
        assert!(!direct.merge_item_stack(&mut incoming, 0, 3, false));
        assert_eq!(incoming, unknown);
        assert!(direct.items.iter().all(Option::is_none));
        assert_eq!(
            direct.add_stack_to_player_inventory(unknown.clone()),
            unknown
        );
        assert!(direct.items.iter().all(Option::is_none));
    }

    #[test]
    fn unknown_items_cannot_participate_in_hotbar_or_collect_clicks() {
        let unknown = stack(32_767, 5);

        let mut clicked_unknown = layouts::player_container(0);
        clicked_unknown.set_slot_contents(9, Some(unknown.clone()));
        clicked_unknown.set_slot_contents(36, Some(stack(1, 3)));
        assert_eq!(clicked_unknown.slot_click(9, 0, 2, false), None);
        assert_eq!(clicked_unknown.slot_stack(9), Some(&unknown));
        assert_eq!(clicked_unknown.slot_stack(36), Some(&stack(1, 3)));

        let mut hotbar_unknown = layouts::player_container(0);
        hotbar_unknown.set_slot_contents(9, Some(stack(1, 3)));
        hotbar_unknown.set_slot_contents(36, Some(unknown.clone()));
        assert_eq!(hotbar_unknown.slot_click(9, 0, 2, false), None);
        assert_eq!(hotbar_unknown.slot_stack(9), Some(&stack(1, 3)));
        assert_eq!(hotbar_unknown.slot_stack(36), Some(&unknown));

        let mut collect = tiny_container(3);
        collect.set_cursor(Some(unknown.clone()));
        collect.set_slot_contents(0, Some(unknown.clone()));
        assert_eq!(collect.slot_click(0, 0, 6, false), None);
        assert_eq!(collect.cursor(), Some(&unknown));
        assert_eq!(collect.slot_stack(0), Some(&unknown));

        assert!(!Container::can_add_item_to_slot(None, &unknown, false));
        assert!(!Container::can_add_item_to_slot(
            Some(&unknown),
            &stack(1, 1),
            false
        ));
    }

    #[test]
    fn unknown_anvil_inputs_clear_fabricated_local_output_and_costs() {
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(stack(32_767, 1)));
        c.set_slot_contents(1, Some(stack(1, 1)));
        assert_eq!(c.slot_stack(0), Some(&stack(32_767, 1)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 1)));
        assert_eq!(c.slot_stack(2), None);
        assert_eq!(c.maximum_cost, Some(0));
        assert_eq!(c.anvil_material_cost, 0);
    }

    #[test]
    fn unknown_cross_slot_inputs_freeze_craft_and_merchant_simulation() {
        let unknown = stack(32_767, 1);

        let mut craft = layouts::player_container(0);
        craft.set_slot_contents(1, Some(stack(5, 1)));
        craft.set_slot_contents(3, Some(stack(5, 1)));
        assert_eq!(craft.slot_stack(0), Some(&stack(280, 4)));
        craft.set_slot_contents(3, Some(unknown.clone()));
        assert_eq!(
            craft.slot_stack(0),
            Some(&stack(280, 4)),
            "an unknown matrix cell must not retract the previous local result"
        );
        assert_eq!(craft.slot_click(0, 0, 0, false), Some(stack(280, 4)));
        assert_eq!(craft.slot_stack(0), Some(&stack(280, 4)));
        assert_eq!(craft.slot_stack(1), Some(&stack(5, 1)));
        assert_eq!(craft.slot_stack(3), Some(&unknown));
        assert_eq!(craft.cursor(), None);

        let mut slots: Vec<Slot> = (0..39).map(|id| Slot::new(id, 0, 0)).collect();
        slots[2] = slots[2].with_role(super::super::slot::SlotRole::CraftResult);
        let mut merchant = Container::new(4, ContainerKind::Merchant, slots);
        merchant.set_merchant_recipe(Some(MerchantRecipe {
            buy_a: stack(265, 2),
            buy_b: Some(stack(331, 1)),
            sell: stack(388, 1),
            uses: 0,
            max_uses: 2,
            disabled: false,
        }));
        merchant.set_slot_contents(0, Some(stack(265, 2)));
        merchant.set_slot_contents(1, Some(stack(331, 1)));
        assert_eq!(merchant.slot_stack(2), Some(&stack(388, 1)));
        merchant.set_slot_contents(1, Some(unknown.clone()));
        assert_eq!(merchant.slot_stack(2), Some(&stack(388, 1)));
        assert_eq!(merchant.slot_click(2, 0, 0, false), Some(stack(388, 1)));
        assert_eq!(merchant.slot_stack(0), Some(&stack(265, 2)));
        assert_eq!(merchant.slot_stack(1), Some(&unknown));
        assert_eq!(merchant.slot_stack(2), Some(&stack(388, 1)));
        assert_eq!(merchant.cursor(), None);
    }

    // ---------------------------------------------------------------- mode 2: hotbar swap

    #[test]
    fn number_key_swap_exchanges_a_slot_with_a_hotbar_slot() {
        let mut c = layouts::player_container(0);
        c.set_slot_contents(9, Some(stack(1, 5))); // first main-inventory slot
        c.set_slot_contents(36, Some(stack(2, 3))); // hotbar key 0
        c.slot_click(9, 0, 2, false);
        assert_eq!(c.slot_stack(9), Some(&stack(2, 3)));
        assert_eq!(c.slot_stack(36), Some(&stack(1, 5)));
    }

    /// `InventoryPlayer.mainInventory` is hotbar-first even though `ContainerPlayer` exposes
    /// main rows first. This drives the live overflow path where a number-key displaces an item
    /// that cannot enter the clicked armour slot.
    #[test]
    fn number_key_overflow_scans_hotbar_then_main_for_partial_and_empty_slots() {
        let mut c = layouts::player_container(0);
        c.set_slot_contents(5, Some(named("diamond_helmet", 1, 0)));
        c.set_slot_contents(36, Some(named("stone", 10, 0))); // key 0, displaced
        c.set_slot_contents(37, Some(named("stone", 60, 0))); // first partial in mainInventory
        c.set_slot_contents(9, Some(named("stone", 60, 0))); // later partial despite lower slot id

        c.slot_click(5, 0, 2, false);

        assert_eq!(c.slot_stack(36), Some(&named("diamond_helmet", 1, 0)));
        assert_eq!(c.slot_stack(37), Some(&named("stone", 64, 0)));
        assert_eq!(c.slot_stack(9), Some(&named("stone", 64, 0)));
        assert_eq!(
            c.slot_stack(38),
            Some(&named("stone", 2, 0)),
            "the first empty hotbar slot receives the remainder"
        );
        assert_eq!(c.slot_stack(5), None);
        assert_eq!(c.slot_stack(10), None, "main-row empty slots come later");
    }

    /// Java's damaged-item fast path copies the entire stack to one empty slot before zeroing
    /// the source. Count two is malformed but network-reachable and distinguishes that assignment
    /// from the ordinary per-slot stack-limit loop.
    #[test]
    fn damaged_stack_fast_path_keeps_the_whole_stack_in_one_hotbar_slot() {
        let mut c = layouts::player_container(0);
        let damaged = named("diamond_sword", 2, 3);
        let leftover = c.add_stack_to_player_inventory(damaged.clone());
        assert_eq!(leftover.count, 0);
        assert_eq!(c.slot_stack(36), Some(&damaged));
        assert_eq!(c.slot_stack(37), None);
        assert_eq!(c.slot_stack(9), None);
    }

    // ---------------------------------------------------------------- mode 3: creative pick

    #[test]
    fn creative_middle_click_duplicates_a_full_stack_without_touching_the_source() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 3)));
        c.slot_click(0, 0, 3, true);
        assert_eq!(c.cursor(), Some(&stack(1, 64)));
        assert_eq!(c.slot_stack(0), Some(&stack(1, 3)), "source is untouched");
    }

    #[test]
    fn creative_middle_click_does_nothing_outside_creative_mode() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 3)));
        c.slot_click(0, 0, 3, false);
        assert_eq!(c.cursor(), None);
    }

    // ---------------------------------------------------------------- mode 4: drop key

    #[test]
    fn drop_key_removes_one_item() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 5)));
        c.slot_click(0, 0, 4, false);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 4)));
    }

    #[test]
    fn ctrl_drop_key_removes_the_whole_stack() {
        let mut c = tiny_container(3);
        c.set_slot_contents(0, Some(stack(1, 5)));
        c.slot_click(0, 1, 4, false);
        assert_eq!(c.slot_stack(0), None);
    }

    // ---------------------------------------------------------------- mode 6: double click

    #[test]
    fn double_click_collects_matching_items_from_every_slot_up_to_the_max_stack() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        c.set_slot_contents(1, Some(stack(1, 40)));
        c.set_slot_contents(2, Some(stack(2, 99))); // different item: untouched
        c.set_slot_contents(3, Some(stack(1, 30)));
        c.slot_click(0, 0, 6, false);
        assert_eq!(
            c.cursor(),
            Some(&stack(1, 64)),
            "10 + 40 + 14 (partial) = 64, capped"
        );
        assert_eq!(
            c.slot_stack(2),
            Some(&stack(2, 99)),
            "different item left alone"
        );
    }

    // ---------------------------------------------------------------- mode 5: drag (H10)

    #[test]
    fn even_split_drag_across_three_slots_conserves_every_item() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 30)));
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 0), 5, false);
        };
        event(&mut c, 0, -999); // start, mode 0 = even split
        event(&mut c, 1, 0);
        event(&mut c, 1, 1);
        event(&mut c, 1, 2);
        event(&mut c, 2, -999); // end
        assert_eq!(c.slot_stack(0), Some(&stack(1, 10)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 10)));
        assert_eq!(c.slot_stack(2), Some(&stack(1, 10)));
        assert_eq!(
            c.cursor(),
            None,
            "30 split evenly three ways leaves nothing over"
        );
        assert_eq!(total_items(&c, 1), 30, "H10: no items created or destroyed");
    }

    #[test]
    fn even_split_drag_with_a_remainder_keeps_the_remainder_on_the_cursor() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 0), 5, false);
        };
        event(&mut c, 0, -999);
        event(&mut c, 1, 0);
        event(&mut c, 1, 1);
        event(&mut c, 1, 2);
        event(&mut c, 2, -999);
        // floor(10/3) = 3 per slot, 9 placed, 1 remains on the cursor.
        assert_eq!(c.slot_stack(0), Some(&stack(1, 3)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 3)));
        assert_eq!(c.slot_stack(2), Some(&stack(1, 3)));
        assert_eq!(c.cursor(), Some(&stack(1, 1)));
        assert_eq!(total_items(&c, 1), 10, "H10: no items created or destroyed");
    }

    #[test]
    fn one_per_slot_drag_mode_places_exactly_one_item_per_slot() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 1), 5, false);
        };
        event(&mut c, 0, -999);
        event(&mut c, 1, 0);
        event(&mut c, 1, 1);
        event(&mut c, 1, 2);
        event(&mut c, 1, 3);
        event(&mut c, 2, -999);
        for i in 0..4 {
            assert_eq!(c.slot_stack(i), Some(&stack(1, 1)), "slot {i}");
        }
        assert_eq!(c.cursor(), Some(&stack(1, 6)));
        assert_eq!(total_items(&c, 1), 10);
    }

    #[test]
    fn creative_fill_drag_mode_is_rejected_outside_creative_mode() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        // Starting a mode-2 drag while not creative must not even enter the drag state, so the
        // "add slot" event that follows is silently ignored (dragEvent stays 0, matching the
        // `else if (this.dragEvent != 0) resetDrag()` / no-op fallthrough in Java).
        c.slot_click(-999, encode_drag(0, 2), 5, false);
        c.slot_click(0, encode_drag(1, 2), 5, false);
        c.slot_click(-999, encode_drag(2, 2), 5, false);
        assert_eq!(
            c.slot_stack(0),
            None,
            "an invalid drag mode must place nothing"
        );
        assert_eq!(
            c.cursor(),
            Some(&stack(1, 10)),
            "and must not touch the cursor either"
        );
    }

    #[test]
    fn creative_fill_drag_mode_fills_every_slot_to_its_max_stack() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 2), 5, true);
        };
        event(&mut c, 0, -999);
        event(&mut c, 1, 0);
        event(&mut c, 1, 1);
        event(&mut c, 2, -999);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 64)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 64)));
        // `j` (the running cursor remainder) is computed identically for every drag mode, so a
        // 10-item cursor filling two slots to 64 each still drives `j` negative and clears it.
        assert_eq!(c.cursor(), None);
    }

    #[test]
    fn a_mismatched_phase_sequence_resets_the_drag_without_moving_anything() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        // start(mode 0), add(slot 0), then *another* start instead of add/end: Java's
        // `(i != 1 || dragEvent != 2) && i != dragEvent` catches this and resets.
        c.slot_click(-999, encode_drag(0, 0), 5, false);
        c.slot_click(0, encode_drag(1, 0), 5, false);
        c.slot_click(-999, encode_drag(0, 0), 5, false); // stray restart
                                                         // Sending "end" now should do nothing useful: the drag was reset, so `dragSlots` is
                                                         // empty and the end-phase branch's `if (!dragSlots.isEmpty())` guard skips entirely.
        c.slot_click(-999, encode_drag(2, 0), 5, false);
        assert_eq!(
            c.slot_stack(0),
            None,
            "the stray restart must have wiped the pending add"
        );
        assert_eq!(
            c.cursor(),
            Some(&stack(1, 10)),
            "and the cursor must be untouched"
        );
    }

    #[test]
    fn releasing_the_cursor_mid_drag_resets_instead_of_placing_a_null_stack() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        c.slot_click(-999, encode_drag(0, 0), 5, false);
        c.slot_click(0, encode_drag(1, 0), 5, false);
        c.set_cursor(None); // e.g. a same-tick server correction emptied it
        c.slot_click(1, encode_drag(1, 0), 5, false); // java: `getItemStack() == null` -> resetDrag
        c.slot_click(-999, encode_drag(2, 0), 5, false);
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
    }

    #[test]
    fn drag_add_phase_skips_a_slot_the_item_cannot_go_into() {
        let mut c = layouts::player_container(0);
        c.set_cursor(Some(stack(1, 10))); // not armour
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 1), 5, false);
        };
        event(&mut c, 0, -999);
        event(&mut c, 1, 5); // armour slot 0 (helmet-only) — stone is not valid here
        event(&mut c, 1, 9); // a real main-inventory slot
        event(&mut c, 2, -999);
        assert_eq!(
            c.slot_stack(5),
            None,
            "the armour slot must refuse a non-armour item"
        );
        assert_eq!(c.slot_stack(9), Some(&stack(1, 1)));
        assert_eq!(total_items(&c, 1), 10);
    }

    #[test]
    fn drag_tops_up_a_slot_that_already_has_a_partial_stack() {
        let mut c = tiny_container(5);
        c.set_slot_contents(0, Some(stack(1, 60)));
        c.set_cursor(Some(stack(1, 8)));
        let event = |c: &mut Container, phase: u8, slot: i32| {
            c.slot_click(slot, encode_drag(phase, 0), 5, false);
        };
        event(&mut c, 0, -999);
        event(&mut c, 1, 0);
        event(&mut c, 1, 1);
        event(&mut c, 2, -999);
        // 8 split evenly over 2 slots = 4 each; slot 0 already has 60, so it clamps to 64
        // (only +4 actually fit) while slot 1 (empty) takes its full +4.
        assert_eq!(c.slot_stack(0), Some(&stack(1, 64)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 4)));
        assert_eq!(
            total_items(&c, 1),
            68,
            "60 pre-existing + 8 from the cursor"
        );
    }

    #[test]
    fn an_empty_add_phase_drag_places_nothing_and_returns_the_full_cursor() {
        let mut c = tiny_container(5);
        c.set_cursor(Some(stack(1, 10)));
        c.slot_click(-999, encode_drag(0, 0), 5, false);
        // no add-slot events
        c.slot_click(-999, encode_drag(2, 0), 5, false);
        assert_eq!(c.cursor(), Some(&stack(1, 10)));
        assert!(c
            .slots()
            .iter()
            .all(|s| c.slot_stack(s.id as usize).is_none()));
    }

    // ------------------------------------------------------- canTakeStack (the anvil output)
    //
    // java: `ContainerRepair.java:71-74` is the only `canTakeStack` override in 1.8.9:
    //   (creative || experienceLevel >= maximumCost) && maximumCost > 0 && getHasStack()
    // Gates at `Container.java:264` (mode 1), `:321` (mode 0), `:394` (mode 2), `:450` (mode 4)
    // and `:462`/`:473` (mode 6).

    /// An anvil whose output slot (index 2) holds `stack` and whose repair costs `cost` levels.
    fn anvil_with_output(cost: i32, level: i32) -> Container {
        let mut c = layouts::anvil_container(3);
        c.set_slot_contents(2, Some(stack(1, 1)));
        c.maximum_cost = Some(cost);
        c.set_player_experience_level(level);
        c
    }

    /// A real client-side `ContainerRepair.updateRepairOutput` result, rather than a hand-set
    /// output/cost pair. Two damaged iron pickaxes cost one level and leave one iron ingot in the
    /// material slot after pickup; exercising this shape catches a gate that is wired only for
    /// synthetic output snapshots.
    fn anvil_with_predicted_material_repair(level: i32) -> (Container, ItemStack) {
        let mut c = layouts::anvil_container(3);
        c.set_slot_contents(0, Some(damaged_stack(257, 200)));
        c.set_slot_contents(1, Some(stack(265, 1)));
        c.set_player_experience_level(level);
        let output = c
            .slot_stack(2)
            .cloned()
            .expect("the damaged pickaxe + iron repair must produce an output");
        assert_eq!(c.maximum_cost, Some(1));
        (c, output)
    }

    #[test]
    fn every_local_anvil_input_mutation_recomputes_output_and_cost_immediately() {
        fn base_anvil() -> Container {
            let mut c = layouts::anvil_container(0);
            c.set_slot_contents(0, Some(damaged_stack(257, 200)));
            assert_eq!(c.slot_stack(2), None);
            assert_eq!(c.maximum_cost, Some(0));
            c
        }

        let assert_repair = |c: &Container| {
            assert_eq!(c.maximum_cost, Some(1));
            assert_eq!(
                c.slot_stack(2).map(|stack| (stack.item_id, stack.damage)),
                Some((257, 138))
            );
        };
        let assert_cleared = |c: &Container| {
            assert_eq!(c.maximum_cost, Some(0));
            assert_eq!(c.slot_stack(2), None);
        };

        // Mode 0: place and remove the material through the cursor.
        let mut pickup = base_anvil();
        pickup.set_cursor(Some(stack(265, 1)));
        pickup.slot_click(1, 0, 0, false);
        assert_repair(&pickup);
        pickup.slot_click(1, 0, 0, false);
        assert_cleared(&pickup);

        // Mode 2: swap the material into and back out of input slot 1.
        let mut hotbar = base_anvil();
        let hotbar_zero = hotbar.slots().len() - 9;
        hotbar.set_slot_contents(hotbar_zero, Some(stack(265, 1)));
        hotbar.slot_click(1, 0, 2, false);
        assert_repair(&hotbar);
        hotbar.slot_click(1, 0, 2, false);
        assert_cleared(&hotbar);

        // Mode 5: completing a normal left-drag over input slot 1.
        let mut drag = base_anvil();
        drag.set_cursor(Some(stack(265, 1)));
        drag.slot_click(-999, encode_drag(0, 0), 5, false);
        drag.slot_click(1, encode_drag(1, 0), 5, false);
        drag.slot_click(-999, encode_drag(2, 0), 5, false);
        assert_repair(&drag);

        // Public/direct transfer and mode 1 share this path. A successful move must recompute;
        // a later full-destination no-op must preserve the already-correct prediction.
        let mut shift = base_anvil();
        shift.set_slot_contents(3, Some(stack(265, 1)));
        assert_eq!(shift.transfer_stack_in_slot(3), Some(stack(265, 1)));
        assert_repair(&shift);
        let output_before = shift.slot_stack(2).cloned();
        shift.set_slot_contents(4, Some(stack(1, 1)));
        assert_eq!(shift.transfer_stack_in_slot(4), None);
        assert_eq!(shift.slot_stack(2), output_before.as_ref());
        assert_eq!(shift.maximum_cost, Some(1));
    }

    /// The wire half. java assigns `itemstack` *inside* the mode-1 gate, so a refused
    /// shift-click reports an **empty** `clicked_item`; the server's own `slotClick` is gated
    /// too, so the bytes match and the transaction is accepted.
    ///
    /// Negative control: with `level = 5` (below) the same call returns `Some`.
    #[test]
    fn shift_clicking_an_unaffordable_anvil_output_moves_nothing_and_reports_an_empty_item() {
        let mut c = anvil_with_output(5, 2);
        let packet = c.click(2, 0, 1, false);
        assert_eq!(packet.clicked_item, None, "java: assigned inside the gate");
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)), "nothing moved");
    }

    #[test]
    fn shift_clicking_an_affordable_anvil_output_still_works() {
        let mut c = anvil_with_output(5, 5);
        let packet = c.click(2, 0, 1, false);
        assert_eq!(packet.clicked_item, Some(stack(1, 1)));
        assert_eq!(
            c.slot_stack(2),
            None,
            "transferred into the player inventory"
        );
    }

    /// The ghost-item half, and the more dangerous one: mode 0's `clicked_item` is snapshotted
    /// *before* the gate, so the packet is byte-identical to vanilla's, the server **accepts**
    /// it, and `isChangingQuantityOnly` means no `SetSlot`/`WindowItems` ever follows. An
    /// ungated client is left holding an item the server never gave it, permanently.
    #[test]
    fn left_clicking_an_unaffordable_anvil_output_leaves_no_ghost_on_the_cursor() {
        let mut c = anvil_with_output(30, 4);
        let packet = c.click(2, 0, 0, false);
        assert_eq!(
            packet.clicked_item,
            Some(stack(1, 1)),
            "java snapshots the slot before the gate, so the wire bytes are unchanged"
        );
        assert_eq!(c.cursor(), None, "the pickup itself must not happen");
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)));
    }

    /// The mode-0 gate covers the occupied-slot branch even when the player is already carrying
    /// a different item. Refusing the click must not accidentally swap/merge the cursor, while
    /// the packet still carries the pre-gate output snapshot.
    #[test]
    fn an_unaffordable_anvil_output_does_not_swap_with_an_existing_cursor() {
        let mut c = anvil_with_output(5, 0);
        c.set_cursor(Some(stack(2, 3)));
        let packet = c.click(2, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(stack(1, 1)));
        assert_eq!(c.cursor(), Some(&stack(2, 3)));
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)));
    }

    /// `maximumCost > 0` is **not** exempted for creative: a zero-cost output cannot be taken
    /// by anybody. This test installs an explicit known zero to distinguish it from the unknown
    /// `None` state used before local repair inputs have been evaluated.
    #[test]
    fn a_zero_cost_anvil_output_cannot_be_taken_even_in_creative() {
        let mut c = anvil_with_output(0, 100);
        assert_eq!(
            c.maximum_cost,
            Some(0),
            "a *known* zero, not an unknown cost"
        );
        c.click(2, 0, 0, true);
        assert_eq!(c.cursor(), None);
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)));
    }

    #[test]
    fn creative_bypasses_the_experience_level_conjunct_but_not_the_others() {
        let mut c = anvil_with_output(30, 0);
        c.click(2, 0, 0, true);
        assert_eq!(c.cursor(), Some(&stack(1, 1)));
    }

    /// Modes 2 and 4 send the same bytes whether or not the gate passes, so they desync
    /// silently in exactly the same way as mode 0.
    #[test]
    fn the_drop_key_and_the_hotbar_swap_also_respect_can_take_stack() {
        let mut c = anvil_with_output(30, 1);
        c.click(2, 0, 4, false); // mode 4: drop one
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)), "mode 4 gated");
        c.click(2, 0, 2, false); // mode 2: swap with hotbar slot 0
        assert_eq!(c.slot_stack(2), Some(&stack(1, 1)), "mode 2 gated");
        assert_eq!(c.slot_stack(c.slots().len() - 9), None);
    }

    /// The five `Container.slotClick` gates must run against the locally predicted anvil result,
    /// not merely an output copied into slot 2. Modes 0/2/4 intentionally keep their vanilla
    /// packet shape while refusing local mutation; mode 1 additionally has the inside-gate
    /// `clicked_item` assignment and therefore emits an empty stack on refusal.
    #[test]
    fn every_pickup_mode_refuses_an_unaffordable_predicted_anvil_output_without_mutation() {
        for mode in [0, 1, 2, 4] {
            let (mut c, expected) = anvil_with_predicted_material_repair(0);
            let packet = c.click(2, 0, mode, false);

            assert_eq!(
                packet.clicked_item,
                if mode == 0 {
                    Some(expected.clone())
                } else {
                    None
                },
                "mode {mode}: Java's packet snapshot"
            );
            assert_eq!(c.cursor(), None, "mode {mode}: cursor unchanged");
            assert_eq!(
                c.slot_stack(2),
                Some(&expected),
                "mode {mode}: output stays"
            );
            assert_eq!(c.slot_stack(0), Some(&damaged_stack(257, 200)));
            assert_eq!(c.slot_stack(1), Some(&stack(265, 1)));
            assert_eq!(c.maximum_cost, Some(1));
            assert_eq!(c.player_experience_level, Some(0));
            assert_eq!(c.slot_stack(c.slots().len() - 9), None);
        }
    }

    /// Once the XP conjunct passes, all four pickup modes consume the same predicted anvil
    /// transaction: one level, the first input, and exactly one material item. The destination
    /// differs by mode but the output is never left duplicated in slot 2.
    #[test]
    fn every_pickup_mode_consumes_an_affordable_predicted_anvil_output_once() {
        for mode in [0, 1, 2, 4] {
            let (mut c, expected) = anvil_with_predicted_material_repair(1);
            let packet = c.click(2, 0, mode, false);

            assert_eq!(
                packet.clicked_item,
                if mode <= 1 {
                    Some(expected.clone())
                } else {
                    None
                }
            );
            assert_eq!(c.slot_stack(0), None, "mode {mode}: repaired item consumed");
            assert_eq!(c.slot_stack(1), None, "mode {mode}: material cost");
            assert_eq!(c.slot_stack(2), None, "mode {mode}: output consumed");
            assert_eq!(c.maximum_cost, Some(0));
            assert_eq!(c.player_experience_level, Some(0));

            match mode {
                0 => assert_eq!(c.cursor(), Some(&expected), "mode 0: cursor destination"),
                1 => assert_eq!(
                    c.slot_stack(38),
                    Some(&expected),
                    "mode 1: reverse player merge destination"
                ),
                2 => assert_eq!(
                    c.slot_stack(c.slots().len() - 9),
                    Some(&expected),
                    "mode 2: hotbar destination"
                ),
                4 => assert_eq!(c.cursor(), None, "mode 4: dropped output is not carried"),
                _ => unreachable!(),
            }
        }
    }

    /// Mode 6's clicked slot condition and sweep both consult `canTakeStack`: an unaffordable
    /// output must not be collected into an already-held matching cursor stack.
    #[test]
    fn double_click_collection_skips_an_unaffordable_anvil_output() {
        let mut c = anvil_with_output(30, 0);
        c.set_slot_contents(2, Some(stack(1, 4)));
        c.set_cursor(Some(stack(1, 1)));
        c.click(2, 0, 6, false);
        assert_eq!(c.cursor(), Some(&stack(1, 1)), "output was not swept");
        assert_eq!(c.slot_stack(2), Some(&stack(1, 4)));
    }

    /// Until a host supplies the player's XP level the level conjunct is treated as satisfied,
    /// so a survival player is not locked out of every anvil. The other two conjuncts still
    /// apply -- see `a_zero_cost_anvil_output_cannot_be_taken_even_in_creative`.
    #[test]
    fn an_unknown_experience_level_does_not_block_an_anvil_take() {
        let mut c = layouts::anvil_container(3);
        c.set_slot_contents(2, Some(stack(1, 1)));
        c.maximum_cost = Some(30);
        c.click(2, 0, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 1)));
    }

    /// With no local anvil inputs, `maximumCost` remains unknown and a vanilla server never sends a second
    /// `S31PacketWindowProperty` for an anvil (`ContainerRepair.java:389-393` -- the only
    /// `sendProgressBarUpdate` call, in `onCraftGuiOpened`, with `maximumCost` still `0`).
    ///
    /// So the gate must stay inert here. Gating on the wire's `0` instead would refuse **every**
    /// anvil take, and because mode 0's `clicked_item` is snapshotted before the gate the server
    /// would *accept* the click, take the item and send nothing back -- an inverted ghost, worse
    /// than the bug. This test is the guard against "fixing" `F-P1-inventory-06-1` that way.
    #[test]
    fn an_unknown_repair_cost_leaves_the_anvil_gate_inert() {
        let mut c = layouts::anvil_container(3);
        c.set_slot_contents(2, Some(stack(1, 1)));
        assert_eq!(c.maximum_cost, None, "nothing may write this from S31");
        assert!(c.can_take_stack(2, false));
        c.click(2, 0, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 1)), "the take still happens");
    }

    /// Every other slot in the game keeps `Slot.java:150`'s `return true`.
    #[test]
    fn can_take_stack_is_true_for_every_role_but_the_anvil_output() {
        // A *known*, unaffordable cost, so the anvil output is the one slot that says no.
        let c = anvil_with_output(5, 0);
        for slot in c.slots() {
            let expected = slot.role != super::super::slot::SlotRole::AnvilOutput;
            assert_eq!(
                c.can_take_stack(usize::from(slot.id), false),
                expected,
                "slot {} role {:?}",
                slot.id,
                slot.role
            );
        }
    }

    /// The marker is an implementation detail of the anvil layout, not a new global slot
    /// subclass. A normal container that happens to use the marker in a fixture must retain
    /// `Slot.java:150`'s unconditional `true` behavior.
    #[test]
    fn anvil_output_marker_does_not_gate_a_non_anvil_container() {
        let mut slots: Vec<Slot> = (0..39u16).map(|id| Slot::new(id, 0, 0)).collect();
        slots[2] = slots[2].with_role(super::super::slot::SlotRole::AnvilOutput);
        let mut c = Container::new(9, ContainerKind::TopThenPlayer { top: 3 }, slots);
        c.set_slot_contents(2, Some(stack(1, 1)));
        c.maximum_cost = Some(5);
        c.set_player_experience_level(0);
        assert!(c.can_take_stack(2, false));
        c.click(2, 0, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 1)));
        assert_eq!(c.slot_stack(2), None);
    }

    // ----------------------------------------- InventoryBasic's 64 clamp, and where it is not

    /// java: `InventoryBasic.setInventorySlotContents` (`:160-170`) clamps an incoming stack to
    /// `getInventoryStackLimit()` — a flat `64` (`:216-219`) — and every slotted window the
    /// client opens is backed by `ContainerLocalMenu extends InventoryBasic`.
    ///
    /// The negative control is the same window's player half, which is `InventoryPlayer` and
    /// does **not** clamp (`InventoryPlayer.java:539-550`). Without it a blanket clamp would
    /// pass this test while diverging from vanilla in the other half of the very same window.
    #[test]
    fn an_oversized_server_stack_is_capped_in_the_container_half_and_not_in_the_player_half() {
        let mut c = layouts::chest_container(1, 3);
        c.set_slot_contents(0, Some(stack(1, 100)));
        assert_eq!(c.slot_stack(0).unwrap().count, 64, "InventoryBasic clamps");
        c.set_slot_contents(30, Some(stack(1, 100)));
        assert_eq!(
            c.slot_stack(30).unwrap().count,
            100,
            "InventoryPlayer does not"
        );
    }

    /// The three windows whose top half is *not* `InventoryBasic`.
    #[test]
    fn craft_grids_craft_results_and_the_anvil_output_are_never_clamped() {
        let mut w = layouts::crafting_table_container(1);
        // 999 is intentionally unregistered: this test isolates InventoryCraftResult and
        // InventoryCrafting's no-clamp behaviour without letting the local recipe predictor
        // replace the authoritative result with a legitimate one-item recipe.
        w.set_slot_contents(0, Some(stack(999, 100))); // InventoryCraftResult
        w.set_slot_contents(4, Some(stack(999, 100))); // InventoryCrafting
        assert_eq!(w.slot_stack(0).unwrap().count, 100);
        assert_eq!(w.slot_stack(4).unwrap().count, 100);

        let mut p = layouts::player_container(0);
        p.set_slot_contents(0, Some(stack(999, 100)));
        p.set_slot_contents(2, Some(stack(999, 100)));
        assert_eq!(p.slot_stack(0).unwrap().count, 100);
        assert_eq!(p.slot_stack(2).unwrap().count, 100);

        // The anvil is the split case: `InventoryBasic("Repair", true, 2)` inputs, an
        // `InventoryCraftResult` output.
        let mut a = layouts::anvil_container(2);
        a.set_slot_contents(0, Some(stack(1, 100)));
        a.set_slot_contents(2, Some(stack(1, 100)));
        assert_eq!(a.slot_stack(0).unwrap().count, 64, "input clamps");
        assert_eq!(a.slot_stack(2).unwrap().count, 100, "output does not");
    }

    /// The other `InventoryBasic`-backed windows, so a future `ContainerKind` cannot quietly
    /// pick the wrong default.
    #[test]
    fn every_other_slotted_window_clamps_its_top_half() {
        for mut c in [
            layouts::furnace_container(1),
            layouts::brewing_stand_container(2),
            layouts::enchanting_table_container(3),
            layouts::beacon_container(4),
            layouts::hopper_container(5, 5),
        ] {
            c.set_slot_contents(0, Some(stack(1, 200)));
            assert_eq!(c.slot_stack(0).unwrap().count, 64, "{:?} slot 0", c.kind);
        }
    }

    // ------------------------------------------------------- ContainerFurnace routing order

    fn named(name: &str, count: i32, damage: i16) -> ItemStack {
        ItemStack {
            item_id: mc_data::item_by_name(name).expect("known 1.8.9 item").0 as i16,
            count,
            damage,
            nbt: None,
        }
    }

    /// java: `ContainerFurnace.transferStackInSlot` (`:114-129`) asks
    /// `getSmeltingResult(...) != null` **before** `isItemFuel(...)`.
    ///
    /// `log` is the case that separates the two orders: it smelts to charcoal
    /// (`FurnaceRecipes.java:44`) *and* is `Material.wood` fuel, so the order alone decides
    /// whether it lands in the input slot or the fuel slot. Testing only `iron_ore` would pass
    /// against a merely-added smelt branch placed *after* the fuel branch.
    #[test]
    fn shift_clicking_a_log_into_a_furnace_fills_the_input_not_the_fuel_slot() {
        let mut c = layouts::furnace_container(1);
        c.set_slot_contents(30, Some(named("log", 8, 0)));
        c.slot_click(30, 0, 1, false);
        assert_eq!(c.slot_stack(0), Some(&named("log", 8, 0)), "input");
        assert_eq!(c.slot_stack(1), None, "not the fuel slot");
        assert_eq!(c.slot_stack(30), None);
    }

    #[test]
    fn a_smeltable_non_fuel_reaches_the_furnace_input_at_all() {
        let mut c = layouts::furnace_container(1);
        c.set_slot_contents(30, Some(named("iron_ore", 3, 0)));
        c.slot_click(30, 0, 1, false);
        assert_eq!(c.slot_stack(0), Some(&named("iron_ore", 3, 0)));
    }

    /// The fuel branch must survive the reorder: coal does not smelt, so it still falls through
    /// to `mergeItemStack(1, 2)`.
    #[test]
    fn a_pure_fuel_still_goes_to_the_fuel_slot() {
        let mut c = layouts::furnace_container(1);
        c.set_slot_contents(30, Some(named("coal", 5, 0)));
        c.slot_click(30, 0, 1, false);
        assert_eq!(c.slot_stack(1), Some(&named("coal", 5, 0)));
        assert_eq!(c.slot_stack(0), None);
    }

    /// Neither smeltable nor fuel: the ordinary hotbar -> main routing, unchanged.
    #[test]
    fn an_item_that_is_neither_still_goes_to_the_main_inventory() {
        let mut c = layouts::furnace_container(1);
        c.set_slot_contents(30, Some(named("diamond", 1, 0)));
        c.slot_click(30, 0, 1, false);
        assert_eq!(c.slot_stack(3), Some(&named("diamond", 1, 0)));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
    }

    /// `compareItemStacks` only ignores metadata when the *key* is `32767`, which is what
    /// `addSmelting` uses. The three `addSmeltingRecipe` keys pin a metadata instead.
    #[test]
    fn the_three_metadata_pinned_smelting_recipes_do_not_accept_other_variants() {
        let mut c = layouts::furnace_container(1);
        for (name, damage, smelts) in [
            ("stonebrick", 0, true),
            ("stonebrick", 1, false), // mossy
            ("stonebrick", 2, false), // already cracked
            ("sponge", 1, true),      // wet
            ("sponge", 0, false),     // dry
            ("fish", 0, true),        // cod
            ("fish", 1, true),        // salmon
            ("fish", 2, false),       // clownfish: !canCook
            ("fish", 3, false),       // pufferfish: !canCook
            ("sand", 1, true),        // red sand: key metadata is 32767
        ] {
            c.set_slot_contents(0, None);
            c.set_slot_contents(3, None);
            c.set_slot_contents(30, Some(named(name, 1, damage)));
            c.slot_click(30, 0, 1, false);
            assert_eq!(
                c.slot_stack(0).is_some(),
                smelts,
                "{name} damage {damage} should {} smelt",
                if smelts { "" } else { "not" }
            );
        }
    }

    // --------------------------------------------- client-side crafting prediction

    fn damaged_stack(item_id: i16, damage: i16) -> ItemStack {
        ItemStack {
            item_id,
            count: 1,
            damage,
            nbt: None,
        }
    }

    fn fill_repair_recipe(c: &mut Container) {
        // Iron pickaxes have maxDamage=250. Two damage-200 tools leave 50 durability each,
        // plus floor(250*5/100)=12, so the result damage is 250-(50+50+12)=138.
        c.set_slot_contents(1, Some(damaged_stack(257, 200)));
        c.set_slot_contents(4, Some(damaged_stack(257, 200)));
    }

    #[test]
    fn repair_recipe_populates_the_player_result_and_click_echo_before_the_round_trip() {
        let mut c = layouts::player_container(0);
        fill_repair_recipe(&mut c);
        let output = damaged_stack(257, 138);
        assert_eq!(c.slot_stack(0), Some(&output));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(output.clone()));
        assert_eq!(c.cursor(), Some(&output));
        assert_eq!(c.slot_stack(1), None, "first ingredient consumed locally");
        assert_eq!(c.slot_stack(4), None, "second ingredient consumed locally");
        assert_eq!(
            c.slot_stack(0),
            None,
            "the exhausted recipe no longer matches"
        );
    }

    #[test]
    fn repair_prediction_works_in_the_workbench_shape_and_retracts_only_its_own_result() {
        let mut c = layouts::crafting_table_container(1);
        fill_repair_recipe(&mut c);
        assert_eq!(c.slot_stack(0), Some(&damaged_stack(257, 138)));
        c.set_slot_contents(4, None);
        assert_eq!(
            c.slot_stack(0),
            None,
            "a stale local prediction is retracted"
        );

        let authoritative = stack(1, 7);
        c.set_slot_contents(0, Some(authoritative.clone()));
        c.set_slot_contents(2, Some(stack(5, 1)));
        assert_eq!(
            c.slot_stack(0),
            Some(&authoritative),
            "an unsupported static recipe result received from the server is preserved"
        );
    }

    #[test]
    fn every_result_pickup_mode_consumes_a_predicted_repair_grid() {
        for (mode, button, destination) in [
            (0, 0, "cursor"),
            (1, 0, "inventory"),
            (2, 0, "hotbar"),
            (4, 0, "drop"),
        ] {
            let mut c = layouts::player_container(0);
            fill_repair_recipe(&mut c);
            let expected = damaged_stack(257, 138);
            let packet = c.click(0, button, mode, false);

            if mode == 0 {
                assert_eq!(c.cursor(), Some(&expected), "{destination}");
            } else if mode == 1 {
                assert!(
                    c.items[9..45]
                        .iter()
                        .any(|stack| stack.as_ref() == Some(&expected)),
                    "{destination}"
                );
            } else if mode == 2 {
                assert_eq!(c.slot_stack(36), Some(&expected), "{destination}");
            }
            if matches!(mode, 0 | 1) {
                assert_eq!(packet.clicked_item, Some(expected.clone()), "{destination}");
            }
            assert_eq!(c.slot_stack(1), None, "{destination}: first input");
            assert_eq!(c.slot_stack(4), None, "{destination}: second input");
            assert_eq!(c.slot_stack(0), None, "{destination}: result exhausted");
        }
    }

    #[test]
    fn only_filled_buckets_have_a_vanilla_container_item() {
        for filled in [326, 327, 335] {
            assert_eq!(
                Container::crafting_container_item(mc_data::ItemId(filled)),
                Some(mc_data::ItemId(325))
            );
        }
        assert_eq!(
            Container::crafting_container_item(mc_data::ItemId(325)),
            None,
            "the empty bucket must not recursively leave another bucket"
        );
        assert_eq!(
            Container::crafting_container_item(mc_data::ItemId(282)),
            None,
            "mushroom stew's bowl is not an Item.containerItem in 1.8.9"
        );
    }

    // -------------------------------------------------- InventoryCraftResult.decrStackSize
    //
    // The backing inventory of a craft result discards `count` and hands back the whole stack.
    // `Container.slotClick` never looks -- it computes `(stackSize + 1) / 2` for a right-click
    // and trusts the return -- so a right-click on a result slot takes *everything*.

    /// java: `InventoryCraftResult.decrStackSize` (`:57-69`) ignores `count`, so
    /// `Container.java:325-326`'s `j2 = (64 + 1) / 2 = 32` still yields all 64.
    ///
    /// The negative control is in the same test: the furnace output, one line down, is backed by
    /// `TileEntityFurnace` and splits normally. Without it this test would also pass against a
    /// blanket "every output slot takes whole", which would be a different bug.
    #[test]
    fn right_clicking_a_craft_result_takes_the_whole_stack_but_a_furnace_output_still_halves() {
        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(0, Some(stack(1, 64)));
        c.click(0, 1, 0, false); // right-click the result
        assert_eq!(c.cursor(), Some(&stack(1, 64)), "all 64, not 32");
        assert_eq!(c.slot_stack(0), None);

        let mut f = layouts::furnace_container(2);
        f.set_slot_contents(2, Some(stack(1, 64)));
        f.click(2, 1, 0, false);
        assert_eq!(f.cursor(), Some(&stack(1, 32)), "TileEntityFurnace splits");
        assert_eq!(f.slot_stack(2), Some(&stack(1, 32)));
    }

    /// The player's own 2x2 result (`ContainerPlayer:16,26`) is the same `InventoryCraftResult`.
    #[test]
    fn the_player_inventory_craft_result_takes_the_whole_stack_too() {
        let mut c = layouts::player_container(0);
        c.set_slot_contents(0, Some(stack(1, 9)));
        c.click(0, 1, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 9)));
        assert_eq!(c.slot_stack(0), None);
    }

    /// The anvil output is `InventoryCraftResult` as well (`ContainerRepair.java:51`), wired as
    /// a plain `Slot` at index `2` (`:65`) -- which works only because every
    /// `InventoryCraftResult` method ignores its index. The `canTakeStack` gate is a separate
    /// concern and is inert here (`an_unknown_repair_cost_leaves_the_anvil_gate_inert`).
    #[test]
    fn right_clicking_a_renamed_stack_off_the_anvil_takes_all_of_it() {
        let mut c = layouts::anvil_container(3);
        c.set_slot_contents(2, Some(stack(1, 64)));
        c.click(2, 1, 0, false);
        assert_eq!(c.cursor(), Some(&stack(1, 64)));
        assert_eq!(c.slot_stack(2), None);
    }

    /// java: mode 4 is `slot3.decrStackSize(clickedButton == 0 ? 1 : stackSize)`
    /// (`Container.java:452`), so `Q` on a craft result drops the entire output, not one item.
    #[test]
    fn q_on_a_craft_result_drops_the_entire_output() {
        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(0, Some(stack(1, 16)));
        c.click(0, 0, 4, false); // mode 4, button 0 = "drop one"
        assert_eq!(c.slot_stack(0), None, "vanilla drops all 16");
        assert_eq!(c.cursor(), None);
    }

    /// java: `Container.mergeItemStack`'s empty-slot branch only checks `itemstack1 == null`;
    /// it does not call `Slot.isItemValid`. Driven directly through a container shaped so that
    /// the distinction is observable, because the built-in shapes pre-filter restricted slots
    /// before invoking this helper.
    #[test]
    fn merging_fills_an_empty_slot_without_rechecking_slot_validity() {
        let mut slots: Vec<Slot> = (0..38u16).map(|i| Slot::new(i, 0, 0)).collect();
        slots[0] = slots[0].with_role(super::super::slot::SlotRole::Output);
        let mut c = Container::new(1, ContainerKind::TopThenPlayer { top: 2 }, slots);
        // Shift-click out of the player half: java merges into `0..2`, and the helper fills
        // slot 0 even though that role would reject a direct click.
        c.set_slot_contents(10, Some(stack(1, 4)));
        c.slot_click(10, 0, 1, false);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 4)));
        assert_eq!(c.slot_stack(1), None);
    }

    // ------------------------------------------------- the one InventoryPlayer, two arrays
    //
    // java has a single `InventoryPlayer` and every container's trailing 36 slots point at it.
    // This port copies instead, so these pin the mapping the copy relies on.

    #[test]
    fn every_container_shape_ends_with_the_same_thirty_six_player_slots() {
        for c in [
            layouts::player_container(0),
            layouts::crafting_table_container(1),
            layouts::chest_container(2, 6),
            layouts::furnace_container(3),
            layouts::brewing_stand_container(4),
            layouts::enchanting_table_container(5),
            layouts::anvil_container(6),
            layouts::beacon_container(7),
            layouts::hopper_container(8, 5),
        ] {
            let section = c.player_section();
            assert_eq!(section.len(), 36, "{:?}", c.kind);
            assert_eq!(section.end, c.slots().len(), "{:?}", c.kind);
            // The last nine are always the hotbar, at java's `8 + l * 18` spacing.
            let hotbar = &c.slots()[section.end - 9..];
            for (l, slot) in hotbar.iter().enumerate() {
                assert_eq!(slot.x, c.slots()[section.end - 9].x + l as i32 * 18);
            }
        }
    }

    #[test]
    fn shift_double_click_backing_groups_match_java_inventory_objects() {
        let player = layouts::player_container(0);
        // ContainerPlayer.java: slot 0 is InventoryCraftResult, slots 1..=4 are one
        // InventoryCrafting, and slots 5..=44 are the shared InventoryPlayer.
        assert!(!player.same_backing_region(0, 1));
        assert!(player.same_backing_region(1, 4));
        assert!(player.same_backing_region(5, 44));
        assert!(!player.same_backing_region(0, 5));
        assert!(!player.same_backing_region(1, 5));

        let workbench = layouts::crafting_table_container(1);
        assert!(!workbench.same_backing_region(0, 1));
        assert!(workbench.same_backing_region(1, 9));
        assert!(workbench.same_backing_region(10, 45));

        let anvil = layouts::anvil_container(2);
        assert!(anvil.same_backing_region(0, 1));
        assert!(!anvil.same_backing_region(0, 2));
        assert!(anvil.same_backing_region(3, 38));
    }

    /// A shift-click out of a chest lands in the player's own inventory, and in vanilla that is
    /// the *same object* the HUD hotbar and `getCurrentItem()` read. Here it has to be copied.
    #[test]
    fn syncing_the_player_section_carries_a_chest_shift_click_back_to_window_zero() {
        let mut chest = layouts::chest_container(3, 3);
        let mut player = layouts::player_container(0);
        chest.set_slot_contents(0, Some(stack(1, 16)));

        // Shift-click the chest slot. `ContainerChest.transferStackInSlot` merges into the
        // player half **reversed**, so for a 3-row chest (player half 27..63) it lands in the
        // last hotbar slot, 62 -- player-section index 35.
        chest.slot_click(0, 0, 1, false);
        assert_eq!(chest.slot_stack(62), Some(&stack(1, 16)));
        assert_eq!(
            player.slot_stack(44),
            None,
            "and window 0 has not heard about it"
        );

        chest.sync_player_section_into(&mut player);
        // Player-section index 35 is window 0's slot 44 -- hotbar slot 8, which is what
        // `refresh_hotbar` reads and therefore what `C08PacketPlayerBlockPlacement` carries.
        assert_eq!(player.slot_stack(44), Some(&stack(1, 16)));
        // And index 0 maps to window 0's slot 9, the first main-inventory slot.
        chest.set_slot_contents(27, Some(stack(2, 1)));
        chest.sync_player_section_into(&mut player);
        assert_eq!(player.slot_stack(9), Some(&stack(2, 1)));
    }

    /// It is a copy in both directions, and it never touches the container-specific slots.
    #[test]
    fn syncing_the_player_section_leaves_the_container_half_alone() {
        let mut player = layouts::player_container(0);
        let mut furnace = layouts::furnace_container(5);
        furnace.set_slot_contents(0, Some(stack(9, 1))); // smelting input
        player.set_slot_contents(36, Some(stack(1, 5))); // hotbar slot 0
        player.set_slot_contents(0, Some(stack(7, 1))); // the 2x2 craft result

        player.sync_player_section_into(&mut furnace);
        assert_eq!(
            furnace.slot_stack(furnace.slots().len() - 9),
            Some(&stack(1, 5))
        );
        assert_eq!(
            furnace.slot_stack(0),
            Some(&stack(9, 1)),
            "the furnace input is not part of InventoryPlayer"
        );
    }

    #[test]
    fn syncing_the_player_section_carries_item_stack_identity_with_the_value() {
        let mut chest = layouts::chest_container(3, 3);
        let mut player = layouts::player_container(0);
        let source_slot = chest.player_section().end - 9;
        let destination_slot = player.player_section().end - 9;
        let replacement = stack(1, 7);

        chest.set_slot_contents(source_slot, Some(replacement.clone()));
        let source_identity = chest
            .slot_stack_identity(source_slot)
            .expect("foreign player slot has an ItemStack identity");
        player.set_slot_contents(destination_slot, Some(stack(2, 3)));
        player.set_slot_contents(destination_slot, Some(stack(3, 3)));
        let old_destination_identity = player
            .slot_stack_identity(destination_slot)
            .expect("window zero starts with its own stack object");
        assert_ne!(source_identity, old_destination_identity);

        chest.sync_player_section_into(&mut player);

        assert_eq!(player.slot_stack(destination_slot), Some(&replacement));
        assert_eq!(
            player.slot_stack_identity(destination_slot),
            Some(source_identity),
            "the shared InventoryPlayer slot must expose the foreign window's ItemStack object"
        );
        assert_ne!(
            player.slot_stack_identity(destination_slot),
            Some(old_destination_identity),
            "a selected-hotbar use must not retain the replaced object"
        );
    }

    // -------------------------------------------------------- onContainerClosed / client close

    /// The base hook is client-visible: it turns the carried `InventoryPlayer.itemStack` into
    /// one local item drop, then clears it. Calling it twice deliberately cannot duplicate that
    /// visual drop -- `ContainerMerchant` reaches `super.onContainerClosed` twice in 1.8.9.
    #[test]
    fn closing_returns_the_cursor_once_and_clears_an_in_progress_drag() {
        let mut c = tiny_container(3);
        c.set_cursor(Some(stack(1, 7)));
        c.slot_click(-999, 0, 5, false); // drag start: event 0 -> state 1
        c.slot_click(0, 1, 5, false); // add slot 0 to that drag
        assert_eq!(c.drag_event, 1);
        assert_eq!(c.drag_slots.len(), 1);
        assert!(c.drag_slots.contains(&0));

        assert_eq!(c.on_container_closed(), Some(stack(1, 7)));
        assert_eq!(c.cursor(), None);
        assert_eq!(c.drag_event, 0);
        assert_eq!(c.drag_mode, -1);
        assert!(c.drag_slots.is_empty());
        assert_eq!(
            c.on_container_closed(),
            None,
            "the second merchant super-call cannot spawn a second cursor drop"
        );
    }

    /// Only the base cursor drop runs on a remote client. The subclass loops which remove a
    /// player/workbench craft matrix or anvil/enchant inputs all sit behind `!worldObj.isRemote`;
    /// copying those loops here would delete live server-synchronised contents before a reopen.
    #[test]
    fn client_close_preserves_container_specific_slots_for_player_crafting_anvil_and_enchanting() {
        let mut player = layouts::player_container(0);
        player.set_slot_contents(0, Some(stack(10, 1)));
        player.set_slot_contents(1, Some(stack(11, 2)));
        player.set_cursor(Some(stack(12, 3)));
        assert_eq!(player.on_container_closed(), Some(stack(12, 3)));
        assert_eq!(player.slot_stack(0), Some(&stack(10, 1)));
        assert_eq!(player.slot_stack(1), Some(&stack(11, 2)));

        let mut workbench = layouts::crafting_table_container(4);
        workbench.set_slot_contents(0, Some(stack(20, 1)));
        workbench.set_slot_contents(9, Some(stack(21, 2)));
        let _ = workbench.on_container_closed();
        assert_eq!(workbench.slot_stack(0), Some(&stack(20, 1)));
        assert_eq!(workbench.slot_stack(9), Some(&stack(21, 2)));

        let mut anvil = layouts::anvil_container(5);
        anvil.set_slot_contents(0, Some(stack(30, 1)));
        anvil.set_slot_contents(1, Some(stack(31, 1)));
        let _ = anvil.on_container_closed();
        assert_eq!(anvil.slot_stack(0), Some(&stack(30, 1)));
        assert_eq!(anvil.slot_stack(1), Some(&stack(31, 1)));

        let mut enchant = layouts::enchanting_table_container(6);
        enchant.set_slot_contents(0, Some(stack(40, 1)));
        enchant.set_slot_contents(1, Some(stack(351, 3)));
        let _ = enchant.on_container_closed();
        assert_eq!(enchant.slot_stack(0), Some(&stack(40, 1)));
        assert_eq!(enchant.slot_stack(1), Some(&stack(351, 3)));
    }

    /// `ContainerChest.onContainerClosed` delegates to `closeInventory`; the local client
    /// backing inventory is `ContainerLocalMenu`/`InventoryBasic`, whose callback has no item
    /// mutation. Its top storage must therefore survive the generic close hook as well.
    #[test]
    fn client_close_does_not_empty_a_chest_top_inventory() {
        let mut chest = layouts::chest_container(7, 3);
        chest.set_slot_contents(0, Some(stack(54, 64)));
        chest.set_cursor(Some(stack(1, 1)));
        assert_eq!(chest.on_container_closed(), Some(stack(1, 1)));
        assert_eq!(chest.slot_stack(0), Some(&stack(54, 64)));
    }

    // ---------------------------------------------------------------- transaction ids

    #[test]
    fn transaction_ids_start_at_one_and_increment() {
        let mut c = tiny_container(1);
        assert_eq!(c.next_transaction_id(), 1);
        assert_eq!(c.next_transaction_id(), 2);
        assert_eq!(c.next_transaction_id(), 3);
    }

    #[test]
    fn player_matrix_predicts_static_recipe_and_consumes_one_per_pickup() {
        let mut c = layouts::player_container(0);
        c.set_slot_contents(1, Some(stack(5, 3)));
        c.set_slot_contents(3, Some(stack(5, 3)));
        assert_eq!(c.slot_stack(0), Some(&stack(280, 4)));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(stack(280, 4)));
        assert_eq!(c.cursor(), Some(&stack(280, 4)));
        assert_eq!(c.slot_stack(1), Some(&stack(5, 2)));
        assert_eq!(c.slot_stack(3), Some(&stack(5, 2)));
        assert_eq!(c.slot_stack(0), Some(&stack(280, 4)));
    }

    #[test]
    fn craft_result_pickup_re_evaluates_a_server_result_when_provenance_is_absent() {
        let mut c = layouts::player_container(0);
        c.set_slot_contents(1, Some(stack(5, 1)));
        c.set_slot_contents(3, Some(stack(5, 1)));
        // A direct result update is authoritative and intentionally clears the local provenance.
        // The SlotCrafting hook still has to consume the matching matrix on pickup.
        c.set_slot_contents(0, Some(stack(280, 4)));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(stack(280, 4)));
        assert_eq!(c.cursor(), Some(&stack(280, 4)));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
        assert_eq!(c.slot_stack(3), None);
    }

    #[test]
    fn static_craft_pickup_consumes_each_ingredient_and_restores_milk_buckets() {
        let mut c = layouts::crafting_table_container(7);
        // `RecipeCake` is shaped as three milk buckets, sugar/egg/sugar, and three wheat.
        for (slot, item) in [
            (1, 335),
            (2, 335),
            (3, 335),
            (4, 353),
            (5, 344),
            (6, 353),
            (7, 296),
            (8, 296),
            (9, 296),
        ] {
            c.set_slot_contents(slot, Some(stack(item, 1)));
        }
        assert_eq!(c.slot_stack(0), Some(&stack(354, 1)));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(stack(354, 1)));
        assert_eq!(c.cursor(), Some(&stack(354, 1)));
        assert_eq!(c.slot_stack(0), None);
        for slot in 1..=3 {
            assert_eq!(
                c.slot_stack(slot),
                Some(&stack(325, 1)),
                "milk bucket {slot}"
            );
        }
        for slot in 4..=9 {
            assert_eq!(c.slot_stack(slot), None, "ingredient slot {slot} consumed");
        }
    }

    #[test]
    fn window_items_sync_recomputes_a_workbench_result_before_the_first_click() {
        let mut c = layouts::crafting_table_container(7);
        let mut items = vec![None; c.slots().len()];
        items[1] = Some(stack(5, 3));
        // Workbench cells are a 3x3 row-major grid: the vertical two-plank stick
        // recipe occupies the first column (slots 1 and 4), not the two corners
        // at slots 1 and 3.
        items[4] = Some(stack(5, 3));
        c.apply_window_items(&mc_protocol::play::s2c::WindowItems {
            window_id: 7,
            items,
        });

        let expected = stack(280, 4);
        assert_eq!(c.slot_stack(0), Some(&expected));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected));
        assert_eq!(c.cursor(), Some(&stack(280, 4)));
    }

    #[test]
    fn unsupported_matrix_preserves_authoritative_result_as_a_negative_control() {
        let mut c = layouts::crafting_table_container(0);
        c.set_slot_contents(0, Some(stack(999, 100)));
        c.set_slot_contents(1, Some(stack(999, 1)));
        assert_eq!(c.slot_stack(0), Some(&stack(999, 100)));
    }

    #[test]
    fn dynamic_rocket_without_charges_has_no_nbt_and_echoes_that_result() {
        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(stack(339, 1))); // paper
        c.set_slot_contents(2, Some(stack(289, 1))); // gunpowder

        let expected = stack(401, 1);
        assert_eq!(c.slot_stack(0), Some(&expected));
        assert_eq!(c.slot_stack(0).and_then(|stack| stack.nbt.as_ref()), None);

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected));
        assert_eq!(c.cursor(), Some(&stack(401, 1)));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
        assert_eq!(c.slot_stack(2), None);
    }

    #[test]
    fn dynamic_firework_star_omits_false_flicker_and_trail_tags() {
        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(damaged_stack(351, 4))); // lapis/blue dye
        c.set_slot_contents(2, Some(stack(289, 1))); // gunpowder

        let mut explosion = NbtCompound::new();
        explosion.insert("Colors", Nbt::IntArray(vec![2_437_522]));
        explosion.insert("Type", Nbt::Byte(0));
        let mut root = NbtCompound::new();
        root.insert("Explosion", Nbt::Compound(explosion));
        let expected = tagged_stack(402, 1, 0, root);
        assert_eq!(c.slot_stack(0), Some(&expected));
        assert!(!c
            .slot_stack(0)
            .and_then(|stack| stack.nbt.as_ref())
            .and_then(|root| root.get_compound("Explosion"))
            .is_some_and(|explosion| {
                explosion.contains_key("Flicker") || explosion.contains_key("Trail")
            }));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected));
    }

    #[test]
    fn dynamic_firework_nbt_reaches_the_click_echo_and_consumes_one_craft() {
        let mut explosion = NbtCompound::new();
        explosion.insert("Colors", Nbt::IntArray(vec![1_973_019]));
        explosion.insert("Type", Nbt::Byte(4));
        let charge = tagged_stack(
            402,
            1,
            0,
            [("Explosion".to_owned(), Nbt::Compound(explosion))]
                .into_iter()
                .collect(),
        );

        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(stack(339, 1))); // paper
        c.set_slot_contents(2, Some(stack(289, 1))); // two slots => Flight 2
        c.set_slot_contents(3, Some(stack(289, 1)));
        c.set_slot_contents(4, Some(charge));

        let mut fireworks = NbtCompound::new();
        fireworks.insert("Flight", Nbt::Byte(2));
        fireworks.insert(
            "Explosions",
            Nbt::list(vec![Nbt::Compound(
                [
                    ("Colors".to_owned(), Nbt::IntArray(vec![1_973_019])),
                    ("Type".to_owned(), Nbt::Byte(4)),
                ]
                .into_iter()
                .collect(),
            )]),
        );
        let expected = tagged_stack(
            401,
            1,
            0,
            [("Fireworks".to_owned(), Nbt::Compound(fireworks))]
                .into_iter()
                .collect(),
        );
        assert_eq!(c.slot_stack(0), Some(&expected));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected));
        assert_eq!(c.cursor().map(|stack| stack.item_id), Some(401));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
        assert_eq!(c.slot_stack(2), None);
        assert_eq!(c.slot_stack(3), None);
        assert_eq!(c.slot_stack(4), None);
    }

    #[test]
    fn dynamic_slot_data_keeps_vanilla_nbt_type_gates_for_dye_and_banner_color() {
        let mut display = NbtCompound::new();
        display.insert("color", Nbt::Byte(7));
        let mut leather_root = NbtCompound::new();
        leather_root.insert("display", Nbt::Compound(display));
        let leather = tagged_stack(298, 1, 0, leather_root);

        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::String("not-an-integer".to_owned()));
        let mut banner_root = NbtCompound::new();
        banner_root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        let banner = tagged_stack(425, 1, 5, banner_root);

        let c = layouts::crafting_table_container(1);
        assert_eq!(c.dynamic_slot_data(Some(&leather)).leather_color, None);
        assert_eq!(
            c.dynamic_slot_data(Some(&banner)).banner_base_color,
            Some(0),
            "a present but nonnumeric Base follows getInteger's zero fallback"
        );
    }

    #[test]
    fn malformed_firework_fade_preserves_the_source_root_without_synthesizing_explosion() {
        let mut source_root = NbtCompound::new();
        source_root.insert("Unrelated", Nbt::Int(7));
        let source = tagged_stack(402, 1, 0, source_root.clone());

        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(source));
        c.set_slot_contents(2, Some(stack(351, 5))); // dye

        let expected = tagged_stack(402, 1, 0, source_root);
        assert_eq!(c.slot_stack(0), Some(&expected));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor(), Some(&expected));
        assert_eq!(
            c.cursor().and_then(|stack| stack.nbt.as_ref()),
            expected.nbt.as_ref()
        );
    }

    #[test]
    fn map_clone_copies_only_the_filled_map_custom_name_into_the_result() {
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Atlas".to_owned()));
        display.insert(
            "Lore",
            Nbt::list(vec![Nbt::String("keep source only".to_owned())]),
        );
        let mut source_root = NbtCompound::new();
        source_root.insert("display", Nbt::Compound(display));
        source_root.insert("map_is_scaling", Nbt::Byte(1));

        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(stack(395, 1))); // blank map
        c.set_slot_contents(2, Some(tagged_stack(358, 1, 12, source_root)));

        let mut expected_display = NbtCompound::new();
        expected_display.insert("Name", Nbt::String("Atlas".to_owned()));
        let mut expected_root = NbtCompound::new();
        expected_root.insert("display", Nbt::Compound(expected_display));
        let expected = tagged_stack(358, 2, 12, expected_root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor().map(|stack| stack.count), Some(2));
        assert_eq!(c.slot_stack(0), None);
    }

    #[test]
    fn map_extending_recomputes_when_the_host_supplies_map_data_scale() {
        let mut c = layouts::crafting_table_container(1);
        for slot in 1..10 {
            c.set_slot_contents(
                slot,
                Some(if slot == 5 {
                    stack(358, 1)
                } else {
                    stack(339, 1)
                }),
            );
        }
        assert_eq!(
            c.slot_stack(0),
            None,
            "missing MapData is a vanilla non-match"
        );

        c.set_map_scale(0, Some(3));
        let mut root = NbtCompound::new();
        root.insert("map_is_scaling", Nbt::Byte(1));
        let expected = tagged_stack(358, 1, 0, root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.slot_stack(0), None);
        for slot in 1..10 {
            assert_eq!(c.slot_stack(slot), None, "slot {slot} consumed");
        }
    }

    #[test]
    fn window_items_banner_add_applies_pattern_nbt_and_consumes_one_each_on_pickup() {
        let mut old_pattern = NbtCompound::new();
        old_pattern.insert("Pattern", Nbt::String("sku".to_owned()));
        old_pattern.insert("Color", Nbt::Int(2));
        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Int(5));
        block_entity.insert(
            "Patterns",
            Nbt::list(vec![Nbt::Compound(old_pattern.clone())]),
        );
        let mut source_root = NbtCompound::new();
        source_root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        source_root.insert("Unrelated", Nbt::Int(17));
        let source = tagged_stack(425, 2, 5, source_root.clone());

        // Apply the same bulk snapshot a live player-inventory screen receives. The result slot
        // is deliberately absent; the matrix callback must derive its NBT-bearing output.
        let mut c = layouts::player_container(9);
        let mut items = vec![None; c.slots().len()];
        items[1] = Some(source);
        items[2] = Some(stack(45, 2)); // brick -> `bri`
        items[3] = Some(ItemStack {
            item_id: 351,
            count: 2,
            damage: 3,
            nbt: None,
        }); // dye metadata is copied verbatim as Color
        c.apply_window_items(&mc_protocol::play::s2c::WindowItems {
            window_id: 9,
            items,
        });

        let mut expected_pattern = NbtCompound::new();
        expected_pattern.insert("Pattern", Nbt::String("bri".to_owned()));
        expected_pattern.insert("Color", Nbt::Int(3));
        let mut expected_block_entity = NbtCompound::new();
        expected_block_entity.insert("Base", Nbt::Int(5));
        expected_block_entity.insert(
            "Patterns",
            Nbt::list(vec![
                Nbt::Compound(old_pattern),
                Nbt::Compound(expected_pattern),
            ]),
        );
        let mut expected_root = NbtCompound::new();
        expected_root.insert("BlockEntityTag", Nbt::Compound(expected_block_entity));
        expected_root.insert("Unrelated", Nbt::Int(17));
        let expected = tagged_stack(425, 1, 5, expected_root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor(), Some(&expected));
        assert_eq!(
            c.slot_stack(0),
            Some(&expected),
            "the second craft remains visible"
        );
        assert_eq!(c.slot_stack(1).map(|stack| stack.count), Some(1));
        assert_eq!(c.slot_stack(2).map(|stack| stack.count), Some(1));
        assert_eq!(c.slot_stack(3).map(|stack| stack.count), Some(1));
    }

    #[test]
    fn window_items_banner_duplicate_copies_nbt_and_returns_patterned_remainder() {
        let mut pattern = NbtCompound::new();
        pattern.insert("Pattern", Nbt::String("bri".to_owned()));
        pattern.insert("Color", Nbt::Int(4));
        let mut block_entity = NbtCompound::new();
        block_entity.insert("Base", Nbt::Int(3));
        block_entity.insert("Patterns", Nbt::list(vec![Nbt::Compound(pattern)]));
        let mut patterned_root = NbtCompound::new();
        patterned_root.insert("BlockEntityTag", Nbt::Compound(block_entity));
        patterned_root.insert("Custom", Nbt::String("keep".to_owned()));
        let patterned = tagged_stack(425, 1, 3, patterned_root.clone());

        let mut c = layouts::player_container(10);
        let mut items = vec![None; c.slots().len()];
        items[1] = Some(patterned);
        items[2] = Some(ItemStack {
            item_id: 425,
            count: 2,
            damage: 3,
            nbt: None,
        }); // blank banner; metadata supplies matching base color
        c.apply_window_items(&mc_protocol::play::s2c::WindowItems {
            window_id: 10,
            items,
        });

        let expected = tagged_stack(425, 1, 3, patterned_root.clone());
        assert_eq!(c.slot_stack(0), Some(&expected));

        // A server result snapshot clears local provenance. Pickup must still run the live
        // dynamic evaluator so SlotCrafting can return this patterned banner as a remainder.
        c.set_slot_contents(0, Some(expected.clone()));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor(), Some(&expected));
        assert_eq!(
            c.slot_stack(1),
            Some(&expected),
            "the patterned banner is the recipe remainder, including its complete NBT"
        );
        assert_eq!(
            c.slot_stack(2)
                .map(|stack| (stack.item_id, stack.count, stack.damage)),
            Some((425, 1, 3))
        );
        assert_eq!(
            c.slot_stack(0),
            Some(&expected),
            "duplicate remains craftable"
        );
    }

    #[test]
    fn window_items_map_extending_copies_source_nbt_and_accepts_negative_scale() {
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Survey".to_owned()));
        let mut source_root = NbtCompound::new();
        source_root.insert("display", Nbt::Compound(display));
        source_root.insert("Custom", Nbt::Int(91));
        source_root.insert("map_is_scaling", Nbt::Byte(0));
        let source = tagged_stack(358, 2, 7, source_root.clone());

        let mut c = layouts::crafting_table_container(11);
        let mut items = vec![None; c.slots().len()];
        for (slot, item) in items.iter_mut().enumerate().take(10).skip(1) {
            *item = Some(if slot == 5 {
                source.clone()
            } else {
                stack(339, 2)
            });
        }
        c.apply_window_items(&mc_protocol::play::s2c::WindowItems {
            window_id: 11,
            items,
        });
        assert_eq!(
            c.slot_stack(0),
            None,
            "MapData is unavailable until supplied"
        );

        // MapData.scale is a signed Java byte. A negative value is unusual but still satisfies
        // RecipesMapExtending's literal `mapdata.scale < 4` predicate.
        c.set_map_scale(7, Some(-1));
        let mut expected_root = source_root.clone();
        expected_root.insert("map_is_scaling", Nbt::Byte(1));
        let expected = tagged_stack(358, 1, 7, expected_root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        // Exercise the direct SetSlot path too: replacing the locally predicted output must not
        // lose the map-extending adapter used by SlotCrafting during pickup.
        c.set_slot_contents(0, Some(expected.clone()));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor(), Some(&expected));
        assert_eq!(
            c.slot_stack(0),
            Some(&expected),
            "the second map craft remains visible"
        );
        for slot in 1..10 {
            assert_eq!(
                c.slot_stack(slot).map(|stack| stack.count),
                Some(1),
                "slot {slot}"
            );
        }
        assert_eq!(
            c.slot_stack(5).and_then(|stack| stack.nbt.as_ref()),
            Some(&source_root)
        );
    }

    #[test]
    fn window_items_leather_dye_applies_color_nbt_and_consumes_inputs_on_pickup() {
        let mut display = NbtCompound::new();
        display.insert("Name", Nbt::String("Hide".to_owned()));
        display.insert(
            "Lore",
            Nbt::list(vec![Nbt::String("source-only".to_owned())]),
        );
        let mut source_root = NbtCompound::new();
        source_root.insert("display", Nbt::Compound(display));
        source_root.insert("Custom", Nbt::Int(23));
        let source = tagged_stack(298, 2, 37, source_root.clone());

        let mut c = layouts::player_container(12);
        let mut items = vec![None; c.slots().len()];
        items[1] = Some(source);
        items[2] = Some(ItemStack {
            item_id: 351,
            count: 2,
            damage: 1,
            nbt: None,
        }); // red dye
        c.apply_window_items(&mc_protocol::play::s2c::WindowItems {
            window_id: 12,
            items,
        });

        let mut expected_display = NbtCompound::new();
        expected_display.insert("Name", Nbt::String("Hide".to_owned()));
        expected_display.insert(
            "Lore",
            Nbt::list(vec![Nbt::String("source-only".to_owned())]),
        );
        expected_display.insert("color", Nbt::Int(0x99_33_33));
        let mut expected_root = NbtCompound::new();
        expected_root.insert("display", Nbt::Compound(expected_display));
        expected_root.insert("Custom", Nbt::Int(23));
        let expected = tagged_stack(298, 1, 37, expected_root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        // Replacing the prediction with an authoritative result exercises the same fallback as
        // a result arriving after the matrix snapshot in a live window.
        c.set_slot_contents(0, Some(expected.clone()));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor(), Some(&expected));
        assert_eq!(
            c.slot_stack(0),
            Some(&expected),
            "the second dye craft remains visible"
        );
        assert_eq!(c.slot_stack(1).map(|stack| stack.count), Some(1));
        assert_eq!(c.slot_stack(2).map(|stack| stack.count), Some(1));
    }

    #[test]
    fn banner_add_does_not_append_to_a_non_compound_typed_patterns_list() {
        let mut patterns = NbtCompound::new();
        patterns.insert("Patterns", Nbt::list(vec![Nbt::Int(99)]));
        let mut source_root = NbtCompound::new();
        source_root.insert("BlockEntityTag", Nbt::Compound(patterns));

        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(tagged_stack(425, 1, 2, source_root.clone())));
        c.set_slot_contents(2, Some(stack(45, 1))); // brick pattern item (`bri`)
        c.set_slot_contents(3, Some(stack(351, 4))); // dye

        let expected = tagged_stack(425, 1, 2, source_root);
        assert_eq!(c.slot_stack(0), Some(&expected));
        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected));
    }

    #[test]
    fn written_book_clone_preserves_source_nbt_for_result_and_remaining_book() {
        let mut pages = NbtCompound::new();
        pages.insert("title", Nbt::String("A".to_owned()));
        let mut source_root = NbtCompound::new();
        source_root.insert("pages", Nbt::list(vec![Nbt::String("hello".to_owned())]));
        source_root.insert("title", Nbt::String("Book".to_owned()));
        source_root.insert("author", Nbt::String("Author".to_owned()));
        source_root.insert("generation", Nbt::Int(0));
        source_root.insert("custom", Nbt::Compound(pages));

        let mut c = layouts::crafting_table_container(1);
        c.set_slot_contents(1, Some(tagged_stack(387, 1, 0, source_root.clone())));
        c.set_slot_contents(2, Some(stack(386, 1)));
        c.set_slot_contents(3, Some(stack(386, 1)));

        let mut expected_root = source_root;
        expected_root.insert("generation", Nbt::Int(1));
        let expected = tagged_stack(387, 2, 0, expected_root);
        assert_eq!(c.slot_stack(0), Some(&expected));

        let packet = c.click(0, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(expected.clone()));
        assert_eq!(c.cursor().map(|stack| stack.count), Some(2));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1).map(|stack| stack.item_id), Some(387));
        assert_eq!(c.slot_stack(1).map(|stack| stack.count), Some(1));
        let mut expected_remaining_root = NbtCompound::new();
        expected_remaining_root.insert("pages", Nbt::list(vec![Nbt::String("hello".to_owned())]));
        expected_remaining_root.insert("title", Nbt::String("Book".to_owned()));
        expected_remaining_root.insert("author", Nbt::String("Author".to_owned()));
        expected_remaining_root.insert("generation", Nbt::Int(0));
        let mut expected_remaining_custom = NbtCompound::new();
        expected_remaining_custom.insert("title", Nbt::String("A".to_owned()));
        expected_remaining_root.insert("custom", Nbt::Compound(expected_remaining_custom));
        assert_eq!(
            c.slot_stack(1).and_then(|stack| stack.nbt.as_ref()),
            Some(&expected_remaining_root)
        );
        assert_eq!(c.slot_stack(2), None);
        assert_eq!(c.slot_stack(3), None);
    }

    #[test]
    fn anvil_material_repair_predicts_cost_and_consumes_inputs_on_pickup() {
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(
            0,
            Some(ItemStack {
                item_id: 257,
                count: 1,
                damage: 200,
                nbt: None,
            }),
        );
        c.set_slot_contents(1, Some(stack(265, 1)));
        assert_eq!(c.maximum_cost, Some(1));
        assert_eq!(
            c.slot_stack(2).map(|s| (s.item_id, s.damage)),
            Some((257, 138))
        );

        c.set_player_experience_level(1);
        let packet = c.click(2, 0, 0, false);
        assert_eq!(packet.clicked_item.as_ref().map(|s| s.damage), Some(138));
        assert_eq!(c.cursor().map(|s| s.damage), Some(138));
        assert_eq!(c.slot_stack(0), None);
        assert_eq!(c.slot_stack(1), None);
        assert_eq!(c.slot_stack(2), None);
        assert_eq!(c.maximum_cost, Some(0));
        assert_eq!(c.player_experience_level, Some(0));
    }

    /// `ContainerRepair.updateRepairOutput` asks `ItemStack.isItemStackDamageable` before the
    /// material-repair branch. A truthy Unbreakable tag therefore rejects an otherwise valid
    /// pickaxe+ingot repair instead of silently stripping the protection and producing output.
    #[test]
    fn anvil_unbreakable_material_is_not_repaired() {
        let mut tag = NbtCompound::new();
        tag.insert("Unbreakable", Nbt::Byte(1));

        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(tagged_stack(257, 1, 200, tag)));
        c.set_slot_contents(1, Some(stack(265, 1))); // iron ingot repairs an iron pickaxe

        assert_eq!(c.maximum_cost, Some(0));
        assert_eq!(c.slot_stack(2), None);
    }

    #[test]
    fn anvil_same_enchantment_merges_and_incompatible_enchantment_is_ignored() {
        fn enchanted(item_id: i16, enchantments: &[(i16, i16)]) -> ItemStack {
            let mut root = NbtCompound::new();
            let list = enchantments
                .iter()
                .map(|(id, level)| {
                    Nbt::Compound(
                        [
                            ("id".to_owned(), Nbt::Short(*id)),
                            ("lvl".to_owned(), Nbt::Short(*level)),
                        ]
                        .into_iter()
                        .collect(),
                    )
                })
                .collect();
            root.insert("ench", Nbt::list(list));
            ItemStack {
                item_id,
                count: 1,
                damage: 0,
                nbt: Some(root),
            }
        }

        let mut merged = layouts::anvil_container(0);
        merged.set_slot_contents(0, Some(enchanted(276, &[(16, 1)])));
        merged.set_slot_contents(1, Some(enchanted(276, &[(16, 1)])));
        let output = merged.slot_stack(2).expect("same enchantment is valid");
        assert_eq!(stack_enchantments(output, false), vec![(16, 2)]);

        let mut incompatible = layouts::anvil_container(0);
        incompatible.set_slot_contents(0, Some(enchanted(276, &[(16, 1)])));
        incompatible.set_slot_contents(1, Some(enchanted(276, &[(17, 1)])));
        let output = incompatible
            .slot_stack(2)
            .expect("the base enchantment remains a valid output");
        assert_eq!(stack_enchantments(output, false), vec![(16, 1)]);
        assert!(!stack_enchantments(output, false)
            .iter()
            .any(|(id, _)| *id == 17));
    }

    #[test]
    fn anvil_enchantment_controls_match_java_subclass_overrides() {
        let sharpness = mc_data::enchantment(16).expect("sharpness");
        let silk_touch = mc_data::enchantment(33).expect("silk touch");
        let sword = stack(276, 1);
        let axe = stack(258, 1);
        let shears = stack(359, 1);
        assert!(enchantment_can_apply(sharpness, &sword));
        assert!(
            enchantment_can_apply(sharpness, &axe),
            "EnchantmentDamage widens WEAPON to ItemAxe"
        );
        assert!(
            enchantment_can_apply(silk_touch, &shears),
            "EnchantmentUntouching widens DIGGER to shears"
        );

        assert!(!enchantments_can_apply_together(0, 1));
        assert!(enchantments_can_apply_together(0, 2));
        assert!(enchantments_can_apply_together(2, 0));
        assert!(!enchantments_can_apply_together(16, 17));
        assert!(!enchantments_can_apply_together(21, 33));
        assert!(
            enchantments_can_apply_together(33, 21),
            "the 1.8.9 LootBonus/SilkTouch overrides are directional"
        );
        assert!(!enchantments_can_apply_together(33, 35));
    }

    fn enchantment_list(entries: &[(i16, i16)]) -> Nbt {
        Nbt::list(
            entries
                .iter()
                .map(|&(id, level)| enchantment_entry(id, level))
                .collect(),
        )
    }

    fn enchanted_book(stored: &[(i16, i16)]) -> ItemStack {
        let mut root = NbtCompound::new();
        root.insert("StoredEnchantments", enchantment_list(stored));
        tagged_stack(ENCHANTED_BOOK_ID, 1, 0, root)
    }

    /// java: `ContainerRepair.updateRepairOutput` reads both books through
    /// `EnchantmentHelper.getEnchantments` (`StoredEnchantments`) and writes the result back with
    /// `setEnchantments`, which for a book goes through `ItemEnchantedBook.addEnchantment` and
    /// never adds an `ench` list. Two Sharpness I books make one Sharpness II book costing 2
    /// (weight 10 -> 1, halved but floored at 1 for a book, times level 2). Negative control: the
    /// old code read the first book's empty `ench`, predicted Sharpness I at cost 1, wrote it to
    /// `ench` and left the copied `StoredEnchantments` at level I.
    #[test]
    fn anvil_combines_two_enchanted_books_through_stored_enchantments() {
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(enchanted_book(&[(16, 1)])));
        c.set_slot_contents(1, Some(enchanted_book(&[(16, 1)])));
        assert_eq!(c.maximum_cost, Some(2));
        let output = c.slot_stack(2).expect("two equal books combine");
        let root = output.nbt.as_ref().expect("the output keeps its tag");
        assert_eq!(stack_enchantments(output, true), vec![(16, 2)]);
        assert!(
            root.get("ench").is_none(),
            "a book never gets an `ench` list"
        );
        assert_eq!(root.get("RepairCost"), Some(&Nbt::Int(1)));
    }

    /// java: `ItemStack.getRepairCost` is `hasKey("RepairCost", 3) ?
    /// getInteger("RepairCost") : 0`: a negative `TAG_Int` is used as stored, any other tag type
    /// counts as 0. Renaming a sword whose cost is -5 costs -5 + 1 = -4 (an output the slot's
    /// `maximumCost > 0` gate refuses) and stamps `-5 * 2 + 1`; a `TAG_Byte` cost of 5 is ignored.
    /// Negative control: the old reader clamped to 0 and coerced the byte (costs 1 and 6).
    #[test]
    fn anvil_repair_cost_is_a_raw_unclamped_tag_int() {
        let renamed = |cost: Nbt| {
            let mut root = NbtCompound::new();
            root.insert("RepairCost", cost);
            let mut c = layouts::anvil_container(0);
            c.set_slot_contents(0, Some(tagged_stack(276, 1, 0, root)));
            c.set_anvil_name("Blade");
            c
        };
        let repair_cost_of = |c: &Container| {
            c.slot_stack(2)
                .and_then(|stack| stack.nbt.as_ref())
                .and_then(|root| root.get("RepairCost"))
                .cloned()
        };

        let negative = renamed(Nbt::Int(-5));
        assert_eq!(negative.maximum_cost, Some(-4));
        assert_eq!(repair_cost_of(&negative), Some(Nbt::Int(-9)));
        assert!(!negative.can_take_stack(2, false));

        let byte = renamed(Nbt::Byte(5));
        assert_eq!(byte.maximum_cost, Some(1));
        assert_eq!(repair_cost_of(&byte), Some(Nbt::Int(1)));
    }

    /// java: `EnchantmentThorns.canApply` accepts every `ItemArmor` and
    /// `EnchantmentDigging.canApply` accepts shears, beyond their ARMOR_TORSO/DIGGER targets, so
    /// a Thorns book enchants leggings (weight 1 -> 8, halved for a book: cost 4). Negative
    /// control: the old predicate refused both and the leggings had no output.
    #[test]
    fn anvil_thorns_fits_any_armour_and_efficiency_fits_shears() {
        let thorns = mc_data::enchantment(7).expect("thorns");
        let efficiency = mc_data::enchantment(32).expect("efficiency");
        let leggings = stack(312, 1);
        assert!(enchantment_can_apply(thorns, &leggings));
        assert!(enchantment_can_apply(efficiency, &stack(359, 1)));

        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(leggings));
        c.set_slot_contents(1, Some(enchanted_book(&[(7, 1)])));
        assert_eq!(c.maximum_cost, Some(4));
        let output = c.slot_stack(2).expect("a Thorns book enchants leggings");
        assert_eq!(stack_enchantments(output, false), vec![(7, 1)]);
    }

    /// java: `getEnchantments` builds a `LinkedHashMap` (a repeated id keeps its
    /// first position and its last level) and `setEnchantments` writes back registered ids only.
    /// Sharpness I, unregistered id 100 and Sharpness III on one sword (map: III) plus a Sharpness
    /// III sword make Sharpness IV at cost 4 with id 100 dropped. Negative control: the old list
    /// merged against the first entry (I -> III, cost 3) and wrote `[(16,3), (100,1), (16,3)]`.
    #[test]
    fn anvil_enchantment_maps_dedupe_ids_and_drop_unregistered_ones() {
        let sword = |enchantments: &[(i16, i16)]| {
            let mut root = NbtCompound::new();
            root.insert("ench", enchantment_list(enchantments));
            tagged_stack(276, 1, 0, root)
        };
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(sword(&[(16, 1), (100, 1), (16, 3)])));
        c.set_slot_contents(1, Some(sword(&[(16, 3)])));
        assert_eq!(c.maximum_cost, Some(4));
        let output = c.slot_stack(2).expect("two enchanted swords combine");
        assert_eq!(stack_enchantments(output, false), vec![(16, 4)]);
    }

    /// java: `StringUtils.isBlank` uses `Character.isWhitespace`, which is not Rust's
    /// `char::is_whitespace`: the no-break spaces and NEL are not blank, U+001C..U+001F are.
    #[test]
    fn anvil_blank_names_follow_java_character_is_whitespace() {
        assert!(java_is_blank(""));
        assert!(java_is_blank(" \u{3000}\u{2028}\u{1F}"));
        for not_blank in ["\u{A0}", "\u{2007}", "\u{202F}", "\u{85}", "a"] {
            assert!(!java_is_blank(not_blank), "{not_blank:?}");
        }
    }

    /// java: `updateRepairOutput` compares the typed name with
    /// `getDisplayName()` in the player's language. A client in French shows a plain apple as
    /// "Pomme": typing the English "Apple" is a one-level rename, and "Pomme" is none.
    #[test]
    fn anvil_rename_test_uses_the_display_name_in_the_players_language() {
        let apple = stack(260, 1);
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(apple.clone()));
        c.set_anvil_input_display_name(Some((apple.clone(), "Pomme".to_owned())));
        c.set_anvil_name("Apple");
        assert_eq!(c.maximum_cost, Some(1), "Apple is not the displayed Pomme");
        let output = c.slot_stack(2).expect("a one-level rename");
        assert_eq!(stack_custom_name(output), Some("Apple"));

        c.set_anvil_name("Pomme");
        assert_eq!(
            c.maximum_cost,
            Some(0),
            "the displayed name itself is no rename"
        );
        assert_eq!(c.slot_stack(2), None);

        // A resolution made for another stack is never applied: the registry name stands in.
        let mut two = apple;
        two.count = 2;
        c.set_slot_contents(0, Some(two));
        c.set_anvil_name("Apple");
        assert_eq!(c.maximum_cost, Some(0));
        assert_eq!(c.slot_stack(2), None);
    }

    #[test]
    fn anvil_name_matching_the_vanilla_display_name_is_not_a_rename() {
        let mut c = layouts::anvil_container(0);
        c.set_slot_contents(0, Some(stack(1, 1)));
        c.set_anvil_name("Stone");
        assert_eq!(c.maximum_cost, Some(0));
        assert_eq!(c.slot_stack(2), None);

        c.set_anvil_name("Polished");
        let output = c.slot_stack(2).expect("renaming costs one level");
        assert_eq!(stack_custom_name(output), Some("Polished"));
    }

    #[test]
    fn anvil_name_matching_a_metadata_display_name_is_not_a_rename() {
        // ItemCloth and its sibling coloured ItemBlock families override
        // Item.getUnlocalizedName(ItemStack).  A one-name-per-id fallback makes every one of
        // these red stacks look like its white/base entry and incorrectly charges a rename.
        for (item_name, expected) in [
            ("wool", "Red Wool"),
            ("stained_glass", "Red Stained Glass"),
            ("stained_hardened_clay", "Red Stained Clay"),
            ("stained_glass_pane", "Red Stained Glass Pane"),
            ("carpet", "Red Carpet"),
        ] {
            let item_id = mc_data::item_by_name(item_name)
                .unwrap_or_else(|| panic!("missing item {item_name}"))
                .0 as i16;
            let mut c = layouts::anvil_container(0);
            c.set_slot_contents(
                0,
                Some(ItemStack {
                    item_id,
                    count: 1,
                    damage: 14,
                    nbt: None,
                }),
            );
            c.set_anvil_name(expected);
            assert_eq!(
                c.maximum_cost,
                Some(0),
                "{item_name}: matching the metadata-aware display name must be a no-op"
            );
            assert_eq!(
                c.slot_stack(2),
                None,
                "{item_name}: a no-op rename must not create an anvil output"
            );
        }
    }

    #[test]
    fn merchant_result_consumes_reversed_buy_slots_and_recomputes_the_retry() {
        let mut slots: Vec<Slot> = (0..39).map(|id| Slot::new(id, 0, 0)).collect();
        slots[2] = slots[2].with_role(super::super::slot::SlotRole::CraftResult);
        let mut c = Container::new(4, ContainerKind::Merchant, slots);
        let recipe = MerchantRecipe {
            buy_a: stack(265, 2),
            buy_b: Some(stack(331, 1)),
            sell: stack(388, 1),
            uses: 0,
            max_uses: 2,
            disabled: false,
        };
        c.set_merchant_recipe(Some(recipe));
        c.set_slot_contents(0, Some(stack(331, 2)));
        c.set_slot_contents(1, Some(stack(265, 4)));
        assert_eq!(c.slot_stack(2), Some(&stack(388, 1)));

        let packet = c.click(2, 0, 0, false);
        assert_eq!(packet.clicked_item, Some(stack(388, 1)));
        assert_eq!(c.cursor(), Some(&stack(388, 1)));
        assert_eq!(c.slot_stack(0), Some(&stack(331, 1)));
        assert_eq!(c.slot_stack(1), Some(&stack(265, 2)));
        assert_eq!(
            c.slot_stack(2),
            Some(&stack(388, 1)),
            "retry remains visible"
        );

        c.click(2, 0, 0, false);
        assert_eq!(c.cursor(), Some(&stack(388, 2)));
        assert_eq!(c.slot_stack(2), None, "the exhausted offer disappears");
    }

    /// java: `InventoryMerchant.resetRecipeAndSlots` matches the offer with
    /// item, metadata and NBT (trying the inputs reversed second), but `SlotMerchantResult`
    /// charges it through `doTrade`, which compares items only and tries the forward orientation
    /// first. Offer: red wool x2 + blue wool x3. With blue x3 in slot 0 and red x2 in slot 1 the
    /// matcher accepts the reversed pair, then `doTrade(blue, red)` takes 2 from blue and 3 from
    /// red: blue x1 is left and red (-1) is cleared. Negative control: the old consumption reused
    /// the matcher, charged red 2 and blue 3, and emptied both slots.
    #[test]
    fn merchant_trade_charges_inputs_by_item_identity_forward_first() {
        let wool = |damage: i16, count: i32| ItemStack {
            item_id: 35,
            count,
            damage,
            nbt: None,
        };
        let mut slots: Vec<Slot> = (0..39).map(|id| Slot::new(id, 0, 0)).collect();
        slots[2] = slots[2].with_role(super::super::slot::SlotRole::CraftResult);
        let mut c = Container::new(4, ContainerKind::Merchant, slots);
        c.set_merchant_recipe(Some(MerchantRecipe {
            buy_a: wool(14, 2),
            buy_b: Some(wool(11, 3)),
            sell: stack(388, 1),
            uses: 0,
            max_uses: 7,
            disabled: false,
        }));

        // Control: the matcher still compares metadata, so the wrong colour yields no offer.
        c.set_slot_contents(0, Some(wool(4, 3)));
        c.set_slot_contents(1, Some(wool(14, 2)));
        assert_eq!(c.slot_stack(2), None);

        c.set_slot_contents(0, Some(wool(11, 3)));
        assert_eq!(c.slot_stack(2), Some(&stack(388, 1)), "matched reversed");
        c.click(2, 0, 0, false);
        assert_eq!(c.cursor(), Some(&stack(388, 1)));
        assert_eq!(c.slot_stack(0), Some(&wool(11, 1)));
        assert_eq!(c.slot_stack(1), None);
    }
}
