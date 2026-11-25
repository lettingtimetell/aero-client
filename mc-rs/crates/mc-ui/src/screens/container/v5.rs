//! Minecraft 1.7.10's windows: which container a 1.7.10 server's `S2DPacketOpenWindow` opens,
//! and the container shapes and rules that differ from 1.8.9's.
//!
//! A 1.7.10 server names a window by number (java17 `S2DPacketOpenWindow`, types 0 to 11) and
//! `NetHandlerPlayClient.handleOpenWindow` builds the client's container from that number alone.
//! [`open_window_v47`] turns the packet into the 1.8.9 [`s2c::OpenWindow`] the rest of the
//! client carries (the same window family, a chat-component title), and
//! [`container_for_open_window_v5`] builds the 1.7.10 container for it. What differs from 1.8.9:
//!
//! - the enchanting table has one slot and no lapis ([`enchanting_table_container_v5`]);
//! - the furnace's fuel slot takes any item ([`furnace_container_v5`]), and its shift-click
//!   routing uses 1.7.10's smelting list ([`is_smeltable_v5`]);
//! - the crafting table, enchanting table and anvil open whatever slot count the packet says,
//!   and the hopper always has five slots;
//! - every container predicts clicks with 1.7.10's rules (`model_v5.rs`).
//!
//! java17: `client/network/NetHandlerPlayClient.java:877-985`, `inventory/Container*.java`.

use mc_protocol::play::s2c;
use mc_protocol::v5::s2c as s2c5;
use mc_version::GameVersion;

use super::layouts;
use super::model::{Container, ContainerKind};
use super::slot::{Slot, SlotRole};
use super::stack_ext::ItemStack;
use crate::font::component::ChatComponent;

/// The 1.8.9 window type each 1.7.10 window number lowers to, by number (java17
/// `NetHandlerPlayClient.handleOpenWindow`'s `switch`). A number past the end opens nothing.
pub const WINDOW_TYPES_V5: [&str; 12] = [
    "minecraft:chest",
    "minecraft:crafting_table",
    "minecraft:furnace",
    "minecraft:dispenser",
    "minecraft:enchanting_table",
    "minecraft:brewing_stand",
    "minecraft:villager",
    "minecraft:beacon",
    "minecraft:anvil",
    "minecraft:hopper",
    "minecraft:dropper",
    "EntityHorse",
];

/// A 1.7.10 `S2DPacketOpenWindow` as the 1.8.9 packet the client carries, or `None` for a
/// window number 1.7.10 does not open (its `switch` has no default).
///
/// - The type is [`WINDOW_TYPES_V5`]'s name. The slot count is what that window really has in
///   1.7.10: the packet's for a chest and a horse, `0` for the three windows 1.8.9 builds on the
///   client (crafting table, enchanting table, anvil: the 1.7.10 server sends 9 for them), the
///   fixed size for the rest (a hopper always has 5).
/// - The title is a text component when the server provided one, else the name the 1.7.10
///   screen draws: a chest's or a horse's title is a translation key, the other windows fall
///   back to their `container.*` key, the villager to `entity.Villager.name`. An empty provided
///   name counts as none for the furnace, brewing stand, villager, beacon and hopper, which test
///   its length; the dispenser, dropper and enchanting table keep it (java17
///   `TileEntityDispenser.isCustomInventoryName` only tests for `null`, and the dropper inherits
///   it; `GuiChest`, `GuiFurnace`, `GuiDispenser`, `GuiBrewingStand`, `GuiHopper`,
///   `GuiScreenHorseInventory`, `GuiMerchant`, `GuiEnchantment`).
/// - A horse window carries the entity id. Whether that entity is a horse is the caller's to
///   check: 1.7.10 opens nothing otherwise.
#[must_use]
pub fn open_window_v47(pkt: &s2c5::OpenWindow) -> Option<s2c::OpenWindow> {
    let inventory_type = *WINDOW_TYPES_V5.get(usize::from(pkt.inventory_type))?;
    let provided = pkt.use_provided_title.then_some(pkt.title.as_str());
    let named = provided.filter(|title| !title.is_empty());
    let (slot_count, title) = match pkt.inventory_type {
        0 | 11 => (pkt.slot_count, title_or_key(provided, &pkt.title)),
        1 => (0, translated("container.crafting")),
        2 => (3, title_or_key(named, "container.furnace")),
        3 => (9, title_or_key(provided, "container.dispenser")),
        4 => (0, title_or_key(provided, "container.enchant")),
        5 => (4, title_or_key(named, "container.brewing")),
        6 => (3, title_or_key(named, "entity.Villager.name")),
        7 => (1, title_or_key(named, "container.beacon")),
        8 => (0, translated("container.repair")),
        9 => (5, title_or_key(named, "container.hopper")),
        _ => (9, title_or_key(provided, "container.dropper")),
    };
    Some(s2c::OpenWindow {
        window_id: pkt.window_id,
        inventory_type: inventory_type.to_owned(),
        window_title: title,
        slot_count,
        entity_id: (pkt.inventory_type == 11).then_some(pkt.entity_id),
    })
}

fn title_or_key(title: Option<&str>, key: &str) -> String {
    match title {
        Some(title) => ChatComponent::text(title).to_json(),
        None => translated(key),
    }
}

fn translated(key: &str) -> String {
    ChatComponent::translate(key, Vec::new()).to_json()
}

/// The 1.7.10 container for a window [`open_window_v47`] lowered, or `None` for a type it never
/// produces. The window's `slot_count` is used only by the chest (`slot_count / 9` rows, as
/// 1.7.10's `GuiChest`) and the horse (2 or 17 slots).
#[must_use]
pub fn container_for_open_window_v5(pkt: &s2c::OpenWindow) -> Option<Container> {
    let id = pkt.window_id;
    let container = match pkt.inventory_type.as_str() {
        "minecraft:chest" => layouts::chest_container(id, pkt.slot_count / 9),
        "minecraft:crafting_table" => layouts::crafting_table_container(id),
        "minecraft:furnace" => furnace_container_v5(id),
        "minecraft:dispenser" | "minecraft:dropper" => layouts::dispenser_container(id),
        "minecraft:enchanting_table" => enchanting_table_container_v5(id),
        "minecraft:brewing_stand" => layouts::brewing_stand_container(id),
        "minecraft:beacon" => layouts::beacon_container(id),
        "minecraft:anvil" => layouts::anvil_container(id),
        "minecraft:hopper" => layouts::hopper_container(id, 5),
        // The villager and the horse have the same shapes in both versions.
        "minecraft:villager" | "EntityHorse" => super::net::container_for_open_window(pkt)?,
        _ => return None,
    };
    Some(container.with_game_version(GameVersion::V1_7_10))
}

/// 1.7.10's enchanting table: the item slot (stack limit 1, any item), then the player's 27 + 9
/// slots, 37 in all. java17: `ContainerEnchantment.<init>`
/// (`inventory/ContainerEnchantment.java:38-62`), whose table inventory has one slot and a stack
/// limit of 1.
///
/// The slot sits where the 1.8.9 panel art this mode reuses draws its item frame (`15, 47`);
/// 1.7.10's own panel has it at `25, 47`. Only the drawing differs.
#[must_use]
pub fn enchanting_table_container_v5(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(37);
    slots.push(Slot::new(0, 15, 47).with_limit(1));
    push_player_slots(&mut slots);
    Container::new(window_id, ContainerKind::EnchantingV5, slots)
        .with_game_version(GameVersion::V1_7_10)
}

/// 1.7.10's furnace: input (0), fuel (1, a plain slot that takes any item, unlike 1.8.9's
/// `SlotFurnaceFuel`), output (2), then the player's slots. java17: `ContainerFurnace.<init>`
/// (`inventory/ContainerFurnace.java:17-35`).
#[must_use]
pub fn furnace_container_v5(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(39);
    slots.push(Slot::new(0, 56, 17));
    slots.push(Slot::new(1, 56, 53));
    slots.push(Slot::new(2, 116, 35).with_role(SlotRole::Output));
    push_player_slots(&mut slots);
    Container::new(window_id, ContainerKind::Furnace, slots).with_game_version(GameVersion::V1_7_10)
}

/// The 27 main and 9 hotbar slots every container ends with, at the usual `84`/`142` rows.
fn push_player_slots(slots: &mut Vec<Slot>) {
    for row in 0..3 {
        for col in 0..9 {
            let id = slots.len() as u16;
            slots.push(Slot::new(id, 8 + col * 18, 84 + row * 18));
        }
    }
    for col in 0..9 {
        let id = slots.len() as u16;
        slots.push(Slot::new(id, 8 + col * 18, 142));
    }
}

/// `FurnaceRecipes.getSmeltingResult(stack) != null` with 1.7.10's list: 1.8.9's minus the two
/// recipes 1.8 added for items 1.7.10 also has, stone bricks to cracked stone bricks and a wet
/// sponge to a sponge. java17: `item/crafting/FurnaceRecipes.java:26-61`.
#[must_use]
pub fn is_smeltable_v5(stack: &ItemStack) -> bool {
    const STONE_BRICK: (i16, i16) = (98, 0);
    const WET_SPONGE: (i16, i16) = (19, 1);
    let key = (stack.item_id, stack.damage);
    key != STONE_BRICK && key != WET_SPONGE && super::slot::is_smeltable(stack)
}

/// The left edge of a 1.7.10 inventory (survival or creative). java17
/// `InventoryEffectRenderer.initGui` moves it right for active potion effects when the screen
/// is laid out, and nothing moves it while the screen stays open; 1.8.9 moves it every tick as
/// effects come and go (`updateActivePotionEffects`), and the slots move with it.
///
/// This client lays an open screen out on every frame, so a new layout here is a new screen
/// size (java17 runs `initGui` again from `setWorldAndResolution`) or a change of the
/// `centered_inventory` mod's centring, which so applies at once as in 1.8.9.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct InitShiftV5 {
    /// The screen size of the latest layout.
    size: (i32, i32),
    /// The size and centring the kept edge was chosen at, and the edge.
    chosen: Option<((i32, i32), bool, i32)>,
}

impl InitShiftV5 {
    /// The screen is laid out at `width` x `height` (its `init`).
    pub(crate) fn laid_out(&mut self, width: i32, height: i32) {
        self.size = (width, height);
    }

    /// The left edge for a screen whose 1.8.9 rule computed `left` now: `left` itself at a new
    /// layout, else the edge chosen at the last one.
    pub(crate) fn left(&mut self, left: i32, centered: bool) -> i32 {
        match self.chosen {
            Some((size, was_centered, kept)) if size == self.size && was_centered == centered => {
                kept
            }
            _ => {
                self.chosen = Some((self.size, centered, left));
                left
            }
        }
    }
}

#[cfg(test)]
#[path = "v5_tests.rs"]
mod tests;
