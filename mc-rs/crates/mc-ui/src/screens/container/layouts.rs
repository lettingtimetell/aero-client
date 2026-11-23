//! Builders for every vanilla container shape S17 covers: the always-open player inventory,
//! chest/double-chest/ender-chest/hopper/dispenser/dropper, furnace, brewing stand, enchanting
//! table, anvil, and beacon.
//!
//! Every slot position below is transcribed from the matching `ContainerXxx` constructor's
//! literal `addSlotToContainer(new Slot(inv, index, x, y))` calls, so a screen built on top of
//! these lines up pixel-for-pixel with vanilla without this port needing its own layout
//! constants rediscovered by trial and error.

use super::model::{Container, ContainerKind};
use super::slot::{Slot, SlotRole};

/// java: `ItemArmor.EMPTY_SLOT_NAMES` (`item/ItemArmor.java:24`), in `armorType` order:
/// helmet, chestplate, leggings, boots. The `minecraft:` namespace is implied.
const ARMOR_SLOT_SPRITES: [&str; 4] = [
    "items/empty_armor_slot_helmet",
    "items/empty_armor_slot_chestplate",
    "items/empty_armor_slot_leggings",
    "items/empty_armor_slot_boots",
];

fn push_slot(slots: &mut Vec<Slot>, x: i32, y: i32) {
    let id = slots.len() as u16;
    slots.push(Slot::new(id, x, y));
}

fn push_slot_with(slots: &mut Vec<Slot>, x: i32, y: i32, f: impl FnOnce(Slot) -> Slot) {
    let id = slots.len() as u16;
    slots.push(f(Slot::new(id, x, y)));
}

/// Appends the 27-slot main inventory then the 9-slot hotbar, at `main_top`/`hotbar_y`. Every
/// vanilla container this port builds ends with exactly this block. java: the
/// `for (l = 0; l < 3; ++l) for (j1 = 0; j1 < 9; ++j1) ...` / `for (i1 = 0; i1 < 9; ++i1) ...`
/// pair repeated verbatim at the end of every `ContainerXxx` constructor.
fn push_player_main_and_hotbar(
    slots: &mut Vec<Slot>,
    left: i32,
    main_top: i32,
    row_height: i32,
    hotbar_y: i32,
) {
    for row in 0..3 {
        for col in 0..9 {
            push_slot(slots, left + col * 18, main_top + row * row_height);
        }
    }
    for col in 0..9 {
        push_slot(slots, left + col * 18, hotbar_y);
    }
}

/// The always-open player inventory screen. java: `ContainerPlayer`.
#[must_use]
pub fn player_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(45);
    // java: `ContainerPlayer.craftResult = new InventoryCraftResult()` (`:16`) behind a
    // `SlotCrafting` (`:26`) -- a whole-stack `decrStackSize`, not the furnace's split.
    push_slot_with(&mut slots, 144, 36, |s| s.with_role(SlotRole::CraftResult));
    for row in 0..2 {
        for col in 0..2 {
            push_slot(&mut slots, 88 + col * 18, 26 + row * 18);
        }
    }
    for armor_index in 0..4u8 {
        push_slot_with(&mut slots, 8, 8 + i32::from(armor_index) * 18, |s| {
            s.with_role(SlotRole::Armor(armor_index))
                .with_limit(1)
                // java: `ItemArmor.EMPTY_SLOT_NAMES[k]` (`ItemArmor.java:24`), returned by the
                // anonymous slot's `getSlotTexture()` (`ContainerPlayer.java:47-51`). These
                // four are the only `getSlotTexture` overrides in 1.8.9, and the sprites are
                // stitched into the block atlas by `ModelBakery.LOCATIONS_BUILTIN_TEXTURES`
                // even though no model references them.
                .with_background(ARMOR_SLOT_SPRITES[armor_index as usize])
        });
    }
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::Player, slots)
}

/// A crafting table. java: `ContainerWorkbench` -- craft result at slot `0`, the 3x3 matrix at
/// `1..=9` (row-major), then the player's 27 main + 9 hotbar. 46 slots.
///
/// Vanilla builds this **entirely client-side**: `EntityPlayerMP.displayGui` sends the
/// three-argument `S2DPacketOpenWindow`, so `slotCount` is `0` and `handleOpenWindow` takes its
/// `!packetIn.hasSlots()` arm into `EntityPlayerSP.displayGui`, which switches over exactly three
/// gui ids and news up the container itself.
#[must_use]
pub fn crafting_table_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(46);
    // java: `ContainerWorkbench.craftResult = new InventoryCraftResult()` (`:15`) behind a
    // `SlotCrafting` (`:25`).
    push_slot_with(&mut slots, 124, 35, |s| s.with_role(SlotRole::CraftResult));
    for row in 0..3 {
        for col in 0..3 {
            push_slot(&mut slots, 30 + col * 18, 17 + row * 18);
        }
    }
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::Workbench, slots)
}

/// A chest with `rows` rows of 9 (`rows * 9` slots). `rows == 3` is a single chest / the ender
/// chest, `rows == 6` is a double chest. java: `ContainerChest`.
#[must_use]
pub fn chest_container(window_id: u8, rows: u8) -> Container {
    let top = usize::from(rows) * 9;
    let mut slots = Vec::with_capacity(top + 36);
    for row in 0..rows {
        for col in 0..9 {
            push_slot(&mut slots, 8 + col * 18, 18 + i32::from(row) * 18);
        }
    }
    let extra = (i32::from(rows) - 4) * 18;
    push_player_main_and_hotbar(&mut slots, 8, 103 + extra, 18, 161 + extra);
    Container::new(window_id, ContainerKind::TopThenPlayer { top }, slots)
}

/// A hopper (5 slots) or a hopper-minecart (also 5). java: `ContainerHopper`.
#[must_use]
pub fn hopper_container(window_id: u8, slot_count: u8) -> Container {
    let top = usize::from(slot_count);
    let mut slots = Vec::with_capacity(top + 36);
    for i in 0..slot_count {
        push_slot(&mut slots, 44 + i32::from(i) * 18, 20);
    }
    push_player_main_and_hotbar(&mut slots, 8, 51, 18, 109);
    Container::new(window_id, ContainerKind::TopThenPlayer { top }, slots)
}

/// A dispenser or a dropper: identical 3x3-slot GUIs in vanilla (`ContainerDispenser` is used
/// for both — `TileEntityDropper` has no `Container` subclass of its own).
#[must_use]
pub fn dispenser_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(45);
    for row in 0..3 {
        for col in 0..3 {
            push_slot(&mut slots, 62 + col * 18, 17 + row * 18);
        }
    }
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::TopThenPlayer { top: 9 }, slots)
}

/// A furnace: input(0), fuel(1), output(2). java: `ContainerFurnace`.
#[must_use]
pub fn furnace_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(39);
    push_slot(&mut slots, 56, 17); // input
    push_slot_with(&mut slots, 56, 53, |s| s.with_role(SlotRole::FurnaceFuel));
    push_slot_with(&mut slots, 116, 35, |s| s.with_role(SlotRole::Output));
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::Furnace, slots)
}

/// A brewing stand: 3 potion-bottle slots then 1 ingredient slot. java:
/// `ContainerBrewingStand`.
#[must_use]
pub fn brewing_stand_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(40);
    push_slot_with(&mut slots, 56, 46, |s| {
        s.with_role(SlotRole::BrewBottle).with_limit(1)
    });
    push_slot_with(&mut slots, 79, 53, |s| {
        s.with_role(SlotRole::BrewBottle).with_limit(1)
    });
    push_slot_with(&mut slots, 102, 46, |s| {
        s.with_role(SlotRole::BrewBottle).with_limit(1)
    });
    push_slot_with(&mut slots, 79, 17, |s| {
        s.with_role(SlotRole::BrewIngredient)
    });
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::BrewingStand, slots)
}

/// An enchanting table: the item slot then the lapis slot. java: `ContainerEnchantment`.
#[must_use]
pub fn enchanting_table_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(38);
    push_slot_with(&mut slots, 15, 47, |s| s.with_limit(1)); // isItemValid always true (default)
    push_slot_with(&mut slots, 35, 47, |s| s.with_role(SlotRole::EnchantLapis));
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::Enchanting, slots)
}

/// An anvil: two input slots then the output slot. java: `ContainerRepair`.
#[must_use]
pub fn anvil_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(39);
    push_slot(&mut slots, 27, 47);
    push_slot(&mut slots, 76, 47);
    push_slot_with(&mut slots, 134, 47, |s| s.with_role(SlotRole::AnvilOutput));
    push_player_main_and_hotbar(&mut slots, 8, 84, 18, 142);
    Container::new(window_id, ContainerKind::Anvil, slots)
}

/// A beacon: the single payment slot. java: `ContainerBeacon`.
#[must_use]
pub fn beacon_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(37);
    push_slot_with(&mut slots, 136, 110, |s| {
        s.with_role(SlotRole::Beacon).with_limit(1)
    });
    push_player_main_and_hotbar(&mut slots, 36, 137, 18, 195);
    Container::new(window_id, ContainerKind::Beacon, slots)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn player_container_has_the_right_slot_count_and_shape() {
        let c = player_container(0);
        assert_eq!(c.slots().len(), 45);
        assert_eq!(c.slots()[0].role, SlotRole::CraftResult);
        assert_eq!(c.slots()[5].role, SlotRole::Armor(0));
        assert_eq!(c.slots()[8].role, SlotRole::Armor(3));
        assert_eq!(c.slots()[5].y, 8);
        assert_eq!(c.slots()[8].y, 62);
        assert_eq!(c.kind, ContainerKind::Player);
    }

    #[test]
    fn the_crafting_table_is_the_result_plus_a_three_by_three_plus_thirty_six() {
        let c = crafting_table_container(4);
        assert_eq!(c.slots().len(), 46);
        assert_eq!(c.kind, ContainerKind::Workbench);
        // java: `new SlotCrafting(..., 0, 124, 35)`, then `30 + j*18, 17 + i*18`.
        assert_eq!(c.slots()[0].role, SlotRole::CraftResult);
        assert_eq!((c.slots()[0].x, c.slots()[0].y), (124, 35));
        assert_eq!((c.slots()[1].x, c.slots()[1].y), (30, 17));
        assert_eq!((c.slots()[9].x, c.slots()[9].y), (30 + 2 * 18, 17 + 2 * 18));
        // `8 + i1*18, 84 + k*18` then `8 + l*18, 142`.
        assert_eq!((c.slots()[10].x, c.slots()[10].y), (8, 84));
        assert_eq!((c.slots()[37].x, c.slots()[37].y), (8, 142));
        for slot in &c.slots()[1..10] {
            assert_eq!(slot.role, SlotRole::Normal, "the 3x3 accepts anything");
        }
    }

    /// java: `ContainerChest`'s `numRows = getSizeInventory() / 9` with no floor, so a menu of
    /// 1..8 slots has **no** container row at all and the player half moves up accordingly.
    #[test]
    fn a_chest_of_fewer_than_nine_slots_has_no_container_row() {
        let c = chest_container(1, 0);
        assert_eq!(c.slots().len(), 36);
        assert_eq!(c.kind, ContainerKind::TopThenPlayer { top: 0 });
        // `i = (0 - 4) * 18 = -72`, so main starts at `103 - 72` and the hotbar at `161 - 72`.
        assert_eq!(c.slots()[0].y, 31);
        assert_eq!(c.slots()[27].y, 89);
    }

    #[test]
    fn single_chest_is_twenty_seven_plus_thirty_six() {
        let c = chest_container(1, 3);
        assert_eq!(c.slots().len(), 27 + 36);
        assert_eq!(c.kind, ContainerKind::TopThenPlayer { top: 27 });
        // java: `i = (numRows - 4) * 18` is *negative* for a 3-row chest (-18), so the player
        // section sits 18px higher than the `161`/`103` baseline a 4-row chest would use.
        assert_eq!(c.slots()[27 + 27].y, 143);
    }

    #[test]
    fn double_chest_is_fifty_four_plus_thirty_six() {
        // The production open-window path computes six rows from the server's 54 slots and
        // calls this same generic builder; there is deliberately no special double-chest path.
        let c = chest_container(2, 6);
        assert_eq!(c.slots().len(), 54 + 36);
        assert_eq!(c.kind, ContainerKind::TopThenPlayer { top: 54 });
    }

    #[test]
    fn hopper_defaults_to_five_slots() {
        let c = hopper_container(3, 5);
        assert_eq!(c.slots().len(), 5 + 36);
        assert_eq!(c.kind, ContainerKind::TopThenPlayer { top: 5 });
    }

    #[test]
    fn dispenser_and_dropper_share_the_nine_slot_shape() {
        let c = dispenser_container(4);
        assert_eq!(c.slots().len(), 9 + 36);
        assert_eq!(c.kind, ContainerKind::TopThenPlayer { top: 9 });
    }

    #[test]
    fn furnace_has_the_three_special_slots_in_order() {
        let c = furnace_container(5);
        assert_eq!(c.slots().len(), 39);
        assert_eq!(c.slots()[1].role, SlotRole::FurnaceFuel);
        assert_eq!(c.slots()[2].role, SlotRole::Output);
    }

    #[test]
    fn brewing_stand_has_three_bottle_slots_and_one_ingredient_slot() {
        let c = brewing_stand_container(6);
        assert_eq!(c.slots().len(), 40);
        for i in 0..3 {
            assert_eq!(c.slots()[i].role, SlotRole::BrewBottle);
            assert_eq!(c.slots()[i].stack_limit, Some(1));
        }
        assert_eq!(c.slots()[3].role, SlotRole::BrewIngredient);
    }

    #[test]
    fn enchanting_table_has_the_lapis_slot_second() {
        let c = enchanting_table_container(7);
        assert_eq!(c.slots().len(), 38);
        assert_eq!(c.slots()[0].stack_limit, Some(1));
        assert_eq!(c.slots()[1].role, SlotRole::EnchantLapis);
    }

    #[test]
    fn anvil_output_is_the_third_slot() {
        let c = anvil_container(8);
        assert_eq!(c.slots().len(), 39);
        assert_eq!(c.slots()[2].role, SlotRole::AnvilOutput);
    }

    #[test]
    fn beacon_is_one_payment_slot_plus_the_full_player_inventory() {
        let c = beacon_container(9);
        assert_eq!(c.slots().len(), 37);
        assert_eq!(c.slots()[0].role, SlotRole::Beacon);
        assert_eq!(c.slots()[0].stack_limit, Some(1));
    }
}
