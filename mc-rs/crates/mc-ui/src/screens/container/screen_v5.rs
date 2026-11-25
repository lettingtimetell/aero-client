//! The 1.7.10 halves of the container screens. A child of `screen`, for [`ContainerScreen`]'s
//! private state; `screen.rs` reaches each function from one line that does nothing for a 1.8.9
//! container.
//!
//! - [`ContainerScreen::for_open_window_v5`]: the screen a 1.7.10 window opens (the 1.8.9 screen
//!   for its family, with the 1.7.10 container and the anvil's 40-character name field).
//! - The enchanting table: `GuiEnchantment.mouseClicked` sends `C11` when
//!   `ContainerEnchantment.enchantItem` passes, which in 1.7.10 needs no lapis and only
//!   `experienceLevel >= cost` (or creative); the plates are lit by the same rule and carry no
//!   lapis icons. (No hover tooltip either: it needs the clue properties 4-6, which a 1.7.10
//!   server never sends.)
//! - The furnace: window properties 0, 1 and 2 are the cook time, the burn time and the
//!   current item's burn time, and a smelt always takes 200 ticks.
//! - Drags: the preview splits by the mode the drag really uses (`interaction.rs`).
//!
//! java17: `client/gui/GuiEnchantment.java`, `client/gui/GuiRepair.java:45`,
//! `inventory/ContainerFurnace.java:76-92`, `tileentity/TileEntityFurnace.java:169-182`.

use mc_protocol::play::s2c;

use super::{ContainerArt, ContainerScreen, EnchantingArt};
use crate::font::{TextMesh, TextSink};
use crate::screens::container::model::{Container, ContainerKind};
use crate::screens::container::v5::container_for_open_window_v5;
use crate::widget::DrawList;

/// `GuiRepair.initGui`'s `nameField.setMaxStringLength(40)` (1.8.9: 30).
const ANVIL_NAME_LENGTH_V5: i32 = 40;

/// `TileEntityFurnace.getCookProgressScaled`'s fixed smelt time.
const SMELT_TICKS_V5: i32 = 200;

impl ContainerScreen {
    /// The screen a 1.7.10 window opens: [`ContainerScreen::for_open_window`]'s screen for the
    /// window's family (panel art, title, the anvil's name field), holding the 1.7.10 container
    /// ([`container_for_open_window_v5`]). `pkt` is the window as
    /// [`crate::screens::container::v5::open_window_v47`] lowered it.
    #[must_use]
    pub fn for_open_window_v5(pkt: &s2c::OpenWindow, title: impl Into<String>) -> Option<Self> {
        let container = container_for_open_window_v5(pkt)?;
        let mut screen = Self::for_open_window(pkt, title)?;
        screen.container = container;
        if let Some(field) = screen.anvil_name.as_mut() {
            field.set_max_length(ANVIL_NAME_LENGTH_V5);
        }
        if let ContainerArt::Furnace(furnace) = &mut screen.art {
            furnace.total_cook_time = SMELT_TICKS_V5;
        }
        Some(screen)
    }

    /// java17 `ContainerEnchantment.enchantItem` (`inventory/ContainerEnchantment.java:177-181`)
    /// for the plate `row`: `Some` of whether `C11` is sent on a 1.7.10 table, `None` otherwise.
    pub(super) fn enchant_gate_v5(&self, row: usize) -> Option<bool> {
        if self.container.kind != ContainerKind::EnchantingV5 {
            return None;
        }
        let ContainerArt::Enchanting(art) = &self.art else {
            return Some(false);
        };
        Some(art.can_activate_v5(row, self.container.slot_stack(0).is_some(), self.creative))
    }

    /// java17 `InventoryEffectRenderer.initGui` shifts the inventory right for active potion
    /// effects once, when the screen is laid out, and never again while it is open (1.8.9 moves
    /// it every tick as effects come and go). A 1.7.10 container keeps the left edge its last
    /// layout chose ([`super::super::v5::InitShiftV5`]: a new size, or the
    /// `centered_inventory` mod's centring changing, which so applies at once as in 1.8.9).
    pub(super) fn keep_init_gui_left_v5(&mut self) {
        if !self.container.game_version().is_1_7_10() {
            return;
        }
        self.gui_left = self
            .init_shift_v5
            .left(self.gui_left, self.effects_centered);
    }

    /// A 1.7.10 drag previews with the limit `MouseState` really kept for it.
    pub(super) fn sync_drag_preview_v5(&mut self) {
        if !self.container.game_version().is_1_7_10() {
            return;
        }
        let limit = self.mouse.drag_splitting_limit();
        if let Some(preview) = self.drag_preview.as_mut() {
            preview.mode = limit;
        }
    }

    /// java17 `ContainerFurnace.updateProgressBar`: `true` when a 1.7.10 furnace took the
    /// property (0 cook time, 1 burn time, 2 the current item's burn time).
    pub(super) fn apply_window_property_v5(&mut self, pkt: &s2c::WindowProperty) -> bool {
        if pkt.window_id != self.container.window_id || !self.container.game_version().is_1_7_10() {
            return false;
        }
        let ContainerArt::Furnace(furnace) = &mut self.art else {
            return false;
        };
        let value = i32::from(pkt.var_value);
        match pkt.var_index {
            0 => furnace.cook_time = value,
            1 => furnace.burn_time = value,
            2 => furnace.current_item_burn_time = value,
            _ => {}
        }
        furnace.total_cook_time = SMELT_TICKS_V5;
        true
    }

    /// A 1.7.10 merchant screen hands its container the whole offer list and the selected
    /// offer: 1.7.10 matches the inputs against every offer while the first is selected
    /// (java17 `MerchantRecipeList.canRecipeBeUsed`, `model_v5.rs`). `false` for any other
    /// screen, which hands over the selected offer itself.
    pub(super) fn sync_merchant_offers_v5(&mut self) -> bool {
        if !self.container.game_version().is_1_7_10() {
            return false;
        }
        let ContainerArt::Merchant(merchant) = &self.art else {
            return false;
        };
        self.container
            .set_merchant_offers_v5(&merchant.recipes, merchant.selected);
        true
    }

    /// A new `MC|TrList` on a 1.7.10 screen: java17 only stores the list
    /// (`NetHandlerPlayClient.handleCustomPayload:1407-1421`), so `GuiMerchant` keeps the page
    /// `page` (only its arrows change it, `actionPerformed:70-87`) and the container keeps its
    /// `currentRecipeIndex` and output (`model_v5.rs`). A page past a shorter list becomes the
    /// first: java17's screen would throw there (`drawGuiContainerBackgroundLayer:117-118` reads
    /// that index from the list), and the first page matches the same offers
    /// (`MerchantRecipeList.canRecipeBeUsed:23` scans the whole list for both). `false` for any
    /// other screen, which starts again at the first page.
    pub(super) fn apply_merchant_list_v5(&mut self, page: usize) -> bool {
        if !self.container.game_version().is_1_7_10() {
            return false;
        }
        let ContainerArt::Merchant(merchant) = &mut self.art else {
            return false;
        };
        merchant.selected = if page < merchant.recipes.len() {
            page
        } else {
            0
        };
        self.container
            .receive_merchant_offers_v5(&merchant.recipes, merchant.selected);
        true
    }
}

impl EnchantingArt {
    /// java17 `ContainerEnchantment.enchantItem`: an offer, an item in the slot, and the
    /// player's level at least the cost (or creative). No lapis, and no 1.8.9
    /// `experienceLevel >= row + 1` rule.
    #[must_use]
    pub fn can_activate_v5(&self, index: usize, has_item: bool, creative: bool) -> bool {
        let Some(&level) = self.enchant_levels.get(index) else {
            return false;
        };
        level > 0 && has_item && (self.player_level >= level || creative)
    }
}

/// The lapis count the 1.8.9 offer rules see. A 1.7.10 table has no lapis, and with an
/// unlimited count those rules (`is_affordable`) are exactly 1.7.10's `experienceLevel >= cost`
/// (java17 `GuiEnchantment.drawGuiContainerBackgroundLayer:152`).
pub(super) fn offer_lapis_v5(container: &Container, lapis: i32) -> i32 {
    if container.kind == ContainerKind::EnchantingV5 {
        i32::MAX
    } else {
        lapis
    }
}

impl ContainerArt {
    /// java17 `GuiEnchantment.drawGuiContainerBackgroundLayer`'s three plates (`:142-185`): a
    /// dark plate for no offer or one the player cannot afford, the lit or hovered plate
    /// otherwise, and no lapis icons. `true` when the container is a 1.7.10 table.
    pub(super) fn draw_enchanting_offers_v5(
        &self,
        d: &mut DrawList,
        (left, top): (i32, i32),
        container: &Container,
        hovered_enchant_plate: Option<usize>,
        mut runes: Option<&mut [TextMesh; 3]>,
    ) -> bool {
        if container.kind != ContainerKind::EnchantingV5 {
            return false;
        }
        let ContainerArt::Enchanting(e) = self else {
            return true;
        };
        let tex = self.texture();
        for row in 0..3usize {
            let x = left + 60;
            let y = top + 14 + 19 * row as i32;
            let level = e.enchant_levels[row];
            let plate_v = if level == 0 || !e.is_affordable(row, i32::MAX) {
                185
            } else if hovered_enchant_plate == Some(row) {
                204
            } else {
                166
            };
            d.blit(&tex, x, y, 0, plate_v, 108, 19);
            if level == 0 {
                continue;
            }
            if let Some(rows) = runes.as_deref_mut() {
                let mesh = std::mem::take(&mut rows[row]);
                for glyph in mesh.glyphs {
                    d.glyph(glyph);
                }
                for bar in mesh.bars {
                    d.bar(bar);
                }
            }
        }
        true
    }
}

#[cfg(test)]
#[path = "screen_v5_tests.rs"]
mod tests;
