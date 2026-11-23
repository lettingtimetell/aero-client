//! Inventory and container screens — S17. java inputs: `inventory/` (`Container.java`,
//! `Slot.java`, `ContainerPlayer` and friends, 30 files/5,221 lines) and `client/gui/inventory/`.
//!
//! # Shape
//!
//! ```text
//!   click (mouse/key) ─┬─> Container::click ──> predicts locally, builds C0EPacketClickWindow
//!                       │        (model.rs: slot_click, H10's drag state machine)
//!                       │
//!   S32PacketConfirmTransaction ─> Container::handle_confirm_transaction ──> ack packet
//!   S2FPacketSetSlot / S30PacketWindowItems ─> Container::apply_set_slot / apply_window_items
//!                       (net.rs — this pair *is* the rollback; see its module doc)
//! ```
//!
//! * [`slot`] — [`slot::Slot`], [`slot::SlotRole`]: display position plus validity/limit rules.
//! * [`model`] — [`model::Container`]: the click state machine (all seven `ClickWindow` modes,
//!   including drag's three phases) and the generic `mergeItemStack` shift-click helper.
//! * [`layouts`] — one builder function per vanilla container shape (player inventory, chest/
//!   double-chest/ender-chest/hopper/dispenser/dropper, furnace, brewing stand, enchanting
//!   table, anvil, beacon), each with the exact slot positions vanilla's Java constructors use.
//! * [`net`] — turning a click into a packet, and applying the server's replies back.
//! * [`interaction`] — desktop mouse events (click/drag/double-click) to `(slot_id,
//!   clicked_button, mode)` triples ready for [`model::Container::click`]; the producer side of
//!   the drag protocol, where [`model`] is the consumer.
//! * [`creative`] — creative-mode tabs, search, and its distinct (non-`slot_click`) click rules.
//! * [`creative_screen`] — [`creative_screen::CreativeScreen`], the on-screen creative
//!   inventory: the `creative_inventory/tabs.png` tab strip, the search box, the scrollbar and
//!   the scrolling item grid.
//! * [`tooltip`] — item tooltip text: name, enchantments, lore, durability, attributes.
//! * [`tooltip_box`] — the popup box vanilla draws around that text (`GuiScreen`'s
//!   `drawHoveringText` border gradient), plus its placement arithmetic.
//! * [`screen`] — [`screen::ContainerScreen`], the `Screen`-trait GUI: panel art per container
//!   kind, the slot-hover highlight, dynamic art (furnace fire, brewing bubbles, enchanting
//!   plates, the anvil cross, the beacon's buttons), and mouse/key routing into the click
//!   protocol.
//! * [`render`] — item rendering in slots: flat 2D icons, a simple isometric 3D path for block
//!   items, stack count, and the durability bar.
//! * [`player_preview`] — [`player_preview::PlayerPreviewRenderer`], the rotating 3-D player
//!   preview both [`screen::ContainerScreen`] (Inventory panel) and [`creative_screen
//!   ::CreativeScreen`] (Inventory tab) embed: `mc-render`'s biped model, CPU-projected into this
//!   crate's own [`crate::widget::DrawList`] under vanilla's `GuiInventory.drawEntityOnScreen`
//!   math.
//! * [`stack_ext`] — small `ItemStack` helper functions this crate needs but `mc-protocol`
//!   deliberately does not carry (it has no `mc-data` dependency).
//!
//! # Hazard H10
//!
//! Drag-click (`ClickWindow` mode `5`) is a three-phase protocol exchange — start, add-slot
//! (repeated once per slot painted over), end — encoded across *multiple* `ClickWindow`
//! packets that share one drag "session"; approximating it with a single packet or the wrong
//! phase transitions produces ghost items that only show up under real network lag, not in a
//! same-machine smoke test. [`model::Container::slot_click`]'s `mode == 5` arm is a
//! line-by-line port of `Container.java:slotClick`'s own mode-5 branch for exactly this reason;
//! see [`model`]'s module doc and its own tests for the phase transitions this covers.

pub mod creative;
// The creative click oracle's replay (test only).
#[cfg(test)]
mod creative_oracle;
pub mod creative_screen;
pub mod interaction;
pub mod layouts;
pub mod model;
pub mod net;
pub mod player_preview;
pub mod render;
pub mod screen;
pub mod slot;
pub mod stack_ext;
pub mod tooltip;
pub mod tooltip_box;
// The `scrollable_tooltips` mod
pub mod tooltip_scroll;

pub use creative_screen::CreativeScreen;
pub use interaction::{MouseState, SlotTarget};
pub use layouts::player_container;
pub use model::{Container, ContainerKind};
pub use net::container_for_open_window;
pub use player_preview::{HorsePreviewSnapshot, PlayerPreviewRenderer, PlayerPreviewSnapshot};
pub use screen::{default_atlas_texture, ContainerArt, ContainerBindings, ContainerScreen};
pub use slot::{Slot, SlotRole};
pub use stack_ext::ItemStack;
