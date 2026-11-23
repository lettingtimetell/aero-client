//! Desktop mouse → click-protocol translation. java: `GuiContainer.java`'s `mouseClicked`,
//! `mouseClickMove`, and `mouseReleased`.
//!
//! [`super::model::Container::slot_click`] (and [`super::model::Container::click`], which wraps
//! it with the transaction id and packet) consume `(slot_id, clicked_button, mode)` triples.
//! Producing the *right* triples — and the right *sequence* of them for a drag — from raw mouse
//! button/position events is a separate, stateful job in vanilla (`GuiContainer` keeps a dozen
//! fields across calls: `dragSplitting`, `lastClickTime`, `doubleClick`, ...), which is what
//! [`MouseState`] reproduces here.
//!
//! # Scope: touchscreen is not reproduced
//!
//! Vanilla's `GuiContainer` also carries a whole parallel touchscreen drag-and-drop path
//! (`clickedSlot`/`draggedStack`/`returningStack`, gated on `gameSettings.touchscreen`) for
//! picking a stack up with one tap and dropping it with another. 1.8.9's PC client defaults
//! this off and it is not part of a normal desktop session; not reproduced.
//!
//! # Drag-splitting's "add" filter is part of the protocol, not a preview optimisation
//!
//! Java's `mouseClickMove` (`:505`) adds a hovered slot to `dragSplittingSlots` only when
//! **all** of `itemstack.stackSize > dragSplittingSlots.size()`, `Container.canAddItemToSlot`,
//! `Slot.isItemValid` and `canDragIntoSlot` hold. An earlier revision of this doc argued the
//! last three could be skipped here because [`super::model::Container::slot_click`]'s own
//! `mode == 5` "add" arm re-validates, so "the *protocol outcome* is identical either way".
//! **That was wrong.** `mouseReleased` emits one mode-5 `C0EPacketClickWindow` *per member of
//! the set*, so an unfiltered set puts extra packets on the wire naming slots the item provably
//! cannot enter, and each one burns a transaction id. The surviving `stackSize > size()` budget
//! is spent on them too, so the stack also ends up split across fewer slots than the player
//! painted. [`MouseState::drag_move`] therefore applies
//! [`super::model::Container::slot_accepts_for_drag`], which is exactly those three checks.
//!
//! # Clicks are applied where they are produced
//!
//! Java's `mouseClicked`/`mouseReleased` call `handleMouseClick` — and therefore
//! `Container.slotClick` — *inline*, so everything after the call sees the mutated container.
//! Two places depend on that and used to be evaluated against a pre-click snapshot here:
//! `mouseReleased`'s shift-double-click sweep (`:539-545`), whose per-slot tests are re-run
//! after each click it just made, and its tail (`:646`), which reads the cursor left behind.
//! [`MouseState::mouse_released`] therefore takes the container by `&mut` and returns finished
//! packets rather than click triples.

use std::collections::BTreeSet;

use mc_protocol::play::c2s;

use super::model::Container;
use super::stack_ext::ItemStack;

/// `(slot_id, clicked_button, mode)`, ready for [`super::model::Container::click`]. `slot_id`
/// carries the `-999`/`-1` sentinels exactly as [`Container::slot_click`] expects.
pub type ClickEvent = (i32, u8, u8);

/// Mouse buttons, matching LWJGL/vanilla's convention: `0` left, `1` right, `2` middle.
pub const LEFT: i32 = 0;
/// Right mouse button.
pub const RIGHT: i32 = 1;

/// How long a second click on the same slot/button still counts as a double-click, in
/// milliseconds. java: `GuiContainer.mouseClicked`'s `250L`.
pub const DOUBLE_CLICK_MS: u64 = 250;

/// Where the mouse is, resolved against the open window's slots.
///
/// Vanilla computes this as two independent facts and folds them into one `int l`
/// (`GuiContainer.java:372`): `slot == null ? -1 : slot.slotNumber`, then `-999` if the point is
/// outside the window rectangle entirely. Those three states are **not** interchangeable — the
/// `-1` case (inside the panel, but on its artwork rather than on a slot) is the one vanilla
/// silently drops on press and still sends on release — so this enum keeps all three.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotTarget {
    /// Hovering a real slot.
    Slot(u16),
    /// Inside the window rectangle but not over a slot — vanilla's `-1`.
    Inside,
    /// Off the window rectangle entirely — vanilla's `-999`.
    Outside,
}

impl SlotTarget {
    fn wire_id(self) -> i32 {
        match self {
            SlotTarget::Slot(id) => i32::from(id),
            SlotTarget::Inside => -1,
            SlotTarget::Outside => -999,
        }
    }

    fn slot_id(self) -> Option<u16> {
        match self {
            SlotTarget::Slot(id) => Some(id),
            SlotTarget::Inside | SlotTarget::Outside => None,
        }
    }
}

/// Per-window mouse state `GuiContainer` keeps across calls. One instance per open container;
/// discard and recreate it when the window changes (matches a fresh `GuiContainer` instance).
#[derive(Debug, Clone, PartialEq)]
pub struct MouseState {
    drag_splitting: bool,
    drag_splitting_slots: BTreeSet<u16>,
    drag_splitting_button: i32,
    drag_splitting_limit: i32,
    ignore_mouse_up: bool,
    last_click_time_ms: u64,
    // Java stores `Slot lastClickSlot`, not the richer hit-test result. Both a point on the
    // panel artwork and a point outside the panel therefore store the same `null` identity.
    last_click_slot: Option<u16>,
    last_click_button: i32,
    double_click: bool,
    shift_clicked_slot: Option<ItemStack>,
}

/// java: `GuiContainer`'s constructor, which is `{ this.inventorySlots = …; this.ignoreMouseUp
/// = true; }`. **`ignoreMouseUp` starts `true`**, so the first release a freshly-opened container
/// screen sees is swallowed -- which matters whenever the gesture that opened the window is
/// itself a click, because the release then lands on the new screen. `#[derive(Default)]` gave
/// `false` and let that release through as a real `C0EPacketClickWindow`.
impl Default for MouseState {
    fn default() -> Self {
        MouseState {
            drag_splitting: false,
            drag_splitting_slots: BTreeSet::new(),
            drag_splitting_button: 0,
            drag_splitting_limit: 0,
            ignore_mouse_up: true,
            last_click_time_ms: 0,
            last_click_slot: None,
            last_click_button: 0,
            double_click: false,
            shift_clicked_slot: None,
        }
    }
}

impl MouseState {
    /// A fresh state, as when a window is first opened.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// java: `GuiContainer.mouseClicked`. `pick_block` is whether the button pressed is bound
    /// to vanilla's "pick block" action (middle-click by default); `shift` is whether shift is
    /// currently held. Returns the one click event this press produces immediately, or `None`
    /// when it only starts accumulating a drag (nothing is sent until release).
    pub fn mouse_clicked(
        &mut self,
        container: &Container,
        target: SlotTarget,
        button: i32,
        pick_block: bool,
        shift: bool,
        now_ms: u64,
    ) -> Option<ClickEvent> {
        self.double_click = self.last_click_slot == target.slot_id()
            && now_ms.saturating_sub(self.last_click_time_ms) < DOUBLE_CLICK_MS
            && self.last_click_button == button;
        self.ignore_mouse_up = false;

        let mut event = None;
        // java: `if (l != -1)` — a press on the panel's own artwork does nothing at all, but
        // still updates the last-click bookkeeping below (the three assignments at the very end
        // of `mouseClicked` sit outside every branch).
        if (button == LEFT || button == RIGHT || pick_block) && target != SlotTarget::Inside {
            let slot_id = target.wire_id();
            if !self.drag_splitting {
                if container.cursor().is_none() {
                    event = Some(if pick_block {
                        (slot_id, button as u8, 3)
                    } else {
                        let shift_click = slot_id != -999 && shift;
                        let mode = if shift_click {
                            self.shift_clicked_slot = target
                                .slot_id()
                                .and_then(|s| container.slot_stack(s as usize).cloned());
                            1
                        } else if slot_id == -999 {
                            4
                        } else {
                            0
                        };
                        (slot_id, button as u8, mode)
                    });
                    self.ignore_mouse_up = true;
                } else {
                    self.drag_splitting = true;
                    self.drag_splitting_button = button;
                    self.drag_splitting_slots.clear();
                    self.drag_splitting_limit = if button == LEFT {
                        0
                    } else if button == RIGHT {
                        1
                    } else {
                        2
                    };
                }
            }
        }

        self.last_click_slot = target.slot_id();
        self.last_click_time_ms = now_ms;
        self.last_click_button = button;
        event
    }

    /// java: `GuiContainer.mouseClickMove` (its non-touchscreen `dragSplitting` branch). Call
    /// while a drag button stays held and the mouse moves onto a new slot.
    ///
    /// The four conditions are transcribed in Java's own left-to-right order from `:505`; see
    /// this module's doc for why the last three are load-bearing on the wire.
    pub fn drag_move(&mut self, container: &Container, target: SlotTarget) {
        if !self.drag_splitting {
            return;
        }
        let Some(slot_id) = target.slot_id() else {
            return;
        };
        let Some(cursor) = container.cursor() else {
            return;
        };
        if cursor.count > self.drag_splitting_slots.len() as i32
            && container.slot_accepts_for_drag(slot_id, cursor)
        {
            self.drag_splitting_slots.insert(slot_id);
        }
    }

    /// java: `GuiContainer.mouseReleased`. `pick_block` is whether `button` is bound to
    /// vanilla's "pick block" action, exactly like [`Self::mouse_clicked`]'s parameter.
    ///
    /// Each click is applied to `container` as it is produced — java calls `handleMouseClick`
    /// inline — and the finished `C0EPacketClickWindow`s are returned in wire order: none for
    /// an ignored release, one for an ordinary click, one per swept slot for a
    /// shift-double-click, or exactly `drag_splitting_slots.len() + 2` for a completed drag
    /// (start, one "add" per slot, end).
    pub fn mouse_released(
        &mut self,
        container: &mut Container,
        target: SlotTarget,
        button: i32,
        pick_block: bool,
        shift: bool,
        creative: bool,
    ) -> Vec<c2s::ClickWindow> {
        let mut packets = Vec::new();
        self.mouse_released_with(
            container,
            target,
            button,
            pick_block,
            shift,
            creative,
            &mut |_, packet| packets.push(packet),
        );
        packets
    }

    /// [`Self::mouse_released`], handing each click to `on_click` the moment it
    /// has been applied, with the container in the state that click left it in. java runs
    /// `detectAndSendChanges` (and so `GuiRepair.renameItem`) inside each `slotClick`, so a
    /// caller that has to put a packet *before* one click's C0E needs exactly this view.
    #[allow(clippy::too_many_arguments)]
    pub fn mouse_released_with(
        &mut self,
        container: &mut Container,
        target: SlotTarget,
        button: i32,
        pick_block: bool,
        shift: bool,
        creative: bool,
        on_click: &mut dyn FnMut(&Container, c2s::ClickWindow),
    ) {
        let slot_id = target.wire_id();

        if self.double_click
            && button == LEFT
            && target
                .slot_id()
                .is_some_and(|s| container.can_merge_slot(s))
        {
            if shift {
                if let (Some(origin), Some(held)) =
                    (target.slot_id(), self.shift_clicked_slot.clone())
                {
                    // java: `for (Slot slot2 : this.inventorySlots.inventorySlots)` — the whole
                    // slot list, **including the clicked slot itself**, with `handleMouseClick`
                    // called inside the loop so each test sees the state the previous click
                    // left. `slot2 != slot` is not one of the conditions.
                    let ids: Vec<u16> = container.slots().iter().map(|slot| slot.id).collect();
                    for id in ids {
                        let index = usize::from(id);
                        if container.can_take_stack(index, creative)
                            && container.slot_stack(index).is_some()
                            && container.same_backing_region(id, origin)
                            && container.slot_stack_would_accept(id, &held)
                        {
                            apply_click(
                                container,
                                on_click,
                                i32::from(id),
                                button as u8,
                                1,
                                creative,
                            );
                        }
                    }
                }
            } else {
                apply_click(container, on_click, slot_id, button as u8, 6, creative);
            }
            self.double_click = false;
            self.last_click_time_ms = 0;
        } else {
            // java `:558-564`: a hard `return`, so the `:646-651` tail below is skipped.
            if self.drag_splitting && self.drag_splitting_button != button {
                self.drag_splitting = false;
                self.drag_splitting_slots.clear();
                self.ignore_mouse_up = true;
                return;
            }

            // java `:566-570`: also a hard `return`. Preserving `last_click_time_ms` here is
            // what makes a second click within 250 ms a double-click at all after a press that
            // dispatched its click immediately — every shift-click, every click on an empty
            // slot, every failed pickup.
            if self.ignore_mouse_up {
                self.ignore_mouse_up = false;
                return;
            }

            if self.drag_splitting && !self.drag_splitting_slots.is_empty() {
                let limit = self.drag_splitting_limit;
                apply_click(
                    container,
                    on_click,
                    -999,
                    encode_drag_button(0, limit),
                    5,
                    creative,
                );
                for sid in self.drag_splitting_slots.clone() {
                    let add = encode_drag_button(1, limit);
                    apply_click(container, on_click, i32::from(sid), add, 5, creative);
                }
                apply_click(
                    container,
                    on_click,
                    -999,
                    encode_drag_button(2, limit),
                    5,
                    creative,
                );
            } else if container.cursor().is_some() {
                let mode = if pick_block {
                    3
                } else {
                    let shift_click = slot_id != -999 && shift;
                    if shift_click {
                        self.shift_clicked_slot = target
                            .slot_id()
                            .and_then(|s| container.slot_stack(usize::from(s)).cloned());
                        1
                    } else {
                        0
                    }
                };
                apply_click(container, on_click, slot_id, button as u8, mode, creative);
            }
        }

        // java `:646-651`, reached only when neither `return` above fired, and reading the
        // cursor **after** the clicks were applied.
        if container.cursor().is_none() {
            self.last_click_time_ms = 0;
        }
        self.drag_splitting = false;
    }
}

/// Apply one click (java: `handleMouseClick`, inline) and hand its packet to
/// `on_click` with the container in the state that click left it in.
fn apply_click(
    container: &mut Container,
    on_click: &mut dyn FnMut(&Container, c2s::ClickWindow),
    slot_id: i32,
    button: u8,
    mode: u8,
    creative: bool,
) {
    let packet = container.click(slot_id, button, mode, creative);
    on_click(container, packet);
}

/// java: `Container.func_94534_d(event, dragMode)`.
fn encode_drag_button(event: u8, drag_mode: i32) -> u8 {
    super::model::encode_drag(event, drag_mode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::container::layouts::{anvil_container, chest_container, player_container};

    fn stack(item_id: i16, count: i32) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage: 0,
            nbt: None,
        }
    }

    /// `(slot_id, used_button, mode)` for each packet, which is what every assertion below is
    /// actually about.
    fn triples(packets: &[c2s::ClickWindow]) -> Vec<(i32, u8, u8)> {
        packets
            .iter()
            .map(|p| (i32::from(p.slot_id), p.used_button, p.mode))
            .collect()
    }

    #[test]
    fn a_plain_left_click_produces_mode_zero() {
        let c = chest_container(0, 3);
        let mut m = MouseState::new();
        let ev = m.mouse_clicked(&c, SlotTarget::Slot(5), LEFT, false, false, 0);
        assert_eq!(ev, Some((5, 0, 0)));
    }

    #[test]
    fn shift_click_produces_mode_one() {
        let c = chest_container(0, 3);
        let mut m = MouseState::new();
        let ev = m.mouse_clicked(&c, SlotTarget::Slot(5), LEFT, false, true, 0);
        assert_eq!(ev, Some((5, 0, 1)));
    }

    #[test]
    fn clicking_outside_with_an_empty_cursor_produces_mode_four_and_the_outside_sentinel() {
        let c = chest_container(0, 3);
        let mut m = MouseState::new();
        let ev = m.mouse_clicked(&c, SlotTarget::Outside, LEFT, false, false, 0);
        assert_eq!(ev, Some((-999, 0, 4)));
    }

    #[test]
    fn pick_block_click_produces_mode_three() {
        let c = chest_container(0, 3);
        let mut m = MouseState::new();
        let ev = m.mouse_clicked(&c, SlotTarget::Slot(2), LEFT, true, false, 0);
        assert_eq!(ev, Some((2, 0, 3)));
    }

    #[test]
    fn clicking_while_holding_an_item_starts_a_drag_instead_of_an_immediate_click() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 10)));
        let mut m = MouseState::new();
        let ev = m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        assert_eq!(
            ev, None,
            "no click event yet — only mouseReleased flushes a drag"
        );
    }

    #[test]
    fn a_full_drag_lifecycle_produces_start_add_add_end_in_order() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 10)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        m.drag_move(&c, SlotTarget::Slot(1));
        m.drag_move(&c, SlotTarget::Slot(2));
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(2), LEFT, false, false, false));
        assert_eq!(events.len(), 4, "{events:?}");
        assert_eq!(events[0], (-999, encode_drag_button(0, 0), 5), "start");
        assert_eq!(events[1], (1, encode_drag_button(1, 0), 5), "add slot 1");
        assert_eq!(events[2], (2, encode_drag_button(1, 0), 5), "add slot 2");
        assert_eq!(events[3], (-999, encode_drag_button(2, 0), 5), "end");
    }

    #[test]
    fn feeding_the_drag_events_into_the_real_container_actually_splits_the_stack() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 10)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        // A real drag's very first `mouseClickMove` still fires over the press slot itself
        // (mouse movement is polled far more finely than "moved to a different slot") before
        // the pointer leaves it, so the starting slot is a drag-splitting target too.
        m.drag_move(&c, SlotTarget::Slot(0));
        m.drag_move(&c, SlotTarget::Slot(1));
        m.mouse_released(&mut c, SlotTarget::Slot(1), LEFT, false, false, false);
        assert_eq!(c.slot_stack(0), Some(&stack(1, 5)));
        assert_eq!(c.slot_stack(1), Some(&stack(1, 5)));
        assert_eq!(c.cursor(), None);
    }

    #[test]
    fn releasing_on_a_different_button_than_the_drag_started_with_cancels_it() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 10)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        m.drag_move(&c, SlotTarget::Slot(1));
        let events = m.mouse_released(&mut c, SlotTarget::Slot(1), RIGHT, false, false, false);
        assert!(
            events.is_empty(),
            "a mismatched release must cancel, not flush, the drag"
        );
    }

    #[test]
    fn a_second_click_on_the_same_slot_within_the_window_is_a_double_click() {
        let mut c = chest_container(0, 3);
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 0);
        m.mouse_released(&mut c, SlotTarget::Slot(3), LEFT, false, false, false);
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 100);
        assert!(m.double_click);
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(3), LEFT, false, false, false));
        assert_eq!(events, vec![(3, 0, 6)], "double click collect");
    }

    /// java: `mouseClicked`'s `if (l != -1)` gate. A press on the panel artwork (inside the
    /// window, off every slot) produces nothing — unlike a press on the *outside*, which is the
    /// "throw the cursor stack on the floor" click.
    #[test]
    fn pressing_the_panel_artwork_produces_no_click_but_outside_produces_mode_four() {
        let c = chest_container(0, 3);
        let mut m = MouseState::new();
        assert_eq!(
            m.mouse_clicked(&c, SlotTarget::Inside, LEFT, false, false, 0),
            None
        );
        assert_eq!(
            m.mouse_clicked(&c, SlotTarget::Outside, LEFT, false, false, 0),
            Some((-999, 0, 4))
        );
    }

    /// The press is dropped, but the bookkeeping is not: vanilla's `lastClickSlot = slot`
    /// (a `null` slot here) runs unconditionally, so a later click on a real slot cannot
    /// double-click against the click that landed on the artwork.
    #[test]
    fn a_press_on_the_panel_artwork_still_resets_the_double_click_tracking() {
        let mut c = chest_container(0, 3);
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 0);
        m.mouse_released(&mut c, SlotTarget::Slot(3), LEFT, false, false, false);
        m.mouse_clicked(&c, SlotTarget::Inside, LEFT, false, false, 50);
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 100);
        assert!(
            !m.double_click,
            "the intervening artwork press must break the double-click chain"
        );
    }

    /// Java's `lastClickSlot` is a nullable `Slot`: panel artwork and the outside region both
    /// produce `slot == null`, even though their wire sentinels differ (`-1` and `-999`). The
    /// second press below is therefore a double-click. Moving onto a real slot before release
    /// makes the double-click branch observable as mode `6`; comparing the full [`SlotTarget`]
    /// instead produces an ordinary mode-`0` click here.
    #[test]
    fn no_slot_regions_share_java_null_identity_for_double_clicks() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 4)));
        c.set_slot_contents(0, Some(stack(1, 1)));
        let mut m = MouseState::new();

        m.mouse_clicked(&c, SlotTarget::Inside, LEFT, false, false, 60_000);
        let first =
            triples(&m.mouse_released(&mut c, SlotTarget::Inside, LEFT, false, false, false));
        assert_eq!(first, vec![(-1, 0, 0)]);

        m.mouse_clicked(&c, SlotTarget::Outside, LEFT, false, false, 60_100);
        assert!(m.double_click, "both presses had Java's null lastClickSlot");
        let second =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(0), LEFT, false, false, false));
        assert_eq!(second, vec![(0, 0, 6)]);
    }

    /// java: `mouseReleased` has no `-1` gate -- it passes `k == -1` straight through, so
    /// releasing a held stack over the panel artwork *is* sent, with slot id `-1`.
    ///
    /// Reaching that branch needs `ignoreMouseUp == false`, and the only thing that clears it is
    /// the unconditional assignment at the top of `mouseClicked`. Pressing on the artwork is
    /// vanilla's own route: `if (l != -1)` skips the whole dispatch block, so nothing is sent and
    /// nothing sets the flag back.
    #[test]
    fn releasing_a_held_stack_on_the_panel_artwork_sends_slot_minus_one() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 4)));
        let mut m = MouseState::new();
        assert_eq!(
            m.mouse_clicked(&c, SlotTarget::Inside, LEFT, false, false, 0),
            None
        );
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Inside, LEFT, false, false, false));
        assert_eq!(events, vec![(-1, 0, 0)]);
    }

    /// java: `GuiContainer`'s constructor sets `ignoreMouseUp = true`, so the release of the very
    /// click that opened the window never reaches `Container.slotClick`.
    #[test]
    fn the_first_release_after_a_window_opens_is_swallowed() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 4)));
        let mut m = MouseState::new();
        assert!(m.ignore_mouse_up, "java: GuiContainer.<init>");
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(0), LEFT, false, false, false));
        assert!(events.is_empty(), "{events:?}");
        assert_eq!(c.cursor(), Some(&stack(1, 4)), "and nothing was placed");
    }

    #[test]
    fn a_second_click_after_the_window_elapses_is_not_a_double_click() {
        let mut c = chest_container(0, 3);
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 0);
        m.mouse_released(&mut c, SlotTarget::Slot(3), LEFT, false, false, false);
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, false, 999);
        assert!(!m.double_click);
    }

    // ------------------------------------------------------------------------------------
    // Regression tests for the divergences fixed in this file. Each one is written so that it
    // FAILS against the previous implementation; the comment on each says how.
    // ------------------------------------------------------------------------------------

    /// java: `GuiContainer.mouseClickMove:505` -- `Container.canAddItemToSlot(slot, itemstack,
    /// true) && slot.isItemValid(itemstack)`. Dragging stone across a chest row that already
    /// holds dirt must skip the dirt slots entirely, so the burst is `start + 1 add + end`, not
    /// `start + 3 adds + end`.
    ///
    /// Negative control: before the filter was wired into `drag_move`, all three swept slots
    /// entered `drag_splitting_slots` and this asserted `5` packets naming slots 0, 1 and 2.
    #[test]
    fn a_drag_skips_slots_that_already_hold_a_different_item() {
        let mut c = chest_container(0, 3);
        c.set_slot_contents(0, Some(stack(3, 1)));
        c.set_slot_contents(1, Some(stack(3, 1)));
        c.set_cursor(Some(stack(1, 8)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        for slot in 0..3u16 {
            m.drag_move(&c, SlotTarget::Slot(slot));
        }
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(2), LEFT, false, false, false));
        assert_eq!(
            events,
            vec![
                (-999, encode_drag_button(0, 0), 5),
                (2, encode_drag_button(1, 0), 5),
                (-999, encode_drag_button(2, 0), 5),
            ],
            "only the empty slot may be named"
        );
        assert_eq!(c.slot_stack(0), Some(&stack(3, 1)), "dirt untouched");
        assert_eq!(
            c.slot_stack(2),
            Some(&stack(1, 8)),
            "the whole stack lands in slot 2"
        );
    }

    /// java: `GuiContainer.mouseClickMove:505` again, this time the `isItemValid` half -- an
    /// armour slot refuses a non-armour item even though it is empty.
    #[test]
    fn a_drag_skips_an_empty_slot_whose_role_refuses_the_item() {
        // The anvil's output slot is `isItemValid == false` and empty.
        let mut c = anvil_container(0);
        c.set_cursor(Some(stack(1, 8)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 0);
        m.drag_move(&c, SlotTarget::Slot(0));
        m.drag_move(&c, SlotTarget::Slot(2));
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(2), LEFT, false, false, false));
        assert_eq!(
            events,
            vec![
                (-999, encode_drag_button(0, 0), 5),
                (0, encode_drag_button(1, 0), 5),
                (-999, encode_drag_button(2, 0), 5),
            ],
            "the output slot must never be named by a drag"
        );
    }

    /// java: `mouseReleased:566-570` -- the `ignoreMouseUp` branch `return`s *before* the
    /// `:646` `lastClickTime = 0` tail. Every press that dispatched its click immediately (here
    /// a shift-click, which leaves the cursor empty) sets `ignoreMouseUp`, so if the tail runs
    /// anyway the clock is zeroed and no second click can ever be a double-click once the
    /// window has been open longer than 250 ms.
    ///
    /// This is the test `F-P2-client.gui.inventory-02-2` asked for: the clock starts at 60 s,
    /// not at `0`, so `now - last_click_time` is not accidentally equal to `now - 0`. Against
    /// the old `else if` chain this assertion fails (`double_click == false`).
    #[test]
    fn an_ignored_release_preserves_the_double_click_clock_on_a_realistic_clock() {
        let mut c = chest_container(0, 3);
        c.set_slot_contents(3, Some(stack(1, 4)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, true, 60_000);
        m.mouse_released(&mut c, SlotTarget::Slot(3), LEFT, false, true, false);
        m.mouse_clicked(&c, SlotTarget::Slot(3), LEFT, false, true, 60_100);
        assert!(
            m.double_click,
            "the ignored release must not have zeroed last_click_time_ms"
        );
    }

    /// java: `mouseReleased:539-545`. The sweep iterates the *whole* slot list; `slot2 != slot`
    /// is not one of its five conditions, so the double-clicked slot is swept too.
    ///
    /// Negative control: the old loop began `if other.id == origin { continue; }`, so slot 40
    /// stayed put and only slot 41 moved.
    #[test]
    fn the_shift_double_click_sweep_includes_the_slot_that_was_clicked() {
        let mut c = chest_container(0, 3);
        // 27 chest slots, then 27 main + 9 hotbar. 27..=53 is the player's main inventory.
        c.set_slot_contents(40, Some(stack(1, 4)));
        c.set_slot_contents(41, Some(stack(1, 4)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(40), LEFT, false, true, 60_000);
        m.mouse_released(&mut c, SlotTarget::Slot(40), LEFT, false, true, false);
        m.mouse_clicked(&c, SlotTarget::Slot(40), LEFT, false, true, 60_100);
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(40), LEFT, false, true, false));
        assert_eq!(
            events,
            vec![(40, 0, 1), (41, 0, 1)],
            "the origin slot is swept first, in slot order"
        );
        assert_eq!(c.slot_stack(40), None, "swept into the chest");
        assert_eq!(c.slot_stack(41), None, "swept into the chest");
    }

    #[test]
    fn the_player_shift_double_click_sweep_excludes_its_separate_craft_result() {
        let mut c = player_container(0);
        c.set_slot_contents(0, Some(stack(3, 4))); // InventoryCraftResult
        c.set_slot_contents(1, Some(stack(3, 4))); // InventoryCrafting
        c.set_slot_contents(2, Some(stack(3, 4))); // InventoryCrafting
        let mut m = MouseState::new();

        // As in GuiContainer, the first shift-click is the press/release pair that establishes
        // the double-click clock; the second press/release emits the sweep itself.
        m.mouse_clicked(&c, SlotTarget::Slot(1), LEFT, false, true, 60_000);
        m.mouse_released(&mut c, SlotTarget::Slot(1), LEFT, false, true, false);
        m.mouse_clicked(&c, SlotTarget::Slot(1), LEFT, false, true, 60_100);
        let events =
            triples(&m.mouse_released(&mut c, SlotTarget::Slot(1), LEFT, false, true, false));

        assert_eq!(
            events,
            vec![(1, 0, 1), (2, 0, 1)],
            "the matrix sweep must not cross into InventoryCraftResult"
        );
        assert_eq!(c.slot_stack(0), Some(&stack(3, 4)));
    }

    /// java: `mouseReleased:558-564` -- cancelling a drag with the wrong button clears the drag,
    /// sets `ignoreMouseUp` and `return`s.
    ///
    /// Honest scope: this pins the branch's own three assignments, **not** the `return`. The
    /// `return` is unobservable on this path and this test would still pass if it were a
    /// fall-through: a drag can only be in progress while the cursor holds a stack, so the tail
    /// it skips (`if cursor.is_none() { last_click_time_ms = 0 }`) is a no-op there anyway, and
    /// the branch already assigns `drag_splitting = false` itself. The observable half of
    /// `F-P2-client.gui.inventory-02-2` is the *other* early return -- see
    /// `an_ignored_release_preserves_the_double_click_clock_on_a_realistic_clock`, which does
    /// fail if its `return` is removed.
    #[test]
    fn a_cancelled_drag_clears_the_drag_and_swallows_the_next_release() {
        let mut c = chest_container(0, 3);
        c.set_cursor(Some(stack(1, 10)));
        let mut m = MouseState::new();
        m.mouse_clicked(&c, SlotTarget::Slot(0), LEFT, false, false, 60_000);
        let events = m.mouse_released(&mut c, SlotTarget::Slot(0), RIGHT, false, false, false);
        assert!(events.is_empty(), "a mismatched release flushes nothing");
        assert!(!m.drag_splitting);
        assert!(m.drag_splitting_slots.is_empty());
        assert!(
            m.ignore_mouse_up,
            "java sets ignoreMouseUp before returning"
        );
        assert_eq!(m.last_click_time_ms, 60_000);
    }
}
