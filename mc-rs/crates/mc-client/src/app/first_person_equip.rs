//! `ItemRenderer.updateEquippedItem`'s test of whether the hand still holds the
//! stack it is drawing, with Java's object identity.
//!
//! Vanilla latches `itemToRender`, an `ItemStack` *object*, and every tick compares it with the
//! inventory's current stack by `getIsItemStackEqual` (item, count, damage and NBT). An object
//! compared with itself is always equal, so a stack the client mutates in place never starts the
//! equip animation by itself: a placed block's `--stackSize`, a sword's hit damage, a bow's shot
//! damage, a thrown snowball, eaten food. A server `S2FPacketSetSlot` instead puts a *new* object
//! in the slot that is equal only by value; after one, the latched object is stale, and the next
//! in-place write does re-equip.
//!
//! mc-rs's player container gives each slot object an identity
//! (`Container::slot_stack_identity`): a replacement renews it, an in-place write
//! (`Container::set_slot_contents_in_place`) keeps it. [`EquipLatch`] holds the latched object's
//! identity and value; the value follows in-place writes while that object is still the slot's,
//! and freezes once it has been replaced, as a Java reference would.
//!
//! Before this, mc-rs keyed the animation on the held stack's *value*, so every predicted
//! placement, hit, shot or throw looked like a new item: the block in hand sank and rose over six
//! ticks per placement while bridging, where vanilla snaps it down and raises it over three
//! (`resetEquippedProgress`). The tests replay `javaref/HeldEquipTraceRef.java`'s traces of the
//! real `ItemRenderer`.

use mc_protocol::ItemStack;

/// Java's `itemToRender` reference: the latched object's identity and its value.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EquipLatch {
    /// The latched object's container identity; meaningless while `value` is `None`.
    identity: u64,
    /// The latched object's value; `None` is Java's `itemToRender == null`.
    value: Option<ItemStack>,
    /// The render phases were rebuilt (a resource reload): the new
    /// first-person phase has no model for the latched stack yet.
    model_stale: bool,
}

impl EquipLatch {
    /// The first-person phase was rebuilt without the latched stack's
    /// model; the next tick that still holds the latched stack selects it again (no equip edge).
    pub fn mark_model_stale(&mut self) {
        self.model_stale = true;
    }

    /// Whether the latched stack's model must be selected again, clearing
    /// the mark.
    pub fn take_model_stale(&mut self) -> bool {
        std::mem::take(&mut self.model_stale)
    }

    /// `itemToRender.getIsItemStackEqual(current)` with `updateEquippedItem`'s null cases (both
    /// null is equal, one null is not). `current` is the held slot's `(identity, stack)`.
    ///
    /// When `current` is the latched object itself it is equal by definition, and its value is
    /// refreshed: Java's reference sees every in-place write.
    pub fn matches(&mut self, current: Option<(u64, &ItemStack)>) -> bool {
        match (&mut self.value, current) {
            (None, None) => true,
            (Some(latched), Some((identity, stack))) if identity == self.identity => {
                if latched != stack {
                    *latched = stack.clone();
                }
                true
            }
            (Some(latched), Some((_, stack))) => latched == stack,
            _ => false,
        }
    }

    /// Latch `current`, as `updateEquippedItem` does once `equippedProgress < 0.1`.
    pub fn latch(&mut self, current: Option<(u64, &ItemStack)>) {
        match current {
            Some((identity, stack)) => {
                self.identity = identity;
                if self.value.as_ref() != Some(stack) {
                    self.value = Some(stack.clone());
                }
            }
            None => {
                self.identity = 0;
                self.value = None;
            }
        }
    }

    /// The latched stack's value, `None` for an empty hand.
    #[must_use]
    pub fn value(&self) -> Option<&ItemStack> {
        self.value.as_ref()
    }
}

/// Whether this tick re-selects the first-person model from the held stack.
///
/// java: `RenderItem.renderItemModelForEntity` looks the model up from `itemToRender` every frame,
/// so a bow follows its pull state (`bow_pulling_N`) and a rod its hook (`fishing_rod_cast`) for
/// as long as it is the latched stack, and a stack changes model at the equip swap. While another
/// stack is still lowering out of the hand (`latch_matches` false), an incoming bow or rod must
/// wait for the swap like anything else: refreshing from the held stack then would draw the
/// incoming item in the outgoing one's place.
#[must_use]
pub fn refreshes_held_model(
    swapped: bool,
    held_bow: bool,
    held_rod: bool,
    latch_matches: bool,
) -> bool {
    swapped || ((held_bow || held_rod) && latch_matches)
}

/// `Minecraft.rightClickMouse`'s tail after `onPlayerRightClick` on a non-air block:
///
/// ```text
/// if (itemstack == null) return;
/// if (itemstack.stackSize == 0) mainInventory[currentItem] = null;
/// else if (itemstack.stackSize != i || playerController.isInCreativeMode())
///     itemRenderer.resetEquippedProgress();
/// ```
///
/// `before` and `after` are the held slot's `(identity, count)` around the call. The Java
/// `itemstack` is the object captured before it; an emptied slot is that object at size 0, and a
/// slot holding a different object leaves the captured one unchanged.
#[must_use]
pub fn block_click_resets_equip(
    before: Option<(u64, i32)>,
    after: Option<(u64, i32)>,
    creative: bool,
) -> bool {
    let Some((object, count_before)) = before else {
        return false;
    };
    let count_after = match after {
        None => 0,
        Some((identity, count)) if identity == object => count,
        Some(_) => count_before,
    };
    count_after != 0 && (count_after != count_before || creative)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mc_render::pipeline::FirstPersonRenderState;

    /// One tick of `HeldEquipTraceRef`: `[prevEquippedProgress, equippedProgress, latched object
    /// is the slot's (1/0)]`.
    type Tick = [f32; 3];

    struct JavaTrace {
        name: &'static str,
        ticks: &'static [Tick],
    }

    // Generated by crates/mc-client/javaref/HeldEquipTraceRef.java; do not edit by hand.
    const JAVA_TRACES: &[JavaTrace] = &[
        JavaTrace {
            name: "bridge_with_echo",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        },
        JavaTrace {
            name: "bridge_no_echo",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        },
        JavaTrace {
            name: "bridge_last_block",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.6, 0.0],
                [0.6, 0.2, 0.0],
                [0.2, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        },
        JavaTrace {
            name: "sword_hits",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.6, 0.0],
                [0.6, 0.2, 0.0],
                [0.2, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.6, 0.0],
                [0.6, 0.2, 0.0],
                [0.2, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        },
        JavaTrace {
            name: "snowball_throws",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        },
        JavaTrace {
            name: "slot_switch",
            ticks: &[
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.6, 0.0],
                [0.6, 0.2, 0.0],
                [0.2, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.6, 0.0],
                [0.6, 0.2, 0.0],
                [0.2, 0.0, 1.0],
                [0.0, 0.4, 1.0],
                [0.4, 0.8, 1.0],
                [0.8, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        },
    ];

    /// The same scheduled events `HeldEquipTraceRef` replays.
    #[derive(Clone, Copy)]
    enum Event {
        /// `rightClickMouse`, BLOCK branch: an ItemBlock placement.
        Place,
        /// `sendUseItem`: a thrown snowball.
        Throw,
        /// `attackTargetEntityWithCurrentItem` on a player: one point of sword damage.
        Hit,
        /// A server `SetSlot` for the held slot with this count and damage.
        SetSlot(i32, i16),
        /// A hotbar key.
        Select(usize),
    }

    fn stack(name: &str, count: i32, damage: i16) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("a real 1.8.9 item");
        ItemStack {
            item_id: id.0 as i16,
            count,
            damage,
            nbt: None,
        }
    }

    const HOTBAR: usize = crate::app::driver::HOTBAR_SLOT_OFFSET;

    /// Replay one scenario through the pieces the driver uses, in `Minecraft.runTick`'s order:
    /// server packets, then input, then `updateEquippedItem` (`Client::tick_first_person`).
    fn replay(slots: &[(usize, ItemStack)], events: &[(usize, Event)], ticks: usize) -> Vec<Tick> {
        let mut container = mc_ui::screens::container::player_container(0);
        for (slot, stack) in slots {
            container.set_slot_contents(HOTBAR + slot, Some(stack.clone()));
        }
        let mut selected = 0usize;
        let mut latch = EquipLatch::default();
        let mut state = FirstPersonRenderState::default();
        let held = |container: &mc_ui::screens::container::Container, selected: usize| {
            let index = HOTBAR + selected;
            container
                .slot_stack(index)
                .cloned()
                .map(|stack| (container.slot_stack_identity(index).unwrap_or(0), stack))
        };
        let update = |container: &mc_ui::screens::container::Container,
                      selected: usize,
                      latch: &mut EquipLatch,
                      state: &mut FirstPersonRenderState| {
            let current = held(container, selected);
            let current_ref = current.as_ref().map(|(id, stack)| (*id, stack));
            let matches = latch.matches(current_ref);
            let identity = current.as_ref().map(|(id, _)| *id);
            if state.update_equipped_item_matching(matches, identity, selected as i32) {
                latch.latch(current_ref);
            }
        };
        // Settle like the Java trace: the starting stack fully raised and latched.
        for _ in 0..4 {
            update(&container, selected, &mut latch, &mut state);
        }
        let mut out = Vec::new();
        for tick in 0..ticks {
            for (_, event) in events.iter().filter(|(at, _)| *at == tick) {
                if let Event::SetSlot(count, damage) = *event {
                    let item = container
                        .slot_stack(HOTBAR + selected)
                        .map_or(0, |stack| stack.item_id);
                    container.set_slot_contents(
                        HOTBAR + selected,
                        Some(ItemStack {
                            item_id: item,
                            count,
                            damage,
                            nbt: None,
                        }),
                    );
                }
            }
            for (_, event) in events.iter().filter(|(at, _)| *at == tick) {
                let index = HOTBAR + selected;
                match *event {
                    Event::SetSlot(..) => {}
                    Event::Select(slot) => selected = slot,
                    Event::Place => {
                        let before = held(&container, selected).map(|(id, s)| (id, s.count));
                        let mut placed = container.slot_stack(index).cloned().expect("a block");
                        placed.count -= 1;
                        container.set_slot_contents_in_place(
                            index,
                            (placed.count > 0).then_some(placed),
                        );
                        let after = held(&container, selected).map(|(id, s)| (id, s.count));
                        if block_click_resets_equip(before, after, false) {
                            state.reset_equipped_progress();
                        }
                    }
                    Event::Throw => {
                        let mut thrown = container.slot_stack(index).cloned().expect("a stack");
                        thrown.count -= 1;
                        container.set_slot_contents_in_place(
                            index,
                            (thrown.count > 0).then_some(thrown),
                        );
                        state.reset_equipped_progress2();
                    }
                    Event::Hit => {
                        let mut sword = container.slot_stack(index).cloned().expect("a sword");
                        sword.damage += 1;
                        container.set_slot_contents_in_place(index, Some(sword));
                    }
                }
            }
            update(&container, selected, &mut latch, &mut state);
            let current = held(&container, selected).map(|(id, _)| id);
            let latched_is_slots = state.item_to_render == current;
            out.push([
                state.prev_equipped_progress,
                state.equipped_progress,
                f32::from(u8::from(latched_is_slots)),
            ]);
        }
        out
    }

    fn assert_matches_java(name: &str, ours: &[Tick]) {
        let java = JAVA_TRACES
            .iter()
            .find(|trace| trace.name == name)
            .expect("a Java trace");
        assert_eq!(ours.len(), java.ticks.len(), "{name}: tick count");
        for (tick, (ours, java)) in ours.iter().zip(java.ticks).enumerate() {
            assert!(
                (ours[0] - java[0]).abs() < 1e-6 && (ours[1] - java[1]).abs() < 1e-6,
                "{name} tick {tick}: equip ({}, {}) vs Java ({}, {})",
                ours[0],
                ours[1],
                java[0],
                java[1]
            );
            assert_eq!(
                ours[2], java[2],
                "{name} tick {tick}: is the latched object the slot's object?"
            );
        }
    }

    #[test]
    fn bridging_snaps_the_block_down_like_java() {
        let wool = stack("wool", 64, 0);
        let placements = [2, 6, 10, 14].map(|tick| (tick, Event::Place));
        let echoes = [(5, 63), (9, 62), (13, 61), (17, 60)]
            .map(|(tick, count)| (tick, Event::SetSlot(count, 0)));
        let events: Vec<_> = placements.into_iter().chain(echoes).collect();
        assert_matches_java(
            "bridge_with_echo",
            &replay(&[(0, wool.clone())], &events, 26),
        );
        let events = [2, 6, 10].map(|tick| (tick, Event::Place));
        assert_matches_java("bridge_no_echo", &replay(&[(0, wool)], &events, 14));
        assert_matches_java(
            "bridge_last_block",
            &replay(&[(0, stack("wool", 1, 0))], &[(2, Event::Place)], 10),
        );
    }

    #[test]
    fn sword_hits_reequip_only_after_a_stale_echo_like_java() {
        let events = [
            (2, Event::Hit),
            (4, Event::Hit),
            (5, Event::SetSlot(1, 1)),
            (5, Event::Hit),
            (7, Event::Hit),
            (9, Event::Hit),
            (10, Event::Hit),
            (12, Event::Hit),
            (15, Event::SetSlot(1, 2)),
            (14, Event::Hit),
        ];
        assert_matches_java(
            "sword_hits",
            &replay(&[(0, stack("iron_sword", 1, 0))], &events, 22),
        );
    }

    #[test]
    fn snowball_throws_rise_at_once_like_java() {
        let events = [
            (2, Event::Throw),
            (5, Event::SetSlot(15, 0)),
            (7, Event::Throw),
            (10, Event::SetSlot(14, 0)),
            (12, Event::Throw),
        ];
        assert_matches_java(
            "snowball_throws",
            &replay(&[(0, stack("snowball", 16, 0))], &events, 18),
        );
    }

    #[test]
    fn switching_between_equal_stacks_does_not_animate_like_java() {
        let slots = [
            (0, stack("iron_sword", 1, 0)),
            (1, stack("iron_sword", 1, 0)),
            (2, stack("bow", 1, 0)),
        ];
        let events = [
            (3, Event::Select(1)),
            (8, Event::Select(2)),
            (16, Event::Select(0)),
        ];
        assert_matches_java("slot_switch", &replay(&slots, &events, 24));
    }

    /// Switching from a sword to a bow: the sword lowers out of the hand before the bow (and its
    /// pull model) comes up. Before, the bow's per-tick pull refresh replaced the sword at once.
    #[test]
    fn an_incoming_bow_waits_for_the_swap() {
        // Sword still latched and lowering, bow selected.
        assert!(!refreshes_held_model(false, true, false, false));
        // The swap edge installs the bow.
        assert!(refreshes_held_model(true, true, false, false));
        // From then on its pull state is refreshed every tick, as is a held rod's hook.
        assert!(refreshes_held_model(false, true, false, true));
        assert!(refreshes_held_model(false, false, true, true));
        assert!(!refreshes_held_model(false, false, true, false));
        // Ordinary items change model only at the swap.
        assert!(!refreshes_held_model(false, false, false, true));
    }

    /// Item 6 ("swapping an item into the held slot doesn't show it"),
    /// through the container click machine, [`EquipLatch`] and the renderer state, in the
    /// driver's order. A different stack clicked into the held slot (a mode-0 swap, a mode-2
    /// number key) lowers the old item and swaps the model in, as a hotbar switch does; a
    /// shift-click that merges more of the same block into the held stack does not animate
    /// (Java: `stackSize +=` on the very object `itemToRender` holds).
    #[test]
    fn a_stack_swapped_into_the_held_slot_reequips_and_a_merge_does_not() {
        use mc_ui::screens::container::Container;
        let held = |container: &Container| {
            container
                .slot_stack(HOTBAR)
                .cloned()
                .map(|stack| (container.slot_stack_identity(HOTBAR).unwrap_or(0), stack))
        };
        // One `updateEquippedItem`; returns whether the model swapped this tick.
        let update =
            |container: &Container, latch: &mut EquipLatch, state: &mut FirstPersonRenderState| {
                let current = held(container);
                let current_ref = current.as_ref().map(|(id, stack)| (*id, stack));
                let matches = latch.matches(current_ref);
                let identity = current.as_ref().map(|(id, _)| *id);
                let swapped = state.update_equipped_item_matching(matches, identity, 0);
                if swapped {
                    latch.latch(current_ref);
                }
                swapped
            };
        // Ticks until the model swaps (`None`: it never does), and the lowest progress seen.
        let run =
            |container: &Container, latch: &mut EquipLatch, state: &mut FirstPersonRenderState| {
                let mut lowest = state.equipped_progress;
                for tick in 0..8 {
                    let swapped = update(container, latch, state);
                    lowest = lowest.min(state.equipped_progress);
                    if swapped {
                        return (Some(tick), lowest);
                    }
                }
                (None, lowest)
            };

        let mut container = mc_ui::screens::container::player_container(0);
        container.set_slot_contents(HOTBAR, Some(stack("iron_sword", 1, 0)));
        container.set_slot_contents(9, Some(stack("bow", 1, 0)));
        container.set_slot_contents(10, Some(stack("wool", 32, 0)));
        container.set_slot_contents(11, Some(stack("wool", 8, 0)));
        let mut latch = EquipLatch::default();
        let mut state = FirstPersonRenderState::default();
        for _ in 0..4 {
            update(&container, &mut latch, &mut state);
        }
        assert_eq!(latch.value(), Some(&stack("iron_sword", 1, 0)));

        // Mode 0: pick the bow up, click it onto the sword (they swap), put the sword down.
        container.slot_click(9, 0, 0, false);
        container.slot_click(HOTBAR as i32, 0, 0, false);
        container.slot_click(12, 0, 0, false);
        let (swapped, lowest) = run(&container, &mut latch, &mut state);
        assert!(swapped.is_some(), "the bow is drawn in the hand");
        assert!(lowest < 0.1, "the sword lowered out first: {lowest}");
        assert_eq!(latch.value(), Some(&stack("bow", 1, 0)));
        assert_eq!(
            run(&container, &mut latch, &mut state).0,
            None,
            "then it settles"
        );

        // Mode 2: hover the sword in slot 12 and press hotbar key 1.
        container.slot_click(12, 0, 2, false);
        let (swapped, _) = run(&container, &mut latch, &mut state);
        assert!(swapped.is_some(), "the number-key swap draws the sword");
        assert_eq!(latch.value(), Some(&stack("iron_sword", 1, 0)));
        for _ in 0..4 {
            update(&container, &mut latch, &mut state);
        }

        // Mode 0 again, with blocks: swap 32 wool into the hand, then shift-click 8 more.
        container.slot_click(10, 0, 0, false);
        container.slot_click(HOTBAR as i32, 0, 0, false);
        container.slot_click(10, 0, 0, false);
        assert!(run(&container, &mut latch, &mut state).0.is_some());
        for _ in 0..4 {
            update(&container, &mut latch, &mut state);
        }
        assert_eq!(state.equipped_progress, 1.0);
        container.slot_click(11, 0, 1, false);
        assert_eq!(container.slot_stack(HOTBAR), Some(&stack("wool", 40, 0)));
        let (swapped, lowest) = run(&container, &mut latch, &mut state);
        assert_eq!(
            swapped, None,
            "a merge into the held stack is the same object"
        );
        assert_eq!(lowest, 1.0, "and does not dip");
        assert_eq!(
            latch.value(),
            Some(&stack("wool", 40, 0)),
            "the latch follows it"
        );
    }

    #[test]
    fn block_click_reset_follows_right_click_mouse() {
        // A placement: the same object, one fewer.
        assert!(block_click_resets_equip(
            Some((7, 64)),
            Some((7, 63)),
            false
        ));
        // Nothing placed: no reset in survival, always in creative.
        assert!(!block_click_resets_equip(
            Some((7, 64)),
            Some((7, 64)),
            false
        ));
        assert!(block_click_resets_equip(Some((7, 64)), Some((7, 64)), true));
        // The last block: the slot is nulled instead, no reset.
        assert!(!block_click_resets_equip(Some((7, 1)), None, true));
        // An empty hand returns early.
        assert!(!block_click_resets_equip(None, Some((9, 1)), true));
        // A different object in the slot leaves the captured stack's size unchanged.
        assert!(!block_click_resets_equip(
            Some((7, 64)),
            Some((8, 12)),
            false
        ));
    }
}
