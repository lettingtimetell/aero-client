//! The wire glue: turning a click into a [`C0EPacketClickWindow`](mc_protocol::play::c2s::ClickWindow)
//! and applying the server's replies back onto a [`Container`].
//!
//! # The rollback mechanism, and what "confirm/reject" actually does
//!
//! Vanilla's click flow (java: `PlayerControllerMP.windowClick`) is: run `slotClick` against
//! the **local** container first (client-side prediction — the player sees the result
//! instantly), *then* send `C0EPacketClickWindow` carrying whatever `slotClick` predicted, and
//! whatever `S32PacketConfirmTransaction` the server later sends back for that
//! `action_number` is only ever used to **acknowledge** — java's own handler
//! (`NetHandlerPlayClient.handleConfirmTransaction`) does nothing but reply with
//! `C0FPacketConfirmTransaction(window_id, action_number, true)` when `accepted == false`.
//! There is no separate "undo my prediction" step anywhere in vanilla. The actual rollback is
//! implicit: a server that rejected a click always follows up with `SetSlot`/`WindowItems`
//! packets carrying the true state, and [`Container::apply_set_slot`]/
//! [`Container::apply_window_items`] **unconditionally overwrite** whatever the local
//! prediction guessed — that overwrite *is* the rollback. [`Container::click`] and
//! [`Container::handle_confirm_transaction`] below reproduce exactly this shape, nothing more.

use mc_protocol::play::{c2s, s2c};
use mc_protocol::PacketBuffer;

use super::layouts;
use super::model::{Container, ContainerKind, MerchantRecipe};
use super::slot::{Slot, SlotRole};

impl Container {
    /// Predict a click locally (mutating `self` exactly as `slot_click` describes) and build
    /// the packet to send for it, allocating the next transaction id. java:
    /// `PlayerControllerMP.windowClick`.
    pub fn click(
        &mut self,
        slot_id: i32,
        used_button: u8,
        mode: u8,
        creative: bool,
    ) -> c2s::ClickWindow {
        let action_number = self.next_transaction_id();
        let clicked_item = self.slot_click(slot_id, used_button, mode, creative);
        c2s::ClickWindow {
            window_id: self.window_id,
            slot_id: slot_id as i16,
            used_button,
            action_number,
            mode,
            clicked_item,
        }
    }

    /// Handle a server `S32PacketConfirmTransaction` **addressed to this container**. Returns
    /// the `C0FPacketConfirmTransaction` acknowledgement to send back, or `None` when there is
    /// nothing to acknowledge (the transaction was for a different window, or it was already
    /// accepted). Never mutates slot contents itself — see the module doc for why that is
    /// correct: the resync comes from [`Self::apply_set_slot`]/[`Self::apply_window_items`],
    /// which the server sends independently of this reply.
    ///
    /// # This is not the whole handler
    ///
    /// Vanilla's resolution step lives one level up, in `NetHandlerPlayClient`, and window `0`
    /// never reaches a *screen* at all. A host must therefore route the packet through
    /// [`confirm_transaction_reply`] rather than calling this method on whatever window happens
    /// to be on screen; see that function's doc.
    #[must_use]
    pub fn handle_confirm_transaction(
        &self,
        pkt: &s2c::ConfirmTransaction,
    ) -> Option<c2s::ConfirmTransaction> {
        if pkt.window_id != self.window_id || pkt.accepted {
            return None;
        }
        Some(c2s::ConfirmTransaction {
            window_id: pkt.window_id,
            action_id: pkt.action_number,
            accepted: true,
        })
    }

    /// Apply a `S2FPacketSetSlot`. `window_id == -1` addresses the cursor stack, matching the
    /// wire's own sentinel; any other mismatched `window_id` is ignored (it is for a window this
    /// container is not).
    pub fn apply_set_slot(&mut self, pkt: &s2c::SetSlot) {
        let updates_enchanting_table = pkt.window_id >= 0
            && pkt.window_id as u8 == self.window_id
            && self.kind == ContainerKind::Enchanting
            && (pkt.slot == 0 || pkt.slot == 1);
        // 1.7.10's table has slot 0 only (`model_v5.rs`).
        let updates_enchanting_table = updates_enchanting_table || self.table_slot_v5(pkt);
        if pkt.window_id == -1 {
            self.set_cursor(pkt.item.clone());
        } else if pkt.window_id >= 0 && pkt.window_id as u8 == self.window_id && pkt.slot >= 0 {
            self.set_slot_contents(pkt.slot as usize, pkt.item.clone());
        }
        if updates_enchanting_table {
            self.enchanting_inputs_changed = true;
        }
    }

    /// Apply a `S30PacketWindowItems`: replace every slot's contents in one shot (vanilla sends
    /// this on window open, and as a bulk resync).
    pub fn apply_window_items(&mut self, pkt: &s2c::WindowItems) {
        if pkt.window_id != self.window_id {
            return;
        }
        for (i, item) in pkt.items.iter().enumerate() {
            self.set_slot_contents(i, item.clone());
        }
        if self.kind == ContainerKind::Enchanting && !pkt.items.is_empty() {
            self.enchanting_inputs_changed = true;
        }
        if self.kind == ContainerKind::EnchantingV5 && !pkt.items.is_empty() {
            self.enchanting_inputs_changed = true;
        }
    }
}

/// java: `NetHandlerPlayClient.handleConfirmTransaction` (`:1174-1196`) — the **whole** handler,
/// resolution step included.
///
/// ```java
/// Container container = null;
/// EntityPlayer entityplayer = this.gameController.thePlayer;
///
/// if (packetIn.getWindowId() == 0)            container = entityplayer.inventoryContainer;
/// else if (packetIn.getWindowId() == entityplayer.openContainer.windowId)
///                                             container = entityplayer.openContainer;
///
/// if (container != null && !packetIn.func_148888_e())
///     this.addToSendQueue(new C0FPacketConfirmTransaction(windowId, actionNumber, true));
/// ```
///
/// Three things about that shape are easy to get wrong, and getting any of them wrong silences
/// the reply:
///
/// 1. **The window-`0` branch cannot fail.** `inventoryContainer` is built in the `EntityPlayer`
///    constructor and lives for the whole session, so a rejected window-`0` transaction is
///    *always* acked — with no GUI open, and regardless of which other window is open. 1.8
///    servers use exactly this packet as a round-trip ping, so gating the reply on "a container
///    screen is open" drops most `C0F`s a session ever owes.
/// 2. **`openContainer` is never null.** It too defaults to `inventoryContainer`, so with no GUI
///    open the `else if` is just `windowId == 0` again — already handled by branch 1. Passing
///    `None` for `open` here models that correctly.
/// 3. **Only a *rejected* transaction is answered.** `func_148888_e()` is `accepted`. The
///    accepted case is silent in vanilla too, which is why a coverage test built on an accepted
///    transaction cannot distinguish a correct handler from one that never replies at all.
///
/// `player` is the always-present window-`0` container; `open` is the container of the window
/// currently on screen, if any.
#[must_use]
pub fn confirm_transaction_reply(
    player: &Container,
    open: Option<&Container>,
    pkt: &s2c::ConfirmTransaction,
) -> Option<c2s::ConfirmTransaction> {
    let container = if pkt.window_id == PLAYER_WINDOW_ID {
        Some(player)
    } else {
        open.filter(|c| c.window_id == pkt.window_id)
    };
    if container.is_none() || pkt.accepted {
        return None;
    }
    Some(c2s::ConfirmTransaction {
        window_id: pkt.window_id,
        action_id: pkt.action_number,
        accepted: true,
    })
}

/// java: the literal `0` in `handleConfirmTransaction`/`handleWindowItems`/`handleSetSlot` —
/// `EntityPlayer.inventoryContainer.windowId`, which `Container`'s constructor leaves at its
/// default.
pub const PLAYER_WINDOW_ID: u8 = 0;

/// Window types that have their own dispatch branch in
/// `NetHandlerPlayClient.handleOpenWindow` and must therefore **not** fall through to
/// [`container_for_open_window`]'s chest fallback.
///
/// The villager branch is handled by the merchant builder below. Horses have a separate arm
/// immediately after this one because their slot layout depends on the entity inventory size.
const OWN_DISPATCH_BRANCH: [&str; 1] = ["minecraft:villager"];

fn merchant_container(window_id: u8) -> Container {
    let mut slots = Vec::with_capacity(39);
    slots.push(Slot::new(0, 36, 53));
    slots.push(Slot::new(1, 62, 53));
    slots.push(Slot::new(2, 120, 53).with_role(SlotRole::CraftResult));
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
    Container::new(window_id, ContainerKind::Merchant, slots)
}

/// Build the `ContainerHorseInventory` shape used by `GuiScreenHorseInventory`.
///
/// Java always creates saddle and armour slots first, then adds fifteen chest slots only when
/// `EntityHorse.isChested()` is true, followed by the player's 27 main slots and nine hotbar
/// slots. The server's horse `OpenWindow` packet reports an inventory size of `2` or `17`, so the
/// packet size is the stable wire-side signal for the chested layout. The model keeps the
/// horse-specific routing and slot roles: generic chest routing would accept arbitrary items into
/// saddle/armour and send the wrong shift-click destination.
fn horse_container(window_id: u8, chested: bool) -> Container {
    let top = if chested { 17 } else { 2 };
    let mut slots = Vec::with_capacity(top + 36);

    // java: ContainerHorseInventory's saddle and armour slots.
    slots.push(
        Slot::new(0, 8, 18)
            .with_role(SlotRole::HorseSaddle)
            .with_limit(1),
    );
    // The packet carries no horse type. Keep armour closed until the tracked entity snapshot
    // resolves `canWearArmor()`; otherwise a llama/donkey could accept one frame of armour.
    slots.push(Slot::new(1, 8, 36).with_role(SlotRole::HorseArmor { can_wear: false }));
    if chested {
        for row in 0..3 {
            for col in 0..5 {
                let id = slots.len() as u16;
                slots.push(Slot::new(id, 80 + col * 18, 18 + row * 18));
            }
        }
    }

    // java: the player inventory uses j = (3 - 4) * 18 = -18 for this GUI.
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
    Container::new(window_id, ContainerKind::Horse { top }, slots)
}

/// Decode the legacy `MC|TrList` payload, returning its window id and offers. The first four
/// bytes are the signed window id; recipe stacks use the protocol-47 `ItemStack` codec.
pub fn merchant_recipes_from_plugin(
    data: &[u8],
) -> Result<(i32, Vec<MerchantRecipe>), mc_protocol::ProtoError> {
    let mut buf = PacketBuffer::from_slice(data);
    let window_id = buf.read_i32()?;
    let count = usize::from(buf.read_u8()?);
    let mut recipes = Vec::with_capacity(count);
    for _ in 0..count {
        let buy_a = buf
            .read_item_stack()?
            .ok_or(mc_protocol::ProtoError::Eof("merchant first buy stack"))?;
        let sell = buf
            .read_item_stack()?
            .ok_or(mc_protocol::ProtoError::Eof("merchant sell stack"))?;
        let has_second = buf.read_bool()?;
        let buy_b = if has_second {
            Some(
                buf.read_item_stack()?
                    .ok_or(mc_protocol::ProtoError::Eof("merchant second buy stack"))?,
            )
        } else {
            None
        };
        let disabled = buf.read_bool()?;
        let uses = buf.read_i32()?;
        let max_uses = buf.read_i32()?;
        recipes.push(MerchantRecipe {
            buy_a,
            buy_b,
            sell,
            uses: if disabled { max_uses } else { uses },
            max_uses,
            disabled,
        });
    }
    Ok((window_id, recipes))
}

/// Build the right [`Container`] shape for a `S2DPacketOpenWindow`, or `None` for a window this
/// section cannot open at all.
///
/// java: `NetHandlerPlayClient.handleOpenWindow` (`:1092`), whose shape is:
///
/// ```text
/// if      "minecraft:container"  -> displayGUIChest(new InventoryBasic(title, slotCount))
/// else if "minecraft:villager"   -> displayVillagerTradeGui(...)
/// else if "EntityHorse"          -> displayGUIHorse(...)
/// else if !packetIn.hasSlots()   -> displayGui(new LocalBlockIntercommunication(...))
/// else                           -> displayGUIChest(new ContainerLocalMenu(guiId, ..., slotCount))
/// ```
///
/// and `displayGUIChest` (`EntityPlayerSP.java:601`) then switches on the *inventory's* own
/// `getGuiID()` — which for a `ContainerLocalMenu` is the packet's `guiId` — to pick between
/// `GuiHopper`/`GuiFurnace`/`GuiBrewingStand`/`GuiBeacon`/`GuiDispenser` and a plain `GuiChest`
/// **default**. So the last `else` is a real, load-bearing fallback and not an error path:
/// `"minecraft:container"` and every unrecognised slotted guiId open a chest of
/// `slotCount / 9` rows.
///
/// That fallback was a `_ => None`, so server- and plugin-created
/// menus — the single most common inventory a hub-server player sees — never opened, while the
/// server believed one was open.
///
/// The order above is part of the protocol. In particular, `"minecraft:container"` is tested
/// before `hasSlots()` and therefore opens even with zero slots, while the three client-built
/// block GUIs are only selected when `hasSlots()` is false. A non-zero-slot packet that happens
/// to use one of those three ids goes through `ContainerLocalMenu` and the ordinary chest
/// fallback in vanilla.
#[must_use]
pub fn container_for_open_window(pkt: &s2c::OpenWindow) -> Option<Container> {
    let kind = pkt.inventory_type.as_str();

    // java: `handleOpenWindow:1097` -- deliberately before `!packetIn.hasSlots()`.
    if kind == "minecraft:container" {
        return Some(layouts::chest_container(pkt.window_id, pkt.slot_count / 9));
    }

    // These have their own dispatch branches. Villagers use a 3-slot merchant inventory.
    if kind == "minecraft:villager" {
        return Some(merchant_container(pkt.window_id));
    }
    // java: `handleOpenWindow` resolves the entity id before calling `displayGUIHorse`. A
    // missing id cannot identify a horse and is routed to the caller's no-screen path (as in
    // vanilla: no screen, no reply). Valid protocol-47 horse inventories are exactly two slots
    // (un-chested) or seventeen slots (chested); refusing malformed sizes avoids opening a
    // screen whose slot numbers could never agree with the server's `ContainerHorseInventory`.
    if kind == "EntityHorse" {
        let _entity_id = pkt.entity_id?;
        return match pkt.slot_count {
            2 => Some(horse_container(pkt.window_id, false)),
            17 => Some(horse_container(pkt.window_id, true)),
            _ => None,
        };
    }
    if OWN_DISPATCH_BRANCH.contains(&kind) {
        return None;
    }

    if pkt.slot_count == 0 {
        // java: `handleOpenWindow`'s `!packetIn.hasSlots()` arm wraps the gui id in a
        // `LocalBlockIntercommunication` and hands it to `EntityPlayerSP.displayGui`, which
        // recognises exactly these three ids.
        return match kind {
            "minecraft:crafting_table" => Some(layouts::crafting_table_container(pkt.window_id)),
            "minecraft:enchanting_table" => {
                Some(layouts::enchanting_table_container(pkt.window_id))
            }
            "minecraft:anvil" => Some(layouts::anvil_container(pkt.window_id)),
            _ => None,
        };
    }

    match kind {
        "minecraft:hopper" => Some(layouts::hopper_container(pkt.window_id, pkt.slot_count)),
        "minecraft:dispenser" | "minecraft:dropper" => {
            Some(layouts::dispenser_container(pkt.window_id))
        }
        "minecraft:furnace" => Some(layouts::furnace_container(pkt.window_id)),
        "minecraft:brewing_stand" => Some(layouts::brewing_stand_container(pkt.window_id)),
        "minecraft:beacon" => Some(layouts::beacon_container(pkt.window_id)),
        // `"minecraft:chest"` and `displayGUIChest`'s default land here. java: `GuiChest`'s
        // `inventoryRows = chestInventory.getSizeInventory() / 9`, whose truncation for a slot
        // count that is not a multiple of nine is vanilla's own.
        _ => Some(layouts::chest_container(pkt.window_id, pkt.slot_count / 9)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::screens::container::layouts::{chest_container, player_container};
    use mc_protocol::item::ItemStack;

    fn stack(item_id: i16, count: i32) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage: 0,
            nbt: None,
        }
    }

    fn rejected(window_id: u8, action_number: i16) -> s2c::ConfirmTransaction {
        s2c::ConfirmTransaction {
            window_id,
            action_number,
            accepted: false,
        }
    }

    // ------------------------------------------------------------------------------------
    // `confirm_transaction_reply` -- java `NetHandlerPlayClient.handleConfirmTransaction`.
    //
    // Note what these do NOT test with: an *accepted* transaction. Vanilla is silent for those
    // too, so an accepted-case test passes identically against a handler that never replies at
    // all -- which is exactly why the pre-existing coverage could not fail on this defect. Every
    // positive assertion below uses `accepted: false`.
    // ------------------------------------------------------------------------------------

    /// The defect this whole workstream is about: window `0` resolves to `inventoryContainer`,
    /// which exists whether or not a GUI is open, so a rejected window-`0` transaction is
    /// **always** acked.
    ///
    /// Negative control: route the same packet through `handle_confirm_transaction` on an open
    /// chest (`window_id == 3`) and it returns `None` -- that is the old behaviour, asserted
    /// below so this test cannot pass by accident.
    #[test]
    fn a_rejected_window_zero_transaction_is_acked_with_no_screen_open() {
        let player = player_container(PLAYER_WINDOW_ID);
        let reply = confirm_transaction_reply(&player, None, &rejected(0, 7));
        assert_eq!(
            reply,
            Some(c2s::ConfirmTransaction {
                window_id: 0,
                action_id: 7,
                accepted: true,
            })
        );

        let chest = chest_container(3, 3);
        assert_eq!(
            chest.handle_confirm_transaction(&rejected(0, 7)),
            None,
            "negative control: the open-screen path alone provably cannot answer window 0"
        );
    }

    /// ... and it is still acked while a *different* window is open, because java tests
    /// `windowId == 0` first and never reaches the `openContainer` comparison.
    #[test]
    fn a_rejected_window_zero_transaction_is_acked_while_a_chest_is_open() {
        let player = player_container(PLAYER_WINDOW_ID);
        let chest = chest_container(3, 3);
        let reply = confirm_transaction_reply(&player, Some(&chest), &rejected(0, 11));
        assert_eq!(reply.map(|r| (r.window_id, r.action_id)), Some((0, 11)));
    }

    /// java: `else if (packetIn.getWindowId() == entityplayer.openContainer.windowId)`.
    #[test]
    fn a_rejected_transaction_for_the_open_window_is_acked() {
        let player = player_container(PLAYER_WINDOW_ID);
        let chest = chest_container(3, 3);
        let reply = confirm_transaction_reply(&player, Some(&chest), &rejected(3, 2));
        assert_eq!(reply.map(|r| (r.window_id, r.action_id)), Some((3, 2)));
    }

    /// A window that is neither `0` nor the open one resolves to no container, so java sends
    /// nothing.
    #[test]
    fn a_rejected_transaction_for_an_unknown_window_is_not_acked() {
        let player = player_container(PLAYER_WINDOW_ID);
        let chest = chest_container(3, 3);
        assert_eq!(
            confirm_transaction_reply(&player, Some(&chest), &rejected(9, 2)),
            None
        );
        assert_eq!(
            confirm_transaction_reply(&player, None, &rejected(9, 2)),
            None
        );
    }

    /// java: `if (container != null && !packetIn.func_148888_e())` -- an accepted transaction is
    /// never answered. This is the one case the old code got right, and on its own it proves
    /// nothing (see the module comment above).
    #[test]
    fn an_accepted_transaction_is_never_answered() {
        let player = player_container(PLAYER_WINDOW_ID);
        let chest = chest_container(3, 3);
        for window_id in [0u8, 3, 9] {
            assert_eq!(
                confirm_transaction_reply(
                    &player,
                    Some(&chest),
                    &s2c::ConfirmTransaction {
                        window_id,
                        action_number: 1,
                        accepted: true,
                    },
                ),
                None,
                "window {window_id}"
            );
        }
    }

    #[test]
    fn click_allocates_a_fresh_transaction_id_each_time() {
        let mut c = chest_container(1, 3);
        let p1 = c.click(-999, 0, 0, false); // click outside window: legal no-op click
        let p2 = c.click(-999, 0, 0, false);
        assert_eq!(p1.action_number, 1);
        assert_eq!(p2.action_number, 2);
        assert_eq!(p1.window_id, 1);
        assert_eq!(p1.slot_id, -999);
    }

    #[test]
    fn click_packet_carries_the_local_prediction() {
        let mut c = chest_container(1, 3);
        c.set_slot_contents(0, Some(stack(1, 10)));
        // Left-click a stacked slot with an empty cursor: picks the whole stack up.
        let pkt = c.click(0, 0, 0, false);
        assert_eq!(pkt.clicked_item, Some(stack(1, 10)));
        assert_eq!(c.cursor(), Some(&stack(1, 10)));
        assert_eq!(c.slot_stack(0), None);
    }

    /// `ContainerRepair`'s output gate is part of the packet snapshot contract as well as the
    /// local state machine. In particular, Java's mode-1 assignment sits inside
    /// `canTakeStack`, while modes 0/2/4 retain their normal packet shape even when their local
    /// mutation is refused.
    #[test]
    fn anvil_click_packets_keep_mode_specific_snapshots_when_xp_is_insufficient() {
        for mode in [0u8, 1, 2, 4] {
            let mut c = crate::screens::container::layouts::anvil_container(7);
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
            c.set_player_experience_level(0);
            let output = c.slot_stack(2).cloned().expect("local anvil output");
            let pkt = c.click(2, 0, mode, false);

            assert_eq!(pkt.window_id, 7);
            assert_eq!(pkt.slot_id, 2);
            assert_eq!(pkt.mode, mode);
            assert_eq!(
                pkt.clicked_item,
                (mode == 0).then_some(output.clone()),
                "mode {mode}: Java snapshot"
            );
            assert_eq!(c.cursor(), None, "mode {mode}: no refused pickup");
            assert_eq!(c.slot_stack(2), Some(&output), "mode {mode}: output stays");
            assert_eq!(c.slot_stack(0).map(|s| s.damage), Some(200));
            assert_eq!(c.slot_stack(1), Some(&stack(265, 1)));
        }
    }

    #[test]
    fn rejected_transaction_is_acknowledged_and_the_next_set_slot_rolls_back_the_guess() {
        let mut c = chest_container(1, 3);
        c.set_slot_contents(5, Some(stack(4, 3)));
        // Client optimistically predicts picking up the stack...
        let pkt = c.click(5, 0, 0, false);
        assert_eq!(c.slot_stack(5), None, "predicted pickup applied locally");
        assert_eq!(pkt.action_number, 1);

        // ...but the server disagrees (say, another player emptied the slot moments earlier).
        let reject = s2c::ConfirmTransaction {
            window_id: 1,
            action_number: pkt.action_number,
            accepted: false,
        };
        let ack = c
            .handle_confirm_transaction(&reject)
            .expect("must acknowledge a rejection");
        assert_eq!(
            ack,
            c2s::ConfirmTransaction {
                window_id: 1,
                action_id: 1,
                accepted: true,
            }
        );

        // The server's resync always follows a rejection: it wins unconditionally.
        c.apply_set_slot(&s2c::SetSlot {
            window_id: 1,
            slot: 5,
            item: None,
        });
        c.apply_set_slot(&s2c::SetSlot {
            window_id: -1,
            slot: 0,
            item: None,
        });
        assert_eq!(c.slot_stack(5), None);
        assert_eq!(
            c.cursor(),
            None,
            "the server also corrected the mispredicted cursor"
        );
    }

    #[test]
    fn accepted_transaction_needs_no_acknowledgement() {
        let c = chest_container(1, 3);
        let accept = s2c::ConfirmTransaction {
            window_id: 1,
            action_number: 1,
            accepted: true,
        };
        assert_eq!(c.handle_confirm_transaction(&accept), None);
    }

    #[test]
    fn confirm_transaction_for_a_different_window_is_ignored() {
        let c = chest_container(1, 3);
        let reject = s2c::ConfirmTransaction {
            window_id: 2,
            action_number: 1,
            accepted: false,
        };
        assert_eq!(c.handle_confirm_transaction(&reject), None);
    }

    #[test]
    fn set_slot_for_a_foreign_window_is_ignored() {
        let mut c = chest_container(1, 3);
        c.set_slot_contents(0, Some(stack(1, 1)));
        c.apply_set_slot(&s2c::SetSlot {
            window_id: 2,
            slot: 0,
            item: Some(stack(99, 1)),
        });
        assert_eq!(
            c.slot_stack(0),
            Some(&stack(1, 1)),
            "unrelated window must not overwrite us"
        );
    }

    #[test]
    fn window_items_replaces_every_slot_at_once() {
        let mut c = chest_container(1, 3);
        c.set_slot_contents(0, Some(stack(1, 1)));
        let mut items = vec![None; c.slots().len()];
        items[3] = Some(stack(5, 2));
        c.apply_window_items(&s2c::WindowItems {
            window_id: 1,
            items,
        });
        assert_eq!(
            c.slot_stack(0),
            None,
            "slot 0 was cleared by the bulk resync"
        );
        assert_eq!(c.slot_stack(3), Some(&stack(5, 2)));
    }

    #[test]
    fn open_window_dispatch_builds_the_matching_shape() {
        let pkt = s2c::OpenWindow {
            window_id: 3,
            inventory_type: "minecraft:furnace".into(),
            window_title: "{\"text\":\"Furnace\"}".into(),
            slot_count: 3,
            entity_id: None,
        };
        let c = container_for_open_window(&pkt).expect("furnace is a known type");
        assert_eq!(c.window_id, 3);
        assert_eq!(c.slots().len(), 39);
    }

    #[test]
    fn open_window_dispatch_returns_none_for_unhandled_types() {
        let pkt = s2c::OpenWindow {
            window_id: 3,
            inventory_type: "minecraft:unknown_local_gui".into(),
            window_title: "{\"text\":\"Unsupported\"}".into(),
            slot_count: 0,
            entity_id: None,
        };
        assert_eq!(container_for_open_window(&pkt), None);
    }

    fn open(inventory_type: &str, slot_count: u8) -> s2c::OpenWindow {
        s2c::OpenWindow {
            window_id: 9,
            inventory_type: inventory_type.to_owned(),
            window_title: r#"{"text":"Menu"}"#.into(),
            slot_count,
            entity_id: None,
        }
    }

    /// java: `handleOpenWindow`'s **first** branch is
    /// `"minecraft:container"` → `displayGUIChest(new InventoryBasic(...))`, and its **last** is
    /// `ContainerLocalMenu` → `displayGUIChest`, whose own default is `GuiChest`. Both were
    /// `None`, so every plugin menu on every hub server opened nothing at all while the server
    /// believed a window was up.
    #[test]
    fn a_plugin_menu_opens_as_a_chest_of_the_declared_size() {
        for kind in [
            "minecraft:container",
            "minecraft:chest",
            "SomeModdedMenu",
            "",
        ] {
            let c = container_for_open_window(&open(kind, 54))
                .unwrap_or_else(|| panic!("{kind} must open"));
            assert_eq!(c.window_id, 9);
            assert_eq!(
                c.slots().len(),
                54 + 36,
                "six rows plus the player's own 36"
            );
        }
        // java: `GuiChest`'s `getSizeInventory() / 9` truncates with **no** floor. This used
        // to assert `9 + 36` behind a `max(1)`, i.e. it pinned the bug; the real behaviour is
        // covered by `a_menu_of_fewer_than_nine_slots_gets_no_container_row`.
        let c = container_for_open_window(&open("minecraft:container", 5)).expect("opens");
        assert_eq!(c.slots().len(), 36);
    }

    /// The fallback must not swallow the two guiIds that have their own vanilla branch: opening
    /// a villager shop or a horse as a chest would put every click on the wrong slot role.
    #[test]
    fn dedicated_villager_and_horse_routes_keep_their_slot_shapes() {
        let villager = container_for_open_window(&open("minecraft:villager", 3))
            .expect("villager has its own merchant container");
        assert_eq!(villager.kind, ContainerKind::Merchant);
        assert_eq!(villager.slots().len(), 39);
        assert_eq!(villager.slots()[2].role, SlotRole::CraftResult);

        let mut horse = open("EntityHorse", 17);
        horse.entity_id = Some(42);
        let horse = container_for_open_window(&horse).expect("chested horse has its own route");
        assert_eq!(horse.kind, ContainerKind::Horse { top: 17 });
        assert_eq!(horse.slots().len(), 53);
        assert_eq!((horse.slots()[0].x, horse.slots()[0].y), (8, 18));
        assert_eq!((horse.slots()[1].x, horse.slots()[1].y), (8, 36));
        assert_eq!((horse.slots()[2].x, horse.slots()[2].y), (80, 18));
        assert_eq!((horse.slots()[16].x, horse.slots()[16].y), (152, 54));
        assert_eq!((horse.slots()[17].x, horse.slots()[17].y), (8, 84));
        assert_eq!((horse.slots()[52].x, horse.slots()[52].y), (152, 142));

        let mut unchested = open("EntityHorse", 2);
        unchested.entity_id = Some(42);
        let unchested =
            container_for_open_window(&unchested).expect("un-chested horse has its own route");
        assert_eq!(unchested.kind, ContainerKind::Horse { top: 2 });
        assert_eq!(unchested.slots().len(), 38);
        assert_eq!((unchested.slots()[2].x, unchested.slots()[2].y), (8, 84));
        assert_eq!(container_for_open_window(&open("EntityHorse", 17)), None);
    }

    #[test]
    fn horse_slots_are_entity_validated_and_hover_only_for_usable_armour() {
        let mut horse_packet = open("EntityHorse", 2);
        horse_packet.entity_id = Some(42);
        let mut horse = container_for_open_window(&horse_packet).expect("horse route");
        assert_eq!(horse.slots()[0].role, SlotRole::HorseSaddle);
        assert!(horse.slots()[0].is_item_valid(&stack(329, 1))); // saddle
        assert!(!horse.slots()[0].is_item_valid(&stack(276, 1))); // sword
        assert!(!horse.slots()[1].is_item_valid(&stack(310, 1))); // unresolved entity state
        assert!(!horse.slots()[1].can_be_hovered());

        horse.set_horse_can_wear_armor(true);
        assert!(horse.slots()[1].is_item_valid(&stack(310, 1))); // ordinary horse
        assert!(horse.slots()[1].can_be_hovered());
        horse.set_horse_can_wear_armor(false);
        assert!(!horse.slots()[1].is_item_valid(&stack(310, 1)));
        assert!(!horse.slots()[1].can_be_hovered());
        // Java's `getSlotAtPosition` still finds this cell for clicks; only the hover paint is
        // suppressed by `Slot.canBeHovered`.
        assert!(horse.slots()[1].x == 8 && horse.slots()[1].y == 36);
    }

    #[test]
    fn horse_shift_click_prioritises_saddle_then_armor_without_generic_chest_fallback() {
        let mut horse_packet = open("EntityHorse", 17);
        horse_packet.entity_id = Some(42);
        let mut horse = container_for_open_window(&horse_packet).expect("horse route");
        let saddle_id = mc_data::item_by_name("saddle").expect("saddle id").0 as i16;
        horse.set_slot_contents(17, Some(stack(saddle_id, 1))); // first player slot
        assert_eq!(horse.slot_stack(17).map(|s| s.item_id), Some(saddle_id));
        assert!(horse.slots()[0].is_item_valid(&stack(saddle_id, 1)));
        let moved = horse
            .transfer_stack_in_slot(17)
            .expect("saddle should transfer");
        assert_eq!(moved.item_id, saddle_id);
        assert_eq!(horse.slot_stack(0).map(|s| s.item_id), Some(saddle_id));

        horse.set_horse_can_wear_armor(true);
        horse.set_slot_contents(18, Some(stack(310, 1))); // next player slot
        horse
            .transfer_stack_in_slot(18)
            .expect("armour should transfer");
        assert_eq!(horse.slot_stack(1).map(|s| s.item_id), Some(310));
        assert!(horse.slot_stack(2).is_none());
    }

    #[test]
    fn merchant_plugin_payload_decodes_offers_and_disabled_compensation() {
        let buy = stack(265, 3);
        let second = stack(331, 2);
        let sell = stack(emerald_id(), 1);
        let mut buf = PacketBuffer::new();
        buf.write_i32(9);
        buf.write_u8(2);
        mc_protocol::item::ItemStack::write(Some(&buy), &mut buf);
        mc_protocol::item::ItemStack::write(Some(&sell), &mut buf);
        buf.write_bool(true);
        mc_protocol::item::ItemStack::write(Some(&second), &mut buf);
        buf.write_bool(false);
        buf.write_i32(2);
        buf.write_i32(7);
        mc_protocol::item::ItemStack::write(Some(&buy), &mut buf);
        mc_protocol::item::ItemStack::write(Some(&sell), &mut buf);
        buf.write_bool(false);
        buf.write_bool(true);
        buf.write_i32(1);
        buf.write_i32(12);

        let (window_id, recipes) =
            merchant_recipes_from_plugin(buf.as_slice()).expect("valid list");
        assert_eq!(window_id, 9);
        assert_eq!(recipes.len(), 2);
        assert_eq!(recipes[0].buy_b, Some(second));
        assert!(!recipes[0].disabled);
        assert_eq!(
            recipes[1].uses, 12,
            "disabled recipes compensate uses to max"
        );
    }

    #[test]
    fn merchant_plugin_payload_rejects_truncated_offer_as_a_negative_control() {
        let mut buf = PacketBuffer::new();
        buf.write_i32(9);
        buf.write_u8(1);
        assert!(merchant_recipes_from_plugin(buf.as_slice()).is_err());
    }

    #[test]
    fn merchant_shift_click_routes_result_reverse_and_buys_forward() {
        let mut merchant = merchant_container(4);
        merchant.set_slot_contents(2, Some(stack(388, 3)));
        let result = merchant.click(2, 0, 1, false);
        assert_eq!(result.clicked_item, Some(stack(388, 3)));
        assert_eq!(merchant.slot_stack(38), Some(&stack(388, 3)));
        assert_eq!(merchant.slot_stack(2), None);

        merchant.set_slot_contents(0, Some(stack(265, 100)));
        assert_eq!(
            merchant.slot_stack(0).unwrap().count,
            64,
            "merchant inventory clamps"
        );
        merchant.click(0, 0, 1, false);
        assert_eq!(merchant.slot_stack(3), Some(&stack(265, 64)));
        assert_eq!(merchant.slot_stack(0), None);
    }

    fn emerald_id() -> i16 {
        388
    }

    /// java: `else if (!packetIn.hasSlots())` routes a zero-slot window to
    /// `EntityPlayerSP.displayGui`, which knows exactly three gui ids. All three must open with
    /// `slot_count == 0`, because `EntityPlayerMP.displayGui` uses the three-argument
    /// `S2DPacketOpenWindow` constructor for every one of them.
    ///
    /// This replaces `a_zero_slot_window_is_still_unhandled`, which **asserted the bug**
    /// (`container_for_open_window(crafting_table, 0) == None`) and so could never fail while
    /// the code was wrong -- fixing the defect is what broke it.
    #[test]
    fn all_three_zero_slot_gui_ids_open_their_own_container() {
        let c = container_for_open_window(&open("minecraft:crafting_table", 0))
            .expect("crafting table");
        assert_eq!(c.slots().len(), 46);
        assert_eq!(
            c.kind,
            crate::screens::container::model::ContainerKind::Workbench
        );
        assert_eq!(
            container_for_open_window(&open("minecraft:enchanting_table", 0))
                .expect("enchanting table")
                .slots()
                .len(),
            38
        );
        assert_eq!(
            container_for_open_window(&open("minecraft:anvil", 0))
                .expect("anvil")
                .slots()
                .len(),
            39
        );
    }

    /// A gui id vanilla's `displayGui` does not know still opens nothing. The exact
    /// `"minecraft:container"` id is different: `handleOpenWindow` tests it before
    /// `hasSlots()` and therefore opens a zero-row chest.
    #[test]
    fn zero_slot_dispatch_preserves_the_java_branch_order() {
        assert_eq!(
            container_for_open_window(&open("minecraft:furnace2", 0)),
            None
        );
        let container = container_for_open_window(&open("minecraft:container", 0))
            .expect("minecraft:container precedes hasSlots in Java");
        assert_eq!(container.slots().len(), 36);

        // Named, normally-slotted machine ids do not bypass the zero-slot branch. Java sends
        // all of these through `displayGui`, which recognises none of them.
        for kind in [
            "minecraft:hopper",
            "minecraft:dispenser",
            "minecraft:furnace",
            "minecraft:brewing_stand",
            "minecraft:beacon",
        ] {
            assert_eq!(
                container_for_open_window(&open(kind, 0)),
                None,
                "{kind} must not open when hasSlots is false"
            );
        }
    }

    /// Conversely, the three ids recognised by `displayGui` are client-built block containers
    /// only on the zero-slot path. If a server sends slots with one of those names, vanilla
    /// wraps it in `ContainerLocalMenu` and `displayGUIChest` takes its default chest branch.
    #[test]
    fn client_built_gui_ids_with_slots_use_the_chest_fallback() {
        for kind in [
            "minecraft:crafting_table",
            "minecraft:enchanting_table",
            "minecraft:anvil",
        ] {
            let container = container_for_open_window(&open(kind, 18))
                .unwrap_or_else(|| panic!("{kind} must open through the slotted fallback"));
            assert_eq!(container.slots().len(), 18 + 36, "{kind}");
            assert_eq!(
                container.kind,
                crate::screens::container::model::ContainerKind::TopThenPlayer { top: 18 },
                "{kind}"
            );
        }
    }

    /// java: `GuiChest`/`ContainerChest` both compute `getSizeInventory() / 9` with no floor.
    /// mc-rs clamped that to `1`, giving a 1..8-slot plugin menu nine phantom top slots and
    /// shifting every player slot 18px down -- so every click on it named the wrong slot id.
    #[test]
    fn a_menu_of_fewer_than_nine_slots_gets_no_container_row() {
        let c = container_for_open_window(&open("minecraft:container", 5)).expect("menu");
        assert_eq!(c.slots().len(), 36, "36 player slots and nothing else");
        assert_eq!(c.slots()[0].y, 31);
    }

    /// Finding F-P1-tileentity-06-5 is specifically the chest wire-to-layout edge. Java uses
    /// `slotCount / 9` twice (`ContainerChest` and `GuiChest`) with ordinary integer truncation,
    /// including the zero-row generic-menu case; there is no `max(1)` fallback. Keep exact legal
    /// row sizes here so a clamp mutant cannot hide behind a test that only inspects one ordinary
    /// six-row chest.
    #[test]
    fn chest_open_window_maps_slot_count_to_vanilla_rows_without_a_floor() {
        for (slot_count, rows) in [(0u8, 0usize), (9, 1), (27, 3), (54, 6)] {
            let container = container_for_open_window(&open("minecraft:container", slot_count))
                .unwrap_or_else(|| panic!("generic chest with {slot_count} slots must open"));
            assert_eq!(
                container.kind,
                ContainerKind::TopThenPlayer { top: rows * 9 },
                "slot_count={slot_count}: top slot count"
            );
            assert_eq!(
                container.slots().len(),
                rows * 9 + 36,
                "slot_count={slot_count}: top rows plus player inventory"
            );
            assert_eq!(
                container.slots()[rows * 9].y,
                103 + (rows as i32 - 4) * 18,
                "slot_count={slot_count}: player inventory follows the truncated row count"
            );
        }

        // Nonmultiples are truncated too; a 1..8-slot plugin menu must not acquire a phantom
        // top row. This is also the direct negative control for `(slot_count / 9).max(1)`.
        let partial = container_for_open_window(&open("minecraft:container", 8)).expect("menu");
        assert_eq!(partial.kind, ContainerKind::TopThenPlayer { top: 0 });
        assert_eq!(partial.slots().len(), 36);
    }

    /// java: `NetHandlerPlayClient.handleOpenWindow`'s eleven ordinary GUI ids plus the
    /// protocol's generic `minecraft:container` and the two entity-specific branches.  Keep the
    /// complete table here because a missing arm silently leaves the server-side open container
    /// alive while this client has no matching slot numbers.  The expected `ContainerKind` also
    /// makes the test bite a regression that accidentally falls through to a chest.
    #[test]
    fn every_vanilla_open_window_id_builds_its_java_container_shape() {
        let cases = [
            (
                "minecraft:container",
                54,
                ContainerKind::TopThenPlayer { top: 54 },
                90,
            ),
            (
                "minecraft:chest",
                27,
                ContainerKind::TopThenPlayer { top: 27 },
                63,
            ),
            ("minecraft:crafting_table", 0, ContainerKind::Workbench, 46),
            ("minecraft:furnace", 3, ContainerKind::Furnace, 39),
            (
                "minecraft:dispenser",
                9,
                ContainerKind::TopThenPlayer { top: 9 },
                45,
            ),
            (
                "minecraft:dropper",
                9,
                ContainerKind::TopThenPlayer { top: 9 },
                45,
            ),
            (
                "minecraft:enchanting_table",
                0,
                ContainerKind::Enchanting,
                38,
            ),
            (
                "minecraft:brewing_stand",
                4,
                ContainerKind::BrewingStand,
                40,
            ),
            ("minecraft:beacon", 1, ContainerKind::Beacon, 37),
            (
                "minecraft:hopper",
                5,
                ContainerKind::TopThenPlayer { top: 5 },
                41,
            ),
            ("minecraft:anvil", 0, ContainerKind::Anvil, 39),
            ("minecraft:villager", 3, ContainerKind::Merchant, 39),
        ];
        for (inventory_type, slot_count, expected_kind, expected_slots) in cases {
            let mut pkt = open(inventory_type, slot_count);
            pkt.window_id = 41;
            if inventory_type == "minecraft:villager" {
                pkt.entity_id = Some(17);
            }
            let container = container_for_open_window(&pkt)
                .unwrap_or_else(|| panic!("{inventory_type} must build a container"));
            assert_eq!(container.window_id, 41, "{inventory_type}: window id");
            assert_eq!(container.kind, expected_kind, "{inventory_type}: kind");
            assert_eq!(
                container.slots().len(),
                expected_slots,
                "{inventory_type}: slots"
            );
        }

        for (slot_count, expected_kind, expected_slots) in [
            (2, ContainerKind::Horse { top: 2 }, 38),
            (17, ContainerKind::Horse { top: 17 }, 53),
        ] {
            let mut pkt = open("EntityHorse", slot_count);
            pkt.window_id = 42;
            pkt.entity_id = Some(99);
            let container = container_for_open_window(&pkt).expect("horse must build");
            assert_eq!(container.kind, expected_kind);
            assert_eq!(container.slots().len(), expected_slots);
        }
    }

    /// java: `handleSetSlot` treats window `-1` as `InventoryPlayer.itemStack`, independently of
    /// the active screen.  Ordinary player-window slots 36..44 are the nine hotbar slots; keep
    /// both halves of that routing in this unit so a future screen gate cannot drop either one.
    #[test]
    fn cursor_and_hotbar_set_slot_packets_survive_high_window_ids() {
        let marker = stack(264, 1);
        for window_id in 36u8..=45 {
            let mut packet = open("minecraft:chest", 27);
            packet.window_id = window_id;
            let mut screen = container_for_open_window(&packet).expect("chest screen");
            screen.apply_set_slot(&s2c::SetSlot {
                window_id: -1,
                slot: -1,
                item: Some(marker.clone()),
            });
            assert_eq!(screen.cursor(), Some(&marker), "window {window_id}: cursor");
        }

        let mut player = player_container(PLAYER_WINDOW_ID);
        for (offset, window_slot) in (36usize..45).enumerate() {
            let item = stack(300 + offset as i16, 1);
            player.apply_set_slot(&s2c::SetSlot {
                window_id: 0,
                slot: window_slot as i16,
                item: Some(item.clone()),
            });
            assert_eq!(
                player.slot_stack(window_slot),
                Some(&item),
                "hotbar slot {window_slot}"
            );
        }
    }
}
