//! Menus, chat and the inventory in replays.
//!
//! A replay is the server's packets plus the client records. Two things the player sees were
//! in neither:
//!
//! * **The screens** the player had open (the inventory, chests, chat with what was being typed,
//!   the pause menu, the Aero mod menu, the others) and the mouse pointer over them. Playback
//!   closed every screen, so a replay never showed them.
//! * **What the player's own clicks did to the inventory.** A vanilla server never echoes an
//!   accepted click back (`NetHandlerPlayServer.processClickWindow` wraps
//!   `detectAndSendChanges()` in `isChangingQuantityOnly = true`), so after the player moved
//!   anything by hand (a potion to the hotbar, loot out of a chest) the replay's inventory was
//!   wrong until the server happened to resend it. The hand then showed the wrong item: a
//!   sword while the player was really eating or drinking.
//!
//! Two record kinds fix both ([`KIND_INVENTORY`], [`KIND_SCREEN`]), written from `build_ui`
//! while recording, after that frame's screen input. Playback applies them in stream order
//! (window-0 slots straight into `Client::player_container`; the screen into
//! [`GuiPlayback`]), and in the Player camera draws the recorded screen with the live drawing
//! code: `build_ui` swaps a display-only copy into `self.ui` (and the recorded pointer into
//! `self.cursor`) for the drawing and swaps the viewer's own back right after, so no input ever
//! reaches the copy.
//!
//! Old replays have neither kind and play exactly as before.
//!
//! # The inventory's layout
//!
//! Where the player's own inventory sits depends on a setting: with a potion effect active,
//! vanilla shifts it right to make room for the effect list (`InventoryEffectRenderer`), and the
//! `centered_inventory` mod keeps it centred (and can hide the list). The live client applies
//! that in the frame's mod hooks (`m2_frame`), before `build_ui`; the display copy is swapped in
//! only inside `build_ui`, so it was always drawn with vanilla's layout, and the recorded pointer
//! (stored relative to the screen's centre) then hovered the wrong slot. The SCREEN record now
//! carries the recording client's layout ([`EffectLayout`]: with the first record, then whenever
//! it changes), and the display copy is laid out with it every frame it is shown. A replay
//! recorded before that uses the viewer's own setting, the best guess for the same player.
//!
//! # Cost
//!
//! Not recording and not watching: one `Option` test and three `bool` tests per frame.
//! Recording: a compare of the 45 inventory slots per frame (plus the open container's), and a
//! record only when something changed.

use std::time::{Duration, Instant};

use mc_protocol::play::s2c::OpenWindow;
use mc_protocol::{ItemStack, PacketBuffer};
use mc_ui::screen_stack::ScreenCtx;
use mc_ui::screens::chat::ChatScreen;
use mc_ui::screens::container::ContainerScreen;
use mc_ui::widget::DrawList;

use super::super::super::{Client, Ui};
use super::codec::{Reader, Writer};
use super::ReplayCameraMode;

/// The INVENTORY record: window-0 slots that changed, as absolute values.
pub(crate) const KIND_INVENTORY: u8 = 6;
/// The SCREEN record: the open screen, the pointer, the open container's slots.
pub(crate) const KIND_SCREEN: u8 = 7;

/// Window 0's slot count (crafting result, 2x2 grid, armour, main inventory, hotbar).
const PLAYER_SLOTS: usize = 45;
/// A container slot index that means "the stack on the pointer".
const CARRIED: u16 = u16::MAX;
/// Pointer-only updates are written at most this often (screen changes always are).
const POINTER_INTERVAL: Duration = Duration::from_millis(2);
/// The mod option that turns the recorded screens off in the viewer.
const OPTION_SHOW: &str = mc_mods::mods::replay::ids::SHOW_SCREENS;
/// The mod option that decides how much typed chat text is recorded.
const OPTION_CHAT_TEXT: &str = mc_mods::mods::replay::ids::CHAT_TEXT;

// ---------------------------------------------------------------------------------------------
// Records
// ---------------------------------------------------------------------------------------------

/// An item stack in the protocol's own encoding (`writeItemStackToBuffer`), length-prefixed so a
/// stack this build cannot read is skipped rather than misreading the rest of the record.
fn write_stack(w: &mut Writer<'_>, stack: Option<&ItemStack>) {
    let mut buf = PacketBuffer::new();
    buf.write_item_stack(stack);
    let bytes = buf.as_slice();
    w.varint(bytes.len() as u64);
    for byte in bytes {
        w.u8(*byte);
    }
}

fn read_stack(r: &mut Reader<'_>) -> Option<Option<ItemStack>> {
    let len = r.varint() as usize;
    let mut bytes = Vec::with_capacity(len.min(4096));
    for _ in 0..len {
        bytes.push(r.u8());
        // Stop at the end of the record: the length is a
        // varint up to 2^64, and the reader answers zeros once short, so a shared replay's one
        // record made this push zeros until memory ran out.
        if r.short() {
            return None;
        }
    }
    PacketBuffer::from_slice(&bytes).read_item_stack().ok()
}

/// [`KIND_INVENTORY`]: window-0 slots (`player_container`) whose contents changed since the last
/// one, or all 45 at the start of a recording.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct InventoryRecord {
    pub(crate) slots: Vec<(u8, Option<ItemStack>)>,
}

impl InventoryRecord {
    const VERSION: u8 = 1;

    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        w.varint(self.slots.len() as u64);
        for (slot, stack) in &self.slots {
            w.u8(*slot);
            write_stack(&mut w, stack.as_ref());
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        let count = (r.varint() as usize).min(PLAYER_SLOTS * 4);
        let mut slots = Vec::with_capacity(count);
        for _ in 0..count {
            let slot = r.u8();
            let stack = read_stack(&mut r)?;
            slots.push((slot, stack));
        }
        (!r.short()).then_some(Self { slots })
    }
}

/// What screen the recording client had up.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) enum ScreenState {
    /// No screen: the game.
    #[default]
    None,
    /// The chat box, with what was typed (see [`ChatText`]).
    Chat { text: String },
    /// The player's own inventory (`GuiInventory`, window 0).
    Inventory,
    /// A server-opened container (chest, furnace, ...).
    Container { window_id: u8 },
    /// The creative inventory (recorded, not drawn).
    Creative,
    /// The pause menu.
    Pause,
    /// The Aero mod menu.
    ModMenu,
    /// Any other screen, by its `Ui::name`.
    Other { name: String },
}

impl ScreenState {
    fn encode(&self, w: &mut Writer<'_>) {
        match self {
            ScreenState::None => w.u8(0),
            ScreenState::Chat { text } => {
                w.u8(1);
                w.str(text);
            }
            ScreenState::Inventory => w.u8(2),
            ScreenState::Container { window_id } => {
                w.u8(3);
                w.u8(*window_id);
            }
            ScreenState::Creative => w.u8(4),
            ScreenState::Pause => w.u8(5),
            ScreenState::ModMenu => w.u8(6),
            ScreenState::Other { name } => {
                w.u8(7);
                w.str(name);
            }
        }
    }

    fn decode(r: &mut Reader<'_>) -> Self {
        match r.u8() {
            1 => ScreenState::Chat { text: r.str() },
            2 => ScreenState::Inventory,
            3 => ScreenState::Container { window_id: r.u8() },
            4 => ScreenState::Creative,
            5 => ScreenState::Pause,
            6 => ScreenState::ModMenu,
            7 => ScreenState::Other { name: r.str() },
            // 0, or a screen kind a newer build added: shown as no screen.
            _ => ScreenState::None,
        }
    }
}

/// How the recording client laid out its own inventory beside active potion effects: the
/// `centered_inventory` mod's arguments to [`ContainerScreen::set_effect_layout`]. Vanilla's
/// layout (`InventoryEffectRenderer`: shifted right, the effect list beside it) is
/// `centered: false, show_effects: true`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EffectLayout {
    /// The inventory stays centred while effects are active.
    pub(crate) centered: bool,
    /// The effect list is drawn beside it.
    pub(crate) show_effects: bool,
}

impl EffectLayout {
    /// One byte, vanilla's being 0: bit 0 centred, bit 1 effect list hidden. Other bits are
    /// left for a newer build and ignored.
    fn to_byte(self) -> u8 {
        u8::from(self.centered) | (u8::from(!self.show_effects) << 1)
    }

    fn from_byte(byte: u8) -> Self {
        Self {
            centered: byte & 1 != 0,
            show_effects: byte & 2 == 0,
        }
    }
}

/// [`KIND_SCREEN`]: what changed about the open screen this frame.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct ScreenRecord {
    /// The screen, when it (or its text) changed.
    pub(crate) state: Option<ScreenState>,
    /// The pointer, in GUI pixels from the centre of the screen, when it moved.
    pub(crate) pointer: Option<(i16, i16)>,
    /// Container slots that changed (absolute values; [`CARRIED`] is the stack on the pointer).
    pub(crate) slots: Vec<(u16, Option<ItemStack>)>,
    /// The recording client's inventory layout, when it changed (and in a recording's first
    /// record). Written last, after the slots: a build that predates it reads the record's other
    /// parts as before and ignores the trailing byte; a replay that predates it has none.
    pub(crate) layout: Option<EffectLayout>,
}

impl ScreenRecord {
    const VERSION: u8 = 1;
    const HAS_STATE: u8 = 1;
    const HAS_POINTER: u8 = 2;
    const HAS_SLOTS: u8 = 4;
    const HAS_LAYOUT: u8 = 8;

    pub(crate) fn encode_into(&self, out: &mut Vec<u8>) {
        let mut w = Writer::reuse(out);
        w.u8(Self::VERSION);
        let mut flags = 0;
        if self.state.is_some() {
            flags |= Self::HAS_STATE;
        }
        if self.pointer.is_some() {
            flags |= Self::HAS_POINTER;
        }
        if !self.slots.is_empty() {
            flags |= Self::HAS_SLOTS;
        }
        if self.layout.is_some() {
            flags |= Self::HAS_LAYOUT;
        }
        w.u8(flags);
        if let Some(state) = &self.state {
            state.encode(&mut w);
        }
        if let Some((dx, dy)) = self.pointer {
            w.svarint(i64::from(dx));
            w.svarint(i64::from(dy));
        }
        if !self.slots.is_empty() {
            w.varint(self.slots.len() as u64);
            for (slot, stack) in &self.slots {
                w.varint(u64::from(*slot));
                write_stack(&mut w, stack.as_ref());
            }
        }
        if let Some(layout) = self.layout {
            w.u8(layout.to_byte());
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Option<Self> {
        let mut r = Reader::new(bytes);
        if r.u8() == 0 {
            return None;
        }
        let flags = r.u8();
        let state = (flags & Self::HAS_STATE != 0).then(|| ScreenState::decode(&mut r));
        let pointer = (flags & Self::HAS_POINTER != 0).then(|| {
            let dx = r.svarint().clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
            let dy = r.svarint().clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16;
            (dx, dy)
        });
        let mut slots = Vec::new();
        if flags & Self::HAS_SLOTS != 0 {
            let count = (r.varint() as usize).min(1024);
            slots.reserve(count);
            for _ in 0..count {
                let slot = r.varint().min(u64::from(u16::MAX)) as u16;
                let stack = read_stack(&mut r)?;
                slots.push((slot, stack));
            }
        }
        let layout = (flags & Self::HAS_LAYOUT != 0).then(|| EffectLayout::from_byte(r.u8()));
        (!r.short()).then_some(Self {
            state,
            pointer,
            slots,
            layout,
        })
    }
}

/// One decoded record of this module's kinds, for playback's work queue.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum GuiRecord {
    Inventory(InventoryRecord),
    Screen(ScreenRecord),
}

impl GuiRecord {
    /// Decode a `StreamId::Client(kind)` payload of one of this module's kinds.
    pub(crate) fn decode(kind: u8, bytes: &[u8]) -> Option<Self> {
        match kind {
            KIND_INVENTORY => InventoryRecord::decode(bytes).map(GuiRecord::Inventory),
            KIND_SCREEN => ScreenRecord::decode(bytes).map(GuiRecord::Screen),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Typed chat: privacy
// ---------------------------------------------------------------------------------------------

/// How much of the typed (not yet sent) chat text a recording keeps
/// (`replay.chat_text`: "Everything", "Hide commands", "Nothing").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChatText {
    /// Every character.
    Everything,
    /// Chat as typed, but a command keeps only its name: `/login hunter2` is recorded as
    /// `/login *******`. The default: a replay is made to be shared, and commands are where
    /// passwords (`/login`, `/register`) and private messages (`/msg`, `/r`) are typed.
    HideCommands,
    /// The chat box is shown open and empty.
    Nothing,
}

impl ChatText {
    pub(crate) fn from_choice(choice: usize) -> Self {
        match choice {
            0 => ChatText::Everything,
            2 => ChatText::Nothing,
            _ => ChatText::HideCommands,
        }
    }

    /// What the recording keeps of `typed`.
    pub(crate) fn apply(self, typed: &str) -> String {
        match self {
            ChatText::Everything => typed.to_owned(),
            ChatText::Nothing => String::new(),
            ChatText::HideCommands => {
                if !typed.starts_with('/') {
                    return typed.to_owned();
                }
                // The command word (and the space after it) stays readable; every character of
                // its arguments, spaces included (so not even the number of words shows),
                // becomes a star.
                let Some(space) = typed.find(' ') else {
                    return typed.to_owned();
                };
                let mut out = String::with_capacity(typed.len());
                out.push_str(&typed[..=space]);
                out.extend(typed[space + 1..].chars().map(|_| '*'));
                out
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Recording side
// ---------------------------------------------------------------------------------------------

/// What this module last wrote while recording (the `Capture::gui`).
#[derive(Debug, Default)]
pub(crate) struct GuiCapture {
    /// The recorder generation these values belong to; a new one starts with full snapshots.
    generation: Option<u64>,
    /// Window-0 slots as last written (`None`: none written yet).
    inventory: Option<Vec<Option<ItemStack>>>,
    /// The screen as last written.
    screen: Option<ScreenState>,
    /// The pointer as last written, and when.
    pointer: Option<(i16, i16)>,
    pointer_at: Option<Instant>,
    /// The open container's slots as last written (window != 0), and the carried stack.
    container: Vec<Option<ItemStack>>,
    carried: Option<ItemStack>,
    /// The inventory layout as last written.
    layout: Option<EffectLayout>,
}

impl Client {
    /// The layout this client gives its own open inventory now: `m2_frame`'s (the
    /// `centered_inventory` mod), vanilla's while the mod is off.
    fn nw_rb_live_layout(&mut self) -> EffectLayout {
        self.m2_sync();
        let (centered, show_effects) = self.m2.inventory_layout();
        EffectLayout {
            centered,
            show_effects,
        }
    }

    /// What screen is up, for the recording.
    fn nw_rb_observed_screen(&self) -> ScreenState {
        match &self.ui {
            Ui::InGame | Ui::Loading => ScreenState::None,
            Ui::Chat(screen) => {
                let policy =
                    ChatText::from_choice(self.mods.settings.choice(MOD_ID, OPTION_CHAT_TEXT));
                ScreenState::Chat {
                    text: policy.apply(&screen.input.text()),
                }
            }
            Ui::Container(screen) => {
                if screen.container.window_id == 0
                    && screen.art == mc_ui::screens::container::ContainerArt::Player
                {
                    ScreenState::Inventory
                } else {
                    ScreenState::Container {
                        window_id: screen.container.window_id,
                    }
                }
            }
            Ui::Creative(_) => ScreenState::Creative,
            Ui::Pause(_) => ScreenState::Pause,
            Ui::ModMenu(_) => ScreenState::ModMenu,
            other => ScreenState::Other {
                name: other.name().to_owned(),
            },
        }
    }

    /// While recording: write the inventory slots and the screen that changed this frame.
    /// From `build_ui`, after this frame's screen input, so playback applies them before the
    /// next tick, as they happened.
    fn nw_rb_capture_gui(&mut self, width: u32, height: u32) {
        let generation = self.tw_rb_generation();
        if self.tw_play.capture.gui.generation != Some(generation) {
            self.tw_play.capture.gui = GuiCapture {
                generation: Some(generation),
                ..GuiCapture::default()
            };
        }
        if self.tw_rb_paused() {
            return;
        }
        // ---- the inventory (window 0) ----
        let slots = self.player_container.slots().len().min(PLAYER_SLOTS);
        let capture = &mut self.tw_play.capture.gui;
        let first = capture.inventory.is_none();
        let written = capture
            .inventory
            .get_or_insert_with(|| vec![None; PLAYER_SLOTS]);
        let changed =
            first || (0..slots).any(|i| self.player_container.slot_stack(i) != written[i].as_ref());
        if changed {
            let mut record = InventoryRecord::default();
            for (i, last) in written.iter_mut().enumerate().take(slots) {
                let now = self.player_container.slot_stack(i);
                if first || now != last.as_ref() {
                    *last = now.cloned();
                    record.slots.push((i as u8, now.cloned()));
                }
            }
            self.tw_rb_push_with(KIND_INVENTORY, |out| record.encode_into(out));
        }
        // ---- the screen ----
        // An unchanged "other" screen is recognised by its static name, without building its
        // state (the chat's text, read through `TextField::text`, is one short string a frame
        // while the chat box is open).
        let unchanged_other = matches!(
            (&self.tw_play.capture.gui.screen, &self.ui),
            (Some(ScreenState::Other { name }), ui) if name == ui.name()
        );
        let state = (!unchanged_other).then(|| self.nw_rb_observed_screen());
        let screen_open = unchanged_other
            || state
                .as_ref()
                .is_some_and(|state| *state != ScreenState::None);
        let pointer = screen_open.then(|| {
            let screen_ctx = ScreenCtx::new(self.screen_resolution(width, height), 0.0);
            let dx = self.cursor.0 - screen_ctx.width / 2;
            let dy = self.cursor.1 - screen_ctx.height / 2;
            (
                dx.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
                dy.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16,
            )
        });
        let now = Instant::now();
        // The in-memory recorder keeps every pointer move, as it keeps every frame record (the
        // fidelity harness compares frames one by one); a file keeps one per 2 ms.
        let throttle = self.tw_play.capture.shim.is_none();
        // A settings-generation compare and two `bool` reads; written
        // with the recording's first record and whenever the setting changes.
        let layout = self.nw_rb_live_layout();
        let mut record = ScreenRecord::default();
        let capture = &mut self.tw_play.capture.gui;
        if capture.layout != Some(layout) {
            capture.layout = Some(layout);
            record.layout = Some(layout);
        }
        let state = state.filter(|state| capture.screen.as_ref() != Some(state));
        let state_changed = state.is_some();
        if let Some(state) = state {
            // A new screen starts its container and carried stack afresh (full snapshot).
            capture.container.clear();
            capture.carried = None;
            capture.screen = Some(state.clone());
            record.state = Some(state);
        }
        if pointer.is_some()
            && pointer != capture.pointer
            && (state_changed
                || !throttle
                || capture
                    .pointer_at
                    .is_none_or(|at| now.duration_since(at) >= POINTER_INTERVAL))
        {
            capture.pointer = pointer;
            capture.pointer_at = Some(now);
            record.pointer = pointer;
        }
        if let Ui::Container(screen) = &self.ui {
            let container = &screen.container;
            if container.window_id != 0 {
                let count = container.slots().len();
                if capture.container.len() != count {
                    capture.container = vec![None; count];
                    // Every slot is written once, empty ones included, when the window opens.
                    for i in 0..count {
                        let now = container.slot_stack(i).cloned();
                        capture.container[i].clone_from(&now);
                        record.slots.push((i as u16, now));
                    }
                } else {
                    for (i, last) in capture.container.iter_mut().enumerate() {
                        let now = container.slot_stack(i);
                        if now != last.as_ref() {
                            *last = now.cloned();
                            record.slots.push((i as u16, now.cloned()));
                        }
                    }
                }
            }
            if container.cursor() != capture.carried.as_ref() {
                capture.carried = container.cursor().cloned();
                record.slots.push((CARRIED, capture.carried.clone()));
            }
        }
        if record.state.is_some()
            || record.pointer.is_some()
            || !record.slots.is_empty()
            || record.layout.is_some()
        {
            self.tw_rb_push_with(KIND_SCREEN, |out| record.encode_into(out));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Playback side
// ---------------------------------------------------------------------------------------------

/// The recorded screen as of where playback is (`Playback::gui`). Plain data, so a checkpoint
/// can copy it; the drawn copy is rebuilt from it.
#[derive(Debug, Clone, Default)]
pub(crate) struct GuiPlayback {
    /// The screen the recording client had up.
    pub(crate) screen: ScreenState,
    /// The pointer, in GUI pixels from the centre of the screen.
    pub(crate) pointer: (i16, i16),
    /// The open container's slots (window != 0) and the stack on the pointer.
    pub(crate) container: Vec<Option<ItemStack>>,
    pub(crate) carried: Option<ItemStack>,
    /// The last `OpenWindow` the replay applied: what a server container screen is drawn from.
    pub(crate) open_window: Option<OpenWindow>,
    /// Bumped whenever the recorded screen itself changes (kind, text, window): the drawn copy
    /// is built afresh, as the live client built a new screen.
    pub(crate) revision: u64,
    /// Bumped whenever the contents it shows change (inventory slots, container slots, the
    /// carried stack): the drawn copy is updated in place, as the live screen was.
    pub(crate) content_revision: u64,
    /// The client tick the screen opened at: a screen's own tick-driven animation (the chat
    /// caret's blink) is as old as the recording's was.
    pub(crate) opened_tick: u64,
    /// The recording client's inventory layout as of here; `None` for a replay
    /// recorded before SCREEN records carried it, which is shown with the viewer's own.
    pub(crate) layout: Option<EffectLayout>,
    /// The drawn copy, kept between frames while nothing changes. Not copied with the rest.
    display: GuiDisplay,
}

/// The display-only copy of the recorded screen. It lives as long as the recorded screen stays
/// up, like the live one: the inventory's player preview looks at the previous frame's pointer
/// (`GuiInventory.oldMouseX`) and its idle animation and the chat caret age with the ticks.
#[derive(Default)]
struct GuiDisplay {
    built: Option<Box<Ui>>,
    /// The [`GuiPlayback::revision`] and `content_revision` it shows.
    revision: u64,
    content_revision: u64,
    /// The client tick it has been ticked to.
    ticked: u64,
}

impl Clone for GuiDisplay {
    fn clone(&self) -> Self {
        GuiDisplay::default()
    }
}

impl std::fmt::Debug for GuiDisplay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuiDisplay")
            .field("built", &self.built.is_some())
            .finish()
    }
}

/// What `build_ui`'s begin hook swapped out, for its end hook to put back.
pub(crate) struct GuiSwap {
    /// The viewer's own screen (`None`: nothing was swapped, only drawn over).
    ui: Option<Ui>,
    cursor: (i32, i32),
    /// The recorded screen, for the end hook's label (screens drawn as "other").
    label: Option<&'static str>,
}

/// The id of the mod whose options these are.
const MOD_ID: &str = "replay";

impl Client {
    /// Playback: apply one of this module's records, in stream order.
    pub(crate) fn nw_rb_apply_gui(&mut self, record: GuiRecord) {
        match record {
            GuiRecord::Inventory(record) => {
                let slots = self.player_container.slots().len();
                let mut changed = false;
                for (slot, stack) in record.slots {
                    let index = usize::from(slot);
                    if index >= slots || self.player_container.slot_stack(index) == stack.as_ref() {
                        continue;
                    }
                    // Java splits and merges a stack in place (the same object, a new count),
                    // and the hand only re-equips for a different object of different value:
                    // keep the slot's identity when the item and its damage stay the same.
                    let same_item = matches!(
                        (self.player_container.slot_stack(index), stack.as_ref()),
                        (Some(old), Some(new)) if old.item_id == new.item_id
                            && old.damage == new.damage
                    );
                    if same_item {
                        self.player_container
                            .set_slot_contents_in_place(index, stack);
                    } else {
                        self.player_container.set_slot_contents(index, stack);
                    }
                    changed = true;
                }
                if changed {
                    self.refresh_hotbar();
                    self.refresh_armor();
                    if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                        playback.gui.content_revision += 1;
                    }
                }
            }
            GuiRecord::Screen(record) => {
                let tick = self.tick;
                let Some(playback) = self.tw_play.playback.as_deref_mut() else {
                    return;
                };
                let gui = &mut playback.gui;
                let mut opened = false;
                if let Some(state) = record.state {
                    // Another screen (not the chat's text changing): it opened afresh.
                    if !same_screen(&gui.screen, &state) {
                        gui.container.clear();
                        gui.carried = None;
                        gui.opened_tick = tick;
                        opened = state != ScreenState::None;
                    }
                    gui.screen = state;
                    gui.revision += 1;
                    gui.content_revision += 1;
                }
                if let Some(pointer) = record.pointer {
                    gui.pointer = pointer;
                }
                if record.layout.is_some() {
                    gui.layout = record.layout;
                }
                for (slot, stack) in record.slots {
                    if slot == CARRIED {
                        gui.carried = stack;
                    } else {
                        let index = usize::from(slot);
                        if gui.container.len() <= index {
                            gui.container.resize(index + 1, None);
                        }
                        gui.container[index] = stack;
                    }
                    gui.content_revision += 1;
                }
                if opened {
                    self.nw_rb_replay_modal_input();
                }
            }
        }
    }

    /// A recorded screen opened: what the live client's `show_screen` did to the game's input
    /// then (`prepare_modal_input`), as far as the replayed game is concerned. java:
    /// `displayGuiScreen` → `setIngameNotInFocus` (`KeyBinding.unPressAllKeys`) and
    /// `leftClickCounter = 10000`, so an attack pressed under an inventory neither digs nor
    /// swings, and a dig in progress is aborted. The viewer's own keyboard and focus state are
    /// left alone.
    fn nw_rb_replay_modal_input(&mut self) {
        self.pending_unpress = true;
        // 1.7.10 support: a 1.7.10 replay resets as 1.7.10 does (`app/interact_v5.rs`).
        self.reset_block_removing_for_version();
        self.controller.lock_out();
    }

    /// Whether the viewer shows recorded screens (the replay mod's "Menus and chat in replays").
    fn nw_rb_show_screens(&self) -> bool {
        self.mods.settings.bool(MOD_ID, OPTION_SHOW)
    }

    /// Build the display-only copy of the recorded screen. `None` when it is not drawn as a
    /// screen (no screen, the creative inventory, a container whose window was opened before
    /// the replay starts).
    fn nw_rb_build_display(&self) -> Option<Ui> {
        let gui = &self.tw_play.playback.as_deref()?.gui;
        match &gui.screen {
            ScreenState::Chat { text } => {
                let limit = i32::try_from(text.encode_utf16().count())
                    .unwrap_or(i32::MAX)
                    .max(256);
                let screen = ChatScreen::with_input_limit(text, 0, &self.font, limit);
                Some(Ui::Chat(Box::new(screen)))
            }
            ScreenState::Inventory => {
                let mut screen = self.new_player_inventory_screen();
                screen.container.set_cursor(gui.carried.clone());
                Some(Ui::Container(Box::new(screen)))
            }
            ScreenState::Container { window_id } => {
                let open = gui
                    .open_window
                    .as_ref()
                    .filter(|open| open.window_id == *window_id)?;
                let title = mc_ui::font::component::ChatComponent::from_json(&open.window_title)
                    .map(|component| component.unformatted_text(&self.assets.language))
                    .unwrap_or_else(|_| open.window_title.clone());
                let mut screen = ContainerScreen::for_open_window(open, title)?;
                screen.set_language(&self.assets.language);
                screen.advanced_tooltips = self.settings.advanced_item_tooltips;
                for (index, stack) in gui.container.iter().enumerate() {
                    if index < screen.container.slots().len() {
                        screen.container.set_slot_contents(index, stack.clone());
                    }
                }
                screen.container.set_cursor(gui.carried.clone());
                Some(Ui::Container(Box::new(screen)))
            }
            ScreenState::Pause => Some(Ui::Pause(Box::new(
                mc_ui::screens::pause::PauseScreen::new(),
            ))),
            ScreenState::None
            | ScreenState::Creative
            | ScreenState::ModMenu
            | ScreenState::Other { .. } => None,
        }
    }

    /// `build_ui`, before the cursor is read: while recording, write this frame's records;
    /// while watching in the Player camera, put the recorded screen and pointer in place for
    /// the drawing. Returns what the end hook puts back.
    pub(crate) fn nw_rb_gui_begin(&mut self, width: u32, height: u32) -> Option<GuiSwap> {
        if self.tw_play.playback.is_none() {
            if self.tw_rb_capturing() {
                self.nw_rb_capture_gui(width, height);
            }
            return None;
        }
        let playback = self.tw_play.playback.as_deref()?;
        let visible = playback.camera_mode == ReplayCameraMode::Pov
            && playback.gui.screen != ScreenState::None
            && matches!(self.ui, Ui::InGame)
            && self.nw_rb_show_screens();
        let menu = visible && playback.gui.screen == ScreenState::ModMenu;
        self.nw_rb_sync_menu(menu);
        if !visible || menu {
            return None;
        }
        let playback = self.tw_play.playback.as_deref()?;
        let (revision, content_revision, opened_tick) = (
            playback.gui.revision,
            playback.gui.content_revision,
            playback.gui.opened_tick,
        );
        let (dx, dy) = playback.gui.pointer;
        let label = match &playback.gui.screen {
            ScreenState::Creative => Some("Creative inventory"),
            ScreenState::Other { name } => Some(other_label(name)),
            _ => None,
        };
        let display = self.nw_rb_display(revision, content_revision, opened_tick);
        let ui = display.map(|display| std::mem::replace(&mut self.ui, display));
        if ui.is_none() && label.is_none() {
            // Nothing to draw after all (a container opened before the replay began).
            return None;
        }
        // Measured with the recorded screen up, as the recording
        // measured it with the live one up: an inventory has its own GUI scale when the
        // independent GUI scale mod sets one, and `build_ui` draws this copy at it.
        let screen_ctx = ScreenCtx::new(self.screen_resolution(width, height), 0.0);
        let pointer = (
            (screen_ctx.width / 2 + i32::from(dx)).clamp(0, screen_ctx.width.max(1) - 1),
            (screen_ctx.height / 2 + i32::from(dy)).clamp(0, screen_ctx.height.max(1) - 1),
        );
        let cursor = std::mem::replace(&mut self.cursor, pointer);
        Some(GuiSwap { ui, cursor, label })
    }

    /// The recorded pointer's size, in multiples of the desktop pointer's at 100%: the window's
    /// display scaling (the desktop's own pointer is that big over the window), or in a video
    /// export the same share of the frame's height as at 1080p.
    fn nw_rb_pointer_size(&self) -> f32 {
        match self.tw_export_offscreen_size() {
            Some((_, height)) => super::pointer::export_size(height),
            None => self.tw_play.pointer.window_scale(),
        }
    }

    /// The layout a recorded inventory is shown with: the recording's, or the viewer's own
    /// setting for a replay recorded before SCREEN records carried it.
    fn nw_rb_shown_layout(&mut self) -> EffectLayout {
        let recorded = self
            .tw_play
            .playback
            .as_deref()
            .and_then(|playback| playback.gui.layout);
        recorded.unwrap_or_else(|| self.nw_rb_live_layout())
    }

    /// This frame's display copy: the one already built while the recorded screen is the same
    /// (updated in place when what it shows changed, ticked up to the client's tick), a new one
    /// when the screen changed. Put back by [`Client::nw_rb_gui_end`].
    fn nw_rb_display(
        &mut self,
        revision: u64,
        content_revision: u64,
        opened_tick: u64,
    ) -> Option<Ui> {
        let (cached, shown_content, ticked) = {
            let display = &mut self.tw_play.playback.as_deref_mut()?.gui.display;
            let cached = display
                .built
                .take()
                .filter(|_| display.revision == revision);
            (cached, display.content_revision, display.ticked)
        };
        let (mut ui, ticked, fresh) = match cached {
            Some(ui) => (*ui, ticked, false),
            None => (self.nw_rb_build_display()?, opened_tick, true),
        };
        // The inventory shows `player_container`, which packets change too (the live screen got
        // those packets directly): compare it every frame (only differing slots are written).
        let inventory = matches!(&ui, Ui::Container(screen) if screen.container.window_id == 0);
        if !fresh && (inventory || shown_content != content_revision) {
            self.nw_rb_refresh_display(&mut ui);
        }
        // The live screen was ticked once per client tick from when it opened: the chat caret
        // (`cursorCounter / 6 % 2`, so only the phase matters) and the inventory preview's
        // idle animation age with it. A screen reached by a long seek starts at most a minute
        // along (its preview's breathing then differs; nothing else does).
        let behind = self.tick.saturating_sub(ticked);
        let steps = match &ui {
            Ui::Chat(_) if fresh => behind % 12,
            _ => behind.min(1_200),
        };
        // As `Client::tick` does for the open screen: the effects, then the screen's tick.
        if steps > 0 {
            if let Ui::Container(screen) = &mut ui {
                screen.set_active_effects(self.player_hud.effects.clone());
            }
        }
        for _ in 0..steps {
            match &mut ui {
                Ui::Chat(screen) => screen.tick(),
                Ui::Container(screen) => mc_ui::screen_stack::Screen::tick(screen.as_mut()),
                _ => {}
            }
        }
        // Laid out as the recording's inventory was, as `m2_frame` lays out
        // the live one before every draw (it never reaches this copy, swapped in only later).
        if let Ui::Container(screen) = &mut ui {
            let layout = self.nw_rb_shown_layout();
            screen.set_effect_layout(layout.centered, layout.show_effects);
        }
        let display = &mut self.tw_play.playback.as_deref_mut()?.gui.display;
        display.revision = revision;
        display.content_revision = content_revision;
        display.ticked = self.tick.max(ticked);
        Some(ui)
    }

    /// Put the recorded contents into a kept display copy: the inventory's from
    /// `player_container` (which the INVENTORY records keep as recorded), a server container's
    /// from the SCREEN records; and the stack on the pointer.
    fn nw_rb_refresh_display(&self, ui: &mut Ui) {
        let Some(gui) = self
            .tw_play
            .playback
            .as_deref()
            .map(|playback| &playback.gui)
        else {
            return;
        };
        let Ui::Container(screen) = ui else {
            return;
        };
        let count = screen.container.slots().len();
        for index in 0..count {
            let stack = if screen.container.window_id == 0 {
                self.player_container.slot_stack(index).cloned()
            } else {
                gui.container.get(index).cloned().flatten()
            };
            if screen.container.slot_stack(index) != stack.as_ref() {
                screen.container.set_slot_contents(index, stack);
            }
        }
        screen.container.set_cursor(gui.carried.clone());
    }

    /// `build_ui`, after the screen was drawn (in the screen's own GUI space): the recorded
    /// pointer, a screen drawn only as a label, then the viewer's own screen and cursor back.
    pub(crate) fn nw_rb_gui_end(
        &mut self,
        swap: Option<GuiSwap>,
        list: &mut DrawList,
        ctx: &ScreenCtx,
    ) {
        let Some(swap) = swap else {
            return;
        };
        if let Some(label) = swap.label {
            // java: `GuiScreen.drawWorldBackground`, the gradient every in-game menu draws first.
            list.gradient_rect(0, 0, ctx.width, ctx.height, 0xC010_1010, 0xD010_1010);
            mc_ui::widget::draw_centered_string(
                list,
                &mut self.font,
                label,
                ctx.width / 2,
                ctx.height / 2 - 4,
                0x00FF_FFFF,
            );
        }
        if !self.tw_play.pointer_hidden {
            // The desktop's standard arrow, in the target's own pixels.
            let size = self.nw_rb_pointer_size();
            let gui_scale = ctx.resolution.scale_factor();
            self.tw_play
                .pointer
                .draw(list, self.cursor, gui_scale, size);
        }
        self.cursor = swap.cursor;
        if let Some(ui) = swap.ui {
            let display = std::mem::replace(&mut self.ui, ui);
            if let Some(playback) = self.tw_play.playback.as_deref_mut() {
                playback.gui.display.built = Some(Box::new(display));
            }
        }
    }

    /// The recorded Aero mod menu: a display-only menu, fully open from its first frame (no
    /// wall-clock opening animation, so an export does not depend on timing), in the slot the
    /// menu's own phase draws a closing menu from. Removed at once when the recorded screen
    /// changes, the camera leaves the player, or the viewer opens its own screen.
    fn nw_rb_sync_menu(&mut self, show: bool) {
        if self.tw_play.menu_shown == show {
            return;
        }
        self.tw_play.menu_shown = show;
        if show {
            let now = Instant::now();
            let opened = now.checked_sub(Duration::from_secs(10)).unwrap_or(now);
            let screen =
                mc_ui::screens::modmenu::ModMenuScreen::new(self.mods.settings.catalog(), opened);
            self.mods.closing_menu = Some(Box::new(screen));
            self.mods.menu_live = true;
        } else if !matches!(self.ui, Ui::ModMenu(_)) {
            self.mods.closing_menu = None;
        }
    }

    /// The screen a replayed tick runs under: the recorded inventory (a copy of the current
    /// one) while the recording had its inventory open, `None` otherwise (the game). The game's
    /// own input differs with a screen up (`handle_game_input` digs and opens the command chat
    /// only with no screen), so a tick the recording ran under its inventory must not run as
    /// "in game". The copy holds exactly `player_container`, so the tick's window-0 sync
    /// (`drain_container_clicks`) changes nothing.
    pub(crate) fn nw_rb_tick_screen(&self) -> Option<Ui> {
        let playback = self.tw_play.playback.as_deref()?;
        matches!(
            playback.gui.screen,
            ScreenState::Inventory | ScreenState::Creative
        )
        .then(|| Ui::Container(Box::new(self.new_player_inventory_screen())))
    }

    /// Whether the recorded mouse pointer is drawn over recorded screens (default: yes). A live
    /// frame never contains it (the system draws the real pointer), so the fidelity harness
    /// turns it off to compare replayed frames with live ones pixel for pixel.
    pub fn replay_set_pointer_visible(&mut self, visible: bool) {
        self.tw_play.pointer_hidden = !visible;
    }

    /// The replay closes or rebuilds its world: no recorded screen may outlive it (the
    /// display-only Aero menu would otherwise stay up over the title screen).
    pub(crate) fn nw_rb_drop_gui(&mut self) {
        self.nw_rb_sync_menu(false);
        if let Some(playback) = self.tw_play.playback.as_deref_mut() {
            playback.gui = GuiPlayback::default();
        }
    }
}

/// Whether `b` is the same screen as `a` (the chat with other text still is).
fn same_screen(a: &ScreenState, b: &ScreenState) -> bool {
    match (a, b) {
        (ScreenState::Chat { .. }, ScreenState::Chat { .. }) => true,
        _ => a == b,
    }
}

/// A readable name for a screen drawn only as a label.
fn other_label(name: &str) -> &'static str {
    match name {
        "options" | "video_settings" | "sound_options" | "controls" | "language"
        | "resource_packs" | "chat_options" | "skin_customisation" | "snooper" => "Options",
        "game_over" => "You died!",
        "sleep" => "Sleeping",
        "edit_sign" => "Editing a sign",
        "book" => "Reading a book",
        "achievements" => "Achievements",
        "stats" => "Statistics",
        "command_block" => "Command block",
        _ => "Menu",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack(item_id: i16, count: i32, damage: i16) -> ItemStack {
        ItemStack {
            item_id,
            count,
            damage,
            nbt: None,
        }
    }

    #[test]
    fn inventory_records_round_trip_and_reject_truncation() {
        let record = InventoryRecord {
            slots: vec![
                (36, Some(stack(276, 1, 0))),
                (40, None),
                (44, Some(stack(373, 1, 16421))),
            ],
        };
        let mut bytes = Vec::new();
        record.encode_into(&mut bytes);
        assert_eq!(InventoryRecord::decode(&bytes), Some(record));
        assert_eq!(InventoryRecord::decode(&bytes[..bytes.len() - 2]), None);
        // A newer build's trailing fields are ignored.
        let mut longer = bytes.clone();
        longer.extend_from_slice(&[9, 9, 9]);
        assert!(InventoryRecord::decode(&longer).is_some());
    }

    #[test]
    fn screen_records_round_trip_every_part() {
        for state in [
            ScreenState::None,
            ScreenState::Chat {
                text: "gg wp".to_owned(),
            },
            ScreenState::Inventory,
            ScreenState::Container { window_id: 7 },
            ScreenState::Creative,
            ScreenState::Pause,
            ScreenState::ModMenu,
            ScreenState::Other {
                name: "options".to_owned(),
            },
        ] {
            let record = ScreenRecord {
                state: Some(state),
                pointer: Some((-120, 33)),
                slots: vec![
                    (0, Some(stack(1, 64, 0))),
                    (CARRIED, Some(stack(322, 3, 1))),
                ],
                layout: Some(EffectLayout {
                    centered: true,
                    show_effects: false,
                }),
            };
            let mut bytes = Vec::new();
            record.encode_into(&mut bytes);
            assert_eq!(ScreenRecord::decode(&bytes), Some(record));
        }
        // A pointer-only update is a few bytes.
        let pointer = ScreenRecord {
            pointer: Some((5, -5)),
            ..ScreenRecord::default()
        };
        let mut bytes = Vec::new();
        pointer.encode_into(&mut bytes);
        assert!(bytes.len() <= 4, "{} bytes", bytes.len());
        assert_eq!(ScreenRecord::decode(&bytes), Some(pointer));
        // An unknown screen kind from a newer build reads as no screen, not as garbage.
        assert_eq!(
            ScreenRecord::decode(&[1, 1, 200]).and_then(|record| record.state),
            Some(ScreenState::None)
        );
    }

    /// The layout is one byte after everything else, behind its own flag. A build
    /// from before it reads the flags it knows (state, pointer, slots) in the same places and
    /// ignores trailing bytes, so it plays a new record exactly as before; a record from before
    /// it decodes with no layout.
    #[test]
    fn the_layout_byte_leaves_the_rest_of_a_screen_record_as_older_builds_read_it() {
        let old = ScreenRecord {
            state: Some(ScreenState::Inventory),
            pointer: Some((-60, 20)),
            slots: vec![(CARRIED, Some(stack(373, 1, 8197)))],
            layout: None,
        };
        let mut old_bytes = Vec::new();
        old.encode_into(&mut old_bytes);
        for (centered, show_effects, byte) in [
            (false, true, 0),
            (true, true, 1),
            (false, false, 2),
            (true, false, 3),
        ] {
            let layout = EffectLayout {
                centered,
                show_effects,
            };
            let new = ScreenRecord {
                layout: Some(layout),
                ..old.clone()
            };
            let mut new_bytes = Vec::new();
            new.encode_into(&mut new_bytes);
            let mut expected = old_bytes.clone();
            expected[1] |= ScreenRecord::HAS_LAYOUT;
            expected.push(byte);
            assert_eq!(new_bytes, expected);
            assert_eq!(ScreenRecord::decode(&new_bytes), Some(new));
        }
        assert_eq!(ScreenRecord::decode(&old_bytes), Some(old));
        // A layout-only record (the setting changed with no screen change) is three bytes.
        let only = ScreenRecord {
            layout: Some(EffectLayout {
                centered: true,
                show_effects: true,
            }),
            ..ScreenRecord::default()
        };
        let mut bytes = Vec::new();
        only.encode_into(&mut bytes);
        assert_eq!(bytes, vec![1, ScreenRecord::HAS_LAYOUT, 1]);
    }

    #[test]
    fn typed_commands_keep_only_their_name() {
        let hide = ChatText::from_choice(1);
        assert_eq!(hide, ChatText::HideCommands);
        assert_eq!(hide.apply("gg everyone"), "gg everyone");
        assert_eq!(hide.apply("/login hunter2"), "/login *******");
        assert_eq!(
            hide.apply("/msg Friend meet at mid"),
            "/msg ******************"
        );
        assert_eq!(hide.apply("/l"), "/l");
        assert_eq!(hide.apply("/l "), "/l ");
        assert_eq!(
            ChatText::from_choice(0).apply("/login hunter2"),
            "/login hunter2"
        );
        assert_eq!(ChatText::from_choice(2).apply("hello"), "");
        // Multi-byte text is starred per character, never split inside one.
        assert_eq!(hide.apply("/r héllo"), "/r *****");
    }

    /// A record whose item stack claims 2^62 bytes, in a few
    /// bytes, is refused at the end of the record instead of reading zeros forever.
    #[test]
    fn a_stack_length_past_the_record_is_refused_not_read_forever() {
        // Version 1, one slot (slot 0), then a stack length varint of 2^62 and nothing else.
        let mut bytes = vec![1, 1, 0];
        let mut length = 1u64 << 62;
        loop {
            let byte = (length & 0x7F) as u8;
            length >>= 7;
            if length == 0 {
                bytes.push(byte);
                break;
            }
            bytes.push(byte | 0x80);
        }
        assert_eq!(InventoryRecord::decode(&bytes), None);
        // The same for a screen record's slots.
        let mut screen = vec![1, ScreenRecord::HAS_SLOTS, 1, 0];
        screen.extend_from_slice(&bytes[3..]);
        assert_eq!(ScreenRecord::decode(&screen), None);
        // A whole, ordinary record still decodes.
        let ordinary = InventoryRecord {
            slots: vec![(36, None)],
        };
        let mut encoded = Vec::new();
        ordinary.encode_into(&mut encoded);
        assert_eq!(InventoryRecord::decode(&encoded), Some(ordinary));
    }
}
