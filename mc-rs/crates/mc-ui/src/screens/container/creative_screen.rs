//! [`CreativeScreen`]: the usable creative-mode inventory — tab strip, search box, scrollbar and
//! item grid. java: `client/gui/inventory/GuiContainerCreative.java`.
//!
//! [`super::creative`] is the data layer (which tabs exist, what is in them, what a click on a
//! tab-source cell does). This module is the screen those rules drive: it draws
//! `creative_inventory/tabs.png`'s twelve tab plates in their vanilla positions, the
//! `tab_items.png`/`tab_item_search.png`/`tab_inventory.png` panel behind the grid, the search
//! field, the scrollbar handle, and every visible stack.
//!
//! # The layout, in vanilla's own numbers
//!
//! ```text
//!   xSize 195, ySize 136              guiLeft = (width - 195) / 2, guiTop = (height - 136) / 2
//!   tab strip   row 0 at guiTop - 28, row 1 at guiTop + ySize - 4; column c at 28*c (+c), or
//!               guiLeft + xSize - 28 for column 5
//!   item grid   5 rows x 9 at (9 + col*18, 18 + row*18)
//!   hotbar      9 at (9 + col*18, 112)
//!   search      (82, 6), 89 x 9, at most 15 characters
//!   scrollbar   handle 12x15 at (175, 18 + 95 * scroll); the drag gutter is (175, 18)-(189, 130)
//! ```
//!
//! # Typing in the search box
//!
//! Plain [`Screen::key_typed`] is handed no [`FontRenderer`], but [`TextField`]'s editing
//! operations need one (they re-measure the visible window after every edit). Earlier revisions
//! of this screen worked around that by **queueing** key presses destined for the search box and
//! draining them at the top of the next [`Screen::draw`], which does have a font — at the cost
//! of a one-frame lag between a keystroke and the item list it filters. [`Screen`] has since
//! grown [`Screen::key_typed_with_font`], which [`ScreenStack`](crate::screen_stack::ScreenStack)
//! calls in `key_typed`'s place; this screen overrides that instead, so a keystroke reaches
//! [`Self::search`] — and [`Self::update_search`] — inside the same call it arrived in. Plain
//! `key_typed` is left at [`Screen`]'s default (Escape pops); nothing but a test calls it on this
//! screen any more.
//!
//! Click-to-position-the-caret inside the search box has the same shape of gap —
//! [`Screen::mouse_clicked`] has no font either — and was closed the same way: [`TextField`]
//! grew [`TextField::mouse_clicked_cached`], which replays a draw-time recording instead of
//! re-measuring, so a click can place the caret with **no** font at all. This screen does not
//! call it, because it does not need to: vanilla pins the field focused and un-losable while the
//! search tab is open (`setCanLoseFocus(false); setFocused(true)`), so typing always reaches it
//! regardless of where — or whether — the caret sits.
//!
//! # Clicks and drawing
//!
//! `GuiContainerCreative` overrides only `handleMouseClick` (and tab presses): every press,
//! drag and release runs `GuiContainer`'s state machine, and every cell is drawn by
//! `GuiContainer.drawSlot`, drag-split preview included. This screen reproduces both; the
//! creative click oracle (`creative_oracle.rs`, from the real 1.8.9 classes) checks the packets.
//! The Inventory tab's player model is [`super::player_preview`]'s (see its doc for its limits).
//! Vanilla 1.8.9's creative screen has no achievements/statistics buttons: `initGui` clears
//! `buttonList` and adds none.

use mc_assets::Language;
use mc_protocol::play::c2s;
use mc_render::entity::SkinModel;
use mc_world::MapData;
use std::collections::BTreeSet;
#[cfg(not(test))]
use std::sync::atomic::{AtomicU8, Ordering};

use crate::font::FontRenderer;
use crate::screen_stack::{Screen, ScreenAction, ScreenCtx};
use crate::screens::hud::{ActiveEffect, INVENTORY_TEXTURE};
use crate::widget::keys::Modifiers;
use crate::widget::{keys, DrawList, MemoryClipboard, TextField, UiTexture};

use super::creative::{self, tab_column, tab_is_first_row, CreativeTab, TABS, TAB_COUNT};
use super::interaction::DOUBLE_CLICK_MS;
use super::model::Container;
use super::player_preview::{PlayerPreviewRenderer, PlayerPreviewSnapshot};
use super::screen::{ContainerBindings, SLOT_HIGHLIGHT, TITLE_COLOR};
#[cfg(test)]
use super::screen::{KEY_CHAT, KEY_DROP, KEY_HOTBAR_1, KEY_INVENTORY};
use super::stack_ext::{self, ItemStack};
use super::tooltip_box::{self, WORLD_DIM_BOTTOM, WORLD_DIM_TOP};
use super::{layouts, render, tooltip};

/// java: `GuiContainerCreative.xSize`.
pub const X_SIZE: i32 = 195;
/// java: `GuiContainerCreative.ySize`.
pub const Y_SIZE: i32 = 136;
/// How many cells the scrollable item grid shows at once: 5 rows of 9.
pub const GRID_CELLS: usize = 45;
/// java: `ContainerCreative.func_148328_e` — more than one screenful means a live scrollbar.
pub const SCROLL_THRESHOLD: usize = 45;
/// The `tabs.png` atlas column at which the scrollbar handle art starts. java:
/// `GuiContainerCreative.java:703` — `232` when the bar is live, `232 + 12` when it is greyed.
pub const SCROLLBAR_U: i32 = 232;

// java: `GuiContainerCreative.selectedTabIndex` is static, so the selection survives screen
// reconstruction for the lifetime of the client process. The live UI is confined to its main
// thread; tests use thread-local storage so parallel test cases cannot rewrite one another's
// remembered tab between construction and assertion.
#[cfg(not(test))]
static SELECTED_TAB_INDEX: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
std::thread_local! {
    static SELECTED_TAB_INDEX: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

#[cfg(not(test))]
fn remembered_selected_tab() -> u8 {
    SELECTED_TAB_INDEX.load(Ordering::Relaxed) % TAB_COUNT
}

#[cfg(test)]
fn remembered_selected_tab() -> u8 {
    SELECTED_TAB_INDEX.with(|tab| tab.get() % TAB_COUNT)
}

#[cfg(not(test))]
fn remember_selected_tab(index: u8) {
    SELECTED_TAB_INDEX.store(index % TAB_COUNT, Ordering::Relaxed);
}

#[cfg(test)]
fn remember_selected_tab(index: u8) {
    SELECTED_TAB_INDEX.with(|tab| tab.set(index % TAB_COUNT));
}

fn tabs_texture() -> UiTexture {
    UiTexture::gui("textures/gui/container/creative_inventory/tabs.png")
}

fn panel_texture(tab: CreativeTab) -> UiTexture {
    UiTexture::gui(&format!(
        "textures/gui/container/creative_inventory/{}",
        tab.background_image()
    ))
}

/// Where one tab plate is drawn, in **window-relative** pixels, together with the `tabs.png`
/// texel it samples.
///
/// java: `GuiContainerCreative.func_147051_a`. Note the two quirks reproduced here: columns
/// `1..=4` are nudged one pixel right *per column* (`l += i`), and column 5 is pinned flush to
/// the panel's right edge instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabPlate {
    /// Left edge relative to `gui_left`.
    pub x: i32,
    /// Top edge relative to `gui_top`.
    pub y: i32,
    /// Source texel column in `tabs.png`.
    pub u: i32,
    /// Source texel row in `tabs.png`: `0` top-row idle, `32` top-row selected, `64` bottom-row
    /// idle, `96` bottom-row selected.
    pub v: i32,
    /// Where the tab's icon item is drawn, relative to `gui_left`/`gui_top`.
    pub icon: (i32, i32),
}

/// The plate geometry for tab `index`, selected or not. java:
/// `GuiContainerCreative.func_147051_a`.
#[must_use]
pub fn tab_plate(index: u8, selected: bool) -> TabPlate {
    let column = tab_column(index);
    let first_row = tab_is_first_row(index);
    let u = column * 28;
    let mut v = if selected { 32 } else { 0 };

    let mut x = 28 * column;
    if column == 5 {
        x = X_SIZE - 28;
    } else if column > 0 {
        x += column;
    }

    let y = if first_row {
        -28
    } else {
        v += 64;
        Y_SIZE - 4
    };

    TabPlate {
        x,
        y,
        u,
        v,
        icon: (x + 6, y + 8 + if first_row { 1 } else { -1 }),
    }
}

/// Whether a window-relative point is on tab `index`'s clickable area.
///
/// java: `GuiContainerCreative.func_147049_a`. Two vanilla quirks, both reproduced and pinned by
/// test: the bounds are **inclusive on both ends** (`>= j && <= j + 28`), making the box 29x33
/// rather than 28x32; and column 5's hit box sits at `xSize - 28 + 2`, two pixels right of where
/// [`tab_plate`] *draws* it.
#[must_use]
pub fn tab_contains(index: u8, x: i32, y: i32) -> bool {
    let column = tab_column(index);
    let mut left = 28 * column;
    if column == 5 {
        left = X_SIZE - 28 + 2;
    } else if column > 0 {
        left += column;
    }
    let top = if tab_is_first_row(index) { -32 } else { Y_SIZE };
    x >= left && x <= left + 28 && y >= top && y <= top + 32
}

/// One clickable cell on the creative screen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreativeSlot {
    /// A read-only cell of the scrollable tab-source grid, `0..45`. Not a server-synced slot:
    /// clicking it runs [`creative::click_tab_source`], not the container state machine.
    Grid(u8),
    /// A player-inventory slot, by its index in the always-open
    /// [`ContainerKind::Player`](super::model::ContainerKind::Player) container.
    Player(u16),
    /// The Inventory tab's destroy-item cell. java: `GuiContainerCreative.field_147064_C`.
    Trash,
}

/// The creative-mode inventory screen.
#[derive(Debug)]
pub struct CreativeScreen {
    /// The real, server-synced player inventory (window `0`). The Inventory tab shows it
    /// directly, and every other tab still shows its hotbar row along the bottom.
    pub player: Container,
    /// The [`UiTexture`] key the host uploaded the block/item atlas under.
    pub atlas: UiTexture,
    /// Vanilla's F3+H "advanced tooltips" setting.
    pub advanced_tooltips: bool,
    /// Current inventory/drop/hotbar/pick-block bindings, refreshed by the host before input.
    bindings: ContainerBindings,
    /// Active translations used by metadata-aware item tooltip names.
    language: Option<Language>,
    /// The search box. Editing goes through [`CreativeScreen::apply_pending_input`].
    pub search: TextField,
    selected_tab: u8,
    item_list: Vec<ItemStack>,
    grid: Vec<Option<ItemStack>>,
    scroll: f32,
    is_scrolling: bool,
    left_button_down: bool,
    /// `GuiContainer`'s desktop drag-splitting state. Creative's screen has a different
    /// `handleMouseClick` tail than ordinary containers, so it cannot reuse the sibling screen's
    /// `MouseState` without also importing that screen's C0E packet path.
    drag_splitting: bool,
    drag_splitting_button: i32,
    drag_splitting_limit: i32,
    drag_splitting_slots: Vec<u16>,
    /// java: `GuiContainer.dragSplittingRemnant`, what the held stack
    /// would keep after the sweep. Recomputed only by [`Self::update_drag_splitting`] (each
    /// accepted move and each slot `drawSlot` drops), as in Java; the held stack shows it while
    /// more than one slot is swept.
    drag_splitting_remnant: i32,
    /// The rest of `GuiContainer`'s mouse state, which vanilla runs
    /// for this screen too (`GuiContainerCreative` overrides `handleMouseClick`, not the state
    /// machine around it): `lastClickSlot`, `lastClickTime`, `lastClickButton`, `doubleClick`,
    /// `ignoreMouseUp` (true in the constructor) and `shiftClickedSlot`.
    last_click_slot: Option<CreativeSlot>,
    last_click_time_ms: u64,
    last_click_button: i32,
    double_click: bool,
    ignore_mouse_up: bool,
    shift_clicked_slot: Option<ItemStack>,
    /// Shift at the last press: the stand-in for Java's release-time `isShiftKeyDown()` when a
    /// release arrives through [`Screen::mouse_released`], which carries no modifiers.
    last_press_shift: bool,
    /// `Minecraft.getSystemTime()` for the 250 ms double-click window: the host's monotonic
    /// milliseconds ([`Self::set_clock_ms`]).
    clock_ms: u64,
    /// `InventoryEffectRenderer`'s live effect collection. Unlike ordinary creative-grid state,
    /// this can change while the screen stays open and therefore also drives the per-tick
    /// `guiLeft` update.
    active_effects: Vec<ActiveEffect>,
    /// The `centered_inventory` mod: keep the inventory centred while effects are
    /// active instead of vanilla's shift to the right. `false` is vanilla.
    effects_centered: bool,
    /// The `centered_inventory` mod with "show effects" off: skip the effect panels.
    /// `false` is vanilla.
    effects_hidden: bool,
    /// Last scaled width supplied to [`Screen::init`], retained because
    /// `GuiContainerCreative.updateScreen` calls `updateActivePotionEffects()` every tick.
    screen_width: i32,
    gui_left: i32,
    gui_top: i32,
    hovered: Option<CreativeSlot>,
    /// NOT VANILLA 1.8.9: the mouse button whose press the hotbar addition in
    /// `mouse_clicked_inner` consumed, so its release is swallowed too.
    hotbar_mouse_button: Option<i32>,
    /// The live `map_<id>` record used by the current advanced item tooltip. Pixel changes do
    /// not affect that text, so this only refreshes when its id or scale changes.
    tooltip_map_data: Option<MapData>,
    outgoing: Vec<c2s::CreativeInventoryAction>,
    pending_search_reset: bool,
    clear_search_on_next_key: bool,
    clipboard: MemoryClipboard,
    /// java: `GuiInventory`'s implicit `mc.thePlayer` — the CPU-projected 3-D player preview
    /// drawn on the Inventory tab only. See [`super::player_preview`].
    player_preview: PlayerPreviewRenderer,
    /// Which skin model (arm width) the preview poses. Set from outside with
    /// [`Self::set_player_preview_skin`]; harmless (and never drawn) on every other tab.
    player_skin: SkinModel,
    /// Ready generation-specific skin texture. `None` preserves the UUID-model Steve/Alex
    /// fallback while profile pixels are absent or still downloading.
    player_skin_texture: Option<UiTexture>,
    /// Live armour/held-item/cape/pose snapshot for the Inventory-tab preview.
    player_preview_snapshot: PlayerPreviewSnapshot,
    /// java: `ageInTicks`'s driver for the preview — ticks since this screen was constructed,
    /// combined with [`ScreenCtx::partial_ticks`] at draw time. Advanced by [`Screen::tick`].
    preview_age_ticks: u32,
    /// The `scrollable_tooltips` mod's state. Off by default, when the tooltip
    /// draw is exactly vanilla's; the client turns it on with `set_options`.
    pub tooltip_scroll: super::tooltip_scroll::TooltipScroll,
}

impl CreativeScreen {
    /// A fresh creative screen on the process's remembered tab, over a new player inventory.
    #[must_use]
    pub fn new() -> Self {
        Self::over(layouts::player_container(0))
    }

    /// A creative screen over an existing, already-synced player inventory container.
    #[must_use]
    pub fn over(player: Container) -> Self {
        let restored_tab = remembered_selected_tab();
        let mut search = TextField::new(0, 0, 0, 89, 9);
        search.set_max_length(15);
        search.background = false;
        search.visible = false;
        search.enabled_color = 0x00FF_FFFF;
        let mut me = Self {
            player,
            atlas: super::screen::default_atlas_texture(),
            advanced_tooltips: false,
            bindings: ContainerBindings::default(),
            language: None,
            search,
            selected_tab: 0,
            item_list: Vec::new(),
            grid: vec![None; GRID_CELLS],
            scroll: 0.0,
            is_scrolling: false,
            left_button_down: false,
            drag_splitting: false,
            drag_splitting_button: 0,
            drag_splitting_limit: 0,
            drag_splitting_slots: Vec::new(),
            drag_splitting_remnant: 0,
            last_click_slot: None,
            last_click_time_ms: 0,
            last_click_button: 0,
            double_click: false,
            ignore_mouse_up: true,
            shift_clicked_slot: None,
            last_press_shift: false,
            clock_ms: 0,
            active_effects: Vec::new(),
            effects_centered: false,
            effects_hidden: false,
            screen_width: 0,
            gui_left: 0,
            gui_top: 0,
            hovered: None,
            hotbar_mouse_button: None,
            tooltip_map_data: None,
            outgoing: Vec::new(),
            pending_search_reset: false,
            clear_search_on_next_key: false,
            clipboard: MemoryClipboard::default(),
            player_preview: PlayerPreviewRenderer::new(),
            player_skin: SkinModel::Classic,
            player_skin_texture: None,
            player_preview_snapshot: PlayerPreviewSnapshot::default(),
            preview_age_ticks: 0,
            tooltip_scroll: super::tooltip_scroll::TooltipScroll::default(),
        };
        me.set_tab(restored_tab);
        // Registering CreativeCrafting immediately calls detectAndSendChanges against the
        // window-0 container's mirror: every occupied slot on the first opening after a join or
        // respawn, afterwards only what changed since the listener last looked.
        me.emit_changes();
        me
    }

    /// Install the host's current rebindable container actions.
    pub fn set_bindings(&mut self, bindings: ContainerBindings) {
        self.bindings = bindings;
    }

    /// Sets which skin model (arm width) the Inventory-tab player preview poses. Only meaningful
    /// while that tab is open — every other tab never draws the preview at all. java: has no
    /// equivalent; vanilla's preview always poses `mc.thePlayer` directly and never needs to be
    /// told which skin that is. See [`super::player_preview`].
    pub fn set_player_preview_skin(&mut self, model: SkinModel) {
        self.player_skin = model;
    }

    /// Sets the ready dynamic skin texture used by the Inventory-tab player preview.
    pub fn set_player_preview_texture(&mut self, texture: Option<UiTexture>) {
        self.player_skin_texture = texture;
    }

    /// Replace the complete live player snapshot consumed by the Inventory-tab preview.
    pub fn set_player_preview_snapshot(&mut self, snapshot: PlayerPreviewSnapshot) {
        self.player_preview_snapshot = snapshot;
    }

    /// Replace the local player's active-effect list used by the inherited
    /// `InventoryEffectRenderer` panel.
    ///
    /// The host already resolves the translated name, amplifier suffix, icon and duration into
    /// [`ActiveEffect`] for the HUD. Supplying that snapshot here keeps the creative screen's
    /// panel on the same tick as the HUD and immediately applies vanilla's horizontal shift.
    pub fn set_active_effects(&mut self, effects: Vec<ActiveEffect>) {
        self.active_effects = effects;
        self.update_active_potion_effects();
    }

    /// The effect snapshot currently rendered beside the creative inventory.
    #[must_use]
    pub fn active_effects(&self) -> &[ActiveEffect] {
        &self.active_effects
    }

    /// `centered_inventory` mod: `centered` keeps the inventory centred while
    /// effects are active (vanilla shifts it right), and `show_effects == false` skips the effect
    /// panels. `(false, true)` is vanilla. Applied at once (search field included), and by every
    /// later effect update.
    pub fn set_effect_layout(&mut self, centered: bool, show_effects: bool) {
        self.effects_centered = centered;
        self.effects_hidden = !show_effects;
        self.update_active_potion_effects();
    }

    // java: InventoryEffectRenderer.java:30-42 plus
    // GuiContainerCreative.java:248-257 (the search-field re-anchor override).
    fn update_active_potion_effects(&mut self) {
        let old_left = self.gui_left;
        self.gui_left = if self.active_effects.is_empty() || self.effects_centered {
            (self.screen_width - X_SIZE) / 2
        } else {
            160 + (self.screen_width - X_SIZE - 200) / 2
        };
        if self.gui_left != old_left {
            self.search.rect.x = self.gui_left + 82;
        }
    }

    // java: InventoryEffectRenderer.java:60-112 (drawActivePotionEffects).
    fn draw_active_potion_effects(&self, d: &mut DrawList, font: &mut FontRenderer) {
        if self.active_effects.is_empty() || self.effects_hidden {
            return;
        }
        let texture = UiTexture::gui(INVENTORY_TEXTURE);
        let x = self.gui_left - 124;
        let mut y = self.gui_top;
        let pitch = if self.active_effects.len() > 5 {
            132 / (self.active_effects.len() as i32 - 1)
        } else {
            33
        };
        for effect in &self.active_effects {
            d.blit(&texture, x, y, 0, 166, 140, 32);
            if effect.icon_index >= 0 {
                d.blit(
                    &texture,
                    x + 6,
                    y + 7,
                    (effect.icon_index % 8) * 18,
                    198 + (effect.icon_index / 8) * 18,
                    18,
                    18,
                );
            }
            font.draw_string_with_shadow(
                d,
                &effect.name,
                (x + 28) as f32,
                (y + 6) as f32,
                0x00FF_FFFF,
            );
            font.draw_string_with_shadow(
                d,
                &effect.duration,
                (x + 28) as f32,
                (y + 16) as f32,
                0x007F_7F7F,
            );
            y += pitch;
        }
    }

    /// Refresh translations used by item tooltip names.
    pub fn set_language(&mut self, language: &Language) {
        self.language = Some(language.clone());
        if self.tab() == CreativeTab::Search {
            self.update_search();
        }
    }

    /// The tab currently open, as an index into [`TABS`].
    #[must_use]
    pub const fn selected_tab(&self) -> u8 {
        self.selected_tab
    }

    /// The tab currently open.
    #[must_use]
    pub fn tab(&self) -> CreativeTab {
        TABS[self.selected_tab as usize % TABS.len()]
    }

    /// Everything the open tab currently lists, before scrolling.
    #[must_use]
    pub fn item_list(&self) -> &[ItemStack] {
        &self.item_list
    }

    /// The 45 cells the grid is showing right now, top-left first.
    #[must_use]
    pub fn grid(&self) -> &[Option<ItemStack>] {
        &self.grid
    }

    /// The scroll position, `0.0` at the top and `1.0` at the bottom.
    #[must_use]
    pub const fn scroll(&self) -> f32 {
        self.scroll
    }

    /// The window's left edge.
    #[must_use]
    pub const fn gui_left(&self) -> i32 {
        self.gui_left
    }

    /// The window's top edge.
    #[must_use]
    pub const fn gui_top(&self) -> i32 {
        self.gui_top
    }

    /// The cell the cursor was over on the last [`Screen::draw`].
    #[must_use]
    pub const fn hovered(&self) -> Option<CreativeSlot> {
        self.hovered
    }

    /// Take every `CreativeInventoryAction` produced since the last call.
    ///
    /// java: `CreativeCrafting.sendSlotContents` -> `PlayerControllerMP.sendSlotPacket`. Vanilla
    /// registers an `ICrafting` listener on the player's container and lets it fire for every
    /// slot the click changed; this port diffs the container around each click, which produces
    /// the same packets for the same reasons.
    pub fn take_outgoing(&mut self) -> Vec<c2s::CreativeInventoryAction> {
        std::mem::take(&mut self.outgoing)
    }

    /// Route a click with the live keyboard modifiers supplied by the host screen dispatcher.
    /// java: `GuiContainerCreative.handleMouseClick` reads `isShiftKeyDown()` at click time.
    pub fn mouse_clicked_with_mods(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        mods: Modifiers,
    ) -> ScreenAction {
        self.mouse_clicked_inner(x, y, button, mods.shift)
    }

    /// java: `GuiContainer.checkHotbarKeys` (`:718`) into `GuiContainerCreative.handleMouseClick`
    /// click type 2. With an empty cursor over a hovered cell, a code bound to `key.hotbar.N`
    /// fills or swaps hotbar slot `N`. Returns whether it fired.
    fn check_hotbar_keys(&mut self, key: i32) -> bool {
        let Some(slot) = self.hovered else {
            return false;
        };
        if self.player.cursor().is_some() {
            return false;
        }
        let Some(hotbar) = self
            .bindings
            .hotbar_keys
            .iter()
            .position(|bound| *bound == key)
        else {
            return false;
        };
        let hotbar = hotbar as u16;
        // java: `handleMouseClick` sets `field_147057_D` first, so after
        // a hotbar key the next key typed on the Search tab replaces the query.
        self.clear_search_on_next_key = true;
        match slot {
            CreativeSlot::Grid(index) => {
                if let Some(mut source) = self.stack_at(CreativeSlot::Grid(index)).cloned() {
                    source.count = stack_ext::max_stack_size(&source);
                    self.player
                        .set_slot_contents(36 + hotbar as usize, Some(source));
                    self.emit_changes();
                }
            }
            CreativeSlot::Player(id) => {
                self.player.slot_click(i32::from(id), hotbar as u8, 2, true);
                if self.tab() == CreativeTab::Inventory {
                    self.emit_changes();
                } else {
                    // `checkHotbarKeys` reaches the same direct C10 tail as a mouse click on a
                    // content-tab hotbar slot, with the hotbar index as `clickedButton`.
                    self.emit_content_tail(id, i32::from(hotbar));
                }
            }
            CreativeSlot::Trash => {}
        }
        true
    }

    fn mouse_clicked_inner(&mut self, x: i32, y: i32, button: i32, shift: bool) -> ScreenAction {
        // NOT VANILLA 1.8.9, a deliberate addition; see
        // `ContainerScreen::mouse_clicked_with_mods`. A mouse button bound to `key.hotbar.N`
        // does what its keyboard counterpart does over a hovered cell, and the press (and its
        // release) does nothing else. Delete this block, the release check in
        // `mouse_released_inner` and `hotbar_mouse_button` to restore 1.8.9 exactly.
        self.hotbar_mouse_button = None;
        if self.check_hotbar_keys(button - 100) {
            self.hotbar_mouse_button = Some(button);
            return ScreenAction::None;
        }
        self.last_press_shift = shift;
        if button == 0 {
            self.left_button_down = true;
            // java: `GuiContainerCreative.mouseClicked` returns before `super.mouseClicked` for a
            // left press on a tab, so none of the bookkeeping below runs.
            if self.tab_at(x, y).is_some() {
                return ScreenAction::None;
            }
            let (bar_left, bar_top, bar_right, bar_bottom) = self.scrollbar_gutter();
            if x >= bar_left && y >= bar_top && x < bar_right && y < bar_bottom {
                self.is_scrolling = self.needs_scroll_bars();
                if self.is_scrolling {
                    self.scroll_from_mouse(y);
                    // java: the scrollbar is `drawScreen`'s; the press still reaches
                    // `GuiContainer.mouseClicked` over no slot, which only keeps its bookkeeping.
                    self.begin_press(None, button);
                    self.end_press(None, button);
                    return ScreenAction::None;
                }
            }
        }
        // java: `GuiContainer.mouseClicked` (`:359-460`).
        let pick_block = button == self.bindings.pick_block_button;
        let slot = self.slot_at(x, y);
        self.begin_press(slot, button);
        // java: `if (mouseButton == 0 || mouseButton == 1 || flag)`: any other button only keeps
        // the bookkeeping (its release can still click, see `mouse_released_inner`).
        if button == 0 || button == 1 || pick_block {
            let outside_panel = self.outside_panel(x, y);
            // java: `if (l != -1)`, a slot or outside the panel, and `else if
            // (!this.dragSplitting)`: a press while a drag is armed changes nothing.
            if (outside_panel || slot.is_some()) && !self.drag_splitting {
                if self.player.cursor().is_none() {
                    // java: an outside press is `handleMouseClick(null, -999, b, 4)` (or 3 for
                    // the pick button), which moves nothing without a cursor stack.
                    match (outside_panel, slot) {
                        (false, Some(slot)) => {
                            if shift && !pick_block {
                                self.shift_clicked_slot = self.stack_at(slot).cloned();
                            }
                            self.click_slot_inner(slot, button, shift, pick_block);
                        }
                        _ => self.empty_handed_click_outside(),
                    }
                    self.ignore_mouse_up = true;
                } else {
                    // java: GuiContainer.mouseClicked does not call handleMouseClick while a
                    // cursor stack is held. It arms drag-splitting and waits for mouseClickMove /
                    // mouseReleased.
                    self.begin_drag(button);
                }
            }
        }
        self.end_press(slot, button);
        ScreenAction::None
    }

    /// java: the start of `GuiContainer.mouseClicked`: `doubleClick` is the same slot (both
    /// `null` counts), within 250 ms, with the same button; then `ignoreMouseUp = false`.
    fn begin_press(&mut self, slot: Option<CreativeSlot>, button: i32) {
        // java: `i - this.lastClickTime < 250L` on signed longs.
        let elapsed = (self.clock_ms as i64).wrapping_sub(self.last_click_time_ms as i64);
        self.double_click = self.last_click_slot == slot
            && elapsed < DOUBLE_CLICK_MS as i64
            && self.last_click_button == button;
        self.ignore_mouse_up = false;
    }

    /// java: the end of `GuiContainer.mouseClicked`, outside every branch.
    fn end_press(&mut self, slot: Option<CreativeSlot>, button: i32) {
        self.last_click_slot = slot;
        self.last_click_time_ms = self.clock_ms;
        self.last_click_button = button;
    }

    /// java: `handleMouseClick(null, -999, button, 4)` (3 for the pick
    /// button) from a press outside the window with nothing held. It sets `field_147057_D` (the
    /// search box clears on the next key) on every tab. On the Inventory tab its last `else`
    /// (`GuiContainerCreative:142-146`) runs `inventoryContainer.slotClick(-999, ...)`, which
    /// changes nothing, then `detectAndSendChanges`, which uploads every slot the listener
    /// mirror has not seen yet.
    fn empty_handed_click_outside(&mut self) {
        self.clear_search_on_next_key = true;
        if self.tab() == CreativeTab::Inventory {
            self.emit_changes();
        }
    }

    fn outside_panel(&self, x: i32, y: i32) -> bool {
        x < self.gui_left
            || x >= self.gui_left + X_SIZE
            || y < self.gui_top
            || y >= self.gui_top + Y_SIZE
    }

    /// Set the double-click clock from the host's monotonic milliseconds
    /// (`Minecraft.getSystemTime()`).
    pub fn set_clock_ms(&mut self, ms: u64) {
        self.clock_ms = ms;
    }

    /// Route a release with the live keyboard modifiers. java: `GuiContainer.mouseReleased`
    /// reads `isShiftKeyDown()` at the release, both for a held stack's click and for the
    /// shift-double-click sweep.
    pub fn mouse_released_with_mods(&mut self, x: i32, y: i32, button: i32, mods: Modifiers) {
        self.mouse_released_inner(x, y, button, mods.shift);
    }

    /// java: `GuiContainer.mouseClicked`'s three drag button cases. The middle button is the
    /// configured pick-block binding in 1.8.9 (`keyCode + 100 == 2`).
    #[must_use]
    fn drag_limit(&self, button: i32) -> Option<i32> {
        if button == self.bindings.pick_block_button {
            Some(2)
        } else {
            match button {
                0 => Some(0),
                1 => Some(1),
                _ => None,
            }
        }
    }

    fn begin_drag(&mut self, button: i32) {
        let Some(limit) = self.drag_limit(button) else {
            return;
        };
        self.drag_splitting = true;
        self.drag_splitting_button = button;
        self.drag_splitting_limit = limit;
        self.drag_splitting_slots.clear();
    }

    fn reset_drag(&mut self) {
        self.drag_splitting = false;
        self.drag_splitting_button = 0;
        self.drag_splitting_limit = 0;
        self.drag_splitting_slots.clear();
    }

    /// Convert a visible creative-screen cell into the underlying player-container slot that
    /// vanilla's `ContainerCreative.canDragIntoSlot` allows. The read-only tab grid is never a
    /// drag destination: its slots are in the scratch inventory and sit above y=90. The
    /// Inventory tab exposes the real player slots and delegates to `ContainerPlayer`.
    #[must_use]
    fn drag_slot_id(&self, slot: CreativeSlot) -> Option<u16> {
        match (self.tab(), slot) {
            (CreativeTab::Inventory, CreativeSlot::Player(id)) if id >= 5 => Some(id),
            (CreativeTab::Inventory, _) => None,
            (_, CreativeSlot::Player(id)) if (36..45).contains(&id) => Some(id),
            _ => None,
        }
    }

    /// java: `GuiContainer.mouseClickMove`'s drag branch (`:505-509`): a slot that passes the
    /// four-condition gate joins `dragSplittingSlots` (a `HashSet`, so a repeat changes nothing)
    /// and `updateDragSplitting` runs, for a slot already in the set too.
    fn add_drag_slot(&mut self, slot: CreativeSlot) {
        let Some(slot_id) = self.drag_slot_id(slot) else {
            return;
        };
        let Some(cursor) = self.player.cursor() else {
            return;
        };
        if cursor.count <= self.drag_splitting_slots.len() as i32
            || !self.player.slot_accepts_for_drag(slot_id, cursor)
        {
            return;
        }
        if !self.drag_splitting_slots.contains(&slot_id) {
            self.drag_splitting_slots.push(slot_id);
        }
        self.update_drag_splitting();
    }

    /// java: `Container.computeStackSize` and the two caps after it in
    /// `GuiContainer.drawSlot` (`:254-266`) and `updateDragSplitting` (`:321-331`): the stack swept
    /// slot `slot_id` would hold, what it holds now, and the yellow count when the item's or the
    /// slot's limit cut the stack.
    fn drag_split_preview(
        &self,
        cursor: &ItemStack,
        slot_id: u16,
    ) -> (ItemStack, i32, Option<i32>) {
        let index = usize::from(slot_id);
        let existing = self.player.slot_stack(index).map_or(0, |stack| stack.count);
        let swept = self.drag_splitting_slots.len().max(1) as i32;
        let mut preview = cursor.clone();
        preview.count = match self.drag_splitting_limit {
            // java: `MathHelper.floor_float((float)stackSize / (float)size)`.
            0 => cursor.count.div_euclid(swept),
            1 => 1,
            // java: `getItem().getItemStackLimit()`, a full stack (the creative-only mode).
            2 => stack_ext::max_stack_size(cursor),
            _ => cursor.count,
        }
        .saturating_add(existing);
        let mut yellow = None;
        let item_max = stack_ext::max_stack_size(&preview);
        if preview.count > item_max {
            yellow = Some(item_max);
            preview.count = item_max;
        }
        if let Some(slot) = self.player.slots().get(index) {
            let slot_max = slot.item_stack_limit(&preview);
            if preview.count > slot_max {
                yellow = Some(slot_max);
                preview.count = slot_max;
            }
        }
        (preview, existing, yellow)
    }

    /// java: `GuiContainer.updateDragSplitting` (`:309-336`): the held
    /// stack less what every swept slot would take. A full-stack sweep can take more than is
    /// held, so the remnant can go below zero (drawn as a red count, as in Java).
    fn update_drag_splitting(&mut self) {
        let Some(cursor) = self.player.cursor() else {
            return;
        };
        if !self.drag_splitting {
            return;
        }
        let mut remnant = cursor.count;
        for &slot_id in &self.drag_splitting_slots {
            let (preview, existing, _) = self.drag_split_preview(cursor, slot_id);
            remnant -= preview.count - existing;
        }
        self.drag_splitting_remnant = remnant;
    }

    fn emit_hotbar_burst(&mut self) {
        // java: GuiContainerCreative.handleMouseClick's raw `getDragEvent(clickedButton) == 2`
        // branch sends all nine creative hotbar slots, including unchanged/empty ones.
        self.outgoing
            .extend((36..45).map(|slot_id| c2s::CreativeInventoryAction {
                slot_id,
                stack: self.player.slot_stack(slot_id as usize).cloned(),
            }));
    }

    /// java: `GuiContainerCreative.mouseReleased` (a left release on a
    /// tab switches to it), then `GuiContainer.mouseReleased` (`:515-652`), whose clicks go
    /// through this screen's `handleMouseClick`.
    fn mouse_released_inner(&mut self, x: i32, y: i32, button: i32, shift: bool) {
        // NOT VANILLA 1.8.9: see the hotbar block in `mouse_clicked_inner`.
        if self.hotbar_mouse_button == Some(button) {
            self.hotbar_mouse_button = None;
            return;
        }
        if button == 0 {
            self.left_button_down = false;
            self.is_scrolling = false;
            if let Some(index) = self.tab_at(x, y) {
                // java: GuiContainerCreative intercepts a left release on a tab before its
                // superclass sees it. Keep that precedence even if a held cursor had armed a
                // drag on an earlier frame.
                self.set_tab(index);
                return;
            }
        }
        let slot = self.slot_at(x, y);
        if self.double_click && button == 0 && slot.is_some_and(Self::can_merge_slot) {
            // java `:533-555`: the second release of a double click on a slot the creative
            // container lets merge (the hotbar row) collects, or with shift sweeps.
            if let Some(slot) = slot {
                if shift {
                    self.shift_double_click_sweep(slot);
                } else {
                    self.double_click_collect(slot);
                }
            }
            self.double_click = false;
            self.last_click_time_ms = 0;
        } else {
            // java `:558-564`: a release of another button cancels the drag, swallows the next
            // release and skips the tail.
            if self.drag_splitting && self.drag_splitting_button != button {
                self.reset_drag();
                self.ignore_mouse_up = true;
                return;
            }
            // java `:566-570`: also a hard return, skipping the tail.
            if self.ignore_mouse_up {
                self.ignore_mouse_up = false;
                return;
            }
            if self.drag_splitting && !self.drag_splitting_slots.is_empty() {
                self.finish_drag_splitting();
            } else if self.player.cursor().is_some() {
                self.release_held_cursor(x, y, button, slot, shift);
            }
        }
        // java `:646-651`.
        if self.player.cursor().is_none() {
            self.last_click_time_ms = 0;
        }
        self.reset_drag();
    }

    /// java: `ContainerCreative.canMergeSlot`, `slotIn.yDisplayPosition > 90`: the hotbar row
    /// (y = 112, on every tab) and the Inventory tab's destroy-item cell (173, 112).
    fn can_merge_slot(slot: CreativeSlot) -> bool {
        matches!(slot, CreativeSlot::Player(36..=44) | CreativeSlot::Trash)
    }

    /// java: `mouseReleased`'s held-stack click, `handleMouseClick(slot, k, state, 3)` for the
    /// pick button, else mode 1 with shift (when `k != -999`), else mode 0.
    fn release_held_cursor(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        slot: Option<CreativeSlot>,
        shift: bool,
    ) {
        let pick_block = button == self.bindings.pick_block_button;
        let outside_panel = self.outside_panel(x, y);
        let shift = shift && !outside_panel && !pick_block;
        if shift {
            // java: `shiftClickedSlot = slot != null && slot.getHasStack() ? slot.getStack() : null`.
            self.shift_clicked_slot = slot.and_then(|slot| self.stack_at(slot).cloned());
        }
        match slot {
            Some(slot) if !outside_panel => self.click_slot_inner(slot, button, shift, pick_block),
            _ => {
                // java: `handleMouseClick(null, k, state, mode)` sets `field_147057_D`. On a
                // content tab its `slotIn == null` branch drops by button whatever the mode, also
                // on the panel artwork (`k == -1`). On the Inventory tab only `-999` with mode 0
                // (made 4) drops, the whole stack; the pick button's mode 3 drops nothing.
                self.clear_search_on_next_key = true;
                if self.tab() != CreativeTab::Inventory || (outside_panel && !pick_block) {
                    self.drop_cursor_outside(button);
                } else {
                    // java: every other Inventory-tab case reaches
                    // the last `else` (`GuiContainerCreative:142-146`):
                    // `inventoryContainer.slotClick(k, state, mode)` changes nothing for a
                    // negative slot id, then `detectAndSendChanges` uploads every slot the
                    // listener mirror has not seen yet.
                    self.emit_changes();
                }
            }
        }
    }

    /// java: `handleMouseClick(slot, k, 0, 6)` from the double-click branch.
    fn double_click_collect(&mut self, slot: CreativeSlot) {
        // java: `handleMouseClick` sets `field_147057_D` first.
        self.clear_search_on_next_key = true;
        match slot {
            // java: the destroy-item cell clears the cursor, as any click on it does.
            CreativeSlot::Trash => self.player.set_cursor(None),
            CreativeSlot::Player(id) if self.tab() == CreativeTab::Inventory => {
                // java: `inventoryContainer.slotClick(slot.slotNumber, 0, 6)`, the player
                // container's collect, then `detectAndSendChanges`.
                self.player.slot_click(i32::from(id), 0, 6, true);
                self.emit_changes();
            }
            CreativeSlot::Player(id) => {
                // java: `ContainerCreative.slotClick(slot, 0, 6)`: its `canMergeSlot` admits only
                // the hotbar row, and the tail uploads the clicked slot alone (the slots the
                // collect drained are not sent).
                self.collect_from_hotbar(id);
                self.emit_content_slot(id);
            }
            CreativeSlot::Grid(_) => {}
        }
    }

    /// java: `Container.slotClick` mode 6 on `ContainerCreative`. With a stack on the cursor and
    /// the clicked slot empty, two passes from index 0 (the first skips full stacks) move
    /// matching items into the cursor, up to its maximum, from the slots `canMergeSlot` admits:
    /// the hotbar, `36..45` here.
    fn collect_from_hotbar(&mut self, clicked: u16) {
        let Some(mut held) = self.player.cursor().cloned() else {
            return;
        };
        if self.player.slot_stack(usize::from(clicked)).is_some() {
            return;
        }
        let max = stack_ext::max_stack_size(&held);
        for pass in 0..2 {
            for id in 36..45u16 {
                if held.count >= max {
                    break;
                }
                let Some(mut existing) = self.player.slot_stack(usize::from(id)).cloned() else {
                    continue;
                };
                if !self.player.slot_stack_would_accept(id, &held)
                    || (pass == 0 && existing.count == stack_ext::max_stack_size(&existing))
                {
                    continue;
                }
                let take = (max - held.count).min(existing.count);
                held.count += take;
                existing.count -= take;
                self.player
                    .set_slot_contents(usize::from(id), (existing.count > 0).then_some(existing));
            }
        }
        self.player.set_cursor(Some(held));
    }

    /// java: the shift branch of the double click (`:535-546`): every slot of the creative
    /// container that can be taken from, holds a stack, shares the clicked slot's inventory and
    /// accepts `shiftClickedSlot` gets a shift-click, in slot order, each seeing the state the
    /// previous one left.
    fn shift_double_click_sweep(&mut self, slot: CreativeSlot) {
        let Some(held) = self.shift_clicked_slot.clone() else {
            return;
        };
        // java: `slot2.inventory == slot.inventory`: the player inventory for a player slot (on
        // the Inventory tab the wrapped slots 5..45, elsewhere the hotbar); the destroy-item
        // cell's scratch inventory holds nothing.
        let CreativeSlot::Player(_) = slot else {
            return;
        };
        let first = if self.tab() == CreativeTab::Inventory {
            5
        } else {
            36
        };
        for id in first..45u16 {
            if self.player.slot_stack(usize::from(id)).is_some()
                && self.player.can_take_stack(usize::from(id), true)
                && self.player.slot_stack_would_accept(id, &held)
            {
                self.click_slot_inner(CreativeSlot::Player(id), 0, true, false);
            }
        }
    }

    /// Apply the mode-5 sequence that vanilla's `GuiContainer.mouseReleased` emits. Inventory-tab
    /// drags use the real player container's listener mirror; content-tab drags use Creative's
    /// direct per-add packet followed by its unconditional nine-slot end burst.
    fn finish_drag_splitting(&mut self) {
        let slots = std::mem::take(&mut self.drag_splitting_slots);
        let limit = self.drag_splitting_limit;
        let content_tab = self.tab() != CreativeTab::Inventory;
        if content_tab {
            // `handleMouseClick(null, -999, start, 5)` has no direct packet. It does, however,
            // advance Container's drag state before each add/end call.
            self.player
                .slot_click(-999, super::model::encode_drag(0, limit), 5, true);
            self.clear_search_on_next_key = true;
            for slot_id in slots {
                self.player.slot_click(
                    slot_id as i32,
                    super::model::encode_drag(1, limit),
                    5,
                    true,
                );
                self.outgoing.push(c2s::CreativeInventoryAction {
                    slot_id: slot_id as i16,
                    stack: self.player.slot_stack(slot_id as usize).cloned(),
                });
            }
            self.player
                .slot_click(-999, super::model::encode_drag(2, limit), 5, true);
            self.emit_hotbar_burst();
        } else {
            self.player
                .slot_click(-999, super::model::encode_drag(0, limit), 5, true);
            // java: the Inventory-tab branch calls detectAndSendChanges immediately after each
            // handleMouseClick, including the otherwise packet-free drag start.
            self.emit_changes();
            for slot_id in slots {
                self.player.slot_click(
                    slot_id as i32,
                    super::model::encode_drag(1, limit),
                    5,
                    true,
                );
                self.emit_changes();
            }
            self.player
                .slot_click(-999, super::model::encode_drag(2, limit), 5, true);
            self.emit_changes();
        }
        self.reset_drag();
    }

    /// Whether the open tab has more items than fit, so the scrollbar is live. java:
    /// `GuiContainerCreative.needsScrollBars`.
    #[must_use]
    pub fn needs_scroll_bars(&self) -> bool {
        self.tab().has_item_grid() && self.item_list.len() > SCROLL_THRESHOLD
    }

    /// Switch tabs, rebuilding the item list and resetting the scroll. java:
    /// `GuiContainerCreative.setCurrentCreativeTab`.
    pub fn set_tab(&mut self, index: u8) {
        // java: `setCurrentCreativeTab` clears `dragSplittingSlots` (`:461`)
        // and nothing else, so no release can reach the old tab's slots. A drag stays armed when
        // a left release over a tab switched (`GuiContainerCreative.mouseReleased` returns before
        // `GuiContainer.mouseReleased` disarms it): the next press then neither clicks nor
        // re-arms, and a release of another button cancels it.
        self.drag_splitting_slots.clear();
        // The Search reset is deferred only because this Rust field needs a font to remeasure.
        // A second tab switch before the next draw cancels that deferred work: Java's
        // `setCurrentCreativeTab` resets the field synchronously only for the Search tab, and a
        // later non-Search switch must not let an old queued reset rebuild this tab's item list.
        self.pending_search_reset = false;
        self.selected_tab = index % TAB_COUNT;
        remember_selected_tab(self.selected_tab);
        let tab = self.tab();
        self.item_list = match tab {
            CreativeTab::Items(label) => creative::stacks_for_tab(label),
            // java: `setCurrentCreativeTab` does `searchField.setText("")` and then
            // `updateCreativeSearch()`, and an empty needle matches every tooltip line, so the
            // tab opens showing the **whole** pool. (The old comment here claimed the opposite
            // and the grid opened blank -- audit `F-P2-creativetab-01-4`.)
            CreativeTab::Search => {
                creative::search_stacks("", self.advanced_tooltips, self.language.as_ref())
            }
            CreativeTab::Inventory => Vec::new(),
        };
        self.search.visible = tab == CreativeTab::Search;
        self.search.can_lose_focus = tab != CreativeTab::Search;
        self.search.set_focused(tab == CreativeTab::Search);
        if tab == CreativeTab::Search {
            self.pending_search_reset = true;
        }
        self.scroll = 0.0;
        self.scroll_to(0.0);
    }

    /// java: `ContainerCreative.scrollTo` — fill the 45 visible cells from `item_list`, offset by
    /// whole rows. Cells past the end of the list are cleared, not left stale.
    pub fn scroll_to(&mut self, scroll: f32) {
        // java: `(itemList.size() + 9 - 1) / 9 - 5` — rows, rounded up, minus the visible five.
        let rows = self.item_list.len().div_ceil(9);
        let overflow = rows as i32 - 5;
        let first_row = ((f64::from(scroll * overflow as f32)) + 0.5) as i32;
        let first_row = first_row.max(0);
        for row in 0..5usize {
            for column in 0..9usize {
                let source = column as i32 + (row as i32 + first_row) * 9;
                let cell = usize::try_from(source)
                    .ok()
                    .and_then(|i| self.item_list.get(i))
                    .cloned();
                self.grid[row * 9 + column] = cell;
            }
        }
    }

    /// Re-run the search against the box's current contents. java:
    /// `GuiContainerCreative.updateCreativeSearch`.
    pub fn update_search(&mut self) {
        self.item_list = creative::search_stacks(
            &self.search.text(),
            self.advanced_tooltips,
            self.language.as_ref(),
        );
        self.scroll = 0.0;
        self.scroll_to(0.0);
    }

    /// Apply a pending tab-switch search reset, queued by [`Self::set_tab`]. Called at the top
    /// of [`Screen::draw`], which has a font and [`Screen::mouse_released`] — the method that
    /// drives tab switching — does not.
    ///
    /// This no longer has anything to do with *typed* keys: those are applied immediately by
    /// [`Screen::key_typed_with_font`] now that it is handed a font, with no queue in between.
    pub fn apply_pending_input(&mut self, font: &FontRenderer) {
        if std::mem::take(&mut self.pending_search_reset) {
            self.search.set_text("", font);
            self.update_search();
        }
    }

    /// Every visible cell and its **window-relative** position. java: `ContainerCreative`'s
    /// constructor for the grid tabs, and `setCurrentCreativeTab`'s slot remapping for the
    /// Inventory tab.
    #[must_use]
    pub fn slot_rects(&self) -> Vec<(CreativeSlot, i32, i32)> {
        let mut out = Vec::with_capacity(GRID_CELLS + 10);
        if self.tab() == CreativeTab::Inventory {
            for index in 5..self.player.slots().len() {
                let j = index as i32;
                let (x, y) = if j < 9 {
                    // java: armour, laid out in two columns of two at 54/27 pixel spacing.
                    let n = j - 5;
                    (9 + (n / 2) * 54, 6 + (n % 2) * 27)
                } else {
                    let k = j - 9;
                    let y = if j >= 36 { 112 } else { 54 + (k / 9) * 18 };
                    (9 + (k % 9) * 18, y)
                };
                out.push((CreativeSlot::Player(index as u16), x, y));
            }
            out.push((CreativeSlot::Trash, 173, 112));
            return out;
        }

        for row in 0..5i32 {
            for column in 0..9i32 {
                out.push((
                    CreativeSlot::Grid((row * 9 + column) as u8),
                    9 + column * 18,
                    18 + row * 18,
                ));
            }
        }
        // The hotbar row is the player's real slots 36..=44.
        for column in 0..9i32 {
            out.push((
                CreativeSlot::Player(36 + column as u16),
                9 + column * 18,
                112,
            ));
        }
        out
    }

    /// The empty-slot placeholder sprite for a creative-screen cell, if it has one. java:
    /// `GuiContainerCreative.CreativeSlot.getSlotTexture` (`:994-997`) forwards to the wrapped
    /// slot, so only the inventory tab's four armour cells answer.
    #[must_use]
    fn slot_background(&self, slot: CreativeSlot) -> Option<&'static str> {
        match slot {
            CreativeSlot::Player(index) => self
                .player
                .slots()
                .get(usize::from(index))
                .and_then(|s| s.background_sprite),
            CreativeSlot::Grid(_) | CreativeSlot::Trash => None,
        }
    }

    /// java: `GuiContainer.isPointInRegion` — one pixel of slack on every side, exactly as
    /// [`super::screen::ContainerScreen::is_point_in_region`] reproduces it.
    #[must_use]
    pub fn is_point_in_region(
        &self,
        left: i32,
        top: i32,
        width: i32,
        height: i32,
        point_x: i32,
        point_y: i32,
    ) -> bool {
        let x = point_x - self.gui_left;
        let y = point_y - self.gui_top;
        x >= left - 1 && x < left + width + 1 && y >= top - 1 && y < top + height + 1
    }

    /// The cell under a screen point, if any.
    #[must_use]
    pub fn slot_at(&self, x: i32, y: i32) -> Option<CreativeSlot> {
        self.slot_rects()
            .into_iter()
            .find(|(_, sx, sy)| self.is_point_in_region(*sx, *sy, 16, 16, x, y))
            .map(|(slot, _, _)| slot)
    }

    /// The tab whose plate contains a screen point, if any. java: the
    /// `for (CreativeTabs t : creativeTabArray) if (func_147049_a(t, i, j))` loops.
    #[must_use]
    pub fn tab_at(&self, x: i32, y: i32) -> Option<u8> {
        let (rx, ry) = (x - self.gui_left, y - self.gui_top);
        (0..TAB_COUNT).find(|index| tab_contains(*index, rx, ry))
    }

    /// The scrollbar's drag gutter, in absolute screen pixels: `(left, top, right, bottom)`.
    /// java: `GuiContainerCreative.drawScreen`'s `k`, `l`, `i1`, `j1`.
    #[must_use]
    pub const fn scrollbar_gutter(&self) -> (i32, i32, i32, i32) {
        let left = self.gui_left + 175;
        let top = self.gui_top + 18;
        (left, top, left + 14, top + 112)
    }

    /// Set the scroll from a mouse Y inside the gutter.
    ///
    /// java: `((mouseY - l) - 7.5F) / ((j1 - l) - 15.0F)` — the `7.5`/`15` are the handle's
    /// half-height and full height, so the handle tracks the cursor's centre rather than its
    /// top edge.
    pub fn scroll_from_mouse(&mut self, mouse_y: i32) {
        let (_, top, _, bottom) = self.scrollbar_gutter();
        let span = (bottom - top) as f32 - 15.0;
        let scroll = ((mouse_y - top) as f32 - 7.5) / span;
        self.scroll = scroll.clamp(0.0, 1.0);
        let s = self.scroll;
        self.scroll_to(s);
    }

    /// java: `GuiContainerCreative.handleMouseInput`'s wheel branch. `delta` is positive when the
    /// wheel turns away from the user, i.e. scroll *up*.
    pub fn scroll_by_wheel(&mut self, delta: f32) {
        if delta == 0.0 || !self.needs_scroll_bars() {
            return;
        }
        // java: `int j = itemList.size() / 9 - 5`, then `currentScroll -= (double)i / (double)j`.
        // `j` is zero for 46..=53 items and the division is done in **double**, so vanilla gets
        // an infinity that the clamp below absorbs rather than an ArithmeticException.
        let steps = (self.item_list.len() / 9) as f64 - 5.0;
        let direction = if delta > 0.0 { 1.0f64 } else { -1.0 };
        let next = f64::from(self.scroll) - direction / steps;
        self.scroll = (next as f32).clamp(0.0, 1.0);
        let s = self.scroll;
        self.scroll_to(s);
    }

    /// The stack a cell currently shows.
    #[must_use]
    pub fn stack_at(&self, slot: CreativeSlot) -> Option<&ItemStack> {
        match slot {
            CreativeSlot::Grid(i) => self.grid.get(i as usize).and_then(Option::as_ref),
            CreativeSlot::Player(id) => self.player.slot_stack(id as usize),
            CreativeSlot::Trash => None,
        }
    }

    /// The tooltip for whatever the cursor is over, or `None`. An item's is suppressed while a
    /// stack is held, exactly as [`super::screen::ContainerScreen::hovered_tooltip`] is.
    #[must_use]
    pub fn hovered_tooltip(&self) -> Option<Vec<String>> {
        let hovered = self.hovered?;
        // java: `GuiContainerCreative.drawScreen:613-616` draws
        // "inventory.binSlot" whenever the mouse is over the destroy cell, with no held-stack
        // test: carrying a stack to it still shows the label.
        if hovered == CreativeSlot::Trash {
            return Some(vec![self.translate("inventory.binSlot")]);
        }
        if self.player.cursor().is_some() {
            return None;
        }
        let stack = self.stack_at(hovered)?;
        let mut lines = self.language.as_ref().map_or_else(
            || {
                tooltip::build_with_map_data(
                    stack,
                    self.advanced_tooltips,
                    self.tooltip_map_data.as_ref(),
                )
            },
            |language| {
                tooltip::build_localized_with_map_data(
                    stack,
                    self.advanced_tooltips,
                    language,
                    self.tooltip_map_data.as_ref(),
                )
            },
        );
        if self.tab() == CreativeTab::Search {
            if let Some(source_tab) = creative::source_tab(stack) {
                lines.insert(
                    1,
                    format!(
                        "\u{a7}l\u{a7}9{}",
                        self.translate(&source_tab.translation_key())
                    ),
                );
            }
        }
        Some(tooltip_box::recolour_tooltip_lines(
            lines,
            tooltip::rarity(stack),
        ))
    }

    /// Returns the map id of the currently tooltip-eligible cell at `mouse`.
    #[must_use]
    pub fn hovered_map_id_at(&self, (mouse_x, mouse_y): (i32, i32)) -> Option<i32> {
        if !self.advanced_tooltips || self.player.cursor().is_some() {
            return None;
        }
        self.slot_rects()
            .into_iter()
            .find(|(_, x, y)| self.is_point_in_region(*x, *y, 16, 16, mouse_x, mouse_y))
            .and_then(|(slot, _, _)| self.stack_at(slot))
            .filter(|stack| stack_ext::props(stack).name == "minecraft:filled_map")
            .map(|stack| i32::from(stack.damage))
    }

    /// Draw this screen using the live record for its hovered filled map, if one exists.
    pub fn draw_with_map_data(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        map_data: Option<&MapData>,
    ) {
        let current = self
            .tooltip_map_data
            .as_ref()
            .map(|data| (data.id, data.scale));
        let incoming = map_data.map(|data| (data.id, data.scale));
        if current != incoming {
            self.tooltip_map_data = map_data.cloned();
        }
        <Self as Screen>::draw(self, d, font, ctx, mouse);
    }

    /// Return signed metadata ids of filled-map stacks that can be visible on this frame.
    ///
    /// The application uses this bounded list to fetch only the corresponding world records;
    /// all item-grid cells are included because scrolling can expose any of them before the next
    /// frame. The player inventory and cursor are included for the Inventory tab and drag path.
    #[must_use]
    pub fn map_ids(&self) -> Vec<i32> {
        let mut ids = BTreeSet::new();
        let mut collect = |stack: &ItemStack| {
            if stack_ext::props(stack).name == "minecraft:filled_map" {
                ids.insert(i32::from(stack.damage));
            }
        };
        for stack in self.grid.iter().filter_map(Option::as_ref) {
            collect(stack);
        }
        for stack in self
            .player
            .slots()
            .iter()
            .filter_map(|slot| self.player.slot_stack(usize::from(slot.id)))
        {
            collect(stack);
        }
        if let Some(stack) = self.player.cursor() {
            collect(stack);
        }
        ids.into_iter().collect()
    }

    /// Retain the host-facing map-record seam without feeding live
    /// pixels into the cells, exactly as [`super::screen::ContainerScreen::set_map_render_data`]
    /// does: `GuiContainerCreative` inherits `GuiContainer.drawSlot`, whose `RenderItem` uses the
    /// static `map_filled` item model. The advanced tooltip's record still arrives through
    /// [`Self::draw_with_map_data`]. The argument is intentionally ignored.
    pub fn set_map_render_data(&mut self, _records: Vec<MapData>) {}

    /// Draw with both the live tooltip record and the bounded set of map records needed for GUI
    /// imagery. This is the application-facing entry point; [`Self::draw_with_map_data`] remains
    /// available for hosts that only need the tooltip seam.
    pub fn draw_with_map_records(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        tooltip_map_data: Option<&MapData>,
        records: Vec<MapData>,
    ) {
        self.set_map_render_data(records);
        self.draw_with_map_data(d, font, ctx, mouse, tooltip_map_data);
    }

    fn translate(&self, key: &str) -> String {
        self.language
            .as_ref()
            .map_or_else(|| key.to_owned(), |language| language.tr(key).to_owned())
    }

    /// Emit a `CreativeInventoryAction` for every player slot whose contents changed since the
    /// listener last looked. java: `inventoryContainer.detectAndSendChanges()` into
    /// `CreativeCrafting.sendSlotContents` (`ItemStack.areItemStacksEqual`, a copy kept).
    ///
    /// The mirror is the window-0 container's own
    /// (`Container.inventoryItemStacks`), not this screen's: the host keeps that container from
    /// one opening to the next, so reopening uploads only what changed meanwhile, as in Java,
    /// instead of every occupied slot again.
    fn emit_changes(&mut self) {
        let changes = self.player.detect_listener_changes();
        self.outgoing
            .extend(
                changes
                    .into_iter()
                    .map(|(slot, stack)| c2s::CreativeInventoryAction {
                        slot_id: slot as i16,
                        stack,
                    }),
            );
    }

    /// Send the content-tab hotbar slot directly after its creative-container click.
    ///
    /// java: `GuiContainerCreative.handleMouseClick`'s final `else` calls
    /// `PlayerControllerMP.sendSlotPacket` for the clicked slot, rather than running
    /// `inventoryContainer.detectAndSendChanges()`. This distinction is observable even when
    /// the slot stays empty (Java still sends `C10(slot, null)`) and also keeps the persistent
    /// `inventoryItemStacks` mirror untouched until the Inventory-tab listener path runs.
    fn emit_content_slot(&mut self, id: u16) {
        self.outgoing.push(c2s::CreativeInventoryAction {
            slot_id: id as i16,
            stack: self.player.slot_stack(id as usize).cloned(),
        });
    }

    /// java: the content-tab tail of
    /// `GuiContainerCreative.handleMouseClick`: `if (Container.getDragEvent(clickedButton) == 2)`
    /// all nine hotbar slots, else the clicked slot. `getDragEvent` is `clickedButton & 3`, so
    /// the middle button, a drag's end and the hotbar keys 3 and 7 (indices 2 and 6, a mode-2
    /// click's `clickedButton`) take the burst.
    fn emit_content_tail(&mut self, id: u16, clicked_button: i32) {
        if clicked_button & 3 == 2 {
            self.emit_hotbar_burst();
        } else {
            self.emit_content_slot(id);
        }
    }

    /// Run a click on one cell. `shift` is vanilla's `clickType == 1`.
    ///
    /// java: `GuiContainerCreative.handleMouseClick`, restricted to the three cases this screen
    /// can reach: the read-only tab grid ([`creative::click_tab_source`]), a real player slot
    /// (the ordinary container state machine with `creative = true`, plus the resulting
    /// `CreativeInventoryAction`s), and the destroy-item cell.
    pub fn click_slot(&mut self, slot: CreativeSlot, button: i32, shift: bool) {
        let pick_block = button == self.bindings.pick_block_button;
        self.click_slot_inner(slot, button, shift, pick_block);
    }

    fn click_slot_inner(&mut self, slot: CreativeSlot, button: i32, shift: bool, pick_block: bool) {
        // java: `handleMouseClick` sets `field_147057_D` before any slot-specific early return.
        // If Search is open, its next keystroke clears the field before processing that key.
        self.clear_search_on_next_key = true;
        match slot {
            CreativeSlot::Grid(index) => {
                let Some(Some(source)) = self.grid.get(index as usize).cloned() else {
                    // java: click type 3 (the middle/pick-block button) returns before the
                    // ordinary cursor/source branches, so an empty virtual cell is a no-op even
                    // when the cursor is held. Left/right releases intentionally clear a held
                    // cursor here, matching the final `else` in handleMouseClick.
                    if !pick_block && self.player.cursor().is_some() {
                        self.player.set_cursor(None);
                    }
                    return;
                };
                let cursor = self.player.cursor().cloned();
                // java: GuiContainer.mouseClicked turns the pick-block mouse binding into
                // clickType 3. The creative grid consumes that mode before its ordinary
                // clickedButton test: only an empty cursor receives a max-size clone.
                if pick_block {
                    if cursor.is_none() {
                        let mut full = source;
                        full.count = stack_ext::max_stack_size(&full);
                        self.player.set_cursor(Some(full));
                    }
                    return;
                }
                let next = creative::click_tab_source(cursor, &source, button, shift);
                self.player.set_cursor(next);
            }
            CreativeSlot::Player(id) => {
                // java: a middle click is delivered as clickType 3. On a content tab the
                // CreativeScreen tail then uploads the entire hotbar (a raw clickedButton of 2
                // also satisfies getDragEvent(2) == 2); the Inventory tab uses the normal
                // inventory-container listener path.
                if pick_block {
                    self.player.slot_click(id as i32, button as u8, 3, true);
                    if self.tab() == CreativeTab::Inventory {
                        self.emit_changes();
                    } else {
                        self.emit_content_tail(id, button);
                    }
                    return;
                }
                if self.tab() != CreativeTab::Inventory && shift {
                    // `ContainerCreative.transferStackInSlot` clears a hotbar slot instead of
                    // running ContainerPlayer's main-inventory merge, then uploads that one slot.
                    // java: `Container.slotClick` runs mode 1 only
                    // for buttons 0 and 1; another button's shift-release still reaches the tail.
                    if button == 0 || button == 1 {
                        self.player.set_slot_contents(id as usize, None);
                    }
                    self.emit_content_tail(id, button);
                    return;
                }
                let mode = u8::from(shift);
                self.player
                    .slot_click(i32::from(id), button as u8, mode, true);
                if self.tab() == CreativeTab::Inventory {
                    self.emit_changes();
                } else {
                    // Java's ContainerCreative tail uploads exactly this mapped hotbar slot,
                    // even when the post-click stack is unchanged or null.
                    self.emit_content_tail(id, button);
                }
            }
            CreativeSlot::Trash => {
                if shift {
                    self.outgoing
                        .extend((0..self.player.slots().len()).map(|index| {
                            c2s::CreativeInventoryAction {
                                slot_id: index as i16,
                                stack: None,
                            }
                        }));
                    return;
                }
                // java: `field_147064_C` is backed by the scratch inventory, so dropping onto it
                // simply discards the cursor stack.
                self.player.set_cursor(None);
            }
        }
    }

    fn drop_cursor_outside(&mut self, button: i32) {
        let Some(mut held) = self.player.cursor().cloned() else {
            return;
        };
        let drop_count = if self.tab() == CreativeTab::Inventory || button == 0 {
            held.count
        } else if button == 1 {
            1
        } else {
            return;
        };
        let mut dropped = held.clone();
        dropped.count = drop_count;
        held.count -= drop_count;
        self.player.set_cursor((held.count != 0).then_some(held));
        self.outgoing.push(c2s::CreativeInventoryAction {
            slot_id: -1,
            stack: Some(dropped),
        });
    }

    fn draw_tab(&self, d: &mut DrawList, font: &mut FontRenderer, index: u8, selected: bool) {
        let plate = tab_plate(index, selected);
        let tex = tabs_texture();
        d.blit(
            &tex,
            self.gui_left + plate.x,
            self.gui_top + plate.y,
            plate.u,
            plate.v,
            28,
            32,
        );
        let (name, meta) = TABS[index as usize].icon_item();
        if let Some(id) = mc_data::item_by_name(name) {
            let stack = ItemStack {
                item_id: id.0 as i16,
                count: 1,
                damage: meta,
                nbt: None,
            };
            render::draw_item_in_slot(
                d,
                &self.atlas,
                font,
                self.gui_left + plate.icon.0,
                self.gui_top + plate.icon.1,
                &stack,
            );
        }
    }

    /// java: `GuiContainer.drawSlot` (`:228-307`) for one creative cell.
    ///
    /// A swept cell shows the stack the sweep would leave there over a `0x80FFFFFF` rect, with a
    /// yellow count when the item's or the slot's limit cut it. A lone swept cell draws nothing
    /// (`:245-248`). A swept cell that can no longer take the held stack leaves the sweep, which
    /// recomputes the remnant (`:268-272`). An empty armour cell draws its silhouette and nothing
    /// else (`flag1`).
    fn draw_slot(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        slot: CreativeSlot,
        x: i32,
        y: i32,
    ) {
        let mut stack = self.stack_at(slot).cloned();
        let mut drag_rect = false;
        let mut yellow = None;
        let swept = match slot {
            CreativeSlot::Player(id)
                if self.drag_splitting && self.drag_splitting_slots.contains(&id) =>
            {
                Some(id)
            }
            _ => None,
        };
        // Only a swept cell copies the held stack (never an unswept one, so no per-cell clone).
        if let Some((id, cursor)) = swept.and_then(|id| Some((id, self.player.cursor()?.clone()))) {
            if self.drag_splitting_slots.len() == 1 {
                return;
            }
            // java: `Container.canAddItemToSlot(slotIn, itemstack1, true) &&
            // this.inventorySlots.canDragIntoSlot(slotIn)`. Only player slots are ever swept, and
            // `ContainerCreative.canDragIntoSlot` admits every one of them.
            if self.player.slot_stack_would_accept(id, &cursor) {
                let (preview, _, cap) = self.drag_split_preview(&cursor, id);
                stack = Some(preview);
                drag_rect = true;
                yellow = cap;
            } else {
                self.drag_splitting_slots.retain(|member| *member != id);
                self.update_drag_splitting();
            }
        }
        if stack.is_none() {
            if let Some(name) = self.slot_background(slot) {
                let uv = super::slot::background_sprite_uv(name);
                d.quad_rect(
                    &self.atlas,
                    (x as f32, y as f32, (x + 16) as f32, (y + 16) as f32),
                    (uv.min_u, uv.min_v, uv.max_u, uv.max_v),
                    0xFFFF_FFFF,
                );
                return; // java: `flag1 = true`
            }
        }
        if drag_rect {
            // java: `drawRect(i, j, i + 16, j + 16, -2130706433)`, under the ghost stack.
            d.gradient_rect(x, y, x + 16, y + 16, SLOT_HIGHLIGHT, SLOT_HIGHLIGHT);
        }
        if let Some(stack) = stack {
            let z_level = render::GUI_SLOT_ITEM_RENDER_Z_LEVEL;
            self.draw_stack(d, font, (x, y), &stack, yellow, z_level);
        }
    }

    /// java: `RenderItem.renderItemAndEffectIntoGUI` then
    /// `renderItemOverlayIntoGUI(..., altText)`: the ordinary cell drawing, or a yellow count in
    /// place of the stack size (the survival screen's capped-count path: the item model, the
    /// durability bar, then `"§e" + count`).
    fn draw_stack(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        (x, y): (i32, i32),
        stack: &ItemStack,
        yellow: Option<i32>,
        z_level: f32,
    ) {
        let Some(count) = yellow else {
            // java: `RenderItem` draws a filled map with the static
            // `map_filled` item model in every `GuiContainer` cell (the survival screen does the
            // same); live map pixels belong to held maps and item frames.
            render::draw_item_in_slot_at_z_level(d, &self.atlas, font, x, y, stack, z_level);
            return;
        };
        render::draw_item_icon_tinted_at_z_level(d, &self.atlas, x, y, stack, 0xFFFF_FFFF, z_level);
        render::draw_durability_bar(d, x, y, stack);
        let mut buf = [0u8; 16];
        let text = yellow_count_text(&mut buf, count);
        render::draw_count_text_with_override(d, font, x, y, stack, Some(text));
    }
}

/// java: `EnumChatFormatting.YELLOW + "" + count`, formatted into a stack
/// buffer (no allocation in the draw path).
fn yellow_count_text(buf: &mut [u8; 16], count: i32) -> &str {
    use std::io::Write as _;
    let len = {
        let mut cursor = std::io::Cursor::new(&mut buf[..]);
        // `§e` (3 bytes) and any i32 (at most 11) fit in 16 bytes.
        let _ = write!(cursor, "\u{a7}e{count}");
        cursor.position() as usize
    };
    std::str::from_utf8(&buf[..len]).unwrap_or("")
}

impl Default for CreativeScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl Screen for CreativeScreen {
    fn name(&self) -> &'static str {
        "container.creative"
    }

    // java: GuiContainerCreative.java:263 (initGui)
    fn init(&mut self, ctx: &ScreenCtx) {
        self.screen_width = ctx.width;
        self.update_active_potion_effects();
        self.gui_top = (ctx.height - Y_SIZE) / 2;
        self.search.rect.x = self.gui_left + 82;
        self.search.rect.y = self.gui_top + 6;
    }

    // java: GuiContainerCreative.java:72-80 (updateScreen). It calls
    // neither `searchField.updateCursorCounter()` nor `GuiContainer.updateScreen`, so the search
    // box's `cursorCounter` stays 0 and its caret is drawn steadily: it never blinks.
    fn tick(&mut self) {
        self.preview_age_ticks = self.preview_age_ticks.wrapping_add(1);
        self.update_active_potion_effects();
    }

    // java: GuiContainerCreative.java:574 (drawScreen) + :676 (drawGuiContainerBackgroundLayer)
    fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        (mouse_x, mouse_y): (i32, i32),
    ) {
        self.apply_pending_input(font);
        if self.is_scrolling {
            self.scroll_from_mouse(mouse_y);
        }

        d.gradient_rect(0, 0, ctx.width, ctx.height, WORLD_DIM_TOP, WORLD_DIM_BOTTOM);

        // java: every *unselected* tab first, then the panel, then the selected tab last, so the
        // open tab's plate paints over its neighbours' edges.
        for index in 0..TAB_COUNT {
            if index != self.selected_tab {
                self.draw_tab(d, font, index, false);
            }
        }

        let tab = self.tab();
        d.blit(
            &panel_texture(tab),
            self.gui_left,
            self.gui_top,
            0,
            0,
            X_SIZE,
            Y_SIZE,
        );
        if self.search.visible {
            self.search.draw(d, font);
        }
        if tab.has_item_grid() {
            let (bar_left, bar_top, _, _) = self.scrollbar_gutter();
            let u = SCROLLBAR_U + if self.needs_scroll_bars() { 0 } else { 12 };
            d.blit(
                &tabs_texture(),
                bar_left,
                bar_top + (95.0 * self.scroll) as i32,
                u,
                0,
                12,
                15,
            );
        }
        self.draw_tab(d, font, self.selected_tab, true);

        // java: `GuiContainerCreative.drawGuiContainerBackgroundLayer`'s
        // `if (creativetabs == CreativeTabs.tabInventory)` block, right after the selected tab's
        // plate. Unlike `GuiInventory`'s survival panel, this call site is not mouse-lagged —
        // vanilla passes this frame's `mouseX`/`mouseY` straight through.
        if tab == CreativeTab::Inventory {
            let (px, py) = (self.gui_left + 43, self.gui_top + 45);
            self.player_preview.draw_with_snapshot(
                d,
                px,
                py,
                20,
                (px - mouse_x) as f32,
                (py - 30 - mouse_y) as f32,
                self.player_skin,
                self.preview_age_ticks as f32 + ctx.partial_ticks,
                self.player_skin_texture.as_ref(),
                &self.player_preview_snapshot,
                &self.atlas,
            );
        }

        // java: `GuiContainerCreative extends GuiContainer`, so every
        // cell goes through the `GuiContainer.drawScreen` slot loop (`:121-139`) the survival
        // screens use: `drawSlot` (the drag preview, the placeholder silhouette that
        // `CreativeSlot.getSlotTexture` (`:994-997`) forwards from the wrapped armour slot, the
        // item), then, for the cell under the mouse, the `0x80FFFFFF` hover rect drawn over all
        // of it with depth off, placeholder cells included. `theSlot` is the last cell hit.
        let mut hovered = None;
        for (slot, x, y) in self.slot_rects() {
            let (ax, ay) = (self.gui_left + x, self.gui_top + y);
            self.draw_slot(d, font, slot, ax, ay);
            if self.is_point_in_region(x, y, 16, 16, mouse_x, mouse_y) {
                hovered = Some(slot);
                d.gradient_rect(ax, ay, ax + 16, ay + 16, SLOT_HIGHLIGHT, SLOT_HIGHLIGHT);
            }
        }
        self.hovered = hovered;

        // java: `drawGuiContainerForegroundLayer` draws the tab's own name over the grid.
        if tab.draws_title() {
            let label = self.translate(&tab.translation_key());
            font.draw_string(
                d,
                &label,
                (self.gui_left + 8) as f32,
                (self.gui_top + 6) as f32,
                TITLE_COLOR,
                false,
            );
        }

        // java: `GuiContainer.drawScreen:145-170`: while more than one
        // slot is swept the held stack shows `dragSplittingRemnant`, with a yellow `0` when
        // nothing would be left (a negative remnant keeps the ordinary red count).
        if let Some(stack) = self.player.cursor() {
            let mut stack = stack.clone();
            let mut yellow = None;
            if self.drag_splitting && self.drag_splitting_slots.len() > 1 {
                stack.count = self.drag_splitting_remnant;
                if stack.count == 0 {
                    yellow = Some(0);
                }
            }
            self.draw_stack(
                d,
                font,
                (mouse_x - 8, mouse_y - 8),
                &stack,
                yellow,
                render::GUI_CURSOR_ITEM_RENDER_Z_LEVEL,
            );
        }

        // `GuiContainerCreative.drawScreen` reaches this through its
        // `InventoryEffectRenderer` superclass before adding the creative-tab hover label.
        self.draw_active_potion_effects(d, font);

        // java: the tab hover label uses `isPointInRegion(j + 3, k + 3, 23, 27, ...)`, a box
        // inset inside the plate, and takes priority over the item tooltip.
        if let Some(index) = (0..TAB_COUNT).find(|index| {
            let column = tab_column(*index);
            let top = if tab_is_first_row(*index) {
                -32
            } else {
                Y_SIZE
            };
            let mut left = 28 * column;
            if column == 5 {
                left = X_SIZE - 28 + 2;
            } else if column > 0 {
                left += column;
            }
            self.is_point_in_region(left + 3, top + 3, 23, 27, mouse_x, mouse_y)
        }) {
            let label = self.translate(&TABS[index as usize].translation_key());
            // Through the `scrollable_tooltips` mod: exactly
            // `tooltip_box::draw` while it is off.
            let (w, h) = (ctx.width, ctx.height);
            self.tooltip_scroll
                .draw(d, font, &[label], mouse_x, mouse_y, w, h);
        } else if let Some(lines) = self.hovered_tooltip() {
            let (w, h) = (ctx.width, ctx.height);
            self.tooltip_scroll
                .draw(d, font, &lines, mouse_x, mouse_y, w, h);
        } else {
            self.tooltip_scroll.clear();
        }
    }

    // java: GuiContainerCreative.java:406 (mouseClicked)
    fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> ScreenAction {
        self.mouse_clicked_inner(x, y, button, false)
    }

    // java: GuiContainerCreative.java:428 (mouseReleased). Without modifiers, the shift of the
    // last press stands in for the release's (`mouse_released_with_mods` has the live one).
    fn mouse_released(&mut self, x: i32, y: i32, button: i32) {
        self.mouse_released_inner(x, y, button, self.last_press_shift);
    }

    fn mouse_dragged(&mut self, x: i32, y: i32, _button: i32, _held_ms: u64) {
        if self.is_scrolling {
            self.scroll_from_mouse(y);
        }
        // java: `GuiContainer.mouseClickMove` runs for every move with a
        // button held, the scrollbar's drag (polled in `drawScreen`) notwithstanding. It only
        // gathers valid slots; the stacks move in `mouseReleased`'s mode-5 sequence.
        if self.drag_splitting {
            if let Some(slot) = self.slot_at(x, y) {
                self.add_drag_slot(slot);
            }
        }
    }

    fn mouse_wheel(&mut self, _x: i32, _y: i32, delta: f32) {
        self.scroll_by_wheel(delta);
    }

    // java: GuiContainerCreative.java:306 (keyTyped)
    //
    // Overrides `key_typed_with_font`, not plain `key_typed`: this is the half of S15's §13 row
    // that stayed open after `TextField::mouse_clicked_cached` closed the click-to-position half
    // (see the module doc). `ScreenStack` calls this method, with a font, so the search field is
    // edited directly instead of being queued for `draw` to apply next frame.
    fn key_typed_with_font(
        &mut self,
        typed: Option<char>,
        key: i32,
        mods: Modifiers,
        font: &FontRenderer,
    ) -> ScreenAction {
        let search_tab = self.tab() == CreativeTab::Search;
        // java: `GuiContainerCreative.keyTyped:308-313`. Off the search tab, a key event while
        // `GameSettings.isKeyDown(keyBindChat)` switches to search and does nothing else: no
        // close, no hotbar swap, and the character is not typed. Only the chat key's own press
        // is seen here; Java would also switch on another key typed while chat is held.
        if !search_tab && self.bindings.chat_key != 0 && key == self.bindings.chat_key {
            if let Some(index) = CreativeTab::Search.index() {
                self.set_tab(index);
            }
            return ScreenAction::None;
        }
        if !search_tab && (key == keys::ESCAPE || key == self.bindings.inventory_key) {
            return ScreenAction::Pop;
        }
        // A key can arrive before the first post-tab-switch draw. Java has already cleared the
        // field in `setCurrentCreativeTab`, so finish the font-dependent deferred reset before
        // this key is appended; otherwise a stale query would briefly receive the new character
        // and then be erased by the next draw.
        if search_tab {
            self.apply_pending_input(font);
        }
        // java: this happens before hotbar shortcuts and before the search field handles the
        // current key. It is distinct from the draw-time reset queued by switching tabs.
        if search_tab && std::mem::take(&mut self.clear_search_on_next_key) {
            self.search.set_text("", font);
        }
        // java: `checkHotbarKeys` runs *before* the search field, so a digit typed while
        // hovering a stack moves the item instead of appearing in the box.
        if self.check_hotbar_keys(key) {
            return ScreenAction::None;
        }
        if search_tab
            && self
                .search
                .key_typed(typed, key, mods, &mut self.clipboard, font)
        {
            self.update_search();
            return ScreenAction::None;
        }
        // java: `super.keyTyped` (which owns Q/drop) is reached only if the search field did not
        // consume the key. Printable Q therefore types into Search instead of dropping a slot.
        // java: `GuiContainer.keyTyped` clicks for the pick and drop
        // keys only `if (theSlot != null && theSlot.getHasStack())`.
        let hovered_has_stack = self
            .hovered
            .is_some_and(|slot| self.stack_at(slot).is_some());
        if key == self.bindings.pick_block_key {
            if let (Some(slot), true) = (self.hovered, hovered_has_stack) {
                // java: `GuiContainer.keyTyped` supplies clickedButton 0 and clickType 3 for a
                // keyboard-bound pick action; it must not make a former middle-click binding
                // continue firing from the mouse.
                self.click_slot_inner(slot, 0, false, true);
            }
            return ScreenAction::None;
        }
        if key == self.bindings.drop_key && hovered_has_stack {
            // java: `handleMouseClick(theSlot, ..., 4)` sets
            // `field_147057_D` before anything else.
            self.clear_search_on_next_key = true;
            match self.hovered {
                Some(CreativeSlot::Grid(index)) => {
                    if let Some(mut dropped) = self.stack_at(CreativeSlot::Grid(index)).cloned() {
                        dropped.count = if mods.ctrl {
                            stack_ext::max_stack_size(&dropped)
                        } else {
                            1
                        };
                        self.outgoing.push(c2s::CreativeInventoryAction {
                            slot_id: -1,
                            stack: Some(dropped),
                        });
                    }
                    return ScreenAction::None;
                }
                Some(CreativeSlot::Player(id)) if self.tab() == CreativeTab::Inventory => {
                    // java: `slotIn.decrStackSize(clickedButton == 0 ? 1 : getMaxStackSize())`
                    // directly, not `Container.slotClick` mode 4 (which needs an empty cursor),
                    // then `sendPacketDropItem`; no `detectAndSendChanges`.
                    if let Some(mut remaining) = self.player.slot_stack(id as usize).cloned() {
                        let amount = if mods.ctrl {
                            stack_ext::max_stack_size(&remaining)
                        } else {
                            1
                        };
                        let mut dropped = remaining.clone();
                        dropped.count = amount.min(remaining.count);
                        remaining.count -= dropped.count;
                        self.player.set_slot_contents(
                            id as usize,
                            (remaining.count > 0).then_some(remaining),
                        );
                        self.outgoing.push(c2s::CreativeInventoryAction {
                            slot_id: -1,
                            stack: Some(dropped),
                        });
                    }
                    return ScreenAction::None;
                }
                Some(CreativeSlot::Player(id)) => {
                    self.player
                        .slot_click(i32::from(id), u8::from(mods.ctrl), 4, true);
                    // Q/drop on a content-tab hotbar cell is handled by the creative container's
                    // direct slot upload, not the player-container diff.
                    self.emit_content_tail(id, i32::from(mods.ctrl));
                    return ScreenAction::None;
                }
                Some(CreativeSlot::Trash) | None => {}
            }
        }
        if key == keys::ESCAPE || key == self.bindings.inventory_key {
            return ScreenAction::Pop;
        }
        ScreenAction::None
    }

    fn pauses_game(&self) -> bool {
        false
    }

    fn allows_user_input(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::super::screen::DEFAULT_PICK_BLOCK_BUTTON;
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(427, 240, 1, false), 0.0)
    }

    fn opened() -> CreativeScreen {
        remember_selected_tab(0);
        let mut s = CreativeScreen::new();
        s.init(&ctx());
        s
    }

    fn stack(name: &str, count: i32) -> ItemStack {
        let id = mc_data::item_by_name(name).expect("known item");
        ItemStack {
            item_id: id.0 as i16,
            count,
            damage: 0,
            nbt: None,
        }
    }

    fn effects(count: usize) -> Vec<ActiveEffect> {
        (0..count)
            .map(|index| ActiveEffect {
                icon_index: -1,
                name: format!("Effect {index}"),
                duration: "1:23".to_owned(),
                beneficial: true,
            })
            .collect()
    }

    // ------------------------------------------------------------ tab strip geometry

    /// The whole point of re-deriving `CreativeTabs`' registration order: the plate positions
    /// are a pure function of the index, so a wrong order puts every tab in the wrong place.
    #[test]
    fn the_top_row_sits_above_the_panel_and_the_bottom_row_below_it() {
        for index in 0..6u8 {
            assert_eq!(tab_plate(index, false).y, -28, "tab {index}");
        }
        for index in 6..12u8 {
            assert_eq!(tab_plate(index, false).y, Y_SIZE - 4, "tab {index}");
        }
    }

    #[test]
    fn selection_and_row_pick_one_of_four_texture_rows() {
        assert_eq!(tab_plate(0, false).v, 0, "top row, idle");
        assert_eq!(tab_plate(0, true).v, 32, "top row, selected");
        assert_eq!(tab_plate(6, false).v, 64, "bottom row, idle");
        assert_eq!(tab_plate(6, true).v, 96, "bottom row, selected");
    }

    /// Columns 1..=4 are nudged one extra pixel right *per column*; column 5 is pinned flush to
    /// the panel's right edge instead of following the pattern.
    #[test]
    fn tab_columns_reproduce_vanillas_per_column_nudge_and_the_pinned_last_column() {
        let xs: Vec<i32> = (0..6u8).map(|i| tab_plate(i, false).x).collect();
        assert_eq!(xs, vec![0, 29, 58, 87, 116, X_SIZE - 28]);
        // The same six x's repeat on the bottom row.
        let bottom: Vec<i32> = (6..12u8).map(|i| tab_plate(i, false).x).collect();
        assert_eq!(bottom, xs);
    }

    #[test]
    fn the_tab_icon_sits_six_pixels_in_and_leans_toward_the_panel() {
        // java: `i1 + 8 + (flag1 ? 1 : -1)` — the top row's icon is a pixel lower, the bottom
        // row's a pixel higher, so both lean toward the item grid.
        assert_eq!(tab_plate(0, false).icon, (6, -28 + 9));
        assert_eq!(tab_plate(6, false).icon, (6, Y_SIZE - 4 + 7));
    }

    /// Two pinned vanilla quirks in one test: the hit box is inclusive at both ends (29x33, not
    /// 28x32), and column 5's hit box is two pixels right of where the plate is drawn.
    #[test]
    fn the_tab_hit_box_is_inclusive_and_column_five_is_offset_from_its_own_art() {
        assert!(tab_contains(0, 0, -32), "top-left corner is inside");
        assert!(tab_contains(0, 28, 0), "the 29th column is still inside");
        assert!(!tab_contains(0, 29, 0));
        assert!(tab_contains(0, 0, -32 + 32), "the 33rd row is still inside");
        assert!(!tab_contains(0, 0, 1));

        let drawn = tab_plate(5, false).x;
        assert!(
            !tab_contains(5, drawn, -10),
            "the plate's own left column is *outside* the hit box"
        );
        assert!(tab_contains(5, drawn + 2, -10));
    }

    #[test]
    fn every_tab_index_has_a_disjoint_hit_box_on_its_own_row() {
        for a in 0..TAB_COUNT {
            for b in 0..TAB_COUNT {
                if a == b {
                    continue;
                }
                let plate = tab_plate(a, false);
                // The centre of a's plate must never land in b's box.
                let (cx, cy) = (plate.x + 14, plate.y + 16);
                assert!(
                    !(tab_contains(b, cx, cy) && !tab_contains(a, cx, cy)),
                    "tab {a}'s centre falls only inside tab {b}"
                );
            }
        }
    }

    #[test]
    fn clicking_a_tab_switches_on_release_not_on_press() {
        let mut s = opened();
        assert_eq!(s.selected_tab(), 0);
        let plate = tab_plate(2, false);
        let (x, y) = (s.gui_left() + plate.x + 14, s.gui_top() + plate.y + 16);
        assert_eq!(s.tab_at(x, y), Some(2));
        s.mouse_clicked(x, y, 0);
        assert_eq!(s.selected_tab(), 0, "a press must not switch tabs");
        s.mouse_released(x, y, 0);
        assert_eq!(s.selected_tab(), 2);
        assert_eq!(s.tab(), CreativeTab::Items("redstone"));
    }

    #[test]
    fn reopening_restores_the_process_wide_selected_tab() {
        remember_selected_tab(0);
        {
            let mut first = CreativeScreen::new();
            first.set_tab(CreativeTab::Inventory.index().unwrap());
            assert_eq!(first.tab(), CreativeTab::Inventory);
        }

        let reopened = CreativeScreen::new();
        assert_eq!(
            reopened.tab(),
            CreativeTab::Inventory,
            "java's static selectedTabIndex survives closing the prior screen"
        );
        assert!(
            reopened
                .slot_rects()
                .iter()
                .any(|(slot, _, _)| *slot == CreativeSlot::Trash),
            "restoring Inventory must also build its special slot layout"
        );
        remember_selected_tab(0);
    }

    // ------------------------------------------------------------ the grid and scrolling

    #[test]
    fn a_content_tab_fills_the_grid_from_the_top_of_its_item_list() {
        let s = opened();
        assert!(!s.item_list().is_empty());
        assert_eq!(s.grid().len(), GRID_CELLS);
        let first = &s.item_list()[0];
        assert_eq!(
            s.grid()[0].as_ref().map(|st| st.item_id),
            Some(first.item_id)
        );
        assert_eq!(s.grid()[0].as_ref().map(|st| st.count), Some(1));
    }

    #[test]
    fn scrolling_to_the_bottom_shows_the_last_row_and_pads_with_empty_cells() {
        let mut s = opened();
        let total = s.item_list().len();
        assert!(total > GRID_CELLS, "building blocks must overflow one page");
        s.scroll_to(1.0);
        let last = s.item_list().last().unwrap();
        assert!(
            s.grid()
                .iter()
                .flatten()
                .any(|st| st.item_id == last.item_id && st.damage == last.damage),
            "the final item has to be on screen at scroll = 1"
        );
        // Cells past the end are cleared, not left holding the previous page's items.
        let filled = s.grid().iter().flatten().count();
        let rows = total.div_ceil(9);
        let expected = total - (rows - 5) * 9;
        assert_eq!(filled, expected.min(GRID_CELLS));
    }

    #[test]
    fn scrolling_never_leaves_a_stale_cell_behind() {
        let mut s = opened();
        s.scroll_to(1.0);
        s.scroll_to(0.0);
        let first = &s.item_list()[0];
        assert_eq!(
            s.grid()[0].as_ref().map(|st| st.item_id),
            Some(first.item_id)
        );
    }

    #[test]
    fn the_scrollbar_is_only_live_when_the_tab_overflows() {
        let mut s = opened();
        assert!(s.needs_scroll_bars(), "building blocks overflows");
        let inventory = CreativeTab::Inventory.index().unwrap();
        s.set_tab(inventory);
        assert!(!s.needs_scroll_bars(), "the inventory tab has no grid");
        let search = CreativeTab::Search.index().unwrap();
        s.set_tab(search);
        assert!(
            s.needs_scroll_bars(),
            "java: an empty needle lists the whole pool, so the search tab overflows too"
        );
    }

    #[test]
    fn dragging_the_scrollbar_maps_the_cursor_centre_to_the_scroll_fraction() {
        let mut s = opened();
        let (left, top, right, bottom) = s.scrollbar_gutter();
        assert_eq!(right - left, 14);
        assert_eq!(bottom - top, 112);
        s.mouse_clicked(left + 5, top + 5, 0);
        // java: `((mouseY - l) - 7.5F) / 97.0F`, so the handle's centre tracks the cursor.
        s.mouse_dragged(left + 5, top + 8, 0, 0);
        assert!((s.scroll() - 0.5 / 97.0).abs() < 1e-4, "{}", s.scroll());
        s.mouse_dragged(left + 5, bottom, 0, 0);
        assert_eq!(s.scroll(), 1.0, "clamped at the bottom");
        s.mouse_dragged(left + 5, top - 50, 0, 0);
        assert_eq!(s.scroll(), 0.0, "clamped at the top");
        s.mouse_released(left + 5, top, 0);
        s.mouse_dragged(left + 5, bottom, 0, 0);
        assert_eq!(s.scroll(), 0.0, "releasing ends the drag");
    }

    #[test]
    fn a_press_outside_the_gutter_does_not_start_a_drag() {
        let mut s = opened();
        let (left, top, _, _) = s.scrollbar_gutter();
        s.mouse_clicked(left - 3, top + 20, 0);
        s.mouse_dragged(left - 3, top + 90, 0, 0);
        assert_eq!(s.scroll(), 0.0);
    }

    #[test]
    fn drawing_an_advanced_grid_map_uses_the_live_map_record() {
        let mut s = opened();
        let mut map = stack("filled_map", 1);
        map.damage = 12;
        s.grid[0] = Some(map);
        s.advanced_tooltips = true;
        let (_, x, y) = s.slot_rects()[0];
        let mouse = (s.gui_left() + x + 8, s.gui_top() + y + 8);
        let mut data = MapData::new(12);
        data.scale = 2;

        s.draw_with_map_data(
            &mut DrawList::new(),
            &mut uniform_font(),
            &ctx(),
            mouse,
            Some(&data),
        );

        assert_eq!(s.hovered_map_id_at(mouse), Some(12));
        let lines = s.hovered_tooltip().expect("filled map tooltip");
        assert_eq!(
            lines.len(),
            4,
            "creative map tooltip has title, two lines, and the advanced id"
        );
        assert!(lines[0].starts_with("\u{a7}f"), "common title: {lines:?}");
        assert_eq!(
            &lines[1..3],
            ["\u{a7}7Scaling at 1:4", "\u{a7}7(Level 2/4)"],
            "creative renderToolTip applies gray in ItemMap line order"
        );
    }

    /// java: `GuiContainerCreative` inherits `GuiContainer.drawSlot`,
    /// whose `RenderItem` draws a filled map with the static `map_filled` model: no live map
    /// texture and no map icons in a cell, even with the map's record at hand (the survival
    /// screen's `drawing_a_filled_map_slot_uses_the_static_item_model_even_with_live_map_data`).
    #[test]
    fn drawing_a_creative_map_cell_uses_the_static_item_model_even_with_live_map_data() {
        let mut s = opened();
        let mut map = stack("filled_map", 1);
        map.damage = -3;
        s.grid[0] = Some(map);
        let mut data = MapData::new(-3);
        data.colors[0] = 8 * 4 + 2;
        data.decorations.push(mc_world::MapDecoration {
            key: "icon-0".to_owned(),
            icon_type: 1,
            x: 0,
            z: 0,
            direction: 0,
        });

        let mut draw = DrawList::new();
        s.draw_with_map_records(
            &mut draw,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            None,
            vec![data],
        );
        assert_eq!(s.map_ids(), vec![-3]);
        assert!(!draw
            .batches()
            .iter()
            .any(|batch| batch.texture == crate::map::dynamic_texture_key(-3)));
        assert!(!draw
            .batches()
            .iter()
            .any(|batch| batch.texture == UiTexture::gui(crate::map::MAP_ICONS_TEXTURE)));
    }

    /// java: `GuiContainerCreative.updateScreen` never calls
    /// `searchField.updateCursorCounter()`, so the focused search box's caret never blinks.
    #[test]
    fn w21_ci_the_search_caret_never_blinks() {
        let mut s = opened();
        s.set_tab(CreativeTab::Search.index().unwrap());
        for tick in 0..40 {
            assert!(s.search.caret_visible(), "tick {tick}");
            s.tick();
        }
    }

    #[test]
    fn the_wheel_scrolls_one_row_at_a_time_and_clamps() {
        let mut s = opened();
        let steps = (s.item_list().len() / 9) as f32 - 5.0;
        s.scroll_by_wheel(-1.0);
        assert!((s.scroll() - 1.0 / steps).abs() < 1e-5, "{}", s.scroll());
        s.scroll_by_wheel(1.0);
        assert!(
            s.scroll().abs() < 1e-7,
            "Java's f32 roundoff: {}",
            s.scroll()
        );
        for _ in 0..500 {
            s.scroll_by_wheel(-1.0);
        }
        assert_eq!(s.scroll(), 1.0, "clamped, never past the end");
    }

    /// java: `j = size / 9 - 5` is **zero** for 46..=53 items, and the division that follows is
    /// done in `double`, so vanilla produces an infinity that the clamp swallows rather than
    /// dividing by an integer zero. Reproduced rather than "fixed".
    #[test]
    fn a_tab_that_overflows_by_less_than_a_full_row_divides_by_zero_and_clamps() {
        let mut s = opened();
        s.item_list = (0..50)
            .filter_map(|i| {
                mc_data::items()
                    .get(i)
                    .map(|p| creative::representative_stack(p.id))
            })
            .collect();
        assert!(s.needs_scroll_bars());
        assert_eq!((s.item_list.len() / 9) as f32 - 5.0, 0.0);
        s.scroll_by_wheel(-1.0);
        assert_eq!(s.scroll(), 1.0, "-1/0 is -inf, so the clamp lands at 1.0");
        s.scroll_by_wheel(1.0);
        assert_eq!(s.scroll(), 0.0);
    }

    // ------------------------------------------------------------ search

    /// Proves the fix for the one-frame typing lag: every character lands in [`TextField::text`]
    /// and filters the grid inside the very call that typed it, with no intervening
    /// [`CreativeScreen::apply_pending_input`] to drain a queue. Contrast the old shape of this
    /// test (still in history), which had to call `apply_pending_input` after the typing loop
    /// before either assertion would see the query.
    #[test]
    fn typing_in_the_search_box_filters_the_grid_on_the_same_frame() {
        let mut s = opened();
        let font = uniform_font();
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.apply_pending_input(&font);
        assert_eq!(
            s.item_list().len(),
            creative::search_items("").len(),
            "java: an empty needle matches every tooltip line, so the tab opens full"
        );
        assert!(s.search.visible && s.search.is_focused());

        for c in "sword".chars() {
            s.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
            // Every single keystroke is already reflected — not just the last one — which is
            // what "no queue" means as opposed to "a shorter queue".
            assert!(s.search.text().ends_with(c));
        }
        assert_eq!(s.search.text(), "sword");
        assert!(!s.item_list().is_empty());
        let diamond = mc_data::item_by_name("diamond_sword").unwrap();
        assert!(s
            .item_list()
            .iter()
            .any(|stack| stack.item_id == diamond.0 as i16));
        assert_eq!(s.scroll(), 0.0, "a new search resets the scroll");
        assert_eq!(
            s.grid()[0].as_ref().map(|st| st.item_id),
            Some(s.item_list()[0].item_id)
        );
    }

    #[test]
    fn the_search_box_is_hidden_and_unfocused_on_every_other_tab() {
        let mut s = opened();
        assert!(!s.search.visible);
        s.set_tab(CreativeTab::Search.index().unwrap());
        assert!(s.search.visible);
        assert!(!s.search.can_lose_focus, "vanilla pins it focused");
        s.set_tab(0);
        assert!(!s.search.visible);
        assert!(!s.search.is_focused());
    }

    #[test]
    fn keys_are_only_applied_on_the_search_tab() {
        let mut s = opened();
        let font = uniform_font();
        for c in "abc".chars() {
            s.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
        }
        assert_eq!(s.search.text(), "", "no typing outside the search tab");
    }

    #[test]
    fn switching_to_the_search_tab_clears_whatever_was_typed_before() {
        let mut s = opened();
        let font = uniform_font();
        let search = CreativeTab::Search.index().unwrap();
        s.set_tab(search);
        for c in "sword".chars() {
            s.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
        }
        assert_eq!(s.search.text(), "sword");
        s.set_tab(0);
        s.set_tab(search);
        s.apply_pending_input(&font);
        assert_eq!(s.search.text(), "");
        assert_eq!(
            s.item_list().len(),
            creative::search_items("").len(),
            "the box is cleared and the search re-run, which lists everything again"
        );
    }

    /// The font-dependent Search reset is queued until `draw`; switching tabs again before that
    /// frame must cancel it. Java's `setCurrentCreativeTab` updates the field and list
    /// synchronously, so a rapid Search -> content switch leaves the content list selected.
    #[test]
    fn switching_away_before_the_queued_search_reset_keeps_the_new_tab_list() {
        let mut s = opened();
        let font = uniform_font();
        let search = CreativeTab::Search.index().unwrap();
        s.set_tab(search);
        assert!(s.pending_search_reset);

        s.set_tab(0);
        let expected = creative::stacks_for_tab("buildingBlocks");
        s.apply_pending_input(&font);

        assert_eq!(s.tab(), CreativeTab::Items("buildingBlocks"));
        assert_eq!(s.item_list(), expected.as_slice());
        assert_eq!(s.grid()[0].as_ref(), expected.first());
    }

    /// The same deferred reset must be drained before a key, because Java clears the field while
    /// entering Search rather than waiting for a render frame. This is the input-side control for
    /// the rapid-switch case above.
    #[test]
    fn the_first_search_key_after_reentering_search_starts_from_an_empty_query() {
        let mut s = opened();
        let font = uniform_font();
        let search = CreativeTab::Search.index().unwrap();
        s.set_tab(search);
        s.apply_pending_input(&font);
        for c in "sword".chars() {
            s.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
        }
        s.set_tab(0);
        s.set_tab(search);

        s.key_typed_with_font(Some('d'), 0, Modifiers::NONE, &font);

        assert_eq!(s.search.text(), "d");
        assert_eq!(
            s.item_list(),
            creative::search_stacks("d", false, s.language.as_ref()).as_slice()
        );
    }

    #[test]
    fn the_first_search_keystroke_after_a_slot_click_replaces_the_query() {
        let mut s = opened();
        let font = uniform_font();
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.apply_pending_input(&font);
        for c in "sword".chars() {
            s.key_typed_with_font(Some(c), 0, Modifiers::NONE, &font);
        }
        assert_eq!(s.search.text(), "sword");

        s.click_slot(CreativeSlot::Grid(0), 0, false);
        s.key_typed_with_font(Some('d'), 0, Modifiers::NONE, &font);
        assert_eq!(
            s.search.text(),
            "d",
            "field_147057_D clears the old query before applying this keystroke"
        );
        assert_eq!(
            s.item_list(),
            creative::search_stacks("d", false, s.language.as_ref())
        );
    }

    #[test]
    fn printable_q_types_on_search_but_still_drops_outside_search() {
        let mut s = opened();
        let font = uniform_font();
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.apply_pending_input(&font);
        s.player.set_slot_contents(36, Some(stack("stone", 3)));
        s.hovered = Some(CreativeSlot::Player(36));

        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::NONE, &font);
        assert_eq!(s.search.text(), "q");
        assert_eq!(s.player.slot_stack(36).map(|stack| stack.count), Some(3));
        assert!(
            s.take_outgoing().is_empty(),
            "the search field consumes printable Q before super.keyTyped can drop"
        );

        s.set_tab(0);
        s.player.set_slot_contents(36, Some(stack("stone", 3)));
        s.hovered = Some(CreativeSlot::Player(36));
        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::NONE, &font);
        assert_eq!(s.player.slot_stack(36).map(|stack| stack.count), Some(2));
        assert_eq!(
            s.take_outgoing().len(),
            1,
            "negative control: Q must retain its ordinary drop behavior off Search"
        );
    }

    // ------------------------------------------------------------ clicking

    #[test]
    fn clicking_a_grid_cell_picks_one_of_that_item_up() {
        let mut s = opened();
        let (_, gx, gy) = s.slot_rects()[0];
        let (x, y) = (s.gui_left() + gx + 8, s.gui_top() + gy + 8);
        assert_eq!(s.slot_at(x, y), Some(CreativeSlot::Grid(0)));
        s.mouse_clicked(x, y, 0);
        let picked = s.player.cursor().expect("a stack was picked up");
        assert_eq!(picked.count, 1);
        assert_eq!(picked.item_id, s.item_list()[0].item_id);
        assert_eq!(picked.damage, s.item_list()[0].damage);
        // Clicking the same cell again increments, exactly as `click_tab_source` says.
        s.mouse_clicked(x, y, 0);
        s.mouse_released(x, y, 0);
        assert_eq!(s.player.cursor().map(|c| c.count), Some(2));
        s.mouse_clicked(x, y, 1);
        s.mouse_released(x, y, 1);
        assert_eq!(s.player.cursor().map(|c| c.count), Some(1));
    }

    /// java: `GuiContainer.mouseClicked` arms `dragSplitting` while a cursor stack is held;
    /// `mouseClickMove` accumulates valid hotbar slots; `mouseReleased` sends start/add/end mode
    /// 5 calls. Creative's final else sends one packet for each add and an unconditional nine
    /// packet hotbar burst for the end call.
    #[test]
    fn content_tab_drag_distributes_and_emits_the_adds_then_nine_slot_end_burst() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("stone", 6)));
        let point = |s: &CreativeScreen, id: u16| {
            let (_, x, y) = s
                .slot_rects()
                .into_iter()
                .find(|(slot, _, _)| *slot == CreativeSlot::Player(id))
                .expect("content tabs expose every hotbar slot");
            (s.gui_left() + x + 8, s.gui_top() + y + 8)
        };
        let first = point(&s, 36);
        let second = point(&s, 37);

        s.mouse_clicked(first.0, first.1, 0);
        assert!(s.take_outgoing().is_empty(), "a drag press sends no C10");
        s.mouse_dragged(first.0, first.1, 0, 0);
        s.mouse_dragged(second.0, second.1, 0, 0);
        s.mouse_released(second.0, second.1, 0);

        assert_eq!(
            s.player.cursor(),
            None,
            "all six items split across two slots"
        );
        assert_eq!(s.player.slot_stack(36).map(|stack| stack.count), Some(3));
        assert_eq!(s.player.slot_stack(37).map(|stack| stack.count), Some(3));
        let out = s.take_outgoing();
        assert_eq!(
            out.len(),
            11,
            "two add packets plus the nine-slot end burst"
        );
        assert_eq!(
            out[..2]
                .iter()
                .map(|packet| (packet.slot_id, packet.stack.is_none()))
                .collect::<Vec<_>>(),
            vec![(36, true), (37, true)],
            "add calls observe the pre-end slot contents"
        );
        assert_eq!(
            out[2..]
                .iter()
                .map(|packet| packet.slot_id)
                .collect::<Vec<_>>(),
            (36..45).collect::<Vec<_>>()
        );
        assert_eq!(out[2].stack.as_ref().map(|stack| stack.count), Some(3));
        assert_eq!(out[3].stack.as_ref().map(|stack| stack.count), Some(3));
        assert!(out[4..].iter().all(|packet| packet.stack.is_none()));
        assert!(!s.drag_splitting && s.drag_splitting_slots.is_empty());
    }

    /// java: `ContainerCreative.canDragIntoSlot` rejects the read-only tab grid (`y <= 90`),
    /// even though GuiContainer still arms dragSplitting on the initial press. With no collected
    /// slots, release falls back to the ordinary grid click and must not manufacture mode-5/C10
    /// traffic. The empty set is the negative control for the positive hotbar drag above.
    #[test]
    fn content_tab_grid_drag_has_no_mode_five_packets_and_falls_back_to_grid_click() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("dirt", 5)));
        let (_, gx, gy) = s.slot_rects()[0];
        let point = (s.gui_left() + gx + 8, s.gui_top() + gy + 8);

        s.mouse_clicked(point.0, point.1, 0);
        assert!(s.drag_splitting);
        s.mouse_dragged(point.0, point.1, 0, 0);
        assert!(s.drag_splitting_slots.is_empty());
        s.mouse_released(point.0, point.1, 0);

        assert!(!s.drag_splitting);
        assert!(s.take_outgoing().is_empty(), "a grid click is cursor-local");
        assert_eq!(
            s.player.cursor(),
            None,
            "different item falls through to cursor discard"
        );
    }

    /// java: the Inventory-tab branch calls `inventoryContainer.slotClick` and
    /// `detectAndSendChanges` for each mode-5 phase instead of Creative's direct hotbar tail.
    /// Only the two changed player slots therefore upload, and the persistent mirror prevents
    /// an unrelated slot from appearing in the drag result.
    #[test]
    fn inventory_tab_drag_uses_the_player_container_diff_for_changed_slots_only() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_cursor(Some(stack("stone", 6)));
        let point = |s: &CreativeScreen, id: u16| {
            let (_, x, y) = s
                .slot_rects()
                .into_iter()
                .find(|(slot, _, _)| *slot == CreativeSlot::Player(id))
                .expect("the Inventory tab exposes the main inventory");
            (s.gui_left() + x + 8, s.gui_top() + y + 8)
        };
        let first = point(&s, 9);
        let second = point(&s, 10);

        s.mouse_clicked(first.0, first.1, 0);
        s.mouse_dragged(first.0, first.1, 0, 0);
        s.mouse_dragged(second.0, second.1, 0, 0);
        s.mouse_released(second.0, second.1, 0);

        assert_eq!(s.player.cursor(), None);
        assert_eq!(s.player.slot_stack(9).map(|stack| stack.count), Some(3));
        assert_eq!(s.player.slot_stack(10).map(|stack| stack.count), Some(3));
        let out = s.take_outgoing();
        assert_eq!(
            out.len(),
            2,
            "detectAndSendChanges uploads just the changed slots"
        );
        assert_eq!(
            out.iter().map(|packet| packet.slot_id).collect::<Vec<_>>(),
            vec![9, 10]
        );
        assert!(out
            .iter()
            .all(|packet| { packet.stack.as_ref().is_some_and(|stack| stack.count == 3) }));
    }

    /// java: the raw `clickedButton == 2` content-tab path also satisfies
    /// `Container.getDragEvent(2) == 2`, so a plain middle click on a hotbar slot emits the same
    /// nine current hotbar stacks after the creative pick operation. The `stone` slot is a
    /// positive control; the eight empty slots prove the burst is not a changed-slot diff.
    #[test]
    fn middle_clicking_a_content_tab_hotbar_slot_emits_the_vanilla_nine_slot_burst() {
        let mut s = opened();
        s.player.set_slot_contents(36, Some(stack("stone", 1)));
        s.take_outgoing();
        let (_, x, y) = s
            .slot_rects()
            .into_iter()
            .find(|(slot, _, _)| *slot == CreativeSlot::Player(36))
            .unwrap();
        s.mouse_clicked(
            s.gui_left() + x + 8,
            s.gui_top() + y + 8,
            DEFAULT_PICK_BLOCK_BUTTON,
        );

        let cursor = s.player.cursor().expect("creative pick gives a full stack");
        assert_eq!(cursor.count, stack_ext::max_stack_size(cursor));
        let out = s.take_outgoing();
        assert_eq!(out.len(), 9);
        assert_eq!(
            out.iter().map(|packet| packet.slot_id).collect::<Vec<_>>(),
            (36..45).collect::<Vec<_>>()
        );
        assert!(out[0].stack.is_some());
        assert!(out[1..].iter().all(|packet| packet.stack.is_none()));
    }

    #[test]
    fn middle_click_clones_a_full_grid_stack_only_when_the_cursor_is_empty() {
        let mut s = opened();
        let source = s.grid()[0].clone().expect("the first grid cell is filled");

        s.click_slot(CreativeSlot::Grid(0), DEFAULT_PICK_BLOCK_BUTTON, false);
        let cloned = s.player.cursor().expect("middle click cloned the source");
        assert_eq!(cloned.item_id, source.item_id);
        assert_eq!(cloned.count, stack_ext::max_stack_size(&source));

        let held = ItemStack { count: 3, ..source };
        s.player.set_cursor(Some(held.clone()));
        s.click_slot(CreativeSlot::Grid(0), DEFAULT_PICK_BLOCK_BUTTON, false);
        assert_eq!(
            s.player.cursor(),
            Some(&held),
            "clickType 3 returns without changing a non-empty cursor"
        );
        assert!(s.take_outgoing().is_empty(), "grid cloning is client-only");
    }

    /// java: `GuiContainerCreative.handleMouseClick` returns immediately for click type `3`,
    /// before inspecting the source stack. A middle-button drag that ends over an empty virtual
    /// grid cell must therefore leave an already-held cursor untouched (the ordinary left/right
    /// empty-cell path intentionally clears it).
    #[test]
    fn middle_release_over_an_empty_grid_cell_keeps_the_held_cursor() {
        let mut s = opened();
        s.grid[0] = None;
        let held = stack("stone", 3);
        s.player.set_cursor(Some(held.clone()));
        let (_, x, y) = s.slot_rects()[0];

        s.mouse_clicked(
            s.gui_left() + x + 8,
            s.gui_top() + y + 8,
            DEFAULT_PICK_BLOCK_BUTTON,
        );
        s.mouse_released(
            s.gui_left() + x + 8,
            s.gui_top() + y + 8,
            DEFAULT_PICK_BLOCK_BUTTON,
        );

        assert_eq!(s.player.cursor(), Some(&held));
        assert!(
            s.take_outgoing().is_empty(),
            "mode-3 virtual click is local-only"
        );
    }

    /// The window point at the middle of `slot`.
    fn centre_of(s: &CreativeScreen, slot: CreativeSlot) -> (i32, i32) {
        let (_, x, y) = s
            .slot_rects()
            .into_iter()
            .find(|(candidate, _, _)| *candidate == slot)
            .expect("the slot is on this tab");
        (s.gui_left() + x + 8, s.gui_top() + y + 8)
    }

    fn click_at(s: &mut CreativeScreen, slot: CreativeSlot, button: i32) {
        let (x, y) = centre_of(s, slot);
        s.mouse_clicked(x, y, button);
        s.mouse_released(x, y, button);
    }

    /// java: `GuiContainerCreative` keeps `GuiContainer`'s mouse state
    /// machine. A second click on the same hotbar slot within 250 ms is a double click there, and
    /// `ContainerCreative.canMergeSlot` (y > 90) admits the hotbar, so the second release is
    /// `handleMouseClick(slot, k, 0, 6)`: a collect over the hotbar into the cursor, after which
    /// the content tab's tail uploads the clicked slot alone.
    #[test]
    fn a_double_click_on_a_content_tab_hotbar_slot_collects_from_the_hotbar() {
        let mut s = opened();
        s.player.set_slot_contents(36, Some(stack("stone", 10)));
        s.player.set_slot_contents(37, Some(stack("stone", 20)));
        s.player.set_slot_contents(38, Some(stack("dirt", 5)));
        s.player.set_slot_contents(9, Some(stack("stone", 30)));
        s.take_outgoing();
        s.set_clock_ms(60_000);
        click_at(&mut s, CreativeSlot::Player(36), 0);
        assert_eq!(s.player.cursor(), Some(&stack("stone", 10)));
        s.set_clock_ms(60_120);
        click_at(&mut s, CreativeSlot::Player(36), 0);

        assert_eq!(
            s.player.cursor(),
            Some(&stack("stone", 30)),
            "10 + the 20 of slot 37; slot 9 is not in the creative container"
        );
        assert_eq!(s.player.slot_stack(36), None);
        assert_eq!(s.player.slot_stack(37), None, "drained, and not uploaded");
        assert_eq!(s.player.slot_stack(38), Some(&stack("dirt", 5)));
        assert_eq!(s.player.slot_stack(9), Some(&stack("stone", 30)));
        let out = s.take_outgoing();
        assert_eq!(
            out.iter()
                .map(|packet| (packet.slot_id, packet.stack.clone()))
                .collect::<Vec<_>>(),
            vec![(36, None), (36, None)],
            "the pick-up's upload, then the collect's upload of the clicked slot"
        );

        // The control: 250 ms later the second click is an ordinary one and puts it back.
        let mut s = opened();
        s.player.set_slot_contents(36, Some(stack("stone", 10)));
        s.player.set_slot_contents(37, Some(stack("stone", 20)));
        s.take_outgoing();
        s.set_clock_ms(60_000);
        click_at(&mut s, CreativeSlot::Player(36), 0);
        s.set_clock_ms(60_250);
        click_at(&mut s, CreativeSlot::Player(36), 0);
        assert_eq!(s.player.cursor(), None);
        assert_eq!(s.player.slot_stack(36), Some(&stack("stone", 10)));
        assert_eq!(s.player.slot_stack(37), Some(&stack("stone", 20)));
    }

    /// On the Inventory tab the hotbar's wrapped slots also merge (y = 112), and the collect is
    /// the player container's own (`inventoryContainer.slotClick(k, 0, 6)`), uploaded by its
    /// `detectAndSendChanges`.
    #[test]
    fn a_double_click_on_an_inventory_tab_hotbar_slot_runs_the_player_containers_collect() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(36, Some(stack("stone", 10)));
        s.player.set_slot_contents(9, Some(stack("stone", 30)));
        s.player.set_slot_contents(40, Some(stack("stone", 5)));
        s.emit_changes();
        s.take_outgoing();
        s.set_clock_ms(60_000);
        click_at(&mut s, CreativeSlot::Player(36), 0);
        s.set_clock_ms(60_100);
        click_at(&mut s, CreativeSlot::Player(36), 0);

        assert_eq!(s.player.cursor(), Some(&stack("stone", 45)));
        let out = s.take_outgoing();
        assert_eq!(
            out.iter()
                .map(|packet| (packet.slot_id, packet.stack.clone()))
                .collect::<Vec<_>>(),
            vec![(36, None), (9, None), (40, None)]
        );
    }

    /// java: `GuiContainer.mouseClicked`'s `else if (!this.dragSplitting)`: a press while a drag
    /// is armed changes nothing, and that button's release cancels the drag (`:558-564`) and sets
    /// `ignoreMouseUp`, which swallows the first button's release too.
    #[test]
    fn a_second_button_during_an_armed_drag_cancels_it_without_a_click() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("stone", 10)));
        s.take_outgoing();
        s.set_clock_ms(60_000);
        let (lx, ly) = centre_of(&s, CreativeSlot::Player(36));
        let (rx, ry) = centre_of(&s, CreativeSlot::Player(37));
        s.mouse_clicked(lx, ly, 0);
        s.set_clock_ms(60_400);
        s.mouse_clicked(rx, ry, 1);
        s.mouse_released(rx, ry, 1);
        s.mouse_released(lx, ly, 0);

        assert_eq!(s.player.cursor(), Some(&stack("stone", 10)));
        assert_eq!(s.player.slot_stack(36), None);
        assert_eq!(s.player.slot_stack(37), None);
        assert!(s.take_outgoing().is_empty());
    }

    /// java: `if (mouseButton == 0 || mouseButton == 1 || flag)`: another mouse button's press
    /// never clicks. With an empty cursor its release does nothing either.
    #[test]
    fn a_side_button_press_does_not_take_from_the_grid() {
        let mut s = opened();
        let (x, y) = centre_of(&s, CreativeSlot::Grid(0));
        s.mouse_clicked(x, y, 3);
        s.mouse_released(x, y, 3);
        assert_eq!(s.player.cursor(), None);
        assert!(s.take_outgoing().is_empty());
    }

    /// java: a pick-button release with a held stack is `handleMouseClick(null, -999, 2, 3)`
    /// outside the panel; on the Inventory tab only mode 4 drops, so nothing is dropped.
    #[test]
    fn a_pick_button_release_outside_the_inventory_tab_drops_nothing() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_cursor(Some(stack("stone", 10)));
        s.take_outgoing();
        let (x, y) = (s.gui_left() - 20, s.gui_top() + 40);
        s.mouse_clicked(x, y, DEFAULT_PICK_BLOCK_BUTTON);
        s.mouse_released(x, y, DEFAULT_PICK_BLOCK_BUTTON);
        assert_eq!(s.player.cursor(), Some(&stack("stone", 10)));
        assert!(s.take_outgoing().is_empty());
        // The control: a left click there drops the whole stack (mode 0 made 4).
        s.mouse_clicked(x, y, 0);
        s.mouse_released(x, y, 0);
        assert_eq!(s.player.cursor(), None);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].slot_id, -1);
        assert_eq!(out[0].stack, Some(stack("stone", 10)));
    }

    /// java: `mouseReleased` reads `isShiftKeyDown()` at the release for a held stack's click: a
    /// stack pressed onto a content-tab hotbar slot without shift and released with it is a
    /// shift-click (`ContainerCreative.transferStackInSlot` clears the slot).
    #[test]
    fn a_held_stack_release_reads_shift_at_the_release() {
        let mut s = opened();
        s.player.set_slot_contents(36, Some(stack("dirt", 3)));
        s.player.set_cursor(Some(stack("stone", 10)));
        s.take_outgoing();
        let (x, y) = centre_of(&s, CreativeSlot::Player(36));
        s.mouse_clicked_with_mods(x, y, 0, Modifiers::NONE);
        s.mouse_released_with_mods(
            x,
            y,
            0,
            Modifiers {
                shift: true,
                ..Modifiers::NONE
            },
        );
        assert_eq!(s.player.slot_stack(36), None);
        assert_eq!(s.player.cursor(), Some(&stack("stone", 10)));
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].slot_id, out[0].stack.clone()), (36, None));
    }

    /// java: a hotbar key over a content-tab hotbar slot is `handleMouseClick(slot, n, i, 2)`, and
    /// the tail tests `Container.getDragEvent(i) == 2`, which is `i & 3 == 2`: hotbar keys 3 and
    /// 7 upload the whole hotbar, the others the clicked slot.
    #[test]
    fn hotbar_keys_3_and_7_over_a_content_tab_hotbar_slot_upload_the_whole_hotbar() {
        let font = uniform_font();
        for (index, burst) in [(2u8, true), (6, true), (1, false), (3, false)] {
            let mut s = opened();
            s.player.set_slot_contents(36, Some(stack("stone", 5)));
            s.player
                .set_slot_contents(36 + usize::from(index), Some(stack("dirt", 3)));
            s.take_outgoing();
            s.hovered = Some(CreativeSlot::Player(36));
            s.key_typed_with_font(
                None,
                KEY_HOTBAR_1 + i32::from(index),
                Modifiers::NONE,
                &font,
            );
            assert_eq!(s.player.slot_stack(36), Some(&stack("dirt", 3)), "swapped");
            let slots: Vec<i16> = s.take_outgoing().iter().map(|p| p.slot_id).collect();
            if burst {
                assert_eq!(slots, (36..45).collect::<Vec<_>>(), "index {index}");
            } else {
                assert_eq!(slots, vec![36], "index {index}");
            }
        }
    }

    /// java: `GuiContainer.keyTyped` drops only `if (theSlot != null && theSlot.getHasStack())`.
    #[test]
    fn the_drop_key_over_an_empty_content_tab_hotbar_slot_sends_nothing() {
        let mut s = opened();
        let font = uniform_font();
        s.take_outgoing();
        s.hovered = Some(CreativeSlot::Player(37));
        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::NONE, &font);
        assert!(s.take_outgoing().is_empty());
    }

    /// java: on the Inventory tab the drop key is `slotIn.decrStackSize(...)` on the hovered slot,
    /// not `Container.slotClick` mode 4, so it takes from the slot even with a stack on the cursor.
    #[test]
    fn the_inventory_tab_drop_key_takes_from_the_slot_even_with_a_held_stack() {
        let mut s = opened();
        let font = uniform_font();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(9, Some(stack("stone", 10)));
        s.player.set_cursor(Some(stack("dirt", 2)));
        s.emit_changes();
        s.take_outgoing();
        s.hovered = Some(CreativeSlot::Player(9));
        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::NONE, &font);
        assert_eq!(s.player.slot_stack(9), Some(&stack("stone", 9)));
        assert_eq!(s.player.cursor(), Some(&stack("dirt", 2)));
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].slot_id, out[0].stack.clone()),
            (-1, Some(stack("stone", 1)))
        );
    }

    #[test]
    fn a_grid_click_sends_no_packet_because_it_is_client_only() {
        let mut s = opened();
        let (_, gx, gy) = s.slot_rects()[0];
        s.mouse_clicked(s.gui_left() + gx + 8, s.gui_top() + gy + 8, 0);
        assert!(
            s.take_outgoing().is_empty(),
            "the pick itself is client-side; the CreativeInventoryAction follows the *drop*"
        );
    }

    #[test]
    fn dropping_a_picked_stack_into_the_hotbar_sends_a_creative_inventory_action() {
        let mut s = opened();
        let grid = s.slot_rects()[0];
        s.mouse_clicked(s.gui_left() + grid.1 + 8, s.gui_top() + grid.2 + 8, 0);
        s.take_outgoing();
        let hotbar = s
            .slot_rects()
            .into_iter()
            .find(|(slot, _, _)| *slot == CreativeSlot::Player(36))
            .expect("the hotbar row is on every grid tab");
        s.mouse_clicked(s.gui_left() + hotbar.1 + 8, s.gui_top() + hotbar.2 + 8, 0);
        s.mouse_released(s.gui_left() + hotbar.1 + 8, s.gui_top() + hotbar.2 + 8, 0);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1, "{out:?}");
        assert_eq!(out[0].slot_id, 36);
        assert!(out[0].stack.is_some());
        assert_eq!(s.player.cursor(), None);
    }

    #[test]
    fn the_trash_cell_only_exists_on_the_inventory_tab_and_discards_the_cursor() {
        let mut s = opened();
        assert!(!s
            .slot_rects()
            .iter()
            .any(|(slot, _, _)| *slot == CreativeSlot::Trash));
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        let trash = s
            .slot_rects()
            .into_iter()
            .find(|(slot, _, _)| *slot == CreativeSlot::Trash)
            .expect("the inventory tab has a destroy-item cell");
        assert_eq!((trash.1, trash.2), (173, 112));
        s.player.set_cursor(Some(stack("stone", 64)));
        s.mouse_clicked(s.gui_left() + trash.1 + 8, s.gui_top() + trash.2 + 8, 0);
        s.mouse_released(s.gui_left() + trash.1 + 8, s.gui_top() + trash.2 + 8, 0);
        assert_eq!(s.player.cursor(), None);
    }

    #[test]
    fn the_inventory_tab_remaps_the_player_container_onto_vanillas_own_positions() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        let rects = s.slot_rects();
        let at = |id: u16| {
            rects
                .iter()
                .find(|(slot, _, _)| *slot == CreativeSlot::Player(id))
                .map(|(_, x, y)| (*x, *y))
        };
        // java: armour (5..8) in two columns of two.
        assert_eq!(at(5), Some((9, 6)));
        assert_eq!(at(6), Some((9, 33)));
        assert_eq!(at(7), Some((63, 6)));
        assert_eq!(at(8), Some((63, 33)));
        // Main inventory 9..35 in three rows, hotbar 36..44 pinned at y = 112.
        assert_eq!(at(9), Some((9, 54)));
        assert_eq!(at(18), Some((9, 72)));
        assert_eq!(at(36), Some((9, 112)));
        assert_eq!(at(44), Some((9 + 8 * 18, 112)));
        // The craft grid and result (0..4) are hidden on this tab.
        for id in 0..5 {
            assert_eq!(at(id), None, "slot {id} must not be shown");
        }
        assert!(!s
            .slot_rects()
            .iter()
            .any(|(s, _, _)| matches!(s, CreativeSlot::Grid(_))));
    }

    #[test]
    fn a_click_on_a_real_player_slot_goes_through_the_container_state_machine() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(20, Some(stack("stone", 30)));
        let rect = s
            .slot_rects()
            .into_iter()
            .find(|(slot, _, _)| *slot == CreativeSlot::Player(20))
            .unwrap();
        s.mouse_clicked(s.gui_left() + rect.1 + 8, s.gui_top() + rect.2 + 8, 1);
        // Right-click with an empty cursor takes half.
        assert_eq!(s.player.cursor().map(|c| c.count), Some(15));
        assert_eq!(s.player.slot_stack(20).map(|c| c.count), Some(15));
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].slot_id, 20);
        assert_eq!(out[0].stack.as_ref().map(|st| st.count), Some(15));
    }

    #[test]
    fn content_tab_click_uploads_the_clicked_hotbar_slot_even_when_it_stays_empty() {
        let mut s = opened();
        // `GuiContainerCreative` does not use the player-container listener for this path. Its
        // final else sends the clicked creative hotbar slot directly, including a null stack.
        s.click_slot(CreativeSlot::Player(36), 0, false);
        let out = s.take_outgoing();
        assert_eq!(
            out.len(),
            1,
            "Java sends one direct C10 for every content-tab click"
        );
        assert_eq!(out[0].slot_id, 36);
        assert!(out[0].stack.is_none());
    }

    #[test]
    fn opening_registers_the_persistent_creative_crafting_mirror() {
        let mut player = layouts::player_container(0);
        player.set_slot_contents(20, Some(stack("stone", 7)));
        let mut s = CreativeScreen::over(player);
        let initial = s.take_outgoing();
        assert_eq!(initial.len(), 1);
        assert_eq!(
            (initial[0].slot_id, initial[0].stack.as_ref().unwrap().count),
            (20, 7)
        );

        // A server write does not mutate Container.inventoryItemStacks. The next inventory-tab
        // detectAndSendChanges therefore echoes it, which a fresh per-click snapshot cannot do.
        s.player.set_slot_contents(21, Some(stack("dirt", 4)));
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.click_slot(CreativeSlot::Player(22), 0, false);
        let echoed = s.take_outgoing();
        assert!(echoed.iter().any(|packet| {
            packet.slot_id == 21 && packet.stack.as_ref().is_some_and(|stack| stack.count == 4)
        }));
    }

    #[test]
    fn shift_clicking_trash_sends_one_clear_for_every_player_container_slot() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.click_slot(CreativeSlot::Trash, 0, true);
        let out = s.take_outgoing();
        assert_eq!(out.len(), 45);
        assert!(out
            .iter()
            .enumerate()
            .all(|(index, packet)| packet.slot_id == index as i16 && packet.stack.is_none()));
    }

    #[test]
    fn shift_clicking_a_content_tab_hotbar_slot_deletes_and_uploads_only_that_slot() {
        let mut s = opened();
        s.player.set_slot_contents(36, Some(stack("stone", 9)));
        s.take_outgoing();
        s.click_slot(CreativeSlot::Player(36), 0, true);
        assert!(s.player.slot_stack(36).is_none());
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].slot_id, 36);
        assert!(out[0].stack.is_none());
    }

    #[test]
    fn number_and_drop_keys_operate_on_grid_cells() {
        let mut s = opened();
        let font = uniform_font();
        s.hovered = Some(CreativeSlot::Grid(0));
        let source = s.grid()[0].clone().unwrap();

        s.key_typed_with_font(Some('3'), KEY_HOTBAR_1 + 2, Modifiers::NONE, &font);
        let hotbar = s.player.slot_stack(38).unwrap();
        assert_eq!(
            (hotbar.item_id, hotbar.damage),
            (source.item_id, source.damage)
        );
        assert_eq!(hotbar.count, stack_ext::max_stack_size(&source));
        assert!(s.take_outgoing().iter().any(|packet| packet.slot_id == 38));

        s.hovered = Some(CreativeSlot::Grid(0));
        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::CTRL, &font);
        let dropped = s.take_outgoing();
        assert_eq!(dropped.len(), 1);
        assert_eq!(dropped[0].slot_id, -1);
        assert_eq!(
            dropped[0].stack.as_ref().map(|stack| stack.count),
            Some(stack_ext::max_stack_size(&source))
        );
    }

    #[test]
    fn clicking_outside_drops_all_or_one_from_the_cursor() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("stone", 3)));
        s.mouse_clicked(s.gui_left() - 10, s.gui_top() + 20, 1);
        s.mouse_released(s.gui_left() - 10, s.gui_top() + 20, 1);
        assert_eq!(s.player.cursor().map(|stack| stack.count), Some(2));
        let one = s.take_outgoing();
        assert_eq!(
            (one[0].slot_id, one[0].stack.as_ref().unwrap().count),
            (-1, 1)
        );

        s.mouse_clicked(s.gui_left() - 10, s.gui_top() + 20, 0);
        s.mouse_released(s.gui_left() - 10, s.gui_top() + 20, 0);
        assert!(s.player.cursor().is_none());
        let rest = s.take_outgoing();
        assert_eq!(
            (rest[0].slot_id, rest[0].stack.as_ref().unwrap().count),
            (-1, 2)
        );
    }

    #[test]
    fn drop_key_on_the_inventory_tab_uses_the_world_drop_sentinel() {
        let mut s = opened();
        let font = uniform_font();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(20, Some(stack("stone", 3)));
        s.hovered = Some(CreativeSlot::Player(20));
        s.key_typed_with_font(Some('q'), KEY_DROP, Modifiers::NONE, &font);
        assert_eq!(s.player.slot_stack(20).map(|stack| stack.count), Some(2));
        let out = s.take_outgoing();
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].slot_id, out[0].stack.as_ref().unwrap().count),
            (-1, 1)
        );
    }

    #[test]
    fn tab_and_destroy_tooltips_use_the_active_language() {
        let mut language = Language::new("test");
        language.insert("itemGroup.buildingBlocks", "Translated Blocks");
        language.insert("inventory.binSlot", "Translated Destroy");
        let mut s = opened();
        s.set_language(&language);
        assert_eq!(
            s.translate(&CreativeTab::Items("buildingBlocks").translation_key()),
            "Translated Blocks"
        );
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.hovered = Some(CreativeSlot::Grid(0));
        assert!(s
            .hovered_tooltip()
            .unwrap()
            .iter()
            .any(|line| line.contains("Translated Blocks")));
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.hovered = Some(CreativeSlot::Trash);
        assert_eq!(
            s.hovered_tooltip(),
            Some(vec!["Translated Destroy".to_owned()])
        );
    }

    // ------------------------------------------------------------ drawing

    /// java: `GuiContainerCreative.updateScreen` calls `updateActivePotionEffects()` every tick,
    /// and its override re-anchors `searchField.xPosition` whenever the inherited method moves
    /// `guiLeft`. Mutating the private snapshot directly is intentional: it proves `tick`, not
    /// merely the public setter's eager refresh, owns the live tracking path.
    #[test]
    fn active_effects_shift_creative_gui_left_each_tick_and_reanchor_search() {
        let mut s = opened();
        let centered = (ctx().width - X_SIZE) / 2;
        assert_eq!(s.gui_left(), centered);
        assert_eq!(s.search.rect.x, centered + 82);

        s.active_effects = effects(1);
        s.tick();
        let shifted = 160 + (ctx().width - X_SIZE - 200) / 2;
        assert_eq!(s.gui_left(), shifted);
        assert_eq!(s.search.rect.x, shifted + 82);

        s.active_effects.clear();
        s.tick();
        assert_eq!(s.gui_left(), centered);
        assert_eq!(s.search.rect.x, centered + 82);
    }

    /// `centered_inventory`: the creative inventory stays centred (search field
    /// included) with effects active, the panels can be hidden, and `(false, true)` restores
    /// vanilla's shift.
    #[test]
    fn the_centered_inventory_layout_keeps_creative_centred_and_reanchors_search() {
        let mut s = opened();
        let centered = (ctx().width - X_SIZE) / 2;
        let shifted = 160 + (ctx().width - X_SIZE - 200) / 2;
        s.set_active_effects(effects(1));
        assert_eq!(s.gui_left(), shifted);
        s.set_effect_layout(true, true);
        assert_eq!(s.gui_left(), centered);
        assert_eq!(s.search.rect.x, centered + 82);
        s.tick();
        assert_eq!(s.gui_left(), centered, "every tick keeps it");
        let mut draw = DrawList::new();
        s.draw_active_potion_effects(&mut draw, &mut uniform_font());
        assert!(
            !draw.is_empty(),
            "panels still drawn, left of the centred inventory"
        );
        s.set_effect_layout(true, false);
        let mut hidden = DrawList::new();
        s.draw_active_potion_effects(&mut hidden, &mut uniform_font());
        assert!(hidden.is_empty());
        s.set_effect_layout(false, true);
        assert_eq!(s.gui_left(), shifted);
        assert_eq!(s.search.rect.x, shifted + 82);
    }

    /// java: `InventoryEffectRenderer.drawActivePotionEffects` starts row zero at `guiTop`,
    /// uses 33px through five effects, then integer pitch `132 / (count - 1)`. These assertions
    /// inspect the emitted inventory-texture plates, so a draw-path disconnect cannot pass by
    /// satisfying layout arithmetic alone.
    #[test]
    fn active_effect_plates_use_vanillas_shifted_origin_and_compressed_pitch() {
        fn plate_origins(count: usize) -> (CreativeScreen, Vec<[f32; 2]>) {
            let mut screen = opened();
            screen.set_active_effects(effects(count));
            let mut draw = DrawList::new();
            screen.draw_active_potion_effects(&mut draw, &mut uniform_font());
            let texture = UiTexture::gui(INVENTORY_TEXTURE);
            let origins = draw
                .batches()
                .iter()
                .filter(|batch| batch.texture == texture)
                .map(|batch| draw.vertices()[batch.range.start].pos)
                .collect();
            (screen, origins)
        }

        let (five, five_rows) = plate_origins(5);
        assert_eq!(five_rows.len(), 5);
        assert_eq!(
            five_rows[0],
            [(five.gui_left() - 124) as f32, five.gui_top() as f32]
        );
        assert_eq!(five_rows[4][1], (five.gui_top() + 4 * 33) as f32);

        let (six, six_rows) = plate_origins(6);
        assert_eq!(six_rows.len(), 6);
        assert_eq!(
            six_rows[0],
            [(six.gui_left() - 124) as f32, six.gui_top() as f32]
        );
        assert_eq!(six_rows[1][1], (six.gui_top() + 132 / 5) as f32);
        assert_eq!(six_rows[5][1], (six.gui_top() + 5 * (132 / 5)) as f32);
    }

    #[test]
    fn drawing_emits_twelve_tab_plates_plus_the_panel() {
        let mut s = opened();
        let mut d = DrawList::new();
        let mut font = uniform_font();
        s.draw(&mut d, &mut font, &ctx(), (0, 0));
        let tabs = tabs_texture();
        let plate_quads: usize = d
            .batches()
            .iter()
            .filter(|b| b.texture == tabs)
            .map(|b| b.range.len())
            .sum();
        // 12 plates + the scrollbar handle, 6 vertices each; the item icons batch separately.
        assert!(
            plate_quads >= 13 * 6,
            "expected at least 13 quads from tabs.png, got {plate_quads} vertices"
        );
        assert!(d
            .batches()
            .iter()
            .any(|b| b.texture == panel_texture(CreativeTab::Items("buildingBlocks"))));
    }

    #[test]
    fn creative_inventory_forwards_and_clears_the_ready_skin_texture() {
        let custom = UiTexture::Resource(mc_assets::ResourceLocation::minecraft(
            "textures/entity/skin/player/profile-3.png",
        ));
        let mut screen = opened();
        screen.set_tab(CreativeTab::Inventory.index().unwrap());
        screen.set_player_preview_texture(Some(custom.clone()));

        let mut ready = DrawList::new();
        screen.draw(&mut ready, &mut uniform_font(), &ctx(), (0, 0));
        assert!(ready.batches().iter().any(|batch| batch.texture == custom));

        screen.set_player_preview_texture(None);
        let mut fallback = DrawList::new();
        screen.draw(&mut fallback, &mut uniform_font(), &ctx(), (0, 0));
        assert!(!fallback
            .batches()
            .iter()
            .any(|batch| batch.texture == custom));
        assert!(fallback
            .batches()
            .iter()
            .any(|batch| { batch.texture == UiTexture::gui("textures/entity/steve.png") }));
    }

    #[test]
    fn creative_inventory_forwards_live_preview_layers_and_pose() {
        let mut screen = opened();
        screen.set_tab(CreativeTab::Inventory.index().unwrap());
        let mut snapshot = PlayerPreviewSnapshot::default();
        snapshot.equipment[4] = Some(stack("diamond_helmet", 1));
        snapshot.params.limb_swing = 4.0;
        snapshot.params.limb_swing_amount = 1.0;
        snapshot.cape = Some(mc_render::entity::layers::CapeState::default());
        snapshot.cape_texture = Some(UiTexture::gui("textures/entity/cape/creative.png"));
        screen.set_player_preview_snapshot(snapshot);

        let mut draw = DrawList::new();
        screen.draw(&mut draw, &mut uniform_font(), &ctx(), (0, 0));
        assert!(draw.vertices().len() > 432);
        assert!(draw.batches().iter().any(|batch| {
            batch.texture == UiTexture::gui("textures/models/armor/diamond_layer_1.png")
        }));
        assert!(draw
            .batches()
            .iter()
            .any(|batch| { batch.texture == UiTexture::gui("textures/entity/cape/creative.png") }));
    }

    #[test]
    fn the_open_tabs_plate_is_drawn_last_so_it_paints_over_its_neighbours() {
        let mut s = opened();
        s.set_tab(3);
        let mut d = DrawList::new();
        let mut font = uniform_font();
        s.draw(&mut d, &mut font, &ctx(), (0, 0));
        let selected = tab_plate(3, true);
        // The scrollbar handle also samples tabs.png and is drawn immediately before the open
        // tab, so the two merge into one batch — the *last six vertices* of it are the plate.
        let tabs = tabs_texture();
        let last = d
            .batches()
            .iter()
            .rev()
            .find(|b| b.texture == tabs)
            .expect("a tabs.png batch");
        let top_left = d.vertices()[last.range.end - 6];
        assert_eq!(
            top_left.pos,
            [
                (s.gui_left() + selected.x) as f32,
                (s.gui_top() + selected.y) as f32
            ],
            "the open tab's plate has to be the last thing drawn from tabs.png"
        );
    }

    #[test]
    fn hovering_a_tab_shows_its_label_instead_of_an_item_tooltip() {
        let mut s = opened();
        let mut font = uniform_font();
        let plate = tab_plate(1, false);
        let point = (s.gui_left() + plate.x + 14, s.gui_top() + plate.y + 16);
        let mut with_tab = DrawList::new();
        s.draw(&mut with_tab, &mut font, &ctx(), point);
        let mut without = DrawList::new();
        s.draw(&mut without, &mut font, &ctx(), (0, 0));
        assert!(
            with_tab.vertices().len() > without.vertices().len(),
            "the hover label has to add geometry"
        );
    }

    #[test]
    fn the_scrollbar_handle_moves_with_the_scroll_and_greys_out_when_inert() {
        let mut s = opened();
        let (bar_left, bar_top, _, _) = s.scrollbar_gutter();
        let handle_y = |s: &mut CreativeScreen| {
            let mut d = DrawList::new();
            let mut font = uniform_font();
            s.draw(&mut d, &mut font, &ctx(), (0, 0));
            d.vertices()
                .iter()
                .find(|v| v.pos[0] == bar_left as f32 && v.pos[1] > (bar_top - 1) as f32)
                .map(|v| v.pos[1])
        };
        let top = handle_y(&mut s).expect("handle drawn");
        assert_eq!(top, bar_top as f32);
        s.scroll_to(1.0);
        s.scroll = 1.0;
        let bottom = handle_y(&mut s).expect("handle drawn");
        assert_eq!(bottom, (bar_top + 95) as f32);
    }

    #[test]
    fn escape_pops_the_screen_from_every_tab() {
        let mut s = opened();
        let font = uniform_font();
        assert_eq!(
            s.key_typed_with_font(None, keys::ESCAPE, Modifiers::NONE, &font),
            ScreenAction::Pop
        );
        s.set_tab(CreativeTab::Search.index().unwrap());
        assert_eq!(
            s.key_typed_with_font(None, keys::ESCAPE, Modifiers::NONE, &font),
            ScreenAction::Pop
        );
    }

    #[test]
    fn the_inventory_key_types_an_e_on_the_search_tab_instead_of_closing() {
        let mut s = opened();
        let font = uniform_font();
        assert_eq!(
            s.key_typed_with_font(None, KEY_INVENTORY, Modifiers::NONE, &font),
            ScreenAction::Pop
        );
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.apply_pending_input(&font);
        s.key_typed_with_font(Some('e'), KEY_INVENTORY, Modifiers::NONE, &font);
        assert_eq!(s.search.text(), "e", "no apply_pending_input in between");
    }

    #[test]
    fn creative_inventory_close_uses_the_rebound_inventory_key() {
        let mut s = opened();
        let font = uniform_font();
        let bindings = ContainerBindings {
            inventory_key: 30,
            ..Default::default()
        };
        s.set_bindings(bindings);

        assert_eq!(
            s.key_typed_with_font(None, KEY_INVENTORY, Modifiers::NONE, &font),
            ScreenAction::None,
            "the former E default must stop closing creative inventory"
        );
        assert_eq!(
            s.key_typed_with_font(None, bindings.inventory_key, Modifiers::NONE, &font),
            ScreenAction::Pop
        );
    }

    /// java: `GuiContainerCreative.keyTyped:308-313`. Off the search tab the chat key switches
    /// to search (focused, cleared) instead of closing or opening chat, and its `t` is not typed.
    /// On the search tab T and E are ordinary letters.
    #[test]
    fn the_chat_key_switches_to_the_search_tab_and_types_nothing() {
        let mut s = opened();
        let font = uniform_font();
        assert_ne!(s.tab(), CreativeTab::Search);
        assert_eq!(
            s.key_typed_with_font(Some('t'), KEY_CHAT, Modifiers::NONE, &font),
            ScreenAction::None
        );
        assert_eq!(s.tab(), CreativeTab::Search);
        assert!(s.search.is_focused());
        s.apply_pending_input(&font);
        assert_eq!(s.search.text(), "", "the switching key is not typed");

        for (typed, key) in [('t', KEY_CHAT), ('e', KEY_INVENTORY)] {
            assert_eq!(
                s.key_typed_with_font(Some(typed), key, Modifiers::NONE, &font),
                ScreenAction::None
            );
        }
        assert_eq!(s.search.text(), "te");
        assert_eq!(s.tab(), CreativeTab::Search);
    }

    #[test]
    fn a_rebound_chat_key_is_the_one_that_switches_to_search() {
        let mut s = opened();
        let font = uniform_font();
        s.set_bindings(ContainerBindings {
            chat_key: 21,
            ..Default::default()
        });
        s.key_typed_with_font(Some('t'), KEY_CHAT, Modifiers::NONE, &font);
        assert_ne!(
            s.tab(),
            CreativeTab::Search,
            "the former default no longer switches"
        );
        s.key_typed_with_font(Some('y'), 21, Modifiers::NONE, &font);
        assert_eq!(s.tab(), CreativeTab::Search);
    }

    /// NOT VANILLA 1.8.9 (the hotbar block in `mouse_clicked_inner`): a mouse button bound to
    /// `key.hotbar.3` does exactly what the keyboard key does over a hovered cell, byte for byte
    /// on the C10 wire, both filling from the grid and swapping on the Inventory tab, and its
    /// press and release do nothing else.
    #[test]
    fn a_mouse_bound_hotbar_key_sends_the_keyboard_keys_exact_creative_burst() {
        fn wire(packets: Vec<c2s::CreativeInventoryAction>) -> Vec<Vec<u8>> {
            packets
                .into_iter()
                .map(|packet| {
                    let mut body = mc_protocol::PacketBuffer::new();
                    mc_protocol::play::C2SPlay::CreativeInventoryAction(packet).write(&mut body);
                    body.as_slice().to_vec()
                })
                .collect()
        }
        let font = uniform_font();
        // Mouse button 4 ("Forward") bound to key.hotbar.3 is stored as `4 - 100`.
        let mut mouse_bound = ContainerBindings::default();
        mouse_bound.hotbar_keys[2] = 4 - 100;

        for (tab, hovered) in [
            (0, CreativeSlot::Grid(0)),
            (
                CreativeTab::Inventory.index().unwrap(),
                CreativeSlot::Player(20),
            ),
        ] {
            let setup = |bindings: ContainerBindings| {
                let mut s = opened();
                s.set_tab(tab);
                s.set_bindings(bindings);
                s.player.set_slot_contents(20, Some(stack("stone", 3)));
                s.take_outgoing();
                s.hovered = Some(hovered);
                s
            };
            let mut keyboard = setup(ContainerBindings::default());
            keyboard.key_typed_with_font(Some('3'), KEY_HOTBAR_1 + 2, Modifiers::NONE, &font);
            let expected = wire(keyboard.take_outgoing());
            assert!(
                !expected.is_empty(),
                "{hovered:?}: the keyboard control sends C10s"
            );

            let mut mouse = setup(mouse_bound);
            let (x, y) = (mouse.gui_left() + 20, mouse.gui_top() + 20);
            assert_eq!(
                mouse.mouse_clicked_with_mods(x, y, 4, Modifiers::NONE),
                ScreenAction::None
            );
            mouse.mouse_released(x, y, 4);
            assert_eq!(wire(mouse.take_outgoing()), expected, "{hovered:?}");
            assert_eq!(mouse.player.slot_stack(38), keyboard.player.slot_stack(38));
            assert_eq!(mouse.player.slot_stack(20), keyboard.player.slot_stack(20));
            assert!(mouse.player.cursor().is_none(), "{hovered:?}");
        }
    }

    /// java: `GuiContainerCreative.CreativeSlot.getSlotTexture` (`:994-997`) delegates to the
    /// wrapped slot, so the creative inventory tab shows the same four armour silhouettes the
    /// survival inventory does. (`drawSlot`'s `flag1` only skips the item: `drawScreen` still
    /// paints the hover rect over a silhouette, see `w21_ci_hover_rect_is_drawn_over_the_cell`.)
    #[test]
    fn the_creative_inventory_tab_shows_the_armour_silhouettes() {
        let mut s = CreativeScreen::new();
        s.set_tab(11); // CreativeTab::Inventory
        assert_eq!(s.tab(), CreativeTab::Inventory);
        let armour: Vec<_> = s
            .slot_rects()
            .into_iter()
            .filter(|(slot, _, _)| s.slot_background(*slot).is_some())
            .collect();
        assert_eq!(armour.len(), 4, "exactly the four armour cells");
        assert_eq!(
            s.slot_background(armour[0].0),
            Some("items/empty_armor_slot_helmet")
        );
        // Every other tab's cells have none.
        s.set_tab(0);
        assert!(s
            .slot_rects()
            .into_iter()
            .all(|(slot, _, _)| s.slot_background(slot).is_none()));
    }

    // ------------------------------------------------- And friends

    /// Press with the held stack on `slots[0]`, then move over each slot in turn (the button
    /// stays down), as `GuiScreen.handleMouseInput` feeds `mouseClickMove`.
    fn w21_ci_sweep(s: &mut CreativeScreen, button: i32, slots: &[u16]) {
        let (x, y) = centre_of(s, CreativeSlot::Player(slots[0]));
        s.mouse_clicked(x, y, button);
        for id in slots {
            let (x, y) = centre_of(s, CreativeSlot::Player(*id));
            s.mouse_dragged(x, y, button, 0);
        }
    }

    /// `(slot, previewed count, yellow count)` for every swept slot, in sweep order.
    fn w21_ci_previews(s: &CreativeScreen) -> Vec<(u16, i32, Option<i32>)> {
        let cursor = s.player.cursor().cloned().expect("a held stack");
        s.drag_splitting_slots
            .iter()
            .map(|id| {
                let (preview, _, yellow) = s.drag_split_preview(&cursor, *id);
                (*id, preview.count, yellow)
            })
            .collect()
    }

    fn w21_ci_is_yellow(color: [f32; 4]) -> bool {
        // `§e`, `0xFFFF55`.
        (color[0] - 1.0).abs() < 1e-3
            && (color[1] - 1.0).abs() < 1e-3
            && (color[2] - 85.0 / 255.0).abs() < 1e-3
    }

    fn w21_ci_is_hover_white(color: [f32; 4]) -> bool {
        color[..3].iter().all(|c| (c - 1.0).abs() < 1e-3) && (color[3] - 128.0 / 255.0).abs() < 1e-3
    }

    /// java: `GuiContainer.drawSlot:243-273` and `updateDragSplitting:309-336`. 64 over three
    /// Inventory-tab slots, the middle one holding 50: 21 each, the middle one capped at a yellow
    /// 64 (21 + 50 = 71), and 64 - 21 - 14 - 21 = 8 left on the cursor. The release then
    /// distributes exactly what was previewed.
    #[test]
    fn w21_ci_a_left_sweep_previews_the_split_the_capped_slot_and_the_remnant() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(10, Some(stack("diamond", 50)));
        s.player.set_cursor(Some(stack("diamond", 64)));
        w21_ci_sweep(&mut s, 0, &[9, 10, 11]);
        assert_eq!(
            w21_ci_previews(&s),
            vec![(9, 21, None), (10, 64, Some(64)), (11, 21, None)]
        );
        assert_eq!(s.drag_splitting_remnant, 8);

        let (x, y) = centre_of(&s, CreativeSlot::Player(11));
        s.mouse_released(x, y, 0);
        assert_eq!(s.player.slot_stack(9), Some(&stack("diamond", 21)));
        assert_eq!(s.player.slot_stack(10), Some(&stack("diamond", 64)));
        assert_eq!(s.player.slot_stack(11), Some(&stack("diamond", 21)));
        assert_eq!(s.player.cursor(), Some(&stack("diamond", 8)));
    }

    /// The right button previews one per slot (the held stack used up: remnant 0), and the
    /// creative-only pick-button sweep a full stack per slot (`computeStackSize` mode 2), which
    /// can take more than is held: the remnant goes negative, as Java's does, and the release
    /// fills every slot and empties the cursor.
    #[test]
    fn w21_ci_right_and_pick_button_sweeps_preview_one_and_a_full_stack_per_slot() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("gold_ingot", 3)));
        w21_ci_sweep(&mut s, 1, &[36, 37, 38]);
        assert_eq!(
            w21_ci_previews(&s),
            vec![(36, 1, None), (37, 1, None), (38, 1, None)]
        );
        assert_eq!(s.drag_splitting_remnant, 0);
        let (x, y) = centre_of(&s, CreativeSlot::Player(38));
        s.mouse_released(x, y, 1);
        assert_eq!(s.player.cursor(), None);
        assert_eq!(s.player.slot_stack(37), Some(&stack("gold_ingot", 1)));

        let mut s = opened();
        s.player.set_cursor(Some(stack("diamond", 5)));
        w21_ci_sweep(&mut s, DEFAULT_PICK_BLOCK_BUTTON, &[39, 40, 41]);
        assert_eq!(
            w21_ci_previews(&s),
            vec![(39, 64, None), (40, 64, None), (41, 64, None)]
        );
        assert_eq!(s.drag_splitting_remnant, 5 - 3 * 64);
        let (x, y) = centre_of(&s, CreativeSlot::Player(41));
        s.mouse_released(x, y, DEFAULT_PICK_BLOCK_BUTTON);
        for id in 39..42 {
            assert_eq!(s.player.slot_stack(id), Some(&stack("diamond", 64)));
        }
        assert_eq!(s.player.cursor(), None);
    }

    /// java: `drawSlot:245-248`: with one swept slot that slot draws nothing at all, not even
    /// its own stack, while every other cell draws as usual.
    #[test]
    fn w21_ci_a_lone_swept_slot_draws_nothing() {
        let mut s = opened();
        s.player.set_slot_contents(37, Some(stack("diamond", 10)));
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, CreativeSlot::Player(37), 0, 0);
        assert!(
            !d.vertices().is_empty(),
            "its count of 10 is drawn normally"
        );

        s.player.set_cursor(Some(stack("diamond", 64)));
        w21_ci_sweep(&mut s, 0, &[37]);
        assert_eq!(s.drag_splitting_slots, vec![37]);
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, CreativeSlot::Player(37), 0, 0);
        assert!(d.vertices().is_empty(), "a lone swept slot draws nothing");
    }

    /// java: `drawSlot:250-303`: a swept cell (two or more swept) draws the `0x80FFFFFF` rect
    /// first, then the stack it would receive; a capped count is yellow.
    #[test]
    fn w21_ci_a_swept_cell_draws_its_rect_under_the_ghost_stack_and_a_yellow_cap() {
        let mut s = opened();
        s.player.set_slot_contents(37, Some(stack("diamond", 50)));
        s.player.set_cursor(Some(stack("diamond", 64)));
        w21_ci_sweep(&mut s, 0, &[36, 37, 38]);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, CreativeSlot::Player(37), 0, 0);
        let v = d.vertices();
        assert!(v.len() > 6, "the rect, then the count text");
        assert!(v[..6].iter().all(|vert| w21_ci_is_hover_white(vert.color)));
        assert!(
            v[6..].iter().any(|vert| w21_ci_is_yellow(vert.color)),
            "21 + 50 is capped at a yellow 64"
        );

        // The same cell outside a sweep: its own white 50, no rect.
        s.reset_drag();
        let mut d = DrawList::new();
        s.draw_slot(&mut d, &mut font, CreativeSlot::Player(37), 0, 0);
        assert!(!d
            .vertices()
            .iter()
            .any(|vert| w21_ci_is_hover_white(vert.color)));
        assert!(!d.vertices().iter().any(|vert| w21_ci_is_yellow(vert.color)));
    }

    /// java: `drawScreen:158-167`: with two or more slots swept the held stack shows the
    /// remnant, a yellow `0` when the sweep uses it all; with one swept slot its full count.
    #[test]
    fn w21_ci_the_held_stack_shows_a_yellow_zero_when_the_sweep_uses_it_all() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("gold_ingot", 3)));
        w21_ci_sweep(&mut s, 1, &[36]);
        let park = (s.gui_left() + 120, s.gui_top() + 70);
        let mut d = DrawList::new();
        s.draw(&mut d, &mut uniform_font(), &ctx(), park);
        assert!(!d.vertices().iter().any(|vert| w21_ci_is_yellow(vert.color)));

        for id in [37, 38] {
            let (x, y) = centre_of(&s, CreativeSlot::Player(id));
            s.mouse_dragged(x, y, 1, 0);
        }
        let mut d = DrawList::new();
        s.draw(&mut d, &mut uniform_font(), &ctx(), park);
        assert!(
            d.vertices().iter().any(|vert| w21_ci_is_yellow(vert.color)),
            "the held stack's yellow 0"
        );
    }

    /// java: `drawSlot:268-272`: a swept slot that can no longer take the held stack (the server
    /// put dirt in it mid-sweep) leaves the sweep when it is drawn, and the remnant is
    /// recomputed without it.
    #[test]
    fn w21_ci_a_swept_slot_that_stops_accepting_leaves_the_sweep_when_drawn() {
        let mut s = opened();
        s.player.set_cursor(Some(stack("diamond", 64)));
        w21_ci_sweep(&mut s, 0, &[36, 37, 38]);
        assert_eq!(s.drag_splitting_remnant, 64 - 3 * 21);
        s.player.set_slot_contents(37, Some(stack("dirt", 1)));
        let mut d = DrawList::new();
        s.draw(&mut d, &mut uniform_font(), &ctx(), (0, 0));
        assert_eq!(s.drag_splitting_slots, vec![36, 38]);
        assert_eq!(s.drag_splitting_remnant, 0);
    }

    /// java: `GuiContainer.drawScreen:126-138` paints the hover rect after `drawSlot`, over the
    /// item, and over an empty armour cell's silhouette too (`flag1` only skips the item): every
    /// quad of the cell's content comes before the half-white rect.
    #[test]
    fn w21_ci_hover_rect_is_drawn_over_the_cell() {
        let inventory = CreativeTab::Inventory.index().unwrap();
        for (tab, id, filled) in [
            (0u8, 36u16, true),
            (inventory, 9, true),
            (inventory, 5, false),
        ] {
            let mut s = opened();
            s.set_tab(tab);
            if filled {
                s.player
                    .set_slot_contents(usize::from(id), Some(stack("stone", 5)));
            }
            let (mx, my) = centre_of(&s, CreativeSlot::Player(id));
            let (cx, cy) = ((mx - 8) as f32, (my - 8) as f32);
            let mut d = DrawList::new();
            s.draw(&mut d, &mut uniform_font(), &ctx(), (mx, my));
            let v = d.vertices();
            let corner = |p: [f32; 2]| {
                (p[0] == cx || p[0] == cx + 16.0) && (p[1] == cy || p[1] == cy + 16.0)
            };
            let interior =
                |p: [f32; 2]| p[0] > cx && p[0] < cx + 16.0 && p[1] > cy && p[1] < cy + 16.0;
            let hover: Vec<usize> = (0..v.len())
                .filter(|i| w21_ci_is_hover_white(v[*i].color) && corner(v[*i].pos))
                .collect();
            // The count text inside the cell, or the silhouette's opaque quad on its corners.
            let content: Vec<usize> = (0..v.len())
                .filter(|i| {
                    let (pos, color) = (v[*i].pos, v[*i].color);
                    !w21_ci_is_hover_white(color)
                        && (interior(pos) || (corner(pos) && (color[3] - 1.0).abs() < 1e-3))
                })
                .collect();
            assert_eq!(
                hover.len(),
                6,
                "tab {tab} slot {id}: one hover rect on the cell"
            );
            assert!(
                !content.is_empty(),
                "tab {tab} slot {id}: the cell's content is drawn"
            );
            assert!(
                content.iter().max() < hover.iter().min(),
                "tab {tab} slot {id}: the hover rect is drawn over the cell's content"
            );
            assert_eq!(s.hovered(), Some(CreativeSlot::Player(id)));
        }
    }

    /// java: a left release over a tab switches in
    /// `GuiContainerCreative.mouseReleased`, which returns before `GuiContainer.mouseReleased`
    /// disarms the drag; `setCurrentCreativeTab` clears only the swept set. So the next press
    /// neither clicks nor re-arms: a right sweep after it still carries the left drag's button,
    /// and its right release cancels the drag instead of dealing one per slot.
    #[test]
    fn w21_ci_a_drag_released_over_a_tab_stays_armed_on_the_new_tab() {
        let mut s = opened();
        s.set_clock_ms(60_000);
        s.player.set_cursor(Some(stack("diamond", 10)));
        let (x, y) = centre_of(&s, CreativeSlot::Player(36));
        s.mouse_clicked(x, y, 0);
        let inventory = CreativeTab::Inventory.index().unwrap();
        let plate = tab_plate(inventory, false);
        let (tx, ty) = (s.gui_left() + plate.x + 14, s.gui_top() + plate.y + 16);
        assert_eq!(s.tab_at(tx, ty), Some(inventory));
        s.mouse_released(tx, ty, 0);
        assert_eq!(s.tab(), CreativeTab::Inventory);
        assert!(s.drag_splitting, "still armed, as in Java");
        assert!(s.drag_splitting_slots.is_empty());

        s.take_outgoing();
        s.set_clock_ms(70_000);
        w21_ci_sweep(&mut s, 1, &[9, 10]);
        assert_eq!(s.drag_splitting_button, 0, "the right press did not re-arm");
        assert_eq!(
            s.drag_splitting_slots,
            vec![9, 10],
            "but the moves still sweep"
        );
        let (x, y) = centre_of(&s, CreativeSlot::Player(10));
        s.mouse_released(x, y, 1);
        assert_eq!(s.player.cursor(), Some(&stack("diamond", 10)), "cancelled");
        assert_eq!(s.player.slot_stack(9), None);
        assert_eq!(s.player.slot_stack(10), None);
        assert!(s.take_outgoing().is_empty());
        assert!(!s.drag_splitting);
    }

    /// java: a press outside the window with nothing held is
    /// `handleMouseClick(null, -999, 0, 4)`. On the Inventory tab it reaches the last `else`
    /// (`GuiContainerCreative:142-146`), whose `detectAndSendChanges` uploads a slot the server
    /// changed since the listener last looked; on an item tab nothing is sent. Every tab sets
    /// `field_147057_D`: the search box clears on the next key.
    #[test]
    fn w21_ci_an_empty_handed_press_outside_flushes_only_the_inventory_tabs_listener() {
        for (tab, expected) in [
            (CreativeTab::Inventory, vec![(21, Some(stack("dirt", 4)))]),
            (CreativeTab::Items("materials"), vec![]),
        ] {
            let mut s = opened();
            s.set_tab(tab.index().unwrap());
            s.set_clock_ms(60_000);
            s.take_outgoing();
            s.player.set_slot_contents(21, Some(stack("dirt", 4)));
            s.clear_search_on_next_key = false;
            let (x, y) = (s.gui_left() - 20, s.gui_top() + 40);
            assert!(s.outside_panel(x, y) && s.tab_at(x, y).is_none());
            s.mouse_clicked(x, y, 0);
            s.mouse_released(x, y, 0);
            let sent: Vec<_> = s
                .take_outgoing()
                .into_iter()
                .map(|packet| (packet.slot_id, packet.stack))
                .collect();
            assert_eq!(sent, expected, "{tab:?}");
            assert!(s.clear_search_on_next_key, "{tab:?}");
        }
    }

    /// java: a held stack released inside the Inventory tab's panel
    /// but over no cell is `handleMouseClick(null, -1, 0, 0)`, which reaches the same last
    /// `else`: nothing moves, the stack stays held, and the listener uploads the slot the server
    /// changed. Released outside, the stack is dropped instead (`clickType` 4), with no flush.
    #[test]
    fn w21_ci_a_held_stack_released_over_no_inventory_cell_flushes_the_listener() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.set_clock_ms(60_000);
        s.player.set_cursor(Some(stack("diamond", 5)));
        s.take_outgoing();
        s.player.set_slot_contents(21, Some(stack("dirt", 4)));
        // Beside the player preview: inside the panel, on no cell.
        let (x, y) = (s.gui_left() + 43, s.gui_top() + 40);
        assert!(!s.outside_panel(x, y) && s.slot_at(x, y).is_none());
        s.mouse_clicked(x, y, 0);
        s.mouse_released(x, y, 0);
        assert_eq!(s.player.cursor(), Some(&stack("diamond", 5)));
        let sent: Vec<_> = s
            .take_outgoing()
            .into_iter()
            .map(|packet| (packet.slot_id, packet.stack))
            .collect();
        assert_eq!(sent, vec![(21, Some(stack("dirt", 4)))]);

        // Outside the panel: the whole stack is dropped and nothing else is sent.
        s.player.set_slot_contents(22, Some(stack("dirt", 2)));
        let (x, y) = (s.gui_left() - 20, s.gui_top() + 40);
        s.set_clock_ms(61_000);
        s.mouse_clicked(x, y, 0);
        s.mouse_released(x, y, 0);
        assert_eq!(s.player.cursor(), None);
        let sent: Vec<_> = s
            .take_outgoing()
            .into_iter()
            .map(|packet| (packet.slot_id, packet.stack))
            .collect();
        assert_eq!(sent, vec![(-1, Some(stack("diamond", 5)))]);
    }

    /// java: `onCraftGuiOpened` runs `detectAndSendChanges` against
    /// `ContainerPlayer.inventoryItemStacks`, which outlives the screen. The first opening
    /// uploads every occupied slot; reopening uploads only what changed meanwhile (here a server
    /// write); after a respawn (a new `ContainerPlayer`) everything is uploaded again.
    #[test]
    fn w21_ci_reopening_uploads_only_what_changed_since_the_listener_last_looked() {
        let slots = |s: &mut CreativeScreen| -> Vec<(i16, Option<ItemStack>)> {
            s.take_outgoing()
                .into_iter()
                .map(|packet| (packet.slot_id, packet.stack))
                .collect()
        };
        let mut player = layouts::player_container(0);
        player.set_slot_contents(20, Some(stack("stone", 7)));
        player.set_slot_contents(36, Some(stack("dirt", 3)));
        let mut s = CreativeScreen::over(player);
        assert_eq!(
            slots(&mut s),
            vec![(20, Some(stack("stone", 7))), (36, Some(stack("dirt", 3)))]
        );

        // Closed: the host keeps the window-0 container; the server changes one slot.
        let mut player = s.player.clone();
        player.set_slot_contents(21, Some(stack("glass", 2)));
        let mut s = CreativeScreen::over(player);
        assert_eq!(slots(&mut s), vec![(21, Some(stack("glass", 2)))]);

        let mut player = s.player.clone();
        let mut s = CreativeScreen::over(player.clone());
        assert!(slots(&mut s).is_empty(), "nothing changed: nothing is sent");

        player.reset_listener_mirror();
        let mut s = CreativeScreen::over(player);
        assert_eq!(
            slots(&mut s).len(),
            3,
            "a respawn's new container uploads everything"
        );
    }

    /// java: `GuiContainerCreative.drawScreen:613-616` labels the destroy
    /// cell with no held-stack test, so carrying a stack to it shows "inventory.binSlot"; an
    /// item's own tooltip stays suppressed while a stack is held.
    #[test]
    fn w21_ci_the_destroy_cell_is_labelled_even_while_a_stack_is_held() {
        let mut s = opened();
        s.set_tab(CreativeTab::Inventory.index().unwrap());
        s.player.set_slot_contents(9, Some(stack("stone", 5)));
        s.player.set_cursor(Some(stack("diamond", 3)));
        let mut font = uniform_font();
        let trash = centre_of(&s, CreativeSlot::Trash);
        s.draw(&mut DrawList::new(), &mut font, &ctx(), trash);
        assert_eq!(
            s.hovered_tooltip(),
            Some(vec!["inventory.binSlot".to_owned()])
        );
        let cell = centre_of(&s, CreativeSlot::Player(9));
        s.draw(&mut DrawList::new(), &mut font, &ctx(), cell);
        assert_eq!(
            s.hovered_tooltip(),
            None,
            "an item's tooltip waits for an empty cursor"
        );
    }

    /// java: `checkHotbarKeys` reaches `handleMouseClick`, which sets
    /// `field_147057_D`: on the Search tab, the next key typed after moving an item with a
    /// number key replaces the query instead of extending it.
    #[test]
    fn w21_ci_a_hotbar_key_on_the_search_tab_makes_the_next_key_replace_the_query() {
        let mut s = opened();
        let mut font = uniform_font();
        s.set_tab(CreativeTab::Search.index().unwrap());
        s.apply_pending_input(&font);
        s.key_typed_with_font(Some('e'), 0, Modifiers::NONE, &font);
        assert_eq!(s.search.text(), "e");
        let (x, y) = centre_of(&s, CreativeSlot::Grid(0));
        s.draw(&mut DrawList::new(), &mut font, &ctx(), (x, y));
        assert!(
            s.stack_at(CreativeSlot::Grid(0)).is_some(),
            "the query lists items"
        );
        s.key_typed_with_font(Some('1'), KEY_HOTBAR_1, Modifiers::NONE, &font);
        assert!(
            s.player.slot_stack(36).is_some(),
            "the number key filled hotbar slot 1"
        );
        assert_eq!(s.search.text(), "e", "and was not typed");
        s.key_typed_with_font(Some('a'), 0, Modifiers::NONE, &font);
        assert_eq!(s.search.text(), "a");
    }
}
