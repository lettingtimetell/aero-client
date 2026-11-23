//! The statistics and achievements screens.
//!
//! java: `client/gui/achievement/GuiStats.java`, `GuiAchievements.java`,
//! `stats/StatFileWriter.java`, `stats/Achievement.java`
//!
//! Both are read-only views over server-supplied data: `S37PacketStatistics` fills the
//! statistics and `S37`'s achievement half fills the achievement grid. Neither screen mutates
//! anything, so both are pure layout over a snapshot — which is all that is here.
//!
//! # What is a port and what is not
//!
//! The four statistics tabs (general, blocks, items, mobs), their column headers, the sort
//! toggle and the `GuiStats`' 288×180 frame are vanilla's. So, now, is the achievements screen's
//! **pannable map**: the tiled block backdrop with its seeded ore scatter and per-row shading,
//! the connecting lines and their arrow-heads, the 26×26 cell plates, the item icon in each
//! cell, and the hover tooltip's three states. See [`AchievementsScreen::draw`] for what is
//! still missing (the accelerating drag smoothing is here; the zoom is modelled and applied but
//! only pixel-verified at 1×).

use mc_assets::{Language, ResourceLocation};

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{
    draw_world_background, ACHIEVEMENT_BACKGROUND_TEXTURE, STATS_ICONS_TEXTURE, TITLE_COLOR,
};
use crate::screens::container::render::{draw_item_icon_tinted, draw_item_icon_tinted_at_z_level};
use crate::screens::container::stack_ext::ItemStack;
use crate::screens::container::tooltip::display_name_localized;
use crate::screens::container::tooltip_box::draw as draw_hovering_text;
use crate::screens::Action;
use crate::widget::draw::{DrawList, Rect, UiCull, UiDepth, UiTexture, GUI_TEXTURE_SIZE};
use crate::widget::Transform;
use crate::widget::{draw_centered_string, draw_string, Button, ScrollList, Scrollbar};

/// The frame is 256×202 of art drawn at 256×202. java: `GuiAchievements` — `i = 256, j = 202`
pub const FRAME_WIDTH: i32 = 256;
/// Frame height.
pub const FRAME_HEIGHT: i32 = 202;

/// Which statistics tab is showing.
// java: GuiStats' four `StatsComponent` lists
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatsTab {
    /// `stats.tooltip.type.statistic` — the general counters.
    General,
    /// Block mining counts.
    Blocks,
    /// Item use / craft / break counts.
    Items,
    /// Mob kill counts.
    Mobs,
}

impl StatsTab {
    /// All four, in vanilla's button order.
    pub const ALL: [StatsTab; 4] = [
        StatsTab::General,
        StatsTab::Blocks,
        StatsTab::Items,
        StatsTab::Mobs,
    ];

    /// The tab button's translation key.
    #[must_use]
    pub const fn translation_key(self) -> &'static str {
        match self {
            StatsTab::General => "stat.generalButton",
            StatsTab::Blocks => "stat.blocksButton",
            StatsTab::Items => "stat.itemsButton",
            StatsTab::Mobs => "stat.mobsButton",
        }
    }

    /// The slot height used by this tab's `GuiSlot`.
    ///
    /// General and mobs deliberately do not share the table tabs' 20-pixel rows: the former
    /// uses ten-pixel rows and the latter uses four font lines. Keeping this beside the tab
    /// identity prevents scroll, hit-testing and draw geometry from drifting apart.
    // java: GuiStats.StatsGeneral/StatsMobsList constructors
    const fn row_height(self) -> i32 {
        match self {
            Self::General => 10,
            Self::Blocks | Self::Items => 20,
            Self::Mobs => 36,
        }
    }

    /// Whether this tab's `GuiSlot` reserves the 20-pixel sortable header.
    // java: GuiStats.Stats constructor; general/mobs leave `hasListHeader` false
    const fn has_header(self) -> bool {
        matches!(self, Self::Blocks | Self::Items)
    }
}

/// One formatted value in a statistics-table column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatValue {
    /// Text drawn in the table.
    pub formatted: String,
    /// Unformatted value used by vanilla's column comparator.
    pub raw: i64,
}

impl StatValue {
    /// A formatted/raw pair.
    #[must_use]
    pub fn new(formatted: impl Into<String>, raw: i64) -> Self {
        Self {
            formatted: formatted.into(),
            raw,
        }
    }
}

/// One statistic row.
#[derive(Debug, Clone, PartialEq)]
pub struct StatRow {
    /// Vanilla's numeric item/entity/registry id. It is the final sort tiebreak.
    pub registry_id: i32,
    /// The already-translated display name.
    pub name: String,
    /// Up to three tab-specific values. Blocks use crafted/used/mined; items use
    /// depleted/crafted/used; mobs use killed/killed-by; general statistics use column zero.
    pub columns: [Option<StatValue>; 3],
    /// Item or block icon for the block/item tables.
    pub icon: Option<ItemStack>,
}

impl StatRow {
    /// A one-column general statistic.
    #[must_use]
    pub fn general(
        registry_id: i32,
        name: impl Into<String>,
        formatted: impl Into<String>,
        raw: i64,
    ) -> Self {
        Self {
            registry_id,
            name: name.into(),
            columns: [Some(StatValue::new(formatted, raw)), None, None],
            icon: None,
        }
    }

    /// A row for one of vanilla's three-column block/item lists.
    #[must_use]
    pub fn item(
        registry_id: i32,
        name: impl Into<String>,
        columns: [Option<StatValue>; 3],
        icon: Option<ItemStack>,
    ) -> Self {
        Self {
            registry_id,
            name: name.into(),
            columns,
            icon,
        }
    }

    /// A three-line mob row. Values should already contain the translated kill phrases.
    #[must_use]
    pub fn mob(
        registry_id: i32,
        name: impl Into<String>,
        killed: StatValue,
        killed_by: StatValue,
    ) -> Self {
        Self {
            registry_id,
            name: name.into(),
            columns: [Some(killed), Some(killed_by), None],
            icon: None,
        }
    }
}

/// One block/item tab's independent three-state sort selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StatSort {
    /// Active header, or `None` for vanilla's registry order.
    pub column: Option<usize>,
    /// `-1` descending, `1` ascending, `0` with no active header.
    pub direction: i8,
}

/// Vanilla's four `IStatType` value formatters.
// java: StatBase.java:24-60
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatFormat {
    /// Integer with `Locale.US` grouping, e.g. `1234 -> "1,234"`.
    Simple,
    /// Game ticks rendered as seconds, minutes, hours, days or years.
    Time,
    /// Centimetres rendered as centimetres, metres or kilometres.
    Distance,
    /// Damage stored in tenths, rendered with two fraction digits.
    Damage,
}

impl StatFormat {
    /// Format one raw `S37PacketStatistics` value exactly as vanilla's `IStatType` does.
    #[must_use]
    pub fn format(self, value: i32) -> String {
        match self {
            Self::Simple => format_grouped_i32(value),
            Self::Time => {
                let seconds = f64::from(value) / 20.0;
                let minutes = seconds / 60.0;
                let hours = minutes / 60.0;
                let days = hours / 24.0;
                let years = days / 365.0;
                if years > 0.5 {
                    format_two_decimals(years, " y")
                } else if days > 0.5 {
                    format_two_decimals(days, " d")
                } else if hours > 0.5 {
                    format_two_decimals(hours, " h")
                } else if minutes > 0.5 {
                    format_two_decimals(minutes, " m")
                } else {
                    // This branch is raw Java double concatenation, not DecimalFormat.
                    format!("{} s", java_plain_double(seconds))
                }
            }
            Self::Distance => {
                let metres = f64::from(value) / 100.0;
                let kilometres = metres / 1_000.0;
                if kilometres > 0.5 {
                    format_two_decimals(kilometres, " km")
                } else if metres > 0.5 {
                    format_two_decimals(metres, " m")
                } else {
                    format!("{value} cm")
                }
            }
            Self::Damage => format_two_decimals(f64::from(value) * 0.1, ""),
        }
    }
}

fn format_grouped_i32(value: i32) -> String {
    let negative = value < 0;
    let digits = i64::from(value).abs().to_string();
    let first = digits.len() % 3;
    let mut out = String::with_capacity(digits.len() + digits.len() / 3 + usize::from(negative));
    if negative {
        out.push('-');
    }
    for (index, byte) in digits.bytes().enumerate() {
        if index > 0 && index % 3 == first % 3 {
            out.push(',');
        }
        out.push(char::from(byte));
    }
    out
}

fn format_two_decimals(value: f64, suffix: &str) -> String {
    format!("{value:.2}{suffix}")
}

fn java_plain_double(value: f64) -> String {
    let rendered = value.to_string();
    if value.is_finite()
        && value.fract() == 0.0
        && !rendered.contains('.')
        && !rendered.contains('e')
        && !rendered.contains('E')
    {
        format!("{rendered}.0")
    } else {
        rendered
    }
}

/// The statistics screen.
#[derive(Debug)]
pub struct StatsScreen {
    /// The four tabs' rows.
    pub rows: std::collections::BTreeMap<u8, Vec<StatRow>>,
    /// Which tab is showing.
    pub tab: StatsTab,
    /// Whether the server has sent the statistics yet.
    pub loading: bool,
    /// The stitched item atlas used by block and item row icons.
    pub atlas_texture: Option<UiTexture>,
    sorts: std::collections::BTreeMap<u8, StatSort>,
    /// One `GuiSlot` per tab. Vanilla keeps four independent list objects, so switching tabs
    /// preserves each tab's scroll offset and drag state instead of sharing one global offset.
    lists: [ScrollList; 4],
    /// Compatibility mirror of the active list's integer scroll position. The real state lives
    /// in [`Self::lists`]; retaining this field keeps the old headless tests and diagnostics able
    /// to inspect the active offset without exposing the widget internals.
    scroll: i32,
    width: i32,
    tabs: Vec<Button>,
    done: Button,
    /// `Stats.field_148218_l`: the three-column header held down by the left mouse button.
    pressed_header: Option<usize>,
    /// A sort header or tab button was pressed but returned no navigation [`Action`]. The host
    /// drains this to play vanilla's `gui.button.press` for otherwise-internal controls.
    internal_button_press: bool,
}

/// The row pitch of the statistics list. java: `GuiStats` — `new Stats(mc, ..., 20)`
pub const STAT_ROW_HEIGHT: i32 = 20;

impl Default for StatsScreen {
    fn default() -> Self {
        Self::new()
    }
}

impl StatsScreen {
    fn make_list(tab: StatsTab) -> ScrollList {
        let mut list = ScrollList::new(0, 0, 32, 0, tab.row_height());
        list.set_header(tab.has_header(), 20);
        list.show_selection_box = false;
        list
    }

    /// An empty, still-loading statistics screen.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rows: std::collections::BTreeMap::new(),
            tab: StatsTab::General,
            loading: true,
            atlas_texture: None,
            sorts: std::collections::BTreeMap::new(),
            lists: std::array::from_fn(|index| Self::make_list(StatsTab::ALL[index])),
            scroll: 0,
            width: 0,
            tabs: Vec::new(),
            done: Button::new(0, 0, 0, "Done"),
            pressed_header: None,
            internal_button_press: false,
        }
    }

    /// Go back to "Downloading statistics..." the way a window resize
    /// does in vanilla.
    ///
    /// java: `GuiScreen.onResize` -> `setWorldAndResolution` clears `buttonList` and calls
    /// `GuiStats.initGui` (`:55-60`) again, which sets the `doesGuiPauseGame` loading flag and
    /// re-sends `C16 REQUEST_STATS` (the host sends it). The next `S37` reaches `doneLoading`
    /// (`:167-176`), which only while loading builds four new `GuiSlot` lists (scroll, drag and
    /// header sort start over), re-adds the buttons and selects the General tab. Until then the
    /// screen draws only the loading text and ignores list input. The item atlas handle is kept.
    pub fn restart_loading(&mut self) {
        let atlas_texture = self.atlas_texture.take();
        *self = Self::new();
        self.atlas_texture = atlas_texture;
    }

    /// Rebuild all four vanilla tabs from the latest `S37PacketStatistics` snapshot.
    ///
    /// The packet carries only `(stat id, value)` pairs. Vanilla joins those ids against
    /// `StatList` after the reply arrives: general counters remain visible at zero, while the
    /// block, item, and mob lists omit rows whose associated counters are all non-positive.
    /// Keeping that join here makes the screen independent of protocol packet types and gives
    /// the host one atomic hand-off from its persistent statistics map.
    pub fn load_statistics(
        &mut self,
        values: &std::collections::HashMap<String, i32>,
        lang: &Language,
    ) {
        let general = mc_data::general_stats()
            .enumerate()
            .map(|(index, info)| {
                let raw = stat_raw(values, info.id);
                StatRow::general(
                    index as i32,
                    translated_stat_name(info, lang),
                    info.format.format(raw),
                    i64::from(raw),
                )
            })
            .collect();

        let blocks = mc_data::block_stats()
            .iter()
            .map(|association| {
                StatRow::item(
                    i32::from(association.item.0),
                    translated_item_name(association.item, lang),
                    [
                        association
                            .crafted
                            .map(|id| formatted_stat_value(values, id)),
                        association.used.map(|id| formatted_stat_value(values, id)),
                        Some(formatted_stat_value(values, association.mined)),
                    ],
                    Some(stat_item_icon(association.item)),
                )
            })
            .collect();

        let items = mc_data::item_stats()
            .iter()
            .map(|association| {
                StatRow::item(
                    i32::from(association.item.0),
                    translated_item_name(association.item, lang),
                    [
                        association
                            .depleted
                            .map(|id| formatted_stat_value(values, id)),
                        association
                            .crafted
                            .map(|id| formatted_stat_value(values, id)),
                        Some(formatted_stat_value(values, association.used)),
                    ],
                    Some(stat_item_icon(association.item)),
                )
            })
            .collect();

        let mobs = mc_data::mob_stats()
            .iter()
            .map(|association| {
                let killed = stat_raw(values, association.killed);
                let killed_by = stat_raw(values, association.killed_by);
                let entity_key = format!("entity.{}.name", association.entity_name);
                let entity_name = lang.tr(&entity_key).to_owned();
                let killed_count = killed.to_string();
                let killed_by_count = killed_by.to_string();
                let killed_text = if killed == 0 {
                    lang.format("stat.entityKills.none", &[&entity_name])
                } else {
                    lang.format("stat.entityKills", &[&killed_count, &entity_name])
                };
                let killed_by_text = if killed_by == 0 {
                    lang.format("stat.entityKilledBy.none", &[&entity_name])
                } else {
                    lang.format("stat.entityKilledBy", &[&entity_name, &killed_by_count])
                };
                StatRow::mob(
                    i32::from(association.entity_id),
                    entity_name,
                    StatValue::new(killed_text, i64::from(killed)),
                    StatValue::new(killed_by_text, i64::from(killed_by)),
                )
            })
            .collect();

        self.set_rows(StatsTab::General, general);
        self.set_rows(StatsTab::Blocks, blocks);
        self.set_rows(StatsTab::Items, items);
        self.set_rows(StatsTab::Mobs, mobs);
    }

    /// Fill one tab's rows. Called when `S37PacketStatistics` arrives.
    pub fn set_rows(&mut self, tab: StatsTab, mut rows: Vec<StatRow>) {
        if tab != StatsTab::General {
            rows.retain(|row| row.columns.iter().flatten().any(|value| value.raw > 0));
        }
        if matches!(tab, StatsTab::Blocks | StatsTab::Items) {
            Self::sort_rows(&mut rows, self.sort_for(tab));
        }
        self.rows.insert(tab as u8, rows);
        self.sync_list_size(tab);
        self.loading = false;
        self.refresh_tab_buttons();
    }

    /// Keep the tab's `GuiSlot` view in step with its joined rows.
    fn sync_list_size(&mut self, tab: StatsTab) {
        let index = tab as usize;
        self.lists[index].size = self.rows.get(&(tab as u8)).map_or(0, Vec::len);
        self.lists[index].set_header(tab.has_header(), 20);
        self.lists[index].clamp_scroll();
        if tab == self.tab {
            self.scroll = self.lists[index].scroll_pixels();
        }
    }

    /// The active tab's vanilla `GuiSlot`.
    #[must_use]
    fn current_list(&self) -> &ScrollList {
        &self.lists[self.tab as usize]
    }

    /// The active tab's vanilla `GuiSlot`.
    fn current_list_mut(&mut self) -> &mut ScrollList {
        &mut self.lists[self.tab as usize]
    }

    fn sort_rows(rows: &mut [StatRow], sort: StatSort) {
        rows.sort_by(|a, b| {
            let ordered = sort.column.map_or(std::cmp::Ordering::Equal, |column| {
                match (&a.columns[column], &b.columns[column]) {
                    (None, Some(_)) => std::cmp::Ordering::Greater,
                    (Some(_), None) => std::cmp::Ordering::Less,
                    (Some(a), Some(b)) if a.raw != b.raw => {
                        if sort.direction < 0 {
                            b.raw.cmp(&a.raw)
                        } else {
                            a.raw.cmp(&b.raw)
                        }
                    }
                    _ => std::cmp::Ordering::Equal,
                }
            });
            ordered.then(a.registry_id.cmp(&b.registry_id))
        });
    }

    /// Cycle one header through descending, ascending and registry order.
    // java: GuiStats.Stats.func_148212_h
    pub fn cycle_sort(&mut self, column: usize) {
        if column >= 3 || !matches!(self.tab, StatsTab::Blocks | StatsTab::Items) {
            return;
        }
        let state = self.sorts.entry(self.tab as u8).or_default();
        if state.column != Some(column) {
            state.column = Some(column);
            state.direction = -1;
        } else if state.direction == -1 {
            state.direction = 1;
        } else {
            *state = StatSort::default();
        }
        self.sort_current_tab();
    }

    /// Compatibility helper: cycle column zero on the current block/item tab.
    pub fn toggle_sort(&mut self) {
        self.cycle_sort(0);
    }

    fn sort_current_tab(&mut self) {
        let key = self.tab as u8;
        let sort = self.sort_for(self.tab);
        if let Some(rows) = self.rows.get_mut(&key) {
            Self::sort_rows(rows, sort);
        }
    }

    /// Sort state for one tab. General and mobs always use registry order.
    #[must_use]
    pub fn sort_for(&self, tab: StatsTab) -> StatSort {
        self.sorts.get(&(tab as u8)).copied().unwrap_or_default()
    }

    fn refresh_tab_buttons(&mut self) {
        for (index, tab) in StatsTab::ALL.iter().copied().enumerate() {
            if let Some(button) = self.tabs.get_mut(index) {
                button.enabled = tab == StatsTab::General
                    || self
                        .rows
                        .get(&(tab as u8))
                        .is_some_and(|rows| !rows.is_empty());
            }
        }
    }

    /// The visible tab's rows.
    #[must_use]
    pub fn current_rows(&self) -> &[StatRow] {
        self.rows
            .get(&(self.tab as u8))
            .map_or(&[][..], Vec::as_slice)
    }

    /// Whether this screen pauses an integrated single-player world.
    ///
    /// java: `GuiStats.doesGuiPauseGame()` returns the inverse of its misleadingly named
    /// `doesGuiPauseGame` loading flag: loading keeps the world ticking, loaded statistics pause.
    #[must_use]
    pub const fn pauses_game(&self) -> bool {
        !self.loading
    }

    /// Drain an internal-control press that needs `gui.button.press` from the host.
    ///
    /// Done returns [`Action::Back`] and is already visible to the generic screen dispatcher;
    /// this signal covers tab buttons and sortable headers, whose action is intentionally local.
    pub fn take_internal_button_press(&mut self) -> bool {
        std::mem::take(&mut self.internal_button_press)
    }

    /// Lay out. java: the five buttons across the bottom at `height - 52` and `height - 28`.
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.width = w;
        for tab in StatsTab::ALL {
            let index = tab as usize;
            let list = &mut self.lists[index];
            // java: `new GuiSlot(mc, width, height, 32, height - 64, slotHeight)`.
            list.set_dimensions(w, h, 32, h - 64);
            list.set_header(tab.has_header(), 20);
            list.size = self.rows.get(&(tab as u8)).map_or(0, Vec::len);
            list.show_selection_box = false;
            list.clamp_scroll();
        }
        self.scroll = self.current_list().scroll_pixels();
        // java: four 80-wide tabs in one row at `height - 52`, starting at `width / 2 - 160`.
        self.tabs = StatsTab::ALL
            .iter()
            .enumerate()
            .map(|(i, tab)| {
                Button::sized(
                    i as i32 + 1,
                    w / 2 - 160 + i as i32 * 80,
                    h - 52,
                    80,
                    20,
                    lang.tr(tab.translation_key()).to_owned(),
                )
            })
            .collect();
        // java: `new GuiButton(0, width / 2 + 4, height - 28, 150, 20, "gui.done")`
        self.done = Button::sized(
            0,
            w / 2 + 4,
            h - 28,
            150,
            20,
            lang.tr("gui.done").to_owned(),
        );
        self.refresh_tab_buttons();
    }

    /// Where row `index` draws.
    #[must_use]
    pub fn row_rect(&self, ctx: &ScreenCtx, index: usize) -> Rect {
        let mut rect = self.current_list().slot_rect(index);
        // `ScrollList::slot_rect` models GuiSlot's visible `slotHeight - 4` draw area. The
        // public stats rectangle remains the whole logical slot so item tooltips and row hit
        // tests include vanilla's four-pixel inter-entry gap.
        rect.height = self.tab.row_height();
        // Keep this method's context argument meaningful to callers that calculate against a
        // freshly sized screen before `init` has run. Once laid out, the list is authoritative.
        if self.width == 0 {
            rect.x = ctx.width / 2 - 108;
        }
        rect
    }

    /// The active table's scrollbar geometry, if its content exceeds the viewport.
    ///
    /// This is intentionally a read-only view of the shared `GuiSlot` arithmetic so callers and
    /// headless tests can inspect the bar without reaching into the screen's per-tab storage.
    #[must_use]
    pub fn scrollbar(&self) -> Option<Scrollbar> {
        let list = self.current_list();
        if list.width <= 0 || list.bottom <= list.top {
            None
        } else {
            list.scrollbar()
        }
    }

    /// Scroll, clamped.
    pub fn scroll_by(&mut self, pixels: i32) {
        self.current_list_mut().scroll_by(pixels);
        self.scroll = self.current_list().scroll_pixels();
    }

    /// Apply a raw wheel event to the active list, preserving GuiSlot's half-slot step and its
    /// four-sided inclusive viewport gate. The client host should call this from its wheel seam.
    // java: GuiSlot.handleMouseInput (`slotHeight / 2`, only in-list)
    pub fn mouse_wheel(&mut self, x: i32, y: i32, delta: f32) {
        if !self.loading {
            self.current_list_mut().mouse_wheel_at(x, y, delta);
            self.scroll = self.current_list().scroll_pixels();
        }
    }

    /// Continue a content or scrollbar drag on the active list.
    // java: GuiSlot.handleMouseInput's `amountScrolled` drag branch
    pub fn mouse_dragged(&mut self, x: i32, y: i32) {
        if !self.loading {
            // `GuiSlot.handleMouseInput` is wrapped in `isMouseYWithinSlotBounds`, so a drag
            // stops being sampled as soon as the cursor leaves the viewport. The shared list
            // stores only the y-coordinate and therefore needs this screen-level x/y gate.
            if self.current_list().is_mouse_within_bounds(x, y) {
                self.current_list_mut().mouse_dragged(y);
            }
            self.scroll = self.current_list().scroll_pixels();
        }
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if self.loading || button != 0 {
            return Action::None;
        }
        if self.done.mouse_pressed(x, y) {
            return Action::Back;
        }
        for (i, b) in self.tabs.iter().enumerate() {
            if b.mouse_pressed(x, y) {
                self.tab = StatsTab::ALL[i];
                self.pressed_header = None;
                self.scroll = self.current_list().scroll_pixels();
                self.internal_button_press = true;
                return Action::None;
            }
        }
        let tab = self.tab;
        let list = self.current_list();
        if tab.has_header() {
            let scroll = list.scroll_pixels();
            let header_bottom = list.top + 4 + list.header_padding - scroll;
            let (content_left, _) = list.content_bounds();
            // java: `GuiSlot.handleMouseInput` subtracts `(width - getListWidth()) / 2`, which
            // is two pixels left of `drawListHeader`'s origin. `Stats.func_148132_a` then accepts
            // the full 36px windows covering the sort arrow *and* the column icon. Header
            // tooltips deliberately retain their narrower, draw-origin-relative 18px windows.
            let local_x = x - content_left;
            let column = if y >= list.top && y < header_bottom && (79..115).contains(&local_x) {
                Some(0)
            } else if y >= list.top && y < header_bottom && (129..165).contains(&local_x) {
                Some(1)
            } else if y >= list.top && y < header_bottom && (179..215).contains(&local_x) {
                Some(2)
            } else {
                None
            };
            if let Some(column) = column {
                self.pressed_header = Some(column);
                self.internal_button_press = true;
                self.cycle_sort(column);
                return Action::None;
            }
        }
        // GuiSlot owns both content and scrollbar dragging. A header click above returned early,
        // so this cannot accidentally arm a drag while sorting.
        self.current_list_mut().mouse_pressed(x, y);
        self.scroll = self.current_list().scroll_pixels();
        Action::None
    }

    /// Release the pressed sort-header art. java: `drawListHeader` clears
    /// `field_148218_l` as soon as the left mouse button is no longer down.
    pub fn mouse_released(&mut self) {
        self.current_list_mut().mouse_released();
        self.pressed_header = None;
    }

    /// Draw.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, true);
        if self.loading {
            let elapsed_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_millis());
            draw_centered_string(
                d,
                font,
                lang.tr("multiplayer.downloadingStats"),
                ctx.width / 2,
                ctx.height / 2,
                TITLE_COLOR,
            );
            draw_centered_string(
                d,
                font,
                AchievementsScreen::loading_spinner(elapsed_ms),
                ctx.width / 2,
                ctx.height / 2 + font.font_height() * 2,
                TITLE_COLOR,
            );
            return;
        }

        // `GuiSlot.drawScreen` paints its scrolling dirt field before the header and entries,
        // then paints the occluding bands and scrollbar after them. Keep that ordering here: the
        // bands are the CPU-draw-list equivalent of vanilla's clipping at the viewport seams.
        let list_index = self.tab as usize;
        let row_count = self.current_rows().len();
        let scroll_pixels = {
            let list = &mut self.lists[list_index];
            list.size = row_count;
            list.set_header(self.tab.has_header(), 20);
            list.draw_background(d);
            list.scroll_pixels()
        };
        self.scroll = scroll_pixels;

        let stats_icons = UiTexture::gui(STATS_ICONS_TEXTURE);
        let list_x = self.current_list().content_bounds().0 + 2;
        let header_y = self.current_list().top + 4 - scroll_pixels;
        if self.tab.has_header() {
            let glyphs = match self.tab {
                StatsTab::Blocks => [18, 36, 54],
                StatsTab::Items => [72, 18, 36],
                _ => unreachable!(),
            };
            for (column, glyph_u) in glyphs.into_iter().enumerate() {
                let x = list_x + [97, 147, 197][column];
                let pressed = self.pressed_header == Some(column);
                d.blit_sized(
                    &stats_icons,
                    x,
                    header_y + 1,
                    0.0,
                    if pressed { 0.0 } else { 18.0 },
                    18,
                    18,
                    128.0,
                    128.0,
                );
                d.blit_sized(
                    &stats_icons,
                    x + i32::from(pressed),
                    header_y + 1 + i32::from(pressed),
                    glyph_u as f32,
                    18.0,
                    18,
                    18,
                    128.0,
                    128.0,
                );
            }
            let sort = self.sort_for(self.tab);
            if let Some(column) = sort.column {
                d.blit_sized(
                    &stats_icons,
                    list_x + [79, 129, 179][column],
                    header_y + 1,
                    18.0,
                    if sort.direction == 1 { 36.0 } else { 0.0 },
                    18,
                    18,
                    128.0,
                    128.0,
                );
            }
        }

        // A `GuiSlot` calls drawSlot for entries whose visible slot rectangle overlaps the
        // viewport. The four-pixel gap is still part of hit-testing, but not of this draw pass.
        let rows: Vec<(usize, Rect, StatRow)> = self
            .current_list()
            .visible_slots()
            .into_iter()
            .filter_map(|index| {
                self.current_rows()
                    .get(index)
                    .cloned()
                    .map(|row| (index, self.row_rect(ctx, index), row))
            })
            .collect();
        for (index, rect, row) in &rows {
            let alternating = if index % 2 == 0 {
                TITLE_COLOR
            } else {
                9_474_192
            };
            match self.tab {
                StatsTab::General => {
                    draw_string(d, font, &row.name, rect.x + 2, rect.y + 1, alternating);
                    if let Some(value) = &row.columns[0] {
                        let width = font.string_width(&value.formatted);
                        draw_string(
                            d,
                            font,
                            &value.formatted,
                            rect.x + 215 - width,
                            rect.y + 1,
                            alternating,
                        );
                    }
                }
                StatsTab::Blocks | StatsTab::Items => {
                    d.blit_sized(
                        &stats_icons,
                        rect.x + 41,
                        rect.y + 1,
                        0.0,
                        0.0,
                        18,
                        18,
                        128.0,
                        128.0,
                    );
                    if let (Some(atlas), Some(icon)) = (&self.atlas_texture, &row.icon) {
                        draw_item_icon_tinted(d, atlas, rect.x + 42, rect.y + 2, icon, 0xFFFF_FFFF);
                    }
                    for (column, right) in [115, 165, 215].into_iter().enumerate() {
                        let text = row.columns[column]
                            .as_ref()
                            .map_or("-", |value| value.formatted.as_str());
                        let width = font.string_width(text);
                        draw_string(
                            d,
                            font,
                            text,
                            rect.x + right - width,
                            rect.y + 5,
                            alternating,
                        );
                    }
                }
                StatsTab::Mobs => {
                    draw_string(d, font, &row.name, rect.x - 8, rect.y + 1, TITLE_COLOR);
                    let line_height = font.font_height();
                    for (column, y) in [rect.y + 1 + line_height, rect.y + 1 + line_height * 2]
                        .into_iter()
                        .enumerate()
                    {
                        if let Some(value) = &row.columns[column] {
                            let color = if value.raw == 0 { 6_316_128 } else { 9_474_192 };
                            draw_string(d, font, &value.formatted, rect.x + 2, y, color);
                        }
                    }
                }
            }
        }

        // The overlay must be after entries: it is what hides a scrolled row's overhanging
        // pixels, supplies GuiSlot's top/bottom fades, and puts the six-pixel bar above content.
        self.lists[list_index].draw_overlay(d);

        // Java's `GuiSlot.func_148142_b` resolves hover text after the overlay and before
        // GuiStats draws its title and buttons. This ordering lets those later controls retain
        // their vanilla z-order when a tooltip reaches the title/button bands.
        if self.tab.has_header() {
            let list = self.current_list();
            let mut tooltip = None;
            if let Some(index) = list.slot_index_at(mouse.0, mouse.1) {
                if mouse.0 >= list_x + 40 && mouse.0 <= list_x + 60 {
                    tooltip = self.current_rows().get(index).map(|row| row.name.clone());
                }
            }
            let header_bottom = list.top + 4 + list.header_padding - list.scroll_pixels();
            if tooltip.is_none()
                && mouse.1 >= list.top
                && mouse.1 <= list.bottom
                && mouse.1 < header_bottom
            {
                let local_x = mouse.0 - list_x;
                let column = if (97..=115).contains(&local_x) {
                    Some(0)
                } else if (147..=165).contains(&local_x) {
                    Some(1)
                } else if (197..=215).contains(&local_x) {
                    Some(2)
                } else {
                    None
                };
                if let Some(column) = column {
                    let key = match (self.tab, column) {
                        (StatsTab::Blocks, 0) => "stat.crafted",
                        (StatsTab::Blocks, 1) => "stat.used",
                        (StatsTab::Blocks, _) => "stat.mined",
                        (StatsTab::Items, 0) => "stat.depleted",
                        (StatsTab::Items, 1) => "stat.crafted",
                        (StatsTab::Items, _) => "stat.used",
                        _ => unreachable!(),
                    };
                    let translated = lang.tr(key).trim();
                    if !translated.is_empty() {
                        tooltip = Some(translated.to_owned());
                    }
                }
            }
            if let Some(tooltip) = tooltip {
                draw_hovering_text(d, font, &[tooltip], mouse.0, mouse.1, ctx.width, ctx.height);
            }
        }

        draw_centered_string(
            d,
            font,
            lang.tr("gui.stats"),
            ctx.width / 2,
            20,
            TITLE_COLOR,
        );
        for b in &mut self.tabs {
            b.draw(d, font, mouse.0, mouse.1);
        }
        self.done.draw(d, font, mouse.0, mouse.1);
    }
}

fn stat_raw(values: &std::collections::HashMap<String, i32>, id: &str) -> i32 {
    values.get(id).copied().unwrap_or(0)
}

fn formatted_stat_value(
    values: &std::collections::HashMap<String, i32>,
    id: &'static str,
) -> StatValue {
    let raw = stat_raw(values, id);
    let info = mc_data::stat_by_id(id).expect("generated association references registered stat");
    StatValue::new(info.format.format(raw), i64::from(raw))
}

fn translated_stat_name(info: &mc_data::StatInfo, lang: &Language) -> String {
    if lang.has_key(info.translation_key) {
        lang.tr(info.translation_key).to_owned()
    } else {
        info.display_en_us.to_owned()
    }
}

fn translated_item_name(item: mc_data::ItemId, lang: &Language) -> String {
    display_name_localized(&stat_item_icon(item), lang).0
}

fn stat_item_icon(item: mc_data::ItemId) -> ItemStack {
    ItemStack {
        item_id: item.0 as i16,
        count: 1,
        damage: 0,
        nbt: None,
    }
}

/// One achievement in the tree.
#[derive(Debug, Clone, PartialEq)]
pub struct AchievementNode {
    /// The `achievement.*` translation key.
    pub key: String,
    /// The grid column vanilla places it at. java: `Achievement.displayColumn`
    pub column: i32,
    /// The grid row. java: `Achievement.displayRow`
    pub row: i32,
    /// The prerequisite's key, if any.
    pub parent: Option<String>,
    /// Whether the player has it.
    pub unlocked: bool,
    /// Whether it is a "special" (gold-bordered) achievement.
    pub special: bool,
    /// The stack drawn in the cell. java: `Achievement.theItemStack`
    pub icon: Option<ItemStack>,
}

/// Vanilla's top-right achievement notification panel.
///
/// The client owns exactly one of these. A new unlock replaces the prior notification, and the
/// inventory hint uses the same panel in permanent mode until explicitly cleared.
// java: GuiAchievement.java
#[derive(Debug, Clone, Default)]
pub struct AchievementToast {
    title: String,
    description: String,
    icon: Option<ItemStack>,
    notification_time_ms: Option<i64>,
    permanent: bool,
}

impl AchievementToast {
    /// Queue a normal three-second "Achievement Get!" notification.
    pub fn display(
        &mut self,
        title: impl Into<String>,
        description: impl Into<String>,
        icon: Option<ItemStack>,
        now_ms: i64,
    ) {
        self.title = title.into();
        self.description = description.into();
        self.icon = icon;
        self.notification_time_ms = Some(now_ms);
        self.permanent = false;
    }

    /// Queue the permanent inventory hint, whose animation begins 2.5 seconds in the future.
    pub fn display_permanent(
        &mut self,
        title: impl Into<String>,
        description: impl Into<String>,
        icon: Option<ItemStack>,
        now_ms: i64,
    ) {
        self.title = title.into();
        self.description = description.into();
        self.icon = icon;
        self.notification_time_ms = Some(now_ms.saturating_add(2_500));
        self.permanent = true;
    }

    /// Remove any active notification. java: `clearAchievements()`
    pub fn clear(&mut self) {
        self.notification_time_ms = None;
    }

    /// Whether a notification is currently queued.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.notification_time_ms.is_some()
    }

    /// Panel y coordinate at `now_ms`, including vanilla's fourth-power slide easing.
    ///
    /// A completed transient notification clears itself and returns `None`. Permanent hints park
    /// at y=0 once their animation reaches the midpoint.
    pub fn panel_y(&mut self, now_ms: i64) -> Option<i32> {
        let start = self.notification_time_ms?;
        // Java deliberately keeps the negative pre-start interval for permanent hints:
        // `Minecraft.getSystemTime() - notificationTime` is signed, so a hint scheduled at
        // now+2500 remains off-screen until that future timestamp. Keep the saturating path for
        // transient notifications (where the negative interval is an expiry condition), but use
        // a widened subtraction for the permanent path so extreme clocks cannot overflow.
        let elapsed_ms = if self.permanent {
            i128::from(now_ms) - i128::from(start)
        } else {
            i128::from(now_ms.saturating_sub(start))
        };
        let mut progress = elapsed_ms as f64 / 3_000.0;
        if !self.permanent && !(0.0..=1.0).contains(&progress) {
            self.clear();
            return None;
        }
        if self.permanent && progress > 0.5 {
            progress = 0.5;
        }

        let mut ease = progress * 2.0;
        if ease > 1.0 {
            ease = 2.0 - ease;
        }
        ease *= 4.0;
        ease = 1.0 - ease;
        if ease < 0.0 {
            ease = 0.0;
        }
        ease *= ease;
        ease *= ease;
        Some(0 - (ease * 36.0) as i32)
    }

    /// Draw over the active screen. The caller supplies wall-clock milliseconds so the animation
    /// continues while the game is paused, just like `Minecraft.getSystemTime()`.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        now_ms: i64,
    ) {
        let Some(y) = self.panel_y(now_ms) else {
            return;
        };
        let x = ctx.width - 160;
        d.blit(
            &UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE),
            x,
            y,
            96,
            202,
            160,
            32,
        );

        if self.permanent {
            font.draw_split_string(d, &self.description, x + 30, y + 7, 120, -1);
        } else {
            draw_string(d, font, &self.title, x + 30, y + 7, -256);
            draw_string(d, font, &self.description, x + 30, y + 18, -1);
        }
        if let (Some(atlas), Some(icon)) = (atlas_texture, self.icon.as_ref()) {
            draw_item_icon_tinted(d, atlas, x + 8, y + 8, icon, 0xFFFF_FFFF);
        }
    }
}

/// The achievements screen's model.
#[derive(Debug, Clone, Default)]
pub struct AchievementGrid {
    /// Every achievement the server knows about.
    pub nodes: Vec<AchievementNode>,
}

/// The pixel pitch of the achievement grid. java: `GuiAchievements` — `i1 = k * 24 - ...`
pub const GRID_PITCH: i32 = 24;

impl AchievementGrid {
    /// A grid over `nodes`.
    #[must_use]
    pub fn new(nodes: Vec<AchievementNode>) -> Self {
        Self { nodes }
    }

    /// Apply a vanilla statistics snapshot to the grid's unlock flags.
    ///
    /// `StatFileWriter.hasAchievementUnlocked` is exactly a positive-value check. Keeping the
    /// join here (rather than making each host mutate `AchievementNode::unlocked` itself) makes
    /// the rendering predicates consume the same full stat id that `S37PacketStatistics` carries
    /// and prevents a future screen consumer from silently falling back to an all-locked tree.
    pub fn apply_statistics(&mut self, values: &std::collections::HashMap<String, i32>) {
        for node in &mut self.nodes {
            node.unlocked = values.get(&node.key).is_some_and(|value| *value > 0);
        }
    }

    /// Whether an achievement is reachable: it has no parent, or its parent is unlocked.
    ///
    /// Vanilla renders unreachable achievements dimmed and hides their description behind
    /// "???"; reachable-but-locked ones show normally. Getting this wrong spoils the tree.
    // java: GuiAchievements.java:340 (`flag`/`flag1`)
    #[must_use]
    pub fn is_reachable(&self, node: &AchievementNode) -> bool {
        match &node.parent {
            None => true,
            Some(parent) => self.nodes.iter().any(|n| &n.key == parent && n.unlocked),
        }
    }

    /// Whether vanilla replaces this achievement's name with `achievement.unknown`.
    ///
    /// This is deliberately an exact-depth test. At depths one and two vanilla shows the real
    /// name with a "requires" line, at depth three it substitutes `???`, and beyond depth three
    /// it does not draw a tooltip at all.
    #[must_use]
    pub fn is_hidden(&self, node: &AchievementNode) -> bool {
        self.locked_depth(node) == 3
    }

    /// The node's position in grid pixels relative to the grid origin.
    #[must_use]
    pub fn node_pos(node: &AchievementNode) -> (i32, i32) {
        (node.column * GRID_PITCH, node.row * GRID_PITCH)
    }

    /// The node whose 22×22 cell contains a grid-space point.
    #[must_use]
    pub fn node_at(&self, gx: i32, gy: i32) -> Option<&AchievementNode> {
        self.nodes.iter().find(|n| {
            let (x, y) = Self::node_pos(n);
            Rect::new(x, y, 22, 22).contains(gx, gy)
        })
    }

    /// How many achievements are unlocked.
    #[must_use]
    pub fn unlocked_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.unlocked).count()
    }

    /// A node by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&AchievementNode> {
        self.nodes.iter().find(|n| n.key == key)
    }

    /// A node's prerequisite, if it names one that exists.
    #[must_use]
    pub fn parent_of(&self, node: &AchievementNode) -> Option<&AchievementNode> {
        node.parent.as_deref().and_then(|k| self.get(k))
    }

    /// How many *consecutive locked* prerequisites sit between this achievement and one the
    /// player has. `0` for an unlocked achievement, and `0` for a locked one whose parent is
    /// unlocked or absent. Vanilla dims a cell by this number and hides its name past 2.
    ///
    /// The walk is capped at the node count: `parent` keys arrive from the server, so a cycle is
    /// possible input, and §11 forbids letting untrusted data hang or panic the client.
    // java: StatFileWriter.java:29 (func_150874_c)
    #[must_use]
    pub fn locked_depth(&self, node: &AchievementNode) -> i32 {
        if node.unlocked {
            return 0;
        }
        let mut depth = 0;
        let mut cursor = self.parent_of(node);
        while let Some(n) = cursor {
            if n.unlocked || depth as usize >= self.nodes.len() {
                break;
            }
            depth += 1;
            cursor = self.parent_of(n);
        }
        depth
    }

    /// The smallest column any achievement occupies — **or zero**.
    ///
    /// Vanilla tracks this in a `static int` that is only ever narrowed by
    /// `Achievement`'s constructor (`if (column < minDisplayColumn)`), and a `static int` starts
    /// at `0`. So a tree entirely to the right of the origin still reports `0`, and the pan
    /// clamp lets the player scroll left of every achievement. Reproduced; pinned by
    /// `the_display_bounds_start_at_zero_the_way_javas_static_ints_do`.
    // java: Achievement.java:69 + AchievementList.java:13
    #[must_use]
    pub fn min_display_column(&self) -> i32 {
        self.nodes.iter().map(|n| n.column).fold(0, i32::min)
    }

    /// See [`AchievementGrid::min_display_column`].
    #[must_use]
    pub fn min_display_row(&self) -> i32 {
        self.nodes.iter().map(|n| n.row).fold(0, i32::min)
    }

    /// See [`AchievementGrid::min_display_column`].
    #[must_use]
    pub fn max_display_column(&self) -> i32 {
        self.nodes.iter().map(|n| n.column).fold(0, i32::max)
    }

    /// See [`AchievementGrid::min_display_column`].
    #[must_use]
    pub fn max_display_row(&self) -> i32 {
        self.nodes.iter().map(|n| n.row).fold(0, i32::max)
    }

    /// The four scroll clamps, in grid pixels: `(min_x, min_y, max_x, max_y)`.
    ///
    /// The asymmetry is vanilla's: the low clamps subtract `112` (half the 224-wide window) and
    /// the high ones subtract `77` (half the 155-tall window), so the *same* pair of constants
    /// is used for both axes even though the window is not square.
    // java: GuiAchievements.java:28 — `field_146572_y` … `field_146560_B`
    #[must_use]
    pub fn scroll_bounds(&self) -> (i32, i32, i32, i32) {
        (
            self.min_display_column() * GRID_PITCH - 112,
            self.min_display_row() * GRID_PITCH - 112,
            self.max_display_column() * GRID_PITCH - 77,
            self.max_display_row() * GRID_PITCH - 77,
        )
    }
}

/// The pannable window inside the frame, in grid pixels. java: the `224.0F`/`155.0F` the tile
/// and node loops compare against.
pub const MAP_WIDTH: i32 = 224;
/// See [`MAP_WIDTH`].
pub const MAP_HEIGHT: i32 = 155;

/// The achievement map is submitted at Java's `translate(..., -200)` z layer.  In the GUI
/// orthographic projection used by the draw list that is depth `0.6`; the map's backdrop,
/// connections, plates, and icons all share it and therefore obey the same `GL_LEQUAL` ordering
/// as vanilla when their rectangles overlap.  Keeping the layer in this screen is important:
/// the frame and tooltip are intentionally overlay passes, while a later map quad must not punch
/// through an earlier icon merely because the CPU list happened to switch textures.
const ACHIEVEMENT_MAP_DEPTH: f32 = 0.6;

/// `RenderItem.renderItemAndEffectIntoGUI` is called while the achievement matrix has already
/// translated to `z = -200` and flattened z with `scale(..., 0)`.  The shared icon helper models
/// the item call's `+100` setup translation and `+50` temporary effect z-level, so this pre-call
/// value makes its emitted depth land back on the map's `0.6` layer.
const ACHIEVEMENT_MAP_ITEM_RENDER_Z_LEVEL: f32 = -350.0;

/// Where the map's local origin sits inside the frame. java: `i1 = k + 16`, `j1 = l + 17`.
///
/// The `16`/`17` asymmetry is vanilla's and it is *not* the same inset the drag hit-test uses
/// (`k = i + 8`, `l = j + 17`) — the draggable strip starts eight pixels left of the drawn map.
pub const MAP_ORIGIN_X: i32 = 16;
/// See [`MAP_ORIGIN_X`].
pub const MAP_ORIGIN_Y: i32 = 17;

/// The eight block textures the backdrop is tiled from, indexed by [`backdrop_tile`].
///
/// Vanilla asks `getBlockModelShapes().getTexture(block.getDefaultState())` for each, which for
/// all eight of these blocks is the one texture named here.
// java: GuiAchievements.java:301 (func_175371_a)
pub const BACKDROP_TEXTURES: [&str; 8] = [
    "textures/blocks/sand",
    "textures/blocks/dirt",
    "textures/blocks/stone",
    "textures/blocks/coal_ore",
    "textures/blocks/iron_ore",
    "textures/blocks/diamond_ore",
    "textures/blocks/redstone_ore",
    "textures/blocks/bedrock",
];

/// Index into [`BACKDROP_TEXTURES`] for one 16×16 backdrop tile.
///
/// `player_id_hash` is `Session.getPlayerID().hashCode()` — the backdrop is *per account*, so
/// two players see different ore scatter on the same tree. `col`/`row` are absolute tile
/// coordinates (`k1 + i4` and `l1 + l3` in the Java).
///
/// Two things here are vanilla and look like mistakes:
///
/// * the seed is `hash + col + row * 16`, so tile `(16, 0)` and tile `(0, 1)` get the **same**
///   seed and therefore the same block — a visible diagonal repeat once the map is wider than
///   sixteen tiles;
/// * row `35` is forced to bedrock wholesale (`l1 + l3 != 35`), a single hard-coded stripe.
///
/// **Deviation, guarding untrusted input:** `nextInt(1 + row)` throws in Java for a
/// non-positive bound. `row` comes from the scroll position, which comes from the achievement
/// coordinates the server sent, so the bound is clamped to `1` here rather than allowed to
/// panic (§11).
// java: GuiAchievements.java:299
#[must_use]
pub fn backdrop_tile(player_id_hash: i32, col: i32, row: i32) -> usize {
    let seed = player_id_hash
        .wrapping_add(col)
        .wrapping_add(row.wrapping_mul(16));
    let mut rng = mc_math::JavaRandom::new(i64::from(seed));
    let roll = rng.next_i32_bound((1 + row).max(1)) + row / 2;
    if roll > 37 || row == 35 {
        return 7; // bedrock
    }
    if roll == 22 {
        if rng.next_i32_bound(2) == 0 {
            5 // diamond ore
        } else {
            6 // redstone ore
        }
    } else if roll == 10 {
        4 // iron ore
    } else if roll == 8 {
        3 // coal ore
    } else if roll > 4 {
        2 // stone
    } else if roll > 0 {
        1 // dirt
    } else {
        0 // sand — the value the variable was initialised with
    }
}

/// The map layer's coordinate frame: vanilla's
/// `translate(i1, j1, -200); scale(1 / zoom, 1 / zoom, 0)`.
///
/// Everything on the map is laid out in **grid-local** pixels and mapped to screen pixels here,
/// which is what lets the layout arithmetic below be a literal transcription of the Java while
/// still honouring the zoom that `DrawList` has no transform for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MapView {
    /// Screen pixel the local origin lands on. java: `(i1, j1)`
    pub origin: (i32, i32),
    /// Scroll in grid pixels, already interpolated, floored and clamped. java: `(i, j)`
    pub scroll: (i32, i32),
    /// java: `field_146570_r`, clamped to `1.0..=2.0`
    pub zoom: f32,
}

impl MapView {
    /// Local x to screen x.
    #[must_use]
    pub fn screen_x(&self, local: f32) -> f32 {
        self.origin.0 as f32 + local / self.zoom
    }

    /// Local y to screen y.
    #[must_use]
    pub fn screen_y(&self, local: f32) -> f32 {
        self.origin.1 as f32 + local / self.zoom
    }

    /// A `drawTexturedModalRect` in local coordinates, against a 256×256 sheet, tinted.
    #[allow(clippy::too_many_arguments)]
    fn blit(
        &self,
        d: &mut DrawList,
        tex: &UiTexture,
        x: i32,
        y: i32,
        u: i32,
        v: i32,
        w: i32,
        h: i32,
        tint: [f32; 4],
    ) {
        let s = 1.0 / GUI_TEXTURE_SIZE;
        d.quad(
            tex,
            [
                [self.screen_x(x as f32), self.screen_y(y as f32)],
                [self.screen_x(x as f32), self.screen_y((y + h) as f32)],
                [self.screen_x((x + w) as f32), self.screen_y(y as f32)],
                [self.screen_x((x + w) as f32), self.screen_y((y + h) as f32)],
            ],
            [
                [u as f32 * s, v as f32 * s],
                [u as f32 * s, (v + h) as f32 * s],
                [(u + w) as f32 * s, v as f32 * s],
                [(u + w) as f32 * s, (v + h) as f32 * s],
            ],
            [tint; 4],
        );
    }

    /// A `drawRect` in local coordinates. Corners may be given in either order, as
    /// `Gui.drawRect` normalises them.
    fn rect(&self, d: &mut DrawList, left: i32, top: i32, right: i32, bottom: i32, argb: u32) {
        let (l, r) = (left.min(right), left.max(right));
        let (t, b) = (top.min(bottom), top.max(bottom));
        d.quad_rect(
            &UiTexture::Solid,
            (
                self.screen_x(l as f32),
                self.screen_y(t as f32),
                self.screen_x(r as f32),
                self.screen_y(b as f32),
            ),
            (0.0, 0.0, 1.0, 1.0),
            argb,
        );
    }

    /// `Gui.drawHorizontalLine` in local coordinates: **inclusive** at both ends.
    fn horizontal_line(&self, d: &mut DrawList, start_x: i32, end_x: i32, y: i32, argb: u32) {
        let (a, b) = (start_x.min(end_x), start_x.max(end_x));
        self.rect(d, a, y, b + 1, y + 1, argb);
    }

    /// `Gui.drawVerticalLine` in local coordinates.
    ///
    /// **Vanilla asymmetry, reproduced:** this one is `drawRect(x, startY + 1, x + 1, endY)`,
    /// so it draws neither its first nor its last row — a vertical link is two pixels shorter
    /// than a horizontal link of the same span. That is what leaves the one-pixel notch at the
    /// corner of every elbow-shaped achievement link.
    // java: Gui.java:35
    fn vertical_line(&self, d: &mut DrawList, x: i32, start_y: i32, end_y: i32, argb: u32) {
        let (a, b) = (start_y.min(end_y), start_y.max(end_y));
        self.rect(d, x, a + 1, x + 1, b, argb);
    }
}

/// The achievements screen: vanilla's pannable parchment map.
#[derive(Debug)]
pub struct AchievementsScreen {
    /// The tree.
    pub grid: AchievementGrid,
    /// Whether the server's statistics reply is still outstanding.
    pub loading: bool,
    /// `Session.getPlayerID().hashCode()`, which seeds the backdrop. Zero is a valid value and
    /// simply produces one particular scatter.
    pub player_id_hash: i32,
    /// Display name of the current inventory binding, used by Taking Inventory's one-off
    /// `IStatStringFormat` description formatter.
    inventory_key_name: Option<String>,
    /// The eased scroll position, in grid pixels. java: `field_146567_u` / `field_146566_v`
    scroll: (f64, f64),
    /// Last tick's, for `partialTicks` interpolation. java: `field_146569_s` / `field_146568_t`
    prev_scroll: (f64, f64),
    /// The drag target the eased position chases. java: `field_146565_w` / `field_146573_x`
    target: (f64, f64),
    /// java: `field_146570_r`
    zoom: f32,
    /// Where the last drag sample was, once a drag is under way.
    // java: `field_146554_D` plus `field_146563_h` / `field_146564_i`
    drag: Option<(i32, i32)>,
    frame: Rect,
    done: Button,
}

impl AchievementsScreen {
    /// A screen over a grid, scrolled to vanilla's opening position.
    // java: GuiAchievements.java:49
    #[must_use]
    pub fn new(grid: AchievementGrid) -> Self {
        // java: `openInventory.displayColumn * 24 - 141 / 2 - 12`, `... * 24 - 141 / 2`.
        // "Taking Inventory" is the tree's root in vanilla; a server that does not send it (or
        // renames it) falls back to the first node, then to the origin.
        let root = grid.get("achievement.openInventory").or(grid.nodes.first());
        let start = (
            f64::from(root.map_or(0, |n| n.column) * GRID_PITCH - 141 / 2 - 12),
            f64::from(root.map_or(0, |n| n.row) * GRID_PITCH - 141 / 2),
        );
        Self {
            grid,
            // java: `private boolean loadingAchievements = true;`
            loading: true,
            player_id_hash: 0,
            inventory_key_name: None,
            scroll: start,
            prev_scroll: start,
            target: start,
            zoom: 1.0,
            drag: None,
            frame: Rect::default(),
            done: Button::new(1, 0, 0, "Done"),
        }
    }

    /// Lay out: a centred 256×202 frame with the Done button under it.
    // java: GuiAchievements.java:63 (initGui)
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.frame = Rect::new(
            (w - FRAME_WIDTH) / 2,
            (h - FRAME_HEIGHT) / 2,
            FRAME_WIDTH,
            FRAME_HEIGHT,
        );
        self.done = Button::sized(
            1,
            w / 2 + 24,
            h / 2 + 74,
            80,
            20,
            lang.tr("gui.done").to_owned(),
        );
    }

    /// Finish the outstanding statistics request and reveal the achievement map.
    ///
    /// java: `IProgressMeter.doneLoading()` / `GuiAchievements.doneLoading()`
    pub fn done_loading(&mut self) {
        self.loading = false;
    }

    /// Supply the current inventory binding's display name for the Taking Inventory tooltip.
    pub fn set_inventory_key_name(&mut self, name: impl Into<String>) {
        self.inventory_key_name = Some(name.into());
    }

    /// Vanilla's seven-frame `IProgressMeter.lanSearchStates` animation.
    #[must_use]
    pub fn loading_spinner(elapsed_ms: u128) -> &'static str {
        const FRAMES: [&str; 7] = [
            "oooooo", "Oooooo", "oOoooo", "ooOooo", "oooOoo", "ooooOo", "oooooO",
        ];
        FRAMES[(elapsed_ms / 150 % FRAMES.len() as u128) as usize]
    }

    /// The frame rectangle.
    #[must_use]
    pub fn frame(&self) -> Rect {
        self.frame
    }

    /// The current zoom, 1×–2×.
    #[must_use]
    pub fn zoom(&self) -> f32 {
        self.zoom
    }

    /// The eased scroll position in grid pixels.
    #[must_use]
    pub fn scroll(&self) -> (f64, f64) {
        self.scroll
    }

    /// One client tick of drag easing: move 85% of the way to the target, or all the way once
    /// the remaining distance is under two pixels.
    // java: GuiAchievements.java:212 (updateScreen)
    pub fn tick(&mut self) {
        if self.loading {
            return;
        }
        self.prev_scroll = self.scroll;
        let dx = self.target.0 - self.scroll.0;
        let dy = self.target.1 - self.scroll.1;
        if dx * dx + dy * dy < 4.0 {
            self.scroll.0 += dx;
            self.scroll.1 += dy;
        } else {
            self.scroll.0 += dx * 0.85;
            self.scroll.1 += dy * 0.85;
        }
    }

    /// Clamp `(x, y)` to the tree's scroll bounds. The upper clamps land on `bound - 1`, not
    /// `bound` — vanilla's `>=` test, reproduced.
    // java: GuiAchievements.java:171
    #[must_use]
    fn clamp_scroll(&self, x: f64, y: f64) -> (f64, f64) {
        let (min_x, min_y, max_x, max_y) = self.grid.scroll_bounds();
        let cx = if x < f64::from(min_x) {
            f64::from(min_x)
        } else if x >= f64::from(max_x) {
            f64::from(max_x - 1)
        } else {
            x
        };
        let cy = if y < f64::from(min_y) {
            f64::from(min_y)
        } else if y >= f64::from(max_y) {
            f64::from(max_y - 1)
        } else {
            y
        };
        (cx, cy)
    }

    /// Nudge the view by a grid-pixel delta, snapping every position together.
    pub fn pan_by(&mut self, dx: i32, dy: i32) {
        self.set_scroll(self.scroll.0 + f64::from(dx), self.scroll.1 + f64::from(dy));
    }

    /// Jump the view, target and interpolation origin included. **Not clamped**, exactly like
    /// every vanilla assignment to these fields — the clamp happens once per frame in
    /// [`AchievementsScreen::draw`], which is what produces the rubber-band snap-back when a
    /// drag overshoots the tree.
    pub fn set_scroll(&mut self, x: f64, y: f64) {
        self.scroll = (x, y);
        self.target = self.scroll;
        self.prev_scroll = self.scroll;
    }

    /// Set the position the eased scroll chases, leaving the live position where it is. This is
    /// the field a drag overshoot and the per-frame clamp both write.
    pub fn set_target(&mut self, x: f64, y: f64) {
        self.target = (x, y);
    }

    /// Clamp the *target* to the tree's bounds. Vanilla does this once per frame at the top of
    /// `drawScreen`, so an overshooting drag moves the live position out of bounds and the
    /// easing in [`AchievementsScreen::tick`] then pulls it back.
    // java: GuiAchievements.java:171
    pub fn clamp_target(&mut self) {
        self.target = self.clamp_scroll(self.target.0, self.target.1);
    }

    /// Drag the map. Vanilla only accepts a drag that *starts* inside a 224×155 strip whose
    /// left edge is `frame.x + 8` — eight pixels left of where the map is actually drawn — and
    /// swallows the first sample of every drag to establish the origin.
    // java: GuiAchievements.java:114
    pub fn mouse_dragged(&mut self, x: i32, y: i32) {
        let strip = Rect::new(self.frame.x + 8, self.frame.y + 17, MAP_WIDTH, MAP_HEIGHT);
        if !strip.contains(x, y) {
            return;
        }
        if let Some((last_x, last_y)) = self.drag {
            // java: `-= (double)((float)(mouseX - field_146563_h) * field_146570_r)` — the
            // delta is scaled by the zoom as a `float`, then widened, so the `f32` rounding is
            // part of the result.
            self.scroll.0 -= f64::from((x - last_x) as f32 * self.zoom);
            self.scroll.1 -= f64::from((y - last_y) as f32 * self.zoom);
            self.target = self.scroll;
            self.prev_scroll = self.scroll;
        }
        self.drag = Some((x, y));
    }

    /// End a drag. java: the `else { field_146554_D = 0; }` arm
    pub fn mouse_released(&mut self) {
        self.drag = None;
    }

    /// Wheel zoom: **scrolling down zooms out**, because LWJGL reports a downward scroll as a
    /// negative `getDWheel` and vanilla *increases* `field_146570_r` on negative — and the map
    /// is drawn at `scale(1 / field_146570_r)`, so a bigger number means a smaller map. One
    /// notch is 0.25, clamped to 1×–2×, and the scroll position absorbs half the size change so
    /// the view stays put around its centre.
    // java: GuiAchievements.java:144
    pub fn zoom_by_wheel(&mut self, wheel: i32) {
        let before = self.zoom;
        if wheel < 0 {
            self.zoom += 0.25;
        } else if wheel > 0 {
            self.zoom -= 0.25;
        }
        self.zoom = self.zoom.clamp(1.0, 2.0);
        if (self.zoom - before).abs() > f32::EPSILON {
            // java: `(f1 - f4) * 0.5F` where `f1 = zoom * 256` and `f4 = before * 256`.
            let dx =
                f64::from((self.zoom * FRAME_WIDTH as f32 - before * FRAME_WIDTH as f32) * 0.5);
            let dy =
                f64::from((self.zoom * FRAME_HEIGHT as f32 - before * FRAME_HEIGHT as f32) * 0.5);
            self.set_scroll(self.scroll.0 - dx, self.scroll.1 - dy);
        }
    }

    /// The map's coordinate frame at `partial_ticks`.
    // java: GuiAchievements.java:243
    #[must_use]
    pub fn map_view(&self, partial_ticks: f32) -> MapView {
        let pt = f64::from(partial_ticks);
        let x = self.prev_scroll.0 + (self.scroll.0 - self.prev_scroll.0) * pt;
        let y = self.prev_scroll.1 + (self.scroll.1 - self.prev_scroll.1) * pt;
        let (x, y) = self.clamp_scroll(mc_math::floor_f64(x).into(), mc_math::floor_f64(y).into());
        MapView {
            origin: (self.frame.x + MAP_ORIGIN_X, self.frame.y + MAP_ORIGIN_Y),
            scroll: (x as i32, y as i32),
            zoom: self.zoom,
        }
    }

    /// The node under a screen-space point, exactly as the draw loop picks the hovered one.
    ///
    /// **Two vanilla details, both reproduced and both pinned by name:** the hit box is
    /// `f3 >= l6 && f3 <= l6 + 22`, i.e. **23** pixels wide, one more than the 22-pixel cell;
    /// and the loop keeps assigning rather than breaking, so when boxes overlap the **last**
    /// node in list order wins. Only nodes the draw loop considers visible can be hovered.
    // java: GuiAchievements.java:478
    #[must_use]
    pub fn node_at_screen(&self, x: i32, y: i32) -> Option<&AchievementNode> {
        self.node_at_screen_at(x, y, 0.0)
    }

    /// [`AchievementsScreen::node_at_screen`] at a given `partial_ticks`.
    #[must_use]
    pub fn node_at_screen_at(
        &self,
        x: i32,
        y: i32,
        partial_ticks: f32,
    ) -> Option<&AchievementNode> {
        let view = self.map_view(partial_ticks);
        let f3 = (x - view.origin.0) as f32 * self.zoom;
        let f4 = (y - view.origin.1) as f32 * self.zoom;
        let mut hovered = None;
        for node in &self.grid.nodes {
            let (lx, ly) = (
                node.column * GRID_PITCH - view.scroll.0,
                node.row * GRID_PITCH - view.scroll.1,
            );
            if !Self::node_visible(lx, ly, self.zoom) {
                continue;
            }
            if f3 >= lx as f32
                && f3 <= (lx + 22) as f32
                && f4 >= ly as f32
                && f4 <= (ly + 22) as f32
            {
                hovered = Some(node);
            }
        }
        hovered
    }

    /// java: `l6 >= -24 && j7 >= -24 && l6 <= 224 * zoom && j7 <= 155 * zoom`
    fn node_visible(local_x: i32, local_y: i32, zoom: f32) -> bool {
        local_x >= -GRID_PITCH
            && local_y >= -GRID_PITCH
            && local_x as f32 <= MAP_WIDTH as f32 * zoom
            && local_y as f32 <= MAP_HEIGHT as f32 * zoom
    }

    /// Run one achievement-map pass under the same depth/cull window as Java's
    /// `drawAchievementScreen`. The shared draw list has no `GEQUAL` state for the initial
    /// backdrop pass, but the visible map layers all end up at the same `-200` z after vanilla's
    /// zero-z scale; using its `LEQUAL` equivalent preserves the observable overlap rule while
    /// retaining a deterministic batch contract for headless callers.
    fn with_map_layer<R>(&self, d: &mut DrawList, draw: impl FnOnce(&mut DrawList) -> R) -> R {
        d.with_item_depth(ACHIEVEMENT_MAP_DEPTH, |d| {
            d.with_render_state(UiDepth::Item, UiCull::Back, draw)
        })
    }

    /// Handle a click.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if button == 0 && !self.loading && self.done.mouse_pressed(x, y) {
            Action::Back
        } else {
            Action::None
        }
    }

    /// Close the entire GUI when the bound inventory key is pressed.
    ///
    /// Unlike the Done button, vanilla calls `displayGuiScreen(null)` here rather than returning
    /// to the parent pause screen.
    // java: GuiAchievements.keyTyped
    #[must_use]
    pub fn key_typed(&self, key: i32, inventory_key: i32) -> Action {
        if key == inventory_key {
            Action::CloseAll
        } else {
            Action::None
        }
    }

    /// Draw the tiled block backdrop.
    ///
    /// **Reproduced verbatim, including the unit mismatch:** the loop bounds step by
    /// `16 / zoom` and subtract the *unscaled* `(i + 288) % 16` remainder, while each tile is
    /// drawn at a plain 16-pixel local pitch. At 1× the two agree exactly; above 1× vanilla
    /// simply draws a few more tiles than it strictly needs.
    // java: GuiAchievements.java:292
    pub fn draw_backdrop(&self, d: &mut DrawList, atlas_texture: &UiTexture, view: &MapView) {
        self.with_map_layer(d, |d| {
            // The window is 155 tall and a tile is at least 8 screen pixels, so ~21 rows is the
            // real maximum; the cap exists only so a hostile zoom cannot spin here.
            const MAX_TILES: i32 = 64;
            let (i, j) = view.scroll;
            let base_col = (i + 288) >> 4;
            let base_row = (j + 288) >> 4;
            let off_x = (i + 288) % 16;
            let off_y = (j + 288) % 16;
            let step = 16.0 / view.zoom;

            // Java does these as two separately rounded `float` operations, so they stay that way
            // here rather than becoming a fused `mul_add`.
            let row_fits = |row: i32| row as f32 * step - (off_y as f32) < MAP_HEIGHT as f32;
            let col_fits = |col: i32| col as f32 * step - (off_x as f32) < MAP_WIDTH as f32;

            let mut row = 0;
            while row < MAX_TILES && row_fits(row) {
                // java: `f2 = 0.6F - (float)(l1 + l3) / 25.0F * 0.3F` — the map darkens with depth.
                let shade = (0.6 - (base_row + row) as f32 / 25.0 * 0.3).max(0.0);
                let mut col = 0;
                while col < MAX_TILES && col_fits(col) {
                    let tile = backdrop_tile(self.player_id_hash, base_col + col, base_row + row);
                    let loc = ResourceLocation::minecraft(BACKDROP_TEXTURES[tile]);
                    let sprite = mc_assets::sprite(&loc);
                    let (x, y) = (col * 16 - off_x, row * 16 - off_y);
                    d.quad(
                        atlas_texture,
                        [
                            [view.screen_x(x as f32), view.screen_y(y as f32)],
                            [view.screen_x(x as f32), view.screen_y((y + 16) as f32)],
                            [view.screen_x((x + 16) as f32), view.screen_y(y as f32)],
                            [
                                view.screen_x((x + 16) as f32),
                                view.screen_y((y + 16) as f32),
                            ],
                        ],
                        [
                            [sprite.uv.min_u, sprite.uv.min_v],
                            [sprite.uv.min_u, sprite.uv.max_v],
                            [sprite.uv.max_u, sprite.uv.min_v],
                            [sprite.uv.max_u, sprite.uv.max_v],
                        ],
                        [[shade, shade, shade, 1.0]; 4],
                    );
                    col += 1;
                }
                row += 1;
            }
        });
    }

    /// Draw the elbow-shaped link from every achievement to its prerequisite, plus the arrow
    /// head that points back along it.
    ///
    /// The colours are vanilla's and they read backwards until you check the Java: an
    /// **unlocked** link is grey `0xFFA0A0A0`, a **reachable** one is bright green
    /// `0xFF00FF00`, and everything else is opaque black. Links more than four locked
    /// prerequisites deep are not drawn at all.
    // java: GuiAchievements.java:348
    pub fn draw_connections(&self, d: &mut DrawList, view: &MapView) {
        self.with_map_layer(d, |d| {
            let tex = UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE);
            let white = [1.0, 1.0, 1.0, 1.0];
            for node in &self.grid.nodes {
                let Some(parent) = self.grid.parent_of(node) else {
                    continue;
                };
                if self.grid.locked_depth(node) > 4 {
                    continue;
                }
                let nx = node.column * GRID_PITCH - view.scroll.0 + 11;
                let ny = node.row * GRID_PITCH - view.scroll.1 + 11;
                let px = parent.column * GRID_PITCH - view.scroll.0 + 11;
                let py = parent.row * GRID_PITCH - view.scroll.1 + 11;
                let color = if node.unlocked {
                    0xFFA0_A0A0
                } else if self.grid.is_reachable(node) {
                    0xFF00_FF00
                } else {
                    0xFF00_0000
                };
                view.horizontal_line(d, nx, px, ny, color);
                view.vertical_line(d, px, ny, py, color);
                // The arrow head, drawn on whichever side the link leaves from. Vanilla's chain
                // is `else if`, so a purely diagonal link only ever gets the horizontal arrow.
                if nx > px {
                    view.blit(d, &tex, nx - 11 - 7, ny - 5, 114, 234, 7, 11, white);
                } else if nx < px {
                    view.blit(d, &tex, nx + 11, ny - 5, 107, 234, 7, 11, white);
                } else if ny > py {
                    view.blit(d, &tex, nx - 5, ny - 11 - 7, 96, 234, 11, 7, white);
                } else if ny < py {
                    view.blit(d, &tex, nx - 5, ny + 11, 96, 241, 11, 7, white);
                }
            }
        });
    }

    /// The cell tint for an achievement: `0.75` unlocked, `1.0` reachable, then `0.3`, `0.2`,
    /// `0.1` by locked depth. `None` means "do not draw this cell at all" — vanilla `continue`s
    /// past anything more than four prerequisites deep.
    // java: GuiAchievements.java:416
    #[must_use]
    pub fn cell_shade(&self, node: &AchievementNode) -> Option<f32> {
        if node.unlocked {
            return Some(0.75);
        }
        if self.grid.is_reachable(node) {
            return Some(1.0);
        }
        match self.grid.locked_depth(node) {
            d if d < 3 => Some(0.3),
            3 => Some(0.2),
            4 => Some(0.1),
            _ => None,
        }
    }

    /// Draw the 26×26 cell plates and the item icon in each.
    ///
    /// The plate keeps [`AchievementsScreen::cell_shade`]'s tint, but the **icon** is forced to
    /// `0.1` whenever the achievement is not reachable — vanilla issues a second
    /// `GlStateManager.color` between the two draws, so an out-of-reach cell shows a readable
    /// frame around a nearly black item. `renderItemOverlays` is deliberately not called here,
    /// which is why no stack count or durability bar appears on the map.
    // java: GuiAchievements.java:406
    pub fn draw_cells(&self, d: &mut DrawList, atlas_texture: Option<&UiTexture>, view: &MapView) {
        self.with_map_layer(d, |d| {
            let tex = UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE);
            for node in &self.grid.nodes {
                let lx = node.column * GRID_PITCH - view.scroll.0;
                let ly = node.row * GRID_PITCH - view.scroll.1;
                if !Self::node_visible(lx, ly, view.zoom) {
                    continue;
                }
                let Some(shade) = self.cell_shade(node) else {
                    continue;
                };
                let plate_u = if node.special { 26 } else { 0 };
                view.blit(
                    d,
                    &tex,
                    lx - 2,
                    ly - 2,
                    plate_u,
                    202,
                    26,
                    26,
                    [shade, shade, shade, 1.0],
                );

                let (Some(atlas), Some(stack)) = (atlas_texture, node.icon.as_ref()) else {
                    continue;
                };
                let icon_shade = if self.grid.is_reachable(node) {
                    shade
                } else {
                    0.1
                };
                // `draw_item_icon_tinted` takes a packed colour, so the shade is quantised to 8 bits
                // here where the plate above keeps its exact float. At the five values vanilla uses
                // the difference is at most 1/255 of a channel.
                let level = (icon_shade * 255.0) as u32;
                let argb = 0xFF00_0000 | (level << 16) | (level << 8) | level;
                // The icon takes integer positions, so it cannot be placed in the map's scaled space
                // by arithmetic the way the plate above is. S15's matrix stack is exactly vanilla's
                // `pushMatrix(); translate(i1, j1); scale(1 / zoom)`, so it goes through that
                // instead and the icon shrinks with the map like everything else.
                d.push_transform(
                    Transform::translate(view.origin.0 as f32, view.origin.1 as f32)
                        .then_scale(1.0 / view.zoom),
                );
                // The map's Java matrix flattens z after translating to -200. Compensate for that
                // outer transform in the shared helper's pre-call z-level so the icon stays on the
                // same depth as its plate and links (and therefore obeys last-wins ordering when
                // cells overlap).
                draw_item_icon_tinted_at_z_level(
                    d,
                    atlas,
                    lx + 3,
                    ly + 3,
                    stack,
                    argb,
                    ACHIEVEMENT_MAP_ITEM_RENDER_Z_LEVEL,
                );
                d.pop_transform();
            }
        });
    }

    /// Draw the hover tooltip: name, wrapped description, and vanilla's three locked variants.
    // java: GuiAchievements.java:497
    pub fn draw_tooltip(
        &self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        node: &AchievementNode,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        let (x, y) = (mouse.0 + 12, mouse.1 - 4);
        let reachable = self.grid.is_reachable(node);
        let depth = self.grid.locked_depth(node);
        let parent_name = self
            .grid
            .parent_of(node)
            .map_or_else(String::new, |p| lang.tr(&p.key).to_owned());
        let requires = lang.format("achievement.requires", &[&parent_name]);
        let mut name = Some(lang.tr(&node.key).to_owned());

        if reachable {
            let width = font.string_width(name.as_deref().unwrap_or("")).max(120);
            let desc = self.achievement_description(node, lang);
            let mut height = font.split_string_height(&desc, width);
            if node.unlocked {
                height += 12;
            }
            d.gradient_rect(
                x - 3,
                y - 3,
                x + width + 3,
                y + height + 3 + 12,
                0xC000_0000,
                0xC000_0000,
            );
            font.draw_split_string(d, &desc, x, y + 12, width, -6_250_336);
            if node.unlocked {
                font.draw_string_with_shadow(
                    d,
                    lang.tr("achievement.taken"),
                    x as f32,
                    (y + height + 4) as f32,
                    -7_302_913,
                );
            }
        } else if depth <= 3 {
            if depth == 3 {
                name = Some(lang.tr("achievement.unknown").to_owned());
            }
            let width = font.string_width(name.as_deref().unwrap_or("")).max(120);
            let height = font.split_string_height(&requires, width);
            d.gradient_rect(
                x - 3,
                y - 3,
                x + width + 3,
                y + height + 12 + 3,
                0xC000_0000,
                0xC000_0000,
            );
            font.draw_split_string(d, &requires, x, y + 12, width, -9_416_624);
        } else {
            name = None;
        }

        if let Some(name) = name {
            let color = if reachable {
                if node.special {
                    -128
                } else {
                    -1
                }
            } else if node.special {
                -8_355_776
            } else {
                -8_355_712
            };
            font.draw_string_with_shadow(d, &name, x as f32, y as f32, color);
        }
    }

    fn achievement_description(&self, node: &AchievementNode, lang: &Language) -> String {
        let desc_key = format!("{}.desc", node.key);
        if node.key == "achievement.openInventory" {
            if let Some(key_name) = self.inventory_key_name.as_deref() {
                return lang.format(&desc_key, &[key_name]);
            }
        }
        lang.tr(&desc_key).to_owned()
    }

    /// Draw the whole screen.
    ///
    /// `atlas_texture` is the uploaded block/item atlas — the same key S17's container screens
    /// use. Without it the backdrop and the cell icons cannot be sampled, so the map draws its
    /// links, plates, frame and text only.
    ///
    /// # Draw order, which is vanilla's and is load-bearing
    ///
    /// The map layer goes down first (backdrop, links, cells), then the 256×202 frame **over**
    /// it — that frame is what hides the tiles that overhang the window, since vanilla clips
    /// with depth tricks this port has no equivalent for. Buttons follow, then the tooltip, and
    /// the title is drawn **last of all**, over the tooltip, because `drawScreen` calls
    /// `drawTitle` after `drawAchievementScreen` returns.
    ///
    /// The map passes carry the draw list's `UiDepth::Item`/`GL_LEQUAL` equivalent at Java's
    /// `-200` layer; the initial `depthFunc(518)` (`GL_GEQUAL`) backdrop pre-pass has no direct
    /// draw-list state, but the world background is depthless in this UI path. The frame and
    /// tooltip remain overlay passes, so their ordering and overlap semantics stay explicit.
    // java: GuiAchievements.java:104 (drawScreen) + :241 (drawAchievementScreen)
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        atlas_texture: Option<&UiTexture>,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, true);
        if self.loading {
            let elapsed_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.as_millis());
            draw_centered_string(
                d,
                font,
                lang.tr("multiplayer.downloadingStats"),
                ctx.width / 2,
                ctx.height / 2,
                TITLE_COLOR,
            );
            draw_centered_string(
                d,
                font,
                Self::loading_spinner(elapsed_ms),
                ctx.width / 2,
                ctx.height / 2 + font.font_height() * 2,
                TITLE_COLOR,
            );
            return;
        }

        self.clamp_target();
        let view = self.map_view(ctx.partial_ticks);
        if let Some(atlas) = atlas_texture {
            self.draw_backdrop(d, atlas, &view);
        }
        self.draw_connections(d, &view);
        self.draw_cells(d, atlas_texture, &view);

        d.blit(
            &UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE),
            self.frame.x,
            self.frame.y,
            0,
            0,
            FRAME_WIDTH,
            FRAME_HEIGHT,
        );
        self.done.draw(d, font, mouse.0, mouse.1);

        if let Some(node) = self.node_at_screen_at(mouse.0, mouse.1, ctx.partial_ticks) {
            let node = node.clone();
            self.draw_tooltip(d, font, &node, mouse, lang);
        }

        // java: drawTitle — `drawString(..., i + 15, j + 5, 4210752)`, left-aligned and unshadowed.
        draw_string(
            d,
            font,
            lang.tr("gui.achievements"),
            self.frame.x + 15,
            self.frame.y + 5,
            4_210_752,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn rows() -> Vec<StatRow> {
        vec![
            StatRow::general(0, "Jumps", "12", 12),
            StatRow::general(1, "Distance Walked", "1.2 km", 1200),
            StatRow::general(2, "Deaths", "3", 3),
        ]
    }

    fn table_row(id: i32, name: &str, values: [Option<i64>; 3]) -> StatRow {
        StatRow::item(
            id,
            name,
            values.map(|value| value.map(|raw| StatValue::new(raw.to_string(), raw))),
            None,
        )
    }

    #[test]
    fn stat_formatters_match_the_live_vanilla_oracle() {
        let cases = [
            (100, "100", "5.0 s", "1.00 m", "10.00"),
            (1_234, "1,234", "1.03 m", "12.34 m", "123.40"),
            (100_000, "100,000", "1.39 h", "1.00 km", "10000.00"),
            (1_728_000, "1,728,000", "1.00 d", "17.28 km", "172800.00"),
            (
                630_720_000,
                "630,720,000",
                "1.00 y",
                "6307.20 km",
                "63072000.00",
            ),
            (
                i32::MAX,
                "2,147,483,647",
                "3.40 y",
                "21474.84 km",
                "214748364.70",
            ),
        ];

        for (raw, simple, time, distance, damage) in cases {
            assert_eq!(StatFormat::Simple.format(raw), simple);
            assert_eq!(StatFormat::Time.format(raw), time);
            assert_eq!(StatFormat::Distance.format(raw), distance);
            assert_eq!(StatFormat::Damage.format(raw), damage);
        }
    }

    #[test]
    fn stat_formatter_thresholds_and_raw_branches_keep_vanillas_asymmetries() {
        assert_eq!(StatFormat::Time.format(1), "0.05 s");
        assert_eq!(StatFormat::Time.format(20), "1.0 s");
        assert_eq!(StatFormat::Time.format(600), "30.0 s");
        assert_eq!(StatFormat::Time.format(620), "0.52 m");
        assert_eq!(StatFormat::Distance.format(50), "50 cm");
        assert_eq!(StatFormat::Distance.format(51), "0.51 m");
        assert_eq!(StatFormat::Distance.format(50_000), "500.00 m");
        assert_eq!(StatFormat::Distance.format(50_001), "0.50 km");
        assert_eq!(StatFormat::Simple.format(i32::MIN), "-2,147,483,648");
    }

    #[test]
    fn live_statistics_snapshot_joins_the_vanilla_registry_into_all_four_tabs() {
        let mut lang = Language::new("test");
        lang.merge_lang_str(
            "stat.leaveGame=Localized Quits\n\
             stat.workbenchInteraction=Localized Workbench Opens\n\
             tile.stone.name=Wrong Generic Stone\n\
             tile.stone.stone.name=Localized Stone\n\
             item.shovelIron.name=Localized Iron Shovel\n\
             item.emptyPotion.name=Localized Water Bottle\n\
             entity.Zombie.name=Localized Zombie\n\
             stat.entityKills=%s kills of %s\n\
             stat.entityKills.none=No kills of %s\n\
             stat.entityKilledBy=%s killed you %s times\n\
             stat.entityKilledBy.none=%s never killed you\n",
        );
        let values = std::collections::HashMap::from([
            ("stat.leaveGame".to_owned(), 1_234),
            ("stat.craftingTableInteraction".to_owned(), 7),
            ("stat.craftItem.minecraft.stone".to_owned(), 2),
            ("stat.useItem.minecraft.stone".to_owned(), 3),
            ("stat.mineBlock.minecraft.stone".to_owned(), 9),
            ("stat.breakItem.minecraft.iron_shovel".to_owned(), 4),
            ("stat.craftItem.minecraft.iron_shovel".to_owned(), 1),
            ("stat.useItem.minecraft.iron_shovel".to_owned(), 6),
            ("stat.useItem.minecraft.potion".to_owned(), 8),
            ("stat.killEntity.Zombie".to_owned(), 2),
            ("stat.unknownFromFutureVersion".to_owned(), 99),
        ]);

        let mut screen = StatsScreen::new();
        screen.load_statistics(&values, &lang);

        assert!(!screen.loading);
        let general = &screen.rows[&(StatsTab::General as u8)];
        assert_eq!(general.len(), 48, "zero-valued general rows remain visible");
        let quits = general
            .iter()
            .find(|row| row.name == "Localized Quits")
            .expect("translated leave-game row");
        assert_eq!(quits.columns[0].as_ref().unwrap().formatted, "1,234");
        assert_eq!(quits.columns[0].as_ref().unwrap().raw, 1_234);
        assert!(
            general
                .iter()
                .any(|row| row.name == "Localized Workbench Opens"),
            "the craftingTableInteraction id uses its distinct workbench translation key"
        );

        let blocks = &screen.rows[&(StatsTab::Blocks as u8)];
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].registry_id, 1);
        assert_eq!(blocks[0].name, "Localized Stone");
        assert_eq!(
            blocks[0]
                .columns
                .each_ref()
                .map(|value| value.as_ref().map(|value| value.raw)),
            [Some(2), Some(3), Some(9)],
            "block columns are crafted, used, mined"
        );
        assert_eq!(blocks[0].icon.as_ref().unwrap().item_id, 1);

        let items = &screen.rows[&(StatsTab::Items as u8)];
        assert_eq!(items.len(), 2);
        let shovel = items
            .iter()
            .find(|row| row.registry_id == 256)
            .expect("iron shovel row");
        assert_eq!(shovel.name, "Localized Iron Shovel");
        assert_eq!(
            shovel
                .columns
                .each_ref()
                .map(|value| value.as_ref().map(|value| value.raw)),
            [Some(4), Some(1), Some(6)],
            "item columns are depleted, crafted, used"
        );
        assert_eq!(shovel.icon.as_ref().unwrap().item_id, 256);
        assert_eq!(
            items
                .iter()
                .find(|row| row.registry_id == 373)
                .expect("potion row")
                .name,
            "Localized Water Bottle",
            "the tooltip uses ItemPotion.getItemStackDisplayName, not the generic item key"
        );

        let mobs = &screen.rows[&(StatsTab::Mobs as u8)];
        assert_eq!(mobs.len(), 1);
        assert_eq!(mobs[0].registry_id, 54);
        assert_eq!(mobs[0].name, "Localized Zombie");
        assert_eq!(
            mobs[0].columns[0].as_ref().unwrap().formatted,
            "2 kills of Localized Zombie"
        );
        assert_eq!(
            mobs[0].columns[1].as_ref().unwrap().formatted,
            "Localized Zombie never killed you",
            "zero uses vanilla's dedicated none phrase"
        );
    }

    /// java: a resize re-runs `GuiStats.initGui`, so the screen shows
    /// the loading text again; the next `S37`'s `doneLoading` builds new lists (no header sort,
    /// no scroll) and selects the General tab.
    #[test]
    fn a_resize_reenters_loading_and_the_next_reply_rebuilds_fresh_lists() {
        let lang = Language::new("test");
        let values = std::collections::HashMap::from([
            ("stat.mineBlock.minecraft.stone".to_owned(), 9),
            ("stat.mineBlock.minecraft.dirt".to_owned(), 4),
        ]);
        let atlas = UiTexture::gui("atlas/test.png");
        let mut screen = StatsScreen::new();
        screen.atlas_texture = Some(atlas.clone());
        screen.init(&ctx(), &lang);
        screen.load_statistics(&values, &lang);
        screen.tab = StatsTab::Blocks;
        screen.cycle_sort(2);
        assert!(!screen.loading);
        assert_ne!(screen.sort_for(StatsTab::Blocks), StatSort::default());
        assert_eq!(screen.current_rows().len(), 2);

        screen.restart_loading();
        assert!(screen.loading, "initGui sets the loading flag again");
        assert!(!screen.pauses_game());
        assert_eq!(screen.tab, StatsTab::General);
        assert!(
            screen.current_rows().is_empty(),
            "nothing but the loading text until the reply"
        );
        assert_eq!(screen.atlas_texture.as_ref(), Some(&atlas));
        screen.init(&ctx(), &lang);
        let (x, y) = (854 / 2 - 160 + 80 + 5, 480 - 52 + 5);
        assert_eq!(
            screen.mouse_clicked(x, y, 0),
            Action::None,
            "the tab buttons are gone while loading"
        );
        assert_eq!(screen.tab, StatsTab::General);

        screen.load_statistics(&values, &lang);
        assert!(!screen.loading);
        assert_eq!(
            screen.tab,
            StatsTab::General,
            "doneLoading shows the general list"
        );
        assert_eq!(
            screen.sort_for(StatsTab::Blocks),
            StatSort::default(),
            "a new GuiSlot carries no header sort"
        );
        assert_eq!(screen.rows[&(StatsTab::Blocks as u8)].len(), 2);
    }

    #[test]
    fn each_item_table_header_cycles_descending_ascending_then_registry_order() {
        let mut s = StatsScreen::new();
        assert!(s.loading);
        s.set_rows(
            StatsTab::Blocks,
            vec![
                table_row(9, "id 9", [Some(12), None, Some(4)]),
                table_row(2, "id 2", [Some(1200), Some(3), Some(4)]),
                table_row(5, "id 5", [Some(3), Some(99), Some(4)]),
            ],
        );
        s.tab = StatsTab::Blocks;
        assert!(!s.loading);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [2, 5, 9],
            "no active header uses registry id"
        );
        s.cycle_sort(0);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [2, 9, 5]
        );
        s.cycle_sort(0);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [5, 9, 2]
        );
        s.cycle_sort(0);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [2, 5, 9]
        );
    }

    #[test]
    fn equal_values_break_the_tie_by_registry_id_and_missing_values_stay_last() {
        let mut s = StatsScreen::new();
        s.set_rows(
            StatsTab::Items,
            vec![
                table_row(54, "Zombie", [Some(7), Some(1), None]),
                table_row(50, "Creeper", [Some(7), Some(1), Some(1)]),
            ],
        );
        s.tab = StatsTab::Items;
        s.cycle_sort(0);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [50, 54]
        );
        s.cycle_sort(2);
        assert_eq!(
            s.current_rows()
                .iter()
                .map(|r| r.registry_id)
                .collect::<Vec<_>>(),
            [50, 54],
            "a missing selected-column stat sorts after a present one"
        );
    }

    #[test]
    fn the_tabs_are_one_vanilla_row_and_switching_resets_the_scroll() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        assert_eq!(s.tabs.len(), 4);
        assert_eq!(
            (s.tabs[0].rect.x, s.tabs[0].rect.y),
            (854 / 2 - 160, 480 - 52)
        );
        assert_eq!(s.tabs[0].rect.width, 80);
        assert_eq!(s.tabs[1].rect.x, 854 / 2 - 80);
        assert_eq!(s.tabs[2].rect.x, 854 / 2);
        assert_eq!(s.tabs[3].rect.x, 854 / 2 + 80);
        assert_eq!(s.tabs[3].rect.y, 480 - 52);
        assert_eq!((s.done.rect.x, s.done.rect.y), (854 / 2 + 4, 480 - 28));
        s.set_rows(StatsTab::General, rows());
        s.scroll_by(100);
        assert!(!s.tabs[3].enabled, "an empty mob list is disabled");
        s.set_rows(
            StatsTab::Mobs,
            vec![StatRow::mob(
                54,
                "Zombie",
                StatValue::new("Killed 1 Zombie", 1),
                StatValue::new("Never killed by Zombie", 0),
            )],
        );
        assert!(s.tabs[3].enabled);
        let b = &s.tabs[3];
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        s.mouse_clicked(x, y, 0);
        assert_eq!(s.tab, StatsTab::Mobs);
        assert_eq!(s.scroll, 0);
        assert!(
            s.take_internal_button_press(),
            "an internal tab switch still plays gui.button.press"
        );
    }

    #[test]
    fn scrolling_clamps_to_the_content() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        s.set_rows(StatsTab::General, rows());
        // Three 20-pixel rows in a ~380-pixel viewport: nothing to scroll.
        s.scroll_by(1000);
        assert_eq!(s.scroll, 0);
        s.set_rows(
            StatsTab::General,
            (0..60)
                .map(|i| StatRow::general(i as i32, format!("s{i}"), "0", i))
                .collect(),
        );
        s.scroll_by(1000);
        assert!(s.scroll > 0);
        s.scroll_by(-10_000);
        assert_eq!(s.scroll, 0);
    }

    #[test]
    fn specialised_tabs_filter_non_positive_rows_and_disable_empty_buttons() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        s.set_rows(StatsTab::General, vec![StatRow::general(0, "Walk", "0", 0)]);
        s.set_rows(
            StatsTab::Blocks,
            vec![
                table_row(1, "all zero", [Some(0), None, Some(-1)]),
                table_row(2, "used", [None, Some(1), None]),
            ],
        );
        s.set_rows(
            StatsTab::Items,
            vec![table_row(3, "zero", [Some(0), None, None])],
        );
        s.set_rows(
            StatsTab::Mobs,
            vec![StatRow::mob(
                54,
                "Zombie",
                StatValue::new("none", 0),
                StatValue::new("none", 0),
            )],
        );

        assert_eq!(s.rows[&(StatsTab::General as u8)].len(), 1);
        assert_eq!(s.rows[&(StatsTab::Blocks as u8)].len(), 1);
        assert!(s.rows[&(StatsTab::Items as u8)].is_empty());
        assert!(s.rows[&(StatsTab::Mobs as u8)].is_empty());
        assert!(s.tabs[0].enabled && s.tabs[1].enabled);
        assert!(!s.tabs[2].enabled && !s.tabs[3].enabled);
        assert_eq!(s.done.id, 0);
        assert_eq!(
            s.tabs.iter().map(|button| button.id).collect::<Vec<_>>(),
            [1, 2, 3, 4]
        );
    }

    #[test]
    fn each_tab_uses_vanillas_row_pitch_and_item_tables_reserve_the_header() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));

        s.tab = StatsTab::General;
        assert_eq!(s.row_rect(&ctx(), 0), Rect::new(319, 36, 220, 10));
        assert_eq!(s.row_rect(&ctx(), 1).y, 46);

        s.tab = StatsTab::Blocks;
        assert_eq!(s.row_rect(&ctx(), 0), Rect::new(319, 56, 220, 20));
        assert_eq!(s.row_rect(&ctx(), 1).y, 76);

        s.tab = StatsTab::Mobs;
        assert_eq!(s.row_rect(&ctx(), 0), Rect::new(319, 36, 220, 36));
        assert_eq!(s.row_rect(&ctx(), 1).y, 72);
    }

    #[test]
    fn stats_tabs_keep_independent_gui_slot_scroll_geometry_and_chrome() {
        let mut s = StatsScreen::new();
        let mut lang = Language::new("en_US");
        lang.insert("gui.stats", "Statistics");
        s.init(&ctx(), &lang);
        s.set_rows(
            StatsTab::General,
            (0..100)
                .map(|id| StatRow::general(id, format!("stat{id}"), "1", 1))
                .collect(),
        );
        s.set_rows(
            StatsTab::Blocks,
            (0..100)
                .map(|id| table_row(id, "Stone", [Some(1), None, None]))
                .collect(),
        );

        assert_eq!(s.lists[StatsTab::General as usize].header_padding, 0);
        assert_eq!(s.lists[StatsTab::General as usize].slot_height, 10);
        assert_eq!(s.lists[StatsTab::Blocks as usize].header_padding, 20);
        assert_eq!(s.lists[StatsTab::Blocks as usize].slot_height, 20);
        assert_eq!(s.lists[StatsTab::Mobs as usize].slot_height, 36);

        // Each GuiSlot retains its own amountScrolled when displaySlot switches.
        s.tab = StatsTab::General;
        s.scroll_by(17);
        assert_eq!(s.scroll, 17);
        assert_eq!(s.row_rect(&ctx(), 0).y, 19);
        s.tab = StatsTab::Blocks;
        assert_eq!(s.row_rect(&ctx(), 0).y, 56);
        s.scroll_by(25);
        assert_eq!(s.row_rect(&ctx(), 0).y, 31);
        s.tab = StatsTab::General;
        assert_eq!(
            s.row_rect(&ctx(), 0).y,
            19,
            "switching tabs restores General's scroll"
        );

        // The shared GuiSlot arithmetic puts the 6-pixel bar at width/2 + 124 and computes a
        // non-minimum thumb from the content height. This is a geometry-only assertion: no GPU
        // or texture upload is involved.
        let bar = s.scrollbar();
        assert!(
            bar.is_some(),
            "the overflowing General list needs a scrollbar"
        );
        let bar = bar.unwrap();
        assert_eq!(bar.x, 854 / 2 + 124);
        assert_eq!((bar.track_top, bar.track_bottom), (32, 416));
        assert_eq!(bar.thumb_height, 384 * 384 / 1000);
    }

    #[test]
    fn stats_wheel_and_bar_drag_use_the_active_tab_slot_and_viewport_gate() {
        let mut s = StatsScreen::new();
        let lang = Language::new("en_US");
        s.init(&ctx(), &lang);
        s.set_rows(
            StatsTab::General,
            (0..100)
                .map(|id| StatRow::general(id, format!("stat{id}"), "1", 1))
                .collect(),
        );
        s.set_rows(
            StatsTab::Blocks,
            (0..100)
                .map(|id| table_row(id, "Stone", [Some(1), None, None]))
                .collect(),
        );

        s.tab = StatsTab::General;
        s.mouse_wheel(200, 100, -1.0);
        assert_eq!(s.scroll, 5, "General uses slotHeight / 2");
        s.mouse_wheel(200, 20, -1.0);
        assert_eq!(s.scroll, 5, "title-area wheel input is ignored");

        s.tab = StatsTab::Blocks;
        s.mouse_wheel(200, 100, -1.0);
        assert_eq!(s.scroll, 10, "table tabs use their 20-pixel slot height");
        let bar = s.scrollbar().expect("overflowing block table");
        let before_drag = s.row_rect(&ctx(), 0).y;
        assert!(s.mouse_clicked(bar.x + 2, 100, 0) == Action::None);
        s.mouse_dragged(bar.x + 2, 140);
        assert!(
            s.row_rect(&ctx(), 0).y < before_drag,
            "dragging the scrollbar down advances the content"
        );
        s.mouse_released();

        let bar = s.scrollbar().expect("bar remains live");
        let before_outside = s.row_rect(&ctx(), 0).y;
        s.mouse_clicked(bar.x + 2, 140, 0);
        s.mouse_dragged(-1, 220);
        assert_eq!(
            s.row_rect(&ctx(), 0).y,
            before_outside,
            "a drag sample outside GuiSlot's viewport is ignored"
        );
        s.mouse_released();

        // A click in the scrollbar column is a drag, never a row click/header sort. The public
        // scrollbar geometry and row origin remain coherent after the gesture.
        assert_eq!(s.sort_for(StatsTab::Blocks), StatSort::default());
    }

    #[test]
    fn stats_rows_are_occluded_by_the_gui_slot_seams_and_scrolling_moves_the_header() {
        let mut s = StatsScreen::new();
        let mut lang = Language::new("en_US");
        lang.insert("gui.stats", "Statistics");
        s.init(&ctx(), &lang);
        s.set_rows(
            StatsTab::Blocks,
            (0..100)
                .map(|id| table_row(id, "Stone", [Some(1), None, None]))
                .collect(),
        );
        s.tab = StatsTab::Blocks;

        // Scroll just far enough for the header origin to move. A stale fixed-at-36 header would
        // put this background at y=37 instead of y=32.
        s.scroll_by(5);
        let mut d = DrawList::new();
        let mut font = uniform_font();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang);
        assert!(
            d.vertices()
                .iter()
                .any(|vertex| vertex.pos == [416.0, 32.0] && vertex.uv == [0.0, 18.0 / 128.0]),
            "the header follows the list scroll"
        );
        assert!(!d
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [416.0, 37.0] && vertex.uv == [0.0, 18.0 / 128.0]));

        // At a larger offset the first plate intentionally overhangs the viewport. GuiSlot's
        // top gradient is submitted after the row, which is the draw-list equivalent of its GL
        // occlusion/clipping pass.
        s.scroll_by(22);
        let mut clipped = DrawList::new();
        s.draw(&mut clipped, &mut font, &ctx(), (0, 0), &lang);
        let row_batch = clipped
            .batches()
            .iter()
            .enumerate()
            .find(|(_, batch)| {
                batch
                    .range
                    .clone()
                    .any(|index| clipped.vertices()[index].pos == [360.0, 30.0])
            })
            .map(|(index, _)| index)
            .expect("the overhanging first row plate");
        let top_overlay_batch = clipped
            .batches()
            .iter()
            .enumerate()
            .find(|(_, batch)| {
                matches!(batch.texture, UiTexture::Solid)
                    && batch
                        .range
                        .clone()
                        .any(|index| clipped.vertices()[index].pos == [0.0, 32.0])
            })
            .map(|(index, _)| index)
            .expect("the top four-pixel fade");
        assert!(
            top_overlay_batch > row_batch,
            "viewport overlay must be submitted after rows so it hides overhangs"
        );
    }

    #[test]
    fn clicking_the_three_exact_header_windows_changes_only_that_tabs_sort() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        s.set_rows(
            StatsTab::Blocks,
            vec![table_row(1, "stone", [Some(1), None, None])],
        );
        s.set_rows(
            StatsTab::Items,
            vec![table_row(2, "stick", [Some(1), None, None])],
        );
        s.tab = StatsTab::Blocks;
        let hit_x = ctx().width / 2 - 110;

        s.mouse_clicked(hit_x + 79, 36, 0);
        assert_eq!(
            s.sort_for(StatsTab::Blocks),
            StatSort {
                column: Some(0),
                direction: -1
            }
        );
        assert_eq!(s.sort_for(StatsTab::Items), StatSort::default());
        assert!(s.take_internal_button_press());
        assert!(
            !s.take_internal_button_press(),
            "the sound intent drains once"
        );
        s.mouse_released();

        s.mouse_clicked(hit_x + 79, 36, 0);
        assert_eq!(s.sort_for(StatsTab::Blocks).direction, 1);
        s.mouse_clicked(hit_x + 79, 36, 0);
        assert_eq!(s.sort_for(StatsTab::Blocks), StatSort::default());

        s.mouse_clicked(hit_x + 78, 36, 0);
        assert_eq!(
            s.sort_for(StatsTab::Blocks),
            StatSort::default(),
            "left neighbour misses"
        );
        s.mouse_clicked(hit_x + 115, 36, 0);
        assert_eq!(
            s.sort_for(StatsTab::Blocks),
            StatSort::default(),
            "exclusive right neighbour misses"
        );

        s.mouse_clicked(hit_x + 129, 55, 0);
        assert_eq!(s.sort_for(StatsTab::Blocks).column, Some(1));
        s.mouse_clicked(hit_x + 179, 56, 0);
        assert_ne!(
            s.sort_for(StatsTab::Blocks).column,
            Some(2),
            "the header ends before y=56"
        );
    }

    #[test]
    fn a_pressed_header_uses_the_depressed_sprites_until_mouse_release() {
        let mut s = StatsScreen::new();
        let mut lang = Language::new("en_US");
        lang.insert("gui.stats", "Statistics");
        s.init(&ctx(), &lang);
        s.set_rows(
            StatsTab::Blocks,
            vec![table_row(1, "Stone", [Some(1), None, None])],
        );
        s.tab = StatsTab::Blocks;
        let hit_x = ctx().width / 2 - 110;
        s.mouse_clicked(hit_x + 79, 36, 0);

        let mut pressed = DrawList::new();
        s.draw(&mut pressed, &mut uniform_font(), &ctx(), (0, 0), &lang);
        assert!(pressed
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [416.0, 37.0] && vertex.uv == [0.0, 0.0]));
        assert!(pressed.vertices().iter().any(|vertex| {
            vertex.pos == [417.0, 38.0] && vertex.uv == [18.0 / 128.0, 18.0 / 128.0]
        }));

        s.mouse_released();
        let mut released = DrawList::new();
        s.draw(&mut released, &mut uniform_font(), &ctx(), (0, 0), &lang);
        assert!(released
            .vertices()
            .iter()
            .any(|vertex| { vertex.pos == [416.0, 37.0] && vertex.uv == [0.0, 18.0 / 128.0] }));
        assert!(!released
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [417.0, 38.0]));
    }

    #[test]
    fn statistics_pause_only_after_the_server_reply_finishes_loading() {
        let mut s = StatsScreen::new();
        assert!(!s.pauses_game(), "loading must leave the world ticking");
        s.set_rows(StatsTab::General, rows());
        assert!(s.pauses_game(), "the loaded statistics screen pauses");
    }

    #[test]
    fn item_tables_draw_the_128_pixel_header_sheet_and_row_plate_at_java_coordinates() {
        let mut s = StatsScreen::new();
        let mut lang = Language::new("en_US");
        lang.insert("gui.stats", "Statistics");
        s.init(&ctx(), &lang);
        s.set_rows(
            StatsTab::Blocks,
            vec![table_row(1, "Stone", [Some(1), None, None])],
        );
        s.tab = StatsTab::Blocks;
        let mut d = DrawList::new();
        let mut font = uniform_font();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang);

        let stats_batch = d
            .batches()
            .iter()
            .find(|batch| {
                matches!(
                    &batch.texture,
                    UiTexture::Resource(location) if location.path == STATS_ICONS_TEXTURE
                )
            })
            .expect("stats-icons header batch");
        let vertices = &d.vertices()[stats_batch.range.clone()];
        assert!(vertices
            .iter()
            .any(|vertex| { vertex.pos == [416.0, 37.0] && vertex.uv == [0.0, 18.0 / 128.0] }));
        assert!(d
            .vertices()
            .iter()
            .any(|vertex| vertex.pos == [360.0, 57.0]));
    }

    #[test]
    fn stats_item_hover_uses_the_single_shared_tooltip_box_renderer() {
        let mut screen = StatsScreen::new();
        let mut language = Language::new("en_US");
        language.insert("gui.stats", "Statistics");
        screen.init(&ctx(), &language);
        screen.set_rows(
            StatsTab::Blocks,
            vec![table_row(1, "Stone", [Some(1), None, None])],
        );
        screen.tab = StatsTab::Blocks;
        let row = screen.row_rect(&ctx(), 0);
        let mouse = (row.x + 45, row.y + 5);
        let mut draw = DrawList::new();
        let mut font = uniform_font();
        screen.draw(&mut draw, &mut font, &ctx(), mouse, &language);

        let expected = crate::screens::container::tooltip_box::layout(
            &font,
            &["Stone".to_owned()],
            mouse.0,
            mouse.1,
            ctx().width,
            ctx().height,
        )
        .expect("one-line tooltip");
        // The list's post-entry fades are also solid quads and vanilla draws the tooltip before
        // the title/buttons. DrawList intentionally merges adjacent Solid batches, so identify
        // the panel by its canonical first vertex rather than assuming a batch boundary.
        let panel_start = draw
            .vertices()
            .iter()
            .position(|vertex| vertex.pos == [(expected.x - 3) as f32, (expected.y - 4) as f32])
            .expect("shared tooltip panel");
        assert_eq!(
            draw.vertices()[panel_start].pos,
            [(expected.x - 3) as f32, (expected.y - 4) as f32],
            "the first canonical top-strip vertex"
        );
        assert_eq!(
            draw.vertices()[panel_start].color,
            crate::widget::draw::unpack_argb(crate::screens::container::tooltip_box::BACKGROUND,)
        );
    }

    #[test]
    fn a_loading_screen_shows_the_downloading_title_and_no_rows() {
        let mut s = StatsScreen::new();
        let mut lang = Language::new("en_US");
        lang.insert("multiplayer.downloadingStats", "Downloading statistics...");
        lang.insert("gui.stats", "Statistics");
        s.init(&ctx(), &lang);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx(), (0, 0), &lang);
        assert!(
            !d.batches().iter().any(|batch| matches!(
                &batch.texture,
                UiTexture::Resource(location)
                    if location.path == crate::widget::WIDGETS_TEXTURE
            )),
            "vanilla has no tab or Done buttons until statistics finish loading"
        );
        let glyphs = |d: &DrawList| {
            d.batches()
                .iter()
                .filter(|b| matches!(b.texture, crate::widget::UiTexture::Font(_)))
                .map(|b| b.range.len())
                .sum::<usize>()
        };
        let before = glyphs(&d);
        s.set_rows(StatsTab::General, rows());
        let mut d2 = DrawList::new();
        s.draw(&mut d2, &mut font, &ctx(), (0, 0), &lang);
        assert!(glyphs(&d2) > before, "rows appear once loaded");
        assert!(d2.batches().iter().any(|batch| matches!(
            &batch.texture,
            UiTexture::Resource(location) if location.path == crate::widget::WIDGETS_TEXTURE
        )));
    }

    #[test]
    fn done_leaves_the_statistics_screen() {
        let mut s = StatsScreen::new();
        s.init(&ctx(), &Language::new("en_US"));
        let (x, y) = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(x, y, 0),
            Action::None,
            "vanilla creates no buttons before the S37 reply"
        );
        s.set_rows(StatsTab::General, rows());
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Back);
    }

    fn node(key: &str, column: i32, row: i32, parent: Option<&str>) -> AchievementNode {
        AchievementNode {
            key: key.into(),
            column,
            row,
            parent: parent.map(ToOwned::to_owned),
            unlocked: false,
            special: false,
            icon: None,
        }
    }

    fn test_icon() -> ItemStack {
        let item = mc_data::item_by_name("chest").expect("chest item");
        ItemStack {
            item_id: item.0 as i16,
            count: 1,
            damage: 0,
            nbt: None,
        }
    }

    fn tree() -> AchievementGrid {
        let mut root = node("achievement.openInventory", 0, 0, None);
        root.unlocked = true;
        AchievementGrid::new(vec![
            root,
            node(
                "achievement.mineWood",
                2,
                1,
                Some("achievement.openInventory"),
            ),
            node(
                "achievement.buildWorkBench",
                4,
                2,
                Some("achievement.mineWood"),
            ),
        ])
    }

    fn real_achievement_grid() -> AchievementGrid {
        AchievementGrid::new(
            mc_data::achievements()
                .iter()
                .map(|info| AchievementNode {
                    key: info.stat_id.to_owned(),
                    column: i32::from(info.display_column),
                    row: i32::from(info.display_row),
                    parent: info.parent.map(|parent| parent.info().stat_id.to_owned()),
                    unlocked: false,
                    special: info.special,
                    icon: None,
                })
                .collect(),
        )
    }

    /// `nodes` plus one far-off achievement, so the scroll clamps leave grid `(0, 0)` reachable.
    ///
    /// Without it the bounds of a three-node tree are so tight that `set_scroll(0, 0)` is
    /// immediately clamped to `max * 24 - 78` — correct behaviour, and the reason the
    /// positional tests below cannot simply assume the scroll they asked for.
    fn roomy(mut nodes: Vec<AchievementNode>) -> AchievementGrid {
        nodes.push(node("spacer", 40, 40, None));
        AchievementGrid::new(nodes)
    }

    /// Reachability and name hiding are separate vanilla rules: a locked prerequisite makes a
    /// node unreachable, but only exactly three consecutive locked prerequisites hide its name.
    #[test]
    fn only_achievements_whose_parent_is_unlocked_are_reachable() {
        let g = tree();
        assert!(g.is_reachable(&g.nodes[0]), "a root is always reachable");
        assert!(g.is_reachable(&g.nodes[1]), "its parent is unlocked");
        assert!(!g.is_reachable(&g.nodes[2]), "its parent is not");
        assert!(
            !g.is_hidden(&g.nodes[1]),
            "reachable but locked is still named"
        );
        assert!(
            !g.is_hidden(&g.nodes[2]),
            "one locked prerequisite still shows the real name"
        );
        assert_eq!(g.unlocked_count(), 1);
    }

    #[test]
    fn only_exactly_three_locked_prerequisites_use_the_unknown_name() {
        let mut nodes = tree().nodes;
        nodes.push(node(
            "achievement.buildPickaxe",
            5,
            3,
            Some("achievement.buildWorkBench"),
        ));
        nodes.push(node(
            "achievement.buildFurnace",
            6,
            4,
            Some("achievement.buildPickaxe"),
        ));
        nodes.push(node(
            "achievement.acquireIron",
            7,
            5,
            Some("achievement.buildFurnace"),
        ));
        let g = AchievementGrid::new(nodes);

        assert_eq!(g.locked_depth(&g.nodes[4]), 3);
        assert!(g.is_hidden(&g.nodes[4]));
        assert_eq!(g.locked_depth(&g.nodes[5]), 4);
        assert!(
            !g.is_hidden(&g.nodes[5]),
            "depth four has no tooltip, so it must not advertise the depth-three substitution"
        );
    }

    #[test]
    fn an_unlocked_achievement_is_never_hidden_even_with_a_locked_parent() {
        let mut g = tree();
        g.nodes[2].unlocked = true;
        assert!(!g.is_hidden(&g.nodes[2]));
    }

    #[test]
    fn statistics_snapshot_drives_real_unlocks_parent_reachability_and_visibility() {
        // This is the negative control for the old all-locked builder: with only the root in the
        // snapshot, the real generated tree must expose exactly the one-step reachable child,
        // shade consecutive locked ancestors, and stop drawing after depth four.
        let mut root_only = real_achievement_grid();
        let mut values = std::collections::HashMap::new();
        values.insert("achievement.openInventory".to_owned(), 1);
        values.insert("achievement.mineWood".to_owned(), 0);
        values.insert("achievement.buildWorkBench".to_owned(), -1);
        root_only.apply_statistics(&values);
        assert_eq!(root_only.unlocked_count(), 1);
        let root_screen = AchievementsScreen::new(root_only);
        let root = root_screen
            .grid
            .get("achievement.openInventory")
            .expect("generated root");
        let mine_wood = root_screen
            .grid
            .get("achievement.mineWood")
            .expect("generated child");
        let build_workbench = root_screen
            .grid
            .get("achievement.buildWorkBench")
            .expect("generated grandchild");
        let build_pickaxe = root_screen
            .grid
            .get("achievement.buildPickaxe")
            .expect("generated depth-two child");
        let build_furnace = root_screen
            .grid
            .get("achievement.buildFurnace")
            .expect("generated depth-three child");
        let full_beacon = root_screen
            .grid
            .get("achievement.fullBeacon")
            .expect("generated deep child");
        assert!(root.unlocked);
        assert_eq!(root_screen.cell_shade(root), Some(0.75));
        assert!(root_screen.grid.is_reachable(mine_wood));
        assert_eq!(root_screen.cell_shade(mine_wood), Some(1.0));
        assert!(!root_screen.grid.is_reachable(build_workbench));
        assert_eq!(root_screen.grid.locked_depth(build_workbench), 1);
        assert_eq!(root_screen.cell_shade(build_workbench), Some(0.3));
        assert_eq!(root_screen.grid.locked_depth(build_pickaxe), 2);
        assert_eq!(root_screen.cell_shade(build_pickaxe), Some(0.3));
        assert_eq!(root_screen.grid.locked_depth(build_furnace), 3);
        assert_eq!(root_screen.cell_shade(build_furnace), Some(0.2));
        assert!(root_screen.grid.locked_depth(full_beacon) > 4);
        assert_eq!(root_screen.cell_shade(full_beacon), None);

        // Positive values are the exact `StatFileWriter.hasAchievementUnlocked` predicate. Once
        // the server reports every achievement, all 34 cells and all 33 parent links become
        // renderable; this bites a regression that silently hardcodes every node to locked.
        let mut complete = real_achievement_grid();
        let values = mc_data::achievements()
            .iter()
            .map(|info| (info.stat_id.to_owned(), 1))
            .collect::<std::collections::HashMap<_, _>>();
        complete.apply_statistics(&values);
        assert_eq!(complete.unlocked_count(), mc_data::achievements().len());
        let complete_screen = AchievementsScreen::new(complete);
        assert!(complete_screen
            .grid
            .nodes
            .iter()
            .all(|node| complete_screen.cell_shade(node) == Some(0.75)));
        for node in &complete_screen.grid.nodes {
            let mut cell = DrawList::new();
            complete_screen.draw_cells(
                &mut cell,
                None,
                &MapView {
                    origin: (0, 0),
                    scroll: (node.column * GRID_PITCH, node.row * GRID_PITCH),
                    zoom: 1.0,
                },
            );
            assert!(
                !cell.is_empty(),
                "unlocked {} must render when centered",
                node.key
            );
        }
        let mut links = DrawList::new();
        complete_screen.draw_connections(
            &mut links,
            &MapView {
                origin: (0, 0),
                scroll: (0, 0),
                zoom: 1.0,
            },
        );
        assert_eq!(
            links.vertices().len() / 6,
            (mc_data::achievements().len() - 1) * 3,
            "every generated non-root parent link remains visible after unlock"
        );
    }

    #[test]
    fn nodes_sit_on_a_twenty_four_pixel_grid_and_hit_test_a_twenty_two_pixel_cell() {
        let g = tree();
        assert_eq!(AchievementGrid::node_pos(&g.nodes[0]), (0, 0));
        assert_eq!(AchievementGrid::node_pos(&g.nodes[1]), (48, 24));
        assert!(g.node_at(0, 0).is_some());
        assert!(g.node_at(21, 21).is_some());
        // The two-pixel gutter between cells belongs to neither.
        assert!(g.node_at(22, 0).is_none());
        assert_eq!(
            g.node_at(50, 26).map(|n| n.key.as_str()),
            Some("achievement.mineWood")
        );
    }

    #[test]
    fn the_frame_is_centred_and_panning_moves_the_hit_test_with_it() {
        let mut s = AchievementsScreen::new(roomy(tree().nodes));
        s.init(&ctx(), &Language::new("en_US"));
        let f = s.frame();
        assert_eq!(f.width, FRAME_WIDTH);
        assert_eq!(f.height, FRAME_HEIGHT);
        assert_eq!(f.x, (854 - FRAME_WIDTH) / 2);
        assert_eq!(f.y, (480 - FRAME_HEIGHT) / 2);
        // Scroll to the origin so grid pixel (0, 0) is the map's own origin, `(x + 16, y + 17)`.
        s.set_scroll(0.0, 0.0);
        let (px, py) = (f.x + MAP_ORIGIN_X, f.y + MAP_ORIGIN_Y);
        assert_eq!(
            s.node_at_screen(px, py).map(|n| n.key.as_str()),
            Some("achievement.openInventory")
        );
        s.pan_by(48, 24);
        assert_eq!(
            s.node_at_screen(px, py).map(|n| n.key.as_str()),
            Some("achievement.mineWood"),
            "panning brings the next node under the cursor"
        );
        // Far off the map, nothing.
        assert!(s.node_at_screen(0, 0).is_none());
    }

    #[test]
    fn the_achievements_screen_draws_its_frame_and_leaves_on_done() {
        let mut lang = Language::new("en_US");
        lang.insert("gui.achievements", "Achievements");
        lang.insert("gui.done", "Done");
        lang.insert("achievement.openInventory", "Taking Inventory");
        let mut s = AchievementsScreen::new(tree());
        assert!(s.loading, "vanilla waits for S37 before revealing the map");
        s.done_loading();
        s.init(&ctx(), &lang);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, None, &ctx(), (0, 0), &lang);
        assert!(d.batches().iter().any(|b| matches!(&b.texture,
            UiTexture::Resource(r) if r.path == ACHIEVEMENT_BACKGROUND_TEXTURE)));
        let (x, y) = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Back);
    }

    #[test]
    fn the_inventory_binding_closes_achievements_to_the_game_not_the_parent() {
        let s = AchievementsScreen::new(tree());
        assert_eq!(s.key_typed(18, 18), Action::CloseAll);
        assert_eq!(s.key_typed(1, 18), Action::None);
    }

    // ------------------------------------------------------- the map's model

    /// `AchievementList`'s bounds live in `static int`s that the `Achievement` constructor only
    /// ever *narrows*, and a `static int` starts at zero — so a tree entirely in the positive
    /// quadrant still reports a minimum of zero and the player can scroll left of all of it.
    #[test]
    fn the_display_bounds_start_at_zero_the_way_javas_static_ints_do() {
        let g = AchievementGrid::new(vec![node("a", 3, 4, None), node("b", 7, 9, None)]);
        assert_eq!(g.min_display_column(), 0, "not 3");
        assert_eq!(g.min_display_row(), 0, "not 4");
        assert_eq!(g.max_display_column(), 7);
        assert_eq!(g.max_display_row(), 9);
        // And a tree entirely in the negative quadrant reports a maximum of zero.
        let neg = AchievementGrid::new(vec![node("a", -3, -4, None)]);
        assert_eq!(neg.max_display_column(), 0);
        assert_eq!(neg.max_display_row(), 0);
    }

    /// Hand-derived from `AchievementList.minDisplayColumn * 24 - 112` and the three siblings:
    /// the low clamps subtract 112 (half the 224-wide window) and the high ones 77 (half the
    /// 155-tall one) — on *both* axes, even though the window is not square.
    #[test]
    fn the_scroll_clamps_use_vanillas_asymmetric_one_twelve_and_seventy_seven() {
        let g = AchievementGrid::new(vec![node("a", -2, -1, None), node("b", 5, 6, None)]);
        assert_eq!(
            g.scroll_bounds(),
            (-2 * 24 - 112, -24 - 112, 5 * 24 - 77, 6 * 24 - 77)
        );
        assert_eq!(g.scroll_bounds(), (-160, -136, 43, 67));
    }

    /// java: `openInventory.displayColumn * 24 - 141 / 2 - 12`.
    #[test]
    fn the_opening_scroll_centres_on_taking_inventory() {
        let mut nodes = tree().nodes;
        nodes[0].column = 3;
        nodes[0].row = 2;
        let s = AchievementsScreen::new(AchievementGrid::new(nodes));
        assert_eq!(
            s.scroll(),
            (f64::from(3 * 24 - 70 - 12), f64::from(2 * 24 - 70))
        );
    }

    #[test]
    fn locked_depth_counts_consecutive_locked_prerequisites() {
        let g = tree();
        assert_eq!(g.locked_depth(&g.nodes[0]), 0, "unlocked is always zero");
        assert_eq!(g.locked_depth(&g.nodes[1]), 0, "its parent is unlocked");
        assert_eq!(g.locked_depth(&g.nodes[2]), 1, "one locked prerequisite");
    }

    /// `parent` keys come off the wire, so a cycle is possible input. The walk must terminate.
    #[test]
    fn a_cyclic_parent_chain_from_the_server_terminates_instead_of_hanging() {
        let g = AchievementGrid::new(vec![node("a", 0, 0, Some("b")), node("b", 1, 0, Some("a"))]);
        assert!(g.locked_depth(&g.nodes[0]) <= 2);
        assert!(!g.is_reachable(&g.nodes[0]));
    }

    /// The current position races ahead 85% per tick until it is within two pixels, then snaps.
    // java: GuiAchievements.updateScreen
    #[test]
    fn scroll_easing_covers_eighty_five_percent_then_snaps_inside_two_pixels() {
        let mut s = AchievementsScreen::new(tree());
        s.done_loading();
        s.set_scroll(0.0, 0.0);
        s.target = (100.0, 0.0);
        s.tick();
        assert!((s.scroll().0 - 85.0).abs() < 1e-9);
        s.tick();
        assert!((s.scroll().0 - 97.75).abs() < 1e-9);
        // dx is now 2.25, and 2.25² = 5.06 ≥ 4, so still eased.
        s.tick();
        assert!(s.scroll().0 < 100.0);
        // Once within two pixels it lands exactly.
        s.target = (s.scroll().0 + 1.0, 0.0);
        let want = s.target.0;
        s.tick();
        assert!((s.scroll().0 - want).abs() < 1e-9);
    }

    /// Vanilla's first drag sample only records the origin; the second is the first that moves
    /// anything, and it moves the map the *opposite* way to the mouse.
    #[test]
    fn dragging_moves_the_map_the_opposite_way_and_the_first_sample_is_swallowed() {
        let mut s = AchievementsScreen::new(tree());
        s.init(&ctx(), &Language::new("en_US"));
        s.set_scroll(0.0, 0.0);
        let f = s.frame();
        let (x, y) = (f.x + 100, f.y + 100);
        s.mouse_dragged(x, y);
        assert_eq!(
            s.scroll(),
            (0.0, 0.0),
            "the first sample only sets the origin"
        );
        s.mouse_dragged(x + 10, y + 4);
        assert_eq!(s.scroll(), (-10.0, -4.0));
        s.mouse_released();
        s.mouse_dragged(x + 30, y + 30);
        assert_eq!(s.scroll(), (-10.0, -4.0), "a fresh drag re-samples first");
    }

    /// The strip you can grab starts eight pixels left of where the map is drawn (`i + 8` vs
    /// `i + 16`) and is exactly 224×155.
    #[test]
    fn the_drag_strip_starts_eight_pixels_left_of_the_drawn_map() {
        let mut s = AchievementsScreen::new(tree());
        s.init(&ctx(), &Language::new("en_US"));
        let f = s.frame();
        s.mouse_dragged(f.x + 8, f.y + 17);
        assert!(s.drag.is_some(), "the strip's own top-left corner grabs");
        s.mouse_released();
        s.mouse_dragged(f.x + 7, f.y + 17);
        assert!(s.drag.is_none(), "one pixel further left does not");
        s.mouse_dragged(f.x + 8 + 224, f.y + 17);
        assert!(s.drag.is_none(), "and the right edge is exclusive");
    }

    #[test]
    fn the_wheel_zooms_out_when_it_scrolls_down_and_clamps_between_one_and_two() {
        let mut s = AchievementsScreen::new(tree());
        assert_eq!(s.zoom(), 1.0);
        s.zoom_by_wheel(1);
        assert_eq!(
            s.zoom(),
            1.0,
            "already at the minimum; a scroll up does nothing"
        );
        s.zoom_by_wheel(-1);
        assert_eq!(s.zoom(), 1.25);
        for _ in 0..10 {
            s.zoom_by_wheel(-1);
        }
        assert_eq!(s.zoom(), 2.0, "clamped at 2x");
        // The scroll absorbs half the size change so the view stays put.
        let before = s.scroll();
        s.zoom_by_wheel(1);
        assert_eq!(s.zoom(), 1.75);
        assert!((s.scroll().0 - (before.0 + 0.25 * 256.0 * 0.5)).abs() < 1e-4);
        assert!((s.scroll().1 - (before.1 + 0.25 * 202.0 * 0.5)).abs() < 1e-4);
    }

    /// The overshoot is what makes the map rubber-band: a drag past the edge moves the live
    /// position out of bounds, and only the *target* is clamped, so the easing pulls it back.
    #[test]
    fn only_the_target_is_clamped_which_is_what_makes_the_map_rubber_band() {
        let mut s = AchievementsScreen::new(tree());
        s.done_loading();
        s.set_scroll(10_000.0, 0.0);
        assert_eq!(s.scroll().0, 10_000.0);
        s.clamp_target();
        let (_, _, max_x, _) = s.grid.scroll_bounds();
        assert_eq!(
            s.target.0,
            f64::from(max_x - 1),
            "the >= test lands on bound - 1"
        );
        s.tick();
        assert!(s.scroll().0 < 10_000.0, "and the easing drags it back");
    }

    #[test]
    fn the_map_view_is_a_pure_translation_at_one_times_zoom() {
        let mut s = AchievementsScreen::new(roomy(tree().nodes));
        s.init(&ctx(), &Language::new("en_US"));
        s.set_scroll(0.0, 0.0);
        let v = s.map_view(0.0);
        assert_eq!(v.zoom, 1.0);
        assert_eq!(v.scroll, (0, 0));
        assert_eq!(v.origin, (s.frame().x + 16, s.frame().y + 17));
        assert_eq!(v.screen_x(0.0), s.frame().x as f32 + 16.0);
        assert_eq!(v.screen_y(155.0), s.frame().y as f32 + 17.0 + 155.0);
    }

    // ------------------------------------------------------- the map's drawing

    /// Two of the backdrop's rules are hard-coded rather than random, and both are visible: a
    /// single bedrock stripe at row 35, and bedrock everywhere the depth roll passes 37.
    #[test]
    fn the_backdrop_forces_row_thirty_five_and_everything_deep_to_bedrock() {
        for hash in [0, 1, -991_726_233, i32::MAX] {
            for col in -3..5 {
                assert_eq!(backdrop_tile(hash, col, 35), 7, "row 35 is always bedrock");
                assert_eq!(backdrop_tile(hash, col, 90), 7, "and so is anything deep");
            }
        }
        // Row 34 is not the stripe.
        let above: Vec<usize> = (0..8).map(|c| backdrop_tile(7, c, 34)).collect();
        assert!(above.iter().any(|&t| t != 7), "{above:?}");
    }

    /// The topmost row can only ever be sand: `nextInt(1) + 0 / 2` is `0`, which falls past
    /// every branch of the cascade to the value the variable was initialised with.
    #[test]
    fn the_shallowest_row_is_always_sand_because_the_roll_cannot_be_positive() {
        for hash in [0, 42, -1] {
            for col in -4..8 {
                assert_eq!(backdrop_tile(hash, col, 0), 0);
            }
        }
    }

    /// Two accounts see different scatter over the same tree, which is the whole reason the
    /// seed involves the player id at all.
    #[test]
    fn two_players_get_different_backdrops_for_the_same_tile() {
        let a: Vec<usize> = (0..32).map(|c| backdrop_tile(0, c, 12)).collect();
        let b: Vec<usize> = (0..32).map(|c| backdrop_tile(999_331, c, 12)).collect();
        assert_ne!(a, b);
    }

    /// The link's two halves come from `drawHorizontalLine`/`drawVerticalLine`, which disagree:
    /// the horizontal one is inclusive at both ends (`endX + 1`), the vertical one draws
    /// `startY + 1 .. endY` and so loses a row at *each* end.
    #[test]
    fn a_vertical_link_is_two_pixels_shorter_than_a_horizontal_link_of_the_same_span() {
        let view = MapView {
            origin: (0, 0),
            scroll: (0, 0),
            zoom: 1.0,
        };
        let mut h = DrawList::new();
        view.horizontal_line(&mut h, 10, 30, 5, 0xFF00_0000);
        let hx: Vec<f32> = h.vertices().iter().map(|v| v.pos[0]).collect();
        let (h0, h1) = (
            hx.iter().copied().fold(f32::MAX, f32::min),
            hx.iter().copied().fold(f32::MIN, f32::max),
        );
        assert_eq!((h0, h1), (10.0, 31.0), "21 pixels for a span of 20");

        let mut v = DrawList::new();
        view.vertical_line(&mut v, 5, 10, 30, 0xFF00_0000);
        let vy: Vec<f32> = v.vertices().iter().map(|q| q.pos[1]).collect();
        let (v0, v1) = (
            vy.iter().copied().fold(f32::MAX, f32::min),
            vy.iter().copied().fold(f32::MIN, f32::max),
        );
        assert_eq!((v0, v1), (11.0, 30.0), "19 pixels for the same span");
    }

    fn colours(d: &DrawList) -> Vec<[f32; 4]> {
        d.vertices().iter().map(|v| v.color).collect()
    }

    /// Grey for done, green for available, black for out of reach — vanilla's, and not the
    /// order anyone guesses.
    #[test]
    fn a_link_is_grey_when_unlocked_green_when_reachable_and_black_otherwise() {
        let view = MapView {
            origin: (0, 0),
            scroll: (0, 0),
            zoom: 1.0,
        };
        let mut g = tree();
        // nodes[1] is reachable (its parent is unlocked); nodes[2] is not.
        let s = AchievementsScreen::new(g.clone());
        let mut d = DrawList::new();
        s.draw_connections(&mut d, &view);
        let c = colours(&d);
        assert!(
            c.contains(&[0.0, 1.0, 0.0, 1.0]),
            "the reachable link is green"
        );
        assert!(
            c.contains(&[0.0, 0.0, 0.0, 1.0]),
            "the unreachable one is black"
        );

        g.nodes[1].unlocked = true;
        let s = AchievementsScreen::new(g);
        let mut d = DrawList::new();
        s.draw_connections(&mut d, &view);
        let grey = 160.0 / 255.0;
        assert!(
            colours(&d).contains(&[grey, grey, grey, 1.0]),
            "an unlocked link turns grey, not brighter"
        );
    }

    #[test]
    fn a_link_more_than_four_prerequisites_deep_is_not_drawn_at_all() {
        let mut nodes = vec![node("n0", 0, 0, None)];
        for i in 1..8 {
            nodes.push(node(&format!("n{i}"), i, 0, Some(&format!("n{}", i - 1))));
        }
        nodes[0].unlocked = true;
        let s = AchievementsScreen::new(AchievementGrid::new(nodes));
        let view = MapView {
            origin: (0, 0),
            scroll: (0, 0),
            zoom: 1.0,
        };
        let mut d = DrawList::new();
        s.draw_connections(&mut d, &view);
        // n1..n5 have locked depth 0..4 and draw; n6 and n7 are 5 and 6 deep and do not.
        assert_eq!(s.grid.locked_depth(&s.grid.nodes[5]), 4);
        assert_eq!(s.grid.locked_depth(&s.grid.nodes[6]), 5);
        // Each drawn link is one horizontal rect, one (degenerate) vertical rect and one arrow.
        let quads = d.vertices().len() / 6;
        assert_eq!(quads, 5 * 3, "five links survive the depth cut");
    }

    #[test]
    fn cell_shades_follow_vanillas_five_step_ladder_and_vanish_past_four() {
        let mut nodes = vec![node("n0", 0, 0, None)];
        for i in 1..8 {
            nodes.push(node(&format!("n{i}"), i, 0, Some(&format!("n{}", i - 1))));
        }
        nodes[0].unlocked = true;
        let s = AchievementsScreen::new(AchievementGrid::new(nodes));
        assert_eq!(s.cell_shade(&s.grid.nodes[0]), Some(0.75), "unlocked");
        assert_eq!(s.cell_shade(&s.grid.nodes[1]), Some(1.0), "reachable");
        assert_eq!(s.cell_shade(&s.grid.nodes[2]), Some(0.3));
        assert_eq!(s.cell_shade(&s.grid.nodes[3]), Some(0.3));
        assert_eq!(s.cell_shade(&s.grid.nodes[4]), Some(0.2));
        assert_eq!(s.cell_shade(&s.grid.nodes[5]), Some(0.1));
        assert_eq!(s.cell_shade(&s.grid.nodes[6]), None, "past four, nothing");
    }

    /// The plate is a 26×26 sprite drawn two pixels up and left of the 22-pixel cell, and a
    /// special achievement takes the second one on the sheet.
    #[test]
    fn a_special_cell_takes_the_second_plate_and_both_overhang_the_cell_by_two() {
        let mut g = tree();
        g.nodes[0].special = true;
        let s = AchievementsScreen::new(g);
        let view = MapView {
            origin: (100, 200),
            scroll: (0, 0),
            zoom: 1.0,
        };
        let mut d = DrawList::new();
        s.draw_cells(&mut d, None, &view);
        // The root is at grid (0, 0), so its plate lands at origin - 2.
        assert_eq!(d.vertices()[0].pos, [98.0, 198.0]);
        assert_eq!(d.vertices()[0].uv, [26.0 / 256.0, 202.0 / 256.0]);
        // Strip order is [TL, BL, TR, BL, BR, TR], so index 2 is the top-right corner.
        let plate_size = d.vertices()[2].pos[0] - d.vertices()[0].pos[0];
        assert_eq!(plate_size, 26.0);
    }

    /// At 2× the map is drawn through `scale(1 / zoom)`, so the plate comes out half size. That
    /// the *item icon* halves with it — it reaches the same scale through S15's matrix stack
    /// rather than through [`MapView`] — needs a baked model, so it is pinned in
    /// `tests/achievements_map.rs` instead.
    #[test]
    fn zooming_out_halves_the_cell_plate() {
        let s = AchievementsScreen::new(tree());
        let width = |zoom: f32| {
            let view = MapView {
                origin: (0, 0),
                scroll: (0, 0),
                zoom,
            };
            let mut d = DrawList::new();
            s.draw_cells(&mut d, None, &view);
            let plate = &d.vertices()[0..6];
            plate.iter().map(|v| v.pos[0]).fold(f32::MIN, f32::max)
                - plate.iter().map(|v| v.pos[0]).fold(f32::MAX, f32::min)
        };
        assert_eq!(width(1.0), 26.0);
        assert_eq!(width(2.0), 13.0);
    }

    /// The hit box vanilla tests is `>= l6 && <= l6 + 22` — twenty-**three** pixels, one more
    /// than the cell it belongs to, so the gutter's first column belongs to the cell after all.
    #[test]
    fn the_hover_box_is_twenty_three_pixels_wide_not_twenty_two() {
        let mut s = AchievementsScreen::new(roomy(vec![node("a", 0, 0, None)]));
        s.init(&ctx(), &Language::new("en_US"));
        s.set_scroll(0.0, 0.0);
        let f = s.frame();
        let (ox, oy) = (f.x + MAP_ORIGIN_X, f.y + MAP_ORIGIN_Y);
        assert!(s.node_at_screen(ox + 22, oy).is_some(), "22 still hits");
        assert!(s.node_at_screen(ox + 23, oy).is_none(), "23 does not");
        assert!(s.node_at_screen(ox, oy + 22).is_some());
        assert!(s.node_at_screen(ox - 1, oy).is_none());
        // `AchievementGrid::node_at`, which the model keeps for grid-space queries, uses the
        // honest 22-pixel cell instead — the two deliberately disagree on that one column.
        assert!(s.grid.node_at(22, 0).is_none());
    }

    /// The hover loop assigns rather than breaking, so when cells overlap the last one wins.
    #[test]
    fn overlapping_cells_hover_the_last_one_in_list_order() {
        let mut s = AchievementsScreen::new(roomy(vec![
            node("first", 0, 0, None),
            node("second", 0, 0, None),
        ]));
        s.init(&ctx(), &Language::new("en_US"));
        s.set_scroll(0.0, 0.0);
        let f = s.frame();
        assert_eq!(
            s.node_at_screen(f.x + MAP_ORIGIN_X, f.y + MAP_ORIGIN_Y)
                .map(|n| n.key.as_str()),
            Some("second")
        );
    }

    /// The title is `drawString` at `(frame.x + 15, frame.y + 5)`, left-aligned, unshadowed —
    /// not a centred string — and it is drawn **after** the tooltip, so it can never be covered.
    #[test]
    fn the_title_is_left_aligned_at_fifteen_five_and_drawn_last() {
        let mut lang = Language::new("en_US");
        lang.insert("gui.achievements", "Achievements");
        lang.insert("gui.done", "Done");
        let mut s = AchievementsScreen::new(tree());
        s.done_loading();
        s.init(&ctx(), &lang);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, None, &ctx(), (0, 0), &lang);
        let f = s.frame();
        assert!(
            matches!(
                d.batches().last().map(|b| &b.texture),
                Some(UiTexture::Font(_))
            ),
            "nothing is drawn after the title"
        );
        // The title's first glyph starts exactly at (frame.x + 15, frame.y + 5), unshadowed —
        // so that vertex exists, and the centred position it would have had does not.
        let has = |p: [f32; 2]| d.vertices().iter().any(|v| v.pos == p);
        assert!(
            has([(f.x + 15) as f32, (f.y + 5) as f32]),
            "left-aligned at 15, 5"
        );
        let centred = (854 - font.string_width("Achievements")) / 2;
        assert!(
            !has([centred as f32, (f.y + 8) as f32]),
            "and not where drawCenteredString would have put it"
        );
    }

    /// The frame goes down *after* the map, because it is what hides the tiles that overhang
    /// the 224×155 window.
    #[test]
    fn the_frame_is_drawn_over_the_map_not_under_it() {
        let mut lang = Language::new("en_US");
        lang.insert("gui.done", "Done");
        let mut s = AchievementsScreen::new(tree());
        s.done_loading();
        s.init(&ctx(), &lang);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, None, &ctx(), (0, 0), &lang);
        let f = s.frame();
        // The frame blit is the one quad at exactly the frame's top-left with UV (0, 0).
        let frame_quad = d
            .vertices()
            .chunks(6)
            .position(|q| q[0].pos == [f.x as f32, f.y as f32] && q[0].uv == [0.0, 0.0])
            .expect("the frame is blitted");
        let plate_quad = d
            .vertices()
            .chunks(6)
            .position(|q| (q[0].uv[1] - 202.0 / 256.0).abs() < 1e-6)
            .expect("at least one cell plate is drawn");
        assert!(plate_quad < frame_quad, "{plate_quad} then {frame_quad}");
    }

    #[test]
    fn overlapping_map_cells_keep_java_depth_and_list_order_for_plates_and_icons() {
        let mut first = node("first", 0, 0, None);
        first.icon = Some(test_icon());
        let mut second = node("second", 0, 0, None);
        second.special = true;
        second.icon = Some(test_icon());
        let screen = AchievementsScreen::new(AchievementGrid::new(vec![first, second]));
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut d = DrawList::new();
        screen.draw_cells(
            &mut d,
            Some(&atlas),
            &MapView {
                origin: (0, 0),
                scroll: (0, 0),
                zoom: 1.0,
            },
        );

        let plate_batches: Vec<_> = d
            .batches()
            .iter()
            .filter(|batch| batch.texture == UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE))
            .collect();
        assert_eq!(
            plate_batches.len(),
            2,
            "one plate batch per side of the icon interleave"
        );
        assert!(plate_batches.iter().all(|batch| {
            batch.depth == UiDepth::Item
                && d.depths()[batch.range.start..batch.range.end]
                    .iter()
                    .all(|depth| (*depth - ACHIEVEMENT_MAP_DEPTH).abs() < f32::EPSILON)
        }));
        let first_plate = &d.vertices()[plate_batches[0].range.start];
        let last_plate = &d.vertices()[plate_batches[1].range.start];
        assert_eq!(first_plate.uv, [0.0, 202.0 / 256.0]);
        assert_eq!(last_plate.uv, [26.0 / 256.0, 202.0 / 256.0]);
        assert!(plate_batches[0].range.end <= plate_batches[1].range.start);

        let icon_batches: Vec<_> = d
            .batches()
            .iter()
            .filter(|batch| batch.texture != UiTexture::gui(ACHIEVEMENT_BACKGROUND_TEXTURE))
            .collect();
        assert_eq!(icon_batches.len(), 2, "one icon pass per overlapping node");
        assert!(icon_batches.iter().all(|batch| {
            batch.depth == UiDepth::Item
                && d.depths()[batch.range.start..batch.range.end]
                    .iter()
                    .all(|depth| (*depth - ACHIEVEMENT_MAP_DEPTH).abs() < 0.01)
        }));
        assert!(icon_batches[0].range.end <= icon_batches[1].range.start);
    }

    #[test]
    fn map_layers_are_item_depth_but_frame_and_hover_are_overlay_ordered() {
        let mut lang = Language::new("en_US");
        lang.insert("gui.achievements", "Achievements");
        lang.insert("gui.done", "Done");
        lang.insert("achievement.openInventory", "Taking Inventory");
        lang.insert("achievement.openInventory.desc", "Open your inventory");
        let mut root = node("achievement.openInventory", 0, 0, None);
        root.unlocked = true;
        root.icon = Some(test_icon());
        let mut child = node("child", 1, 0, Some("achievement.openInventory"));
        child.icon = Some(test_icon());
        let mut screen = AchievementsScreen::new(AchievementGrid::new(vec![root, child]));
        screen.done_loading();
        screen.init(&ctx(), &lang);
        let atlas = UiTexture::gui("textures/atlas/blocks.png");
        let mut d = DrawList::new();
        let mut font = uniform_font();
        let mouse = (
            screen.frame().x + MAP_ORIGIN_X,
            screen.frame().y + MAP_ORIGIN_Y,
        );
        screen.draw(&mut d, &mut font, Some(&atlas), &ctx(), mouse, &lang);

        let map_batches: Vec<_> = d
            .batches()
            .iter()
            .enumerate()
            .filter(|(_, batch)| batch.depth == UiDepth::Item)
            .collect();
        assert!(!map_batches.is_empty(), "map emits depth-tested batches");
        assert!(map_batches.iter().all(|(_, batch)| {
            d.depths()[batch.range.start..batch.range.end]
                .iter()
                .all(|depth| (*depth - ACHIEVEMENT_MAP_DEPTH).abs() < 0.01)
        }));

        let frame_index = d
            .batches()
            .iter()
            .enumerate()
            .find(|(_, batch)| {
                batch.depth == UiDepth::Overlay
                    && matches!(
                        &batch.texture,
                        UiTexture::Resource(location)
                            if location.path == ACHIEVEMENT_BACKGROUND_TEXTURE
                    )
                    && batch.range.clone().any(|index| {
                        d.vertices()[index].pos == [screen.frame.x as f32, screen.frame.y as f32]
                    })
            })
            .map(|(index, _)| index)
            .expect("frame overlay batch");
        assert!(map_batches.iter().all(|(index, _)| *index < frame_index));

        let title_start = d
            .batches()
            .iter()
            .enumerate()
            .rev()
            .find(|(_, batch)| matches!(batch.texture, UiTexture::Font(_)))
            .map(|(index, _)| index)
            .expect("title font batch");
        assert!(
            title_start > frame_index,
            "title follows frame and hover content"
        );
    }

    #[test]
    fn a_loading_screen_draws_no_map() {
        let mut lang = Language::new("en_US");
        lang.insert("multiplayer.downloadingStats", "Downloading statistics...");
        let mut s = AchievementsScreen::new(tree());
        s.loading = true;
        s.init(&ctx(), &lang);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, None, &ctx(), (0, 0), &lang);
        assert!(
            !d.batches().iter().any(|b| matches!(&b.texture,
                UiTexture::Resource(r) if r.path == ACHIEVEMENT_BACKGROUND_TEXTURE)),
            "no frame, no plates, no links while the stats are still downloading"
        );
        assert_eq!(s.mouse_clicked(0, 0, 0), Action::None);
    }

    #[test]
    fn loading_spinner_uses_vanillas_seven_frames_for_one_hundred_fifty_ms_each() {
        let expected = [
            "oooooo", "Oooooo", "oOoooo", "ooOooo", "oooOoo", "ooooOo", "oooooO",
        ];
        for (frame, expected) in expected.into_iter().enumerate() {
            let start = frame as u128 * 150;
            assert_eq!(AchievementsScreen::loading_spinner(start), expected);
            assert_eq!(AchievementsScreen::loading_spinner(start + 149), expected);
        }
        assert_eq!(AchievementsScreen::loading_spinner(1_050), "oooooo");
    }

    #[test]
    fn transient_achievement_toast_slides_in_parks_and_expires_after_three_seconds() {
        let mut toast = AchievementToast::default();
        toast.display("Achievement Get!", "Taking Inventory", None, 10_000);

        // A normal notification is not allowed to render before its timestamp; vanilla clears
        // it immediately when the signed elapsed value is negative.
        assert_eq!(toast.panel_y(9_999), None);
        assert!(!toast.is_active());
        toast.display("Achievement Get!", "Taking Inventory", None, 10_000);
        assert_eq!(toast.panel_y(10_000), Some(-36));
        assert_eq!(toast.panel_y(10_001), Some(-35));
        assert_eq!(toast.panel_y(10_375), Some(0));
        assert_eq!(toast.panel_y(11_500), Some(0));
        assert_eq!(toast.panel_y(12_625), Some(0));
        assert_eq!(toast.panel_y(13_000), Some(-36));
        assert_eq!(toast.panel_y(13_001), None);
        assert!(!toast.is_active());
    }

    #[test]
    fn permanent_achievement_hint_starts_late_and_never_slides_back_out() {
        let mut toast = AchievementToast::default();
        toast.display_permanent("Taking Inventory", "Press E", None, 10_000);

        // The permanent notification is scheduled at 12_500. Its negative pre-start progress is
        // intentionally retained, which keeps the panel at its fully off-screen y coordinate.
        assert_eq!(toast.panel_y(12_499), Some(-36));
        assert_eq!(toast.panel_y(12_500), Some(-36));
        assert_eq!(toast.panel_y(12_501), Some(-35));
        assert_eq!(toast.panel_y(12_875), Some(0));
        assert_eq!(toast.panel_y(i64::MAX), Some(0));
        assert!(toast.is_active());
        toast.clear();
        assert_eq!(toast.panel_y(i64::MAX), None);
    }

    #[test]
    fn achievement_toast_uses_the_top_right_background_slice() {
        let mut toast = AchievementToast::default();
        toast.display("Achievement Get!", "Taking Inventory", None, 1_000);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        toast.draw(&mut d, &mut font, None, &ctx(), 1_375);

        let background = d
            .batches()
            .iter()
            .find(|batch| {
                matches!(
                    &batch.texture,
                    UiTexture::Resource(location)
                        if location.path == ACHIEVEMENT_BACKGROUND_TEXTURE
                )
            })
            .expect("toast background");
        let first = d.vertices()[background.range.start];
        assert_eq!(first.pos, [(854 - 160) as f32, 0.0]);
        assert_eq!(first.uv, [96.0 / 256.0, 202.0 / 256.0]);
    }

    #[test]
    fn the_tooltip_names_a_reachable_achievement_and_hides_an_unreachable_one() {
        let mut lang = Language::new("en_US");
        lang.insert("achievement.mineWood", "Getting Wood");
        lang.insert(
            "achievement.mineWood.desc",
            "Attack a tree until a block breaks",
        );
        lang.insert("achievement.buildWorkBench", "Benchmarking");
        lang.insert("achievement.requires", "Requires %s");
        let g = tree();
        let s = AchievementsScreen::new(g);
        let mut font = uniform_font();

        let mut reachable = DrawList::new();
        s.draw_tooltip(&mut reachable, &mut font, &s.grid.nodes[1], (40, 40), &lang);
        assert!(!reachable.is_empty());

        // Two prerequisites deep: the name is still shown, but the body is "Requires ...".
        let mut locked = DrawList::new();
        s.draw_tooltip(&mut locked, &mut font, &s.grid.nodes[2], (40, 40), &lang);
        assert!(!locked.is_empty());

        // The tooltip box always starts at (mouse.x + 12 - 3, mouse.y - 4 - 3).
        assert_eq!(
            reachable.vertices()[0].pos,
            [(40 + 12 - 3) as f32, (40 - 4 - 3) as f32]
        );
    }

    #[test]
    fn taking_inventory_description_formats_the_current_inventory_binding() {
        let mut lang = Language::new("en_US");
        lang.insert(
            "achievement.openInventory.desc",
            "Press '%1$s' to open your inventory.",
        );
        let mut s = AchievementsScreen::new(tree());
        assert_eq!(
            s.achievement_description(&s.grid.nodes[0], &lang),
            "Press '%1$s' to open your inventory.",
            "without the driver seam the untranslated placeholder stays visible"
        );
        s.set_inventory_key_name("Button 3");
        assert_eq!(
            s.achievement_description(&s.grid.nodes[0], &lang),
            "Press 'Button 3' to open your inventory."
        );
    }
}
