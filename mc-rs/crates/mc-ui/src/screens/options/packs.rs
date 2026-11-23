//! The resource-pack screen: two lists, available on the left and selected on the right.
//!
//! java: `client/gui/GuiScreenResourcePacks.java`, `GuiResourcePackAvailable.java`,
//! `GuiResourcePackSelected.java`, `ResourcePackListEntry.java`
//!
//! Vanilla moves packs through the icon controls and scrolls each list by dragging. The
//! *ordering semantics* have to be right:
//! the selected list is drawn top-to-bottom but applied bottom-to-top, so the pack at the top
//! of the right-hand column is the one that wins. Getting that backwards silently inverts every
//! texture override.

use mc_assets::{Language, PACK_FORMAT_1_8_9};
use mc_render::settings::GameSettings;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_dirt_background, TITLE_COLOR};
use crate::screens::yes_no::YesNoScreen;
use crate::screens::Action;
use crate::widget::draw::{DrawList, Rect, UiTexture};
use crate::widget::{draw_centered_string, draw_string, Button, ScrollList, ScrollbarX};

/// One pack in either column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackEntry {
    /// The file name, which is what `options.txt` stores.
    pub name: String,
    /// The `pack.mcmeta` description.
    pub description: String,
    /// Whether its `pack_format` matches this client.
    pub compatible: bool,
    /// Declared pack format, when metadata supplied one. Needed to choose the old/new warning.
    pub pack_format: Option<i64>,
    /// Texture uploaded for the pack's root `pack.png`, or the default-pack fallback.
    pub icon: Option<UiTexture>,
    /// `ResourcePackListEntryDefault.func_148310_d()` is false: the built-in row is a permanent
    /// floor under the user's selected packs and has no arrow controls.
    builtin_default: bool,
}

impl PackEntry {
    /// A pack entry.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, compatible: bool) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            compatible,
            pack_format: None,
            icon: None,
            builtin_default: false,
        }
    }

    /// Vanilla's synthetic, permanently-selected `ResourcePackListEntryDefault` row.
    #[must_use]
    pub fn default_pack(description: impl Into<String>) -> Self {
        Self {
            name: "Default".to_owned(),
            description: description.into(),
            compatible: true,
            pack_format: Some(i64::from(PACK_FORMAT_1_8_9)),
            icon: None,
            builtin_default: true,
        }
    }

    /// Whether this is the synthetic built-in row, which must never reach `options.txt`.
    #[must_use]
    pub fn is_default(&self) -> bool {
        self.builtin_default
    }

    fn has_controls(&self) -> bool {
        !self.builtin_default
    }

    /// Attach the declared pack format and derive compatibility from 1.8.9's exact format.
    #[must_use]
    pub fn with_pack_format(mut self, pack_format: i64) -> Self {
        self.pack_format = Some(pack_format);
        self.compatible = pack_format == i64::from(PACK_FORMAT_1_8_9);
        self
    }

    /// Attach the dynamic texture the resource-pack repository uploaded for `pack.png`.
    #[must_use]
    pub fn with_icon(mut self, icon: UiTexture) -> Self {
        self.icon = Some(icon);
        self
    }

    /// Vanilla's normal-colour warning description for a pack from a different version.
    // java: ResourcePackListEntry.java:66
    #[must_use]
    pub fn description_line(&self, lang: &Language) -> String {
        if self.compatible {
            self.description.clone()
        } else {
            let age = if self
                .pack_format
                .is_some_and(|format| format > i64::from(PACK_FORMAT_1_8_9))
            {
                "resourcePack.incompatible.new"
            } else {
                "resourcePack.incompatible.old"
            };
            lang.tr(age).to_owned()
        }
    }
}

/// Which column a row lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackColumn {
    /// The left-hand "Available" list.
    Available,
    /// The right-hand "Selected" list.
    Selected,
}

/// The resource-pack screen.
#[derive(Debug)]
pub struct ResourcePacksScreen {
    /// Packs on disk but not enabled.
    pub available: Vec<PackEntry>,
    /// Enabled packs, **highest priority first** — the order the screen shows.
    pub selected: Vec<PackEntry>,
    /// Vanilla backs each column with its own `GuiSlot`.
    lists: [ScrollList; 2],
    open_folder: Button,
    done: Button,
    /// `gameSettings.touchscreen`: keeps every row's icon controls visible.
    touchscreen: bool,
    /// Vanilla substitutes the built-in pack's image when a pack has no readable `pack.png`.
    default_icon: Option<UiTexture>,
    /// The modal `GuiYesNo` shown before an incompatible pack can be enabled.
    confirmation: Option<PackConfirmation>,
    confirmation_text: ConfirmationText,
    screen_size: (i32, i32),
    /// Whether anything changed and the packs need reloading.
    pub dirty: bool,
}

#[derive(Debug)]
struct PackConfirmation {
    pack_name: String,
    dialog: YesNoScreen,
}

#[derive(Debug, Default)]
struct ConfirmationText {
    title: String,
    old: String,
    new: String,
    yes: String,
    no: String,
}

/// A column is 200 wide. java: `new GuiResourcePackAvailable(mc, 200, height, list)`
pub const COLUMN_WIDTH: i32 = 200;
/// A row is 36 tall. java: `ResourcePackListEntry` is 32 of content plus a 4-pixel gap.
pub const ROW_HEIGHT: i32 = 36;
/// `setHasListHeader(true, (int)(FONT_HEIGHT * 1.5F))`, with `FONT_HEIGHT == 9`.
pub const HEADER_PADDING: i32 = 13;
/// Width used by `trimStringToWidth` and `listFormattedStringToWidth` for pack text.
pub const PACK_TEXT_WIDTH: i32 = 157;
const DESCRIPTION_COLOR: i32 = 0x0080_8080;
const RESOURCE_PACKS_TEXTURE: &str = "textures/gui/resource_packs.png";
/// Java literal `-8978432`; note this is `0xFF770000`, not the audit prose's `0xFF700000`.
const INCOMPATIBLE_PLATE: u32 = 0xFF77_0000;

impl ResourcePacksScreen {
    /// A screen over the packs found on disk and the ones `options.txt` has enabled.
    ///
    /// `enabled` is `GameSettings::resource_packs`, which vanilla stores **lowest priority
    /// first** (it is applied in order, later packs overriding earlier ones). The right-hand
    /// column shows the reverse.
    #[must_use]
    pub fn new(found: Vec<PackEntry>, enabled: &[String]) -> Self {
        let mut selected = Vec::new();
        // Walk `enabled` backwards: the last entry is the highest priority, so it goes on top.
        for name in enabled.iter().rev() {
            if let Some(p) = found.iter().find(|p| &p.name == name) {
                selected.push(p.clone());
            }
        }
        let available: Vec<PackEntry> = found
            .into_iter()
            .filter(|p| !enabled.iter().any(|n| n == &p.name))
            .collect();
        // java: GuiScreenResourcePacks.initGui appends ResourcePackListEntryDefault after all
        // reversed user packs. Its metadata failure string is the exact fallback used by
        // ResourcePackListEntryDefault.func_148311_a; the client can replace it once it has read
        // the built-in pack metadata.
        selected.push(PackEntry::default_pack("\u{a7}cMissing pack.mcmeta :("));
        let mut lists = [
            Self::new_column_list(available.len()),
            Self::new_column_list(selected.len()),
        ];
        // The real dimensions arrive in `init`; keep the pre-layout lists harmless.
        lists[0].set_slot_x_bounds_from_left(0);
        lists[1].set_slot_x_bounds_from_left(0);
        Self {
            available,
            selected,
            lists,
            open_folder: Button::new(2, 0, 0, ""),
            done: Button::new(1, 0, 0, ""),
            touchscreen: false,
            default_icon: None,
            confirmation: None,
            confirmation_text: ConfirmationText::default(),
            screen_size: (0, 0),
            dirty: false,
        }
    }

    /// Carry `GameSettings::touchscreen` into the row renderer.
    #[must_use]
    pub fn with_touchscreen(mut self, touchscreen: bool) -> Self {
        self.touchscreen = touchscreen;
        self
    }

    /// Supply the built-in resource pack's `pack.png` texture as the fallback for entries whose
    /// own image was absent or unreadable.
    #[must_use]
    pub fn with_default_icon(mut self, icon: UiTexture) -> Self {
        self.default_icon = Some(icon);
        self
    }

    /// Replace the built-in row's description with the default pack's `pack.mcmeta` text.
    #[must_use]
    pub fn with_default_description(mut self, description: impl Into<String>) -> Self {
        self.set_default_description(description);
        self
    }

    /// Replace the built-in row's description after construction.
    pub fn set_default_description(&mut self, description: impl Into<String>) {
        if let Some(entry) = self.selected.iter_mut().find(|entry| entry.is_default()) {
            entry.description = description.into();
        }
    }

    /// Replace the built-in fallback icon after the screen has been constructed.
    pub fn set_default_icon(&mut self, icon: Option<UiTexture>) {
        self.default_icon = icon;
    }

    /// Whether the incompatible-pack `GuiYesNo` currently owns input and drawing.
    #[must_use]
    pub fn confirmation_open(&self) -> bool {
        self.confirmation.is_some()
    }

    fn new_column_list(size: usize) -> ScrollList {
        let mut list = ScrollList::new(COLUMN_WIDTH, 0, 32, 0, ROW_HEIGHT);
        list.size = size;
        list.list_width = COLUMN_WIDTH;
        list.show_selection_box = false;
        list.scrollbar_placement = ScrollbarX::FromRight { inset: 6 };
        list.set_header(true, HEADER_PADDING);
        list
    }

    fn list(&self, column: PackColumn) -> &ScrollList {
        &self.lists[column as usize]
    }

    fn list_mut(&mut self, column: PackColumn) -> &mut ScrollList {
        &mut self.lists[column as usize]
    }

    /// The packs in one column.
    #[must_use]
    pub fn column(&self, column: PackColumn) -> &[PackEntry] {
        match column {
            PackColumn::Available => &self.available,
            PackColumn::Selected => &self.selected,
        }
    }

    /// Where row `index` of `column` draws.
    #[must_use]
    pub fn row_rect(&self, column: PackColumn, index: usize) -> Rect {
        self.list(column).slot_rect(index)
    }

    /// Scroll one column, clamped to its content.
    pub fn scroll_by(&mut self, column: PackColumn, pixels: i32) {
        self.list_mut(column).scroll_by(pixels);
    }

    /// The `resourcePacks` value to write back to `options.txt`, lowest priority first.
    #[must_use]
    pub fn to_settings_order(&self) -> Vec<String> {
        self.selected
            .iter()
            .rev()
            .filter(|p| !p.is_default())
            .map(|p| p.name.clone())
            .collect()
    }

    /// Write the current selection into `settings`.
    pub fn apply(&self, settings: &mut GameSettings) {
        settings.resource_packs = self.to_settings_order();
        settings.incompatible_resource_packs = self
            .selected
            .iter()
            .filter(|p| !p.is_default() && !p.compatible)
            .map(|p| p.name.clone())
            .collect();
    }

    /// Lay out.
    // java: GuiScreenResourcePacks.java:44
    pub fn init(&mut self, ctx: &ScreenCtx, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        self.screen_size = (w, h);
        self.confirmation_text = ConfirmationText {
            title: lang
                .tr("resourcePack.incompatible.confirm.title")
                .to_owned(),
            old: lang.tr("resourcePack.incompatible.confirm.old").to_owned(),
            new: lang.tr("resourcePack.incompatible.confirm.new").to_owned(),
            yes: lang.tr("gui.yes").to_owned(),
            no: lang.tr("gui.no").to_owned(),
        };
        if let Some(confirmation) = &mut self.confirmation {
            confirmation.dialog.resize(w, h);
        }
        // Vanilla places the two 200-wide columns either side of the centre with an 8 px gap.
        let origins = [w / 2 - 4 - COLUMN_WIDTH, w / 2 + 4];
        for (list, origin) in self.lists.iter_mut().zip(origins) {
            list.set_dimensions(COLUMN_WIDTH, h, 32, h - 55 + 4);
            list.list_width = COLUMN_WIDTH;
            list.show_selection_box = false;
            list.scrollbar_placement = ScrollbarX::FromRight { inset: 6 };
            list.set_header(true, HEADER_PADDING);
            list.set_slot_x_bounds_from_left(origin);
            list.clamp_scroll();
        }
        self.open_folder = Button::sized(
            2,
            w / 2 - 154,
            h - 48,
            150,
            20,
            lang.tr("resourcePack.openFolder").to_owned(),
        );
        self.done = Button::sized(
            1,
            w / 2 + 4,
            h - 48,
            150,
            20,
            lang.tr("gui.done").to_owned(),
        );
    }

    /// Move a pack from available to selected, at the **top** (highest priority).
    // java: GuiScreenResourcePacks — the entry's right-arrow button
    pub fn enable(&mut self, index: usize) -> bool {
        if index >= self.available.len() {
            return false;
        }
        let p = self.available.remove(index);
        self.selected.insert(0, p);
        self.sync_sizes();
        true
    }

    /// Move a pack from selected back to available.
    pub fn disable(&mut self, index: usize) -> bool {
        if index >= self.selected.len() || !self.selected[index].has_controls() {
            return false;
        }
        let p = self.selected.remove(index);
        self.available.insert(0, p);
        self.sync_sizes();
        true
    }

    /// Move a selected pack one place up or down the priority order.
    pub fn reorder(&mut self, index: usize, up: bool) -> bool {
        if index >= self.selected.len() || !self.selected[index].has_controls() {
            return false;
        }
        let target = if up {
            if index == 0 {
                return false;
            }
            index - 1
        } else {
            if index + 1 >= self.selected.len() {
                return false;
            }
            index + 1
        };
        if !self.selected[target].has_controls() {
            return false;
        }
        self.selected.swap(index, target);
        self.dirty = true;
        true
    }

    fn sync_sizes(&mut self) {
        self.dirty = true;
        self.lists[PackColumn::Available as usize].size = self.available.len();
        self.lists[PackColumn::Selected as usize].size = self.selected.len();
        self.lists[PackColumn::Available as usize].clamp_scroll();
        self.lists[PackColumn::Selected as usize].clamp_scroll();
    }

    /// Which actionable 32-pixel icon square a point falls in.
    ///
    /// `GuiSlot` accepts two pixels to the left of the drawn row (`left..left + 2`) and passes a
    /// negative relative x to the entry. `ResourcePackListEntry` only applies an upper bound
    /// (`relativeX <= 32`), so those two pixels deliberately remain actionable here.
    #[must_use]
    pub fn hit(&self, x: i32, y: i32) -> Option<(PackColumn, usize)> {
        for column in [PackColumn::Available, PackColumn::Selected] {
            let list = self.list(column);
            if let Some(index) = list.slot_index_at(x, y) {
                if self.column(column)[index].has_controls() && x - list.slot_rect(index).x <= 32 {
                    return Some((column, index));
                }
            }
        }
        None
    }

    /// Handle a click on the screen or either list entry.
    pub fn mouse_clicked(&mut self, x: i32, y: i32, button: i32) -> Action {
        if self.confirmation.is_some() {
            let result = self
                .confirmation
                .as_mut()
                .and_then(|pending| pending.dialog.click(x, y, button));
            if let Some(result) = result {
                let pending = self
                    .confirmation
                    .take()
                    .expect("a dialog result requires a pending pack");
                if result.confirmed {
                    if let Some(index) = self
                        .available
                        .iter()
                        .position(|entry| entry.name == pending.pack_name)
                    {
                        self.enable(index);
                    }
                }
            }
            return Action::None;
        }

        // java: `GuiScreenResourcePacks.actionPerformed` case 1 — copy the selection into
        // `gameSettings`, `saveOptions()`, `refreshResources()`, `displayGuiScreen(parentScreen)`.
        // The caller does the first three (it owns the settings and the pack stack); the
        // *screen's* half of that is only "leave", so this is `Back` rather than `SaveSettings`.
        if button == 0 {
            if self.done.mouse_pressed(x, y) {
                return Action::Back;
            }
            if self.open_folder.mouse_pressed(x, y) {
                // Its own intent; see `Action::OpenResourcePackFolder`.
                return Action::OpenResourcePackFolder;
            }
            // Only the left button arms GuiSlot's content/scrollbar drag. Entry controls below
            // intentionally accept every button: ResourcePackListEntry.mousePressed ignores its
            // `mouseEvent` argument, and GuiScreenResourcePacks forwards all buttons to it.
            for column in [PackColumn::Available, PackColumn::Selected] {
                let list = self.list(column);
                if x >= list.left && x <= list.right && y >= list.top && y <= list.bottom {
                    self.list_mut(column).mouse_pressed(x, y);
                    break;
                }
            }
        }
        match self.hit(x, y) {
            Some((PackColumn::Available, i)) => {
                if self.available[i].compatible {
                    self.enable(i);
                } else {
                    // java: ResourcePackListEntry.mousePressed calls markChanged before opening
                    // GuiYesNo, including when the player ultimately cancels.
                    self.dirty = true;
                    let message = if self.available[i]
                        .pack_format
                        .is_some_and(|format| format > i64::from(PACK_FORMAT_1_8_9))
                    {
                        self.confirmation_text.new.clone()
                    } else {
                        self.confirmation_text.old.clone()
                    };
                    let mut dialog = YesNoScreen::new(
                        self.confirmation_text.title.clone(),
                        message,
                        self.confirmation_text.yes.clone(),
                        self.confirmation_text.no.clone(),
                        0,
                    );
                    dialog.resize(self.screen_size.0, self.screen_size.1);
                    let confirmation = PackConfirmation {
                        pack_name: self.available[i].name.clone(),
                        dialog,
                    };
                    // The click armed the underlying GuiSlot before the entry opened its child
                    // screen. Release that drag now; input must belong exclusively to GuiYesNo.
                    self.mouse_released();
                    self.confirmation = Some(confirmation);
                }
                self.mouse_released();
                Action::None
            }
            Some((PackColumn::Selected, i)) => {
                let r = self.row_rect(PackColumn::Selected, i);
                let relative_x = x - r.x;
                let relative_y = y - r.y;
                if relative_x < 16 {
                    self.disable(i);
                } else if relative_x > 16 && relative_y < 16 {
                    self.reorder(i, true);
                } else if relative_x > 16 && relative_y > 16 {
                    self.reorder(i, false);
                }
                self.mouse_released();
                Action::None
            }
            None => Action::None,
        }
    }

    /// Continue a drag in the resource-pack column under the cursor.
    ///
    /// java: `GuiSlot.handleMouseInput` gates its drag branch on
    /// `isMouseYWithinSlotBounds(mouseY)`, which checks both coordinates even though the helper's
    /// name mentions only the y axis. Keeping the gate here matters when a held pointer leaves a
    /// column and crosses the centre gap or the button area: the active list must retain its last
    /// drag anchor until the pointer re-enters, rather than jumping while it is outside.
    pub fn mouse_dragged(&mut self, x: i32, y: i32) {
        if self.confirmation.is_some() {
            return;
        }
        for list in &mut self.lists {
            if list.is_mouse_within_bounds(x, y) {
                list.mouse_dragged(y);
            }
        }
    }

    /// Release both independent column drags.
    pub fn mouse_released(&mut self) {
        if self.confirmation.is_some() {
            return;
        }
        for list in &mut self.lists {
            list.mouse_released();
        }
    }

    /// Scroll whichever column the cursor is over.
    ///
    /// java: `GuiScreenResourcePacks.handleMouseInput` forwards the wheel to *both* lists and each
    /// `GuiSlot.handleMouseInput` ignores it unless the cursor is inside its own bounds; the step
    /// is `GuiSlot`'s own `slotHeight / 2`.
    // java: GuiSlot.java:283
    pub fn mouse_wheel(&mut self, x: i32, y: i32, delta: f32) {
        if self.confirmation.is_some() || delta == 0.0 {
            return;
        }
        for list in &mut self.lists {
            if x >= list.left && x <= list.right && y >= list.top && y <= list.bottom {
                list.mouse_wheel(delta);
                return;
            }
        }
    }

    fn header_position(&self, column: PackColumn) -> (i32, i32) {
        let list = self.list(column);
        let row_x = list.left + list.width / 2 - list.list_width / 2 + 2;
        let scrolled_y = list.top + 4 - list.scroll_pixels();
        (row_x + list.width / 2, (list.top + 3).min(scrolled_y))
    }

    fn display_name(font: &FontRenderer, name: &str) -> String {
        if font.string_width(name) <= PACK_TEXT_WIDTH {
            name.to_owned()
        } else {
            let ellipsis = "...";
            format!(
                "{}{}",
                font.trim_to_width(name, PACK_TEXT_WIDTH - font.string_width(ellipsis)),
                ellipsis
            )
        }
    }

    fn entry_name(entry: &PackEntry, hovered: bool, lang: &Language) -> String {
        if hovered && !entry.compatible {
            lang.tr("resourcePack.incompatible").to_owned()
        } else {
            entry.name.clone()
        }
    }

    fn description_lines(
        entry: &PackEntry,
        hovered: bool,
        lang: &Language,
        font: &FontRenderer,
    ) -> Vec<String> {
        let description = if hovered && !entry.compatible {
            entry.description_line(lang)
        } else {
            entry.description.clone()
        };
        font.wrap_to_lines(&description, PACK_TEXT_WIDTH)
            .into_iter()
            .take(2)
            .collect()
    }

    fn description_y(row_y: i32, line: usize) -> i32 {
        row_y + 12 + 10 * line as i32
    }

    fn draw_entry_overlay(
        &self,
        d: &mut DrawList,
        column: PackColumn,
        index: usize,
        row: Rect,
        mouse: (i32, i32),
    ) {
        if !self.column(column)[index].has_controls()
            || (!self.touchscreen
                && self.list(column).slot_index_at(mouse.0, mouse.1) != Some(index))
        {
            return;
        }

        let texture = UiTexture::gui(RESOURCE_PACKS_TEXTURE);
        let relative_x = mouse.0 - row.x;
        let relative_y = mouse.1 - row.y;
        d.rect(row.x, row.y, row.x + 32, row.y + 32, 0xA090_9090);

        let mut arrow = |u: f32, active: bool| {
            d.blit_sized(
                &texture,
                row.x,
                row.y,
                u,
                if active { 32.0 } else { 0.0 },
                32,
                32,
                256.0,
                256.0,
            );
        };
        match column {
            PackColumn::Available => arrow(0.0, relative_x < 32),
            PackColumn::Selected => {
                arrow(32.0, relative_x < 16);
                if index > 0 && self.selected[index - 1].has_controls() {
                    arrow(96.0, relative_x > 16 && relative_x < 32 && relative_y < 16);
                }
                if index + 1 < self.selected.len() && self.selected[index + 1].has_controls() {
                    arrow(64.0, relative_x > 16 && relative_x < 32 && relative_y > 16);
                }
            }
        }
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
        if let Some(confirmation) = &mut self.confirmation {
            confirmation.dialog.draw_dialog(d, font, mouse);
            return;
        }

        draw_dirt_background(d, ctx.width, ctx.height, 0.0);
        for column in [PackColumn::Available, PackColumn::Selected] {
            let c = column as usize;
            self.lists[c].draw_background(d);

            let header = match column {
                PackColumn::Available => lang.tr("resourcePack.available.title"),
                PackColumn::Selected => lang.tr("resourcePack.selected.title"),
            };
            let (header_x, header_y) = self.header_position(column);
            draw_centered_string(
                d,
                font,
                &format!("\u{a7}n\u{a7}l{header}"),
                header_x,
                header_y,
                TITLE_COLOR,
            );

            let rows: Vec<(usize, Rect, PackEntry, bool)> = self.lists[c]
                .visible_slots()
                .into_iter()
                .map(|i| {
                    let p = &self.column(column)[i];
                    (
                        i,
                        self.row_rect(column, i),
                        p.clone(),
                        self.touchscreen
                            || self.lists[c].slot_index_at(mouse.0, mouse.1) == Some(i),
                    )
                })
                .collect();
            for (index, r, entry, hovered) in rows {
                if !entry.compatible {
                    d.rect(
                        r.x - 1,
                        r.y - 1,
                        r.x + COLUMN_WIDTH - 9,
                        r.y + r.height + 1,
                        INCOMPATIBLE_PLATE,
                    );
                }

                if let Some(icon) = entry.icon.as_ref().or(self.default_icon.as_ref()) {
                    d.blit_sized(icon, r.x, r.y, 0.0, 0.0, 32, 32, 32.0, 32.0);
                }
                self.draw_entry_overlay(d, column, index, r, mouse);

                let name = Self::display_name(font, &Self::entry_name(&entry, hovered, lang));
                let description = Self::description_lines(&entry, hovered, lang, font);
                draw_string(d, font, &name, r.x + 32 + 2, r.y + 1, TITLE_COLOR);
                for (i, line) in description.iter().enumerate() {
                    draw_string(
                        d,
                        font,
                        line,
                        r.x + 32 + 2,
                        Self::description_y(r.y, i),
                        DESCRIPTION_COLOR,
                    );
                }
            }
            self.lists[c].draw_overlay(d);
        }
        draw_centered_string(
            d,
            font,
            lang.tr("resourcePack.title"),
            ctx.width / 2,
            16,
            TITLE_COLOR,
        );
        draw_centered_string(
            d,
            font,
            lang.tr("resourcePack.folderInfo"),
            ctx.width / 2 - 77,
            ctx.height - 26,
            DESCRIPTION_COLOR,
        );
        self.open_folder.draw(d, font, mouse.0, mouse.1);
        self.done.draw(d, font, mouse.0, mouse.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::{ScaledResolution, OPTIONS_BACKGROUND};

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn found() -> Vec<PackEntry> {
        vec![
            PackEntry::new("alpha.zip", "first", true),
            PackEntry::new("beta.zip", "second", true),
            PackEntry::new("old.zip", "ancient", false),
        ]
    }

    /// The load-bearing one. `options.txt` lists packs lowest-priority first; the screen shows
    /// them highest first. A port that skipped the reversal would silently invert every
    /// override.
    #[test]
    fn the_selected_column_is_the_reverse_of_the_settings_order() {
        let s = ResourcePacksScreen::new(found(), &["alpha.zip".into(), "beta.zip".into()]);
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["beta.zip", "alpha.zip", "Default"],
            "the last entry in options.txt wins, so it is shown on top"
        );
        assert_eq!(s.to_settings_order(), ["alpha.zip", "beta.zip"]);
        assert_eq!(
            s.available
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["old.zip"]
        );
    }

    #[test]
    fn a_settings_round_trip_is_lossless() {
        let order = vec!["alpha.zip".to_owned(), "old.zip".to_owned()];
        let s = ResourcePacksScreen::new(found(), &order);
        assert_eq!(s.to_settings_order(), order);
        let mut settings = GameSettings::default();
        s.apply(&mut settings);
        assert_eq!(settings.resource_packs, order);
        assert_eq!(settings.incompatible_resource_packs, ["old.zip"]);
    }

    #[test]
    fn enabling_a_pack_puts_it_at_the_top_of_the_priority_order() {
        let mut s = ResourcePacksScreen::new(found(), &["alpha.zip".into()]);
        assert_eq!(s.selected.len(), 2, "the Default row is always the floor");
        let i = s
            .available
            .iter()
            .position(|p| p.name == "beta.zip")
            .unwrap();
        assert!(s.enable(i));
        assert_eq!(s.selected[0].name, "beta.zip");
        assert_eq!(s.to_settings_order(), ["alpha.zip", "beta.zip"]);
        assert!(s.dirty);
    }

    #[test]
    fn disabling_returns_a_pack_to_the_available_column() {
        let mut s = ResourcePacksScreen::new(found(), &["alpha.zip".into(), "beta.zip".into()]);
        assert!(s.disable(0));
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha.zip", "Default"]
        );
        assert_eq!(
            s.available
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["beta.zip", "old.zip"],
            "vanilla returns a disabled pack to index zero, not the bottom"
        );
        assert!(!s.disable(99));
        assert!(!s.enable(99));
    }

    #[test]
    fn reordering_moves_one_place_and_stops_at_the_ends() {
        let mut s = ResourcePacksScreen::new(
            found(),
            &["alpha.zip".into(), "beta.zip".into(), "old.zip".into()],
        );
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["old.zip", "beta.zip", "alpha.zip", "Default"]
        );
        assert!(!s.reorder(0, true));
        assert!(!s.reorder(2, false));
        assert!(s.reorder(0, false));
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["beta.zip", "old.zip", "alpha.zip", "Default"]
        );
    }

    #[test]
    fn an_unknown_enabled_pack_is_dropped_rather_than_faked() {
        // A pack named in options.txt that is no longer on disk simply disappears; vanilla does
        // the same, and inventing an entry for it would write a phantom back on save.
        let s = ResourcePacksScreen::new(found(), &["gone.zip".into(), "alpha.zip".into()]);
        assert_eq!(s.to_settings_order(), ["alpha.zip"]);
    }

    #[test]
    fn incompatible_warning_age_comes_from_the_raw_pack_format_without_colour_codes() {
        let mut lang = Language::new("en_US");
        lang.insert("resourcePack.incompatible", "Incompatible");
        lang.insert(
            "resourcePack.incompatible.old",
            "(Made for an older version)",
        );
        lang.insert(
            "resourcePack.incompatible.new",
            "(Made for a newer version)",
        );
        let p = PackEntry::new("old.zip", "ancient", false);
        assert_eq!(INCOMPATIBLE_PLATE as i32, -8_978_432);
        let line = p.description_line(&lang);
        assert_eq!(line, "(Made for an older version)");
        assert!(
            !line.contains('\u{a7}'),
            "the empty ChatStyle adds no colour code"
        );
        let future = PackEntry::new("future.zip", "future", true).with_pack_format(99);
        assert!(!future.compatible);
        assert!(future
            .description_line(&lang)
            .ends_with("(Made for a newer version)"));
        let ok = PackEntry::new("a.zip", "fine", true);
        assert_eq!(ok.description_line(&lang), "fine");
    }

    #[test]
    fn the_two_columns_sit_either_side_of_the_centre() {
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &Language::new("en_US"));
        assert_eq!(
            (s.lists[0].left, s.lists[1].left),
            (854 / 2 - 204, 854 / 2 + 4)
        );
        let a = s.row_rect(PackColumn::Available, 0);
        assert_eq!((a.x, a.right()), (854 / 2 - 202, 854 / 2 - 2));
        let b = s.row_rect(PackColumn::Selected, 0);
        assert_eq!((b.x, b.right()), (854 / 2 + 6, 854 / 2 + 206));
        assert_eq!(a.y, 32 + 4 + HEADER_PADDING);
        assert_eq!(s.row_rect(PackColumn::Available, 1).y, 85);
        assert_eq!(s.lists[0].scrollbar_x(), 854 / 2 - 10);
        assert_eq!(s.lists[1].scrollbar_x(), 854 / 2 + 198);
        assert_eq!(s.done.rect.x, 854 / 2 + 4);
        assert_eq!(s.open_folder.rect.x, 854 / 2 - 154);
        assert_eq!(s.done.rect.y, 480 - 48);
    }

    /// java: `actionPerformed` case 1 leaves for `parentGuiScreen`. Applying the selection and
    /// reloading the packs is the *caller's* half — it owns `GameSettings` and the pack stack —
    /// so what the screen reports is only "I am done".
    #[test]
    fn done_leaves_for_the_parent_screen() {
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &Language::new("en_US"));
        let (x, y) = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::Back);
    }

    /// The button has its own intent: an `OpenUrl` value can also come
    /// from a server's chat or book link.
    #[test]
    fn open_pack_folder_asks_for_the_folder_not_a_url() {
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &Language::new("en_US"));
        let (x, y) = (s.open_folder.rect.x + 1, s.open_folder.rect.y + 1);
        assert_eq!(s.mouse_clicked(x, y, 0), Action::OpenResourcePackFolder);
    }

    /// The wheel picks its column from the cursor's x, exactly as two independent `GuiSlot`s do.
    #[test]
    fn the_wheel_scrolls_only_the_column_the_cursor_is_over() {
        let many: Vec<PackEntry> = (0..12)
            .map(|i| PackEntry::new(format!("pack{i}.zip"), "d", true))
            .collect();
        let enabled: Vec<String> = (0..6).map(|i| format!("pack{i}.zip")).collect();
        let mut s = ResourcePacksScreen::new(many, &enabled);
        // A short screen so both columns really do overflow.
        let short = ScreenCtx::new(ScaledResolution::new(854, 200, 1, false), 0.0);
        s.init(&short, &Language::new("en_US"));
        let before = (
            s.row_rect(PackColumn::Available, 0).y,
            s.row_rect(PackColumn::Selected, 0).y,
        );
        s.mouse_wheel(854 / 2 + 100, 50, -1.0);
        let after = (
            s.row_rect(PackColumn::Available, 0).y,
            s.row_rect(PackColumn::Selected, 0).y,
        );
        assert_eq!(before.0, after.0, "the left column must not move");
        assert!(after.1 < before.1, "the right column scrolled down");
        // Outside the list bounds the wheel does nothing at all.
        let held = s.row_rect(PackColumn::Selected, 0).y;
        s.mouse_wheel(854 / 2 + 100, 5, -1.0);
        assert_eq!(s.row_rect(PackColumn::Selected, 0).y, held);
    }

    #[test]
    fn the_header_padding_moves_rows_and_extends_the_scroll_range_by_thirteen() {
        let many: Vec<PackEntry> = (0..6)
            .map(|i| PackEntry::new(format!("pack{i}.zip"), "d", true))
            .collect();
        let short = ScreenCtx::new(ScaledResolution::new(854, 200, 1, false), 0.0);
        let mut s = ResourcePacksScreen::new(many, &[]);
        s.init(&short, &Language::new("en_US"));
        let list = &s.lists[PackColumn::Available as usize];
        assert_eq!(list.header_padding, 13);
        assert!(list.has_header);
        assert_eq!(list.content_height(), 6 * ROW_HEIGHT + 13);
        assert_eq!(list.max_scroll(), 6 * ROW_HEIGHT + 13 - (149 - 32 - 4));
        assert_eq!(s.row_rect(PackColumn::Available, 0).y, 49);
        assert_eq!(s.header_position(PackColumn::Available).1, 35);
    }

    #[test]
    fn only_the_icon_square_moves_a_pack_and_selected_quadrants_reorder_it() {
        let mut s = ResourcePacksScreen::new(
            found(),
            &["alpha.zip".into(), "beta.zip".into(), "old.zip".into()],
        );
        s.init(&ctx(), &Language::new("en_US"));

        let middle = s.row_rect(PackColumn::Selected, 1);
        let before = s.to_settings_order();
        s.mouse_clicked(middle.x + 33, middle.y + 4, 0);
        assert_eq!(
            s.to_settings_order(),
            before,
            "pixel 33 is outside the icon"
        );

        s.mouse_clicked(middle.x + 17, middle.y + 4, 0);
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["beta.zip", "old.zip", "alpha.zip", "Default"],
            "upper-right moves the middle row one place up"
        );

        let top = s.row_rect(PackColumn::Selected, 0);
        let held = s.to_settings_order();
        s.mouse_clicked(top.x + 16, top.y + 4, 0);
        assert_eq!(
            s.to_settings_order(),
            held,
            "the strict x=16 boundary is intentionally inert"
        );

        let top = s.row_rect(PackColumn::Selected, 0);
        s.mouse_clicked(top.x + 17, top.y + 17, 0);
        assert_eq!(
            s.selected
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            ["old.zip", "beta.zip", "alpha.zip", "Default"],
            "lower-right moves the top row down"
        );

        let second = s.row_rect(PackColumn::Selected, 1);
        s.mouse_clicked(second.x + 15, second.y + 17, 0);
        assert_eq!(
            s.available[0].name, "beta.zip",
            "left half disables to index zero"
        );
    }

    #[test]
    fn clicking_an_available_icon_enables_it_but_clicking_the_text_does_not() {
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &Language::new("en_US"));
        let r = s.row_rect(PackColumn::Available, 0);
        s.mouse_clicked(r.x + 80, r.y + 4, 0);
        assert_eq!(s.to_settings_order(), Vec::<String>::new());
        s.mouse_clicked(r.x + 32, r.y + 4, 0);
        assert_eq!(s.selected[0].name, "alpha.zip", "relative x=32 is included");
    }

    /// `GuiListExtended` starts the entry's relative-x origin two pixels right of the list's
    /// content bound, while `ResourcePackListEntry.mousePressed` accepts every `relativeX <=
    /// 32`. The otherwise surprising left fringe and four-pixel inter-row gap are therefore
    /// actionable; the next pixel outside the list and the first pixel beyond the icon are not.
    #[test]
    fn pack_controls_keep_the_java_list_origin_and_icon_boundaries() {
        let mut left_fringe = ResourcePacksScreen::new(found(), &[]);
        left_fringe.init(&ctx(), &Language::new("en_US"));
        let row = left_fringe.row_rect(PackColumn::Available, 0);
        left_fringe.mouse_clicked(row.x - 2, row.y + 34, 0);
        assert_eq!(
            left_fringe.selected[0].name, "alpha.zip",
            "relative x=-2 and y=34 reach the entry exactly as GuiListExtended does"
        );

        let mut outside_list = ResourcePacksScreen::new(found(), &[]);
        outside_list.init(&ctx(), &Language::new("en_US"));
        let row = outside_list.row_rect(PackColumn::Available, 0);
        outside_list.mouse_clicked(row.x - 3, row.y + 4, 0);
        assert!(
            outside_list.to_settings_order().is_empty(),
            "one pixel left of GuiSlot's content bound is not an entry click"
        );

        let mut beyond_icon = ResourcePacksScreen::new(found(), &[]);
        beyond_icon.init(&ctx(), &Language::new("en_US"));
        let row = beyond_icon.row_rect(PackColumn::Available, 0);
        beyond_icon.mouse_clicked(row.x + 33, row.y + 4, 0);
        assert!(
            beyond_icon.to_settings_order().is_empty(),
            "the exact <= 32 icon test must not expand into the pack text"
        );
    }

    #[test]
    fn entry_controls_accept_every_mouse_button_but_screen_buttons_are_left_only() {
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &Language::new("en_US"));
        let available = s.row_rect(PackColumn::Available, 0);
        s.mouse_clicked(available.x + 4, available.y + 4, 1);
        assert_eq!(
            s.selected[0].name, "alpha.zip",
            "ResourcePackListEntry ignores the forwarded mouseEvent argument"
        );

        let selected = s.row_rect(PackColumn::Selected, 0);
        s.mouse_clicked(selected.x + 4, selected.y + 4, 2);
        assert!(
            s.to_settings_order().is_empty(),
            "middle click reaches the left-arrow zone too"
        );

        let done = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(s.mouse_clicked(done.0, done.1, 1), Action::None);
    }

    #[test]
    fn a_content_drag_scrolls_one_to_one_and_release_disarms_it() {
        let many: Vec<PackEntry> = (0..12)
            .map(|i| PackEntry::new(format!("pack{i}.zip"), "d", true))
            .collect();
        let short = ScreenCtx::new(ScaledResolution::new(854, 200, 1, false), 0.0);
        let mut s = ResourcePacksScreen::new(many, &[]);
        s.init(&short, &Language::new("en_US"));
        let row = s.row_rect(PackColumn::Available, 0);
        s.mouse_clicked(row.x + 100, row.y + 4, 0);
        s.mouse_dragged(row.x + 100, row.y - 16);
        assert_eq!(s.lists[0].scroll_pixels(), 20);
        s.mouse_released();
        s.mouse_dragged(row.x + 100, row.y - 40);
        assert_eq!(s.lists[0].scroll_pixels(), 20, "release ends the drag");
    }

    #[test]
    fn a_pack_drag_stops_at_the_column_boundary_until_the_pointer_reenters() {
        let many: Vec<PackEntry> = (0..12)
            .map(|i| PackEntry::new(format!("pack{i}.zip"), "d", true))
            .collect();
        let short = ScreenCtx::new(ScaledResolution::new(854, 200, 1, false), 0.0);
        let mut s = ResourcePacksScreen::new(many, &[]);
        s.init(&short, &Language::new("en_US"));
        let row = s.row_rect(PackColumn::Available, 0);
        s.mouse_clicked(row.x + 100, row.y + 4, 0);
        s.mouse_dragged(row.x + 100, row.y - 16);
        assert_eq!(s.lists[0].scroll_pixels(), 20);

        // Java's `isMouseYWithinSlotBounds` rejects this x even though y remains inside the
        // viewport. A pre-fix implementation that forwarded y unconditionally moved the list to
        // 12 here, making the next re-entry jump by the wrong delta.
        s.mouse_dragged(row.x - 3, row.y - 8);
        assert_eq!(
            s.lists[0].scroll_pixels(),
            20,
            "leaving the column horizontally must not advance its drag"
        );

        // Re-entering resumes from the last in-bounds y anchor, as GuiSlot does.
        s.mouse_dragged(row.x + 100, row.y - 8);
        assert_eq!(s.lists[0].scroll_pixels(), 12);
    }

    #[test]
    fn pack_text_uses_vanillas_157_pixel_trim_wrap_and_ten_pixel_pitch() {
        let font = uniform_font();
        let name = "a".repeat(40);
        let shown = ResourcePacksScreen::display_name(&font, &name);
        assert!(shown.ends_with("..."));
        assert!(font.string_width(&shown) <= PACK_TEXT_WIDTH);
        assert_ne!(shown, name, "the untrimmed-name mutant must bite");

        let entry = PackEntry::new("p", "one two three four five six seven", true);
        let lines =
            ResourcePacksScreen::description_lines(&entry, false, &Language::new("en_US"), &font);
        assert_eq!(lines.len(), 2);
        assert!(lines
            .iter()
            .all(|line| font.string_width(line) <= PACK_TEXT_WIDTH));
        assert_eq!(
            ResourcePacksScreen::description_y(100, 1),
            122,
            "the second line is y+22, never y+21"
        );
        assert_eq!(DESCRIPTION_COLOR, 0x0080_8080);
    }

    #[test]
    fn an_incompatible_warning_replaces_the_description_only_while_hovered() {
        let mut lang = Language::new("en_US");
        lang.insert("resourcePack.incompatible", "Incompatible");
        lang.insert("resourcePack.incompatible.old", "Old version");
        let entry = PackEntry::new("old.zip", "original description", false);
        let font = uniform_font();
        assert_eq!(
            ResourcePacksScreen::description_lines(&entry, false, &lang, &font),
            ["original description"]
        );
        let hovered = ResourcePacksScreen::description_lines(&entry, true, &lang, &font);
        assert_eq!(hovered, ["Old version"]);
        assert_eq!(
            ResourcePacksScreen::entry_name(&entry, true, &lang),
            "Incompatible"
        );
        assert_eq!(
            ResourcePacksScreen::entry_name(&entry, false, &lang),
            "old.zip"
        );
    }

    #[test]
    fn an_incompatible_pack_requires_the_exact_old_or_new_confirmation() {
        let mut lang = Language::new("en_US");
        lang.insert(
            "resourcePack.incompatible.confirm.title",
            "Confirm incompatible pack",
        );
        lang.insert(
            "resourcePack.incompatible.confirm.old",
            "This is the OLD warning",
        );
        lang.insert(
            "resourcePack.incompatible.confirm.new",
            "This is the much longer NEW warning for a future pack",
        );
        lang.insert("gui.yes", "Yes");
        lang.insert("gui.no", "No");

        let packs = vec![
            PackEntry::new("old.zip", "old", true).with_pack_format(0),
            PackEntry::new("new.zip", "new", true).with_pack_format(2),
        ];
        let mut s = ResourcePacksScreen::new(packs, &[]);
        s.init(&ctx(), &lang);

        let old = s.row_rect(PackColumn::Available, 0);
        assert_eq!(s.mouse_clicked(old.x + 4, old.y + 4, 0), Action::None);
        assert!(s.confirmation_open());
        assert!(
            s.to_settings_order().is_empty(),
            "opening GuiYesNo must not enable the pack"
        );
        assert!(s.dirty, "vanilla markChanged runs before the confirmation");

        let mut old_draw = DrawList::new();
        s.draw(&mut old_draw, &mut uniform_font(), &ctx(), (0, 0), &lang);
        let old_vertices = old_draw.vertices().len();

        // GuiYesNo's cancel button is width/2+5 .. +155 at height/6+96 .. +116.
        s.mouse_clicked(854 / 2 + 6, 480 / 6 + 97, 0);
        assert!(!s.confirmation_open());
        assert!(
            s.to_settings_order().is_empty(),
            "cancel leaves the pack available"
        );

        let new = s.row_rect(PackColumn::Available, 1);
        s.mouse_clicked(new.x + 4, new.y + 4, 0);
        assert!(s.confirmation_open());
        let mut new_draw = DrawList::new();
        s.draw(&mut new_draw, &mut uniform_font(), &ctx(), (0, 0), &lang);
        assert_ne!(
            old_vertices,
            new_draw.vertices().len(),
            "the distinct old/new localized messages must reach the dialog renderer"
        );

        // Confirm is width/2-155 .. -5 at the same y.
        s.mouse_clicked(854 / 2 - 154, 480 / 6 + 97, 0);
        assert!(!s.confirmation_open());
        assert_eq!(s.selected[0].name, "new.zip");
        assert_eq!(s.available[0].name, "old.zip");
    }

    #[test]
    fn draw_emits_each_columns_gui_slot_chrome_and_the_hover_arrow_texture() {
        let mut lang = Language::new("en_US");
        lang.insert("resourcePack.available.title", "Available Resource Packs");
        lang.insert("resourcePack.selected.title", "Selected Resource Packs");
        lang.insert("resourcePack.folderInfo", "Place resource pack files here");
        lang.insert("resourcePack.title", "Select Resource Packs");
        let mut s = ResourcePacksScreen::new(found(), &[]);
        s.init(&ctx(), &lang);
        let row = s.row_rect(PackColumn::Available, 0);
        let mut d = DrawList::new();
        let mut font = uniform_font();
        s.draw(&mut d, &mut font, &ctx(), (row.x + 4, row.y + 4), &lang);

        let paths: Vec<_> = d
            .batches()
            .iter()
            .filter_map(|batch| match &batch.texture {
                UiTexture::Resource(location) => Some(location.path.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            d.batches()
                .iter()
                .filter(|batch| {
                    matches!(&batch.texture, UiTexture::Resource(location) if location.path == OPTIONS_BACKGROUND)
                })
                .map(|batch| batch.range.len())
                .sum::<usize>()
                >= 7 * 6,
            "full-screen dirt plus two column backgrounds and four overlay bands"
        );
        assert!(
            paths.contains(&RESOURCE_PACKS_TEXTURE),
            "hovering a row emits the resource_packs.png arrow art"
        );
        assert!(
            s.lists[0].scrollbar().is_none(),
            "the short list needs no bar"
        );
    }

    #[test]
    fn an_uploaded_pack_icon_is_blitted_at_exactly_thirty_two_by_thirty_two() {
        let icon = UiTexture::gui("dynamic/test_pack_icon.png");
        let entry = PackEntry::new("pack.zip", "description", true).with_icon(icon.clone());
        let mut s = ResourcePacksScreen::new(vec![entry], &[]);
        s.init(&ctx(), &Language::new("en_US"));
        let row = s.row_rect(PackColumn::Available, 0);
        let mut d = DrawList::new();
        s.draw(
            &mut d,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &Language::new("en_US"),
        );
        let batch = d
            .batches()
            .iter()
            .find(|batch| batch.texture == icon)
            .expect("pack icon batch");
        let vertices = &d.vertices()[batch.range.clone()];
        let min_x = vertices
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::INFINITY, f32::min);
        let max_x = vertices
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = vertices
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            (min_x, min_y, max_x, max_y),
            (
                row.x as f32,
                row.y as f32,
                (row.x + 32) as f32,
                (row.y + 32) as f32
            )
        );
    }

    #[test]
    fn a_missing_pack_icon_uses_the_builtin_pack_image_but_never_replaces_a_real_icon() {
        let fallback = UiTexture::gui("dynamic/default_pack_icon.png");
        let own = UiTexture::gui("dynamic/own_pack_icon.png");
        let entries = vec![
            PackEntry::new("missing.zip", "description", true),
            PackEntry::new("present.zip", "description", true).with_icon(own.clone()),
        ];
        let mut s = ResourcePacksScreen::new(entries, &[]).with_default_icon(fallback.clone());
        s.init(&ctx(), &Language::new("en_US"));
        let mut d = DrawList::new();
        s.draw(
            &mut d,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &Language::new("en_US"),
        );

        assert_eq!(
            d.batches()
                .iter()
                .filter(|batch| batch.texture == fallback)
                .count(),
            2,
            "the missing user icon and built-in Default row both use the built-in image"
        );
        assert_eq!(
            d.batches()
                .iter()
                .filter(|batch| batch.texture == own)
                .count(),
            1,
            "an uploaded pack.png wins over the fallback"
        );
    }

    #[test]
    fn the_default_row_is_a_permanent_unarrowed_floor_and_never_reaches_settings() {
        let fallback = UiTexture::gui("dynamic/default_pack_icon.png");
        let mut s = ResourcePacksScreen::new(Vec::new(), &[])
            .with_default_description("The built-in resources")
            .with_default_icon(fallback.clone())
            .with_touchscreen(true);
        s.init(&ctx(), &Language::new("en_US"));

        assert_eq!(s.selected.len(), 1);
        assert!(s.selected[0].is_default());
        assert_eq!(s.selected[0].name, "Default");
        assert_eq!(s.selected[0].description, "The built-in resources");
        assert!(s.to_settings_order().is_empty());
        let mut settings = GameSettings::default();
        s.apply(&mut settings);
        assert!(settings.resource_packs.is_empty());
        assert!(settings.incompatible_resource_packs.is_empty());

        let row = s.row_rect(PackColumn::Selected, 0);
        assert_eq!(s.hit(row.x + 4, row.y + 4), None);
        s.mouse_clicked(row.x + 4, row.y + 4, 0);
        assert_eq!(s.selected.len(), 1, "the default row cannot be removed");
        assert!(!s.dirty, "clicking the inert row cannot schedule a reload");

        let mut draw = DrawList::new();
        s.draw(
            &mut draw,
            &mut uniform_font(),
            &ctx(),
            (row.x + 4, row.y + 4),
            &Language::new("en_US"),
        );
        assert_eq!(
            draw.batches()
                .iter()
                .filter(|batch| batch.texture == fallback)
                .count(),
            1,
            "the default row draws the built-in pack image"
        );
        assert!(
            !draw.batches().iter().any(|batch| {
                matches!(&batch.texture, UiTexture::Resource(location) if location.path == RESOURCE_PACKS_TEXTURE)
            }),
            "func_148310_d=false suppresses arrow art even on touchscreen"
        );

        let mut with_user = ResourcePacksScreen::new(found(), &["alpha.zip".to_owned()]);
        with_user.init(&ctx(), &Language::new("en_US"));
        let user = with_user.row_rect(PackColumn::Selected, 0);
        with_user.mouse_clicked(user.x + 17, user.y + 17, 0);
        assert_eq!(
            with_user
                .selected
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["alpha.zip", "Default"],
            "the last user pack cannot move below ResourcePackListEntryDefault"
        );
        assert!(!with_user.dirty, "the blocked move is not a change");
    }

    #[test]
    fn touchscreen_keeps_the_icon_controls_visible_without_a_hover() {
        fn draws_arrow(screen: &mut ResourcePacksScreen) -> bool {
            screen.init(&ctx(), &Language::new("en_US"));
            let mut d = DrawList::new();
            screen.draw(
                &mut d,
                &mut uniform_font(),
                &ctx(),
                (0, 0),
                &Language::new("en_US"),
            );
            d.batches().iter().any(|batch| {
                matches!(&batch.texture, UiTexture::Resource(location) if location.path == RESOURCE_PACKS_TEXTURE)
            })
        }

        let mut ordinary = ResourcePacksScreen::new(found(), &[]);
        assert!(!draws_arrow(&mut ordinary));
        let mut touch = ResourcePacksScreen::new(found(), &[]).with_touchscreen(true);
        assert!(draws_arrow(&mut touch));
    }

    #[test]
    fn touchscreen_uses_incompatibility_warning_text_without_pointer_hover() {
        fn font_vertices(draw: &DrawList) -> usize {
            draw.batches()
                .iter()
                .filter(|batch| matches!(&batch.texture, UiTexture::Font(_)))
                .map(|batch| batch.range.len())
                .sum()
        }

        let mut lang = Language::new("en_US");
        lang.insert("resourcePack.incompatible", "Incompatible");
        lang.insert(
            "resourcePack.incompatible.old",
            "Made for an older version of Minecraft",
        );
        let entry = PackEntry::new("p", "short", false);

        let mut ordinary = ResourcePacksScreen::new(vec![entry.clone()], &[]);
        ordinary.init(&ctx(), &lang);
        let mut ordinary_draw = DrawList::new();
        ordinary.draw(
            &mut ordinary_draw,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &lang,
        );

        let mut touchscreen = ResourcePacksScreen::new(vec![entry], &[]).with_touchscreen(true);
        touchscreen.init(&ctx(), &lang);
        let mut touchscreen_draw = DrawList::new();
        touchscreen.draw(
            &mut touchscreen_draw,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &lang,
        );

        assert!(
            font_vertices(&touchscreen_draw) > font_vertices(&ordinary_draw),
            "Java treats touchscreen rows as selected, so incompatible warning name/description must replace the normal text"
        );
    }

    #[test]
    fn draw_really_emits_both_translated_header_strings() {
        fn language(headers: (&str, &str)) -> Language {
            let mut lang = Language::new("en_US");
            lang.insert("resourcePack.available.title", headers.0);
            lang.insert("resourcePack.selected.title", headers.1);
            lang.insert("resourcePack.folderInfo", "");
            lang.insert("resourcePack.title", "");
            lang.insert("resourcePack.openFolder", "");
            lang.insert("gui.done", "");
            lang
        }

        let mut with = ResourcePacksScreen::new(Vec::new(), &[]);
        with.init(&ctx(), &language(("A", "B")));
        let mut with_draw = DrawList::new();
        with.draw(
            &mut with_draw,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &language(("A", "B")),
        );

        let mut without = ResourcePacksScreen::new(Vec::new(), &[]);
        without.init(&ctx(), &language(("", "")));
        let mut without_draw = DrawList::new();
        without.draw(
            &mut without_draw,
            &mut uniform_font(),
            &ctx(),
            (0, 0),
            &language(("", "")),
        );

        assert_eq!(
            with_draw.vertices().len() - without_draw.vertices().len(),
            2 * 2 * (2 * 6 + 6),
            "two bold-underlined one-glyph headers: doubled glyph plus underline, in both shadow and foreground passes"
        );
    }

    #[test]
    fn overflowing_columns_get_independent_right_inset_scrollbars() {
        let many: Vec<PackEntry> = (0..12)
            .map(|i| PackEntry::new(format!("pack{i}.zip"), "d", true))
            .collect();
        let enabled: Vec<String> = (0..6).map(|i| format!("pack{i}.zip")).collect();
        let short = ScreenCtx::new(ScaledResolution::new(854, 200, 1, false), 0.0);
        let mut s = ResourcePacksScreen::new(many, &enabled);
        s.init(&short, &Language::new("en_US"));
        for list in &s.lists {
            let bar = list.scrollbar().expect("six rows plus header overflow");
            assert_eq!(bar.x, list.right - 6);
            assert_eq!((bar.track_top, bar.track_bottom), (32, 149));
        }
    }
}
