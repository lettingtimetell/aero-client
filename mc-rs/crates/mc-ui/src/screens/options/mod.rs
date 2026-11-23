//! The options screens: root, video, controls, sound, chat, language, resource packs.
//!
//! java: `client/gui/GuiOptions.java`, `GuiVideoSettings.java`, `GuiControls.java`,
//! `GuiLanguage.java`, `GuiScreenResourcePacks.java`, `GuiScreenOptionsSounds.java`,
//! `ScreenChatOptions.java`, `GuiOptionsRowList.java`, `GuiOptionButton.java`,
//! `GuiOptionSlider.java`, `GuiSnooper.java`
//!
//! [`model`] holds the option table — what an option is called, what it is worth, and what
//! pressing it does. Everything here is layout on top of that.

pub mod controls;
pub mod language;
pub mod model;
pub mod packs;
pub mod sounds;

use mc_assets::Language;
use mc_protocol::play::s2c::ServerDifficultyLevel;
use mc_render::settings::GameSettings;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::{draw_centered_string, draw_string, Button, ScrollList, ScrollbarX, Slider};

pub use model::{OptionId, OptionKind};

/// Which options sub-screen to open.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OptionsPage {
    /// `GuiVideoSettings`
    Video,
    /// `GuiControls`
    Controls,
    /// `GuiScreenOptionsSounds`
    Sound,
    /// `ScreenChatOptions`
    Chat,
    /// `GuiLanguage`
    Language,
    /// `GuiScreenResourcePacks`
    ResourcePacks,
    /// `GuiSnooper`
    Snooper,
    /// `GuiCustomizeSkin`
    SkinCustomisation,
}

/// The options root screen.
// java: GuiOptions.java:18
#[derive(Debug)]
pub struct OptionsScreen {
    fov: Slider,
    /// The server-retained difficulty shown by the in-world, read-only row.
    difficulty: ServerDifficultyLevel,
    difficulty_button: Option<Button>,
    buttons: Vec<Button>,
    layout_size: Option<(i32, i32)>,
    /// Whether a world is loaded, which decides the background and the difficulty button.
    pub in_world: bool,
}

/// Button ids, matching vanilla's.
mod ids {
    pub const CONTROLS: i32 = 100;
    pub const VIDEO: i32 = 101;
    pub const LANGUAGE: i32 = 102;
    pub const CHAT: i32 = 103;
    pub const SNOOPER: i32 = 104;
    pub const RESOURCE_PACKS: i32 = 105;
    pub const SOUNDS: i32 = 106;
    pub const DIFFICULTY: i32 = 108;
    pub const SKIN: i32 = 110;
    /// Vanilla's "Super Secret Settings..." button. It is no longer laid
    /// out: the Super Secret Settings mod (Graphics window) picks a shader from a list instead.
    /// Kept so the tests can name the empty slot.
    #[cfg(test)]
    pub const SUPER_SECRET: i32 = 8_675_309;
    pub const DONE: i32 = 200;
    /// "Accounts...": the Accounts screen (no vanilla equivalent),
    /// in the empty slot vanilla's stream button had, so it is reachable without the Aero title
    /// screen's account corner too. An id vanilla never uses.
    pub const ACCOUNTS: i32 = 9_100;
}

impl Default for OptionsScreen {
    fn default() -> Self {
        Self::new(false)
    }
}

impl OptionsScreen {
    /// A root options screen.
    #[must_use]
    pub fn new(in_world: bool) -> Self {
        Self::new_with_difficulty(in_world, ServerDifficultyLevel::Normal)
    }

    /// Build an options screen with the latest server difficulty snapshot.
    ///
    /// `GuiOptions` displays this row only over a world, and multiplayer cannot change it, so
    /// the control is deliberately disabled. The plain [`Self::new`] constructor keeps the
    /// vanilla default (`Normal`) for callers that have not received a world snapshot yet.
    #[must_use]
    pub fn new_with_difficulty(in_world: bool, difficulty: ServerDifficultyLevel) -> Self {
        Self {
            fov: Slider::new(0, 0, 0, String::new(), 0.0, 1.0, 0.5),
            difficulty,
            difficulty_button: None,
            buttons: Vec::new(),
            layout_size: None,
            in_world,
        }
    }

    /// Replace the retained server difficulty and refresh the disabled row, if present.
    ///
    /// A `DifficultyChanged` packet may arrive while this screen is open; the driver owns that
    /// event and can call this snapshot update without making the multiplayer control clickable.
    pub fn set_difficulty(&mut self, difficulty: ServerDifficultyLevel, lang: &Language) {
        self.difficulty = difficulty;
        if let Some(button) = &mut self.difficulty_button {
            button.text = difficulty_label(difficulty, lang);
        }
    }

    /// The server difficulty snapshot backing the read-only multiplayer row.
    #[must_use]
    pub const fn difficulty(&self) -> ServerDifficultyLevel {
        self.difficulty
    }

    /// Lay out FOV plus the server-retained in-world difficulty row.
    // java: GuiOptions.java:39
    pub fn init(&mut self, ctx: &ScreenCtx, settings: &GameSettings, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        let opt = OptionId::Fov;
        let keep_drag = self.layout_size == Some((w, h)) && self.fov.dragging;
        let drag_position = self.fov.position();
        self.fov = Slider::new(
            0,
            w / 2 - 155,
            h / 6 - 12,
            opt.display_string(settings, lang),
            0.0,
            1.0,
            opt.normalize(opt.float_value(settings)),
        );
        if keep_drag {
            self.fov.set_position(drag_position);
            self.fov.dragging = true;
        }
        self.layout_size = Some((w, h));
        self.difficulty_button = self.in_world.then(|| {
            let mut button = Button::sized(
                ids::DIFFICULTY,
                w / 2 + 5,
                h / 6 - 12,
                150,
                20,
                difficulty_label(self.difficulty, lang),
            );
            // java: GuiOptions disables the difficulty button in multiplayer.
            button.enabled = false;
            button
        });
        // The stream and Realms buttons are omitted: both are permanently out of scope (§2).
        // So is "Super Secret Settings...": the client's Super Secret
        // Settings mod picks the shader from a list now, and
        // its slot right of Skin Customization stays empty like the stream button's.
        // Every remaining button keeps vanilla's exact slot.
        let tr = |k: &str| lang.tr(k).to_owned();
        self.buttons = vec![
            Button::sized(
                ids::SKIN,
                w / 2 - 155,
                h / 6 + 48 - 6,
                150,
                20,
                tr("options.skinCustomisation"),
            ),
            Button::sized(
                ids::SOUNDS,
                w / 2 - 155,
                h / 6 + 72 - 6,
                150,
                20,
                tr("options.sounds"),
            ),
            // In the stream button's slot, right of Sounds.
            Button::sized(
                ids::ACCOUNTS,
                w / 2 + 5,
                h / 6 + 72 - 6,
                150,
                20,
                crate::screens::accounts::options_button_label(lang),
            ),
            Button::sized(
                ids::VIDEO,
                w / 2 - 155,
                h / 6 + 96 - 6,
                150,
                20,
                tr("options.video"),
            ),
            Button::sized(
                ids::CONTROLS,
                w / 2 + 5,
                h / 6 + 96 - 6,
                150,
                20,
                tr("options.controls"),
            ),
            Button::sized(
                ids::LANGUAGE,
                w / 2 - 155,
                h / 6 + 120 - 6,
                150,
                20,
                tr("options.language"),
            ),
            Button::sized(
                ids::CHAT,
                w / 2 + 5,
                h / 6 + 120 - 6,
                150,
                20,
                tr("options.chat.title"),
            ),
            Button::sized(
                ids::RESOURCE_PACKS,
                w / 2 - 155,
                h / 6 + 144 - 6,
                150,
                20,
                tr("options.resourcepack"),
            ),
            Button::sized(
                ids::SNOOPER,
                w / 2 + 5,
                h / 6 + 144 - 6,
                150,
                20,
                tr("options.snooper.view"),
            ),
            Button::new(ids::DONE, w / 2 - 100, h / 6 + 168, tr("gui.done")),
        ];
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        if button != 0 {
            return Action::None;
        }
        if self.fov.mouse_pressed(x, y) {
            self.commit_fov(settings, lang);
            return Action::None;
        }
        for b in &self.buttons {
            if b.mouse_pressed(x, y) {
                return match b.id {
                    ids::VIDEO => Action::OpenOptionsPage(OptionsPage::Video),
                    ids::CONTROLS => Action::OpenOptionsPage(OptionsPage::Controls),
                    ids::SOUNDS => Action::OpenOptionsPage(OptionsPage::Sound),
                    ids::CHAT => Action::OpenOptionsPage(OptionsPage::Chat),
                    ids::LANGUAGE => Action::OpenOptionsPage(OptionsPage::Language),
                    ids::RESOURCE_PACKS => Action::OpenOptionsPage(OptionsPage::ResourcePacks),
                    ids::SNOOPER => Action::OpenOptionsPage(OptionsPage::Snooper),
                    ids::SKIN => Action::OpenOptionsPage(OptionsPage::SkinCustomisation),
                    ids::ACCOUNTS => {
                        Action::Accounts(crate::screens::accounts::AccountsAction::Open)
                    }
                    ids::DONE => Action::Back,
                    _ => Action::None,
                };
            }
        }
        Action::None
    }

    /// Continue an FOV drag.
    pub fn mouse_dragged(&mut self, x: i32, settings: &mut GameSettings, lang: &Language) {
        // java: GuiOptionSlider.mouseDragged only mutates GameSettings while the button is
        // visible and was selected by mousePressed. A drag that began elsewhere must not
        // re-commit the slider's stale layout position when the cursor crosses the FOV row.
        if self.fov.dragging && self.fov.button.visible {
            self.fov.mouse_dragged(x);
            self.commit_fov(settings, lang);
        }
    }

    /// End an FOV drag.
    pub fn mouse_released(&mut self) {
        self.fov.mouse_released();
    }

    fn commit_fov(&mut self, settings: &mut GameSettings, lang: &Language) {
        commit_slider(OptionId::Fov, &mut self.fov, settings, lang);
    }

    /// Fine adjustment (X1, item 33): Left/Right (Shift: x10) move
    /// the FOV slider, this screen's only slider, one degree. `SaveSettings` when it moved.
    pub fn slider_key(
        &mut self,
        key: i32,
        shift: bool,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        match arrow_steps(key, shift) {
            Some(steps) if self.fov.button.visible => {
                step_slider(OptionId::Fov, &mut self.fov, steps, settings, lang);
                Action::SaveSettings
            }
            _ => Action::None,
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
        draw_world_background(d, ctx.width, ctx.height, self.in_world);
        draw_centered_string(
            d,
            font,
            lang.tr("options.title"),
            ctx.width / 2,
            15,
            TITLE_COLOR,
        );
        self.fov.draw(d, font, mouse.0, mouse.1);
        if let Some(button) = &mut self.difficulty_button {
            button.draw(d, font, mouse.0, mouse.1);
        }
        for b in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
    }
}

fn difficulty_label(difficulty: ServerDifficultyLevel, lang: &Language) -> String {
    let key = match difficulty {
        ServerDifficultyLevel::Peaceful => "options.difficulty.peaceful",
        ServerDifficultyLevel::Easy => "options.difficulty.easy",
        ServerDifficultyLevel::Normal => "options.difficulty.normal",
        ServerDifficultyLevel::Hard => "options.difficulty.hard",
    };
    format!("{}: {}", lang.tr("options.difficulty"), lang.tr(key))
}

/// One locally available value shown by the snooper page.
///
/// Vanilla obtains these pairs from `PlayerUsageSnooper`. The port deliberately does not
/// collect or upload telemetry, so the live client supplies an empty list. Keeping the entry
/// type here lets deterministic callers (and tests) exercise the exact TreeMap ordering and
/// value-column trimming without smuggling a network concern into the UI crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnooperEntry {
    /// The key shown in the left column.
    pub key: String,
    /// The value shown in the right column, before the vanilla width trim.
    pub value: String,
}

impl SnooperEntry {
    /// Build a key/value pair for a local, already-collected value.
    #[must_use]
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

/// The client-information page opened by `GuiOptions` button 104.
///
/// This is intentionally a read-only local view. The vanilla page renders the current client
/// and (when present) integrated-server snooper maps; this multiplayer client has no telemetry
/// collector, so [`Self::new`] starts with no rows while still preserving the page's controls,
/// geometry, and route. [`Self::with_entries`] is useful to an embedding caller that has its own
/// explicitly consented local values.
// java: `client/gui/GuiSnooper.java`
#[derive(Debug)]
pub struct SnooperScreen {
    /// Whether the page is over a running world rather than the title/menu background.
    pub in_world: bool,
    /// Locally supplied values, in Java `TreeMap` order after construction.
    pub entries: Vec<SnooperEntry>,
    /// The scrolling list used by vanilla's inner `GuiSnooper.List`.
    pub list: ScrollList,
    /// Wrapped explanation lines (`options.snooper.desc`).
    pub description: Vec<String>,
    /// The on/off control (`GuiSnooper` button id 1).
    pub toggle: Button,
    /// The Done control (`GuiSnooper` button id 2).
    pub done: Button,
    title: String,
    /// Whether value-list gestures are live. The toggle itself stays enabled when off so it can
    /// be turned back on; vanilla's button id 1 has that same property.
    values_enabled: bool,
}

impl SnooperScreen {
    /// Build an empty, non-telemetry snooper page.
    #[must_use]
    pub fn new(in_world: bool) -> Self {
        Self::with_entries(in_world, std::iter::empty::<SnooperEntry>())
    }

    /// Build a page over local values. Keys are sorted by Java's UTF-16 `String` ordering.
    #[must_use]
    pub fn with_entries(in_world: bool, entries: impl IntoIterator<Item = SnooperEntry>) -> Self {
        let mut entries = entries.into_iter().collect::<Vec<_>>();
        // `TreeMap<String, String>` uses `String.compareTo`, i.e. UTF-16 code units. Rust's
        // default `str::cmp` is scalar-based and differs for supplementary characters.
        entries.sort_by(|a, b| a.key.encode_utf16().cmp(b.key.encode_utf16()));
        // `TreeMap.put` keeps one value per key, with the last insertion winning.
        let mut unique: Vec<SnooperEntry> = Vec::with_capacity(entries.len());
        for entry in entries {
            if let Some(previous) = unique.last_mut() {
                if previous.key == entry.key {
                    *previous = entry;
                    continue;
                }
            }
            unique.push(entry);
        }
        Self {
            in_world,
            entries: unique,
            list: ScrollList::new(0, 0, 80, 0, crate::font::FONT_HEIGHT + 1),
            description: Vec::new(),
            toggle: Button::new(1, 0, 0, ""),
            done: Button::new(2, 0, 0, "Done"),
            title: String::new(),
            values_enabled: true,
        }
    }

    /// Replace the local values and restore TreeMap order.
    pub fn set_entries(&mut self, entries: impl IntoIterator<Item = SnooperEntry>) {
        self.entries = entries.into_iter().collect();
        self.entries
            .sort_by(|a, b| a.key.encode_utf16().cmp(b.key.encode_utf16()));
        let mut unique: Vec<SnooperEntry> = Vec::with_capacity(self.entries.len());
        for entry in self.entries.drain(..) {
            if let Some(previous) = unique.last_mut() {
                if previous.key == entry.key {
                    *previous = entry;
                    continue;
                }
            }
            unique.push(entry);
        }
        self.entries = unique;
    }

    /// Whether the value-list controls are enabled for the current setting.
    #[must_use]
    pub fn values_enabled(&self) -> bool {
        self.values_enabled
    }

    /// Lay out the page and update translated labels.
    ///
    /// Vanilla's `initGui` trims each value to `width - 220`, while the two controls are always
    /// 150×20 at `width / 2 - 152` and `width / 2 + 2`, both at `height - 30`.
    // java: `GuiSnooper.initGui`
    pub fn init(
        &mut self,
        ctx: &ScreenCtx,
        settings: &GameSettings,
        font: &FontRenderer,
        lang: &Language,
    ) {
        let (w, h) = (ctx.width, ctx.height);
        self.title = lang.tr("options.snooper.title").to_owned();
        self.description = font.wrap_to_lines(lang.tr("options.snooper.desc"), w - 30);
        self.list.set_dimensions(w, h, 80, h - 40);
        self.list.slot_height = font.font_height() + 1;
        self.list.scrollbar_placement = ScrollbarX::FromRight { inset: 10 };
        self.list.size = self.entries.len();
        // There is no telemetry-backed value source in this port. Keep the list visibly inert
        // while snooping is disabled, but leave button 1 enabled so the player can re-enable it.
        self.values_enabled = settings.snooper_enabled;
        self.list.enabled = self.values_enabled;
        self.toggle = Button::sized(
            1,
            w / 2 - 152,
            h - 30,
            150,
            20,
            OptionId::SnooperEnabled.display_string(settings, lang),
        );
        self.done = Button::sized(2, w / 2 + 2, h - 30, 150, 20, lang.tr("gui.done"));
    }

    /// Handle a click, returning the application intent for the client driver.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        if button != 0 {
            return Action::None;
        }
        if self.done.mouse_pressed(x, y) {
            return Action::Back;
        }
        if self.toggle.mouse_pressed(x, y) {
            OptionId::SnooperEnabled.cycle(settings, 1);
            self.toggle.text = OptionId::SnooperEnabled.display_string(settings, lang);
            self.values_enabled = settings.snooper_enabled;
            self.list.enabled = self.values_enabled;
            return Action::SaveSettings;
        }
        if self.values_enabled {
            // The rows are informational, just like `GuiSnooper.List.elementClicked`; this call
            // still claims a press on the list's drag surface so scrollbar movement is reachable.
            self.list.mouse_pressed(x, y);
        }
        Action::None
    }

    /// Continue a list drag, if values are enabled.
    pub fn mouse_dragged(&mut self, y: i32) {
        if self.values_enabled {
            self.list.mouse_dragged(y);
        }
    }

    /// End a list drag.
    pub fn mouse_released(&mut self) {
        self.list.mouse_released();
    }

    /// Scroll only while the pointer is inside the `GuiSlot` viewport and values are enabled.
    pub fn mouse_wheel_at(&mut self, x: i32, y: i32, delta: f32) {
        if self.values_enabled {
            self.list.mouse_wheel_at(x, y, delta);
        }
    }

    /// Draw the dirt/world background, rows, title, description, and controls.
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, self.in_world);
        self.list.size = self.entries.len();
        // java: `GuiSnooper.List.drawBackground` is intentionally empty; the outer
        // `GuiSnooper.drawScreen` dirt background remains visible behind the rows.
        for index in self.list.visible_slots() {
            let y = self.list.slot_rect(index).y;
            let entry = &self.entries[index];
            draw_string(d, font, &entry.key, 10, y, TITLE_COLOR);
            let value = font.trim_to_width(&entry.value, ctx.width - 220);
            draw_string(d, font, &value, 230, y, TITLE_COLOR);
        }
        self.list.draw_overlay(d);
        draw_centered_string(d, font, &self.title, ctx.width / 2, 8, TITLE_COLOR);
        let mut y = 22;
        for line in &self.description {
            draw_centered_string(d, font, line, ctx.width / 2, y, 8_421_504);
            y += font.font_height();
        }
        self.toggle.draw(d, font, mouse.0, mouse.1);
        self.done.draw(d, font, mouse.0, mouse.1);
        let _ = lang; // labels are refreshed by `init`; retain the language in the draw seam.
    }
}

/// Vanilla's video-settings option order.
// java: GuiVideoSettings.java:16
pub const VIDEO_OPTIONS: [OptionId; 16] = [
    OptionId::Graphics,
    OptionId::RenderDistance,
    OptionId::AmbientOcclusion,
    OptionId::FramerateLimit,
    OptionId::Anaglyph,
    OptionId::ViewBobbing,
    OptionId::GuiScale,
    OptionId::Gamma,
    OptionId::RenderClouds,
    OptionId::Particles,
    OptionId::Fullscreen,
    OptionId::Vsync,
    OptionId::MipmapLevels,
    OptionId::BlockAlternatives,
    OptionId::UseVbo,
    OptionId::EntityShadows,
];

/// Vanilla's chat-settings option order.
// java: ScreenChatOptions.java:12
pub const CHAT_OPTIONS: [OptionId; 10] = [
    OptionId::ChatVisibility,
    OptionId::ChatColor,
    OptionId::ChatLinks,
    OptionId::ChatOpacity,
    OptionId::ChatLinksPrompt,
    OptionId::ChatScale,
    OptionId::ChatHeightFocused,
    OptionId::ChatHeightUnfocused,
    OptionId::ChatWidth,
    OptionId::ReducedDebugInfo,
];

/// Vanilla's skin-customisation overlay order.
// java: GuiCustomizeSkin.java:27-30
pub const SKIN_OPTIONS: [OptionId; 7] = [
    OptionId::ModelPartCape,
    OptionId::ModelPartJacket,
    OptionId::ModelPartLeftSleeve,
    OptionId::ModelPartRightSleeve,
    OptionId::ModelPartLeftPantsLeg,
    OptionId::ModelPartRightPantsLeg,
    OptionId::ModelPartHat,
];

/// The five float options whose vanilla setter calls `GuiNewChat.refreshChat` immediately.
// java: GameSettings.java:278-306
pub const CHAT_REFRESH_OPTIONS: [OptionId; 5] = [
    OptionId::ChatOpacity,
    OptionId::ChatHeightFocused,
    OptionId::ChatHeightUnfocused,
    OptionId::ChatWidth,
    OptionId::ChatScale,
];

/// Side effects an option-list screen asks its host to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionListEvent {
    /// Re-wrap the complete received chat backlog and reset its scroll position.
    RefreshChat,
}

/// One control on the two-column option list: either a slider or a cycling button.
#[derive(Debug)]
pub enum OptionRow {
    /// A slider.
    Slider(OptionId, Slider),
    /// A cycling or toggling button.
    Button(OptionId, Button),
}

impl OptionRow {
    /// Which option this row edits.
    #[must_use]
    pub fn id(&self) -> OptionId {
        match self {
            OptionRow::Slider(id, _) | OptionRow::Button(id, _) => *id,
        }
    }
}

/// Commit the cursor's raw slider position, then put the handle over the value that
/// `GameSettings` actually retained.
///
/// Stepped options round in `denormalizeValue`; vanilla's `GuiOptionSlider.mouseDragged` writes
/// the value and immediately calls `normalizeValue` again. Without that second pass the label
/// reports (for example) seven chunks while the handle remains between seven and eight.
fn commit_slider(
    option: OptionId,
    slider: &mut Slider,
    settings: &mut GameSettings,
    lang: &Language,
) {
    option.set_float_value(settings, option.denormalize(slider.position()));
    slider.set_position(option.normalize(option.float_value(settings)));
    slider.button.text = option.display_string(settings, lang);
}

/// Left/Right as a number of slider steps (Shift: ten), `None` for any other key (X1, item 33: fine adjustment of the option sliders).
#[must_use]
pub fn arrow_steps(key: i32, shift: bool) -> Option<i32> {
    let steps = if shift { 10 } else { 1 };
    match key {
        crate::widget::keys::LEFT => Some(-steps),
        crate::widget::keys::RIGHT => Some(steps),
        _ => None,
    }
}

/// Move an option slider `steps` of the option's label step ([`OptionId::display_steps`]: one
/// percent of sensitivity, one FOV degree, one render-distance chunk) and commit it like a
/// drag.
pub fn step_slider(
    option: OptionId,
    slider: &mut Slider,
    steps: i32,
    settings: &mut GameSettings,
    lang: &Language,
) {
    if let Some(divisions) = option.display_steps() {
        slider.step_position(steps, divisions);
        commit_slider(option, slider, settings, lang);
    }
}

/// A two-column option collection, either scrolling like `GuiOptionsRowList` or fixed like
/// `ScreenChatOptions`.
///
/// Rows are laid out **paired**: option `i` goes in column `i % 2` of row `i / 2`, at
/// `width / 2 - 155 + (i % 2) * 160`. That is the same formula the non-scrolling screens use;
/// only the vertical origin differs.
// java: GuiOptionsRowList.java:20
#[derive(Debug)]
pub struct OptionList {
    /// The controls, in option order.
    pub rows: Vec<OptionRow>,
    list: ScrollList,
    options: Vec<OptionId>,
    scrolling: bool,
    events: Vec<OptionListEvent>,
    /// The row of the slider pressed last: the arrow keys move it when none is hovered
    /// (X1, item 33).
    last_slider: Option<usize>,
}

impl OptionList {
    /// Build a list over `options`.
    #[must_use]
    pub fn new(options: &[OptionId]) -> Self {
        let mut list = ScrollList::new(0, 0, 32, 0, 25);
        list.show_selection_box = false;
        // java: GuiOptionsRowList overrides both GuiSlot values so its bar clears the right row.
        list.list_width = 400;
        list.scrollbar_placement = ScrollbarX::Centered { offset: 32 };
        Self {
            rows: Vec::new(),
            list,
            options: options.to_vec(),
            scrolling: true,
            events: Vec::new(),
            last_slider: None,
        }
    }

    /// Build the fixed two-column grid used by `ScreenChatOptions`.
    #[must_use]
    fn new_fixed(options: &[OptionId]) -> Self {
        let mut this = Self::new(options);
        this.scrolling = false;
        this
    }

    fn record_change(&mut self, option: OptionId) {
        if CHAT_REFRESH_OPTIONS.contains(&option) {
            self.events.push(OptionListEvent::RefreshChat);
        }
    }

    /// Lay the rows out for a screen size.
    pub fn init(&mut self, ctx: &ScreenCtx, settings: &GameSettings, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        let (top, bottom) = if self.scrolling {
            (32, h - 32)
        } else {
            // Five fixed pairs occupy h/6..h/6+116. The Done button begins four pixels later.
            (h / 6, h / 6 + 116)
        };
        // `Client::build_ui` currently repeats this layout call before each draw. Vanilla only
        // re-runs `initGui` when the screen opens or its resolution changes, so preserve the
        // selected slider across an unchanged layout; an actual resize still creates fresh rows.
        let keep_gesture = !self.rows.is_empty()
            && self.list.width == w
            && self.list.height == h
            && self.list.top == top
            && self.list.bottom == bottom;
        let old_rows = std::mem::take(&mut self.rows);
        self.list.set_dimensions(w, h, top, bottom);
        self.list.size = self.options.len().div_ceil(2);
        if !keep_gesture {
            self.list.enabled = true;
        }
        for (i, opt) in self.options.iter().copied().enumerate() {
            let x = w / 2 - 155 + (i as i32 % 2) * 160;
            let y = self.row_y(i);
            self.rows.push(match opt.kind() {
                OptionKind::Float { .. } => {
                    let mut slider = Slider::new(
                        i as i32,
                        x,
                        y,
                        opt.display_string(settings, lang),
                        0.0,
                        1.0,
                        opt.normalize(opt.float_value(settings)),
                    );
                    if keep_gesture {
                        if let Some(OptionRow::Slider(old_opt, old)) = old_rows.get(i) {
                            if *old_opt == opt && old.dragging {
                                slider.set_position(old.position());
                                slider.dragging = true;
                            }
                        }
                    }
                    OptionRow::Slider(opt, slider)
                }
                _ => OptionRow::Button(
                    opt,
                    Button::sized(i as i32, x, y, 150, 20, opt.display_string(settings, lang)),
                ),
            });
        }
    }

    fn row_y(&self, index: usize) -> i32 {
        if self.scrolling {
            // java: GuiOptionsRowList's entries are drawn at the slot's y; the slot pitch is 25.
            self.list.top + 4 - self.list.scroll_pixels() + (index as i32 / 2) * 25
        } else {
            // java: ScreenChatOptions.initGui, with no GuiSlot offset or scrolling.
            self.list.top + (index as i32 / 2) * 24
        }
    }

    /// Re-place every row after a scroll and refresh its label.
    pub fn relayout(&mut self, settings: &GameSettings, lang: &Language) {
        for i in 0..self.rows.len() {
            let y = self.row_y(i);
            match &mut self.rows[i] {
                OptionRow::Slider(opt, s) => {
                    s.button.rect.y = y;
                    s.button.text = opt.display_string(settings, lang);
                }
                OptionRow::Button(opt, b) => {
                    b.rect.y = y;
                    b.text = opt.display_string(settings, lang);
                }
            }
        }
    }

    /// Handle a click. Returns `Some(option)` when a value changed.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Option<OptionId> {
        if y < self.list.top || y > self.list.bottom {
            return None;
        }
        for (index, row) in self.rows.iter_mut().enumerate() {
            match row {
                OptionRow::Slider(opt, s) => {
                    if s.mouse_pressed(x, y) {
                        commit_slider(*opt, s, settings, lang);
                        let changed = *opt;
                        self.list.enabled = false;
                        self.last_slider = Some(index);
                        self.record_change(changed);
                        return Some(changed);
                    }
                }
                OptionRow::Button(opt, b) => {
                    if b.mouse_pressed(x, y) {
                        opt.cycle(settings, 1);
                        b.text = opt.display_string(settings, lang);
                        let changed = *opt;
                        self.list.enabled = false;
                        self.record_change(changed);
                        return Some(changed);
                    }
                }
            }
        }
        if self.scrolling {
            self.list.mouse_pressed(x, y);
        }
        None
    }

    /// Continue a slider drag.
    pub fn mouse_dragged(
        &mut self,
        x: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Option<OptionId> {
        for row in &mut self.rows {
            if let OptionRow::Slider(opt, s) = row {
                // java: GuiOptionSlider.mouseDragged has an outer visible guard; a hidden row
                // must not commit its last layout position while another button is held.
                if s.dragging && s.button.visible {
                    s.mouse_dragged(x);
                    commit_slider(*opt, s, settings, lang);
                    let changed = *opt;
                    self.record_change(changed);
                    return Some(changed);
                }
            }
        }
        None
    }

    /// Continue a content or scrollbar drag and move the row widgets with the viewport.
    pub fn mouse_list_dragged(&mut self, y: i32, settings: &GameSettings, lang: &Language) {
        if self.scrolling {
            self.list.mouse_dragged(y);
            self.relayout(settings, lang);
        }
    }

    /// End any slider drag.
    pub fn mouse_released(&mut self) {
        for row in &mut self.rows {
            if let OptionRow::Slider(_, s) = row {
                s.mouse_released();
            }
        }
        self.list.mouse_released();
        self.list.enabled = true;
    }

    /// Fine adjustment (X1, item 33): Left/Right (Shift: x10) move
    /// the slider under the pointer, else the one pressed last, one label step. Returns the
    /// option that changed.
    pub fn slider_key(
        &mut self,
        key: i32,
        shift: bool,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Option<OptionId> {
        let steps = arrow_steps(key, shift)?;
        let hovered = self.rows.iter().position(|row| {
            matches!(row, OptionRow::Slider(_, s) if s.button.visible && s.button.is_mouse_over())
        });
        let index = hovered.or(self.last_slider)?;
        let OptionRow::Slider(opt, s) = self.rows.get_mut(index)? else {
            return None;
        };
        let option = *opt;
        step_slider(option, s, steps, settings, lang);
        self.last_slider = Some(index);
        self.record_change(option);
        Some(option)
    }

    /// Scroll.
    pub fn mouse_wheel(&mut self, delta: f32, settings: &GameSettings, lang: &Language) {
        if self.scrolling {
            self.list.mouse_wheel(delta);
            self.relayout(settings, lang);
        }
    }

    /// Scroll only while the pointer is inside GuiSlot's viewport.
    pub fn mouse_wheel_at(
        &mut self,
        x: i32,
        y: i32,
        delta: f32,
        settings: &GameSettings,
        lang: &Language,
    ) {
        if (self.list.left..=self.list.right).contains(&x)
            && (self.list.top..=self.list.bottom).contains(&y)
        {
            self.mouse_wheel(delta, settings, lang);
        }
    }

    /// Draw the rows that fall inside the viewport.
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse: (i32, i32)) {
        if self.scrolling {
            self.list.draw_background(d);
        }
        let (top, bottom) = (self.list.top, self.list.bottom);
        for row in &mut self.rows {
            let y = match row {
                OptionRow::Slider(_, s) => s.button.rect.y,
                OptionRow::Button(_, b) => b.rect.y,
            };
            if self.scrolling && (y + 20 < top || y > bottom) {
                continue;
            }
            match row {
                OptionRow::Slider(_, s) => s.draw(d, font, mouse.0, mouse.1),
                OptionRow::Button(_, b) => b.draw(d, font, mouse.0, mouse.1),
            }
        }
        if self.scrolling {
            self.list.draw_overlay(d);
        }
    }
}

/// The shared host for video's scrolling [`OptionList`] and chat's fixed two-column grid.
#[derive(Debug)]
pub struct OptionListScreen {
    /// The option rows.
    pub list: OptionList,
    /// Title translation key.
    pub title_key: &'static str,
    /// Whether a world is loaded.
    pub in_world: bool,
    done: Button,
    /// Vanilla's fixed-grid Done-button offset from `height / 6`.
    ///
    /// `GuiCustomizeSkin` uses a shorter layout than `ScreenChatOptions`; scrolling pages
    /// replace this with their bottom-anchored control in [`Self::init`].
    fixed_done_y_offset: i32,
}

impl OptionListScreen {
    /// The video-settings screen.
    #[must_use]
    pub fn video(in_world: bool) -> Self {
        Self::new(&VIDEO_OPTIONS, "options.videoTitle", in_world)
    }

    /// The chat-settings screen.
    #[must_use]
    pub fn chat(in_world: bool) -> Self {
        Self {
            list: OptionList::new_fixed(&CHAT_OPTIONS),
            title_key: "options.chat.title",
            in_world,
            done: Button::new(200, 0, 0, "Done"),
            fixed_done_y_offset: 120,
        }
    }

    /// The skin-customisation screen.
    #[must_use]
    pub fn skin(in_world: bool) -> Self {
        Self {
            list: OptionList::new_fixed(&SKIN_OPTIONS),
            // java: `GuiCustomizeSkin.initGui:27` titles the screen with
            // `options.skinCustomisation.title` ("Skin Customization"); the `...` key is only
            // GuiOptions' button label.
            title_key: "options.skinCustomisation.title",
            in_world,
            done: Button::new(200, 0, 0, "Done"),
            fixed_done_y_offset: 96,
        }
    }

    /// A screen over an arbitrary option list.
    #[must_use]
    pub fn new(options: &[OptionId], title_key: &'static str, in_world: bool) -> Self {
        Self {
            list: OptionList::new(options),
            title_key,
            in_world,
            done: Button::new(200, 0, 0, "Done"),
            fixed_done_y_offset: 120,
        }
    }

    /// Drain host-owned side effects in the order the UI produced them.
    ///
    /// The host must call this after both `OptionListScreen::mouse_clicked` and the directly
    /// forwarded `OptionList::mouse_dragged` callback. Vanilla refreshes once per callback,
    /// including repeated drag frames at the same cursor position.
    pub fn take_events(&mut self) -> Vec<OptionListEvent> {
        std::mem::take(&mut self.list.events)
    }

    /// Lay out.
    pub fn init(&mut self, ctx: &ScreenCtx, settings: &GameSettings, lang: &Language) {
        self.list.init(ctx, settings, lang);
        let done_y = if self.list.scrolling {
            ctx.height - 27
        } else {
            ctx.height / 6 + self.fixed_done_y_offset
        };
        self.done = Button::new(
            200,
            ctx.width / 2 - 100,
            done_y,
            lang.tr("gui.done").to_owned(),
        );
    }

    /// Handle a click.
    pub fn mouse_clicked(
        &mut self,
        x: i32,
        y: i32,
        button: i32,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        if button != 0 {
            return Action::None;
        }
        // java: `GuiVideoSettings.actionPerformed` id 200 — `saveOptions()` then
        // `displayGuiScreen(parentGuiScreen)`. The save is the caller's (it owns the path
        // `options.txt` lives at); what the screen reports is "leave".
        if self.done.mouse_pressed(x, y) {
            return Action::Back;
        }
        match self.list.mouse_clicked(x, y, settings, lang) {
            // A GUI-scale or unicode-font change invalidates every laid-out screen.
            Some(OptionId::GuiScale | OptionId::ForceUnicodeFont) => Action::Relayout,
            // java: every `GameSettings.setOptionValue`/`setOptionFloatValue` ends in
            // `saveOptions()`, so a changed option is on disk before the screen is left.
            Some(_) => Action::SaveSettings,
            None => Action::None,
        }
    }

    /// Left/Right step a slider (see [`OptionList::slider_key`]); `SaveSettings` when one
    /// moved. The host drains [`OptionListScreen::take_events`] afterwards, as after a drag.
    pub fn slider_key(
        &mut self,
        key: i32,
        shift: bool,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        match self.list.slider_key(key, shift, settings, lang) {
            Some(_) => Action::SaveSettings,
            None => Action::None,
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
        draw_world_background(d, ctx.width, ctx.height, self.in_world);
        self.list.draw(d, font, mouse);
        draw_centered_string(
            d,
            font,
            lang.tr(self.title_key),
            ctx.width / 2,
            if self.list.scrolling { 5 } else { 20 },
            TITLE_COLOR,
        );
        self.done.draw(d, font, mouse.0, mouse.1);
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

    fn lang() -> Language {
        Language::new("en_US")
    }

    fn difficulty_lang() -> Language {
        let mut l = lang();
        l.insert("options.difficulty", "Difficulty");
        l.insert("options.difficulty.peaceful", "Peaceful");
        l.insert("options.difficulty.easy", "Easy");
        l.insert("options.difficulty.normal", "Normal");
        l.insert("options.difficulty.hard", "Hard");
        l
    }

    #[test]
    fn in_world_difficulty_row_shows_retained_value_in_vanilla_slot_and_is_disabled() {
        let l = difficulty_lang();
        let mut s = OptionsScreen::new_with_difficulty(true, ServerDifficultyLevel::Hard);
        s.init(&ctx(), &GameSettings::default(), &l);
        let button = s
            .difficulty_button
            .as_ref()
            .expect("in-world options include the server difficulty row");
        assert_eq!(button.text, "Difficulty: Hard");
        assert!(
            !button.enabled,
            "multiplayer cannot change server difficulty"
        );
        assert_eq!(
            button.hover_state(true),
            0,
            "disabled rows use the grey widget state"
        );
        assert_eq!(
            button.rect.x,
            854 / 2 + 5,
            "difficulty occupies the right top slot"
        );
        assert_eq!(
            button.rect.y,
            480 / 6 - 12,
            "difficulty keeps GuiOptions' row pitch"
        );

        let (x, y) = (button.rect.x + 1, button.rect.y + 1);
        let mut settings = GameSettings::default();
        assert_eq!(s.mouse_clicked(x, y, 0, &mut settings, &l), Action::None);
        assert_eq!(s.difficulty, ServerDifficultyLevel::Hard);

        s.set_difficulty(ServerDifficultyLevel::Peaceful, &l);
        assert_eq!(
            s.difficulty_button.as_ref().unwrap().text,
            "Difficulty: Peaceful"
        );
    }

    #[test]
    fn difficulty_row_formats_all_server_values_and_defaults_to_normal() {
        let l = difficulty_lang();
        for (difficulty, label) in [
            (ServerDifficultyLevel::Peaceful, "Difficulty: Peaceful"),
            (ServerDifficultyLevel::Easy, "Difficulty: Easy"),
            (ServerDifficultyLevel::Normal, "Difficulty: Normal"),
            (ServerDifficultyLevel::Hard, "Difficulty: Hard"),
        ] {
            let mut s = OptionsScreen::new_with_difficulty(true, difficulty);
            s.init(&ctx(), &GameSettings::default(), &l);
            assert_eq!(s.difficulty_button.as_ref().unwrap().text, label);
        }

        let mut default = OptionsScreen::new(true);
        default.init(&ctx(), &GameSettings::default(), &l);
        assert_eq!(
            default.difficulty_button.as_ref().unwrap().text,
            "Difficulty: Normal",
            "the no-snapshot constructor matches Environment's vanilla default"
        );
    }

    #[test]
    fn no_world_options_omit_server_difficulty_row() {
        let l = difficulty_lang();
        let mut s = OptionsScreen::new_with_difficulty(false, ServerDifficultyLevel::Hard);
        s.init(&ctx(), &GameSettings::default(), &l);
        assert!(s.difficulty_button.is_none());
        assert!(s.buttons.iter().all(|button| button.id != ids::DIFFICULTY));
    }

    fn snooper_lang() -> Language {
        let mut l = lang();
        l.insert("options.snooper", "Allow Snooper");
        l.insert("options.snooper.title", "Machine Specs Collection");
        l.insert(
            "options.snooper.desc",
            "Local values are shown below. Telemetry is never uploaded by this client.",
        );
        l.insert("options.on", "ON");
        l.insert("options.off", "OFF");
        l.insert("gui.done", "Done");
        l
    }

    #[test]
    fn the_accounts_button_takes_the_stream_buttons_empty_slot_and_opens_the_screen() {
        let mut s = OptionsScreen::new(false);
        let settings = GameSettings::default();
        s.init(&ctx(), &settings, &lang());
        let (w, h) = (854, 480);
        let accounts = s
            .buttons
            .iter()
            .find(|b| b.id == ids::ACCOUNTS)
            .expect("the Accounts button");
        // Vanilla's stream button: right of Sounds.
        assert_eq!((accounts.rect.x, accounts.rect.y), (w / 2 + 5, h / 6 + 66));
        assert_eq!(accounts.text, "Accounts...");
        let (x, y) = (accounts.rect.x + 1, accounts.rect.y + 1);
        // No other button shares the slot.
        assert_eq!(
            s.buttons.iter().filter(|b| b.rect.contains(x, y)).count(),
            1
        );
        let mut settings = settings;
        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &lang()),
            Action::Accounts(crate::screens::accounts::AccountsAction::Open)
        );
    }

    #[test]
    fn the_root_options_buttons_land_on_vanillas_grid() {
        let mut s = OptionsScreen::new(false);
        s.init(&ctx(), &GameSettings::default(), &lang());
        let (w, h) = (854, 480);
        let by = |id: i32| s.buttons.iter().find(|b| b.id == id).expect("button");
        assert_eq!(
            (s.fov.button.rect.x, s.fov.button.rect.y),
            (w / 2 - 155, h / 6 - 12)
        );
        assert_eq!(
            (by(ids::VIDEO).rect.x, by(ids::VIDEO).rect.y),
            (w / 2 - 155, h / 6 + 90)
        );
        // Super Secret Settings moved to the mod menu: its slot, right of
        // Skin Customization, is empty and no button carries its id.
        assert!(s.buttons.iter().all(|b| b.id != ids::SUPER_SECRET));
        assert!(s
            .buttons
            .iter()
            .all(|b| !b.rect.contains(w / 2 + 5 + 75, h / 6 + 42 + 10)));
        assert_eq!(
            (by(ids::SKIN).rect.x, by(ids::SKIN).rect.y),
            (w / 2 - 155, h / 6 + 42)
        );
        assert_eq!(
            (by(ids::CONTROLS).rect.x, by(ids::CONTROLS).rect.y),
            (w / 2 + 5, h / 6 + 90)
        );
        assert_eq!(by(ids::LANGUAGE).rect.y, h / 6 + 114);
        assert_eq!(by(ids::CHAT).rect.x, w / 2 + 5);
        assert_eq!(by(ids::RESOURCE_PACKS).rect.y, h / 6 + 138);
        assert_eq!(
            (by(ids::SNOOPER).rect.x, by(ids::SNOOPER).rect.y),
            (w / 2 + 5, h / 6 + 138)
        );
        assert_eq!(
            (by(ids::DONE).rect.x, by(ids::DONE).rect.y),
            (w / 2 - 100, h / 6 + 168)
        );
        assert_eq!(by(ids::DONE).rect.width, 200);
    }

    #[test]
    fn every_root_button_opens_its_page() {
        let mut s = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings::default();
        s.init(&ctx(), &settings, &l);
        let click = |s: &mut OptionsScreen, id: i32, settings: &mut GameSettings| {
            let b = s.buttons.iter().find(|b| b.id == id).unwrap();
            let (x, y) = (b.rect.x + 1, b.rect.y + 1);
            s.mouse_clicked(x, y, 0, settings, &lang())
        };
        assert_eq!(
            click(&mut s, ids::VIDEO, &mut settings),
            Action::OpenOptionsPage(OptionsPage::Video)
        );
        assert_eq!(
            click(&mut s, ids::CONTROLS, &mut settings),
            Action::OpenOptionsPage(OptionsPage::Controls)
        );
        assert_eq!(
            click(&mut s, ids::LANGUAGE, &mut settings),
            Action::OpenOptionsPage(OptionsPage::Language)
        );
        assert_eq!(
            click(&mut s, ids::RESOURCE_PACKS, &mut settings),
            Action::OpenOptionsPage(OptionsPage::ResourcePacks)
        );
        assert_eq!(
            click(&mut s, ids::SNOOPER, &mut settings),
            Action::OpenOptionsPage(OptionsPage::Snooper)
        );
        // The old Super Secret Settings slot does nothing now.
        assert_eq!(
            s.mouse_clicked(854 / 2 + 5 + 75, 480 / 6 + 42 + 10, 0, &mut settings, &l),
            Action::None
        );
        assert_eq!(click(&mut s, ids::DONE, &mut settings), Action::Back);
    }

    #[test]
    fn dragging_the_fov_slider_writes_degrees_back_to_the_settings() {
        let mut s = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings::default();
        s.init(&ctx(), &settings, &l);
        // Press at the far right of the slider track.
        let (x, y) = (s.fov.button.rect.right() - 1, s.fov.button.rect.y + 5);
        s.mouse_clicked(x, y, 0, &mut settings, &l);
        assert_eq!(settings.fov, 110.0, "the right end is Quake Pro");
        s.mouse_dragged(s.fov.button.rect.x, &mut settings, &l);
        assert_eq!(settings.fov, 30.0);
        s.mouse_released();
    }

    #[test]
    fn an_unrelated_drag_does_not_recommit_the_fov_layout_position() {
        let mut s = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings::default();
        s.init(&ctx(), &settings, &l);

        // Simulate a settings update while the cursor is held from a press outside this slider.
        // Vanilla's GuiOptionSlider.mouseDragged is a no-op because dragging is false; before
        // the gate above, the stale position reconstructed during init overwrote 110 degrees.
        settings.fov = 110.0;
        s.mouse_dragged(s.fov.button.rect.x + 4, &mut settings, &l);
        assert_eq!(settings.fov, 110.0);
        assert!(!s.fov.dragging);
    }

    #[test]
    fn the_fov_handle_is_renormalized_from_the_snapped_setting() {
        let mut s = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings::default();
        s.init(&ctx(), &settings, &l);

        // A position of 50/142 maps between whole FOV degrees. GameSettings snaps the value;
        // GuiOptionSlider then normalizes that snapped setting back into sliderValue.
        let x = s.fov.button.rect.x + 4 + 50;
        let y = s.fov.button.rect.y + 5;
        let raw_position = s.fov.position_for(x);
        s.mouse_clicked(x, y, 0, &mut settings, &l);

        let snapped_position = OptionId::Fov.normalize(settings.fov);
        assert_ne!(
            raw_position, snapped_position,
            "the control must exercise a real snap"
        );
        assert_eq!(s.fov.position(), snapped_position);
    }

    #[test]
    fn a_stepped_option_handle_tracks_the_value_written_on_click_and_drag() {
        let mut list = OptionList::new(&[OptionId::RenderDistance]);
        let l = lang();
        let mut settings = GameSettings::default();
        list.init(&ctx(), &settings, &l);

        let OptionRow::Slider(_, slider) = &list.rows[0] else {
            panic!("render distance is a slider")
        };
        let y = slider.button.rect.y + 5;
        let x = slider.button.rect.x + 4 + 50;
        let raw_position = slider.position_for(x);
        assert_eq!(
            list.mouse_clicked(x, y, &mut settings, &l),
            Some(OptionId::RenderDistance)
        );
        let snapped_position =
            OptionId::RenderDistance.normalize(OptionId::RenderDistance.float_value(&settings));
        assert_ne!(
            raw_position, snapped_position,
            "the control must exercise a real snap"
        );
        let OptionRow::Slider(_, slider) = &list.rows[0] else {
            unreachable!()
        };
        assert_eq!(slider.position(), snapped_position);

        let x = slider.button.rect.x + 4 + 73;
        let raw_position = slider.position_for(x);
        assert_eq!(
            list.mouse_dragged(x, &mut settings, &l),
            Some(OptionId::RenderDistance)
        );
        let snapped_position =
            OptionId::RenderDistance.normalize(OptionId::RenderDistance.float_value(&settings));
        assert_ne!(
            raw_position, snapped_position,
            "the drag must exercise a real snap"
        );
        let OptionRow::Slider(_, slider) = &list.rows[0] else {
            unreachable!()
        };
        assert_eq!(slider.position(), snapped_position);
    }

    #[test]
    fn a_hidden_option_slider_does_not_commit_during_a_drag() {
        let mut list = OptionList::new(&[OptionId::RenderDistance]);
        let l = lang();
        let mut settings = GameSettings::default();
        list.init(&ctx(), &settings, &l);

        let (x, y) = match &list.rows[0] {
            OptionRow::Slider(_, slider) => (slider.button.rect.x + 30, slider.button.rect.y + 5),
            _ => panic!("render distance is a slider"),
        };
        assert_eq!(
            list.mouse_clicked(x, y, &mut settings, &l),
            Some(OptionId::RenderDistance)
        );
        settings.render_distance_chunks = mc_render::settings::native_render_distance_max();
        if let OptionRow::Slider(_, slider) = &mut list.rows[0] {
            slider.button.visible = false;
        } else {
            unreachable!();
        }

        assert_eq!(
            list.mouse_dragged(x, &mut settings, &l),
            None,
            "GuiOptionSlider.mouseDragged is inert while hidden"
        );
        assert_eq!(
            settings.render_distance_chunks,
            mc_render::settings::native_render_distance_max()
        );
    }

    #[test]
    fn repeated_root_layout_during_a_frame_does_not_end_the_slider_drag() {
        let mut screen = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings::default();
        let ctx = ctx();
        screen.init(&ctx, &settings, &l);
        let y = screen.fov.button.rect.y + 5;
        let x = screen.fov.button.rect.x + 30;
        screen.mouse_clicked(x, y, 0, &mut settings, &l);
        assert!(screen.fov.dragging);

        // The client currently asks screens to lay themselves out before every draw. Repeating
        // an unchanged layout must not replace the selected GuiOptionSlider mid-gesture.
        screen.init(&ctx, &settings, &l);
        assert!(screen.fov.dragging);
        screen.mouse_dragged(screen.fov.button.rect.right(), &mut settings, &l);
        assert_eq!(settings.fov, 110.0);
    }

    #[test]
    fn repeated_option_list_layout_during_a_frame_preserves_the_selected_slider() {
        let mut list = OptionList::new(&[OptionId::RenderDistance]);
        let l = lang();
        let mut settings = GameSettings::default();
        let ctx = ctx();
        list.init(&ctx, &settings, &l);
        let OptionRow::Slider(_, slider) = &list.rows[0] else {
            panic!("render distance is a slider")
        };
        let (x, y) = (slider.button.rect.x + 30, slider.button.rect.y + 5);
        list.mouse_clicked(x, y, &mut settings, &l);

        list.init(&ctx, &settings, &l);
        let OptionRow::Slider(_, slider) = &list.rows[0] else {
            unreachable!()
        };
        assert!(slider.dragging);
        let x = slider.button.rect.right();
        assert_eq!(
            list.mouse_dragged(x, &mut settings, &l),
            Some(OptionId::RenderDistance)
        );
        assert_eq!(
            settings.render_distance_chunks,
            mc_render::settings::native_render_distance_max()
        );
    }

    #[test]
    fn the_video_list_pairs_options_into_two_columns() {
        let mut s = OptionListScreen::video(false);
        let settings = GameSettings::default();
        s.init(&ctx(), &settings, &lang());
        assert_eq!(s.list.rows.len(), VIDEO_OPTIONS.len());
        let x_of = |r: &OptionRow| match r {
            OptionRow::Slider(_, s) => (s.button.rect.x, s.button.rect.y),
            OptionRow::Button(_, b) => (b.rect.x, b.rect.y),
        };
        let (x0, y0) = x_of(&s.list.rows[0]);
        let (x1, y1) = x_of(&s.list.rows[1]);
        let (x2, y2) = x_of(&s.list.rows[2]);
        assert_eq!(x0, 854 / 2 - 155);
        assert_eq!(x1, 854 / 2 - 155 + 160, "the second column is 160 across");
        assert_eq!(y0, y1, "a pair shares a row");
        assert_eq!(x2, x0);
        assert_eq!(y2 - y0, 25, "the row pitch is 25");
        // Sliders and buttons are chosen by the option's kind.
        assert!(matches!(
            s.list.rows[0],
            OptionRow::Button(OptionId::Graphics, _)
        ));
        assert!(matches!(
            s.list.rows[1],
            OptionRow::Slider(OptionId::RenderDistance, _)
        ));
    }

    #[test]
    fn chat_options_match_vanillas_exact_set_and_order() {
        assert_eq!(
            CHAT_OPTIONS,
            [
                OptionId::ChatVisibility,
                OptionId::ChatColor,
                OptionId::ChatLinks,
                OptionId::ChatOpacity,
                OptionId::ChatLinksPrompt,
                OptionId::ChatScale,
                OptionId::ChatHeightFocused,
                OptionId::ChatHeightUnfocused,
                OptionId::ChatWidth,
                OptionId::ReducedDebugInfo,
            ]
        );
    }

    #[test]
    fn skin_customisation_has_all_seven_live_model_part_toggles() {
        assert_eq!(
            SKIN_OPTIONS,
            [
                OptionId::ModelPartCape,
                OptionId::ModelPartJacket,
                OptionId::ModelPartLeftSleeve,
                OptionId::ModelPartRightSleeve,
                OptionId::ModelPartLeftPantsLeg,
                OptionId::ModelPartRightPantsLeg,
                OptionId::ModelPartHat,
            ]
        );

        let l = lang();
        for option in SKIN_OPTIONS {
            let mut screen = OptionListScreen::skin(false);
            let mut settings = GameSettings::default();
            screen.init(&ctx(), &settings, &l);
            let index = SKIN_OPTIONS
                .iter()
                .position(|candidate| *candidate == option)
                .expect("skin option index");
            let OptionRow::Button(_, button) = &screen.list.rows[index] else {
                panic!("skin model parts must be buttons")
            };
            let click = (button.rect.x + 1, button.rect.y + 1);

            assert!(option.bool_value(&settings));
            assert_eq!(
                screen.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
                Action::SaveSettings
            );
            assert!(!option.bool_value(&settings), "{option:?} toggles off");
            assert_eq!(
                screen.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
                Action::SaveSettings
            );
            assert!(option.bool_value(&settings), "{option:?} toggles on");
        }
    }

    #[test]
    fn done_button_uses_skin_layout_offset_without_moving_video_or_chat() {
        let settings = GameSettings::default();
        let l = lang();
        for height in [240_i32, 480, 720] {
            let c = ScreenCtx::new(ScaledResolution::new(854, height as u32, 1, false), 0.0);

            let mut skin = OptionListScreen::skin(false);
            skin.init(&c, &settings, &l);
            assert_eq!(skin.done.rect.y, height / 6 + 96);

            let mut chat = OptionListScreen::chat(false);
            chat.init(&c, &settings, &l);
            assert_eq!(chat.done.rect.y, height / 6 + 120);
            assert_ne!(
                chat.done.rect.y, skin.done.rect.y,
                "chat keeps ScreenChatOptions' +120 Done offset"
            );

            let mut video = OptionListScreen::video(false);
            video.init(&c, &settings, &l);
            assert_eq!(video.done.rect.y, height - 27);
            assert_ne!(
                video.done.rect.y, skin.done.rect.y,
                "video keeps its bottom-anchored Done button"
            );
        }
    }

    /// java: `GuiCustomizeSkin.initGui:27` draws
    /// `I18n.format("options.skinCustomisation.title")` at y 20, while GuiOptions' button that
    /// opens it is `options.skinCustomisation` (en_US: "Skin Customization..."). The screen used
    /// the button's key and showed the ellipsis.
    #[test]
    fn the_skin_screen_is_titled_with_its_own_title_key() {
        let mut l = lang();
        l.insert("options.skinCustomisation", "Skin Customization...");
        l.insert("options.skinCustomisation.title", "Skin Customization");
        let screen = OptionListScreen::skin(false);
        assert_eq!(screen.title_key, "options.skinCustomisation.title");
        assert_eq!(l.tr(screen.title_key), "Skin Customization");
    }

    #[test]
    fn exactly_five_chat_float_options_request_refresh() {
        assert_eq!(
            CHAT_REFRESH_OPTIONS,
            [
                OptionId::ChatOpacity,
                OptionId::ChatHeightFocused,
                OptionId::ChatHeightUnfocused,
                OptionId::ChatWidth,
                OptionId::ChatScale,
            ]
        );
    }

    #[test]
    fn chat_slider_press_and_every_drag_callback_emit_refresh_intent() {
        let mut s = OptionListScreen::chat(false);
        let mut settings = GameSettings::default();
        let l = lang();
        s.init(&ctx(), &settings, &l);
        let index = CHAT_OPTIONS
            .iter()
            .position(|option| *option == OptionId::ChatWidth)
            .unwrap();
        let OptionRow::Slider(_, slider) = &s.list.rows[index] else {
            panic!("chat width must be a slider")
        };
        let (x, y) = (slider.button.rect.x + 40, slider.button.rect.y + 5);

        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &l),
            Action::SaveSettings
        );
        assert_eq!(s.take_events(), vec![OptionListEvent::RefreshChat]);

        // GuiOptionSlider.mouseDragged calls setOptionFloatValue on every callback, even if the
        // cursor did not move. GameSettings therefore refreshes chat once per callback too.
        assert_eq!(
            s.list.mouse_dragged(x, &mut settings, &l),
            Some(OptionId::ChatWidth)
        );
        assert_eq!(
            s.list.mouse_dragged(x, &mut settings, &l),
            Some(OptionId::ChatWidth)
        );
        assert_eq!(
            s.take_events(),
            vec![OptionListEvent::RefreshChat, OptionListEvent::RefreshChat]
        );
    }

    #[test]
    fn non_refresh_options_emit_no_chat_intent() {
        let mut settings = GameSettings::default();
        let l = lang();

        let mut chat = OptionListScreen::chat(false);
        chat.init(&ctx(), &settings, &l);
        let OptionRow::Button(_, visibility) = &chat.list.rows[0] else {
            panic!("chat visibility must be a button")
        };
        let click = (visibility.rect.x + 1, visibility.rect.y + 1);
        assert_eq!(
            chat.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
            Action::SaveSettings
        );
        assert!(chat.take_events().is_empty());

        let mut video = OptionListScreen::video(false);
        video.init(&ctx(), &settings, &l);
        let index = VIDEO_OPTIONS
            .iter()
            .position(|option| *option == OptionId::RenderDistance)
            .unwrap();
        let OptionRow::Slider(_, distance) = &video.list.rows[index] else {
            panic!("render distance must be a slider")
        };
        let click = (distance.button.rect.x + 40, distance.button.rect.y + 1);
        assert_eq!(
            video.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
            Action::SaveSettings
        );
        assert!(video.take_events().is_empty());
    }

    #[test]
    fn chat_options_use_the_fixed_five_row_screen_chat_options_grid() {
        let mut s = OptionListScreen::chat(false);
        let settings = GameSettings::default();
        let l = lang();
        let c = ctx();
        s.init(&c, &settings, &l);

        let position = |row: &OptionRow| match row {
            OptionRow::Slider(_, slider) => (slider.button.rect.x, slider.button.rect.y),
            OptionRow::Button(_, button) => (button.rect.x, button.rect.y),
        };
        let left = c.width / 2 - 155;
        let right = left + 160;
        let top = c.height / 6;
        assert!(!s.list.scrolling, "ScreenChatOptions has no GuiSlot");
        assert_eq!(s.list.rows.len(), 10);
        for pair in 0..5 {
            assert_eq!(
                position(&s.list.rows[pair * 2]),
                (left, top + pair as i32 * 24)
            );
            assert_eq!(
                position(&s.list.rows[pair * 2 + 1]),
                (right, top + pair as i32 * 24)
            );
        }
        assert_eq!(
            (s.done.rect.x, s.done.rect.y, s.done.rect.width),
            (c.width / 2 - 100, top + 120, 200)
        );

        let before = position(&s.list.rows[0]);
        s.list.mouse_wheel(-1.0, &settings, &l);
        assert_eq!(
            position(&s.list.rows[0]),
            before,
            "the fixed page cannot scroll"
        );
        assert_eq!(s.list.list.scroll_pixels(), 0);
    }

    #[test]
    fn options_row_list_uses_its_four_hundred_pixel_column_and_shifted_bar() {
        let mut s = OptionListScreen::video(false);
        s.init(&ctx(), &GameSettings::default(), &lang());
        assert_eq!(s.list.list.list_width, 400);
        assert_eq!(s.list.list.scrollbar_x(), 854 / 2 + 124 + 32);
        assert_eq!(
            s.list.list.scrollbar_x(),
            854 / 2 + 156,
            "one pixel beyond the right button's x+155 edge"
        );
    }

    #[test]
    fn options_row_list_drags_and_ignores_wheel_outside_its_viewport() {
        let mut s = OptionListScreen::video(false);
        let settings = GameSettings::default();
        let l = lang();
        let short = ScreenCtx::new(ScaledResolution::new(854, 160, 1, false), 0.0);
        s.init(&short, &settings, &l);
        let y0 = match &s.list.rows[0] {
            OptionRow::Slider(_, slider) => slider.button.rect.y,
            OptionRow::Button(_, button) => button.rect.y,
        };

        let start_y = s.list.list.top + 5;
        let mut changed = settings.clone();
        assert_eq!(s.list.mouse_clicked(10, start_y, &mut changed, &l), None);
        s.list.mouse_list_dragged(start_y - 20, &settings, &l);
        let dragged_y = match &s.list.rows[0] {
            OptionRow::Slider(_, slider) => slider.button.rect.y,
            OptionRow::Button(_, button) => button.rect.y,
        };
        assert_eq!(dragged_y, y0 - 20);
        s.list.mouse_released();

        let after_drag = s.list.list.scroll_pixels();
        s.list
            .mouse_wheel_at(10, s.list.list.top - 1, -1.0, &settings, &l);
        assert_eq!(s.list.list.scroll_pixels(), after_drag);
        s.list
            .mouse_wheel_at(10, s.list.list.top, -1.0, &settings, &l);
        assert_eq!(s.list.list.scroll_pixels(), after_drag + 12);
    }

    #[test]
    fn clicking_a_video_row_changes_the_setting_and_its_label() {
        let mut s = OptionListScreen::video(false);
        let mut settings = GameSettings::default();
        let mut l = lang();
        l.insert("options.graphics", "Graphics");
        l.insert("options.graphics.fancy", "Fancy");
        l.insert("options.graphics.fast", "Fast");
        s.init(&ctx(), &settings, &l);
        assert!(settings.fancy_graphics);
        let OptionRow::Button(_, b) = &s.list.rows[0] else {
            panic!("expected a button")
        };
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &l),
            Action::SaveSettings,
            "java: setOptionValue ends in saveOptions()"
        );
        assert!(!settings.fancy_graphics);
        let OptionRow::Button(_, b) = &s.list.rows[0] else {
            panic!()
        };
        assert_eq!(b.text, "Graphics: Fast");
    }

    #[test]
    fn changing_the_gui_scale_asks_for_a_relayout() {
        let mut s = OptionListScreen::video(false);
        let mut settings = GameSettings::default();
        let l = lang();
        s.init(&ctx(), &settings, &l);
        let idx = VIDEO_OPTIONS
            .iter()
            .position(|o| *o == OptionId::GuiScale)
            .unwrap();
        let OptionRow::Button(_, b) = &s.list.rows[idx] else {
            panic!()
        };
        let (x, y) = (b.rect.x + 1, b.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &l),
            Action::Relayout
        );
        assert_eq!(settings.gui_scale, 1);
    }

    /// java: `GuiVideoSettings.actionPerformed` id 200 is `saveOptions()` **then**
    /// `displayGuiScreen(parentGuiScreen)` — the screen leaves. Every value change on the way
    /// there has already asked for a save of its own.
    #[test]
    fn done_returns_to_the_parent_screen() {
        let mut s = OptionListScreen::chat(false);
        let mut settings = GameSettings::default();
        let l = lang();
        s.init(&ctx(), &settings, &l);
        let click = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
            Action::Back
        );
    }

    #[test]
    fn scrolling_the_video_list_moves_every_row_together() {
        let mut s = OptionListScreen::video(false);
        let settings = GameSettings::default();
        let l = lang();
        // A short screen, so the eight rows really do overflow the viewport.
        let short = ScreenCtx::new(ScaledResolution::new(854, 160, 1, false), 0.0);
        s.init(&short, &settings, &l);
        let y_of = |s: &OptionListScreen, i: usize| match &s.list.rows[i] {
            OptionRow::Slider(_, w) => w.button.rect.y,
            OptionRow::Button(_, w) => w.rect.y,
        };
        let before = (y_of(&s, 0), y_of(&s, 2));
        s.list.mouse_wheel(-1.0, &settings, &l);
        let after = (y_of(&s, 0), y_of(&s, 2));
        assert!(after.0 < before.0, "scrolling down moves rows up");
        assert_eq!(
            before.0 - after.0,
            before.1 - after.1,
            "every row moves by the same amount"
        );
    }

    /// The rows are drawn *between* the two halves of the list frame, not before both.
    ///
    /// [`ScrollList::draw_background`](crate::widget::ScrollList::draw_background) emits an
    /// opaque tiled quad across the whole viewport, and [`DrawList`] has no z-order — batches
    /// are submitted and drawn front to back. So a screen that drew its rows and *then* called
    /// `draw_frame` produced a full complement of row vertices that were painted over: sixteen
    /// working options, none of them visible. A vertex-count assertion cannot see that; the
    /// batch order can.
    #[test]
    fn the_list_background_never_lands_on_top_of_a_row() {
        let mut s = OptionListScreen::video(false);
        let settings = GameSettings::default();
        let l = lang();
        let ctx = ctx();
        s.init(&ctx, &settings, &l);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &ctx, (0, 0), &l);

        let is = |i: usize, path: &str| {
            matches!(&d.batches()[i].texture,
                crate::widget::UiTexture::Resource(r) if r.path == path)
        };
        // Where the first row's button art starts.
        let first_row = (0..d.batches().len())
            .find(|i| is(*i, crate::widget::WIDGETS_TEXTURE))
            .expect("the rows draw buttons and sliders");
        let rows_begin = d.batches()[first_row].range.start;

        // `draw_overlay` legitimately re-uses the dirt texture after the rows for the two bands
        // that clip scrolled-out entries — but those sit strictly outside `top..bottom`. Any
        // dirt quad drawn after the rows that reaches *into* the viewport is covering them.
        let (top, bottom) = (s.list.list.top as f32, s.list.list.bottom as f32);
        for (i, batch) in d.batches().iter().enumerate() {
            if !is(i, crate::widget::OPTIONS_BACKGROUND) || batch.range.start < rows_begin {
                continue;
            }
            for quad in d.vertices()[batch.range.clone()].chunks(6) {
                let lo = quad.iter().map(|v| v.pos[1]).fold(f32::MAX, f32::min);
                let hi = quad.iter().map(|v| v.pos[1]).fold(f32::MIN, f32::max);
                assert!(
                    lo >= bottom || hi <= top,
                    "a dirt quad spanning y {lo}..{hi} is drawn after the rows and reaches into \
                     the viewport {top}..{bottom} — draw_background belongs before the rows and \
                     only draw_overlay after"
                );
            }
        }
    }

    #[test]
    fn rows_scrolled_out_of_the_viewport_are_not_drawn() {
        let mut s = OptionListScreen::video(false);
        let settings = GameSettings::default();
        let l = lang();
        // A short screen so most rows fall outside.
        let short = ScreenCtx::new(ScaledResolution::new(854, 240, 1, false), 0.0);
        s.init(&short, &settings, &l);
        let mut font = uniform_font();
        let mut d = DrawList::new();
        s.draw(&mut d, &mut font, &short, (0, 0), &l);
        let drawn = d
            .batches()
            .iter()
            .filter(|b| {
                matches!(&b.texture, crate::widget::UiTexture::Resource(r)
                if r.path == crate::widget::WIDGETS_TEXTURE)
            })
            .count();
        assert!(drawn > 0, "something was drawn");
        // 240 - 32 - 32 = 176 px of viewport at a 25 px pitch is 8 rows, i.e. at most 16 of the
        // 16 options plus the Done button — but definitely not all 16 pairs.
        assert!(s.list.rows.len() == 16);
    }

    #[test]
    fn snooper_uses_vanillas_geometry_and_java_sorted_value_rows() {
        let l = snooper_lang();
        let mut s = SnooperScreen::with_entries(
            false,
            [
                SnooperEntry::new("zeta", "last"),
                SnooperEntry::new("alpha", "first"),
                SnooperEntry::new("alpha", "last-alpha"),
            ],
        );
        let settings = GameSettings::default();
        s.init(&ctx(), &settings, &uniform_font(), &l);

        assert_eq!(
            s.entries.iter().map(|e| e.key.as_str()).collect::<Vec<_>>(),
            ["alpha", "zeta"]
        );
        assert_eq!(s.entries[0].value, "last-alpha");
        assert_eq!(
            (s.list.top, s.list.bottom, s.list.slot_height),
            (80, 440, 10)
        );
        assert_eq!(s.list.scrollbar_x(), 854 - 10);
        assert_eq!(
            (s.toggle.rect.x, s.toggle.rect.y, s.toggle.rect.width),
            (854 / 2 - 152, 480 - 30, 150)
        );
        assert_eq!((s.done.rect.x, s.done.rect.y), (854 / 2 + 2, 480 - 30));
        assert_eq!(
            s.list.slot_rect(0).y,
            84,
            "GuiSlot starts rows four pixels below top"
        );
        assert_eq!(s.list.slot_rect(1).y, 94, "FONT_HEIGHT + 1 pitch");
    }

    #[test]
    fn snooper_toggle_is_live_when_off_but_value_controls_are_disabled() {
        let l = snooper_lang();
        let mut settings = GameSettings {
            snooper_enabled: false,
            ..Default::default()
        };
        let mut s = SnooperScreen::with_entries(false, [SnooperEntry::new("key", "value")]);
        s.init(&ctx(), &settings, &uniform_font(), &l);
        assert!(!s.values_enabled());
        assert!(
            !s.list.enabled,
            "the telemetry value list is inert while snooping is off"
        );
        assert!(
            s.toggle.enabled,
            "button 1 must remain usable to re-enable snooping"
        );

        let before = s.list.amount_scrolled();
        s.mouse_wheel_at(400, s.list.top, -1.0);
        assert_eq!(
            s.list.amount_scrolled(),
            before,
            "disabled values do not scroll"
        );

        let click = (s.toggle.rect.x + 1, s.toggle.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(click.0, click.1, 0, &mut settings, &l),
            Action::SaveSettings
        );
        assert!(settings.snooper_enabled);
        assert!(s.values_enabled());
        assert!(s.list.enabled);
        assert!(s.toggle.text.ends_with("ON"));
    }

    #[test]
    fn snooper_done_and_viewport_input_return_the_expected_actions() {
        let l = snooper_lang();
        let mut entries = Vec::new();
        for i in 0..64 {
            entries.push(SnooperEntry::new(
                format!("key-{i:02}"),
                format!("value-{i:02}"),
            ));
        }
        let mut s = SnooperScreen::with_entries(false, entries);
        let settings = GameSettings::default();
        let short = ScreenCtx::new(ScaledResolution::new(854, 180, 1, false), 0.0);
        s.init(&short, &settings, &uniform_font(), &l);
        s.mouse_wheel_at(400, s.list.top - 1, -1.0);
        assert_eq!(
            s.list.amount_scrolled(),
            0.0,
            "above the list is outside its viewport"
        );
        s.mouse_wheel_at(400, s.list.top, -1.0);
        assert_eq!(
            s.list.amount_scrolled(),
            5.0,
            "GuiSnooper's ten-pixel rows scroll by five"
        );

        let done = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(done.0, done.1, 0, &mut GameSettings::default(), &l),
            Action::Back
        );
        assert_eq!(
            s.mouse_clicked(done.0, done.1, 1, &mut GameSettings::default(), &l),
            Action::None
        );
    }


    /// Left/Right step the options screen's FOV slider one degree (Shift: ten) and ask for a
    /// save; other keys do nothing.
    #[test]
    fn the_arrow_keys_step_the_fov_one_degree() {
        use crate::widget::keys;
        let mut s = OptionsScreen::new(false);
        let l = lang();
        let mut settings = GameSettings {
            fov: 70.0,
            ..GameSettings::default()
        };
        s.init(&ctx(), &settings, &l);
        assert_eq!(
            s.slider_key(keys::RIGHT, false, &mut settings, &l),
            Action::SaveSettings
        );
        assert_eq!(settings.fov, 71.0);
        s.slider_key(keys::LEFT, true, &mut settings, &l);
        assert_eq!(settings.fov, 61.0);
        assert_eq!(
            s.slider_key(keys::A, false, &mut settings, &l),
            Action::None
        );
        assert_eq!(settings.fov, 61.0);
    }

    /// In an option list the arrows move the slider pressed last, by its own step (render
    /// distance: one chunk); with nothing pressed or hovered they do nothing.
    #[test]
    fn the_arrow_keys_step_the_last_pressed_list_slider() {
        use crate::widget::keys;
        let mut s = OptionListScreen::video(false);
        let l = lang();
        let mut settings = GameSettings {
            render_distance_chunks: 8,
            ..GameSettings::default()
        };
        s.init(&ctx(), &settings, &l);
        assert_eq!(
            s.slider_key(keys::RIGHT, false, &mut settings, &l),
            Action::None,
            "nothing pressed or hovered yet"
        );
        let index = s
            .list
            .rows
            .iter()
            .position(|row| row.id() == OptionId::RenderDistance)
            .expect("render distance row");
        let OptionRow::Slider(_, slider) = &s.list.rows[index] else {
            panic!("render distance is a slider");
        };
        let rect = slider.button.rect;
        let (x, y) = (slider.handle_x() + 4, rect.y + 5);
        s.mouse_clicked(x, y, 0, &mut settings, &l);
        s.list.mouse_released();
        assert_eq!(settings.render_distance_chunks, 8, "pressed on the handle");
        assert_eq!(
            s.slider_key(keys::RIGHT, false, &mut settings, &l),
            Action::SaveSettings
        );
        assert_eq!(settings.render_distance_chunks, 9);
        s.slider_key(keys::LEFT, false, &mut settings, &l);
        s.slider_key(keys::LEFT, false, &mut settings, &l);
        assert_eq!(settings.render_distance_chunks, 7);
    }
}
