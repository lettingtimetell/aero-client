//! The controls screen: three mouse options and the rebindable key list.
//!
//! java: `client/gui/GuiControls.java`, `GuiKeyBindingList.java`
//!
//! The list is one row per binding, grouped by category with a centred header row per group.
//! Clicking a binding's button arms it; the very next key or mouse press becomes the new
//! binding. `mc-render`'s [`KeyBindings`] owns the model, including conflict detection.

use mc_assets::Language;
use mc_render::input::{Input, KeyBindings};
use mc_render::settings::GameSettings;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::screens::options::model::OptionId;
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::keys;
use crate::widget::{draw_centered_string, Button, ScrollList, ScrollbarX, Slider};

/// The three options above the key list. java: `GuiControls.optionsArr`
pub const CONTROL_OPTIONS: [OptionId; 3] = [
    OptionId::InvertMouse,
    OptionId::Sensitivity,
    OptionId::Touchscreen,
];

/// One row of the key list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyRow {
    /// A centred category header, e.g. "Movement".
    Category(String),
    /// A rebindable action, identified by its `key.*` description.
    Binding(String),
}

/// The controls screen.
#[derive(Debug)]
pub struct ControlsScreen {
    /// The rows, headers included.
    pub rows: Vec<KeyRow>,
    /// The binding currently waiting for a key, if any. java: `GuiControls.buttonId`
    pub arming: Option<String>,
    /// Whether a world is loaded.
    pub in_world: bool,
    list: ScrollList,
    label_width: i32,
    sliders: Vec<(OptionId, Slider)>,
    buttons: Vec<(OptionId, Button)>,
    done: Button,
    reset_all: Button,
}

impl ControlsScreen {
    /// A controls screen over `bindings`.
    #[must_use]
    pub fn new(bindings: &KeyBindings, in_world: bool) -> Self {
        Self {
            // `init` has the active language and replaces this with vanilla's translated sort.
            // Keeping the raw-key order here makes the value useful before its first layout too.
            rows: Self::build_rows_by(bindings, str::to_owned),
            arming: None,
            in_world,
            list: ScrollList::new(0, 0, 63, 0, 20),
            label_width: 0,
            sliders: Vec::new(),
            buttons: Vec::new(),
            done: Button::new(200, 0, 0, "Done"),
            reset_all: Button::new(201, 0, 0, "Reset"),
        }
    }

    /// Group the bindings by category, in vanilla's sorted order.
    ///
    /// Vanilla sorts `KeyBinding[]` with `compareTo`, which orders by *translated* category
    /// first and translated description second. Ordering by raw translation keys instead would
    /// shuffle every category's entries and can also shuffle the categories when the language
    /// changes.
    // java: GuiKeyBindingList.java:18
    #[must_use]
    pub fn build_rows(bindings: &KeyBindings, lang: &Language) -> Vec<KeyRow> {
        // `KeyBinding.compareTo` calls `I18n.format(key, new Object[0])`, not the raw
        // translation accessor. The distinction is observable for a resource-pack language
        // entry containing a formatter conversion: Java catches the missing-argument error and
        // sorts the resulting `"Format error: ..."` string.
        Self::build_rows_by(bindings, |key| lang.format(key, &[]))
    }

    fn build_rows_by(bindings: &KeyBindings, translate: impl Fn(&str) -> String) -> Vec<KeyRow> {
        let mut sorted = bindings.iter().collect::<Vec<_>>();
        // java: KeyBinding.compareTo compares the translated category, then the translated
        // description. Java String ordering is over UTF-16 code units, not Unicode scalars.
        sorted.sort_by(|a, b| {
            let a_category = translate(a.category());
            let b_category = translate(b.category());
            java_string_cmp(&a_category, &b_category).then_with(|| {
                let a_description = translate(a.description());
                let b_description = translate(b.description());
                java_string_cmp(&a_description, &b_description)
            })
        });

        let mut rows = Vec::new();
        let mut category = None;
        for binding in sorted {
            if category != Some(binding.category()) {
                category = Some(binding.category());
                rows.push(KeyRow::Category(binding.category().to_owned()));
            }
            rows.push(KeyRow::Binding(binding.description().to_owned()));
        }
        rows
    }

    /// Lay out.
    // java: GuiControls.java:38
    pub fn init(
        &mut self,
        ctx: &ScreenCtx,
        settings: &GameSettings,
        font: &FontRenderer,
        lang: &Language,
    ) {
        let (w, h) = (ctx.width, ctx.height);
        self.rows = Self::build_rows(&settings.keys, lang);
        self.list.set_dimensions(w, h, 63, h - 32);
        self.list.list_width = 220 + 32;
        self.list.scrollbar_placement = ScrollbarX::Centered { offset: 15 };
        self.list.show_selection_box = false;
        self.list.size = self.rows.len();
        self.label_width = self
            .rows
            .iter()
            .filter_map(|r| match r {
                KeyRow::Binding(d) => Some(font.string_width(lang.tr(d))),
                KeyRow::Category(_) => None,
            })
            .max()
            .unwrap_or(0);

        self.done = Button::sized(
            200,
            w / 2 - 155,
            h - 29,
            150,
            20,
            lang.tr("gui.done").to_owned(),
        );
        self.reset_all = Button::sized(
            201,
            w / 2 - 155 + 160,
            h - 29,
            150,
            20,
            lang.tr("controls.resetAll").to_owned(),
        );
        // `Client::build_ui` re-runs this before every draw, but vanilla re-runs `initGui` only
        // on open and resize. Rebuilding the grabbed `GuiOptionSlider` cleared its `dragging`
        // flag one frame after the press, so sensitivity could be clicked but never dragged.
        let drag = self
            .sliders
            .iter()
            .find(|(_, s)| s.dragging)
            .map(|(opt, s)| (*opt, s.button.rect, s.position()));
        self.sliders.clear();
        self.buttons.clear();
        for (i, opt) in CONTROL_OPTIONS.iter().copied().enumerate() {
            let x = w / 2 - 155 + (i as i32 % 2) * 160;
            let y = 18 + 24 * (i as i32 / 2);
            match opt.kind() {
                crate::screens::options::OptionKind::Float { .. } => {
                    let mut s = Slider::new(
                        i as i32,
                        x,
                        y,
                        String::new(),
                        0.0,
                        1.0,
                        opt.normalize(opt.float_value(settings)),
                    );
                    s.button.text = opt.display_string(settings, lang);
                    self.sliders.push((opt, s));
                }
                _ => self.buttons.push((
                    opt,
                    Button::sized(i as i32, x, y, 150, 20, opt.display_string(settings, lang)),
                )),
            }
        }
        // An unchanged layout keeps the grabbed slider; a resize builds fresh buttons, as
        // `initGui` does.
        if let Some((dragged, rect, position)) = drag {
            if let Some((_, s)) = self
                .sliders
                .iter_mut()
                .find(|(opt, s)| *opt == dragged && s.button.rect == rect)
            {
                s.set_position(position);
                s.dragging = true;
            }
        }
    }

    /// Whether every binding is at its default, which is what greys out "Reset All".
    // java: GuiControls.java:162
    #[must_use]
    pub fn all_default(bindings: &KeyBindings) -> bool {
        bindings.iter().all(|b| b.binding() == b.default_binding())
    }

    /// The label for a binding's button, with vanilla's `> … <` and red-conflict decoration.
    // java: GuiKeyBindingList.KeyEntry.drawEntry
    #[must_use]
    pub fn binding_label(&self, bindings: &KeyBindings, description: &str) -> String {
        let Some(b) = bindings.get(description) else {
            return String::new();
        };
        let name = binding_display_name(b.binding());
        if self.arming.as_deref() == Some(description) {
            format!("\u{a7}f> \u{a7}e{name}\u{a7}f <")
        } else if b.binding().is_bound() && bindings.is_conflicting(description) {
            format!("\u{a7}c{name}")
        } else {
            name
        }
    }

    /// Where a row's two buttons sit, given the row's screen `y`.
    ///
    /// Vanilla's numbers: the label right-aligns to `x + 90`, the rebind button starts at
    /// `x + 105` and is 75 wide, and the reset button starts at `x + 190` and is 50 wide.
    #[must_use]
    pub fn row_button_xs(&self, index: usize) -> (i32, i32) {
        let r = self.list.slot_rect(index);
        (r.x + 105, r.x + 190)
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
        // An armed binding swallows the next click, which is how the mouse buttons get bound.
        if let Some(desc) = self.arming.take() {
            // java: setOptionKeyBinding(buttonId, -100 + mouseButton)
            settings
                .keys
                .set_binding(&desc, Input::Mouse(button.max(0) as u16));
            return Action::SaveSettings;
        }
        if button != 0 {
            return Action::None;
        }
        if self.done.mouse_pressed(x, y) {
            return Action::Back;
        }
        if self.reset_all.mouse_pressed(x, y) {
            settings.keys.reset_all();
            return Action::SaveSettings;
        }
        for (opt, s) in &mut self.sliders {
            if s.mouse_pressed(x, y) {
                opt.set_float_value(settings, opt.denormalize(s.position()));
                s.button.text = opt.display_string(settings, lang);
                return Action::None;
            }
        }
        for (opt, b) in &mut self.buttons {
            if b.mouse_pressed(x, y) {
                opt.cycle(settings, 1);
                b.text = opt.display_string(settings, lang);
                return Action::SaveSettings;
            }
        }
        if y < self.list.top || y > self.list.bottom {
            return Action::None;
        }
        for index in self.list.visible_slots() {
            let KeyRow::Binding(desc) = &self.rows[index] else {
                continue;
            };
            let r = self.list.slot_rect(index);
            let (rebind_x, reset_x) = self.row_button_xs(index);
            if (r.y..r.y + 20).contains(&y) {
                if (rebind_x..rebind_x + 75).contains(&x) {
                    self.arming = Some(desc.clone());
                    // GuiListExtended fences list dragging until this claimed press is released.
                    self.list.enabled = false;
                    return Action::None;
                }
                if (reset_x..reset_x + 50).contains(&x) {
                    let desc = desc.clone();
                    if let Some(b) = settings.keys.get_mut(&desc) {
                        b.reset();
                    }
                    self.list.enabled = false;
                    return Action::SaveSettings;
                }
            }
        }
        self.list.mouse_pressed(x, y);
        Action::None
    }

    /// Handle a key press.
    ///
    /// While armed, Escape *unbinds* rather than cancelling — vanilla's
    /// `setOptionKeyBinding(buttonId, 0)`.
    // java: GuiControls.java:127
    pub fn key_typed(&mut self, key: i32, settings: &mut GameSettings) -> Action {
        self.key_typed_with_char(None, key, settings)
    }

    /// Handle a key press together with the character produced by the active keyboard layout.
    ///
    /// LWJGL can report a printable character with scan code zero (notably for dead keys and
    /// AltGr combinations). Vanilla preserves those as `typedChar + 256`; code zero with no
    /// character changes no binding, but still disarms the waiting row.
    // java: GuiControls.java:127-152
    pub fn key_typed_with_char(
        &mut self,
        typed: Option<char>,
        key: i32,
        settings: &mut GameSettings,
    ) -> Action {
        if let Some(desc) = self.arming.take() {
            let input = if key == keys::ESCAPE {
                Some(Input::Unbound)
            } else if key != 0 {
                Some(Input::from_lwjgl(key))
            } else {
                typed
                    .filter(|c| *c != '\0')
                    // Java's `char` is one UTF-16 code unit. A supplementary Rust scalar is
                    // represented by its first surrogate, exactly what the LWJGL-era API could
                    // have supplied in one event.
                    .and_then(|c| {
                        let mut units = [0; 2];
                        c.encode_utf16(&mut units).first().copied()
                    })
                    .map(|unit| Input::from_lwjgl(i32::from(unit) + 256))
            };
            if let Some(input) = input {
                settings.keys.set_binding(&desc, input);
            }
            return Action::SaveSettings;
        }
        if key == keys::ESCAPE {
            // GuiControls delegates to GuiScreen.keyTyped, whose Escape path closes every GUI;
            // it does not activate the Done button or return only one parent level.
            Action::CloseAll
        } else {
            Action::None
        }
    }

    /// Scroll.
    pub fn mouse_wheel(&mut self, delta: f32) {
        self.list.mouse_wheel(delta);
    }

    /// Scroll only while the cursor is inside the list viewport, as `GuiSlot.handleMouseInput`
    /// does before it reads the wheel delta.
    pub fn mouse_wheel_at(&mut self, x: i32, y: i32, delta: f32) {
        self.list.mouse_wheel_at(x, y, delta);
    }

    /// Continue a drag of the sensitivity slider.
    ///
    /// java: `GuiSlider.mouseDragged`, reached through `GuiScreen.mouseClickMove`. Without it the
    /// mouse-sensitivity slider could only be *clicked* to a position, never dragged to one —
    /// which is how everybody actually sets it.
    pub fn mouse_dragged(&mut self, x: i32, settings: &mut GameSettings, lang: &Language) {
        for (opt, s) in &mut self.sliders {
            if s.dragging {
                s.mouse_dragged(x);
                opt.set_float_value(settings, opt.denormalize(s.position()));
                s.button.text = opt.display_string(settings, lang);
            }
        }
    }

    /// Continue a content or scrollbar drag begun by [`Self::mouse_clicked`].
    pub fn mouse_list_dragged(&mut self, y: i32) {
        self.list.mouse_dragged(y);
    }

    /// End any slider drag. java: `GuiSlider.mouseReleased`.
    pub fn mouse_released(&mut self) {
        for (_, s) in &mut self.sliders {
            s.mouse_released();
        }
        self.list.mouse_released();
        self.list.enabled = true;
    }

    /// Fine adjustment (X1, item 33): Left/Right
    /// (Shift: x10) move the slider under the pointer, else the first (sensitivity, the only
    /// one), one label step: one percent, so every whole percentage is reachable at any GUI
    /// scale. The driver calls this only for keys no armed binding took. `SaveSettings` when it
    /// moved.
    pub fn slider_key(
        &mut self,
        key: i32,
        shift: bool,
        settings: &mut GameSettings,
        lang: &Language,
    ) -> Action {
        let Some(steps) = super::arrow_steps(key, shift) else {
            return Action::None;
        };
        let index = self
            .sliders
            .iter()
            .position(|(_, s)| s.button.visible && s.button.is_mouse_over())
            .unwrap_or(0);
        match self.sliders.get_mut(index) {
            Some((opt, s)) if s.button.visible => {
                super::step_slider(*opt, s, steps, settings, lang);
                Action::SaveSettings
            }
            _ => Action::None,
        }
    }

    /// Draw.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        ctx: &ScreenCtx,
        mouse: (i32, i32),
        settings: &GameSettings,
        lang: &Language,
    ) {
        draw_world_background(d, ctx.width, ctx.height, self.in_world);
        self.list.size = self.rows.len();
        self.list.draw_background(d);
        for index in self.list.visible_slots() {
            let r = self.list.slot_rect(index);
            match &self.rows[index] {
                KeyRow::Category(key) => {
                    let text = lang.tr(key).to_owned();
                    let x = ctx.width / 2 - font.string_width(&text) / 2;
                    draw_key_row_text(d, font, &text, x, category_text_y(r.y, font.font_height()));
                }
                KeyRow::Binding(desc) => {
                    let label = lang.tr(desc).to_owned();
                    draw_key_row_text(
                        d,
                        font,
                        &label,
                        r.x + 90 - self.label_width,
                        binding_text_y(r.y, font.font_height()),
                    );
                    let (rebind_x, reset_x) = self.row_button_xs(index);
                    let mut rebind = Button::sized(
                        0,
                        rebind_x,
                        r.y,
                        75,
                        20,
                        self.binding_label(&settings.keys, desc),
                    );
                    rebind.draw(d, font, mouse.0, mouse.1);
                    let mut reset = Button::sized(
                        0,
                        reset_x,
                        r.y,
                        50,
                        20,
                        lang.tr("controls.reset").to_owned(),
                    );
                    reset.enabled = settings
                        .keys
                        .get(desc)
                        .is_some_and(|b| b.binding() != b.default_binding());
                    reset.draw(d, font, mouse.0, mouse.1);
                }
            }
        }
        self.list.draw_overlay(d);
        draw_centered_string(
            d,
            font,
            lang.tr("controls.title"),
            ctx.width / 2,
            8,
            TITLE_COLOR,
        );
        for (_, s) in &mut self.sliders {
            s.draw(d, font, mouse.0, mouse.1);
        }
        for (_, b) in &mut self.buttons {
            b.draw(d, font, mouse.0, mouse.1);
        }
        self.reset_all.enabled = !Self::all_default(&settings.keys);
        self.done.draw(d, font, mouse.0, mouse.1);
        self.reset_all.draw(d, font, mouse.0, mouse.1);
    }
}

fn binding_display_name(input: Input) -> String {
    match input {
        // java: GameSettings.getKeyDisplayString(int), whose `key >= 256` branch treats the
        // stored value as a character rather than asking LWJGL for a key name.
        Input::RawLwjgl(code) if code >= 256 => {
            let unit = (code - 256) as u16;
            String::from_utf16_lossy(&[unit]).to_uppercase()
        }
        _ => input.display_name(),
    }
}

fn java_string_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

fn binding_text_y(row_y: i32, font_height: i32) -> i32 {
    // GuiSlot passes `slotHeight - 4` (16), not the list's 20-pixel pitch, to drawEntry.
    row_y + (20 - 4) / 2 - font_height / 2
}

fn category_text_y(row_y: i32, font_height: i32) -> i32 {
    row_y + (20 - 4) - font_height - 1
}

fn draw_key_row_text(d: &mut DrawList, font: &mut FontRenderer, text: &str, x: i32, y: i32) -> i32 {
    // GuiKeyBindingList uses FontRenderer.drawString(...), whose dropShadow argument is false.
    font.draw_string(d, text, x as f32, y as f32, TITLE_COLOR, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn screen(settings: &GameSettings) -> ControlsScreen {
        let mut s = ControlsScreen::new(&settings.keys, false);
        s.init(&ctx(), settings, &uniform_font(), &Language::new("en_US"));
        s
    }

    #[test]
    fn the_rows_are_grouped_by_category_with_a_header_each() {
        let settings = GameSettings::default();
        let s = screen(&settings);
        let headers: Vec<&String> = s
            .rows
            .iter()
            .filter_map(|r| match r {
                KeyRow::Category(c) => Some(c),
                KeyRow::Binding(_) => None,
            })
            .collect();
        assert_eq!(headers.len(), settings.keys.categories().len());
        assert!(
            matches!(s.rows[0], KeyRow::Category(_)),
            "a header comes first"
        );
        let bindings = s
            .rows
            .iter()
            .filter(|r| matches!(r, KeyRow::Binding(_)))
            .count();
        assert_eq!(bindings, settings.keys.iter().count());
    }

    #[test]
    fn rows_follow_translated_category_and_description_order() {
        use mc_render::input::keys as binding_keys;

        let settings = GameSettings::default();
        let mut lang = Language::new("test");
        for category in settings.keys.categories() {
            lang.insert(category, format!("ZZZ {category}"));
        }
        lang.insert("key.categories.movement", "AAA movement");
        for (key, label) in [
            (binding_keys::FORWARD, "Zulu"),
            (binding_keys::LEFT, "Yankee"),
            (binding_keys::BACK, "Xray"),
            (binding_keys::RIGHT, "Whiskey"),
            (binding_keys::JUMP, "Alpha"),
            (binding_keys::SNEAK, "Bravo"),
            (binding_keys::SPRINT, "Charlie"),
        ] {
            lang.insert(key, label);
        }

        let rows = ControlsScreen::build_rows(&settings.keys, &lang);
        assert_eq!(
            rows.iter().take(8).cloned().collect::<Vec<_>>(),
            vec![
                KeyRow::Category("key.categories.movement".into()),
                KeyRow::Binding(binding_keys::JUMP.into()),
                KeyRow::Binding(binding_keys::SNEAK.into()),
                KeyRow::Binding(binding_keys::SPRINT.into()),
                KeyRow::Binding(binding_keys::RIGHT.into()),
                KeyRow::Binding(binding_keys::BACK.into()),
                KeyRow::Binding(binding_keys::LEFT.into()),
                KeyRow::Binding(binding_keys::FORWARD.into()),
            ]
        );
    }

    #[test]
    fn translated_category_sort_uses_i18n_format_with_empty_arguments() {
        let settings = GameSettings::default();
        let mut lang = Language::new("test");
        for category in settings.keys.categories() {
            lang.insert(category, format!("ZZZ {category}"));
        }
        lang.insert("key.categories.gameplay", "Alpha category");
        lang.insert("key.categories.movement", "%s");

        let rows = ControlsScreen::build_rows(&settings.keys, &lang);
        assert_eq!(
            rows.first(),
            Some(&KeyRow::Category("key.categories.gameplay".into())),
            "Java sorts `Alpha category` before the formatted `Format error: %s`; a raw-translation mutant sorts `%s` first"
        );
    }

    #[test]
    fn translated_description_sort_uses_i18n_format_with_empty_arguments() {
        use mc_render::input::keys as binding_keys;

        let settings = GameSettings::default();
        let mut lang = Language::new("test");
        for category in settings.keys.categories() {
            lang.insert(category, format!("ZZZ {category}"));
        }
        lang.insert("key.categories.movement", "AAA movement");
        for binding in settings
            .keys
            .iter()
            .filter(|binding| binding.category() == "key.categories.movement")
        {
            lang.insert(binding.description(), "ZZZ binding");
        }
        lang.insert(binding_keys::JUMP, "Alpha binding");
        lang.insert(binding_keys::FORWARD, "%s");

        let rows = ControlsScreen::build_rows(&settings.keys, &lang);
        assert_eq!(
            &rows[..3],
            [
                KeyRow::Category("key.categories.movement".into()),
                KeyRow::Binding(binding_keys::JUMP.into()),
                KeyRow::Binding(binding_keys::FORWARD.into()),
            ],
            "description compareTo formats both labels before comparing them"
        );
    }

    #[test]
    fn translated_sort_uses_java_utf16_order() {
        assert_eq!(
            java_string_cmp("\u{1f600}", "\u{e000}"),
            std::cmp::Ordering::Less,
            "Java compares the surrogate D83D before E000; Rust scalar order is the opposite"
        );
    }

    #[test]
    fn key_list_uses_vanillas_scrollbar_override() {
        let settings = GameSettings::default();
        let s = screen(&settings);
        assert_eq!(s.list.scrollbar_x(), 854 / 2 + 124 + 15);
        assert_eq!(s.list.list_width, 252);
    }

    #[test]
    fn row_text_uses_the_sixteen_pixel_draw_entry_height_and_no_shadow() {
        assert_eq!(binding_text_y(100, 9), 104);
        assert_eq!(category_text_y(100, 9), 106);

        let mut font = uniform_font();
        let mut d = DrawList::new();
        draw_key_row_text(&mut d, &mut font, "A", 0, 0);
        assert_eq!(
            d.vertices().len(),
            6,
            "one glyph is one quad; a shadow mutant emits a second six-vertex quad"
        );
    }

    #[test]
    fn the_list_drags_and_wheel_events_are_bounded_to_its_viewport() {
        let settings = GameSettings::default();
        let mut s = ControlsScreen::new(&settings.keys, false);
        let short = ScreenCtx::new(ScaledResolution::new(854, 160, 1, false), 0.0);
        s.init(&short, &settings, &uniform_font(), &Language::new("en_US"));

        let start_y = s.list.top + 5;
        let mut scratch = settings.clone();
        s.mouse_clicked(10, start_y, 0, &mut scratch, &Language::new("en_US"));
        s.mouse_list_dragged(start_y - 20);
        assert_eq!(s.list.scroll_pixels(), 20);
        s.mouse_released();

        let after_drag = s.list.scroll_pixels();
        s.mouse_wheel_at(10, s.list.top - 1, -1.0);
        assert_eq!(s.list.scroll_pixels(), after_drag, "title strip is outside");
        s.mouse_wheel_at(10, s.list.top, -1.0);
        assert_eq!(s.list.scroll_pixels(), after_drag + 10);
    }

    #[test]
    fn a_claimed_key_entry_press_fences_list_dragging_until_release() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let index = s
            .rows
            .iter()
            .position(|row| matches!(row, KeyRow::Binding(_)))
            .unwrap();
        let row = s.list.slot_rect(index);
        let (rebind_x, _) = s.row_button_xs(index);
        s.mouse_clicked(
            rebind_x + 1,
            row.y + 1,
            0,
            &mut settings,
            &Language::new("en_US"),
        );
        assert!(!s.list.enabled);
        s.mouse_released();
        assert!(s.list.enabled);
    }

    #[test]
    fn arming_a_binding_captures_the_next_key() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let index = s
            .rows
            .iter()
            .position(|r| matches!(r, KeyRow::Binding(_)))
            .unwrap();
        let r = s.list.slot_rect(index);
        let (rebind_x, _) = s.row_button_xs(index);
        assert_eq!(
            s.mouse_clicked(
                rebind_x + 2,
                r.y + 2,
                0,
                &mut settings,
                &Language::new("en_US")
            ),
            Action::None
        );
        assert!(s.arming.is_some());
        let desc = s.arming.clone().unwrap();
        // LWJGL 33 is 'F'.
        assert_eq!(s.key_typed(33, &mut settings), Action::SaveSettings);
        assert!(s.arming.is_none());
        assert_eq!(
            settings.keys.get(&desc).unwrap().binding(),
            Input::from_lwjgl(33)
        );
    }

    #[test]
    fn escape_while_armed_unbinds_instead_of_cancelling() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        s.arming = Some(mc_render::input::keys::FORWARD.to_owned());
        s.key_typed(keys::ESCAPE, &mut settings);
        assert_eq!(
            settings
                .keys
                .get(mc_render::input::keys::FORWARD)
                .unwrap()
                .binding(),
            Input::Unbound
        );
        // With nothing armed, inherited GuiScreen Escape closes the whole screen stack.
        assert_eq!(s.key_typed(keys::ESCAPE, &mut settings), Action::CloseAll);
    }

    #[test]
    fn zero_scancode_binds_the_layout_character_plus_256() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let desc = mc_render::input::keys::FORWARD.to_owned();
        s.arming = Some(desc.clone());

        assert_eq!(
            s.key_typed_with_char(Some('é'), 0, &mut settings),
            Action::SaveSettings
        );
        assert_eq!(
            settings.keys.get(&desc).unwrap().binding(),
            Input::RawLwjgl(256 + 'é' as i32),
            "a from_lwjgl(0) mutant would unbind this instead"
        );
        assert_eq!(s.binding_label(&settings.keys, &desc), "É");
    }

    #[test]
    fn zero_scancode_without_a_character_disarms_without_changing_the_binding() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let desc = mc_render::input::keys::FORWARD.to_owned();
        let before = settings.keys.get(&desc).unwrap().binding();
        s.arming = Some(desc.clone());

        assert_eq!(
            s.key_typed_with_char(None, 0, &mut settings),
            Action::SaveSettings
        );
        assert_eq!(settings.keys.get(&desc).unwrap().binding(), before);
        assert!(s.arming.is_none());
    }

    #[test]
    fn nonzero_scancode_wins_over_the_typed_character() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let desc = mc_render::input::keys::FORWARD.to_owned();
        s.arming = Some(desc.clone());

        s.key_typed_with_char(Some('x'), 33, &mut settings);
        assert_eq!(
            settings.keys.get(&desc).unwrap().binding(),
            Input::from_lwjgl(33)
        );
    }

    #[test]
    fn arming_then_clicking_binds_a_mouse_button() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let desc = mc_render::input::keys::ATTACK.to_owned();
        s.arming = Some(desc.clone());
        s.mouse_clicked(0, 0, 1, &mut settings, &Language::new("en_US"));
        assert_eq!(settings.keys.get(&desc).unwrap().binding(), Input::Mouse(1));
    }

    #[test]
    fn a_conflicting_binding_is_labelled_red_and_the_armed_one_gets_arrows() {
        let mut settings = GameSettings::default();
        let forward = mc_render::input::keys::FORWARD;
        let back = mc_render::input::keys::BACK;
        let s0 = screen(&settings);
        assert!(!s0
            .binding_label(&settings.keys, forward)
            .starts_with('\u{a7}'));

        let f = settings.keys.get(forward).unwrap().binding();
        settings.keys.set_binding(back, f);
        let mut s = screen(&settings);
        assert!(
            s.binding_label(&settings.keys, forward)
                .starts_with("\u{a7}c"),
            "a duplicate binding is red"
        );
        s.arming = Some(forward.to_owned());
        let label = s.binding_label(&settings.keys, forward);
        assert!(label.starts_with("\u{a7}f> \u{a7}e") && label.ends_with("\u{a7}f <"));
    }

    #[test]
    fn reset_all_is_only_enabled_once_something_is_rebound() {
        let mut settings = GameSettings::default();
        assert!(ControlsScreen::all_default(&settings.keys));
        settings
            .keys
            .set_binding(mc_render::input::keys::JUMP, Input::from_lwjgl(19));
        assert!(!ControlsScreen::all_default(&settings.keys));

        let mut s = screen(&settings);
        let (x, y) = (s.reset_all.rect.x + 1, s.reset_all.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &Language::new("en_US")),
            Action::SaveSettings
        );
        assert!(ControlsScreen::all_default(&settings.keys));
    }

    #[test]
    fn the_per_row_reset_button_restores_only_that_binding() {
        let mut settings = GameSettings::default();
        let mut s = screen(&settings);
        let index = s
            .rows
            .iter()
            .position(|r| matches!(r, KeyRow::Binding(_)))
            .unwrap();
        let KeyRow::Binding(desc) = s.rows[index].clone() else {
            unreachable!()
        };
        settings.keys.set_binding(&desc, Input::from_lwjgl(88));
        let r = s.list.slot_rect(index);
        let (_, reset_x) = s.row_button_xs(index);
        s.mouse_clicked(
            reset_x + 2,
            r.y + 2,
            0,
            &mut settings,
            &Language::new("en_US"),
        );
        let b = settings.keys.get(&desc).unwrap();
        assert_eq!(b.binding(), b.default_binding());
    }

    #[test]
    fn the_three_mouse_options_sit_above_the_list() {
        let settings = GameSettings::default();
        let s = screen(&settings);
        assert_eq!(s.sliders.len(), 1, "sensitivity is the only slider");
        assert_eq!(s.buttons.len(), 2);
        assert_eq!(
            s.buttons[0].1.rect,
            crate::widget::Rect::new(854 / 2 - 155, 18, 150, 20)
        );
        assert_eq!(s.sliders[0].1.button.rect.x, 854 / 2 - 155 + 160);
        assert_eq!(s.sliders[0].1.button.rect.y, 18);
        assert_eq!(s.buttons[1].1.rect.y, 42, "the third wraps to the next row");
    }

    /// `Client::build_ui` calls `init` before every frame. `GuiOptionSlider.dragging` must
    /// survive that until release, or sensitivity can be clicked to a spot but never dragged.
    #[test]
    fn a_sensitivity_drag_survives_the_per_frame_relayout() {
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        let font = uniform_font();
        let mut s = screen(&settings);
        let rect = s.sliders[0].1.button.rect;
        let x_at = |fraction: f32| rect.x + 4 + ((rect.width - 8) as f32 * fraction) as i32;

        s.mouse_clicked(x_at(0.2), rect.y + 5, 0, &mut settings, &l);
        for fraction in [0.6, 0.9] {
            s.init(&ctx(), &settings, &font, &l);
            s.mouse_dragged(x_at(fraction), &mut settings, &l);
            let sensitivity = settings.mouse_sensitivity;
            assert!(
                (sensitivity - fraction).abs() < 0.01,
                "{fraction}: {sensitivity}"
            );
        }
        s.mouse_released();
        s.init(&ctx(), &settings, &font, &l);
        s.mouse_dragged(x_at(0.1), &mut settings, &l);
        assert!((settings.mouse_sensitivity - 0.9).abs() < 0.01, "released");
    }
}
