//! The language screen: a scrolling list of locales plus the force-unicode toggle.
//!
//! java: `client/gui/GuiLanguage.java`
//!
//! Selecting a language is what makes the unicode-font flag change on its own, so this screen
//! is the one that has to tell the caller to re-lay-out everything —
//! [`common::unicode_font_enabled`](crate::screens::common::unicode_font_enabled) is
//! `language.is_unicode() || settings.force_unicode_font`, and *both* halves move here.

use mc_assets::{Language, LanguageInfo, DEFAULT_LANGUAGE};
use mc_render::settings::GameSettings;

use crate::font::FontRenderer;
use crate::screen_stack::ScreenCtx;
use crate::screens::common::{draw_world_background, TITLE_COLOR};
use crate::screens::options::model::OptionId;
use crate::screens::Action;
use crate::widget::draw::DrawList;
use crate::widget::{draw_centered_string, Button, ScrollList};

/// The language screen.
#[derive(Debug)]
pub struct LanguageScreen {
    /// The available locales, sorted by code — vanilla sorts `langCodeList` the same way.
    pub languages: Vec<LanguageInfo>,
    /// Whether a world is loaded behind this screen.
    pub in_world: bool,
    list: ScrollList,
    unicode: Button,
    done: Button,
}

impl LanguageScreen {
    /// A screen over the locales the resource packs advertise.
    #[must_use]
    pub fn new(mut languages: Vec<LanguageInfo>) -> Self {
        languages.sort_by(|a, b| a.code.cmp(&b.code));
        let mut list = ScrollList::new(0, 0, 32, 0, 18);
        list.size = languages.len();
        Self {
            languages,
            in_world: false,
            list,
            unicode: Button::new(100, 0, 0, ""),
            done: Button::new(6, 0, 0, "Done"),
        }
    }

    /// Select the vanilla world-background branch for an in-game language screen.
    #[must_use]
    pub fn with_in_world(mut self, in_world: bool) -> Self {
        self.in_world = in_world;
        self
    }

    /// Lay out.
    // java: GuiLanguage.java:45
    pub fn init(&mut self, ctx: &ScreenCtx, settings: &GameSettings, lang: &Language) {
        let (w, h) = (ctx.width, ctx.height);
        // java: `super(mc, width, height, 32, height - 65 + 4, 18)`
        self.list.set_dimensions(w, h, 32, h - 65 + 4);
        self.list.size = self.languages.len();
        self.unicode = Button::sized(
            100,
            w / 2 - 155,
            h - 38,
            150,
            20,
            OptionId::ForceUnicodeFont.display_string(settings, lang),
        );
        self.done = Button::sized(
            6,
            w / 2 - 155 + 160,
            h - 38,
            150,
            20,
            lang.tr("gui.done").to_owned(),
        );
        self.select_code(&settings.language);
    }

    /// Highlight the row for `code`, if it is present.
    pub fn select_code(&mut self, code: &str) {
        self.list.selected = self
            .languages
            .iter()
            .position(|l| l.code == code)
            .map_or(-1, |i| i as i32);
    }

    /// Replace the list after a resource-pack reload without retaining stale pack entries.
    pub fn replace_languages(&mut self, mut languages: Vec<LanguageInfo>) {
        let selected_code = self.selected().map(|language| language.code.clone());
        languages.sort_by(|a, b| a.code.cmp(&b.code));
        self.languages = languages;
        self.list.size = self.languages.len();
        let code = selected_code
            .filter(|code| self.languages.iter().any(|language| language.code == *code))
            .unwrap_or_else(|| DEFAULT_LANGUAGE.to_owned());
        self.select_code(&code);
    }

    /// The selected locale, if any.
    #[must_use]
    pub fn selected(&self) -> Option<&LanguageInfo> {
        usize::try_from(self.list.selected)
            .ok()
            .and_then(|i| self.languages.get(i))
    }

    /// Handle a click.
    ///
    /// Returns [`Action::Relayout`] when the *effective* unicode flag changes — either because
    /// the toggle moved or because the newly-picked language disagrees with the old one about
    /// it. That is the point of this screen from the font's perspective.
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
        if self.unicode.mouse_pressed(x, y) {
            OptionId::ForceUnicodeFont.cycle(settings, 1);
            self.unicode.text = OptionId::ForceUnicodeFont.display_string(settings, lang);
            return Action::Relayout;
        }
        if self.list.mouse_pressed(x, y) {
            return Action::None;
        }
        if let Some(index) = self.list.slot_index_at(x, y) {
            self.list.selected = index as i32;
            if let Some(info) = self.languages.get(index) {
                if settings.language != info.code {
                    settings.language = info.code.clone();
                    return Action::Relayout;
                }
            }
        }
        Action::None
    }

    /// Scroll.
    pub fn mouse_wheel(&mut self, delta: f32) {
        self.list.mouse_wheel(delta);
    }

    /// Scroll only when the pointer is inside `GuiSlot`'s viewport.
    pub fn mouse_wheel_at(&mut self, x: i32, y: i32, delta: f32) {
        self.list.mouse_wheel_at(x, y, delta);
    }

    /// Continue a content or scrollbar drag begun by [`Self::mouse_clicked`].
    pub fn mouse_dragged(&mut self, y: i32) {
        self.list.mouse_dragged(y);
    }

    /// End a list drag.
    pub fn mouse_released(&mut self) {
        self.list.mouse_released();
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
        self.list.size = self.languages.len();
        self.list.draw_background(d);
        for index in self.list.visible_slots() {
            let r = self.list.slot_rect(index);
            if self.list.selected == index as i32 {
                self.list.draw_selection_box(d, index);
            }
            let label = self.languages[index].to_string();
            // java: GuiLanguage.List.drawSlot temporarily forces bidi for every language name,
            // then restores the selected locale's direction. A Hebrew name must remain readable
            // while the active locale is English (and vice versa); leaving the caller's flag in
            // place makes the list's visual order depend on which language is currently selected.
            let active_bidi = font.bidi_flag();
            font.set_bidi_flag(true);
            draw_centered_string(d, font, &label, ctx.width / 2, r.y + 1, TITLE_COLOR);
            font.set_bidi_flag(active_bidi);
        }
        self.list.draw_overlay(d);
        draw_centered_string(
            d,
            font,
            lang.tr("options.language"),
            ctx.width / 2,
            16,
            TITLE_COLOR,
        );
        let warning = format!("({})", lang.tr("options.languageWarning"));
        draw_centered_string(
            d,
            font,
            &warning,
            ctx.width / 2,
            ctx.height - 56,
            0x0080_8080,
        );
        self.unicode.draw(d, font, mouse.0, mouse.1);
        self.done.draw(d, font, mouse.0, mouse.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::draw::UiTexture;
    use crate::widget::test_support::uniform_font;
    use crate::widget::ScaledResolution;

    fn ctx() -> ScreenCtx {
        ScreenCtx::new(ScaledResolution::new(854, 480, 1, false), 0.0)
    }

    fn infos() -> Vec<LanguageInfo> {
        vec![
            LanguageInfo {
                code: "zh_CN".into(),
                name: "Chinese Simplified".into(),
                region: "China".into(),
                bidirectional: false,
            },
            LanguageInfo {
                code: "en_US".into(),
                name: "English".into(),
                region: "US".into(),
                bidirectional: false,
            },
            LanguageInfo {
                code: "ar_SA".into(),
                name: "Arabic".into(),
                region: "Saudi Arabia".into(),
                bidirectional: true,
            },
        ]
    }

    #[test]
    fn the_list_is_sorted_by_code_and_preselects_the_active_language() {
        let mut s = LanguageScreen::new(infos());
        assert_eq!(
            s.languages
                .iter()
                .map(|l| l.code.as_str())
                .collect::<Vec<_>>(),
            ["ar_SA", "en_US", "zh_CN"]
        );
        let settings = GameSettings::default();
        s.init(&ctx(), &settings, &Language::new("en_US"));
        assert_eq!(s.selected().map(|l| l.code.as_str()), Some("en_US"));
    }

    #[test]
    fn picking_a_new_language_writes_the_setting_and_asks_for_a_relayout() {
        let mut s = LanguageScreen::new(infos());
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let r = s.list.slot_rect(2); // zh_CN
        let action = s.mouse_clicked(854 / 2, r.y + 2, 0, &mut settings, &l);
        assert_eq!(action, Action::Relayout);
        assert_eq!(settings.language, "zh_CN");
        // Re-picking the same one changes nothing.
        assert_eq!(
            s.mouse_clicked(854 / 2, r.y + 2, 0, &mut settings, &l),
            Action::None
        );
    }

    #[test]
    fn reload_replaces_stale_languages_and_preserves_or_falls_back_selection() {
        let mut screen = LanguageScreen::new(infos());
        screen.select_code("zh_CN");
        let replacement = vec![
            LanguageInfo {
                code: "aa_AA".into(),
                name: "First".into(),
                region: "Test".into(),
                bidirectional: false,
            },
            LanguageInfo {
                code: "zh_CN".into(),
                name: "Overridden Chinese".into(),
                region: "New Region".into(),
                bidirectional: false,
            },
            infos()
                .into_iter()
                .find(|info| info.code == "en_US")
                .unwrap(),
        ];
        screen.replace_languages(replacement);
        assert_eq!(screen.languages.len(), 3, "stale pack entries are removed");
        assert_eq!(
            screen.list.size, 3,
            "the scroll-list extent follows replacement"
        );
        assert_eq!(
            screen
                .languages
                .iter()
                .map(|info| info.code.as_str())
                .collect::<Vec<_>>(),
            ["aa_AA", "en_US", "zh_CN"],
            "replacement remains sorted by code"
        );
        assert_eq!(
            screen.selected().map(|info| info.code.as_str()),
            Some("zh_CN")
        );
        assert_eq!(
            screen.selected().map(|info| info.name.as_str()),
            Some("Overridden Chinese")
        );

        screen.replace_languages(vec![infos()
            .into_iter()
            .find(|info| info.code == "en_US")
            .unwrap()]);
        assert_eq!(
            screen.selected().map(|info| info.code.as_str()),
            Some("en_US")
        );
    }

    #[test]
    fn the_unicode_toggle_flips_the_setting_and_asks_for_a_relayout() {
        let mut s = LanguageScreen::new(infos());
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let (x, y) = (s.unicode.rect.x + 1, s.unicode.rect.y + 1);
        assert_eq!(
            s.mouse_clicked(x, y, 0, &mut settings, &l),
            Action::Relayout
        );
        assert!(settings.force_unicode_font);
    }

    #[test]
    fn done_leaves_without_touching_the_settings() {
        let mut s = LanguageScreen::new(infos());
        let mut settings = GameSettings::default();
        let l = Language::new("en_US");
        s.init(&ctx(), &settings, &l);
        let before = settings.language.clone();
        let (x, y) = (s.done.rect.x + 1, s.done.rect.y + 1);
        assert_eq!(s.mouse_clicked(x, y, 0, &mut settings, &l), Action::Back);
        assert_eq!(settings.language, before);
    }

    #[test]
    fn the_two_bottom_buttons_land_on_vanillas_coordinates() {
        let mut s = LanguageScreen::new(infos());
        s.init(&ctx(), &GameSettings::default(), &Language::new("en_US"));
        assert_eq!(
            (s.unicode.rect.x, s.unicode.rect.y),
            (854 / 2 - 155, 480 - 38)
        );
        assert_eq!(s.done.rect.x, 854 / 2 - 155 + 160);
        assert_eq!(s.done.rect.width, 150);
        // The list stops 61 pixels above the bottom: `height - 65 + 4`.
        assert_eq!(s.list.bottom, 480 - 61);
    }

    #[test]
    fn list_dragging_and_wheel_bounds_match_gui_slot() {
        let languages = (0..30)
            .map(|i| LanguageInfo {
                code: format!("lang_{i:02}"),
                name: format!("Language {i}"),
                region: "Test".into(),
                bidirectional: false,
            })
            .collect();
        let mut s = LanguageScreen::new(languages);
        let short = ScreenCtx::new(ScaledResolution::new(320, 160, 1, false), 0.0);
        s.init(&short, &GameSettings::default(), &Language::new("en_US"));

        let start_y = s.list.top + 5;
        assert!(!s.list.mouse_pressed(10, start_y), "content, not scrollbar");
        s.mouse_dragged(start_y - 20);
        assert_eq!(s.list.scroll_pixels(), 20);
        s.mouse_released();

        let after_drag = s.list.scroll_pixels();
        s.mouse_wheel_at(10, s.list.bottom + 1, -1.0);
        assert_eq!(
            s.list.scroll_pixels(),
            after_drag,
            "bottom controls are outside"
        );
        s.mouse_wheel_at(10, s.list.bottom, -1.0);
        assert_eq!(s.list.scroll_pixels(), after_drag + 9);
    }

    #[test]
    fn background_switches_from_dirt_to_the_world_gradient_in_game() {
        let settings = GameSettings::default();
        let lang = Language::new("en_US");

        let mut out_of_world = LanguageScreen::new(infos());
        out_of_world.init(&ctx(), &settings, &lang);
        let mut dirt_draw = DrawList::new();
        out_of_world.draw(&mut dirt_draw, &mut uniform_font(), &ctx(), (0, 0), &lang);
        assert!(matches!(
            dirt_draw.batches().first().map(|batch| &batch.texture),
            Some(UiTexture::Resource(_))
        ));

        let mut in_world = LanguageScreen::new(infos()).with_in_world(true);
        in_world.init(&ctx(), &settings, &lang);
        let mut world_draw = DrawList::new();
        in_world.draw(&mut world_draw, &mut uniform_font(), &ctx(), (0, 0), &lang);
        assert_eq!(
            world_draw.batches().first().map(|batch| &batch.texture),
            Some(&UiTexture::Solid),
            "a hard-coded dirt call starts with the options-background texture"
        );
    }
}
