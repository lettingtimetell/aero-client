//! Toggle Sprint HUD element: the "[Sprinting (Toggled)]" text (`HudData::sprint_status`).
//!
//! The client (`mc-client/src/app/mods/m3_controls.rs`) decides the
//! status every frame while the mod is on; this element only turns it into cached text. The
//! texts and their priority (research `toggle_sprint.md`): flying, then riding, then sneaking,
//! then sprinting.
//!
//! * The status is a `Copy` enum and every text is a `&'static str`, so a frame whose status
//!   did not change costs one compare: no formatting, no measuring.
//! * The style is the shared readout style with this mod's own defaults (white, shadow,
//!   brackets, no box). See `mc-mods/src/mods/toggle_sprint.rs`.

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::DrawList;

/// What the toggle-sprint text says. Ordered as the client ranks them: an earlier variant wins
/// when several apply (flying beats riding beats sneaking beats sprinting).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SprintStatus {
    /// Nothing to show: not sprinting or sneaking, and the toggle is off.
    #[default]
    Hidden,
    /// `abilities.flying` (creative or spectator flight).
    Flying,
    /// The player is riding an entity.
    Riding,
    /// Sneaking because toggle sneak is on.
    SneakToggled,
    /// Sneaking because the sneak key is held.
    SneakHeld,
    /// The sprint toggle is on (shown even while standing still, so the player can see the
    /// toggle is armed).
    SprintToggled,
    /// Sprinting with the sprint key physically held.
    SprintHeld,
    /// Sprinting from vanilla's double-tap of forward.
    SprintVanilla,
}

impl SprintStatus {
    /// The undecorated text (brackets come from the style). Empty for [`SprintStatus::Hidden`].
    #[must_use]
    pub fn text(self) -> &'static str {
        match self {
            SprintStatus::Hidden => "",
            SprintStatus::Flying => "Flying",
            SprintStatus::Riding => "Riding",
            SprintStatus::SneakToggled => "Sneaking (Toggled)",
            SprintStatus::SneakHeld => "Sneaking (Key Held)",
            SprintStatus::SprintToggled => "Sprinting (Toggled)",
            SprintStatus::SprintHeld => "Sprinting (Key Held)",
            SprintStatus::SprintVanilla => "Sprinting (Vanilla)",
        }
    }
}

/// The `toggle_sprint` element.
#[derive(Debug)]
pub struct ToggleSprintElement {
    style: HudTextStyle,
    /// The `show_text` option.
    show_text: bool,
    /// The status `text` was built for; `None` forces a rebuild.
    shown: Option<SprintStatus>,
    /// The font layout generation `size` was measured with.
    font_generation: u64,
    text: String,
    size: (f32, f32),
}

impl Default for ToggleSprintElement {
    fn default() -> Self {
        ToggleSprintElement {
            style: HudTextStyle::default(),
            show_text: true,
            shown: None,
            font_generation: u64::MAX,
            text: String::new(),
            size: (0.0, 0.0),
        }
    }
}

impl HudElement for ToggleSprintElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("toggle_sprint", "toggle_sprint")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "toggle_sprint");
        self.show_text = settings.bool("toggle_sprint", "show_text");
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        // The editor shows the most common text so the element can be placed while standing
        // still in a menu.
        let status = if ctx.editing {
            SprintStatus::SprintToggled
        } else {
            ctx.data.sprint_status
        };
        if !self.show_text || status == SprintStatus::Hidden {
            return None;
        }
        if self.shown != Some(status) || font.layout_generation() != self.font_generation {
            self.shown = Some(status);
            self.font_generation = font.layout_generation();
            self.style.decorate(status.text(), &mut self.text);
            self.size = self.style.size(font, &self.text);
        }
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        self.style.draw(d, font, &self.text, 0.0, 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings, editing: bool) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (640.0, 360.0),
            pixel_scale: 3.0,
            editing,
        }
    }

    fn data(status: SprintStatus) -> HudData {
        HudData {
            sprint_status: status,
            ..HudData::default()
        }
    }

    #[test]
    fn every_status_has_its_text_and_hidden_has_none() {
        assert_eq!(SprintStatus::default(), SprintStatus::Hidden);
        assert_eq!(SprintStatus::Hidden.text(), "");
        assert_eq!(SprintStatus::Flying.text(), "Flying");
        assert_eq!(SprintStatus::Riding.text(), "Riding");
        assert_eq!(SprintStatus::SneakToggled.text(), "Sneaking (Toggled)");
        assert_eq!(SprintStatus::SneakHeld.text(), "Sneaking (Key Held)");
        assert_eq!(SprintStatus::SprintToggled.text(), "Sprinting (Toggled)");
        assert_eq!(SprintStatus::SprintHeld.text(), "Sprinting (Key Held)");
        assert_eq!(SprintStatus::SprintVanilla.text(), "Sprinting (Vanilla)");
    }

    #[test]
    fn the_default_style_is_bracketed_text_without_a_box() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ToggleSprintElement::default();
        element.sync(&settings);
        let size = element
            .layout(
                &ctx(&data(SprintStatus::SprintToggled), &settings, false),
                &font,
            )
            .expect("a toggled sprint is shown");
        assert_eq!(element.text, "[Sprinting (Toggled)]");
        assert_eq!(
            size,
            (
                font.string_width("[Sprinting (Toggled)]") as f32,
                font.font_height() as f32
            ),
            "no background box by default: the text's own size"
        );
    }

    #[test]
    fn the_text_is_rebuilt_only_when_the_status_changes() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ToggleSprintElement::default();
        element.sync(&settings);
        let held = data(SprintStatus::SprintHeld);
        element.layout(&ctx(&held, &settings, false), &font);
        assert_eq!(element.text, "[Sprinting (Key Held)]");
        let ptr = element.text.as_ptr();
        let capacity = element.text.capacity();
        let len = element.text.len();
        for _ in 0..3 {
            element.layout(&ctx(&held, &settings, false), &font);
        }
        assert_eq!(
            (
                element.text.as_ptr(),
                element.text.capacity(),
                element.text.len()
            ),
            (ptr, capacity, len),
            "same status: no rebuild"
        );
        element.layout(&ctx(&data(SprintStatus::Flying), &settings, false), &font);
        assert_eq!(element.text, "[Flying]");
    }

    #[test]
    fn hidden_or_switched_off_draws_nothing_but_the_editor_shows_a_sample() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ToggleSprintElement::default();
        element.sync(&settings);
        let hidden = data(SprintStatus::Hidden);
        assert_eq!(element.layout(&ctx(&hidden, &settings, false), &font), None);
        assert!(element
            .layout(&ctx(&hidden, &settings, true), &font)
            .is_some());
        assert_eq!(
            element.text, "[Sprinting (Toggled)]",
            "the editor's sample text"
        );

        settings.set_bool("toggle_sprint", "show_text", false);
        element.sync(&settings);
        let sneaking = data(SprintStatus::SneakToggled);
        assert_eq!(
            element.layout(&ctx(&sneaking, &settings, false), &font),
            None,
            "show_text off hides even an active status"
        );
    }

    #[test]
    fn style_options_reach_the_draw() {
        let mut settings = ModSettings::new();
        settings.set_bool("toggle_sprint", "brackets", false);
        settings.set_bool("toggle_sprint", "background", true);
        settings.set_color("toggle_sprint", "text_color", 0xFF00_FF00);
        let mut font = uniform_font();
        let mut element = ToggleSprintElement::default();
        element.sync(&settings);
        let vanilla = data(SprintStatus::SprintVanilla);
        let size = element
            .layout(&ctx(&vanilla, &settings, false), &font)
            .unwrap();
        assert_eq!(element.text, "Sprinting (Vanilla)");
        assert_eq!(size.1, crate::modhud::style::BOX_HEIGHT, "the box is on");
        let mut d = DrawList::new();
        element.draw(&ctx(&vanilla, &settings, false), &mut font, &mut d);
        assert!(d.vertices().iter().any(|v| v.color == [0.0, 1.0, 0.0, 1.0]));
    }
}
