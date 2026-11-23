//! A checkbox.
//!
//! **1.8.9 has no checkbox.** Every boolean option in the vanilla 1.8.9 GUI is a
//! `GuiOptionButton` that cycles its own label (`"Fullscreen: ON"`), and `GuiCheckBox` only
//! arrives in 1.9. S15's task list asks for one, so this is a forward-port of 1.9's
//! `client/gui/GuiCheckBox.java` — the drawing is two nested `drawRect`s plus a centred `x`,
//! which is exactly what that class does and needs no artwork the 1.8.9 assets lack.
//!
//! [`ToggleButton`] is the *actual* 1.8.9 shape and is what an options screen should use.

use crate::font::FontRenderer;
use crate::widget::button::{draw_centered_string, draw_string, Button, TEXT_IDLE};
use crate::widget::draw::{DrawList, Rect};

/// The box is 11×11.
pub const BOX_SIZE: i32 = 11;
/// Outline colour, `0xFFA0A0A0`.
pub const OUTLINE: u32 = 0xFFA0_A0A0;
/// Fill colour, opaque black.
pub const FILL: u32 = 0xFF00_0000;

/// A labelled checkbox.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Checkbox {
    /// The underlying button state (id, position, label, enabled/visible).
    pub button: Button,
    /// Whether the box is ticked.
    pub checked: bool,
    hovered: bool,
}

impl Checkbox {
    /// A checkbox whose clickable area spans the box plus its label.
    ///
    /// The width needs the font, because it is `box + 2 + string_width(label)`.
    // java (1.9): GuiCheckBox.java:16
    #[must_use]
    pub fn new(
        id: i32,
        x: i32,
        y: i32,
        label: impl Into<String>,
        checked: bool,
        font: &FontRenderer,
    ) -> Self {
        let label = label.into();
        let width = BOX_SIZE + 2 + font.string_width(&label);
        Self {
            button: Button::sized(id, x, y, width, BOX_SIZE, label),
            checked,
            hovered: false,
        }
    }

    /// Re-measure the label after changing it.
    pub fn relayout(&mut self, font: &FontRenderer) {
        self.button.rect.width = BOX_SIZE + 2 + font.string_width(&self.button.text);
    }

    /// Press: toggles when hit. Returns whether it was hit.
    // java (1.9): GuiCheckBox.java:44
    pub fn mouse_pressed(&mut self, x: i32, y: i32) -> bool {
        if self.button.mouse_pressed(x, y) {
            self.checked = !self.checked;
            true
        } else {
            false
        }
    }

    /// Draw the box, the tick and the label.
    // java (1.9): GuiCheckBox.java:26
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse_x: i32, mouse_y: i32) {
        if !self.button.visible {
            return;
        }
        let r = self.button.rect;
        self.hovered = r.contains(mouse_x, mouse_y);
        d.rect(r.x, r.y, r.x + BOX_SIZE, r.y + r.height, OUTLINE);
        d.rect(
            r.x + 1,
            r.y + 1,
            r.x + BOX_SIZE - 1,
            r.y + r.height - 1,
            FILL,
        );
        if self.checked {
            draw_centered_string(d, font, "x", r.x + BOX_SIZE / 2, r.y + 1, TEXT_IDLE);
        }
        let color = self.button.text_color();
        draw_string(
            d,
            font,
            &self.button.text,
            r.x + BOX_SIZE + 2,
            r.y + 2,
            color,
        );
    }

    /// Whether the pointer was over the checkbox at the last [`draw`](Self::draw).
    ///
    /// A checkbox has no hover artwork — 1.9's does not either — but screens still want to know.
    #[must_use]
    pub fn is_mouse_over(&self) -> bool {
        self.hovered
    }
}

/// Vanilla 1.8.9's actual boolean widget: a button whose label carries the value.
///
/// java: `client/gui/GuiOptionButton.java` + `settings/GameSettings.getKeyBinding`
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToggleButton {
    /// The underlying button.
    pub button: Button,
    /// The option name, without the value suffix.
    pub label: String,
    /// The current value.
    pub value: bool,
    /// Text for `true`. Vanilla uses the translated `options.on`.
    pub on_text: String,
    /// Text for `false`.
    pub off_text: String,
}

impl ToggleButton {
    /// A 150×20 option button.
    #[must_use]
    pub fn new(id: i32, x: i32, y: i32, label: impl Into<String>, value: bool) -> Self {
        let label = label.into();
        let mut t = Self {
            button: Button::sized(id, x, y, 150, 20, ""),
            label,
            value,
            on_text: "ON".to_owned(),
            off_text: "OFF".to_owned(),
        };
        t.refresh();
        t
    }

    /// Recompute the label from the current value.
    pub fn refresh(&mut self) {
        let v = if self.value {
            &self.on_text
        } else {
            &self.off_text
        };
        self.button.text = format!("{}: {}", self.label, v);
    }

    /// Set the value and refresh the label.
    pub fn set(&mut self, value: bool) {
        self.value = value;
        self.refresh();
    }

    /// Press: flips the value when hit.
    pub fn mouse_pressed(&mut self, x: i32, y: i32) -> bool {
        if self.button.mouse_pressed(x, y) {
            self.set(!self.value);
            true
        } else {
            false
        }
    }

    /// Draw as a normal button.
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse_x: i32, mouse_y: i32) {
        self.button.draw(d, font, mouse_x, mouse_y);
    }
}

/// A rectangle helper shared by both toggles, exported for screens that want their own.
#[must_use]
pub fn checkbox_bounds(x: i32, y: i32, label_width: i32) -> Rect {
    Rect::new(x, y, BOX_SIZE + 2 + label_width, BOX_SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn font() -> FontRenderer {
        crate::widget::test_support::uniform_font()
    }

    #[test]
    fn a_checkbox_is_the_box_plus_two_plus_its_label() {
        let f = font();
        let c = Checkbox::new(1, 10, 20, "Fancy", false, &f);
        // Five 5-pixel glyphs.
        assert_eq!(c.button.rect, Rect::new(10, 20, 11 + 2 + 25, 11));
        assert_eq!(checkbox_bounds(10, 20, 25), c.button.rect);
    }

    #[test]
    fn clicking_anywhere_on_the_row_toggles_it() {
        let f = font();
        let mut c = Checkbox::new(1, 10, 20, "Fancy", false, &f);
        assert!(c.mouse_pressed(12, 22), "on the box");
        assert!(c.checked);
        assert!(c.mouse_pressed(40, 25), "on the label");
        assert!(!c.checked);
        assert!(!c.mouse_pressed(9, 20), "one pixel left of the box");
        assert!(!c.checked, "a miss must not toggle");
        assert!(!c.mouse_pressed(10, 31), "one pixel below");
    }

    #[test]
    fn a_disabled_checkbox_refuses_clicks() {
        let f = font();
        let mut c = Checkbox::new(1, 0, 0, "x", false, &f);
        c.button.enabled = false;
        assert!(!c.mouse_pressed(2, 2));
        assert!(!c.checked);
    }

    #[test]
    fn drawing_emits_the_outline_fill_and_tick() {
        let f = font();
        let mut font = font();
        let mut c = Checkbox::new(1, 0, 0, "AB", true, &f);
        let mut d = DrawList::new();
        c.draw(&mut d, &mut font, 2, 2);
        assert!(c.is_mouse_over());
        // Two rects, then the 'x' glyph pair (shadow + main), then two label glyph pairs.
        assert!(d.vertices().len() >= 4 * 6);
        d.clear();
        c.checked = false;
        c.draw(&mut d, &mut font, 500, 500);
        assert!(!c.is_mouse_over());
    }

    #[test]
    fn relayout_tracks_a_changed_label() {
        let f = font();
        let mut c = Checkbox::new(1, 0, 0, "AB", false, &f);
        assert_eq!(c.button.rect.width, 11 + 2 + 10);
        c.button.text = "ABCD".into();
        c.relayout(&f);
        assert_eq!(c.button.rect.width, 11 + 2 + 20);
    }

    #[test]
    fn the_vanilla_toggle_button_carries_its_value_in_the_label() {
        let mut t = ToggleButton::new(3, 0, 0, "Fullscreen", false);
        assert_eq!(t.button.text, "Fullscreen: OFF");
        assert!(t.mouse_pressed(10, 10));
        assert!(t.value);
        assert_eq!(t.button.text, "Fullscreen: ON");
        t.on_text = "AN".into();
        t.set(true);
        assert_eq!(t.button.text, "Fullscreen: AN");
        assert!(!t.mouse_pressed(1000, 1000));
        assert!(t.value, "a miss does not toggle");
    }
}
