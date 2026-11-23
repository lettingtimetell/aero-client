//! Buttons and sliders.
//!
//! java: `client/gui/GuiButton.java`, `client/gui/GuiOptionSlider.java`
//!
//! Both are pure state plus arithmetic: the hit test, the hover state and the slider's
//! position→value mapping are all exactly reproducible without a GPU, which is what the tests
//! at the bottom of this file check.

use std::cell::Cell;

use crate::font::FontRenderer;
use crate::widget::draw::{DrawList, Rect, UiTexture, WIDGETS_TEXTURE};

std::thread_local! {
    /// One input-dispatch-local replacement for Java's direct `GuiButton.playPressSound` call.
    ///
    /// Screens deliberately return application actions rather than owning audio. Recording the
    /// primitive hit here preserves the missing fact without making every screen's public return
    /// type carry it. The driver brackets exactly one mouse dispatch with
    /// [`capture_button_press`], so direct widget tests and later dispatches cannot leak a hit.
    static BUTTON_PRESS_OBSERVED: Cell<bool> = const { Cell::new(false) };

    /// The pointer's x in GUI pixels **with its sub-pixel fraction** (X1, item 33). See [`set_precise_cursor_x`].
    static PRECISE_CURSOR_X: Cell<Option<f32>> = const { Cell::new(None) };
}

/// Tell the option sliders where the pointer really is: `gui_x` is the cursor's x in GUI pixels
/// as an exact fraction (the physical pixel times `scaled_width / display_width`, from
/// [`crate::widget::ScaledResolution::cursor_to_gui_precise`]). The driver sets it once per
/// frame next to its integer GUI cursor.
///
/// Vanilla maps a slider from whole GUI pixels, so at GUI scale 2 a 150-pixel slider has 143
/// positions for sensitivity's 201 values and some percentages cannot be reached. With this,
/// [`Slider::position_for`] resolves physical pixels. It is used only while the integer
/// `mouse_x` a screen passes is this value floored, so a stale value or a test's synthetic
/// coordinate falls back to vanilla's whole pixels.
pub fn set_precise_cursor_x(gui_x: Option<f32>) {
    PRECISE_CURSOR_X.with(|x| x.set(gui_x.filter(|v| v.is_finite())));
}

/// `mouse_x` refined by [`set_precise_cursor_x`] when the two agree.
fn precise_x(mouse_x: i32) -> f32 {
    PRECISE_CURSOR_X
        .with(Cell::get)
        .filter(|x| x.floor() as i32 == mouse_x)
        .unwrap_or(mouse_x as f32)
}

/// A slider position (`0..=1`) snapped to the nearest of `divisions` equal steps (item 33):
/// "Sensitivity: 87%" then stores exactly step 174 of 200.
///
/// The step is returned as the smallest `f32` at or above `k / divisions`, because that
/// quotient can round a hair below `k` steps (53 % is 0.265, stored as 0.26499998) and
/// vanilla's labels truncate (`(int)(value * 200.0F)`), which would show 52 % and make 53 %
/// unreachable.
#[must_use]
pub fn snap_position(position: f32, divisions: f32) -> f32 {
    if !(divisions.is_finite() && divisions >= 1.0) {
        return position.clamp(0.0, 1.0);
    }
    step_at(position.clamp(0.0, 1.0) * divisions, divisions)
}

/// The position of the step nearest `steps` (in units of `1 / divisions`); see
/// [`snap_position`].
fn step_at(steps: f32, divisions: f32) -> f32 {
    let k = steps.round().clamp(0.0, divisions);
    let mut position = (k / divisions).clamp(0.0, 1.0);
    while position < 1.0 && position * divisions < k {
        position = f32::from_bits(position.to_bits() + 1);
    }
    position
}

/// Run one screen mouse dispatch and report whether an enabled, visible [`Button`] accepted it.
///
/// This is intentionally about the widget hit, not the screen's resulting action: vanilla plays
/// `gui.button.press` before `actionPerformed`, including buttons whose handler changes only
/// local screen state. Chat components, list rows, text fields, and container slots never call
/// [`Button::mouse_pressed`] and therefore remain silent.
pub fn capture_button_press<T>(dispatch: impl FnOnce() -> T) -> (T, bool) {
    BUTTON_PRESS_OBSERVED.with(|observed| observed.set(false));
    let result = dispatch();
    let hit = BUTTON_PRESS_OBSERVED.with(|observed| observed.replace(false));
    (result, hit)
}

/// Vanilla's default button size.
pub const DEFAULT_BUTTON_WIDTH: i32 = 200;
/// Vanilla's default button height.
pub const DEFAULT_BUTTON_HEIGHT: i32 = 20;

/// Label colour for an enabled, un-hovered button. `0xE0E0E0`.
pub const TEXT_IDLE: i32 = 14_737_632;
/// Label colour for a disabled button. `0xA0A0A0`.
pub const TEXT_DISABLED: i32 = 10_526_880;
/// Label colour for a hovered button. `0xFFFFA0`.
pub const TEXT_HOVERED: i32 = 16_777_120;

/// A vanilla push button.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Button {
    /// The id passed back to the screen's action handler.
    pub id: i32,
    /// Position and size in GUI pixels.
    pub rect: Rect,
    /// The label. Already translated — widgets do not touch [`mc_assets::Language`].
    pub text: String,
    /// A disabled button draws greyed out and refuses clicks.
    pub enabled: bool,
    /// An invisible button draws nothing and refuses clicks.
    pub visible: bool,
    hovered: bool,
}

impl Button {
    /// A 200×20 button.
    #[must_use]
    pub fn new(id: i32, x: i32, y: i32, text: impl Into<String>) -> Self {
        Self::sized(id, x, y, DEFAULT_BUTTON_WIDTH, DEFAULT_BUTTON_HEIGHT, text)
    }

    /// A button of an explicit size.
    #[must_use]
    pub fn sized(
        id: i32,
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        text: impl Into<String>,
    ) -> Self {
        Self {
            id,
            rect: Rect::new(x, y, width, height),
            text: text.into(),
            enabled: true,
            visible: true,
            hovered: false,
        }
    }

    /// Which of the three 20-pixel rows of `widgets.png` to sample: 0 disabled, 1 idle,
    /// 2 hovered.
    // java: GuiButton.java:59 (getHoverState)
    #[must_use]
    pub fn hover_state(&self, mouse_over: bool) -> u32 {
        if !self.enabled {
            0
        } else if mouse_over {
            2
        } else {
            1
        }
    }

    /// The hover flag as of the last [`draw`](Self::draw). Vanilla's `isMouseOver`.
    #[must_use]
    pub fn is_mouse_over(&self) -> bool {
        self.hovered
    }

    /// Whether `(x, y)` is inside the button's rectangle, ignoring enabled/visible.
    #[must_use]
    pub fn contains(&self, x: i32, y: i32) -> bool {
        self.rect.contains(x, y)
    }

    /// Vanilla's `mousePressed`: true when the button is live and the point is inside.
    // java: GuiButton.java:126
    #[must_use]
    pub fn mouse_pressed(&self, x: i32, y: i32) -> bool {
        let pressed = self.enabled && self.visible && self.contains(x, y);
        if pressed {
            BUTTON_PRESS_OBSERVED.with(|observed| observed.set(true));
        }
        pressed
    }

    /// Label colour for the current state.
    // java: GuiButton.java:93
    #[must_use]
    pub fn text_color(&self) -> i32 {
        if !self.enabled {
            TEXT_DISABLED
        } else if self.hovered {
            TEXT_HOVERED
        } else {
            TEXT_IDLE
        }
    }

    /// Update the cached hover flag and emit the button's artwork and label.
    ///
    /// The label is centred on `x + width/2` and sits at `y + (height - 8) / 2` — note the
    /// literal 8, not [`crate::font::FONT_HEIGHT`]: vanilla centres on the glyph box, not on
    /// the line box, which is why a 20-pixel button puts text at `y + 6`.
    // java: GuiButton.java:78 (drawButton)
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse_x: i32, mouse_y: i32) {
        if !self.visible {
            return;
        }
        self.hovered = self.contains(mouse_x, mouse_y);
        let state = self.hover_state(self.hovered);
        d.blit_button(&UiTexture::gui(WIDGETS_TEXTURE), self.rect, state);
        let color = self.text_color();
        let cx = self.rect.x + self.rect.width / 2;
        let cy = self.rect.y + (self.rect.height - 8) / 2;
        draw_centered_string(d, font, &self.text, cx, cy, color);
    }
}

/// `Gui.drawCenteredString`: centred on `x`, with a drop shadow.
///
/// The half-width uses **integer** division, so an odd-width string sits half a pixel left of
/// true centre. Vanilla does this; matching it keeps every screen's text on the same pixels.
// java: Gui.java:122
pub fn draw_centered_string(
    d: &mut DrawList,
    font: &mut FontRenderer,
    text: &str,
    x: i32,
    y: i32,
    color: i32,
) -> i32 {
    let x = x - font.string_width(text) / 2;
    font.draw_string_with_shadow(d, text, x as f32, y as f32, color)
}

/// `Gui.drawString`: left-aligned, with a drop shadow.
// java: Gui.java:130
pub fn draw_string(
    d: &mut DrawList,
    font: &mut FontRenderer,
    text: &str,
    x: i32,
    y: i32,
    color: i32,
) -> i32 {
    font.draw_string_with_shadow(d, text, x as f32, y as f32, color)
}

/// A horizontal option slider, drawn as a flat button with a draggable 8×20 handle.
///
/// The label is supplied by the settings model and is deliberately not synthesized here.
/// Vanilla's reachable `GuiOptionSlider` asks `GameSettings` for its
/// complete display string; the similarly named `GuiSlider` formatter belongs exclusively to
/// the unreachable single-player custom-world screen.
#[derive(Debug, Clone)]
pub struct Slider {
    /// The underlying button (position, size, label, enabled/visible).
    pub button: Button,
    /// Value at position 0.
    pub min: f32,
    /// Value at position 1.
    pub max: f32,
    /// Normalised handle position in `0..=1`.
    position: f32,
    /// Whether the handle is currently being dragged.
    pub dragging: bool,
}

impl Slider {
    /// Vanilla's slider is 150×20.
    pub const DEFAULT_WIDTH: i32 = 150;
    /// The handle is 8 pixels wide (two 4-pixel halves of `widgets.png`).
    pub const HANDLE_WIDTH: i32 = 8;

    /// A slider spanning `min..=max`, starting at `value`.
    // java: GuiOptionSlider.java:21
    #[must_use]
    pub fn new(
        id: i32,
        x: i32,
        y: i32,
        name: impl Into<String>,
        min: f32,
        max: f32,
        value: f32,
    ) -> Self {
        let text = name.into();
        Self {
            button: Button::sized(id, x, y, Self::DEFAULT_WIDTH, DEFAULT_BUTTON_HEIGHT, text),
            min,
            max,
            position: if (max - min).abs() < f32::EPSILON {
                0.0
            } else {
                (value - min) / (max - min)
            },
            dragging: false,
        }
    }

    /// The value the handle is currently over.
    // java: GuiOptionSlider.java:39 (sliderValue mapped through the option)
    #[must_use]
    pub fn value(&self) -> f32 {
        self.min + (self.max - self.min) * self.position
    }

    /// The normalised handle position.
    #[must_use]
    pub fn position(&self) -> f32 {
        self.position
    }

    /// Move the handle to `value`.
    pub fn set_value(&mut self, value: f32) {
        self.position = if (self.max - self.min).abs() < f32::EPSILON {
            0.0
        } else {
            (value - self.min) / (self.max - self.min)
        };
    }

    /// Move the handle to a normalised position.
    pub fn set_position(&mut self, position: f32) {
        self.position = position;
    }

    /// Where the left edge of the handle sits, in GUI pixels.
    ///
    /// The handle travels `width - 8` pixels, so it stays inside the track at both ends.
    // java: GuiOptionSlider.java:52
    #[must_use]
    pub fn handle_x(&self) -> i32 {
        self.button.rect.x
            + (self.position * (self.button.rect.width - Self::HANDLE_WIDTH) as f32) as i32
    }

    /// The position a click at `mouse_x` maps to, clamped to `0..=1`. Vanilla's arithmetic,
    /// on the sub-pixel pointer x when the driver supplied one ([`set_precise_cursor_x`]).
    // java: GuiOptionSlider.java:42, 73
    #[must_use]
    pub fn position_for(&self, mouse_x: i32) -> f32 {
        let travel = (self.button.rect.width - Self::HANDLE_WIDTH) as f32;
        if travel <= 0.0 {
            return 0.0;
        }
        ((precise_x(mouse_x) - (self.button.rect.x + 4) as f32) / travel).clamp(0.0, 1.0)
    }

    /// Fine adjustment (item 33): move the handle `steps` of `1 / divisions` of the travel
    /// (the option's display step: 200 for sensitivity's 0..200 %), starting from the division
    /// nearest the handle, clamped to the ends.
    pub fn step_position(&mut self, steps: i32, divisions: f32) {
        if divisions.is_finite() && divisions >= 1.0 {
            let at = (self.position * divisions).round() + steps as f32;
            self.position = step_at(at, divisions);
        }
    }

    /// Press: jumps the handle to the click and starts dragging. Returns whether it was hit.
    // java: GuiOptionSlider.java:66 (mousePressed)
    pub fn mouse_pressed(&mut self, x: i32, y: i32) -> bool {
        if self.button.mouse_pressed(x, y) {
            self.set_position(self.position_for(x));
            self.dragging = true;
            true
        } else {
            false
        }
    }

    /// Drag: follows the mouse while held. A no-op when not dragging.
    // java: GuiOptionSlider.java:36 (mouseDragged)
    pub fn mouse_dragged(&mut self, x: i32) {
        if self.dragging && self.button.visible {
            self.set_position(self.position_for(x));
        }
    }

    /// Release: stops dragging.
    // java: GuiOptionSlider.java:88 (mouseReleased)
    pub fn mouse_released(&mut self) {
        self.dragging = false;
    }

    /// Draw the flat track, the handle, and the label.
    ///
    /// A slider's `getHoverState` is hard-coded to 0, so it always uses the *disabled* button
    /// row as its track — that is why sliders look flat next to buttons.
    // java: GuiOptionSlider.java:28
    pub fn draw(&mut self, d: &mut DrawList, font: &mut FontRenderer, mouse_x: i32, mouse_y: i32) {
        if !self.button.visible {
            return;
        }
        self.button.hovered = self.button.contains(mouse_x, mouse_y);
        let tex = UiTexture::gui(WIDGETS_TEXTURE);
        d.blit_button(&tex, self.button.rect, 0);
        let hx = self.handle_x();
        d.blit(&tex, hx, self.button.rect.y, 0, 66, 4, 20);
        d.blit(&tex, hx + 4, self.button.rect.y, 196, 66, 4, 20);
        let color = self.button.text_color();
        let cx = self.button.rect.x + self.button.rect.width / 2;
        let cy = self.button.rect.y + (self.button.rect.height - 8) / 2;
        draw_centered_string(d, font, &self.button.text, cx, cy, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_default_button_is_two_hundred_by_twenty() {
        let b = Button::new(1, 10, 20, "Done");
        assert_eq!(b.rect, Rect::new(10, 20, 200, 20));
        assert!(b.enabled && b.visible);
    }

    #[test]
    fn the_hit_rectangle_is_half_open() {
        let b = Button::new(0, 100, 50, "x");
        assert!(b.mouse_pressed(100, 50), "top-left corner");
        assert!(b.mouse_pressed(299, 69), "bottom-right pixel");
        assert!(!b.mouse_pressed(300, 69), "one past the right edge");
        assert!(!b.mouse_pressed(299, 70), "one past the bottom edge");
        assert!(!b.mouse_pressed(99, 50));
        assert!(!b.mouse_pressed(100, 49));
    }

    #[test]
    fn a_disabled_or_hidden_button_refuses_clicks_but_still_hit_tests() {
        let mut b = Button::new(0, 0, 0, "x");
        b.enabled = false;
        assert!(!b.mouse_pressed(10, 10));
        assert!(b.contains(10, 10), "contains ignores enabled");
        b.enabled = true;
        b.visible = false;
        assert!(!b.mouse_pressed(10, 10));
    }

    #[test]
    fn a_dispatch_scope_reports_the_widget_hit_independently_of_its_result() {
        let mut button = Button::new(0, 0, 0, "actionless");
        let (result, hit) = capture_button_press(|| {
            assert!(button.mouse_pressed(10, 10));
            None::<u8>
        });
        assert_eq!(result, None, "the screen may produce no application action");
        assert!(hit, "the accepted GuiButton hit is an independent fact");

        button.enabled = false;
        let (_, hit) = capture_button_press(|| button.mouse_pressed(10, 10));
        assert!(!hit, "disabled buttons neither accept nor sound");

        button.enabled = true;
        assert!(
            button.mouse_pressed(10, 10),
            "leave an unscoped stale marker"
        );
        let (_, hit) = capture_button_press(|| false);
        assert!(
            !hit,
            "each dispatch clears observations made outside its scope"
        );
    }

    #[test]
    fn hover_state_picks_the_texture_row() {
        let mut b = Button::new(0, 0, 0, "x");
        assert_eq!(b.hover_state(false), 1);
        assert_eq!(b.hover_state(true), 2);
        b.enabled = false;
        assert_eq!(b.hover_state(true), 0, "disabled wins over hovered");
    }

    #[test]
    fn text_colour_follows_the_state() {
        let mut b = Button::new(0, 0, 0, "x");
        assert_eq!(b.text_color(), TEXT_IDLE);
        b.hovered = true;
        assert_eq!(b.text_color(), TEXT_HOVERED);
        b.enabled = false;
        assert_eq!(b.text_color(), TEXT_DISABLED);
    }

    #[test]
    fn slider_position_maps_to_value_linearly() {
        let s = Slider::new(0, 0, 0, "FOV", 30.0, 110.0, 70.0);
        assert!((s.position() - 0.5).abs() < 1e-6);
        assert!((s.value() - 70.0).abs() < 1e-4);
    }

    #[test]
    fn a_click_four_pixels_in_is_position_zero() {
        let mut s = Slider::new(0, 100, 0, "x", 0.0, 1.0, 0.0);
        // The track is 150 wide and the handle 8, so the travel is 142 and the anchor is x+4.
        assert_eq!(s.position_for(104), 0.0);
        assert_eq!(s.position_for(104 + 142), 1.0);
        assert!((s.position_for(104 + 71) - 0.5).abs() < 1e-6);
        // Outside the track clamps rather than extrapolating.
        assert_eq!(s.position_for(0), 0.0);
        assert_eq!(s.position_for(10_000), 1.0);

        assert!(s.mouse_pressed(104 + 71, 10));
        assert!((s.position() - 0.5).abs() < 1e-6);
        assert!(s.dragging);
    }

    #[test]
    fn the_handle_stays_inside_the_track_at_both_ends() {
        let mut s = Slider::new(0, 100, 0, "x", 0.0, 1.0, 0.0);
        s.set_position(0.0);
        assert_eq!(s.handle_x(), 100);
        s.set_position(1.0);
        assert_eq!(s.handle_x(), 100 + 150 - 8, "the right edge is flush");
    }

    #[test]
    fn dragging_only_moves_the_handle_while_held() {
        let mut s = Slider::new(0, 0, 0, "x", 0.0, 1.0, 0.0);
        s.mouse_dragged(75);
        assert_eq!(s.position(), 0.0, "not dragging yet");
        assert!(s.mouse_pressed(4, 10));
        s.mouse_dragged(4 + 142);
        assert_eq!(s.position(), 1.0);
        s.mouse_released();
        s.mouse_dragged(4);
        assert_eq!(s.position(), 1.0, "released, so the drag is ignored");
    }

    #[test]
    fn a_press_outside_the_slider_does_not_start_a_drag() {
        let mut s = Slider::new(0, 0, 0, "x", 0.0, 1.0, 0.5);
        assert!(!s.mouse_pressed(500, 500));
        assert!(!s.dragging);
        assert_eq!(s.position(), 0.5);
    }

    #[test]
    fn a_degenerate_range_does_not_divide_by_zero() {
        let mut s = Slider::new(0, 0, 0, "x", 5.0, 5.0, 5.0);
        assert_eq!(s.position(), 0.0);
        assert_eq!(s.value(), 5.0);
        s.set_value(9.0);
        assert_eq!(s.position(), 0.0);
        // A zero-width slider cannot divide by its travel either.
        let mut z = Slider::new(0, 0, 0, "x", 0.0, 1.0, 0.0);
        z.button.rect.width = 8;
        assert_eq!(z.position_for(4), 0.0);
    }

    #[test]
    fn the_complete_settings_label_is_preserved_while_the_value_changes() {
        let mut s = Slider::new(7, 0, 0, "FOV: 90", 0.0, 1.0, 0.75);
        assert_eq!(s.button.text, "FOV: 90");
        s.set_value(1.0);
        assert_eq!(s.button.text, "FOV: 90");
        assert!(s.mouse_pressed(4, 10));
        s.mouse_dragged(146);
        assert_eq!(s.button.text, "FOV: 90");
    }

    #[test]
    fn a_failed_press_does_not_cancel_an_existing_drag() {
        let mut s = Slider::new(0, 0, 0, "x", 0.0, 1.0, 0.0);
        assert!(s.mouse_pressed(4, 10));
        assert!(!s.mouse_pressed(500, 500));
        assert!(s.dragging);
        s.mouse_dragged(146);
        assert_eq!(s.position(), 1.0);
    }

    #[test]
    fn an_active_drag_ignores_enabled_but_respects_visible() {
        let mut s = Slider::new(0, 0, 0, "x", 0.0, 1.0, 0.0);
        assert!(s.mouse_pressed(4, 10));
        s.button.enabled = false;
        s.mouse_dragged(146);
        assert_eq!(
            s.position(),
            1.0,
            "GuiOptionSlider checks only visible while dragging"
        );

        s.button.visible = false;
        s.mouse_dragged(4);
        assert_eq!(s.position(), 1.0, "a hidden slider freezes until release");
        s.mouse_released();
        assert!(!s.dragging);
    }
}
