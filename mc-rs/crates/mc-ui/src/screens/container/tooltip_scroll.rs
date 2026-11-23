//! Scrollable item tooltips: the `scrollable_tooltips` mod.
//!
//! Vanilla places a tooltip with [`tooltip_box::layout`] and never moves it again, so a tooltip
//! taller than the screen (a book with thirty enchantments, a server's lore wall) is cut off at
//! the top, and one near the top or left edge can hang off the screen. With the mod on, the mouse
//! wheel moves the tooltip that is showing, but only along an axis where it does not fit:
//!
//! * wheel up moves it down (towards its top), wheel down moves it up;
//! * Shift + wheel moves it sideways (wheel up = right, towards its left edge);
//! * the offset is clamped so the panel stops two pixels inside the screen edge it is being
//!   pulled from;
//! * another tooltip (a different item, a different line list) starts at vanilla's position.
//!
//! The wheel is only taken while the tooltip shown last frame overflows along the wheel's axis
//! ([`TooltipScroll::wheel`] returns `false` otherwise), so the creative inventory keeps scrolling
//! its item grid whenever the hovered tooltip fits.
//!
//! **Off** ([`TooltipScroll::default`]): [`TooltipScroll::draw`] is exactly
//! [`tooltip_box::draw`] and [`TooltipScroll::wheel`] returns `false` at once. Nothing is hashed
//! or stored.

use super::tooltip_box::{self, TooltipLayout};
use crate::font::FontRenderer;
use crate::widget::DrawList;

/// How far inside the screen edge a scrolled panel stops.
const MARGIN: i32 = 2;

/// The scroll state one container screen keeps for its tooltip.
#[derive(Debug, Clone, Default)]
pub struct TooltipScroll {
    /// The mod is on. Set by the client ([`TooltipScroll::set_options`]).
    enabled: bool,
    /// Pixels per wheel notch.
    step: i32,
    /// The current offset from vanilla's position, GUI pixels (positive = right / down).
    offset: (i32, i32),
    /// Identity of the tooltip the offset belongs to (a hash of its lines).
    key: Option<u64>,
    /// Vanilla's (unscrolled) layout of the tooltip drawn last frame and the screen size, or
    /// `None` when no tooltip was drawn: what the wheel acts on.
    last: Option<(TooltipLayout, i32, i32)>,
}

impl TooltipScroll {
    /// Apply the mod's settings. Turning it off forgets any offset.
    pub fn set_options(&mut self, enabled: bool, step: i32) {
        if !enabled && self.enabled {
            *self = TooltipScroll::default();
        }
        self.enabled = enabled;
        self.step = step.max(1);
    }

    /// Whether the mod is on for this screen.
    #[must_use]
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The offset currently applied to the tooltip, GUI pixels.
    #[must_use]
    pub fn offset(&self) -> (i32, i32) {
        self.offset
    }

    /// Draw `lines` as the tooltip at the mouse: [`tooltip_box::draw`], moved by the scroll offset
    /// while the mod is on.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        d: &mut DrawList,
        font: &mut FontRenderer,
        lines: &[String],
        mouse_x: i32,
        mouse_y: i32,
        screen_width: i32,
        screen_height: i32,
    ) {
        if !self.enabled {
            tooltip_box::draw(
                d,
                font,
                lines,
                mouse_x,
                mouse_y,
                screen_width,
                screen_height,
            );
            return;
        }
        let Some(layout) =
            tooltip_box::layout(font, lines, mouse_x, mouse_y, screen_width, screen_height)
        else {
            self.clear();
            return;
        };
        let key = hash_lines(lines);
        if self.key != Some(key) {
            self.key = Some(key);
            self.offset = (0, 0);
        }
        self.offset = clamp_offset(self.offset, &layout, screen_width, screen_height);
        self.last = Some((layout, screen_width, screen_height));
        let moved = TooltipLayout {
            x: layout.x + self.offset.0,
            y: layout.y + self.offset.1,
            ..layout
        };
        tooltip_box::draw_layout(d, font, lines, &moved);
    }

    /// No tooltip this frame: the wheel goes back to the screen, and the next tooltip (even the
    /// same item again) starts at vanilla's position.
    pub fn clear(&mut self) {
        self.last = None;
        self.key = None;
        self.offset = (0, 0);
    }

    /// The mouse wheel (`delta` in notches, positive = up). Moves the tooltip shown last frame
    /// when it overflows the screen along the wheel's axis (`sideways` = Shift held) and returns
    /// `true`, i.e. the screen must not also use this wheel event. Returns `false`, touching
    /// nothing, when the mod is off, no tooltip is showing, or it fits along that axis.
    pub fn wheel(&mut self, delta: f32, sideways: bool) -> bool {
        if !self.enabled || delta == 0.0 || !delta.is_finite() {
            return false;
        }
        let Some((layout, width, height)) = self.last else {
            return false;
        };
        let (lo, hi) = axis_range(&layout, width, height, sideways);
        if lo == 0 && hi == 0 {
            return false;
        }
        let amount = (delta * self.step as f32).round() as i32;
        if sideways {
            self.offset.0 = (self.offset.0 + amount).clamp(lo, hi);
        } else {
            self.offset.1 = (self.offset.1 + amount).clamp(lo, hi);
        }
        true
    }
}

/// The offsets that keep the panel's far edge reachable along one axis: `(lo, hi)` with
/// `lo <= 0 <= hi`, both zero when the panel is fully on screen along that axis.
fn axis_range(layout: &TooltipLayout, width: i32, height: i32, sideways: bool) -> (i32, i32) {
    let (left, top, right, bottom) = layout.panel_bounds();
    let (start, end, extent) = if sideways {
        (left, right, width)
    } else {
        (top, bottom, height)
    };
    let lo = ((extent - MARGIN) - end).min(0);
    let hi = (MARGIN - start).max(0);
    (lo, hi.max(lo))
}

fn clamp_offset(offset: (i32, i32), layout: &TooltipLayout, width: i32, height: i32) -> (i32, i32) {
    let (x_lo, x_hi) = axis_range(layout, width, height, true);
    let (y_lo, y_hi) = axis_range(layout, width, height, false);
    (offset.0.clamp(x_lo, x_hi), offset.1.clamp(y_lo, y_hi))
}

/// FNV-1a over the lines (with a separator), so a different list of lines is a different tooltip.
fn hash_lines(lines: &[String]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for line in lines {
        for byte in line.bytes().chain(std::iter::once(0xFF)) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::widget::test_support::uniform_font;

    fn lines(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("line {i}")).collect()
    }

    fn on() -> TooltipScroll {
        let mut scroll = TooltipScroll::default();
        scroll.set_options(true, 10);
        scroll
    }

    #[test]
    fn off_is_exactly_vanillas_draw_and_never_takes_the_wheel() {
        let mut font = uniform_font();
        let mut scroll = TooltipScroll::default();
        for (n, (mx, my)) in [(1, (10, 10)), (40, (100, 200)), (3, (630, 5))] {
            let mut ours = DrawList::new();
            scroll.draw(&mut ours, &mut font, &lines(n), mx, my, 640, 360);
            let mut vanilla = DrawList::new();
            tooltip_box::draw(&mut vanilla, &mut font, &lines(n), mx, my, 640, 360);
            assert_eq!(ours, vanilla);
            assert!(!scroll.wheel(1.0, false));
            assert!(!scroll.wheel(-1.0, true));
        }
    }

    #[test]
    fn a_tooltip_that_fits_does_not_take_the_wheel_even_when_on() {
        let mut font = uniform_font();
        let mut scroll = on();
        let mut d = DrawList::new();
        scroll.draw(&mut d, &mut font, &lines(3), 100, 100, 640, 360);
        assert!(
            !scroll.wheel(1.0, false),
            "fits: the creative grid keeps the wheel"
        );
        assert!(!scroll.wheel(1.0, true));
        assert_eq!(scroll.offset(), (0, 0));
        let mut vanilla = DrawList::new();
        tooltip_box::draw(&mut vanilla, &mut font, &lines(3), 100, 100, 640, 360);
        assert_eq!(d, vanilla, "on but unscrolled: vanilla's pixels");
    }

    #[test]
    fn a_tall_tooltip_scrolls_down_to_its_top_and_back_to_vanillas_place() {
        let mut font = uniform_font();
        let mut scroll = on();
        let text = lines(50); // 8 + 2 + 49 * 10 = 500 pixels tall on a 360-pixel screen
        let mut d = DrawList::new();
        scroll.draw(&mut d, &mut font, &text, 100, 100, 640, 360);
        let layout = tooltip_box::layout(&font, &text, 100, 100, 640, 360).unwrap();
        let top = layout.panel_bounds().1;
        assert!(top < 0, "vanilla cuts the top off");
        // Wheel down: already at the bottom, nothing to reveal but the wheel is still taken.
        assert!(scroll.wheel(-1.0, false));
        assert_eq!(scroll.offset(), (0, 0));
        // Wheel up: moves down by one step per notch, stopping with the top 2 px inside.
        assert!(scroll.wheel(1.0, false));
        assert_eq!(scroll.offset(), (0, 10));
        assert!(scroll.wheel(3.0, false));
        assert_eq!(scroll.offset(), (0, 40));
        for _ in 0..100 {
            scroll.wheel(1.0, false);
        }
        assert_eq!(scroll.offset(), (0, MARGIN - top));
        // The draw really moves the panel.
        let mut moved = DrawList::new();
        scroll.draw(&mut moved, &mut font, &text, 100, 100, 640, 360);
        let min_y = moved
            .vertices()
            .iter()
            .map(|v| v.pos[1])
            .fold(f32::MAX, f32::min);
        assert_eq!(min_y, MARGIN as f32, "the panel's top edge, 2 px inside");
        // Sideways does nothing for a tooltip that fits horizontally.
        assert!(!scroll.wheel(1.0, true));
    }

    #[test]
    fn another_tooltip_or_no_tooltip_resets_the_offset() {
        let mut font = uniform_font();
        let mut scroll = on();
        let text = lines(50);
        let mut d = DrawList::new();
        scroll.draw(&mut d, &mut font, &text, 100, 100, 640, 360);
        scroll.wheel(2.0, false);
        assert_eq!(scroll.offset(), (0, 20));
        // Same lines, next frame: kept.
        scroll.draw(&mut d, &mut font, &text, 101, 100, 640, 360);
        assert_eq!(scroll.offset(), (0, 20));
        // A different item: back to vanilla's place.
        let mut other = lines(50);
        other[0] = "\u{a7}bDiamond Sword".to_owned();
        scroll.draw(&mut d, &mut font, &other, 101, 100, 640, 360);
        assert_eq!(scroll.offset(), (0, 0));
        scroll.wheel(2.0, false);
        scroll.clear();
        assert!(
            !scroll.wheel(1.0, false),
            "no tooltip shown: the wheel is the screen's"
        );
        scroll.draw(&mut d, &mut font, &other, 101, 100, 640, 360);
        assert_eq!(scroll.offset(), (0, 0));
    }

    #[test]
    fn a_wide_tooltip_scrolls_sideways_with_shift() {
        let mut font = uniform_font();
        let mut scroll = on();
        // One very long line: wider than the screen, so vanilla's 28-pixel jump pushes it off
        // the left edge.
        let text = ["W".repeat(150)];
        let mut d = DrawList::new();
        scroll.draw(&mut d, &mut font, &text, 300, 100, 640, 360);
        let layout = tooltip_box::layout(&font, &text, 300, 100, 640, 360).unwrap();
        let (left, _, right, _) = layout.panel_bounds();
        assert!(left < 0 || right > 640);
        assert!(!scroll.wheel(1.0, false), "fits vertically");
        assert!(scroll.wheel(1.0, true));
        let (lo, hi) = axis_range(&layout, 640, 360, true);
        assert!(lo < 0 || hi > 0);
        for _ in 0..200 {
            scroll.wheel(1.0, true);
        }
        assert_eq!(scroll.offset().0, hi);
        for _ in 0..400 {
            scroll.wheel(-1.0, true);
        }
        assert_eq!(scroll.offset().0, lo);
    }

    #[test]
    fn turning_the_mod_off_forgets_the_offset() {
        let mut font = uniform_font();
        let mut scroll = on();
        let mut d = DrawList::new();
        scroll.draw(&mut d, &mut font, &lines(50), 100, 100, 640, 360);
        scroll.wheel(1.0, false);
        scroll.set_options(false, 10);
        assert_eq!(scroll.offset(), (0, 0));
        assert!(!scroll.enabled());
        assert!(!scroll.wheel(1.0, false));
    }
}
