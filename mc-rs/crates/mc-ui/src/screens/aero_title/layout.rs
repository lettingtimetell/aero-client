//! Where the title screen's pieces go: a pure function of the window size, so it is tested
//! without a GPU and lives in exactly one place.
//!
//! **The layout.** Everything hangs off one left margin. The title's capitals are centred 1/8 of
//! the height from the top (the user's "7/8ths to the top"). The four buttons stack in a column
//! directly under it, left edges on the title's ink, with a little extra air before Quit Game,
//! the one destructive button. That keeps the whole menu in the top-left quarter, reading as one
//! block with the title, and leaves the rest of the screen to the greenhouse. A column under a
//! left-aligned title is also the one arrangement that never covers the scene's middle, where
//! the path and the waterfalls are.
//!
//! Sizes are in design pixels for a 1920x1080 window, scaled by [`scale`].

use super::AeroButton;
use crate::screens::modmenu::paint::Rect;

/// Left margin of the title's ink and of the button column.
pub const MARGIN_X: f32 = 128.0;
/// The title's em size.
pub const TITLE_SIZE: f32 = 116.0;
/// From the title's baseline to the first button's top.
pub const TITLE_TO_BUTTONS: f32 = 66.0;
/// One button.
pub const BUTTON_W: f32 = 340.0;
/// One button.
pub const BUTTON_H: f32 = 56.0;
/// Between buttons.
pub const BUTTON_GAP: f32 = 12.0;
/// Extra space before Quit Game.
pub const QUIT_GAP: f32 = 18.0;
/// Button corner radius.
pub const BUTTON_RADIUS: f32 = 16.0;
/// Distance of the footer text's bottom from the window's bottom.
pub const FOOTER_BOTTOM: f32 = 22.0;

/// Physical pixels per design pixel: the height decides (a 1080p window is 1), unless the window
/// is so narrow that the title would not fit.
#[must_use]
pub fn scale(screen: (f32, f32)) -> f32 {
    let fit = (screen.1 / 1080.0).min(screen.0 / 1536.0);
    if fit.is_finite() {
        fit.clamp(0.4, 4.0)
    } else {
        1.0
    }
}

/// One frame's layout, in physical pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Layout {
    /// Physical pixels per design pixel.
    pub s: f32,
    /// The window.
    pub screen: (f32, f32),
    /// Left edge of the title's ink and of the buttons.
    pub left: f32,
    /// The title's em size.
    pub title_size: f32,
    /// The title's baseline: its capitals are centred on `screen.1 / 8`.
    pub title_baseline: f32,
    /// The buttons, in [`AeroButton::ALL`] order.
    pub buttons: [Rect; AeroButton::COUNT],
    /// Button corner radius.
    pub radius: f32,
    /// The footer text's bottom edge.
    pub footer_bottom: f32,
}

impl Layout {
    /// The layout for a `screen`-sized window, given the title font's cap height at an em size.
    #[must_use]
    pub fn new(screen: (f32, f32), cap_height: impl Fn(f32) -> f32) -> Layout {
        let s = scale(screen);
        let left = (MARGIN_X * s).round();
        let title_size = TITLE_SIZE * s;
        let cap = cap_height(title_size);
        let title_baseline = (screen.1 / 8.0 + cap * 0.5).round();
        let mut y = title_baseline + TITLE_TO_BUTTONS * s;
        let (w, h) = ((BUTTON_W * s).round(), (BUTTON_H * s).round());
        let mut buttons = [Rect::default(); AeroButton::COUNT];
        for (index, button) in buttons.iter_mut().enumerate() {
            if index == AeroButton::Quit.index() {
                y += QUIT_GAP * s;
            }
            *button = Rect::new(left, y.round(), w, h);
            y += h + BUTTON_GAP * s;
        }
        Layout {
            s,
            screen,
            left,
            title_size,
            title_baseline,
            buttons,
            radius: BUTTON_RADIUS * s,
            footer_bottom: screen.1 - FOOTER_BOTTOM * s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(size: f32) -> f32 {
        size * 0.727
    }

    #[test]
    fn at_1080p_the_title_is_centred_an_eighth_down_and_the_column_sits_under_it() {
        let layout = Layout::new((1920.0, 1080.0), cap);
        assert_eq!(layout.s, 1.0);
        let cap_centre = layout.title_baseline - cap(layout.title_size) * 0.5;
        assert!((cap_centre - 135.0).abs() <= 1.0, "{cap_centre}");
        for button in layout.buttons {
            assert_eq!(button.x, layout.left, "one left edge");
            assert!(button.y > layout.title_baseline);
        }
        // In order, not overlapping, Quit a little further away.
        let b = layout.buttons;
        let gap = |i: usize| b[i + 1].y - b[i].bottom();
        assert_eq!(gap(0), gap(1));
        let quit = AeroButton::Quit.index();
        assert!(gap(quit - 1) > gap(quit - 2));
        // The whole menu stays in the top part of the screen: the greenhouse keeps the rest.
        // (Replays, and later Mods, joined the column; it was the top
        // half with four buttons.)
        let last = b[AeroButton::COUNT - 1];
        assert!(last.bottom() < 1080.0 * 0.62, "{}", last.bottom());
    }

    #[test]
    fn the_layout_scales_with_the_height_and_stays_inside_small_windows() {
        let big = Layout::new((3840.0, 2160.0), cap);
        assert_eq!(big.s, 2.0);
        assert_eq!(big.buttons[0].w, 2.0 * BUTTON_W);
        for screen in [
            (1280.0, 720.0),
            (854.0, 480.0),
            (800.0, 1000.0),
            (3440.0, 1440.0),
        ] {
            let layout = Layout::new(screen, cap);
            let last = layout.buttons[AeroButton::COUNT - 1];
            assert!(last.bottom() < screen.1 * 0.75, "{screen:?}: {last:?}");
            assert!(last.right() < screen.0, "{screen:?}: {last:?}");
            assert!(layout.footer_bottom > last.bottom());
        }
    }
}
