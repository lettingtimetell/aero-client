//! The FPS readout (`fps`), plus the readout box `cps`, `ping` and `reach_display` share.
//! The framework's reference HUD element: copy its shape for any text readout.
//!
//! * options come from the settings in `sync`, never per frame;
//! * the string is rebuilt only when the number, the mode or the font changes;
//! * with the background on, the box is at least [`READOUT_MIN_WIDTH`] wide and the text is
//!   centred in it, so the box does not twitch as the number gains or loses a digit (a fixed
//!   56-pixel box, research `fps.md`).
//!
//! The number is `HudData::fps`, the F3 figure: frames presented in the last whole second.

use std::fmt::Write as _;

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::DrawList;

/// The narrowest a readout box gets, in GUI pixels (the static FPS/CPS box).
pub const READOUT_MIN_WIDTH: f32 = 56.0;

/// The size of a one-line readout: at least `min_width` wide while it has a box (the text is
/// centred by [`draw_readout`]), just the text without one.
pub(crate) fn readout_size(
    style: &HudTextStyle,
    font: &FontRenderer,
    text: &str,
    min_width: f32,
) -> (f32, f32) {
    let (w, h) = style.size(font, text);
    if style.background.is_some() {
        (w.max(min_width), h)
    } else {
        (w, h)
    }
}

/// Draw a readout at local `(0, 0)`, `width` wide as [`readout_size`] returned it.
pub(crate) fn draw_readout(
    style: &HudTextStyle,
    d: &mut DrawList,
    font: &mut FontRenderer,
    text: &str,
    width: f32,
) {
    if style.background.is_some() {
        style.draw_centered(d, font, text, 0.0, 0.0, width);
    } else {
        style.draw(d, font, text, 0.0, 0.0);
    }
}

/// What the `display` option shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Display {
    /// `240 FPS`.
    #[default]
    Fps,
    /// `4.17 ms`: the mean frame time over the same second.
    FrameTime,
    /// `240 FPS (4.17 ms)`.
    Both,
}

/// `"<fps> FPS"`, `"<ms> ms"` or both, in the shared readout box.
#[derive(Debug)]
pub struct FpsElement {
    style: HudTextStyle,
    display: Display,
    /// The value `text` was built from; `None` forces a rebuild.
    shown: Option<u32>,
    /// The font layout generation `size` was measured with.
    font_generation: u64,
    text: String,
    scratch: String,
    size: (f32, f32),
}

impl Default for FpsElement {
    fn default() -> Self {
        FpsElement {
            style: HudTextStyle::default(),
            display: Display::Fps,
            shown: None,
            font_generation: u64::MAX,
            text: String::new(),
            scratch: String::new(),
            size: (0.0, 0.0),
        }
    }
}

/// The readout for `fps` before brackets.
fn write_text(out: &mut String, fps: u32, display: Display) {
    match display {
        Display::Fps => {
            let _ = write!(out, "{fps} FPS");
        }
        Display::FrameTime => write_frame_time(out, fps),
        Display::Both => {
            let _ = write!(out, "{fps} FPS (");
            write_frame_time(out, fps);
            out.push(')');
        }
    }
}

/// The mean frame time of the second the FPS was counted in, `1000 / fps`, with two decimals
/// (`0.80 ms`); `-- ms` before the first whole second.
fn write_frame_time(out: &mut String, fps: u32) {
    if fps == 0 {
        out.push_str("-- ms");
    } else {
        let _ = write!(out, "{:.2} ms", 1000.0 / f64::from(fps));
    }
}

impl HudElement for FpsElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("fps", "fps")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "fps");
        self.display = match settings.choice("fps", "display") {
            1 => Display::FrameTime,
            2 => Display::Both,
            _ => Display::Fps,
        };
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let fps = ctx.data.fps;
        if self.shown != Some(fps) || font.layout_generation() != self.font_generation {
            self.shown = Some(fps);
            self.font_generation = font.layout_generation();
            self.scratch.clear();
            write_text(&mut self.scratch, fps, self.display);
            self.style.decorate(&self.scratch, &mut self.text);
            self.size = readout_size(&self.style, font, &self.text, READOUT_MIN_WIDTH);
        }
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        draw_readout(&self.style, d, font, &self.text, self.size.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (640.0, 360.0),
            pixel_scale: 3.0,
            editing: false,
        }
    }

    #[test]
    fn the_string_is_rebuilt_only_when_the_number_changes() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = FpsElement::default();
        element.sync(&settings);
        let mut data = HudData {
            fps: 240,
            ..HudData::default()
        };
        let first = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(element.text, "240 FPS");
        let ptr = element.text.as_ptr();
        let capacity = element.text.capacity();
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(
            (element.text.as_ptr(), element.text.capacity()),
            (ptr, capacity),
            "same number: no rebuild"
        );
        data.fps = 1000;
        let second = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(element.text, "1000 FPS");
        // The fixed box: a fourth digit does not move the edge.
        assert_eq!(first, (READOUT_MIN_WIDTH, crate::modhud::style::BOX_HEIGHT));
        assert_eq!(second, first);
    }

    #[test]
    fn the_display_option_picks_fps_frame_time_or_both() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let mut element = FpsElement::default();
        let data = HudData {
            fps: 1250,
            ..HudData::default()
        };
        settings.set_choice("fps", "display", 1);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "0.80 ms");
        settings.set_choice("fps", "display", 2);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "1250 FPS (0.80 ms)");
        // Before the first whole second there is no frame time to show.
        let zero = HudData::default();
        element.layout(&ctx(&zero, &settings), &font);
        assert_eq!(element.text, "0 FPS (-- ms)");
        // Brackets wrap the whole readout.
        settings.set_choice("fps", "display", 0);
        settings.set_bool("fps", "brackets", true);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "[1250 FPS]");
    }

    #[test]
    fn a_long_readout_grows_the_box_past_its_minimum() {
        let mut settings = ModSettings::new();
        settings.set_choice("fps", "display", 2);
        let font = uniform_font();
        let mut element = FpsElement::default();
        element.sync(&settings);
        let data = HudData {
            fps: 1250,
            ..HudData::default()
        };
        let (w, _) = element.layout(&ctx(&data, &settings), &font).unwrap();
        let text_width = font.string_width("1250 FPS (0.80 ms)") as f32;
        assert!(text_width + 7.0 > READOUT_MIN_WIDTH);
        assert_eq!(w, text_width + 7.0, "4 px padding left, 3 right");
    }

    #[test]
    fn style_options_reach_the_draw() {
        let mut settings = ModSettings::new();
        settings.set_bool("fps", "background", false);
        settings.set_color("fps", "text_color", 0xFF00_FF00);
        let mut font = uniform_font();
        let mut element = FpsElement::default();
        element.sync(&settings);
        let data = HudData {
            fps: 60,
            ..HudData::default()
        };
        let size = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(
            size.1,
            font.font_height() as f32,
            "no box: text height only"
        );
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings), &mut font, &mut d);
        assert!(!d.is_empty());
        // The shadow pass is drawn darker, the foreground pass in the chosen green.
        assert!(d.vertices().iter().any(|v| v.color == [0.0, 1.0, 0.0, 1.0]));
    }
}
