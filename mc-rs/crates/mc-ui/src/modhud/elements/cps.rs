//! The CPS readout (`cps`): `12 CPS`, or `12 | 4 CPS` with right clicks, in the shared
//! readout box (see [`super::fps`]).
//!
//! The counts are `HudData::cps_left`/`cps_right`. The client fills them (`mc-client/src/app/mods/m1_hud.rs`) from presses of the attack and use **bindings** that
//! reach the game (no screen open), time-stamped when the press arrived and counted over a
//! rolling one-second window. `keystrokes` shows the same numbers.

use std::fmt::Write as _;

use mc_mods::ModSettings;

use super::fps::{draw_readout, readout_size, READOUT_MIN_WIDTH};
use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::DrawList;

/// The `cps` element.
#[derive(Debug)]
pub struct CpsElement {
    style: HudTextStyle,
    show_right: bool,
    /// `(left, right)` the text was built from (`right` is `None` while it is not shown);
    /// `None` forces a rebuild.
    shown: Option<(u32, Option<u32>)>,
    font_generation: u64,
    text: String,
    scratch: String,
    size: (f32, f32),
}

impl Default for CpsElement {
    fn default() -> Self {
        CpsElement {
            style: HudTextStyle::default(),
            show_right: false,
            shown: None,
            font_generation: u64::MAX,
            text: String::new(),
            scratch: String::new(),
            size: (0.0, 0.0),
        }
    }
}

impl HudElement for CpsElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("cps", "cps")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "cps");
        self.show_right = settings.bool("cps", "show_right");
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let key = (
            ctx.data.cps_left,
            self.show_right.then_some(ctx.data.cps_right),
        );
        if self.shown != Some(key) || font.layout_generation() != self.font_generation {
            self.shown = Some(key);
            self.font_generation = font.layout_generation();
            self.scratch.clear();
            let _ = match key {
                (left, Some(right)) => write!(self.scratch, "{left} | {right} CPS"),
                (left, None) => write!(self.scratch, "{left} CPS"),
            };
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
            screen: (960.0, 540.0),
            pixel_scale: 2.0,
            editing: false,
        }
    }

    #[test]
    fn left_only_by_default_and_rebuilt_only_on_change() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = CpsElement::default();
        element.sync(&settings);
        let mut data = HudData {
            cps_left: 12,
            cps_right: 4,
            ..HudData::default()
        };
        let size = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(element.text, "12 CPS");
        assert_eq!(size, (READOUT_MIN_WIDTH, crate::modhud::style::BOX_HEIGHT));
        let ptr = element.text.as_ptr();
        // A right click changes nothing while right clicks are not shown.
        data.cps_right = 7;
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text.as_ptr(), ptr);
        assert_eq!(element.text, "12 CPS");
        data.cps_left = 13;
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "13 CPS");
    }

    #[test]
    fn right_clicks_and_brackets() {
        let mut settings = ModSettings::new();
        settings.set_bool("cps", "show_right", true);
        settings.set_bool("cps", "brackets", true);
        let mut font = uniform_font();
        let mut element = CpsElement::default();
        element.sync(&settings);
        let data = HudData {
            cps_left: 12,
            cps_right: 4,
            ..HudData::default()
        };
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "[12 | 4 CPS]");
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings), &mut font, &mut d);
        assert!(!d.is_empty());
    }
}
