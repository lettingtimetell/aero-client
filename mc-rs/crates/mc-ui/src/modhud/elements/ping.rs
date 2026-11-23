//! The ping readout (`ping`): `45 ms` in the shared readout box (see [`super::fps`]).
//!
//! The number is your own tab-list latency (`HudData::ping_ms`, from `S38PacketPlayerListItem`):
//! the server's smoothed keep-alive round trip, the figure the tab list's bars show. It changes
//! rarely (vanilla servers broadcast it every 30 s) and is shown as sent, never smoothed or
//! measured another way: a second connection is something a vanilla client never opens.
//!
//! "Colour by ping" colours the number only, with the font's own colour codes (`§a` green,
//! `§e` yellow, `§6` gold, `§c` red), then `§r` restores the text colour for ` ms`.

use std::fmt::Write as _;

use mc_mods::ModSettings;

use super::fps::{draw_readout, readout_size, READOUT_MIN_WIDTH};
use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::DrawList;

/// The colour code for a latency: green below 80 ms, yellow below 150, gold below 250, red.
#[must_use]
fn ping_color_code(ms: i32) -> char {
    match ms {
        i32::MIN..=79 => 'a',
        80..=149 => 'e',
        150..=249 => '6',
        _ => 'c',
    }
}

/// The `ping` element.
#[derive(Debug)]
pub struct PingElement {
    style: HudTextStyle,
    color_by_ping: bool,
    /// The latency the text was built from; `None` forces a rebuild.
    shown: Option<Option<i32>>,
    font_generation: u64,
    text: String,
    scratch: String,
    size: (f32, f32),
}

impl Default for PingElement {
    fn default() -> Self {
        PingElement {
            style: HudTextStyle::default(),
            color_by_ping: true,
            shown: None,
            font_generation: u64::MAX,
            text: String::new(),
            scratch: String::new(),
            size: (0.0, 0.0),
        }
    }
}

impl HudElement for PingElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("ping", "ping")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "ping");
        self.color_by_ping = settings.bool("ping", "color_by_ping");
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let ping = ctx.data.ping_ms;
        if self.shown != Some(ping) || font.layout_generation() != self.font_generation {
            self.shown = Some(ping);
            self.font_generation = font.layout_generation();
            self.scratch.clear();
            let _ = match ping {
                // Before the server's first latency arrives.
                None => write!(self.scratch, "-- ms"),
                Some(ms) if self.color_by_ping => {
                    write!(self.scratch, "\u{a7}{}{ms}\u{a7}r ms", ping_color_code(ms))
                }
                Some(ms) => write!(self.scratch, "{ms} ms"),
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
    fn thresholds_follow_the_spec() {
        assert_eq!(ping_color_code(0), 'a');
        assert_eq!(ping_color_code(79), 'a');
        assert_eq!(ping_color_code(80), 'e');
        assert_eq!(ping_color_code(149), 'e');
        assert_eq!(ping_color_code(150), '6');
        assert_eq!(ping_color_code(249), '6');
        assert_eq!(ping_color_code(250), 'c');
        assert_eq!(
            ping_color_code(-1),
            'a',
            "a server's placeholder -1 is not a red alert"
        );
    }

    #[test]
    fn the_number_is_coloured_and_the_suffix_keeps_the_text_colour() {
        let settings = ModSettings::new();
        let mut font = uniform_font();
        let mut element = PingElement::default();
        element.sync(&settings);
        let data = HudData {
            ping_ms: Some(45),
            ..HudData::default()
        };
        let size = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(element.text, "\u{a7}a45\u{a7}r ms");
        // Colour codes take no width.
        assert_eq!(font.string_width(&element.text), font.string_width("45 ms"));
        assert_eq!(size.0, READOUT_MIN_WIDTH);
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings), &mut font, &mut d);
        let green = [85.0 / 255.0, 1.0, 85.0 / 255.0, 1.0];
        let colors: Vec<[f32; 4]> = d.vertices().iter().map(|v| v.color).collect();
        assert!(colors.contains(&green), "the number is drawn in §a green");
        assert!(
            colors.contains(&[1.0, 1.0, 1.0, 1.0]),
            "` ms` is drawn white"
        );
    }

    #[test]
    fn no_latency_yet_and_plain_mode() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let mut element = PingElement::default();
        element.sync(&settings);
        let mut data = HudData::default();
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "-- ms");
        settings.set_bool("ping", "color_by_ping", false);
        element.sync(&settings);
        data.ping_ms = Some(312);
        element.layout(&ctx(&data, &settings), &font);
        assert_eq!(element.text, "312 ms");
    }
}
