//! The coordinates readout (`coordinates`), vertical by default:
//!
//! ```text
//! X: 128
//! Y: 64
//! Z: -302
//! NE (+X -Z)
//! ```
//!
//! * With 0 decimals the numbers are **block** coordinates, `floor` of the feet position, which
//!   is F3's `Block:` line (`-0.3` is block `-1`). With 1 or 2 decimals they are the exact feet
//!   position, F3's `XYZ:`.
//! * The direction is 8-way (what players use in call-outs), with the axes you move along.
//! * Hidden while the server has set reduced debug info, as F3 hides them then.
//! * The lines are rebuilt only when a shown number, the direction or an option changes, so a
//!   player standing still costs no formatting.

use std::fmt::Write as _;

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::style::{BOX_PAD_X, BOX_TEXT_Y};
use crate::modhud::{HudCtx, HudElement, HudTextStyle};
use crate::widget::{DrawList, UiTexture};

/// Pixels from one line to the next.
const LINE_PITCH: f32 = 10.0;

/// The 8-way facing names and the axes moving forward follows, indexed by
/// [`direction_index`]. Vanilla yaw: 0 is south (+Z), 90 west (-X), 180 north, 270 east.
const DIRECTIONS: [(&str, &str); 8] = [
    ("S", "+Z"),
    ("SW", "-X +Z"),
    ("W", "-X"),
    ("NW", "-X -Z"),
    ("N", "-Z"),
    ("NE", "+X -Z"),
    ("E", "+X"),
    ("SE", "+X +Z"),
];

/// Which of [`DIRECTIONS`] a yaw faces: eight 45-degree sectors centred on the compass points.
#[must_use]
pub fn direction_index(yaw: f32) -> usize {
    let w = f64::from(yaw).rem_euclid(360.0);
    ((w + 22.5) / 45.0).floor() as usize % 8
}

/// `scaled / 10^decimals` with exactly `decimals` places, from the already-rounded integer, so
/// there is never a `-0.0`.
fn write_fixed(out: &mut String, scaled: i64, decimals: u32) {
    if decimals == 0 {
        let _ = write!(out, "{scaled}");
        return;
    }
    let unit = 10_u64.pow(decimals);
    let sign = if scaled < 0 { "-" } else { "" };
    let magnitude = scaled.unsigned_abs();
    let _ = write!(
        out,
        "{sign}{}.{:0width$}",
        magnitude / unit,
        magnitude % unit,
        width = decimals as usize
    );
}

/// What the lines were built from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shown {
    /// The three numbers as shown: block coordinates, or the position times `10^decimals`
    /// rounded. `None` while hidden by reduced debug info (the editor shows `?`).
    values: Option<[i64; 3]>,
    direction: usize,
}

/// The `coordinates` element.
#[derive(Debug)]
pub struct CoordinatesElement {
    style: HudTextStyle,
    horizontal: bool,
    decimals: u32,
    show_direction: bool,
    shown: Option<Shown>,
    font_generation: u64,
    /// `X: ..`, `Y: ..`, `Z: ..` and the direction line.
    lines: [String; 4],
    widths: [f32; 4],
    /// The horizontal layout's single line.
    joined: String,
    size: (f32, f32),
}

impl Default for CoordinatesElement {
    fn default() -> Self {
        CoordinatesElement {
            style: HudTextStyle::default(),
            horizontal: false,
            decimals: 0,
            show_direction: true,
            shown: None,
            font_generation: u64::MAX,
            lines: Default::default(),
            widths: [0.0; 4],
            joined: String::new(),
            size: (0.0, 0.0),
        }
    }
}

impl CoordinatesElement {
    fn line_count(&self) -> usize {
        if self.show_direction {
            4
        } else {
            3
        }
    }

    fn rebuild(&mut self, shown: Shown, font: &FontRenderer) {
        for (axis, line) in ["X: ", "Y: ", "Z: "].iter().zip(&mut self.lines) {
            line.clear();
            line.push_str(axis);
        }
        match shown.values {
            Some(values) => {
                for (value, line) in values.iter().zip(&mut self.lines) {
                    write_fixed(line, *value, self.decimals);
                }
            }
            None => {
                for line in &mut self.lines[..3] {
                    line.push('?');
                }
            }
        }
        let (name, axes) = DIRECTIONS[shown.direction % 8];
        self.lines[3].clear();
        let _ = write!(self.lines[3], "{name} ({axes})");
        let count = self.line_count();
        for (width, line) in self.widths.iter_mut().zip(&self.lines) {
            *width = font.string_width(line).max(0) as f32;
        }
        if self.horizontal {
            self.joined.clear();
            for (i, line) in self.lines[..count].iter().enumerate() {
                if i > 0 {
                    self.joined.push_str("  ");
                }
                self.joined.push_str(line);
            }
            self.size = self.style.size(font, &self.joined);
        } else {
            let widest = self.widths[..count].iter().copied().fold(0.0, f32::max);
            let lines = count as f32;
            self.size = if self.style.background.is_some() {
                (
                    widest + BOX_PAD_X * 2.0 - 1.0,
                    8.0 + LINE_PITCH * lines - 2.0,
                )
            } else {
                (
                    widest,
                    LINE_PITCH * (lines - 1.0) + font.font_height() as f32,
                )
            };
        }
    }
}

impl HudElement for CoordinatesElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("coordinates", "coordinates")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.style = HudTextStyle::read(settings, "coordinates");
        self.horizontal = settings.choice("coordinates", "layout") == 1;
        self.decimals = settings.f32("coordinates", "decimals").clamp(0.0, 2.0) as u32;
        self.show_direction = settings.bool("coordinates", "show_direction");
        self.shown = None;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        let hidden = ctx.data.reduced_debug_info;
        if hidden && !ctx.editing {
            return None;
        }
        let values = (!hidden).then(|| {
            if self.decimals == 0 {
                ctx.data.block_position().map(i64::from)
            } else {
                let unit = 10_f64.powi(self.decimals as i32);
                ctx.data.position.map(|v| (v * unit).round() as i64)
            }
        });
        let shown = Shown {
            values,
            direction: if self.show_direction {
                direction_index(ctx.data.yaw)
            } else {
                0
            },
        };
        if self.shown != Some(shown) || font.layout_generation() != self.font_generation {
            self.shown = Some(shown);
            self.font_generation = font.layout_generation();
            self.rebuild(shown, font);
        }
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        mc_core::profile_scope!("hud.mod.coordinates");
        if self.horizontal {
            self.style.draw(d, font, &self.joined, 0.0, 0.0);
            return;
        }
        let (x, y) = match self.style.background {
            Some(argb) => {
                d.quad_rect(
                    &UiTexture::Solid,
                    (0.0, 0.0, self.size.0, self.size.1),
                    (0.0, 0.0, 1.0, 1.0),
                    argb,
                );
                (BOX_PAD_X, BOX_TEXT_Y)
            }
            None => (0.0, 0.0),
        };
        for (i, line) in self.lines[..self.line_count()].iter().enumerate() {
            font.draw_string(
                d,
                line,
                x,
                y + LINE_PITCH * i as f32,
                self.style.color as i32,
                self.style.shadow,
            );
        }
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

    fn at(position: [f64; 3], yaw: f32) -> HudData {
        HudData {
            position,
            yaw,
            ..HudData::default()
        }
    }

    #[test]
    fn eight_way_directions_follow_vanilla_yaw() {
        let name = |yaw: f32| DIRECTIONS[direction_index(yaw)].0;
        assert_eq!(name(0.0), "S");
        assert_eq!(name(45.0), "SW");
        assert_eq!(name(90.0), "W");
        assert_eq!(name(135.0), "NW");
        assert_eq!(name(180.0), "N");
        assert_eq!(name(225.0), "NE");
        assert_eq!(name(270.0), "E");
        assert_eq!(name(315.0), "SE");
        assert_eq!(name(-90.0), "E", "negative yaw wraps");
        assert_eq!(name(22.4), "S");
        assert_eq!(name(22.6), "SW");
        assert_eq!(name(350.0 + 720.0), "S");
    }

    #[test]
    fn block_coordinates_floor_like_f3_and_are_cached() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = CoordinatesElement::default();
        element.sync(&settings);
        let data = at([128.7, 64.0, -301.2], 225.0);
        let size = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(
            element.lines,
            ["X: 128", "Y: 64", "Z: -302", "NE (+X -Z)"].map(String::from)
        );
        assert_eq!(size.1, 46.0, "four lines in the box");
        let widest = font.string_width("NE (+X -Z)") as f32;
        assert_eq!(size.0, widest + 7.0);
        // Moving inside the same block re-formats nothing.
        let ptr = element.lines[0].as_ptr();
        let moved = at([128.1, 64.5, -301.9], 226.0);
        element.layout(&ctx(&moved, &settings), &font);
        assert_eq!(element.lines[0].as_ptr(), ptr);
        assert_eq!(
            element.shown,
            Some(Shown {
                values: Some([128, 64, -302]),
                direction: 5,
            })
        );
        // -0.3 is block -1 (floor), not a truncated 0.
        let negative = at([-0.3, 0.0, 0.2], 0.0);
        element.layout(&ctx(&negative, &settings), &font);
        assert_eq!(element.lines[0], "X: -1");
    }

    #[test]
    fn decimals_horizontal_and_no_minus_zero() {
        let mut settings = ModSettings::new();
        settings.set_f32("coordinates", "decimals", 1.0);
        settings.set_choice("coordinates", "layout", 1);
        settings.set_bool("coordinates", "show_direction", false);
        let font = uniform_font();
        let mut element = CoordinatesElement::default();
        element.sync(&settings);
        let data = at([-0.04, 64.25, -12.36], 0.0);
        let size = element.layout(&ctx(&data, &settings), &font).unwrap();
        assert_eq!(element.joined, "X: 0.0  Y: 64.3  Z: -12.4");
        assert_eq!(size.1, crate::modhud::style::BOX_HEIGHT);
        let mut fixed = String::new();
        write_fixed(&mut fixed, -5, 2);
        assert_eq!(fixed, "-0.05");
    }

    #[test]
    fn reduced_debug_info_hides_it_except_as_a_placeholder_in_the_editor() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = CoordinatesElement::default();
        element.sync(&settings);
        let data = HudData {
            reduced_debug_info: true,
            ..at([1.0, 2.0, 3.0], 0.0)
        };
        assert!(element.layout(&ctx(&data, &settings), &font).is_none());
        let editing = HudCtx {
            editing: true,
            ..ctx(&data, &settings)
        };
        assert!(element.layout(&editing, &font).is_some());
        assert_eq!(element.lines[0], "X: ?");
    }
}
