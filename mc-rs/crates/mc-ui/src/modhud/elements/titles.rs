//! Titles HUD element (`titles`): the S45 title and subtitle
//! as one draggable element.
//!
//! While the mod is on, the client sets `Hud::titles_detached` so the vanilla draw is skipped, and
//! hands this element both strings plus the alpha vanilla would use this frame (`Hud::title_alpha`
//! at the frame's partial ticks) through [`HudData::titles`](crate::modhud::HudData). The timers
//! are untouched: `Hud::tick` and the packet handlers still own them.
//!
//! # Geometry
//!
//! Vanilla translates to the screen centre `C = (W / 2, H / 2)` (Java int division) and draws
//! the title at 4x with its text at `y = -10`, the subtitle at 2x at `y = 5`, both centred with
//! `-width / 2` and shadowed. Each line here keeps that arrangement around a local centre, with
//! its own size multiplier (`title_scale` and `subtitle_scale`, 100 % = vanilla), so the box
//! spans from the title's top (`-10 * title_px`) to the subtitle's shadow (`+14 * subtitle_px`):
//! 68 pixels at the defaults. The box gets one extra pixel in either direction when the screen's
//! parity needs it, so the default centred placement starts on a whole pixel and the local
//! centre lands exactly on vanilla's `C`. See `the_default_placement_draws_vanillas_exact_quads`.

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::widget::{DrawList, Transform};

/// The title and subtitle as the client sees them this frame (`HudData::titles`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HudTitles {
    /// `GuiIngame.displayedTitle`, formatting codes included.
    pub title: String,
    /// `GuiIngame.displayedSubTitle`.
    pub subtitle: String,
    /// The fade alpha vanilla would draw both with this frame (`Hud::title_alpha`), `None` while
    /// vanilla would draw nothing.
    pub alpha: Option<i32>,
}

/// Vanilla's title scale. java: `GlStateManager.scale(4.0F, 4.0F, 4.0F)`
const TITLE_PX: f32 = 4.0;
/// Vanilla's subtitle scale. java: `GlStateManager.scale(2.0F, 2.0F, 2.0F)`
const SUBTITLE_PX: f32 = 2.0;
/// The editor's sample lines.
const SAMPLE_TITLE: &str = "Title";
const SAMPLE_SUBTITLE: &str = "Subtitle";

/// The `titles` element.
#[derive(Debug)]
pub struct TitlesElement {
    hide: bool,
    shadow: bool,
    /// The title's scale (vanilla 4), from `title_scale`.
    title_px: f32,
    /// The subtitle's scale (vanilla 2), from `subtitle_scale`.
    subtitle_px: f32,
    /// The strings the widths were measured for, with the font layout generation.
    measured_title: String,
    measured_subtitle: String,
    font_generation: u64,
    title_width: i32,
    subtitle_width: i32,
    /// Vanilla's `C` in local coordinates.
    centre: (f32, f32),
    /// This frame draws the editor sample, not the live title.
    sample: bool,
}

impl Default for TitlesElement {
    fn default() -> Self {
        TitlesElement {
            hide: false,
            shadow: true,
            title_px: TITLE_PX,
            subtitle_px: SUBTITLE_PX,
            measured_title: String::new(),
            measured_subtitle: String::new(),
            font_generation: u64::MAX,
            title_width: 0,
            subtitle_width: 0,
            centre: (0.0, 0.0),
            sample: false,
        }
    }
}

impl TitlesElement {
    fn lines<'a>(&self, ctx: &'a HudCtx<'_>) -> (&'a str, &'a str) {
        if self.sample {
            (SAMPLE_TITLE, SAMPLE_SUBTITLE)
        } else {
            (
                ctx.data.titles.title.as_str(),
                ctx.data.titles.subtitle.as_str(),
            )
        }
    }
}

impl HudElement for TitlesElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("titles", "titles")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.hide = settings.bool("titles", "hide");
        self.shadow = settings.bool("titles", "text_shadow");
        self.title_px = TITLE_PX * settings.f32("titles", "title_scale") / 100.0;
        self.subtitle_px = SUBTITLE_PX * settings.f32("titles", "subtitle_scale") / 100.0;
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if self.hide {
            return None;
        }
        self.sample = ctx.data.titles.alpha.is_none();
        if self.sample && !ctx.editing {
            return None;
        }
        let (title, subtitle) = self.lines(ctx);
        if self.font_generation != font.layout_generation()
            || self.measured_title != title
            || self.measured_subtitle != subtitle
        {
            self.font_generation = font.layout_generation();
            self.measured_title.clear();
            self.measured_title.push_str(title);
            self.measured_subtitle.clear();
            self.measured_subtitle.push_str(subtitle);
            self.title_width = font.string_width(title).max(0);
            self.subtitle_width = font.string_width(subtitle).max(0);
        }
        let widest = (self.title_px * self.title_width as f32)
            .max(self.subtitle_px * self.subtitle_width as f32)
            .ceil() as i32;
        let tall = (10.0 * self.title_px + 14.0 * self.subtitle_px).ceil() as i32;
        // Match the screen's parity on both axes so the centred box starts on a whole pixel and
        // `floor(box / 2)` is vanilla's `W / 2` (resp. the title top plus 40) on screen.
        let (screen_w, screen_h) = (ctx.screen.0 as i32, ctx.screen.1 as i32);
        let width = widest + ((screen_w - widest) & 1);
        let height = tall + ((screen_h - tall) & 1);
        self.centre = ((width / 2) as f32, 10.0 * self.title_px);
        Some((width as f32, height as f32))
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        let alpha = if self.sample {
            255
        } else {
            ctx.data.titles.alpha.unwrap_or(255)
        };
        // java: `16777215 | (i2 << 24 & -16777216)`
        let color = 0x00FF_FFFF | ((alpha << 24) & 0xFF00_0000_u32 as i32);
        let (title, subtitle) = self.lines(ctx);
        let anchor = Transform::translate(self.centre.0, self.centre.1);
        let (title_width, subtitle_width, shadow) =
            (self.title_width, self.subtitle_width, self.shadow);
        d.with_transform(anchor.then_scale(self.title_px), |d| {
            font.draw_string(d, title, -(title_width / 2) as f32, -10.0, color, shadow);
        });
        d.with_transform(anchor.then_scale(self.subtitle_px), |d| {
            font.draw_string(
                d,
                subtitle,
                -(subtitle_width / 2) as f32,
                5.0,
                color,
                shadow,
            );
        });
    }

    /// Vanilla draws titles with F3 open (after the debug screen), so this element does too.
    fn draws_under_debug(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(
        data: &'a HudData,
        settings: &'a ModSettings,
        screen: (f32, f32),
        editing: bool,
    ) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen,
            pixel_scale: 2.0,
            editing,
        }
    }

    fn titles(title: &str, subtitle: &str, alpha: Option<i32>) -> HudData {
        HudData {
            titles: HudTitles {
                title: title.to_owned(),
                subtitle: subtitle.to_owned(),
                alpha,
            },
            ..HudData::default()
        }
    }

    #[test]
    fn nothing_without_a_title_and_a_sample_in_the_editor() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = TitlesElement::default();
        element.sync(&settings);
        let none = HudData::default();
        assert_eq!(
            element.layout(&ctx(&none, &settings, (640.0, 360.0), false), &font),
            None
        );
        let size = element
            .layout(&ctx(&none, &settings, (640.0, 360.0), true), &font)
            .expect("the editor gets a sample");
        assert!(element.sample);
        assert_eq!(size.1, 68.0, "40 above the centre, 28 below");
    }

    #[test]
    fn hide_draws_nothing_even_in_the_editor() {
        let mut settings = ModSettings::new();
        settings.set_bool("titles", "hide", true);
        let font = uniform_font();
        let mut element = TitlesElement::default();
        element.sync(&settings);
        let data = titles("Victory!", "You won", Some(255));
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), true), &font),
            None
        );
    }

    #[test]
    fn the_box_follows_the_line_sizes_and_the_screen_parity() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let data = titles("ab", "abcdefghij", Some(255));
        let mut element = TitlesElement::default();
        element.sync(&settings);
        let title_w = font.string_width("ab") as f32;
        let subtitle_w = font.string_width("abcdefghij") as f32;
        let wide = (4.0 * title_w).max(2.0 * subtitle_w);
        let size = element
            .layout(&ctx(&data, &settings, (640.0, 360.0), false), &font)
            .unwrap();
        assert_eq!(size, (wide, 68.0));
        let odd = element
            .layout(&ctx(&data, &settings, (641.0, 361.0), false), &font)
            .unwrap();
        assert_eq!(odd, (wide + 1.0, 69.0), "one parity pixel on each odd axis");

        settings.set_f32("titles", "title_scale", 50.0);
        settings.set_f32("titles", "subtitle_scale", 200.0);
        element.sync(&settings);
        let size = element
            .layout(&ctx(&data, &settings, (640.0, 360.0), false), &font)
            .unwrap();
        // Title at 2x (20 above the centre), subtitle at 4x (56 below).
        assert_eq!(size.1, 20.0 + 56.0);
        assert_eq!(size.0, (2.0 * title_w).max(4.0 * subtitle_w));
    }

    #[test]
    fn the_strings_are_measured_only_when_they_change() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = TitlesElement::default();
        element.sync(&settings);
        let data = titles("\u{a7}6Victory", "\u{a7}7You won", Some(200));
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let ptrs = (
            element.measured_title.as_ptr(),
            element.measured_subtitle.as_ptr(),
        );
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        assert_eq!(
            (
                element.measured_title.as_ptr(),
                element.measured_subtitle.as_ptr()
            ),
            ptrs
        );
    }

    /// The mod on at its defaults must reproduce `Hud::draw_with_spectator_and_crosshair`'s
    /// title block exactly, on every screen parity.
    #[test]
    fn the_default_placement_draws_vanillas_exact_quads() {
        use crate::widget::draw_centered_string;
        let mut settings = ModSettings::new();
        settings.set_enabled("titles", true);
        for (w, h) in [(640, 360), (641, 361), (427, 240), (854, 481)] {
            for (title, subtitle) in [
                ("Victory!", "\u{a7}7You were the last one standing"),
                ("x", ""),
                ("", "Only a subtitle"),
            ] {
                let alpha = 0x80;
                let data = titles(title, subtitle, Some(alpha));
                let mut hud = crate::modhud::ModHud::new();
                hud.sync(&settings);
                let mut font = uniform_font();
                let mut ours = DrawList::new();
                hud.draw(
                    &ctx(&data, &settings, (w as f32, h as f32), false),
                    &mut font,
                    &mut ours,
                );

                let mut font = uniform_font();
                let mut vanilla = DrawList::new();
                let color = 0x00FF_FFFF | (alpha << 24);
                let anchor = Transform::translate((w / 2) as f32, (h / 2) as f32);
                vanilla.push_transform(anchor.then_scale(4.0));
                draw_centered_string(&mut vanilla, &mut font, title, 0, -10, color);
                vanilla.pop_transform();
                vanilla.push_transform(anchor.then_scale(2.0));
                draw_centered_string(&mut vanilla, &mut font, subtitle, 0, 5, color);
                vanilla.pop_transform();
                crate::modhud::elements::action_bar::tests::assert_same_quads(
                    ours.vertices(),
                    vanilla.vertices(),
                    &format!("{title:?}/{subtitle:?} on {w}x{h} must land on vanilla's pixels"),
                );
            }
        }
    }

    #[test]
    fn options_reach_the_draw() {
        let mut settings = ModSettings::new();
        let mut font = uniform_font();
        let data = titles("Victory", "GG", Some(255));
        let mut element = TitlesElement::default();
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut shadowed = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut shadowed,
        );
        settings.set_bool("titles", "text_shadow", false);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut flat = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut flat,
        );
        assert!(flat.vertices().len() < shadowed.vertices().len());
        // The fade alpha reaches every vertex.
        let data = titles("Victory", "GG", Some(0x40));
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut faded = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut faded,
        );
        assert!(faded
            .vertices()
            .iter()
            .all(|v| (v.color[3] - 0x40 as f32 / 255.0).abs() < 1e-3));
    }
}
