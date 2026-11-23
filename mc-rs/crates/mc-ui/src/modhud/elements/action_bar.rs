//! Action Bar HUD element (`action_bar`): vanilla's overlay
//! message above the hotbar as a draggable element.
//!
//! While the mod is on, the client sets `Hud::action_bar_detached` so the vanilla draw is skipped,
//! and hands this element the message plus the colour vanilla would use this frame
//! (`Hud::record_color` at the frame's partial ticks: the 60-tick fade and the jukebox rainbow)
//! through [`HudData::action_bar`](crate::modhud::HudData). The message's lifetime is untouched:
//! `Hud::tick` still counts it down exactly as before.
//!
//! At the default placement (bottom centre, 63 pixels up, scale 1) every glyph lands on vanilla's
//! pixel: the box is the text's width, plus one pixel when the screen width's parity needs it, so
//! the centred box starts on a whole pixel, and the text sits inside it at vanilla's integer
//! `width / 2 - text_width / 2`. See `the_default_placement_draws_vanillas_exact_quads`.
//!
//! In the HUD editor with no message showing, a white sample ("Action Bar") is drawn instead.

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::widget::DrawList;

/// The overlay message as the client sees it this frame (`HudData::action_bar`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HudActionBar {
    /// `GuiIngame.recordPlaying`, formatting codes included.
    pub text: String,
    /// The colour and fade alpha vanilla would draw it with this frame (`Hud::record_color`),
    /// `None` while vanilla would draw nothing.
    pub color: Option<i32>,
}

/// Eight glyph rows plus the one-pixel shadow row.
const HEIGHT: f32 = 9.0;
/// The editor's sample text.
const SAMPLE: &str = "Action Bar";
/// The sample's colour: opaque white, vanilla's non-jukebox colour.
const SAMPLE_COLOR: i32 = 0xFFFF_FFFF_u32 as i32;

/// The `action_bar` element.
#[derive(Debug)]
pub struct ActionBarElement {
    hide: bool,
    shadow: bool,
    /// The text `text_width` was measured for, with the font layout generation.
    measured: String,
    font_generation: u64,
    text_width: i32,
    /// The text's x inside the box (0 or 1: see the module docs).
    text_x: i32,
    /// This frame draws the editor sample, not the live message.
    sample: bool,
}

impl Default for ActionBarElement {
    fn default() -> Self {
        ActionBarElement {
            hide: false,
            shadow: false,
            measured: String::new(),
            font_generation: u64::MAX,
            text_width: 0,
            text_x: 0,
            sample: false,
        }
    }
}

impl HudElement for ActionBarElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("action_bar", "action_bar")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.hide = settings.bool("action_bar", "hide");
        self.shadow = settings.bool("action_bar", "text_shadow");
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if self.hide {
            return None;
        }
        let bar = &ctx.data.action_bar;
        let sample = bar.color.is_none() || bar.text.is_empty();
        if sample && !ctx.editing {
            return None;
        }
        let text = if sample { SAMPLE } else { bar.text.as_str() };
        if self.font_generation != font.layout_generation() || self.measured != text {
            self.font_generation = font.layout_generation();
            self.measured.clear();
            self.measured.push_str(text);
            self.text_width = font.string_width(text).max(0);
        }
        self.sample = sample;
        // A box whose width has the screen width's parity is centred on a whole pixel, at
        // `(W - w) / 2`. Vanilla's text starts at `W / 2 - tw / 2` (Java int division); the
        // difference is 1 only for an even screen and an odd text width.
        let screen_width = ctx.screen.0 as i32;
        let mut width = self.text_width;
        if (screen_width - width) & 1 != 0 {
            width += 1;
        }
        let vanilla_x = screen_width / 2 - self.text_width / 2;
        let box_x = (screen_width - width) / 2;
        self.text_x = (vanilla_x - box_x).clamp(0, width - self.text_width);
        Some((width as f32, HEIGHT))
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        let (text, color) = if self.sample {
            (SAMPLE, SAMPLE_COLOR)
        } else {
            let bar = &ctx.data.action_bar;
            (bar.text.as_str(), bar.color.unwrap_or(SAMPLE_COLOR))
        };
        font.draw_string(d, text, self.text_x as f32, 0.0, color, self.shadow);
    }

    /// Vanilla draws the overlay message with F3 open (`GuiIngame` draws it after the debug
    /// screen), so this element does too.
    fn draws_under_debug(&self) -> bool {
        true
    }
}

#[cfg(test)]
pub(crate) mod tests {
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

    fn message(text: &str, color: Option<i32>) -> HudData {
        HudData {
            action_bar: HudActionBar {
                text: text.to_owned(),
                color,
            },
            ..HudData::default()
        }
    }

    #[test]
    fn nothing_without_a_message_and_a_sample_in_the_editor() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ActionBarElement::default();
        element.sync(&settings);
        let none = HudData::default();
        assert_eq!(
            element.layout(&ctx(&none, &settings, (640.0, 360.0), false), &font),
            None
        );
        // A faded-out message (colour `None`) is not drawn either.
        let faded = message("Hello", None);
        assert_eq!(
            element.layout(&ctx(&faded, &settings, (640.0, 360.0), false), &font),
            None
        );
        let size = element.layout(&ctx(&none, &settings, (640.0, 360.0), true), &font);
        assert!(element.sample);
        let width = font.string_width(SAMPLE);
        let parity = (640 - width) & 1;
        assert_eq!(size, Some(((width + parity) as f32, HEIGHT)));
    }

    /// Same quads as vanilla's: identical UVs and colours, positions equal up to float
    /// association (`(box + local) + advance` against `absolute + advance`: a few ulps, far
    /// below the rasteriser's pixel-centre test, so the pixels are identical).
    pub(crate) fn assert_same_quads(
        ours: &[crate::widget::UiVertex],
        vanilla: &[crate::widget::UiVertex],
        what: &str,
    ) {
        assert_eq!(ours.len(), vanilla.len(), "{what}: vertex count");
        for (i, (a, b)) in ours.iter().zip(vanilla).enumerate() {
            assert_eq!((a.uv, a.color), (b.uv, b.color), "{what}: vertex {i}");
            let off = (a.pos[0] - b.pos[0]).abs().max((a.pos[1] - b.pos[1]).abs());
            assert!(
                off < 1e-3,
                "{what}: vertex {i} at {:?}, vanilla {:?}",
                a.pos,
                b.pos
            );
        }
    }

    #[test]
    fn hide_draws_nothing_even_in_the_editor() {
        let mut settings = ModSettings::new();
        settings.set_bool("action_bar", "hide", true);
        let font = uniform_font();
        let mut element = ActionBarElement::default();
        element.sync(&settings);
        let data = message("Hello", Some(-1));
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), true), &font),
            None
        );
    }

    #[test]
    fn the_text_is_measured_only_when_it_changes() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = ActionBarElement::default();
        element.sync(&settings);
        let data = message("\u{a7}aKills: 3", Some(-1));
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let (ptr, capacity) = (element.measured.as_ptr(), element.measured.capacity());
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        assert_eq!(
            (element.measured.as_ptr(), element.measured.capacity()),
            (ptr, capacity)
        );
        assert_eq!(element.text_width, font.string_width("\u{a7}aKills: 3"));
    }

    /// The mod on at its defaults must look exactly like the vanilla draw it replaces:
    /// `Hud::draw_with_spectator_and_crosshair`'s `font.draw_string(message, W / 2 - tw / 2,
    /// H - 72, colour, false)`, for every parity of screen and text width.
    #[test]
    fn the_default_placement_draws_vanillas_exact_quads() {
        let mut settings = ModSettings::new();
        settings.set_enabled("action_bar", true);
        for (w, h) in [(640, 360), (641, 361), (427, 240), (854, 480)] {
            for text in ["Hello", "Hello!", "\u{a7}cKilled by Steve", "x"] {
                let color = 0xC0FF_FFFF_u32 as i32;
                let data = message(text, Some(color));
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
                let x = w / 2 - font.string_width(text) / 2;
                font.draw_string(&mut vanilla, text, x as f32, (h - 72) as f32, color, false);
                assert_same_quads(
                    ours.vertices(),
                    vanilla.vertices(),
                    &format!("{text:?} on {w}x{h} must land on vanilla's pixels"),
                );
            }
        }
    }

    #[test]
    fn the_shadow_option_reaches_the_draw() {
        let mut settings = ModSettings::new();
        let mut font = uniform_font();
        let data = message("Hello", Some(-1));
        let mut element = ActionBarElement::default();
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut flat = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut flat,
        );
        settings.set_bool("action_bar", "text_shadow", true);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut shadowed = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut shadowed,
        );
        assert!(
            shadowed.vertices().len() > flat.vertices().len(),
            "the shadow pass adds glyph quads"
        );
    }
}
