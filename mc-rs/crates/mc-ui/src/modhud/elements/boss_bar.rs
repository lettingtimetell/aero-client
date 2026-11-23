//! Boss Bar HUD element (`boss_bar`): the vanilla boss bar as a draggable element.
//!
//! While the mod is on, the client detaches the vanilla bar (`Hud::set_boss_bar_detached`),
//! consumes its 100-frame lifetime exactly as vanilla's draw would (`Hud::consume_boss_frame`)
//! and hands the result to this element through [`HudData::boss`](crate::modhud::HudData).
//! The paint is the vanilla HUD's own [`paint_boss_bar`], so at the default placement (top
//! centre, 2 pixels down, scale 1) every pixel is vanilla's: the box is the bar's 182 pixels
//! (or the name's width when that is wider), plus one pixel when the screen width's parity
//! needs it, so a centred box lands on vanilla's integer `width / 2 - 91`.
//!
//! In the HUD editor with no boss active, a sample ("Boss Bar", full health) is drawn instead;
//! the real boss state is never touched.

use mc_mods::ModSettings;

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudElement};
use crate::screens::hud::{paint_boss_bar, BossBarStyle};
use crate::widget::DrawList;

/// The boss bar as the client sees it this frame (`HudData::boss`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HudBoss {
    /// Vanilla would draw the bar this frame: a boss is set and the frame the client consumed
    /// was still inside its 100-frame lifetime.
    pub shown: bool,
    /// `BossStatus.bossName`, formatting codes included.
    pub name: String,
    /// `BossStatus.healthScale` (current / max health, unclamped like vanilla's).
    pub health_scale: f32,
}

/// The bar's width. java: `int j = 182` in `renderBossHealth`.
const BAR_WIDTH: i32 = 182;
/// Name row (y 0..9) plus the bar (y 10..15).
const HEIGHT: f32 = 15.0;
/// The editor's sample name.
const SAMPLE_NAME: &str = "Boss Bar";

/// The `boss_bar` element.
#[derive(Debug)]
pub struct BossBarElement {
    hide: bool,
    style: BossBarStyle,
    /// The name `name_width` was measured for, with the font layout generation.
    measured: String,
    font_generation: u64,
    name_width: i32,
    /// This frame's box width.
    width: i32,
    /// This frame draws the editor sample, not the live boss.
    sample: bool,
}

impl Default for BossBarElement {
    fn default() -> Self {
        BossBarElement {
            hide: false,
            style: BossBarStyle::VANILLA,
            measured: String::new(),
            font_generation: u64::MAX,
            name_width: 0,
            width: BAR_WIDTH,
            sample: false,
        }
    }
}

impl HudElement for BossBarElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("boss_bar", "boss_bar")
    }

    fn sync(&mut self, settings: &ModSettings) {
        self.hide = settings.bool("boss_bar", "hide");
        self.style = BossBarStyle {
            bar: settings.bool("boss_bar", "show_bar"),
            text: settings.bool("boss_bar", "show_text"),
            shadow: settings.bool("boss_bar", "text_shadow"),
        };
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if self.hide || !(self.style.bar || self.style.text) {
            return None;
        }
        let boss = &ctx.data.boss;
        let sample = !boss.shown;
        if sample && !ctx.editing {
            return None;
        }
        let name = if sample {
            SAMPLE_NAME
        } else {
            boss.name.as_str()
        };
        if self.font_generation != font.layout_generation() || self.measured != name {
            self.font_generation = font.layout_generation();
            self.measured.clear();
            self.measured.push_str(name);
            self.name_width = font.string_width(name);
        }
        self.sample = sample;
        // A long name widens the box (centred on the bar), so the editor's box covers it.
        let mut width = BAR_WIDTH;
        if self.style.text {
            width = width.max(self.name_width);
        }
        // Vanilla centres with integer division (`W / 2 - 91`). With `W - width` even, a box
        // centred by the anchor maths lands on a whole pixel and its local centre `width / 2`
        // is exactly vanilla's `W / 2`.
        if (ctx.screen.0 as i32 - width) & 1 != 0 {
            width += 1;
        }
        self.width = width;
        Some((width as f32, HEIGHT))
    }

    fn draw(&mut self, ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        let (name, health_scale) = if self.sample {
            (SAMPLE_NAME, 1.0)
        } else {
            (ctx.data.boss.name.as_str(), ctx.data.boss.health_scale)
        };
        paint_boss_bar(d, font, self.width / 2, 0, name, health_scale, self.style);
    }

    /// Vanilla draws the boss bar with F3 open, so this element does too.
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

    fn boss(name: &str, health_scale: f32) -> HudData {
        HudData {
            boss: HudBoss {
                shown: true,
                name: name.to_owned(),
                health_scale,
            },
            ..HudData::default()
        }
    }

    #[test]
    fn nothing_without_a_boss_and_a_sample_in_the_editor() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = BossBarElement::default();
        element.sync(&settings);
        let data = HudData::default();
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font),
            None
        );
        let size = element.layout(&ctx(&data, &settings, (640.0, 360.0), true), &font);
        assert_eq!(size, Some((182.0, 15.0)));
        assert!(element.sample);
    }

    #[test]
    fn hide_and_both_halves_off_draw_nothing_even_in_the_editor() {
        let mut settings = ModSettings::new();
        let font = uniform_font();
        let data = boss("Wither", 0.5);
        let mut element = BossBarElement::default();
        settings.set_bool("boss_bar", "hide", true);
        element.sync(&settings);
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), true), &font),
            None
        );
        settings.set_bool("boss_bar", "hide", false);
        settings.set_bool("boss_bar", "show_bar", false);
        settings.set_bool("boss_bar", "show_text", false);
        element.sync(&settings);
        assert_eq!(
            element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font),
            None
        );
    }

    #[test]
    fn the_box_keeps_vanillas_integer_centre_on_odd_widths_and_grows_for_long_names() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = BossBarElement::default();
        element.sync(&settings);
        let short = boss("Dragon", 1.0);
        assert_eq!(
            element.layout(&ctx(&short, &settings, (640.0, 360.0), false), &font),
            Some((182.0, 15.0))
        );
        assert_eq!(
            element.layout(&ctx(&short, &settings, (641.0, 360.0), false), &font),
            Some((183.0, 15.0)),
            "odd screen: one parity pixel"
        );
        // 40 glyphs of the 5-pixel test font: 200 pixels, wider than the bar.
        let long = boss(&"W".repeat(40), 1.0);
        assert_eq!(
            element.layout(&ctx(&long, &settings, (640.0, 360.0), false), &font),
            Some((200.0, 15.0))
        );
        // Name hidden: the bar alone decides.
        let mut settings = ModSettings::new();
        settings.set_bool("boss_bar", "show_text", false);
        element.sync(&settings);
        assert_eq!(
            element.layout(&ctx(&long, &settings, (640.0, 360.0), false), &font),
            Some((182.0, 15.0))
        );
    }

    #[test]
    fn the_name_is_measured_only_when_it_changes() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = BossBarElement::default();
        element.sync(&settings);
        let data = boss("Wither", 0.25);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let (ptr, capacity) = (element.measured.as_ptr(), element.measured.capacity());
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        assert_eq!(
            (element.measured.as_ptr(), element.measured.capacity()),
            (ptr, capacity)
        );
        assert_eq!(element.name_width, font.string_width("Wither"));
    }

    /// The mod-off path: `Hud::draw_boss_health` (now through the shared `paint_boss_bar`) emits
    /// exactly the vertices the pre-wave body did, kept here verbatim as the reference.
    #[test]
    fn the_vanilla_boss_draw_is_byte_identical_to_the_pre_wave_body() {
        use crate::screen_stack::ScreenCtx;
        use crate::screens::hud::Hud;
        use crate::widget::{ScaledResolution, UiTexture};

        fn pre_wave(
            d: &mut DrawList,
            font: &mut FontRenderer,
            ctx: &ScreenCtx,
            name: &str,
            scale: f32,
        ) {
            let tex = UiTexture::icons();
            let width = 182;
            let x = ctx.width / 2 - width / 2;
            let y = 12;
            let filled = (scale * 183.0) as i32;
            d.blit(tex, x, y, 0, 74, width, 5);
            d.blit(tex, x, y, 0, 74, width, 5);
            if filled > 0 {
                d.blit(tex, x, y, 0, 79, filled, 5);
            }
            font.draw_string_with_shadow(
                d,
                name,
                (ctx.width / 2 - font.string_width(name) / 2) as f32,
                (y - 10) as f32,
                0x00FF_FFFF,
            );
        }

        for (w, h) in [(854, 480), (855, 481), (1920, 1080)] {
            let ctx = ScreenCtx::new(ScaledResolution::new(w, h, 0, false), 0.0);
            for (name, scale) in [("Wither", 0.5), ("\u{a7}cDragon", 1.0), ("x", 0.0)] {
                let mut hud = Hud::new();
                hud.set_boss_status(scale * 10.0, 10.0, name, false);
                let mut now = DrawList::new();
                hud.draw_boss_health(&mut now, &mut uniform_font(), &ctx);
                let mut reference = DrawList::new();
                pre_wave(
                    &mut reference,
                    &mut uniform_font(),
                    &ctx,
                    name,
                    hud.boss_health_scale(),
                );
                assert_eq!(now, reference, "{name} on {w}x{h}");
            }
        }
    }

    /// F3 hides the mod HUD, but not the vanilla pieces vanilla draws under F3.
    #[test]
    fn only_the_vanilla_stand_ins_draw_while_f3_is_open() {
        let mut settings = ModSettings::new();
        settings.set_enabled("fps", true);
        settings.set_enabled("boss_bar", true);
        settings.set_enabled("scoreboard", true);
        let mut hud = crate::modhud::ModHud::new();
        hud.sync(&settings);
        let mut font = uniform_font();
        let data = HudData {
            sidebar: Some(crate::screens::hud::Sidebar::from_scores(
                "Board",
                &[("a".to_owned(), 1)],
                str::to_owned,
            )),
            sidebar_revision: 1,
            ..boss("Wither", 1.0)
        };
        let slot = |hud: &crate::modhud::ModHud, id: &str| {
            hud.slots()
                .iter()
                .find(|slot| mc_mods::catalog::all()[slot.mod_index].id == id)
                .and_then(|slot| slot.rect)
        };
        let mut d = DrawList::new();
        hud.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut d,
        );
        assert!(slot(&hud, "fps").is_some() && slot(&hud, "boss_bar").is_some());
        let mut d = DrawList::new();
        hud.draw_under_debug(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut d,
        );
        assert!(
            slot(&hud, "fps").is_none(),
            "an ordinary mod element hides under F3"
        );
        assert!(slot(&hud, "boss_bar").is_some());
        assert!(slot(&hud, "scoreboard").is_some());
    }

    #[test]
    fn options_reach_the_draw() {
        let mut settings = ModSettings::new();
        let mut font = uniform_font();
        let data = boss("Wither", 0.5);
        let mut element = BossBarElement::default();
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut full = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut full,
        );

        settings.set_bool("boss_bar", "show_bar", false);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut text_only = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut text_only,
        );
        assert!(text_only
            .batches()
            .iter()
            .all(|batch| batch.texture != *crate::widget::UiTexture::icons()));

        settings.set_bool("boss_bar", "show_bar", true);
        settings.set_bool("boss_bar", "text_shadow", false);
        element.sync(&settings);
        element.layout(&ctx(&data, &settings, (640.0, 360.0), false), &font);
        let mut flat = DrawList::new();
        element.draw(
            &ctx(&data, &settings, (640.0, 360.0), false),
            &mut font,
            &mut flat,
        );
        assert!(
            flat.vertices().len() < full.vertices().len(),
            "no shadow pass: fewer glyph quads"
        );
    }
}
