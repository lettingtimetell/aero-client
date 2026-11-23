//! Status Effects (`potion_status`): your active potion effects on the HUD, the list vanilla
//! only shows inside the inventory.
//!
//! ```text
//! [icon] Speed II
//!        1:30
//! [icon] Strength
//!        0:07        <- pulses below `blink_seconds`
//! ```
//!
//! * One row per effect, 20 px tall with a 2 px gap: the vanilla 18×18 icon cell from
//!   `textures/gui/container/inventory.png` (`InventoryEffectRenderer`'s UV), the name with its
//!   level, and under it the time left in `duration_color`.
//! * The order is `HudData::effects`', sorted by potion id like the inventory's own list.
//! * A name is rebuilt only when its row's effect or level changes and a duration once per whole
//!   second, so the frames in between format nothing.
//! * Blink: below `blink_seconds` the text alternates between full and 30 % alpha every 500 ms.
//!   The icon stays solid, and capped (`**:**`) effects never blink.
//! * Anchored on the right third of the screen it mirrors: icons at the screen edge, the text
//!   right-aligned next to them.

use std::fmt::Write as _;

use mc_mods::{HudAnchor, ModSettings};

use crate::font::FontRenderer;
use crate::modhud::{HudCtx, HudEffect, HudElement, HudTextStyle};
use crate::screens::hud::INVENTORY_TEXTURE;
use crate::widget::{DrawList, UiTexture};

/// From one row's top to the next: a 20-pixel row and a 2-pixel gap.
const ROW_PITCH: f32 = 22.0;
/// The icon cell in `inventory.png`.
const ICON: i32 = 18;
/// Where the text column starts while icons are shown.
const TEXT_X: f32 = 20.0;
/// Padding around the whole list while the background box is on.
const PAD: f32 = 3.0;
/// The dim half of a blink, as a fraction of the text colour's own alpha.
const BLINK_ALPHA: f32 = 0.3;

/// What the HUD editor shows while you have no effects: `Speed II 1:30` and a blinking
/// `Strength 0:07` (research `HUD-EDITOR.md`).
const PREVIEW: [HudEffect; 2] = [
    HudEffect {
        id: 1,
        amplifier: 1,
        duration_ticks: 90 * 20,
        duration_max: false,
    },
    HudEffect {
        id: 5,
        amplifier: 0,
        duration_ticks: 7 * 20,
        duration_max: false,
    },
];

/// Roman numerals I to X (`enchantment.level.1` .. `.10`, the only ones vanilla's lang has).
pub(crate) const ROMAN: [&str; 10] = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

/// Append the level suffix of a zero-based `amplifier`: nothing for level I (or a negative
/// amplifier, as vanilla's tooltip), else ` II` with `roman` (I to X) or ` 2`. Levels above X
/// are always Arabic. Vanilla's inventory stops at IV; servers hand out Speed V and higher.
pub(crate) fn write_level(out: &mut String, amplifier: i32, roman: bool) {
    let level = i64::from(amplifier) + 1;
    if level <= 1 {
        return;
    }
    if roman && level <= 10 {
        out.push(' ');
        out.push_str(ROMAN[(level - 1) as usize]);
    } else {
        let _ = write!(out, " {level}");
    }
}

/// Vanilla's `Potion.getDurationString`: `m:ss` (minutes keep counting past an hour, `75:00`),
/// or `**:**` for a capped duration.
pub(crate) fn write_duration(out: &mut String, ticks: i32, capped: bool) {
    if capped {
        out.push_str("**:**");
    } else {
        let seconds = ticks.max(0) / 20;
        let _ = write!(out, "{}:{:02}", seconds / 60, seconds % 60);
    }
}

/// `argb` with its alpha scaled by `factor`. The font draws an alpha below 4 opaque (vanilla's
/// `drawString` quirk), so such an input counts as opaque and the result never drops below 4.
pub(crate) fn scale_alpha(argb: u32, factor: f32) -> u32 {
    let alpha = match argb >> 24 {
        a if a < 4 => 255,
        a => a,
    };
    let scaled = ((alpha as f32 * factor).round() as u32).clamp(4, 255);
    (argb & 0x00FF_FFFF) | (scaled << 24)
}

/// One effect's cached text.
#[derive(Debug, Default)]
struct Row {
    /// `(id, amplifier)` the name was built for.
    name_key: Option<(u8, u8)>,
    name: String,
    name_width: f32,
    /// The whole seconds the duration was built for (`None` inside: capped).
    duration_key: Option<Option<i32>>,
    duration: String,
    duration_width: f32,
    /// The cell in `inventory.png`, `None` for the effects vanilla has no icon for (instant
    /// health and damage, saturation).
    icon: Option<i32>,
    /// In the dim half of a blink this frame.
    dim: bool,
}

/// The `potion_status` element.
#[derive(Debug)]
pub struct PotionStatusElement {
    style: HudTextStyle,
    show_icons: bool,
    roman: bool,
    blink: bool,
    blink_time_only: bool,
    effect_name_color: bool,
    blink_ticks: i32,
    duration_color: u32,
    hide_permanent: bool,
    /// Anchored on the right: icons at the right edge, text right-aligned.
    mirrored: bool,
    /// `inventory.png`, built once (`UiTexture::gui` allocates).
    texture: UiTexture,
    /// Grown on demand and never shrunk, so rows keep their strings from frame to frame.
    rows: Vec<Row>,
    /// Rows in use this frame.
    count: usize,
    font_generation: u64,
    size: (f32, f32),
}

impl Default for PotionStatusElement {
    fn default() -> Self {
        PotionStatusElement {
            style: HudTextStyle::default(),
            show_icons: true,
            roman: true,
            blink: true,
            blink_time_only: false,
            effect_name_color: false,
            blink_ticks: 10 * 20,
            duration_color: 0xFFAA_AAAA,
            hide_permanent: false,
            mirrored: false,
            texture: UiTexture::gui(INVENTORY_TEXTURE),
            rows: Vec::new(),
            count: 0,
            font_generation: u64::MAX,
            size: (0.0, 0.0),
        }
    }
}

impl PotionStatusElement {
    /// Force every row's text to be rebuilt.
    fn forget_text(&mut self) {
        for row in &mut self.rows {
            row.name_key = None;
            row.duration_key = None;
        }
    }

    /// Where the text column starts, inside the padding.
    fn text_offset(&self) -> f32 {
        if self.show_icons {
            TEXT_X
        } else {
            0.0
        }
    }
}

impl HudElement for PotionStatusElement {
    fn id(&self) -> (&'static str, &'static str) {
        ("potion_status", "potion_status")
    }

    fn sync(&mut self, settings: &ModSettings) {
        const ID: &str = "potion_status";
        self.style = HudTextStyle::read(settings, ID);
        self.show_icons = settings.bool(ID, "show_icons");
        self.roman = settings.choice(ID, "numerals") == 0;
        self.blink = settings.bool(ID, "blink");
        self.blink_time_only = settings.bool(ID, "blink_time_only");
        self.effect_name_color = settings.bool(ID, "effect_name_color");
        self.blink_ticks = (settings.f32(ID, "blink_seconds").clamp(1.0, 30.0) * 20.0) as i32;
        self.duration_color = settings.color(ID, "duration_color");
        self.hide_permanent = settings.bool(ID, "hide_permanent");
        self.mirrored = matches!(
            settings.placement(ID, ID).anchor,
            HudAnchor::TopRight | HudAnchor::CenterRight | HudAnchor::BottomRight
        );
        self.forget_text();
    }

    fn layout(&mut self, ctx: &HudCtx<'_>, font: &FontRenderer) -> Option<(f32, f32)> {
        if font.layout_generation() != self.font_generation {
            self.font_generation = font.layout_generation();
            self.forget_text();
        }
        let hide_permanent = self.hide_permanent;
        let listed = move |effect: &&HudEffect| !(hide_permanent && effect.duration_max);
        let effects: &[HudEffect] = if ctx.editing && !ctx.data.effects.iter().any(|e| listed(&e)) {
            &PREVIEW
        } else {
            &ctx.data.effects
        };
        let (roman, blink, blink_ticks) = (self.roman, self.blink, self.blink_ticks);
        let dim_phase = (ctx.data.now_ms / 500) % 2 == 1;
        let mut count = 0;
        for effect in effects.iter().filter(listed) {
            if count == self.rows.len() {
                self.rows.push(Row::default());
            }
            let row = &mut self.rows[count];
            count += 1;
            let name_key = (effect.id, effect.amplifier);
            if row.name_key != Some(name_key) {
                row.name_key = Some(name_key);
                let potion = mc_data::potion(effect.id);
                row.icon = potion.and_then(|potion| potion.status_icon_index);
                row.name.clear();
                match potion {
                    Some(potion) => row.name.push_str(potion.display_name),
                    None => {
                        let _ = write!(row.name, "Effect {}", effect.id);
                    }
                }
                write_level(&mut row.name, i32::from(effect.amplifier), roman);
                row.name_width = font.string_width(&row.name).max(0) as f32;
            }
            let seconds = (!effect.duration_max).then(|| effect.duration_ticks.max(0) / 20);
            if row.duration_key != Some(seconds) {
                row.duration_key = Some(seconds);
                row.duration.clear();
                write_duration(
                    &mut row.duration,
                    effect.duration_ticks,
                    effect.duration_max,
                );
                row.duration_width = font.string_width(&row.duration).max(0) as f32;
            }
            row.dim =
                blink && dim_phase && !effect.duration_max && effect.duration_ticks <= blink_ticks;
        }
        self.count = count;
        if count == 0 {
            return None;
        }
        let widest = self.rows[..count]
            .iter()
            .map(|row| row.name_width.max(row.duration_width))
            .fold(0.0, f32::max);
        let (width, height) = (self.text_offset() + widest, count as f32 * ROW_PITCH - 2.0);
        self.size = if self.style.background.is_some() {
            (width + 2.0 * PAD, height + 2.0 * PAD)
        } else {
            (width, height)
        };
        Some(self.size)
    }

    fn draw(&mut self, _ctx: &HudCtx<'_>, font: &mut FontRenderer, d: &mut DrawList) {
        let pad = match self.style.background {
            Some(argb) => {
                d.quad_rect(
                    &UiTexture::Solid,
                    (0.0, 0.0, self.size.0, self.size.1),
                    (0.0, 0.0, 1.0, 1.0),
                    argb,
                );
                PAD
            }
            None => 0.0,
        };
        let inner_right = self.size.0 - pad;
        let text_offset = self.text_offset();
        for (n, row) in self.rows[..self.count].iter().enumerate() {
            let top = pad + n as f32 * ROW_PITCH;
            if let Some(index) = row.icon.filter(|_| self.show_icons) {
                let x = if self.mirrored {
                    inner_right - ICON as f32
                } else {
                    pad
                };
                d.blit(
                    &self.texture,
                    x as i32,
                    top as i32 + 1,
                    (index % 8) * ICON,
                    198 + (index / 8) * ICON,
                    ICON,
                    ICON,
                );
            }
            let (name_x, duration_x) = if self.mirrored {
                let right = inner_right - text_offset;
                (right - row.name_width, right - row.duration_width)
            } else {
                (pad + text_offset, pad + text_offset)
            };
            let name_color = if self.effect_name_color {
                row.name_key
                    .and_then(|(id, _)| mc_data::potion(id))
                    .map_or(self.style.color, |potion| {
                        (self.style.color & 0xFF00_0000) | potion.liquid_color
                    })
            } else {
                self.style.color
            };
            let name_color = if row.dim && !self.blink_time_only {
                scale_alpha(name_color, BLINK_ALPHA)
            } else {
                name_color
            };
            let duration_color = if row.dim {
                scale_alpha(self.duration_color, BLINK_ALPHA)
            } else {
                self.duration_color
            };
            let shadow = self.style.shadow;
            font.draw_string(d, &row.name, name_x, top + 1.0, name_color as i32, shadow);
            let (y, color) = (top + 11.0, duration_color as i32);
            font.draw_string(d, &row.duration, duration_x, y, color, shadow);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modhud::HudData;
    use crate::widget::test_support::uniform_font;

    fn ctx<'a>(data: &'a HudData, settings: &'a ModSettings, editing: bool) -> HudCtx<'a> {
        HudCtx {
            data,
            settings,
            screen: (960.0, 540.0),
            pixel_scale: 2.0,
            editing,
        }
    }

    fn effect(id: u8, amplifier: u8, seconds: i32) -> HudEffect {
        HudEffect {
            id,
            amplifier,
            duration_ticks: seconds * 20,
            duration_max: false,
        }
    }

    fn with_effects(effects: Vec<HudEffect>, now_ms: u64) -> HudData {
        HudData {
            effects,
            now_ms,
            ..HudData::default()
        }
    }

    #[test]
    fn levels_and_durations_format_like_the_spec() {
        let level = |amplifier: i32, roman: bool| {
            let mut out = String::from("Speed");
            write_level(&mut out, amplifier, roman);
            out
        };
        assert_eq!(level(0, true), "Speed");
        assert_eq!(level(1, true), "Speed II");
        assert_eq!(level(4, true), "Speed V", "past vanilla's IV");
        assert_eq!(level(9, true), "Speed X");
        assert_eq!(level(10, true), "Speed 11", "above X is always Arabic");
        assert_eq!(level(1, false), "Speed 2");
        assert_eq!(
            level(-1, true),
            "Speed",
            "a negative amplifier shows no level"
        );
        let duration = |ticks: i32, capped: bool| {
            let mut out = String::new();
            write_duration(&mut out, ticks, capped);
            out
        };
        assert_eq!(duration(95 * 20, false), "1:35");
        assert_eq!(duration(75 * 60 * 20, false), "75:00");
        assert_eq!(duration(19, false), "0:00");
        assert_eq!(duration(32_767, true), "**:**");
        assert_eq!(scale_alpha(0xFFAA_BBCC, 0.3), 0x4DAA_BBCC);
        assert_eq!(
            scale_alpha(0x00FF_FFFF, 0.3),
            0x4DFF_FFFF,
            "alpha 0 draws opaque"
        );
    }

    #[test]
    fn rows_follow_the_effects_and_text_is_cached() {
        let settings = ModSettings::new();
        let font = uniform_font();
        let mut element = PotionStatusElement::default();
        element.sync(&settings);
        assert!(element
            .layout(&ctx(&HudData::default(), &settings, false), &font)
            .is_none());
        let mut data = with_effects(vec![effect(1, 1, 90), effect(5, 0, 7)], 0);
        data.effects[0].duration_ticks += 10;
        let (w, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(element.count, 2);
        assert_eq!(element.rows[0].name, "Speed II");
        assert_eq!(element.rows[0].duration, "1:30");
        assert_eq!(element.rows[0].icon, Some(0));
        assert_eq!(element.rows[1].name, "Strength");
        assert_eq!(element.rows[1].duration, "0:07");
        // Background on by default: 3 px around two 20 px rows and their 2 px gap.
        assert_eq!(h, 2.0 * ROW_PITCH - 2.0 + 2.0 * PAD);
        // The widest text decides: "Strength" (the font's space is narrower than a letter).
        let widest = font
            .string_width("Speed II")
            .max(font.string_width("Strength"));
        assert_eq!(widest, font.string_width("Strength"));
        assert_eq!(w, TEXT_X + widest as f32 + 2.0 * PAD);
        // A tick later, inside the same second: nothing is re-formatted (the marks survive).
        element.rows[0].name.push('!');
        element.rows[0].duration.push('!');
        data.effects[0].duration_ticks -= 1;
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[0].duration, "1:30!");
        // The next second rebuilds the duration only.
        data.effects[0].duration_ticks -= 10;
        element.layout(&ctx(&data, &settings, false), &font);
        assert_eq!(element.rows[0].duration, "1:29");
        assert_eq!(
            element.rows[0].name, "Speed II!",
            "same effect and level: same name"
        );
        // Effects without a status icon keep the text column aligned.
        let healing = with_effects(vec![effect(6, 0, 1)], 0);
        element.layout(&ctx(&healing, &settings, false), &font);
        assert_eq!(element.rows[0].icon, None);
        assert_eq!(element.rows[0].name, "Instant Health");
    }

    #[test]
    fn options_numerals_icons_and_endless_effects() {
        let mut settings = ModSettings::new();
        settings.set_choice("potion_status", "numerals", 1);
        settings.set_bool("potion_status", "show_icons", false);
        settings.set_bool("potion_status", "background", false);
        settings.set_bool("potion_status", "hide_permanent", true);
        let font = uniform_font();
        let mut element = PotionStatusElement::default();
        element.sync(&settings);
        let lobby_speed = HudEffect {
            duration_max: true,
            ..effect(1, 1, 0)
        };
        let data = with_effects(vec![lobby_speed, effect(8, 2, 180)], 0);
        let (w, h) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        assert_eq!(element.count, 1, "the endless speed is hidden");
        assert_eq!(element.rows[0].name, "Jump Boost 3");
        assert_eq!(element.rows[0].duration, "3:00");
        assert_eq!((w, h), (font.string_width("Jump Boost 3") as f32, 20.0));
        // Only endless effects left: nothing to draw in game.
        let only_endless = with_effects(vec![lobby_speed], 0);
        assert!(element
            .layout(&ctx(&only_endless, &settings, false), &font)
            .is_none());
    }

    #[test]
    fn effect_coloured_names_can_stay_steady_while_only_the_time_blinks() {
        let mut settings = ModSettings::new();
        settings.set_bool("potion_status", "show_icons", false);
        settings.set_bool("potion_status", "text_shadow", false);
        settings.set_bool("potion_status", "background", false);
        settings.set_bool("potion_status", "effect_name_color", true);
        settings.set_bool("potion_status", "blink_time_only", true);
        settings.set_color("potion_status", "text_color", 0x80FF_FFFF);
        let mut font = uniform_font();
        let mut element = PotionStatusElement::default();
        element.sync(&settings);
        let mut render = |now_ms| {
            let data = with_effects(vec![effect(1, 0, 7)], now_ms);
            let context = ctx(&data, &settings, false);
            element.layout(&context, &font).unwrap();
            let mut d = DrawList::new();
            element.draw(&context, &mut font, &mut d);
            d
        };
        let bright = render(0);
        let dim = render(500);
        let expected_rgb = mc_data::potion(1).unwrap().liquid_color;
        let rgb = [
            ((expected_rgb >> 16) & 255) as f32 / 255.0,
            ((expected_rgb >> 8) & 255) as f32 / 255.0,
            (expected_rgb & 255) as f32 / 255.0,
        ];
        let mut name_vertices = 0;
        let mut time_vertices = 0;
        for (a, b) in bright.vertices().iter().zip(dim.vertices()) {
            if a.pos[1] < 10.0 {
                assert_eq!(a.color, b.color, "name stays steady");
                assert_eq!(&a.color[..3], &rgb);
                assert!((a.color[3] - 128.0 / 255.0).abs() < 1e-6);
                name_vertices += 1;
            } else {
                assert!(b.color[3] < a.color[3], "time still blinks");
                time_vertices += 1;
            }
        }
        assert!(name_vertices > 0 && time_vertices > 0);
    }

    #[test]
    fn ending_effects_blink_and_the_editor_shows_a_preview() {
        let settings = ModSettings::new();
        let mut font = uniform_font();
        let mut element = PotionStatusElement::default();
        element.sync(&settings);
        let alphas = |element: &mut PotionStatusElement, font: &mut FontRenderer, data| {
            let mut d = DrawList::new();
            element.layout(&ctx(&data, &settings, false), font).unwrap();
            element.draw(&ctx(&data, &settings, false), font, &mut d);
            d.vertices()
                .iter()
                .map(|v| (v.color[3] * 255.0).round() as u32)
                .collect::<Vec<_>>()
        };
        // 7 s left is below the default 10 s: dim on the odd half-seconds only.
        let bright = alphas(
            &mut element,
            &mut font,
            with_effects(vec![effect(5, 0, 7)], 0),
        );
        let dim = alphas(
            &mut element,
            &mut font,
            with_effects(vec![effect(5, 0, 7)], 500),
        );
        assert!(!bright.contains(&77));
        assert!(dim.contains(&77), "30 % of the opaque text colour");
        // 30 s left does not blink.
        let calm = alphas(
            &mut element,
            &mut font,
            with_effects(vec![effect(5, 0, 30)], 500),
        );
        assert!(!calm.contains(&77));
        // The editor previews two effects while there are none.
        let empty = HudData::default();
        assert!(element
            .layout(&ctx(&empty, &settings, true), &font)
            .is_some());
        assert_eq!(element.count, 2);
        assert_eq!(element.rows[0].name, "Speed II");
        assert_eq!(element.rows[1].duration, "0:07");
    }

    #[test]
    fn right_anchors_mirror_the_row() {
        let mut settings = ModSettings::new();
        settings.set_placement(
            "potion_status",
            "potion_status",
            mc_mods::HudPlacement::new(HudAnchor::CenterRight, -2.0, 0.0),
        );
        let mut font = uniform_font();
        let mut element = PotionStatusElement::default();
        element.sync(&settings);
        let data = with_effects(vec![effect(1, 0, 60)], 0);
        let (w, _) = element
            .layout(&ctx(&data, &settings, false), &font)
            .unwrap();
        let mut d = DrawList::new();
        element.draw(&ctx(&data, &settings, false), &mut font, &mut d);
        let icon = d
            .batches()
            .iter()
            .find(|batch| batch.texture == UiTexture::gui(INVENTORY_TEXTURE))
            .expect("the icon is drawn from inventory.png");
        let left = d.vertices()[icon.range.clone()]
            .iter()
            .map(|v| v.pos[0])
            .fold(f32::MAX, f32::min);
        assert_eq!(left, w - PAD - ICON as f32, "icon at the right edge");
    }
}
